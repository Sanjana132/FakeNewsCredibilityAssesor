"""
Phase 9 — Retrieval Tools for the LangGraph Agent.

Four async tools that the RETRIEVE node fans out over in parallel:
  1. GoogleFactCheckTool  — Google Fact Check Tools API (1000 req/day free)
  2. WikipediaTool        — Wikipedia search + page summary API
  3. FAISSTool            — Local FAISS index of scraped PolitiFact/Snopes
  4. NewsAPITool          — newsapi.org /everything endpoint

Each tool:
  • Is an async coroutine (awaitable)
  • Has a 10-second timeout
  • Fails gracefully: returns [] on any error
  • Returns a list of SourceResult dicts

Install:
    pip install aiohttp wikipedia-api sentence-transformers faiss-cpu
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_HERE    = Path(__file__).resolve().parent.parent
DATA_DIR  = _HERE / "data"

TIMEOUT_S = 10

# Wikipedia (and good API etiquette generally) require a descriptive User-Agent;
# requests without one get a 403. Sent on every outbound HTTP call.
HTTP_HEADERS = {
    "User-Agent": (
        "FakeNewsCredibilityBot/1.0 "
        "(academic research; contact sanj18reddy@gmail.com)"
    )
}


@dataclass
class SourceResult:
    title:     str
    snippet:   str
    url:       str
    score:     float        # verdict credibility score 0–1 if known, else None
    source:    str          # "google_fc" | "wikipedia" | "faiss" | "newsapi"
    relevance: float = 1.0  # cosine similarity or API rank-normalised

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GOOGLE FACT CHECK TOOLS API
# ─────────────────────────────────────────────────────────────────────────────

class GoogleFactCheckTool:
    """
    POST to factchecktools.googleapis.com/v1alpha1/claims:search.
    Requires: GOOGLE_FACTCHECK_API_KEY env var.
    Free tier: 1 000 requests/day.
    """
    _url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(self):
        self._key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")

    async def __call__(self, query: str, k: int = 5) -> list[SourceResult]:
        if not self._key:
            return []
        import aiohttp
        params = {"query": query, "key": self._key, "pageSize": k}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                        self._url, params=params, headers=HTTP_HEADERS,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    data = await r.json(content_type=None)
        except Exception:
            return []

        results = []
        for claim in (data.get("claims") or []):
            review = (claim.get("claimReview") or [{}])[0]
            rating_raw = review.get("textualRating", "").lower()
            # Map common ratings to [0,1]
            score = _rating_to_score(rating_raw)
            results.append(SourceResult(
                title=claim.get("text", "")[:200],
                snippet=review.get("textualRating", ""),
                url=review.get("url", ""),
                score=score,
                source="google_fc",
                relevance=1.0 - (len(results) * 0.05),
            ))
        return results[:k]


def _rating_to_score(rating: str) -> float:
    """Heuristic: map free-text rating to [0,1]."""
    for kw, val in [
        ("true", 1.0), ("correct", 1.0), ("accurate", 0.9),
        ("mostly true", 0.8), ("mostly correct", 0.8),
        ("mixed", 0.6), ("partially", 0.6), ("half", 0.6),
        ("mostly false", 0.3), ("misleading", 0.35),
        ("false", 0.1), ("incorrect", 0.1), ("fabricated", 0.0),
        ("pants", 0.0), ("scam", 0.0),
    ]:
        if kw in rating:
            return val
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 2. WIKIPEDIA TOOL
# ─────────────────────────────────────────────────────────────────────────────

class WikipediaTool:
    """
    Wikipedia REST API: search titles then fetch intro summary.
    No key required. Rate limit: be polite (handled by TIMEOUT_S + single req).
    """
    _search_url = "https://en.wikipedia.org/w/api.php"
    _summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

    async def __call__(self, query: str, k: int = 3) -> list[SourceResult]:
        import aiohttp
        results = []
        try:
            async with aiohttp.ClientSession() as sess:
                # Step 1: search
                params = {
                    "action": "query", "list": "search",
                    "srsearch": query, "srlimit": k,
                    "format": "json",
                }
                async with sess.get(
                        self._search_url, params=params, headers=HTTP_HEADERS,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    data = await r.json(content_type=None)

                hits = (data.get("query") or {}).get("search") or []

                for hit in hits[:k]:
                    title = hit.get("title", "")
                    snippet = _strip_html(hit.get("snippet", ""))
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    results.append(SourceResult(
                        title=title, snippet=snippet[:300],
                        url=url, score=None,
                        source="wikipedia",
                        relevance=1.0 - (len(results) * 0.1),
                    ))
        except Exception:
            pass
        return results


def _strip_html(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAISS TOOL
# ─────────────────────────────────────────────────────────────────────────────

class FAISSTool:
    """
    Query the local FAISS index built by speaker_scraper.py.
    Uses all-MiniLM-L6-v2 to embed the query and performs cosine search.
    Falls back gracefully if index file is absent.
    """
    def __init__(self):
        self._index = None
        self._meta  = None
        self._model = None
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        self._loaded = True
        index_path = DATA_DIR / "faiss.index"
        meta_path  = DATA_DIR / "faiss_meta.json"
        if not index_path.exists():
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self._index = faiss.read_index(str(index_path))
            self._meta  = json.loads(meta_path.read_text())
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._index = None

    async def __call__(self, query: str, k: int = 5) -> list[SourceResult]:
        self._lazy_load()
        if self._index is None:
            return []
        try:
            import numpy as np
            emb = self._model.encode([query], normalize_embeddings=True,
                                      show_progress_bar=False)
            emb = emb.astype("float32")
            scores, indices = self._index.search(emb, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                m = self._meta[idx]
                results.append(SourceResult(
                    title=m.get("claim", "")[:200],
                    snippet=f"Verdict: {m.get('verdict', '')}",
                    url=m.get("url", ""),
                    score=m.get("verdict_score"),
                    source="faiss",
                    relevance=float(score),
                ))
            return results
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEWS API TOOL
# ─────────────────────────────────────────────────────────────────────────────

class NewsAPITool:
    """
    newsapi.org /v2/everything endpoint.
    Requires: NEWSAPI_KEY env var.
    Free tier: 100 requests/day, last 30 days.
    """
    _url = "https://newsapi.org/v2/everything"

    def __init__(self):
        self._key = os.environ.get("NEWSAPI_KEY", "")

    async def __call__(self, query: str, k: int = 5) -> list[SourceResult]:
        if not self._key:
            return []
        import aiohttp
        params = {
            "q": query, "apiKey": self._key,
            "pageSize": k, "sortBy": "relevancy",
            "language": "en",
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                        self._url, params=params, headers=HTTP_HEADERS,
                        timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as r:
                    data = await r.json(content_type=None)
        except Exception:
            return []

        results = []
        for art in (data.get("articles") or [])[:k]:
            results.append(SourceResult(
                title=art.get("title", "")[:200],
                snippet=(art.get("description") or "")[:300],
                url=art.get("url", ""),
                score=None,
                source="newsapi",
                relevance=1.0 - (len(results) * 0.05),
            ))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_google = GoogleFactCheckTool()
_wiki   = WikipediaTool()
_faiss  = FAISSTool()
_news   = NewsAPITool()


async def retrieve_all(query: str, k_each: int = 3) -> dict[str, list[SourceResult]]:
    """
    Fan out all 4 tools concurrently.
    Returns dict keyed by tool name; any tool that errors returns [].
    """
    gfc, wik, fai, nws = await asyncio.gather(
        _google(query, k_each),
        _wiki(query, k_each),
        _faiss(query, k_each),
        _news(query, k_each),
        return_exceptions=True,
    )

    def _safe(result):
        return result if isinstance(result, list) else []

    return {
        "google_fc":  _safe(gfc),
        "wikipedia":  _safe(wik),
        "faiss":      _safe(fai),
        "newsapi":    _safe(nws),
    }
