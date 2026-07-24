"""
Phase 9 — Source Retrieval Layer

Three-layer retrieval cascade:
  Layer 1: Google Fact Check API   (most authoritative; 1k/day free)
  Layer 2: FAISS local index       (always available; scraped metadata)
  Layer 3: Mistral correction summary (LLM synthesis of all evidence)

Each layer is called only if the previous layer returns fewer than
min_results relevant hits (score < 0.5 and relevance > 0.5).

Usage from graph.py:
    from agent.source_retrieval import retrieve_sources
    evidence = await retrieve_sources(claim, speaker, context)
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from .tools import retrieve_all, SourceResult

# Stopwords stripped when building keyword queries and scoring relevance, so a
# claim searches on its salient terms rather than "the/of/that/…" noise.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "they", "them", "their", "he",
    "she", "his", "her", "you", "your", "we", "our", "not", "no", "do", "does",
    "did", "has", "have", "had", "will", "would", "can", "could", "should", "may",
    "might", "about", "into", "than", "then", "there", "here", "what", "which",
    "who", "whom", "how", "when", "where", "why", "all", "any", "some", "more",
    "most", "other", "such", "only", "own", "same", "just", "said", "says", "say",
    "new", "one", "two", "contain", "contains", "let", "lets", "make", "makes",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower())
            if w not in _STOPWORDS}


def _keywords(text: str, limit: int = 12) -> list[str]:
    """Ordered, de-duplicated content words — a focused search query."""
    seen: set[str] = set()
    out: list[str] = []
    for w in re.findall(r"[A-Za-z]{3,}", text or ""):
        lw = w.lower()
        if lw in _STOPWORDS or lw in seen:
            continue
        seen.add(lw)
        out.append(w)
        if len(out) >= limit:
            break
    return out


@dataclass
class EvidenceBundle:
    claim:          str
    sources:        list[dict]             = field(default_factory=list)
    has_fc_result:  bool                   = False
    fc_score:       Optional[float]        = None
    sources_used:   list[str]              = field(default_factory=list)
    summary:        str                    = ""

    def to_dict(self) -> dict:
        return {
            "claim":         self.claim,
            "sources":       self.sources,
            "has_fc_result": self.has_fc_result,
            "fc_score":      self.fc_score,
            "sources_used":  self.sources_used,
            "summary":       self.summary,
        }


async def retrieve_sources(claim: str, speaker: str = "",
                            context: str = "",
                            k_each: int = 3,
                            min_results: int = 1) -> EvidenceBundle:
    """
    Fan out all four retrieval tools, collate and deduplicate results.

    Returns an EvidenceBundle with:
      .sources      — list of SourceResult dicts, sorted by relevance
      .has_fc_result — True if at least one fact-check verdict was found
      .fc_score     — mean verdict score from fact-check sources (or None)
    """
    query = _build_query(claim, speaker, context)
    tool_results = await retrieve_all(query, k_each=k_each)

    all_sources: list[SourceResult] = []
    sources_used: list[str] = []
    for tool_name, results in tool_results.items():
        if results:
            all_sources.extend(results)
            sources_used.append(tool_name)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    deduped: list[SourceResult] = []
    for sr in all_sources:
        if sr.url and sr.url not in seen_urls:
            seen_urls.add(sr.url)
            deduped.append(sr)
        elif not sr.url:
            deduped.append(sr)

    # ── Relevance rerank/filter ──────────────────────────────────────────────
    # Keyword-search tools (Wikipedia/News) occasionally return tangential hits
    # (e.g. a TV-episode list for a "microchip" claim). Score each source by how
    # many of the claim's content words appear in its title (weighted) + snippet,
    # drop non-fact-check sources with zero overlap, and sort by that relevance.
    claim_words = _content_words(claim)

    def _is_fc(s: SourceResult) -> bool:
        return s.source in ("google_fc", "faiss") and s.score is not None

    def _overlap(s: SourceResult) -> float:
        title_hits = len(claim_words & _content_words(s.title))
        snip_hits  = len(claim_words & _content_words(s.snippet))
        return float(2 * title_hits + snip_hits)

    for s in deduped:
        s.relevance = _overlap(s)

    # Keep fact-checks always; keep others only if they share ≥1 content word.
    kept = [s for s in deduped if _is_fc(s) or s.relevance > 0]
    # Guard: never drop everything just because overlap was thin — fall back to
    # the original hits if the filter left no non-fact-check sources.
    if not any(not _is_fc(s) for s in kept) and deduped:
        kept = deduped

    kept.sort(key=lambda s: (0 if _is_fc(s) else 1, -s.relevance))

    fc_sources = [s for s in kept if _is_fc(s)]
    fc_score   = float(sum(s.score for s in fc_sources) / len(fc_sources)) \
                 if fc_sources else None

    bundle = EvidenceBundle(
        claim=claim,
        sources=[s.to_dict() for s in kept[:k_each * 4]],
        has_fc_result=bool(fc_sources),
        fc_score=fc_score,
        sources_used=sources_used,
    )
    return bundle


def _build_query(claim: str, speaker: str, context: str) -> str:
    """
    Build a focused keyword query from the claim's content words (+ speaker).
    Dropping stopwords sharpens keyword-search tools like Wikipedia — the full
    sentence with "the/that/…" tends to surface tangential articles. Context
    (e.g. "a speech") is intentionally excluded as it only adds noise.
    """
    parts = _keywords(claim, limit=12)
    if speaker and speaker not in ("unknown", "anonymous_social_media", ""):
        parts.append(speaker)
    return " ".join(parts) or claim[:200]


def format_evidence_for_llm(bundle: EvidenceBundle, max_sources: int = 4) -> str:
    """
    Format the evidence bundle into a text block suitable for the
    Mistral REASON prompt.
    """
    if not bundle.sources:
        return "No external sources found for this claim."

    lines = ["Retrieved evidence:\n"]
    for i, src in enumerate(bundle.sources[:max_sources], 1):
        verdict_str = (f"  Verdict score: {src['score']:.2f}"
                       if src.get("score") is not None else "")
        lines.append(
            f"[{i}] {src.get('title', '')} ({src.get('source', '')})\n"
            f"  {src.get('snippet', '')}{verdict_str}\n"
            f"  URL: {src.get('url', 'N/A')}\n"
        )
    return "\n".join(lines)
