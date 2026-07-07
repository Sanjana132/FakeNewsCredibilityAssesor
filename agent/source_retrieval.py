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
from dataclasses import dataclass, field
from typing import Optional

from .tools import retrieve_all, SourceResult


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

    # Sort: fact-check sources first, then by relevance
    deduped.sort(key=lambda s: (
        0 if s.source in ("google_fc", "faiss") else 1,
        -s.relevance,
    ))

    fc_sources = [s for s in deduped
                  if s.source in ("google_fc", "faiss") and s.score is not None]
    fc_score   = float(sum(s.score for s in fc_sources) / len(fc_sources)) \
                 if fc_sources else None

    bundle = EvidenceBundle(
        claim=claim,
        sources=[s.to_dict() for s in deduped[:k_each * 4]],
        has_fc_result=bool(fc_sources),
        fc_score=fc_score,
        sources_used=sources_used,
    )
    return bundle


def _build_query(claim: str, speaker: str, context: str) -> str:
    """Build a concise search query from claim metadata."""
    parts = [claim[:200]]
    if speaker and speaker not in ("unknown", "anonymous_social_media", ""):
        parts.append(speaker)
    if context and context not in ("unknown", ""):
        parts.append(context)
    return " ".join(parts)


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
