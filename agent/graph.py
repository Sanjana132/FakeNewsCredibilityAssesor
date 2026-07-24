"""
Phase 9 — LangGraph Agent Pipeline

6-node credibility assessment graph:

  DECOMPOSE → CLASSIFY → RETRIEVE → SCORE → REASON(cond) → COMPILE

  DECOMPOSE : Parse and sanitise the input claim.
  CLASSIFY  : Run DeBERTa (Phase 5) → point score + MC Dropout CI.
  RETRIEVE  : asyncio.gather over 4 tools (FC API, Wiki, FAISS, NewsAPI).
  SCORE     : Fuse DeBERTa score with fact-check evidence score (if found).
  REASON    : Conditional — only if fused_score < 0.5. Calls Mistral-7B
              adapter to generate a justification paragraph.
  COMPILE   : Package everything into a final AssessmentResult.

Install:
    pip install langgraph langchain-core

Usage:
    from agent.graph import run_assessment
    result = await run_assessment("Obama tripled the deficit", "Barack Obama",
                                  "a campaign rally")
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from agent.source_retrieval import retrieve_sources, format_evidence_for_llm


# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # Input
    claim:          str
    speaker:        str
    context:        str
    prior:          float

    # CLASSIFY output
    deberta_score:  Optional[float]
    deberta_lower:  Optional[float]
    deberta_upper:  Optional[float]
    deberta_std:    Optional[float]

    # RETRIEVE output
    evidence:       Optional[dict]

    # SCORE output
    fused_score:    Optional[float]
    fused_lower:    Optional[float]
    fused_upper:    Optional[float]
    fc_score:       Optional[float]

    # REASON output
    explanation:    Optional[str]
    sources_used:   list[str]

    # COMPILE output
    verdict:        Optional[str]
    elapsed_ms:     Optional[float]
    errors:         list[str]


# ─────────────────────────────────────────────────────────────────────────────
# NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def node_decompose(state: AgentState) -> AgentState:
    """Sanitise and normalise inputs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cred123", _HERE / "data_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    claim   = str(state.get("claim", "")).strip()[:2000]
    speaker = str(state.get("speaker", "")).strip()
    context = mod.normalise_context(state.get("context", "unknown"))
    prior   = float(mod.get_context_prior(context))

    return {**state, "claim": claim, "speaker": speaker,
            "context": context, "prior": prior,
            "errors": state.get("errors", [])}


def node_classify(state: AgentState) -> AgentState:
    """Run DeBERTa with MC Dropout."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase5", _HERE / "deberta_model.py")
    p5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p5)

    errors = list(state.get("errors", []))
    model_dir = _HERE / "models"
    tok_path  = model_dir / "deberta_tokenizer"
    wt_path   = model_dir / "deberta_best.pt"

    if not tok_path.exists() or not wt_path.exists():
        errors.append("DeBERTa weights not found — using prior as score")
        prior = state.get("prior", 0.5)
        return {**state,
                "deberta_score": prior, "deberta_lower": prior - 0.15,
                "deberta_upper": prior + 0.15, "deberta_std": 0.08,
                "errors": errors}

    try:
        import torch
        from transformers import AutoTokenizer
        device = p5.detect_device()
        tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)
        model = p5.DeBERTaCredibilityModel()
        ckpt  = torch.load(wt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        # Build features
        from data_pipeline import extract_all_features, get_context_prior
        feats_dict = extract_all_features(state["claim"], state["context"])
        feats = [feats_dict.get(c, 0.0) for c in p5.FEAT_COLS
                 if c not in ("context_credibility_prior", "token_length_approx")]
        prior = float(get_context_prior(state["context"]))
        feats.append(prior)
        # token_length_approx
        feats.append(len(state["claim"].split()) * 1.3)

        result = p5.predict_with_uncertainty(
            model, tokenizer,
            state["claim"], state.get("speaker", ""),
            state["context"], prior, feats, device=device)

        return {**state,
                "deberta_score": result["mean"],
                "deberta_lower": result["lower"],
                "deberta_upper": result["upper"],
                "deberta_std":   result["std"],
                "errors": errors}
    except Exception as e:
        errors.append(f"DeBERTa classify failed: {e}")
        prior = state.get("prior", 0.5)
        return {**state,
                "deberta_score": prior, "deberta_lower": prior - 0.15,
                "deberta_upper": prior + 0.15, "deberta_std": 0.08,
                "errors": errors}


async def node_retrieve(state: AgentState) -> AgentState:
    """Fan out all 4 retrieval tools concurrently."""
    errors = list(state.get("errors", []))
    try:
        bundle = await retrieve_sources(
            state["claim"], state.get("speaker", ""),
            state.get("context", "unknown"), k_each=3)
        return {**state,
                "evidence":     bundle.to_dict(),
                "fc_score":     bundle.fc_score,
                "sources_used": bundle.sources_used,
                "errors": errors}
    except Exception as e:
        errors.append(f"Retrieval failed: {e}")
        return {**state, "evidence": None, "fc_score": None,
                "sources_used": [], "errors": errors}


def node_score(state: AgentState) -> AgentState:
    """Fuse DeBERTa score with fact-check evidence."""
    d_score  = state.get("deberta_score", 0.5)
    fc_score = state.get("fc_score")
    d_lower  = state.get("deberta_lower", d_score - 0.15)
    d_upper  = state.get("deberta_upper", d_score + 0.15)

    if fc_score is not None:
        # Weight: 0.6 DeBERTa + 0.4 fact-check when a verdict is available.
        fused = 0.6 * d_score + 0.4 * fc_score
        # Re-centre the DeBERTa confidence interval on the fused score, then
        # shrink its width when the fact-check agrees (agreement → 1 narrows to
        # 70% of the original width; full disagreement keeps the full width).
        half_width = (d_upper - d_lower) / 2.0
        agreement  = 1.0 - abs(d_score - fc_score)   # 1.0 = identical scores
        ci_shrink  = 0.7 + 0.3 * (1.0 - agreement)   # ∈ [0.7, 1.0]
        half_width *= ci_shrink
        fused_lower = fused - half_width
        fused_upper = fused + half_width
    else:
        fused       = d_score
        fused_lower = d_lower
        fused_upper = d_upper

    fused = round(max(0.0, min(1.0, fused)), 4)
    return {**state,
            "fused_score":  fused,
            "fused_lower":  round(max(0.0, min(1.0, fused_lower)), 4),
            "fused_upper":  round(max(0.0, min(1.0, fused_upper)), 4)}


async def node_reason(state: AgentState) -> AgentState:
    """Generate LLM justification (only called if fused_score < 0.5)."""
    errors = list(state.get("errors", []))
    try:
        from llm_finetune import generate_explanation
        evidence_text = ""
        if state.get("evidence"):
            from agent.source_retrieval import EvidenceBundle, format_evidence_for_llm
            bundle_dict = state["evidence"]
            bundle = EvidenceBundle(
                claim=bundle_dict["claim"],
                sources=bundle_dict.get("sources", []),
                has_fc_result=bundle_dict.get("has_fc_result", False),
                fc_score=bundle_dict.get("fc_score"),
                sources_used=bundle_dict.get("sources_used", []),
            )
            evidence_text = "\n\n" + format_evidence_for_llm(bundle)

        augmented_claim = state["claim"] + evidence_text
        explanation = generate_explanation(
            augmented_claim,
            state.get("speaker", ""),
            state.get("context", "unknown"),
            state.get("fused_score", 0.5),
            max_new_tokens=256,
        )
        return {**state, "explanation": explanation, "errors": errors}
    except Exception as e:
        errors.append(f"LLM reason failed: {e}")
        return {**state,
                "explanation": _heuristic_explanation(state),
                "errors": errors}


def _heuristic_explanation(state: AgentState) -> str:
    """Fallback when LLM is unavailable."""
    score = state.get("fused_score", 0.5)
    verdict = _score_to_verdict(score)
    sources = state.get("sources_used", [])
    src_str = f" Sources checked: {', '.join(sources)}." if sources else ""
    return (
        f"Based on our credibility model, this claim scores {score:.2f}/1.0 "
        f"({verdict}).{src_str} "
        f"[Full LLM justification requires the fine-tuned Mistral adapter — "
        f"run python llm_finetune.py --train]"
    )


def node_compile(state: AgentState) -> AgentState:
    """Assemble the final result."""
    score   = state.get("fused_score", 0.5)
    verdict = _score_to_verdict(score)
    return {**state, "verdict": verdict}


def _score_to_verdict(score: float) -> str:
    if score < 0.35:  return "Likely False"
    if score < 0.65:  return "Unverified / Mixed"
    return "Likely True"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_graph():
    """Build the LangGraph StateGraph (lazy-import to avoid hard dependency)."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None

    g = StateGraph(AgentState)
    g.add_node("decompose", node_decompose)
    g.add_node("classify",  node_classify)
    g.add_node("retrieve",  node_retrieve)
    g.add_node("score",     node_score)
    g.add_node("reason",    node_reason)
    g.add_node("compile",   node_compile)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "classify")
    g.add_edge("classify",  "retrieve")
    g.add_edge("retrieve",  "score")
    g.add_conditional_edges(
        "score",
        lambda s: "reason" if (s.get("fused_score", 0.5) < 0.5) else "compile",
        {"reason": "reason", "compile": "compile"},
    )
    g.add_edge("reason",  "compile")
    g.add_edge("compile", END)

    return g.compile()


_GRAPH = None  # compiled lazily on first call


async def run_assessment(claim: str, speaker: str = "",
                          context: str = "unknown") -> dict:
    """
    Main entry point for the agent pipeline.
    Returns a dict with the full AssessmentResult.
    """
    global _GRAPH
    t0 = time.perf_counter()

    initial_state: AgentState = {
        "claim":   claim,
        "speaker": speaker,
        "context": context,
        "errors":  [],
    }

    if _GRAPH is not None:
        final = await _GRAPH.ainvoke(initial_state)
    else:
        # Fallback: run nodes manually without LangGraph
        state = node_decompose(initial_state)
        state = node_classify(state)
        state = await node_retrieve(state)
        state = node_score(state)
        if state.get("fused_score", 0.5) < 0.5:
            state = await node_reason(state)
        else:
            state["explanation"] = None
        final = node_compile(state)

    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "claim":          final.get("claim"),
        "speaker":        final.get("speaker"),
        "context":        final.get("context"),
        "score":          final.get("fused_score"),
        "lower_90ci":     final.get("fused_lower"),
        "upper_90ci":     final.get("fused_upper"),
        "deberta_score":  final.get("deberta_score"),
        "fc_score":       final.get("fc_score"),
        "verdict":        final.get("verdict"),
        "explanation":    final.get("explanation"),
        "sources":        (final.get("evidence") or {}).get("sources", []),
        "sources_used":   final.get("sources_used", []),
        "errors":         final.get("errors", []),
        "elapsed_ms":     round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase 9 — LangGraph Agent")
    ap.add_argument("--claim",   required=True)
    ap.add_argument("--speaker", default="")
    ap.add_argument("--context", default="unknown")
    args = ap.parse_args()

    result = asyncio.run(run_assessment(args.claim, args.speaker, args.context))
    print(json.dumps(result, indent=2, ensure_ascii=False))
