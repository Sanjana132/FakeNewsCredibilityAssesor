"""
Agent pipeline — the credibility-verification BRAIN.

    assess(claim) -> Verdict

One place that owns the whole decision flow, independent of any UI:
    gate → score (DeBERTa + conformal) → retrieve → NLI-confirm → decide tier.

The returned Verdict is pure data; presentation layers (Gradio, FastAPI) format
it. This is deliberately separate from gradio_app.py so both front-ends call the
SAME logic instead of diverging.

Tier of the returned verdict (confidence order):
    "factcheck"  — a stance-confirmed published fact-check drives the verdict
    "evidence"   — no confirmed fact-check, but news/wiki snippets lean support/refute
    "stylistic"  — fallback: the DeBERTa style prior only (labelled as such)
    (ok=False)   — the input wasn't a checkable claim (gate rejected it)
"""
from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent.parent   # repo root


# ── .env loader (so any front-end that imports the pipeline gets the keys) ──────
def _load_env_file(path: Path) -> None:
    """Minimal, dependency-free .env loader. Existing env vars win (setdefault)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file(_HERE / ".env")


# ── cached model singletons (loaded on first assess) ────────────────────────────
_MODEL = None
_TOK = None
_DEVICE = None
_P5 = None


def _load_once():
    global _MODEL, _TOK, _DEVICE, _P5
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("phase5", _HERE / "deberta_model.py")
    _P5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_P5)

    _DEVICE = _P5.detect_device(os.environ.get("MODEL_DEVICE"))
    tok_path = _HERE / "models" / "deberta_tokenizer"
    wt_path = _HERE / "models" / "deberta_best.pt"
    if not tok_path.exists() or not wt_path.exists():
        raise FileNotFoundError(
            "models/deberta_best.pt or deberta_tokenizer/ not found — train "
            "Phase 5 or copy the trained models/ folder in first.")

    _TOK = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)
    model = _P5.DeBERTaCredibilityModel()
    ckpt = torch.load(wt_path, map_location=_DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(_DEVICE).eval()
    _MODEL = model
    print(f"[pipeline] model loaded on {_DEVICE}")


def score(text: str, speaker: str = "", context: str = "unknown") -> dict:
    """DeBERTa credibility score with (conformal-repaired) uncertainty interval."""
    _load_once()
    from data_pipeline import (
        normalise_context, get_context_prior, extract_all_features)
    ctx = normalise_context(context)
    prior = float(get_context_prior(ctx))
    fd = extract_all_features(text, ctx)
    fd["context_credibility_prior"] = prior
    fd["token_length_approx"] = len(text) / 4.0
    feats = [float(fd.get(c, 0.0)) for c in _P5.FEAT_COLS]
    return _P5.predict_with_uncertainty(
        _MODEL, _TOK, text, speaker, ctx, prior, feats, device=_DEVICE)


async def retrieve(text: str):
    """Best-effort source retrieval (Wikipedia / FactCheck / FAISS / News)."""
    try:
        from agent.source_retrieval import retrieve_sources
        return await retrieve_sources(text, k_each=4)
    except Exception as e:
        print(f"[pipeline] retrieval unavailable: {e}")
        return None


# ── input gate: refuse non-claims (the regressor can't abstain on its own) ──────
def _looks_like_word(w: str) -> bool:
    w = w.lower()
    if len(w) == 1:
        return w in {"a", "i"}
    if not any(v in w for v in "aeiouy"):
        return False                        # no vowel → 'asdfg', 'qwrt'
    if re.search(r"(.)\1\1", w):
        return False                        # 3+ identical chars in a row
    if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", w):
        return False                        # 5+ consonants in a row → mash
    return True


def is_checkable_claim(text: str) -> tuple[bool, str]:
    """(ok, reason). reason is '' when ok."""
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 3:
        return False, "give me a full statement — a sentence or more, not a word or two"
    real = [w for w in words if _looks_like_word(w)]
    if len(real) / len(words) < 0.6:
        return False, "this doesn't read as natural-language text"
    return True, ""


def is_factcheck(s: dict) -> bool:
    return s.get("source") in ("google_fc", "faiss") and s.get("score") is not None


# ── the verdict ─────────────────────────────────────────────────────────────────
@dataclass
class Verdict:
    ok: bool                              # False → gate rejected the input
    reason: str = ""                      # gate rejection reason ("__too_short__" special)
    tier: str = ""                        # "factcheck" | "evidence" | "stylistic"
    score: float = 0.5                    # headline number (verdict or prior)
    prior: float = 0.5                    # DeBERTa stylistic prior
    lo: float = 0.0
    hi: float = 0.0                       # uncertainty interval on the prior
    confidence: float = 0.0               # NLI match confidence (factcheck tier)
    n_matched: int = 0
    matches: list = field(default_factory=list)      # stance-confirmed fact-checks
    fc_srcs: list = field(default_factory=list)       # unconfirmed fact-checks
    other_srcs: list = field(default_factory=list)    # non-fact-check sources
    evidence: object = None               # stance.EvidenceStance (leaning) or None
    stance_available: bool = False        # did the NLI model load?
    bundle: object = None                 # raw EvidenceBundle


async def assess(claim_text: str) -> Verdict:
    """Run the full pipeline on a claim and return a structured Verdict."""
    msg = (claim_text or "").strip()
    if len(msg) < 12:
        return Verdict(ok=False, reason="__too_short__")
    ok, why = is_checkable_claim(msg)
    if not ok:
        return Verdict(ok=False, reason=why)

    claim = msg if len(msg) <= 1200 else msg[:1200]

    # 1. stylistic prior (always computed; headline only if nothing better)
    r = score(claim)
    prior, lo, hi = r["mean"], r["lower"], r["upper"]

    # 2. retrieve, then NLI-confirm which fact-checks address THIS claim
    bundle = await retrieve(claim)
    fc_srcs = [s for s in bundle.sources if is_factcheck(s)] if bundle else []
    other_srcs = [s for s in bundle.sources if not is_factcheck(s)] if bundle else []

    st = None
    if fc_srcs:
        from agent import stance
        st = stance.assess(claim, fc_srcs)

    if st is not None and st.available and st.verdict is not None:
        return Verdict(
            ok=True, tier="factcheck", score=st.verdict, prior=prior, lo=lo, hi=hi,
            confidence=st.confidence, n_matched=st.n_matched, matches=st.matches,
            other_srcs=other_srcs, bundle=bundle, stance_available=True)

    # 3. no confirmed fact-check → soft evidence leaning, else stylistic prior
    es = None
    if other_srcs:
        from agent import stance
        es = stance.assess_evidence(claim, other_srcs)
    tier = "evidence" if (es is not None and es.available and es.leaning) else "stylistic"
    return Verdict(
        ok=True, tier=tier, score=prior, prior=prior, lo=lo, hi=hi,
        fc_srcs=fc_srcs, other_srcs=other_srcs, evidence=es, bundle=bundle,
        stance_available=(st.available if st is not None else False))
