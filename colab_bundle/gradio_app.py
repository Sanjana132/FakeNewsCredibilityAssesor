"""
Phase 12 — Chatbot Demo

A conversational UI for the credibility detector: paste a news statement or
article and the bot replies with

  • a credibility score (0–1) + verdict + 90% confidence interval,
  • a short interpretation, and
  • sources to check the claim against (fact-checks / evidence), pulled from the
    retrieval agent — the disputing ones surfaced first for low-credibility claims.

The DeBERTa model is loaded ONCE and cached, so replies are fast after the first.

Run:
    MODEL_DEVICE=cpu python gradio_app.py          # loads the model directly
    python gradio_app.py --share                   # public Gradio link

Richer sources come from optional retrieval keys / index:
    GOOGLE_FACTCHECK_API_KEY   (free 1k/day)  — real fact-check verdicts
    NEWSAPI_KEY                (free 100/day)
    a FAISS index built via  python speaker_scraper.py --build-faiss
Wikipedia works with no key (needs `aiohttp`). Everything degrades gracefully.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gradio as gr

# ── cached model singletons (loaded on first message) ────────────────────────
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

    spec = importlib.util.spec_from_file_location("phase5", _HERE / "phase5_deberta.py")
    _P5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_P5)

    _DEVICE = _P5.detect_device(os.environ.get("MODEL_DEVICE"))
    tok_path = _HERE / "models" / "deberta_tokenizer"
    wt_path  = _HERE / "models" / "deberta_best.pt"
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
    print(f"[gradio] model loaded on {_DEVICE}")


def _score(text: str, speaker: str = "", context: str = "unknown") -> dict:
    _load_once()
    from credibility_detector_phases123 import (
        normalise_context, get_context_prior, extract_all_features)
    ctx   = normalise_context(context)
    prior = float(get_context_prior(ctx))
    fd = extract_all_features(text, ctx)
    fd["context_credibility_prior"] = prior
    fd["token_length_approx"] = len(text) / 4.0
    feats = [float(fd.get(c, 0.0)) for c in _P5.FEAT_COLS]
    return _P5.predict_with_uncertainty(
        _MODEL, _TOK, text, speaker, ctx, prior, feats, device=_DEVICE)


async def _retrieve(text: str):
    """Best-effort source retrieval via the agent (Wikipedia/FactCheck/FAISS/News)."""
    try:
        from agent.source_retrieval import retrieve_sources
        return await retrieve_sources(text, k_each=4)
    except Exception as e:
        print(f"[gradio] retrieval unavailable: {e}")
        return None


# ── formatting ────────────────────────────────────────────────────────────────

def _verdict(score: float) -> tuple[str, str]:
    if score < 0.35:
        return "🔴", "Likely False"
    if score < 0.65:
        return "🟡", "Uncertain / Mixed"
    return "🟢", "Likely Credible"


def _meter(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def _format_sources(bundle, low_credibility: bool) -> str:
    if bundle is None or not getattr(bundle, "sources", None):
        return ("\n\n_No external sources retrieved (offline, or rate-limited). "
                "Set `GOOGLE_FACTCHECK_API_KEY` / `NEWSAPI_KEY` or build the FAISS "
                "index for fact-check verdicts._")

    # Fact-check verdicts (with a numeric rating) first — those actually dispute
    # or support the claim — then general evidence, capped at 5.
    srcs = bundle.sources
    rated = [s for s in srcs if s.get("score") is not None]
    other = [s for s in srcs if s.get("score") is None]
    ordered = (rated + other)[:5]

    header = ("\n\n**🔎 Sources to check this claim against:**"
              if low_credibility else "\n\n**🔎 Related sources:**")
    lines = [header]
    for s in ordered:
        tag = s.get("source", "web").replace("_", " ").upper()
        title = (s.get("title") or "").strip() or "(untitled)"
        url = s.get("url") or ""
        rating = ""
        if s.get("score") is not None:
            fc = s["score"]
            flag = "disputes" if fc < 0.4 else "supports" if fc > 0.6 else "mixed on"
            rating = f" · fact-check **{flag}** this ({fc:.2f})"
        link = f" — [link]({url})" if url else ""
        lines.append(f"- **[{tag}]** {title}{rating}{link}")
    return "\n".join(lines)


async def respond(message: str, history):
    msg = (message or "").strip()
    if len(msg) < 12:
        return ("👋 Paste a **news statement or article** (a full sentence or more) "
                "and I'll rate how credible it is and point you to sources to check it.")

    # Score the core claim (long articles are truncated to the model's window).
    claim = msg if len(msg) <= 1200 else msg[:1200]
    r = _score(claim)
    score, lo, hi = r["mean"], r["lower"], r["upper"]
    emoji, label = _verdict(score)

    reply = [
        f"{emoji} **{label} — credibility {score:.2f} / 1.00**",
        f"`{_meter(score)}`  (90% CI {lo:.2f}–{hi:.2f})",
    ]
    if score < 0.35:
        reply.append("\nThis reads as **low-credibility** — the language and "
                     "framing pattern-match claims fact-checkers rate false.")
    elif score < 0.65:
        reply.append("\nThis is **uncertain** — treat it as unverified until "
                     "corroborated by the sources below.")
    else:
        reply.append("\nThis reads as **plausible/credible**, but still worth "
                     "confirming against the sources below.")

    bundle = await _retrieve(claim)
    reply.append(_format_sources(bundle, low_credibility=score < 0.5))
    reply.append("\n\n_ℹ️ The score reflects how fact-checkers would likely rate "
                 "this claim, not absolute truth. Verify important claims yourself._")
    return "\n".join(reply)


# ── UI ────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "Vaccines contain microchips that let the government track people.",
    "The unemployment rate fell to a record low last month.",
    "Scientists have confirmed that drinking coffee cures cancer.",
    "The senate passed the infrastructure bill with bipartisan support.",
    "5G towers are spreading the virus across major cities.",
]


def build_ui() -> gr.ChatInterface:
    # Gradio 6 ChatInterface always uses the "messages" format and has no `type`
    # or `theme` kwarg (theme is set on launch/Blocks instead).
    return gr.ChatInterface(
        fn=respond,
        title="🔍 Fake News & Source Credibility Detector",
        description=(
            "Paste a **news statement or article**. The bot returns a credibility "
            "score (0 = false, 1 = credible), a verdict with a confidence interval, "
            "and sources to check the claim against."
        ),
        examples=EXAMPLES,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Credibility chatbot demo")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public Gradio link")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (overrides MODEL_DEVICE)")
    args = ap.parse_args()

    if args.device:
        os.environ["MODEL_DEVICE"] = args.device
    os.environ.setdefault("MODEL_DEVICE", "cpu")   # MPS NaNs for DeBERTa on Mac

    print(f"Starting credibility chatbot on port {args.port} "
          f"(device={os.environ['MODEL_DEVICE']})")
    build_ui().launch(server_port=args.port, share=args.share)
