"""
Phase 12 — Gradio Demo

Interactive UI for the credibility detector.
Calls the FastAPI /assess endpoint (must be running) or
falls back to direct model inference if the API is unreachable.

Run:
    # Option A: with API running
    API_URL=http://localhost:8000 python gradio_app.py

    # Option B: standalone (loads model directly)
    python gradio_app.py --standalone

Install:
    pip install gradio requests
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gradio as gr

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")

CONTEXT_OPTIONS = [
    "unknown", "a speech", "a TV interview", "a campaign rally",
    "a press release", "a Twitter post", "a Facebook post", "a debate",
    "an ad", "a news conference", "a radio interview", "a campaign website",
    "an op-ed", "a town hall meeting", "a newsletter", "a stump speech",
    "a WhatsApp forward", "a viral image", "a blog post", "factual claim",
    "online media", "a social media post",
]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

def _score_via_api(statement: str, speaker: str, context: str) -> dict:
    import requests
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = requests.post(
        f"{API_URL}/assess",
        json={"statement": statement, "speaker": speaker,
               "context": context, "use_llm": True},
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _score_standalone(statement: str, speaker: str, context: str) -> dict:
    """Direct model inference — no API required."""
    spec = importlib.util.spec_from_file_location(
        "phase5", _HERE / "phase5_deberta.py")
    p5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p5)

    from credibility_detector_phases123 import (
        normalise_context, get_context_prior, extract_all_features)

    context_slot = normalise_context(context)
    prior        = float(get_context_prior(context_slot))
    feats_d      = extract_all_features(statement, context_slot)
    feats        = [feats_d.get(c, 0.0) for c in p5.FEAT_COLS]

    import torch
    from transformers import AutoTokenizer
    tok_path = _HERE / "models" / "deberta_tokenizer"
    wt_path  = _HERE / "models" / "deberta_best.pt"

    if not tok_path.exists():
        return {"score": prior, "lower_90ci": prior - 0.15,
                "upper_90ci": prior + 0.15, "verdict": "Unverified / Mixed",
                "explanation": "DeBERTa weights not found.", "sources": [],
                "model_used": "prior-only"}

    device    = p5.detect_device()
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)
    model     = p5.DeBERTaCredibilityModel()
    ckpt      = torch.load(wt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(device)

    result = p5.predict_with_uncertainty(
        model, tokenizer, statement, speaker, context_slot, prior, feats,
        device=device)

    score   = result["mean"]
    verdict = ("Likely False"        if score < 0.35 else
               "Unverified / Mixed"  if score < 0.65 else
               "Likely True")
    return {
        "score":       result["mean"],
        "lower_90ci":  result["lower"],
        "upper_90ci":  result["upper"],
        "verdict":     verdict,
        "explanation": None,
        "sources":     [],
        "model_used":  "DeBERTa-standalone",
        "elapsed_ms":  0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLER
# ─────────────────────────────────────────────────────────────────────────────

_USE_API = True  # set to False with --standalone


def assess(statement: str, speaker: str, context: str):
    if not statement or len(statement.strip()) < 10:
        return ("⚠️ Please enter a claim of at least 10 characters.",
                "", "", "", "", "")

    try:
        if _USE_API:
            data = _score_via_api(statement.strip(), speaker.strip(), context)
        else:
            data = _score_standalone(statement.strip(), speaker.strip(), context)
    except Exception as e:
        return (f"❌ Error: {e}", "", "", "", "", "")

    score   = data.get("score", 0.5)
    lower   = data.get("lower_90ci", 0.0)
    upper   = data.get("upper_90ci", 1.0)
    verdict = data.get("verdict", "Unknown")
    explain = data.get("explanation") or ""
    sources = data.get("sources", [])

    # Score gauge text
    emoji = "✅" if score >= 0.65 else "⚠️" if score >= 0.35 else "❌"
    score_text  = f"{emoji}  **{verdict}**\n\n"
    score_text += f"Credibility score: **{score:.3f}** / 1.000\n"
    score_text += f"90% CI: [{lower:.3f} – {upper:.3f}]\n"
    score_text += f"Model: {data.get('model_used', 'N/A')}"

    ci_text = f"{lower:.3f} ↔ {upper:.3f}  (90% confidence interval)"

    # Context prior
    prior_text = f"Context prior: {data.get('context_prior_used', 0.5):.3f}"

    # Sources
    if sources:
        src_lines = []
        for s in sources[:5]:
            badge = f"[{s.get('source','?').upper()}]"
            score_badge = (f" (verdict: {s['score']:.2f})"
                           if s.get("score") is not None else "")
            url   = s.get("url", "")
            link  = f" → {url}" if url else ""
            src_lines.append(f"**{badge}** {s.get('title','')}{score_badge}{link}")
        sources_text = "\n\n".join(src_lines)
    else:
        sources_text = "_No external sources retrieved._"

    return score_text, ci_text, prior_text, explain or "_No LLM explanation generated._", sources_text, data.get("elapsed_ms", 0)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
            title="Fake News & Source Credibility Detector",
            theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# 🔍 Fake News & Source Credibility Detector\n"
            "Enter a claim to assess its credibility (0.0 = false, 1.0 = true).\n"
            "The model uses DeBERTa-v3 + source retrieval + LLM justification."
        )

        with gr.Row():
            with gr.Column(scale=2):
                statement_input = gr.Textbox(
                    label="Claim / Statement",
                    placeholder="e.g. The unemployment rate fell to a record low.",
                    lines=3, max_lines=6)
                speaker_input = gr.Textbox(
                    label="Speaker (optional)",
                    placeholder="e.g. Barack Obama")
                context_input = gr.Dropdown(
                    label="Context / Venue",
                    choices=CONTEXT_OPTIONS,
                    value="unknown")
                submit_btn = gr.Button("Assess", variant="primary")

            with gr.Column(scale=3):
                score_out    = gr.Markdown(label="Verdict & Score")
                ci_out       = gr.Textbox(label="Confidence Interval", interactive=False)
                prior_out    = gr.Textbox(label="Context Prior", interactive=False)
                explain_out  = gr.Markdown(label="LLM Explanation")
                sources_out  = gr.Markdown(label="Retrieved Sources")
                elapsed_out  = gr.Number(label="Response time (ms)", interactive=False)

        submit_btn.click(
            fn=assess,
            inputs=[statement_input, speaker_input, context_input],
            outputs=[score_out, ci_out, prior_out,
                     explain_out, sources_out, elapsed_out],
        )

        gr.Examples(
            examples=[
                ["The unemployment rate fell to a record low.", "Joe Biden", "a speech"],
                ["Vaccines contain microchips.", "anonymous", "a WhatsApp forward"],
                ["The Earth is flat and governments are hiding it.", "", "online media"],
                ["Scientists confirm coffee cures cancer.", "", "a blog post"],
            ],
            inputs=[statement_input, speaker_input, context_input],
        )

        gr.Markdown(
            "---\n"
            "*Model: DeBERTa-v3-base fine-tuned on LIAR-2 + MultiFC + FEVER + AVeriTeC.*\n"
            "*Baseline MAE: 0.2867 | Target MAE: <0.24 on T4 GPU.*"
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gradio Demo")
    ap.add_argument("--standalone", action="store_true",
                    help="Load model directly (no API server needed)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio tunnel")
    args = ap.parse_args()

    _USE_API = not args.standalone  # type: ignore[assignment]

    demo = build_ui()
    print(f"Starting Gradio demo on port {args.port}")
    print(f"Backend: {'standalone (direct model)' if args.standalone else API_URL}")
    demo.launch(server_port=args.port, share=args.share)
