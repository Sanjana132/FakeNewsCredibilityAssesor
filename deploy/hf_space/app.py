"""
Hugging Face Spaces entry point.

Downloads the trained model artifacts from the HF Model Hub at startup, then
launches the Gradio credibility chatbot. The heavy weights live in a separate
model repo (they are too large for the Space git repo); set MODEL_REPO to point
at yours.
"""
import os
from pathlib import Path

os.environ.setdefault("MODEL_DEVICE", "cpu")   # Spaces free tier is CPU-only

_HERE   = Path(__file__).resolve().parent
MODELS  = _HERE / "models"
MODELS.mkdir(exist_ok=True)

# HF model repo holding deberta_best.pt, deberta_tokenizer/, speaker_profiles.json
MODEL_REPO = os.environ.get("MODEL_REPO", "SanjanaR132/credibility-detector-deberta")


def _provision() -> None:
    """Fetch model artifacts from the Hub (once per container start)."""
    if (MODELS / "deberta_best.pt").exists():
        return
    from huggingface_hub import snapshot_download
    print(f"[space] downloading model artifacts from {MODEL_REPO} …")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(MODELS),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),   # only needed if the repo is private
    )
    print("[space] model ready.")


_provision()

import gradio_app

demo = gradio_app.build_ui()
demo.queue()            # serialise requests on the single free-tier CPU
demo.launch()
