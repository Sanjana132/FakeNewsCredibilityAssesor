"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 6 — Context Encoder                                              ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Learns a 16-dim embedding for each of the 22 canonical CONTEXT_SLOTS.  ║
║  Trained end-to-end with DeBERTa (Phase 5) — the embeddings capture     ║
║  how credibility-relevant each venue is, independently of the priors.   ║
║                                                                          ║
║  Standalone use:                                                         ║
║    enc = ContextEncoder()                                                ║
║    emb = enc("a campaign rally")   # → tensor of shape (16,)            ║
║                                                                          ║
║  Also exports compute_interaction_features() so phase5 can call it      ║
║  without importing the whole encoder.                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

Run standalone:
    python context_encoder.py                # prints slot-embedding norms
    python context_encoder.py --visualise    # saves eda_output/18_ctx_embed.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE    = Path(__file__).resolve().parent
DATA_DIR  = _HERE / "data"
MODEL_DIR = _HERE / "models"
EDA_DIR   = _HERE / "eda_output"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_HERE))

# Keep in sync with data_pipeline.CONTEXT_SLOTS
CONTEXT_SLOTS = [
    "a speech", "a TV interview", "a campaign rally", "a press release",
    "a Twitter post", "a Facebook post", "a debate", "an ad",
    "a news conference", "a radio interview", "a campaign website",
    "an op-ed", "a town hall meeting", "a newsletter", "a stump speech",
    "a WhatsApp forward", "a viral image", "a blog post", "factual claim",
    "online media", "a social media post", "unknown",
]
SLOT_TO_IDX = {s: i for i, s in enumerate(CONTEXT_SLOTS)}
N_SLOTS     = len(CONTEXT_SLOTS)   # 22
EMBED_DIM   = 16


class ContextEncoder(nn.Module):
    """
    Learnable embedding for canonical context slots.

    During DeBERTa training (Phase 5) this module is instantiated and its
    parameters are included in the optimiser — so the embeddings adapt to
    the credibility-prediction task rather than being fixed priors.

    At inference, the embedding is concatenated to the DeBERTa [CLS] token
    before the regression head:

        fusion_input = [CLS (768) | ctx_embed (16) | feat (13)] → 797-dim
    """

    def __init__(self, n_slots: int = N_SLOTS, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(n_slots, embed_dim, padding_idx=N_SLOTS - 1)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def slot_to_idx(self, slot: str) -> int:
        return SLOT_TO_IDX.get(slot, SLOT_TO_IDX["unknown"])

    def forward(self, slot_ids: torch.Tensor) -> torch.Tensor:
        """slot_ids: (B,) int tensor → (B, embed_dim)"""
        return self.embed(slot_ids)

    def encode_string(self, slot: str,
                      device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Convenience: encode a single string slot → (embed_dim,) tensor."""
        idx = torch.tensor([self.slot_to_idx(slot)], dtype=torch.long).to(device)
        return self.forward(idx).squeeze(0)

    def save(self, path: Path = MODEL_DIR / "context_encoder.pt") -> None:
        torch.save(self.state_dict(), path)
        print(f"  Saved context encoder → {path}")

    @classmethod
    def load(cls, path: Path = MODEL_DIR / "context_encoder.pt",
             device: torch.device = torch.device("cpu")) -> "ContextEncoder":
        enc = cls()
        if path.exists():
            enc.load_state_dict(torch.load(path, map_location=device,
                                           weights_only=True))
        enc.to(device).eval()
        return enc


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_context_priors_from_cache() -> dict:
    cache = DATA_DIR / "context_priors.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        return data.get("priors", {}), data.get("global_mean", 0.5)
    return {}, 0.5


def compute_interaction_features(context_slot: str, vader_compound: float,
                                  vader_pos: float, vader_neg: float) -> dict:
    """
    Reproduce the 3 interaction features from Phase 2, but from a canonical
    slot string + VADER scores (so the API can call this without the full
    preprocessing pipeline).
    """
    priors, global_mean = get_context_priors_from_cache()
    prior = priors.get(context_slot, global_mean)
    extremity = abs(vader_pos - vader_neg)
    return {
        "context_credibility_prior":    round(prior, 4),
        "context_sentiment_risk":       round(prior * extremity, 4),
        "context_adjusted_sentiment":   round(vader_compound * prior, 4),
        "persuasive_context_flag":      int(prior < global_mean),
    }


def slot_ids_from_series(contexts, device: torch.device) -> torch.Tensor:
    """Convert a pandas Series / list of slot strings → (N,) int tensor."""
    ids = [SLOT_TO_IDX.get(str(c), SLOT_TO_IDX["unknown"]) for c in contexts]
    return torch.tensor(ids, dtype=torch.long).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualise_embeddings(enc: ContextEncoder) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    weights = enc.embed.weight.detach().numpy()  # (22, 16)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=60, c="#378ADD", alpha=0.8)
    for i, slot in enumerate(CONTEXT_SLOTS):
        ax.annotate(slot, (coords[i, 0], coords[i, 1]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")
    ax.set_title("Context slot embeddings (PCA 2D projection)\n"
                 "Clusters = similar credibility-signal venues",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.tight_layout()
    out = EDA_DIR / "18_ctx_embeddings.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 6 — Context Encoder")
    ap.add_argument("--visualise", action="store_true",
                    help="Plot PCA of slot embeddings")
    args = ap.parse_args()

    print("=" * 60)
    print("  Context Encoder — slot inventory")
    print("=" * 60)
    enc = ContextEncoder.load()

    priors, gm = get_context_priors_from_cache()
    print(f"\n  {'Slot':<30} {'Prior':>6}  {'Embed norm':>10}")
    print(f"  {'─'*30} {'─'*6}  {'─'*10}")
    for i, slot in enumerate(CONTEXT_SLOTS):
        idx  = torch.tensor([i])
        norm = enc(idx).norm().item()
        prior = priors.get(slot, gm)
        print(f"  {slot:<30} {prior:>6.3f}  {norm:>10.4f}")

    if args.visualise:
        print("\n  Generating embedding plot…")
        visualise_embeddings(enc)

    print("\n✓ Context encoder ready.")


if __name__ == "__main__":
    main()
