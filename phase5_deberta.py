"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 5 — DeBERTa-v3-base Fine-tuning                                 ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Trains a credibility regression model (0.0–1.0 score output).          ║
║                                                                          ║
║  Why DeBERTa over RoBERTa:                                               ║
║   • Disentangled attention: content and position treated separately.     ║
║     Handles "not true" vs "true" better than standard attention.         ║
║   • +2–4 F1 points over RoBERTa on NLU benchmarks at same compute cost. ║
║   • GLUE score 91.9 vs RoBERTa 88.5.                                    ║
║                                                                          ║
║  Why NOT frontier LLMs (Claude / GPT-4o) for scoring:                   ║
║   • Cost: GPT-4o = ~$5k/1M statements. DeBERTa = ~$4 (server power).   ║
║   • Latency: GPT-4o = 800ms–3s. DeBERTa = ~40ms.                       ║
║   • SHAP: impossible on closed API. DeBERTa = full gradient access.     ║
║   • Calibration: LLM scores are uncalibrated opinions. DeBERTa is       ║
║     MSE-trained against ground-truth labels.                             ║
║                                                                          ║
║  Context × sentiment handled at TWO levels:                              ║
║   1. Structured input prefix: [speaker: X | context: Y | prior: 0.40]   ║
║      DeBERTa attends to speaker, context, and prior alongside text.      ║
║   2. Engineered features appended to the [CLS] embedding before          ║
║      the regression head — including context_sentiment_risk and          ║
║      context_adjusted_sentiment (the rally boasting features).           ║
║                                                                          ║
║  MC Dropout for confidence intervals:                                    ║
║   • Runs 20 stochastic forward passes at inference                       ║
║   • Returns mean ± std → "0.32 ± 0.08" instead of just "0.32"           ║
║   • Honest about model uncertainty — no competing project does this      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Outputs:                                                                ║
║    ./models/deberta_best.pt          (model weights)                     ║
║    ./models/deberta_tokenizer/       (tokenizer)                         ║
║    ./models/deberta_results.json     (metrics vs baseline)               ║
║    ./eda_output/12_deberta_shap.html (token SHAP highlights)             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Hardware:                                                               ║
║   • GPU (CUDA): full training ~4hrs on T4 (Google Colab free)            ║
║   • Apple Silicon (MPS): replace DEVICE = "cuda" → "mps", ~15hrs        ║
║   • CPU: not recommended for training — use Colab T4                     ║
╚══════════════════════════════════════════════════════════════════════════╝

Install:
    pip install transformers torch peft accelerate scikit-learn shap
    pip install transformers-interpret mlflow

Run (Colab T4 recommended for training):
    python phase5_deberta.py --train
    python phase5_deberta.py --evaluate
    python phase5_deberta.py --compare-roberta     # ablation
    python phase5_deberta.py --predict "Obama tripled the debt" \\
        --speaker "Barack Obama" --context "a campaign rally"

MacBook Air (MPS):
    python phase5_deberta.py --train --device mps
"""

import argparse
import json
import os
import warnings
from pathlib import Path

# Must be set before torch / MPS is initialised.
# Makes unsupported MPS ops (e.g. DeBERTa gather) silently fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from utils.seed import set_seed
set_seed(42)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
DATA_DIR  = _HERE / "data"
MODEL_DIR = _HERE / "models"
EDA_DIR   = _HERE / "eda_output"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "microsoft/deberta-v3-base"
ROBERTA_MODEL = "roberta-base"   # used in ablation only

MAX_LEN    = 128    # 128 is 4× faster than 256 (attention is O(n²)); claims avg <60 tokens
BATCH_SIZE = 16
EPOCHS     = 8
LR         = 2e-5
WARMUP     = 0.15   # 15% warmup: smoother start prevents early instability
FREEZE_N   = 4      # freeze bottom 4 layers; layers 4-11 + head are trainable
DROPOUT    = 0.1
MC_PASSES  = 20     # forward passes for MC Dropout confidence intervals
ALPHA_LOSS = 0.7    # MSE weight in combined loss (0.7 MSE + 0.3 MAE)
PATIENCE   = 3      # early stopping: stop if val_MAE doesn't improve for N epochs

# ── Engineered feature columns (from Phase 2) ─────────────────────────────────
# These are appended to the [CLS] embedding before the regression head.
# Critically includes context×sentiment interaction features that handle
# the rally-boasting problem (high positive sentiment in low-accountability venue).
FEAT_COLS = [
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "pos_word_count", "neg_word_count",
    "pos_neg_ratio", "sentiment_extremity",
    "context_sentiment_risk",       # ctx_prior × extremity — rally boast detector
    "context_adjusted_sentiment",   # vader × ctx_prior — discounts persuasive contexts
    "persuasive_context_flag",      # binary: 1 if rally/ad/WhatsApp
    "context_credibility_prior",    # data-driven source accountability score
    "token_length_approx",
]


def detect_device(preferred: str = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS with PYTORCH_ENABLE_MPS_FALLBACK=1 (set above) works for DeBERTa:
    # unsupported gather ops fall back to CPU transparently.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CredibilityDataset(Dataset):
    """
    Tokenises statements for DeBERTa with a structured prefix.

    Input format:
        [speaker: Barack Obama | context: a campaign rally | prior: 0.40] statement text

    Why structured prefix:
    • DeBERTa's attention mechanism attends across the entire input.
      Prepending speaker, context, and the numerical context prior as text
      allows the model to jointly attend to WHO said it, WHERE it was said,
      and HOW accountable that venue is — alongside the statement text itself.
    • prior: 0.40 for a campaign rally vs prior: 0.78 for a press release
      explicitly signals the accountability level before the text is processed.
    • context_sentiment_risk is fed as a numerical engineered feature
      (not text) — the model sees both the text-based signal AND the number.

    Also returns engineered features as a separate tensor for the fusion head.
    """
    def __init__(self, df: pd.DataFrame, tokenizer,
                 max_len: int = MAX_LEN, use_prefix: bool = True):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.use_prefix = use_prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Build input text with structured prefix ───────────────────────
        text = str(row.get("text_deberta", row.get("text", "")))
        if self.use_prefix:
            parts = []
            speaker = str(row.get("speaker", "")).strip()
            context = str(row.get("context", "")).strip()
            prior   = row.get("context_credibility_prior", 0.5)
            if speaker: parts.append(f"speaker: {speaker}")
            if context: parts.append(f"context: {context}")
            parts.append(f"prior: {prior:.2f}")
            text = "[" + " | ".join(parts) + "] " + text

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Engineered features tensor ────────────────────────────────────
        feat_vals = []
        for col in FEAT_COLS:
            v = row.get(col, 0.0)
            feat_vals.append(float(v) if not pd.isna(v) else 0.0)
        features = torch.tensor(feat_vals, dtype=torch.float)

        # DeBERTa may or may not produce token_type_ids
        token_type_ids = enc.get(
            "token_type_ids",
            torch.zeros(self.max_len, dtype=torch.long)
        )
        if hasattr(token_type_ids, "squeeze"):
            token_type_ids = token_type_ids.squeeze(0)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids,
            "features":       features,
            "label":          torch.tensor(float(row["credibility_score"]),
                                           dtype=torch.float),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class DeBERTaCredibilityModel(nn.Module):
    """
    DeBERTa-v3-base with a fusion regression head.

    Architecture:
        Input: [PREFIX + STATEMENT TEXT]
            ↓
        DeBERTa encoder
            ↓
        [CLS] hidden state (768-dim)
            ↓
        Concat with engineered features (13-dim) → 781-dim
            ↓
        Linear(781 → 256) → GELU → Dropout
            ↓
        Linear(256 → 1) → Sigmoid → credibility score (0–1)

    Design decisions:
    • [CLS] token — represents the whole sequence. Standard for classification/regression.
    • Concatenate engineered features AFTER the encoder, not before.
      Before: they'd be lost during attention — the encoder ignores arbitrary vectors.
      After: the regression head can learn how much weight to give text vs features.
    • GELU activation — outperforms ReLU in transformer fine-tuning tasks.
    • Sigmoid output — constrains score to (0,1) matching our target range.
      Without sigmoid, model can predict negative values or >1.
    • Combined MSE + MAE loss — MSE penalises large errors heavily,
      MAE is robust to outliers. Weighted combination (0.7 MSE + 0.3 MAE) balances both.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 n_features: int = len(FEAT_COLS), dropout: float = DROPOUT):
        super().__init__()
        self.config  = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden       = self.config.hidden_size   # 768 for base models

        # Fusion head: [CLS] (768) + engineered features (13) → score (1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + n_features, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                features=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_emb = out.last_hidden_state[:, 0, :]  # [CLS] token, shape (B, 768)
        # MPS: DeBERTa relative attention can produce NaN via -inf softmax
        cls_emb = torch.nan_to_num(cls_emb, nan=0.0, posinf=1.0, neginf=-1.0)

        if features is not None:
            combined = torch.cat([cls_emb, features], dim=1)  # (B, 781)
        else:
            combined = cls_emb

        return self.head(combined).squeeze(-1)  # (B,)


def combined_loss(preds: torch.Tensor, targets: torch.Tensor,
                  alpha: float = ALPHA_LOSS) -> torch.Tensor:
    """alpha × MSE + (1-alpha) × MAE. alpha=0.7 weights MSE more heavily."""
    return alpha * nn.MSELoss()(preds, targets) + \
           (1 - alpha) * nn.L1Loss()(preds, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LAYER FREEZING
# ─────────────────────────────────────────────────────────────────────────────

def freeze_layers(model: DeBERTaCredibilityModel, n: int = FREEZE_N) -> None:
    """
    Freeze embedding layer + bottom N transformer layers.

    WHY freeze:
    DeBERTa was pre-trained on large text corpora — the bottom layers capture
    universal linguistic patterns (syntax, morphology) that transfer perfectly.
    Fine-tuning them risks catastrophic forgetting on small datasets (~23k rows).
    Layers 6–11 + the regression head remain trainable — they adapt to the
    credibility scoring task while preserving lower-level representations.

    DeBERTa-v3-base: 12 transformer layers (0–11).
    Freezing bottom 6 → layers 6–11 + head are trainable.
    """
    for p in model.encoder.embeddings.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.encoder.encoder.layer):
        if i < n:
            for p in layer.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Frozen bottom {n} layers. "
          f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss  = 0.0
    valid_steps = 0
    pbar = tqdm(loader, desc="  train", unit="batch", leave=False)
    for batch in pbar:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        ttype = batch["token_type_ids"].to(device)
        feats = batch["features"].to(device)
        lbls  = batch["label"].to(device)

        optimizer.zero_grad()
        preds = model(ids, mask, ttype, feats)
        loss  = combined_loss(preds, lbls)
        if torch.isnan(loss) or torch.isinf(loss):
            pbar.set_postfix(loss="NaN-skip")
            continue
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss  += loss.item()
        valid_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", ok=valid_steps)
    return total_loss / max(valid_steps, 1)


@torch.no_grad()
def evaluate_model(model, loader, device) -> dict:
    model.eval()
    preds_all, lbls_all = [], []
    total_loss = 0.0
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        ttype = batch["token_type_ids"].to(device)
        feats = batch["features"].to(device)
        lbls  = batch["label"].to(device)
        preds = model(ids, mask, ttype, feats)
        preds = torch.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
        total_loss += combined_loss(preds, lbls).item()
        preds_all.extend(preds.cpu().numpy())
        lbls_all.extend(lbls.cpu().numpy())

    p = np.array(preds_all); y = np.array(lbls_all)
    def bucket(a): return np.where(a<0.35,0,np.where(a<0.65,1,2))
    return {
        "loss":     round(total_loss / len(loader), 4),
        "MAE":      round(mean_absolute_error(y, p), 4),
        "Pearson_r": round(pearsonr(y, p)[0], 4),
        "Spearman_r":round(spearmanr(y, p)[0], 4),
        "Macro_F1": round(f1_score(bucket(y), bucket(p), average="macro"), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MC DROPOUT — CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_uncertainty(model, tokenizer, text: str,
                              speaker: str = "", context: str = "",
                              prior: float = 0.50,
                              features: list = None,
                              n_passes: int = MC_PASSES,
                              device: torch.device = torch.device("cpu")) -> dict:
    """
    MC Dropout: enable dropout at inference, run N forward passes,
    return mean ± std as confidence interval.

    Example output: {"mean": 0.32, "std": 0.08, "lower": 0.18, "upper": 0.46}
    This means: "0.32 ± 0.08 (90% CI: 0.18–0.46)"

    WHY confidence intervals:
    A system that says "score: 0.32" is less informative and less trustworthy
    than one that says "score: 0.32 ± 0.08". The latter tells the user:
    "we're reasonably confident this is false, but there's some uncertainty."
    For a fact-checking system, calibrated uncertainty is as important as
    the point estimate. No competing portfolio project does this.
    """
    # Build prefix
    parts = []
    if speaker: parts.append(f"speaker: {speaker}")
    if context: parts.append(f"context: {context}")
    parts.append(f"prior: {prior:.2f}")
    full_text = "[" + " | ".join(parts) + "] " + text

    enc = tokenizer(full_text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    # enc["token_type_ids"] from return_tensors="pt" already has shape (1, MAX_LEN).
    # Calling .unsqueeze(0) on it would produce (1, 1, MAX_LEN) → dimension error.
    ttype_raw = enc.get("token_type_ids", None)
    ttype = (ttype_raw if ttype_raw is not None
             else torch.zeros(1, MAX_LEN, dtype=torch.long)).to(device)

    feat_tensor = None
    if features is not None:
        feat_tensor = torch.tensor([features], dtype=torch.float).to(device)

    model.train()   # ← enables dropout (key MC Dropout step)
    scores = []
    with torch.no_grad():
        for _ in range(n_passes):
            s = model(ids, mask, ttype, feat_tensor)
            scores.append(s.item())
    model.eval()

    arr = np.array(scores)
    return {
        "mean":  round(float(arr.mean()), 4),
        "std":   round(float(arr.std()),  4),
        "lower": round(float(np.percentile(arr, 5)),  4),   # 90% CI
        "upper": round(float(np.percentile(arr, 95)), 4),
        "verdict": ("False" if arr.mean() < 0.35 else
                    "Half True" if arr.mean() < 0.65 else "True"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. SHAP TOKEN EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_shap_highlights(model, tokenizer, statement: str,
                         speaker: str = "", context: str = "",
                         prior: float = 0.50) -> str:
    """
    Token-level SHAP attributions using transformers-interpret.

    Returns HTML string with tokens highlighted:
    • Red tokens   → pushed score DOWN (credibility-lowering phrases)
    • Green tokens → pushed score UP   (credibility-raising phrases)

    The HTML is returned for embedding in the API response and Gradio UI.
    Example output would show:
      "50 percent" in red (pushed toward false)
      "peer reviewed" in green (pushed toward true)
      "[context: a campaign rally]" in red (low-accountability venue)

    WHY this is powerful for the rally boasting problem:
    If a politician says "best economy ever" at a rally, SHAP will show:
    • "best ever" → weak negative attribution (overused boast language)
    • "[context: a campaign rally]" → negative attribution
    • "[prior: 0.40]" → negative attribution (low accountability)
    The model explains WHY it's sceptical, not just that it is.
    """
    try:
        from transformers_interpret import SequenceClassificationExplainer

        # Wrap regression model as a 3-class classifier for SHAP
        class _Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, input_ids, attention_mask):
                # No features in SHAP wrapper (attribution on text only)
                out = self.m.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                # Use only first 768 dims (no engineered features for SHAP)
                dummy = torch.zeros(cls.shape[0], len(FEAT_COLS)).to(cls.device)
                score = self.m.head(torch.cat([cls, dummy], dim=1))
                false_l = 1.0 - score; true_l = score
                mixed_l = 1.0 - torch.abs(score - 0.5) * 2
                return torch.cat([false_l, mixed_l, true_l], dim=1)

        wrapper = _Wrapper(model)
        explainer = SequenceClassificationExplainer(
            wrapper, tokenizer,
            custom_labels=["false", "mixed", "true"]
        )
        parts = []
        if speaker: parts.append(f"speaker: {speaker}")
        if context: parts.append(f"context: {context}")
        parts.append(f"prior: {prior:.2f}")
        full = "[" + " | ".join(parts) + "] " + statement
        explainer(full)
        return explainer.visualize()   # returns HTML
    except ImportError:
        return "<p>Install transformers-interpret for SHAP token highlights.</p>"
    except Exception as e:
        return f"<p>SHAP error: {e}</p>"


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train(model_name: str = DEFAULT_MODEL,
          use_prefix: bool = True,
          device_str: str = None) -> dict:

    device = detect_device(device_str)
    print(f"\n{'='*60}")
    print(f"  PHASE 5 — Fine-tuning: {model_name}")
    print(f"  Device: {device} | Epochs: {EPOCHS} | LR: {LR} | Batch: {BATCH_SIZE}")
    print(f"  Max seq len: {MAX_LEN} | Frozen layers: {FREEZE_N}")
    print(f"  Prefix: {'enabled' if use_prefix else 'disabled'}")
    print(f"{'='*60}\n")

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")

    # Fill missing feature columns
    for df in [train_df, val_df, test_df]:
        for col in FEAT_COLS:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Tokenizer
    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Datasets & loaders
    train_ds = CredibilityDataset(train_df, tokenizer, use_prefix=use_prefix)
    val_ds   = CredibilityDataset(val_df,   tokenizer, use_prefix=use_prefix)
    test_ds  = CredibilityDataset(test_df,  tokenizer, use_prefix=use_prefix)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # Model
    print("Loading model…")
    model = DeBERTaCredibilityModel(model_name=model_name).to(device)
    freeze_layers(model, FREEZE_N)

    # Optimiser — separate LRs for encoder vs head
    # Lower LR for encoder: preserve pre-trained representations
    # Head LR capped at 5× encoder LR (was 10×) to reduce early instability
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LR},
        {"params": model.head.parameters(),    "lr": LR * 5},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # MLflow tracking
    run_name = f"{model_name.split('/')[-1]}" + \
               ("-prefix" if use_prefix else "-noprefix")
    mlflow.set_experiment("credibility-detector")

    best_mae   = float("inf")
    best_epoch = 0
    history    = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name, "epochs": EPOCHS, "lr": LR,
            "batch_size": BATCH_SIZE, "max_len": MAX_LEN,
            "freeze_layers": FREEZE_N, "use_prefix": use_prefix,
            "loss_alpha": ALPHA_LOSS,
        })

        no_improve = 0
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_epoch(model, train_loader, optimizer,
                                   scheduler, device)
            val_m   = evaluate_model(model, val_loader, device)

            row = {"epoch": epoch, "train_loss": round(tr_loss, 4), **val_m}
            history.append(row)

            print(f"  Epoch {epoch}/{EPOCHS}  "
                  f"train_loss={tr_loss:.4f}  "
                  f"val_MAE={val_m['MAE']:.4f}  "
                  f"Pearson_r={val_m['Pearson_r']:.4f}  "
                  f"F1={val_m['Macro_F1']:.4f}")

            mlflow.log_metrics({
                "train_loss":  tr_loss,
                "val_MAE":     val_m["MAE"],
                "val_pearson": val_m["Pearson_r"],
                "val_F1":      val_m["Macro_F1"],
            }, step=epoch)

            if val_m["MAE"] < best_mae:
                best_mae   = val_m["MAE"]
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(),
                           MODEL_DIR / "deberta_best.pt")
                tokenizer.save_pretrained(
                    MODEL_DIR / "deberta_tokenizer")
                print(f"  ★ New best — saved (val MAE={best_mae:.4f})")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stop: no improvement for {PATIENCE} epochs.")
                    break

        print(f"\n  Best epoch: {best_epoch}  |  Best val MAE: {best_mae:.4f}")

        # Load best weights, evaluate on test
        model.load_state_dict(torch.load(
            MODEL_DIR / "deberta_best.pt", map_location=device))
        test_m = evaluate_model(model, test_loader, device)
        print(f"  Test MAE={test_m['MAE']:.4f}  "
              f"Pearson_r={test_m['Pearson_r']:.4f}  "
              f"F1={test_m['Macro_F1']:.4f}")

        # Load baseline for delta comparison
        baseline_mae = None
        bp = MODEL_DIR / "baseline_results.json"
        if bp.exists():
            baseline_mae = json.loads(bp.read_text())["benchmark_mae"]
            delta = baseline_mae - best_mae
            print(f"\n  Baseline val MAE : {baseline_mae:.4f}")
            print(f"  DeBERTa val MAE  : {best_mae:.4f}")
            print(f"  Improvement      : {delta:+.4f} "
                  f"({'✓ beats baseline' if delta > 0 else '✗ underperforms baseline'})")

        mlflow.log_metric("test_MAE", test_m["MAE"])
        mlflow.log_metric("best_val_MAE", best_mae)

    def _to_py(obj):
        """Recursively convert numpy scalar types to Python natives for JSON."""
        if isinstance(obj, dict):  return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_to_py(v) for v in obj]
        if isinstance(obj, float): return obj
        if hasattr(obj, "item"):   return obj.item()   # numpy/torch scalar
        return obj

    results = _to_py({
        "model":         model_name,
        "best_epoch":    best_epoch,
        "best_val_MAE":  float(best_mae),
        "test":          test_m,
        "baseline_MAE":  baseline_mae,
        "improvement":   round(float((baseline_mae or 0) - best_mae), 4),
        "history":       history,
    })
    (MODEL_DIR / "deberta_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n  Saved: models/deberta_best.pt")
    print(f"  Saved: models/deberta_tokenizer/")
    print(f"  Saved: models/deberta_results.json")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. ROBERTA ABLATION  (DeBERTa vs RoBERTa delta)
# ─────────────────────────────────────────────────────────────────────────────

def run_roberta_ablation(device_str: str = None) -> None:
    """
    Train both DeBERTa-v3-base and RoBERTa-base with identical settings.
    Reports the MAE delta — this is the quantitative evidence that
    DeBERTa's disentangled attention helps on fact-checking tasks.

    Expected result on LIAR-2: DeBERTa ~0.02–0.04 MAE better than RoBERTa.
    Include this table in your README.
    """
    print("\n  Running RoBERTa ablation…")
    deberta_r = train(model_name=DEFAULT_MODEL, device_str=device_str)
    roberta_r = train(model_name=ROBERTA_MODEL, device_str=device_str)

    deberta_mae = deberta_r["best_val_MAE"]
    roberta_mae = roberta_r["best_val_MAE"]
    delta = roberta_mae - deberta_mae

    print(f"\n{'─'*45}")
    print(f"  DeBERTa-v3-base val MAE : {deberta_mae:.4f}")
    print(f"  RoBERTa-base val MAE    : {roberta_mae:.4f}")
    print(f"  Delta (RoBERTa - DeBERTa): {delta:+.4f}")
    print(f"  {'DeBERTa wins' if delta > 0 else 'RoBERTa wins'}")
    print(f"{'─'*45}")
    print("  Interview talking point:")
    print("  'DeBERTa's disentangled attention handles negation")
    print("   and word-order better — critical for detecting")
    print(f"   misleading claims. Delta: {delta:+.4f} MAE.'")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SINGLE PREDICTION (inference)
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(statement: str, speaker: str = "",
                   context: str = "", device_str: str = None) -> None:
    """Load best checkpoint and predict with MC Dropout confidence interval."""
    device = detect_device(device_str)
    ckpt   = MODEL_DIR / "deberta_best.pt"
    tok_dir = MODEL_DIR / "deberta_tokenizer"
    if not ckpt.exists():
        print("No trained model found. Run: python phase5_deberta.py --train")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=False)
    model     = DeBERTaCredibilityModel().to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()

    # Context prior for the given context
    from credibility_detector_phases123 import get_context_prior
    prior = get_context_prior(context)

    result = predict_with_uncertainty(
        model, tokenizer, statement,
        speaker=speaker, context=context, prior=prior,
        device=device,
    )

    print(f"\n  Statement  : {statement}")
    print(f"  Speaker    : {speaker or 'unknown'}")
    print(f"  Context    : {context or 'unknown'}")
    print(f"  Ctx prior  : {prior:.2f}")
    print(f"\n  Score      : {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  90% CI     : [{result['lower']:.4f}, {result['upper']:.4f}]")
    print(f"  Verdict    : {result['verdict']}")

    # SHAP highlights
    shap_html = get_shap_highlights(model, tokenizer, statement,
                                     speaker, context, prior)
    html_path = EDA_DIR / "12_deberta_shap.html"
    html_path.write_text(f"<html><body>{shap_html}</body></html>")
    print(f"  SHAP saved : {html_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 5 — DeBERTa-v3 Fine-tuning")
    ap.add_argument("--train",           action="store_true")
    ap.add_argument("--evaluate",        action="store_true")
    ap.add_argument("--compare-roberta", action="store_true",
                    help="Run DeBERTa vs RoBERTa ablation")
    ap.add_argument("--no-prefix",       action="store_true",
                    help="Disable speaker+context prefix (ablation)")
    ap.add_argument("--predict",         type=str, default=None,
                    help="Statement to predict")
    ap.add_argument("--speaker",         type=str, default="")
    ap.add_argument("--context",         type=str, default="")
    ap.add_argument("--device",          type=str, default=None,
                    help="cuda / mps / cpu (auto-detected if omitted)")
    args = ap.parse_args()

    if args.compare_roberta:
        run_roberta_ablation(device_str=args.device)
    elif args.train:
        train(model_name=DEFAULT_MODEL,
              use_prefix=not args.no_prefix,
              device_str=args.device)
    elif args.evaluate:
        device = detect_device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR/"deberta_tokenizer"), use_fast=False)
        model = DeBERTaCredibilityModel().to(device)
        model.load_state_dict(torch.load(
            str(MODEL_DIR/"deberta_best.pt"), map_location=device))
        model.eval()
        test_df = pd.read_csv(DATA_DIR/"test.csv")
        for col in FEAT_COLS:
            if col not in test_df.columns: test_df[col] = 0.0
        test_ds = CredibilityDataset(test_df, tokenizer)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)
        m = evaluate_model(model, test_loader, device)
        print("\n  Test metrics:")
        for k, v in m.items():
            print(f"    {k}: {v}")
    elif args.predict:
        predict_single(args.predict, args.speaker,
                       args.context, args.device)
    else:
        print("Specify --train, --evaluate, --predict, or --compare-roberta")
        print("Example: python phase5_deberta.py --train")
        print("Example: python phase5_deberta.py --predict 'Taxes cut 50%' "
              "--speaker 'Donald Trump' --context 'a campaign rally'")


if __name__ == "__main__":
    main()
