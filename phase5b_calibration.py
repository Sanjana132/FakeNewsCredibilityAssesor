"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 5b — Calibration & Uncertainty Quantification                    ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Measures (not claims) calibration of the DeBERTa regression model.     ║
║                                                                          ║
║  Checks:                                                                 ║
║   1. Reliability diagram — are predicted scores ≈ actual means?         ║
║   2. ECE (Expected Calibration Error) — target < 0.05                   ║
║   3. MC Dropout 90% CI coverage — target ≈ 0.90                         ║
║   4. Temperature scaling — if ECE > 0.05 or coverage off by >5 pts      ║
║                                                                          ║
║  Outputs:                                                                ║
║    eda_output/16_reliability.png   (reliability diagram + ECE)           ║
║    eda_output/17_ci_coverage.png   (CI width distribution)               ║
║    models/calibration.json         (ECE, coverage, temperature T)        ║
╚══════════════════════════════════════════════════════════════════════════╝

Run after Phase 5 training:
    python phase5b_calibration.py
    python phase5b_calibration.py --apply-temp-scaling
    python phase5b_calibration.py --n-bins 15 --n-passes 30
"""

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

warnings.filterwarnings("ignore")

_HERE    = Path(__file__).resolve().parent
DATA_DIR  = _HERE / "data"
MODEL_DIR = _HERE / "models"
EDA_DIR   = _HERE / "eda_output"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(_HERE))
from utils.seed import set_seed
set_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PHASE 5 MODEL
# ─────────────────────────────────────────────────────────────────────────────

def _load_phase5():
    """Import phase5_deberta as a module without running main()."""
    spec = importlib.util.spec_from_file_location(
        "phase5_deberta", _HERE / "phase5_deberta.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_and_tokenizer(p5, device_str: str = None):
    device = p5.detect_device(device_str)
    tok_path = MODEL_DIR / "deberta_tokenizer"
    wt_path  = MODEL_DIR / "deberta_best.pt"

    if not tok_path.exists() or not wt_path.exists():
        raise FileNotFoundError(
            "models/deberta_best.pt or deberta_tokenizer/ not found. "
            "Run Phase 5 training first."
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)

    import torch
    model = p5.DeBERTaCredibilityModel()
    ckpt = torch.load(wt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────────────────────
# 2. COLLECT PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(p5, model, tokenizer, device, split: str = "test",
                        n_passes: int = 20, max_rows: int = None):
    """Run MC Dropout on the chosen split and return (y_true, y_mean, y_std)."""
    import torch
    from torch.utils.data import DataLoader

    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run Phases 1-2 first")
    df = pd.read_csv(path)
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)

    ds = p5.CredibilityDataset(df, tokenizer, use_prefix=True)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    feat_cols = p5.FEAT_COLS
    y_true, y_means, y_stds = [], [], []

    print(f"  Running {n_passes}×MC Dropout on {len(df):,} {split} rows…")
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        ttype = batch.get("token_type_ids",
                          torch.zeros_like(ids)).to(device)
        feats = batch["features"].to(device)
        lbls  = batch["label"].numpy()

        model.train()  # enable dropout
        passes = []
        with torch.no_grad():
            for _ in range(n_passes):
                p = model(ids, mask, ttype, feats)
                passes.append(torch.nan_to_num(p, nan=0.5).cpu().numpy())
        model.eval()

        passes = np.stack(passes, axis=0)  # (n_passes, B)
        y_true.extend(lbls)
        y_means.extend(passes.mean(axis=0).tolist())
        y_stds.extend(passes.std(axis=0).tolist())

    return np.array(y_true), np.array(y_means), np.array(y_stds)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALIBRATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray,
                n_bins: int = 10) -> tuple:
    """
    Expected Calibration Error for regression:
      bin predictions into n_bins equal-width buckets,
      compute |mean_prediction - mean_actual| per bin, weight by count.

    Returns (ECE, bin_stats_df).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    rows = []
    for lo, hi in zip(bin_lowers, bin_uppers):
        mask = (y_pred >= lo) & (y_pred < hi)
        if not mask.any():
            continue
        mean_pred   = y_pred[mask].mean()
        mean_actual = y_true[mask].mean()
        count       = mask.sum()
        rows.append({
            "bin_lower": lo, "bin_upper": hi,
            "bin_mid":   (lo + hi) / 2,
            "mean_pred": mean_pred, "mean_actual": mean_actual,
            "count":     count,
            "abs_diff":  abs(mean_pred - mean_actual),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return 0.0, df
    ece = float((df["count"] * df["abs_diff"]).sum() / y_pred.shape[0])
    return ece, df


def compute_ci_coverage(y_true: np.ndarray, y_mean: np.ndarray,
                        y_std: np.ndarray, z: float = 1.645) -> float:
    """
    Fraction of test rows where true label falls inside the 90% CI
    [mean - z*std, mean + z*std].  Target ≈ 0.90.
    """
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    covered = ((y_true >= lower) & (y_true <= upper)).mean()
    return float(covered)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

def fit_temperature(y_val_true: np.ndarray, y_val_pred: np.ndarray) -> float:
    """
    Find scalar T that minimises MSE on val set after logit → T-scale → sigmoid.
    Valid only for predictions in (0, 1).
    """
    eps = 1e-6
    y_val_pred_clipped = np.clip(y_val_pred, eps, 1 - eps)
    logits = logit(y_val_pred_clipped)

    def mse(T):
        scaled = expit(logits / T)
        return float(np.mean((scaled - y_val_true) ** 2))

    result = minimize_scalar(mse, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(y_pred: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-6
    logits = logit(np.clip(y_pred, eps, 1 - eps))
    return expit(logits / T)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability(bin_df: pd.DataFrame, ece: float,
                     bin_df_cal: pd.DataFrame = None, ece_cal: float = None,
                     n_bins: int = 10) -> None:
    fig, axes = plt.subplots(1, 2 if bin_df_cal is not None else 1,
                             figsize=(12 if bin_df_cal is not None else 6, 5))
    if bin_df_cal is None:
        axes = [axes]

    for ax, bdf, label, ece_val in zip(
            axes,
            [bin_df, bin_df_cal] if bin_df_cal is not None else [bin_df],
            ["Before calibration", "After temperature scaling"],
            [ece, ece_cal] if ece_cal is not None else [ece]):
        if bdf is None or bdf.empty:
            continue
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.bar(bdf["bin_mid"], bdf["abs_diff"], width=1/n_bins,
               alpha=0.4, color="#E24B4A", label="Gap")
        ax.step(bdf["bin_mid"], bdf["mean_actual"], where="mid",
                color="#185FA5", lw=2, label="Actual mean")
        ax.step(bdf["bin_mid"], bdf["mean_pred"], where="mid",
                color="#639922", lw=2, ls="--", label="Predicted mean")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted credibility score")
        ax.set_ylabel("Mean actual credibility")
        ax.set_title(f"{label}\nECE = {ece_val:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("Reliability Diagram — DeBERTa Credibility Regressor",
                 fontweight="bold")
    plt.tight_layout()
    out = EDA_DIR / "16_reliability.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {out.name}")


def plot_ci_coverage(y_true: np.ndarray, y_mean: np.ndarray,
                     y_std: np.ndarray, coverage: float) -> None:
    lower = y_mean - 1.645 * y_std
    upper = y_mean + 1.645 * y_std
    ci_width = upper - lower
    covered  = (y_true >= lower) & (y_true <= upper)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MC Dropout 90% CI Analysis", fontweight="bold")

    ax = axes[0]
    ax.hist(ci_width, bins=40, color="#378ADD", alpha=0.75, edgecolor="white")
    ax.axvline(ci_width.mean(), color="#E24B4A", ls="--", lw=1.5,
               label=f"Mean width = {ci_width.mean():.3f}")
    ax.set_xlabel("CI width (upper - lower)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of 90% CI widths")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.sort(y_mean)
    ax.scatter(y_mean[~covered], y_true[~covered],
               c="#E24B4A", s=8, alpha=0.5, label="Outside CI")
    ax.scatter(y_mean[covered],  y_true[covered],
               c="#639922", s=8, alpha=0.3, label="Inside CI")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Predicted score"); ax.set_ylabel("True score")
    ax.set_title(f"Coverage = {coverage:.3f}  (target ≥ 0.90)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = EDA_DIR / "17_ci_coverage.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 5b — Calibration")
    ap.add_argument("--device",           default=None,
                    help="cpu | cuda | mps")
    ap.add_argument("--n-bins",    type=int, default=10,
                    help="Number of calibration bins (default 10)")
    ap.add_argument("--n-passes",  type=int, default=20,
                    help="MC Dropout passes (default 20)")
    ap.add_argument("--max-rows",  type=int, default=None,
                    help="Cap test rows for speed testing")
    ap.add_argument("--apply-temp-scaling", action="store_true",
                    help="Fit temperature on val set and re-evaluate")
    args = ap.parse_args()

    print("=" * 60)
    print("  PHASE 5b — Calibration & Uncertainty Quantification")
    print("=" * 60)

    print("\nLoading Phase 5 module and model…")
    p5     = _load_phase5()
    model, tokenizer, device = load_model_and_tokenizer(p5, args.device)
    print(f"  Device: {device}")

    # ── Test set predictions ──────────────────────────────────────────────
    y_true, y_mean, y_std = collect_predictions(
        p5, model, tokenizer, device,
        split="test", n_passes=args.n_passes, max_rows=args.max_rows)

    ece, bin_df = compute_ece(y_true, y_mean, n_bins=args.n_bins)
    coverage    = compute_ci_coverage(y_true, y_mean, y_std)
    mae         = float(np.abs(y_true - y_mean).mean())

    print(f"\n  ── Test set metrics ─────────────────────────")
    print(f"  MAE (point estimate)  : {mae:.4f}")
    print(f"  ECE ({args.n_bins} bins)           : {ece:.4f}  "
          f"({'✓ OK' if ece <= 0.05 else '⚠ HIGH — consider temperature scaling'})")
    print(f"  90% CI coverage       : {coverage:.4f}  "
          f"({'✓ OK' if abs(coverage - 0.90) <= 0.05 else '⚠ OFF TARGET'})")

    # ── Temperature scaling (optional) ───────────────────────────────────
    T = 1.0
    ece_cal = None
    bin_df_cal = None

    if args.apply_temp_scaling or ece > 0.05 or abs(coverage - 0.90) > 0.05:
        print("\n  Fitting temperature scaling on val set…")
        y_val_true, y_val_mean, _ = collect_predictions(
            p5, model, tokenizer, device,
            split="val", n_passes=args.n_passes, max_rows=args.max_rows)
        T = fit_temperature(y_val_true, y_val_mean)
        print(f"  Temperature T = {T:.4f}")

        y_mean_cal   = apply_temperature(y_mean, T)
        ece_cal, bin_df_cal = compute_ece(y_true, y_mean_cal, n_bins=args.n_bins)
        coverage_cal = compute_ci_coverage(y_true, y_mean_cal, y_std)
        mae_cal      = float(np.abs(y_true - y_mean_cal).mean())

        print(f"\n  ── After temperature scaling (T={T:.3f}) ──────")
        print(f"  MAE (calibrated)      : {mae_cal:.4f}")
        print(f"  ECE ({args.n_bins} bins)           : {ece_cal:.4f}")
        print(f"  90% CI coverage       : {coverage_cal:.4f}")
    else:
        print("  Calibration looks good — no temperature scaling needed.")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n  Generating plots…")
    plot_reliability(bin_df, ece, bin_df_cal, ece_cal, n_bins=args.n_bins)
    plot_ci_coverage(y_true, y_mean, y_std, coverage)

    # ── Save results ──────────────────────────────────────────────────────
    cal_results = {
        "n_test":    int(y_true.shape[0]),
        "n_bins":    args.n_bins,
        "n_passes":  args.n_passes,
        "mae":       round(mae, 4),
        "ECE":       round(ece, 4),
        "ECE_target":           0.05,
        "ECE_pass":             bool(ece <= 0.05),
        "CI_coverage_90":       round(coverage, 4),
        "CI_coverage_target":   0.90,
        "CI_coverage_pass":     bool(abs(coverage - 0.90) <= 0.05),
        "temperature":          round(T, 4),
        "ECE_after_scaling":    round(ece_cal, 4) if ece_cal is not None else None,
    }
    out_path = MODEL_DIR / "calibration.json"
    out_path.write_text(json.dumps(cal_results, indent=2))
    print(f"\n  Saved: models/calibration.json")

    print("\n  Summary:")
    print(f"    ECE          : {ece:.4f}  (target < 0.05) "
          f"{'✓' if ece <= 0.05 else '✗'}")
    print(f"    90% coverage : {coverage:.4f}  (target ≈ 0.90) "
          f"{'✓' if abs(coverage-0.90) <= 0.05 else '✗'}")
    if T != 1.0:
        print(f"    Temperature T: {T:.4f}  (applied to production scorer)")
    print("\n✓ Phase 5b complete.")


if __name__ == "__main__":
    main()
