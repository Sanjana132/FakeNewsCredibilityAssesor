"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 4 — TF-IDF Baseline Model                                        ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TF-IDF + Ridge regressor baseline.                                     ║
║  Sets the benchmark MAE that DeBERTa (Phase 6) must beat.               ║
║  Includes SHAP LinearExplainer to validate n-gram credibility signals.  ║
║                                                                          ║
║  Why TF-IDF not Word2Vec:                                                ║
║   • Maximally simple → clean comparison floor for DeBERTa delta         ║
║   • SHAP gives human-readable n-gram attributions (50% → false,         ║
║     peer reviewed → true) — Word2Vec dimensions are opaque              ║
║   • Word2Vec would be a weak middle-ground, neither simple nor powerful  ║
║                                                                          ║
║  Also tests: does adding context × sentiment features help TF-IDF?      ║
║  (ablation: TF-IDF text only vs TF-IDF + engineered features)           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Outputs:                                                                ║
║    ./models/baseline_tfidf.pkl    (vectorizer + model)                  ║
║    ./models/baseline_results.json (metrics)                              ║
║    ./eda_output/11_shap_tfidf.png (SHAP bar chart)                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Install:
    pip install scikit-learn shap pandas numpy matplotlib joblib

Run:
    python phase4_baseline.py
    python phase4_baseline.py --ablation   # text-only vs text+features
"""

import argparse
import json
import warnings
from pathlib import Path

from utils.seed import set_seed
set_seed(42)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib

warnings.filterwarnings("ignore")

# Paths resolved relative to this script file so it works from any cwd
_HERE      = Path(__file__).resolve().parent
DATA_DIR   = _HERE / "data"
MODEL_DIR  = _HERE / "models"
EDA_DIR    = _HERE / "eda_output"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Engineered feature columns from Phase 2 ──────────────────────────────────
# These are the 13 numerical features including context × sentiment interactions.
# The context_sentiment_risk and context_adjusted_sentiment features directly
# tackle the rally-boasting problem: high positive/negative sentiment in a
# low-accountability context (rally, WhatsApp) gets a penalised signal.
ENGINEERED_FEATURES = [
    # VADER (4)
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    # Hu & Liu opinion lexicon (2)
    "pos_word_count", "neg_word_count",
    # Derived sentiment (2)
    "pos_neg_ratio", "sentiment_extremity",
    # Context × sentiment interactions (3) — NEW, addresses rally boast problem
    "context_sentiment_risk",       # ctx_prior × extremity: low = emotional + low-accountability
    "context_adjusted_sentiment",   # vader_compound × ctx_prior: discounts rally boasts
    "persuasive_context_flag",      # 1 if rally / ad / WhatsApp / social media
    # Context and history (2)
    "context_credibility_prior",    # data-driven prior per context venue
    "token_length_approx",          # text length proxy
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run Phases 1 & 2 first")
    df = pd.read_csv(path)
    # Fill empty preprocessed text with raw text fallback
    df["text_tfidf"] = df["text_tfidf"].fillna("").astype(str)
    # Fill missing engineered features with column median or 0
    for col in ENGINEERED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(train_texts, val_texts, test_texts,
                        max_features: int = 50_000, ngram_range=(1, 3),
                        min_df: int = 2, max_df: float = 0.9):
    """
    Fit TF-IDF on train, transform val and test.

    Hyperparameters chosen (tuned for lower val MAE):
    • max_features=50k — large enough for political/scientific vocabulary
    • ngram_range=(1,3) — trigrams add "not enough info", "half of all",
      "no evidence that" — high-signal phrases for credibility
    • sublinear_tf=True — log-normalise term frequencies (standard for short text)
    • min_df=2 — ignore terms appearing only once (noise)
    • max_df=0.9 — drop terms in >90% of docs (near-stopwords that add no signal
      and otherwise soak up Ridge weight, nudging MAE up)
    """
    print(f"  Fitting TF-IDF (max={max_features:,}, ngram={ngram_range}, "
          f"min_df={min_df}, max_df={max_df})…")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
        strip_accents="unicode",
        analyzer="word",
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)
    print(f"  TF-IDF matrix: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    return vectorizer, X_train, X_val, X_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(X_train, y_train) -> RidgeCV:
    """
    RidgeCV finds the best regularisation alpha on training data via LOOCV.
    Ridge chosen over Lasso because:
    • TF-IDF features are correlated (synonyms appear together)
    • Ridge handles multicollinearity better than Lasso
    • L2 penalty distributes weight across correlated features
    """
    print("  Training RidgeCV regressor…")
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error")
    ridge_cv.fit(X_train, y_train)
    print(f"  Best alpha: {ridge_cv.alpha_:.4f}")
    return ridge_cv


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def bucket(arr):
    """Convert continuous scores to 3 classes for F1 evaluation."""
    return np.where(arr < 0.35, 0, np.where(arr < 0.65, 1, 2))


def evaluate(model, X, y_true, split_name: str) -> dict:
    """Compute MAE, RMSE, Pearson r, Spearman r, 3-class macro F1."""
    from sklearn.metrics import f1_score
    y_pred = np.clip(model.predict(X), 0.0, 1.0)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = mean_squared_error(y_true, y_pred) ** 0.5
    pr, _  = pearsonr(y_true, y_pred)
    sr, _  = spearmanr(y_true, y_pred)
    f1     = f1_score(bucket(np.array(y_true)),
                      bucket(y_pred), average="macro")
    metrics = {
        "split":       split_name,
        "MAE":         round(mae,  4),
        "RMSE":        round(rmse, 4),
        "Pearson_r":   round(pr,   4),
        "Spearman_r":  round(sr,   4),
        "Macro_F1_3class": round(f1, 4),
        "n":           len(y_true),
    }
    print(f"  [{split_name}] MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"r={pr:.4f}  F1={f1:.4f}")
    return metrics


def evaluate_per_dataset(model, X, df: pd.DataFrame) -> dict:
    """
    Break down MAE / F1 by source dataset. Shows whether the baseline's headline
    number is propped up by an easy corpus (FEVER) while it struggles on the hard
    political claims (LIAR-2) — the same slice DeBERTa is expected to win on.
    """
    from sklearn.metrics import f1_score
    if "dataset" not in df.columns:
        return {}
    y_pred = np.clip(model.predict(X), 0.0, 1.0)
    y_true = df["credibility_score"].values
    out = {}
    print("\n  Per-dataset test breakdown:")
    print(f"  {'dataset':<12} {'n':>7} {'MAE':>8} {'F1':>8}")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*8}")
    for ds in sorted(df["dataset"].unique()):
        mask = (df["dataset"] == ds).values
        if mask.sum() < 5:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        mae = float(mean_absolute_error(yt, yp))
        try:
            f1 = float(f1_score(bucket(yt), bucket(yp), average="macro"))
        except ValueError:
            f1 = float("nan")
        out[ds] = {"n": int(mask.sum()), "MAE": round(mae, 4),
                   "Macro_F1_3class": round(f1, 4)}
        print(f"  {ds:<12} {int(mask.sum()):>7,} {mae:>8.4f} {f1:>8.4f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def shap_analysis(model, X_train, vectorizer, feat_names_extra: list,
                  n_top: int = 20) -> None:
    """
    SHAP LinearExplainer for the Ridge model on TF-IDF features.

    Produces two outputs:
    1. Console: top N features with SHAP value direction and interpretation
    2. Plot: SHAP bar chart saved to eda_output/11_shap_tfidf.png

    WHY SHAP matters here:
    The bar chart should show patterns like:
      → "50 percent", "all immigrants", "tripled" push toward FALSE (low score)
      → "peer reviewed", "confirmed by", "official data" push toward TRUE (high score)
      → "campaign rally" context prior pushes toward FALSE
      → "context_sentiment_risk" high value pushes toward FALSE
    These are your sanity-check: the model learned meaningful credibility signals,
    not dataset artefacts.

    Context × sentiment features will appear in the chart alongside TF-IDF n-grams,
    showing their relative importance — this directly validates the interaction
    features we added to address the rally boasting problem.
    """
    print("\n  Computing SHAP values (LinearExplainer)…")
    # Sample for SHAP — materialising dense (n_rows × 50k) kills RAM on large sets
    MAX_SHAP_ROWS = 2_000
    rng = np.random.default_rng(42)
    if X_train.shape[0] > MAX_SHAP_ROWS:
        idx = rng.choice(X_train.shape[0], MAX_SHAP_ROWS, replace=False)
        X_shap = X_train[idx]
        print(f"  (Sampled {MAX_SHAP_ROWS:,} rows from {X_train.shape[0]:,} for SHAP)")
    else:
        X_shap = X_train
    explainer   = shap.LinearExplainer(model, X_shap,
                                        feature_perturbation="interventional")
    shap_values = explainer(X_shap)

    # Feature names: TF-IDF vocab + engineered feature names
    tfidf_names = vectorizer.get_feature_names_out().tolist()
    all_names   = tfidf_names + feat_names_extra
    shap_vals   = shap_values.values   # shape (n_samples, n_features)

    # Mean absolute SHAP value per feature — importance ranking
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-n_top:][::-1]

    print(f"\n  Top {n_top} features by mean |SHAP|:")
    print(f"  {'Feature':<35} {'Mean |SHAP|':>12}  Direction")
    print(f"  {'─'*35} {'─'*12}  {'─'*30}")
    for i in top_idx:
        name     = all_names[i] if i < len(all_names) else f"feat_{i}"
        mean_val = shap_vals[:, i].mean()
        direction = ("↑ toward TRUE" if mean_val > 0.01 else
                     "↓ toward FALSE" if mean_val < -0.01 else "neutral")
        print(f"  {name:<35} {mean_abs[i]:>12.5f}  {direction}")

    # ── SHAP bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    top_names = [all_names[i] if i < len(all_names) else f"feat_{i}"
                 for i in top_idx]
    top_vals  = [shap_vals[:, i].mean() for i in top_idx]
    colors    = ["#E24B4A" if v < 0 else "#639922" for v in top_vals]
    y_pos     = np.arange(len(top_names))
    ax.barh(y_pos, top_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.axvline(0, color="#2C2C2A", lw=0.8)
    ax.set_xlabel("Mean SHAP value\n(positive = toward TRUE, negative = toward FALSE)")
    ax.set_title(f"TF-IDF baseline — top {n_top} features by SHAP value\n"
                 "(red = credibility-lowering, green = credibility-raising)")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "11_shap_tfidf.png", bbox_inches="tight", dpi=130)
    plt.close()
    print(f"\n  Saved: 11_shap_tfidf.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ABLATION — text only vs text + engineered features
# ─────────────────────────────────────────────────────────────────────────────

def ablation_study(train_df, val_df, test_df) -> None:
    """
    Compare two variants:
    A) TF-IDF text only          (no engineered features)
    B) TF-IDF text + engineered  (VADER + context × sentiment interactions)

    The delta on val MAE tells you how much the context interaction features
    actually help over pure text — justifying the rally boasting analysis.
    """
    print("\n" + "─"*55)
    print("  ABLATION: text-only vs text + engineered features")
    print("─"*55)

    vectorizer, X_tr, X_va, X_te = build_tfidf_matrix(
        train_df["text_tfidf"], val_df["text_tfidf"], test_df["text_tfidf"]
    )
    y_tr = train_df["credibility_score"].values
    y_va = val_df["credibility_score"].values

    results = {}

    # Variant A: text only
    print("\n  A) Text only:")
    model_a = train_ridge(X_tr, y_tr)
    r_a = evaluate(model_a, X_va, y_va, "val_text_only")
    results["text_only"] = r_a

    # Variant B: text + engineered features
    print("\n  B) Text + engineered features:")
    scaler = StandardScaler()
    eng_tr = csr_matrix(scaler.fit_transform(
        train_df[ENGINEERED_FEATURES].fillna(0).values))
    eng_va = csr_matrix(scaler.transform(
        val_df[ENGINEERED_FEATURES].fillna(0).values))
    X_tr_combined = hstack([X_tr, eng_tr])
    X_va_combined = hstack([X_va, eng_va])
    model_b = train_ridge(X_tr_combined, y_tr)
    r_b = evaluate(model_b, X_va_combined, y_va, "val_text+features")
    results["text_plus_features"] = r_b

    delta_mae = r_a["MAE"] - r_b["MAE"]
    print(f"\n  MAE delta (text_only - text+features): {delta_mae:+.4f}")
    if delta_mae > 0.005:
        print("  → Engineered features HELP (+ive delta means lower MAE with features)")
    else:
        print("  → Minimal benefit from engineered features on this dataset")
    print("  Note: context×sentiment features matter more on the full dataset")
    print("  where persuasive-context statements are well-represented.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 4 — TF-IDF Baseline")
    ap.add_argument("--ablation", action="store_true",
                    help="Run ablation: text-only vs text+features")
    ap.add_argument("--no-shap",  action="store_true",
                    help="Skip SHAP (faster for large datasets)")
    ap.add_argument("--max-features", type=int, default=50_000,
                    help="TF-IDF vocabulary cap (default 50k)")
    ap.add_argument("--ngram-max",    type=int, default=3,
                    help="Max n-gram size, e.g. 3 → (1,3) (default 3)")
    args = ap.parse_args()

    print("=" * 60)
    print("  PHASE 4 — TF-IDF Baseline Model")
    print("=" * 60)

    # Load splits
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")
    print(f"  Loaded: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    y_train = train_df["credibility_score"].values
    y_val   = val_df["credibility_score"].values
    y_test  = test_df["credibility_score"].values

    # ── Build features ────────────────────────────────────────────────────
    vectorizer, X_tr, X_va, X_te = build_tfidf_matrix(
        train_df["text_tfidf"], val_df["text_tfidf"], test_df["text_tfidf"],
        max_features=args.max_features, ngram_range=(1, args.ngram_max),
    )

    # Append engineered features (scaled dense → sparse → hstack)
    print("  Appending engineered features…")
    scaler = StandardScaler()
    eng_tr = csr_matrix(scaler.fit_transform(
        train_df[ENGINEERED_FEATURES].fillna(0).values))
    eng_va = csr_matrix(scaler.transform(
        val_df[ENGINEERED_FEATURES].fillna(0).values))
    eng_te = csr_matrix(scaler.transform(
        test_df[ENGINEERED_FEATURES].fillna(0).values))

    X_tr_full = hstack([X_tr, eng_tr])
    X_va_full = hstack([X_va, eng_va])
    X_te_full = hstack([X_te, eng_te])
    print(f"  Combined matrix: {X_tr_full.shape[1]:,} features total "
          f"(TF-IDF + {len(ENGINEERED_FEATURES)} engineered)")

    # ── Train ──────────────────────────────────────────────────────────────
    model = train_ridge(X_tr_full, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n  Evaluation:")
    train_metrics = evaluate(model, X_tr_full, y_train, "train")
    val_metrics   = evaluate(model, X_va_full, y_val,   "val")
    test_metrics  = evaluate(model, X_te_full, y_test,  "test")

    # Per-dataset breakdown on the test split
    per_dataset = evaluate_per_dataset(model, X_te_full, test_df)

    results = {
        "model":      "TF-IDF + Ridge (with engineered features)",
        "alpha":      model.alpha_,
        "n_features": X_tr_full.shape[1],
        "train":      train_metrics,
        "val":        val_metrics,
        "test":       test_metrics,
        "per_dataset": per_dataset,
        "benchmark_mae": val_metrics["MAE"],  # DeBERTa must beat this
    }

    print(f"\n  ★ BENCHMARK VAL MAE = {val_metrics['MAE']:.4f}")
    print(f"    DeBERTa (Phase 6) must achieve MAE < {val_metrics['MAE']:.4f}")
    print(f"    Expected DeBERTa improvement: 0.04–0.08 MAE points")

    # ── SHAP ───────────────────────────────────────────────────────────────
    if not args.no_shap:
        shap_analysis(
            model, X_tr_full, vectorizer,
            feat_names_extra=ENGINEERED_FEATURES,
            n_top=20,
        )

    # ── Ablation ───────────────────────────────────────────────────────────
    if args.ablation:
        ablation_results = ablation_study(train_df, val_df, test_df)
        results["ablation"] = ablation_results

    # ── Save ───────────────────────────────────────────────────────────────
    joblib.dump({
        "vectorizer": vectorizer,
        "scaler":     scaler,
        "model":      model,
        "feature_cols": ENGINEERED_FEATURES,
    }, MODEL_DIR / "baseline_tfidf.pkl")

    (MODEL_DIR / "baseline_results.json").write_text(
        json.dumps(results, indent=2))

    print(f"\n  Saved: models/baseline_tfidf.pkl")
    print(f"  Saved: models/baseline_results.json")
    print("✓ Phase 4 complete. Next: python deberta_model.py")


if __name__ == "__main__":
    main()
