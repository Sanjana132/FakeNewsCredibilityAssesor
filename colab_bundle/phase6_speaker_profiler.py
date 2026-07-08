"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 6 — Speaker & Source Credibility Profiler                        ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Aggregates per-speaker credibility from the labelled dataset AND        ║
║  from DeBERTa predictions (if available) to build durable source         ║
║  profiles that persist beyond individual claims.                         ║
║                                                                          ║
║  WHY speaker profiles matter:                                            ║
║   • A single claim is ambiguous — a pattern of 200 claims is not.       ║
║   • Profiles catch repeat offenders without re-running DeBERTa          ║
║     on every new claim from the same speaker.                            ║
║   • Bayesian-shrunk scores prevent one outlier claim from tanking         ║
║     a speaker with 500 labelled statements.                              ║
║                                                                          ║
║  Outputs:                                                                ║
║    models/speaker_profiles.json   (per-speaker stats, used by Phase 8)  ║
║    eda_output/13_top_speakers.png (credibility bar chart, top/bottom 20) ║
║    eda_output/14_speaker_scatter.png (claims count vs mean score)        ║
║    eda_output/15_context_breakdown.png (score distribution per venue)    ║
╚══════════════════════════════════════════════════════════════════════════╝

Install:
    pip install pandas numpy matplotlib seaborn scipy

Run:
    python phase6_speaker_profiler.py
    python phase6_speaker_profiler.py --min-claims 5   # raise minimum threshold
    python phase6_speaker_profiler.py --with-predictions  # use DeBERTa scores if available
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

_HERE    = Path(__file__).resolve().parent
DATA_DIR  = _HERE / "data"
MODEL_DIR = _HERE / "models"
EDA_DIR   = _HERE / "eda_output"
EDA_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_PRIOR      = 0.5   # Bayesian shrinkage prior
MIN_CLAIMS_DEFAULT = 3    # speakers with fewer claims get heavy shrinkage


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_all_splits() -> pd.DataFrame:
    dfs = []
    for split in ("train", "val", "test"):
        p = DATA_DIR / f"{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run Phases 1-2 first")
        df = pd.read_csv(p)
        df["split"] = split
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined):,} rows "
          f"({combined['speaker'].nunique():,} unique speakers)")
    return combined


def attach_deberta_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Phase 5 produced a predictions file, merge those scores in.
    Falls back to label-based credibility_score if file is missing.
    """
    pred_path = MODEL_DIR / "deberta_test_preds.csv"
    if not pred_path.exists():
        print("  No DeBERTa prediction file found — using label scores.")
        df["score_to_use"] = df["credibility_score"]
        return df
    preds = pd.read_csv(pred_path)
    merged = df.merge(preds[["text", "deberta_score"]], on="text", how="left")
    merged["score_to_use"] = merged["deberta_score"].fillna(
        merged["credibility_score"])
    n_pred = merged["deberta_score"].notna().sum()
    print(f"  Merged DeBERTa scores for {n_pred:,} / {len(merged):,} rows.")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPEAKER PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_score(scores: pd.Series, global_mean: float,
                   kappa: float = 10.0) -> float:
    """
    Shrink speaker mean toward global_mean proportionally to claim count.
    kappa = effective prior sample size.  Large kappa → stronger shrinkage.

    shrunk = (n * mu_speaker + kappa * mu_global) / (n + kappa)

    A speaker with 2 claims gets a score ≈ 0.8×global + 0.2×observed.
    A speaker with 100 claims gets a score ≈ 0.09×global + 0.91×observed.
    """
    n   = len(scores)
    mu  = scores.mean()
    return (n * mu + kappa * global_mean) / (n + kappa)


def build_speaker_profiles(df: pd.DataFrame) -> pd.DataFrame:
    global_mean = df["score_to_use"].mean()
    print(f"  Global mean credibility: {global_mean:.4f}")

    records = []
    for speaker, grp in df.groupby("speaker"):
        if speaker in ("", "unknown", "anonymous", "anonymous_social_media"):
            continue
        scores  = grp["score_to_use"].dropna()
        if len(scores) == 0:
            continue

        # Core stats
        n       = len(scores)
        mean_s  = scores.mean()
        std_s   = scores.std() if n > 1 else 0.0
        bayes_s = bayesian_score(scores, global_mean)

        # Trend: slope of score vs claim index (positive = improving over time)
        if n >= 5:
            idx  = np.arange(n)
            trend = float(np.polyfit(idx, scores.values, 1)[0])
        else:
            trend = 0.0

        # Top subjects
        if "subject" in grp.columns:
            subjects = (grp["subject"].dropna()
                        .str.strip().value_counts().head(3).index.tolist())
        else:
            subjects = []

        # Most common context
        context_mode = (grp["context"].dropna().mode().iloc[0]
                        if "context" in grp.columns and len(grp) > 0
                        else "unknown")

        # Dataset origin
        sources = grp["dataset"].value_counts().to_dict() if "dataset" in grp.columns else {}

        # Verdict distribution
        verdict = {
            "false_pct":   round((scores < 0.35).mean() * 100, 1),
            "mixed_pct":   round(((scores >= 0.35) & (scores < 0.65)).mean() * 100, 1),
            "true_pct":    round((scores >= 0.65).mean() * 100, 1),
        }

        records.append({
            "speaker":        speaker,
            "n_claims":       n,
            "mean_score":     round(mean_s,  4),
            "bayes_score":    round(bayes_s, 4),
            "std_score":      round(std_s,   4),
            "trend":          round(trend,   6),
            "top_subjects":   subjects,
            "context_mode":   context_mode,
            "verdict":        verdict,
            "sources":        sources,
            "job":            grp["speaker_job"].dropna().mode().iloc[0]
                              if "speaker_job" in grp.columns
                              and grp["speaker_job"].notna().any() else "",
        })

    profiles = pd.DataFrame(records).sort_values("bayes_score")
    print(f"  Built {len(profiles):,} speaker profiles.")
    return profiles, global_mean


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {"false": "#E24B4A", "mixed": "#F5A623", "true": "#639922"}


def plot_top_bottom_speakers(profiles: pd.DataFrame, n: int = 20) -> None:
    """Horizontal bar chart: top N most credible and bottom N least credible."""
    qual = profiles[profiles["n_claims"] >= MIN_CLAIMS_DEFAULT].copy()
    top    = qual.nlargest(n, "bayes_score")
    bottom = qual.nsmallest(n, "bayes_score")
    combined = pd.concat([bottom, top])

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, sub, title, cmap_edge in [
        (axes[0], bottom, f"Least Credible (bottom {n})", "#E24B4A"),
        (axes[1], top,   f"Most Credible (top {n})",      "#639922"),
    ]:
        colors = [PALETTE["false"] if s < 0.35
                  else PALETTE["mixed"] if s < 0.65
                  else PALETTE["true"]
                  for s in sub["bayes_score"]]
        y = np.arange(len(sub))
        ax.barh(y, sub["bayes_score"], color=colors, edgecolor="white", alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r['speaker']} ({r['n_claims']})"
             for _, r in sub.iterrows()],
            fontsize=8)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#888", lw=0.8, ls="--")
        ax.set_xlabel("Bayesian credibility score")
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("Speaker Credibility Profiles (Bayesian-shrunk scores)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = EDA_DIR / "13_top_speakers.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {out.name}")


def plot_count_vs_score(profiles: pd.DataFrame) -> None:
    """Scatter: claim count vs bayes_score, annotate outliers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["false"] if s < 0.35
              else PALETTE["mixed"] if s < 0.65
              else PALETTE["true"]
              for s in profiles["bayes_score"]]
    ax.scatter(profiles["n_claims"], profiles["bayes_score"],
               c=colors, alpha=0.6, s=20, edgecolors="none")

    # Annotate top-5 largest and most extreme
    notable = pd.concat([
        profiles.nlargest(5, "n_claims"),
        profiles.nlargest(3, "bayes_score"),
        profiles.nsmallest(3, "bayes_score"),
    ]).drop_duplicates("speaker")
    for _, row in notable.iterrows():
        ax.annotate(row["speaker"], (row["n_claims"], row["bayes_score"]),
                    fontsize=6, alpha=0.8,
                    xytext=(4, 2), textcoords="offset points")

    ax.axhline(0.5, color="#888", lw=0.8, ls="--", label="neutral (0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of claims (log scale)")
    ax.set_ylabel("Bayesian credibility score")
    ax.set_title("Speaker claim volume vs credibility")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = EDA_DIR / "14_speaker_scatter.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {out.name}")


def plot_context_breakdown(df: pd.DataFrame) -> None:
    """Box plot of credibility score distribution per context venue."""
    if "context" not in df.columns:
        return
    top_contexts = (df["context"].value_counts().head(12).index.tolist())
    sub = df[df["context"].isin(top_contexts)].copy()
    order = (sub.groupby("context")["score_to_use"]
             .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=sub, x="context", y="score_to_use",
                order=order, palette="RdYlGn", ax=ax, linewidth=0.7,
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax.axhline(0.5, color="#555", lw=0.8, ls="--")
    ax.set_xlabel("Context / venue")
    ax.set_ylabel("Credibility score")
    ax.set_title("Credibility distribution by context venue\n"
                 "(lower = more false claims issued in this venue)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    out = EDA_DIR / "15_context_breakdown.png"
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAVE & PRINT
# ─────────────────────────────────────────────────────────────────────────────

def save_profiles(profiles: pd.DataFrame, global_mean: float,
                  min_claims: int) -> None:
    records = []
    for _, r in profiles.iterrows():
        records.append({
            "speaker":     r["speaker"],
            "n_claims":    int(r["n_claims"]),
            "mean_score":  float(r["mean_score"]),
            "bayes_score": float(r["bayes_score"]),
            "std_score":   float(r["std_score"]),
            "trend":       float(r["trend"]),
            "job":         r.get("job", ""),
            "top_subjects":r.get("top_subjects", []),
            "context_mode":r.get("context_mode", ""),
            "verdict":     r.get("verdict", {}),
            "sources":     {k: int(v) for k, v in r.get("sources", {}).items()},
        })

    out = {
        "global_mean":  round(float(global_mean), 4),
        "min_claims":   min_claims,
        "n_profiles":   len(records),
        "profiles":     records,
    }
    path = MODEL_DIR / "speaker_profiles.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved: models/speaker_profiles.json  ({len(records):,} profiles)")


def print_summary(profiles: pd.DataFrame, min_claims: int) -> None:
    qual = profiles[profiles["n_claims"] >= min_claims]
    print(f"\n  ── Top 10 least credible speakers (≥{min_claims} claims) ──")
    for _, r in qual.nsmallest(10, "bayes_score").iterrows():
        bar = "█" * int(r["bayes_score"] * 20)
        print(f"  {r['speaker']:<35} {r['bayes_score']:.3f}  {bar} "
              f"({r['n_claims']} claims)")

    print(f"\n  ── Top 10 most credible speakers (≥{min_claims} claims) ──")
    for _, r in qual.nlargest(10, "bayes_score").iterrows():
        bar = "█" * int(r["bayes_score"] * 20)
        print(f"  {r['speaker']:<35} {r['bayes_score']:.3f}  {bar} "
              f"({r['n_claims']} claims)")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 6 — Speaker Credibility Profiler")
    ap.add_argument("--min-claims",       type=int,  default=MIN_CLAIMS_DEFAULT,
                    help="Minimum claims for a speaker to appear in rankings")
    ap.add_argument("--with-predictions", action="store_true",
                    help="Use DeBERTa predicted scores instead of labels")
    args = ap.parse_args()

    print("=" * 60)
    print("  PHASE 6 — Speaker & Source Credibility Profiler")
    print("=" * 60)

    df = load_all_splits()
    if args.with_predictions:
        df = attach_deberta_predictions(df)
    else:
        df["score_to_use"] = df["credibility_score"]

    profiles, global_mean = build_speaker_profiles(df)

    print("\n  Generating visualisations…")
    plot_top_bottom_speakers(profiles, n=20)
    plot_count_vs_score(profiles)
    plot_context_breakdown(df)

    print_summary(profiles, args.min_claims)
    save_profiles(profiles, global_mean, args.min_claims)

    print("\n✓ Phase 6 complete. Next: python phase7_shap_explainer.py")


if __name__ == "__main__":
    main()
