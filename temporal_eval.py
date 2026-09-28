#!/usr/bin/env python
"""
Temporal-drift evaluation for the DeBERTa credibility scorer.

The main model is trained/tested on a RANDOM split, which hides temporal drift:
train and test both span 2012–2021, so the model "sees the future" during
training. This script measures the drift that the random split hides, three ways:

  --drift        (CPU, instant, NO model)  How much the DATA itself shifts across
                 a time cutoff: label-prior shift, subject-topic shift, and a PSI
                 on the credibility-score distribution. This is the drift the
                 model would face on future claims.

  --eval-model   (CPU, uses the trained model)  Runs the DEPLOYED scorer on
                 LIAR-2's held-out TEST rows (never trained on) and reports MAE
                 BY YEAR. A rising error toward recent years is a drift signal.
                 Caveat printed: the model saw all years in training, so this
                 shows era-difficulty, not clean forward-drift.

  --make-split   Writes a TEMPORAL split (train ≤ cutoff, test > cutoff) plus a
                 matched RANDOM split, so you can retrain on Colab and quote the
                 real number: temporal-test MAE − random-test MAE = the drift gap.

Data source: chengxuphd/liar2 (has a 'date' field, ~2012–2021, 100% coverage).

Usage:
    python temporal_eval.py --drift --cutoff 2019
    python temporal_eval.py --eval-model --sample 400
    python temporal_eval.py --make-split --cutoff 2019
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
DATA_DIR = _HERE / "data"


def _year(s) -> int | None:
    m = re.search(r"(?:19|20)\d{2}", str(s))
    return int(m.group(0)) if m else None


def load_liar2_with_dates(sample: int | None = None) -> pd.DataFrame:
    """LIAR-2 with statement, year, credibility score, and original split."""
    from datasets import load_dataset
    from data_pipeline import LIAR2_INT_TO_LABEL, LIAR2_SCORE

    frames = []
    for split in ("train", "validation", "test"):
        ds = load_dataset("chengxuphd/liar2", split=split)
        df = ds.to_pandas()[["statement", "date", "label", "subject", "context"]]
        df["orig_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["cred"] = df["label"].map(
        lambda i: LIAR2_SCORE.get(LIAR2_INT_TO_LABEL.get(int(i), None)))
    df["year"] = df["date"].map(_year)
    df = df.dropna(subset=["year", "cred", "statement"])
    df = df[df["statement"].str.strip().str.len() > 0]
    df["year"] = df["year"].astype(int)
    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
    return df


def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two [0,1] distributions.
    <0.1 negligible · 0.1–0.25 moderate · >0.25 significant shift."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    pa = np.histogram(a, edges)[0] / max(len(a), 1) + 1e-6
    pb = np.histogram(b, edges)[0] / max(len(b), 1) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def drift_report(df: pd.DataFrame, cutoff: int) -> None:
    past = df[df["year"] <= cutoff]
    fut = df[df["year"] > cutoff]
    print(f"\n{'='*64}\n  DATA DRIFT across cutoff {cutoff}  (no model needed)\n{'='*64}")
    print(f"  train-era (≤{cutoff}): {len(past):,} claims, {past['year'].min()}–{cutoff}")
    print(f"  future    (>{cutoff}): {len(fut):,} claims, {cutoff+1}–{fut['year'].max()}")
    if len(past) == 0 or len(fut) == 0:
        print("  ⚠️ one side is empty — pick a cutoff inside the data range."); return

    # 1. Label-prior shift (mean credibility of claims per era)
    print(f"\n  Mean credibility label:  ≤{cutoff} = {past['cred'].mean():.3f}"
          f"   >{cutoff} = {fut['cred'].mean():.3f}"
          f"   Δ = {fut['cred'].mean() - past['cred'].mean():+.3f}")

    # 2. PSI on the credibility-score distribution
    psi = _psi(past["cred"].values, fut["cred"].values)
    verdict = ("negligible" if psi < 0.1 else
               "MODERATE" if psi < 0.25 else "SIGNIFICANT")
    print(f"  PSI (label distribution): {psi:.3f}  → {verdict} shift")

    # 3. Topic shift — subjects that rise/fall most between eras
    def _top_subjects(frame):
        s = (frame["subject"].fillna("").str.split(r"[;,]")
             .explode().str.strip().replace("", np.nan).dropna())
        return s.value_counts(normalize=True)
    ps, fs = _top_subjects(past), _top_subjects(fut)
    common = ps.index.union(fs.index)
    shift = (fs.reindex(common).fillna(0) - ps.reindex(common).fillna(0))
    print(f"\n  Biggest topic shifts (share in >{cutoff} minus share in ≤{cutoff}):")
    for subj, d in shift.sort_values(ascending=False).head(4).items():
        print(f"    ▲ {subj:<22} {d:+.3f}")
    for subj, d in shift.sort_values().head(4).items():
        print(f"    ▼ {subj:<22} {d:+.3f}")
    print("\n  → This is the shift the model faces on future claims. A random\n"
          "    split never tests it, because it mixes these eras into train+test.")


def eval_model(df: pd.DataFrame, sample: int | None) -> None:
    print(f"\n{'='*64}\n  MODEL ERROR BY YEAR  (deployed scorer, held-out LIAR-2 test)\n{'='*64}")
    test = df[df["orig_split"] == "test"].copy()
    if sample:
        test = test.sample(min(sample, len(test)), random_state=42).reset_index(drop=True)
    print(f"  Scoring {len(test):,} held-out claims (the model never trained on these)…")

    import gradio_app as g
    preds = []
    for i, row in enumerate(test.itertuples(index=False), 1):
        ctx = (getattr(row, "context", "") or "unknown")
        preds.append(g._score(row.statement, context=ctx)["mean"])
        if i % 100 == 0:
            print(f"    …{i}/{len(test)}")
    test["pred"] = preds
    test["abs_err"] = (test["pred"] - test["cred"]).abs()

    from sklearn.metrics import mean_absolute_error
    overall = mean_absolute_error(test["cred"], test["pred"])
    print(f"\n  Overall MAE: {overall:.4f}\n")
    print(f"  {'year':>6} {'n':>6} {'MAE':>8}")
    print("  " + "-" * 22)
    for yr, grp in test.groupby("year"):
        print(f"  {yr:>6} {len(grp):>6} {grp['abs_err'].mean():>8.4f}")
    # trend: correlation of year with error
    if test["year"].nunique() > 2:
        r = np.corrcoef(test["year"], test["abs_err"])[0, 1]
        trend = ("error RISES for recent years (drift signal)" if r > 0.05 else
                 "error FALLS for recent years" if r < -0.05 else "no clear trend")
        print(f"\n  year↔error correlation: {r:+.3f}  → {trend}")
    print("\n  ⚠️ Caveat: the model was trained on a RANDOM split, so it saw all\n"
          "    years. This shows era-difficulty, not clean forward-drift. For that,\n"
          "    use --make-split and retrain (below).")


def make_split(df: pd.DataFrame, cutoff: int) -> None:
    past = df[df["year"] <= cutoff].copy()
    fut = df[df["year"] > cutoff].copy()
    DATA_DIR.mkdir(exist_ok=True)
    cols = ["statement", "cred", "year", "subject", "context"]
    past[cols].to_csv(DATA_DIR / f"liar2_temporal_train_le{cutoff}.csv", index=False)
    fut[cols].to_csv(DATA_DIR / f"liar2_temporal_test_gt{cutoff}.csv", index=False)
    # matched random split of the SAME rows, for a fair baseline
    shuf = df.sample(frac=1.0, random_state=42)
    n_test = len(fut)
    shuf.iloc[n_test:][cols].to_csv(DATA_DIR / "liar2_random_train.csv", index=False)
    shuf.iloc[:n_test][cols].to_csv(DATA_DIR / "liar2_random_test.csv", index=False)
    print(f"\n{'='*64}\n  WROTE TEMPORAL + RANDOM SPLITS (cutoff {cutoff})\n{'='*64}")
    print(f"  Temporal:  train ≤{cutoff} = {len(past):,}   test >{cutoff} = {len(fut):,}")
    print(f"  Random:    train = {len(shuf)-n_test:,}   test = {n_test:,} (same sizes)")
    print("\n  To get the real drift number, retrain the model on each TRAIN set and\n"
          "  compare TEST MAE (do this on Colab GPU):")
    print("    temporal-test MAE − random-test MAE  =  the temporal-drift penalty")
    print("  A positive gap = the model is worse on genuinely future claims than a\n"
          "  random split makes it look. That gap is the honest, quotable number.")


def main():
    ap = argparse.ArgumentParser(description="Temporal-drift evaluation")
    ap.add_argument("--drift", action="store_true", help="data-drift report (no model)")
    ap.add_argument("--eval-model", action="store_true", help="model MAE by year")
    ap.add_argument("--make-split", action="store_true", help="write temporal+random splits")
    ap.add_argument("--cutoff", type=int, default=2019, help="year cutoff (default 2019)")
    ap.add_argument("--sample", type=int, default=400, help="rows to score in --eval-model")
    args = ap.parse_args()

    if not (args.drift or args.eval_model or args.make_split):
        args.drift = True  # sensible default

    print("Loading LIAR-2 with dates…")
    df = load_liar2_with_dates()
    print(f"  {len(df):,} dated claims, {df['year'].min()}–{df['year'].max()}")

    if args.drift:
        drift_report(df, args.cutoff)
    if args.make_split:
        make_split(df, args.cutoff)
    if args.eval_model:
        eval_model(df, args.sample)


if __name__ == "__main__":
    main()
