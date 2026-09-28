#!/usr/bin/env python
"""
Rigorous temporal-drift experiment (Colab GPU).

Trains the SAME model, SAME recipe, on two splits of the SAME LIAR-2 rows:
  • TEMPORAL: train on claims dated ≤ cutoff, test on claims dated > cutoff
  • RANDOM:   matched-size shuffled split spanning all years (the usual setup)

    temporal-test MAE − random-test MAE  =  the temporal-drift penalty

A positive gap is the honest number a random split hides: how much worse the
model really is on genuinely FUTURE claims. LIAR-2 only (the sole dated source),
so the dataset is held constant and the split type is the only variable.

Design: reuses production featurization and training with zero reimplementation —
it reads the Phase 1–3 featurized data/*.csv, joins the `date`→`year` back from
LIAR-2 on HF, re-splits, and calls deberta_model.train() for each split.

Run AFTER Phase 1–3 has produced data/{train,val,test}.csv:
    python temporal_retrain.py --device cuda --cutoff 2019
    python temporal_retrain.py --device cuda --cutoff 2019 --epochs 2   # quicker
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
DATA = _HERE / "data"
MODELS = _HERE / "models"
_BAK = ".bak_temporal"


def _year(s) -> int | None:
    m = re.search(r"(?:19|20)\d{2}", str(s))
    return int(m.group(0)) if m else None


def _norm(t) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())


def load_featurized_liar2_with_year() -> pd.DataFrame:
    """Featurized LIAR-2 rows (from Phase 1–3) with `year` joined back by text."""
    parts = [pd.read_csv(DATA / f) for f in ("train.csv", "val.csv", "test.csv")
             if (DATA / f).exists()]
    if not parts:
        sys.exit("ERROR: run Phase 1–3 first (need featurized data/train.csv etc.)")
    df = pd.concat(parts, ignore_index=True)
    if "dataset" in df.columns:
        df = df[df["dataset"] == "liar2"].copy()
    if "text" not in df.columns or "credibility_score" not in df.columns:
        sys.exit("ERROR: data/*.csv lack featurized columns — run Phase 2.")

    from datasets import load_dataset
    ymap: dict[str, int] = {}
    for sp in ("train", "validation", "test"):
        ds = load_dataset("chengxuphd/liar2", split=sp)
        for st, dt in zip(ds["statement"], ds["date"]):
            y = _year(dt)
            if y:
                ymap[_norm(st)] = y

    df["year"] = df["text"].map(lambda t: ymap.get(_norm(t)))
    rate = df["year"].notna().mean()
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    print(f"  featurized LIAR-2 rows with a matched year: {len(df):,} ({rate:.0%} joined)")
    if len(df) < 500:
        sys.exit("ERROR: too few rows matched a year — check the text join.")
    return df


def _carve_val(train_df: pd.DataFrame, frac: float = 0.1, seed: int = 42):
    val = train_df.sample(frac=frac, random_state=seed)
    tr = train_df.drop(val.index)
    return tr.reset_index(drop=True), val.reset_index(drop=True)


def _write_splits(train_df, val_df, test_df) -> None:
    for f in ("train.csv", "val.csv", "test.csv"):
        src = DATA / f
        if src.exists() and not (DATA / (f + _BAK)).exists():
            shutil.copy(src, DATA / (f + _BAK))
    train_df.to_csv(DATA / "train.csv", index=False)
    val_df.to_csv(DATA / "val.csv", index=False)
    test_df.to_csv(DATA / "test.csv", index=False)


def _restore_splits() -> None:
    for f in ("train.csv", "val.csv", "test.csv"):
        bak = DATA / (f + _BAK)
        if bak.exists():
            shutil.move(str(bak), str(DATA / f))


def run_experiment(name, train_df, val_df, test_df, device, epochs) -> dict:
    import deberta_model as D
    print(f"\n{'#'*60}\n#  Training [{name}]  "
          f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}\n{'#'*60}")
    _write_splits(train_df, val_df, test_df)
    try:
        res = D.train(device_str=device, epochs=epochs)
    finally:
        _restore_splits()
    (MODELS / f"temporal_{name}_results.json").write_text(
        json.dumps(res, indent=2, default=str))
    return res["test"]


def main():
    ap = argparse.ArgumentParser(description="Temporal vs random retrain")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cutoff", type=int, default=2019)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs (default = production EPOCHS)")
    args = ap.parse_args()

    print("Loading featurized LIAR-2 with dates…")
    df = load_featurized_liar2_with_year()
    past = df[df["year"] <= args.cutoff]
    fut = df[df["year"] > args.cutoff]
    print(f"  train-era ≤{args.cutoff}: {len(past):,}   future >{args.cutoff}: {len(fut):,}")
    if len(fut) < 100 or len(past) < 100:
        sys.exit("ERROR: cutoff leaves too few rows on one side — pick another.")

    # TEMPORAL split: train on the past (carve a val), test on the future.
    t_train, t_val = _carve_val(past)
    t_test = fut.reset_index(drop=True)

    # RANDOM split: identical sizes, shuffled across all years — the fair baseline.
    shuf = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    r_test = shuf.iloc[: len(t_test)]
    r_train, r_val = _carve_val(shuf.iloc[len(t_test):])

    try:
        tm = run_experiment("temporal", t_train, t_val, t_test, args.device, args.epochs)
        rm = run_experiment("random", r_train, r_val, r_test, args.device, args.epochs)
    finally:
        _restore_splits()  # belt-and-suspenders: never leave swapped data behind

    gap = tm["MAE"] - rm["MAE"]
    print(f"\n{'='*60}\n  TEMPORAL-DRIFT PENALTY — LIAR-2, cutoff {args.cutoff}\n{'='*60}")
    print(f"  Random-split   test MAE : {rm['MAE']:.4f}   (train & test span all years)")
    print(f"  Temporal-split test MAE : {tm['MAE']:.4f}   (train ≤{args.cutoff}, test >{args.cutoff})")
    print(f"  DRIFT PENALTY (temporal − random): {gap:+.4f}"
          f"  → {'model is worse on genuine future claims' if gap > 0 else 'no penalty detected'}")
    print("  Quote this gap: it's the temporal drift a random split hides.")
    print("\n  ⚠️ This retrains the working model twice and overwrites "
          "models/deberta_best.pt — re-run Phase 5 to restore the production model.")


if __name__ == "__main__":
    main()
