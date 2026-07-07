"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CREDIBILITY DETECTOR — PHASES 1, 2 & 3 (COMBINED)                     ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  PHASE 1 — Dataset loading (LIAR-2, MultiFC, FEVER, AVeriTeC)            ║
║            FakeNewsNet REMOVED. Context weights as FEATURE not label.    ║
║  PHASE 2 — Preprocessing + feature engineering                           ║
║            autocorrect REMOVED (DeBERTa path). TextBlob REMOVED.        ║
║            3 context×sentiment interaction features ADDED.               ║
║  PHASE 3 — EDA (descriptive stats + 10 visualisation plots)              ║
╚══════════════════════════════════════════════════════════════════════════╝

Install:
    pip install datasets pandas numpy scikit-learn nltk vaderSentiment
    pip install matplotlib seaborn scipy langdetect transformers
    python -m nltk.downloader stopwords punkt punkt_tab opinion_lexicon

Run full pipeline:
    python credibility_detector_phases123.py

Run individual phases:
    python credibility_detector_phases123.py --phase 1 --local
    python credibility_detector_phases123.py --phase 2 --relevance
    python credibility_detector_phases123.py --phase 3

Dev / quick run (uses local cache, small sample):
    python credibility_detector_phases123.py --local --sample 500
"""

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

from utils.seed import set_seed
set_seed(42)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# One-time NLTK downloads
for _pkg in ["stopwords","punkt","punkt_tab","opinion_lexicon"]:
    nltk.download(_pkg, quiet=True)

# ── Global objects (created once, reused across all functions) ─────────────
STOP_WORDS  = set(stopwords.words("english"))
STEMMER     = PorterStemmer()
VADER       = SentimentIntensityAnalyzer()
POS_WORDS   = set(opinion_lexicon.positive())
NEG_WORDS   = set(opinion_lexicon.negative())

_HERE    = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
EDA_DIR  = _HERE / "eda_output"

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CREDIT_COLS = [
    "true_counts","mostly_true_counts","half_true_counts",
    "mostly_false_counts","false_counts","pants_on_fire_counts",
]
SCHEMA_COLS = [
    "text","label_original","credibility_score",
    "speaker","speaker_job","subject","context","state_info",
    "justification","split","dataset",
] + CREDIT_COLS

LABEL_ORDER = [
    "pants-fire","false","barely-true","half-true","mostly-true","true",
    "pants on fire","mostly true","half true","mostly false",  # MultiFC variants
    "supports","refutes","not_enough_info",                   # FEVER
    "supported","refuted","conflicting_evidence","not_enough_evidence",  # AVeriTeC
]

# ── Context priors: DATA-DRIVEN, not hardcoded ──────────────────────────────
# IMPORTANT: No credibility weights are assigned here.
# Priors are computed from training labels in build_data_driven_priors()
# and cached to ./data/context_priors.json.
# At runtime, get_context_prior() reads the cache — all numbers come from data.

_CONTEXT_PRIORS_CACHE: dict = {}   # populated by load_or_build_priors()
_GLOBAL_MEAN_CACHE: float = 0.5   # updated when priors are built

def build_data_driven_priors(train_df: pd.DataFrame,
                              min_count: int = 5) -> tuple:
    """
    Compute mean credibility per context from training labels.
    Uses Bayesian shrinkage: contexts with few examples are pulled
    toward the global mean to avoid noisy estimates.

    shrinkage = n / (n + 10)
    prior = shrinkage × sample_mean + (1-shrinkage) × global_mean

    Returns (priors_dict, global_mean).
    All numbers come from YOUR data, not from the developer.
    """
    global _CONTEXT_PRIORS_CACHE, _GLOBAL_MEAN_CACHE
    global_mean = float(train_df["credibility_score"].mean())
    priors = {}
    if "context" in train_df.columns:
        # Normalise raw context strings to canonical CONTEXT_SLOTS before groupby.
        # Without this, raw values like "Press Release" never match "a press release"
        # and all priors collapse to global_mean — making them not data-driven.
        tmp_ctx = (train_df["context"]
                   .fillna("").astype(str).str.strip()
                   .replace({"nan": "", "None": "", "none": ""})
                   .apply(normalise_context))
        _tmp = pd.DataFrame({
            "context": tmp_ctx.values,
            "score":   train_df["credibility_score"].values,
        })
        grouped = _tmp.groupby("context")["score"].agg(["mean", "count"])
        for ctx, row in grouped.iterrows():
            if ctx not in set(CONTEXT_SLOTS): continue
            n = float(row["count"])
            shrinkage = n / (n + 10.0)
            priors[ctx] = round(shrinkage * row["mean"] + (1-shrinkage) * global_mean, 4)
    for slot in CONTEXT_SLOTS:
        if slot not in priors:
            priors[slot] = round(global_mean, 4)
    _CONTEXT_PRIORS_CACHE = priors
    _GLOBAL_MEAN_CACHE    = global_mean
    cache_path = DATA_DIR / "context_priors.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"priors": priors, "global_mean": global_mean}, indent=2))
    return priors, global_mean


def load_or_build_priors(train_df: pd.DataFrame = None) -> dict:
    """Load cached priors or build from train_df if cache missing."""
    global _CONTEXT_PRIORS_CACHE, _GLOBAL_MEAN_CACHE
    if _CONTEXT_PRIORS_CACHE:
        return _CONTEXT_PRIORS_CACHE
    cache_path = DATA_DIR / "context_priors.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        _CONTEXT_PRIORS_CACHE = data["priors"]
        _GLOBAL_MEAN_CACHE    = data.get("global_mean", 0.5)
        return _CONTEXT_PRIORS_CACHE
    if train_df is not None:
        priors, _ = build_data_driven_priors(train_df)
        return priors
    return {}   # truly no data available

CONTEXT_SLOTS = [
    "a speech", "a TV interview", "a campaign rally", "a press release",
    "a Twitter post", "a Facebook post", "a debate", "an ad",
    "a news conference", "a radio interview", "a campaign website",
    "an op-ed", "a town hall meeting", "a newsletter", "a stump speech",
    "a WhatsApp forward", "a viral image", "a blog post", "factual claim",
    "online media", "a social media post", "unknown",
]


def get_context_prior(context: str) -> float:
    """
    Look up the DATA-DRIVEN prior for a context string.
    Reads from _CONTEXT_PRIORS_CACHE (built from training labels).
    Falls back to _GLOBAL_MEAN_CACHE for unseen contexts.
    No hardcoded numbers.
    """
    priors = _CONTEXT_PRIORS_CACHE
    if not priors:
        cache_path = DATA_DIR / "context_priors.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            priors = data.get("priors", {})
    # Guard: NaN floats arrive when context column is missing in CSV rows
    if context is None or (isinstance(context, float) and np.isnan(context)):
        return _GLOBAL_MEAN_CACHE
    ctx = str(context).strip()
    return priors.get(ctx, _GLOBAL_MEAN_CACHE)


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = 0 if col in CREDIT_COLS else ""
    df["credibility_score"] = pd.to_numeric(
        df["credibility_score"], errors="coerce").fillna(0.5)
    for c in CREDIT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # Fill ALL string columns — NaN floats break str methods downstream
    str_cols = set(SCHEMA_COLS) - set(CREDIT_COLS) - {"credibility_score"}
    for c in str_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()
    # Specifically ensure context is never NaN or empty — fall back to "unknown"
    df["context"] = df["context"].replace("", "unknown")
    return df[SCHEMA_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

LIAR2_INT_TO_LABEL = {
    0:"pants-fire",1:"false",2:"barely-true",
    3:"half-true",4:"mostly-true",5:"true",
}
LIAR2_SCORE = {
    "pants-fire":0.0,"false":0.2,"barely-true":0.4,
    "half-true":0.6,"mostly-true":0.8,"true":1.0,
}
MULTIFC_MAP = {
    # Clearly true
    "true":1.0,"correct":1.0,"accurate":1.0,"verified":1.0,"confirmed":1.0,
    "promise kept":0.9,
    # Mostly true
    "mostly true":0.8,"mostly correct":0.8,"largely true":0.8,"mostly accurate":0.8,
    # Mixed / partially true
    "partially true":0.6,"half true":0.6,"mixed":0.6,"mixture":0.6,
    "previously true":0.6,"needs context":0.5,
    # Uncertain / unverified
    "unproven":0.5,"unclear":0.5,"unverified":0.5,"outdated":0.5,
    # Misleading / mostly false
    "mostly false":0.4,"largely false":0.4,"understated":0.4,
    "misleading":0.35,"misleads":0.35,"overstated":0.35,"exaggerated":0.35,
    "overstates":0.35,"cherry pick":0.35,"cherry picks":0.35,
    # False
    "false":0.2,"incorrect":0.2,"inaccurate":0.2,"distorts":0.2,
    # Satire / fiction / fabricated
    "satire":0.15,"fiction":0.1,"legend":0.1,
    "fabricated":0.0,"pants on fire":0.0,"scam":0.0,"fake":0.0,
}
# HuggingFace FEVER uses "REFUTED" (past tense), not "REFUTES".
# Both forms are included so the code works regardless of dataset version.
FEVER_SCORE = {
    "SUPPORTS":1.0,"REFUTES":0.0,"REFUTED":0.0,"NOT ENOUGH INFO":0.5,
    "supports":1.0,"refutes":0.0,"refuted":0.0,"not_enough_info":0.5,
}
AVERITEC_SCORE = {
    "supported":1.0,"refuted":0.0,
    "conflicting_evidence":0.35,"conflicting evidence/cherrypicking":0.35,
    "not_enough_evidence":0.5,"not enough evidence":0.5,
}


def _load_from_hf(dataset_id: str, config: str = None,
                   trust_remote_code: bool = False):
    """Try HuggingFace, return raw DatasetDict or None."""
    try:
        from datasets import load_dataset
        kwargs = {}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        return load_dataset(dataset_id, **kwargs) if config is None \
            else load_dataset(dataset_id, config, **kwargs)
    except Exception as e:
        print(f"  HF unavailable ({str(e)[:70]})")
        return None


def load_liar2(local: bool = False, sample: int = None) -> pd.DataFrame:
    """
    LIAR-2 — ~23k PolitiFact political statements, 6-class integer labels.

    Key design notes:
    • Labels stored as integers 0–5 in HF dataset — mapped to string then score.
    • Credit history (past PolitiFact ratings per speaker) stored as 6 count cols.
    • Justification field used as LLM fine-tuning target in Phase 7.
    • All 3 HF splits merged here; re-split in make_splits() after combining.
    """
    print("Loading LIAR-2…")
    if not local:
        raw = _load_from_hf("chengxuphd/liar2")
        if raw is not None:
            records = []
            for sname, sdata in raw.items():
                rows = sdata if sample is None else \
                    sdata.shuffle(seed=42).select(range(min(sample//3, len(sdata))))
                for row in rows:
                    ls = LIAR2_INT_TO_LABEL.get(row.get("label",-1))
                    if ls is None: continue
                    stmt = (row.get("statement") or "").strip()
                    if not stmt: continue
                    credit = {c: int(row.get(c, 0) or 0) for c in CREDIT_COLS}
                    records.append({
                        "text": stmt, "label_original": ls,
                        "credibility_score": LIAR2_SCORE[ls],
                        "speaker": (row.get("speaker") or "").strip(),
                        "speaker_job": (row.get("job_title") or
                                        row.get("speaker_job_title") or "").strip(),
                        "subject": (row.get("subject") or "").strip(),
                        "context": (row.get("context") or "").strip(),
                        "state_info": (row.get("state_info") or "").strip(),
                        "justification": (row.get("justification") or "").strip(),
                        "split": sname, "dataset": "liar2", **credit,
                    })
            df = pd.DataFrame(records)
            print(f"  LIAR-2 (HF): {len(df):,} rows")
            return _ensure_schema(df)

    path = DATA_DIR / "liar2_raw.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run with --hf or place CSV in ./data/")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if sample: df = df.sample(min(sample, len(df)), random_state=42)
    print(f"  LIAR-2 (local): {len(df):,} rows")
    return _ensure_schema(df)


def load_multifc(local: bool = False, sample: int = None) -> pd.DataFrame:
    """
    MultiFC — ~36k claims from 26 global fact-checking orgs.
    Covers health, science, finance — fills LIAR-2's domain gaps.
    Includes WhatsApp/Facebook/Instagram contexts absent from LIAR-2.
    """
    print("Loading MultiFC…")
    if not local:
        raw = _load_from_hf("pszemraj/multi_fc")
        if raw is not None:
            records = []
            for sname, sdata in raw.items():
                rows = sdata if sample is None else \
                    sdata.shuffle(seed=42).select(range(min(sample//3, len(sdata))))
                for row in rows:
                    # Actual field is "label" not "gold_label"
                    rl = (row.get("label") or "").lower().strip()
                    score = MULTIFC_MAP.get(rl)
                    if score is None:
                        for k, v in MULTIFC_MAP.items():
                            if k in rl: score = v; rl = k; break
                    if score is None: continue
                    claim = (row.get("claim") or "").strip()
                    if not claim: continue
                    reason_text = (row.get("reason") or "").strip()
                    # `reason` is the fact-checking explanation paragraph,
                    # NOT a venue descriptor. Only use it as context when it's
                    # short enough to look like a venue type; otherwise default
                    # to "a social media post" (MultiFC is primarily web claims).
                    # This prevents normalise_context from calling the NLI model
                    # on every 500-word paragraph in the dataset.
                    ctx = reason_text if len(reason_text) < 80 \
                        else "a social media post"
                    records.append({
                        "text": claim, "label_original": rl,
                        "credibility_score": score,
                        # Actual field is "speaker" not "claimant"
                        "speaker": (row.get("speaker") or "").strip()
                                   or "anonymous_social_media",
                        "speaker_job": str(row.get("checker") or "").strip(),
                        # Actual field is "article title" (with space) not "main_text"
                        "subject": str(row.get("article title") or
                                       row.get("categories") or "")[:60].strip(),
                        "context": ctx,
                        "state_info": "", "split": sname, "dataset": "multifc",
                        "justification": reason_text,
                        **{c: 0 for c in CREDIT_COLS},
                    })
            df = pd.DataFrame(records)
            print(f"  MultiFC (HF): {len(df):,} rows")
            return _ensure_schema(df)

    path = DATA_DIR / "multifc_raw.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found."); return pd.DataFrame()
    df = pd.read_csv(path)
    if sample: df = df.sample(min(sample, len(df)), random_state=42)
    print(f"  MultiFC (local): {len(df):,} rows")
    return _ensure_schema(df)


def load_fever(local: bool = False, sample: int = None,
               cap: int = 50_000) -> pd.DataFrame:
    """
    FEVER — ~185k Wikipedia factual claims, 3 labels.
    Capped at 50k to prevent FEVER dominating combined dataset.
    No justification field — FEVER rows excluded from LLM fine-tuning (Phase 7).

    Uses lucadiliello/fever (Parquet mirror) — the official fever/fever relies on
    a deprecated loading script that modern datasets versions refuse to run.
    Labels in this mirror are integers: 0=SUPPORTS, 1=NOT ENOUGH INFO, 2=REFUTES.
    """
    # Integer label map for lucadiliello/fever
    _INT_SCORE = {0: 1.0, 1: 0.5, 2: 0.0}
    _INT_LABEL = {0: "supports", 1: "not_enough_info", 2: "refutes"}

    print(f"Loading FEVER (cap={cap:,})…")
    if not local:
        raw = _load_from_hf("lucadiliello/fever")
        if raw is not None:
            records = []; seen = 0
            for sname, sdata in raw.items():
                if seen >= cap: break
                for row in sdata.shuffle(seed=42):
                    if seen >= cap: break
                    try:
                        lbl_int = int(row.get("label"))
                    except (TypeError, ValueError):
                        continue
                    score = _INT_SCORE.get(lbl_int)
                    if score is None: continue
                    claim = (row.get("claim") or "").strip()
                    if not claim: continue
                    # evidence is a list of strings in this mirror
                    evidence = row.get("evidence") or []
                    if isinstance(evidence, list):
                        justification = " ".join(
                            str(e) for e in evidence[:3] if e)[:300]
                    else:
                        justification = ""
                    records.append({
                        "text": claim,
                        "label_original": _INT_LABEL[lbl_int],
                        "credibility_score": score,
                        "speaker": "wikipedia_claim", "speaker_job": "",
                        "subject": "factual knowledge", "context": "factual claim",
                        "state_info": "", "justification": justification,
                        "split": sname, "dataset": "fever",
                        **{c: 0 for c in CREDIT_COLS},
                    })
                    seen += 1
            df = pd.DataFrame(records)
            print(f"  FEVER (HF): {len(df):,} rows")
            return _ensure_schema(df)

    path = DATA_DIR / "fever_raw.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found."); return pd.DataFrame()
    df = pd.read_csv(path)
    if sample: df = df.sample(min(sample, len(df)), random_state=42)
    if len(df) > cap: df = df.sample(cap, random_state=42)
    print(f"  FEVER (local): {len(df):,} rows")
    return _ensure_schema(df)


def load_averitec(local: bool = False, sample: int = None) -> pd.DataFrame:
    """
    AVeriTeC — ~4.5k web-verified claims with Q&A evidence chains.
    Best for: grounding claims in web evidence → feeds RAG pipeline.
    Q&A pairs used as justification for LLM fine-tuning (Phase 7).
    """
    print("Loading AVeriTeC…")
    if not local:
        raw = _load_from_hf("pminervini/averitec")
        if raw is not None:
            records = []
            for sname, sdata in raw.items():
                rows = sdata if sample is None else \
                    sdata.shuffle(seed=42).select(range(min(sample//3, len(sdata))))
                for row in rows:
                    rl = (row.get("label") or "").lower().strip()
                    score = AVERITEC_SCORE.get(rl)
                    if score is None:
                        for k, v in AVERITEC_SCORE.items():
                            if k[:8] in rl: score = v; break
                    if score is None: continue
                    claim = (row.get("claim") or "").strip()
                    if not claim: continue
                    qa = row.get("questions_and_answers") or []
                    evidence = []
                    for item in qa[:3]:
                        if isinstance(item, dict):
                            for ans in (item.get("answers") or [])[:1]:
                                if isinstance(ans, dict):
                                    evidence.append(
                                        (ans.get("answer") or "").strip())
                    justification = " ".join(filter(None, evidence))[:500]
                    records.append({
                        "text": claim,
                        "label_original": rl.replace("/","_").replace(" ","_"),
                        "credibility_score": score,
                        "speaker": (row.get("speaker_name") or
                                    row.get("claimant") or "").strip(),
                        "speaker_job": (row.get("speaker_description") or "")[:100],
                        "subject": (row.get("fact_checking_article") or "")[:60].strip(),
                        "context": "online media", "state_info": "",
                        "justification": justification,
                        "split": sname, "dataset": "averitec",
                        **{c: 0 for c in CREDIT_COLS},
                    })
            df = pd.DataFrame(records)
            print(f"  AVeriTeC (HF): {len(df):,} rows")
            return _ensure_schema(df)

    path = DATA_DIR / "averitec_raw.csv"
    if not path.exists():
        print(f"  ERROR: {path} not found."); return pd.DataFrame()
    df = pd.read_csv(path)
    if sample: df = df.sample(min(sample, len(df)), random_state=42)
    print(f"  AVeriTeC (local): {len(df):,} rows")
    return _ensure_schema(df)


def add_context_prior_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add context_credibility_prior as a NUMERICAL FEATURE.
    NEVER modifies credibility_score (the target label).
    Design: lower prior for WhatsApp/social media, higher for BBC/press release.
    """
    df = df.copy()
    df["context_credibility_prior"] = df["context"].apply(get_context_prior)
    corr = df["context_credibility_prior"].corr(df["credibility_score"])
    print(f"\n  context_credibility_prior: Pearson r vs label = {corr:.4f}")
    return df


def compute_historical_credibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted average of LIAR-2 credit history counts → single score.
    Non-LIAR-2 rows get NaN (filled by speaker fallback in Phase 9).
    """
    weights = {
        "true_counts":1.0,"mostly_true_counts":0.8,"half_true_counts":0.6,
        "mostly_false_counts":0.4,"false_counts":0.2,"pants_on_fire_counts":0.0,
    }
    def _calc(row):
        if row["dataset"] != "liar2": return np.nan
        total = sum(int(row[c]) for c in weights)
        if total == 0: return np.nan
        return round(sum(int(row[c])*w for c,w in weights.items())/total, 4)
    df = df.copy()
    df["historical_credibility"] = df.apply(_calc, axis=1)
    n = df["historical_credibility"].notna().sum()
    print(f"  Historical credibility computed for {n:,} LIAR-2 rows")
    return df


def combine_datasets(*dfs) -> pd.DataFrame:
    valid = [d for d in dfs if d is not None and len(d) > 0]
    print(f"\nCombining {len(valid)} datasets…")
    combined = pd.concat(valid, ignore_index=True)
    print(f"  Raw: {len(combined):,} rows")
    combined["text"] = (
        combined["text"].fillna("").str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"http\S+", "", regex=True)
        .str.replace(r"www\.\S+", "", regex=True).str.strip()
    )
    before = len(combined)
    combined = combined[combined["text"].str.len() >= 20].copy()
    print(f"  Dropped {before-len(combined):,} short-text rows")
    combined["_h"] = combined["text"].apply(
        lambda t: hashlib.md5(t.lower().encode()).hexdigest()[:12])
    before = len(combined)
    combined = combined.drop_duplicates("_h").drop(columns=["_h"]).reset_index(drop=True)
    print(f"  Removed {before-len(combined):,} duplicates | Final: {len(combined):,} rows")
    print("\n  Rows by dataset:")
    for ds, cnt in combined["dataset"].value_counts().items():
        print(f"    {ds:<15} {cnt:>7,}  ({cnt/len(combined)*100:.1f}%)")
    return combined


def make_splits(df, val_size=0.10, test_size=0.10, seed=42):
    """
    Joint stratification on credibility_bucket × dataset.
    Ensures every split has representative mix of score ranges AND sources.
    """
    df = df.copy()
    df["_sb"] = pd.cut(df["credibility_score"],
                        bins=[-0.01,.35,.65,1.01], labels=["low","medium","high"])
    df["_strata"] = df["_sb"].astype(str) + "_" + df["dataset"]
    rare = df["_strata"].value_counts()[lambda x: x < 5].index
    df.loc[df["_strata"].isin(rare), "_strata"] = "other"
    tv, test = train_test_split(df, test_size=test_size,
                                stratify=df["_strata"], random_state=seed)
    train, val = train_test_split(tv, test_size=val_size/(1-test_size),
                                  stratify=tv["_strata"], random_state=seed)
    for s in [train, val, test]:
        s.drop(columns=["_sb","_strata"], inplace=True)
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    print(f"\nSplits:")
    for nm, sdf in [("train",train),("val",val),("test",test)]:
        print(f"  {nm:<6} {len(sdf):>6,} | mean={sdf['credibility_score'].mean():.3f} "
              f"| {sdf['dataset'].value_counts().to_dict()}")
    return train, val, test


def run_phase1(local: bool = False, sample: int = None,
               skip_fever: bool = False) -> None:
    print("\n" + "="*60)
    print("  PHASE 1 — Dataset Loading & Merging")
    print("="*60)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    liar2    = load_liar2   (local=local, sample=sample)
    multifc  = load_multifc (local=local, sample=sample)
    averitec = load_averitec(local=local, sample=sample)
    fever    = None if skip_fever else load_fever(local=local, sample=sample)

    combined = combine_datasets(liar2, multifc, averitec, fever)
    combined = compute_historical_credibility(combined)

    train, val, test = make_splits(combined)

    # Build priors from TRAIN ONLY — prevents val/test label leakage into
    # context_credibility_prior and all three derived interaction features.
    _priors, _gm = build_data_driven_priors(train)
    print(f"  Context priors built | global mean: {_gm:.4f}")

    # Stamp prior feature on every split using the train-only cache.
    # Phase 2 will re-read the same JSON cache, so values stay consistent.
    for _df in (train, val, test):
        _df["context_credibility_prior"] = _df["context"].apply(get_context_prior)
    corr = train["context_credibility_prior"].corr(train["credibility_score"])
    print(f"  context_credibility_prior: Pearson r vs label = {corr:.4f}")

    # Save
    stats = {
        "total":len(combined),"train":len(train),"val":len(val),"test":len(test),
        "datasets": combined["dataset"].value_counts().to_dict(),
        "credibility_score": {
            "mean":round(combined["credibility_score"].mean(),4),
            "std": round(combined["credibility_score"].std(),4),
            "skew":round(combined["credibility_score"].skew(),4),
        },
        "label_dist": combined["label_original"].value_counts().to_dict(),
        "context_prior_corr": round(corr, 4),
    }
    for nm, df in [("train",train),("val",val),("test",test)]:
        df.to_csv(DATA_DIR / f"{nm}.csv", index=False)
        print(f"  Saved data/{nm}.csv  ({len(df):,} rows)")
    (DATA_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("  Saved data/dataset_stats.json")
    print("✓ Phase 1 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — PREPROCESSING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# ── Context normaliser (fast substring rules + zero-shot NLI fallback) ──────
_CONTEXT_RULES = [
    (["whatsapp","wapp","whats app","wa.me"],  "a WhatsApp forward"),
    (["telegram"],                             "a Telegram forward"),
    (["instagram","ig post"],                  "a social media post"),
    (["tiktok","tik tok"],                     "a social media post"),
    (["twitter","tweet","x.com","@"],          "a Twitter post"),
    (["facebook","fb post"],                   "a Facebook post"),
    (["viral image","meme","screenshot"],      "a viral image"),
    (["blog","substack","medium.com"],         "a blog post"),
    (["campaign rally","rally"],               "a campaign rally"),
    (["press release"],                        "a press release"),
    (["news conference","press conference","briefing","presser"], "a news conference"),
    (["tv interview","television interview",
      "bbc.com","cnn.com","nytimes.com","reuters.com","reuters",
      "apnews.com","ap.org","washingtonpost.com","theguardian.com",
      "bloomberg.com","politico.com","nbcnews.com","abcnews.com"],
     "a TV interview"),
    (["radio","podcast"],                      "a radio interview"),
    (["debate"],                               "a debate"),
    (["op-ed","opinion piece","column"],       "an op-ed"),
    (["stump speech"],                         "a stump speech"),
    (["town hall","townhall"],                 "a town hall meeting"),
    (["newsletter","email blast"],             "a newsletter"),
    (["campaign website","campaign site"],     "a campaign website"),
    (["advertisement","commercial","ad "],     "an ad"),
    (["speech","address","remarks"],           "a speech"),
    # LIAR-2 specific patterns
    (["news release","press statement","official statement"],           "a press release"),
    (["talk show"],                                                     "a TV interview"),
    (["interview"],                                                     "a TV interview"),
    (["campaign event","campaign speech","fundraiser","fund-raiser",
      "convention"],                                                    "a campaign rally"),
    (["web posting","web ad","web site","website","online"],            "online media"),
    (["mailer","direct mail","tv ad","radio ad","television ad",
      "campaign ad","political ad"],                                    "an ad"),
    (["statement","announcement","filing"],                             "a press release"),
    (["hearing","committee","senate floor","house floor",
      "congress","legislature","legislative session"],                  "a news conference"),
    (["email","e-mail"],                                                "a newsletter"),
    (["book","report","study","paper","journal"],                       "a press release"),
    (["floor speech","floor statement"],                                "a speech"),
]
_known_slots = set(CONTEXT_SLOTS)

def normalise_context(context) -> str:
    """
    Map any value (str, float NaN, None, URL) → canonical CONTEXT_SLOTS entry.
    """
    if context is None:
        return "unknown"
    if isinstance(context, float):
        return "unknown"
    ctx = str(context).strip()
    if not ctx or ctx in _known_slots: return ctx or "unknown"
    if len(ctx) > 120:
        return "unknown"
    ctx_low = ctx.lower()
    for triggers, slot in _CONTEXT_RULES:
        if any(t in ctx_low for t in triggers): return slot
    return "unknown"


# ── Preprocessing steps ──────────────────────────────────────────────────────
_PUNCT_RE  = re.compile(r"[^\w\s\-]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def _remove_punctuation(t: str) -> str:
    return _PUNCT_RE.sub(" ", t).strip()

def _lowercase(t: str) -> str:
    return t.lower()

def _remove_stopwords(t: str) -> str:
    return " ".join(w for w in t.split() if w not in STOP_WORDS)

def _remove_non_english(t: str) -> str:
    return " ".join(w for w in t.split() if not _NON_ASCII.search(w))

def _normalise_repeats(t: str) -> str:
    return _REPEAT_RE.sub(r"\1\1", t)

def _tokenize(t: str) -> list:
    return word_tokenize(t)

def _stem(tokens: list) -> list:
    return [STEMMER.stem(tk) for tk in tokens]


def preprocess_for_tfidf(text: str) -> str:
    """
    7 steps: punct → lower → stopwords → non-english → repeats → tokenize → STEM.
    Stem reduces vocabulary for sparse TF-IDF features.
    """
    if not isinstance(text, str) or not text.strip(): return ""
    t = _remove_punctuation(text)
    t = _lowercase(t)
    t = _remove_stopwords(t)
    t = _remove_non_english(t)
    t = _normalise_repeats(t)
    toks = _tokenize(t)
    return " ".join(_stem(toks))


def preprocess_for_deberta(text: str) -> str:
    """
    5 steps: punct → lower → stopwords → non-english → repeats.
    NO stemming — DeBERTa's BPE tokeniser handles morphology.
    NO autocorrect — slow (~4-8s/stmt); DeBERTa is robust to minor spelling.
    """
    if not isinstance(text, str) or not text.strip(): return ""
    t = _remove_punctuation(text)
    t = _lowercase(t)
    t = _remove_stopwords(t)
    t = _remove_non_english(t)
    t = _normalise_repeats(t)
    return t.strip()


def is_english_doc(text: str) -> bool:
    try:
        from langdetect import detect
        return detect(text) == "en"
    except Exception:
        return True


# ── Feature engineering ──────────────────────────────────────────────────────

def extract_all_features(text, context) -> dict:
    """
    Extract 11 features from RAW text (not preprocessed):

    VADER (4):
      vader_compound, vader_pos, vader_neg, vader_neu

    Hu & Liu opinion lexicon (2):
      pos_word_count, neg_word_count

    Derived sentiment (2):
      pos_neg_ratio       : pos/(pos+neg), 0.5 if no opinion words
      sentiment_extremity : |vader_pos - vader_neg|

    Context × sentiment interactions (3) ← NEW (addresses rally boast problem):
      context_sentiment_risk     : context_prior × sentiment_extremity
                                   Low = high emotion in low-accountability venue → RED FLAG
      context_adjusted_sentiment : vader_compound × context_prior
                                   Discounts boastful rally sentiment,
                                   preserves measured BBC/press release sentiment
      persuasive_context_flag    : 1 if persuasive context (rally, ad, WhatsApp)

    WHY RAW TEXT for features:
      VADER and Hu&Liu use punctuation + casing for accuracy.
      Stripping them first (as in TF-IDF preprocessing) hurts feature quality.
    """
    # Coerce NaN / non-string values (arrive from CSV missing fields)
    if not isinstance(text, str):
        text = "" if (text is None or
                      (isinstance(text, float) and np.isnan(text))) else str(text)
    if not isinstance(context, str):
        context = "" if (context is None or
                         (isinstance(context, float) and np.isnan(context)))                   else str(context)

    if not text.strip():
        ctx_prior = get_context_prior(context)
        return {
            "vader_compound":0.0,"vader_pos":0.0,"vader_neg":0.0,"vader_neu":1.0,
            "pos_word_count":0,"neg_word_count":0,"pos_neg_ratio":0.5,
            "sentiment_extremity":0.0,
            "context_sentiment_risk":0.0,
            "context_adjusted_sentiment":0.0,
            "persuasive_context_flag":0,
        }

    vs      = VADER.polarity_scores(text)
    tokens  = text.lower().split()
    pos_cnt = sum(1 for t in tokens if t in POS_WORDS)
    neg_cnt = sum(1 for t in tokens if t in NEG_WORDS)
    total   = pos_cnt + neg_cnt
    extremity = round(abs(vs["pos"] - vs["neg"]), 4)
    ctx_prior   = get_context_prior(context)   # data-driven from training labels
    # persuasive_context_flag: data-driven — 1 if this context has a
    # below-average prior (learned from labels, not hardcoded)
    is_persuasive = int(ctx_prior < _GLOBAL_MEAN_CACHE)

    return {
        # VADER
        "vader_compound":           round(vs["compound"], 4),
        "vader_pos":                round(vs["pos"], 4),
        "vader_neg":                round(vs["neg"], 4),
        "vader_neu":                round(vs["neu"], 4),
        # Hu & Liu
        "pos_word_count":           pos_cnt,
        "neg_word_count":           neg_cnt,
        # Derived
        "pos_neg_ratio":            round(pos_cnt / total, 4) if total > 0 else 0.5,
        "sentiment_extremity":      extremity,
        # Context × sentiment interactions (NEW)
        "context_sentiment_risk":   round(ctx_prior * extremity, 4),
        "context_adjusted_sentiment": round(vs["compound"] * ctx_prior, 4),
        "persuasive_context_flag":  is_persuasive,
    }


ALL_FEATURES = [
    "vader_compound","vader_pos","vader_neg","vader_neu",
    "pos_word_count","neg_word_count","pos_neg_ratio","sentiment_extremity",
    "context_sentiment_risk","context_adjusted_sentiment","persuasive_context_flag",
    "context_credibility_prior","token_length_approx",
]


def relevance_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson + Spearman correlation of every feature vs credibility_score.
    Prints verdict: KEEP (|r|≥0.05 and p<0.05) or DROP.
    """
    rows = []
    for feat in ALL_FEATURES:
        if feat not in df.columns: continue
        x = df[feat].fillna(0)
        y = df["credibility_score"]
        pr, pp = stats.pearsonr(x, y)
        sr, _  = stats.spearmanr(x, y)
        verdict = ("KEEP — strong" if abs(pr) >= 0.15 and pp < 0.05 else
                   "KEEP — weak"   if abs(pr) >= 0.05 and pp < 0.05 else "DROP")
        rows.append({"feature":feat,"pearson_r":round(pr,4),
                     "spearman_r":round(sr,4),"p_value":f"{pp:.2e}",
                     "sig":"✓" if pp<0.05 else "✗","verdict":verdict})
    tbl = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    print(f"\n  {'Feature':<35} {'Pearson r':>10} {'Spearman r':>11} {'Sig':>4}  Verdict")
    print(f"  {'─'*35} {'─'*10} {'─'*11} {'─'*4}  {'─'*18}")
    for _, r in tbl.iterrows():
        print(f"  {r['feature']:<35} {r['pearson_r']:>10.4f} "
              f"{r['spearman_r']:>11.4f} {r['sig']:>4}  {r['verdict']}")
    return tbl


def apply_preprocessing(df: pd.DataFrame, filter_en: bool = True) -> pd.DataFrame:
    df = df.copy()
    n_before = len(df)

    # English filter
    if filter_en:
        df["_en"] = df["text"].apply(is_english_doc)
        removed = (~df["_en"]).sum()
        df = df[df["_en"]].drop(columns=["_en"]).reset_index(drop=True)
        if removed: print(f"  Removed {removed:,} non-English rows")

    # Sanitise text and context columns — NaN floats arrive from CSV
    # when source datasets have missing fields.
    df["text"]    = df["text"].fillna("").astype(str)
    df["context"] = (df["context"]
                     .fillna("")
                     .astype(str)
                     .str.strip()
                     .replace({"nan": "", "None": "", "none": ""}))

    # Apply normalise_context to EVERY row (not via mask) so pandas
    # never calls the function with a float — it always receives a str.
    df["context"] = df["context"].apply(normalise_context)
    df["context_credibility_prior"] = df["context"].apply(get_context_prior)

    # Text preprocessing
    df["text_tfidf"]   = df["text"].apply(preprocess_for_tfidf)
    df["text_deberta"] = df["text"].apply(preprocess_for_deberta)

    # Sentiment + interaction features
    feat_df = df.apply(
        lambda row: pd.Series(extract_all_features(row["text"], row["context"])),
        axis=1
    )
    df = pd.concat([df, feat_df], axis=1)

    # Token length proxy
    df["token_length_approx"] = (df["text_deberta"].str.len() / 4).round(0).astype(int)
    too_long = (df["token_length_approx"] > 512).sum()
    if too_long:
        print(f"  {too_long:,} rows likely >512 tokens — will be truncated by DeBERTa")

    # Drop rows where preprocessing emptied the text
    df = df[(df["text_tfidf"].str.len() > 3) &
            (df["text_deberta"].str.len() > 3)].reset_index(drop=True)
    print(f"  Preprocessed: {len(df):,} rows (dropped {n_before-len(df):,})")
    return df


def run_phase2(relevance: bool = False, filter_en: bool = True) -> None:
    print("\n" + "="*60)
    print("  PHASE 2 — Preprocessing & Feature Engineering")
    print("="*60)

    for split in ["train","val","test"]:
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            print(f"  {path} not found — run Phase 1 first"); continue
        print(f"\n[{split.upper()}]")
        df = pd.read_csv(path)
        df = apply_preprocessing(df, filter_en=filter_en)
        df.to_csv(path, index=False)
        print(f"  Saved {path}")

    if relevance:
        print("\n" + "="*60)
        print("  FEATURE RELEVANCE — train set")
        print("="*60)
        relevance_analysis(pd.read_csv(DATA_DIR / "train.csv"))

    print("✓ Phase 2 complete.")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════

LABEL_COLOR_MAP = {
    "pants-fire":"#E24B4A","false":"#F09595","barely-true":"#EF9F27",
    "half-true":"#FAC775","mostly-true":"#97C459","true":"#639922",
    "pants on fire":"#E24B4A","mostly true":"#97C459","half true":"#FAC775",
    "mostly false":"#EF9F27","supports":"#639922","refutes":"#E24B4A",
    "not_enough_info":"#FAC775","supported":"#639922","refuted":"#E24B4A",
    "conflicting_evidence":"#EF9F27","not_enough_evidence":"#FAC775",
}
DS_COLORS = {"liar2":"#7B5CF5","multifc":"#185FA5","fever":"#1D9E75","averitec":"#D85A30"}


def descriptive_stats(df: pd.DataFrame) -> None:
    cs = df["credibility_score"]
    sep = "─"*55
    print(f"\n{sep}")
    print("  DATASET OVERVIEW")
    print(sep)
    print(f"  Total rows         : {len(df):,}")
    print(f"  Memory usage       : {df.memory_usage(deep=True).sum()/1e6:.2f} MB")
    print(f"  Duplicate rows     : {df.duplicated(subset=['text']).sum():,}")
    if "dataset" in df.columns:
        print("\n  Rows by source:")
        for ds, cnt in df["dataset"].value_counts().items():
            print(f"    {ds:<15} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    print(f"\n{sep}")
    print("  CREDIBILITY SCORE")
    print(sep)
    pct = cs.describe(percentiles=[.05,.25,.5,.75,.95])
    for k,v in [("Count",f"{int(pct['count']):,}"),
                ("Mean", f"{pct['mean']:.4f}"),
                ("Median",f"{pct['50%']:.4f}"),
                ("Std",  f"{pct['std']:.4f}"),
                ("Min",  f"{pct['min']:.4f}"),
                ("Max",  f"{pct['max']:.4f}"),
                ("5th pct",f"{pct['5%']:.4f}"),
                ("25th pct",f"{pct['25%']:.4f}"),
                ("75th pct",f"{pct['75%']:.4f}"),
                ("95th pct",f"{pct['95%']:.4f}"),
                ("IQR",  f"{pct['75%']-pct['25%']:.4f}"),
                ("Skewness",f"{cs.skew():.4f}"),
                ("Kurtosis",f"{cs.kurt():.4f}")]:
        print(f"  {k:<15}: {v}")

    from scipy.stats import shapiro
    w, p = shapiro(cs.sample(min(5000, len(cs)), random_state=42))
    print(f"  Shapiro-Wilk  : W={w:.4f}, p={p:.2e} "
          f"({'not normal' if p<0.05 else 'normal'})")

    print(f"\n{sep}")
    print("  LABEL DISTRIBUTION")
    print(sep)
    lc = df["label_original"].value_counts()
    for lbl, cnt in lc.items():
        bar = "█" * int(cnt/len(df)*50)
        print(f"  {str(lbl):<25} {cnt:>6,}  ({cnt/len(df)*100:4.1f}%)  {bar}")
    imbr = lc.max()/lc.min()
    print(f"\n  Imbalance ratio: {imbr:.2f}× "
          f"({'⚠ high' if imbr>3 else '✓ acceptable'})")

    print(f"\n{sep}")
    print("  CONTEXT × SENTIMENT INTERACTION (key features)")
    print(sep)
    for feat in ["context_sentiment_risk","context_adjusted_sentiment",
                 "persuasive_context_flag","context_credibility_prior"]:
        if feat in df.columns:
            col = df[feat]
            corr = col.corr(df["credibility_score"])
            print(f"  {feat:<35} mean={col.mean():.3f}  r={corr:.4f}")

    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    print(f"\n{sep}")
    print("  MISSING VALUES")
    print(sep)
    if len(has_missing) == 0:
        print("  ✓ No missing values")
    else:
        for col in has_missing.index:
            print(f"  {col:<30} {has_missing[col]:>6,}  ({has_missing[col]/len(df)*100:.1f}%)")
    print()


def plot_all(df: pd.DataFrame) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    cs = df["credibility_score"]

    # ── Plot 1: Credibility score distribution ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Credibility score distribution", fontweight="bold")
    ax = axes[0]
    ax.hist(cs, bins=30, color="#378ADD", alpha=0.72, edgecolor="white", lw=0.4)
    ax2 = ax.twinx(); cs.plot.kde(ax=ax2, color="#185FA5", lw=2)
    ax2.set_ylabel("Density", fontsize=9); ax2.set_ylim(bottom=0)
    ax.axvline(cs.mean(),   color="#E24B4A", ls="--", lw=1.3, label=f"Mean {cs.mean():.2f}")
    ax.axvline(cs.median(), color="#639922", ls="--", lw=1.3, label=f"Median {cs.median():.2f}")
    ax.legend(fontsize=8); ax.set_xlabel("Credibility score"); ax.set_ylabel("Count")
    ax.set_title("Histogram + KDE")
    axes[1].boxplot(cs, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#B5D4F4", color="#0C447C"),
                    medianprops=dict(color="#E24B4A", lw=2),
                    flierprops=dict(marker="o", color="#888780", ms=3, alpha=0.3))
    axes[1].set_title("Box plot"); axes[1].set_xticks([])
    plt.tight_layout()
    plt.savefig(EDA_DIR / "01_credibility_distribution.png", bbox_inches="tight")
    plt.close(); print("  Saved: 01_credibility_distribution.png")

    # ── Plot 2: Label distribution ────────────────────────────────────────
    lc = df["label_original"].value_counts()
    colors = [LABEL_COLOR_MAP.get(str(l), "#888780") for l in lc.index]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(lc.index.astype(str), lc.values, color=colors,
                  edgecolor="white", lw=0.5)
    for bar, val in zip(bars, lc.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=8)
    ax.set_title("Label distribution"); ax.set_xlabel("Label")
    ax.set_ylabel("Count"); ax.set_ylim(0, lc.max()*1.2)
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(EDA_DIR / "02_label_distribution.png", bbox_inches="tight")
    plt.close(); print("  Saved: 02_label_distribution.png")

    # ── Plot 3: Credibility by dataset (KDE overlay) ──────────────────────
    if "dataset" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        for ds, col in DS_COLORS.items():
            sub = df[df["dataset"]==ds]["credibility_score"]
            if len(sub) < 10: continue
            sub.plot.kde(ax=ax, label=f"{ds} (n={len(sub):,})", color=col, lw=2)
            ax.axvline(sub.mean(), color=col, ls="--", lw=1, alpha=0.6)
        ax.set_title("Credibility score by dataset (KDE)")
        ax.set_xlabel("Credibility score"); ax.set_xlim(-0.1, 1.1)
        ax.legend(fontsize=9); plt.tight_layout()
        plt.savefig(EDA_DIR / "03_credibility_by_dataset.png", bbox_inches="tight")
        plt.close(); print("  Saved: 03_credibility_by_dataset.png")

    # ── Plot 4: Speaker vs credibility (top/bottom 15) ─────────────────────
    if "speaker" in df.columns:
        liar2 = df[df["dataset"]=="liar2"] if "dataset" in df.columns else df
        spk = (liar2[liar2["speaker"].str.len() > 0]
               .groupby("speaker")["credibility_score"]
               .agg(mean="mean", count="count")
               .query("count >= 5").sort_values("mean"))
        if len(spk) >= 10:
            bottom = spk.head(10); top = spk.tail(10).iloc[::-1]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Speaker credibility (min 5 statements)", fontweight="bold")
            for ax, data, title, color in zip(axes,
                    [bottom, top],
                    ["Bottom 10 (least credible)", "Top 10 (most credible)"],
                    ["#E24B4A", "#639922"]):
                bars = ax.barh(data.index, data["mean"],
                               color=color, alpha=0.8, edgecolor="white")
                for bar, (_, row) in zip(bars, data.iterrows()):
                    ax.text(bar.get_width()+0.01,
                            bar.get_y()+bar.get_height()/2,
                            f"n={int(row['count'])}", va="center", fontsize=7.5)
                ax.set_xlim(0, 1.1); ax.set_title(title)
                ax.set_xlabel("Mean credibility score")
                ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            plt.savefig(EDA_DIR / "04_speaker_credibility.png", bbox_inches="tight")
            plt.close(); print("  Saved: 04_speaker_credibility.png")

    # ── Plot 5: Subject vs credibility ─────────────────────────────────────
    if "subject" in df.columns:
        df_subj = df.copy()
        df_subj["subject"] = df_subj["subject"].str.split(",")
        df_subj = df_subj.explode("subject")
        df_subj["subject"] = df_subj["subject"].str.strip()
        subj = (df_subj[df_subj["subject"].str.len() > 0]
                .groupby("subject")["credibility_score"]
                .agg(mean="mean", count="count")
                .query("count >= 20").sort_values("mean"))
        if len(subj) > 0:
            colors = ["#E24B4A" if v < 0.45 else
                      "#EF9F27" if v < 0.60 else "#639922"
                      for v in subj["mean"]]
            fig, ax = plt.subplots(figsize=(10, max(4, len(subj)*0.35)))
            ax.barh(subj.index, subj["mean"], color=colors,
                    alpha=0.8, edgecolor="white")
            ax.axvline(0.5, color="#888780", ls="--", lw=1)
            ax.set_xlabel("Mean credibility score")
            ax.set_title("Subject vs credibility (min 20 statements)")
            ax.set_xlim(0, 1); ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            plt.savefig(EDA_DIR / "05_subject_credibility.png", bbox_inches="tight")
            plt.close(); print("  Saved: 05_subject_credibility.png")

    # ── Plot 6: Context vs credibility ─────────────────────────────────────
    if "context" in df.columns:
        ctx = (df.groupby("context")["credibility_score"]
               .agg(mean="mean", count="count")
               .query("count >= 10").sort_values("mean"))
        if len(ctx) > 0:
            colors = ["#E24B4A" if v < 0.40 else
                      "#EF9F27" if v < 0.55 else "#639922"
                      for v in ctx["mean"]]
            fig, ax = plt.subplots(figsize=(10, max(4, len(ctx)*0.38)))
            ax.barh(ctx.index, ctx["mean"], color=colors,
                    alpha=0.8, edgecolor="white")
            ax.axvline(0.5, color="#888780", ls="--", lw=1,
                       label="Neutral prior (0.5)")
            ax.set_xlabel("Mean credibility score")
            ax.set_title("Context vs credibility\n"
                         "(WhatsApp/social media ← low | "
                         "press conference/BBC → high)")
            ax.set_xlim(0, 1); ax.tick_params(axis="y", labelsize=8)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(EDA_DIR / "06_context_credibility.png", bbox_inches="tight")
            plt.close(); print("  Saved: 06_context_credibility.png")

    # ── Plot 7: Context × sentiment interaction ───────────────────────────
    feats_available = [f for f in [
        "context_sentiment_risk","context_adjusted_sentiment",
        "sentiment_extremity","persuasive_context_flag",
        "vader_compound","context_credibility_prior",
    ] if f in df.columns]
    if feats_available:
        n = len(feats_available)
        fig, axes = plt.subplots(1, n, figsize=(n*3.5, 4))
        if n == 1: axes = [axes]
        fig.suptitle("Context × Sentiment features vs credibility score",
                     fontweight="bold")
        for ax, feat in zip(axes, feats_available):
            ax.scatter(df[feat], df["credibility_score"],
                       alpha=0.15, s=8, color="#378ADD")
            # Trend line
            xv = df[feat].fillna(0)
            m, b = np.polyfit(xv, df["credibility_score"], 1)
            xs = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xs, m*xs+b, color="#E24B4A", lw=1.5)
            r = xv.corr(df["credibility_score"])
            ax.set_title(f"{feat}\nr={r:.3f}", fontsize=9)
            ax.set_xlabel(feat.replace("_"," "), fontsize=8)
            ax.set_ylabel("Credibility score", fontsize=8)
        plt.tight_layout()
        plt.savefig(EDA_DIR / "07_context_sentiment_interactions.png",
                    bbox_inches="tight")
        plt.close(); print("  Saved: 07_context_sentiment_interactions.png")

    # ── Plot 8: Context × label heatmap ─────────────────────────────────
    top_ctx = df["context"].value_counts().head(8).index.tolist()
    top_ds  = df["dataset"].unique().tolist() if "dataset" in df.columns else []
    if top_ctx and "label_original" in df.columns:
        sub = df[df["context"].isin(top_ctx)]
        pivot = (sub.groupby(["context","dataset"])["credibility_score"]
                 .mean().unstack(fill_value=None) if "dataset" in df.columns
                 else sub.groupby(["context","label_original"])["credibility_score"]
                 .count().unstack(fill_value=0))
        if pivot.shape[0] > 1 and pivot.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(pivot.astype(float), annot=True, fmt=".2f",
                        cmap="RdYlGn", center=0.5,
                        vmin=0, vmax=1, linewidths=0.4, ax=ax,
                        annot_kws={"size":8})
            ax.set_title("Context × Dataset — mean credibility score")
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelrotation=0,  labelsize=8)
            plt.tight_layout()
            plt.savefig(EDA_DIR / "08_context_dataset_heatmap.png",
                        bbox_inches="tight")
            plt.close(); print("  Saved: 08_context_dataset_heatmap.png")

    # ── Plot 9: VADER by label (violin) ──────────────────────────────────
    if "vader_compound" in df.columns and "label_original" in df.columns:
        top_labels = df["label_original"].value_counts().head(6).index.tolist()
        sub = df[df["label_original"].isin(top_labels)].copy()
        if len(sub) > 50:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Sentiment features by label (violin)", fontweight="bold")
            for ax, feat, title in zip(axes,
                    ["vader_compound","context_sentiment_risk"],
                    ["VADER compound sentiment by label",
                     "context_sentiment_risk by label\n"
                     "(low = high emotion in low-credibility venue)"]):
                if feat not in sub.columns: continue
                label_colors = {l: LABEL_COLOR_MAP.get(str(l), "#888780")
                                for l in top_labels}
                parts = ax.violinplot(
                    [sub[sub["label_original"]==l][feat].dropna().values
                     for l in top_labels],
                    positions=range(len(top_labels)),
                    showmedians=True, showextrema=True)
                for i, (pc, lbl) in enumerate(zip(parts["bodies"], top_labels)):
                    pc.set_facecolor(label_colors[lbl])
                    pc.set_alpha(0.75)
                ax.set_xticks(range(len(top_labels)))
                ax.set_xticklabels([str(l) for l in top_labels],
                                   rotation=20, fontsize=8)
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("Label"); ax.set_ylabel(feat)
            plt.tight_layout()
            plt.savefig(EDA_DIR / "09_sentiment_by_label.png", bbox_inches="tight")
            plt.close(); print("  Saved: 09_sentiment_by_label.png")

    # ── Plot 10: Correlation heatmap ──────────────────────────────────────
    num_cols = [c for c in ALL_FEATURES + ["credibility_score","historical_credibility"]
                if c in df.columns]
    if len(num_cols) >= 3:
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols)),
                                        max(6, len(num_cols)-1)))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    linewidths=0.4, annot_kws={"size":7}, ax=ax)
        ax.set_title("Pearson correlation heatmap (lower triangle)")
        ax.tick_params(axis="x", labelrotation=40, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0,  labelsize=7)
        plt.tight_layout()
        plt.savefig(EDA_DIR / "10_correlation_heatmap.png", bbox_inches="tight")
        plt.close(); print("  Saved: 10_correlation_heatmap.png")


def run_phase3() -> None:
    print("\n" + "="*60)
    print("  PHASE 3 — Exploratory Data Analysis")
    print("="*60)
    path = DATA_DIR / "train.csv"
    if not path.exists():
        print("  train.csv not found — run Phases 1 & 2 first"); return
    df = pd.read_csv(path)
    descriptive_stats(df)
    print("\nGenerating plots…")
    plot_all(df)
    print(f"\n✓ Phase 3 complete. Plots saved to {EDA_DIR.resolve()}/")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Credibility Detector — Phases 1, 2 & 3"
    )
    ap.add_argument("--phase",    type=int, choices=[1,2,3],
                    default=None, help="Run a single phase (default: all)")
    ap.add_argument("--local",    action="store_true",
                    help="Use local CSV cache instead of HuggingFace")
    ap.add_argument("--sample",   type=int, default=None,
                    help="Sample N rows per dataset (dev runs)")
    ap.add_argument("--no-fever", action="store_true",
                    help="Skip FEVER dataset (fastest load)")
    ap.add_argument("--relevance",action="store_true",
                    help="Print feature relevance table after Phase 2")
    ap.add_argument("--no-english-filter", action="store_true",
                    help="Skip langdetect English filtering")
    args = ap.parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║  CREDIBILITY DETECTOR  —  Phases 1, 2 & 3   ║")
    print("╚══════════════════════════════════════════════╝")

    phases = [args.phase] if args.phase else [1, 2, 3]

    if 1 in phases:
        run_phase1(local=args.local, sample=args.sample,
                   skip_fever=args.no_fever)
    if 2 in phases:
        run_phase2(relevance=args.relevance,
                   filter_en=not args.no_english_filter)
    if 3 in phases:
        run_phase3()

    print("\n╔══════════════════════════════╗")
    print("║  All selected phases done.   ║")
    print("║  Next: phase4_baseline.py    ║")
    print("╚══════════════════════════════╝")


if __name__ == "__main__":
    main()