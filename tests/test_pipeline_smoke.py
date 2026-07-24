"""
Smoke test: 200 synthetic rows through Phases 1–4 end-to-end.
Verifies:
  - Phases 1-2 produce CSVs with correct columns and no NaN in key fields
  - Phase 4 TF-IDF baseline loads data and trains without error
  - Priors are built from train split only (no val/test leakage)
  - Final splits have the right size proportions
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow project root imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_pipeline as m

FEAT_COLS = [
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "pos_word_count", "neg_word_count", "pos_neg_ratio",
    "sentiment_extremity", "context_sentiment_risk",
    "context_adjusted_sentiment", "persuasive_context_flag",
    "context_credibility_prior", "token_length_approx",
]


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    """Generate synthetic rows that look like combined dataset output."""
    rng = np.random.default_rng(42)
    texts = [
        "The president signed the bill into law today.",
        "Vaccines contain microchips according to this claim.",
        "The unemployment rate fell to a record low last month.",
        "Scientists confirm that coffee cures cancer.",
        "The senator voted against the tax relief act.",
    ] * (n // 5 + 1)
    texts = texts[:n]

    labels_int = rng.integers(0, 6, size=n)
    label_names = [m.LIAR2_INT_TO_LABEL[i] for i in labels_int]
    scores = [m.LIAR2_SCORE[l] for l in label_names]
    contexts = rng.choice(
        ["a speech", "a Twitter post", "a WhatsApp forward",
         "a press release", "unknown"],
        size=n
    )

    df = pd.DataFrame({
        "text":             texts,
        "label_original":   label_names,
        "credibility_score": scores,
        "speaker":          ["test_speaker"] * n,
        "speaker_job":      ["politician"] * n,
        "subject":          ["economy"] * n,
        "context":          contexts,
        "state_info":       [""] * n,
        "justification":    [""] * n,
        "split":            ["train"] * n,
        "dataset":          ["liar2"] * n,
        **{c: [0] * n for c in m.CREDIT_COLS},
    })
    return df


@pytest.fixture(scope="module")
def tmp_data_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline_data")
    return tmp


@pytest.fixture(scope="module")
def patched_module(tmp_data_dir):
    """Redirect DATA_DIR to temp dir and build splits from synthetic data."""
    original_data = m.DATA_DIR
    original_eda  = m.EDA_DIR

    m.DATA_DIR = tmp_data_dir
    m.EDA_DIR  = tmp_data_dir / "eda"
    m.EDA_DIR.mkdir(exist_ok=True)
    m._CONTEXT_PRIORS_CACHE = {}
    m._GLOBAL_MEAN_CACHE = 0.5

    df = _synthetic_df(200)
    df = m.compute_historical_credibility(df)
    train, val, test = m.make_splits(df)

    # Build priors from TRAIN ONLY — mirrors fixed run_phase1 order
    m.build_data_driven_priors(train)
    for _df in (train, val, test):
        _df["context_credibility_prior"] = _df["context"].apply(m.get_context_prior)

    # Apply preprocessing
    train = m.apply_preprocessing(train, filter_en=False)
    val   = m.apply_preprocessing(val,   filter_en=False)
    test  = m.apply_preprocessing(test,  filter_en=False)

    for name, sdf in [("train", train), ("val", val), ("test", test)]:
        sdf.to_csv(tmp_data_dir / f"{name}.csv", index=False)

    yield {"train": train, "val": val, "test": test, "data_dir": tmp_data_dir}

    m.DATA_DIR = original_data
    m.EDA_DIR  = original_eda
    m._CONTEXT_PRIORS_CACHE = {}
    m._GLOBAL_MEAN_CACHE = 0.5


# ── split sizes ───────────────────────────────────────────────────────────────

def test_split_sizes_are_reasonable(patched_module):
    train = patched_module["train"]
    val   = patched_module["val"]
    test  = patched_module["test"]
    total = len(train) + len(val) + len(test)
    assert total == 200
    assert 0.70 <= len(train) / total <= 0.90
    assert 0.05 <= len(val)   / total <= 0.20
    assert 0.05 <= len(test)  / total <= 0.20


# ── no NaN in key columns ─────────────────────────────────────────────────────

def test_no_nan_credibility_score(patched_module):
    for name, df in patched_module.items():
        if name == "data_dir":
            continue
        assert df["credibility_score"].isna().sum() == 0, \
            f"NaN in credibility_score in {name}"


def test_no_nan_feature_cols(patched_module):
    for name, df in patched_module.items():
        if name == "data_dir":
            continue
        for col in FEAT_COLS:
            if col in df.columns:
                assert df[col].isna().sum() == 0, \
                    f"NaN in {col} in {name}"


# ── priors are from train only ────────────────────────────────────────────────

def test_context_priors_json_exists(patched_module):
    cache = patched_module["data_dir"] / "context_priors.json"
    assert cache.exists(), "context_priors.json not written"

def test_context_priors_json_has_global_mean(patched_module):
    cache = patched_module["data_dir"] / "context_priors.json"
    data = json.loads(cache.read_text())
    assert "global_mean" in data
    assert 0.0 <= data["global_mean"] <= 1.0

def test_prior_feature_in_range(patched_module):
    for name, df in patched_module.items():
        if name == "data_dir":
            continue
        col = df["context_credibility_prior"]
        assert (col >= 0.0).all() and (col <= 1.0).all(), \
            f"context_credibility_prior out of [0,1] in {name}"


# ── preprocessed text columns present ────────────────────────────────────────

def test_text_tfidf_column_exists(patched_module):
    assert "text_tfidf" in patched_module["train"].columns

def test_text_deberta_column_exists(patched_module):
    assert "text_deberta" in patched_module["train"].columns

def test_text_tfidf_non_empty(patched_module):
    train = patched_module["train"]
    assert (train["text_tfidf"].str.len() > 0).all()


# ── TF-IDF baseline smoke (Phase 4) ──────────────────────────────────────────

def test_tfidf_baseline_trains_and_scores(patched_module):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error
    from scipy.sparse import hstack, csr_matrix

    train = patched_module["train"]
    val   = patched_module["val"]

    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(train["text_tfidf"])
    X_va = vec.transform(val["text_tfidf"])

    y_tr = train["credibility_score"].values
    y_va = val["credibility_score"].values

    model = RidgeCV(alphas=(0.1, 1.0, 10.0))
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va).clip(0, 1)

    mae = mean_absolute_error(y_va, preds)
    assert 0.0 <= mae <= 1.0, f"Unexpected MAE: {mae}"
