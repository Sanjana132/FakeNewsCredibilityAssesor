"""
Tests for extract_all_features:
  - NaN / None text and context handled gracefully (no crash, all-zero output)
  - Interaction arithmetic verified against known inputs
  - persuasive_context_flag = 1 iff context prior < global mean
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import credibility_detector_phases123 as module


FEAT_KEYS = {
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "pos_word_count", "neg_word_count", "pos_neg_ratio",
    "sentiment_extremity",
    "context_sentiment_risk", "context_adjusted_sentiment",
    "persuasive_context_flag",
}


def setup_module(_m):
    """Install a trivial prior cache so the module doesn't need train data."""
    module._CONTEXT_PRIORS_CACHE = {slot: 0.5 for slot in module.CONTEXT_SLOTS}
    module._GLOBAL_MEAN_CACHE = 0.5
    # Give "a WhatsApp forward" a low prior (persuasive context)
    module._CONTEXT_PRIORS_CACHE["a WhatsApp forward"] = 0.3
    # Give "a press release" a high prior (non-persuasive)
    module._CONTEXT_PRIORS_CACHE["a press release"] = 0.7


def teardown_module(_m):
    module._CONTEXT_PRIORS_CACHE = {}
    module._GLOBAL_MEAN_CACHE = 0.5


# ── null / missing inputs ────────────────────────────────────────────────────

def test_none_text_returns_all_keys():
    feats = module.extract_all_features(None, "a speech")
    assert FEAT_KEYS.issubset(set(feats.keys()))

def test_nan_float_text_returns_all_keys():
    feats = module.extract_all_features(float("nan"), "unknown")
    assert FEAT_KEYS.issubset(set(feats.keys()))

def test_empty_string_text_neutral_compound():
    feats = module.extract_all_features("", "unknown")
    assert feats["vader_compound"] == 0.0

def test_none_context_no_crash():
    feats = module.extract_all_features("The sky is blue.", None)
    assert FEAT_KEYS.issubset(set(feats.keys()))

def test_nan_context_no_crash():
    feats = module.extract_all_features("The sky is blue.", math.nan)
    assert FEAT_KEYS.issubset(set(feats.keys()))


# ── feature value ranges ─────────────────────────────────────────────────────

def test_vader_components_sum_to_one():
    feats = module.extract_all_features("This is a perfectly neutral statement.", "unknown")
    total = feats["vader_pos"] + feats["vader_neg"] + feats["vader_neu"]
    assert abs(total - 1.0) < 1e-3, f"VADER components sum to {total}, expected ~1.0"

def test_compound_in_minus_one_to_one():
    for text in ["This is great!", "Terrible disaster.", "Neutral statement."]:
        feats = module.extract_all_features(text, "unknown")
        assert -1.0 <= feats["vader_compound"] <= 1.0

def test_pos_neg_ratio_bounds():
    feats = module.extract_all_features("Excellent wonderful amazing", "unknown")
    assert 0.0 <= feats["pos_neg_ratio"] <= 1.0

def test_sentiment_extremity_non_negative():
    feats = module.extract_all_features("This is horrifying and terrible!", "unknown")
    assert feats["sentiment_extremity"] >= 0.0

def test_persuasive_flag_is_binary():
    for text in ["Great.", "Awful.", "Neutral."]:
        feats = module.extract_all_features(text, "a WhatsApp forward")
        assert feats["persuasive_context_flag"] in (0, 1)


# ── interaction arithmetic ───────────────────────────────────────────────────

def test_context_sentiment_risk_equals_prior_times_extremity():
    text = "This is absolutely fantastic!"
    feats = module.extract_all_features(text, "a WhatsApp forward")
    prior = module.get_context_prior("a WhatsApp forward")
    expected = round(prior * feats["sentiment_extremity"], 4)
    assert abs(feats["context_sentiment_risk"] - expected) < 1e-3

def test_context_adjusted_sentiment_equals_compound_times_prior():
    text = "Absolutely certain this is true."
    feats = module.extract_all_features(text, "a press release")
    prior = module.get_context_prior("a press release")
    expected = round(feats["vader_compound"] * prior, 4)
    assert abs(feats["context_adjusted_sentiment"] - expected) < 1e-3


# ── persuasive_context_flag logic ────────────────────────────────────────────

def test_persuasive_flag_one_for_low_prior_context():
    # WhatsApp prior set to 0.3 in setup — below global mean 0.5
    feats = module.extract_all_features("Check this out!", "a WhatsApp forward")
    assert feats["persuasive_context_flag"] == 1

def test_persuasive_flag_zero_for_high_prior_context():
    # Press release prior set to 0.7 in setup — above global mean 0.5
    feats = module.extract_all_features("The report confirms findings.", "a press release")
    assert feats["persuasive_context_flag"] == 0

def test_persuasive_flag_threshold_is_global_mean():
    # At exactly global mean (0.5), flag should be 0 (not strictly below)
    module._CONTEXT_PRIORS_CACHE["a debate"] = 0.5
    feats = module.extract_all_features("A debate claim.", "a debate")
    assert feats["persuasive_context_flag"] == 0


# ── all features in ALL_FEATURES list ────────────────────────────────────────

def test_all_feature_cols_present():
    feats = module.extract_all_features("Ordinary claim.", "a speech")
    # ALL_FEATURES has 13 columns; extract_all_features returns 11
    # (context_credibility_prior and token_length_approx are added separately)
    for key in [
        "vader_compound","vader_pos","vader_neg","vader_neu",
        "pos_word_count","neg_word_count","pos_neg_ratio",
        "sentiment_extremity","context_sentiment_risk",
        "context_adjusted_sentiment","persuasive_context_flag",
    ]:
        assert key in feats, f"Missing key: {key}"
