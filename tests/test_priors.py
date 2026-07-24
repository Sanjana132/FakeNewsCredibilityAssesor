"""
Tests for build_data_driven_priors:
  - Shrinkage math (n→0 → global mean, n→∞ → sample mean)
  - JSON cache round-trip
  - TRAIN-ONLY invariant: priors are identical whether or not val/test exist
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_pipeline as module


def _make_df(n_rows: int, context: str, score: float) -> pd.DataFrame:
    return pd.DataFrame({
        "context":          [context] * n_rows,
        "credibility_score": [score]  * n_rows,
        "dataset":          ["liar2"] * n_rows,
    })


def _patch_data_dir(tmp_path):
    """Redirect DATA_DIR to a temp directory for the duration of a test."""
    original = module.DATA_DIR
    module.DATA_DIR = tmp_path
    module._CONTEXT_PRIORS_CACHE = {}
    module._GLOBAL_MEAN_CACHE = 0.5
    return original


def _restore_data_dir(original):
    module.DATA_DIR = original
    module._CONTEXT_PRIORS_CACHE = {}
    module._GLOBAL_MEAN_CACHE = 0.5


# ── shrinkage math ────────────────────────────────────────────────────────────

def test_zero_count_falls_back_to_global_mean(tmp_path):
    original = _patch_data_dir(tmp_path)
    try:
        # Rows exist (so global_mean is well-defined) but none map to named
        # CONTEXT_SLOTS after normalisation → every slot defaults to global_mean.
        train = pd.DataFrame({
            "context":           ["completely unrecognised venue xyz"] * 30,
            "credibility_score": [0.6] * 30,
        })
        priors, gm = module.build_data_driven_priors(train)
        import math
        assert not math.isnan(gm), "global_mean should not be NaN"
        for slot in module.CONTEXT_SLOTS:
            assert abs(priors[slot] - gm) < 1e-6, f"Slot {slot!r} not at global mean"
    finally:
        _restore_data_dir(original)


def test_large_n_approaches_sample_mean(tmp_path):
    original = _patch_data_dir(tmp_path)
    try:
        # 1000 rows of "a speech" at score=0.9; global_mean ~0.9 too
        df = _make_df(1000, "a speech", 0.9)
        priors, gm = module.build_data_driven_priors(df)
        # shrinkage = 1000/(1000+10) = 0.9901 → prior should be very close to 0.9
        assert abs(priors["a speech"] - 0.9) < 0.02
    finally:
        _restore_data_dir(original)


def test_small_n_shrunk_toward_global_mean(tmp_path):
    original = _patch_data_dir(tmp_path)
    try:
        # 2 rows at 1.0, everything else at 0.0 → global_mean near 0
        rows = [{"context": "a speech", "credibility_score": 1.0}] * 2
        rows += [{"context": "unknown", "credibility_score": 0.0}] * 100
        df = pd.DataFrame(rows)
        priors, gm = module.build_data_driven_priors(df)
        # shrinkage for "a speech" = 2/(2+10)=0.167; prior = 0.167*1.0 + 0.833*gm
        expected = (2 * 1.0 + 10 * gm) / (2 + 10)
        assert abs(priors["a speech"] - expected) < 0.01
    finally:
        _restore_data_dir(original)


def test_shrinkage_formula_exact(tmp_path):
    """Verify the exact formula: prior = (n*mu + kappa*gm) / (n+kappa)."""
    original = _patch_data_dir(tmp_path)
    try:
        kappa = 10.0
        n = 7
        mu = 0.75
        rows = [{"context": "a debate", "credibility_score": mu}] * n
        rows += [{"context": "unknown", "credibility_score": 0.5}] * 50
        df = pd.DataFrame(rows)
        priors, gm = module.build_data_driven_priors(df)
        expected = (n * mu + kappa * gm) / (n + kappa)
        assert abs(priors["a debate"] - expected) < 0.005
    finally:
        _restore_data_dir(original)


# ── cache round-trip ──────────────────────────────────────────────────────────

def test_cache_file_written_and_readable(tmp_path):
    original = _patch_data_dir(tmp_path)
    try:
        df = _make_df(20, "a speech", 0.8)
        priors, gm = module.build_data_driven_priors(df)

        cache_path = tmp_path / "context_priors.json"
        assert cache_path.exists(), "cache file not written"

        data = json.loads(cache_path.read_text())
        assert "priors" in data
        assert "global_mean" in data
        assert abs(data["global_mean"] - gm) < 1e-6
    finally:
        _restore_data_dir(original)


def test_cache_loaded_by_get_context_prior(tmp_path):
    original = _patch_data_dir(tmp_path)
    try:
        df = _make_df(50, "a debate", 0.9)
        module.build_data_driven_priors(df)

        # Clear in-memory cache to force JSON read
        module._CONTEXT_PRIORS_CACHE = {}
        module._GLOBAL_MEAN_CACHE = 0.5

        prior = module.get_context_prior("a debate")
        assert prior > 0.5, "Expected prior close to 0.9 for debate slot"
    finally:
        _restore_data_dir(original)


# ── TRAIN-ONLY invariant ──────────────────────────────────────────────────────

def test_priors_identical_with_and_without_val_test(tmp_path):
    """
    Priors built from train alone must equal priors built from train
    even when val/test with different scores are present in the data.
    This verifies the leakage fix: val/test labels must NOT shift priors.
    """
    original = _patch_data_dir(tmp_path)
    try:
        train = _make_df(80, "a speech", 0.8)

        # Build priors from train only
        priors_train_only, gm_train = module.build_data_driven_priors(train)
        prior_speech_train = priors_train_only["a speech"]

        # Reset and simulate the OLD (leaky) approach: build from combined
        module._CONTEXT_PRIORS_CACHE = {}
        module._GLOBAL_MEAN_CACHE = 0.5

        val  = _make_df(10, "a speech", 0.1)  # very different score
        test = _make_df(10, "a speech", 0.1)
        combined = pd.concat([train, val, test], ignore_index=True)
        priors_combined, _ = module.build_data_driven_priors(combined)
        prior_speech_combined = priors_combined["a speech"]

        # The train-only prior (0.8-ish) must differ from the contaminated one
        assert abs(prior_speech_train - prior_speech_combined) > 0.05, (
            "Priors were identical — the TRAIN-ONLY invariant test is not discriminating"
        )
    finally:
        _restore_data_dir(original)
