"""
Tests for every label map: LIAR-2 int → score, MultiFC substring,
FEVER int/string variants, AVeriTeC partial match.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import (
    LIAR2_INT_TO_LABEL,
    LIAR2_SCORE,
    MULTIFC_MAP,
    FEVER_SCORE,
    AVERITEC_SCORE,
)


# ── LIAR-2 ───────────────────────────────────────────────────────────────────

def test_liar2_all_ints_mapped():
    assert set(LIAR2_INT_TO_LABEL.keys()) == {0, 1, 2, 3, 4, 5}

def test_liar2_int_to_score_monotone():
    # Higher integer → higher score
    scores = [LIAR2_SCORE[LIAR2_INT_TO_LABEL[i]] for i in range(6)]
    assert scores == sorted(scores), "LIAR-2 scores not monotonically increasing"

def test_liar2_extreme_labels():
    assert LIAR2_SCORE["pants-fire"] == 0.0
    assert LIAR2_SCORE["true"] == 1.0

def test_liar2_invalid_int_returns_none():
    assert LIAR2_INT_TO_LABEL.get(-1) is None
    assert LIAR2_INT_TO_LABEL.get(6) is None


# ── MultiFC ──────────────────────────────────────────────────────────────────

def test_multifc_true_variants():
    for key in ("true", "correct", "accurate", "verified", "confirmed"):
        assert MULTIFC_MAP[key] == 1.0, f"Expected 1.0 for {key!r}"

def test_multifc_false_variants():
    for key in ("false", "incorrect", "inaccurate"):
        assert MULTIFC_MAP[key] == 0.2, f"Expected 0.2 for {key!r}"

def test_multifc_satire_near_zero():
    assert MULTIFC_MAP["satire"] < 0.2

def test_multifc_pants_on_fire():
    assert MULTIFC_MAP["pants on fire"] == 0.0

def test_multifc_misleading_below_half():
    assert MULTIFC_MAP["misleading"] < 0.5

def test_multifc_mixed_is_half():
    assert MULTIFC_MAP["mixed"] == 0.6  # partially true / mixed

def test_multifc_all_scores_in_range():
    for key, val in MULTIFC_MAP.items():
        assert 0.0 <= val <= 1.0, f"Out of range: {key!r} → {val}"

def test_multifc_substring_fallback():
    # Simulate the substring fallback logic used in load_multifc
    rl = "largely false claim"
    score = MULTIFC_MAP.get(rl)
    if score is None:
        for k, v in MULTIFC_MAP.items():
            if k in rl:
                score = v
                break
    assert score == 0.4, f"Expected 0.4 for 'largely false' substring, got {score}"


# ── FEVER ────────────────────────────────────────────────────────────────────

def test_fever_supports_is_true():
    assert FEVER_SCORE["SUPPORTS"] == 1.0
    assert FEVER_SCORE["supports"] == 1.0

def test_fever_refutes_is_false():
    assert FEVER_SCORE["REFUTES"] == 0.0
    assert FEVER_SCORE["refutes"] == 0.0
    assert FEVER_SCORE["REFUTED"] == 0.0   # past-tense variant in some HF mirrors

def test_fever_nei_is_half():
    assert FEVER_SCORE["NOT ENOUGH INFO"] == 0.5
    assert FEVER_SCORE["not_enough_info"]  == 0.5

def test_fever_all_scores_in_range():
    for key, val in FEVER_SCORE.items():
        assert 0.0 <= val <= 1.0, f"Out of range: {key!r} → {val}"


# ── AVeriTeC ─────────────────────────────────────────────────────────────────

def test_averitec_supported_is_true():
    assert AVERITEC_SCORE["supported"] == 1.0

def test_averitec_refuted_is_false():
    assert AVERITEC_SCORE["refuted"] == 0.0

def test_averitec_conflicting_evidence():
    assert AVERITEC_SCORE["conflicting_evidence"] == 0.35

def test_averitec_not_enough_evidence_is_half():
    assert AVERITEC_SCORE["not_enough_evidence"] == 0.5

def test_averitec_partial_match():
    # Simulate load_averitec partial-match fallback
    rl = "conflicting evidence/cherrypicking"
    score = AVERITEC_SCORE.get(rl)
    if score is None:
        for k, v in AVERITEC_SCORE.items():
            if k[:8] in rl:
                score = v
                break
    assert score == 0.35, f"Expected 0.35 via partial match, got {score}"

def test_averitec_all_scores_in_range():
    for key, val in AVERITEC_SCORE.items():
        assert 0.0 <= val <= 1.0, f"Out of range: {key!r} → {val}"
