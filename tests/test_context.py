"""
Tests for normalise_context — covers every edge-case the function must handle.
"""
import math
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import normalise_context, CONTEXT_SLOTS

_KNOWN = set(CONTEXT_SLOTS)


# ── null / missing inputs ────────────────────────────────────────────────────

def test_none_returns_unknown():
    assert normalise_context(None) == "unknown"

def test_float_nan_returns_unknown():
    assert normalise_context(float("nan")) == "unknown"

def test_nan_float_via_math():
    assert normalise_context(math.nan) == "unknown"

def test_empty_string_returns_unknown():
    assert normalise_context("") == "unknown"

def test_whitespace_only_returns_unknown():
    assert normalise_context("   ") == "unknown"

def test_string_nan_returns_unknown():
    # CSV round-trip can produce the literal string "nan"
    result = normalise_context("nan")
    # "nan" doesn't match any CONTEXT_SLOT so it should fall through to unknown
    assert result == "unknown"

def test_string_none_returns_unknown():
    result = normalise_context("None")
    assert result == "unknown"


# ── canonical pass-through ───────────────────────────────────────────────────

def test_canonical_slot_passes_through():
    for slot in CONTEXT_SLOTS:
        if slot == "unknown":
            continue
        assert normalise_context(slot) == slot, f"Failed for slot: {slot!r}"

def test_unknown_slot_is_identity():
    assert normalise_context("unknown") == "unknown"


# ── URL / domain matching ────────────────────────────────────────────────────

def test_bbc_url_maps_to_tv_interview():
    assert normalise_context("story posted on bbc.com") == "a TV interview"

def test_reuters_url_maps_to_tv_interview():
    assert normalise_context("Reuters news item") == "a TV interview"

def test_cnn_maps_to_tv_interview():
    assert normalise_context("cnn.com interview") == "a TV interview"

def test_whatsapp_maps_correctly():
    assert normalise_context("shared via WhatsApp") == "a WhatsApp forward"

def test_wa_me_link_maps_to_whatsapp():
    # wa.me is the WhatsApp link shortener
    assert normalise_context("https://wa.me/share/something") == "a WhatsApp forward"


# ── LIAR-2 specific patterns ─────────────────────────────────────────────────

def test_news_release_maps_to_press_release():
    assert normalise_context("news release from governor") == "a press release"

def test_talk_show_maps_to_tv_interview():
    assert normalise_context("a talk show appearance") == "a TV interview"

def test_campaign_event_maps_to_rally():
    assert normalise_context("at a campaign event in Iowa") == "a campaign rally"

def test_web_posting_maps_to_online_media():
    assert normalise_context("web posting on party site") == "online media"

def test_hearing_maps_to_news_conference():
    assert normalise_context("senate floor hearing") == "a news conference"

def test_email_maps_to_newsletter():
    assert normalise_context("campaign email blast") == "a newsletter"

def test_floor_speech_maps_to_speech():
    assert normalise_context("floor speech in the senate") == "a speech"


# ── long-string truncation ───────────────────────────────────────────────────

def test_very_long_string_returns_unknown():
    long_text = "This is a paragraph that goes on and on and contains many words " * 5
    assert normalise_context(long_text) == "unknown"

def test_string_at_exactly_120_chars_still_matched():
    # 120 chars that contain a known trigger word should still be matched
    text = ("speech " * 17)[:120]  # exactly 120 chars, contains "speech"
    result = normalise_context(text)
    assert result in _KNOWN


# ── fallback for unrecognised content ────────────────────────────────────────

def test_completely_unknown_returns_unknown():
    assert normalise_context("quizzical mumbo jumbo text") == "unknown"

def test_output_always_in_slots():
    samples = [
        "foo bar baz", "bbc.com article", "rally speech campaign",
        None, float("nan"), "", "a TV interview", "an interview",
        "congressional hearing floor", "email from senator",
    ]
    for s in samples:
        result = normalise_context(s)
        assert result in _KNOWN, f"Got {result!r} for input {s!r}"
