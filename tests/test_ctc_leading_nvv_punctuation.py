import sys
from pathlib import Path


from ctc_prealign import (  # noqa: E402
    _strip_leading_punctuation_after_tags,
    preprocess_asr_for_mfa,
)


def test_known_leading_nvv_is_preserved_and_uppercased():
    assert preprocess_asr_for_mfa("[Breathing]你好") == "BREATHING 你好"
    assert preprocess_asr_for_mfa("[Cough]你好") == "COUGH 你好"
    assert preprocess_asr_for_mfa("[Laughter]你好") == "LAUGHTER 你好"


def test_real_leading_punctuation_is_still_removed_once():
    assert _strip_leading_punctuation_after_tags("，你好") == ("你好", "，")


def test_punctuation_after_known_nvv_is_not_misclassified_as_leading():
    assert _strip_leading_punctuation_after_tags(
        "[Breathing]，你好") == ("[Breathing]，你好", None)


def test_preprocess_keeps_non_nvv_punctuation_unchanged():
    assert preprocess_asr_for_mfa("，你好") == "，你好"


def test_multiple_control_tags_and_known_nvv_are_preserved():
    source = "<|zh|> <|sad|> [Breathing]你好"
    assert _strip_leading_punctuation_after_tags(source) == (source, None)
    assert _strip_leading_punctuation_after_tags(
        "<|zh|> <|sad|> ，你好") == ("<|zh|> <|sad|> 你好", "，")


def test_known_cough_and_laughter_are_protected():
    for name in ("Cough", "Laughter"):
        source = f"[{name}]，你好"
        assert _strip_leading_punctuation_after_tags(source) == (source, None)


def test_unknown_and_malformed_tags_keep_legacy_bracket_behavior():
    assert _strip_leading_punctuation_after_tags("[Unknown]你好") == (
        "Unknown]你好", "[")
    assert _strip_leading_punctuation_after_tags("[Breathing你好") == (
        "Breathing你好", "[")
