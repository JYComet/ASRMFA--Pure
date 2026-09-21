import pytest

from scripts.ja_tts_export import build_quality_masks
from scripts.ja_en_schema import JAContractError, stable_digest
from scripts.ja_prosody import resolve_mora_tones


def test_prosody_prediction_and_measurement_masks_are_separate():
    masks = build_quality_masks({"accent_predicted": {"value": [1], "provenance": "text-frontend"}, "f0_measured": None})
    assert masks["accent_predicted_known_mask"] is True
    assert masks["f0_known_mask"] is False
    assert masks["accent_predicted_provenance"] == "text-frontend"
    assert masks["f0_measured_provenance"] is None


def test_unknown_is_emitted_only_when_no_eligible_tone_source_exists():
    graph = {"locked_reading_digest": stable_digest("ア"), "mora_nodes": [
        {"mora_id": "m-0", "kana": "ア", "kind": "regular", "f0_observed": False},
    ]}
    rows = resolve_mora_tones(graph, {"accent_evidence_valid": False}, None, None)
    assert rows == [{"mora_id": "m-0", "kana": "ア", "kind": "regular", "mora_index": 0,
                     "tone": "UNK", "tone_known": False, "tone_source": "unknown",
                     "f0_observed": False, "tone_provenance": None, "overridden_sources": []}]


def test_invalid_phrase_nucleus_is_rejected():
    with pytest.raises(JAContractError, match="accent_phrase_unresolved"):
        from scripts.ja_prosody import mora_tones_from_phrase
        mora_tones_from_phrase(2, 3)
