from scripts.ja_tts_export import build_quality_masks


def test_prosody_prediction_and_measurement_masks_are_separate():
    masks = build_quality_masks({"accent_predicted": {"value": [1], "provenance": "text-frontend"}, "f0_measured": None})
    assert masks["accent_predicted_known_mask"] is True
    assert masks["f0_known_mask"] is False
    assert masks["accent_predicted_provenance"] == "text-frontend"
    assert masks["f0_measured_provenance"] is None
