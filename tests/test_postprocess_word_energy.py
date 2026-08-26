from types import SimpleNamespace
import wave

import numpy as np

from scripts.postprocess_textgrids import (
    Interval,
    Tier,
    TextGrid,
    _rms_frames_in_span,
    _word_energy_audit,
    _word_energy_noise_model,
    _word_rms,
    load_audio,
)


def _args(**overrides):
    values = {
        "filter_word_energy_ratio": 2.0,
        "enable_word_in_silence_filter": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _lineage(spans, *, status="verified", reasons=None):
    owners = {}
    source = []
    for ordinal, (start, end, label) in enumerate(spans):
        source.append({"ordinal": ordinal, "start": start, "end": end,
                       "text": label})
        owners[str(ordinal)] = [{
            "phone_ordinal": ordinal,
            "lexical_ordinal": ordinal,
            "label": "a",
            "start": start,
            "end": end,
        }]
    return {"schema": "source-phone-lineage-v1", "status": status,
            "owners": owners, "source_intervals": source,
            "reasons": reasons or []}


def test_word_rms_is_10ms_frame_rms_not_mean_absolute():
    audio = np.tile(np.array([0.0, 1.0], dtype=np.float32), 80)
    assert abs(_word_rms(audio, 16000, 0.0, 0.01) - 2 ** -0.5) < 1e-5
    values, spans = _rms_frames_in_span(audio, 16000, 0.0, 0.01)
    assert len(values) == 1 and spans == [(0.0, 0.01)]


def test_noise_pool_prefers_explicit_silence_and_ratio_has_no_hidden_floor():
    words = Tier("words", 0.0, 0.4, [
        Interval(0.0, 0.1, "<sp0>"), Interval(0.1, 0.2, "ni3"),
        Interval(0.2, 0.4, "<sp1>"),
    ])
    audio = np.full(8000, 0.1, dtype=np.float32)
    audio[1600:3200] = 0.15
    model = _word_energy_noise_model(audio, 16000, words, _args())
    assert model["source"] == "explicit_silence_owner"
    assert model["frame_count"] == 30
    assert model["ratio"] == 2.0
    assert np.isclose(model["threshold"], 0.2)

    fallback = _word_energy_noise_model(
        audio, 16000,
        Tier("words", 0.0, 0.4, [Interval(0.0, 0.4, "ni3")]),
        _args())
    assert fallback["source"] == "global_audio_percentile_15"
    assert fallback["percentile"] == 15


def test_merge_dilution_uses_premerge_lexical_core():
    words = Tier("words", 0.0, 0.6, [Interval(0.0, 0.05, "<sp0>"),
                                      Interval(0.05, 0.5, "ni3"),
                                      Interval(0.5, 0.6, "<sp0>")])
    words._word_energy_premerge_spans = {"0": [0.05, 0.15]}
    words._word_energy_merge_ledger = [{
        "lexical_ordinal": 0,
        "left_lexical_ordinal": 0,
        "operation": "energy_short_sp_merge",
        "policy": None,
    }]
    tg = TextGrid(0.0, 0.6, [words])
    tg._phone_lineage = _lineage([(0.05, 0.15, "ni3")])
    audio = np.full(9600, 0.1, dtype=np.float32)
    audio[800:2400] = 0.5
    audit = _word_energy_audit(words, _args(), audio, 16000, textgrid=tg)
    item = audit["items"][0]
    assert item["classification"] == "silence_merge_dilution"
    assert item["resulting_reason"] is None


def test_low_energy_and_lineage_hole_are_distinct_hard_results():
    words = Tier("words", 0.0, 0.3, [Interval(0.0, 0.1, "<sp0>"),
                                      Interval(0.1, 0.2, "ni3"),
                                      Interval(0.2, 0.3, "<sp0>")])
    audio = np.zeros(4800, dtype=np.float32)
    low_tg = TextGrid(0.0, 0.3, [words])
    low_tg._phone_lineage = _lineage([(0.1, 0.2, "ni3")])
    low = _word_energy_audit(words, _args(), audio, 16000, textgrid=low_tg)
    assert low["items"][0]["classification"] == "true_low_energy"
    assert low["items"][0]["resulting_reason"] == "word_in_silence"

    hole_tg = TextGrid(0.0, 0.3, [words])
    hole_tg._phone_lineage = _lineage(
        [(0.1, 0.2, "ni3")], status="rejected",
        reasons=[{"label": "a", "reason": "phone_lineage_ambiguous"}])
    unresolved = _word_energy_audit(words, _args(), audio, 16000,
                                    textgrid=hole_tg)
    assert unresolved["items"][0]["classification"] == "word_energy_evidence_unresolved"


def test_thirty_ms_source_phone_boundary_mismatch_requires_three_frames():
    words = Tier("words", 0.0, 0.2, [Interval(0.0, 0.03, "<sp0>"),
                                      Interval(0.03, 0.1, "ni3"),
                                      Interval(0.1, 0.2, "<sp0>")])
    tg = TextGrid(0.0, 0.2, [words])
    tg._phone_lineage = _lineage([(0.0, 0.1, "ni3")])
    audio = np.full(3200, 0.1, dtype=np.float32)
    audio[:1600] = 0.5
    audit = _word_energy_audit(words, _args(), audio, 16000, textgrid=tg)
    assert audit["items"][0]["classification"] == "word_energy_boundary_mismatch"

    words20 = Tier("words", 0.0, 0.2, [Interval(0.0, 0.02, "<sp0>"),
                                         Interval(0.02, 0.1, "ni3"),
                                         Interval(0.1, 0.2, "<sp0>")])
    tg20 = TextGrid(0.0, 0.2, [words20])
    tg20._phone_lineage = _lineage([(0.0, 0.1, "ni3")])
    audit20 = _word_energy_audit(words20, _args(), audio, 16000, textgrid=tg20)
    assert audit20["items"][0]["classification"] != "word_energy_boundary_mismatch"


def _ctc_anchored_word_grid(source_span):
    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.48, "<sp1>"),
        Interval(0.48, 0.63, "ni3"),
        Interval(0.63, 0.8, "<sp0>"),
    ])
    words._ctc_word_authority = [{
        "lexical_ordinal": 0,
        "text": "ni3",
        "ctc_span": [0.50, 0.63],
        "resolved_span": [0.48, 0.63],
    }]
    tg = TextGrid(0.0, 0.8, [words])
    tg._phone_lineage = _lineage([(*source_span, "ni3")])
    audio = np.full(12800, 0.1, dtype=np.float32)
    audio[int(0.48 * 16000):int(0.63 * 16000)] = 0.5
    return words, tg, audio


def test_valid_ctc_span_suppresses_mfa_overhang_boundary_mismatch():
    words, tg, audio = _ctc_anchored_word_grid((0.48, 0.71))
    audit = _word_energy_audit(words, _args(), audio, 16000, textgrid=tg)
    item = audit["items"][0]
    assert item["ctc_span"] == [0.5, 0.63]
    assert item["classification"] != "word_energy_boundary_mismatch"
    assert item["diagnostics"]["source_phone_outside_ctc_or_final"] is True


def test_valid_ctc_span_suppresses_source_phone_hole_hard_reason():
    words, tg, audio = _ctc_anchored_word_grid((0.52, 0.58))
    audit = _word_energy_audit(words, _args(), audio, 16000, textgrid=tg)
    item = audit["items"][0]
    assert item["classification"] != "word_energy_evidence_unresolved"
    assert item["diagnostics"]["phone_hole_or_audio_hole_suppressed"] is True
    assert "phone_hole" in item["lineage"]["diagnostic_reasons"]


def test_english_nvv_adjacency_crosses_silence_without_skipping_chinese_owner():
    words = Tier("words", 0.0, 0.4, [
        Interval(0.0, 0.1, "OK"), Interval(0.1, 0.2, "<sp1>"),
        Interval(0.2, 0.3, "you3"), Interval(0.3, 0.4, "<LAUGHTER>")])
    tg = TextGrid(0.0, 0.4, [words])
    tg._phone_lineage = _lineage([
        (0.0, 0.1, "OK"), (0.2, 0.3, "you3"), (0.3, 0.4, "<LAUGHTER>")])
    audit = _word_energy_audit(
        words, _args(), np.full(6400, 0.5, dtype=np.float32), 16000,
        textgrid=tg)
    items = {item["word"]: item for item in audit["items"]}
    assert items["OK"]["classification"] == "not_applicable"
    assert items["<LAUGHTER>"]["classification"] == "not_applicable"
    assert items["you3"]["english_nvv_adjacency"]["previous"] == "OK"
    assert items["you3"]["english_nvv_adjacency"]["crossed_silence"] is True


def test_disabled_detector_keeps_diagnostic_but_no_filter_result():
    words = Tier("words", 0.0, 0.3, [Interval(0.0, 0.1, "<sp0>"),
                                      Interval(0.1, 0.2, "ni3"),
                                      Interval(0.2, 0.3, "<sp0>")])
    tg = TextGrid(0.0, 0.3, [words])
    tg._phone_lineage = _lineage([(0.1, 0.2, "ni3")])
    audit = _word_energy_audit(words, _args(enable_word_in_silence_filter=False),
                               np.zeros(4800, dtype=np.float32), 16000,
                               textgrid=tg)
    assert audit["enabled"] is False
    assert audit["items"][0]["classification"] == "true_low_energy"


def test_memory_and_wav_audio_use_the_same_energy_helper(tmp_path):
    words = Tier("words", 0.0, 0.3, [
        Interval(0.0, 0.1, "<sp0>"), Interval(0.1, 0.2, "ni3"),
        Interval(0.2, 0.3, "<sp0>"),
    ])
    tg = TextGrid(0.0, 0.3, [words])
    tg._phone_lineage = _lineage([(0.1, 0.2, "ni3")])
    audio = np.full(4800, 0.1, dtype=np.float32)
    audio[1600:3200] = 0.5
    path = tmp_path / "energy.wav"
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())
    wav_audio, wav_sr = load_audio(path)
    memory = _word_energy_audit(words, _args(), audio, 16000, textgrid=tg)
    from_wav = _word_energy_audit(words, _args(), wav_audio, wav_sr, textgrid=tg)
    assert memory["noise_model"]["source"] == from_wav["noise_model"]["source"]
    assert memory["items"][0]["classification"] == from_wav["items"][0]["classification"]
    assert np.isclose(memory["noise_model"]["threshold"],
                      from_wav["noise_model"]["threshold"], rtol=1e-3)
