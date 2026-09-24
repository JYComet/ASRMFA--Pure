"""Regressions from the Chinese postprocessing pipeline on Japanese data."""

import wave

import numpy as np

from scripts.ja_audio import audio_stage, inspect_wav
from scripts.ja_tts_export import build_five_track_tiers


def _record():
    return {
        "sample_rate": 16000, "frame_count": 32000,
        "text_layers": {"orig_text": "さくら、world。"},
        "words": [
            {"unit_id": "ja", "source_text": "さくら", "kana": "サクラ", "language": "ja", "char_span": [0, 3], "start_sample": 8000, "end_sample": 16000},
            {"unit_id": "en", "source_text": "world", "kana": "", "language": "en", "char_span": [4, 9], "start_sample": 20000, "end_sample": 28000},
        ],
        "native_phones": [
            {"phone_id": "p0", "language": "ja", "native_phone": "s", "phone_kana": "サ", "phone_tone": "H", "start_sample": 8000, "end_sample": 16000},
            {"phone_id": "p1", "language": "en", "native_phone": "W", "phone_kana": "", "phone_tone": "NA", "start_sample": 20000, "end_sample": 28000},
        ],
    }


def test_reference_punctuation_replaces_local_sp_and_leading_sp1():
    tiers = {tier["name"]: tier["intervals"] for tier in build_five_track_tiers(_record())}
    assert [row["text"] for row in tiers["original_text"]] == ["<sp1>", "さくら", "、", "world", "。"]
    assert [row["text"] for row in tiers["mfa_phone"]] == ["<sp1>", "ja:s", "<sp1>", "en:W", "<sp1>"]
    assert [row["text"] for row in tiers["kana"]] == ["<sp1>", "サクラ", "、", "", "。"]
    assert all((left["end_sample"] == right["start_sample"]) for left, right in zip(tiers["original_text"], tiers["original_text"][1:]))


def test_no_reference_punctuation_does_not_invent_one():
    record = _record()
    record["text_layers"]["orig_text"] = "さくら world"
    record["words"][1]["char_span"] = [4, 9]
    labels = [row["text"] for row in build_five_track_tiers(record)[0]["intervals"]]
    assert labels[2] == "<sp1>"


def test_audio_stage_adds_half_second_edge_silence(tmp_path):
    source = tmp_path / "source.wav"
    signal = np.full(16000, 1000, dtype="<i2")
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(signal.tobytes())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"uid":"u1","wav":"' + str(source) + '","text":"あ"}\n', encoding="utf-8")
    stage_dir = tmp_path / "stages" / "audio"
    result = audio_stage({"input_manifest": str(manifest)}, stage_dir)
    assert result.status == "COMPLETE"
    for path in (next(stage_dir.glob("train__*.wav")), next(stage_dir.glob("alignment__*.wav"))):
        assert inspect_wav(path)["frames"] == 32000
        with wave.open(str(path), "rb") as inp:
            samples = np.frombuffer(inp.readframes(inp.getnframes()), dtype="<i2")
        assert np.all(samples[:8000] == 0) and np.all(samples[-8000:] == 0)



def test_edge_receipt_replays_both_axes_and_detects_tampering(tmp_path):
    from scripts.ja_audio import make_audio_receipt, prepare_alignment_wav, prepare_training_wav
    from scripts.verify_ja_en_tts import _replay_pcm16_transform
    source = tmp_path / "source.wav"
    values = np.full(16000, 1000, dtype="<i2")
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(values.tobytes())
    train = tmp_path / "train.wav"
    alignment = tmp_path / "alignment.wav"
    train_result = prepare_training_wav(source, train, normalize_edges=True)
    align_result = prepare_alignment_wav(source, alignment, normalize_edges=True)
    receipt = make_audio_receipt("u1", source, train, alignment,
                                 train_transform=train_result["sample_transform"],
                                 alignment_transform=align_result["sample_transform"])
    assert _replay_pcm16_transform(str(source), str(train), receipt["train_transform"])
    assert _replay_pcm16_transform(str(source), str(alignment), receipt["alignment_transform"])
    assert receipt["sample_transform"]["output_start"] == 8000
    tampered = dict(receipt["alignment_transform"])
    tampered["edge_normalization"] = {**tampered["edge_normalization"], "pad_before_frames": 7999}
    try:
        valid = _replay_pcm16_transform(str(source), str(alignment), tampered)
    except ValueError:
        valid = False
    assert not valid



def test_v2_verifier_accepts_edge_recipe_and_checks_shifted_source_axis(tmp_path):
    from scripts.ja_audio import make_audio_receipt, prepare_alignment_wav, prepare_training_wav
    from scripts.verify_ja_en_tts import verify_training_record
    source = tmp_path / "source.wav"
    values = np.full(16000, 1000, dtype="<i2")
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(values.tobytes())
    train = tmp_path / "train.wav"
    alignment = tmp_path / "alignment.wav"
    tr = prepare_training_wav(source, train, normalize_edges=True)
    ar = prepare_alignment_wav(source, alignment, normalize_edges=True)
    receipt = make_audio_receipt("u1", source, train, alignment,
                                 train_transform=tr["sample_transform"],
                                 alignment_transform=ar["sample_transform"])
    record = _record()
    record.update(schema="tts-training-record-v2", uid="u1",
                  train_wav=receipt["train"], alignment_wav=receipt["alignment"],
                  audio_transform=receipt, source_sample_rate=16000,
                  train_sample_rate=16000, quality_masks={},
                  moras=[], basic_phones=[], duration_groups=[])
    record["words"] = [{"unit_id": "ja", "source_text": "あ", "kana": "ア", "language": "ja",
                        "start_sample": 8000, "end_sample": 24000}]
    record["native_phones"] = [{"phone_id": "p0", "unit_id": "ja", "language": "ja",
                                "native_phone": "a", "phone_kana": "ア", "phone_tone": "UNK",
                                "start_sample": 8000, "end_sample": 24000,
                                "train_start_sample": 8000, "train_end_sample": 24000,
                                "source_start_sample": 0, "source_end_sample": 16000}]
    errors = verify_training_record(record)
    assert not any(error["code"] in {"schema_unknown", "audio_transform_invalid"} for error in errors), errors
    record["native_phones"][0]["source_start_sample"] = 8000
    errors = verify_training_record(record)
    assert any(error["code"] == "audio_transform_invalid" for error in errors)



def test_independent_grid_audit_rejects_missing_sp_and_wrong_punctuation(tmp_path):
    from scripts.ja_tts_export import render_textgrid
    from scripts.verify_ja_en_tts import _verify_v2_textgrid
    record = _record()
    record["uid"] = "u1"
    path = tmp_path / "u1.TextGrid"
    path.write_text(render_textgrid(record), encoding="utf-8")
    errors = []
    _verify_v2_textgrid(record, tmp_path, errors)
    assert errors == []
    path.write_text(path.read_text(encoding="utf-8").replace('text = "、"', 'text = "<sp1>"', 1), encoding="utf-8")
    errors = []
    _verify_v2_textgrid(record, tmp_path, errors)
    assert any(error["code"] == "tts_invalid" for error in errors)



def test_no_pause_punctuation_stays_in_original_text_without_carving_word():
    record = _record()
    record["frame_count"] = 28000
    record["words"][1]["start_sample"] = 16000
    record["native_phones"][1]["start_sample"] = 16000
    record["text_layers"]["orig_text"] = "さくら、world。"
    tiers = {tier["name"]: tier["intervals"] for tier in build_five_track_tiers(record)}
    assert [(row["start_sample"], row["end_sample"], row["text"]) for row in tiers["original_text"]] == [
        (0, 8000, "<sp1>"), (8000, 16000, "さくら、"), (16000, 28000, "world。")]
    assert [(row["start_sample"], row["end_sample"]) for row in tiers["mfa_phone"] if row.get("segment_id")] == [
        (8000, 16000), (16000, 28000)]



def test_48k_train_and_16k_alignment_share_source_rate_edge_scan(tmp_path):
    from scripts.ja_audio import prepare_alignment_wav, prepare_training_wav
    from scripts.verify_ja_en_tts import _replay_pcm16_transform
    source = tmp_path / "source48.wav"
    values = np.concatenate([np.zeros(4800, dtype="<i2"),
                             np.full(4800, 1000, dtype="<i2"),
                             np.zeros(4800, dtype="<i2")])
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(48000); out.writeframes(values.tobytes())
    train = tmp_path / "train.wav"
    alignment = tmp_path / "alignment.wav"
    tr = prepare_training_wav(source, train, normalize_edges=True)["sample_transform"]
    ar = prepare_alignment_wav(source, alignment, normalize_edges=True)["sample_transform"]
    assert tr["edge_normalization"] == ar["edge_normalization"]
    assert abs(tr["output_start"] / 48000 - ar["output_start"] / 16000) <= 1 / 16000
    assert _replay_pcm16_transform(str(source), str(train), tr)
    assert _replay_pcm16_transform(str(source), str(alignment), ar)



def test_japanese_reference_quotes_keep_leading_sp_and_terminal_marks():
    record = _record()
    record["frame_count"] = 20000
    record["text_layers"]["orig_text"] = "「さくら」。"
    record["words"] = [dict(record["words"][0], char_span=[1, 4])]
    record["native_phones"] = record["native_phones"][:1]
    tiers = {tier["name"]: tier["intervals"] for tier in build_five_track_tiers(record)}
    assert [row["text"] for row in tiers["original_text"]] == ["<sp1>", "「さくら", "」。"]
    assert [row["text"] for row in tiers["kana"]] == ["<sp1>", "「サクラ", "」。"]
