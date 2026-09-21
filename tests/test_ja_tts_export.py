import json
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.ja_tts_export import export_tts_artifacts, build_training_record, handle_tts
from scripts.ja_audio import make_audio_receipt


def _wav(path: Path, frames: int = 16000) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
        out.writeframes((np.zeros(frames, dtype="<i2")).tobytes())


def _alignment():
    return {
        "schema": "ja-en-alignment-v2", "uid": "u1",
        "words": [{"unit_id": "w0", "text": "さくら", "language": "ja", "start_sample": 0, "end_sample": 8000}],
        "phones": [
            {"unit_id": "w0", "phone_id": "p0", "phone": "s", "native_phone": "s", "language": "ja", "start_sample": 0, "end_sample": 4000, "raw_interval_id": 1, "alias": "ju_000000", "mora_ids": ["m0"]},
            {"unit_id": "w0", "phone_id": "p1", "phone": "a", "native_phone": "a", "language": "ja", "start_sample": 4000, "end_sample": 8000, "raw_interval_id": 2, "alias": "ju_000000", "mora_ids": ["m0"]},
        ],
        "languages": [{"language": "ja", "unit_ids": ["w0"]}],
        "mora_graph": {"moras": [{"mora_id": "m0", "surface": "さ", "phone_ids": ["p0", "p1"]}], "relations": [{"mora_id": "m0", "phone_id": "p0"}, {"mora_id": "m0", "phone_id": "p1"}]},
    }


def test_export_writes_native_jsonl_and_three_textgrid_tiers(tmp_path: Path):
    train, alignment = tmp_path / "train.wav", tmp_path / "alignment.wav"
    _wav(train); _wav(alignment)
    alignment_data = _alignment()
    alignment_data.update({"selected_reading": "さくら", "locked_aliases": [{"alias": "ju_000000", "pronunciation": ["s", "a"]}], "native_inventory": {"ja": ["s", "a"], "en": []}, "raw_mfa": {"phones": [{"phone_id": "p0", "raw_interval_id": 1, "unit_id": "w0"}, {"phone_id": "p1", "raw_interval_id": 2, "unit_id": "w0"}]}, "reading_evidence": {"selected_reading": "さくら", "status": "manual_verified"}, "partition": {"verified": ["u1"], "rejected": [], "unresolved": []}})
    receipt = make_audio_receipt("u1", train, train, alignment, alignment_transform={"method": "identity_fixture_v1", "source_start": 0, "source_end": 16000, "output_start": 0, "output_frames": 16000})
    record = build_training_record(alignment_data, train_wav=train, alignment_wav=alignment, speaker="spk", audio_receipt=receipt)
    assert record["schema"] == "tts-training-record-v1"
    assert [p["duration_samples"] for p in record["phones"]] == [4000, 4000]
    assert record["quality_masks"]["accent_predicted_known_mask"] is False
    paths = export_tts_artifacts(record, tmp_path / "out")
    row = json.loads((tmp_path / "out" / "tts_training_records.jsonl").read_text())
    assert row["phones"][0]["native_phone"] == "s"
    grid = (tmp_path / "out" / "u1.TextGrid").read_text()
    assert all(f'name = "{tier}"' in grid for tier in ("words", "phones", "language"))
    assert paths["textgrid"].is_file()


def test_tts_stage_consumes_authoritative_alignment_and_declares_real_outputs(tmp_path: Path):
    train, alignment_wav = tmp_path / "train.wav", tmp_path / "alignment.wav"
    _wav(train); _wav(alignment_wav)
    alignment = _alignment()
    alignment.update({"uid": "stage-1", "selected_reading": "さくら", "locked_aliases": [{"alias": "ju_000000", "pronunciation": ["s", "a"]}], "native_inventory": {"ja": ["s", "a"], "en": []}, "raw_mfa": {"textgrid_path": str(tmp_path / "stage" / "stage-1.TextGrid"), "phones": [{"phone_id": "p0", "raw_interval_id": 1, "unit_id": "w0"}, {"phone_id": "p1", "raw_interval_id": 2, "unit_id": "w0"}]}, "reading_evidence": {"selected_reading": "さくら", "status": "manual_verified"}, "partition": {"verified": ["stage-1"], "rejected": [], "unresolved": []}, "train_wav": str(train), "alignment_wav": str(alignment_wav)})
    alignment["audio_receipt"] = make_audio_receipt("stage-1", train, train, alignment_wav, alignment_transform={"method": "identity_fixture_v1", "source_start": 0, "source_end": 16000, "output_start": 0, "output_frames": 16000})
    source = tmp_path / "alignment.jsonl"; source.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    stage = tmp_path / "stage"; stage.mkdir()
    result = handle_tts({"tts": {"alignment_jsonl": str(source)}}, stage)
    assert result.status == "COMPLETE"
    receipt = json.loads((stage / "receipt.json").read_text())
    assert {Path(row["path"]).name for row in receipt["outputs"]} == {"tts_training_records.jsonl", "stage-1.TextGrid"}


def test_export_rejects_missing_authoritative_alias_and_audio_receipts(tmp_path: Path):
    train, alignment = tmp_path / "train.wav", tmp_path / "alignment.wav"
    _wav(train); _wav(alignment)
    with pytest.raises(ValueError, match="authoritative audio-transform"):
        build_training_record(_alignment(), train_wav=train, alignment_wav=alignment)
