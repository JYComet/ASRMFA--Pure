import json
import hashlib
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.ja_tts_export import export_tts_artifacts, build_training_record, handle_tts
from scripts.ja_audio import make_audio_receipt
from scripts.ja_en_schema import make_receipt


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
    raw_grid = tmp_path / "raw.TextGrid"; raw_grid.write_text('File type = "ooTextFile"\n', encoding="utf-8")
    alignment = _alignment()
    alignment.update({"uid": "stage-1", "selected_reading": "さくら", "locked_aliases": [{"alias": "ju_000000", "unit_id": "w0", "pronunciation": ["s", "a"]}], "native_inventory": {"ja": ["s", "a"], "en": []}, "raw_mfa": {"runs": [{"run_id": "ja-fixture", "raw_textgrid": {"path": str(raw_grid), "sha256": hashlib.sha256(raw_grid.read_bytes()).hexdigest()}}]}, "reading_evidence": {"selected_reading": "さくら", "status": "manual_verified"}, "partition": {"verified": ["w0"], "rejected": [], "unresolved": []}, "train_wav": str(train), "alignment_wav": str(alignment_wav)})
    alignment["audio_receipt"] = make_audio_receipt("stage-1", train, train, alignment_wav, alignment_transform={"method": "identity_fixture_v1", "source_start": 0, "source_end": 16000, "output_start": 0, "output_frames": 16000})
    alignment["train_wav"] = alignment["audio_receipt"]["train"]
    alignment["alignment_wav"] = alignment["audio_receipt"]["alignment"]
    alignment["schema"] = "ja-prosody-alignment-v1"
    alignment["native_phones"] = alignment.pop("phones")
    for index, phone in enumerate(alignment["native_phones"]):
        mora_id, basic_id, kana = ("m0", "bp0", "さ") if index == 0 else ("m1", "bp1", "く")
        phone.update({"token_id": "w0", "mora_ids": [mora_id], "basic_phone_ids": [basic_id], "phone_kana": kana, "phone_tone": "H"})
    alignment.update({
        "moras": [{"mora_id": "m0", "kana": "さ", "tone": "H"}, {"mora_id": "m1", "kana": "く", "tone": "H"}],
        "basic_phones": [{"basic_phone_id": "bp0", "mora_id": "m0", "symbol": "s"}, {"basic_phone_id": "bp1", "mora_id": "m1", "symbol": "a"}],
        "duration_groups": [], "tone_sources": [{"entry_id": "fixture"}],
        "mora_graph": {"moras": [{"mora_id": "m0"}, {"mora_id": "m1"}], "relations": [{"mora_id": "m0", "phone_id": "p0"}, {"mora_id": "m1", "phone_id": "p1"}]},
        "frontend": {"fixture": True}, "model_ids": {"mfa": "fixture"}, "dict_ids": {"ja": "fixture"}, "seams": [],
        "source_receipt": make_receipt(stage="merge", status="COMPLETE"),
    })
    source = tmp_path / "alignment.jsonl"; source.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    stage = tmp_path / "stage"; stage.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{"uid": "stage-1", "wav": str(train), "text": "さくら", "reading_lock": {"uid": "stage-1", "status": "COMPLETE", "selected_reading": "さくら"}}]}), encoding="utf-8")
    result = handle_tts({"input_manifest": str(manifest), "workspace": str(tmp_path), "tts": {"alignment_jsonl": str(source)}, "stage_inputs": {"tts": {"alignment_jsonl": str(source)}}}, stage)
    assert result.status == "COMPLETE"
    receipt = json.loads((stage / "receipt.json").read_text())
    assert {Path(row["path"]).name for row in receipt["outputs"]} == {"tts_training_records.jsonl", "stage-1.TextGrid"}
    tampered = {**alignment, "basic_phones": []}
    source.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    rejected = handle_tts({"input_manifest": str(manifest), "workspace": str(tmp_path), "tts": {"alignment_jsonl": str(source)}, "stage_inputs": {"tts": {"alignment_jsonl": str(source)}}}, tmp_path / "tampered")
    assert rejected.status == "REJECTED"
    source.write_text(json.dumps(alignment) + "\n", encoding="utf-8")


@pytest.mark.parametrize("tamper", ("rebind", "projection", "coverage", "duration"))
def test_tts_handler_rejects_semantically_inconsistent_complete_prosody_graph(tmp_path: Path, tamper: str):
    """All referenced IDs can exist while immutable prosody relations disagree."""
    # Reuse the complete registered-handler fixture and transform only one
    # immutable semantic claim at a time.
    test_tts_stage_consumes_authoritative_alignment_and_declares_real_outputs(tmp_path)
    source = tmp_path / "alignment.jsonl"
    artifact = json.loads(source.read_text(encoding="utf-8"))
    if tamper == "rebind":
        artifact["native_phones"][0].update({"mora_ids": ["m1"], "basic_phone_ids": ["bp1"], "phone_kana": "く"})
    elif tamper == "projection":
        artifact["native_phones"][0].update({"phone_kana": "tampered", "phone_tone": "L"})
    elif tamper == "coverage":
        artifact["mora_graph"]["relations"].pop()
    else:
        artifact["native_phones"] = [artifact["native_phones"][0]]
        artifact["native_phones"][0].update({"mora_ids": ["m0", "m1"], "basic_phone_ids": ["bp0", "bp1"], "phone_kana": "さ|く", "phone_tone": "H|H"})
        artifact["mora_graph"]["relations"] = [{"mora_id": "m0", "phone_id": "p0"}, {"mora_id": "m1", "phone_id": "p0"}]
        artifact["duration_groups"] = [{"duration_group_id": "duration-group-p0", "native_phone_id": "p0", "basic_phone_ids": ["bp0"], "total_duration_samples": 1}]
    source.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    assert handle_tts({"input_manifest": str(manifest), "workspace": str(tmp_path), "tts": {"alignment_jsonl": str(source)}, "stage_inputs": {"tts": {"alignment_jsonl": str(source)}}}, tmp_path / f"bad-{tamper}").status == "REJECTED"


def test_tts_stage_rejects_merge_v3_on_production_path(tmp_path: Path):
    source = tmp_path / "merge.jsonl"
    source.write_text('{"schema":"ja-en-alignment-v3"}\n', encoding="utf-8")
    stage = tmp_path / "stage"; stage.mkdir()
    assert handle_tts({"tts": {"alignment_jsonl": str(source)}, "stage_inputs": {"tts": {}}}, stage).status == "REJECTED"


def test_export_rejects_missing_authoritative_alias_and_audio_receipts(tmp_path: Path):
    train, alignment = tmp_path / "train.wav", tmp_path / "alignment.wav"
    _wav(train); _wav(alignment)
    with pytest.raises(ValueError, match="authoritative audio-transform"):
        build_training_record(_alignment(), train_wav=train, alignment_wav=alignment)
