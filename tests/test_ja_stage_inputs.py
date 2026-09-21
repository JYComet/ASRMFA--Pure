from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import pytest

from scripts.ja_en_schema import JAContractError, artifact_record
from scripts.ja_en_stage_inputs import (
    assemble_prosody_rows,
    assemble_tts_rows,
    prepare_alignment_requests,
    prepare_anchor_requests,
    prepare_merge_requests,
)
from scripts.ja_prosody import handle_prosody
from scripts.ja_en_schema import stable_digest


def _wav(path: Path, frames: int = 3200) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * frames)
    return path


def _unit(uid: str, language: str, index: int, start: int, end: int, text: str, alias: str) -> dict:
    return {
        "unit_id": f"{uid}:unit_{index:04d}",
        "text": text,
        "language": language,
        "route": language,
        "char_span": [start, end],
        "start_sample": start * 100,
        "end_sample": end * 100,
        "alias": alias,
        "pronunciation": ["a" if language == "en" else "a"],
    }


def _fixture(tmp_path: Path) -> tuple[dict, list[dict]]:
    rows = []
    specs = [("u-ja", "東京", [("ja", "東京", "ju_000001")]),
             ("u-en", "hello", [("en", "hello", "eu_000002")]),
             ("u-mix", "東京hello", [("ja", "東京", "ju_000003"), ("en", "hello", "eu_000004")])]
    for uid, text, parts in specs:
        wav_path = _wav(tmp_path / f"{uid}.wav", 3200)
        units = []
        cursor = 0
        for idx, (lang, unit_text, alias) in enumerate(parts):
            units.append(_unit(uid, lang, idx, cursor, cursor + len(unit_text), unit_text, alias))
            cursor += len(unit_text)
        receipt = {
            "schema": "audio-transform-receipt-v2", "uid": uid,
            "source": artifact_record(wav_path), "train": artifact_record(wav_path),
            "alignment": artifact_record(wav_path),
            "sample_transform": {"kind": "identity", "source_rate": 16000, "target_rate": 16000},
        }
        rows.append({"uid": uid, "text": text, "wav": str(wav_path), "speaker": "fixture"})
        # These are the same file kinds emitted by W1/W2.  The manifest above
        # intentionally contains no frozen reading, frontend, semantic, or
        # transform evidence.
        stage_root = tmp_path / "stages"
        for section in ("audio", "reading", "frontend", "semantic"):
            (stage_root / section).mkdir(parents=True, exist_ok=True)
        with (stage_root / "audio" / "audio_transform_receipts.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"uid": uid, "audio_receipt": receipt}, ensure_ascii=False) + "\n")
        with (stage_root / "reading" / "locked_readings.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"uid": uid, "reading_lock": {"schema": "ja-reading-selection-v2", "uid": uid,
                      "status": "COMPLETE", "selected_reading": text,
                      "candidates": [{"reading": text, "source": "fixture"}]}}, ensure_ascii=False) + "\n")
        with (stage_root / "frontend" / "frontend_contracts.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"uid": uid, "frontend": {"text_layer_digest": f"frontend-{uid}", "frontend_commit": "fixture"},
                      "lexical_units": units}, ensure_ascii=False) + "\n")
        with (stage_root / "semantic" / "semantic_graphs.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"uid": uid, "semantic_graph": {"schema": "ja-semantic-phone-graph-v1", "uid": uid,
                      "nodes": [], "edges": []}}, ensure_ascii=False) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": rows}, ensure_ascii=False), encoding="utf-8")
    config = {"input_manifest": str(manifest), "workspace": str(tmp_path),
              "asr": {"qwen_forced_aligner": {"model": "fixture-qwen", "device": "cpu", "dtype": "float32"}},
              "align": {"padding_ms": 100}, "merge": {"initial_padding_ms": 40}}
    return config, rows


def test_multi_uid_anchor_routes_and_alignment_crops(tmp_path: Path):
    config, _ = _fixture(tmp_path)
    anchors = prepare_anchor_requests(config, tmp_path)
    assert [row["uid"] for row in anchors] == ["u-ja", "u-en", "u-mix"]
    assert anchors[0]["route"] == "ja"
    assert anchors[1]["route"] == "en"
    assert anchors[2]["route"] == "mixed"

    alignments = prepare_alignment_requests(config, tmp_path, anchors)
    mixed = next(row for row in alignments if row["uid"] == "u-mix")
    assert [run["language"] for run in mixed["runs"]] == ["ja", "en"]
    assert mixed["runs"][1]["context_start_sample"] < mixed["runs"][1]["ownership_start_sample"]
    assert mixed["runs"][0]["global_offset_sample"] == mixed["runs"][0]["context_start_sample"]
    assert Path(mixed["runs"][0]["crop"]["path"]).read_bytes().startswith(b"RIFF")
    assert all(run["crop"]["sample_rate"] == 16000 and run["crop"]["channels"] == 1 for run in mixed["runs"])


def test_merge_expected_aliases_and_serializable_two_sided_retry(tmp_path: Path):
    config, _ = _fixture(tmp_path)
    anchors = prepare_anchor_requests(config, tmp_path)
    alignment_requests = prepare_alignment_requests(config, tmp_path, anchors)
    # This alias is intentionally absent from observed phones.  It must remain
    # in the frozen expected set passed to the merge stage.
    merge = prepare_merge_requests(config, tmp_path, alignment_requests)
    mixed = next(row for row in merge if row["uid"] == "u-mix")
    assert set(mixed["expected_languages"]) == {"ju_000003", "eu_000004"}
    assert set(mixed["expected_aliases"]) == {"ju_000003", "eu_000004"}
    assert mixed["rerun_plan"]["initial_padding_samples"] == 1280
    assert mixed["rerun_plan"]["left"]["padding_samples"] == 2560
    json.dumps(mixed["rerun_plan"])
    assert mixed["expected_units"]


def test_merge_request_preserves_locked_token_alias_and_all_semantic_graphs(tmp_path: Path):
    graph = {
        "schema": "ja-semantic-phone-graph-v2", "uid": "u1", "token_id": "tok-1",
        "native_phone_templates": [{"native_phone_id": "np0", "native_phone": "a",
                                    "token_id": "tok-1", "basic_phone_ids": ["bp0"],
                                    "mora_ids": ["m0"], "transform": "identity"}],
    }
    request = {
        "uid": "u1", "runs": [{"run_id": "ja-run", "language": "ja",
            "ownership_start_sample": 0, "ownership_end_sample": 100,
            "aliases": [{"alias": "ju_000001", "unit_id": "unit-1", "token_id": "tok-1",
                         "language": "ja", "pronunciation": ["a"]}]}],
        "semantic_graphs": [graph],
    }
    merge = prepare_merge_requests({"merge": {"initial_padding_ms": 40}}, tmp_path, [request])[0]
    assert merge["locked_aliases"] == [{"alias": "ju_000001", "unit_id": "unit-1", "token_id": "tok-1",
                                         "language": "ja", "pronunciation": ["a"]}]
    assert merge["semantic_graphs"] == [graph]


def _merged_row(tmp_path: Path, uid: str = "u-ja") -> dict:
    wav_path = _wav(tmp_path / f"{uid}-merged.wav", 3200)
    alias = "ju_000001"
    phone = {"phone_id": f"{uid}:phone_0", "uid": uid, "unit_id": f"{uid}:unit_0000",
             "alias": alias, "language": "ja", "native_phone": "t", "phone": "t",
             "start_sample": 100, "end_sample": 300, "raw_interval_index": 3,
             "raw_artifact": {"path": str(wav_path), "sha256": artifact_record(wav_path)["sha256"]}}
    audio = artifact_record(wav_path)
    with wave.open(str(wav_path), "rb") as handle:
        audio.update({"sample_rate": handle.getframerate(), "channels": handle.getnchannels(),
                      "sample_width": handle.getsampwidth(), "frames": handle.getnframes()})
    receipt = {"schema": "audio-transform-receipt-v2", "uid": uid, "source": audio,
               "train": audio, "alignment": audio,
               "sample_transform": {"kind": "identity", "source_rate": 16000, "target_rate": 16000}}
    return {"schema": "ja-en-alignment-v3", "uid": uid,
            "words": [{"unit_id": f"{uid}:unit_0000", "alias": alias, "text": "東京",
                        "language": "ja", "start_sample": 0, "end_sample": 1000}],
            # ``phones`` remains only as the legacy TTS-v1 adapter input; the
            # production identity validator consumes native_phones.
            "phones": [phone], "native_phones": [dict(phone, token_id=f"{uid}:unit_0000")], "languages": ["ja"], "durations": [200],
            "locked_aliases": [{"alias": alias, "unit_id": f"{uid}:unit_0000", "language": "ja", "pronunciation": ["t"]}],
            "native_inventory": {"language": "ja", "phones": ["t"], "source": "fixture"},
            "raw_mfa": {"runs": [{"run_id": "ja-run", "raw_textgrid": phone["raw_artifact"]}]},
            "reading_evidence": {"selected_reading": "東京", "source": "fixture"},
            "selected_reading": "東京", "partition": {"verified": [alias], "rejected": [], "unresolved": []},
            "mora_graph": {"moras": [], "relations": []}, "audio_receipt": receipt,
            "alignment_wav": audio, "train_wav": audio,
            "frontend": {"text_layer_digest": "x", "frontend_commit": "fixture"},
            "model_ids": {"qwen": "fixture-qwen", "mfa": "fixture-mfa"},
            "dict_ids": {"ja": "fixture-dict"}, "seams": [],
            "source_receipt": {"schema": "ja-pipeline-receipt-v1", "stage": "audio", "status": "COMPLETE",
                               "inputs": {"artifacts": []}, "outputs": [audio], "params": {}, "tools": [], "commands": [], "errors": []}}


def test_tts_assembly_rejects_pre_prosody_rows(tmp_path: Path):
    config, _ = _fixture(tmp_path)
    row = _merged_row(tmp_path)
    with pytest.raises((JAContractError, ValueError)):
        assemble_tts_rows(config, tmp_path, [row])


def test_alignment_rejects_tampered_upstream_audio_hash(tmp_path: Path):
    config, rows = _fixture(tmp_path)
    anchors = prepare_anchor_requests(config, tmp_path)
    path = Path(rows[0]["wav"])
    path.write_bytes(path.read_bytes() + b"tampered")
    rejected = prepare_alignment_requests(config, tmp_path, anchors)
    bad = next(row for row in rejected if row["uid"] == "u-ja")
    assert bad["status"] == "REJECTED"
    assert bad["errors"][0]["code"] == "receipt_hash_mismatch"


def _prosody_graph(uid: str = "mixed-1") -> dict:
    reading = "コー"
    digest = stable_digest(reading)
    return {
        "schema": "ja-semantic-phone-graph-v2", "uid": uid, "token_id": "tok-1",
        "locked_reading": reading, "locked_reading_digest": digest,
        "mora_nodes": [
            {"mora_id": "m-1", "kana": "コ", "kind": "regular", "mora_index": 0,
             "accent_phrase_id": "ap-1", "mora_index_in_phrase": 1},
            {"mora_id": "m-2", "kana": "ー", "kind": "long_extension", "mora_index": 1,
             "accent_phrase_id": "ap-1", "mora_index_in_phrase": 2},
        ],
        "basic_phone_nodes": [
            {"basic_phone_id": "bp-1", "mora_id": "m-1", "realization": "observed"},
            {"basic_phone_id": "bp-2", "mora_id": "m-2", "realization": "merged"},
        ],
        "native_phone_templates": [
            {"native_phone_id": "np-1", "native_phone": "k", "token_id": "tok-1", "alias": "ju-1",
             "mora_ids": ["m-1"], "basic_phone_ids": ["bp-1"], "transform": "identity"},
            {"native_phone_id": "np-2", "native_phone": "o\u02d0", "token_id": "tok-1", "alias": "ju-1",
             "mora_ids": ["m-1", "m-2"], "basic_phone_ids": ["bp-1", "bp-2"], "transform": "long_vowel_merge"},
        ],
    }


def _prosody_frontend(reading: str = "コー") -> dict:
    digest = stable_digest(reading)
    return {"accent_evidence_valid": True, "locked_reading_digest": digest,
            "accent_evidence": {
                "adapter_version": "fixture-v1", "provider_evidence_sha256": "provider-sha",
                "unit_evidence_sha256": "unit-sha",
                "provider_identity": {"provider": "fixture", "provider_revision": "r1"},
                "accent_phrases": [{"accent_phrase_id": "ap-1", "mora_count": 2, "nucleus": 1}],
                "moras": [
                    {"accent_phrase_id": "ap-1", "mora_index_in_phrase": 1, "mora_count": 2, "nucleus": 1},
                    {"accent_phrase_id": "ap-1", "mora_index_in_phrase": 2, "mora_count": 2, "nucleus": 1},
                ],
            }}


def _prosody_alignment(uid: str = "mixed-1") -> dict:
    phones = []
    for index, (template, label, mora_ids, basic_ids) in enumerate((
        ("np-1", "k", ["m-1"], ["bp-1"]),
        ("np-1", "k", ["m-1"], ["bp-1"]),
        ("np-2", "o\u02d0", ["m-1", "m-2"], ["bp-1", "bp-2"]),
        ("np-2", "o\u02d0", ["m-1", "m-2"], ["bp-1", "bp-2"]),
    )):
        phones.append({"phone_id": f"p-{index}", "uid": uid, "token_id": "tok-1", "alias": "ju-1",
                       "language": "ja", "native_phone": label, "native_phone_template_id": template,
                       "mora_ids": mora_ids, "basic_phone_ids": basic_ids,
                       "transform": "identity" if template == "np-1" else "long_vowel_merge",
                       "start_sample": index * 100, "end_sample": (index + 1) * 100,
                       "raw_interval_id": index + 1})
    return {"schema": "ja-en-alignment-v3", "uid": uid, "words": [], "native_phones": phones}


def _write_prosody_sources(workspace: Path, alignment: dict) -> dict:
    for stage in ("semantic", "frontend", "reading", "merge"):
        (workspace / "stages" / stage).mkdir(parents=True, exist_ok=True)
    (workspace / "stages" / "semantic" / "semantic_graphs.jsonl").write_text(
        json.dumps({"uid": alignment["uid"], "semantic_graph": _prosody_graph(alignment["uid"])}, ensure_ascii=False) + "\n", encoding="utf-8")
    (workspace / "stages" / "frontend" / "frontend_contracts.jsonl").write_text(
        json.dumps({"uid": alignment["uid"], "frontend": _prosody_frontend()}, ensure_ascii=False) + "\n", encoding="utf-8")
    (workspace / "stages" / "reading" / "locked_readings.jsonl").write_text(
        json.dumps({"uid": alignment["uid"], "reading_lock": {"uid": alignment["uid"], "selected_reading": "コー"}}, ensure_ascii=False) + "\n", encoding="utf-8")
    source = workspace / "stages" / "merge" / "merged_alignments.jsonl"
    source.write_text(json.dumps(alignment, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"prosody": {"algorithm_version": "ja-mora-tone-v1", "alignment_jsonl": str(source)}}


def test_assemble_prosody_rows_joins_by_uid_token_and_alias(tmp_path: Path):
    alignment = _prosody_alignment()
    config = _write_prosody_sources(tmp_path, alignment)
    rows = assemble_prosody_rows(config, tmp_path, [alignment])
    assert [row["uid"] for row in rows] == ["mixed-1"]
    assert rows[0]["schema"] == "ja-prosody-alignment-v1"
    assert [row["raw_interval_id"] for row in rows[0]["native_phones"]] == [1, 2, 3, 4]
    assert rows[0]["native_phones"][2]["phone_tone"] == "H|L"


def test_prosody_handler_preserves_each_failed_uid_in_ledger(tmp_path: Path):
    valid = _prosody_alignment("good")
    invalid = _prosody_alignment("bad")
    config = _write_prosody_sources(tmp_path, valid)
    source = Path(config["prosody"]["alignment_jsonl"])
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (valid, invalid)) + "\n", encoding="utf-8")
    stage_dir = tmp_path / "stages" / "prosody"
    result = handle_prosody(config, stage_dir)
    assert result.status == "PARTIAL"
    receipt = json.loads((stage_dir / "receipt.json").read_text(encoding="utf-8"))
    assert [error["uid"] for error in receipt["errors"]] == ["bad"]
    rows = [json.loads(line) for line in (stage_dir / "prosody_alignments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["uid"] for row in rows] == ["good"]


def test_prosody_handler_consumes_multi_uid_merge_aggregate_and_keeps_merge_ledger(tmp_path: Path):
    alignment = _prosody_alignment("good")
    config = _write_prosody_sources(tmp_path, alignment)
    source = tmp_path / "stages" / "merge" / "ja_en_alignments.json"
    source.write_text(json.dumps({"schema": "ja-en-alignment-v3", "alignments": [alignment]}), encoding="utf-8")
    config["prosody"]["alignment_jsonl"] = str(source)
    (tmp_path / "stages" / "merge" / "uid_errors.json").write_text(json.dumps({
        "schema": "ja-en-uid-error-ledger-v1", "errors": [{"uid": "rejected", "code": "seam_rejected", "message": "fixture"}],
    }), encoding="utf-8")
    result = handle_prosody(config, tmp_path / "stages" / "prosody")
    assert result.status == "PARTIAL"
    receipt = json.loads((tmp_path / "stages" / "prosody" / "receipt.json").read_text(encoding="utf-8"))
    assert [row["uid"] for row in receipt["errors"]] == ["rejected"]


def test_prosody_aggregate_unions_expected_and_blocked_merge_uid_ledger(tmp_path: Path):
    alignment = _prosody_alignment("good")
    config = _write_prosody_sources(tmp_path, alignment)
    source = tmp_path / "stages" / "merge" / "ja_en_alignments.json"
    source.write_text(json.dumps({"alignments": [alignment]}), encoding="utf-8")
    config["prosody"]["alignment_jsonl"] = str(source)
    (tmp_path / "stages" / "merge" / "uid_errors.json").write_text(json.dumps({
        "expected_uids": ["good", "blocked"], "blocked_uids": ["blocked"], "errors": [],
    }), encoding="utf-8")
    result = handle_prosody(config, tmp_path / "stages" / "prosody")
    assert result.status == "PARTIAL"
    ledger = json.loads((tmp_path / "stages" / "prosody" / "uid_errors.json").read_text(encoding="utf-8"))
    assert ledger["expected_uids"] == ["blocked", "good"]
    assert ledger["blocked_uids"] == ["blocked"]
    assert ledger["errors"][0]["uid"] == "blocked"
