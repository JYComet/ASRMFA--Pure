import json
from pathlib import Path

import pytest

from scripts.ja_en_schema import JAContractError
from scripts.run_ja_en_pipeline import _prepare_stage_config, preflight
from tests.test_ja_en_foundation import _config


def test_no_reference_manifest_is_allowed_only_for_fallback(tmp_path):
    config_path, config = _config(tmp_path)
    manifest = Path(config["input_manifest"])
    manifest.write_text(json.dumps({"uid": "u1", "wav": "/mnt/source/read-only.wav"}) + "\n", encoding="utf-8")
    with pytest.raises(JAContractError, match="non-empty text"):
        preflight(config, config_path=config_path)
    config["asr"]["reference_mode"] = "fallback"
    _, _, rows, _ = preflight(config, config_path=config_path)
    assert rows[0].get("text") is None


def test_fallback_downstream_reads_only_resolved_asr_manifest(tmp_path):
    config_path, config = _config(tmp_path)
    config["asr"]["reference_mode"] = "fallback"
    resolved = tmp_path / "workspace/stages/asr/resolved_manifest.jsonl"
    resolved.parent.mkdir(parents=True)
    resolved.write_text(json.dumps({"uid": "u1", "wav": "/mnt/source/read-only.wav",
                                    "text": "さくら", "text_origin": "qwen3asr_transcript_only"}) + "\n", encoding="utf-8")
    prepared = _prepare_stage_config(config, "frontend", tmp_path / "workspace")
    assert prepared["input_manifest"] == str(resolved)


def test_blind_transcripts_are_bound_to_all_uids_and_preserve_source_manifest(tmp_path):
    from scripts.ja_asr_crossval import resolve_blind_transcripts

    run_root = tmp_path / "qwen3asr"
    (run_root / "transcripts").mkdir(parents=True)
    (run_root / ".qwen3asr_run_receipt.json").write_text(json.dumps({"status": "COMPLETE", "failed_count": 0}), encoding="utf-8")
    transcript = run_root / "transcripts/a_qwen3.txt"
    transcript.write_text("桜。\n", encoding="utf-8")
    import hashlib
    (run_root / "qwen3asr_manifest.json").write_text(json.dumps({"source_stems": ["a"], "success": ["a"], "failed": [],
        "success_records": [{"stem": "a", "path": "transcripts/a_qwen3.txt",
                             "sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()}]}), encoding="utf-8")
    source = [{"uid": "a", "wav": "/data/a.wav"}]
    resolved = resolve_blind_transcripts(source, {"a": "a"}, run_root)
    assert resolved[0]["text"] == "桜。"
    assert resolved[0]["text_origin"] == "qwen3asr_transcript_only"
    assert "text" not in source[0]
    with pytest.raises(ValueError, match="missing|incomplete"):
        resolve_blind_transcripts(source + [{"uid": "b", "wav": "/data/b.wav"}],
                                  {"a": "a", "b": "b"}, run_root)


def test_locked_frontend_reconstruction_uses_blind_asr_text(tmp_path, monkeypatch):
    import scripts.run_ja_en_pipeline as pipeline
    _, config = _config(tmp_path)
    config["asr"]["reference_mode"] = "fallback"
    workspace = tmp_path / "workspace"
    reading = workspace / "stages/reading"
    reading.mkdir(parents=True)
    (reading / "locked_readings.jsonl").write_text(json.dumps({"uid": "u1", "status": "blocked"}) + "\n", encoding="utf-8")
    resolved = workspace / "stages/asr/resolved_manifest.jsonl"
    resolved.parent.mkdir(parents=True)
    resolved.write_text(json.dumps({"uid": "u1", "wav": "/data/a.wav", "text": "さくら"}) + "\n", encoding="utf-8")
    seen = {}
    def fake_frontend(stage_config, stage_dir):
        seen["manifest"] = stage_config["input_manifest"]
        pipeline.atomic_write_json(stage_dir / "receipt.json", pipeline.make_receipt(stage="frontend", status="COMPLETE"), workspace=workspace)
        return pipeline.StageResult("frontend", "COMPLETE", str(stage_dir / "receipt.json"))
    monkeypatch.setitem(pipeline._STAGE_REGISTRY, "frontend", (fake_frontend, "frontend"))
    result = pipeline._run_locked_frontend_reconstruction(config, workspace)
    assert result.status == "COMPLETE"
    assert seen["manifest"] == str(resolved)


def test_failed_blind_asr_blocks_downstream_without_source_text(tmp_path, monkeypatch):
    import scripts.run_ja_en_pipeline as pipeline
    config_path, config = _config(tmp_path)
    config["asr"]["reference_mode"] = "fallback"
    Path(config["input_manifest"]).write_text(json.dumps({"uid": "u1", "wav": "/data/a.wav"}) + "\n", encoding="utf-8")
    def blocked_asr(_config, stage_dir):
        pipeline.atomic_write_json(stage_dir / "receipt.json", pipeline.make_receipt(stage="asr", status="BLOCKED"), workspace=stage_dir.parent.parent)
        return pipeline.StageResult("asr", "BLOCKED", str(stage_dir / "receipt.json"))
    def frontend_must_not_run(*_args):
        raise AssertionError("frontend saw the unresolved no-reference manifest")
    monkeypatch.setitem(pipeline._STAGE_REGISTRY, "asr", (blocked_asr, "asr"))
    monkeypatch.setitem(pipeline._STAGE_REGISTRY, "frontend", (frontend_must_not_run, "frontend"))
    results = pipeline.dispatch(config, config_path, ["asr", "frontend"])
    assert [(row.stage, row.status) for row in results] == [("asr", "BLOCKED"), ("frontend", "BLOCKED")]


def test_blind_transcript_only_cannot_claim_three_family_profile(tmp_path):
    from scripts.ja_asr_crossval import asr_stage
    manifest = tmp_path / "blind.jsonl"
    manifest.write_text(json.dumps({"uid": "u1", "wav": "/data/a.wav"}) + "\n", encoding="utf-8")
    stage_dir = tmp_path / "stages/asr"
    result = asr_stage({"input_manifest": str(manifest), "asr": {"reference_mode": "fallback",
                        "profile": "baseline_3family"}}, stage_dir)
    assert result.status == "BLOCKED"
    receipt = json.loads((stage_dir / "receipt.json").read_text(encoding="utf-8"))
    assert "qwen_only_dev" in receipt["errors"][0]["message"]
