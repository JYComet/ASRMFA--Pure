from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ja_en_schema import JAContractError, atomic_write_json, make_receipt, stable_digest
from scripts.run_ja_en_pipeline import (
    _autoload_stages,
    _prepare_stage_config,
    config_identity,
    _stage_cache_identity,
    dispatch,
    main,
    preflight,
    stage_registry,
    validate_resume,
)


def _config(tmp_path: Path) -> tuple[Path, dict]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "u1", "wav": "/mnt/source/read-only.wav", "text": "さくら"}) + "\n", encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": "ja-supply-chain-lock-v1", "status": "ok", "policy": {}, "resources": [{"id": "prod", "kind": "model"}, {"id": "julius", "kind": "diagnostic"}]}), encoding="utf-8")
    config = {
        "pipeline": "ja_en_tts", "workspace": str(tmp_path / "workspace"),
        "input_manifest": str(manifest), "supply_chain_lock": str(lock),
        "asr": {"profile": "qwen_only_dev", "family_vote_policy": "one_per_family"},
        "frontend": {
            "provider": "fixture", "commit": "pinned", "use_vanilla": False,
            "use_tsqyomi": False, "use_sudachi_kanji_yomi": False,
            "predict_nani": False, "normalize_mode": "None", "use_read_as_pron": False,
            "revert_long_vowels": False, "revert_yotsugana": False,
            "run_marine": False, "reject_unbound_spans": True,
        },
        "mixed": {"enabled": False}, "julius_diagnostic": {"enabled": False, "write_back": False},
    }
    return tmp_path / "config.json", config


def test_production_snapshot_ignores_julius_but_rejects_tampering(tmp_path: Path):
    config_path, config = _config(tmp_path)
    results = dispatch(config, config_path, ["inventory"])
    assert results[0].status == "BLOCKED"
    workspace = Path(config["workspace"])
    _, manifest_path, rows, _ = preflight(config, config_path=config_path)
    config["julius_diagnostic"] = {"enabled": True, "write_back": False, "converter_commit": "changed"}
    validate_resume(workspace, config_identity(config, manifest_path, rows), allow_new=False)
    snapshot = workspace / "stages" / "inventory" / "run_contract" / "production_config.json"
    snapshot.write_text("tampered", encoding="utf-8")
    with pytest.raises(JAContractError) as error:
        validate_resume(workspace, config_identity(config, manifest_path, rows), allow_new=False)
    assert error.value.code in {"receipt_hash_mismatch", "receipt_output_missing"}


def test_reading_cache_binds_candidate_analysis_not_rewritten_frontend_receipt(tmp_path: Path):
    workspace = tmp_path / "workspace"
    analysis = workspace / "stages" / "frontend" / "frontend_analysis.json"
    atomic_write_json(analysis, {"schema": "frontend-candidate-analysis-v1", "records": []}, workspace=workspace)
    frontend_receipt = workspace / "stages" / "frontend" / "receipt.json"
    reading_receipt = workspace / "stages" / "reading" / "receipt.json"
    atomic_write_json(frontend_receipt, make_receipt(stage="frontend", status="COMPLETE", outputs=[analysis]), workspace=workspace)
    atomic_write_json(reading_receipt, make_receipt(stage="reading", status="COMPLETE"), workspace=workspace)
    identity = {"schema": "test", "digest": "fixed"}
    first = _stage_cache_identity("reading", identity, {}, workspace)
    changed = make_receipt(stage="frontend", status="COMPLETE", outputs=[analysis], params={"reconstruction": "changed"})
    atomic_write_json(frontend_receipt, changed, workspace=workspace)
    assert _stage_cache_identity("reading", identity, {}, workspace) == first


def test_legacy_workspace_is_stale_after_prosody_stage_is_added(tmp_path: Path):
    config_path, config = _config(tmp_path)
    workspace = Path(config["workspace"])
    workspace.mkdir(parents=True)
    legacy_identity = {"schema": "ja-en-run-identity-v1", "production_stages": [
        "inventory", "audio", "asr", "reading", "frontend", "semantic",
        "anchors", "align", "merge", "tts", "verify",
    ]}
    atomic_write_json(workspace / ".ja_en_run_identity.json", {
        **legacy_identity, "identity_digest": stable_digest(legacy_identity)
    }, workspace=workspace)
    _, manifest_path, rows, _ = preflight(config, config_path=config_path)
    with pytest.raises(JAContractError) as error:
        validate_resume(workspace, config_identity(config, manifest_path, rows), allow_new=False)
    assert error.value.code in {"resume_identity_drift", "resume_stale"}


def test_prosody_resources_invalidate_only_prosody_downstream(tmp_path: Path):
    identity = {"config": {"mfa": {"japanese_dictionary": "dictionary-a"},
                           "prosody": {"manual_overrides": "tones-a"}}}
    changed = {"config": {"mfa": {"japanese_dictionary": "dictionary-a"},
                          "prosody": {"manual_overrides": "tones-b"}}}
    stages = ("merge", "prosody", "tts", "verify")
    before = {stage: _stage_cache_identity(stage, identity, {}, tmp_path) for stage in stages}
    after = {stage: _stage_cache_identity(stage, changed, {}, tmp_path) for stage in stages}
    assert before["merge"] == after["merge"]
    assert all(before[stage] != after[stage] for stage in ("prosody", "tts", "verify"))


def test_mfa_dictionary_invalidates_align_and_all_downstream(tmp_path: Path):
    identity = {"config": {"mfa": {"japanese_dictionary": "dictionary-a"}, "prosody": {}}}
    changed = {"config": {"mfa": {"japanese_dictionary": "dictionary-b"}, "prosody": {}}}
    stages = ("semantic", "align", "merge", "prosody", "tts", "verify")
    before = {stage: _stage_cache_identity(stage, identity, {}, tmp_path) for stage in stages}
    after = {stage: _stage_cache_identity(stage, changed, {}, tmp_path) for stage in stages}
    assert before["semantic"] == after["semantic"]
    assert all(before[stage] != after[stage] for stage in ("align", "merge", "prosody", "tts", "verify"))


def test_workspace_override_rejects_nonempty_target_without_resume(tmp_path: Path):
    config_path, config = _config(tmp_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "evidence.txt").write_text("keep", encoding="utf-8")
    assert main(["--config", str(config_path), "--workspace", str(occupied)]) == 2
    assert (occupied / "evidence.txt").read_text(encoding="utf-8") == "keep"


def test_autoload_registers_prosody_stage():
    _autoload_stages()
    assert stage_registry()["prosody"]["owner"] == "scripts.ja_prosody"


def test_tts_stage_input_is_only_the_prosody_artifact(tmp_path: Path):
    artifact = tmp_path / "stages" / "prosody" / "prosody_alignments.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"schema":"ja-prosody-alignment-v1"}\n', encoding="utf-8")
    prepared = _prepare_stage_config({}, "tts", tmp_path)
    assert prepared["tts"]["alignment_jsonl"] == str(artifact)
    assert prepared["stage_inputs"]["tts"] == {"alignment_jsonl": str(artifact)}
