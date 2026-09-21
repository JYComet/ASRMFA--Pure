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


def test_mfa_dictionary_invalidates_semantic_and_all_downstream(tmp_path: Path):
    identity = {"config": {"mfa": {"japanese_dictionary": "dictionary-a"}, "prosody": {}}}
    changed = {"config": {"mfa": {"japanese_dictionary": "dictionary-b"}, "prosody": {}}}
    stages = ("semantic", "align", "merge", "prosody", "tts", "verify")
    before = {stage: _stage_cache_identity(stage, identity, {}, tmp_path) for stage in stages}
    after = {stage: _stage_cache_identity(stage, changed, {}, tmp_path) for stage in stages}
    assert all(before[stage] != after[stage] for stage in ("semantic", "align", "merge", "prosody", "tts", "verify"))


def test_real_identity_scopes_assets_to_their_first_consuming_stage(tmp_path: Path):
    config_path, config = _config(tmp_path)
    dictionary, metadata, archive, runtime, frontend_model, tones = (tmp_path / "dictionary", tmp_path / "metadata", tmp_path / "archive", tmp_path / "runtime", tmp_path / "frontend.model", tmp_path / "tones.json")
    for path, value in ((dictionary, "dictionary-a"), (metadata, "metadata-a"), (archive, "archive-a"), (runtime, "runtime-a"), (frontend_model, "frontend-a"), (tones, "tones-a")):
        path.write_text(value, encoding="utf-8")
    config["mfa"] = {"japanese_dictionary": str(dictionary), "japanese_metadata": str(metadata), "japanese_acoustic": str(archive), "runtime_python": str(runtime)}
    config["frontend"]["accent_model"] = str(frontend_model)
    config["prosody"] = {"manual_overrides": str(tones)}
    manifest_path = Path(config["input_manifest"])
    rows = [{"uid": "u1", "wav": "/mnt/source/read-only.wav", "text": "さくら"}]
    workspace = Path(config["workspace"])
    before = config_identity(config, manifest_path, rows)
    dictionary.write_text("dictionary-b", encoding="utf-8")
    after_dictionary = config_identity(config, manifest_path, rows)
    assert _stage_cache_identity("semantic", before, config, workspace) != _stage_cache_identity("semantic", after_dictionary, config, workspace)
    for asset in (metadata, archive):
        dictionary.write_text("dictionary-a", encoding="utf-8"); metadata.write_text("metadata-a", encoding="utf-8"); archive.write_text("archive-a", encoding="utf-8")
        asset.write_text("changed", encoding="utf-8")
        changed = config_identity(config, manifest_path, rows)
        assert _stage_cache_identity("semantic", before, config, workspace) != _stage_cache_identity("semantic", changed, config, workspace)
    metadata.write_text("metadata-a", encoding="utf-8"); archive.write_text("archive-a", encoding="utf-8"); runtime.write_text("runtime-b", encoding="utf-8")
    after_runtime = config_identity(config, manifest_path, rows)
    assert _stage_cache_identity("semantic", before, config, workspace) == _stage_cache_identity("semantic", after_runtime, config, workspace)
    assert _stage_cache_identity("align", before, config, workspace) != _stage_cache_identity("align", after_runtime, config, workspace)
    runtime.write_text("runtime-a", encoding="utf-8"); frontend_model.write_text("frontend-b", encoding="utf-8")
    after_frontend = config_identity(config, manifest_path, rows)
    assert _stage_cache_identity("reading", before, config, workspace) == _stage_cache_identity("reading", after_frontend, config, workspace)
    assert _stage_cache_identity("frontend", before, config, workspace) != _stage_cache_identity("frontend", after_frontend, config, workspace)
    frontend_model.write_text("frontend-a", encoding="utf-8"); tones.write_text("tones-b", encoding="utf-8")
    after_tones = config_identity(config, manifest_path, rows)
    assert _stage_cache_identity("merge", before, config, workspace) == _stage_cache_identity("merge", after_tones, config, workspace)
    assert _stage_cache_identity("prosody", before, config, workspace) != _stage_cache_identity("prosody", after_tones, config, workspace)


def test_persisted_stage_cache_decision_starts_at_semantic_for_dictionary_drift(tmp_path: Path):
    config_path, config = _config(tmp_path)
    dictionary = tmp_path / "dictionary"; dictionary.write_text("a", encoding="utf-8")
    config["mfa"] = {"japanese_dictionary": str(dictionary)}
    manifest_path = Path(config["input_manifest"])
    rows = [{"uid": "u1", "wav": "/mnt/source/read-only.wav", "text": "さくら"}]
    workspace = Path(config["workspace"]); workspace.mkdir()
    before = config_identity(config, manifest_path, rows)
    persisted = {stage: _stage_cache_identity(stage, before, config, workspace) for stage in ("reading", "frontend", "semantic", "align", "merge", "prosody", "tts", "verify")}
    atomic_write_json(workspace / ".ja_en_stage_cache.json", {"schema": "ja-stage-cache-v1", "stages": persisted}, workspace=workspace)
    dictionary.write_text("b", encoding="utf-8")
    after = config_identity(config, manifest_path, rows)
    cached = json.loads((workspace / ".ja_en_stage_cache.json").read_text())["stages"]
    first_invalidated = next(stage for stage in persisted if cached[stage] != _stage_cache_identity(stage, after, config, workspace))
    assert first_invalidated == "semantic"


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


def test_resume_compatibility_rejects_stage_order_drift(tmp_path: Path):
    config_path, config = _config(tmp_path)
    _, manifest_path, rows, workspace = preflight(config, config_path=config_path)
    identity = config_identity(config, manifest_path, rows)
    workspace.mkdir()
    from scripts.run_ja_en_pipeline import _compatibility_identity
    payload = {"compatibility_digest": stable_digest(_compatibility_identity(identity))}
    atomic_write_json(workspace / ".ja_en_run_identity.json", payload, workspace=workspace)
    identity["production_stages"] = list(reversed(identity["production_stages"]))
    with pytest.raises(JAContractError, match="resume_identity_drift"):
        validate_resume(workspace, identity, allow_new=False)
