from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ja_en_schema import JAContractError, atomic_write_json, make_receipt
from scripts.run_ja_en_pipeline import (
    config_identity,
    _stage_cache_identity,
    dispatch,
    preflight,
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
