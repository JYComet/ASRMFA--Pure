import json
from pathlib import Path

import pytest

from scripts.ja_en_schema import (
    ERROR_CODES,
    JAContractError,
    PRODUCTION_STAGES,
    SCHEMAS,
    atomic_write_json,
    cache_key,
    make_receipt,
    validate_config,
    validate_exact_partition,
    validate_receipt,
)
from scripts.run_ja_en_pipeline import (
    _load_yaml,
    config_identity,
    dispatch,
    inspect_workspace,
    preflight,
    validate_resume,
)


def _config(tmp_path: Path) -> tuple[Path, dict]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "u1", "wav": "/mnt/source/read-only.wav", "text": "さくら"}) + "\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""pipeline: ja_en_tts
workspace: {tmp_path / 'workspace'}
input_manifest: {manifest}
supply_chain_lock: {tmp_path / 'lock.json'}
asr:
  profile: qwen_only_dev
  family_vote_policy: one_per_family
frontend:
  provider: pyopenjtalk-plus
  commit: pinned
  use_vanilla: false
  use_tsqyomi: false
  use_sudachi_kanji_yomi: false
  predict_nani: false
  normalize_mode: None
  use_read_as_pron: false
  revert_long_vowels: false
  revert_yotsugana: false
  run_marine: false
  reject_unbound_spans: true
mixed:
  enabled: false
julius_diagnostic:
  enabled: false
  write_back: false
""",
        encoding="utf-8",
    )
    return config_path, _load_yaml(config_path)


def test_five_track_contract_versions_and_stage_order():
    assert {
        "ja-semantic-phone-graph-v2", "ja-en-alignment-v3",
        "ja-prosody-alignment-v1", "tts-training-record-v2",
        "five-track-textgrid-v1",
    } <= SCHEMAS
    assert PRODUCTION_STAGES[-4:] == ("merge", "prosody", "tts", "verify")
    assert {
        "accent_phrase_unresolved", "tone_cardinality_mismatch",
        "native_basic_mapping_ambiguous", "phone_tone_projection_lossy",
        "five_track_boundary_mismatch", "tone_provenance_missing",
    } <= ERROR_CODES


def test_prosody_config_has_closed_keys(tmp_path):
    _, config = _config(tmp_path)
    config["prosody"] = {
        "algorithm_version": "ja-mora-tone-v1",
        "manual_overrides": None,
        "accent_lexicon": None,
        "allow_unknown_tones": True,
    }
    assert validate_config(config)["prosody"]["algorithm_version"] == "ja-mora-tone-v1"
    config["prosody"]["nearest_neighbor_fill"] = True
    with pytest.raises(JAContractError, match="config_unknown_key"):
        validate_config(config)


def test_partition_is_exact_and_rejects_overlap():
    assert validate_exact_partition(["a", "b"], {"verified": ["a"], "unresolved": ["b"]}) == {
        "verified": ["a"],
        "unresolved": ["b"],
    }
    with pytest.raises(JAContractError) as error:
        validate_exact_partition(["a", "b"], {"verified": ["a"], "rejected": ["a"]})
    assert error.value.code == "partition_not_exact"


def test_receipt_records_hash_size_and_rejects_tampering(tmp_path):
    output = tmp_path / "stages" / "inventory" / "inventory.json"
    atomic_write_json(output, {"uid": "u1"}, workspace=tmp_path)
    receipt = make_receipt(stage="inventory", status="COMPLETE", outputs=[output])
    validate_receipt(receipt, workspace=tmp_path)
    output.write_text("tampered", encoding="utf-8")
    with pytest.raises(JAContractError) as error:
        validate_receipt(receipt, workspace=tmp_path)
    assert error.value.code == "receipt_hash_mismatch"


def test_source_absolute_path_is_allowed_but_symlink_manifest_is_rejected(tmp_path):
    config_path, config = _config(tmp_path)
    _, manifest_path, rows, workspace = preflight(config, config_path=config_path)
    assert rows[0]["wav"].startswith("/mnt/")
    link = tmp_path / "manifest-link.jsonl"
    link.symlink_to(manifest_path)
    config["input_manifest"] = str(link)
    with pytest.raises(JAContractError) as error:
        preflight(config, config_path=config_path)
    assert error.value.code == "manifest_symlink"


def test_dispatch_skeleton_and_inspect_are_deterministic(tmp_path):
    config_path, config = _config(tmp_path)
    results = dispatch(config, config_path, ["inventory"])
    assert results[0].status == "BLOCKED"
    workspace = Path(config["workspace"])
    report = inspect_workspace(workspace)
    assert report["stages"]["inventory"]["status"] == "BLOCKED"
    assert {Path(row["path"]).name for row in report["stages"]["inventory"]["outputs"]} == {
        "production_config.json", "production_supply_chain_lock.json",
    }
    identity = json.loads((workspace / ".ja_en_run_identity.json").read_text(encoding="utf-8"))
    _, manifest_path, rows, _ = preflight(config, config_path=config_path)
    validate_resume(workspace, config_identity(config, manifest_path, rows), allow_new=False)
    (workspace / "unexpected.cache").write_text("stale", encoding="utf-8")
    with pytest.raises(JAContractError) as error:
        validate_resume(workspace, config_identity(config, manifest_path, rows), allow_new=False)
    assert error.value.code == "resume_extra_file"
    assert identity["identity_digest"] == cache_key(identity["identity"])


def test_config_rejects_duplicate_keys_and_julius_writeback(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("pipeline: ja_en_tts\npipeline: nope\n", encoding="utf-8")
    with pytest.raises(JAContractError) as error:
        _load_yaml(duplicate)
    assert error.value.code == "config_malformed"
    with pytest.raises(JAContractError) as error:
        validate_config(
            {
                "pipeline": "ja_en_tts",
                "workspace": "/tmp/run",
                "input_manifest": "/tmp/manifest.jsonl",
                "supply_chain_lock": "/tmp/lock.json",
                "frontend": {
                    "provider": "pyopenjtalk-plus",
                    "commit": "pinned",
                    "use_vanilla": False,
                    "use_tsqyomi": False,
                    "use_sudachi_kanji_yomi": False,
                    "predict_nani": False,
                    "normalize_mode": "None",
                    "use_read_as_pron": False,
                    "revert_long_vowels": False,
                    "revert_yotsugana": False,
                    "run_marine": False,
                    "reject_unbound_spans": True,
                },
                "julius_diagnostic": {"enabled": True, "write_back": True},
            }
        )
    assert error.value.code == "julius_writeback_forbidden"


def test_root_receipt_is_canonical_and_binds_declared_run_inputs(tmp_path):
    config_path, config = _config(tmp_path)
    results = dispatch(config, config_path, ["inventory"])
    assert results[0].status == "BLOCKED"
    workspace = Path(config["workspace"])
    root_receipt = json.loads((workspace / "receipt.json").read_text(encoding="utf-8"))
    assert not (workspace / ".ja_en_pipeline_receipt.json").exists()
    assert root_receipt["inputs"]["stages"] == ["inventory"]
    input_paths = {row["path"] for row in root_receipt["inputs"]["artifacts"]}
    assert str(config_path) not in input_paths
    assert any(path.endswith("production_config.json") for path in input_paths)


def test_config_identity_refuses_symlinked_asset_directories(tmp_path):
    config_path, config = _config(tmp_path)
    target = tmp_path / "model-target"
    target.mkdir()
    link = tmp_path / "model-link"
    link.symlink_to(target, target_is_directory=True)
    config["asr"] = {
        "profile": "qwen_only_dev",
        "family_vote_policy": "one_per_family",
        "qwen_model": str(link),
    }
    _, manifest_path, rows, _ = preflight(config, config_path=config_path)
    with pytest.raises(JAContractError) as error:
        config_identity(config, manifest_path, rows)
    assert error.value.code == "config_malformed"
