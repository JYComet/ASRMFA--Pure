import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_ja_supply_chain import (
    SupplyChainError,
    freeze_lock,
    load_lock,
    verify_lock,
)


def _lock(path: Path, *, license_status="approved", digest=None):
    return {
        "schema": "ja-supply-chain-lock-v1",
        "status": "frozen",
        "resources": [{
            "id": "example",
            "kind": "source",
            "source_url": "https://example.invalid/project",
            "commit": "a" * 40,
            "tree_sha256": digest or "0" * 64,
            "license": {"status": license_status, "spdx": "MIT", "file_sha256": "1" * 64},
            "artifacts": [],
        }],
    }


def test_strict_lock_rejects_unknown_license(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(_lock(path, license_status="unreviewed")), encoding="utf-8")
    with pytest.raises(SupplyChainError, match="license"):
        verify_lock(load_lock(path), strict=True)


def test_artifact_hash_is_checked(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"known")
    lock = _lock(path=tmp_path, digest=hashlib.sha256(b"known").hexdigest())
    lock["resources"][0]["artifacts"] = [{"id": "asset", "path": str(asset), "sha256": "0" * 64}]
    with pytest.raises(SupplyChainError, match="hash"):
        verify_lock(lock, strict=False)


def test_load_lock_rejects_schema_and_duplicate_resource_ids(tmp_path):
    path = tmp_path / "lock.json"
    payload = _lock(path)
    payload["resources"].append(dict(payload["resources"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SupplyChainError, match="duplicate"):
        load_lock(path)


def test_strict_requires_active_artifact_binding(tmp_path):
    payload = _lock(tmp_path, license_status="approved")
    payload["resources"][0]["license"]["file_sha256"] = hashlib.sha256(b"lic").hexdigest()
    with pytest.raises(SupplyChainError, match="artifact"):
        verify_lock(payload, strict=True)


def test_freeze_lock_binds_real_artifact_and_license_hashes(tmp_path):
    asset = tmp_path / "asset.bin"; asset.write_bytes(b"asset")
    license_file = tmp_path / "LICENSE"; license_file.write_bytes(b"license")
    payload = _lock(tmp_path, license_status="unreviewed")
    frozen = freeze_lock(payload, {"example": {"artifacts": [str(asset)], "license_path": str(license_file)}})
    resource = frozen["resources"][0]
    assert resource["artifacts"][0]["sha256"] == hashlib.sha256(b"asset").hexdigest()
    assert resource["license"]["file_sha256"] == hashlib.sha256(b"license").hexdigest()


def test_disabled_diagnostic_resource_does_not_block_license_gate(tmp_path):
    payload = _lock(tmp_path, license_status="approved")
    payload["resources"][0]["kind"] = "diagnostic"
    payload["status"] = "frozen"
    report = verify_lock(payload, strict=True, active_config={"julius_diagnostic": {"enabled": False}})
    assert report["status"] == "PASS"


def test_model_directory_binding_rejects_added_removed_and_changed_files(tmp_path):
    model = tmp_path / "model"; model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    (model / "config.json").write_bytes(b"{}")
    license_file = tmp_path / "LICENSE"; license_file.write_bytes(b"license")
    payload = _lock(tmp_path, license_status="approved")
    payload["resources"][0]["kind"] = "model"
    frozen = freeze_lock(payload, {"example": {"artifacts": [{"id": "model", "path": str(model)}], "license_path": str(license_file)}})
    assert verify_lock(frozen, strict=True)["status"] == "PASS"
    (model / "extra.bin").write_bytes(b"extra")
    with pytest.raises(SupplyChainError, match="inventory"):
        verify_lock(frozen, strict=True)
    (model / "extra.bin").unlink()
    (model / "weights.bin").write_bytes(b"changed")
    with pytest.raises(SupplyChainError, match="inventory|hash"):
        verify_lock(frozen, strict=True)


def test_runtime_symlink_binding_detects_retarget_and_requires_environment_receipt(tmp_path):
    target_a = tmp_path / "python-a"; target_a.write_bytes(b"python-a")
    target_b = tmp_path / "python-b"; target_b.write_bytes(b"python-b")
    invoked = tmp_path / "python"; invoked.symlink_to(target_a)
    env = tmp_path / "environment.json"; env.write_text("{}", encoding="utf-8")
    license_file = tmp_path / "LICENSE"; license_file.write_bytes(b"license")
    payload = _lock(tmp_path, license_status="approved")
    payload["resources"][0]["kind"] = "runtime"
    frozen = freeze_lock(payload, {"example": {"runtime": {
        "invoked_path": str(invoked), "package_env_receipt": str(env)}, "license_path": str(license_file)}})
    assert verify_lock(frozen, strict=True, active_config={"runtime_python": str(invoked)})["status"] == "PASS"
    invoked.unlink(); invoked.symlink_to(target_b)
    with pytest.raises(SupplyChainError, match="runtime"):
        verify_lock(frozen, strict=True)


def test_config_manifest_and_lock_are_not_model_asset_bindings(tmp_path):
    payload = _lock(tmp_path, license_status="approved")
    report = verify_lock(payload, strict=False, active_config={
        "input_manifest": str(tmp_path / "manifest.jsonl"),
        "supply_chain_lock": str(tmp_path / "lock.json"),
        "julius_diagnostic": {"enabled": False},
    })
    assert report["status"] == "PARTIAL"
