#!/usr/bin/env python3
"""Validate the Japanese pipeline's explicit, auditable supply-chain lock.

The lock is intentionally data-only.  This verifier never downloads or imports
runtime packages; it checks that every declared resource has an immutable
revision/hash and that code, wheels, dictionaries, models and licences are
audited independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


class SupplyChainError(ValueError):
    """A fail-closed lock or artifact error."""


_KNOWN_LICENSE_STATES = {"approved", "reviewed", "unreviewed", "unknown", "missing", "blocked"}
_RESOURCE_KINDS = {"source", "wheel", "binary", "model", "dictionary", "runtime", "frontend", "diagnostic"}
_HEX64 = set("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(path: Path) -> tuple[list[dict[str, str]], str]:
    """Return a stable regular-file inventory and digest for a model tree."""
    if path.is_symlink() or not path.is_dir():
        raise SupplyChainError(f"directory artifact is missing or symlinked: {path}")
    rows: list[dict[str, str]] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink():
            raise SupplyChainError(f"directory artifact contains symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(path).as_posix()
            rows.append({"path": relative, "sha256": _sha256(candidate)})
    encoded = "".join(f"{row['path']}\t{row['sha256']}\n" for row in rows).encode("utf-8")
    return rows, hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def freeze_lock(lock: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a catalogue lock to local files without inventing licence approval.

    ``bindings`` is a JSON object keyed by resource id.  Each value may contain
    ``tree_path``, ``license_path`` and ``artifacts`` (list of file paths or
    ``{"id", "path"}`` records).  The resulting lock remains blocked when a
    license is unreviewed, but all hashes and paths are concrete and auditable.
    """
    result = json.loads(json.dumps(lock))
    for resource in result.get("resources", []):
        binding = bindings.get(resource.get("id"))
        if not isinstance(binding, Mapping):
            continue
        tree_path = binding.get("tree_path")
        if tree_path:
            candidate = Path(str(tree_path)).expanduser().absolute()
            if candidate.is_dir():
                tree_files, tree_hash = _directory_inventory(candidate)
                resource["tree_sha256"] = tree_hash
                resource["tree_files"] = tree_files
            elif candidate.is_file() and not candidate.is_symlink():
                resource["tree_sha256"] = _sha256(candidate)
            else:
                raise SupplyChainError(f"{resource['id']}: tree_path is missing or symlinked")
        license_path = binding.get("license_path")
        if license_path:
            candidate = Path(str(license_path)).expanduser().absolute()
            if not candidate.is_file() or candidate.is_symlink():
                raise SupplyChainError(f"{resource['id']}: license_path is missing or symlinked")
            resource.setdefault("license", {})["path"] = str(candidate)
            resource["license"]["file_sha256"] = _sha256(candidate)
        artifact_rows = []
        for item in binding.get("artifacts", []) or []:
            if isinstance(item, str):
                item = {"id": Path(item).name, "path": item}
            if not isinstance(item, Mapping) or not item.get("path"):
                raise SupplyChainError(f"{resource['id']}: invalid artifact binding")
            candidate = Path(str(item["path"])).expanduser().absolute()
            if candidate.is_dir():
                files, tree_hash = _directory_inventory(candidate)
                artifact_rows.append({"id": str(item.get("id", candidate.name)), "kind": "directory",
                                      "path": str(candidate), "sha256": tree_hash, "files": files})
            elif candidate.is_file() and not candidate.is_symlink():
                artifact_rows.append({"id": str(item.get("id", candidate.name)), "path": str(candidate), "sha256": _sha256(candidate)})
            else:
                raise SupplyChainError(f"{resource['id']}: artifact is missing or symlinked: {candidate}")
        if artifact_rows:
            resource["artifacts"] = artifact_rows
        runtime = binding.get("runtime") or binding.get("runtime_binding")
        if isinstance(runtime, Mapping):
            invoked = Path(str(runtime.get("invoked_path", runtime.get("path", "")))).expanduser().absolute()
            if not invoked.exists():
                raise SupplyChainError(f"{resource['id']}: runtime invoked_path is missing")
            resolved = invoked.resolve()
            if not resolved.is_file():
                raise SupplyChainError(f"{resource['id']}: runtime resolved_path is not a file")
            runtime_row: dict[str, Any] = {"invoked_path": str(invoked), "resolved_path": str(resolved),
                                           "resolved_sha256": _sha256(resolved)}
            receipt = runtime.get("package_env_receipt", runtime.get("packageenv_receipt"))
            if receipt:
                receipt_path = Path(str(receipt)).expanduser().absolute()
                if receipt_path.is_symlink() or not receipt_path.is_file():
                    raise SupplyChainError(f"{resource['id']}: package environment receipt is missing or symlinked")
                runtime_row["package_env_receipt"] = {"path": str(receipt_path), "sha256": _sha256(receipt_path)}
            resource["runtime_binding"] = runtime_row
    result["status"] = "frozen"
    return result


def _digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64 for c in value.lower()):
        raise SupplyChainError(f"{field} must be a lowercase SHA-256 digest")
    return value.lower()


def load_lock(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise SupplyChainError(f"lock file is missing or symlinked: {candidate}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplyChainError(f"invalid lock JSON: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "ja-supply-chain-lock-v1":
        raise SupplyChainError("lock schema must be ja-supply-chain-lock-v1")
    resources = payload.get("resources")
    if not isinstance(resources, list) or not resources:
        raise SupplyChainError("lock resources must be a non-empty list")
    seen: set[str] = set()
    for index, item in enumerate(resources):
        if not isinstance(item, Mapping):
            raise SupplyChainError(f"resource {index} must be an object")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise SupplyChainError(f"resource {index} has no id")
        if identifier in seen:
            raise SupplyChainError(f"duplicate resource id: {identifier}")
        seen.add(identifier)
    return dict(payload)


def verify_lock(lock: Mapping[str, Any], *, strict: bool = False, base_dir: str | Path | None = None,
                active_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Verify a loaded lock and return a JSON-serialisable report.

    ``strict`` is the production gate: every resource must have an approved
    licence.  In non-strict mode unreviewed licences remain visible as
    warnings, so a caller cannot mistake a development check for approval.
    """

    if lock.get("schema") != "ja-supply-chain-lock-v1":
        raise SupplyChainError("lock schema must be ja-supply-chain-lock-v1")
    root = Path(base_dir).expanduser().absolute() if base_dir else Path.cwd()
    resources = lock.get("resources")
    if not isinstance(resources, list) or not resources:
        raise SupplyChainError("lock resources must be a non-empty list")
    seen: set[str] = set()
    warnings: list[str] = []
    checked = 0
    unbound = 0
    for index, raw in enumerate(resources):
        if not isinstance(raw, Mapping):
            raise SupplyChainError(f"resource {index} must be an object")
        resource = dict(raw)
        identifier = resource.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise SupplyChainError(f"resource {index} has no id")
        if identifier in seen:
            raise SupplyChainError(f"duplicate resource id: {identifier}")
        seen.add(identifier)
        kind = resource.get("kind")
        if kind not in _RESOURCE_KINDS:
            raise SupplyChainError(f"{identifier}: unsupported resource kind {kind!r}")
        if not isinstance(resource.get("source_url"), str) or not resource["source_url"]:
            raise SupplyChainError(f"{identifier}: source_url is required")
        revision = resource.get("commit") or resource.get("tag") or resource.get("revision")
        if not isinstance(revision, str) or not revision:
            raise SupplyChainError(f"{identifier}: pinned commit/tag/revision is required")
        if resource.get("tree_sha256") is not None:
            _digest(resource["tree_sha256"], field=f"{identifier}.tree_sha256")
        license_info = resource.get("license")
        if not isinstance(license_info, Mapping):
            raise SupplyChainError(f"{identifier}: license record is required")
        state = str(license_info.get("status", "unknown")).lower()
        if state not in _KNOWN_LICENSE_STATES:
            raise SupplyChainError(f"{identifier}: unknown license status {state!r}")
        license_hash = license_info.get("file_sha256")
        if license_hash is not None:
            _digest(license_hash, field=f"{identifier}.license.file_sha256")
        optional_diagnostic = kind == "diagnostic" and not bool((active_config or {}).get("julius_diagnostic", {}).get("enabled", False))
        if state not in {"approved", "reviewed"} and not optional_diagnostic:
            message = f"{identifier}: license status is {state}"
            if strict:
                raise SupplyChainError(message)
            warnings.append(message)
        artifacts = resource.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise SupplyChainError(f"{identifier}: artifacts must be a list")
        runtime_binding = resource.get("runtime_binding") if kind == "runtime" else None
        if strict and kind != "diagnostic" and not artifacts and not isinstance(runtime_binding, Mapping):
            raise SupplyChainError(f"{identifier}: production resource has no bound artifact")
        if not artifacts and kind != "diagnostic" and not isinstance(runtime_binding, Mapping):
            unbound += 1
            warnings.append(f"{identifier}: no local artifact is bound")
        evidence_hashes = resource.get("artifact_hashes", [])
        if not isinstance(evidence_hashes, list):
            raise SupplyChainError(f"{identifier}: artifact_hashes must be a list")
        for evidence in evidence_hashes:
            if not isinstance(evidence, Mapping) or not isinstance(evidence.get("id"), str):
                raise SupplyChainError(f"{identifier}: artifact evidence needs id")
            _digest(evidence.get("sha256"), field=f"{identifier}.artifact_hashes.sha256")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
                raise SupplyChainError(f"{identifier}: artifact path is required")
            expected = _digest(artifact.get("sha256"), field=f"{identifier}.artifact.sha256")
            artifact_path = _resolve_path(artifact["path"], root)
            if artifact.get("kind") == "directory" or artifact_path.is_dir():
                if artifact_path.is_symlink() or not artifact_path.is_dir():
                    raise SupplyChainError(f"{identifier}: artifact directory missing or symlinked: {artifact_path}")
                actual_files, actual = _directory_inventory(artifact_path)
                expected_files = artifact.get("files")
                if not isinstance(expected_files, list) or actual_files != expected_files:
                    raise SupplyChainError(f"{identifier}: directory artifact file inventory mismatch for {artifact_path}")
            else:
                if artifact_path.is_symlink() or not artifact_path.is_file():
                    raise SupplyChainError(f"{identifier}: artifact missing: {artifact_path}")
                actual = _sha256(artifact_path)
            if actual != expected:
                raise SupplyChainError(f"{identifier}: artifact hash mismatch for {artifact_path}")
            checked += 1
        if isinstance(runtime_binding, Mapping):
            invoked = _resolve_path(runtime_binding.get("invoked_path"), root)
            resolved = _resolve_path(runtime_binding.get("resolved_path"), root)
            if not invoked.exists() or resolved.is_symlink() or not resolved.is_file() or invoked.resolve() != resolved:
                raise SupplyChainError(f"{identifier}: runtime invoked/resolved path changed")
            if _sha256(resolved) != _digest(runtime_binding.get("resolved_sha256"), field=f"{identifier}.runtime_binding.resolved_sha256"):
                raise SupplyChainError(f"{identifier}: runtime resolved hash mismatch")
            receipt = runtime_binding.get("package_env_receipt")
            if strict and not isinstance(receipt, Mapping):
                raise SupplyChainError(f"{identifier}: package environment receipt is required")
            if isinstance(receipt, Mapping):
                receipt_path = _resolve_path(receipt.get("path"), root)
                if receipt_path.is_symlink() or not receipt_path.is_file() or _sha256(receipt_path) != _digest(receipt.get("sha256"), field=f"{identifier}.package_env_receipt.sha256"):
                    raise SupplyChainError(f"{identifier}: package environment receipt mismatch")
        license_path = license_info.get("path")
        if strict and not license_path and not optional_diagnostic:
            raise SupplyChainError(f"{identifier}: license artifact path is not bound")
        if license_path:
            candidate = Path(str(license_path)).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            expected = _digest(license_info.get("file_sha256"), field=f"{identifier}.license.file_sha256")
            if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
                raise SupplyChainError(f"{identifier}: license file hash mismatch or missing")
    if active_config:
        configured_paths = _config_paths(active_config)
        bound_paths = {str(Path(str(artifact.get("path"))).expanduser().absolute())
                       for resource in resources for artifact in resource.get("artifacts", [])
                       if isinstance(artifact, Mapping) and artifact.get("path")}
        for resource in resources:
            runtime_binding = resource.get("runtime_binding") if isinstance(resource, Mapping) else None
            if isinstance(runtime_binding, Mapping):
                for key in ("invoked_path", "resolved_path"):
                    if runtime_binding.get(key):
                        bound_paths.add(str(_resolve_path(runtime_binding[key], root).absolute()))
        for configured in configured_paths:
            path = Path(configured)
            if not path.exists():
                if strict:
                    raise SupplyChainError(f"active config asset is missing: {configured}")
                warnings.append(f"active config asset is missing: {configured}")
                continue
            if not any(configured == bound or Path(bound).is_relative_to(path) for bound in bound_paths):
                if strict:
                    raise SupplyChainError(f"active config asset is not bound in lock: {configured}")
                warnings.append(f"active config asset is not bound in lock: {configured}")
    if strict and lock.get("status") not in {"frozen", "verified"}:
        raise SupplyChainError("strict verification requires lock status frozen or verified")
    return {"schema": "ja-supply-chain-report-v1", "status": "PARTIAL" if warnings or unbound else "PASS", "strict": strict,
            "resources": len(resources), "artifacts_checked": checked, "unbound_resources": unbound, "warnings": warnings}


def _load_active_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SupplyChainError(f"active config cannot be read: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SupplyChainError("active config must be a mapping")
    return value


def _config_paths(value: Any, key: str = "") -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for name, child in value.items():
            result.update(_config_paths(child, str(name)))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_config_paths(child, key))
        return result
    asset_key = key.lower()
    # Manifests, the lock itself, and report/override files are bound by their
    # stage receipts or run identity. They are not model artifacts and must
    # not become impossible self-bindings in the production lock.
    excluded = ("manifest", "supply_chain_lock", "manual_overrides", "gold_manifest", "source_inventory", "receipt")
    is_asset_key = not any(token in asset_key for token in excluded) and any(
        token in asset_key for token in ("model", "acoustic", "dictionary", "wheel", "binary", "runtime", "mfa_root", "source_wav", "wav"))
    if is_asset_key and isinstance(value, str) and ("/" in value or "\\" in value):
        return {str(Path(value).expanduser().absolute())}
    return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--config", type=Path, help="active pipeline config; strict mode binds configured asset paths")
    parser.add_argument("--freeze-bindings", type=Path, help="JSON bindings used to materialize a local frozen lock")
    parser.add_argument("--output", type=Path, help="output path for --freeze-bindings")
    args = parser.parse_args(argv)
    try:
        payload = load_lock(args.lock)
        if args.freeze_bindings:
            bindings = json.loads(args.freeze_bindings.read_text(encoding="utf-8"))
            if not isinstance(bindings, Mapping):
                raise SupplyChainError("freeze bindings must be an object keyed by resource id")
            frozen = freeze_lock(payload, bindings)
            destination = (args.output or args.lock).expanduser().absolute()
            destination.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"status": "FROZEN", "lock": str(destination)}, ensure_ascii=False))
            return 0
        if args.strict and args.config:
            config = _load_active_config(args.config.expanduser().absolute())
            configured_lock = Path(str(config.get("supply_chain_lock", ""))).expanduser().absolute()
            if configured_lock != args.lock.expanduser().absolute():
                raise SupplyChainError("active config supply_chain_lock does not match --lock")
        report = verify_lock(payload, strict=args.strict, base_dir=args.lock.parent,
                             active_config=config if args.strict and args.config else None)
    except SupplyChainError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
