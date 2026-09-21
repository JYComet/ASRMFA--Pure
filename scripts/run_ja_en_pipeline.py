"""Executable foundation and dispatcher for the Japanese/English pipeline.

The dispatcher deliberately does not load ASR, frontend, or MFA packages.
It validates the immutable run contract first and invokes registered stage
functions in isolated stages/<name> namespaces. Feature stages can be
registered by their owning modules without changing this core file.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import importlib
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping
import os

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from .ja_en_schema import (
        JAContractError,
        PRODUCTION_STAGES,
        SCHEMAS,
        STAGE_NAMES,
        StageResult,
        atomic_write_json,
        exclusive_lock,
        ensure_output_path,
        list_relative_files,
        load_json,
        canonical_json,
        make_receipt,
        artifact_record,
        sha256_file,
        stable_digest,
        validate_config,
        validate_manifest,
        validate_receipt,
    )
except ImportError:  # direct script execution
    from ja_en_schema import (
        JAContractError,
        PRODUCTION_STAGES,
        SCHEMAS,
        STAGE_NAMES,
        StageResult,
        atomic_write_json,
        exclusive_lock,
        ensure_output_path,
        list_relative_files,
        load_json,
        canonical_json,
        make_receipt,
        artifact_record,
        sha256_file,
        stable_digest,
        validate_config,
        validate_manifest,
        validate_receipt,
    )


StageHandler = Callable[[Mapping[str, Any], Path], StageResult | Mapping[str, Any] | None]
_STAGE_REGISTRY: dict[str, tuple[StageHandler, str]] = {}


def register_stage(name: str, handler: StageHandler, *, output_namespace: str | None = None) -> None:
    """Register one stage and its exclusive output namespace."""
    if name not in STAGE_NAMES:
        raise ValueError(f"unknown Japanese/English stage: {name}")
    if not callable(handler):
        raise TypeError("stage handler must be callable")
    namespace = output_namespace or name
    if not namespace or "/" in namespace or "\\" in namespace or namespace in {".", ".."}:
        raise ValueError("stage namespace must be a single directory name")
    if name in _STAGE_REGISTRY and getattr(_STAGE_REGISTRY[name][0], "__name__", "") != "_skeleton_handler":
        raise ValueError(f"stage already registered: {name}")
    _STAGE_REGISTRY[name] = (handler, namespace)


def stage_registry() -> dict[str, dict[str, Any]]:
    return {
        name: {"owner": getattr(handler, "__module__", "unknown"), "namespace": namespace}
        for name, (handler, namespace) in sorted(_STAGE_REGISTRY.items())
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise JAContractError("config_malformed", "PyYAML is required to read YAML config", str(path))

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise JAContractError("config_malformed", f"duplicate key {key!r}", str(path))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.load(handle, Loader=UniqueKeyLoader)
    except JAContractError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise JAContractError("config_malformed", str(exc), str(path)) from exc
    return validate_config(value, config_path=path)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise JAContractError("manifest_invalid", str(exc), f"{path}:{line_number}") from exc
            return validate_manifest(rows)
        with path.open("r", encoding="utf-8") as handle:
            return validate_manifest(json.load(handle))
    except JAContractError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise JAContractError("manifest_invalid", str(exc), str(path)) from exc


def config_identity(config: Mapping[str, Any], manifest_path: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    # Source paths are retained verbatim: authorized source inputs may be
    # outside the output workspace and must not be rewritten.
    production_config = dict(config)
    # Julius is a diagnostic-only branch.  Its enablement and converter must
    # never invalidate production MFA/TTS identities; Julius owners build a
    # separate diagnostic cache key from the full config.
    production_config.pop("julius_diagnostic", None)

    def path_identity(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = Path(value).expanduser()
        if not candidate.exists():
            return {"path": value, "exists": False}
        if candidate.is_symlink():
            resolved = candidate.resolve()
            if resolved.is_file():
                return {"path": value, "exists": True, "symlink": True, "resolved": str(resolved),
                        "size": resolved.stat().st_size, "sha256": sha256_file(resolved)}
            # A symlinked model/data directory can retarget without changing
            # the link itself. Refuse it so production identities cannot miss
            # a tree mutation; runtime executable symlinks remain hash-bound
            # by the file branch above.
            if resolved.is_dir():
                raise JAContractError("config_malformed", "symlinked directories are not permitted for configured assets", str(candidate))
            return {"path": value, "exists": True, "symlink": True, "resolved": str(resolved)}
        if candidate.is_file():
            return {"path": value, "exists": True, "size": candidate.stat().st_size, "sha256": sha256_file(candidate)}
        if candidate.is_dir():
            entries = []
            for child in sorted(candidate.rglob("*")):
                if child.is_symlink():
                    target = child.resolve()
                    if target.is_dir():
                        raise JAContractError("config_malformed", "symlinked directories are not permitted inside configured assets", str(child))
                    entry = {"path": child.relative_to(candidate).as_posix(), "symlink": True, "resolved": str(target)}
                    if target.is_file():
                        entry.update({"size": target.stat().st_size, "sha256": sha256_file(target)})
                    entries.append(entry)
                elif child.is_file():
                    entries.append({"path": child.relative_to(candidate).as_posix(), "size": child.stat().st_size, "sha256": sha256_file(child)})
            return {"path": value, "exists": True, "directory": True, "entries": entries, "digest": stable_digest(entries)}
        return {"path": value, "exists": True, "other": True}

    def runtime_identity(value: str) -> Any:
        base = path_identity(value)
        executable = Path(value).expanduser()
        if not executable.is_file():
            return base
        probe = (
            "import hashlib, importlib.metadata as m, json, pathlib, sys; "
            "def digest(p): "
            " h=hashlib.sha256(); "
            " with open(p,'rb') as f: "
            "  [h.update(chunk) for chunk in iter(lambda:f.read(1024*1024),b'')]; "
            " return h.hexdigest(); "
            "rows=[]; "
            "for_dist = sorted(m.distributions(), key=lambda d: (str(d.metadata.get('Name','')).lower(), str(d.version))); "
            "for d in for_dist: "
            " files=sorted(str(f) for f in (d.files or [])); "
            " bound=[]; "
            " for f in (d.files or []): "
            "  p=pathlib.Path(d.locate_file(f)); "
            "  if p.is_file() and (str(f).endswith('.py') or str(f).endswith(('.so','.pyd','.pyi')) or str(f).endswith('.dist-info/RECORD')): "
            "   try: bound.append({'path':str(f),'size':p.stat().st_size,'sha256':digest(p)}) "
            "   except OSError: pass; "
            " rows.append({'name':str(d.metadata.get('Name','')),'version':str(d.version),'files':files,'bound_files':bound}); "
            "print(json.dumps({'executable': sys.executable, 'version': sys.version, 'distributions': rows}, sort_keys=True))"
        )
        try:
            completed = subprocess.run([str(executable), "-c", probe], capture_output=True, text=True, timeout=60, check=False)
            payload = json.loads(completed.stdout) if completed.returncode == 0 else None
            if not isinstance(payload, Mapping):
                return {"asset": base, "probe": "failed", "returncode": completed.returncode}
            return {"asset": base, "probe": "ok", "digest": stable_digest(payload), "details": payload}
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            return {"asset": base, "probe": "failed", "error": type(exc).__name__}

    source_artifacts = []
    for row in manifest:
        for field in ("wav", "audio", "source_wav", "script"):
            value = row.get(field)
            if isinstance(value, str):
                source_artifacts.append(path_identity(value))
    model_artifacts = {
        key: path_identity(value)
        for section in ("asr", "mfa", "frontend")
        for key, value in (config.get(section, {}) or {}).items()
        if key.endswith(("_model", "_acoustic", "_dictionary", "_wheel", "_binary", "_aligner", "_runtime", "_runtime_python"))
    }
    implementation_files = []
    for filename in (
        "ja_en_schema.py", "run_ja_en_pipeline.py", "ja_audio.py", "ja_asr_crossval.py",
        "ja_frontend.py", "ja_text_layers.py", "ja_phone_adapter.py", "ja_en_anchors.py",
        "ja_asr_provider_worker.py", "ja_canary_gate.py", "align_japanese_mfa.py",
        "merge_ja_en_mfa.py", "ja_tts_export.py", "verify_ja_en_tts.py", "verify_ja_supply_chain.py",
        "ja_en_stage_inputs.py",
    ):
        candidate = Path(__file__).resolve().parent / filename
        if candidate.is_file():
            implementation_files.append({"path": str(candidate), "sha256": sha256_file(candidate), "size": candidate.stat().st_size})
    config_paths: dict[str, Any] = {}
    def lock_identity(value: str) -> Any:
        candidate = Path(value).expanduser()
        if not candidate.is_file() or candidate.is_symlink():
            return path_identity(value)
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return path_identity(value)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("resources"), list):
            return path_identity(value)
        # Diagnostic Julius resources are intentionally outside production
        # cache identity. The complete lock is still supplied to verification.
        production = {key: payload.get(key) for key in ("schema", "status", "policy")}
        production["resources"] = [row for row in payload["resources"] if not isinstance(row, Mapping) or row.get("kind") != "diagnostic"]
        return {"path": value, "exists": True, "scope": "production", "sha256": stable_digest(production)}
    def collect_paths(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                collect_paths(child, f"{key}.{child_key}" if key else str(child_key))
        elif isinstance(value, list):
            for child in value:
                collect_paths(child, key)
        elif isinstance(value, str) and (
            key.endswith(("_path", "_file", "_model", "_acoustic", "_dictionary", "_metadata", "_wheel", "_binary", "_aligner", "_manifest", "_lock", "_receipt", ".model", ".runtime_python", ".runtime"))
            or key in {"input_manifest", "supply_chain_lock", "mapping_file", "gold_manifest", "manual_overrides", "runtime_python"}
        ):
            if key.endswith("runtime_python") or key in {"runtime_python", "runtime"}:
                config_paths[key] = runtime_identity(value)
            else:
                config_paths[key] = lock_identity(value) if key.endswith("_lock") or key.endswith(".lock") else path_identity(value)
    collect_paths(production_config)
    return {
        "schema": "ja-pipeline-receipt-v1",
        "pipeline": config["pipeline"],
        "config": production_config,
        "config_digest": stable_digest(production_config),
        "manifest_path": str(manifest_path),
        "manifest_digest": stable_digest(manifest),
        "source_artifacts": source_artifacts,
        "model_artifacts": model_artifacts,
        "config_artifacts": config_paths,
        "implementation_files": implementation_files,
        "schemas": sorted(SCHEMAS),
        "stage_registry": stage_registry(),
    }


def preflight(config: Mapping[str, Any], *, config_path: Path | None = None) -> tuple[dict[str, Any], Path, list[dict[str, Any]], Path]:
    validated = validate_config(config, config_path=config_path)
    base = config_path.parent if config_path is not None else Path.cwd()
    def resolve(value: Any, key: str = "") -> Any:
        if isinstance(value, Mapping):
            return {str(child_key): resolve(child, str(child_key)) for child_key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, key) for child in value]
        if isinstance(value, str) and (
            key in {"workspace", "input_manifest", "supply_chain_lock", "mapping_file", "gold_manifest", "manual_overrides", "runtime_python"}
            or key in {"model", "aligner", "runtime"}
            or key.endswith(("_path", "_file", "_model", "_acoustic", "_dictionary", "_metadata", "_wheel", "_binary", "_manifest", "_lock", "_receipt"))
        ):
            candidate = Path(value).expanduser()
            return str(candidate if candidate.is_absolute() else (base / candidate).absolute())
        return value
    validated = resolve(validated)
    manifest_path = Path(str(validated["input_manifest"])).expanduser().absolute()
    if manifest_path.is_symlink():
        raise JAContractError("manifest_symlink", "manifest is a symlink", str(manifest_path))
    if not manifest_path.is_file():
        raise JAContractError("manifest_invalid", "manifest file does not exist", str(manifest_path))
    manifest = _load_manifest(manifest_path)
    for row in manifest:
        for field in ("wav", "audio", "source_wav", "script"):
            if isinstance(row.get(field), str):
                candidate = Path(row[field]).expanduser()
                if not candidate.is_absolute():
                    candidate = manifest_path.parent / candidate
                if candidate.is_symlink():
                    raise JAContractError("manifest_symlink", "source input is symlinked", str(candidate.absolute()))
                row[field] = str(candidate.absolute())
    workspace = Path(str(validated["workspace"])).expanduser().absolute()
    if workspace.exists() and workspace.is_symlink():
        raise JAContractError("manifest_symlink", "workspace is a symlink", str(workspace))
    return validated, manifest_path, manifest, workspace


def _identity_path(workspace: Path) -> Path:
    return workspace / ".ja_en_run_identity.json"


def _production_lock_payload(lock_path: Path) -> dict[str, Any]:
    if not lock_path.is_file():
        return {"schema": "ja-supply-chain-lock-v1", "status": "missing", "policy": {}, "resources": [], "source_path": str(lock_path)}
    payload = load_json(lock_path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("resources"), list):
        raise JAContractError("supply_chain_invalid", "supply-chain lock has no resource list", str(lock_path))
    return {
        key: payload.get(key) for key in ("schema", "status", "policy")
    } | {
        "resources": [row for row in payload["resources"] if not isinstance(row, Mapping) or row.get("kind") != "diagnostic"]
    }


def _ensure_production_snapshots(workspace: Path, config: Mapping[str, Any], *, resume: bool) -> tuple[Path, Path]:
    """Freeze production inputs while leaving diagnostic config independently scoped."""
    inventory_dir = workspace / "stages" / "inventory" / "run_contract"
    config_snapshot = inventory_dir / "production_config.json"
    lock_snapshot = inventory_dir / "production_supply_chain_lock.json"
    if resume:
        if not config_snapshot.is_file() or not lock_snapshot.is_file():
            raise JAContractError("resume_stale", "production input snapshots are missing", str(inventory_dir))
        return config_snapshot, lock_snapshot
    production = dict(config)
    production.pop("julius_diagnostic", None)
    atomic_write_json(config_snapshot, production, workspace=workspace)
    lock_path = Path(str(config["supply_chain_lock"]))
    atomic_write_json(lock_snapshot, _production_lock_payload(lock_path), workspace=workspace)
    return config_snapshot, lock_snapshot


def validate_resume(workspace: Path, identity: Mapping[str, Any], *, allow_new: bool = True) -> None:
    """Validate resume state before any stage handler can be imported/loaded."""
    identity_path = _identity_path(workspace)
    if not workspace.exists():
        if allow_new:
            return
        raise JAContractError("resume_stale", "workspace is missing", str(workspace))
    if workspace.is_symlink():
        raise JAContractError("manifest_symlink", "workspace is a symlink", str(workspace))
    if not identity_path.is_file() or identity_path.is_symlink():
        if allow_new and workspace.is_dir():
            residual = {child.name for child in workspace.iterdir() if child.name != ".ja_en.lock"}
            if not residual:
                return
        raise JAContractError("resume_stale", "run identity is missing", str(identity_path))
    existing = load_json(identity_path)
    if not isinstance(existing, Mapping) or existing.get("identity_digest") != stable_digest(identity):
        raise JAContractError("resume_identity_drift", "run identity changed", str(identity_path))
    for root_receipt in (workspace / ".ja_en_pipeline_receipt.json", workspace / "receipt.json"):
        if root_receipt.is_file():
            validate_receipt(load_json(root_receipt), workspace=workspace)
    allowed = {".ja_en_run_identity.json", ".ja_en_pipeline_receipt.json", ".ja_en_stage_cache.json", "receipt.json", ".ja_en.lock"}
    allowed_prefixes = {f"stages/{ns}/" for _, ns in _STAGE_REGISTRY.values()}
    for relative in list_relative_files(workspace):
        if relative in allowed or any(relative.startswith(prefix) for prefix in allowed_prefixes):
            continue
        raise JAContractError("resume_extra_file", f"unexpected workspace file {relative}", str(workspace / relative))
    # A stage namespace is owned by one stage.  Its receipt defines every
    # output path; unlisted files are stale/extra and fail before a handler can
    # load a model.
    for stage, (_, namespace) in _STAGE_REGISTRY.items():
        stage_dir = workspace / "stages" / namespace
        if not stage_dir.exists():
            continue
        receipt_path = stage_dir / "receipt.json"
        if not receipt_path.is_file():
            raise JAContractError("resume_stale", "stage receipt is missing", str(stage_dir))
        receipt = load_json(receipt_path)
        validate_receipt(receipt, workspace=workspace)
        expected = {receipt_path.relative_to(workspace).as_posix()}
        for output in receipt.get("outputs", []):
            expected.add(Path(output["path"]).absolute().relative_to(workspace.absolute()).as_posix())
        actual = {path for path in list_relative_files(stage_dir)}
        actual = {f"stages/{namespace}/{path}" for path in actual}
        if actual != expected:
            raise JAContractError("resume_extra_file", f"stage output set changed: {sorted(actual ^ expected)}", str(stage_dir))


def _autoload_stages() -> None:
    """Load stage registrations without importing model runtimes."""
    modules = (
        "ja_audio", "ja_asr_crossval", "ja_frontend", "ja_phone_adapter",
        "ja_en_anchors", "align_japanese_mfa", "merge_ja_en_mfa",
        "run_julius_diagnostic", "ja_tts_export",
    )
    for name in modules:
        try:
            module = importlib.import_module(f"scripts.{name}")
        except ModuleNotFoundError as exc:
            if exc.name not in {f"scripts.{name}", "scripts"}:
                raise
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError:
                if name == "run_julius_diagnostic":
                    continue
                raise
        registrar = getattr(module, "register_stages", None)
        if callable(registrar):
            registrar(register_stage)
    if getattr(_STAGE_REGISTRY.get("verify", (None,))[0], "__name__", "") == "_skeleton_handler":
        register_stage("verify", _verify_stage, output_namespace="verify")


def _verify_stage(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    try:
        from scripts.verify_ja_en_tts import verify_workspace
    except ModuleNotFoundError:
        from verify_ja_en_tts import verify_workspace

    receipt_path = stage_dir / "receipt.json"
    target = stage_dir.parent.parent
    try:
        report = verify_workspace(target)
    except Exception as exc:
        report = {"ok": False, "status": "REJECTED", "errors": [{"code": "verifier_failed", "message": str(exc)}]}
    report_path = stage_dir / "independent_verification.json"
    atomic_write_json(report_path, report, workspace=stage_dir.parent.parent)
    publish = config.get("publish", {}) if isinstance(config.get("publish"), Mapping) else {}
    profile = (config.get("asr") or {}).get("profile")
    allowed = bool(report.get("ok")) and bool(report.get("release_ready", report.get("ok")))
    # qwen_only_dev and fixture/synthetic evidence can execute verification but
    # are never promoted to production COMPLETE.
    if profile == "qwen_only_dev" or config.get("synthetic_fixture") is True:
        allowed = False
    status = "COMPLETE" if allowed and publish.get("require_independent_verifier", True) else ("BLOCKED" if not report.get("ok") else "PARTIAL")
    errors = [] if status == "COMPLETE" else [{"code": "verifier_failed" if not report.get("ok") else "publish_blocked", "message": "independent verifier did not authorize publication"}]
    receipt = make_receipt(stage="verify", status=status, outputs=[report_path], params={"implementation": "independent-verifier", "release_ready": allowed}, errors=errors)
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult("verify", status, str(receipt_path))


def _workspace_output_paths(workspace: Path) -> list[Path]:
    paths = []
    operational = {"receipt.json", ".ja_en_pipeline_receipt.json", ".ja_en_run_identity.json", ".ja_en_stage_cache.json", ".ja_en.lock"}
    for path in workspace.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name not in operational:
            paths.append(path)
    return sorted(paths)


def _stage_receipt(workspace: Path, stage: str) -> Mapping[str, Any] | None:
    namespace = _STAGE_REGISTRY.get(stage, (None, stage))[1]
    path = workspace / "stages" / namespace / "receipt.json"
    if not path.is_file():
        return None
    receipt = load_json(path)
    validate_receipt(receipt, workspace=workspace)
    return receipt


def _stage_cache_identity(stage: str, identity: Mapping[str, Any], config: Mapping[str, Any], workspace: Path) -> str:
    upstream: dict[str, Any] = {}
    for predecessor in PRODUCTION_STAGES:
        if predecessor == stage:
            break
        if stage == "reading" and predecessor == "frontend":
            # Reading consumes the immutable candidate analysis payload. The
            # frontend receipt is later rewritten for locked reconstruction;
            # binding the whole receipt here creates a frontend↔reading cycle.
            continue
        receipt = _stage_receipt(workspace, predecessor)
        if receipt is not None:
            upstream[predecessor] = stable_digest(receipt)
    if stage == "reading":
        analysis_path = workspace / "stages" / _STAGE_REGISTRY.get("frontend", (None, "frontend"))[1] / "frontend_analysis.json"
        if analysis_path.is_file():
            upstream["frontend_candidate_analysis"] = artifact_record(analysis_path)
    if stage == "frontend":
        reading_receipt = _stage_receipt(workspace, "reading")
        if reading_receipt is not None:
            upstream["reading_locked_reconstruction"] = stable_digest(reading_receipt)
    payload: dict[str, Any] = {"stage": stage, "global_identity": identity, "upstream_receipts": upstream}
    if stage == "julius":
        payload["diagnostic_config"] = config.get("julius_diagnostic", {})
    return stable_digest(payload)


def _run_stage(handler: StageHandler, config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    stage = stage_dir.name if stage_dir.name in STAGE_NAMES else "inventory"
    try:
        result = handler(config, stage_dir)
    except JAContractError as exc:
        receipt_path = stage_dir / "receipt.json"
        atomic_write_json(receipt_path, make_receipt(stage=stage, status="REJECTED", params={"implementation": "dispatcher"}, errors=[exc.as_dict()]), workspace=stage_dir.parent.parent)
        return StageResult(stage, "REJECTED", str(receipt_path))
    except Exception as exc:
        receipt_path = stage_dir / "receipt.json"
        atomic_write_json(receipt_path, make_receipt(stage=stage, status="REJECTED", params={"implementation": "dispatcher"}, errors=[{"code": "verifier_failed", "message": f"{type(exc).__name__}: {exc}"}]), workspace=stage_dir.parent.parent)
        return StageResult(stage, "REJECTED", str(receipt_path))
    if isinstance(result, StageResult):
        return result
    if isinstance(result, Mapping):
        return StageResult(stage=stage_dir.name, status=str(result.get("status", "PARTIAL")), receipt_path=result.get("receipt_path"))
    return StageResult(stage=stage_dir.name, status="PARTIAL")


def _run_locked_frontend_reconstruction(config: Mapping[str, Any], workspace: Path) -> StageResult:
    """Adapt W1 lock rows to the W2 request envelope without changing evidence."""
    frontend_namespace = _STAGE_REGISTRY["frontend"][1]
    frontend_dir = workspace / "stages" / frontend_namespace
    reading_path = workspace / "stages" / _STAGE_REGISTRY["reading"][1] / "locked_readings.jsonl"
    if not reading_path.is_file():
        return StageResult("frontend", "BLOCKED", str(frontend_dir / "receipt.json"))
    adapter_path = frontend_dir / "locked_readings_request.jsonl"
    analysis_path = frontend_dir / "frontend_analysis.json"
    analyses: dict[str, Mapping[str, Any]] = {}
    if analysis_path.is_file():
        payload = load_json(analysis_path)
        analyses = {str(row.get("uid")): row for row in payload.get("records", []) if isinstance(row, Mapping)}
    adapted: list[str] = []
    for line in reading_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        analysis = analyses.get(str(row.get("uid")), {})
        row.setdefault("frontend_profile", analysis.get("frontend_profile", row.get("analysis_profile")))
        row.setdefault("frontend_options_digest", analysis.get("frontend_options_digest"))
        row.setdefault("analysis_digest", analysis.get("analysis_digest", row.get("analysis_version_digest")))
        adapted.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    adapter_path.write_text("\n".join(adapted) + ("\n" if adapted else ""), encoding="utf-8")
    adapted_config = copy.deepcopy(dict(config))
    adapted_config.setdefault("frontend", {})["locked_readings_path"] = str(adapter_path)
    result = _run_stage(_STAGE_REGISTRY["frontend"][0], adapted_config, frontend_dir)
    receipt_path = frontend_dir / "receipt.json"
    if receipt_path.is_file():
        receipt = dict(load_json(receipt_path))
        outputs = list(receipt.get("outputs", []))
        if str(adapter_path) not in {str(item.get("path")) for item in outputs if isinstance(item, Mapping)}:
            outputs.append(artifact_record(adapter_path))
            receipt["outputs"] = outputs
            atomic_write_json(receipt_path, receipt, workspace=workspace)
    return result


def _normalize_stage_receipt(stage: str, stage_dir: Path, result: StageResult, workspace: Path) -> StageResult:
    """Check producer receipts and fail closed on omitted output artifacts."""
    receipt_path = stage_dir / "receipt.json"
    if not receipt_path.is_file():
        return result
    receipt = dict(load_json(receipt_path))
    if stage == "tts":
        missing = False
        for row in receipt.get("outputs", []):
            path = Path(str(row.get("path", "")))
            if not path.is_file():
                missing = True
        # A missing upstream merge is a normal DAG block.  REJECTED is
        # reserved for a producer that claimed completion while omitting or
        # tampering with its declared artifacts.
        if receipt.get("status") in {"COMPLETE", "PARTIAL"} and (missing or not receipt.get("outputs")):
            receipt["status"] = "REJECTED"
            receipt.setdefault("errors", []).append({"code": "tts_invalid", "message": "TTS receipt references missing output artifacts"})
        atomic_write_json(receipt_path, receipt, workspace=workspace)
        return StageResult(result.stage, receipt["status"], str(receipt_path))
    return result


def _run_uid_batch(stage: str, requests: Sequence[Mapping[str, Any]], config: Mapping[str, Any], workspace: Path) -> StageResult:
    """Run a single-UID handler in isolated workspaces and aggregate receipts."""
    handler, namespace = _STAGE_REGISTRY[stage]
    root_stage = workspace / "stages" / namespace
    output_paths: list[Path] = []
    statuses: list[str] = []
    plans: list[dict[str, Any]] = []
    merged_payloads: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    # A bridge request may be rejected for one UID while its siblings remain
    # runnable.  Keep that row in the stage ledger and never let it disappear
    # merely because the handler only accepts executable requests.
    blocked_uids: list[str] = []
    for request in requests:
        uid = str(request.get("uid"))
        request_status = str(request.get("status", "READY"))
        if request_status in {"REJECTED", "BLOCKED", "UNRESOLVED"}:
            blocked_uids.append(uid)
            for error in request.get("errors", []):
                if isinstance(error, Mapping):
                    errors.append({"uid": uid, **dict(error)})
            continue
        isolated = Path(tempfile.mkdtemp(prefix=f"ja-en-{stage}-{uid}-"))
        isolated_stage = isolated / "stages" / namespace
        isolated_stage.mkdir(parents=True, exist_ok=True)
        stage_config = copy.deepcopy(dict(config))
        request_payload = dict(request)
        if stage == "anchors":
            qwen = request.get("qwen", {}) if isinstance(request.get("qwen"), Mapping) else {}
            request_payload = {"uid": uid, "audio": request["audio"]["path"],
                               "spoken_text": request["canonical_spoken_text"], "units": request["lexical_units"],
                               "route": request["route"], "sample_rate": request["audio"]["sample_rate"],
                               "total_samples": request["audio"]["frames"], "qwen_forced_aligner": qwen.get("model"),
                               "device": qwen.get("device"), "dtype": qwen.get("dtype"),
                               "qwen_runtime_python": (config.get("asr") or {}).get("qwen_runtime_python")}
        elif stage == "align":
            mfa = config.get("mfa", {}) if isinstance(config.get("mfa"), Mapping) else {}
            runs = []
            for run in request.get("runs", []):
                crop_dir = Path(str(run["crop"]["path"])).parent
                runs.append({**{key: run[key] for key in ("run_id", "language", "unit_ids", "ownership_start_sample", "ownership_end_sample", "context_start_sample", "context_end_sample")},
                             "aliases": run["aliases"], "corpus_dir": str(crop_dir),
                             "acoustic_model": mfa.get("japanese_acoustic" if run["language"] == "ja" else "english_acoustic"),
                             "dictionary": str(run["locked_dictionary"]["path"]),
                             "native_inventory_path": mfa.get("japanese_metadata" if run["language"] == "ja" else "english_metadata"),
                             "output_dir": str(isolated_stage / "mfa_output" / str(run["run_id"])),
                             "temporary_directory": tempfile.mkdtemp(prefix=f"ja-en-mfa-{run['run_id']}-"),
                             "runtime_python": mfa.get("runtime_python"), "sample_rate": run["crop"]["sample_rate"],
                             "offset_sample": run["global_offset_sample"], "total_samples": run["crop"]["frames"]})
            request_payload = {"uid": uid, "runs": runs}
        elif stage == "merge":
            align_dir = workspace / "stages" / "align" / "uids" / uid
            request_payload = {"uid": uid,
                "japanese_ledger": str(align_dir / "strict_ja_mfa.json") if (align_dir / "strict_ja_mfa.json").is_file() else [],
                "english_ledger": str(align_dir / "strict_en_mfa.json") if (align_dir / "strict_en_mfa.json").is_file() else [],
                "expected_languages": request.get("expected_languages", {}), "ownership": [0, max((int(v["end_sample"]) for v in request.get("ownership", {}).values()), default=1)],
                "sample_rate": int((request.get("source_alignment") or {}).get("sample_rate", 16000)),
                "words": request.get("expected_units", []), "seams": request.get("seams", []),
                "rerun_plan": request.get("rerun_plan")}
        stage_config.setdefault("stage_inputs", {})[stage] = request_payload
        result = _run_stage(handler, stage_config, isolated_stage)
        statuses.append(result.status)
        source_receipt = isolated_stage / "receipt.json"
        if source_receipt.is_file():
            payload = load_json(source_receipt)
            errors.extend(payload.get("errors", []))
        destination = root_stage / "uids" / uid
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(isolated_stage.rglob("*")):
            if not source.is_file() or source.name == "receipt.json":
                continue
            relative = source.relative_to(isolated_stage)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            output_paths.append(target)
            if stage == "anchors" and target.name == "anchor_plan.json":
                plans.append(load_json(target))
            if stage == "merge" and target.name == "ja_en_alignment.json":
                merged_payloads.append(load_json(target))
        # The temporary execution directory is intentionally outside the
        # workspace.  It is not part of the receipt and can be reclaimed by
        # the host after this run.
        shutil.rmtree(isolated, ignore_errors=True)
    if stage == "anchors" and plans:
        aggregate = root_stage / "anchor_plans.json"
        atomic_write_json(aggregate, {"schema": "ja-en-alignment-plan-v2", "plans": plans}, workspace=workspace)
        output_paths.append(aggregate)
    if stage == "merge" and merged_payloads:
        aggregate = root_stage / "ja_en_alignments.json"
        atomic_write_json(aggregate, {"schema": "ja-en-alignment-v2", "alignments": merged_payloads}, workspace=workspace)
        output_paths.append(aggregate)
    if blocked_uids:
        blocked_path = root_stage / "uid_errors.json"
        atomic_write_json(blocked_path, {
            "schema": "ja-en-uid-error-ledger-v1",
            "stage": stage,
            "expected_uids": [str(request.get("uid")) for request in requests],
            "blocked_uids": sorted(set(blocked_uids)),
            "errors": errors,
        }, workspace=workspace)
        output_paths.append(blocked_path)
    if not statuses:
        status = "BLOCKED"
    elif all(value == "COMPLETE" for value in statuses) and not blocked_uids:
        status = "COMPLETE"
    elif all(value == "BLOCKED" for value in statuses):
        status = "BLOCKED"
    else:
        status = "PARTIAL"
    receipt = make_receipt(stage=stage, status=status, outputs=output_paths,
                           params={"uid_count": len(requests), "uid_scoped": True,
                                   "blocked_uid_count": len(set(blocked_uids))}, errors=errors)
    receipt_path = root_stage / "receipt.json"
    atomic_write_json(receipt_path, receipt, workspace=workspace)
    return StageResult(stage, status, str(receipt_path))


def _append_unlisted_stage_outputs(stage_dir: Path, workspace: Path) -> None:
    receipt_path = stage_dir / "receipt.json"
    if not receipt_path.is_file():
        return
    receipt = dict(load_json(receipt_path))
    outputs = list(receipt.get("outputs", []))
    declared = {str(row.get("path")) for row in outputs if isinstance(row, Mapping)}
    for path in sorted(stage_dir.rglob("*")):
        if path.is_file() and path.name != "receipt.json" and str(path.absolute()) not in declared:
            outputs.append(artifact_record(path))
    if outputs != receipt.get("outputs", []):
        receipt["outputs"] = outputs
        atomic_write_json(receipt_path, receipt, workspace=workspace)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _uid_error_rows(workspace: Path, stage: str, schema: str) -> list[dict[str, Any]]:
    """Adapt an upstream UID error ledger to the next stage contract."""
    path = workspace / "stages" / stage / "uid_errors.json"
    if not path.is_file() or path.is_symlink():
        return []
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        return []
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for error in payload.get("errors", []):
        if isinstance(error, Mapping) and error.get("uid"):
            uid = str(error["uid"])
            by_uid.setdefault(uid, []).append({key: value for key, value in error.items() if key != "uid"})
    return [{"schema": schema, "uid": uid, "status": "REJECTED", "errors": errors}
            for uid, errors in sorted(by_uid.items())]


def _prepare_stage_config(config: Mapping[str, Any], stage: str, workspace: Path) -> Mapping[str, Any]:
    """Build declared mechanical inputs from immutable upstream artifacts."""
    prepared = copy.deepcopy(dict(config))
    stage_inputs = prepared.setdefault("stage_inputs", {})
    try:
        from .ja_en_stage_inputs import assemble_tts_rows, prepare_alignment_requests, prepare_anchor_requests, prepare_merge_requests
    except ImportError:
        try:
            from ja_en_stage_inputs import assemble_tts_rows, prepare_alignment_requests, prepare_anchor_requests, prepare_merge_requests
        except ImportError:  # pragma: no cover - compatibility with foundation-only installs
            assemble_tts_rows = prepare_alignment_requests = prepare_anchor_requests = prepare_merge_requests = None
    if prepare_anchor_requests is not None and stage == "anchors" and "anchors" not in prepared and "anchors" not in stage_inputs:
        requests = prepare_anchor_requests(prepared, workspace)
        stage_inputs["_uid_requests"] = requests
        ready_requests = [request for request in requests if request.get("status") not in {"REJECTED", "BLOCKED", "UNRESOLVED"}]
        if len(ready_requests) == 1 and len(requests) == 1:
            request = ready_requests[0]
            qwen = request.get("qwen", {})
            stage_inputs["anchors"] = {
                "uid": request["uid"], "audio": request["audio"]["path"],
                "spoken_text": request["canonical_spoken_text"], "units": request["lexical_units"],
                "route": request["route"], "sample_rate": request["audio"]["sample_rate"],
                "total_samples": request["audio"]["frames"], "qwen_forced_aligner": qwen.get("model"),
                "device": qwen.get("device"), "dtype": qwen.get("dtype"),
                "qwen_runtime_python": (prepared.get("asr") or {}).get("qwen_runtime_python"),
            }
    if prepare_alignment_requests is not None and stage == "align" and "align" not in prepared and "align" not in stage_inputs:
        plan_path = workspace / "stages" / "anchors" / "anchor_plan.json"
        aggregate_path = workspace / "stages" / "anchors" / "anchor_plans.json"
        if aggregate_path.is_file():
            plans = list(load_json(aggregate_path).get("plans", []))
        elif plan_path.is_file():
            plans = [load_json(plan_path)]
        else:
            plans = []
        if plans:
            requests = prepare_alignment_requests(prepared, workspace, plans)
            requests.extend(_uid_error_rows(workspace, "anchors", "ja-en-alignment-request-v1"))
            stage_inputs["_uid_requests"] = requests
            mfa = prepared.get("mfa", {}) if isinstance(prepared.get("mfa"), Mapping) else {}
            runs = []
            for request in requests:
                if request.get("status") in {"REJECTED", "BLOCKED", "UNRESOLVED"}:
                    continue
                for run in request["runs"]:
                    run_dir = Path(run["crop"]["path"]).parent
                    runs.append({**{key: run[key] for key in ("run_id", "language", "unit_ids", "ownership_start_sample", "ownership_end_sample", "context_start_sample", "context_end_sample")},
                                 "aliases": run["aliases"], "corpus_dir": str(run_dir),
                                 "acoustic_model": mfa.get("japanese_acoustic" if run["language"] == "ja" else "english_acoustic"),
                                 "dictionary": str(run["locked_dictionary"]["path"]),
                                 "native_inventory_path": mfa.get("japanese_metadata" if run["language"] == "ja" else "english_metadata"),
                                 "output_dir": str(workspace / "stages" / "align" / "mfa_output" / run["run_id"]),
                                 "temporary_directory": tempfile.mkdtemp(prefix=f"ja-en-mfa-{run['run_id']}-"),
                                 "runtime_python": mfa.get("runtime_python"), "sample_rate": run["crop"]["sample_rate"],
                                 "offset_sample": run["global_offset_sample"], "total_samples": run["crop"]["frames"]})
            ready_requests = [request for request in requests if request.get("status") not in {"REJECTED", "BLOCKED", "UNRESOLVED"}]
            if runs and len(ready_requests) == 1:
                stage_inputs["align"] = {"uid": ready_requests[0]["uid"], "runs": runs}
        else:
            blocked = _uid_error_rows(workspace, "anchors", "ja-en-alignment-request-v1")
            if blocked:
                stage_inputs["_uid_requests"] = blocked
    if prepare_merge_requests is not None and stage == "merge" and "merge" not in prepared and "merge" not in stage_inputs:
        plan_path = workspace / "stages" / "anchors" / "anchor_plan.json"
        aggregate_path = workspace / "stages" / "anchors" / "anchor_plans.json"
        plans = list(load_json(aggregate_path).get("plans", [])) if aggregate_path.is_file() else ([load_json(plan_path)] if plan_path.is_file() else [])
        if plans:
            alignment_requests = prepare_alignment_requests(prepared, workspace, plans)
            merge_requests = prepare_merge_requests(prepared, workspace, alignment_requests)
            merge_requests.extend(_uid_error_rows(workspace, "anchors", "ja-en-merge-request-v1"))
            merge_requests.extend(_uid_error_rows(workspace, "align", "ja-en-merge-request-v1"))
            stage_inputs["_uid_requests"] = merge_requests
            ready_requests = [request for request in merge_requests if request.get("status") not in {"REJECTED", "BLOCKED", "UNRESOLVED"}]
            if len(ready_requests) == 1:
                request = ready_requests[0]
                align_dir = workspace / "stages" / "align"
                stage_inputs["merge"] = {"uid": request["uid"],
                    "japanese_ledger": str(align_dir / "strict_ja_mfa.json") if (align_dir / "strict_ja_mfa.json").is_file() else [],
                    "english_ledger": str(align_dir / "strict_en_mfa.json") if (align_dir / "strict_en_mfa.json").is_file() else [],
                    "expected_languages": request["expected_languages"], "ownership": [0, int(request["runs"][0]["source_audio"]["frames"])],
                    "sample_rate": int(request["runs"][0]["source_audio"]["sample_rate"]), "words": plans[0].get("units", []),
                    "seams": request.get("seams", []), "rerun_plan": request.get("rerun_plan")}
        else:
            blocked = (_uid_error_rows(workspace, "anchors", "ja-en-merge-request-v1") +
                       _uid_error_rows(workspace, "align", "ja-en-merge-request-v1"))
            if blocked:
                stage_inputs["_uid_requests"] = blocked
    if assemble_tts_rows is not None and stage == "tts" and "tts" not in prepared and "tts" not in stage_inputs:
        aggregate = workspace / "stages" / "merge" / "ja_en_alignments.json"
        merged = list(load_json(aggregate).get("alignments", [])) if aggregate.is_file() else []
        single = workspace / "stages" / "merge" / "ja_en_alignment.json"
        if not merged and single.is_file():
            merged = [load_json(single)]
        if merged:
            records: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for alignment in merged:
                uid = str(alignment.get("uid", "")) if isinstance(alignment, Mapping) else ""
                try:
                    records.extend(assemble_tts_rows(prepared, workspace, [alignment]))
                except Exception as exc:
                    error = exc.as_dict() if isinstance(exc, JAContractError) else {"code": "publish_blocked", "message": str(exc)}
                    errors.append({"uid": uid, **error})
            if records:
                # assemble_tts_rows writes the same canonical file for each
                # UID.  Re-write the aggregate once so multi-UID order and
                # the complete partition are explicit in the final artifact.
                rows_path = workspace / "stages" / "tts" / "tts_training_records.jsonl"
                atomic_write_bytes(rows_path, b"".join(canonical_json(row) for row in records), workspace=workspace)
                stage_inputs["tts"] = {"alignment_jsonl": str(rows_path)}
            if errors:
                error_path = workspace / "stages" / "tts" / "uid_errors.json"
                atomic_write_json(error_path, {"schema": "ja-en-uid-error-ledger-v1", "stage": "tts",
                                                "expected_uids": [str(row.get("uid", "")) for row in merged],
                                                "blocked_uids": sorted({str(row.get("uid", "")) for row in errors}),
                                                "errors": errors}, workspace=workspace)
                stage_inputs["_tts_uid_errors"] = errors
            if not records:
                stage_inputs["_tts_bridge_error"] = {"code": "publish_blocked", "message": "no merged UID produced an authoritative TTS record"}
            stage_inputs["tts"] = {"alignment_jsonl": str(workspace / "stages" / "tts" / "tts_training_records.jsonl")}
    if stage == "anchors" and "anchors" not in prepared and "anchors" not in stage_inputs:
        manifest_path = Path(str(prepared["input_manifest"]))
        rows = _load_manifest(manifest_path)
        records_path = workspace / "stages" / "frontend" / "frontend_reconstruction.json"
        if not records_path.is_file():
            records_path = workspace / "stages" / "frontend" / "frontend_analysis.json"
        audio_rows = {str(row.get("uid")): row for row in _jsonl_rows(workspace / "stages" / "audio" / "audio_transform_receipts.jsonl")}
        records = load_json(records_path).get("records", []) if records_path.is_file() else []
        if len(rows) == 1 and len(records) == 1:
            source = audio_rows.get(str(rows[0].get("uid")), {})
            alignment = source.get("alignment", {}) if isinstance(source, Mapping) else {}
            units = []
            for index, unit in enumerate(records[0].get("units", [])):
                if unit.get("lexical_status") != "lexical":
                    continue
                units.append({
                    "unit_id": str(unit.get("token_id", f"tok_{index:06d}")),
                    "text": str(unit.get("caller_surface", unit.get("surface", ""))),
                    "char_span": list(unit.get("canonical_span", unit.get("orig_span", [0, 0]))),
                    "language": "en" if unit.get("language") == "en" else "ja",
                })
            if isinstance(alignment, Mapping) and units:
                stage_inputs["anchors"] = {
                    "uid": str(rows[0].get("uid")),
                    "audio": alignment.get("path"),
                    "spoken_text": str(rows[0].get("text", rows[0].get("orig_text", ""))),
                    "units": units,
                    "route": "mixed" if {unit["language"] for unit in units} == {"ja", "en"} else units[0]["language"],
                    "sample_rate": int(alignment.get("sample_rate", 16000)),
                    "total_samples": int(alignment.get("frames", 0)),
                    "qwen_forced_aligner": (prepared.get("asr") or {}).get("qwen_forced_aligner"),
                    "qwen_runtime_python": (prepared.get("asr") or {}).get("qwen_runtime_python"),
                    "device": (prepared.get("asr") or {}).get("device", "cuda:0"),
                    "dtype": (prepared.get("asr") or {}).get("dtype", "bfloat16"),
                }
    if stage == "align" and "align" not in prepared and "align" not in stage_inputs:
        plan_path = workspace / "stages" / "anchors" / "anchor_plan.json"
        alias_path = workspace / "stages" / "semantic" / "alias_map.jsonl"
        if plan_path.is_file() and alias_path.is_file():
            plan = load_json(plan_path)
            aliases = _jsonl_rows(alias_path)
            audio_rows = {str(row.get("uid")): row for row in _jsonl_rows(workspace / "stages" / "audio" / "audio_transform_receipts.jsonl")}
            audio = audio_rows.get(str(plan.get("uid")), {}).get("alignment", {})
            mfa = prepared.get("mfa", {}) if isinstance(prepared.get("mfa"), Mapping) else {}
            runs = []
            for run in plan.get("runs", []):
                language = str(run.get("language"))
                selected = [row for row in aliases if row.get("token_id") in set(run.get("unit_ids", []))]
                if not selected:
                    continue
                run_id = str(run["run_id"])
                corpus = workspace / "stages" / "align" / "prepared_corpus" / run_id
                corpus.mkdir(parents=True, exist_ok=True)
                wav = Path(str(audio.get("path", "")))
                local_wav = corpus / f"{run_id}.wav"
                if wav.is_file() and not local_wav.exists():
                    shutil.copyfile(wav, local_wav)
                (corpus / f"{run_id}.lab").write_text(" ".join(row["alias"] for row in selected) + "\n", encoding="utf-8")
                runs.append({
                    **{key: run[key] for key in ("run_id", "language", "unit_ids", "ownership_start_sample", "ownership_end_sample", "context_start_sample", "context_end_sample") if key in run},
                    "aliases": selected,
                    "corpus_dir": str(corpus),
                    "acoustic_model": mfa.get("japanese_acoustic" if language == "ja" else "english_acoustic"),
                    "dictionary": str(workspace / "stages" / "semantic" / ("locked.dict" if language == "ja" else "english_locked.dict")),
                    "native_inventory_path": mfa.get("japanese_metadata" if language == "ja" else "english_metadata"),
                    "output_dir": str(workspace / "stages" / "align" / "mfa_output" / run_id),
                    # MFA creates database/cache symlinks in its temporary
                    # tree. Keep that scratch directory outside the strict
                    # stage namespace: only copied corpus, raw TextGrid and
                    # ledgers are publishable artifacts.
                    "temporary_directory": tempfile.mkdtemp(prefix=f"ja-en-mfa-{run_id}-"),
                    "runtime_python": mfa.get("runtime_python"),
                    "sample_rate": int(audio.get("sample_rate", 16000)),
                })
            if runs:
                stage_inputs["align"] = {"uid": plan.get("uid"), "runs": runs}
    if stage == "merge" and "merge" not in prepared and "merge" not in stage_inputs:
        align_dir = workspace / "stages" / "align"
        plan_path = workspace / "stages" / "anchors" / "anchor_plan.json"
        if plan_path.is_file():
            plan = load_json(plan_path)
            expected: dict[str, str] = {}
            alias_rows = _jsonl_rows(workspace / "stages" / "semantic" / "alias_map.jsonl")
            # Cardinality is frozen by the semantic occurrence map.  Deriving
            # it from observed MFA phones would make a missing run disappear
            # from the expected set and could falsely validate a partial merge.
            for row in alias_rows:
                alias = row.get("alias")
                language = row.get("language")
                if isinstance(alias, str) and language in {"ja", "en"}:
                    expected[alias] = language
            audio_rows = {str(row.get("uid")): row for row in _jsonl_rows(workspace / "stages" / "audio" / "audio_transform_receipts.jsonl")}
            total = int(audio_rows.get(str(plan.get("uid")), {}).get("alignment", {}).get("frames", 0))
            if expected and total:
                ja_ledger = align_dir / "strict_ja_mfa.json"
                en_ledger = align_dir / "strict_en_mfa.json"
                stage_inputs["merge"] = {
                    "uid": plan.get("uid"), "japanese_ledger": str(ja_ledger) if ja_ledger.is_file() else [],
                    "english_ledger": str(en_ledger) if en_ledger.is_file() else [], "expected_languages": expected,
                    "ownership": [0, total], "sample_rate": int(audio_rows[str(plan.get("uid"))]["alignment"].get("sample_rate", 16000)),
                    "words": plan.get("units", []), "seams": plan.get("seams", []),
                }
    if stage == "tts" and "tts" not in prepared and "tts" not in stage_inputs:
        merged = workspace / "stages" / "merge" / "ja_en_alignment.json"
        if merged.is_file():
            payload = load_json(merged)
            audio_rows = {str(row.get("uid")): row for row in _jsonl_rows(workspace / "stages" / "audio" / "audio_transform_receipts.jsonl")}
            audio = audio_rows.get(str(payload.get("uid")), {})
            row = {**payload, "train_wav": audio.get("train", {}).get("path"), "alignment_wav": audio.get("alignment", {}).get("path"), "audio_receipt": audio}
            adapter = workspace / "stages" / "merge" / "alignment.jsonl"
            adapter.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            _append_unlisted_stage_outputs(workspace / "stages" / "merge", workspace)
            stage_inputs["tts"] = {"alignment_jsonl": str(adapter)}
    return prepared


def dispatch(config: Mapping[str, Any], config_path: Path, stages: list[str], *, resume: bool = False, autoload_handlers: bool = False, _lock_held: bool = False) -> list[StageResult]:
    if autoload_handlers:
        _autoload_stages()
    validated, manifest_path, manifest, workspace = preflight(config, config_path=config_path)
    if not _lock_held:
        # Identity, resume validation, stage execution and both root receipts
        # must observe one atomic run lock.  The inner stage-loop lock is a
        # same-process re-entry handled by exclusive_lock.
        with exclusive_lock(workspace / ".ja_en.lock"):
            return dispatch(validated, config_path, stages, resume=resume, autoload_handlers=False, _lock_held=True)
    workspace.mkdir(parents=True, exist_ok=True)
    identity = config_identity(validated, manifest_path, manifest)
    if resume:
        validate_resume(workspace, identity, allow_new=False)
    else:
        validate_resume(workspace, identity, allow_new=True)
    production_config_snapshot, production_lock_snapshot = _ensure_production_snapshots(workspace, validated, resume=resume)
    contract_inputs = [artifact_record(production_config_snapshot), artifact_record(production_lock_snapshot), artifact_record(manifest_path)]
    if not resume:
        identity_payload = {
            "schema": "ja-pipeline-receipt-v1",
            "identity_digest": stable_digest(identity),
            "identity": identity,
        }
        atomic_write_json(_identity_path(workspace), identity_payload, workspace=workspace)
    results: list[StageResult] = []
    cache_path = workspace / ".ja_en_stage_cache.json"
    stage_cache: dict[str, str] = {}
    if resume:
        if not cache_path.is_file() or cache_path.is_symlink():
            raise JAContractError("cache_miss", "stage cache is missing before resume", str(cache_path))
        cached = load_json(cache_path)
        if not isinstance(cached, Mapping) or not isinstance(cached.get("stages", {}), Mapping):
            raise JAContractError("cache_tampered", "stage cache is malformed", str(cache_path))
        stage_cache = {str(key): str(value) for key, value in cached["stages"].items()}
    with exclusive_lock(workspace / ".ja_en.lock"):
        for stage in stages:
            if stage not in STAGE_NAMES:
                raise JAContractError("config_malformed", f"unknown stage {stage!r}", "$.stage")
            handler, namespace = _STAGE_REGISTRY[stage]
            stage_dir = workspace / "stages" / namespace
            stage_dir.mkdir(parents=True, exist_ok=True)
            if stage == "verify":
                provisional = make_receipt(
                    stage="verify", status="PARTIAL",
                    inputs={"stages": stages, "artifacts": contract_inputs},
                    outputs=_workspace_output_paths(workspace),
                    params={"authoritative_root_receipt": "receipt.json", "pre_verification": True},
                    errors=[{"code": "publish_blocked", "message": "independent verification is pending"}],
                )
                atomic_write_json(workspace / "receipt.json", provisional, workspace=workspace)
                # The independent verifier checks the complete stage DAG. A
                # provisional verify receipt makes that DAG explicit without
                # making verification depend on a prior COMPLETE result.
                atomic_write_json(
                    stage_dir / "receipt.json",
                    make_receipt(
                        stage="verify", status="RUNNING", params={"authoritative_root_receipt": "receipt.json"},
                        errors=[{"code": "publish_blocked", "message": "verification is pending"}],
                    ), workspace=workspace,
                )
            prior = _stage_receipt(workspace, stage) if resume else None
            current_cache = _stage_cache_identity(stage, identity, validated, workspace)
            if prior and prior.get("status") == "COMPLETE" and stage_cache.get(stage) == current_cache:
                results.append(StageResult(stage=stage, status="COMPLETE", receipt_path=str(stage_dir / "receipt.json")))
                continue
            stage_config = _prepare_stage_config(validated, stage, workspace)
            uid_requests = stage_config.get("stage_inputs", {}).get("_uid_requests") if isinstance(stage_config.get("stage_inputs"), Mapping) else None
            if (isinstance(uid_requests, list) and stage in {"anchors", "align", "merge"}
                    and (len(uid_requests) > 1 or any(isinstance(row, Mapping) and row.get("status") in {"REJECTED", "BLOCKED", "UNRESOLVED"} for row in uid_requests))):
                stage_result = _run_uid_batch(stage, uid_requests, stage_config, workspace)
            elif stage == "tts" and isinstance(stage_config.get("stage_inputs"), Mapping) and stage_config["stage_inputs"].get("_tts_bridge_error"):
                # W3 could not produce an authoritative record for any UID.
                # Persist the reason as a stage receipt without invoking an
                # exporter that would otherwise manufacture fallback evidence.
                bridge_error = dict(stage_config["stage_inputs"]["_tts_bridge_error"])
                tts_errors = list(stage_config["stage_inputs"].get("_tts_uid_errors", []))
                if not tts_errors:
                    tts_errors = [bridge_error]
                receipt = make_receipt(stage="tts", status="BLOCKED", params={"implementation": "w3-authoritative-bridge"}, errors=tts_errors)
                atomic_write_json(stage_dir / "receipt.json", receipt, workspace=workspace)
                stage_result = StageResult("tts", "BLOCKED", str(stage_dir / "receipt.json"))
            else:
                stage_result = _normalize_stage_receipt(stage, stage_dir, _run_stage(handler, stage_config, stage_dir), workspace)
                if stage == "tts" and isinstance(stage_config.get("stage_inputs"), Mapping) and stage_config["stage_inputs"].get("_tts_uid_errors"):
                    receipt_path = stage_dir / "receipt.json"
                    receipt = dict(load_json(receipt_path)) if receipt_path.is_file() else make_receipt(stage="tts", status="PARTIAL")
                    receipt["status"] = "PARTIAL" if receipt.get("status") == "COMPLETE" else receipt.get("status", "PARTIAL")
                    receipt.setdefault("errors", []).extend(stage_config["stage_inputs"]["_tts_uid_errors"])
                    atomic_write_json(receipt_path, receipt, workspace=workspace)
                    stage_result = StageResult("tts", receipt["status"], str(receipt_path))
            _append_unlisted_stage_outputs(stage_dir, workspace)
            results.append(stage_result)
            stage_cache[stage] = current_cache
            # The frontend stage has an explicit candidate-analysis pass before
            # W1 selection and a locked-reading reconstruction pass afterwards.
            if stage == "reading" and "frontend" in stages and _STAGE_REGISTRY["frontend"][0].__name__ != "_skeleton_handler":
                frontend_dir = workspace / "stages" / _STAGE_REGISTRY["frontend"][1]
                results.append(_normalize_stage_receipt("frontend", frontend_dir, _run_locked_frontend_reconstruction(validated, workspace), workspace))
                stage_cache["frontend"] = _stage_cache_identity("frontend", identity, validated, workspace)
        atomic_write_json(cache_path, {"schema": "ja-stage-cache-v1", "stages": stage_cache}, workspace=workspace)
    # A stage can appear twice because reading triggers the locked frontend
    # reconstruction.  The final occurrence is the authoritative DAG state.
    stage_summary = {result.stage: result.status for result in results}
    statuses = list(stage_summary.values())
    required_statuses = [stage_summary.get(stage, "BLOCKED") for stage in PRODUCTION_STAGES]
    full_dag_invoked = set(stages).issuperset(PRODUCTION_STAGES)
    final_status = "COMPLETE" if full_dag_invoked and required_statuses and all(value == "COMPLETE" for value in required_statuses) else ("BLOCKED" if required_statuses and all(value == "BLOCKED" for value in required_statuses) else "PARTIAL")
    final_receipt = make_receipt(
        stage="verify",
        status=final_status,
        inputs={"stages": stages, "artifacts": contract_inputs},
        params={"autoload_handlers": autoload_handlers, "authoritative_root_receipt": "receipt.json", "independent_verifier_stage": "stages/verify/receipt.json"},
        errors=[] if final_status == "COMPLETE" else [{"code": "publish_blocked", "message": "one or more required stages are not COMPLETE", "stages": stage_summary}],
    )
    final_receipt["stage_results"] = stage_summary
    final_receipt["publishable"] = final_status == "COMPLETE"
    final_receipt["outputs"] = [artifact_record(path) for path in _workspace_output_paths(workspace)]
    atomic_write_json(workspace / "receipt.json", final_receipt, workspace=workspace)
    return results


def inspect_workspace(workspace: Path) -> dict[str, Any]:
    if workspace.is_symlink():
        raise JAContractError("manifest_symlink", "workspace is a symlink", str(workspace))
    receipts: dict[str, Any] = {}
    for stage in STAGE_NAMES:
        path = workspace / "stages" / _STAGE_REGISTRY.get(stage, (None, stage))[1] / "receipt.json"
        if path.exists():
            receipt = load_json(path)
            validate_receipt(receipt, workspace=workspace)
            receipts[stage] = receipt
    return {"workspace": str(workspace), "stages": receipts, "files": list_relative_files(workspace)}


def _skeleton_handler(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    stage = stage_dir.name
    receipt_path = stage_dir / "receipt.json"
    receipt = make_receipt(
        stage=stage if stage in STAGE_NAMES else "inventory",
        status="BLOCKED",
        params={"implementation": "foundation-skeleton", "synthetic": True},
        errors=[{"code": "publish_blocked", "message": "stage implementation is not registered"}],
    )
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage=stage, status="BLOCKED", receipt_path=str(receipt_path))


for _name in STAGE_NAMES:
    register_stage(_name, _skeleton_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Japanese/English ASR + MFA + TTS pipeline foundation")
    parser.add_argument("--config", required=False, help="YAML pipeline config")
    parser.add_argument("--check", action="store_true", help="validate config and manifest without writing")
    parser.add_argument("--inspect", metavar="WORKSPACE", help="inspect stage receipts without loading models")
    parser.add_argument("--stage", default=",".join(PRODUCTION_STAGES), help="comma-separated stages")
    parser.add_argument("--resume", action="store_true", help="resume an existing, identity-matching workspace")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            print(json.dumps(inspect_workspace(Path(args.inspect).expanduser().absolute()), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if not args.config:
            parser.error("--config is required unless --inspect is used")
        config_path = Path(args.config).expanduser().absolute()
        config = _load_yaml(config_path)
        _autoload_stages()
        validated, manifest_path, manifest, workspace = preflight(config, config_path=config_path)
        if args.check:
            print(json.dumps({"status": "OK", "config": str(config_path), "manifest": str(manifest_path), "items": len(manifest), "stages": stage_registry()}, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        stages = [part.strip() for part in args.stage.split(",") if part.strip()]
        results = dispatch(validated, config_path, stages, resume=args.resume, autoload_handlers=False)
        status = str(load_json(workspace / "receipt.json").get("status", "PARTIAL"))
        print(json.dumps({"status": status, "workspace": str(workspace), "results": [r.__dict__ for r in results]}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if status == "COMPLETE" else 2
    except JAContractError as exc:
        print(json.dumps({"status": "REJECTED", "error": exc.as_dict()}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
