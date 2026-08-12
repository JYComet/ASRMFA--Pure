#!/usr/bin/env python3
"""Prepare and execute the isolated, non-publishing GPU-1000 MFA run.

This deliberately is a small wrapper around ``run_pipeline.py``.  It is not a
second pipeline and it never makes a source tree writable, publishes output,
or retries a run.  The real ``run`` command is intentionally a separately
authorised operation; the other commands are useful with temporary fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from types import SimpleNamespace
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - reported by command
    yaml = None

PROJECT = Path(__file__).resolve().parent.parent
SPEAKERS = ("ria", "花礼", "雪狐桑")
QUOTAS = {"ria": 334, "花礼": 333, "雪狐桑": 333}
SEED = "gpu1000-v1"
SELECTOR = "gpu1000-balanced-hash-v1"
PRELIMINARY_SELECTION_DIGEST = "2114f03a1d2ff4b85660c06b76f657c8c1a3414709c9ccc3754fe2c223139335"
ENV_ALLOWLIST = ("PATH", "HOME", "CUDA_VISIBLE_DEVICES", "CONDA_PREFIX",
                 "LD_LIBRARY_PATH", "PYTHONPATH", "LANG", "LC_ALL", "NUMBA_CACHE_DIR")
TELEMETRY_INTERVAL_SECONDS = 2
CONTINUATION_SCHEMA = "gpu1000-continuation-v1"
CONTINUATION_SCOPE_SCHEMA = "gpu1000-continuation-scope-v1"
CONTINUATION_PREFLIGHT_V2_SCHEMA = "gpu1000-continuation-preflight-v2"

try:
    from scripts.pipeline_utils import make_mfa_alignment_axis_receipt, make_mfa_input_axis_receipt, validate_strict_mfa_textgrid, make_pipeline_accounting_receipt, validate_pipeline_accounting_receipt
except ImportError:  # direct script execution
    from pipeline_utils import make_mfa_alignment_axis_receipt, make_mfa_input_axis_receipt, validate_strict_mfa_textgrid, make_pipeline_accounting_receipt, validate_pipeline_accounting_receipt


class SafetyError(RuntimeError):
    pass


def _continuation_scope(path: Path) -> dict[str, Any]:
    """Load the one-stem continuation scope without accepting implicit scope."""
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"continuation scope unreadable: {exc}")
    if isinstance(raw, list):
        raw = {"stems": raw}
    if isinstance(raw, dict) and isinstance(raw.get("scope"), dict):
        nested = dict(raw["scope"]); nested.setdefault("schema", raw.get("schema", CONTINUATION_SCOPE_SCHEMA)); raw = nested
    if isinstance(raw, dict) and isinstance(raw.get("retry"), dict):
        nested = dict(raw["retry"]); nested.setdefault("schema", raw.get("schema", CONTINUATION_SCOPE_SCHEMA)); raw = nested
    if not isinstance(raw, dict) or raw.get("schema", CONTINUATION_SCOPE_SCHEMA) != CONTINUATION_SCOPE_SCHEMA:
        raise SafetyError("continuation scope schema invalid")
    stems = raw.get("stems")
    if not isinstance(stems, list) or len(stems) != 1 or not isinstance(stems[0], str) or not stems[0]:
        raise SafetyError("continuation scope must contain exactly one stem")
    return raw


def _continuation_grid_evidence(root: Path, expected_count: int = 999) -> tuple[list[Path], str]:
    workspace = root / "workspace"
    # Retained MFA output is authoritative.  The ctc_pretg fallback exists
    # only for old fixtures and is never mixed with MFA shard output.
    shard_grids = sorted(workspace.glob("mfa_shards/*/shard_*/output/*.TextGrid"), key=lambda p: p.name)
    retry_grids = sorted(workspace.glob("mfa_shards/*/retry_missing/output/*.TextGrid"), key=lambda p: p.name)
    grids = shard_grids + retry_grids
    if grids and (len(shard_grids) != 990 or len(retry_grids) != 9):
        raise SafetyError("MFA retained grid origin must be exactly 990 shard + 9 retry_missing")
    if not grids:
        grids = sorted((workspace / "ctc_pretg").glob("*.TextGrid"), key=lambda p: p.name)
    if len(grids) != expected_count or len({p.stem for p in grids}) != expected_count:
        raise SafetyError(f"continuation requires exactly {expected_count} original TextGrids")
    if any(p.is_symlink() or not p.is_file() for p in grids):
        raise SafetyError("continuation original TextGrid namespace is not ordinary files")
    for grid in grids:
        if grid.read_text(encoding="utf-8", errors="replace").lstrip().startswith("File type"):
            if validate_strict_mfa_textgrid(grid):
                raise SafetyError(f"invalid retained MFA TextGrid: {grid.name}")
    entries = [{"stem": p.stem, "relative": str(p.relative_to(root)), "sha256": sha_file(p)} for p in grids]
    return grids, digest(entries)


def _continuation_namespace_errors(root: Path) -> list[str]:
    forbidden = ("ctc_pretg_adj", "mfa", "mfa_input", "mfa_output", "aligned", "align_en",
                 "postprocess", "strict_ok_runs", "output", "filtered")
    errors = []
    for name in forbidden:
        paths = (root / "workspace" / name,)
        for path in paths:
            if not path.exists():
                continue
            if name == "aligned" and path.is_dir() and not any(path.iterdir()):
                continue
            if name == "ctc_pretg_adj" and path.is_dir() and not any(path.iterdir()):
                continue
            if name == "strict_ok_runs" and path.is_dir():
                receipts = list(path.glob("*/output/.pipeline_run_receipt_v2.json"))
                filtered = list(path.glob("*/filtered/*"))
                historical = [p for p in receipts if not p.parent.parent.name.startswith("continuation_")]
                failed = [p for p in receipts if p.parent.parent.name.startswith("continuation_")]
                if ((len(receipts) == 1 and len(historical) == 1) or
                        (len(receipts) == 2 and len(historical) == 1 and len(failed) == 1)) and not filtered:
                    continue
            errors.append(f"downstream_namespace_present:{name}")
    for name in ("aligned", "en_phones"):
        path = root / "workspace" / name
        if path.exists() and (path.is_symlink() or not path.is_dir() or any(path.iterdir())):
            errors.append(f"continuation_namespace_not_empty_canonical:{name}")
    # Process markers imply a continuation was already attempted.  Never
    # resume in place or infer that a partial process is reusable.
    for marker in ("continuation_started.json", "continuation_receipt.json", "continuation_gate.json"):
        if (root / marker).exists(): errors.append(f"continuation_marker_present:{marker}")
    return errors


def _empty_namespace_binding(root: Path, name: str) -> dict[str, Any]:
    path = root / "workspace" / name
    if not path.exists(): return {"state": "absent", "name": name}
    if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise SafetyError(f"continuation namespace must be empty ordinary directory: {name}")
    stat = path.stat()
    return {"state": "empty", "name": name, "path": str(path.resolve()), "dev": stat.st_dev,
            "ino": stat.st_ino, "mode": stat.st_mode, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "empty_tree_digest": sha_tree(path)}


def _discover_continuation_scope(root: Path, scope_path: Path | None) -> tuple[Path, dict[str, Any]]:
    if scope_path is None:
        candidates = sorted((root / "workspace" / "mfa_logs").glob("*/mfa_missing_retry_plan.json"))
        candidates += [root / "retry_plan.json", root / "mfa_retry_plan.json",
                       root / "continuation_scope.json", root / "workspace" / "retry_plan.json"]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise SafetyError("immutable retry plan must resolve to exactly one scope")
        scope_path = matches[0]
    payload = read_json(scope_path)
    if scope_path.name == "mfa_missing_retry_plan.json" and isinstance(payload, dict):
        coordinator = payload.get("coordinator", {}) if isinstance(payload, dict) else {}
        history = coordinator.get("history", []) if isinstance(coordinator, dict) else []
        if not history: history = payload.get("history", payload.get("attempts", []))
        latest = history[-1] if isinstance(history, list) and history else payload
        missing = latest.get("missing", []) if isinstance(latest, dict) else []
        missing = [item.get("stem") if isinstance(item, dict) else item for item in missing]
        payload = {"schema": CONTINUATION_SCOPE_SCHEMA, "stems": missing,
                   "retry_plan": str(scope_path.resolve()), **({"lab": latest.get("lab"), "wav": latest.get("wav")} if isinstance(latest, dict) else {})}
        # Keep the immutable retry-plan path as the provenance binding.  The
        # normalized one-stem scope is intentionally in-memory only: discovery
        # is a read-only operation and must not create a /tmp artifact.
        return scope_path.resolve(), _continuation_scope_payload(payload)
    return scope_path.resolve(), _continuation_scope(scope_path)


def _continuation_scope_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-normalized scope without writing it to disk."""
    if not isinstance(payload, dict) or payload.get("schema", CONTINUATION_SCOPE_SCHEMA) != CONTINUATION_SCOPE_SCHEMA:
        raise SafetyError("continuation scope schema invalid")
    stems = payload.get("stems")
    if not isinstance(stems, list) or len(stems) != 1 or not isinstance(stems[0], str) or not stems[0]:
        raise SafetyError("continuation scope must contain exactly one stem")
    return payload


def _resolve_current_retry_inputs(root: Path, stem: str) -> tuple[Path, Path]:
    """Resolve exactly one ordinary LAB/WAV pair from retained retry input."""
    retry_roots = sorted((root / "workspace" / "mfa_shards").glob("*/retry_missing"))
    retry_roots += [root / "workspace" / "retry_missing", root / "workspace" / "retained_retry_missing"]
    labs: list[Path] = []
    wavs: list[Path] = []
    for retry_root in retry_roots:
        if not retry_root.is_dir() or retry_root.is_symlink():
            continue
        labs.extend(retry_root.rglob(f"{stem}.lab"))
        wavs.extend(retry_root.rglob(f"{stem}.wav"))
    labs = [p for p in labs if p.is_file() and not p.is_symlink()]
    wavs = [p for p in wavs if p.is_file() and not p.is_symlink()]
    if len(labs) != 1 or len(wavs) != 1:
        raise SafetyError("continuation requires exactly one retained retry LAB/WAV pair")
    lab, wav = labs[0].resolve(), wavs[0].resolve()
    if lab.stem != stem or lab.suffix != ".lab" or wav.stem != stem or wav.suffix != ".wav":
        raise SafetyError("retained retry input stem/extension mismatch")
    return lab, wav


def _bind_retry_inputs(scope: dict[str, Any], root: Path, proof: dict[str, Any]) -> dict[str, Any]:
    stem = scope["stems"][0]
    lab, wav = _resolve_current_retry_inputs(root, stem)
    by_role = {
        str(item.get("role")): item for item in proof.get("inputs", [])
        if isinstance(item, dict) and item.get("stem") == stem
    }
    for role, path in (("anchor.lab", lab), ("mfa_axis_audio", wav)):
        record = by_role.get(role)
        if not isinstance(record, dict) or not record.get("sha256"):
            raise SafetyError(f"proven retry input role missing: {role}")
        if sha_file(path) != record["sha256"]:
            raise SafetyError(f"current retry input hash mismatch: {role}")
    bound = dict(scope)
    bound["lab"] = str(lab)
    bound["wav"] = str(wav)
    bound["lab_sha256"] = sha_file(lab)
    bound["wav_sha256"] = sha_file(wav)
    return bound


def continuation_preflight(root: Path, scope_path: Path | None = None, *, expected_grids: int = 999,
                           python: str = sys.executable, proven_mfa_retry_receipt: Path | None = None) -> dict[str, Any]:
    """Validate immutable failed-run evidence before a one-stem continuation."""
    root = root.resolve(); errors: list[str] = []
    if proven_mfa_retry_receipt is None or not proven_mfa_retry_receipt.is_file():
        raise SafetyError("continuation requires a proven MFA retry receipt")
    try:
        manifest, evidence, _ = state(root)
        run = read_json(root / "run_receipt.json")
    except (OSError, json.JSONDecodeError, SafetyError) as exc:
        raise SafetyError(f"continuation preflight evidence missing: {exc}")
    if run.get("returncode") != 1:
        errors.append("original_run_not_rc1")
    original_receipt_sha256 = sha_file(root / "run_receipt.json")
    namespace_bindings: dict[str, Any] = {}
    for namespace in ("aligned", "en_phones"):
        try: namespace_bindings[namespace] = _empty_namespace_binding(root, namespace)
        except SafetyError as exc: errors.append(str(exc))
    retained_strict = sorted((root / "workspace" / "strict_ok_runs").glob("*/output/.pipeline_run_receipt_v2.json"))
    retained_strict_binding = {"path": str(retained_strict[0].resolve()), "sha256": sha_file(retained_strict[0])} if len(retained_strict) == 1 else None
    try:
        scope_path, scope = _discover_continuation_scope(root, scope_path)
        scope_stem = scope["stems"][0]
    except SafetyError as exc:
        errors.append(str(exc)); scope, scope_stem = {}, ""
    try:
        grids, grid_digest = _continuation_grid_evidence(root, expected_grids)
        grid_stems = {p.stem for p in grids}
    except SafetyError as exc:
        errors.append(str(exc)); grids, grid_digest, grid_stems = [], "", set()
    selected = {row.get("stem") for row in manifest.get("samples", []) if isinstance(row, dict)}
    if len(selected) != 1000: errors.append("original_selection_not_exact1000")
    if scope_stem and (scope_stem in grid_stems or scope_stem not in selected):
        errors.append("continuation_scope_not_single_missing_stem")
    errors.extend(_continuation_namespace_errors(root))
    current_code_hashes = {key: sha_file(PROJECT / name) for name, key in
                           (("scripts/run_pipeline.py", "run_pipeline_sha256"),
                            ("scripts/ctc_prealign.py", "ctc_prealign_sha256"),
                            ("scripts/gpu1000_orchestrate.py", "orchestrator_sha256"))}
    baseline_keys = ("run_pipeline_sha256", "ctc_prealign_sha256")
    if not all(isinstance(evidence.get(key), str) and evidence.get(key) for key in baseline_keys):
        errors.append("prepared_code_hashes_missing")
    proven_root = proven_mfa_retry_receipt.parent if proven_mfa_retry_receipt else Path(".")
    proven_files = [proven_mfa_retry_receipt] if proven_mfa_retry_receipt else []
    proven_hashes = {str(path): sha_file(path) for path in proven_files if path.is_file()}
    artifacts = {}
    proof_payload: dict[str, Any] = {}
    if proven_mfa_retry_receipt and proven_mfa_retry_receipt.is_file():
        try:
            proof = read_json(proven_mfa_retry_receipt); proof_payload = proof if isinstance(proof, dict) else {}; cmd = proof.get("command", [])
            if scope:
                try:
                    scope = _bind_retry_inputs(scope, root, proof)
                except SafetyError as exc:
                    errors.append(str(exc))
            for label, record in (("mfa_executable", proof.get("mfa_executable", {})), ("mfa_dependency", proof.get("mfa_dependency", {})), ("dictionary", proof.get("dictionary", {})), ("model", proof.get("model", {}))):
                path = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path("")
                if not path.is_file(): errors.append(f"proven_mfa_artifact_missing:{label}")
                else:
                    actual = sha_file(path); artifacts[label] = {"path": str(path), "sha256": actual}
                    if record.get("sha256") and actual != record.get("sha256"): errors.append(f"proven_mfa_artifact_hash:{label}")
            proof_inputs = proof.get("inputs", [])
            for item in proof_inputs:
                if isinstance(item, dict) and item.get("stem") == scope_stem:
                    src = Path(str(item.get("source", "")))
                    if src.is_file() and item.get("sha256") != sha_file(src): errors.append(f"scope_input_hash_mismatch:{item.get('role')}")
                    # Current-root retry paths may differ from the original
                    # proof paths; hash equality is the binding authority.
            if not isinstance(cmd, list) or len(cmd) < 5: errors.append("proven_mfa_command_missing")
            else:
                if cmd[0] != proof.get("mfa_executable", {}).get("path"): errors.append("mfa_executable_path_mismatch")
                if cmd[3] != proof.get("dictionary", {}).get("path") or cmd[4] != proof.get("model", {}).get("path"): errors.append("mfa_model_dictionary_path_mismatch")
                for value in (cmd[0], cmd[3], cmd[4]):
                    path = Path(str(value))
                    if not path.exists(): errors.append(f"proven_mfa_artifact_missing:{value}")
                    elif path == Path(str(proof.get("mfa_executable", {}).get("path", ""))) and path.is_file() and proof.get("mfa_executable", {}).get("sha256") and sha_file(path) != proof.get("mfa_executable", {}).get("sha256"):
                        errors.append(f"proven_mfa_artifact_hash:{value}")
            for key in ("MFA_ROOT_DIR", "NUMBA_CACHE_DIR"):
                value = proof.get("environment", {}).get(key)
                if not value: errors.append(f"proven_mfa_env_missing:{key}")
                elif not Path(value).is_dir() or not os.access(value, os.W_OK): errors.append(f"proven_mfa_env_unwritable:{key}")
        except (OSError, json.JSONDecodeError): errors.append("proven_mfa_receipt_invalid")
    receipt = {"schema": "gpu1000-continuation-preflight-v1", "root": str(root),
               "scope_path": str(scope_path), "scope": scope, "expected_original_grids": expected_grids,
               "retry_plan_path": scope.get("retry_plan", str(scope_path)),
               "retry_plan_sha256": sha_file(Path(scope["retry_plan"])) if isinstance(scope.get("retry_plan"), str) and Path(scope["retry_plan"]).is_file() else "",
               "original_grid_count": len(grids), "original_grid_digest": grid_digest,
               "original_run_receipt_sha256": original_receipt_sha256,
               "selected_digest": manifest.get("selection_digest"), "errors": sorted(set(errors)),
               "prepared_code_hashes": {key: evidence.get(key) for key in current_code_hashes},
               "current_code_hashes": current_code_hashes,
               "code_hash_allowlist": "prepared_baseline_and_current_recorded",
               "proven_mfa_evidence": proven_hashes,
               "proven_mfa_retry_receipt": str(proven_mfa_retry_receipt.resolve()),
               "proven_mfa_artifacts": artifacts,
               "namespace_bindings": namespace_bindings,
               "retained_strict_binding": retained_strict_binding,
               "mfa_interpreter": str(Path(python).resolve()),
               "ok": not errors, "continuation_status": "READY" if not errors else "BLOCKED"}
    return receipt


def _continuation_v2_blockers(root: Path) -> list[str]:
    """Return mutation markers/namespaces that make a fresh v2 impossible."""
    blockers: list[str] = []
    for name in ("continuation_started.json", "continuation_result.json",
                 "alignment_merge_receipt.json"):
        if (root / name).exists():
            blockers.append(f"continuation_marker_present:{name}")
    workspace = root / "workspace"
    for name in ("continuation_singleton_corpus", "continuation_singleton_audio",
                 "continuation_singleton_mfa", "continuation_singleton_temp",
                 "continuation_singleton_mfa_root", "continuation_singleton_numba_cache",
                 "continuation_singleton_mfa_rescue", "continuation_singleton_temp_rescue",
                 "continuation_temp", "continuation_singleton_attempts.json",
                 ".mfa_alignment_axis_receipt.json", ".continuation_source_receipt.json",
                 ".mfa_input_axis_receipt.json"):
        if (workspace / name).exists():
            blockers.append(f"continuation_namespace_present:{name}")
    return blockers


def continuation_preflight_v2(root: Path, scope_path: Path | None = None, *, expected_grids: int = 999,
                              python: str = sys.executable,
                              proven_mfa_retry_receipt: Path | None = None) -> dict[str, Any]:
    """Create one immutable fresh preflight bound to the existing v1 receipt."""
    root = root.resolve()
    v1_path = root / "continuation_preflight_receipt.json"
    if not v1_path.is_file():
        raise SafetyError("continuation-preflight-v2 requires existing v1 receipt")
    blockers = _continuation_v2_blockers(root)
    if blockers:
        raise SafetyError("v2 preflight blocked: " + "; ".join(blockers))
    try:
        v1 = read_json(v1_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"v1 preflight unreadable: {exc}")
    if not isinstance(v1, dict) or v1.get("schema") != "gpu1000-continuation-preflight-v1":
        raise SafetyError("existing preflight is not v1")
    if proven_mfa_retry_receipt is None:
        candidate = v1.get("proven_mfa_retry_receipt")
        if isinstance(candidate, str) and candidate:
            proven_mfa_retry_receipt = Path(candidate)
    fresh = continuation_preflight(root, scope_path, expected_grids=expected_grids,
                                   python=python, proven_mfa_retry_receipt=proven_mfa_retry_receipt)
    if not fresh.get("ok"):
        raise SafetyError("fresh continuation preflight failed: " + "; ".join(fresh.get("errors", [])))
    receipt = {
        "schema": CONTINUATION_PREFLIGHT_V2_SCHEMA,
        "root": str(root),
        "reason": "code_changed_before_continuation_started",
        "v1_binding": {"path": str(v1_path.resolve()), "sha256": sha_file(v1_path)},
        "fresh_preflight": fresh,
        "fresh_preflight_digest": digest(fresh),
        "continuation_status": "READY",
        "ok": True,
    }
    write_once(root / "continuation_preflight_receipt_v2.json", receipt)
    return receipt


def continuation_batch_policy(*, initial_pass: bool, hard_failures: int = 0,
                              singleton: bool = True) -> dict[str, Any]:
    """Return the narrow singleton-then-expand policy; never infer expansion."""
    if not singleton:
        raise SafetyError("continuation must start as singleton")
    policy = {"singleton": {"count": 1},
              "initial": {"count": 100, "split": [20, 80]},
              "expanded": {"count": 1000, "split": [200, 800]},
              "expanded_allowed": bool(initial_pass and hard_failures == 0),
              "condition": "initial_pass_and_zero_hard_failures"}
    return policy


def _transform_retry_command(command: list[str], *, corpus: Path, audio: Path,
                             output: Path, temp: Path, proof: dict[str, Any]) -> list[str]:
    """Transform only proven path tokens in the retry command."""
    if not isinstance(command, list) or len(command) < 6:
        raise SafetyError("proven retry command missing positional MFA paths")
    result = [str(token) for token in command]
    replacements: dict[str, str] = {}
    old_corpus = proof.get("corpus") or proof.get("input_corpus")
    old_audio = proof.get("audio") or proof.get("audio_directory")
    old_output = proof.get("output") or proof.get("output_directory")
    old_temp = proof.get("temporary_directory") or proof.get("temp")
    if not old_corpus and len(command) > 2:
        old_corpus = command[2]
    if not old_output and len(command) > 5:
        old_output = command[5]
    for flag, target in (("--audio_directory", "audio"),
                         ("--temporary_directory", "temp")):
        if flag in result:
            value = result[result.index(flag) + 1]
            if target == "audio" and not old_audio:
                old_audio = value
            if target == "temp" and not old_temp:
                old_temp = value
    for index, old in ((2, old_corpus), (5, old_output)):
        if old:
            replacements[str(old)] = str((corpus if index == 2 else output).resolve())
            replacements[str(Path(str(old)).resolve())] = str((corpus if index == 2 else output).resolve())
    # The positional corpus is also the historical audio directory in old
    # proofs; explicit --audio_directory wins when present.
    if old_audio:
        replacements[str(old_audio)] = str(audio.resolve())
        replacements[str(Path(str(old_audio)).resolve())] = str(audio.resolve())
    if old_temp:
        replacements[str(old_temp)] = str(temp.resolve())
        replacements[str(Path(str(old_temp)).resolve())] = str(temp.resolve())
    result[2] = str(corpus.resolve())
    result[5] = str(output.resolve())
    for index, token in enumerate(result):
        result[index] = replacements.get(str(Path(token).resolve()) if token.startswith("/") else token, token)
    # Explicitly bind the positional replacements after token normalization.
    result[2], result[5] = str(corpus.resolve()), str(output.resolve())
    return result


def _revalidate_singleton_evidence(scope: dict[str, Any], proof: dict[str, Any]) -> None:
    """Recheck immutable proof and current input hashes immediately pre-run."""
    for key in ("mfa_executable", "mfa_dependency", "dictionary", "model"):
        record = proof.get(key, {})
        if key == "mfa_dependency" and not isinstance(record, dict):
            continue
        if key == "mfa_dependency" and not record.get("path"):
            continue
        path = Path(str(record.get("path", "")))
        if not path.exists():
            raise SafetyError(f"singleton proof artifact missing: {key}")
        actual = sha_tree(path) if path.is_dir() else sha_file(path)
        expected = record.get("sha256")
        if expected and actual != expected:
            raise SafetyError(f"singleton proof artifact hash changed: {key}")
    exe_parent = Path(str(proof["mfa_executable"]["path"])).resolve().parent
    proof_env = proof.get("environment", {})
    if not isinstance(proof_env, dict):
        raise SafetyError("singleton proof environment invalid")
    path_prefix = proof_env.get("PATH_prefix", str(exe_parent))
    if Path(str(path_prefix)).resolve() != exe_parent:
        raise SafetyError("singleton proof PATH_prefix is not MFA executable parent")
    stem = scope["stems"][0]
    by_role = {str(item.get("role")): item for item in proof.get("inputs", [])
               if isinstance(item, dict) and item.get("stem") == stem}
    for role, key in (("anchor.lab", "lab"), ("mfa_axis_audio", "wav")):
        path = Path(str(scope.get(key, "")))
        record = by_role.get(role, {})
        if not path.is_file() or path.is_symlink() or sha_file(path) != record.get("sha256"):
            raise SafetyError(f"singleton retry input changed: {role}")


def _execute_singleton_mfa(root: Path, scope: dict[str, Any], python: str = sys.executable, proof: dict[str, Any] | None = None) -> Path:
    """Run only the one-stem MFA rescue from retained LAB/16k WAV evidence."""
    stem = scope["stems"][0]
    # Always resolve the current-root retained retry pair.  Scope/proof paths
    # are provenance only; an external or stale path must never become the
    # execution input merely because it remains readable.
    lab, wav = _resolve_current_retry_inputs(root, stem)
    if isinstance(proof, dict):
        bound_scope = dict(scope); bound_scope["lab"] = str(lab); bound_scope["wav"] = str(wav)
        _revalidate_singleton_evidence(bound_scope, proof)
    workspace = root / "workspace"
    corpus = workspace / "continuation_singleton_corpus"
    audio = workspace / "continuation_singleton_audio"
    output = workspace / "continuation_singleton_mfa"
    temp_dir = workspace / "continuation_singleton_temp"
    mfa_root = workspace / "continuation_singleton_mfa_root"
    numba = workspace / "continuation_singleton_numba_cache"
    if any(path.exists() for path in (corpus, audio, output, temp_dir, mfa_root, numba)):
        raise SafetyError("singleton MFA namespace already exists")
    for path in (corpus, audio, output, temp_dir, mfa_root, numba):
        path.mkdir()
    shutil.copy2(lab, corpus / lab.name); shutil.copy2(wav, audio / wav.name)
    cfg = load_yaml(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    try:
        import importlib
        cfg = importlib.import_module("scripts.run_pipeline").load_config(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    except Exception:
        pass
    mfa_cfg = cfg.get("mfa", {}) if isinstance(cfg.get("mfa"), dict) else {}
    en_cfg = cfg.get("mfa_en", {}) if isinstance(cfg.get("mfa_en"), dict) else {}
    dictionary = Path(str(mfa_cfg.get("mfa_dict", cfg.get("mfa_dict", "dict/mfa_ipa.dict"))))
    model_name = str(mfa_cfg.get("acoustic_model", cfg.get("acoustic_model", "mandarin_mfa")))
    model = Path(model_name)
    if not dictionary.is_absolute(): dictionary = (PROJECT / dictionary).resolve()
    if not model.is_absolute():
        extracted = Path(str(cfg.get("models_dir", PROJECT / "models/mfa"))).resolve() / "extracted_models" / "acoustic" / f"{model_name}_acoustic"
        model = extracted if extracted.is_dir() else Path(model_name)
    mfa_bin, _ = resolve_mfa_dependency(python)
    proof_env = proof.get("environment", {}) if isinstance(proof, dict) else {}
    if not isinstance(proof_env, dict):
        raise SafetyError("singleton proof environment invalid")
    if isinstance(proof, dict) and isinstance(proof.get("mfa_executable"), dict):
        mfa_bin = str(proof["mfa_executable"].get("path", mfa_bin))
        if proof.get("dictionary", {}).get("path"): dictionary = Path(proof["dictionary"]["path"])
        if proof.get("model", {}).get("path"): model = Path(proof["model"]["path"])
    if not mfa_bin or not dictionary.is_file() or (model.is_absolute() and not model.exists()):
        raise SafetyError("singleton MFA dependencies unavailable")
    command = proof.get("command") if isinstance(proof, dict) else None
    if not isinstance(command, list):
        raise SafetyError("singleton MFA requires proven retry command")
    base_cmd = _transform_retry_command(command, corpus=corpus, audio=audio,
                                        output=output, temp=temp_dir, proof=proof)
    attempts = []
    env = safe_env()
    exe_parent = Path(mfa_bin).resolve().parent
    proof_env = proof.get("environment", {}) if isinstance(proof, dict) else {}
    path_prefix = Path(str(proof_env.get("PATH_prefix", exe_parent))).resolve()
    if path_prefix != exe_parent:
        raise SafetyError("singleton proof PATH_prefix is not MFA executable parent")
    safe_path = env.get("PATH", os.environ.get("PATH", ""))
    env["PATH"] = str(path_prefix) + (os.pathsep + safe_path if safe_path else "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(proof_env.get(key, "1"))
    env["NUMBA_CACHE_DIR"] = str(numba.resolve()); env["MFA_ROOT_DIR"] = str(mfa_root.resolve())
    completed = subprocess.run(base_cmd, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    attempts.append({"attempt": "20_80", "argv": base_cmd, "returncode": completed.returncode,
                     "stderr": (completed.stderr or "")[-1000:], "stdout": (completed.stdout or "")[-1000:]})
    grid = output / stem / f"{stem}.TextGrid"
    if not grid.is_file():
        candidates = list(output.rglob(f"{stem}.TextGrid")); grid = candidates[0] if len(candidates) == 1 else grid
    if completed.returncode != 0 and "NoAlignmentsError" in (completed.stderr or ""):
        rescue_output = root / "workspace" / "continuation_singleton_mfa_rescue"
        rescue_temp = root / "workspace" / "continuation_singleton_temp_rescue"
        if rescue_output.exists() or rescue_temp.exists():
            raise SafetyError("singleton MFA rescue namespace already exists")
        rescue_output.mkdir(); rescue_temp.mkdir()
        rescue_cmd = list(base_cmd); rescue_cmd[5] = str(rescue_output)
        if "--beam" in rescue_cmd:
            beam_index = rescue_cmd.index("--beam") + 1
            if rescue_cmd[beam_index] == "20":
                rescue_cmd[beam_index] = "200"
        if "--retry_beam" in rescue_cmd:
            retry_beam_index = rescue_cmd.index("--retry_beam") + 1
            if rescue_cmd[retry_beam_index] == "80":
                rescue_cmd[retry_beam_index] = "800"
        if "--temporary_directory" in rescue_cmd:
            rescue_cmd[rescue_cmd.index("--temporary_directory") + 1] = str(rescue_temp)
        completed = subprocess.run(rescue_cmd, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        attempts.append({"attempt": "200_800", "argv": rescue_cmd, "returncode": completed.returncode,
                         "stderr": (completed.stderr or "")[-1000:], "stdout": (completed.stdout or "")[-1000:]})
        grid = rescue_output / stem / f"{stem}.TextGrid"
        if not grid.is_file():
            candidates = list(rescue_output.rglob(f"{stem}.TextGrid")); grid = candidates[0] if len(candidates) == 1 else grid
    receipt_path = root / "workspace" / "continuation_singleton_attempts.json"
    write_once(receipt_path, {"schema": "gpu1000-singleton-attempts-v1", "stem": stem, "attempts": attempts,
                              "interpreter": str(Path(python).resolve()), "dictionary": str(dictionary),
                              "dictionary_sha256": sha_file(dictionary), "model": str(model),
                              "model_sha256": sha_tree(model) if model.is_dir() else (sha_file(model) if model.is_file() else "registry"),
                              "numba_cache_dir": env["NUMBA_CACHE_DIR"]})
    if completed.returncode != 0: raise SafetyError(f"singleton MFA failed rc={completed.returncode}")
    if not grid.is_file() or grid.is_symlink(): raise SafetyError("singleton MFA produced no bound TextGrid")
    if validate_strict_mfa_textgrid(grid): raise SafetyError("singleton MFA produced invalid TextGrid")
    return grid


def _merged_continuation_root(root: Path, continuation_root: Path | None, preflight: dict[str, Any],
                              scope: dict[str, Any]) -> dict[str, Any]:
    """Atomically install exact-1000 MFA evidence into the same run root."""
    root = root.resolve(); continuation_root = root if continuation_root is None else continuation_root.resolve()
    if continuation_root != root:
        raise SafetyError("continuation must operate in the original root")
    missing = scope["stems"][0]
    grid_path = scope.get("textgrid") or scope.get("grid")
    if not isinstance(grid_path, str): raise SafetyError("scope must bind the continuation TextGrid path")
    grid_path = Path(grid_path).resolve()
    if not grid_path.is_file() or grid_path.stem != missing or grid_path.is_symlink():
        raise SafetyError("continuation TextGrid path is missing, aliased, or stem-mismatched")
    manifest = read_json(root / "selected_manifest.json")
    samples = manifest.get("samples", [])
    workspace = root / "workspace"
    temp = Path(tempfile.mkdtemp(prefix="aligned.partial-", dir=str(workspace)))
    try:
        grid_dir = temp
        old_grids, _ = _continuation_grid_evidence(root)
        for source_grid in old_grids + [grid_path]:
            destination = grid_dir / source_grid.name
            if destination.exists(): raise SafetyError(f"continuation grid collision: {source_grid.name}")
            shutil.copy2(source_grid, destination)
        merged = sorted(grid_dir.glob("*.TextGrid"), key=lambda p: p.name)
        selected = {row.get("stem") for row in samples if isinstance(row, dict)}
        if len(merged) != 1000 or {p.stem for p in merged} != selected:
            raise SafetyError("continuation merged TextGrid set is not exact1000")
        # Re-read copied inputs and bind every source hash to the frozen
        # selection manifest; no source tree is consulted during continuation.
        for row in samples:
            for suffix, key in ((".wav", "wav_sha256"), (".txt", "txt_sha256")):
                path = root / "input" / f"{row['stem']}{suffix}"
                if not path.is_file() or sha_file(path) != row.get(key):
                    raise SafetyError(f"continuation source hash mismatch: {row.get('stem')}{suffix}")
        source_hashes = [{"stem": row["stem"], "wav_sha256": row["wav_sha256"], "txt_sha256": row["txt_sha256"]} for row in samples]
        source_receipt = {"schema": "gpu1000-continuation-source-v1", "count": len(source_hashes),
                          "stems": sorted(selected), "stems_digest": digest(sorted(selected)),
                          "source_hashes": source_hashes, "source_hashes_digest": digest(source_hashes),
                          "original_run_receipt_sha256": preflight["original_run_receipt_sha256"]}
        write_once(root / "workspace" / ".continuation_source_receipt.json", source_receipt)
        axis = {"schema": "gpu1000-continuation-axis-v1", "source_role": "mfa_axis_audio",
                "axis_root": str((root / "workspace" / "aligned").resolve()), "stems": sorted(selected),
                "stems_digest": digest(sorted(selected)), "source_hashes_digest": digest(source_hashes),
                "lineage_original_run_receipt_sha256": preflight["original_run_receipt_sha256"]}
        # Keep both input and alignment axes explicit for downstream consumers.
        input_path = root / "workspace" / "ctc_pretg" / ".mfa_input_axis_receipt.json"
        if input_path.is_file():
            input_axis = read_json(input_path)
            if sorted(input_axis.get("stems", [])) != sorted(selected):
                raise SafetyError("retained MFA input-axis receipt does not bind exact1000")
        else:
            audio_rows = []
            for row in samples:
                wav_path = root / "input" / f"{row['stem']}.wav"
                try:
                    with wave.open(str(wav_path), "rb") as handle:
                        duration = handle.getnframes() / handle.getframerate()
                        metadata = {"duration_s": duration, "sample_rate": handle.getframerate(),
                                    "frames": handle.getnframes(), "channels": handle.getnchannels(),
                                    "sample_width": handle.getsampwidth()}
                    audio_rows.append({"stem": row["stem"], "path": str(wav_path.resolve()),
                                       "sha256": row["wav_sha256"], **metadata})
                except (OSError, EOFError, wave.Error):
                    audio_rows.append({"stem": row["stem"], "path": str(wav_path.resolve()), "sha256": row["wav_sha256"]})
            input_axis = make_mfa_input_axis_receipt(sorted(selected), audio_rows, axis_root=root / "input")
            input_axis["source_hashes_digest"] = axis["source_hashes_digest"]
            input_path.parent.mkdir(parents=True, exist_ok=True)
            write_once(input_path, input_axis)
        input_hash = sha_file(input_path)
        aligned = root / "workspace" / "aligned"
        if aligned.exists() and (not aligned.is_dir() or any(aligned.iterdir())):
            raise SafetyError("aligned namespace must be absent or empty")
        if aligned.exists(): aligned.rmdir()
        os.replace(temp, aligned)
        alignment_rows = [{"stem": p.stem, "path": str((aligned / p.name).resolve()), "sha256": sha_file(aligned / p.name)} for p in merged]
        axis["axis_root"] = str(aligned.resolve()); axis["alignments"] = alignment_rows
        try:
            alignment_axis = make_mfa_alignment_axis_receipt(input_axis, alignment_rows, alignment_root=aligned)
        except (TypeError, ValueError) as exc:
            raise SafetyError(f"alignment axis binding failed: {exc}")
        alignment_path = root / "workspace" / ".mfa_alignment_axis_receipt.json"
        write_once(alignment_path, alignment_axis)
        alignment_hash = sha_file(alignment_path)
        lineage = {"schema": "gpu1000-alignment-merge-v1", "status": "READY_FOR_DOWNSTREAM",
                   "original_root": str(root), "original_run_receipt_sha256": preflight["original_run_receipt_sha256"],
                   "original_grid_digest": preflight["original_grid_digest"], "scope": scope,
                   "scope_count": 1, "merged_count": 1000, "selection_digest": manifest.get("selection_digest"),
                   "source_hashes_digest": axis["source_hashes_digest"], "source_receipt_sha256": sha_file(root / "workspace" / ".continuation_source_receipt.json"),
                   "axis_receipt_sha256": alignment_hash,
                   "preflight_v2_path": preflight.get("preflight_v2_path", ""),
                   "preflight_v2_sha256": preflight.get("preflight_v2_sha256", ""),
                   "preflight_v2_digest": preflight.get("preflight_v2_digest", ""),
                   "fresh_preflight_digest": preflight.get("fresh_preflight_digest", ""),
                   "strict_receipt_sha256": "",
                   "input_axis_receipt_sha256": input_hash, "aligned_path": str(aligned.resolve()),
                   "downstream_policy": ["align_en", "postprocess", "strict_output"],
                   "batch_policy": continuation_batch_policy(initial_pass=False)}
        write_once(root / "alignment_merge_receipt.json", lineage)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return read_json(root / "alignment_merge_receipt.json")


def continue_after_mfa(root: Path, scope_path: Path | None = None, continuation_root: Path | None = None,
                       *, execute: bool = False, python: str = sys.executable, proven_mfa_retry_receipt: Path | None = None) -> dict[str, Any]:
    """Bind one MFA result and create a fresh continuation namespace.

    ``execute`` is intentionally opt-in and currently only records the
    downstream allowlist; no GPU/CTC process is ever launched here.
    """
    root = root.resolve()
    v2_path = root / "continuation_preflight_receipt_v2.json"
    if not v2_path.is_file():
        raise SafetyError("continuation requires stored v2 preflight receipt")
    try:
        v2 = read_json(v2_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"stored v2 preflight unreadable: {exc}")
    if not isinstance(v2, dict) or v2.get("schema") != CONTINUATION_PREFLIGHT_V2_SCHEMA:
        raise SafetyError("stored v2 preflight schema invalid")
    v1_binding = v2.get("v1_binding", {})
    v1_path = Path(str(v1_binding.get("path", ""))).resolve()
    if v1_path != root / "continuation_preflight_receipt.json" or not v1_path.is_file() or sha_file(v1_path) != v1_binding.get("sha256"):
        raise SafetyError("stored v1 preflight binding invalid")
    fresh_bound = v2.get("fresh_preflight")
    if not isinstance(fresh_bound, dict) or digest(fresh_bound) != v2.get("fresh_preflight_digest"):
        raise SafetyError("stored v2 fresh preflight digest invalid")
    if proven_mfa_retry_receipt is None:
        try: proven_mfa_retry_receipt = Path(str(fresh_bound["proven_mfa_retry_receipt"]))
        except (KeyError, TypeError): pass
    preflight = continuation_preflight(root, scope_path, python=python, proven_mfa_retry_receipt=proven_mfa_retry_receipt)
    if not preflight["ok"]: raise SafetyError("continuation preflight failed: " + "; ".join(preflight["errors"]))
    if preflight != fresh_bound or digest(preflight) != v2.get("fresh_preflight_digest"):
        raise SafetyError("fresh preflight differs from stored v2 evidence")
    if v2.get("continuation_status") != "READY":
        raise SafetyError("stored v2 preflight is not READY")
    stored_digest = digest(v2)
    for namespace, binding in preflight.get("namespace_bindings", {}).items():
        if _empty_namespace_binding(root, namespace) != binding:
            raise SafetyError(f"continuation namespace changed after preflight: {namespace}")
    write_once(root / "continuation_started.json", {"schema": "gpu1000-continuation-start-v1",
        "preflight_digest": stored_digest, "preflight_receipt_sha256": sha_file(v2_path),
        "preflight_v2_path": str(v2_path.resolve()), "preflight_v2_sha256": sha_file(v2_path),
        "fresh_preflight_digest": v2.get("fresh_preflight_digest"),
        "original_run_receipt_sha256": preflight["original_run_receipt_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    _, scope = _discover_continuation_scope(root, scope_path)
    if execute:
        if scope.get("textgrid") or scope.get("grid"):
            raise SafetyError("execute mode must produce singleton MFA TextGrid; external grid is forbidden")
        proof_payload = read_json(proven_mfa_retry_receipt) if proven_mfa_retry_receipt else None
        scope = dict(scope); scope["textgrid"] = str(_execute_singleton_mfa(root, scope, python, proof_payload).resolve())
    preflight = dict(preflight)
    preflight["preflight_v2_path"] = str(v2_path.resolve())
    preflight["preflight_v2_sha256"] = sha_file(v2_path)
    preflight["preflight_v2_digest"] = stored_digest
    preflight["fresh_preflight_digest"] = v2.get("fresh_preflight_digest")
    receipt = _merged_continuation_root(root, continuation_root, preflight, scope)
    if execute:
        downstream = _run_permitted_downstream(root, python)
        return finalize_continuation(root, downstream=downstream)
    receipt["execution_requested"] = False
    return receipt


def finalize_continuation(root: Path, *, downstream: dict[str, Any] | None = None) -> dict[str, Any]:
    """Promote READY_FOR_DOWNSTREAM only after one bound strict receipt exists."""
    root = root.resolve(); result_path = root / "continuation_result.json"; merge_path = root / "alignment_merge_receipt.json"
    if result_path.exists(): raise SafetyError("continuation result already exists")
    if not merge_path.is_file(): raise SafetyError("alignment merge receipt missing")
    result = read_json(merge_path)
    if result.get("status") != "READY_FOR_DOWNSTREAM": raise SafetyError("continuation is not awaiting downstream completion")
    designated = downstream.get("strict_receipt") if isinstance(downstream, dict) else (
        result.get("downstream", {}).get("strict_receipt") if isinstance(result.get("downstream"), dict) else None)
    strict = [Path(designated)] if isinstance(designated, str) and Path(designated).is_file() else []
    if strict and not strict[0].is_relative_to(root / "workspace" / "strict_ok_runs"):
        raise SafetyError("designated strict receipt escapes continuation strict namespace")
    if strict and not strict[0].parent.parent.name.startswith("continuation_"):
        raise SafetyError("designated strict receipt is not from a new continuation run")
    if not strict:
        raise SafetyError("continuation finalization requires designated downstream strict receipt")
    if len(strict) != 1: raise SafetyError("strict output receipt must be exactly one")
    try:
        strict_payload = read_json(strict[0])
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"strict output receipt unreadable: {exc}")
    if not isinstance(strict_payload, dict) or strict_payload.get("schema") != "pipeline-run-receipt-v2":
        raise SafetyError("strict output receipt schema invalid")
    promoted = dict(result); promoted["schema"] = CONTINUATION_SCHEMA; promoted["status"] = "PASS_WITH_CONTINUATION"
    promoted["strict_receipt_sha256"] = sha_file(strict[0]); promoted["strict_receipt_path"] = str(strict[0].resolve())
    if downstream is not None: promoted["downstream"] = downstream
    write_once(result_path, promoted)
    return promoted


def _textgrid_xmax(path: Path) -> float:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("xmax ="):
            try: values.append(float(line.split("=", 1)[1].strip()))
            except ValueError: pass
    if not values: raise SafetyError(f"axis recovery TextGrid xmax missing: {path.name}")
    return max(values)


def axis_recovery_preflight(root: Path) -> dict[str, Any]:
    """Read-only validation for repairing legacy alignment-axis metadata."""
    root = root.resolve(); workspace = root / "workspace"
    old_path = workspace / ".mfa_alignment_axis_receipt.json"
    input_path = workspace / "ctc_pretg" / ".mfa_input_axis_receipt.json"
    aligned = workspace / "aligned"
    if not old_path.is_file() or not input_path.is_file() or not aligned.is_dir():
        raise SafetyError("axis recovery evidence missing")
    try: old = read_json(old_path); input_axis = read_json(input_path)
    except (OSError, json.JSONDecodeError) as exc: raise SafetyError(f"axis recovery receipt unreadable: {exc}")
    if not isinstance(old, dict) or not isinstance(input_axis, dict): raise SafetyError("axis recovery receipt invalid")
    stems = sorted(input_axis.get("stems", []))
    rows = old.get("alignments", [])
    if old.get("stems") != stems or not isinstance(rows, list) or len(rows) != len(stems):
        raise SafetyError("axis recovery old receipt is not exact selected universe")
    audio = {row.get("stem"): row for row in input_axis.get("audio", []) if isinstance(row, dict)}
    corrected: list[dict[str, Any]] = []
    missing_required = 0
    for row in rows:
        stem = row.get("stem") if isinstance(row, dict) else None
        path = Path(str(row.get("path", ""))).resolve() if isinstance(row, dict) else Path("")
        binding = audio.get(stem)
        if not isinstance(stem, str) or not path.is_file() or path.is_symlink() or path.parent != aligned.resolve() or not isinstance(binding, dict):
            raise SafetyError(f"axis recovery row binding invalid: {stem}")
        actual_grid_hash = sha_file(path)
        if row.get("sha256") != actual_grid_hash:
            raise SafetyError(f"axis recovery grid hash changed: {stem}")
        audio_path = Path(str(binding.get("path", "")))
        if not audio_path.is_file() or audio_path.is_symlink() or sha_file(audio_path) != binding.get("sha256"):
            raise SafetyError(f"axis recovery audio hash changed: {stem}")
        xmax = _textgrid_xmax(path)
        duration = binding.get("duration_s")
        if duration is None or abs(float(xmax) - float(duration)) > 0.003:
            raise SafetyError(f"axis recovery TextGrid/audio axis drift: {stem}")
        corrected.append({"stem": stem, "path": str(path), "sha256": actual_grid_hash,
                          "xmax": xmax, "audio_sha256": binding.get("sha256"),
                          "audio_duration_s": duration})
        if not ("audio_sha256" in row and "xmax" in row):
            missing_required += 1
    corrected.sort(key=lambda row: row["stem"])
    return {"schema": "gpu1000-axis-recovery-preflight-v1", "root": str(root),
            "old_axis_path": str(old_path.resolve()), "old_axis_sha256": sha_file(old_path),
            "input_axis_path": str(input_path.resolve()), "input_axis_sha256": sha_file(input_path),
            "aligned_path": str(aligned.resolve()), "stems": stems, "corrected_rows": corrected,
            "missing_required_fields": missing_required,
            "needs_repair": missing_required > 0,
            "ok": True}


def _legacy_resume_matches(old: dict[str, Any], fresh: dict[str, Any]) -> bool:
    if not isinstance(old, dict) or old.get("schema") != fresh.get("schema"):
        return False
    extension_keys = {"alignment_axis_path", "alignment_axis_sha256", "failed_strict_path",
                      "failed_strict_sha256", "failed_report_sha256", "failed_manifest_sha256",
                      "english_manifest_path", "english_manifest_sha256", "english_ledger_digest",
                      "orchestrator_delta_reason"}
    old_cmp = {key: value for key, value in old.items() if key not in extension_keys}
    fresh_cmp = {key: value for key, value in fresh.items() if key not in extension_keys}
    old_codes = dict(old_cmp.get("current_code_hashes", {})); fresh_codes = dict(fresh_cmp.get("current_code_hashes", {}))
    old_codes.pop("orchestrator_sha256", None); fresh_codes.pop("orchestrator_sha256", None)
    old_cmp["current_code_hashes"] = old_codes; fresh_cmp["current_code_hashes"] = fresh_codes
    return old_cmp == fresh_cmp


def recover_alignment_axis(root: Path) -> dict[str, Any]:
    preflight = axis_recovery_preflight(root)
    root = root.resolve(); workspace = root / "workspace"
    old_resume_path = root / "downstream_resume_preflight_receipt.json"
    if not old_resume_path.is_file():
        raise SafetyError("axis recovery requires existing downstream resume receipt")
    resume_evidence = _downstream_resume_preflight(root)
    if not resume_evidence.get("ok"):
        raise SafetyError("axis recovery requires valid old downstream evidence: " + "; ".join(resume_evidence.get("errors", [])))
    if not preflight.get("needs_repair"):
        raise SafetyError("axis recovery is only valid for missing alignment-axis fields")
    try: old_resume = read_json(old_resume_path)
    except (OSError, json.JSONDecodeError) as exc: raise SafetyError(f"old downstream resume unreadable: {exc}")
    if not _legacy_resume_matches(old_resume, resume_evidence):
        raise SafetyError("old downstream resume evidence is stale or tampered")
    output_path = workspace / ".mfa_alignment_axis_receipt_recovered.json"
    chain_path = workspace / ".mfa_alignment_axis_recovery_receipt.json"
    if output_path.exists() or chain_path.exists():
        raise SafetyError("axis recovery receipt already exists")
    input_axis = read_json(Path(preflight["input_axis_path"]))
    corrected = make_mfa_alignment_axis_receipt(input_axis, preflight["corrected_rows"], alignment_root=Path(preflight["aligned_path"]))
    write_once(output_path, corrected)
    chain = {"schema": "gpu1000-axis-recovery-v1", "preflight": preflight,
             "old_resume_path": str(old_resume_path.resolve()), "old_resume_sha256": sha_file(old_resume_path),
             "failed_strict_path": resume_evidence["failed_strict_path"],
             "failed_strict_sha256": resume_evidence["failed_strict_sha256"],
             "failed_report_sha256": resume_evidence["failed_report_sha256"],
             "failed_manifest_sha256": resume_evidence["failed_manifest_sha256"],
             "legacy_resume_policy": "old shared fields exact; failed/alignment-axis/english extensions permitted; upstream code hashes exact; orchestrator delta permitted",
             "corrected_axis_path": str(output_path.resolve()), "corrected_axis_sha256": sha_file(output_path),
             "corrected_axis_digest": digest(corrected)}
    write_once(chain_path, chain)
    return chain


def _downstream_resume_preflight(root: Path, *, python: str = sys.executable) -> dict[str, Any]:
    """Read-only proof that an existing merge may resume downstream only."""
    root = root.resolve(); workspace = root / "workspace"
    errors: list[str] = []
    def load_optional(path: Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            errors.append(f"resume_{label}_unreadable")
            return {}
        if not isinstance(value, dict):
            errors.append(f"resume_{label}_invalid")
            return {}
        return value
    v2_path = root / "continuation_preflight_receipt_v2.json"
    started_path = root / "continuation_started.json"
    merge_path = root / "alignment_merge_receipt.json"
    result_path = root / "continuation_result.json"
    if result_path.exists(): errors.append("continuation_result_already_exists")
    for label, path in (("v2", v2_path), ("started", started_path), ("merge", merge_path)):
        if not path.is_file(): errors.append(f"resume_{label}_missing")
    v2: dict[str, Any] = {}; started: dict[str, Any] = {}; merge: dict[str, Any] = {}
    if v2_path.is_file():
        try: v2 = read_json(v2_path)
        except (OSError, json.JSONDecodeError): errors.append("resume_v2_unreadable")
        if not isinstance(v2, dict) or v2.get("schema") != CONTINUATION_PREFLIGHT_V2_SCHEMA:
            errors.append("resume_v2_schema_invalid"); v2 = {}
    if started_path.is_file():
        try: started = read_json(started_path)
        except (OSError, json.JSONDecodeError): errors.append("resume_started_unreadable")
        if not isinstance(started, dict): errors.append("resume_started_invalid"); started = {}
    if merge_path.is_file():
        try: merge = read_json(merge_path)
        except (OSError, json.JSONDecodeError): errors.append("resume_merge_unreadable")
        if not isinstance(merge, dict): errors.append("resume_merge_invalid"); merge = {}
    v2_sha = sha_file(v2_path) if v2_path.is_file() else ""
    v2_digest = digest(v2) if v2 else ""
    fresh = v2.get("fresh_preflight") if isinstance(v2, dict) else None
    if not isinstance(fresh, dict) or digest(fresh) != v2.get("fresh_preflight_digest"):
        errors.append("resume_v2_fresh_digest_invalid")
    if started.get("preflight_v2_path") != str(v2_path.resolve()) or started.get("preflight_v2_sha256") != v2_sha:
        errors.append("resume_started_v2_binding_invalid")
    if started.get("fresh_preflight_digest") != v2.get("fresh_preflight_digest"):
        errors.append("resume_started_fresh_digest_invalid")
    expected_codes = fresh.get("current_code_hashes", {}) if isinstance(fresh, dict) else {}
    current_codes = {key: sha_file(PROJECT / name) for name, key in
                     (("scripts/run_pipeline.py", "run_pipeline_sha256"),
                      ("scripts/ctc_prealign.py", "ctc_prealign_sha256"),
                      ("scripts/gpu1000_orchestrate.py", "orchestrator_sha256"))}
    for key in ("run_pipeline_sha256", "ctc_prealign_sha256"):
        if expected_codes.get(key) != current_codes.get(key):
            errors.append(f"resume_current_code_hash_mismatch:{key}")
    attempts_path = workspace / "continuation_singleton_attempts.json"
    attempts = []
    if not attempts_path.is_file():
        errors.append("resume_singleton_attempts_missing")
    else:
        try:
            attempts_payload = read_json(attempts_path)
            if isinstance(attempts_payload, dict) and attempts_payload.get("schema") != "gpu1000-singleton-attempts-v1":
                errors.append("resume_singleton_attempts_schema_invalid")
            attempts = attempts_payload.get("attempts", [])
        except (OSError, json.JSONDecodeError, AttributeError): errors.append("resume_singleton_attempts_unreadable")
    def beam_value(attempt: Any, flag: str) -> str | None:
        argv = attempt.get("argv", []) if isinstance(attempt, dict) else []
        try: return str(argv[argv.index(flag) + 1])
        except (ValueError, IndexError, TypeError): return None
    if (not isinstance(attempts, list) or len(attempts) != 2 or
            attempts[0].get("attempt") != "20_80" or attempts[0].get("returncode") != 1 or
            "NoAlignmentsError" not in str(attempts[0].get("stderr", "")) or
            beam_value(attempts[0], "--beam") != "20" or beam_value(attempts[0], "--retry_beam") != "80" or
            attempts[1].get("attempt") != "200_800" or attempts[1].get("returncode") != 0 or
            beam_value(attempts[1], "--beam") != "200" or beam_value(attempts[1], "--retry_beam") != "800"):
        errors.append("resume_singleton_attempts_invalid")
    if merge.get("status") != "READY_FOR_DOWNSTREAM": errors.append("resume_merge_not_ready")
    if merge.get("preflight_v2_path") != str(v2_path.resolve()) or merge.get("preflight_v2_sha256") != v2_sha:
        errors.append("resume_merge_v2_binding_invalid")
    if merge.get("preflight_v2_digest") != v2_digest: errors.append("resume_merge_v2_digest_invalid")
    manifest = load_optional(root / "selected_manifest.json", "manifest")
    selected = {row.get("stem") for row in manifest.get("samples", []) if isinstance(row, dict)}
    aligned = workspace / "aligned"
    grids = sorted(aligned.glob("*.TextGrid")) if aligned.is_dir() else []
    if len(selected) != 1000 or len(grids) != 1000 or {p.stem for p in grids} != selected or any(p.is_symlink() or not p.is_file() for p in grids):
        errors.append("resume_aligned_not_exact1000")
    source_path = workspace / ".continuation_source_receipt.json"
    input_path = workspace / "ctc_pretg" / ".mfa_input_axis_receipt.json"
    axis_path = workspace / ".mfa_alignment_axis_receipt.json"
    recovery_path = workspace / ".mfa_alignment_axis_receipt_recovered.json"
    recovery_chain = workspace / ".mfa_alignment_axis_recovery_receipt.json"
    for label, path in (("source", source_path), ("input_axis", input_path), ("axis", axis_path)):
        if not path.is_file(): errors.append(f"resume_{label}_receipt_missing")
    source = load_optional(source_path, "source_receipt")
    input_axis = load_optional(input_path, "input_axis")
    axis = load_optional(recovery_path if recovery_path.is_file() else axis_path, "alignment_axis")
    if recovery_path.is_file():
        chain = load_optional(recovery_chain, "axis_recovery")
        if (chain.get("corrected_axis_path") != str(recovery_path.resolve()) or
                chain.get("corrected_axis_sha256") != sha_file(recovery_path)):
            errors.append("resume_axis_recovery_binding_invalid")
    if source.get("schema") != "gpu1000-continuation-source-v1" or source.get("count") != 1000 or source.get("stems") != sorted(selected):
        errors.append("resume_source_receipt_invalid")
    if input_axis.get("stems") != sorted(selected): errors.append("resume_input_axis_invalid")
    audio_rows = input_axis.get("audio", []) if isinstance(input_axis, dict) else []
    if not isinstance(audio_rows, list) or {row.get("stem") for row in audio_rows if isinstance(row, dict)} != selected:
        errors.append("resume_input_axis_audio_invalid")
    else:
        for row in audio_rows:
            audio_path = Path(str(row.get("path", "")))
            if not audio_path.is_file() or audio_path.is_symlink() or sha_file(audio_path) != row.get("sha256"):
                errors.append(f"resume_input_audio_hash_invalid:{row.get('stem')}")
    alignments = axis.get("alignments", [])
    if (axis.get("stems") != sorted(selected) or axis.get("alignment_root") != str(aligned.resolve()) or
            not isinstance(alignments, list) or len(alignments) != 1000 or
            {row.get("stem") for row in alignments if isinstance(row, dict)} != selected):
        errors.append("resume_alignment_axis_invalid")
    else:
        for row in alignments:
            path = Path(str(row.get("path", ""))).resolve()
            if not path.is_file() or path.parent != aligned.resolve() or path.is_symlink() or sha_file(path) != row.get("sha256"):
                errors.append(f"resume_alignment_hash_invalid:{row.get('stem')}")
    axis_hash = sha_file(recovery_path) if recovery_path.is_file() else (sha_file(axis_path) if axis_path.is_file() else "")
    if merge.get("source_receipt_sha256") != (sha_file(source_path) if source_path.is_file() else "") or (merge.get("axis_receipt_sha256") != (sha_file(axis_path) if axis_path.is_file() else "") and not recovery_path.is_file()):
        errors.append("resume_merge_receipt_hashes_invalid")
    if merge.get("input_axis_receipt_sha256") != (sha_file(input_path) if input_path.is_file() else ""):
        errors.append("resume_input_axis_hash_invalid")
    english_manifest_path = workspace / "en_phones" / "en_alignment_manifest.json"
    english_manifest_sha = ""
    english_ledger_digest = ""
    if english_manifest_path.is_file() and not english_manifest_path.is_symlink():
        english_manifest = load_optional(english_manifest_path, "english_manifest")
        ledgers = english_manifest.get("stem_ledgers", [])
        ledger_stems = {row.get("stem") for row in ledgers if isinstance(row, dict)} if isinstance(ledgers, list) else set()
        if (english_manifest.get("schema") != "strict-en-mfa-v1" or english_manifest.get("status") != "success" or
                english_manifest.get("strict_provenance") is not True or ledger_stems != selected or len(ledgers) != 1000):
            errors.append("resume_english_manifest_invalid")
        else:
            canonical_ledgers = []
            for row in sorted(ledgers, key=lambda item: item.get("stem", "")):
                path = Path(str(row.get("path", ""))).resolve()
                if path.parent != english_manifest_path.parent.resolve() or path.is_symlink() or not path.is_file() or sha_file(path) != row.get("sha256"):
                    errors.append(f"resume_english_ledger_invalid:{row.get('stem')}")
                canonical_ledgers.append({"stem": row.get("stem"), "path": str(path), "sha256": row.get("sha256")})
            english_ledger_digest = digest(canonical_ledgers)
        english_manifest_sha = sha_file(english_manifest_path)
    strict_receipts = sorted((workspace / "strict_ok_runs").glob("*/output/.pipeline_run_receipt_v2.json"))
    historical_receipts = [p for p in strict_receipts if not p.parent.parent.name.startswith("continuation_")]
    failed_receipts = [p for p in strict_receipts if p.parent.parent.name.startswith("continuation_")]
    if len(historical_receipts) != 1 or len(failed_receipts) != 1:
        errors.append("resume_historical_strict_namespace_invalid")
    failed_receipt = failed_receipts[0] if len(failed_receipts) == 1 else None
    failed_report = failed_receipt.parent / "postprocess_report.jsonl" if failed_receipt else None
    failed_manifest = failed_receipt.parent / "strict_ok_manifest.json" if failed_receipt else None
    for label, path in (("failed_report", failed_report), ("failed_manifest", failed_manifest)):
        if path is None or not path.is_file(): errors.append(f"resume_{label}_missing")
    return {"schema": "gpu1000-downstream-resume-preflight-v1", "root": str(root),
            "v2_path": str(v2_path.resolve()), "v2_sha256": v2_sha, "v2_digest": v2_digest,
            "started_path": str(started_path.resolve()), "started_sha256": sha_file(started_path) if started_path.is_file() else "",
            "merge_path": str(merge_path.resolve()), "merge_sha256": sha_file(merge_path) if merge_path.is_file() else "",
            "current_code_hashes": current_codes, "historical_strict_path": str(historical_receipts[0].resolve()) if len(historical_receipts) == 1 else "",
            "failed_strict_path": str(failed_receipt.resolve()) if failed_receipt else "",
            "failed_strict_sha256": sha_file(failed_receipt) if failed_receipt else "",
            "failed_report_sha256": sha_file(failed_report) if failed_report and failed_report.is_file() else "",
            "failed_manifest_sha256": sha_file(failed_manifest) if failed_manifest and failed_manifest.is_file() else "",
            "english_manifest_path": str(english_manifest_path.resolve()) if english_manifest_path.is_file() else "",
            "english_manifest_sha256": english_manifest_sha, "english_ledger_digest": english_ledger_digest,
            "alignment_axis_path": str((recovery_path if recovery_path.is_file() else axis_path).resolve()),
            "alignment_axis_sha256": axis_hash,
            "orchestrator_delta_reason": "downstream_resume_code_drift_allowed_after_v2; upstream hashes remain bound",
            "errors": sorted(set(errors)), "ok": not errors,
            "continuation_status": "READY" if not errors else "BLOCKED"}


def downstream_resume_preflight(root: Path, *, python: str = sys.executable) -> dict[str, Any]:
    receipt = _downstream_resume_preflight(root, python=python)
    if not receipt["ok"]:
        raise SafetyError("downstream resume preflight failed: " + "; ".join(receipt["errors"]))
    path = root.resolve() / "downstream_resume_preflight_receipt.json"
    write_once(path, receipt)
    return receipt


def downstream_resume_preflight_v2(root: Path, *, python: str = sys.executable, write: bool = True) -> dict[str, Any]:
    """Create the post-axis-recovery resume receipt without touching old evidence."""
    root = root.resolve(); workspace = root / "workspace"
    chain_path = workspace / ".mfa_alignment_axis_recovery_receipt.json"
    old_resume_path = root / "downstream_resume_preflight_receipt.json"
    if not chain_path.is_file() or not old_resume_path.is_file():
        raise SafetyError("downstream resume v2 requires axis recovery and old resume receipts")
    try: chain = read_json(chain_path)
    except (OSError, json.JSONDecodeError) as exc: raise SafetyError(f"axis recovery receipt unreadable: {exc}")
    if chain.get("schema") != "gpu1000-axis-recovery-v1": raise SafetyError("axis recovery receipt schema invalid")
    if chain.get("old_resume_path") != str(old_resume_path.resolve()) or chain.get("old_resume_sha256") != sha_file(old_resume_path):
        raise SafetyError("axis recovery old resume binding invalid")
    fresh = _downstream_resume_preflight(root, python=python)
    if not fresh.get("ok"): raise SafetyError("downstream resume v2 preflight failed: " + "; ".join(fresh.get("errors", [])))
    receipt = {"schema": "gpu1000-downstream-resume-preflight-v2", "root": str(root),
               "reason": "axis_recovery_corrected_alignment_receipt",
               "old_resume_path": str(old_resume_path.resolve()), "old_resume_sha256": sha_file(old_resume_path),
               "axis_recovery_path": str(chain_path.resolve()), "axis_recovery_sha256": sha_file(chain_path),
               "fresh_resume": fresh, "fresh_resume_digest": digest(fresh),
               "alignment_axis_path": fresh["alignment_axis_path"], "alignment_axis_sha256": fresh["alignment_axis_sha256"],
               "current_code_hashes": fresh["current_code_hashes"], "continuation_status": "READY", "ok": True}
    if write:
        write_once(root / "downstream_resume_preflight_receipt_v2.json", receipt)
    return receipt


def continue_downstream(root: Path, *, python: str = sys.executable) -> dict[str, Any]:
    root = root.resolve(); path = root / "downstream_resume_preflight_receipt.json"
    v2_path = root / "downstream_resume_preflight_receipt_v2.json"
    if (root / "workspace" / ".mfa_alignment_axis_recovery_receipt.json").is_file() and not v2_path.is_file():
        raise SafetyError("corrected axis requires downstream resume v2 receipt")
    if v2_path.is_file():
        try: stored_v2 = read_json(v2_path)
        except (OSError, json.JSONDecodeError) as exc: raise SafetyError(f"downstream resume v2 unreadable: {exc}")
        if not isinstance(stored_v2, dict) or stored_v2.get("schema") != "gpu1000-downstream-resume-preflight-v2":
            raise SafetyError("downstream resume v2 schema invalid")
        fresh_v2 = downstream_resume_preflight_v2(root, python=python, write=False)
        if stored_v2 != fresh_v2: raise SafetyError("downstream resume v2 drift")
        downstream = _run_permitted_downstream(root, python, alignment_axis_path=Path(str(stored_v2["alignment_axis_path"])))
        return finalize_continuation(root, downstream=downstream)
    if not path.is_file(): raise SafetyError("downstream resume requires stored preflight receipt")
    try: stored = read_json(path)
    except (OSError, json.JSONDecodeError) as exc: raise SafetyError(f"downstream resume receipt unreadable: {exc}")
    fresh = _downstream_resume_preflight(root, python=python)
    if stored != fresh or stored.get("continuation_status") != "READY":
        raise SafetyError("downstream resume preflight drift or invalid status")
    alignment_axis_path = Path(str(stored.get("alignment_axis_path", "")))
    downstream = _run_permitted_downstream(root, python, alignment_axis_path=alignment_axis_path)
    return finalize_continuation(root, downstream=downstream)


def _run_permitted_downstream(root: Path, python: str, *, alignment_axis_path: Path | None = None) -> dict[str, Any]:
    """Run only align_en → postprocess → strict_ok in a fresh output path."""
    import importlib
    try:
        pipeline = importlib.import_module("scripts.run_pipeline")
    except ModuleNotFoundError:
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        pipeline = importlib.import_module("scripts.run_pipeline")
    cfg = load_yaml(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    cfg = pipeline.load_config(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    run_id = "continuation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "workspace" / "strict_ok_runs" / run_id / "output"
    filtered = output.parent / "filtered"; output.mkdir(parents=True); filtered.mkdir()
    workspace = root / "workspace"; ctc = workspace / "ctc_pretg"
    manifest = read_json(root / "selected_manifest.json")
    expected_stems = sorted(row.get("stem") for row in manifest.get("samples", []) if isinstance(row, dict) and isinstance(row.get("stem"), str))
    args = SimpleNamespace(python=python, overwrite=False)
    source_receipt_path = ctc / ".pipeline_run_receipt_v2.json"
    if not source_receipt_path.is_file():
        raise SafetyError("continuation requires immutable CTC accounting receipt")
    source_receipt = read_json(source_receipt_path)
    if not isinstance(source_receipt, dict) or source_receipt.get("schema") != "pipeline-run-receipt-v2":
        raise SafetyError("continuation CTC accounting receipt schema invalid")
    if validate_pipeline_accounting_receipt(source_receipt):
        raise SafetyError("continuation CTC accounting receipt failed canonical validation")
    source_stems = source_receipt.get("source", {}).get("stems", [])
    eligible_stems = source_receipt.get("eligible", {}).get("stems", [])
    exclusions = source_receipt.get("exclusions", [])
    prior_output = source_receipt.get("output", {}).get("stems", [])
    prior_filtered = source_receipt.get("filtered", {}).get("stems", [])
    accounting = make_pipeline_accounting_receipt(source_stems, eligible_stems, exclusions, prior_output, prior_filtered,
        run_id=run_id, mode="gpu1000_continuation", route=["align_en", "postprocess", "strict_ok"],
        paths={"output": str(output.resolve()), "filtered": str(filtered.resolve())},
        shards=source_receipt.get("shards"), extra={"lineage": "continuation"})
    if validate_pipeline_accounting_receipt(accounting):
        raise SafetyError("constructed continuation accounting receipt failed validation")
    write_once(output / ".pipeline_run_receipt_v2.json", accounting)
    (output / ".pipeline_run_receipt_v2.json").chmod(0o644)
    input_axis_path = ctc / ".mfa_input_axis_receipt.json"
    if not input_axis_path.is_file(): raise SafetyError("continuation input-axis receipt missing")
    input_axis = read_json(input_axis_path)
    mfa_axis_root = Path(str(input_axis.get("axis_root", ""))).resolve()
    tts_root = Path(str(input_axis.get("tts_authoritative_audio_root", root / "input"))).resolve()
    if not mfa_axis_root.is_dir() or not tts_root.is_dir(): raise SafetyError("continuation axis roots unavailable")
    ctx = {"workspace": workspace, "data_dir": root / "input", "expected_stems": expected_stems,
           "workspace": workspace, "ctc_pretg": ctc, "ctc_pretg_adj": ctc,
           "aligned_dir": workspace / "aligned", "mfa_audio_dir": mfa_axis_root,
           "mfa_axis_audio_dir": mfa_axis_root, "tts_authoritative_audio_dir": tts_root,
           "raw_text_dir": root / "input", "output_dir": output, "filtered_dir": filtered,
           "temp_dir": workspace / "continuation_temp", "models_dir": (Path(str(cfg.get("models_dir", "."))) if Path(str(cfg.get("models_dir", "."))).is_absolute() else (PROJECT / str(cfg.get("models_dir", ".")))).resolve(),
           "mfa_dict": Path(cfg.get("mfa_dict", "dict")), "axis_contract_required": True,
           "accounting_required": True, "strict_ready": True,
           "mfa_input_axis_receipt_path": ctc / ".mfa_input_axis_receipt.json",
           "mfa_alignment_axis_receipt_path": (alignment_axis_path.resolve() if alignment_axis_path is not None else workspace / ".mfa_alignment_axis_receipt.json"),
           "accounting_receipt_path": output / ".pipeline_run_receipt_v2.json",
           "accounting_source_receipt_path": source_receipt_path}
    ctx["temp_dir"].mkdir(parents=True, exist_ok=True)
    mfa_python = Path(python).resolve()
    commands = [("align_en", pipeline.step_mfa_align_en), ("postprocess", pipeline.step_postprocess),
                ("strict_ok", pipeline.step_strict_ok)]
    forbidden_tokens = {"trim", "resample", "prealign", "normalize", "adjust", "--force", "--overwrite", "--no-cache", "--output-staging"}
    records = []
    for name, func in commands:
        started = datetime.now(timezone.utc).isoformat()
        reusable_manifest = workspace / "en_phones" / "en_alignment_manifest.json"
        if name == "align_en" and reusable_manifest.is_file() and not reusable_manifest.is_symlink():
            records.append({"step": name, "argv": [name, str(reusable_manifest.resolve())], "returncode": 0,
                            "reused": True, "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat()})
            continue
        rc = func(args, cfg, mfa_python, ctx)
        argv = [name, str(output.resolve()), str(filtered.resolve()), str((workspace / "aligned").resolve())]
        if any(token == forbidden for token in argv for forbidden in forbidden_tokens):
            raise SafetyError(f"forbidden upstream token in downstream argv: {argv}")
        records.append({"step": name, "argv": argv, "returncode": rc, "started_at_utc": started,
                        "ended_at_utc": datetime.now(timezone.utc).isoformat()})
        if rc != 0: raise SafetyError(f"continuation downstream {name} failed rc={rc}")
    strict_receipts = sorted(output.parent.glob("output/.pipeline_run_receipt_v2.json"))
    if len(strict_receipts) != 1: raise SafetyError("continuation strict receipt not exactly one")
    return {"records": records, "strict_receipt": str(strict_receipts[0].resolve()),
            "strict_receipt_sha256": sha_file(strict_receipts[0]), "output": str(output.resolve()),
            "filtered": str(filtered.resolve())}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_tree(path: Path) -> str:
    if path.is_file():
        return sha_file(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(child.relative_to(path)).encode("utf-8")); h.update(b"\0")
        h.update(sha_file(child).encode("ascii")); h.update(b"\n")
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_once(path: Path, value: Any, *, text: bool = False) -> None:
    if path.exists():
        raise SafetyError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if text else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o444)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def require_new_root(root: Path) -> None:
    if root.exists():
        raise SafetyError(f"prepare target must not exist: {root}")
    if root.is_symlink():
        raise SafetyError(f"prepare target may not be a symlink: {root}")


def source_samples(source: Path, quotas: dict[str, int], explicit_stems: list[str] | None = None) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Freeze eligible sources and record, rather than hide, bad source entries.

    A lone WAV must not make a 54k cache unusable: it is an explicit source
    exclusion.  A duplicate *eligible* global identity is different -- there
    is no authoritative choice, so preparation fails before it creates a root.
    """
    candidates: dict[str, list[dict[str, str]]] = {}
    exclusions: list[dict[str, str]] = []
    for speaker in SPEAKERS:
        folder = source / speaker
        if not folder.is_dir():
            raise SafetyError(f"missing speaker source directory: {folder}")
        rows: list[dict[str, str]] = []
        wavs = sorted(folder.glob("*.wav"), key=lambda p: (nfc(p.name), str(p)))
        paired_txt: set[Path] = set()
        for wav in wavs:
            stem = nfc(wav.stem)
            if wav.stem != stem or wav.name != nfc(wav.name):
                exclusions.append({"speaker": speaker, "reason": "non_nfc_wav_name", "path": str(wav)})
                continue
            if not wav.is_file() or wav.is_symlink():
                exclusions.append({"speaker": speaker, "reason": "nonordinary_wav", "path": str(wav)})
                continue
            txt = wav.with_suffix(".txt")
            paired_txt.add(txt)
            if txt.is_symlink() or (txt.exists() and not txt.is_file()):
                exclusions.append({"speaker": speaker, "reason": "nonordinary_sibling_txt", "path": str(txt)})
                continue
            if not txt.exists():
                exclusions.append({"speaker": speaker, "reason": "missing_sibling_txt", "path": str(wav)})
                continue
            if txt.stem != stem or txt.name != nfc(txt.name):
                exclusions.append({"speaker": speaker, "reason": "non_nfc_or_mismatched_sibling_txt", "path": str(txt)})
                continue
            rows.append({"speaker": speaker, "stem": stem, "wav": str(wav), "txt": str(txt),
                         "rank": hashlib.sha256(f"{SEED}\0{speaker}\0{stem}".encode("utf-8")).hexdigest()})
        for txt in sorted(folder.glob("*.txt"), key=lambda p: (nfc(p.name), str(p))):
            if txt not in paired_txt:
                exclusions.append({"speaker": speaker, "reason": "orphan_txt", "path": str(txt)})
        candidates[speaker] = rows
    by_stem: dict[str, list[dict[str, str]]] = {}
    for rows in candidates.values():
        for row in rows: by_stem.setdefault(row["stem"], []).append(row)
    ambiguous = {stem for stem, rows in by_stem.items() if len(rows) > 1}
    for stem in sorted(ambiguous):
        for row in sorted(by_stem[stem], key=lambda r: (r["speaker"], r["wav"])):
            exclusions.append({"speaker": row["speaker"], "reason": "duplicate_global_eligible_stem",
                               "path": row["wav"], "stem": stem})
    if ambiguous:
        raise SafetyError("duplicate global eligible stem authority ambiguity: " + ", ".join(sorted(ambiguous)))
    result: dict[str, list[dict[str, str]]] = {}
    for speaker in SPEAKERS:
        rows = candidates[speaker]
        if explicit_stems is None:
            if len(rows) < quotas[speaker]:
                raise SafetyError(f"{speaker} has {len(rows)} eligible samples; need {quotas[speaker]}")
            result[speaker] = sorted(rows, key=lambda r: (r["rank"], r["stem"]))[:quotas[speaker]]
    if explicit_stems is not None:
        if len(explicit_stems) != len(set(explicit_stems)):
            raise SafetyError("explicit selected-stems list contains duplicates")
        available = {row["stem"]: row for rows in candidates.values() for row in rows}
        missing = sorted(set(explicit_stems) - set(available))
        if missing: raise SafetyError("explicit selected stem is not eligible: " + ", ".join(missing[:10]))
        result = {speaker: [] for speaker in SPEAKERS}
        for stem in explicit_stems: result[available[stem]["speaker"]].append(available[stem])
    return result, sorted(exclusions, key=lambda r: (r["speaker"], r["reason"], r["path"]))


def balanced_quotas(count: int) -> dict[str, int]:
    base, remainder = divmod(count, len(SPEAKERS))
    return {speaker: base + (1 if index < remainder else 0) for index, speaker in enumerate(SPEAKERS)}


def shard_sizes(count: int) -> list[int]:
    base, remainder = divmod(count, 8)
    return [base + (1 if gpu < remainder else 0) for gpu in range(8)]


def load_explicit_stems(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
        stems = value.get("stems") if isinstance(value, dict) else value
    except json.JSONDecodeError:
        stems = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(stems, list) or not all(isinstance(stem, str) and stem for stem in stems):
        raise SafetyError("selected-stems must be a JSON list/{stems:list} or nonempty newline list")
    return [nfc(stem) for stem in stems]


def git_evidence() -> dict[str, Any]:
    def out(*argv: str) -> str:
        try:
            return subprocess.run(argv, cwd=PROJECT, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"
    return {"head": out("git", "rev-parse", "HEAD").strip(),
            "status_porcelain": out("git", "status", "--porcelain"),
            "diff": out("git", "diff", "--binary"),
            "untracked": out("git", "ls-files", "--others", "--exclude-standard")}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SafetyError("PyYAML is required")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SafetyError(f"config is not a mapping: {path}")
    return data


def materialize_config(template: Path, root: Path) -> dict[str, Any]:
    cfg = load_yaml(template)
    cfg["mode"] = "nvrasr_fallback"
    cfg["data_dir"] = str(root / "input")
    cfg["workspace"] = str(root / "workspace")
    cfg["output_dir"] = str(root / "configured_output_never_publish")
    cfg["output_staging"] = False
    cfg["use_cache"] = False
    ctc = cfg.setdefault("ctc_prealign", {})
    ctc.update({"enabled": True, "all_gpus": True, "limit": 0})
    if "/mnt/Raw" in json.dumps(cfg, ensure_ascii=False):
        raise SafetyError("resolved config contains forbidden /mnt/Raw")
    return cfg


def prepare(args: argparse.Namespace) -> int:
    root, source, template = args.root.resolve(), args.source.resolve(), args.config.resolve()
    require_new_root(root)
    explicit_stems = load_explicit_stems(args.selected_stems) if getattr(args, "selected_stems", None) else None
    count = args.count if getattr(args, "count", None) is not None else (len(explicit_stems) if explicit_stems is not None else 1000)
    if count != 1000 and not 8 <= count <= 64:
        raise SafetyError("confirmation count must be 8..64; full run count is exactly 1000")
    if explicit_stems is not None and len(explicit_stems) != count:
        raise SafetyError("--count does not match explicit selected-stems length")
    quotas = balanced_quotas(count) if explicit_stems is None else {speaker: 0 for speaker in SPEAKERS}
    picked, exclusions = source_samples(source, quotas, explicit_stems)
    selected = [row for speaker in SPEAKERS for row in picked[speaker]]
    # A sorted identity order is the contract used by sharding and analysis.
    selected.sort(key=lambda r: (r["speaker"], r["stem"]))
    if len(selected) != count or len({row["stem"] for row in selected}) != count:
        raise SafetyError("selected identity count is not globally unique")
    names = [f"{row['stem']}{suffix}" for row in selected for suffix in (".wav", ".txt")]
    if len(names) != len(set(names)) or any(name != nfc(name) for name in names):
        raise SafetyError("selected destination filename collision or NFC collision")
    root.mkdir(parents=True)
    copied: list[dict[str, str]] = []
    for row in selected:
        destination = root / "input"; destination.mkdir(parents=True, exist_ok=True)
        source_relative = str(Path(row["wav"]).relative_to(source))
        copied_row = {"speaker": row["speaker"], "stem": row["stem"], "source_relative_wav": source_relative,
                      "source_relative_txt": str(Path(row["txt"]).relative_to(source))}
        for kind in ("wav", "txt"):
            src, dst = Path(row[kind]), destination / Path(row[kind]).name
            shutil.copy2(src, dst)  # ordinary copy: links are rejected above
            if dst.is_symlink() or sha_file(src) != sha_file(dst):
                raise SafetyError(f"copy integrity failure: {src} -> {dst}")
            copied_row[f"{kind}_destination"] = str(dst)
            copied_row[f"{kind}_sha256"] = sha_file(dst)
        copied.append(copied_row)
    cfg = materialize_config(template, root)
    config_path = root / "resolved_gpu1000_nvrasr_fallback.yaml"
    if yaml is None: raise SafetyError("PyYAML is required")
    write_once(config_path, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=True), text=True)
    identity = [{key: row[key] for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                for row in copied]
    manifest = {"schema": "gpu1000-selection-v1", "selector": SELECTOR, "seed": SEED,
                "preliminary_selection_digest": PRELIMINARY_SELECTION_DIGEST,
                "source": str(source), "quotas": {speaker: len(picked[speaker]) for speaker in SPEAKERS},
                "count": len(copied), "run_label": "full1000" if count == 1000 else "confirmation_nonpublish",
                "samples": copied, "selection_digest": digest(identity)}
    write_once(root / "selected_manifest.json", manifest)
    write_once(root / "source_inventory.json", {"schema": "gpu1000-source-inventory-v1", "source": str(source),
                                                  "eligible_counts": {speaker: len(picked[speaker]) for speaker in SPEAKERS},
                                                  "exclusions": exclusions, "exclusions_digest": digest(exclusions)})
    write_once(root / "selected_stems.json", {"stems": [r["stem"] for r in copied], "digest": digest(identity)})
    # ctc_prealign discovers the flat directory by sorted WAV pathname.  The
    # selection contract may retain speaker grouping, but the shard plan must
    # mirror that consumer order exactly or its all-GPU receipt cannot bind.
    ctc_order = sorted(copied, key=lambda row: Path(row["wav_destination"]).name)
    offset = 0; shards = []
    for gpu, size in enumerate(shard_sizes(count)):
        shards.append({"gpu": gpu, "stems": [r["stem"] for r in ctc_order[offset:offset + size]]}); offset += size
    write_once(root / "shard_plan.json", {"schema": "gpu1000-shard-plan-v1", "shards": shards,
                                           "digest": digest(shards)})
    evidence = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "git": git_evidence(),
                "config_template": str(template), "config_template_sha256": sha_file(template),
                "resolved_config_sha256": sha_file(config_path),
                "run_pipeline_sha256": sha_file(PROJECT / "scripts/run_pipeline.py"),
                "ctc_prealign_sha256": sha_file(PROJECT / "scripts/ctc_prealign.py"),
                "model_and_dictionary": provenance_paths(cfg),
                "source_exclusions": exclusions, "source_exclusions_digest": digest(exclusions)}
    write_once(root / "prepare_evidence.json", evidence)
    print(json.dumps({"root": str(root), "count": len(copied), "selection_digest": manifest["selection_digest"]}))
    return 0


def provenance_paths(cfg: dict[str, Any]) -> list[dict[str, str]]:
    raw = [cfg.get("models_dir"), cfg.get("mfa_dict"), cfg.get("pinyin_dict"),
           cfg.get("ctc_prealign", {}).get("model_path"), cfg.get("mfa_en", {}).get("dictionary"),
           cfg.get("mfa_en", {}).get("acoustic_model"), cfg.get("mfa_en", {}).get("g2p_model")]
    result = []
    for value in raw:
        if value:
            p = Path(str(value))
            result.append({"path": str(p), "exists": str(p.exists()),
                           "sha256": sha_tree(p) if p.exists() else ""})
    return result


def state(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name in ("selected_manifest.json", "prepare_evidence.json", "source_inventory.json", "shard_plan.json",
                 "resolved_gpu1000_nvrasr_fallback.yaml"):
        if not (root / name).is_file(): raise SafetyError(f"not an exclusive prepared root; missing {name}")
    return read_json(root / "selected_manifest.json"), read_json(root / "prepare_evidence.json"), read_json(root / "shard_plan.json")


def validate_prepared(root: Path) -> list[str]:
    manifest, evidence, shards = state(root)
    errors: list[str] = []
    samples = manifest.get("samples", [])
    count = manifest.get("count")
    if not isinstance(count, int) or count != len(samples) or (count != 1000 and not 8 <= count <= 64): errors.append("selection_count_invalid")
    if len({r.get("stem") for r in samples}) != count: errors.append("selected_stems_not_globally_unique")
    if count == 1000 and manifest.get("quotas") != QUOTAS: errors.append("full_selection_quota_invalid")
    if count != 1000 and manifest.get("run_label") != "confirmation_nonpublish": errors.append("confirmation_run_not_labeled_nonpublish")
    identity = [{key: row.get(key) for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                for row in samples]
    if digest(identity) != manifest.get("selection_digest"): errors.append("selection_manifest_digest_mismatch")
    wanted = {r["stem"]: r for r in samples}
    input_dir = root / "input"
    expected_names = {f"{stem}{suffix}" for stem in wanted for suffix in (".wav", ".txt")}
    actual_names = {entry.name for entry in input_dir.iterdir()} if input_dir.is_dir() else set()
    if actual_names != expected_names or len(actual_names) != count * 2:
        errors.append("flat_input_namespace_not_exact")
    for entry in input_dir.iterdir() if input_dir.is_dir() else ():
        if entry.is_dir() or entry.is_symlink() or not entry.is_file(): errors.append(f"flat_input_nonordinary_entry:{entry.name}")
    for stem, row in wanted.items():
        for suffix, key in ((".wav", "wav_sha256"), (".txt", "txt_sha256")):
            file = input_dir / f"{stem}{suffix}"
            if (not file.is_file() or file.is_symlink() or str(file) != row.get(key.replace("_sha256", "_destination"))
                    or sha_file(file) != row.get(key)):
                errors.append(f"input_hash_mismatch:{stem}{suffix}")
    shard_rows = shards.get("shards", [])
    members = [stem for shard in shard_rows for stem in shard.get("stems", [])]
    expected_sizes = shard_sizes(count) if isinstance(count, int) else []
    if (len(shard_rows) != 8 or [len(s.get("stems", [])) for s in shard_rows] != expected_sizes
            or set(members) != set(wanted) or len(members) != len(set(members))):
        errors.append("shard_plan_not_eight_disjoint_expected_sizes")
    if sha_file(root / "resolved_gpu1000_nvrasr_fallback.yaml") != evidence.get("resolved_config_sha256"):
        errors.append("resolved_config_hash_mismatch")
    for name, path in (("run_pipeline_sha256", PROJECT / "scripts/run_pipeline.py"),
                       ("ctc_prealign_sha256", PROJECT / "scripts/ctc_prealign.py")):
        if sha_file(path) != evidence.get(name): errors.append(f"code_hash_mismatch:{path.name}")
    cfg = load_yaml(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    if (cfg.get("data_dir") != str(root / "input") or cfg.get("workspace") != str(root / "workspace")
            or cfg.get("output_dir") != str(root / "configured_output_never_publish")
            or cfg.get("output_staging") is not False or cfg.get("use_cache") is not False):
        errors.append("resolved_config_safety_values_invalid")
    ctc = cfg.get("ctc_prealign", {})
    if not (ctc.get("enabled") is True and ctc.get("all_gpus") is True and ctc.get("limit") == 0):
        errors.append("resolved_ctc_all_gpus_limit_invalid")
    if (root / "workspace").exists() or (root / "configured_output_never_publish").exists():
        errors.append("prepared_root_not_exclusive_workspace_or_output_exists")
    if "/mnt/Raw" in (root / "resolved_gpu1000_nvrasr_fallback.yaml").read_text(encoding="utf-8"):
        errors.append("forbidden_publish_path")
    return errors


def _command_error(label: str, completed: subprocess.CompletedProcess[str]) -> SafetyError:
    """Keep diagnostic evidence useful without embedding unbounded command output."""
    stderr = (completed.stderr or "").strip()[:2000]
    stdout = (completed.stdout or "").strip()[:2000]
    return SafetyError(f"{label} rc={completed.returncode} stderr={stderr!r} stdout={stdout!r}")


def gpu_snapshot(fake: str | None = None) -> list[dict[str, Any]]:
    if fake:
        data = json.loads(Path(fake).read_text(encoding="utf-8"))
        return data["gpus"] if isinstance(data, dict) else data
    gpu_proc = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
                               "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    if gpu_proc.returncode: raise _command_error("nvidia-smi gpu query failed", gpu_proc)
    rows: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for line in gpu_proc.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 5: raise SafetyError(f"nvidia-smi gpu query malformed row: {line!r}")
        row = {"index": int(values[0]), "uuid": values[1], "memory_free_mib": int(values[2]),
               "memory_used_mib": int(values[3]),
               "utilization_gpu_pct": None if values[4] in ("", "N/A", "[Not Supported]") else int(values[4]),
               "compute_pids": []}
        rows.append(row); by_uuid[values[1]] = row
    apps_proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if apps_proc.returncode: raise _command_error("nvidia-smi compute-app query failed", apps_proc)
    for line in apps_proc.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if not line.strip(): continue
        if len(values) != 2: raise SafetyError(f"nvidia-smi compute-app query malformed row: {line!r}")
        pid, uuid = values
        if uuid not in by_uuid: raise SafetyError(f"nvidia-smi compute-app references unknown gpu uuid: {uuid!r}")
        by_uuid[uuid]["compute_pids"].append(pid)
    return rows


def resolve_mfa_dependency(python: str) -> tuple[str | None, str]:
    """Resolve MFA in the exact interpreter environment, never ambient PATH."""
    executable = Path(python).resolve()
    sibling = executable.parent / "mfa"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling), "sibling_executable"
    probe = subprocess.run([str(executable), "-c", "import montreal_forced_aligner"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if probe.returncode == 0:
        return f"module:{executable}:montreal_forced_aligner", "module_import"
    return None, f"missing sibling {sibling}; module probe {_command_error('mfa module import failed', probe)}"


def preflight(args: argparse.Namespace) -> int:
    root = args.root.resolve(); errors = validate_prepared(root)
    cfg = load_yaml(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    if args.fake_gpus:
        mfa_path, mfa_resolution = None, "not_checked_fake_mode"
    else:
        mfa_path, mfa_resolution = resolve_mfa_dependency(getattr(args, "python", sys.executable))
    if not args.fake_gpus:
        for record in provenance_paths(cfg):
            if record["exists"] != "True": errors.append(f"missing_dependency:{record['path']}")
        if mfa_path is None: errors.append("missing_dependency:mfa")
    free = shutil.disk_usage(root).free
    if free < 100 * 1024**3: errors.append("insufficient_disk_less_than_100GiB")
    try:
        gpus = gpu_snapshot(args.fake_gpus)
        indices = sorted(g.get("index") for g in gpus)
        if indices != list(range(8)): errors.append("need_exactly_gpus_0_through_7")
        for gpu in gpus:
            if int(gpu.get("memory_free_mib", 0)) < 40 * 1024: errors.append(f"gpu{gpu.get('index')}_free_memory_below_40GiB")
            if gpu.get("compute_pids"): errors.append(f"gpu{gpu.get('index')}_has_compute_pids")
    except (ValueError, OSError, SafetyError, json.JSONDecodeError) as exc: errors.append(f"gpu_probe_failed:{exc}")
    receipt = {"schema": "gpu1000-preflight-v1", "at_utc": datetime.now(timezone.utc).isoformat(),
               "root": str(root), "errors": errors, "gpus": gpus if 'gpus' in locals() else [],
               "mfa_interpreter": str(Path(getattr(args, "python", sys.executable)).resolve()),
               "resolved_mfa_path": mfa_path, "mfa_resolution": mfa_resolution}
    write_once(root / "preflight_receipt.json", receipt)
    if errors: raise SafetyError("preflight failed: " + "; ".join(errors))
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def safe_env() -> dict[str, str]:
    return {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}


def telemetry_loop(path: Path, stop: threading.Event, fake: str | None) -> None:
    while not stop.is_set():
        try: snapshot: dict[str, Any] = {"at_utc": datetime.now(timezone.utc).isoformat(), "gpus": gpu_snapshot(fake)}
        except Exception as exc: snapshot = {"at_utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
        with path.open("a", encoding="utf-8") as out: out.write(json.dumps(snapshot) + "\n")
        stop.wait(TELEMETRY_INTERVAL_SECONDS)


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if (root / "run_receipt.json").exists() or (root / "run_started.json").exists():
        raise SafetyError("same run root cannot be run twice")
    if not (root / "preflight_receipt.json").is_file(): raise SafetyError("run requires successful preflight receipt")
    if read_json(root / "preflight_receipt.json").get("errors"): raise SafetyError("run requires successful preflight")
    errors = validate_prepared(root)
    if errors: raise SafetyError("prepared state changed: " + "; ".join(errors))
    config = root / "resolved_gpu1000_nvrasr_fallback.yaml"
    if "/mnt/Raw" in config.read_text(encoding="utf-8"): raise SafetyError("refusing /mnt/Raw publication")
    argv = [args.python, str(PROJECT / "scripts/run_pipeline.py"), "--config", str(config), "--python", args.python,
            "--no-cache", "--no-output-staging", "--validate"]
    write_once(root / "run_started.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
                                            "environment": safe_env(), "telemetry_interval_seconds": TELEMETRY_INTERVAL_SECONDS})
    if args.dry_run:
        write_once(root / "run_receipt.json", {"schema": "gpu1000-run-v1", "dry_run": True, "argv": argv,
                                                "returncode": 0, "started_at_utc": datetime.now(timezone.utc).isoformat(), "ended_at_utc": datetime.now(timezone.utc).isoformat()})
        return 0
    stdout, stderr, telemetry = root / "pipeline.stdout.log", root / "pipeline.stderr.log", root / "nvidia-smi.telemetry.jsonl"
    stop = threading.Event(); watcher = threading.Thread(target=telemetry_loop, args=(telemetry, stop, args.fake_gpus), daemon=True)
    started = datetime.now(timezone.utc).isoformat(); watcher.start()
    try:
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            completed = subprocess.run(argv, cwd=PROJECT, env=safe_env(), stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False)
    finally:
        stop.set(); watcher.join(timeout=20)
    write_once(root / "run_receipt.json", {"schema": "gpu1000-run-v1", "argv": argv, "environment": safe_env(), "returncode": completed.returncode,
                                             "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                             "stdout_sha256": sha_file(stdout), "stderr_sha256": sha_file(stderr), "telemetry_sha256": sha_file(telemetry)})
    return completed.returncode


def _continuation_preflight_command(args: argparse.Namespace) -> int:
    receipt = continuation_preflight(args.root, args.scope, expected_grids=args.expected_grids, python=args.python, proven_mfa_retry_receipt=args.proven_mfa_retry_receipt)
    # A preflight receipt is evidence, not permission to mutate the failed
    # root.  Refuse to overwrite an existing receipt.
    write_once(args.root.resolve() / "continuation_preflight_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False)); return 0 if receipt["ok"] else 2


def _continuation_preflight_v2_command(args: argparse.Namespace) -> int:
    receipt = continuation_preflight_v2(args.root, args.scope, expected_grids=args.expected_grids,
                                        python=args.python, proven_mfa_retry_receipt=args.proven_mfa_retry_receipt)
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def _continue_after_mfa_command(args: argparse.Namespace) -> int:
    receipt = continue_after_mfa(args.root, args.scope, execute=args.execute, python=args.python, proven_mfa_retry_receipt=args.proven_mfa_retry_receipt)
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def _downstream_resume_preflight_command(args: argparse.Namespace) -> int:
    receipt = downstream_resume_preflight(args.root, python=args.python)
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def _continue_downstream_command(args: argparse.Namespace) -> int:
    receipt = continue_downstream(args.root, python=args.python)
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def _axis_recovery_command(args: argparse.Namespace) -> int:
    receipt = recover_alignment_axis(args.root)
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def _downstream_resume_preflight_v2_command(args: argparse.Namespace) -> int:
    receipt = downstream_resume_preflight_v2(args.root, python=args.python)
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--root", type=Path, required=True); p.add_argument("--source", type=Path, default=Path("/mnt/nvme3/mfa_audio_cache_ria")); p.add_argument("--config", type=Path, default=PROJECT / "configs/gpu1000_nvrasr_fallback.yaml"); p.add_argument("--count", type=int); p.add_argument("--selected-stems", type=Path); p.set_defaults(func=prepare)
    p = sub.add_parser("preflight"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable); p.add_argument("--fake-gpus", type=str); p.set_defaults(func=preflight)
    p = sub.add_parser("run"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable); p.add_argument("--fake-gpus", type=str); p.add_argument("--dry-run", action="store_true"); p.set_defaults(func=run)
    p = sub.add_parser("continuation-preflight"); p.add_argument("--root", type=Path, required=True); p.add_argument("--scope", type=Path); p.add_argument("--python", default=sys.executable); p.add_argument("--proven-mfa-retry-receipt", type=Path, required=True); p.add_argument("--expected-grids", type=int, default=999)
    p.set_defaults(func=lambda a: _continuation_preflight_command(a))
    p = sub.add_parser("continuation-preflight-v2"); p.add_argument("--root", type=Path, required=True); p.add_argument("--scope", type=Path); p.add_argument("--python", default=sys.executable); p.add_argument("--proven-mfa-retry-receipt", type=Path); p.add_argument("--expected-grids", type=int, default=999)
    p.set_defaults(func=lambda a: _continuation_preflight_v2_command(a))
    p = sub.add_parser("continue-after-mfa"); p.add_argument("--root", type=Path, required=True); p.add_argument("--scope", type=Path); p.add_argument("--python", default=sys.executable); p.add_argument("--proven-mfa-retry-receipt", type=Path); p.add_argument("--execute", action="store_true")
    p.set_defaults(func=lambda a: _continue_after_mfa_command(a))
    p = sub.add_parser("downstream-resume-preflight"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable)
    p.set_defaults(func=lambda a: _downstream_resume_preflight_command(a))
    p = sub.add_parser("continue-downstream"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable)
    p.set_defaults(func=lambda a: _continue_downstream_command(a))
    p = sub.add_parser("axis-recovery"); p.add_argument("--root", type=Path, required=True)
    p.set_defaults(func=lambda a: _axis_recovery_command(a))
    p = sub.add_parser("downstream-resume-preflight-v2"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable)
    p.set_defaults(func=lambda a: _downstream_resume_preflight_v2_command(a))
    p = sub.add_parser("analyze"); p.add_argument("--root", type=Path, required=True); p.add_argument("--json-out", type=Path); p.add_argument("--markdown-out", type=Path); p.set_defaults(func=lambda a: __import__("analyze_gpu1000_run").analyze_command(a))
    args = parser.parse_args()
    try: return args.func(args)
    except SafetyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
