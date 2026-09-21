"""Contracts shared by the Japanese/English pipeline stages.

This module intentionally contains only deterministic, dependency-light
plumbing.  Stage implementations may depend on it, but it must not import a
model runtime (MFA, ASR, or a frontend).  Keeping the contract here makes it
possible to validate a run before loading a model and gives the independent
verifier a small surface to audit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCHEMAS = frozenset(
    {
        "ja-supply-chain-lock-v1",
        "ja-asr-evidence-v2",
        "ja-reading-selection-v2",
        "ja-frontend-contract-v2",
        "ja-semantic-phone-graph-v1",
        "ja-en-alignment-plan-v2",
        "strict-ja-mfa-v2",
        "strict-en-mfa-v2",
        "julius-diagnostic-v1",
        "ja-en-alignment-v2",
        "tts-training-record-v1",
        "audio-transform-receipt-v2",
        "ja-pipeline-receipt-v1",
    }
)

# Stable codes are part of the public ledger.  Do not use exception class
# names or Python error text as a machine-readable failure reason.
ERROR_CODES = frozenset(
    {
        "config_malformed",
        "config_unknown_key",
        "config_path_invalid",
        "supply_chain_invalid",
        "supply_chain_license_unknown",
        "supply_chain_hash_mismatch",
        "dependency_ambiguous",
        "provider_ambiguous",
        "runtime_capability_missing",
        "model_asset_missing",
        "model_asset_hash_mismatch",
        "dictionary_asset_missing",
        "source_audio_invalid",
        "audio_transform_invalid",
        "asr_provider_failed",
        "asr_family_duplicate",
        "reading_unresolved",
        "anchor_invalid",
        "seam_rejected",
        "julius_diagnostic_unavailable",
        "julius_writeback_forbidden",
        "tts_invalid",
        "verifier_failed",
        "schema_unknown",
        "schema_invalid",
        "manifest_invalid",
        "manifest_duplicate_id",
        "manifest_path_invalid",
        "manifest_symlink",
        "partition_not_exact",
        "receipt_invalid",
        "receipt_missing",
        "receipt_input_missing",
        "receipt_hash_mismatch",
        "receipt_output_missing",
        "receipt_output_symlink",
        "resume_stale",
        "resume_extra_file",
        "resume_identity_drift",
        "cache_tampered",
        "cache_miss",
        "lock_busy",
        "lock_stale",
        "frontend_provider_ambiguous",
        "frontend_capability_missing",
        "frontend_span_unbound",
        "frontend_empty_lexical_phones",
        "frontend_representation_drift",
        "reading_ambiguous",
        "semantic_parse_failed",
        "semantic_relation_ambiguous",
        "mfa_phone_unsupported",
        "mfa_inventory_mismatch",
        "dictionary_roundtrip_failed",
        "mora_phone_relation_unresolved",
        "anchor_conflict",
        "alignment_invalid",
        "publish_blocked",
    }
)

PRODUCTION_STAGES = (
    "inventory",
    "audio",
    "asr",
    "reading",
    "frontend",
    "semantic",
    "anchors",
    "align",
    "merge",
    "tts",
    "verify",
)
STAGE_NAMES = PRODUCTION_STAGES + ("julius",)

# Required fields are deliberately small and stable.  Stage owners may add
# evidence fields, but these fields are the cross-stage join keys consumed by
# the verifier.  The dictionaries are also useful to downstream workers as a
# single source of truth for serialized JSONL records.
SCHEMA_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "ja-supply-chain-lock-v1": frozenset({"schema", "status", "resources"}),
    "ja-asr-evidence-v2": frozenset({"schema", "uid", "providers"}),
    "ja-reading-selection-v2": frozenset({"schema", "uid", "status", "candidates", "selected_reading"}),
    "ja-frontend-contract-v2": frozenset({"schema", "caller_text", "text_layer_digest", "frontend_commit", "options", "units"}),
    "ja-semantic-phone-graph-v1": frozenset({"schema", "uid", "nodes", "edges"}),
    "ja-en-alignment-plan-v2": frozenset({"schema", "uid", "runs", "seams"}),
    "strict-ja-mfa-v2": frozenset({"schema", "uid", "language", "runs", "ledger"}),
    "strict-en-mfa-v2": frozenset({"schema", "uid", "language", "runs", "ledger"}),
    "julius-diagnostic-v1": frozenset({"schema", "uid", "status", "production_write_back"}),
    "ja-en-alignment-v2": frozenset({"schema", "uid", "words", "phones", "languages"}),
    "tts-training-record-v1": frozenset({"schema", "uid", "train_wav", "alignment_wav", "phones", "durations", "mora_graph", "quality_masks"}),
    "audio-transform-receipt-v2": frozenset({"schema", "uid", "source", "train", "alignment", "sample_transform"}),
    "ja-pipeline-receipt-v1": frozenset({"schema", "stage", "status", "inputs", "outputs", "params", "tools", "commands", "errors"}),
}

ALIAS_PREFIXES = {"ja": "ju_", "en": "eu_"}


class JAContractError(ValueError):
    """A contract failure with a stable code and JSON path."""

    def __init__(self, code: str, message: str, path: str = "$") -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown stable error code: {code}")
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def canonical_json(value: Any) -> bytes:
    """Encode JSON in the one representation used for identities and receipts."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise JAContractError("schema_invalid", f"non-canonical JSON value: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _absolute(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().absolute()


def reject_symlink(path: os.PathLike[str] | str, *, code: str = "manifest_symlink") -> Path:
    candidate = _absolute(path)
    if candidate.is_symlink():
        raise JAContractError(code, "symlinks are not permitted", str(candidate))
    return candidate


def ensure_output_path(path: os.PathLike[str] | str, workspace: os.PathLike[str] | str) -> Path:
    """Validate an output path is a real file location below the run workspace."""

    candidate = reject_symlink(path)
    root = reject_symlink(workspace).resolve()
    try:
        candidate.parent.resolve().relative_to(root)
    except ValueError as exc:
        raise JAContractError("manifest_path_invalid", "output escapes workspace", str(candidate)) from exc
    # An existing parent component may itself be a symlink, even if the final
    # path does not exist yet.
    current = candidate.parent
    while current != root and current != current.parent:
        if current.is_symlink():
            raise JAContractError("manifest_symlink", "output parent is a symlink", str(current))
        current = current.parent
    return candidate


def atomic_write_bytes(path: os.PathLike[str] | str, data: bytes, *, workspace: os.PathLike[str] | str | None = None) -> Path:
    destination = _absolute(path)
    if workspace is not None:
        destination = ensure_output_path(destination, workspace)
    elif destination.exists() and destination.is_symlink():
        raise JAContractError("manifest_symlink", "cannot replace symlink", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any], *, workspace: os.PathLike[str] | str | None = None) -> Path:
    return atomic_write_bytes(path, canonical_json(payload), workspace=workspace)


def load_json(path: os.PathLike[str] | str, *, code: str = "receipt_invalid") -> Any:
    candidate = reject_symlink(path, code="receipt_output_symlink")
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise JAContractError(code, str(exc), str(candidate)) from exc


def validate_schema(payload: Mapping[str, Any], expected: str | None = None) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise JAContractError("schema_invalid", "payload must be an object")
    schema = payload.get("schema")
    if schema not in SCHEMAS:
        raise JAContractError("schema_unknown", f"unsupported schema {schema!r}", "$.schema")
    if expected is not None and schema != expected:
        raise JAContractError("schema_invalid", f"expected {expected!r}, got {schema!r}", "$.schema")
    return payload


def validate_record(payload: Mapping[str, Any], expected: str | None = None) -> Mapping[str, Any]:
    """Validate the cross-stage fields of a JSON object or JSONL row."""

    validate_schema(payload, expected)
    required = SCHEMA_REQUIRED_FIELDS[payload["schema"]]
    missing = sorted(required - set(payload))
    if missing:
        raise JAContractError("schema_invalid", f"missing required fields: {missing}", "$")
    return payload


def make_record(schema: str, **fields: Any) -> dict[str, Any]:
    if schema not in SCHEMA_REQUIRED_FIELDS:
        raise JAContractError("schema_unknown", f"unsupported schema {schema!r}", "$.schema")
    return {"schema": schema, **fields}


def make_occurrence_alias(language: str, ordinal: int) -> str:
    if language not in ALIAS_PREFIXES or type(ordinal) is not int or ordinal < 0:
        raise JAContractError("schema_invalid", "language must be ja/en and ordinal a non-negative integer")
    return f"{ALIAS_PREFIXES[language]}{ordinal:06d}"


def validate_occurrence_alias(alias: str, language: str | None = None) -> str:
    if not isinstance(alias, str) or not alias.isascii() or alias.count("_") != 1:
        raise JAContractError("schema_invalid", "alias must be ASCII with one prefix separator", "$.alias")
    prefix, ordinal = alias.split("_", 1)
    if f"{prefix}_" not in ALIAS_PREFIXES.values() or not ordinal.isdigit():
        raise JAContractError("schema_invalid", "invalid occurrence alias", "$.alias")
    if language is not None and ALIAS_PREFIXES.get(language) != f"{prefix}_":
        raise JAContractError("schema_invalid", "alias language mismatch", "$.alias")
    return alias


def validate_alias_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Require one locked pronunciation row for every occurrence alias."""

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        alias = row.get("alias")
        try:
            validate_occurrence_alias(alias)
        except JAContractError as exc:
            raise JAContractError(exc.code, exc.message, f"$[{index}].alias") from exc
        if alias in seen:
            raise JAContractError("schema_invalid", "duplicate occurrence alias", f"$[{index}].alias")
        if not isinstance(row.get("pronunciation"), list) or not row["pronunciation"]:
            raise JAContractError("dictionary_roundtrip_failed", "alias needs one non-empty pronunciation", f"$[{index}]")
        seen.add(alias)
        result.append(dict(row))
    return result


def validate_exact_partition(
    expected: Sequence[str] | set[str],
    buckets: Mapping[str, Sequence[str] | set[str]],
) -> dict[str, list[str]]:
    """Require disjoint buckets whose union is exactly ``expected``."""

    expected_list = list(expected)
    if len(expected_list) != len(set(expected_list)):
        raise JAContractError("partition_not_exact", "expected set contains duplicate identifiers")
    expected_set = set(expected_list)
    seen: set[str] = set()
    normalized: dict[str, list[str]] = {}
    for name, values in buckets.items():
        current = list(values)
        if len(current) != len(set(current)):
            raise JAContractError("partition_not_exact", f"duplicate item in bucket {name}", f"$.{name}")
        overlap = seen.intersection(current)
        if overlap:
            raise JAContractError("partition_not_exact", f"overlap {sorted(overlap)!r}", f"$.{name}")
        unknown = set(current) - expected_set
        if unknown:
            raise JAContractError("partition_not_exact", f"unknown items {sorted(unknown)!r}", f"$.{name}")
        seen.update(current)
        normalized[name] = sorted(current)
    if seen != expected_set:
        missing = sorted(expected_set - seen)
        extra = sorted(seen - expected_set)
        raise JAContractError("partition_not_exact", f"missing={missing!r}, extra={extra!r}")
    return normalized


def validate_manifest(manifest: Any, *, workspace: os.PathLike[str] | str | None = None) -> list[dict[str, Any]]:
    """Validate source rows while allowing authorized absolute source paths.

    Source audio is an input and may live outside the output workspace.  Only
    output paths are containment checked by this foundation layer.
    """

    rows = manifest.get("items") if isinstance(manifest, Mapping) else manifest
    if not isinstance(rows, list):
        raise JAContractError("manifest_invalid", "manifest must be a list or {items: [...]} ")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise JAContractError("manifest_invalid", "row must be an object", f"$[{index}]")
        uid = row.get("uid") or row.get("id")
        if not isinstance(uid, str) or not uid:
            raise JAContractError("manifest_invalid", "uid is required", f"$[{index}].uid")
        if uid in seen:
            raise JAContractError("manifest_duplicate_id", uid, f"$[{index}].uid")
        seen.add(uid)
        if not isinstance(row.get("text", row.get("orig_text")), str) or not row.get("text", row.get("orig_text")).strip():
            raise JAContractError("manifest_invalid", "non-empty text is required", f"$[{index}].text")
        audio_fields = [row.get(key) for key in ("wav", "audio", "source_wav")]
        if not any(isinstance(value, str) and value.strip() for value in audio_fields):
            raise JAContractError("manifest_invalid", "a non-empty audio path is required", f"$[{index}].audio")
        for key in ("wav", "audio", "source_wav", "script", "text"):
            value = row.get(key)
            if value is not None and not isinstance(value, str):
                raise JAContractError("manifest_invalid", "path/text field must be a string", f"$[{index}].{key}")
        result.append(dict(row))
    return result


def validate_config(config: Any, *, config_path: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise JAContractError("config_malformed", "top-level config must be a mapping")
    required = {"pipeline", "workspace", "input_manifest", "supply_chain_lock"}
    missing = sorted(required - set(config))
    if missing:
        raise JAContractError("config_malformed", f"missing required keys: {missing}")
    allowed_top = required | {
        "asr", "frontend", "mfa", "mixed", "julius_diagnostic", "publish",
        "stage_inputs", "synthetic_fixture", "inventory", "audio", "reading",
        "semantic", "anchors", "align", "merge", "tts", "verify", "cache",
        "mapping_file", "gold_manifest", "manual_overrides", "code_root",
    }
    unknown_top = sorted(set(config) - allowed_top)
    if unknown_top:
        raise JAContractError("config_unknown_key", f"unknown top-level keys: {unknown_top}", "$")
    if config.get("pipeline") != "ja_en_tts":
        raise JAContractError("config_malformed", "pipeline must be ja_en_tts", "$.pipeline")
    workspace = config.get("workspace")
    if not isinstance(workspace, str) or not workspace.strip():
        raise JAContractError("config_malformed", "workspace must be a path", "$.workspace")
    for field in ("input_manifest", "supply_chain_lock"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise JAContractError("config_malformed", f"{field} must be a path", f"$.{field}")
    for key in ("asr", "frontend", "mfa", "mixed", "julius_diagnostic", "publish"):
        if key in config and not isinstance(config[key], Mapping):
            raise JAContractError("config_malformed", "section must be a mapping", f"$.{key}")
    asr = config.get("asr", {})
    if asr and asr.get("profile") not in {"qwen_only_dev", "baseline_3family", "extended_5model"}:
        raise JAContractError("config_malformed", "unsupported ASR profile", "$.asr.profile")
    if asr and asr.get("family_vote_policy") != "one_per_family":
        raise JAContractError("config_malformed", "family_vote_policy must be one_per_family", "$.asr.family_vote_policy")
    frontend = config.get("frontend", {})
    for field in ("provider", "commit"):
        if not isinstance(frontend.get(field), str) or not frontend[field]:
            raise JAContractError("config_malformed", f"frontend {field} is required", f"$.frontend.{field}")
    frontend_flags = (
        "use_vanilla", "use_tsqyomi", "use_sudachi_kanji_yomi", "predict_nani",
        "use_read_as_pron", "revert_long_vowels", "revert_yotsugana", "run_marine",
        "reject_unbound_spans",
    )
    for flag in frontend_flags:
        if flag not in frontend:
            raise JAContractError("config_malformed", "frontend option must be explicit", f"$.frontend.{flag}")
        if type(frontend[flag]) is not bool:
            raise JAContractError("config_malformed", "frontend option must be boolean", f"$.frontend.{flag}")
    if frontend.get("reject_unbound_spans") is not True:
        raise JAContractError("config_malformed", "reject_unbound_spans must be true", "$.frontend.reject_unbound_spans")
    if frontend.get("normalize_mode") not in {"None", "NFC", "NFKC"}:
        raise JAContractError("config_malformed", "normalize_mode must be explicit", "$.frontend.normalize_mode")
    mixed = config.get("mixed", {})
    julius = config.get("julius_diagnostic", {})
    if type(mixed.get("enabled", False)) is not bool or type(julius.get("enabled", False)) is not bool:
        raise JAContractError("config_malformed", "enabled must be boolean")
    if "allow_phone_clipping" in mixed and type(mixed["allow_phone_clipping"]) is not bool:
        raise JAContractError("config_malformed", "allow_phone_clipping must be boolean", "$.mixed.allow_phone_clipping")
    if mixed.get("allow_phone_clipping", False) is not False:
        raise JAContractError("config_malformed", "phone clipping is forbidden", "$.mixed.allow_phone_clipping")
    if mixed.get("enabled", False) and mixed.get("anchor_strategy") != "dual_full_utterance_v1":
        raise JAContractError("config_malformed", "unsupported mixed anchor strategy", "$.mixed.anchor_strategy")
    if type(julius.get("write_back", False)) is not bool:
        raise JAContractError("config_malformed", "write_back must be boolean", "$.julius_diagnostic.write_back")
    if julius.get("enabled", False) and julius.get("write_back", False):
        raise JAContractError("julius_writeback_forbidden", "Julius write_back is forbidden", "$.julius_diagnostic.write_back")
    # config_path is deliberately only used to reject a symlink config; source
    # manifests and source WAVs may be outside the workspace.
    if config_path is not None:
        reject_symlink(config_path, code="config_path_invalid")
    return dict(config)


def artifact_record(path: os.PathLike[str] | str) -> dict[str, Any]:
    candidate = reject_symlink(path, code="receipt_output_symlink")
    row: dict[str, Any] = {"path": str(candidate), "exists": candidate.exists()}
    if candidate.is_file():
        row.update({"size": candidate.stat().st_size, "sha256": sha256_file(candidate)})
    else:
        row.update({"size": 0, "sha256": None})
    return row


def make_receipt(
    *,
    stage: str,
    status: str,
    inputs: Mapping[str, Any] | None = None,
    outputs: Sequence[os.PathLike[str] | str] | None = None,
    params: Mapping[str, Any] | None = None,
    tools: Sequence[str] | None = None,
    commands: Sequence[str] | None = None,
    errors: Sequence[Mapping[str, Any] | str] | None = None,
) -> dict[str, Any]:
    if stage not in STAGE_NAMES:
        raise JAContractError("receipt_invalid", f"unknown stage {stage!r}", "$.stage")
    if status not in {"PENDING", "RUNNING", "PARTIAL", "COMPLETE", "REJECTED", "BLOCKED"}:
        raise JAContractError("receipt_invalid", f"unknown status {status!r}", "$.status")
    output_rows = [artifact_record(output) for output in outputs or ()]
    input_payload = dict(inputs or {})
    if "artifacts" not in input_payload:
        input_payload["artifacts"] = []
    return {
        "schema": "ja-pipeline-receipt-v1",
        "stage": stage,
        "status": status,
        "inputs": input_payload,
        "outputs": output_rows,
        "params": dict(params or {}),
        "tools": list(tools or []),
        "commands": list(commands or []),
        "errors": [e if isinstance(e, Mapping) else {"message": str(e)} for e in (errors or ())],
    }


def validate_receipt(receipt: Mapping[str, Any], *, workspace: os.PathLike[str] | str | None = None) -> Mapping[str, Any]:
    validate_schema(receipt, "ja-pipeline-receipt-v1")
    for field in ("stage", "status", "inputs", "outputs", "params", "tools", "commands", "errors"):
        if field not in receipt:
            raise JAContractError("receipt_invalid", f"missing field {field}", f"$.{field}")
    if receipt.get("stage") not in STAGE_NAMES:
        raise JAContractError("receipt_invalid", "receipt stage is unknown", "$.stage")
    if receipt.get("status") not in {"PENDING", "RUNNING", "PARTIAL", "COMPLETE", "REJECTED", "BLOCKED"}:
        raise JAContractError("receipt_invalid", "receipt status is unknown", "$.status")
    for index, output in enumerate(receipt["outputs"]):
        if not isinstance(output, Mapping) or not output.get("path"):
            raise JAContractError("receipt_invalid", "invalid output row", f"$.outputs[{index}]")
        if output.get("exists") is not True or not isinstance(output.get("sha256"), str) or len(output["sha256"]) != 64:
            raise JAContractError("receipt_output_missing", "receipt output must exist with a SHA-256", f"$.outputs[{index}]")
        path = reject_symlink(output["path"], code="receipt_output_symlink")
        if workspace is not None:
            ensure_output_path(path, workspace)
        if output.get("exists"):
            if not path.is_file():
                raise JAContractError("receipt_output_missing", "output is missing", str(path))
            if output.get("size") != path.stat().st_size or output.get("sha256") != sha256_file(path):
                raise JAContractError("receipt_hash_mismatch", "output receipt does not match file", str(path))
    output_paths = [str(row["path"]) for row in receipt["outputs"]]
    if len(output_paths) != len(set(output_paths)):
        raise JAContractError("receipt_invalid", "duplicate receipt output path", "$.outputs")
    artifacts = receipt["inputs"].get("artifacts", []) if isinstance(receipt["inputs"], Mapping) else []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or not artifact.get("path"):
            raise JAContractError("receipt_invalid", "invalid input artifact", f"$.inputs.artifacts[{index}]")
        if artifact.get("exists") is not True or not isinstance(artifact.get("sha256"), str) or len(artifact["sha256"]) != 64:
            raise JAContractError("receipt_input_missing", "receipt input must exist with a SHA-256", f"$.inputs.artifacts[{index}]")
        path = reject_symlink(artifact["path"], code="receipt_output_symlink")
        if not path.is_file():
            raise JAContractError("receipt_input_missing", "input artifact is missing", str(path))
        if artifact.get("size") != path.stat().st_size or artifact.get("sha256") != sha256_file(path):
            raise JAContractError("receipt_hash_mismatch", "input receipt does not match file", str(path))
    return receipt


def cache_key(identity: Mapping[str, Any]) -> str:
    return stable_digest(identity)


def validate_cache_identity(identity: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if cache_key(identity) != cache_key(expected):
        raise JAContractError("resume_identity_drift", "cache identity changed")


def list_relative_files(root: os.PathLike[str] | str) -> list[str]:
    base = reject_symlink(root)
    if not base.exists():
        return []
    files: list[str] = []
    for path in base.rglob("*"):
        if path.is_symlink():
            raise JAContractError("resume_extra_file", "symlink in workspace", str(path))
        if path.is_file():
            files.append(path.relative_to(base).as_posix())
    return sorted(files)


@contextlib.contextmanager
def exclusive_lock(path: os.PathLike[str] | str) -> Iterator[Path]:
    lock = _absolute(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    owned = False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        owned = True
    except FileExistsError as exc:
        # Dispatch may enter the stage loop through a lock held by its outer
        # run context.  Re-entry is safe only for this process; another PID
        # remains a hard failure and requires explicit stale-lock recovery.
        try:
            existing = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if not isinstance(existing, Mapping) or existing.get("pid") != os.getpid():
            raise JAContractError("lock_busy", "another stage owns the lock", str(lock)) from exc
    try:
        if owned:
            os.write(fd, canonical_json({"pid": os.getpid(), "created": time.time()}))
            os.close(fd)
        yield lock
    finally:
        if owned:
            with contextlib.suppress(FileNotFoundError):
                lock.unlink()


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    receipt_path: str | None = None


def verify_output_set(root: os.PathLike[str] | str, expected: Sequence[str]) -> None:
    actual = set(list_relative_files(root))
    wanted = set(expected)
    if actual != wanted:
        raise JAContractError("resume_extra_file", f"expected={sorted(wanted)!r}, actual={sorted(actual)!r}")


__all__ = [
    "ALIAS_PREFIXES", "ERROR_CODES", "PRODUCTION_STAGES", "SCHEMA_REQUIRED_FIELDS", "STAGE_NAMES", "SCHEMAS", "JAContractError",
    "StageResult", "atomic_write_bytes", "atomic_write_json", "cache_key", "canonical_json",
    "artifact_record", "ensure_output_path", "exclusive_lock", "list_relative_files", "load_json", "make_receipt",
    "reject_symlink", "sha256_bytes", "sha256_file", "stable_digest", "validate_cache_identity",
    "validate_alias_rows", "validate_config", "validate_exact_partition", "validate_manifest", "validate_occurrence_alias", "validate_receipt",
    "validate_record", "validate_schema", "verify_output_set", "make_occurrence_alias", "make_record",
]
