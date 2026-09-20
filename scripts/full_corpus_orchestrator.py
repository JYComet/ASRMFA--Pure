"""Resumable Qwen3-only planning, preflight, canary and serial runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_RUN_PIPELINE = Path(__file__).resolve().parent / "run_pipeline.py"

try:
    from scripts.full_corpus_inventory import scan_sources, write_frozen_inventory
    from scripts.speaker_namespace import publication_speaker
except ModuleNotFoundError:
    from full_corpus_inventory import scan_sources, write_frozen_inventory
    from speaker_namespace import publication_speaker


@dataclass(frozen=True)
class ChunkLimits:
    max_files: int
    max_hours: float
    max_bytes: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text_mode: str
    items: tuple
    total_seconds: float
    total_bytes: int


class _PrefetchTask:
    """One daemon staging task that never writes shared run status."""

    def __init__(self, chunk_id: str, target, *args, **kwargs):
        self.chunk_id = chunk_id
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name=f"prefetch-{chunk_id}", daemon=True)

    def _run(self):
        try:
            self._result = self._target(*self._args, **self._kwargs)
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()

    def start(self):
        self._thread.start()
        return self

    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError(f"prefetch is still running: {self.chunk_id}")
        if self._error is not None:
            raise self._error
        return self._result


def _prefetch_can_promote(task: _PrefetchTask, producer, downstream) -> bool:
    return task.done() and producer is None and downstream is None


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stems_digest(stems) -> str:
    return hashlib.sha256(json.dumps(sorted(stems), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _item_from_json(row):
    from types import SimpleNamespace
    data = dict(row)
    for key in ("source_path", "reference_path"):
        if data.get(key):
            data[key] = Path(data[key])
    return SimpleNamespace(**data)


def _durable_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        fd = os.open(path.parent, os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


def _event(root: Path, event: dict) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _set_status(root: Path, payload: dict) -> None:
    _durable_json(Path(root) / "status.json", payload)


def _bound_digest(root: Path, chunk: Chunk, config: dict) -> dict:
    return {"chunk_id": chunk.chunk_id,
            "inventory_sha256": (config.get("_inventory_sha256")
                                 or _sha256(Path(root) / "frozen_inventory.json")),
            "chunk_stems_digest": _stems_digest([_get(item, "run_stem") for item in chunk.items]),
            "chunk_stems": sorted(_get(item, "run_stem") for item in chunk.items),
            "config_sha256": (config.get("_config_sha256")
                              if "_config_sha256" in config
                              else (_sha256(Path(config["config_path"]))
                                    if config.get("config_path")
                                    and Path(config["config_path"]).is_file() else None))}


def _gamesl_bound(root: Path, chunk: Chunk, config: dict) -> dict:
    """Bind a GAMESL receipt to only the padded items in its frozen chunk."""
    bound = _bound_digest(root, chunk, config)
    stems = sorted(_get(item, "run_stem") for item in chunk.items
                   if _get(item, "needs_gamesl_padding", False))
    bound["chunk_stems"] = stems
    bound["chunk_stems_digest"] = _stems_digest(stems)
    return bound


def _load_valid_gamesl_receipt(path: Path, bound: dict):
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = ("chunk_id", "inventory_sha256", "config_sha256")
        if any(receipt.get(key) != bound.get(key) for key in stable_keys):
            return None
        rows = receipt.get("replacements", [])
        by_stem = {row.get("stem"): row for row in rows}
        wanted = list(bound.get("chunk_stems", []))
        if len(by_stem) != len(rows) or any(stem not in by_stem for stem in wanted):
            return None
        selected = [by_stem[stem] for stem in wanted]
        for row in selected:
            target = Path(row["target"])
            if not target.is_file() or row.get("new_sha256") != _sha256(target):
                return None
        if all(receipt.get(key) == value for key, value in bound.items()):
            return receipt
        projected = dict(receipt)
        projected.update(bound)
        projected["replacements"] = selected
        projected["published_count"] = len(selected)
        projected["replaced_count"] = sum(
            row.get("old_sha256") is not None for row in selected)
        return projected
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _load_bound_gamesl_receipt_fast(path: Path, bound: dict):
    """Validate an already-published GAMESL receipt without rereading its WAVs.

    Prepared MFA chunks published and hashed these immutable targets during the
    prepare phase.  The hot path only needs to prove that this receipt belongs
    to the same frozen chunk and contains exactly its padded stems.
    """
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = ("chunk_id", "inventory_sha256", "config_sha256")
        if any(receipt.get(key) != bound.get(key) for key in stable_keys):
            return None
        rows = receipt.get("replacements", [])
        by_stem = {row.get("stem"): row for row in rows}
        wanted = list(bound.get("chunk_stems", []))
        if len(by_stem) != len(rows) or any(stem not in by_stem for stem in wanted):
            return None
        selected = [by_stem[stem] for stem in wanted]
        if all(receipt.get(key) == value for key, value in bound.items()):
            return receipt
        projected = dict(receipt)
        projected.update(bound)
        projected["replacements"] = selected
        projected["published_count"] = len(selected)
        projected["replaced_count"] = sum(
            row.get("old_sha256") is not None for row in selected)
        return projected
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None


def _load_valid_output_receipt(path: Path, bound: dict):
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if any(receipt.get(key) != value for key, value in bound.items()):
            return None
        rows = receipt.get("replacements", [])
        if bound.get("chunk_stems") and len(rows) != len(bound["chunk_stems"]):
            return None
        for row in rows:
            target = Path(row["target"])
            if not target.is_file() or row.get("new_sha256") != _sha256(target):
                return None
        if bound.get("chunk_stems") and sorted(row.get("stem") for row in rows) != bound["chunk_stems"]:
            return None
        return receipt
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _load_bound_output_receipt_fast(path: Path, bound: dict):
    """Validate a durable completed-publication receipt without rehashing NAS."""
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if any(receipt.get(key) != value for key, value in bound.items()):
            return None
        rows = receipt.get("replacements", [])
        stems = [row.get("stem") for row in rows]
        wanted = list(bound.get("chunk_stems", []))
        if len(stems) != len(set(stems)) or sorted(stems) != wanted:
            return None
        return receipt
    except (OSError, KeyError, json.JSONDecodeError, TypeError):
        return None


def build_chunks(items, limits: ChunkLimits) -> list[Chunk]:
    if limits.max_files <= 0 or limits.max_hours <= 0 or limits.max_bytes <= 0:
        raise ValueError("chunk limits must be positive")
    chunks = []
    for mode in ("reference", "fallback"):
        current, seconds, total = [], 0.0, 0
        for item in sorted((x for x in items if _get(x, "text_mode") == mode), key=lambda x: _get(x, "run_stem")):
            duration = float(_get(item, "duration_seconds", 0.0) or 0.0)
            size = int(_get(item, "audio_bytes", 0) or 0)
            if current and (len(current) >= limits.max_files
                            or seconds + duration > limits.max_hours * 3600
                            or total + size > limits.max_bytes):
                chunks.append(_make_chunk(mode, current, seconds, total))
                current, seconds, total = [], 0.0, 0
            current.append(item)
            seconds += duration
            total += size
        if current:
            chunks.append(_make_chunk(mode, current, seconds, total))
    return chunks


def _make_chunk(mode, items, seconds, total):
    digest = hashlib.sha256("\n".join(_get(x, "run_stem") for x in items).encode()).hexdigest()[:16]
    return Chunk(f"{mode}-{digest}", mode, tuple(items), seconds, total)


def _strip_forbidden(value):
    if isinstance(value, dict):
        return {key: _strip_forbidden(val) for key, val in value.items()
                if "nvasr" not in str(key).lower()
                and "nvasr" not in str(val).lower()}
    if isinstance(value, list):
        return [_strip_forbidden(val) for val in value]
    if isinstance(value, str) and "nvasr" in value.lower():
        return None
    return copy.deepcopy(value)


def materialize_pipeline_config(chunk: Chunk, task_config: dict, path: Path) -> dict:
    source = task_config.get("pipeline", task_config)
    # The task YAML is intentionally richer than run_pipeline's schema.  Do
    # not leak orchestration/selection keys into a child pipeline config.
    allowed = {
        "mode", "reference_mode", "workspace", "data_dir", "audio_dir",
        "pinyin_dir", "aligned_dir", "output_dir", "filtered_dir",
        "validate_dir", "temp_dir", "models_dir", "mfa_dict", "acoustic_model",
        "mfa", "mfa_en", "ctc_prealign", "ctc_adjust", "normalize",
        "normalize_ria", "normalize_en", "normalize_punct", "pad_silence",
        "postprocess", "audio_axis", "output_staging", "keep_16k_audio",
        "require_fresh_workspace", "streaming", "pipelined", "batch",
        "use_cache", "nvme_cache", "auto_cache", "strict_ctc_ready",
        "python_path", "trim", "prepare", "resample", "output_spec",
    }
    cfg = _strip_forbidden({key: source[key] for key in allowed if key in source})
    if not isinstance(cfg, dict):
        cfg = {}
    run_dir = Path(path).parent / chunk.chunk_id
    cfg.update({
        "mode": "full",
        "reference_mode": "authority" if chunk.text_mode == "reference" else "fallback",
        "data_dir": str(Path(source.get("chunk_input", task_config.get("chunk_input", run_dir / "input")))),
        "workspace": str(run_dir / "workspace"),
        "output_dir": str(run_dir / "output"),
        "output_staging": False,
    })
    trim = cfg.setdefault("trim", {})
    trim["normalize_edges"] = False
    # Staging has already established the authoritative audio axis (including
    # exact 0.5 s game-audio edges).  Keep every internal pause so Qwen, MFA,
    # the final TextGrid, and the published WAV all share that same axis.
    trim["max_silence_sec"] = 1_000_000_000.0
    qwen = source.get("qwen", {}) or {}
    ctc = cfg.get("ctc_prealign", {}) or {}
    ctc.update({
        "enabled": True,
        "provider": "qwen3_hf",
        "all_gpus": True,
        "python": qwen.get("python", source.get("python_path", "python")),
        "model_path": qwen.get("asr_model", ctc.get("model_path", "")),
        "forced_aligner_model_path": qwen.get("forced_aligner_model", ctc.get("forced_aligner_model_path", "")),
        "dtype": qwen.get("dtype", ctc.get("dtype", "bfloat16")),
        "language": qwen.get("language", ctc.get("language", "Chinese")),
        "batch_size": qwen.get("batch_size", ctc.get("batch_size", 4)),
        "context": qwen.get("context", ctc.get("context", "")),
        "max_new_tokens": qwen.get("max_new_tokens", ctc.get("max_new_tokens", 2048)),
        "limit": qwen.get("limit", ctc.get("limit", 0)),
        "timeout": qwen.get("timeout", ctc.get("timeout", 86400)),
        "allow_item_failures": qwen.get(
            "allow_item_failures", ctc.get("allow_item_failures", True)),
        "nvv_enabled": False,
        "reference_nvv_enabled": False,
    })
    cfg["ctc_prealign"] = ctc
    cfg.setdefault("ctc_adjust", {"enabled": True, "limit": 0})
    # The MFA interpreter belongs to the scheduler.  run_pipeline uses the
    # top-level python_path for its own subprocess environment and does not
    # accept an mfa.python task key.
    cfg.setdefault("python_path", str(source.get("python_path", source.get("mfa", {}).get("python", "python"))))
    if isinstance(cfg.get("mfa"), dict):
        cfg["mfa"].pop("python", None)
    cfg = _strip_forbidden(cfg)
    if "nvasr" in json.dumps(cfg, ensure_ascii=False).lower():
        raise ValueError("forbidden NVASR identity in resolved config")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg


def _load_frozen_items(root: Path):
    path = Path(root) / "frozen_inventory.json"
    if not path.is_file():
        raise RuntimeError("frozen inventory is required")
    return [_item_from_json(row) for row in json.loads(path.read_text(encoding="utf-8")).get("items", [])]


def _select_canary(items, spec):
    candidates = [item for item in items
                  if _get(item, "source_id") == spec.get("source_id")
                  and _get(item, "speaker") == spec.get("speaker")]
    if spec.get("relative_path"):
        candidates = [item for item in candidates
                      if (_get(item, "source_relative_path") == spec["relative_path"]
                          or "/".join(Path(_get(item, "source_relative_path")).parts[1:]) == spec["relative_path"])]
    elif spec.get("relative_stem"):
        candidates = [item for item in candidates
                      if Path(_get(item, "source_relative_path")).stem == spec["relative_stem"]]
    if len(candidates) != 1:
        raise ValueError(f"canary identity is not unique: {spec}")
    return candidates[0]


def _partition_stems(value):
    return set(value.get("stems", [])) if isinstance(value, dict) else set()


def _validate_pipeline_receipt(path: Path, stems, *, reference_mode=None, producer=False):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid pipeline accounting receipt: {path}") from exc
    if payload.get("schema") != "pipeline-run-receipt-v2":
        raise RuntimeError(f"unsupported pipeline accounting schema: {path}")
    if payload.get("run_health") != "healthy" or payload.get("silent_loss") != 0:
        raise RuntimeError(f"unhealthy pipeline accounting receipt: {path}")
    expected = set(stems)
    eligible = _partition_stems(payload.get("eligible"))
    output = _partition_stems(payload.get("output"))
    filtered = _partition_stems(payload.get("filtered"))
    if eligible != expected or output & filtered or output | filtered != expected:
        raise RuntimeError(f"pipeline accounting denominator mismatch: {path}")
    for name, values in (("source", payload.get("source", {}).get("stems", [])),
                         ("eligible", payload.get("eligible", {}).get("stems", [])),
                         ("output", payload.get("output", {}).get("stems", [])),
                         ("filtered", payload.get("filtered", {}).get("stems", []))):
        part = payload.get(name, {})
        if part.get("count") != len(values) or part.get("stems_digest") != _stems_digest(values):
            raise RuntimeError(f"pipeline {name} partition count/digest mismatch: {path}")
    derived = payload.get("derived", {})
    expected_derived = (("eligible_count", eligible), ("output_count", output), ("filtered_count", filtered))
    if derived and any(derived.get(key) != len(value) for key, value in expected_derived if key in derived):
        raise RuntimeError(f"pipeline derived counts mismatch: {path}")
    extra = payload.get("extra", {})
    processed = set(extra.get("processed_stems", []))
    if processed and processed != expected:
        raise RuntimeError(f"pipeline processed stem mismatch: {path}")
    expected_mode = {"reference": "authority", "fallback": "fallback"}.get(reference_mode, reference_mode)
    fingerprint_mode = payload.get("fingerprints", {}).get("inputs", {}).get("reference_mode")
    actual_mode = extra.get("reference_mode") or fingerprint_mode
    if expected_mode and actual_mode not in {expected_mode, reference_mode, "auto"}:
        raise RuntimeError(f"pipeline reference mode mismatch: {path}")
    if producer:
        route = payload.get("route", [])
        if "qwen3_hf" not in route or "qwen3_forced_aligner" not in route:
            raise RuntimeError(f"Qwen producer route missing: {path}")
        for key in ("identity_digest", "forced_aligner_model_tree_digest"):
            if not extra.get(key):
                raise RuntimeError(f"pipeline evidence digest missing: {key}")
    return payload


def _qwen_evidence(workspace: Path, stems, reference_mode=None):
    raw = Path(workspace) / "ctc_pretg"
    manifest = raw / "manifest.json"
    identity = raw / ".qwen3_hf_identity.json"
    producer_receipt = raw / ".pipeline_run_receipt_v2.json"
    if not manifest.is_file() or not identity.is_file() or not producer_receipt.is_file():
        raise RuntimeError("sealed Qwen manifest, identity, and accounting evidence are required")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("Qwen manifest must be a sealed list")
    present = {row.get("stem") or Path(row.get("audio", "")).stem for row in rows if isinstance(row, dict)}
    for row in rows:
        if not row.get("_words") or not row.get("duration_s") or not (row.get("text_normalized") or row.get("text_asr") or row.get("text_original")):
            raise RuntimeError("Qwen manifest row has no sealed timestamps/text")
    identity_payload = json.loads(identity.read_text(encoding="utf-8"))
    if identity_payload.get("schema") != "qwen3-hf-prealign-identity-v1" or identity_payload.get("provider") != "qwen3_hf":
        raise RuntimeError("sealed producer is not Qwen3")
    if "nvasr" in json.dumps(identity_payload, ensure_ascii=False).lower():
        raise RuntimeError("forbidden producer identity")
    models = identity_payload.get("models", {})
    if not models.get("forced_aligner_tree_digest") or not (models.get("asr_tree_digest") or models.get("asr") == "not_required"):
        raise RuntimeError("Qwen model tree digests are missing")
    for key in ("inputs", "references_digest", "output_digest", "identity_digest"):
        if not identity_payload.get(key):
            raise RuntimeError(f"Qwen identity field is missing: {key}")
    if identity_payload.get("identity_digest") != json.loads(producer_receipt.read_text(encoding="utf-8")).get("extra", {}).get("identity_digest"):
        raise RuntimeError("Qwen identity digest does not match producer receipt")
    producer_accounting = _validate_pipeline_receipt(
        producer_receipt, stems, reference_mode=reference_mode, producer=True)
    producer_output = sorted(_partition_stems(producer_accounting.get("output")))
    producer_filtered = sorted(_partition_stems(producer_accounting.get("filtered")))
    if present != set(producer_output):
        raise RuntimeError("Qwen manifest output membership mismatch")
    receipts = sorted(raw.glob("*.qwen3*.json")) or [manifest]
    paths = [manifest, identity, producer_receipt, *receipts]
    return {
        "qwen_manifest": str(manifest),
        "qwen_identity": str(identity),
        "raw_timestamps": [str(path) for path in receipts],
        "producer_accounting": str(producer_receipt),
        "output_stems": producer_output,
        "filtered_stems": producer_filtered,
        "sha256": {str(path): _sha256(path) for path in paths},
    }


def _result_roots(workspace: Path, stems=None, reference_mode=None):
    candidates = sorted(Path(workspace).glob("**/.pipeline_run_receipt_v2.json"), key=lambda p: p.stat().st_mtime)
    for receipt in reversed(candidates):
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if stems is not None:
                _validate_pipeline_receipt(receipt, stems, reference_mode=reference_mode)
            paths = payload.get("paths", {})
            output = Path(paths.get("output", receipt.parent))
            filtered = Path(paths.get("filtered", receipt.parent.parent / "filtered"))
            if output.is_dir() and filtered.is_dir():
                return output, filtered
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"strict output receipt is missing: {workspace}")


def _final_evidence(workspace: Path, output: Path, qwen: dict):
    accounting = output / ".pipeline_run_receipt_v2.json"
    report = output / "postprocess_report.jsonl"
    strict = output / "strict_ok_manifest.json"
    if not accounting.is_file() or not report.is_file():
        raise RuntimeError("final accounting and postprocess evidence are required")
    evidence = dict(qwen)
    evidence.update({
        "accounting": str(accounting),
        "strict_manifest": str(strict) if strict.is_file() else str(accounting),
        "postprocess_report": str(report),
        "punctuation_evidence": str(report),
    })
    evidence.setdefault("sha256", {}).update({str(path): _sha256(path)
                                               for path in (accounting, report, strict) if path.is_file()})
    return evidence


def _launch(root, command, label):
    log = Path(root) / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    _event(root, {"event": "started", "stage": label, "pid": process.pid,
                  "command": command, "log": str(log)})
    return process, log


def _downstream_command(python: str, config_path: Path, *,
                        refresh_ctc_work: bool = False,
                        skip_to: str = "normalize_punct",
                        stop_after: str | None = None,
                        validate: bool = True) -> list[str]:
    command = [python, str(_RUN_PIPELINE), "--config", str(config_path),
               "--python", python, "--skip-to", skip_to]
    if stop_after:
        command.extend(("--stop-after", stop_after))
    if validate:
        command.append("--validate")
    if refresh_ctc_work:
        command.append("--refresh-ctc-work")
    return command


def _prepare_command(python: str, config_path: Path, *, qwen_sealed: bool) -> list[str]:
    if qwen_sealed:
        return _downstream_command(
            python, config_path, refresh_ctc_work=True,
            skip_to="normalize_punct", stop_after="adjust")
    return [python, str(_RUN_PIPELINE), "--config", str(config_path),
            "--python", python, "--stop-after", "adjust", "--validate"]


def _prepared_chunk(chunk: Chunk, record: dict) -> tuple[Chunk, list[dict]]:
    ready = list(record.get("ready_stems", []))
    failures = list(record.get("stage_failures", []))
    failure_stems = [row.get("stem") for row in failures]
    expected = {_get(item, "run_stem") for item in chunk.items}
    if (len(ready) != len(set(ready)) or len(failure_stems) != len(set(failure_stems))
            or set(ready) & set(failure_stems)
            or set(ready) | set(failure_stems) != expected):
        raise RuntimeError(f"prepared chunk partition mismatch: {chunk.chunk_id}")
    return _subset_chunk(
        chunk, [item for item in chunk.items if _get(item, "run_stem") in set(ready)]), failures


def _mark_terminal(root, status, chunk, state, reason=None):
    status["terminal"][chunk.chunk_id] = state
    if state != "complete":
        for item in chunk.items:
            stem = _get(item, "run_stem")
            status["items"][stem] = {"state": state, **({"reason": reason} if reason else {})}
            _event(root, {"event": "item_terminal", "chunk": chunk.chunk_id, "stem": stem,
                          "state": state, "reason": reason})
    elif any(_get(item, "run_stem") not in status["items"] for item in chunk.items):
        raise RuntimeError("chunk marked complete without per-item terminal records")
    _event(root, {"event": "chunk_terminal", "chunk": chunk.chunk_id, "state": state})
    _set_status(root, status)


def _subset_chunk(chunk: Chunk, items) -> Chunk:
    items = tuple(items)
    return Chunk(
        chunk.chunk_id,
        chunk.text_mode,
        items,
        sum(float(_get(item, "duration_seconds", 0.0) or 0.0) for item in items),
        sum(int(_get(item, "audio_bytes", 0) or 0) for item in items),
    )


def _publication_bound_chunk(chunk: Chunk, publication) -> Chunk:
    """Return the exact public-output denominator for a sealed publication."""
    public_stems = set(publication.accepted_stems) | set(publication.filtered_stems)
    return _subset_chunk(
        chunk, [item for item in chunk.items
                if _get(item, "run_stem") in public_stems])


def _publication_rows(current: dict, frozen: dict) -> list[dict]:
    rows = []
    for item, staged in zip(current["pipeline_chunk"].items, current["receipts"]):
        row = dict(frozen[_get(item, "run_stem")].__dict__)
        row["pipeline_wav"] = staged.pipeline_wav
        rows.append(row)
    return rows


def _validate_completed_chunk(current: dict, frozen: dict, validate_chunk_result):
    """Validate an existing final receipt, including producer-side filters."""
    stems = [_get(item, "run_stem") for item in current["pipeline_chunk"].items]
    output, filtered = _result_roots(
        current["workspace"], stems, current["pipeline_chunk"].text_mode)
    evidence = _final_evidence(current["workspace"], output, current["qwen"])
    rows = _publication_rows(current, frozen)
    producer_failures = {
        stem: "qwen_producer_filtered"
        for stem in current["qwen"].get("filtered_stems", [])
    }
    return validate_chunk_result({
        "chunk_id": current["chunk"].chunk_id,
        "items": rows,
        "output_root": output,
        "filtered_root": filtered,
        "evidence": evidence,
        "failed": producer_failures,
    }, {"items": rows})


def _publish_completed_chunk(current: dict, root: Path, config: dict,
                             frozen: dict, validate_chunk_result,
                             publish_gamesl, publish_chunk) -> dict:
    """Run final validation and NAS publication outside the compute lane."""
    publication = current.get("recovered_publication") or _validate_completed_chunk(
        current, frozen, validate_chunk_result)
    gamesl_receipt = None
    gamesl_resumed = False
    if config.get("gamesl_root"):
        if current.get("prepared_for_mfa", False):
            gamesl_receipt = _load_bound_gamesl_receipt_fast(
                current["gamesl_receipt_path"], current["gamesl_bound"])
            if gamesl_receipt is None:
                raise RuntimeError(
                    "prepared GAMESL receipt binding is missing or invalid: "
                    f"{current['chunk'].chunk_id}")
            gamesl_resumed = True
        else:
            gamesl_receipt = _load_valid_gamesl_receipt(
                current["gamesl_receipt_path"], current["gamesl_bound"])
            gamesl_resumed = gamesl_receipt is not None
            if gamesl_receipt is None:
                gamesl_receipt = publish_gamesl(
                    current["receipts"], Path(config["gamesl_root"]),
                    root / "rollback" / "gamesl")
                gamesl_receipt.update(current["gamesl_bound"])
                _durable_json(current["gamesl_receipt_path"], gamesl_receipt)
    publication_receipt = publish_chunk(
        publication, Path(config["output_root"]), root / "rollback" / "output")
    publication_receipt.update(_bound_digest(
        root, _publication_bound_chunk(current["chunk"], publication), config))
    _durable_json(current["output_receipt_path"], publication_receipt)
    return {
        "publication": publication,
        "publication_receipt": publication_receipt,
        "gamesl_receipt": gamesl_receipt,
        "gamesl_resumed": gamesl_resumed,
    }


def _load_frozen_stage_receipt(item, input_dir: Path, *, verify_content: bool = True,
                               resolution_digest: str | None = None):
    """Reuse a frozen, NVMe-local stage artifact without rereading its NAS source.

    The reuse is only valid for the resolution table that produced the artifact.
    When a repair supplies its own table, a receipt bound to a different one is
    rejected here so the item is re-normalized from source instead of silently
    keeping the stale text.
    """
    input_dir = Path(input_dir)
    stem = _get(item, "run_stem")
    receipt_path = input_dir / f"{stem}.stage_receipt.json"
    pipeline = input_dir / f"{stem}.wav"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    try:
        try:
            from scripts.full_corpus_stage import StageReceipt
        except ModuleNotFoundError:
            from full_corpus_stage import StageReceipt
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (saved.get("run_stem") != stem
                or saved.get("source_sha256") != _get(item, "audio_sha256")
                or Path(saved.get("pipeline_wav", "")).resolve() != pipeline.resolve()
                or not pipeline.is_file() or pipeline.is_symlink()
                or not saved.get("pipeline_wav_sha256")
                or (verify_content
                    and saved.get("pipeline_wav_sha256") != _sha256(pipeline))):
            return None
        gamesl = Path(saved["gamesl_wav"]) if saved.get("gamesl_wav") else None
        if gamesl is not None and (not gamesl.is_file() or gamesl.is_symlink()
                                   or not saved.get("gamesl_wav_sha256")
                                   or (verify_content and saved.get("gamesl_wav_sha256")
                                       != _sha256(gamesl))):
            return None
        if _get(item, "needs_gamesl_padding", False) != (gamesl is not None):
            return None
        if resolution_digest is not None and (
                saved.get("macro_resolution_digest") != resolution_digest):
            return None
        text_path = Path(saved["text_path"]) if saved.get("text_path") else None
        if _get(item, "reference_path") is not None:
            expected_text = input_dir / f"{stem}.txt"
            if (text_path is None or text_path.resolve() != expected_text.resolve()
                    or not text_path.is_file() or text_path.is_symlink()
                    or not saved.get("text_sha256")
                    or (verify_content and saved.get("text_sha256") != _sha256(text_path))
                    or not saved.get("normalized_text")):
                return None
        elif text_path is not None or saved.get("text_sha256") or saved.get("normalized_text"):
            return None
        return StageReceipt(
            stem, _get(item, "source_id"), _get(item, "game"),
            _get(item, "speaker"), pipeline, gamesl,
            saved["pipeline_wav_sha256"], saved.get("gamesl_wav_sha256"),
            text_path, saved.get("text_sha256"), saved.get("normalized_text"),
            saved["source_sha256"], saved.get("macro_resolution_digest"))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _collect_stage_chunk_items(chunk: Chunk, frozen: dict, base: Path, stage_fn, *,
                               verify_content: bool = True, resolutions=None,
                               resolution_digest: str | None = None):
    """Stage a chunk privately; the caller later merges terminal status."""
    ready_items, receipts, failure_rows = [], [], []
    input_dir = Path(base) / "input"
    for item in chunk.items:
        stem = _get(item, "run_stem")
        try:
            receipt = _load_frozen_stage_receipt(
                frozen[stem], input_dir, verify_content=verify_content,
                resolution_digest=resolution_digest)
            if receipt is None:
                receipt = stage_fn(frozen[stem], input_dir, Path(base) / "gamesl_stage",
                                   resolutions=resolutions,
                                   resolution_digest=resolution_digest)
        except Exception as exc:
            for suffix in (".wav", ".txt", ".stage_receipt.json"):
                (input_dir / f"{stem}{suffix}").unlink(missing_ok=True)
            failure_rows.append({
                "stem": stem,
                "state": "pipeline_failure",
                "reason": f"stage_failure:{type(exc).__name__}:{exc}",
            })
            continue
        ready_items.append(item)
        receipts.append(receipt)
    return _subset_chunk(chunk, ready_items), receipts, failure_rows


def _apply_stage_result(root: Path, status: dict, chunk: Chunk, result):
    pipeline_chunk, receipts, failure_rows = result
    status["terminal"].pop(chunk.chunk_id, None)
    ready = {_get(item, "run_stem") for item in pipeline_chunk.items}
    for stem in ready:
        status["items"].pop(stem, None)
    for row in failure_rows:
        status["items"][row["stem"]] = {
            "state": row["state"], "reason": row["reason"]}
        _event(root, {"event": "item_terminal", "chunk": chunk.chunk_id, **row})
    _set_status(root, status)
    return pipeline_chunk, receipts, len(failure_rows)


def _stage_chunk_items(root: Path, status: dict, chunk: Chunk, frozen: dict,
                       base: Path, stage_fn):
    """Stage each item independently and account for input failures."""
    return _apply_stage_result(
        root, status, chunk,
        _collect_stage_chunk_items(chunk, frozen, base, stage_fn))


def _validate_canary_receipt(path: Path, root: Path, config: dict, items) -> dict:
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("canary receipt is unreadable") from exc
    if (receipt.get("schema") != "qwen3-0915all-canary-v1" or receipt.get("success") is not True
            or receipt.get("exact_count") != 2 or receipt.get("public_output") is not False
            or receipt.get("five_tier_validated") is not True):
        raise RuntimeError("canary receipt is not the exact successful pair")
    expected_items = [_select_canary(items, spec) for spec in config.get("canaries", [])]
    results = receipt.get("results", [])
    expected_keys = [(item.source_id, item.speaker, item.text_mode, item.run_stem) for item in expected_items]
    actual_keys = [(row.get("source_id"), row.get("speaker"), row.get("text_mode"), row.get("run_stem")) for row in results]
    if len(results) != 2 or actual_keys != expected_keys:
        raise RuntimeError("canary identities or modes are not exact")
    if receipt.get("inventory_sha256") != _sha256(root / "frozen_inventory.json"):
        raise RuntimeError("canary inventory binding mismatch")
    if config.get("config_path") and receipt.get("config_sha256") != _sha256(Path(config["config_path"])):
        raise RuntimeError("canary config binding mismatch")
    try:
        from scripts.full_corpus_publish import _validate_evidence, _validate_grid
    except ModuleNotFoundError:
        from full_corpus_publish import _validate_evidence, _validate_grid
    for item, row in zip(expected_items, results):
        input_wav = Path(row["input_wav"]); output_grid = Path(row["output_textgrid"])
        if not input_wav.is_file() or row.get("input_wav_sha256") != _sha256(input_wav):
            raise RuntimeError(f"canary input artifact mismatch: {item.run_stem}")
        if not output_grid.is_file() or row.get("output_textgrid_sha256") != _sha256(output_grid):
            raise RuntimeError(f"canary output artifact mismatch: {item.run_stem}")
        if output_grid.stem != item.run_stem:
            raise RuntimeError(f"canary output identity mismatch: {item.run_stem}")
        try:
            _validate_grid(output_grid, dict(vars(item), pipeline_wav=input_wav))
        except ValueError as exc:
            raise RuntimeError(f"canary five-tier/axis validation failed: {item.run_stem}") from exc
        if item.needs_gamesl_padding and (not row.get("padded_verification", {}).get("head_ok") or not row.get("padded_verification", {}).get("tail_ok")):
            raise RuntimeError(f"canary padded timeline verification missing: {item.run_stem}")
        gamesl_wav = Path(row["gamesl_wav"]) if row.get("gamesl_wav") else None
        if item.needs_gamesl_padding:
            if (gamesl_wav is None or not gamesl_wav.is_file()
                    or row.get("gamesl_wav_sha256") != _sha256(gamesl_wav)
                    or row.get("gamesl_wav_sha256") != row.get("input_wav_sha256")):
                raise RuntimeError(f"canary GAMESL timeline/hash mismatch: {item.run_stem}")
        elif gamesl_wav is not None or row.get("gamesl_wav_sha256"):
            raise RuntimeError(f"v5 canary unexpectedly has GAMESL output: {item.run_stem}")
        _validate_evidence(row.get("evidence"), [item.run_stem], [dict(item.__dict__, pipeline_wav=input_wav)])
        if item.text_mode == "reference":
            text_path = Path(row["normalized_text_path"]) if row.get("normalized_text_path") else None
            if (not row.get("normalized_text") or text_path is None or not text_path.is_file()
                    or row.get("normalized_text_sha256") != _sha256(text_path)
                    or text_path.read_text(encoding="utf-8") != row.get("normalized_text")):
                raise RuntimeError("authority canary normalized text evidence is missing or changed")
        if item.text_mode == "fallback" and (not row.get("qwen_transcript") or not row.get("qwen_timestamps")):
            raise RuntimeError("fallback canary transcript/timestamp evidence is missing")
        for public_root in (config.get("output_root"), config.get("gamesl_root")):
            if public_root and Path(row["output_textgrid"]).resolve().is_relative_to(Path(public_root).resolve()):
                raise RuntimeError("canary wrote a public output")
            if public_root and gamesl_wav and gamesl_wav.resolve().is_relative_to(Path(public_root).resolve()):
                raise RuntimeError("canary wrote a public GAMESL output")
    return receipt


def _fresh_canary_root(root: Path) -> Path:
    """Return the stable first canary root or a unique private rerun root."""
    root = Path(root)
    first = root / "canaries"
    if not first.exists():
        return first
    candidate = root / f"canaries-rerun-{time.time_ns()}"
    while candidate.exists():
        candidate = root / f"canaries-rerun-{time.time_ns()}"
    return candidate


def run_canary(config: dict) -> int:
    root = Path(config["run_root"])
    items = _load_frozen_items(root)
    canaries = config.get("canaries", [])
    if len(canaries) != 2 or [(x.get("source_id"), x.get("speaker")) for x in canaries] != [("wuwa", "今汐"), ("v5_0707", "LAria")]:
        raise ValueError("exact fixed canary pair is required")
    canary_receipt_path = root / "canary_receipt.json"
    if canary_receipt_path.is_file():
        try:
            _validate_canary_receipt(canary_receipt_path, root, config, items)
            return 0
        except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
    private = _fresh_canary_root(root)
    private.mkdir(parents=True, exist_ok=True)
    try:
        from scripts.full_corpus_stage import stage_item, verify_padded_wav
        from scripts.full_corpus_publish import validate_chunk_result
    except ModuleNotFoundError:
        from full_corpus_stage import stage_item, verify_padded_wav
        from full_corpus_publish import validate_chunk_result
    results = []
    for index, spec in enumerate(canaries):
        item = _select_canary(items, spec)
        if index == 0 and item.text_mode != "reference":
            raise ValueError("Wuthering Waves canary must use authority reference mode")
        if index == 1 and item.text_mode != "fallback":
            raise ValueError("LAria canary must use Qwen fallback mode")
        base = private / f"canary-{index}"
        staged = stage_item(item, base / "input", base / "gamesl_stage")
        chunk = Chunk(f"canary-{index}", item.text_mode, (item,), item.duration_seconds, item.audio_bytes)
        cfg_path = private / f"canary-{index}.yaml"
        cfg = materialize_pipeline_config(chunk, {**config, "chunk_input": str(base / "input")}, cfg_path)
        py = str(config.get("mfa", {}).get("python", "python"))
        pre = _prepare_command(py, cfg_path, qwen_sealed=False)
        down = _downstream_command(py, cfg_path, skip_to="align")
        if config.get("dry_run"):
            results.append({"run_stem": item.run_stem, "source_id": item.source_id, "speaker": item.speaker, "planned": True})
            continue
        process, log = _launch(root, pre, f"canary-{index}-prealign")
        rc = process.wait()
        if rc:
            return rc
        qwen = _qwen_evidence(Path(cfg["workspace"]), [item.run_stem], item.text_mode)
        process, log = _launch(root, down, f"canary-{index}-downstream")
        rc = process.wait()
        if rc:
            return rc
        output, filtered = _result_roots(Path(cfg["workspace"]), [item.run_stem], item.text_mode)
        evidence = _final_evidence(Path(cfg["workspace"]), output, qwen)
        row = dict(item.__dict__)
        row["pipeline_wav"] = staged.pipeline_wav
        publication = validate_chunk_result({"chunk_id": chunk.chunk_id, "items": [row],
                                             "output_root": output, "filtered_root": filtered,
                                             "evidence": evidence}, {"items": [row]})
        if len(publication.accepted_stems) != 1 or publication.filtered_stems:
            raise RuntimeError("canary must be accepted")
        manifest_rows = json.loads(Path(evidence["qwen_manifest"]).read_text(encoding="utf-8"))
        manifest_row = next(row for row in manifest_rows if (row.get("stem") or Path(row.get("audio", "")).stem) == item.run_stem)
        results.append({"run_stem": item.run_stem, "source_id": item.source_id,
                        "speaker": item.speaker, "text_mode": item.text_mode,
                        "input_wav": str(staged.pipeline_wav),
                        "input_wav_sha256": staged.pipeline_wav_sha256,
                        "gamesl_wav": str(staged.gamesl_wav) if staged.gamesl_wav else None,
                        "gamesl_wav_sha256": staged.gamesl_wav_sha256,
                        "padded_verification": verify_padded_wav(staged.pipeline_wav, .5) if item.needs_gamesl_padding else None,
                        "normalized_text": staged.normalized_text,
                        "normalized_text_path": str(staged.text_path) if staged.text_path else None,
                        "normalized_text_sha256": staged.text_sha256,
                        "qwen_transcript": manifest_row.get("text_asr") or manifest_row.get("text_normalized"),
                        "qwen_timestamps": manifest_row.get("_words"),
                        "output_textgrid": str(publication.accepted[0].source),
                        "output_textgrid_sha256": _sha256(publication.accepted[0].source),
                        "evidence": evidence})
    if not config.get("dry_run"):
        _durable_json(root / "canary_receipt.json", {"schema": "qwen3-0915all-canary-v1",
                      "exact_count": 2, "success": True, "five_tier_validated": True,
                      "public_output": False, "inventory_sha256": _sha256(root / "frozen_inventory.json"),
                      "config_sha256": _sha256(Path(config["config_path"])) if config.get("config_path") and Path(config["config_path"]).is_file() else None,
                      "results": results})
    return 0


def _validate_preflight_receipt(path: Path, root: Path, config: dict, items, chunks_path: Path) -> dict:
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("preflight receipt is unreadable") from exc
    expected = {
        "schema": "qwen3-0915all-preflight-v1",
        "status": "passed",
        "inventory_sha256": _sha256(root / "frozen_inventory.json"),
        "chunks_sha256": _sha256(chunks_path),
        "config_sha256": _sha256(Path(config["config_path"])) if config.get("config_path") and Path(config["config_path"]).is_file() else None,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("preflight receipt binding is invalid")
    selected = [_select_canary(items, spec) for spec in config.get("canaries", [])]
    expected_canaries = [{"source_id": item.source_id, "speaker": item.speaker,
                          "run_stem": item.run_stem, "text_mode": item.text_mode}
                         for item in selected]
    if (receipt.get("canaries") != expected_canaries
            or receipt.get("source_deferred_to_stage") != len(items)
            or receipt.get("canary_hash_verified") != len(expected_canaries)
            or receipt.get("source_hash_policy") != "frozen_at_prepare_reverified_at_stage"):
        raise RuntimeError("preflight inventory/canary evidence is incomplete")
    required = [str(value) for value in (
        config.get("qwen", {}).get("python"), config.get("mfa", {}).get("python"),
        config.get("qwen", {}).get("asr_model"), config.get("qwen", {}).get("forced_aligner_model"),
        config.get("mfa_dictionary"), config.get("pinyin_dictionary"), config.get("mfa_models")) if value]
    if receipt.get("dependencies") != required or any(not Path(value).exists() for value in required):
        raise RuntimeError("preflight dependency evidence is stale")
    resolved = receipt.get("resolved_configs", {})
    for mode in ("reference", "fallback"):
        resolved_path = root / "preflight" / f"{mode}.yaml"
        if not resolved_path.is_file() or resolved.get(mode) != _sha256(resolved_path):
            raise RuntimeError("preflight resolved config evidence is stale")
    if receipt.get("process_scan", {}).get("status") != "safe" or receipt.get("process_scan", {}).get("conflicts"):
        raise RuntimeError("preflight process scan is not safe")
    required_gpu_count = int(config.get("required_gpu_count", 0) or 0)
    if receipt.get("gpu_count", 0) < required_gpu_count:
        raise RuntimeError("preflight GPU evidence is insufficient")
    minimum_free = int(config.get("minimum_free_bytes", 0) or 0)
    for target in (config.get("run_root"), config.get("output_root"), config.get("gamesl_root")):
        if target and shutil.disk_usage(Path(target).parent).free < minimum_free:
            raise RuntimeError(f"disk free-space threshold is no longer satisfied: {target}")
    if "WutheringWaves2.2_CN" in str(config.get("sources", {}).get("wuwa", "")):
        if receipt.get("wuwa_denominator") != {"expected": 12138, "observed": 12138, "status": "passed"}:
            raise RuntimeError("preflight Wuthering Waves denominator evidence is invalid")
    return receipt


def _run_contract(config: dict):
    """Load and validate the immutable inventory/chunk execution contract."""
    root = Path(config["run_root"])
    canary = root / "canary_receipt.json"
    payload = json.loads((root / "frozen_inventory.json").read_text(encoding="utf-8"))
    items = payload.get("items", [])
    frozen_items = [_item_from_json(row) for row in items]
    if not canary.is_file():
        raise RuntimeError("successful exact canary receipt is required")
    _validate_canary_receipt(canary, root, config, frozen_items)
    preflight = root / "preflight_receipt.json"
    if not preflight.is_file():
        raise RuntimeError("successful preflight receipt is required")
    limits = config.get("chunk_limits", {})
    chunks = build_chunks(items, ChunkLimits(
        int(limits.get("max_files", 10000)),
        float(limits.get("max_audio_hours", 20)),
        int(limits.get("max_bytes", 10**11))))
    chunks_file = root / "chunks.json"
    if not chunks_file.is_file():
        raise RuntimeError("sealed chunk manifest is required")
    _validate_preflight_receipt(preflight, root, config, frozen_items, chunks_file)
    rows = json.loads(chunks_file.read_text(encoding="utf-8")).get("chunks", [])
    expected = {chunk.chunk_id: sorted(_get(item, "run_stem") for item in chunk.items)
                for chunk in chunks}
    actual = {row.get("chunk_id"): sorted(row.get("items", [])) for row in rows}
    if len(rows) != len(actual) or actual != expected:
        raise RuntimeError("dynamic chunks differ from sealed chunks manifest")
    binding = {
        "inventory_sha256": _sha256(root / "frozen_inventory.json"),
        "chunks_sha256": _sha256(chunks_file),
        "config_sha256": (_sha256(Path(config["config_path"]))
                          if config.get("config_path")
                          and Path(config["config_path"]).is_file() else None),
        "canary_sha256": _sha256(canary),
        "preflight_sha256": _sha256(preflight),
    }
    return root, payload, frozen_items, chunks, binding


def _prepared_input_receipt(root: Path, chunk: Chunk, staged_chunk: Chunk,
                            mfa_chunk: Chunk,
                            failures: list[dict], workspace: Path,
                            receipts, gamesl_receipt: dict, qwen: dict,
                            config: dict) -> dict:
    public_gamesl = {row.get("stem"): row for row in gamesl_receipt.get("replacements", [])}
    stage_by_stem = {receipt.run_stem: receipt for receipt in receipts}
    rows = []
    for item in mfa_chunk.items:
        stem = _get(item, "run_stem")
        stage = stage_by_stem[stem]
        lab = workspace / "ctc_pretg_adj" / f"{stem}.lab"
        textgrid = workspace / "ctc_pretg_adj" / f"{stem}.TextGrid"
        tokens = workspace / "ctc_pretg_adj" / f"{stem}_tokens.jsonl"
        mfa_wav = workspace / "audio_16k" / f"{stem}.wav"
        required = (lab, textgrid, tokens, mfa_wav)
        if any(path.is_symlink() or not path.is_file() for path in required):
            raise RuntimeError(f"prepared input is incomplete: {stem}")
        public = public_gamesl.get(stem)
        authoritative = Path(public["target"]) if public else Path(stage.pipeline_wav)
        authoritative_sha = public.get("new_sha256") if public else stage.pipeline_wav_sha256
        if authoritative.is_symlink() or not authoritative.is_file():
            raise RuntimeError(f"prepared authoritative audio is unavailable: {stem}")
        rows.append({
            "stem": stem,
            "authoritative_audio": str(authoritative),
            "authoritative_audio_sha256": authoritative_sha,
            "pipeline_audio": str(stage.pipeline_wav),
            "pipeline_audio_sha256": stage.pipeline_wav_sha256,
            "normalized_text": str(lab), "normalized_text_sha256": _sha256(lab),
            "normalized_timestamps": str(textgrid),
            "normalized_timestamps_sha256": _sha256(textgrid),
            "normalized_tokens": str(tokens), "normalized_tokens_sha256": _sha256(tokens),
            "mfa_audio": str(mfa_wav), "mfa_audio_sha256": _sha256(mfa_wav),
        })
    bound = _bound_digest(root, chunk, config)
    return {
        "schema": "qwen3-mfa-prepared-input-v1", **bound,
        "ready_stems": sorted(_get(item, "run_stem") for item in staged_chunk.items),
        "mfa_ready_stems": sorted(row["stem"] for row in rows),
        "stage_failures": failures,
        "rows": rows,
        "qwen_identity": qwen.get("identity"),
        "qwen_output_stems": sorted(qwen.get("output_stems", [])),
        "gamesl_receipt": str(root / "chunks" / chunk.chunk_id / "gamesl_publication.json"),
        "ctc_work_receipt": str(workspace / "ctc_pretg_adj" / ".ctc_work_receipt.json"),
    }


def _load_prepared_input(path: Path, chunk: Chunk, root: Path, config: dict):
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "qwen3-mfa-prepared-input-v1":
            return None
        bound = _bound_digest(root, chunk, config)
        if any(receipt.get(key) != value for key, value in bound.items()):
            return None
        prepared, failures = _prepared_chunk(chunk, receipt)
        rows = receipt.get("rows", [])
        mfa_ready = list(receipt.get("mfa_ready_stems", []))
        if (not set(mfa_ready).issubset({_get(item, "run_stem") for item in prepared.items})
                or sorted(row.get("stem") for row in rows) != sorted(mfa_ready)):
            return None
        for row in rows:
            for path_key, hash_key in (
                    ("normalized_text", "normalized_text_sha256"),
                    ("normalized_timestamps", "normalized_timestamps_sha256"),
                    ("normalized_tokens", "normalized_tokens_sha256")):
                artifact = Path(row[path_key])
                if artifact.is_symlink() or not artifact.is_file() or _sha256(artifact) != row[hash_key]:
                    return None
        return receipt
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return None


def run_prepare_all(config: dict) -> int:
    """Prepare every chunk through normalized/adjusted Qwen timestamps before MFA."""
    root, payload, frozen_items, chunks, binding = _run_contract(config)
    phase_path = root / "prepare_status.json"
    if phase_path.is_file():
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        if (phase.get("schema") != "qwen3-mfa-prepare-status-v1"
                or phase.get("chunk_count") != len(chunks)
                or any(phase.get(key) != value for key, value in binding.items())):
            raise RuntimeError("existing prepare status is not bound to this run")
    else:
        phase = {"schema": "qwen3-mfa-prepare-status-v1", "state": "running",
                 "chunk_count": len(chunks), "chunks": {}, **binding}
    expected_chunk_ids = {chunk.chunk_id for chunk in chunks}
    phase_rows = phase.get("chunks", {})
    if (set(phase_rows) == expected_chunk_ids
            and all(row.get("state") in {"prepared", "already_complete"}
                    for row in phase_rows.values())):
        # The phase receipt is written only after every chunk has a durable
        # prepared-input receipt. Do not reopen and rehash all normalized
        # artifacts every time the MFA supervisor restarts.
        phase["state"] = "complete"
        _durable_json(phase_path, phase)
        _event(root, {"event": "prepare_phase_reused",
                      "chunk_count": len(chunks), "fast_resume": True})
        return 0
    phase["state"] = "running"
    _durable_json(phase_path, phase)
    main_status_path = root / "status.json"
    main_status = (json.loads(main_status_path.read_text(encoding="utf-8"))
                   if main_status_path.is_file() else {"terminal": {}})
    frozen = {item.run_stem: item for item in frozen_items}
    try:
        from scripts.full_corpus_stage import stage_item, publish_gamesl
    except ModuleNotFoundError:
        from full_corpus_stage import stage_item, publish_gamesl

    prefetch = None
    producer = None
    next_index = 0
    hard_failure = False
    while next_index < len(chunks) or prefetch or producer:
        if prefetch is None and next_index < len(chunks):
            chunk = chunks[next_index]
            next_index += 1
            if main_status.get("terminal", {}).get(chunk.chunk_id) == "complete":
                phase["chunks"][chunk.chunk_id] = {"state": "already_complete"}
                _durable_json(phase_path, phase)
                continue
            base = root / "chunks" / chunk.chunk_id
            prepared_path = base / "prepared_input.json"
            prior = _load_prepared_input(prepared_path, chunk, root, config)
            if prior is not None:
                phase["chunks"][chunk.chunk_id] = {
                    "state": "prepared", "receipt": str(prepared_path),
                    "ready_stems": prior["ready_stems"],
                    "stage_failures": prior["stage_failures"]}
                _durable_json(phase_path, phase)
                _event(root, {"event": "prepare_chunk_reused", "chunk": chunk.chunk_id})
                continue
            _event(root, {"event": "prepare_prefetch_started", "chunk": chunk.chunk_id,
                          "while_chunk": producer["chunk"].chunk_id if producer else None})
            task = _PrefetchTask(
                chunk.chunk_id, _collect_stage_chunk_items,
                chunk, frozen, base, stage_item).start()
            prefetch = {"chunk": chunk, "base": base, "task": task}

        if prefetch and prefetch["task"].done() and producer is None:
            current = prefetch
            prefetch = None
            chunk, base = current["chunk"], current["base"]
            try:
                pipeline_chunk, receipts, failures = current["task"].result()
                _event(root, {"event": "prepare_stage_completed", "chunk": chunk.chunk_id,
                              "ready_count": len(pipeline_chunk.items),
                              "failed_count": len(failures)})
                gamesl_bound = _gamesl_bound(root, pipeline_chunk, config)
                gamesl_path = base / "gamesl_publication.json"
                gamesl_receipt = _load_valid_gamesl_receipt(gamesl_path, gamesl_bound)
                if gamesl_receipt is None:
                    gamesl_receipt = publish_gamesl(
                        receipts, Path(config["gamesl_root"]), root / "rollback" / "gamesl")
                    gamesl_receipt.update(gamesl_bound)
                    _durable_json(gamesl_path, gamesl_receipt)
                _event(root, {"event": "prepare_gamesl_published", "chunk": chunk.chunk_id,
                              "published_count": gamesl_receipt.get("published_count", 0)})
                if not pipeline_chunk.items:
                    receipt = _prepared_input_receipt(
                        root, chunk, pipeline_chunk, pipeline_chunk, failures,
                        base / "workspace",
                        receipts, gamesl_receipt, {}, config)
                    _durable_json(base / "prepared_input.json", receipt)
                    phase["chunks"][chunk.chunk_id] = {
                        "state": "prepared", "receipt": str(base / "prepared_input.json"),
                        "ready_stems": [], "stage_failures": failures}
                    _durable_json(phase_path, phase)
                    continue
                cfg_path = root / "configs" / f"{chunk.chunk_id}.yaml"
                cfg = materialize_pipeline_config(
                    pipeline_chunk, {**config, "chunk_input": str(base / "input")}, cfg_path)
                workspace = Path(cfg["workspace"])
                qwen_sealed = (workspace / "ctc_pretg" / "manifest.json").is_file()
                command = _prepare_command(
                    str(config.get("mfa", {}).get("python", "python")), cfg_path,
                    qwen_sealed=qwen_sealed)
                process, log = _launch(root, command, f"{chunk.chunk_id}-prepare")
                producer = {"chunk": chunk, "pipeline_chunk": pipeline_chunk,
                            "receipts": receipts, "failures": failures,
                            "gamesl_receipt": gamesl_receipt, "workspace": workspace,
                            "process": process, "log": log}
                _event(root, {"event": "prepare_started", "chunk": chunk.chunk_id,
                              "pid": process.pid, "qwen_sealed": qwen_sealed})
            except Exception as exc:
                hard_failure = True
                phase["chunks"][chunk.chunk_id] = {
                    "state": "prepare_failure", "reason": f"{type(exc).__name__}:{exc}"}
                _durable_json(phase_path, phase)
                _event(root, {"event": "prepare_failed", "chunk": chunk.chunk_id,
                              "reason": f"{type(exc).__name__}:{exc}"})

        if producer and producer["process"].poll() is not None:
            current = producer
            producer = None
            chunk = current["chunk"]
            rc = current["process"].returncode
            _event(root, {"event": "prepare_finished", "chunk": chunk.chunk_id,
                          "pid": current["process"].pid, "return_code": rc})
            if rc:
                hard_failure = True
                phase["chunks"][chunk.chunk_id] = {
                    "state": "prepare_failure", "reason": "prepare_pipeline_failed"}
            else:
                try:
                    qwen = _qwen_evidence(
                        current["workspace"],
                        [_get(item, "run_stem") for item in current["pipeline_chunk"].items],
                        current["pipeline_chunk"].text_mode)
                    qwen_ready = set(qwen.get("output_stems", []))
                    prepared_chunk = _subset_chunk(
                        current["pipeline_chunk"],
                        [item for item in current["pipeline_chunk"].items
                         if _get(item, "run_stem") in qwen_ready])
                    failures = list(current["failures"])
                    receipt = _prepared_input_receipt(
                        root, chunk, current["pipeline_chunk"], prepared_chunk,
                        failures, current["workspace"],
                        current["receipts"], current["gamesl_receipt"], qwen, config)
                    prepared_path = root / "chunks" / chunk.chunk_id / "prepared_input.json"
                    _durable_json(prepared_path, receipt)
                    phase["chunks"][chunk.chunk_id] = {
                        "state": "prepared", "receipt": str(prepared_path),
                        "ready_stems": receipt["ready_stems"],
                        "stage_failures": failures}
                except Exception as exc:
                    hard_failure = True
                    phase["chunks"][chunk.chunk_id] = {
                        "state": "prepare_failure", "reason": f"{type(exc).__name__}:{exc}"}
            _durable_json(phase_path, phase)

        if prefetch or producer:
            time.sleep(float(config.get("scheduler", {}).get("poll_interval_sec", 0.05)))

    expected_states = {"prepared", "already_complete"}
    if (set(phase.get("chunks", {})) != {chunk.chunk_id for chunk in chunks}
            or any(row.get("state") not in expected_states
                   for row in phase.get("chunks", {}).values())):
        hard_failure = True
    phase["state"] = "complete_with_failures" if hard_failure else "complete"
    _durable_json(phase_path, phase)
    return 1 if hard_failure else 0


def run_full(config: dict) -> int:
    root = Path(config["run_root"])
    receipt = root / "canary_receipt.json"
    payload = json.loads((root / "frozen_inventory.json").read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not receipt.is_file():
        raise RuntimeError("successful exact canary receipt is required")
    _validate_canary_receipt(receipt, root, config, [_item_from_json(row) for row in items])
    preflight_path = root / "preflight_receipt.json"
    if not preflight_path.is_file():
        raise RuntimeError("successful preflight receipt is required")
    limits = config.get("chunk_limits", {})
    chunks = build_chunks(items, ChunkLimits(int(limits.get("max_files", 10000)), float(limits.get("max_audio_hours", 20)), int(limits.get("max_bytes", 10**11))))
    chunks_file = root / "chunks.json"
    if not chunks_file.is_file():
        raise RuntimeError("sealed chunk manifest is required")
    _validate_preflight_receipt(preflight_path, root, config,
                                [_item_from_json(row) for row in items], chunks_file)
    chunks_payload = json.loads(chunks_file.read_text(encoding="utf-8"))
    expected_chunk_rows = {chunk.chunk_id: sorted(_get(item, "run_stem") for item in chunk.items) for chunk in chunks}
    chunk_rows = chunks_payload.get("chunks", [])
    actual_chunk_rows = {row.get("chunk_id"): sorted(row.get("items", [])) for row in chunk_rows}
    if len(chunk_rows) != len(actual_chunk_rows) or actual_chunk_rows != expected_chunk_rows:
        raise RuntimeError("dynamic chunks differ from sealed chunks manifest")
    binding = {"inventory_sha256": _sha256(root / "frozen_inventory.json"),
               "chunks_sha256": _sha256(chunks_file),
               "config_sha256": _sha256(Path(config["config_path"])) if config.get("config_path") and Path(config["config_path"]).is_file() else None,
               "canary_sha256": _sha256(receipt), "preflight_sha256": _sha256(preflight_path)}
    # These two inputs are sealed by the run contract. Reuse their digests for
    # every per-chunk receipt instead of rereading the large inventory from NAS.
    config["_inventory_sha256"] = binding["inventory_sha256"]
    config["_config_sha256"] = binding["config_sha256"]
    status_path = root / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (status.get("schema") != "qwen3-0915all-status-v1" or status.get("chunk_count") != len(chunks)
                or any(status.get(key) != value for key, value in binding.items())):
            raise RuntimeError("existing status is not bound to this inventory/chunk plan")
        status["state"] = "running"
    else:
        status = {"schema": "qwen3-0915all-status-v1", "state": "running", "terminal": {}, "items": {}, "chunk_count": len(chunks), **binding}
    _set_status(root, status)
    prepare_phase_path = root / "prepare_status.json"
    prepare_records = None
    if prepare_phase_path.is_file():
        prepare_phase = json.loads(prepare_phase_path.read_text(encoding="utf-8"))
        if (prepare_phase.get("schema") != "qwen3-mfa-prepare-status-v1"
                or prepare_phase.get("state") != "complete"
                or prepare_phase.get("chunk_count") != len(chunks)
                or any(prepare_phase.get(key) != value for key, value in binding.items())):
            raise RuntimeError("MFA cannot start before the all-chunk prepare phase is complete")
        prepare_records = prepare_phase.get("chunks", {})
    try:
        from scripts.full_corpus_stage import stage_item, publish_gamesl
        from scripts.full_corpus_publish import validate_chunk_result, publish_chunk, build_final_report
    except ModuleNotFoundError:
        from full_corpus_stage import stage_item, publish_gamesl
        from full_corpus_publish import validate_chunk_result, publish_chunk, build_final_report
    frozen = {row["run_stem"]: _item_from_json(row) for row in items}
    producer = None
    downstream = None
    prefetch = None
    publication_task = None
    next_index = 0
    publications = []
    publication_receipts = []
    failed = any(record.get("state") in {"producer_failure", "pipeline_failure", "mfa_failure"}
                 for record in status.get("items", {}).values())
    while next_index < len(chunks) or prefetch or producer or downstream or publication_task:
        if publication_task and publication_task["task"].done():
            completed = publication_task
            publication_task = None
            current = completed["current"]
            try:
                result = completed["task"].result()
                publication = result["publication"]
                _event(root, {"event": "chunk_validated", "chunk": current["chunk"].chunk_id,
                              "accepted": list(publication.accepted_stems),
                              "filtered": list(publication.filtered_stems),
                              "failed": list(publication.failed_stems)})
                if result["gamesl_receipt"] is not None:
                    _event(root, {"event": "gamesl_published",
                                  "chunk": current["chunk"].chunk_id,
                                  "items": list(current["gamesl_bound"]["chunk_stems"]),
                                  "receipt": str(current["gamesl_receipt_path"]),
                                  "resumed": result["gamesl_resumed"],
                                  "after_compute": True,
                                  "background": True})
                publication_receipts.append(result["publication_receipt"])
                _event(root, {"event": "output_published",
                              "chunk": current["chunk"].chunk_id,
                              "background": True})
                publications.append(publication)
                for stem in publication.accepted_stems:
                    status["items"][stem] = {"state": "accepted"}
                    _event(root, {"event": "item_terminal", "chunk": current["chunk"].chunk_id,
                                  "stem": stem, "state": "accepted"})
                for stem in publication.filtered_stems:
                    status["items"][stem] = {"state": "filtered"}
                    _event(root, {"event": "item_terminal", "chunk": current["chunk"].chunk_id,
                                  "stem": stem, "state": "filtered"})
                for row in publication.failed:
                    status["items"][row["stem"]] = {
                        "state": "producer_failure", "reason": row["reason"]}
                    _event(root, {"event": "item_terminal", "chunk": current["chunk"].chunk_id,
                                  "stem": row["stem"], "state": "producer_failure",
                                  "reason": row["reason"]})
                failed = failed or bool(publication.failed)
                _mark_terminal(root, status, current["chunk"], "complete")
            except Exception as exc:
                failed = True
                _mark_terminal(root, status, current["pipeline_chunk"],
                               "pipeline_failure", str(exc))

        # Audio for exactly one next chunk may move from NAS to NVMe while the
        # current chunk computes. Qwen/MFA/postprocess remain serial, while the
        # prior chunk's final validation and NAS publication use another lane.
        if prefetch is None and next_index < len(chunks):
            chunk = chunks[next_index]
            next_index += 1
            base = root / "chunks" / chunk.chunk_id
            output_receipt_path = base / "output_publication.json"
            if status.get("terminal", {}).get(chunk.chunk_id) == "complete":
                public_items = [item for item in chunk.items
                                if status.get("items", {}).get(_get(item, "run_stem"), {}).get("state")
                                in {"accepted", "filtered"}]
                output_bound = _bound_digest(
                    root, _subset_chunk(chunk, public_items), config)
                prior_publication = (
                    _load_bound_output_receipt_fast(output_receipt_path, output_bound)
                    if prepare_records is not None
                    else _load_valid_output_receipt(output_receipt_path, output_bound))
                if prior_publication is None:
                    raise RuntimeError(f"completed chunk receipt/target mismatch: {chunk.chunk_id}")
                publication_receipts.append(prior_publication)
                _event(root, {"event": "chunk_resume_skipped", "chunk": chunk.chunk_id,
                              "receipt": str(output_receipt_path)})
                continue
            prepared_failures = []
            stage_chunk = chunk
            if prepare_records is not None:
                record = prepare_records.get(chunk.chunk_id, {})
                if record.get("state") != "prepared":
                    raise RuntimeError(f"missing prepared MFA input: {chunk.chunk_id}")
                stage_chunk, prepared_failures = _prepared_chunk(chunk, record)
            _event(root, {"event": "prefetch_started", "stage": "nas_to_nvme",
                          "chunk": chunk.chunk_id, "item_count": len(stage_chunk.items),
                          "while_chunk": (producer or downstream or {}).get("chunk").chunk_id
                          if (producer or downstream) else None})
            _event(root, {"event": "stage_started", "chunk": chunk.chunk_id,
                          "item_count": len(chunk.items), "background": bool(producer or downstream)})
            task = _PrefetchTask(
                chunk.chunk_id, _collect_stage_chunk_items,
                stage_chunk, frozen, base, stage_item,
                verify_content=prepare_records is None).start()
            prefetch = {"chunk": chunk, "base": base,
                        "output_receipt_path": output_receipt_path,
                        "task": task, "reported": False, "retries": 0,
                        "stage_chunk": stage_chunk,
                        "prepared_failures": prepared_failures,
                        "prepared_for_mfa": prepare_records is not None}
        if prefetch and prefetch["task"].done() and not prefetch["reported"]:
            prefetch["reported"] = True
            _event(root, {"event": "prefetch_completed", "stage": "nas_to_nvme",
                          "chunk": prefetch["chunk"].chunk_id})
        if prefetch and _prefetch_can_promote(
                prefetch["task"], producer, downstream):
            current_prefetch = prefetch
            prefetch = None
            chunk = current_prefetch["chunk"]
            base = current_prefetch["base"]
            output_receipt_path = current_prefetch["output_receipt_path"]
            try:
                staged_result = current_prefetch["task"].result()
            except Exception as exc:
                if current_prefetch["retries"] < 1:
                    current_prefetch["retries"] += 1
                    current_prefetch["reported"] = False
                    current_prefetch["task"] = _PrefetchTask(
                        chunk.chunk_id, _collect_stage_chunk_items,
                        current_prefetch.get("stage_chunk", chunk),
                        frozen, base, stage_item,
                        verify_content=not current_prefetch.get(
                            "prepared_for_mfa", False)).start()
                    prefetch = current_prefetch
                    _event(root, {"event": "prefetch_retried", "chunk": chunk.chunk_id,
                                  "reason": f"{type(exc).__name__}:{exc}"})
                else:
                    failed = True
                    _mark_terminal(root, status, chunk, "pipeline_failure", str(exc))
                continue
            try:
                pipeline_chunk, receipts, stage_failures = _apply_stage_result(
                    root, status, chunk, staged_result)
                for row in current_prefetch.get("prepared_failures", []):
                    status["items"][row["stem"]] = {
                        "state": row["state"], "reason": row["reason"]}
                    _event(root, {"event": "item_terminal", "chunk": chunk.chunk_id, **row})
                _set_status(root, status)
                prepared_failure_count = len(current_prefetch.get("prepared_failures", []))
                failed = failed or bool(stage_failures or prepared_failure_count)
                _event(root, {"event": "stage_completed", "chunk": chunk.chunk_id,
                              "input_count": len(chunk.items),
                              "ready_count": len(pipeline_chunk.items),
                              "failed_count": stage_failures + prepared_failure_count})
                output_bound = _bound_digest(root, pipeline_chunk, config)
                if not pipeline_chunk.items:
                    empty_receipt = {"published_count": 0, "replaced_count": 0,
                                     "replacements": [], **output_bound}
                    _durable_json(output_receipt_path, empty_receipt)
                    publication_receipts.append(empty_receipt)
                    _mark_terminal(root, status, chunk, "complete")
                    continue
                gamesl_bound = _gamesl_bound(root, pipeline_chunk, config)
                gamesl_receipt_path = base / "gamesl_publication.json"
                cfg_path = root / "configs" / f"{chunk.chunk_id}.yaml"
                cfg = materialize_pipeline_config(pipeline_chunk, {**config, "chunk_input": str(base / "input")}, cfg_path)
                workspace = Path(cfg["workspace"])
                py = str(config.get("mfa", {}).get("python", "python"))
                sealed_manifest = workspace / "ctc_pretg" / "manifest.json"
                if sealed_manifest.exists():
                    qwen = _qwen_evidence(workspace, [_get(item, "run_stem") for item in pipeline_chunk.items], chunk.text_mode)
                    _event(root, {"event": "qwen_evidence_sealed", "chunk": chunk.chunk_id, "paths": qwen})
                    producer = {"chunk": chunk, "pipeline_chunk": pipeline_chunk,
                                "receipts": receipts, "cfg": cfg, "workspace": workspace,
                                "output_bound": output_bound,
                                "output_receipt_path": output_receipt_path,
                                "gamesl_bound": gamesl_bound,
                                "gamesl_receipt_path": gamesl_receipt_path,
                                "process": None, "log": None, "qwen": qwen, "resumed": True,
                                "prepared_for_mfa": current_prefetch.get("prepared_for_mfa", False)}
                    _event(root, {"event": "qwen_evidence_reused", "chunk": chunk.chunk_id,
                                  "paths": qwen})
                else:
                    command = [py, str(_RUN_PIPELINE), "--config", str(cfg_path), "--python", py, "--stop-after", "prealign", "--validate"]
                    process, log = _launch(root, command, f"{chunk.chunk_id}-prealign")
                    producer = {"chunk": chunk, "pipeline_chunk": pipeline_chunk,
                                "receipts": receipts, "cfg": cfg, "workspace": workspace,
                                "output_bound": output_bound,
                                "output_receipt_path": output_receipt_path,
                                "gamesl_bound": gamesl_bound,
                                "gamesl_receipt_path": gamesl_receipt_path,
                                "process": process, "log": log}
            except Exception as exc:
                failed = True
                _mark_terminal(root, status, chunk, "pipeline_failure", str(exc))
            if producer is not None:
                # Re-enter admission before polling even an already-sealed
                # producer so NAS prefetch overlaps recovery validation,
                # GAMESL publication, and ordinary Qwen/MFA computation.
                continue
        if producer and (producer["process"] is None or producer["process"].poll() is not None):
            current = producer
            producer = None
            rc = 0 if current["process"] is None else current["process"].returncode
            _event(root, {"event": "finished", "lane": "gpu", "stage": current["chunk"].chunk_id + "-prealign", "pid": current["process"].pid if current["process"] else None, "return_code": rc, "log": str(current["log"]) if current["log"] else None, "resumed": current.get("resumed", False)})
            if rc:
                failed = True
                _mark_terminal(root, status, current["pipeline_chunk"], "producer_failure", "qwen_prealign_failed")
            else:
                try:
                    qwen = current.get("qwen") or _qwen_evidence(
                        current["workspace"],
                        [_get(item, "run_stem") for item in current["pipeline_chunk"].items],
                        current["pipeline_chunk"].text_mode)
                    current = {**current, "qwen": qwen}
                    try:
                        recovered = _validate_completed_chunk(
                            current, frozen, validate_chunk_result)
                    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                        _event(root, {"event": "downstream_recovery_unavailable",
                                      "chunk": current["chunk"].chunk_id,
                                      "reason": f"{type(exc).__name__}:{exc}"})
                        py = str(config.get("mfa", {}).get("python", "python"))
                        cfg_path = root / "configs" / f"{current['chunk'].chunk_id}.yaml"
                        prepared_for_mfa = bool(current.get("prepared_for_mfa", False))
                        refresh_ctc_work = bool(current.get("resumed", False)) and not prepared_for_mfa
                        command = _downstream_command(
                            py, cfg_path, refresh_ctc_work=refresh_ctc_work,
                            skip_to="align" if prepared_for_mfa else "normalize_punct",
                            validate=not prepared_for_mfa)
                        process, log = _launch(root, command, f"{current['chunk'].chunk_id}-downstream")
                        _event(root, {"event": "downstream_started", "chunk": current["chunk"].chunk_id,
                                      "pid": process.pid,
                                      "refresh_ctc_work": refresh_ctc_work,
                                      "prepared_for_mfa": prepared_for_mfa})
                        downstream = {**current, "process": process, "log": log}
                    else:
                        _event(root, {"event": "downstream_evidence_reused",
                                      "chunk": current["chunk"].chunk_id})
                        downstream = {**current, "process": None, "log": None,
                                      "recovered_publication": recovered}
                except Exception as exc:
                    failed = True
                    _mark_terminal(root, status, current["pipeline_chunk"], "producer_failure", str(exc))
        if (downstream and publication_task is None
                and (downstream["process"] is None
                     or downstream["process"].poll() is not None)):
            current = downstream
            downstream = None
            rc = 0 if current["process"] is None else current["process"].returncode
            _event(root, {"event": "finished", "lane": "cpu", "stage": current["chunk"].chunk_id + "-downstream", "pid": current["process"].pid if current["process"] else None, "return_code": rc, "log": str(current["log"]) if current["log"] else None, "resumed": current["process"] is None})
            if rc:
                failed = True
                _mark_terminal(root, status, current["pipeline_chunk"], "pipeline_failure", "downstream_failed")
            else:
                task = _PrefetchTask(
                    current["chunk"].chunk_id, _publish_completed_chunk,
                    current, root, config, frozen, validate_chunk_result,
                    publish_gamesl, publish_chunk).start()
                publication_task = {"current": current, "task": task}
                _event(root, {"event": "publication_started",
                              "chunk": current["chunk"].chunk_id,
                              "background": True})
        if prefetch or producer or downstream or publication_task:
            time.sleep(float(config.get("scheduler", {}).get("poll_interval_sec", 0.05)))
    expected = {_get(item, "run_stem") for item in items}
    if set(status["items"]) != expected:
        failed = True
    status["state"] = "complete_with_failures" if failed else "complete"
    _set_status(root, status)
    _durable_json(root / "final_report.json", _status_report(payload, status, publications,
                                                               publication_receipts))
    return 1 if failed else 0


def _status_report(payload, status, publications, publication_receipts=None):
    counts = {"input": len(payload.get("items", [])), "accepted": 0, "filtered": 0,
              "producer_failure": 0, "pipeline_or_mfa_failure": 0,
              "excluded": len(payload.get("excluded", [])), "invalid": len(payload.get("invalid", []))}
    reasons = {}
    for record in status.get("items", {}).values():
        state = record.get("state")
        if state in {"accepted", "filtered"}:
            counts[state] += 1
        elif state == "producer_failure":
            counts[state] += 1
        elif state in {"pipeline_failure", "mfa_failure"}:
            counts["pipeline_or_mfa_failure"] += 1
        if record.get("reason"):
            reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    groups = {}
    for item in payload.get("items", []):
        public_speaker = publication_speaker(
            item.get("game"), item.get("speaker") or "_default")
        key = (item.get("source_id"), item.get("game"), public_speaker,
               item.get("text_mode"))
        record = groups.setdefault(key, {"source_id": key[0], "game": key[1], "speaker": key[2], "text_mode": key[3], "input": 0, "duration_seconds": 0.0})
        record["input"] += 1; record["duration_seconds"] += float(item.get("duration_seconds", 0.0) or 0.0)
    return {"schema": "qwen3-0915all-report-v1", "counts": counts,
            "reasons": reasons, "status": status,
            "terminal_stems": sorted(status.get("items", {})),
            "groups": list(groups.values()),
            "publication_receipts": list(publication_receipts or []),
            "excluded": payload.get("excluded", []), "invalid": payload.get("invalid", [])}


def prepare(config: dict, sample_only: int = 0, dry_run: bool = False) -> dict:
    root = Path(tempfile.mkdtemp(prefix="qwen3-0915all-preview-")) if (sample_only or dry_run) else Path(config["run_root"])
    local = copy.deepcopy(config)
    local["_sample_only"] = sample_only
    snapshot = scan_sources(local)
    write_frozen_inventory(snapshot, root / "frozen_inventory.json")
    limits = config.get("chunk_limits", {})
    chunks = build_chunks(snapshot.items, ChunkLimits(int(limits.get("max_files", 10000)), float(limits.get("max_audio_hours", 20)), int(limits.get("max_bytes", 10**11))))
    _durable_json(root / "chunks.json", {"schema": "qwen3-0915all-chunks-v1", "chunks": [
        {"chunk_id": chunk.chunk_id, "text_mode": chunk.text_mode, "items": [_get(item, "run_stem") for item in chunk.items], "total_seconds": chunk.total_seconds, "total_bytes": chunk.total_bytes}
        for chunk in chunks]})
    print(json.dumps({"preview_root": str(root), "items": len(snapshot.items), "excluded": len(snapshot.excluded), "chunks": len(chunks)}, ensure_ascii=False))
    return {"root": root, "snapshot": snapshot, "chunks": chunks}


def _check_no_symlink_components(path: Path):
    path = Path(path)
    for parent in [path, *path.parents]:
        if parent == parent.parent:
            break
        if parent.is_symlink():
            raise ValueError(f"symlink path component: {parent}")


def preflight(config: dict) -> int:
    if "nvasr" in json.dumps(config, ensure_ascii=False).lower():
        raise ValueError("NVASR identity is forbidden")
    for key in ("run_root", "output_root", "gamesl_root"):
        if config.get(key):
            _check_no_symlink_components(Path(config[key]))
    write_roots = [Path(config[key]).resolve() for key in ("run_root", "output_root", "gamesl_root") if config.get(key)]
    for index, left in enumerate(write_roots):
        for right in write_roots[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("run, output, and GAMESL roots must be disjoint")
    for key in ("gamedata", "v5_0707", "wuwa"):
        root = Path(config.get("sources", {}).get(key, ""))
        _check_no_symlink_components(root)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError(f"source root unavailable: {key}")
        for write_root in write_roots:
            if root.resolve() == write_root or root.resolve() in write_root.parents or write_root in root.resolve().parents:
                raise ValueError(f"source/write root overlap: {root} / {write_root}")
    required = (config.get("qwen", {}).get("python"), config.get("mfa", {}).get("python"),
                config.get("qwen", {}).get("asr_model"), config.get("qwen", {}).get("forced_aligner_model"),
                config.get("mfa_dictionary"), config.get("pinyin_dictionary"), config.get("mfa_models"))
    for path in required:
        if path and not Path(path).exists():
            raise ValueError(f"required dependency missing: {path}")
    canaries = config.get("canaries", [])
    if len(canaries) != 2 or canaries[0].get("relative_stem") != "zh_vo_Chengxiaoshan_main_1_1_145_11" or canaries[1].get("relative_path") != "wavs/LAria_00001.wav":
        raise ValueError("canary contract mismatch")
    inventory_path = Path(config.get("run_root", "")) / "frozen_inventory.json"
    selected = []
    source_deferred_to_stage = 0
    canary_hash_verified = 0
    if inventory_path.is_file():
        frozen_items = _load_frozen_items(Path(config["run_root"]))
        selected = [_select_canary(frozen_items, spec) for spec in canaries]
        # Preparing the frozen inventory already hashed every byte.  Re-read
        # only the two fixed canaries here.  The complete audio/reference
        # hashes for all items are compared to that inventory immediately
        # before each item is staged, which also catches edits made during a
        # multi-day run.
        for item in selected:
            if not item.source_path.is_file() or _sha256(item.source_path) != item.audio_sha256:
                raise ValueError(f"frozen canary source hash drift: {item.source_path}")
            if item.reference_path is not None and (
                    not item.reference_path.is_file()
                    or _sha256(item.reference_path) != item.reference_sha256):
                raise ValueError(f"frozen canary reference hash drift: {item.reference_path}")
            canary_hash_verified += 1
        source_deferred_to_stage = len(frozen_items)
        if selected[0].text_mode != "reference" or selected[1].text_mode != "fallback":
            raise ValueError("canary text mode contract mismatch")
        # Validate both generated modes against run_pipeline's actual schema.
        try:
            from scripts.run_pipeline import validate_config
        except ModuleNotFoundError:
            from run_pipeline import validate_config
        for mode in ("reference", "fallback"):
            probe = Chunk("preflight-" + mode, mode, (selected[0] if mode == "reference" else selected[1],), 0, 0)
            resolved = materialize_pipeline_config(probe, {**config, "chunk_input": str(Path(config["run_root"]) / "preflight-input")}, Path(config["run_root"]) / "preflight" / (mode + ".yaml"))
            errors = validate_config(resolved, "full")
            if errors:
                raise ValueError("resolved pipeline config invalid: " + "; ".join(errors))
    if config.get("required_gpu_count"):
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if probe.returncode or len([line for line in probe.stdout.splitlines() if "GPU" in line]) < int(config["required_gpu_count"]):
            raise ValueError("eight GPUs are required")
    process_probe = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False)
    config_text = str(config.get("config_path", "")); workspace_prefix = str(config.get("run_root", ""))
    conflicts = [line.strip() for line in process_probe.stdout.splitlines()
                 if "run_pipeline.py" in line and (config_text in line or workspace_prefix in line)]
    if conflicts:
        raise ValueError("conflicting pipeline process already uses this run root")
    minimum_free = int(config.get("minimum_free_bytes", 0) or 0)
    if minimum_free:
        disk_samples = {str(path): shutil.disk_usage(Path(path).parent).free for path in (config["run_root"], config["output_root"], config["gamesl_root"])}
        for path in disk_samples:
            if disk_samples[path] < minimum_free:
                raise ValueError(f"disk free-space threshold failed: {path}")
    else:
        disk_samples = {str(path): shutil.disk_usage(Path(path).parent).free for path in (config["run_root"], config["output_root"], config["gamesl_root"])}
    # The production source has a sealed expected eligible denominator.  Keep
    # this check out of small temporary fixtures used by unit tests.
    wuwa_root = Path(config.get("sources", {}).get("wuwa", ""))
    if "WutheringWaves2.2_CN" in str(wuwa_root):
        inventory = Path(config["run_root"]) / "frozen_inventory.json"
        if inventory.is_file():
            eligible = sum(row.get("source_id") == "wuwa" for row in json.loads(inventory.read_text(encoding="utf-8")).get("items", []))
        else:
            eligible = sum(row.source_id == "wuwa" for row in scan_sources(config).items)
        if eligible != 12138:
            report = Path(config["run_root"]) / "preflight_wuwa_drift.json"
            _durable_json(report, {"expected": 12138, "observed": eligible, "source": str(wuwa_root), "blocked": True})
            raise ValueError(f"Wuthering Waves eligible denominator drift: expected 12138, observed {eligible}")
        wuwa_gate = {"expected": 12138, "observed": eligible, "status": "passed"}
    else:
        wuwa_gate = {"expected": None, "observed": None, "status": "not_applicable"}
    receipts = {
        "schema": "qwen3-0915all-preflight-v1",
        "config_sha256": _sha256(Path(config["config_path"])) if config.get("config_path") and Path(config["config_path"]).is_file() else None,
        "inventory_sha256": _sha256(inventory_path) if inventory_path.is_file() else None,
        "chunks_sha256": _sha256(Path(config["run_root"]) / "chunks.json") if (Path(config["run_root"]) / "chunks.json").is_file() else None,
        "canaries": [{"source_id": item.source_id, "speaker": item.speaker, "run_stem": item.run_stem, "text_mode": item.text_mode} for item in selected] if inventory_path.is_file() else [],
        "status": "passed",
        "process_scan": {"status": "safe", "conflicts": conflicts},
        "gpu_count": int(config.get("required_gpu_count", 0) or 0),
        "dependencies": [str(path) for path in required if path],
        "disk_free_bytes": disk_samples,
        "source_deferred_to_stage": source_deferred_to_stage,
        "canary_hash_verified": canary_hash_verified,
        "source_hash_policy": "frozen_at_prepare_reverified_at_stage",
        "resolved_configs": {mode: _sha256(Path(config["run_root"]) / "preflight" / (mode + ".yaml")) for mode in ("reference", "fallback") if (Path(config["run_root"]) / "preflight" / (mode + ".yaml")).is_file()},
        "wuwa_denominator": wuwa_gate,
    }
    _durable_json(Path(config["run_root"]) / "preflight_receipt.json", receipts)
    return 0


def audit(config: dict) -> int:
    root = Path(config["run_root"])
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    payload = json.loads((root / "frozen_inventory.json").read_text(encoding="utf-8"))
    expected = {row["run_stem"] for row in payload.get("items", [])}
    states = status.get("items", {})
    allowed = {"accepted", "filtered", "producer_failure", "pipeline_failure", "mfa_failure"}
    if status.get("state") not in {"complete", "complete_with_failures"} or set(states) != expected:
        raise ValueError("terminal conservation is incomplete")
    if any(row.get("state") not in allowed for row in states.values()):
        raise ValueError("unknown terminal item state")
    chunks_path = root / "chunks.json"
    if chunks_path.is_file():
        chunk_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        expected_chunks = {row["chunk_id"] for row in chunk_payload.get("chunks", [])}
        if set(status.get("terminal", {})) != expected_chunks:
            raise ValueError("terminal chunk denominator is incomplete or duplicated")
    if len(states) != len(set(states)):
        raise ValueError("duplicate terminal item records")
    report_path = root / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else _status_report(payload, status, [])
    if report.get("schema") != "qwen3-0915all-report-v1" or set(report.get("terminal_stems", [])) != expected:
        raise ValueError("final report denominator is incomplete")
    if report.get("status") != status:
        raise ValueError("final report status binding mismatch")
    counts = report.get("counts", {})
    terminal_count = sum(int(counts.get(key, 0) or 0) for key in ("accepted", "filtered", "producer_failure", "pipeline_or_mfa_failure"))
    if terminal_count != len(expected):
        raise ValueError("final report state counts do not conserve inputs")
    public_states = {stem: row for stem, row in states.items()
                     if row.get("state") in {"accepted", "filtered"}}
    inventory_map = {row["run_stem"]: row for row in payload.get("items", [])}
    published = {}
    for publication in report.get("publication_receipts", []):
        if not isinstance(publication, dict):
            raise ValueError("final report publication receipt is invalid")
        if publication.get("inventory_sha256") != _sha256(root / "frozen_inventory.json"):
            raise ValueError("publication receipt inventory binding mismatch")
        if config.get("config_path") and publication.get("config_sha256") != _sha256(Path(config["config_path"])):
            raise ValueError("publication receipt config binding mismatch")
        receipt_stems = sorted(publication.get("chunk_stems", []))
        rows = publication.get("replacements", [])
        if sorted(row.get("stem") for row in rows) != receipt_stems:
            raise ValueError("publication receipt stem set mismatch")
        for row in rows:
            stem = row.get("stem")
            if stem in published or stem not in public_states:
                raise ValueError("duplicate or unknown published stem")
            item = inventory_map[stem]
            state = public_states[stem]["state"]
            expected_speaker = publication_speaker(
                item.get("game"), item.get("speaker") or "_default")
            if row.get("kind") != state or row.get("speaker") != expected_speaker:
                raise ValueError("publication receipt state/speaker mismatch")
            target = Path(row.get("target", ""))
            if config.get("output_root"):
                suffix = Path("_filtered") if state == "filtered" else Path()
                wanted = (Path(config["output_root"]) / suffix /
                          expected_speaker / f"{stem}.TextGrid").resolve()
                if target.resolve() != wanted:
                    raise ValueError("publication receipt target mapping mismatch")
            if not target.is_file() or row.get("new_sha256") != _sha256(target):
                raise ValueError("final report publication target is missing or tampered")
            published[stem] = row
    if set(published) != set(public_states):
        raise ValueError("published output denominator is incomplete")
    _durable_json(Path(config.get("reports_root", root / "reports")) / "final_report.json", report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "canary", "prepare-all", "run", "audit"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-only", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["config_path"] = str(Path(args.config).resolve())
    config["dry_run"] = args.dry_run
    if args.command == "prepare":
        prepare(config, args.sample_only, args.dry_run)
        return 0
    if args.command == "preflight":
        return preflight(config)
    if args.command == "canary":
        return run_canary(config)
    if args.command == "prepare-all":
        return run_prepare_all(config)
    if args.command == "run":
        prepare_rc = run_prepare_all(config)
        if prepare_rc:
            return prepare_rc
        return run_full(config)
    return audit(config)


if __name__ == "__main__":
    raise SystemExit(main())
