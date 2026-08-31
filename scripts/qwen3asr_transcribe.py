#!/usr/bin/env python3
"""Isolated, resumable Qwen3-ASR transcript-only runner.

This module deliberately has no import-time dependency on ``qwen_asr`` or
``torch``.  The official Transformers backend is loaded only after all input,
model, and capability checks have passed.  It never reads or writes CTC, MFA,
TextGrid, postprocess, or ordinary pipeline receipt artifacts.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types
import gc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from pipeline_utils import (compute_model_tree_digest, read_wav_metadata,
                                stable_json_digest)
except ModuleNotFoundError:  # package import: ``from scripts import qwen3asr_transcribe``
    from .pipeline_utils import (compute_model_tree_digest, read_wav_metadata,
                                 stable_json_digest)


SCHEMA = "qwen3asr-mode-v1"
CHECKPOINT_SCHEMA = "qwen3asr-checkpoint-v1"
MANIFEST_SCHEMA = "qwen3asr-manifest-v1"
RECEIPT_SCHEMA = "qwen3asr-run-receipt-v1"
ANCHORED_PROFILE = "anchored_nvv"
TRANSCRIPT_PROFILE = "transcript_only"
ANCHORED_CHECKPOINT_SCHEMA = "qwen3asr-anchored-nvv-checkpoint-v1"
ANCHORED_MANIFEST_SCHEMA = "qwen3asr-anchored-nvv-manifest-v1"
ANCHORED_RECEIPT_SCHEMA = "qwen3asr-anchored-nvv-run-receipt-v1"
ANCHOR_SCHEMA = "qwen3asr-anchored-nvv-anchor-v1"
TIMELINE_SCHEMA = "nvasr-candidate-timeline-v1"
FUSION_SCHEMA = "qwen3asr-anchored-nvv-v1"
PROVIDER_CONTRACT_VERSION = "qwen3asr-provider-contract-v2"
ZERO_WIDTH_POLICY_VERSION = "zero-width-span-policy-v1"
SUPPORTED_QWEN_ASR_VERSION = "0.0.6"
BACKEND_NAME = "transformers-qwen3asr"


class Qwen3ASRError(RuntimeError):
    """A fail-closed qwen3asr configuration, identity, or run error."""


class TranscriptResultError(ValueError):
    """The backend returned a result that cannot produce a transcript."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _resolve_path(raw: str | os.PathLike[str], project_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _qcfg(cfg: dict[str, Any]) -> dict[str, Any]:
    value = cfg.get("qwen3asr", {})
    if not isinstance(value, dict):
        raise Qwen3ASRError("qwen3asr config section must be a mapping")
    return value


def _source_inventory(data_dir: Path) -> list[dict[str, Any]]:
    """Freeze and validate one unambiguous WAV for every source stem."""
    if data_dir.is_symlink():
        raise Qwen3ASRError(f"qwen3asr data_dir must not be a symlink: {data_dir}")
    root = data_dir.resolve(strict=True)
    if not root.is_dir():
        raise Qwen3ASRError(f"qwen3asr data_dir must be a regular directory: {data_dir}")
    candidates = sorted(
        (path for path in root.rglob("*")
         if path.suffix.lower() == ".wav" and path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    records: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for wav_path in candidates:
        if wav_path.is_symlink():
            raise Qwen3ASRError(f"qwen3asr source WAV symlink is not allowed: {wav_path}")
        stem = wav_path.stem
        if not stem or stem in {".", ".."}:
            raise Qwen3ASRError(f"invalid WAV source stem: {wav_path.name!r}")
        if stem in seen:
            raise Qwen3ASRError(
                f"duplicate qwen3asr source stem {stem!r}: {seen[stem]} and {wav_path}")
        try:
            metadata = read_wav_metadata(wav_path)
            if metadata["channels"] < 1 or metadata["sample_rate"] < 1:
                raise ValueError("missing channel/rate")
            if metadata["frames"] < 1:
                raise ValueError("empty audio")
        except Exception as exc:
            raise Qwen3ASRError(f"invalid WAV source {wav_path}: {exc}") from exc
        seen[stem] = wav_path
        relative = wav_path.relative_to(root).as_posix()
        records.append({
            "stem": stem,
            "wav_relative_path": relative,
            "size": wav_path.stat().st_size,
            "sha256": metadata["sha256"],
        })
    return sorted(records, key=lambda record: record["stem"])


def _model_identity(model_path: Path, *, label: str = "qwen3asr") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if model_path.is_symlink() or not model_path.is_dir():
        raise Qwen3ASRError(
            f"{label} model path must be an existing local non-symlink directory: {model_path}")
    try:
        for child in model_path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"symlink not allowed in model tree: {child}")
        digest, files = compute_model_tree_digest(model_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Qwen3ASRError(f"invalid {label} model tree {model_path}: {exc}") from exc
    if not files:
        raise Qwen3ASRError(f"{label} model directory is empty: {model_path}")
    return {"path": str(model_path.resolve()), "tree_digest": digest}, files


def _installed_qwen_version() -> str:
    try:
        return importlib.metadata.version("qwen-asr")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Qwen3ASRError(
            "qwen-asr is not installed; install the isolated dependency with "
            "pip install -r requirements-qwen3asr.txt") from exc


def _validate_runtime_config(cfg: dict[str, Any], *, require_runtime: bool = False,
                             device_override: str | None = None,
                             profile_override: str | None = None) -> dict[str, Any]:
    qcfg = _qcfg(cfg)
    profile = str(profile_override or qcfg.get("profile", TRANSCRIPT_PROFILE)).strip()
    if profile not in {TRANSCRIPT_PROFILE, ANCHORED_PROFILE}:
        raise Qwen3ASRError(
            f"qwen3asr.profile must be {TRANSCRIPT_PROFILE!r} or {ANCHORED_PROFILE!r}")
    backend = str(qcfg.get("backend", "transformers"))
    if backend != "transformers":
        raise Qwen3ASRError(
            "qwen3asr v1 supports only backend=transformers "
            "(Qwen3ASRModel.from_pretrained); vLLM and online APIs are unsupported")
    model_raw = qcfg.get("model_path")
    if not model_raw:
        raise Qwen3ASRError("qwen3asr.model_path is required and must point to a local model directory")
    device = str(device_override or qcfg.get("device", "cuda:0"))
    dtype = str(qcfg.get("dtype", "bfloat16"))
    language = qcfg.get("language")
    context = qcfg.get("context", "")
    batch_size = qcfg.get("batch_size", 1)
    max_new_tokens = qcfg.get("max_new_tokens", 2048)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise Qwen3ASRError("qwen3asr.batch_size must be a positive integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise Qwen3ASRError("qwen3asr.max_new_tokens must be a positive integer")
    if not isinstance(context, str):
        raise Qwen3ASRError("qwen3asr.context must be a string")
    if language is not None and not isinstance(language, str):
        raise Qwen3ASRError("qwen3asr.language must be a string or null")
    if isinstance(language, str):
        language = language.strip()
        if not language or language.casefold() == "auto":
            language = None
    forced_aligner_raw = qcfg.get("forced_aligner_model_path")
    nvasr_raw = qcfg.get("nvasr_model_path")
    nvv_bias = qcfg.get("nvv_bias", 4.0)
    pause_threshold = qcfg.get("pause_threshold", 8)
    if profile == ANCHORED_PROFILE:
        if isinstance(nvv_bias, bool) or not isinstance(nvv_bias, (int, float)):
            raise Qwen3ASRError("qwen3asr.nvv_bias must be a finite number")
        if not isinstance(pause_threshold, int) or isinstance(pause_threshold, bool) \
                or pause_threshold <= 0:
            raise Qwen3ASRError("qwen3asr.pause_threshold must be a positive integer")
        if not (float("-inf") < float(nvv_bias) < float("inf")):
            raise Qwen3ASRError("qwen3asr.nvv_bias must be finite")
        if language is None or language.casefold() != "chinese":
            raise Qwen3ASRError(
                "qwen3asr anchored_nvv requires language=Chinese (auto detection is forbidden)")
        if not forced_aligner_raw:
            raise Qwen3ASRError(
                "qwen3asr anchored_nvv requires forced_aligner_model_path")
        if not nvasr_raw:
            raise Qwen3ASRError(
                "qwen3asr anchored_nvv requires nvasr_model_path")
    else:
        nvv_bias, pause_threshold = 4.0, 8
    if require_runtime:
        version = _installed_qwen_version()
        if version != SUPPORTED_QWEN_ASR_VERSION:
            raise Qwen3ASRError(
                f"unsupported qwen-asr version {version!r}; v1 requires "
                f"qwen-asr=={SUPPORTED_QWEN_ASR_VERSION}")
    return {
        "model_path": Path(str(model_raw)),
        "backend": BACKEND_NAME,
        "device": device,
        "dtype": dtype,
        "language": language,
        "context": context,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "profile": profile,
        "forced_aligner_model_path": (
            Path(str(forced_aligner_raw)) if forced_aligner_raw else None
        ),
        "nvasr_model_path": Path(str(nvasr_raw)) if nvasr_raw else None,
        "nvv_bias": float(nvv_bias),
        "pause_threshold": pause_threshold,
        "qwen_config": qcfg,
    }


def _validate_torch_capability(settings: dict[str, Any], torch: Any) -> Any:
    """Validate dtype/device availability without constructing model weights."""
    dtype = getattr(torch, settings["dtype"], None)
    if dtype is None:
        raise Qwen3ASRError(f"torch does not expose configured dtype {settings['dtype']!r}")
    cuda_match = re.fullmatch(r"cuda:(\d+)", settings["device"])
    if cuda_match:
        index = int(cuda_match.group(1))
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            raise Qwen3ASRError(
                f"configured qwen3asr device {settings['device']} requires CUDA, but CUDA is unavailable")
        count = int(cuda.device_count())
        if index >= count:
            raise Qwen3ASRError(
                f"configured qwen3asr device {settings['device']} is unavailable; "
                f"torch reports {count} CUDA device(s)")
    return dtype


def _load_backend(settings: dict[str, Any]) -> tuple[Any, str]:
    """Load only the official Qwen3ASRModel Transformers backend."""
    try:
        package = importlib.import_module("qwen_asr")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise Qwen3ASRError(
            "qwen3asr runtime unavailable; install requirements-qwen3asr.txt and "
            "use a torch-enabled runtime") from exc
    model_class = getattr(package, "Qwen3ASRModel", None)
    if model_class is None or not hasattr(model_class, "from_pretrained"):
        raise Qwen3ASRError(
            "installed qwen_asr does not expose Qwen3ASRModel.from_pretrained")
    dtype = _validate_torch_capability(settings, torch)
    try:
        model = model_class.from_pretrained(
            str(settings["model_path"]),
            dtype=dtype,
            device_map=settings["device"],
            max_inference_batch_size=settings["batch_size"],
            max_new_tokens=settings["max_new_tokens"],
        )
    except Exception as exc:
        raise Qwen3ASRError(f"Qwen3ASRModel.from_pretrained failed: {exc}") from exc
    return model, _installed_qwen_version()


def _close_provider(provider: Any) -> None:
    """Best-effort provider cleanup; cleanup errors never mask inference errors."""
    if provider is None:
        return
    for name in ("close", "release", "shutdown"):
        method = getattr(provider, name, None)
        if callable(method):
            try:
                method()
            except BaseException:
                pass
            break
    try:
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except BaseException:
        pass
    try:
        gc.collect()
    except BaseException:
        pass


class _ForcedAlignerAdapter:
    """JSON boundary around the official Qwen3 forced-aligner result."""

    def __init__(self, model: Any):
        self.model = model

    def align(self, audio: Path, text: str, language: str = "Chinese") -> list[dict[str, Any]]:
        align = getattr(self.model, "align", None)
        if not callable(align):
            raise Qwen3ASRError("Qwen3ForcedAligner does not expose align")
        result = align(str(audio), text, language=language)

        def looks_like_item(item: Any) -> bool:
            if isinstance(item, dict):
                return (any(key in item for key in ("text", "unit"))
                        and any(key in item for key in
                                ("start_time", "start_s", "start")))
            return all(hasattr(item, name) for name in ("text", "start_time", "end_time"))

        def batch_items(batch: Any) -> Any:
            if isinstance(batch, dict):
                items = batch.get("items", batch.get("segments"))
            else:
                items = getattr(batch, "items", None)
            if callable(items):
                items = items()
            return items

        # Official qwen-asr returns one batch result even for one audio file:
        # [ForcedAlignResult(items=[ForcedAlignItem(...)])].  Direct item
        # lists remain supported for isolated fakes and older integrations.
        if isinstance(result, dict):
            result = [result] if looks_like_item(result) else batch_items(result)
        if isinstance(result, (list, tuple)):
            if not all(looks_like_item(item) for item in result):
                if len(result) != 1:
                    raise Qwen3ASRError(
                        "Qwen3ForcedAligner.align must return exactly one batch result")
                result = batch_items(result[0])
        elif all(hasattr(result, name) for name in ("text", "start_time", "end_time")):
            result = [result]
        else:
            result = batch_items(result)
        if not isinstance(result, (list, tuple)):
            raise Qwen3ASRError("Qwen3ForcedAligner.align returned no item sequence")
        if not result:
            raise Qwen3ASRError("Qwen3ForcedAligner.align returned no items")

        raw_items = []
        for index, item in enumerate(result):
            item_text = getattr(item, "text", None)
            start = getattr(item, "start_time", None)
            end = getattr(item, "end_time", None)
            if isinstance(item, dict):
                item_text = item.get("text", item.get("unit"))
                start = item.get("start_time", item.get("start_s", item.get("start")))
                end = item.get("end_time", item.get("end_s", item.get("end")))
            if hasattr(start, "total_seconds"):
                start = start.total_seconds()
            if hasattr(end, "total_seconds"):
                end = end.total_seconds()
            if not isinstance(item_text, str) or not item_text:
                raise Qwen3ASRError(f"forced-aligner item {index} has no text")
            try:
                start_s, end_s = float(start), float(end)
            except (TypeError, ValueError) as exc:
                raise Qwen3ASRError(
                    f"forced-aligner item {index} has invalid timestamps") from exc
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                raise Qwen3ASRError(
                    f"forced-aligner item {index} has non-finite timestamps")
            if end_s < start_s:
                raise Qwen3ASRError(
                    f"forced-aligner item {index} has a negative duration")
            raw_items.append({
                "unit": item_text,
                "raw_start_s": start_s,
                "raw_end_s": end_s,
            })

        quantum_s = None
        normalized = []
        previous_raw_start = None
        previous_final_end = None
        for index, item in enumerate(raw_items):
            raw_start = item["raw_start_s"]
            raw_end = item["raw_end_s"]
            if previous_raw_start is not None and raw_start < previous_raw_start:
                raise Qwen3ASRError(
                    f"forced-aligner item {index} is not monotonic")
            previous_raw_start = raw_start

            adjustment = None
            if raw_end > raw_start:
                start_s, end_s = raw_start, raw_end
            else:
                if quantum_s is None:
                    try:
                        timestamp_segment_time = getattr(
                            self.model, "timestamp_segment_time")
                        quantum_s = float(timestamp_segment_time) / 1000.0
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise Qwen3ASRError(
                            "forced-aligner zero-duration items require model.timestamp_segment_time"
                        ) from exc
                    if not math.isfinite(quantum_s) or quantum_s <= 0:
                        raise Qwen3ASRError(
                            "forced-aligner model.timestamp_segment_time must be finite and positive")

                next_raw_start = (
                    raw_items[index + 1]["raw_start_s"]
                    if index + 1 < len(raw_items) else None
                )
                right_end = raw_start + quantum_s
                right_available = (
                    next_raw_start is not None
                    and right_end <= next_raw_start
                    and (previous_final_end is None or raw_start >= previous_final_end)
                )
                if right_available:
                    start_s, end_s = raw_start, right_end
                    adjustment = {
                        "reason": "zero_duration_expand_right",
                        "quantum_s": quantum_s,
                        "raw_start_s": raw_start,
                        "raw_end_s": raw_end,
                    }
                else:
                    start_s, end_s = raw_start - quantum_s, raw_start
                    if start_s < 0 or (previous_final_end is not None
                                       and start_s < previous_final_end):
                        raise Qwen3ASRError(
                            f"forced-aligner zero-duration item {index} has no full quantum gap")
                    adjustment = {
                        "reason": "zero_duration_expand_left",
                        "quantum_s": quantum_s,
                        "raw_start_s": raw_start,
                        "raw_end_s": raw_end,
                    }

            if (not math.isfinite(start_s) or not math.isfinite(end_s)
                    or end_s <= start_s
                    or (previous_final_end is not None and start_s < previous_final_end)):
                raise Qwen3ASRError(
                    f"forced-aligner item {index} has invalid final timing span")
            normalized_item = {
                "unit": item["unit"],
                "start_s": start_s,
                "end_s": end_s,
                "raw_start_s": raw_start,
                "raw_end_s": raw_end,
            }
            if adjustment is not None:
                normalized_item["timing_adjustment"] = adjustment
            normalized.append(normalized_item)
            previous_final_end = end_s
        return normalized

    def close(self) -> None:
        model = self.model
        for name in ("close", "release", "shutdown"):
            method = getattr(model, name, None)
            if callable(method):
                try:
                    method()
                except BaseException:
                    pass
                break
        self.model = None


def _load_forced_aligner(settings: dict[str, Any]) -> _ForcedAlignerAdapter:
    try:
        package = importlib.import_module("qwen_asr")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise Qwen3ASRError(
            "anchored_nvv forced aligner runtime requires qwen_asr and torch") from exc
    model_class = getattr(package, "Qwen3ForcedAligner", None)
    if model_class is None or not hasattr(model_class, "from_pretrained"):
        raise Qwen3ASRError("installed qwen_asr lacks Qwen3ForcedAligner.from_pretrained")
    dtype = _validate_torch_capability(settings, torch)
    try:
        model = model_class.from_pretrained(
            str(settings["forced_aligner_model_path"]),
            dtype=dtype,
            device_map=settings["device"],
        )
    except Exception as exc:
        raise Qwen3ASRError(f"Qwen3ForcedAligner.from_pretrained failed: {exc}") from exc
    return _ForcedAlignerAdapter(model)


class _NVASRAdapter:
    """FunASR adapter producing only the foundation candidate timeline."""

    def __init__(self, model: Any):
        self.model = model

    def transcribe(self, audio_path: Path, stem: str, **_: Any) -> dict[str, Any]:
        generated = self.model.generate(input=str(audio_path))
        if isinstance(generated, tuple) and len(generated) == 2:
            generated = generated[0]
        if isinstance(generated, dict):
            generated = [generated]
        if not isinstance(generated, (list, tuple)) or len(generated) != 1:
            raise Qwen3ASRError(
                f"NVASR generate must return exactly one result for {stem}")
        result = generated[0]
        if not isinstance(result, dict):
            raise Qwen3ASRError("NVASR generate result must be an object")
        key = result.get("key", result.get("wav_file", result.get("audio")))
        if key is None:
            raise Qwen3ASRError(f"NVASR result has no key for {stem}")
        if Path(str(key)).stem != stem and str(key) != stem:
            raise Qwen3ASRError(f"NVASR result key/stem mismatch for {stem}: {key!r}")
        timeline = result.get("nvasr_candidate_timeline")
        if timeline is None:
            raise Qwen3ASRError(f"NVASR patched inference returned no candidate timeline for {stem}")
        if not isinstance(timeline, dict):
            raise Qwen3ASRError("NVASR candidate timeline must be an object")
        timeline = dict(timeline)
        timeline_stem = timeline.get("stem")
        if timeline_stem is not None and Path(str(timeline_stem)).stem != stem \
                and str(timeline_stem) != stem:
            raise Qwen3ASRError(
                f"NVASR candidate timeline stem mismatch for {stem}: {timeline_stem!r}")
        timeline["stem"] = stem
        return timeline

    def close(self) -> None:
        model = self.model
        for name in ("close", "release", "shutdown"):
            method = getattr(model, name, None)
            if callable(method):
                try:
                    method()
                except BaseException:
                    pass
                break
        self.model = None


def _load_nvasr(settings: dict[str, Any]) -> _NVASRAdapter:
    try:
        funasr = importlib.import_module("funasr")
        ctc = importlib.import_module("ctc_prealign")
    except ImportError:
        try:
            funasr = importlib.import_module("funasr")
            ctc = importlib.import_module("scripts.ctc_prealign")
        except ImportError as exc:
            raise Qwen3ASRError(
                "anchored_nvv NVASR runtime requires funasr and ctc_prealign") from exc
    auto_model = getattr(funasr, "AutoModel", None)
    if auto_model is None:
        raise Qwen3ASRError("installed funasr does not expose AutoModel")
    try:
        model = auto_model(
            model=str(settings["nvasr_model_path"]), device=settings["device"])
        patched = ctc.make_patched_inference(
            ref_texts={}, bias_value=settings["nvv_bias"],
            pause_threshold=settings["pause_threshold"],
            enable_nvv=True, reference_only=False)
        target = getattr(model, "model", None)
        if target is None:
            raise Qwen3ASRError("FunASR AutoModel has no model object to patch")
        target.inference = types.MethodType(patched, target)
    except Qwen3ASRError:
        _close_provider(locals().get("model"))
        raise
    except Exception as exc:
        _close_provider(locals().get("model"))
        raise Qwen3ASRError(f"FunASR NVASR initialization failed: {exc}") from exc
    return _NVASRAdapter(model)


def _call_backend(backend: Any, audio_paths: list[Path], settings: dict[str, Any]) -> Any:
    transcribe: Callable[..., Any] = getattr(backend, "transcribe", backend)
    # Keep this call intentionally explicit: timestamps are outside the v1
    # contract and must never be requested from the backend.
    return transcribe(
        audio=[str(path) for path in audio_paths],
        language=settings["language"],
        context=settings["context"],
        return_time_stamps=False,
    )


def _result_text_and_language(result: Any) -> tuple[str, str]:
    if isinstance(result, str):
        text = result
        language = ""
    elif isinstance(result, dict):
        text = result.get("text")
        language = result.get("language", result.get("detected_language", ""))
    else:
        text = getattr(result, "text", None)
        language = getattr(result, "language",
                           getattr(result, "detected_language", ""))
    if not isinstance(text, str) or not text.strip():
        raise TranscriptResultError("empty transcript")
    return text.strip(), language if isinstance(language, str) else ""


def _sanitize_message(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:500]


def _failure_record(stem: str, code: str, exc: BaseException) -> dict[str, str]:
    return {
        "stem": stem,
        "code": code,
        "exception_type": type(exc).__name__,
        "message": _sanitize_message(exc),
    }


def _checkpoint_payload(identity: dict[str, Any], successes: dict[str, dict[str, Any]],
                        failures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "successes": [successes[stem] for stem in sorted(successes)],
        "failures": [failures[stem] for stem in sorted(failures)],
        "success_stems": sorted(successes),
        "failed_stems": sorted(failures),
    }


def _output_file(output_root: Path, stem: str) -> Path:
    return output_root / "transcripts" / f"{stem}_qwen3.txt"


def _load_resume(output_root: Path, identity: dict[str, Any],
                 source_stems: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    checkpoint = output_root / "qwen3asr_checkpoint.json"
    known = {"transcripts", checkpoint.name, "qwen3asr_manifest.json", ".qwen3asr_run_receipt.json"}
    if output_root.exists():
        unexpected = sorted(child.name for child in output_root.iterdir()
                            if child.name not in known)
        if unexpected:
            raise Qwen3ASRError(
                f"qwen3asr output root contains unexpected artifacts: {unexpected}")
        transcripts = output_root / "transcripts"
        if transcripts.exists() and (transcripts.is_symlink() or not transcripts.is_dir()):
            raise Qwen3ASRError("qwen3asr transcripts namespace must be a regular directory")
    if not checkpoint.exists():
        if output_root.exists() and any(child.name in known for child in output_root.iterdir()):
            raise Qwen3ASRError(
                f"qwen3asr output has artifacts but no matching checkpoint: {output_root}")
        return {}, {}
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen3ASRError(f"cannot read qwen3asr checkpoint: {checkpoint}: {exc}") from exc
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("identity") != identity:
        raise Qwen3ASRError("qwen3asr checkpoint identity mismatch; recovery is required")
    successes = {row["stem"]: row for row in payload.get("successes", [])
                 if isinstance(row, dict) and isinstance(row.get("stem"), str)}
    failures = {row["stem"]: row for row in payload.get("failures", [])
                if isinstance(row, dict) and isinstance(row.get("stem"), str)}
    if (set(successes) | set(failures)) != source_stems or set(successes) & set(failures):
        raise Qwen3ASRError("qwen3asr checkpoint does not partition the frozen source stems")
    for stem, row in successes.items():
        path = _output_file(output_root, stem)
        if (row.get("path") != f"transcripts/{stem}_qwen3.txt"
                or path.is_symlink() or not path.is_file()
                or row.get("size") != path.stat().st_size
                or row.get("sha256") != _sha256_file(path)):
            raise Qwen3ASRError(
                f"qwen3asr success artifact missing or tampered before inference: {stem}")
    transcripts = output_root / "transcripts"
    if transcripts.exists():
        expected = {f"{stem}_qwen3.txt" for stem in successes}
        actual = {child.name for child in transcripts.iterdir()}
        if actual != expected:
            raise Qwen3ASRError(
                "qwen3asr transcripts namespace contains an unaccounted artifact")
    return successes, failures


def _chinese_lexical_units(text: str) -> list[str]:
    """Return the Qwen-authoritative lexical sequence for anchored_nvv.

    The anchored profile intentionally passes only CJK characters to the
    forced aligner.  Qwen punctuation remains in the raw transcript evidence
    but is excluded because the official aligner removes punctuation.
    """
    units = [char for char in text if (
        "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"
    )]
    if not units:
        raise TranscriptResultError("anchored_nvv Qwen transcript is not Chinese")
    punctuation = set("，。！？、；：…,.!?;:")
    invalid = [char for char in text if not char.isspace() and char not in units
               and char not in punctuation]
    if invalid:
        raise TranscriptResultError(
            "anchored_nvv Qwen transcript contains non-Chinese lexical material: "
            + "".join(invalid[:8])
        )
    return units


def _provider_callable(provider: Any, names: tuple[str, ...]) -> Callable[..., Any]:
    if provider is None:
        raise Qwen3ASRError(
            "anchored_nvv provider is not configured; production smoke must supply "
            "the forced-aligner and NVASR providers explicitly"
        )
    if callable(provider):
        return provider
    for name in names:
        candidate = getattr(provider, name, None)
        if callable(candidate):
            return candidate
    raise Qwen3ASRError(
        f"anchored_nvv provider must be callable or expose one of {names}"
    )


def _invoke_provider(provider: Any, names: tuple[str, ...], **kwargs: Any) -> Any:
    """Call a provider with only the keyword parameters it declares.

    This keeps the provider boundary small and makes fake providers useful in
    tests without swallowing exceptions raised by the provider itself.
    """
    target = _provider_callable(provider, names)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(**kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD
           for parameter in parameters.values()):
        return target(**kwargs)
    accepted = {
        name: value for name, value in kwargs.items()
        if name in parameters
    }
    return target(**accepted)


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Qwen3ASRError(f"anchored_nvv artifact is missing or unsafe: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_artifact_record(root: Path, record: dict[str, Any], *, stem: str,
                              phase: str) -> None:
    if not isinstance(record, dict):
        raise Qwen3ASRError(f"anchored_nvv {phase} record is malformed for {stem}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
        raise Qwen3ASRError(f"anchored_nvv {phase} artifact path is unsafe for {stem}")
    path = root / raw_path
    if path.is_symlink() or not path.is_file():
        raise Qwen3ASRError(f"anchored_nvv {phase} artifact is missing for {stem}")
    if record.get("size") != path.stat().st_size or record.get("sha256") != _sha256_file(path):
        raise Qwen3ASRError(f"anchored_nvv {phase} artifact is tampered for {stem}")


def _anchored_paths(root: Path, stem: str) -> dict[str, Path]:
    return {
        "anchor": root / "anchors" / f"{stem}.qwen3_forced_aligner.json",
        "candidate": root / "nvasr_candidates" / f"{stem}.candidate_timeline.json",
        "fused": root / "fused" / f"{stem}.anchored_nvv.json",
    }


def _anchored_checkpoint_payload(identity: dict[str, Any],
                                 records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": ANCHORED_CHECKPOINT_SCHEMA,
        "identity": identity,
        "records": [records[stem] for stem in sorted(records)],
    }


def _load_anchored_resume(root: Path, identity: dict[str, Any],
                          source_stems: set[str]) -> dict[str, dict[str, Any]]:
    checkpoint = root / "anchored_nvv_checkpoint.json"
    known = {
        "anchors", "nvasr_candidates", "fused", checkpoint.name,
        "anchored_nvv_manifest.json", ".anchored_nvv_run_receipt.json",
    }
    if root.exists():
        unexpected = sorted(child.name for child in root.iterdir()
                            if child.name not in known)
        if unexpected:
            raise Qwen3ASRError(
                f"anchored_nvv output root contains unexpected artifacts: {unexpected}")
        for namespace in ("anchors", "nvasr_candidates", "fused"):
            path = root / namespace
            if path.exists() and (path.is_symlink() or not path.is_dir()):
                raise Qwen3ASRError(f"anchored_nvv namespace must be a regular directory: {namespace}")
    if not checkpoint.exists():
        if root.exists() and any(root.iterdir()):
            raise Qwen3ASRError(
                "anchored_nvv output has artifacts but no matching checkpoint")
        return {}
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen3ASRError(f"cannot read anchored_nvv checkpoint: {exc}") from exc
    if payload.get("schema") != ANCHORED_CHECKPOINT_SCHEMA or payload.get("identity") != identity:
        raise Qwen3ASRError("anchored_nvv checkpoint identity mismatch; recovery is required")
    records: dict[str, dict[str, Any]] = {}
    for row in payload.get("records", []):
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
            raise Qwen3ASRError("anchored_nvv checkpoint contains a malformed record")
        stem = row["stem"]
        if stem in records or stem not in source_stems:
            raise Qwen3ASRError("anchored_nvv checkpoint does not match source stems")
        phases = row.get("phases", {})
        if not isinstance(phases, dict):
            raise Qwen3ASRError(f"anchored_nvv phases are malformed for {stem}")
        qwen_evidence = row.get("qwen")
        if qwen_evidence is not None:
            if (not isinstance(qwen_evidence, dict)
                    or not isinstance(qwen_evidence.get("text"), str)
                    or not isinstance(qwen_evidence.get("language"), str)
                    or not isinstance(qwen_evidence.get("units"), list)
                    or any(not isinstance(unit, str) for unit in qwen_evidence["units"])):
                raise Qwen3ASRError(f"anchored_nvv Qwen evidence is malformed for {stem}")
            if _chinese_lexical_units(qwen_evidence["text"]) != qwen_evidence["units"]:
                raise Qwen3ASRError(f"anchored_nvv Qwen lexical evidence is inconsistent for {stem}")
        for phase, record in phases.items():
            if phase not in {"anchor", "candidate", "fused"}:
                raise Qwen3ASRError(f"anchored_nvv checkpoint has unknown phase {phase}")
            _validate_artifact_record(root, record, stem=stem, phase=phase)
            expected_path = _anchored_paths(root, stem)[phase].relative_to(root).as_posix()
            if record.get("path") != expected_path:
                raise Qwen3ASRError(
                    f"anchored_nvv {phase} artifact path does not match stem {stem}")
            try:
                phase_payload = json.loads((root / record["path"]).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise Qwen3ASRError(
                    f"anchored_nvv {phase} artifact is not valid JSON for {stem}") from exc
            if phase == "anchor":
                if (phase_payload.get("schema") != ANCHOR_SCHEMA
                        or phase_payload.get("stem") != stem
                        or phase_payload.get("profile") != ANCHORED_PROFILE
                        or phase_payload.get("lexical_authority") != "qwen"
                        or phase_payload.get("timing_label") != "qwen3_forced_aligner"
                        or not isinstance(phase_payload.get("qwen_lexical_units"), list)
                        or not all(isinstance(unit, str)
                                   for unit in phase_payload["qwen_lexical_units"])):
                    raise Qwen3ASRError(f"anchored_nvv anchor evidence is malformed for {stem}")
                _chinese_lexical_units("".join(phase_payload.get("qwen_lexical_units", [])))
            elif phase == "candidate":
                _normalize_candidate_timeline(phase_payload, stem=stem)
            elif (phase_payload.get("schema") != FUSION_SCHEMA
                  or phase_payload.get("stem") != stem
                  or phase_payload.get("lexical_authority") != "qwen"
                  or phase_payload.get("lexical_timing_source") != "qwen3_forced_aligner"
                  or phase_payload.get("status") not in {"COMPLETE", "FAILED"}):
                raise Qwen3ASRError(f"anchored_nvv fused evidence is malformed for {stem}")
            if phase == "fused":
                conservation = phase_payload.get("candidate_conservation")
                accepted = phase_payload.get("accepted")
                rejected = phase_payload.get("rejected")
                valid_timing_sources = {
                    "nvasr_ctc_free_decode",
                    "nvasr_blank_pause_heuristic",
                }
                if (not isinstance(conservation, dict)
                        or conservation.get("exactly_once") is not True
                        or not isinstance(accepted, list)
                        or not isinstance(rejected, list)
                        or any(
                            not isinstance(candidate, dict)
                            or candidate.get("timing_source") not in valid_timing_sources
                            for candidate in [*accepted, *rejected]
                        )):
                    raise Qwen3ASRError(
                        f"anchored_nvv fused evidence is malformed for {stem}")
        records[stem] = row
    # Any file in an allowed namespace must be accounted for by the checkpoint.
    expected = {
        record["path"] for row in records.values()
        for record in row.get("phases", {}).values()
    }
    actual = {
        path.relative_to(root).as_posix()
        for namespace in ("anchors", "nvasr_candidates", "fused")
        if (root / namespace).exists()
        for path in (root / namespace).iterdir()
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise Qwen3ASRError("anchored_nvv artifact namespace contains an unaccounted artifact")
    return records


def _load_fusion_module() -> Any:
    try:
        return importlib.import_module("qwen3asr_fusion")
    except ModuleNotFoundError:
        return importlib.import_module("scripts.qwen3asr_fusion")


def _normalize_candidate_timeline(raw: Any, *, stem: str) -> dict[str, Any]:
    """Accept a canonical timeline or normalize raw CTC evidence lazily."""
    if isinstance(raw, dict) and "candidate_timeline" in raw:
        raw = raw["candidate_timeline"]
    if isinstance(raw, dict) and raw.get("schema") == TIMELINE_SCHEMA:
        timeline = dict(raw)
        timeline.setdefault("stem", stem)
    elif isinstance(raw, dict) and "frame_token_ids" in raw:
        try:
            ctc = importlib.import_module("ctc_prealign")
        except ModuleNotFoundError:
            ctc = importlib.import_module("scripts.ctc_prealign")
        timeline = ctc.extract_nvasr_candidate_timeline(
            raw["frame_token_ids"], raw.get("diagnostic_text", raw.get("diagnostic", "")),
            token_decoder=raw.get("token_decoder"),
            token_surfaces=raw.get("token_surfaces"), stem=stem,
        )
    else:
        raise Qwen3ASRError(
            f"NVASR provider must return {TIMELINE_SCHEMA} evidence for {stem}")
    if timeline.get("stem") != stem:
        raise Qwen3ASRError(f"NVASR candidate timeline stem mismatch for {stem}")
    fusion = _load_fusion_module()
    try:
        return fusion.validate_candidate_timeline(timeline)
    except Exception as exc:
        raise Qwen3ASRError(f"invalid NVASR candidate timeline for {stem}: {exc}") from exc


def _write_anchored_manifest_and_receipt(root: Path, identity: dict[str, Any],
                                         records: dict[str, dict[str, Any]],
                                         source_stems: list[str], argv: list[str]) -> int:
    success = sorted(stem for stem in source_stems
                     if "fused" in records.get(stem, {}).get("phases", {})
                     and records[stem].get("status") == "COMPLETE")
    failed = sorted(set(source_stems) - set(success))
    manifest = {
        "schema": ANCHORED_MANIFEST_SCHEMA,
        "identity": identity,
        "source_stems": source_stems,
        "success": success,
        "failed": failed,
        "records": [records[stem] for stem in source_stems if stem in records],
    }
    manifest_path = root / "anchored_nvv_manifest.json"
    checkpoint_path = root / "anchored_nvv_checkpoint.json"
    _atomic_write_json(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)
    complete = not failed
    receipt = {
        "schema": ANCHORED_RECEIPT_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": list(argv),
        "profile": ANCHORED_PROFILE,
        "status": "COMPLETE" if complete else "PARTIAL",
        "return_code": 0 if complete else 1,
        "identity": identity,
        "identity_digest": stable_json_digest(identity),
        "source_stems": source_stems,
        "source_stems_digest": stable_json_digest(source_stems),
        "success": success,
        "failed": failed,
        "manifest": {"path": manifest_path.relative_to(root).as_posix(),
                     "sha256": manifest_sha256},
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    _atomic_write_json(root / ".anchored_nvv_run_receipt.json", receipt)
    print(f"anchored_nvv {receipt['status']}: {success} success, {failed} failed -> {root}")
    return int(receipt["return_code"])


def run_anchored_nvv(
    cfg: dict[str, Any], project_root: Path, *,
    data_dir: Path | None = None, output_dir: Path | None = None,
    backend: Any | None = None,
    forced_aligner_backend: Any | None = None,
    nvasr_backend: Any | None = None,
    qwen_asr_version: str | None = None,
    device_override: str | None = None,
    argv: list[str] | None = None,
    profile_override: str | None = None,
) -> int:
    """Run the explicit Chinese Qwen/NVASR anchored profile.

    Heavy providers are deliberately phase-serial: all Qwen lexical work is
    completed before forced alignment, all forced alignment before NVASR, and
    fusion runs only after both evidence streams are persisted.  Provider
    objects are seams for the isolated runtime and tests; they are never
    imported at module load time.
    """
    settings = _validate_runtime_config(
        cfg, device_override=device_override, profile_override=profile_override)
    if settings["profile"] != ANCHORED_PROFILE:
        raise Qwen3ASRError("run_anchored_nvv requires qwen3asr.profile=anchored_nvv")
    settings["model_path"] = _resolve_path(settings["model_path"], project_root)
    forced_path = _resolve_path(settings["forced_aligner_model_path"], project_root)
    nvasr_path = _resolve_path(settings["nvasr_model_path"], project_root)
    settings["forced_aligner_model_path"] = forced_path
    settings["nvasr_model_path"] = nvasr_path
    model_info, model_files = _model_identity(settings["model_path"])
    aligner_info, aligner_files = _model_identity(
        forced_path, label="qwen3asr forced_aligner")
    nvasr_info, nvasr_files = _model_identity(nvasr_path, label="qwen3asr nvasr")
    source_root = data_dir or _resolve_path(cfg.get("data_dir", ""), project_root)
    sources = _source_inventory(source_root)
    if not sources:
        raise Qwen3ASRError("anchored_nvv source inventory is empty; no output was created")
    version = qwen_asr_version or _installed_qwen_version()
    if version != SUPPORTED_QWEN_ASR_VERSION:
        raise Qwen3ASRError(
            f"unsupported qwen-asr version {version!r}; v1 requires "
            f"qwen-asr=={SUPPORTED_QWEN_ASR_VERSION}")
    source_stems = [record["stem"] for record in sources]
    identity = {
        "schema": ANCHORED_PROFILE,
        "profile": ANCHORED_PROFILE,
        "source_stems": source_stems,
        "source_stems_digest": stable_json_digest(source_stems),
        "wav_files": sources,
        "wav_files_digest": stable_json_digest(sources),
        "model": {**model_info, "files": model_files},
        "forced_aligner_model": {**aligner_info, "files": aligner_files},
        "nvasr_model": {**nvasr_info, "files": nvasr_files},
        "qwen_model_tree_digest": model_info["tree_digest"],
        "forced_aligner_model_tree_digest": aligner_info["tree_digest"],
        "nvasr_model_tree_digest": nvasr_info["tree_digest"],
        "qwen_asr_version": version,
        "anchor_schema": ANCHOR_SCHEMA,
        "timeline_schema": TIMELINE_SCHEMA,
        "fusion_schema": FUSION_SCHEMA,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "zero_width_policy_version": ZERO_WIDTH_POLICY_VERSION,
        "backend": settings["backend"],
        "device": settings["device"],
        "dtype": settings["dtype"],
        "language": settings["language"],
        "context_hash": stable_json_digest(settings["context"]),
        "batch_size": settings["batch_size"],
        "max_new_tokens": settings["max_new_tokens"],
        "timing_label": "qwen3_forced_aligner",
        "lexical_authority": "qwen",
        "nvv_bias": settings["nvv_bias"],
        "pause_threshold": settings["pause_threshold"],
    }
    root_candidate = output_dir or _resolve_path(
        _qcfg(cfg).get("output_dir", "output/qwen3asr"), project_root)
    if root_candidate.is_symlink():
        raise Qwen3ASRError(f"anchored_nvv output root must not be a symlink: {root_candidate}")
    root = root_candidate.resolve(strict=False)
    records = _load_anchored_resume(root, identity, set(source_stems))
    for stem in source_stems:
        records.setdefault(stem, {"stem": stem, "status": "PENDING", "phases": {}})
    pending_qwen = [stem for stem in source_stems
                    if "anchor" not in records[stem].get("phases", {})
                    and "qwen" not in records[stem]]
    root.mkdir(parents=True, exist_ok=True)
    for namespace in ("anchors", "nvasr_candidates", "fused"):
        (root / namespace).mkdir(exist_ok=True)
    source_root = source_root.resolve()
    paths_by_stem = {
        record["stem"]: source_root / record["wav_relative_path"]
        for record in sources
    }

    def save_checkpoint() -> None:
        _atomic_write_json(
            root / "anchored_nvv_checkpoint.json",
            _anchored_checkpoint_payload(identity, records),
        )

    def fail(stem: str, code: str, exc: BaseException) -> None:
        records[stem]["status"] = "FAILED"
        records[stem]["failure"] = _failure_record(stem, code, exc)

    # Phase 1: persist Qwen lexical evidence in the checkpoint.  It is not a
    # standalone artifact: the locked namespace exposes only the anchor,
    # candidate, and fused evidence files.
    qwen_provider = backend
    qwen_owned = False
    if pending_qwen and qwen_provider is None:
        qwen_provider, loaded_version = _load_backend(settings)
        qwen_owned = True
        if qwen_provider is None:
            raise Qwen3ASRError("Qwen provider factory returned None")
        if loaded_version != version:
            _close_provider(qwen_provider)
            raise Qwen3ASRError(
                f"qwen-asr version changed while loading: {version!r} -> {loaded_version!r}")
    try:
        for start in range(0, len(pending_qwen), settings["batch_size"]):
            batch_stems = pending_qwen[start:start + settings["batch_size"]]
            batch_paths = [paths_by_stem[stem] for stem in batch_stems]
            try:
                raw_results = _call_backend(qwen_provider, batch_paths, settings)
                if isinstance(raw_results, (str, dict)) or not hasattr(raw_results, "__len__"):
                    raw_results = [raw_results]
                if len(raw_results) != len(batch_stems):
                    raise ValueError(
                        f"expected {len(batch_stems)} result(s), got {len(raw_results)}")
            except Exception as exc:
                for stem in batch_stems:
                    fail(stem, "qwen_backend_exception", exc)
                save_checkpoint()
                continue
            for stem, raw_result in zip(batch_stems, raw_results):
                try:
                    text, detected_language = _result_text_and_language(raw_result)
                    units = _chinese_lexical_units(text)
                    records[stem]["qwen"] = {
                        "text": text, "language": detected_language, "units": units,
                    }
                    records[stem].pop("failure", None)
                    records[stem]["status"] = "PENDING"
                except Exception as exc:
                    fail(stem, "qwen_backend_exception", exc)
            save_checkpoint()
    finally:
        if qwen_owned:
            _close_provider(qwen_provider)
            qwen_provider = None

    # Phase 2: run the explicit forced aligner only after every pending Qwen
    # batch has completed and its lexical evidence is checkpointed.
    pending_aligner = [stem for stem in source_stems
                       if "anchor" not in records[stem].get("phases", {})
                       and "qwen" in records[stem]]
    forced_provider = forced_aligner_backend
    forced_owned = False
    if pending_aligner and forced_provider is None:
        forced_provider = _load_forced_aligner(settings)
        forced_owned = True
        if forced_provider is None:
            raise Qwen3ASRError("forced-aligner provider factory returned None")
    try:
        for stem in pending_aligner:
            try:
                qwen_evidence = records[stem]["qwen"]
                forced = _invoke_provider(
                    forced_provider, ("align", "forced_align", "run"),
                    audio=paths_by_stem[stem], audio_path=paths_by_stem[stem],
                    text=qwen_evidence["text"], qwen_text=qwen_evidence["text"],
                    language="Chinese", units=qwen_evidence["units"],
                    qwen_lexical_units=qwen_evidence["units"], stem=stem,
                    model_path=forced_path, device=settings["device"],
                )
                if isinstance(forced, dict):
                    forced = forced.get("forced_aligner_items", forced.get("items", forced))
                if not isinstance(forced, (list, tuple)):
                    raise ValueError("forced aligner must return a sequence of timed items")
                anchor = {
                    "schema": ANCHOR_SCHEMA,
                    "profile": ANCHORED_PROFILE,
                    "stem": stem,
                    "qwen_text": qwen_evidence["text"],
                    "qwen_language": qwen_evidence["language"],
                    "qwen_lexical_units": qwen_evidence["units"],
                    "lexical_authority": "qwen",
                    "timing_label": "qwen3_forced_aligner",
                    "forced_aligner_items": list(forced),
                }
                target = _anchored_paths(root, stem)["anchor"]
                _atomic_write_json(target, anchor)
                records[stem]["phases"]["anchor"] = _artifact_record(root, target)
                records[stem].pop("failure", None)
                records[stem]["status"] = "PENDING"
            except Exception as exc:
                fail(stem, "forced_aligner_exception", exc)
            save_checkpoint()
    finally:
        if forced_owned:
            _close_provider(forced_provider)
            forced_provider = None

    # Phase 3: NVASR candidate extraction, after all lexical/aligner work.
    pending_candidates = [stem for stem in source_stems
                          if "anchor" in records[stem].get("phases", {})
                          and "candidate" not in records[stem].get("phases", {})]
    nvasr_provider = nvasr_backend
    nvasr_owned = False
    if pending_candidates and nvasr_provider is None:
        nvasr_provider = _load_nvasr(settings)
        nvasr_owned = True
        if nvasr_provider is None:
            raise Qwen3ASRError("NVASR provider factory returned None")
    try:
        for stem in pending_candidates:
            try:
                anchor_payload = json.loads(
                    _anchored_paths(root, stem)["anchor"].read_text(encoding="utf-8"))
                raw = _invoke_provider(
                    nvasr_provider, ("extract_candidate_timeline", "transcribe", "infer", "run"),
                    audio=paths_by_stem[stem], audio_path=paths_by_stem[stem], stem=stem,
                    model_path=settings["nvasr_model_path"],
                    device=settings["device"],
                    qwen_lexical_units=anchor_payload["qwen_lexical_units"],
                )
                timeline = _normalize_candidate_timeline(raw, stem=stem)
                target = _anchored_paths(root, stem)["candidate"]
                _atomic_write_json(target, timeline)
                records[stem]["phases"]["candidate"] = _artifact_record(root, target)
                records[stem].pop("failure", None)
                records[stem]["status"] = "PENDING"
            except Exception as exc:
                fail(stem, "nvasr_candidate_exception", exc)
            save_checkpoint()
    finally:
        if nvasr_owned:
            _close_provider(nvasr_provider)
            nvasr_provider = None

    # Phase 4: provider-neutral, fail-closed fusion.
    fusion = _load_fusion_module()
    for stem in source_stems:
        row = records[stem]
        if not {"anchor", "candidate"}.issubset(row.get("phases", {})):
            continue
        if "fused" in row.get("phases", {}):
            continue
        try:
            paths = _anchored_paths(root, stem)
            anchor = json.loads(paths["anchor"].read_text(encoding="utf-8"))
            timeline = json.loads(paths["candidate"].read_text(encoding="utf-8"))
            result = fusion.fuse_qwen_nvasr_candidates(
                anchor["qwen_lexical_units"], anchor["forced_aligner_items"], timeline)
            target = paths["fused"]
            _atomic_write_json(target, result)
            row["phases"]["fused"] = _artifact_record(root, target)
            if result.get("status") != "COMPLETE":
                raise Qwen3ASRError("candidate rejection made anchored_nvv stem FAILED")
            row["status"] = "COMPLETE"
            row.pop("failure", None)
        except Exception as exc:
            fail(stem, "fusion_rejected_or_invalid", exc)
        save_checkpoint()

    save_checkpoint()
    return _write_anchored_manifest_and_receipt(
        root, identity, records, source_stems, list(argv or []))


def check_qwen3asr(cfg: dict[str, Any], project_root: Path, *,
                   device_override: str | None = None,
                   profile_override: str | None = None) -> tuple[bool, str]:
    """Validate capability and local model availability without creating output."""
    try:
        settings = _validate_runtime_config(
            cfg, device_override=device_override, profile_override=profile_override)
        settings["model_path"] = _resolve_path(settings["model_path"], project_root)
        model_info, _ = _model_identity(settings["model_path"])
        aligner_info = None
        nvasr_info = None
        if settings["profile"] == ANCHORED_PROFILE:
            forced_path = _resolve_path(settings["forced_aligner_model_path"], project_root)
            aligner_info, _ = _model_identity(
                forced_path, label="qwen3asr forced_aligner")
            nvasr_path = _resolve_path(settings["nvasr_model_path"], project_root)
            nvasr_info, _ = _model_identity(nvasr_path, label="qwen3asr nvasr")
        version = _installed_qwen_version()
        if version != SUPPORTED_QWEN_ASR_VERSION:
            raise Qwen3ASRError(
                f"unsupported qwen-asr version {version!r}; v1 requires "
                f"qwen-asr=={SUPPORTED_QWEN_ASR_VERSION}")
        package = importlib.import_module("qwen_asr")
        model_class = getattr(package, "Qwen3ASRModel", None)
        if model_class is None or not hasattr(model_class, "from_pretrained"):
            raise Qwen3ASRError("qwen_asr lacks Qwen3ASRModel.from_pretrained")
        torch = importlib.import_module("torch")
        _validate_torch_capability(settings, torch)
        detail = (f"qwen3asr check OK: profile={settings['profile']}, backend={BACKEND_NAME}, "
                      f"qwen-asr=={SUPPORTED_QWEN_ASR_VERSION}, "
                      f"device={settings['device']}, dtype={settings['dtype']}, "
                      f"model={model_info['path']}")
        if aligner_info is not None:
            detail += f", forced_aligner_model={aligner_info['path']}"
        if nvasr_info is not None:
            detail += f", nvasr_model={nvasr_info['path']}"
        return True, detail
    except (Qwen3ASRError, ImportError) as exc:
        return False, str(exc)


def run_qwen3asr(cfg: dict[str, Any], project_root: Path, *,
                 data_dir: Path | None = None, output_dir: Path | None = None,
                 backend: Any | None = None, qwen_asr_version: str | None = None,
                 device_override: str | None = None,
                 argv: list[str] | None = None,
                 forced_aligner_backend: Any | None = None,
                 nvasr_backend: Any | None = None,
                 profile_override: str | None = None) -> int:
    """Run or resume the isolated transcript-only mode.

    ``backend`` and ``qwen_asr_version`` are test seams; production uses the
    official lazy-loaded backend above.
    """
    settings = _validate_runtime_config(
        cfg, device_override=device_override, profile_override=profile_override)
    if settings["profile"] == ANCHORED_PROFILE:
        return run_anchored_nvv(
            cfg, project_root, data_dir=data_dir, output_dir=output_dir,
            backend=backend, forced_aligner_backend=forced_aligner_backend,
            nvasr_backend=nvasr_backend, qwen_asr_version=qwen_asr_version,
            device_override=device_override, argv=argv,
            profile_override=ANCHORED_PROFILE,
        )
    settings["model_path"] = _resolve_path(settings["model_path"], project_root)
    source_root = data_dir or _resolve_path(cfg.get("data_dir", ""), project_root)
    sources = _source_inventory(source_root)
    if not sources:
        raise Qwen3ASRError("qwen3asr source inventory is empty; no output was created")
    model_info, model_files = _model_identity(settings["model_path"])
    version = qwen_asr_version or _installed_qwen_version()
    if version != SUPPORTED_QWEN_ASR_VERSION:
        raise Qwen3ASRError(
            f"unsupported qwen-asr version {version!r}; v1 requires "
            f"qwen-asr=={SUPPORTED_QWEN_ASR_VERSION}")
    source_stems = [record["stem"] for record in sources]
    identity = {
        "schema": SCHEMA,
        "source_stems": source_stems,
        "source_stems_digest": stable_json_digest(source_stems),
        "wav_files": sources,
        "wav_files_digest": stable_json_digest(sources),
        "model": {**model_info, "files": model_files},
        "qwen_asr_version": version,
        "backend": settings["backend"],
        "device": settings["device"],
        "dtype": settings["dtype"],
        "language": settings["language"],
        "context_hash": stable_json_digest(settings["context"]),
        "batch_size": settings["batch_size"],
        "max_new_tokens": settings["max_new_tokens"],
    }
    root_candidate = output_dir or _resolve_path(
        _qcfg(cfg).get("output_dir", "output/qwen3asr"), project_root)
    if root_candidate.is_symlink():
        raise Qwen3ASRError(
            f"qwen3asr output root must not be a symlink: {root_candidate}")
    root = root_candidate.resolve(strict=False)
    successes, failures = _load_resume(root, identity, set(source_stems))
    pending = [stem for stem in source_stems if stem not in successes]
    # Identity and every prior success artifact are verified before weights
    # are loaded.  A complete resume therefore requires no backend at all.
    if pending and backend is None:
        backend, loaded_version = _load_backend(settings)
        if loaded_version != version:
            raise Qwen3ASRError(
                f"qwen-asr version changed while loading: {version!r} -> {loaded_version!r}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    # ``sources_root`` is derived once after source validation to avoid any
    # backend access to a mutable or differently resolved input root.
    sources_root = source_root.resolve()
    paths_by_stem = {
        record["stem"]: sources_root / record["wav_relative_path"]
        for record in sources
    }
    for start in range(0, len(pending), settings["batch_size"]):
        batch_stems = pending[start:start + settings["batch_size"]]
        batch_paths = [paths_by_stem[stem] for stem in batch_stems]
        try:
            raw_results = _call_backend(backend, batch_paths, settings)
        except Exception as exc:
            for stem in batch_stems:
                failures[stem] = _failure_record(stem, "backend_exception", exc)
                successes.pop(stem, None)
        else:
            if isinstance(raw_results, (str, dict)) or not hasattr(raw_results, "__len__"):
                raw_results = [raw_results]
            if len(raw_results) != len(batch_stems):
                exc = ValueError(
                    f"expected {len(batch_stems)} result(s), got {len(raw_results)}")
                for stem in batch_stems:
                    failures[stem] = _failure_record(stem, "batch_length_mismatch", exc)
                    successes.pop(stem, None)
            else:
                for stem, raw_result in zip(batch_stems, raw_results):
                    try:
                        text, detected_language = _result_text_and_language(raw_result)
                        payload = (text + "\n").encode("utf-8")
                        target = _output_file(root, stem)
                        _atomic_write_bytes(target, payload)
                        successes[stem] = {
                            "stem": stem, "path": f"transcripts/{stem}_qwen3.txt",
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "language": detected_language,
                        }
                        failures.pop(stem, None)
                    except Exception as exc:
                        failures[stem] = _failure_record(
                            stem, "invalid_transcript", exc)
                        successes.pop(stem, None)
        _atomic_write_json(root / "qwen3asr_checkpoint.json",
                           _checkpoint_payload(identity, successes, failures))

    # Also refresh the atomic checkpoint for a fully-resumed run that had no
    # pending batches.
    _atomic_write_json(root / "qwen3asr_checkpoint.json",
                       _checkpoint_payload(identity, successes, failures))

    success_stems = sorted(successes)
    failed_stems = sorted(set(source_stems) - set(success_stems))
    if any(stem not in failures for stem in failed_stems):
        raise Qwen3ASRError("qwen3asr internal accounting error: missing failure record")
    success_records = [successes[stem] for stem in success_stems]
    failed_records = [failures[stem] for stem in failed_stems]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "identity": identity,
        "source_stems": source_stems,
        "success": success_stems,
        "failed": failed_stems,
        "success_records": success_records,
        "failed_records": failed_records,
        "global_reasons": [],
    }
    manifest_path = root / "qwen3asr_manifest.json"
    checkpoint_path = root / "qwen3asr_checkpoint.json"
    _atomic_write_json(manifest_path, manifest)
    complete = not failed_stems
    manifest_sha256 = _sha256_file(manifest_path)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": list(argv or []),
        "status": "COMPLETE" if complete else "PARTIAL",
        "return_code": 0 if complete else 1,
        "identity": identity,
        "identity_digest": stable_json_digest(identity),
        "model": {
            "path": model_info["path"],
            "tree_digest": model_info["tree_digest"],
            "file_count": len(model_files),
        },
        "source_count": len(source_stems),
        "success_count": len(success_stems),
        "failed_count": len(failed_stems),
        "source_stems": source_stems,
        "source_stems_digest": stable_json_digest(source_stems),
        "success_stems": success_stems,
        "success_stems_digest": stable_json_digest(success_stems),
        "failed_stems": failed_stems,
        "failed_stems_digest": stable_json_digest(failed_stems),
        "transcript_records_digest": stable_json_digest(success_records),
        "manifest": {"path": manifest_path.relative_to(root).as_posix(),
                     "sha256": manifest_sha256},
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    _atomic_write_json(root / ".qwen3asr_run_receipt.json", receipt)
    print(f"qwen3asr {receipt['status']}: {success_stems} success, {failed_stems} failed -> {root}")
    return int(receipt["return_code"])


def _configured_python(args: Any, cfg: dict[str, Any], project_root: Path) -> Path | None:
    raw = getattr(args, "python", None) or _qcfg(cfg).get("python")
    if not raw:
        return None
    candidate = _resolve_path(str(raw), project_root)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Qwen3ASRError(f"configured qwen3asr Python does not exist: {candidate}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise Qwen3ASRError(
            f"configured qwen3asr Python is not an executable file: {resolved}")
    current = Path(sys.executable).resolve(strict=True)
    return None if resolved == current else resolved


def run_cli(args: Any, cfg: dict[str, Any], project_root: Path, *, check: bool = False) -> int:
    forbidden = []
    for name, flag in (("step", "--step"), ("skip_to", "--skip-to"),
                       ("stop_after", "--stop-after"), ("scan_only", "--scan-only"),
                       ("ctc_ready", "--ctc-ready"), ("force", "--force"),
                       ("overwrite", "--overwrite"), ("workspace", "--workspace")):
        value = getattr(args, name, None)
        if value not in (None, False, ""):
            forbidden.append(flag)
    forbidden.extend(
        f"--skip-{name[5:]}" for name in vars(args)
        if name.startswith("skip_") and getattr(args, name, False)
    )
    if forbidden:
        print("ERROR: qwen3asr forbids pipeline flags: " + ", ".join(sorted(set(forbidden))))
        return 1
    try:
        target_python = _configured_python(args, cfg, project_root)
        if target_python is not None:
            return int(subprocess.run([str(target_python), *sys.argv]).returncode)
        device_override = getattr(args, "device", None)
        if check:
            ok, message = check_qwen3asr(
                cfg, project_root, device_override=device_override,
                profile_override=getattr(args, "qwen3asr_profile", None))
            print(message)
            return 0 if ok else 1
        data_dir = _resolve_path(args.data_dir, project_root) if args.data_dir else None
        output_dir = _resolve_path(args.output_dir, project_root) if args.output_dir else None
        return run_qwen3asr(
            cfg, project_root, data_dir=data_dir, output_dir=output_dir,
            device_override=device_override, argv=list(sys.argv),
            profile_override=getattr(args, "qwen3asr_profile", None))
    except (OSError, ValueError, Qwen3ASRError) as exc:
        print(f"ERROR: qwen3asr: {exc}")
        return 1
