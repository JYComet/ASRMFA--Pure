#!/usr/bin/env python3
"""Qwen3 HF lexical prealign producer for the existing MFA route.

The producer keeps the existing six-file CTC bundle contract so the remaining
normalize/adjust/MFA/postprocess stages do not need a second pipeline.  Qwen
ForcedAligner supplies lexical spans only; MFA remains the phoneme aligner.
NVV discovery is intentionally absent from this provider.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from qwen3_timestamp_normalization import (
        normalize_qwen_input_text,
        normalize_timestamps,
        SCHEMA as TIMESTAMP_SCHEMA,
    )
except ModuleNotFoundError:
    from .qwen3_timestamp_normalization import (
        normalize_qwen_input_text,
        normalize_timestamps,
        SCHEMA as TIMESTAMP_SCHEMA,
    )

try:
    from english_units import parse_english_units, project_authority_semantics
    from pipeline_utils import validate_ctc_transcript_bundle, CTC_SUFFIXES
except ModuleNotFoundError:
    from .english_units import parse_english_units, project_authority_semantics
    from .pipeline_utils import validate_ctc_transcript_bundle, CTC_SUFFIXES

try:
    from pipeline_utils import (compute_model_tree_digest,
                                make_ctc_normalization_marker,
                                make_pipeline_accounting_receipt,
                                make_pipeline_run_id,
                                read_wav_metadata,
                                write_ctc_run_receipt,
                                write_pipeline_accounting_receipt)
except ModuleNotFoundError:
    from .pipeline_utils import (compute_model_tree_digest,
                                 make_ctc_normalization_marker,
                                 make_pipeline_accounting_receipt,
                                 make_pipeline_run_id,
                                 read_wav_metadata,
                                 write_ctc_run_receipt,
                                 write_pipeline_accounting_receipt)

try:
    from qwen3_hf_backend import (Qwen3HFError, Qwen3HFSettings,
                                   load_backend, runtime_capabilities, validate_settings)
except ModuleNotFoundError:
    from .qwen3_hf_backend import (Qwen3HFError, Qwen3HFSettings,
                                    load_backend, runtime_capabilities, validate_settings)


class Qwen3PrealignError(RuntimeError):
    """A malformed Qwen prealign configuration or output."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _duration(path: Path) -> float:
    try:
        meta = read_wav_metadata(path)
        frames, rate = int(meta["frames"]), int(meta["sample_rate"])
    except Exception:
        with wave.open(str(path), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
    if frames < 1 or rate < 1:
        raise Qwen3PrealignError(f"empty or invalid WAV: {path}")
    return frames / rate


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF)


def _is_punct(text: str) -> bool:
    return bool(text) and all(not ch.isalnum() and not _is_cjk(ch)
                              and not ch.isspace() for ch in text)


def _lexical_text(text: str) -> str:
    """Remove punctuation while retaining CJK and whitespace for alignment."""
    return "".join(ch for ch in text if ch.isspace() or ch.isalnum() or _is_cjk(ch))


def _pinyin(unit: str) -> str:
    if not unit:
        raise Qwen3PrealignError("empty lexical unit")
    if all(_is_cjk(ch) for ch in unit):
        try:
            from pypinyin import Style, lazy_pinyin
            values = lazy_pinyin(unit, style=Style.TONE3,
                                 neutral_tone_with_five=True)
            return " ".join(values)
        except ImportError as exc:
            raise Qwen3PrealignError("pypinyin is required for qwen3_hf") from exc
    return unit.strip()


def contextual_pinyin(text: str) -> list[str]:
    """Read CJK phrases without joining across punctuation or English."""
    from pypinyin import Style, lazy_pinyin
    values = []
    for phrase in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", text):
        values.extend(lazy_pinyin(phrase, style=Style.TONE3,
                                  neutral_tone_with_five=True))
    return values


def reproject_pinyin_rows(rows: list[dict], source_text: str) -> list[dict]:
    """Correct only derived readings, keeping Qwen text/timing immutable."""
    plan = _lexical_plan(source_text)
    if len(rows) != len(plan):
        raise Qwen3PrealignError("pinyin projection lexical count mismatch")
    readings = iter(contextual_pinyin(source_text))
    result = []
    for row, item in zip(rows, plan):
        if row.get("provider") != "qwen3_hf" or row.get("unit") != item["unit"]:
            raise Qwen3PrealignError("pinyin projection source unit mismatch")
        projected = dict(row)
        word = (item["authority"].alignment_token if item["authority"]
                else next(readings))
        if word != row.get("word"):
            projected["word"] = word
            projected["pinyin_projection"] = {
                "schema": "qwen3-contextual-pinyin-v1",
                "original_word": row["word"], "normalized_word": word,
                "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                "policy": "preserve_punctuation_and_english_phrase_boundaries",
            }
        result.append(projected)
    return result


def _fallback_units(text: str) -> list[str]:
    """Build the expected lexical sequence for strict aligner cardinality."""
    units: list[str] = []
    english: list[str] = []
    for char in text:
        if _is_cjk(char):
            if english:
                units.append("".join(english)); english = []
            units.append(char)
        elif char.isalnum() or char in "'-_":
            english.append(char)
        elif english:
            units.append("".join(english)); english = []
    if english:
        units.append("".join(english))
    return units


def _source_wavs(audio_dir: Path) -> list[Path]:
    if not audio_dir.is_dir() or audio_dir.is_symlink():
        raise Qwen3PrealignError(f"invalid audio root: {audio_dir}")
    files = sorted((p for p in audio_dir.rglob("*")
                    if p.suffix.lower() == ".wav" and p.is_file()), key=lambda p: p.stem)
    if any(p.is_symlink() for p in files):
        raise Qwen3PrealignError("source WAV symlinks are not supported")
    stems = [p.stem for p in files]
    if len(stems) != len(set(stems)):
        raise Qwen3PrealignError("duplicate WAV stems in qwen3_hf source")
    return files


def _select_wavs(files: list[Path], section: dict[str, Any],
                 project_root: Path) -> list[Path]:
    """Apply the same frozen stem selector semantics as the legacy producer."""
    stems_file = section.get("stems_file")
    if stems_file:
        path = Path(str(stems_file))
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file():
            raise Qwen3PrealignError(f"qwen3_hf stems_file is missing: {path}")
        selected = {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")}
        missing = selected - {p.stem for p in files}
        if missing:
            raise Qwen3PrealignError(f"selected source WAVs are missing: {sorted(missing)}")
        if section.get("offset", 0) or section.get("limit", 0):
            raise Qwen3PrealignError("stems_file cannot be combined with limit/offset")
        files = [path for path in files if path.stem in selected]
    offset = int(section.get("offset", 0) or 0)
    limit = int(section.get("limit", 0) or 0)
    if offset < 0 or limit < 0:
        raise Qwen3PrealignError("qwen3_hf offset/limit must be non-negative")
    if offset:
        files = files[offset:]
    if limit:
        files = files[:limit]
    return files


def _references(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*.txt"), key=lambda p: p.as_posix()):
        if path.is_symlink() or path.stem in result:
            raise Qwen3PrealignError(f"ambiguous/symlink reference: {path}")
        result[path.stem] = path.read_text(encoding="utf-8-sig").strip()
    return result


def _lexical_plan(text: str) -> list[dict[str, Any]]:
    """Use the shared MFA lexical contract before requesting model times."""
    authorities = parse_english_units(text)
    plan = []
    for item in project_authority_semantics(text):
        kind, surface = item["kind"], item["surface"]
        if kind == "nvv":
            raise Qwen3PrealignError("qwen3_hf cannot align reference NVV labels")
        if kind == "cjk":
            plan.append({"unit": surface, "authority": None})
        elif kind == "english":
            authority = authorities[item["reference_ordinal"]]
            plan.append({"unit": surface.replace("-", ""), "authority": authority})
        elif kind == "other" and any(ch.isalnum() for ch in surface):
            raise Qwen3PrealignError(
                f"unsupported MFA lexical unit {surface!r}; use Chinese characters or English words")
    if not plan:
        raise Qwen3PrealignError("no spoken lexical units in transcript")
    return plan


def _write_textgrid(path: Path, duration: float, rows: list[dict[str, Any]]) -> None:
    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for row in rows:
        start, end = row["start_s"], row["end_s"]
        if start < cursor - 1e-9:
            raise Qwen3PrealignError("overlapping Qwen lexical intervals")
        if start > cursor:
            intervals.append((cursor, start, ""))
        intervals.append((start, end, row["word"]))
        cursor = end
    # TextGrid boundaries are serialized to six decimal places.  A smaller
    # floating-point remainder would round both endpoints to the same value
    # and create an invalid zero-duration trailing interval.
    if duration - cursor >= 1e-6:
        intervals.append((cursor, duration, ""))
    if not intervals:
        raise Qwen3PrealignError("Qwen alignment produced no lexical interval")
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0", f"xmax = {duration:.6f}", "tiers? <exists>",
        "size = 2", "item []:", "    item [1]:",
        '        class = "IntervalTier"', '        name = "words"',
        "        xmin = 0", f"        xmax = {duration:.6f}",
        f"        intervals: size = {len(intervals)}",
    ]
    for index, (start, end, text) in enumerate(intervals, 1):
        lines.extend([f"        intervals [{index}]:", f"            xmin = {start:.6f}",
                      f"            xmax = {end:.6f}",
                      f'            text = "{text.replace(chr(34), chr(34) * 2)}"'])
    lines.extend(["    item [2]:", '        class = "IntervalTier"',
                  '        name = "pauses"', "        xmin = 0",
                  f"        xmax = {duration:.6f}",
                  '        intervals: size = 1', "        intervals [1]:",
                  "            xmin = 0", f"            xmax = {duration:.6f}",
                  '            text = ""'])
    _atomic_text(path, "\n".join(lines) + "\n")


def _model_evidence(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_dir() or path.is_symlink():
        raise Qwen3PrealignError(f"qwen3_hf model tree is missing: {path}")
    try:
        digest, files = compute_model_tree_digest(path)
        if not files:
            raise ValueError("empty model directory")
        return digest, files
    except (OSError, ValueError) as exc:
        raise Qwen3PrealignError(f"cannot fingerprint model tree {path}: {exc}") from exc


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _artifact_digest(output_dir: Path, stems: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    suffixes = (*CTC_SUFFIXES, "_ref.txt")
    for stem in sorted(stems):
        for suffix in suffixes:
            path = output_dir / f"{stem}{suffix}"
            if not path.is_file() or path.is_symlink():
                rows.append({"name": path.name, "sha256": None})
            else:
                rows.append({"name": path.name, "sha256": _sha256(path)})
    for name in ("manifest.json", ".ctc_normalized"):
        path = output_dir / name
        rows.append({"name": name, "sha256": _sha256(path) if path.is_file() and not path.is_symlink() else None})
    return _stable_digest(rows)


def _identity_payload(*, settings: Qwen3HFSettings, runtime: dict[str, Any],
                      asr_digest: str, aligner_digest: str,
                      input_digest: str, reference_digest: str,
                      output_digest: str, stems: list[str]) -> dict[str, Any]:
    return {
        "schema": "qwen3-hf-prealign-identity-v1",
        "provider": "qwen3_hf",
        "runtime": runtime,
        "models": {"asr_tree_digest": asr_digest,
                    "forced_aligner_tree_digest": aligner_digest},
        "settings": {"model_path": str(settings.model_path),
                      "forced_aligner_model_path": str(settings.forced_aligner_model_path),
                      "device": settings.device, "dtype": settings.dtype,
                      "language": settings.language,
                      "max_new_tokens": settings.max_new_tokens,
                      "batch_size": settings.batch_size,
                      "context": settings.context},
        "forced_aligner_device": settings.forced_aligner_device or settings.device,
        "inputs": {"stems": sorted(stems), "digest": input_digest},
        "references_digest": reference_digest,
        "output_digest": output_digest,
    }


def _write_identity(path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body["identity_digest"] = _stable_digest(payload)
    _atomic_text(path, json.dumps(body, ensure_ascii=False, indent=2) + "\n")


_PROCESS_QWEN_BACKEND = None
_PROCESS_QWEN_SETTINGS = None


def _initialize_qwen_process(settings, require_asr, factory):
    """One persistent model pair per spawned process; HF loading is not thread-safe."""
    global _PROCESS_QWEN_BACKEND, _PROCESS_QWEN_SETTINGS
    import atexit
    _PROCESS_QWEN_SETTINGS = settings
    _PROCESS_QWEN_BACKEND = factory(settings, require_asr=require_asr)
    close = getattr(_PROCESS_QWEN_BACKEND, "close", None)
    if callable(close):
        atexit.register(close)
    print(f"  Qwen GPU worker ready: pid={os.getpid()} device={settings.device}", flush=True)


def _infer_with_qwen_backend(model, wav, reference, reference_mode, settings):
    is_reference = bool(reference) and reference_mode != "fallback"
    asr_text = "" if is_reference else model.transcribe(
        wav, language=None if settings.language.lower() == "auto" else settings.language,
        context=settings.context)
    authority = reference if is_reference else asr_text
    plan = _lexical_plan(normalize_qwen_input_text(authority))
    language = settings.language
    if language.lower() == "auto":
        language = getattr(model, "last_language", None) or "Chinese"
    aligned = model.align(wav, " ".join(item["unit"] for item in plan), language=language)
    return authority, asr_text, is_reference, plan, aligned


def _infer_qwen_process(wav, reference, reference_mode):
    try:
        return _infer_with_qwen_backend(
            _PROCESS_QWEN_BACKEND, wav, reference, reference_mode, _PROCESS_QWEN_SETTINGS)
    except Exception as exc:
        # Use a stable, pickleable error type across the process boundary.
        return Qwen3PrealignError(f"{type(exc).__name__}: {exc}")


def infer_qwen_items(wavs, refs, reference_mode, settings, *, devices,
                     backend=None, backend_factory=None):
    """Bounded, ordered inference with one model owner per visible GPU.

    batch_size controls queued items per device; native model requests remain
    individual utterances. Only the parent producer publishes/checkpoints.
    Native multi-GPU workers use spawned processes to isolate model loading.
    A failed item is returned as an exception and is never sent to another ASR.
    """
    native_processes = backend is None and backend_factory is None
    backend_factory = backend_factory or load_backend
    devices = list(devices)[:max(1, len(wavs))]
    if not devices or (backend is not None and len(devices) != 1):
        raise Qwen3PrealignError("invalid Qwen device/backend allocation")
    owned = {}
    need_asr = any(reference_mode == "fallback" or not refs.get(w.stem) for w in wavs)

    def infer(wav, device_index):
        try:
            if device_index not in owned:
                owned[device_index] = backend or backend_factory(
                    replace(settings, device=devices[device_index]), require_asr=need_asr)
            return _infer_with_qwen_backend(
                owned[device_index], wav, refs.get(wav.stem), reference_mode, settings)
        except Exception as exc:
            return exc

    native_processes = native_processes and len(devices) > 1
    if native_processes:
        import multiprocessing as mp
        executors = [ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"),
            initializer=_initialize_qwen_process,
            initargs=(replace(settings, device=device), need_asr, backend_factory))
            for device in devices]
    else:
        executors = [ThreadPoolExecutor(max_workers=1) for _ in devices]
    queue = deque()
    source = iter(enumerate(wavs))
    def submit_next():
        item = next(source, None)
        if item is None:
            return False
        index, wav = item
        device_index = index % len(devices)
        if native_processes:
            future = executors[device_index].submit(
                _infer_qwen_process, wav, refs.get(wav.stem), reference_mode)
        else:
            future = executors[device_index].submit(infer, wav, device_index)
        queue.append((wav, future))
        return True
    try:
        for _ in range(max(1, settings.batch_size) * len(devices)):
            if not submit_next():
                break
        while queue:
            wav, future = queue.popleft()
            yield wav, future.result()
            submit_next()
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
        if backend is None:
            for model in owned.values():
                close = getattr(model, "close", None)
                if callable(close):
                    close()


def run_qwen3_hf(cfg: dict[str, Any], project_root: Path,
                 *, backend: Any | None = None,
                 argv: list[str] | None = None) -> int:
    section = cfg.get("qwen3_hf", cfg.get("ctc_prealign", {}))
    if not isinstance(section, dict):
        raise Qwen3PrealignError("qwen3_hf configuration must be a mapping")
    data_dir = Path(section.get("data_dir", cfg.get("data_dir", "")))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    audio_dir = Path(section.get("audio_dir", cfg.get("audio_dir", data_dir)))
    if not audio_dir.is_absolute():
        audio_dir = project_root / audio_dir
    output_dir = Path(section.get("output_dir", cfg.get("output_dir", "ctc_pretg")))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    reference_mode = str(section.get("reference_mode", cfg.get("reference_mode", "auto")))
    if reference_mode not in {"auto", "authority", "fallback"}:
        raise Qwen3PrealignError("reference_mode must be auto, authority, or fallback")
    if section.get("overwrite"):
        raise Qwen3PrealignError("Qwen identity is immutable; choose a fresh output directory")
    if bool(section.get("nvv_enabled", False)) or section.get("reference_nvv_enabled") is True:
        raise Qwen3PrealignError("qwen3_hf does not support NVV labels")
    wavs = _select_wavs(_source_wavs(audio_dir), section, project_root)
    if not wavs:
        raise Qwen3PrealignError("qwen3_hf source inventory is empty")
    refs = _references(data_dir)
    expected = {wav.stem for wav in wavs}
    eligible = expected.copy()
    exclusions: dict[str, str] = {}
    if reference_mode == "authority":
        for stem in sorted(expected):
            if stem not in refs or not refs[stem]:
                eligible.discard(stem)
                exclusions[stem] = "missing_reference"
    for stem in eligible:
        if refs.get(stem) and reference_mode != "fallback":
            _lexical_plan(normalize_qwen_input_text(refs[stem]))  # reject unsupported reference/NVV before loading weights
    model_value = str(section.get("model_path", "")).strip()
    aligner_value = str(section.get("forced_aligner_model_path", "")).strip()
    requires_asr_model = any(reference_mode == "fallback" or not refs.get(stem)
                             for stem in eligible)
    if (requires_asr_model and not model_value) or not aligner_value:
        raise Qwen3PrealignError(
            "qwen3_hf model_path and forced_aligner_model_path are required")
    settings = Qwen3HFSettings(
        model_path=Path(model_value),
        forced_aligner_model_path=Path(aligner_value),
        device=str(section.get("device", "cuda:0")), dtype=str(section.get("dtype", "bfloat16")),
        language=(str(section.get("language")) if section.get("language") else "auto"),
        max_new_tokens=int(section.get("max_new_tokens", 2048)),
        batch_size=int(section.get("batch_size", 1)),
        context=str(section.get("context", "")),
        forced_aligner_device=section.get("forced_aligner_device"),
    )
    model_path = settings.model_path if settings.model_path.is_absolute() else project_root / settings.model_path
    aligner_path = (settings.forced_aligner_model_path if settings.forced_aligner_model_path.is_absolute()
                    else project_root / settings.forced_aligner_model_path)
    if model_path != settings.model_path or aligner_path != settings.forced_aligner_model_path:
        settings = replace(settings, model_path=model_path,
                           forced_aligner_model_path=aligner_path)
    validate_settings(settings)
    devices = [settings.device]
    if section.get("all_gpus", False):
        if settings.forced_aligner_device is not None:
            raise Qwen3PrealignError("all_gpus cannot share a fixed forced_aligner_device")
        import torch
        count = torch.cuda.device_count()
        if not count:
            raise Qwen3PrealignError("all_gpus requires a visible CUDA GPU")
        devices = [f"cuda:{index}" for index in range(count)]
    asr_digest, asr_files = (_model_evidence(settings.model_path)
                             if requires_asr_model else ("not_required", []))
    aligner_digest, aligner_files = _model_evidence(settings.forced_aligner_model_path)
    producer_model_path = (settings.model_path if requires_asr_model
                           else settings.forced_aligner_model_path)
    producer_model_digest = asr_digest if requires_asr_model else aligner_digest
    producer_model_files = asr_files if requires_asr_model else aligner_files
    runtime = section.get("runtime_capabilities") if backend is not None else None
    if not isinstance(runtime, dict):
        runtime = runtime_capabilities()
    runtime = dict(runtime)
    runtime["inference_devices"] = devices
    runtime["producer_sha256"] = _sha256(Path(__file__))
    runtime["backend_sha256"] = _sha256(Path(__file__).with_name("qwen3_hf_backend.py"))
    runtime["timestamp_normalization"] = TIMESTAMP_SCHEMA
    runtime["timestamp_normalization_sha256"] = _sha256(
        Path(__file__).with_name("qwen3_timestamp_normalization.py"))
    for package_name in ("torch", "pypinyin"):
        try:
            runtime[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            runtime[package_name] = "unavailable"
    dictionary_value = str(cfg.get("mfa_dict", ""))
    dictionary_path = Path(dictionary_value) if dictionary_value else None
    if dictionary_path is not None:
        if not dictionary_path.is_absolute():
            dictionary_path = project_root / dictionary_path
        if not dictionary_path.is_file() or dictionary_path.is_symlink():
            raise Qwen3PrealignError(f"invalid MFA dictionary: {dictionary_path}")
    dictionary_digest = _sha256(dictionary_path) if dictionary_path else _stable_digest(None)
    runtime["dictionary_digest"] = dictionary_digest
    input_rows = [{"stem": wav.stem, "path": str(wav.resolve()),
                   "sha256": _sha256(wav), "duration_s": _duration(wav)} for wav in wavs]
    for row in input_rows:
        if row["duration_s"] > 300.0:
            raise Qwen3PrealignError(
                f"Qwen3 ForcedAligner supports at most 300 seconds: {row['stem']}")
    input_digest = _stable_digest(input_rows)
    reference_rows = [{"stem": stem, "text": refs[stem]} for stem in sorted(refs)
                      if stem in eligible]
    reference_digest = _stable_digest({"mode": reference_mode, "references": reference_rows})
    if section.get("check_only"):
        print(json.dumps({"provider": "qwen3_hf", "runtime": runtime,
                          "selected": len(wavs), "eligible": len(eligible),
                          "models": {"asr": asr_digest, "aligner": aligner_digest}}, indent=2))
        return 0
    if not eligible:
        raise Qwen3PrealignError("no eligible audio with reference text")
    selected_stems = sorted(eligible)
    old_identity_path = output_dir / ".qwen3_hf_identity.json"
    if output_dir.is_symlink():
        raise Qwen3PrealignError("Qwen output directory must not be a symlink")
    prior_produced: list[str] = []
    prior_records: list[dict[str, Any]] = []
    if old_identity_path.is_file():
        try:
            if old_identity_path.is_symlink():
                raise ValueError("symlink identity")
            old = json.loads(old_identity_path.read_text(encoding="utf-8"))
            recorded_digest = old.pop("identity_digest")
            if recorded_digest != _stable_digest(old):
                raise ValueError("tampered identity")
            allowed = {f"{stem}{suffix}" for stem in selected_stems
                       for suffix in (*CTC_SUFFIXES, "_ref.txt")}
            allowed.update({"manifest.json", ".ctc_normalized", ".qwen3_hf_identity.json",
                            ".ctc_run_receipt.json", ".pipeline_run_receipt_v2.json",
                            ".ctc_raw_manifest.json", "summary.txt"})
            if any(path.name not in allowed or not path.is_file() or path.is_symlink()
                   for path in output_dir.iterdir()):
                raise ValueError("unexpected or tampered output namespace")
            prior_records = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            prior_produced = [Path(record["lab"]).stem for record in prior_records]
            if len(set(prior_produced)) != len(prior_produced) or not set(prior_produced) <= eligible:
                raise ValueError("changed output membership")
            candidate = _identity_payload(
                settings=settings, runtime=runtime, asr_digest=asr_digest,
                aligner_digest=aligner_digest, input_digest=input_digest,
                reference_digest=reference_digest,
                output_digest=_artifact_digest(output_dir, prior_produced),
                stems=selected_stems)
            if old != candidate:
                raise ValueError("changed configuration or tampered output")
            for name in (".ctc_run_receipt.json", ".pipeline_run_receipt_v2.json"):
                receipt_path = output_dir / name
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt_path.is_symlink() or receipt.get("extra", {}).get("identity_digest") != recorded_digest:
                    raise ValueError("tampered producer receipt")
                if name == ".ctc_run_receipt.json":
                    if (receipt.get("model") != {"path": str(producer_model_path.resolve()),
                                                "tree_digest": producer_model_digest, "files": producer_model_files}
                            or receipt.get("dictionary", {}).get("digest") != dictionary_digest
                            or receipt.get("input_stems") != selected_stems
                            or receipt.get("output_stems") != sorted(prior_produced)
                            or receipt.get("extra", {}).get("provider") != "qwen3_hf"
                            or receipt.get("extra", {}).get("forced_aligner_model") != {
                                "path": str(settings.forced_aligner_model_path),
                                "tree_digest": aligner_digest, "files": aligner_files}):
                        raise ValueError("tampered model/output receipt")
            if set(prior_produced) == eligible:
                print("Qwen3 HF prealign identity matches; reusing existing artifacts")
                return 0
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Qwen3PrealignError(f"Qwen identity changed or artifacts tampered: {exc}; use a fresh output directory") from exc
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise Qwen3PrealignError("missing Qwen identity in nonempty output directory; use a fresh directory")
    pending = eligible - set(prior_produced)
    inferences = infer_qwen_items([wav for wav in wavs if wav.stem in pending],
                                 refs, reference_mode, settings,
                                 devices=devices, backend=backend)
    produced: list[str] = list(prior_produced)
    failures: list[str] = []
    records: list[dict[str, Any]] = list(prior_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for wav, inference in inferences:
            stem = wav.stem
            if stem not in pending:
                continue
            try:
                if isinstance(inference, Exception):
                    raise inference
                authority, asr_text, is_reference, plan, aligned = inference
                expected_units = [item["unit"] for item in plan]
                if len(aligned) != len(expected_units):
                    raise Qwen3PrealignError(
                        f"ForcedAligner unit count {len(aligned)} != lexical text {len(expected_units)}")
                rows: list[dict[str, Any]] = []
                pinyin_values = iter(contextual_pinyin(
                    normalize_qwen_input_text(authority)))
                for ordinal, (item, planned) in enumerate(zip(aligned, plan)):
                    expected_unit = planned["unit"]
                    unit = str(item["unit"])
                    if unit.replace(" ", "") != expected_unit.replace(" ", ""):
                        raise Qwen3PrealignError(
                            f"ForcedAligner unit mismatch: {unit!r} != {expected_unit!r}")
                    start, end = float(item["start_s"]), float(item["end_s"])
                    raw_start = float(item.get("raw_start_s", start))
                    raw_end = float(item.get("raw_end_s", end))
                    duration = next(row["duration_s"] for row in input_rows
                                    if row["stem"] == stem)
                    if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end):
                        raise Qwen3PrealignError(f"invalid Qwen span for {stem}: {start}..{end}")
                    if (not math.isfinite(raw_start) or not math.isfinite(raw_end)
                            or raw_start < 0 or raw_end < raw_start):
                        raise Qwen3PrealignError(
                            f"invalid raw Qwen span for {stem}: {raw_start}..{raw_end}")
                    end = min(end, duration)
                    canonical = planned["authority"]
                    row = {"unit": expected_unit,
                           "word": canonical.alignment_token if canonical else next(pinyin_values),
                           "start_s": start, "end_s": end,
                           "provider": "qwen3_hf", "lexical_timing_source": "qwen3_forced_aligner_hf",
                           "raw_start_s": raw_start, "raw_end_s": raw_end}
                    if canonical is not None:
                        row["surface_text"] = canonical.surface_text
                    timing_adjustment = item.get("timing_adjustment")
                    if timing_adjustment is not None:
                        if not isinstance(timing_adjustment, dict):
                            raise Qwen3PrealignError(
                                f"invalid Qwen timing adjustment for {stem}")
                        row["timing_adjustment"] = dict(timing_adjustment)
                    if is_reference and canonical is not None:
                        unit = replace(canonical, source_ctc_ordinals=(ordinal,),
                                       canonical_start=start, canonical_end=end).to_dict()
                        row.update({"canonical_unit": unit, "canonical_unit_sha256": _stable_digest(unit),
                                    "source_ctc_ordinals": [ordinal],
                                    "source_ordinal_provider": "qwen3_hf",  # legacy field denotes row ordinals, not CTC frames
                                    "canonical_span": [start, end], "surface_text": canonical.surface_text,
                                    "reference_identity": hashlib.sha256(authority.encode("utf-8")).hexdigest(),
                                    "reference_ordinal": canonical.reference_ordinal})
                    rows.append(row)
                duration = next(row["duration_s"] for row in input_rows
                                if row["stem"] == stem)
                rows, normalized_text, pause_entries = normalize_timestamps(rows, authority, duration)
                _write_textgrid(output_dir / f"{stem}.TextGrid", duration, rows)
                _atomic_text(output_dir / f"{stem}.lab", " ".join(r["word"] for r in rows) + "\n")
                token_rows = [{**row, "text": row["unit"],
                               "start_s": row["start_s"], "end_s": row["end_s"],
                               "start_ms": row["start_s"] * 1000,
                               "end_ms": row["end_s"] * 1000,
                               "source": "qwen3_forced_aligner_hf"} for row in rows]
                _atomic_text(output_dir / f"{stem}_tokens.jsonl",
                             "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in token_rows))
                _atomic_text(output_dir / f"{stem}_punct.json",
                             json.dumps(pause_entries, ensure_ascii=False) + "\n")
                _atomic_text(output_dir / f"{stem}_text_cn.txt", normalized_text + "\n")
                _atomic_text(output_dir / f"{stem}_text_raw.txt", authority + "\n")
                if is_reference:
                    _atomic_text(output_dir / f"{stem}_ref.txt", normalized_text + "\n")
                errors = validate_ctc_transcript_bundle(output_dir, stem, _require_processed=False)
                if errors:
                    raise Qwen3PrealignError("; ".join(errors))
                records.append({"audio": str(wav),
                                "textgrid": str(output_dir / f"{stem}.TextGrid"),
                                "lab": str(output_dir / f"{stem}.lab"),
                                "text_asr": asr_text,
                                "text_original": authority,
                                "text_normalized": normalized_text,
                                "timestamp_normalization": TIMESTAMP_SCHEMA,
                                "content_authority": "reference" if is_reference else "qwen3",
                                "provider": "qwen3_hf",
                                "lexical_timing_source": "qwen3_forced_aligner_hf",
                                "duration_s": duration, "n_words": len(rows), "n_punct": len(pause_entries),
                                "_words": [{"word": row["word"], "start": row["start_s"], "end": row["end_s"]} for row in rows]})
                produced.append(stem)
            except Exception as exc:
                print(f"FAIL {stem}: {exc}")
                failures.append(stem)
                for suffix in (*CTC_SUFFIXES, "_ref.txt"):
                    (output_dir / f"{stem}{suffix}").unlink(missing_ok=True)
    finally:
        inferences.close()
    manifest = sorted(records, key=lambda record: record["lab"])
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_digest = _sha256(output_dir / "manifest.json")
    _atomic_text(output_dir / ".ctc_normalized",
                 make_ctc_normalization_marker(len(produced), manifest_digest))
    output_digest = _artifact_digest(output_dir, sorted(produced))
    identity = _identity_payload(
        settings=settings, runtime=runtime, asr_digest=asr_digest,
        aligner_digest=aligner_digest, input_digest=input_digest,
        reference_digest=reference_digest, output_digest=output_digest,
        stems=selected_stems)
    _write_identity(output_dir / ".qwen3_hf_identity.json", identity)
    write_ctc_run_receipt(output_dir, actual_argv=argv or sys.argv,
                          asr_python=sys.executable, model_path=producer_model_path,
                          model_tree_digest=producer_model_digest, model_file_manifest=producer_model_files,
                          dict_path=dictionary_path or Path(""), dict_digest=dictionary_digest,
                          input_stems=sorted(eligible), output_stems=sorted(produced),
                          extra={"provider": "qwen3_hf", "runtime": runtime,
                                 "forced_aligner_model": {
                                     "path": str(settings.forced_aligner_model_path),
                                     "tree_digest": aligner_digest,
                                     "files": aligner_files},
                                 "identity_digest": _stable_digest(identity)})
    receipt = make_pipeline_accounting_receipt(
        sorted(expected), sorted(eligible), exclusions, sorted(produced),
        sorted(set(eligible) - set(produced)), run_id=make_pipeline_run_id(),
        mode="ctc_prealign", route=["qwen3_hf", "qwen3_forced_aligner"],
        paths={"output": str(output_dir), "filtered": str(output_dir)},
        shards=[{"shard_id": "single", "stems": sorted(eligible)}],
        extra={"provider": "qwen3_hf", "processed_stems": sorted(eligible),
               "reference_mode": reference_mode, "nvv_enabled": False,
               "forced_aligner_model_tree_digest": aligner_digest,
               "identity_digest": _stable_digest(identity)})
    write_pipeline_accounting_receipt(output_dir, receipt)
    summary = f"Files: {len(expected)} total, {len(produced)} OK, {len(failures)} failed\n"
    _atomic_text(output_dir / "summary.txt", summary)
    return 1 if failures and not section.get("allow_item_failures", False) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default="", help="ASR weights; required only without reference text")
    parser.add_argument("--forced-aligner-model-path", required=True)
    parser.add_argument("--dict-path", default="")
    parser.add_argument("--check", action="store_true", help="check local inputs/runtime without loading weights or creating output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forced-aligner-device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--context", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--all-gpus", action="store_true")
    parser.add_argument("--reference-mode", default="auto")
    parser.add_argument("--stems-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-nvv", action="store_true")
    parser.add_argument("--allow-item-failures", action="store_true")
    args = parser.parse_args(argv)
    cfg = {"data_dir": str(args.data_dir), "audio_dir": str(args.audio_dir or args.data_dir),
           "output_dir": str(args.output_dir), "mfa_dict": args.dict_path,
           "reference_mode": args.reference_mode,
           "ctc_prealign": {"model_path": args.model_path,
                             "forced_aligner_model_path": args.forced_aligner_model_path,
                             "device": args.device, "dtype": args.dtype,
                             "forced_aligner_device": args.forced_aligner_device,
                             "language": args.language, "context": args.context,
                             "max_new_tokens": args.max_new_tokens,
                             "batch_size": args.batch_size,
                             "all_gpus": args.all_gpus,
                             "check_only": args.check,
                             "nvv_enabled": False,
                             "allow_item_failures": args.allow_item_failures,
                             "stems_file": str(args.stems_file) if args.stems_file else "",
                             "limit": args.limit, "offset": args.offset}}
    try:
        return run_qwen3_hf(cfg, Path.cwd(), argv=[sys.argv[0], *(argv or sys.argv[1:])])
    except (OSError, ValueError, Qwen3HFError, Qwen3PrealignError) as exc:
        print(f"ERROR: qwen3_hf: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
