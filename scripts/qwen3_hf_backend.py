#!/usr/bin/env python3
"""Lazy adapters for the native Qwen3 ASR and ForcedAligner models.

The regular MFA environment must be able to import this module without
installing ``qwen_asr`` or a recent Transformers build.  Heavy imports are
therefore kept inside the factories.  The adapter deliberately exposes only
the two operations the prealign producer needs: transcription and lexical
timestamp alignment.  Phone alignment remains MFA's responsibility.
"""

from __future__ import annotations

import importlib
import gc
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Qwen3HFError(RuntimeError):
    """A fail-closed Qwen native backend error."""


@dataclass(frozen=True)
class Qwen3HFSettings:
    model_path: Path
    forced_aligner_model_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    language: str | None = "Chinese"
    max_new_tokens: int = 2048
    batch_size: int = 1
    context: str = ""
    forced_aligner_device: str | None = None


def validate_settings(settings: Qwen3HFSettings) -> None:
    """Validate paths and inference values before importing model packages."""
    if not settings.model_path:
        raise Qwen3HFError("qwen3_hf model_path is required")
    if not settings.forced_aligner_model_path:
        raise Qwen3HFError("qwen3_hf forced_aligner_model_path is required")
    if settings.max_new_tokens < 1 or settings.batch_size < 1:
        raise Qwen3HFError("qwen3_hf generation values must be positive")
    if not isinstance(settings.dtype, str) or not settings.dtype.strip():
        raise Qwen3HFError("qwen3_hf dtype must be a non-empty string")
    if not isinstance(settings.device, str) or not settings.device.strip():
        raise Qwen3HFError("qwen3_hf device must be a non-empty string")
    if settings.forced_aligner_device is not None and (
            not isinstance(settings.forced_aligner_device, str)
            or not settings.forced_aligner_device.strip()):
        raise Qwen3HFError("qwen3_hf forced_aligner_device must be a non-empty string")


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Qwen3HFError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise Qwen3HFError(f"{field} must be finite")
    return result


def normalize_transcript(value: Any) -> str:
    """Extract a transcript from official Qwen result objects or test fakes."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = value.get("transcription", value.get("text", value.get("transcript", "")))
    else:
        text = getattr(value, "text", None)
        if text is None:
            text = getattr(value, "transcript", None)
    if not isinstance(text, str) or not text.strip():
        raise Qwen3HFError("Qwen ASR returned an empty transcript")
    return text.strip()


def normalize_alignment_items(
        value: Any, *, timestamp_segment_time_ms: Any = None) -> list[dict[str, Any]]:
    """Normalize native ForcedAligner items to producer JSON rows.

    Native Qwen output is lexical (Chinese characters and English words).  It
    does not expose a phoneme sequence.  Model-quantized zero-width items are
    expanded only into a complete adjacent timestamp quantum while preserving
    their raw point; NaN, infinity and negative spans remain invalid.
    """
    if isinstance(value, dict):
        if "items" in value or "segments" in value:
            value = value.get("items", value.get("segments"))
        elif any(key in value for key in ("text", "unit", "word")):
            value = [value]
    elif not isinstance(value, (list, tuple)):
        items = getattr(value, "items", None)
        if callable(items):
            items = items()
        value = items
    if not isinstance(value, (list, tuple)) or not value:
        raise Qwen3HFError("Qwen ForcedAligner returned no lexical items")

    raw_items: list[dict[str, Any]] = []
    previous_start = -math.inf
    previous_end = -math.inf
    for index, item in enumerate(value):
        if isinstance(item, dict):
            unit = item.get("text", item.get("unit", item.get("word")))
            start_raw = item.get("start_time", item.get("start_s", item.get("start")))
            end_raw = item.get("end_time", item.get("end_s", item.get("end")))
        else:
            unit = getattr(item, "text", None) or getattr(item, "unit", None)
            start_raw = getattr(item, "start_time", None)
            if start_raw is None:
                start_raw = getattr(item, "start_s", None)
            end_raw = getattr(item, "end_time", None)
            if end_raw is None:
                end_raw = getattr(item, "end_s", None)
        if not isinstance(unit, str) or not unit.strip():
            raise Qwen3HFError(f"ForcedAligner item {index} has no lexical text")
        start = _number(start_raw, f"ForcedAligner item {index} start")
        end = _number(end_raw, f"ForcedAligner item {index} end")
        if start < 0 or end < start:
            raise Qwen3HFError(
                f"ForcedAligner item {index} has invalid span {start}..{end}")
        if start + 1e-9 < previous_start or end + 1e-9 < previous_end:
            raise Qwen3HFError(f"ForcedAligner items are not monotonic at {index}")
        previous_start, previous_end = start, end
        raw_items.append({
            "unit": unit,
            "raw_start_s": start,
            "raw_end_s": end,
        })

    quantum_s = None
    normalized: list[dict[str, Any]] = []
    previous_final_end = -math.inf
    for index, item in enumerate(raw_items):
        start = item.get("adjusted_start_s", item["raw_start_s"])
        end = item.get("adjusted_end_s", item["raw_end_s"])
        adjustment = item.get("timing_adjustment")
        if end == start:
            if quantum_s is None:
                quantum_ms = _number(
                    timestamp_segment_time_ms,
                    "ForcedAligner timestamp_segment_time")
                quantum_s = quantum_ms / 1000.0
                if quantum_s <= 0:
                    raise Qwen3HFError(
                        "ForcedAligner timestamp_segment_time must be positive")
            next_start = (raw_items[index + 1]["raw_start_s"]
                          if index + 1 < len(raw_items) else None)
            right_end = round(start + quantum_s, 9)
            if (next_start is not None and right_end <= next_start + 1e-9
                    and start + 1e-9 >= previous_final_end):
                end = right_end
                reason = "zero_duration_expand_right"
            else:
                left_start = start - quantum_s
                if left_start < 0 or left_start + 1e-9 < previous_final_end:
                    previous = normalized[-1] if normalized else None
                    previous_duration = (
                        previous["end_s"] - previous["start_s"]
                        if previous is not None else -math.inf)
                    can_borrow_left = (
                        previous is not None
                        and abs(previous["end_s"] - start) <= 1e-9
                        and previous_duration + 1e-9 >= 2 * quantum_s)
                    next_item = raw_items[index + 1] if index + 1 < len(raw_items) else None
                    next_duration = (
                        next_item["raw_end_s"] - next_item["raw_start_s"]
                        if next_item is not None else -math.inf)
                    can_borrow_right = (
                        next_item is not None
                        and abs(next_item["raw_start_s"] - start) <= 1e-9
                        and next_duration + 1e-9 >= 2 * quantum_s)
                    if can_borrow_left:
                        start = round(previous["end_s"] - quantum_s, 9)
                        end = previous["end_s"]
                        previous["end_s"] = start
                        previous["timing_adjustment"] = {
                            "reason": "yield_right_quantum_to_zero_duration",
                            "quantum_s": quantum_s,
                            "raw_start_s": previous["raw_start_s"],
                            "raw_end_s": previous["raw_end_s"],
                        }
                        reason = "zero_duration_borrow_left"
                    elif can_borrow_right:
                        end = right_end
                        next_item["adjusted_start_s"] = end
                        next_item["timing_adjustment"] = {
                            "reason": "yield_left_quantum_to_zero_duration",
                            "quantum_s": quantum_s,
                            "raw_start_s": next_item["raw_start_s"],
                            "raw_end_s": next_item["raw_end_s"],
                        }
                        reason = "zero_duration_borrow_right"
                    else:
                        run_end = index + 1
                        while (run_end < len(raw_items)
                               and raw_items[run_end]["raw_start_s"] == start
                               and raw_items[run_end]["raw_end_s"] == start):
                            run_end += 1
                        right_neighbor = raw_items[run_end] if run_end < len(raw_items) else None
                        left_touches = (
                            previous is not None
                            and abs(previous["end_s"] - start) <= 1e-9)
                        right_touches = (
                            right_neighbor is not None
                            and abs(right_neighbor["raw_start_s"] - start) <= 1e-9
                            and right_neighbor["raw_end_s"] > start)
                        if not left_touches and not right_touches:
                            left_bound = max(
                                0.0,
                                previous_final_end if math.isfinite(previous_final_end)
                                else start,
                            )
                            right_bound = (right_neighbor["raw_start_s"]
                                           if right_neighbor is not None else start)
                            sub_start = max(left_bound, start - quantum_s / 2)
                            sub_end = min(right_bound, start + quantum_s / 2)
                            if sub_end > sub_start + 1e-9:
                                start, end = sub_start, sub_end
                                reason = "zero_duration_expand_subquantum_gap"
                                adjustment = {
                                    "reason": reason,
                                    "quantum_s": quantum_s,
                                    "raw_start_s": item["raw_start_s"],
                                    "raw_end_s": item["raw_end_s"],
                                }
                                normalized_item = {
                                    "unit": item["unit"],
                                    "start_s": start,
                                    "end_s": end,
                                    "raw_start_s": item["raw_start_s"],
                                    "raw_end_s": item["raw_end_s"],
                                    "timing_adjustment": adjustment,
                                }
                                normalized.append(normalized_item)
                                previous_final_end = end
                                continue
                            raise Qwen3HFError(
                                f"ForcedAligner zero-duration item {index} has no usable adjacent support")
                        zero_count = run_end - index
                        outer_start = previous["start_s"] if left_touches else start
                        outer_end = right_neighbor["raw_end_s"] if right_touches else start
                        participant_count = zero_count + int(left_touches) + int(right_touches)
                        slot = (outer_end - outer_start) / participant_count
                        if slot <= 1e-9:
                            raise Qwen3HFError(
                                f"ForcedAligner zero-duration item {index} has no positive adjacent support")
                        boundaries = [
                            round(outer_start + slot * offset, 9)
                            for offset in range(participant_count + 1)
                        ]
                        boundaries[0], boundaries[-1] = outer_start, outer_end
                        zero_offset = 0
                        if left_touches:
                            previous["end_s"] = boundaries[1]
                            previous["timing_adjustment"] = {
                                "reason": "redistribute_shared_boundary_for_zero_duration",
                                "quantum_s": quantum_s,
                                "raw_start_s": previous["raw_start_s"],
                                "raw_end_s": previous["raw_end_s"],
                            }
                            zero_offset = 1
                        for offset, zero_index in enumerate(range(index, run_end), start=zero_offset):
                            zero_item = raw_items[zero_index]
                            zero_item["adjusted_start_s"] = boundaries[offset]
                            zero_item["adjusted_end_s"] = boundaries[offset + 1]
                            zero_item["timing_adjustment"] = {
                                "reason": "zero_duration_redistribute_shared_boundary",
                                "quantum_s": quantum_s,
                                "raw_start_s": zero_item["raw_start_s"],
                                "raw_end_s": zero_item["raw_end_s"],
                            }
                        if right_touches:
                            right_neighbor["adjusted_start_s"] = boundaries[-2]
                            right_neighbor["timing_adjustment"] = {
                                "reason": "redistribute_shared_boundary_for_zero_duration",
                                "quantum_s": quantum_s,
                                "raw_start_s": right_neighbor["raw_start_s"],
                                "raw_end_s": right_neighbor["raw_end_s"],
                            }
                        start = item["adjusted_start_s"]
                        end = item["adjusted_end_s"]
                        reason = "zero_duration_redistribute_shared_boundary"
                else:
                    start = left_start
                    reason = "zero_duration_expand_left"
            adjustment = {
                "reason": reason,
                "quantum_s": quantum_s,
                "raw_start_s": item["raw_start_s"],
                "raw_end_s": item["raw_end_s"],
            }
        if end <= start:
            raise Qwen3HFError(
                f"ForcedAligner item {index} has non-positive final span {start}..{end}")
        normalized_item = {
            "unit": item["unit"],
            "start_s": start,
            "end_s": end,
            "raw_start_s": item["raw_start_s"],
            "raw_end_s": item["raw_end_s"],
        }
        if adjustment is not None:
            normalized_item["timing_adjustment"] = adjustment
        normalized.append(normalized_item)
        previous_final_end = end
    return normalized


def runtime_capabilities() -> dict[str, str]:
    """Check native API availability without resolving or loading weights."""
    try:
        package = importlib.import_module("transformers")
    except ImportError as exc:
        raise Qwen3HFError("Install requirements-qwen3-hf.txt in a separate environment") from exc
    version = str(getattr(package, "__version__", ""))
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (5, 13, 0):
        raise Qwen3HFError(f"Native Qwen3 requires Transformers >=5.13.0; found {version!r}")
    for name in ("AutoProcessor", "AutoModelForMultimodalLM", "AutoModelForTokenClassification"):
        if not callable(getattr(getattr(package, name, None), "from_pretrained", None)):
            raise Qwen3HFError(f"Transformers lacks {name}.from_pretrained; install native Qwen support")
    return {"backend": "transformers-native-qwen3-hf", "transformers_version": version}


class Qwen3HFBackend:
    """Native Transformers calls; reference-only runs never load ASR weights."""

    def __init__(self, *, settings: Qwen3HFSettings, torch: Any,
                 asr_processor: Any, asr_model: Any,
                 aligner_processor: Any, forced_aligner: Any):
        self.settings = settings
        self.torch = torch
        self.asr_processor = asr_processor
        self.asr_model = asr_model
        self.aligner_processor = aligner_processor
        self.forced_aligner = forced_aligner
        self.last_language = settings.language

    def transcribe(self, audio: Path, *, language: str | None = None,
                   context: str = "") -> str:
        if self.asr_model is None:
            raise Qwen3HFError("ASR is not loaded in this reference-only run")
        inputs = self.asr_processor.apply_transcription_request(
            audio=str(audio), language=language, prompt=context or self.settings.context or None)
        inputs = inputs.to(self.asr_model.device, self.asr_model.dtype)
        with self.torch.inference_mode():
            generated = self.asr_model.generate(
                **inputs, max_new_tokens=self.settings.max_new_tokens)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        parsed = self.asr_processor.decode(generated, return_format="parsed")
        if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
            raise Qwen3HFError("Native ASR returned an invalid single-audio result")
        self.last_language = language or parsed[0].get("language")
        return normalize_transcript(parsed[0])

    def align(self, audio: Path, text: str, *, language: str | None = "Chinese") -> list[dict[str, Any]]:
        if self.forced_aligner is None:
            raise Qwen3HFError("ForcedAligner is closed")
        inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=str(audio), transcript=text, language=language or self.last_language)
        inputs = inputs.to(self.forced_aligner.device, self.forced_aligner.dtype)
        with self.torch.inference_mode():
            outputs = self.forced_aligner(**inputs)
        decoded = self.aligner_processor.decode_forced_alignment(
            logits=outputs.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
            timestamp_token_id=self.forced_aligner.config.timestamp_token_id)
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise Qwen3HFError("Native ForcedAligner returned an invalid batch")
        return normalize_alignment_items(
            decoded[0],
            timestamp_segment_time_ms=getattr(
                self.forced_aligner.config, "timestamp_segment_time", None),
        )

    def close(self) -> None:
        self.asr_model = self.forced_aligner = None
        self.asr_processor = self.aligner_processor = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _torch_dtype(torch: Any, name: str) -> Any:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise Qwen3HFError(f"torch does not expose dtype {name!r}")
    return dtype


def load_backend(settings: Qwen3HFSettings, *, require_asr: bool = True) -> Qwen3HFBackend:
    """Load native model classes only after producer checks have completed."""
    validate_settings(settings)
    paths = [settings.forced_aligner_model_path]
    if require_asr:
        paths.append(settings.model_path)
    for path in paths:
        if not path.is_dir() or path.is_symlink():
            raise Qwen3HFError(f"Expected existing local non-symlink model directory: {path}")
    runtime_capabilities()
    try:
        package = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise Qwen3HFError(
            "qwen3_hf requires native Transformers and torch in its isolated environment") from exc
    dtype = _torch_dtype(torch, settings.dtype)
    common = {"dtype": dtype, "device_map": settings.device, "local_files_only": True}
    asr = asr_processor = None
    try:
        aligner_processor = package.AutoProcessor.from_pretrained(
            str(settings.forced_aligner_model_path), local_files_only=True)
        for method in ("prepare_forced_aligner_inputs", "decode_forced_alignment"):
            if not callable(getattr(aligner_processor, method, None)):
                raise Qwen3HFError(f"Native ForcedAligner processor lacks {method}; check Transformers/model format")
        if require_asr:
            asr_processor = package.AutoProcessor.from_pretrained(
                str(settings.model_path), local_files_only=True)
            if not callable(getattr(asr_processor, "apply_transcription_request", None)):
                raise Qwen3HFError("ASR processor lacks apply_transcription_request; use a native -hf checkpoint")
        aligner = package.AutoModelForTokenClassification.from_pretrained(
            str(settings.forced_aligner_model_path),
            **{**common, "device_map": settings.forced_aligner_device or settings.device})
        aligner.eval()
        if require_asr:
            asr = package.AutoModelForMultimodalLM.from_pretrained(str(settings.model_path), **common)
            asr.eval()
    except Exception as exc:
        raise Qwen3HFError(f"failed to load native Qwen3 models: {exc}") from exc
    return Qwen3HFBackend(settings=settings, torch=torch, asr_processor=asr_processor,
                         asr_model=asr, aligner_processor=aligner_processor, forced_aligner=aligner)
