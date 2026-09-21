"""Qwen3 ForcedAligner anchors and language-run planning for the JA/EN flow.

The ForcedAligner is used only for lexical timing.  This module deliberately
keeps its model import lazy so unit tests can exercise the exact character and
sample contracts without loading a GPU model.
"""

from __future__ import annotations

import importlib
import json
import math
import argparse
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .ja_en_schema import JAContractError, StageResult, atomic_write_json, make_receipt
except ImportError:  # pragma: no cover
    from ja_en_schema import JAContractError, StageResult, atomic_write_json, make_receipt


class AnchorConflictError(ValueError):
    """The independent language passes disagree on a seam beyond the limit."""


ANCHOR_PLAN_FILENAME = "anchor_plan.json"
ANCHOR_RUNS_FILENAME = "language_runs.json"
ANCHOR_REQUIRED_KEYS = frozenset({"schema", "uid", "route", "runs", "seams", "boundary_evidence", "passes", "units"})


class AnchorMappingError(ValueError):
    """A lexical anchor cannot be mapped to the frozen character stream."""


def _number(value: Any, name: str) -> float:
    if hasattr(value, "total_seconds"):
        value = value.total_seconds()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnchorMappingError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise AnchorMappingError(f"{name} is not finite")
    return result


def _item_text(item: Any) -> str:
    if isinstance(item, Mapping):
        value = item.get("unit", item.get("text", item.get("word")))
    else:
        value = getattr(item, "unit", None) or getattr(item, "text", None)
    if not isinstance(value, str) or not value:
        raise AnchorMappingError("forced-aligner item has no lexical text")
    return value


def _item_time(item: Any, key: str) -> float:
    aliases = {
        "start": ("start_s", "start_time", "start"),
        "end": ("end_s", "end_time", "end"),
    }
    if isinstance(item, Mapping):
        value = next((item.get(name) for name in aliases[key] if name in item), None)
    else:
        value = next((getattr(item, name, None) for name in aliases[key]
                      if hasattr(item, name)), None)
    return _number(value, f"forced-aligner {key}")


def normalize_anchor_items(items: Any) -> list[dict[str, Any]]:
    """Normalize native Qwen item objects without changing lexical text."""
    if isinstance(items, Mapping):
        items = items.get("items", items.get("segments", [items]))
    elif not isinstance(items, (list, tuple)):
        value = getattr(items, "items", None)
        items = value() if callable(value) else value
    if not isinstance(items, (list, tuple)) or not items:
        raise AnchorMappingError("forced-aligner returned no lexical items")
    # Native qwen_asr returns a one-audio batch: [ForcedAlignResult(items=[])].
    if len(items) == 1:
        batch = items[0]
        if isinstance(batch, Mapping) and "items" in batch:
            items = batch["items"]
        elif not isinstance(batch, Mapping) and not hasattr(batch, "text") and not hasattr(batch, "unit"):
            value = getattr(batch, "items", None)
            if callable(value):
                value = value()
            if isinstance(value, (list, tuple)):
                items = value
    result: list[dict[str, Any]] = []
    previous_start = -math.inf
    previous_end = -math.inf
    for index, item in enumerate(items):
        text = _item_text(item)
        start = _item_time(item, "start")
        end = _item_time(item, "end")
        if start < 0 or end <= start:
            raise AnchorMappingError(f"forced-aligner item {index} has invalid span")
        if start < previous_start or end < previous_end:
            raise AnchorMappingError("forced-aligner items are not monotonic")
        result.append({"unit": text, "start_s": start, "end_s": end})
        previous_start, previous_end = start, end
    return result


def _validate_units(spoken_text: str, units: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    previous_end = 0
    for index, unit in enumerate(units):
        text = unit.get("text", unit.get("unit"))
        span = unit.get("char_span")
        if not isinstance(text, str) or not text or not isinstance(span, (list, tuple)) or len(span) != 2:
            raise AnchorMappingError(f"unit {index} lacks text/char_span")
        start, end = span
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or end > len(spoken_text):
            raise AnchorMappingError(f"unit {index} has invalid char_span")
        if start < previous_end or spoken_text[start:end] != text:
            raise AnchorMappingError("unit char_span is not an exact monotonic lexical stream")
        spans.append((text, start, end))
        previous_end = end
    return spans


def align_units_to_lexical_items(
    spoken_text: str,
    units: Sequence[Mapping[str, Any]],
    items: Any,
    *,
    sample_rate: int,
    target_language: str | None = None,
) -> list[dict[str, Any]]:
    """Map Qwen lexical items to caller units by exact character stream.

    No substring search or fuzzy matching is used.  Item text must consume the
    same stream as ``spoken_text`` in order, and each caller unit must carry an
    exact half-open character span.
    """
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    spans = _validate_units(spoken_text, units)
    normalized = normalize_anchor_items(items)
    # Qwen emits lexical units and may omit separators.  The frozen stream is
    # therefore the exact concatenation of caller unit text, while char_span
    # still points into the original caller text.  No search is performed.
    lexical_text = "".join(span[0] for span in spans)
    cursor = 0
    mapped_items: list[tuple[int, int, float, float]] = []
    for item in normalized:
        text = item["unit"]
        end_cursor = cursor + len(text)
        if end_cursor > len(lexical_text) or lexical_text[cursor:end_cursor] != text:
            raise AnchorMappingError("forced-aligner lexical character stream mismatch")
        mapped_items.append((cursor, end_cursor, item["start_s"], item["end_s"]))
        cursor = end_cursor
    if cursor != len(lexical_text):
        raise AnchorMappingError("forced-aligner lexical character stream is incomplete")

    if target_language is not None and target_language not in {"ja", "en"}:
        raise ValueError("target_language must be ja, en, or None")
    result: list[dict[str, Any]] = []
    item_index = 0
    unit_cursor = 0
    for unit_index, (text, start, end) in enumerate(spans):
        unit_stream_start = unit_cursor
        unit_stream_end = unit_cursor + len(text)
        selected: list[tuple[float, float]] = []
        while item_index < len(mapped_items) and mapped_items[item_index][1] <= unit_stream_start:
            item_index += 1
        probe = item_index
        selected_indices: list[int] = []
        while probe < len(mapped_items) and mapped_items[probe][0] < unit_stream_end:
            item_start, item_end, begin, finish = mapped_items[probe]
            crosses = item_start < unit_stream_start or item_end > unit_stream_end
            is_target = target_language is None or units[unit_index].get("language") == target_language
            if crosses and is_target:
                raise AnchorMappingError(f"anchor item crosses unit char span for unit {unit_index}")
            selected.append((begin, finish))
            selected_indices.append(probe)
            probe += 1
        if not selected:
            raise AnchorMappingError(f"unit {unit_index} has no lexical anchor")
        crosses = any(mapped_items[index][0] < unit_stream_start or mapped_items[index][1] > unit_stream_end for index in selected_indices)
        start_s = min(value[0] for value in selected)
        end_s = max(value[1] for value in selected)
        result.append({
            **dict(units[unit_index]),
            "anchor_start_s": None if crosses else start_s,
            "anchor_end_s": None if crosses else end_s,
            "start_sample": None if crosses else int(round(start_s * sample_rate)),
            "end_sample": None if crosses else int(round(end_s * sample_rate)),
            "crosses_non_target_unit": bool(crosses),
        })
        crossing_indices = [index for index in selected_indices if mapped_items[index][0] < unit_stream_start or mapped_items[index][1] > unit_stream_end]
        item_index = min(crossing_indices) if crossing_indices else probe
        unit_cursor = unit_stream_end
    return result


def _pass_boundary_edge(
    spoken_text: str,
    units: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    left_index: int,
    sample_rate: int,
) -> dict[str, Any] | None:
    """Return real lexical-item edge timing at one caller-unit boundary."""
    spans = _validate_units(spoken_text, units)
    left_stream_end = sum(len(text) for text, _, _ in spans[: left_index + 1])
    right_stream_start = left_stream_end
    cursor = 0
    previous: Mapping[str, Any] | None = None
    for item in normalize_anchor_items(items):
        end = cursor + len(item["unit"])
        if end == left_stream_end:
            previous = item
        if cursor == right_stream_start and previous is not None:
            return {
                "left_end_sample": int(round(_number(previous["end_s"], "end") * sample_rate)),
                "right_start_sample": int(round(_number(item["start_s"], "start") * sample_rate)),
                "left_item": previous["unit"], "right_item": item["unit"],
                "char_boundary": left_stream_end,
            }
        cursor = end
    return None


def run_dual_forced_alignment(
    aligner: Any,
    audio: Path,
    spoken_text: str,
    *,
    sample_rate: int,
) -> dict[str, list[dict[str, Any]]]:
    """Run both full-utterance language passes against identical inputs."""
    if not isinstance(spoken_text, str) or not spoken_text:
        raise ValueError("spoken_text must be non-empty")
    audio_arg = str(audio)
    result: dict[str, list[dict[str, Any]]] = {}
    for language in ("Japanese", "English"):
        method = getattr(aligner, "align", None)
        if not callable(method):
            raise RuntimeError("Qwen ForcedAligner does not expose align")
        result[language] = normalize_anchor_items(method(audio_arg, spoken_text, language=language))
    return result


def run_forced_alignment(
    aligner: Any,
    audio: Path,
    spoken_text: str,
    *,
    route: str,
    sample_rate: int,
) -> dict[str, list[dict[str, Any]]]:
    """Route pure utterances to one pass and mixed utterances to both."""
    if route not in {"ja", "en", "mixed"}:
        raise ValueError("route must be ja, en, or mixed")
    if route == "mixed":
        return run_dual_forced_alignment(aligner, audio, spoken_text, sample_rate=sample_rate)
    language = "Japanese" if route == "ja" else "English"
    method = getattr(aligner, "align", None)
    if not callable(method):
        raise RuntimeError("Qwen ForcedAligner does not expose align")
    return {language: normalize_anchor_items(method(str(audio), spoken_text, language=language))}


def build_anchor_plan(
    *,
    uid: str,
    aligner: Any,
    audio: Path,
    spoken_text: str,
    units: Sequence[Mapping[str, Any]],
    route: str,
    sample_rate: int,
    total_samples: int,
    padding_samples: int = 0,
    max_disagreement_ms: int = 80,
) -> dict[str, Any]:
    """Produce the deterministic Stage 5 plan consumed by language runners."""
    passes = run_forced_alignment(aligner, audio, spoken_text, route=route, sample_rate=sample_rate)
    mapped: dict[str, list[dict[str, Any]]] = {}
    for language, items in passes.items():
        target = "ja" if language == "Japanese" else "en"
        mapped[language] = align_units_to_lexical_items(
            spoken_text, units, items, sample_rate=sample_rate, target_language=target
        )
    if not mapped:
        raise AnchorMappingError("no language pass produced anchors")
    routed_units: list[dict[str, Any]] = []
    for unit in units:
        language = unit.get("language")
        pass_name = "Japanese" if language == "ja" else "English"
        selected_rows = mapped.get(pass_name)
        if selected_rows is None:
            raise AnchorMappingError(f"missing {pass_name} pass for {unit.get('unit_id')}")
        selected = next((row for row in selected_rows if row.get("unit_id") == unit.get("unit_id")), None)
        if selected is None:
            raise AnchorMappingError(f"missing anchor for {unit.get('unit_id')}")
        routed_units.append({**dict(unit), **{key: value for key, value in selected.items() if key not in {"unit_id", "text", "char_span", "language"}}})
    runs = build_language_runs(routed_units, padding_samples=padding_samples, total_samples=total_samples)
    # A pure utterance owns the full immutable WAV axis; Qwen lexical anchors
    # are internal guidance and may begin after leading silence or end before
    # trailing silence that MFA must still align.
    if len(runs) == 1:
        runs[0]["ownership_start_sample"] = 0
        runs[0]["ownership_end_sample"] = total_samples
    seams: list[dict[str, Any]] = []
    boundary_evidence: list[dict[str, Any]] = []
    for left, right in zip(routed_units, routed_units[1:]):
        if left.get("language") == right.get("language"):
            continue
        boundaries: dict[str, list[dict[str, Any]]] = {}
        left_index = next((index for index, unit in enumerate(units) if unit.get("unit_id") == left.get("unit_id")), None)
        if left_index is None:
            raise AnchorConflictError("seam unit is absent from frozen lexical units")
        for name, raw_items in passes.items():
            edge = _pass_boundary_edge(spoken_text, units, raw_items, left_index, sample_rate)
            if edge is None:
                raise AnchorConflictError(f"{name} pass lacks a real lexical item edge at language seam")
            boundaries[name] = [{"left_unit_id": left["unit_id"], "right_unit_id": right["unit_id"], **edge}]
        boundary_evidence.append({"left_unit_id": left["unit_id"], "right_unit_id": right["unit_id"], **boundaries})
        if len(boundaries) == 2:
            new_seams = compute_integer_seams(boundaries["Japanese"], boundaries["English"], max_disagreement_ms=max_disagreement_ms, sample_rate=sample_rate)
            for seam in new_seams:
                seam["seam_id"] = f"seam_{len(seams):04d}"
                seams.append(seam)
    # The seam is the ownership boundary.  Context padding remains attached
    # to both runs, but no run owns samples beyond this integer seam.
    run_by_unit: dict[str, dict[str, Any]] = {
        unit_id: run for run in runs for unit_id in run["unit_ids"]
    }
    for seam in seams:
        left_run = run_by_unit[seam["left_unit_id"]]
        right_run = run_by_unit[seam["right_unit_id"]]
        boundary = int(seam["ownership_seam_sample"])
        left_run["ownership_end_sample"] = boundary
        right_run["ownership_start_sample"] = boundary
        if left_run["ownership_start_sample"] > boundary or right_run["ownership_end_sample"] < boundary:
            raise AnchorMappingError("computed seam falls outside unit ownership")
    for run in runs:
        run["context_start_sample"] = max(0, run["ownership_start_sample"] - padding_samples)
        run["context_end_sample"] = min(total_samples, run["ownership_end_sample"] + padding_samples)
    return {"schema": "ja-en-alignment-plan-v2", "uid": uid, "route": route, "runs": runs, "seams": seams,
            "boundary_evidence": boundary_evidence, "passes": mapped, "units": routed_units}


def _boundary(row: Mapping[str, Any], side: str) -> int:
    names = (f"{side}_sample", f"{side}_end_sample" if side == "left" else f"{side}_start_sample")
    for name in names:
        if name in row:
            value = row[name]
            if type(value) is int and value >= 0:
                return value
    raise AnchorMappingError(f"seam row lacks {side} boundary")


def compute_integer_seams(
    japanese_boundaries: Sequence[Mapping[str, Any]],
    english_boundaries: Sequence[Mapping[str, Any]],
    *,
    max_disagreement_ms: int = 80,
    sample_rate: int,
) -> list[dict[str, Any]]:
    """Combine dual-pass seam midpoints using integer sample arithmetic."""
    if len(japanese_boundaries) != len(english_boundaries):
        raise AnchorConflictError("dual pass seam cardinality mismatch")
    limit = int(max_disagreement_ms * sample_rate / 1000)
    seams: list[dict[str, Any]] = []
    for index, (ja, en) in enumerate(zip(japanese_boundaries, english_boundaries)):
        if ja.get("left_unit_id") != en.get("left_unit_id") or ja.get("right_unit_id") != en.get("right_unit_id"):
            raise AnchorConflictError(f"seam {index} unit mismatch")
        ja_mid = (_boundary(ja, "left") + _boundary(ja, "right")) // 2
        en_mid = (_boundary(en, "left") + _boundary(en, "right")) // 2
        disagreement = abs(ja_mid - en_mid)
        if disagreement > limit:
            raise AnchorConflictError(f"seam {index} disagreement exceeds {max_disagreement_ms} ms")
        seams.append({
            "seam_id": ja.get("seam_id", f"seam_{index:04d}"),
            "left_unit_id": ja["left_unit_id"],
            "right_unit_id": ja["right_unit_id"],
            "japanese_midpoint_sample": ja_mid,
            "english_midpoint_sample": en_mid,
            "ownership_seam_sample": (ja_mid + en_mid) // 2,
            "disagreement_samples": disagreement,
        })
    return seams


def build_language_runs(
    units: Sequence[Mapping[str, Any]],
    *,
    padding_samples: int,
    total_samples: int,
) -> list[dict[str, Any]]:
    """Group adjacent units by language while retaining fixed ownership."""
    if type(padding_samples) is not int or padding_samples < 0 or type(total_samples) is not int or total_samples <= 0:
        raise ValueError("invalid run padding or total samples")
    runs: list[dict[str, Any]] = []
    for unit in units:
        language = unit.get("language")
        if language not in {"ja", "en"}:
            raise ValueError("language run must use ja or en")
        start, end = unit.get("start_sample"), unit.get("end_sample")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or end > total_samples:
            raise ValueError("unit sample span is invalid")
        if runs and runs[-1]["language"] == language:
            run = runs[-1]
            run["unit_ids"].append(unit["unit_id"])
            run["ownership_end_sample"] = end
            run["context_end_sample"] = min(total_samples, end + padding_samples)
        else:
            runs.append({
                "run_id": f"{language}_run_{len(runs):04d}",
                "language": language,
                "unit_ids": [unit["unit_id"]],
                "ownership_start_sample": start,
                "ownership_end_sample": end,
                "context_start_sample": max(0, start - padding_samples),
                "context_end_sample": min(total_samples, end + padding_samples),
            })
    return runs


class QwenForcedAligner:
    """Lazy native SDK wrapper used by production callers."""

    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def from_pretrained(cls, model_path: Path, **kwargs: Any) -> "QwenForcedAligner":
        package = importlib.import_module("qwen_asr")
        model_class = getattr(package, "Qwen3ForcedAligner", None)
        if model_class is None or not callable(getattr(model_class, "from_pretrained", None)):
            raise RuntimeError("qwen_asr lacks Qwen3ForcedAligner.from_pretrained")
        return cls(model_class.from_pretrained(str(model_path), **kwargs))

    def align(self, audio: str, text: str, *, language: str) -> Any:
        return self.model.align(audio, text, language=language)


def handle_anchors(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    """Run anchors from a prepared input mapping or fail closed.

    The root dispatcher supplies ``anchors`` (or ``stage_inputs.anchors``) with
    ``audio``, ``spoken_text``, ``units``, ``route``, ``sample_rate`` and
    ``total_samples``.  A callable ``qwen_aligner`` is accepted for isolated
    tests; production loads the native SDK model from the configured path.
    """
    receipt_path = stage_dir / "receipt.json"
    raw = config.get("anchors")
    if not isinstance(raw, Mapping):
        raw = (config.get("stage_inputs") or {}).get("anchors") if isinstance(config.get("stage_inputs"), Mapping) else None
    if raw is None and (stage_dir / "input.json").is_file():
        raw = json.loads((stage_dir / "input.json").read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        receipt = make_receipt(stage="anchors", status="BLOCKED", params={"implementation": "qwen-dual-full-utterance-v1"}, errors=[{"code": "publish_blocked", "message": "anchor inputs/model runner not supplied"}])
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult(stage="anchors", status="BLOCKED", receipt_path=str(receipt_path))
    try:
        aligner = raw.get("qwen_aligner") or raw.get("aligner") or config.get("qwen_aligner")
        if aligner is None:
            model_path = raw.get("qwen_forced_aligner") or (config.get("asr") or {}).get("qwen_forced_aligner")
            if not model_path:
                raise ValueError("qwen_forced_aligner path is required")
            runtime_python = raw.get("qwen_runtime_python") or (config.get("asr") or {}).get("qwen_runtime_python")
            if runtime_python:
                request = {"audio": str(raw["audio"]), "spoken_text": str(raw.get("spoken_text", raw.get("text", ""))), "route": str(raw.get("route", "mixed")), "sample_rate": int(raw["sample_rate"]), "model": str(model_path), "device": str(raw.get("device", "cuda:0")), "dtype": str(raw.get("dtype", "bfloat16"))}
                worker_input = stage_dir / "qwen_worker_input.json"
                worker_output = stage_dir / "qwen_worker_output.json"
                atomic_write_json(worker_input, request, workspace=stage_dir.parent.parent)
                command = [str(runtime_python), str(Path(__file__).resolve()), "--worker-input", str(worker_input), "--worker-output", str(worker_output)]
                completed = subprocess.run(command, check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or "Qwen anchor worker failed")
                worker_payload = json.loads(worker_output.read_text(encoding="utf-8"))
                if worker_payload.get("status") != "COMPLETE":
                    raise RuntimeError(str(worker_payload.get("error", "Qwen anchor worker rejected")))
                aligner = _StaticPassAligner(worker_payload["passes"])
            else:
                aligner = QwenForcedAligner.from_pretrained(Path(str(model_path)))
        units = raw.get("units")
        if isinstance(units, (str, Path)):
            units = json.loads(Path(units).read_text(encoding="utf-8"))
        if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
            raise ValueError("anchors.units must be a sequence")
        plan = build_anchor_plan(
            uid=str(raw.get("uid", config.get("uid", ""))), aligner=aligner,
            audio=Path(str(raw["audio"])), spoken_text=str(raw.get("spoken_text", raw.get("text", ""))),
            units=units, route=str(raw.get("route", "mixed")),
            sample_rate=int(raw["sample_rate"]), total_samples=int(raw["total_samples"]),
            padding_samples=int(raw.get("padding_samples", 0)),
            max_disagreement_ms=int(raw.get("max_disagreement_ms", 80)),
        )
        plan_path = stage_dir / ANCHOR_PLAN_FILENAME
        runs_path = stage_dir / ANCHOR_RUNS_FILENAME
        atomic_write_json(plan_path, plan, workspace=stage_dir.parent.parent)
        atomic_write_json(runs_path, {"schema": plan["schema"], "uid": plan["uid"], "runs": plan["runs"], "seams": plan["seams"]}, workspace=stage_dir.parent.parent)
        stage_outputs = [plan_path, runs_path]
        for worker_artifact in (stage_dir / "qwen_worker_input.json", stage_dir / "qwen_worker_output.json"):
            if worker_artifact.is_file():
                stage_outputs.append(worker_artifact)
        receipt = make_receipt(stage="anchors", status="COMPLETE", inputs={"uid": plan["uid"], "route": plan["route"]}, outputs=stage_outputs, params={"implementation": "qwen-dual-full-utterance-v1", "anchor_max_disagreement_ms": int(raw.get("max_disagreement_ms", 80))})
    except Exception as exc:
        receipt = make_receipt(stage="anchors", status="REJECTED", params={"implementation": "qwen-dual-full-utterance-v1"}, errors=[{"code": "anchor_invalid", "message": str(exc)}])
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="anchors", status=receipt["status"], receipt_path=str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("anchors", handle_anchors, output_namespace="anchors")


class _StaticPassAligner:
    def __init__(self, passes: Mapping[str, Any]):
        self.passes = passes

    def align(self, audio: str, text: str, *, language: str) -> Any:
        return self.passes[language]


def _worker_main(input_path: Path, output_path: Path) -> int:
    request = json.loads(input_path.read_text(encoding="utf-8"))
    try:
        torch = importlib.import_module("torch")
        dtype_name = str(request.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        model = QwenForcedAligner.from_pretrained(Path(request["model"]), dtype=dtype, device_map=request.get("device", "cuda:0"))
        passes = run_forced_alignment(model, Path(request["audio"]), request["spoken_text"], route=request["route"], sample_rate=int(request["sample_rate"]))
        output_path.write_text(json.dumps({"status": "COMPLETE", "passes": passes}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        output_path.write_text(json.dumps({"status": "REJECTED", "error": str(exc)}, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen JA/EN anchor worker")
    parser.add_argument("--worker-input", required=True)
    parser.add_argument("--worker-output", required=True)
    args = parser.parse_args(argv)
    return _worker_main(Path(args.worker_input), Path(args.worker_output))


__all__ = [
    "AnchorConflictError", "AnchorMappingError", "QwenForcedAligner", "align_units_to_lexical_items",
    "build_language_runs", "compute_integer_seams", "handle_anchors", "normalize_anchor_items",
    "register_stages", "run_dual_forced_alignment", "run_forced_alignment", "build_anchor_plan",
    "ANCHOR_PLAN_FILENAME", "ANCHOR_RUNS_FILENAME", "ANCHOR_REQUIRED_KEYS",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
