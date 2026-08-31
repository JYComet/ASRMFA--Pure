"""Provider-neutral Qwen lexical/NVASR candidate fusion.

The NVASR producer supplies lexical occurrences and candidate neighbors, not
Qwen gap numbers.  This module globally aligns those occurrences to the Qwen
sequence using only explicit lexical equality.  Qwen remains the sole lexical
authority: NVASR lexical substitutions are evidence for alignment only and
are never copied into the fused sequence.

No ASR provider is imported here.  In particular, this module does not use
ordinal zipping or timestamp-nearest matching.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import re
from typing import Any


TIMELINE_SCHEMA = "nvasr-candidate-timeline-v1"
FUSION_SCHEMA = "qwen3asr-anchored-nvv-v1"
TIMING_LABEL = "qwen3_forced_aligner"
QUERY_FRAMES = 4
FRAME_MS = 60
FRAME_SECONDS = FRAME_MS / 1000.0
_COORD_TOLERANCE = 1e-9
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_LATIN_NUMERIC_RE = re.compile(r"[A-Za-z0-9]+")


class FusionError(ValueError):
    """Raised when global evidence is malformed."""


def candidate_id_for_occurrence(occurrence: int) -> str:
    """Return the stable identifier for a zero-based candidate occurrence."""

    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
        raise FusionError("candidate occurrence must be a non-negative integer")
    return f"nvasr-candidate-{occurrence:04d}"


def _error(path: str, message: str) -> FusionError:
    return FusionError(f"{path}: {message}")


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise _error(path, "must be finite")
    return number


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(path, "must be an integer")
    return value


def _nonempty_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "must be a non-empty string")
    return value


def _optional_ordinal(value: Any, path: str) -> int | None:
    if value is None:
        return None
    ordinal = _integer(value, path)
    if ordinal < 0:
        raise _error(path, "must be non-negative or null")
    return ordinal


def _coordinate_pair(
    record: Mapping[str, Any], *, prefix: str, path: str,
) -> tuple[int, float]:
    frame_key = f"{prefix}_frame"
    seconds_key = f"{prefix}_s"
    if frame_key not in record or seconds_key not in record:
        raise _error(path, f"requires both {frame_key} and {seconds_key}")
    frame = _integer(record[frame_key], f"{path}.{frame_key}")
    seconds = _finite_number(record[seconds_key], f"{path}.{seconds_key}")
    return frame, seconds


def _coordinates(record: Mapping[str, Any], path: str) -> dict[str, Any]:
    points = {
        name: _coordinate_pair(record, prefix=name, path=path)
        for name in ("raw_start", "raw_end", "speech_start", "speech_end")
    }
    raw_start_frame, raw_start_s = points["raw_start"]
    raw_end_frame, raw_end_s = points["raw_end"]
    speech_start_frame, speech_start_s = points["speech_start"]
    speech_end_frame, speech_end_s = points["speech_end"]
    for name, start_frame, end_frame, start_s, end_s in (
        ("raw", raw_start_frame, raw_end_frame, raw_start_s, raw_end_s),
        ("speech", speech_start_frame, speech_end_frame, speech_start_s, speech_end_s),
    ):
        if start_frame < 0 or end_frame < start_frame:
            raise _error(path, f"{name} frame span is invalid")
        if start_s < 0 or end_s < start_s:
            raise _error(path, f"{name} second span is invalid")
        if not math.isclose(start_s, start_frame * FRAME_SECONDS,
                            rel_tol=0.0, abs_tol=_COORD_TOLERANCE):
            raise _error(path, f"{name} start seconds do not match frame coordinate")
        if not math.isclose(end_s, end_frame * FRAME_SECONDS,
                            rel_tol=0.0, abs_tol=_COORD_TOLERANCE):
            raise _error(path, f"{name} end seconds do not match frame coordinate")
    if raw_start_frame < QUERY_FRAMES:
        raise _error(path, "raw frame coordinate precedes the query prefix")
    if speech_start_frame != raw_start_frame - QUERY_FRAMES:
        raise _error(path, "speech start frame is not raw start minus query_frames")
    if speech_end_frame != raw_end_frame - QUERY_FRAMES:
        raise _error(path, "speech end frame is not raw end minus query_frames")
    return {
        "raw_start_frame": raw_start_frame,
        "raw_end_frame": raw_end_frame,
        "raw_start_s": raw_start_s,
        "raw_end_s": raw_end_s,
        "speech_start_frame": speech_start_frame,
        "speech_end_frame": speech_end_frame,
        "speech_start_s": speech_start_s,
        "speech_end_s": speech_end_s,
    }


def _validate_lexical_occurrences(
    occurrences: Any,
) -> list[dict[str, Any]]:
    if isinstance(occurrences, (str, bytes)) or not isinstance(occurrences, Sequence):
        raise FusionError("lexical_occurrences must be a sequence")
    normalized: list[dict[str, Any]] = []
    previous_raw_end = -1
    for index, occurrence in enumerate(occurrences):
        path = f"lexical_occurrences[{index}]"
        if not isinstance(occurrence, Mapping):
            raise _error(path, "must be an object")
        ordinal = _integer(occurrence.get("lexical_ordinal"), f"{path}.lexical_ordinal")
        if ordinal != index:
            raise _error(path, "lexical_ordinal must be contiguous and ordered")
        surface = _nonempty_text(occurrence.get("surface"), f"{path}.surface")
        kind = _nonempty_text(occurrence.get("kind", "lexical"), f"{path}.kind")
        token_id = _integer(occurrence.get("token_id"), f"{path}.token_id")
        token_ids = occurrence.get("token_ids")
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence) or not token_ids:
            raise _error(path, "token_ids must be a non-empty sequence")
        token_ids = [_integer(token, f"{path}.token_ids") for token in token_ids]
        coordinates = _coordinates(occurrence, path)
        if coordinates["raw_start_frame"] < previous_raw_end:
            raise _error(path, "lexical occurrences are not temporally ordered")
        previous_raw_end = coordinates["raw_end_frame"]
        normalized_row = {
            "lexical_ordinal": index,
            "surface": surface,
            "kind": kind,
            "token_id": token_id,
            "token_ids": token_ids,
            **coordinates,
        }
        for key in (
            "lexical_occurrence_id", "occurrence_id", "surface_occurrence",
            "surface_occurrence_index", "surface_occurrence_id", "source",
        ):
            if key in occurrence:
                normalized_row[key] = occurrence[key]
        normalized.append(normalized_row)
    return normalized


def _validate_timeline(timeline: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(timeline, Mapping):
        raise FusionError("candidate timeline must be an object")
    if timeline.get("schema") != TIMELINE_SCHEMA:
        raise FusionError(f"candidate timeline schema must be {TIMELINE_SCHEMA}")
    if timeline.get("query_frames") != QUERY_FRAMES:
        raise FusionError("query_frames must be exactly 4")
    if timeline.get("frame_ms") != FRAME_MS:
        raise FusionError("frame_ms must be exactly 60")
    stem = _nonempty_text(timeline.get("stem"), "stem")
    duration_s = _finite_number(timeline.get("duration_s"), "duration_s")
    if duration_s < 0:
        raise FusionError("duration_s must be non-negative")
    diagnostic = timeline.get("diagnostic", timeline.get("diagnostic_text"))
    if not isinstance(diagnostic, str):
        raise _error("diagnostic", "must be text")
    lexical_occurrences = _validate_lexical_occurrences(
        timeline.get("lexical_occurrences")
    )
    candidates = timeline.get("candidates")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise FusionError("candidates must be a sequence")
    normalized_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_raw_start = -1
    for index, candidate in enumerate(candidates):
        path = f"candidates[{index}]"
        if not isinstance(candidate, Mapping):
            raise _error(path, "must be an object")
        candidate_id = _nonempty_text(candidate.get("candidate_id"), f"{path}.candidate_id")
        if candidate_id != candidate_id_for_occurrence(index):
            raise _error(path, "candidate_id is not a deterministic occurrence id")
        if candidate_id in seen_ids:
            raise _error(path, "candidate_id is duplicated")
        seen_ids.add(candidate_id)
        occurrence = _integer(candidate.get("occurrence"), f"{path}.occurrence")
        if occurrence != index:
            raise _error(path, "occurrence is not ordered")
        label = _nonempty_text(candidate.get("label", candidate.get("surface")), f"{path}.label")
        surface = _nonempty_text(candidate.get("surface", label), f"{path}.surface")
        source = _nonempty_text(candidate.get("source"), f"{path}.source")
        kind = _nonempty_text(candidate.get("kind"), f"{path}.kind")
        token_id = _integer(candidate.get("token_id"), f"{path}.token_id")
        token_ids = candidate.get("token_ids")
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence) or not token_ids:
            raise _error(path, "token_ids must be a non-empty sequence")
        token_ids = [_integer(token, f"{path}.token_ids") for token in token_ids]
        for neighbor_key in ("left_lexical_ordinal", "right_lexical_ordinal"):
            if neighbor_key not in candidate:
                raise _error(path, f"requires {neighbor_key}")
        left = _optional_ordinal(
            candidate.get("left_lexical_ordinal"), f"{path}.left_lexical_ordinal"
        )
        right = _optional_ordinal(
            candidate.get("right_lexical_ordinal"), f"{path}.right_lexical_ordinal"
        )
        if left is not None and left >= len(lexical_occurrences):
            raise _error(path, "left_lexical_ordinal is out of range")
        if right is not None and right >= len(lexical_occurrences):
            raise _error(path, "right_lexical_ordinal is out of range")
        coordinates = _coordinates(candidate, path)
        if coordinates["raw_start_frame"] < previous_raw_start:
            raise _error(path, "candidates are not temporally ordered")
        previous_raw_start = coordinates["raw_start_frame"]
        candidate_outside_duration = (
            coordinates["speech_start_s"] < -_COORD_TOLERANCE
            or coordinates["speech_end_s"] > duration_s + _COORD_TOLERANCE
        )
        candidate_diagnostic = candidate.get("diagnostic", diagnostic)
        if not isinstance(candidate_diagnostic, str):
            raise _error(f"{path}.diagnostic", "must be text")
        normalized_candidates.append({
            "candidate_id": candidate_id,
            "occurrence": index,
            "label": label,
            "surface": surface,
            "source": source,
            "kind": kind,
            "token_id": token_id,
            "token_ids": token_ids,
            "left_lexical_ordinal": left,
            "right_lexical_ordinal": right,
            "diagnostic": candidate_diagnostic,
            "candidate_outside_duration": candidate_outside_duration,
            **coordinates,
        })
    return {
        "schema": TIMELINE_SCHEMA,
        "stem": stem,
        "query_frames": QUERY_FRAMES,
        "frame_ms": FRAME_MS,
        "duration_s": duration_s,
        "diagnostic": diagnostic,
        "lexical_occurrences": lexical_occurrences,
        "candidates": normalized_candidates,
    }


def validate_candidate_timeline(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached v3 candidate timeline."""

    return _validate_timeline(timeline)


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _is_latin_numeric(text: str) -> bool:
    return bool(_LATIN_NUMERIC_RE.fullmatch(text))


def _lexical_match(nvasr_surface: str, qwen_surface: str) -> bool:
    """Match only exact CJK or case-insensitive ASCII Latin/numeric units."""

    nvasr_surface = nvasr_surface.strip()
    qwen_surface = qwen_surface.strip()
    if _is_cjk(nvasr_surface) or _is_cjk(qwen_surface):
        return nvasr_surface == qwen_surface
    if _is_latin_numeric(nvasr_surface) and _is_latin_numeric(qwen_surface):
        return nvasr_surface.casefold() == qwen_surface.casefold()
    return False


def _optimal_monotonic_mappings(
    nvasr_surfaces: Sequence[str], qwen_units: Sequence[str],
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate unique maximum-cardinality monotonic lexical mappings."""

    n_count = len(nvasr_surfaces)
    q_count = len(qwen_units)
    scores = [[0] * (q_count + 1) for _ in range(n_count + 1)]
    for n_index in range(n_count - 1, -1, -1):
        for q_index in range(q_count - 1, -1, -1):
            score = max(scores[n_index + 1][q_index], scores[n_index][q_index + 1])
            if _lexical_match(nvasr_surfaces[n_index], qwen_units[q_index]):
                score = max(score, 1 + scores[n_index + 1][q_index + 1])
            scores[n_index][q_index] = score

    memo: dict[tuple[int, int], set[tuple[tuple[int, int], ...]]] = {}

    def enumerate_from(n_index: int, q_index: int) -> set[tuple[tuple[int, int], ...]]:
        key = (n_index, q_index)
        if key in memo:
            return memo[key]
        if n_index == n_count or q_index == q_count:
            memo[key] = {()}
            return memo[key]
        target = scores[n_index][q_index]
        mappings: set[tuple[tuple[int, int], ...]] = set()
        if scores[n_index + 1][q_index] == target:
            mappings.update(enumerate_from(n_index + 1, q_index))
        if scores[n_index][q_index + 1] == target:
            mappings.update(enumerate_from(n_index, q_index + 1))
        if (_lexical_match(nvasr_surfaces[n_index], qwen_units[q_index])
                and 1 + scores[n_index + 1][q_index + 1] == target):
            for suffix in enumerate_from(n_index + 1, q_index + 1):
                mappings.add(((n_index, q_index),) + suffix)
        memo[key] = mappings
        return mappings

    return sorted(enumerate_from(0, 0))


def _mapping_lookup(mapping: tuple[tuple[int, int], ...]) -> dict[int, int]:
    return dict(mapping)


def _forced_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise FusionError("forced_aligner_items must be a sequence")
    normalized: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, item in enumerate(items):
        path = f"forced_aligner_items[{index}]"
        if not isinstance(item, Mapping):
            raise _error(path, "must be an object")
        unit = _nonempty_text(item.get("unit", item.get("text")), f"{path}.unit")
        start_s = _finite_number(item.get("start_s", item.get("start")), f"{path}.start_s")
        end_s = _finite_number(item.get("end_s", item.get("end")), f"{path}.end_s")
        if start_s < 0 or end_s <= start_s:
            raise _error(path, "forced-aligner span must satisfy 0 <= start_s < end_s")
        if start_s < previous_end - _COORD_TOLERANCE:
            raise _error(path, "forced-aligner items are not monotonic")
        previous_end = end_s
        normalized.append({"unit": unit, "start_s": start_s, "end_s": end_s})
    return normalized


def _qwen_units(units: Sequence[str]) -> list[str]:
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise FusionError("qwen_lexical_units must be an explicit sequence")
    return [_nonempty_text(unit, f"qwen_lexical_units[{index}]")
            for index, unit in enumerate(units)]


def _failure_result(
    *, timeline: dict[str, Any], qwen_units: list[str],
    aligner_items: list[dict[str, Any]], accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]], alignment: list[tuple[tuple[int, int], ...]],
) -> dict[str, Any]:
    all_rows = accepted + rejected
    return {
        "schema": FUSION_SCHEMA,
        "status": "FAILED",
        "stem": timeline["stem"],
        "lexical_authority": "qwen",
        "qwen_lexical_units": qwen_units,
        "forced_aligner_items": aligner_items,
        "duration_s": timeline["duration_s"],
        "lexical_timing_source": TIMING_LABEL,
        "fused_lexical_units": qwen_units,
        "lexical_alignment": [list(pair) for pair in alignment[0]] if alignment else [],
        "optimal_mapping_count": len(alignment),
        "accepted": accepted,
        "rejected": rejected,
        "candidate_conservation": {
            "input": len(timeline["candidates"]),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "exactly_once": len(all_rows) == len(timeline["candidates"])
            and len({row["candidate_id"] for row in all_rows})
            == len(timeline["candidates"]),
        },
    }


_CANDIDATE_TIMING_SOURCES = {
    "ctc": "nvasr_ctc_free_decode",
    "blank_run": "nvasr_blank_pause_heuristic",
}


def _derive_placement(
    candidate: Mapping[str, Any],
    *,
    left_qwen: int,
    right_qwen: int,
    aligner_items: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> tuple[int | None, str | None, int | None, dict[str, float] | None]:
    """Derive one Qwen boundary from an explicitly mapped owner pair."""

    if candidate["speech_start_s"] < -_COORD_TOLERANCE or candidate["speech_end_s"] > duration_s + _COORD_TOLERANCE:
        return None, "candidate_outside_duration", None, None

    envelope_start = aligner_items[left_qwen]["end_s"]
    envelope_end = aligner_items[right_qwen]["start_s"]
    if envelope_end < envelope_start - _COORD_TOLERANCE:
        return None, "invalid_qwen_anchor_envelope", None, None
    start = candidate["speech_start_s"]
    end = candidate["speech_end_s"]
    if start < envelope_start - _COORD_TOLERANCE or end > envelope_end + _COORD_TOLERANCE:
        return None, "candidate_outside_qwen_anchor_envelope", None, None

    unmatched = list(range(left_qwen + 1, right_qwen))
    if not unmatched:
        return right_qwen, "inter_anchor", None, None

    overlaps = [
        ordinal for ordinal in unmatched
        if start < aligner_items[ordinal]["end_s"] - _COORD_TOLERANCE
        and end > aligner_items[ordinal]["start_s"] + _COORD_TOLERANCE
    ]
    if len(overlaps) > 1:
        reason = "straddling_qwen_ambiguity" if len(overlaps) == 2 else "multi_qwen_overlap"
        return None, reason, None, None
    if len(overlaps) == 1:
        overlap_ordinal = overlaps[0]
        target_start = aligner_items[overlap_ordinal]["start_s"]
        candidate_duration = end - start
        projected_end = target_start
        projected_start = max(envelope_start, projected_end - candidate_duration)
        return overlap_ordinal, "overlap", overlap_ordinal, {
            "qwen_projected_start_s": projected_start,
            "qwen_projected_end_s": projected_end,
        }

    first = unmatched[0]
    last = unmatched[-1]
    if end <= aligner_items[first]["start_s"] + _COORD_TOLERANCE:
        return first, "before", None, None
    if start >= aligner_items[last]["end_s"] - _COORD_TOLERANCE:
        return right_qwen, "after", None, None
    for ordinal, next_ordinal in zip(unmatched, unmatched[1:]):
        if (
            start >= aligner_items[ordinal]["end_s"] - _COORD_TOLERANCE
            and end <= aligner_items[next_ordinal]["start_s"] + _COORD_TOLERANCE
        ):
            return next_ordinal, "inter_anchor", None, None
    return None, "straddling_qwen_ambiguity", None, None


def fuse_qwen_nvasr_candidates(
    qwen_lexical_units: Sequence[str],
    forced_aligner_items: Sequence[Mapping[str, Any]],
    candidate_timeline: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse candidates using all globally optimal monotonic lexical mappings.

    Candidate neighbor ordinals are NVASR ordinals.  A candidate is accepted
    only if every optimal mapping gives one unambiguous owner pair and its
    speech interval has one unambiguous relation to the Qwen anchors.  Qwen
    lexical units are never replaced.  Candidate-level failures are
    structured and fail the stem; malformed global evidence raises
    :class:`FusionError`.
    """

    qwen_units = _qwen_units(qwen_lexical_units)
    aligner_items = _forced_items(forced_aligner_items)
    if len(qwen_units) != len(aligner_items):
        raise FusionError("Qwen lexical units do not exactly match forced-aligner item count")
    if [item["unit"] for item in aligner_items] != qwen_units:
        raise FusionError("Qwen lexical units do not exactly match forced-aligner units")
    timeline = _validate_timeline(candidate_timeline)
    if any(item["end_s"] > timeline["duration_s"] + _COORD_TOLERANCE
           for item in aligner_items):
        raise FusionError("forced-aligner item exceeds duration_s")
    nvasr_occurrences = timeline["lexical_occurrences"]
    nvasr_surfaces = [row["surface"] for row in nvasr_occurrences]
    mappings = _optimal_monotonic_mappings(nvasr_surfaces, qwen_units)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    previous_boundary = -1
    previous_end = -1.0
    last_same_boundary: dict[int, dict[str, Any]] = {}

    for candidate in timeline["candidates"]:
        left = candidate["left_lexical_ordinal"]
        right = candidate["right_lexical_ordinal"]
        reason: str | None = None
        boundary: int | None = None
        temporal_relation: str | None = None
        overlap_ordinal: int | None = None
        projection: dict[str, float] | None = None
        timing_source = _CANDIDATE_TIMING_SOURCES.get(candidate["source"])
        if timing_source is None:
            reason = "unsupported_candidate_source"
        if reason is None and left is None and right is None:
            reason = "one_sided_lexical_edge"
        elif reason is None and (left is None or right is None):
            owner = right if left is None else left
            assert owner is not None
            owner_positions = {
                _mapping_lookup(mapping)[owner]
                for mapping in mappings
                if owner in _mapping_lookup(mapping)
            }
            if len(owner_positions) != 1:
                reason = (
                    "unmapped_lexical_neighbor"
                    if not owner_positions else "ambiguous_lexical_mapping"
                )
            else:
                owner_qwen = next(iter(owner_positions))
                if left is None and owner_qwen != 0:
                    reason = "edge_owner_not_qwen_first"
                elif right is None and owner_qwen != len(qwen_units) - 1:
                    reason = "edge_owner_not_qwen_last"
                else:
                    if left is None:
                        edge_start = 0.0
                        edge_end = aligner_items[0]["start_s"]
                        edge_boundary = 0
                        edge_relation = "before"
                    else:
                        edge_start = aligner_items[-1]["end_s"]
                        edge_end = timeline["duration_s"]
                        edge_boundary = len(qwen_units)
                        edge_relation = "after"
                    if (
                        candidate["speech_start_s"] < edge_start - _COORD_TOLERANCE
                        or candidate["speech_end_s"] > edge_end + _COORD_TOLERANCE
                    ):
                        reason = "candidate_outside_qwen_edge"
                    else:
                        boundary = edge_boundary
                        temporal_relation = edge_relation
        elif reason is None and right != left + 1:
            reason = "skipped_nvasr_lexical_region"
        elif reason is None and not mappings:
            reason = "unmapped_lexical_neighbor"
        elif reason is None:
            owner_pairs: set[tuple[int, int]] = set()
            saw_unmapped = False
            for mapping in mappings:
                lookup = _mapping_lookup(mapping)
                if left not in lookup or right not in lookup:
                    saw_unmapped = True
                    continue
                q_left = lookup[left]
                q_right = lookup[right]
                owner_pairs.add((q_left, q_right))
            if saw_unmapped:
                reason = "unmapped_lexical_neighbor"
            elif len(owner_pairs) > 1:
                reason = "ambiguous_lexical_mapping"
            elif len(owner_pairs) != 1:
                reason = "ambiguous_lexical_mapping"
            else:
                q_left, q_right = next(iter(owner_pairs))
                boundary, temporal_relation, overlap_ordinal, projection = _derive_placement(
                    candidate,
                    left_qwen=q_left,
                    right_qwen=q_right,
                    aligner_items=aligner_items,
                    duration_s=timeline["duration_s"],
                )
                if boundary is None:
                    reason = temporal_relation
                    temporal_relation = None
                    overlap_ordinal = None
                    projection = None
                elif boundary < previous_boundary:
                    reason = "non_monotonic_insertion_boundary"
                elif boundary == previous_boundary and candidate["speech_start_s"] + _COORD_TOLERANCE < previous_end:
                    reason = "ambiguous_candidate_timing"

        row = deepcopy(candidate)
        row["timing_source"] = timing_source
        row.pop("candidate_outside_duration", None)
        if boundary is not None:
            row["qwen_insertion_boundary"] = boundary
            row["qwen_left_ordinal"] = boundary - 1 if boundary else None
            row["qwen_right_ordinal"] = boundary if boundary < len(qwen_units) else None
            row["temporal_relation"] = temporal_relation
            if overlap_ordinal is not None:
                row["qwen_overlap_ordinal"] = overlap_ordinal
            if projection is not None:
                row.update(projection)
        if reason is not None:
            if reason == "ambiguous_candidate_timing" and boundary is not None:
                previous = last_same_boundary.get(boundary)
                if previous is not None:
                    accepted.remove(previous)
                    previous["reason"] = reason
                    rejected.append(previous)
            row["reason"] = reason
            rejected.append(row)
            continue
        row["insertion_index"] = boundary + sum(
            1 for prior in accepted if prior.get("qwen_insertion_boundary") == boundary
        )
        accepted.append(row)
        last_same_boundary[boundary] = row
        previous_boundary = boundary
        previous_end = candidate["speech_end_s"]

    if rejected:
        return _failure_result(
            timeline=timeline, qwen_units=qwen_units,
            aligner_items=aligner_items, accepted=accepted,
            rejected=rejected, alignment=mappings,
        )

    by_boundary: dict[int, list[dict[str, Any]]] = {}
    for row in accepted:
        by_boundary.setdefault(row["qwen_insertion_boundary"], []).append(row)
    fused: list[str] = []
    for boundary in range(len(qwen_units) + 1):
        fused.extend(row["label"] for row in by_boundary.get(boundary, ()))
        if boundary < len(qwen_units):
            fused.append(qwen_units[boundary])
    return {
        "schema": FUSION_SCHEMA,
        "status": "COMPLETE",
        "stem": timeline["stem"],
        "lexical_authority": "qwen",
        "qwen_lexical_units": qwen_units,
        "forced_aligner_items": aligner_items,
        "duration_s": timeline["duration_s"],
        "lexical_timing_source": TIMING_LABEL,
        "fused_lexical_units": fused,
        "lexical_alignment": [list(pair) for pair in mappings[0]] if mappings else [],
        "optimal_mapping_count": len(mappings),
        "accepted": accepted,
        "rejected": [],
        "candidate_conservation": {
            "input": len(timeline["candidates"]),
            "accepted": len(accepted),
            "rejected": 0,
            "exactly_once": len(accepted) == len(timeline["candidates"])
            and len({row["candidate_id"] for row in accepted})
            == len(timeline["candidates"]),
        },
    }


fuse_candidate_timeline = fuse_qwen_nvasr_candidates
validate_nvasr_candidate_timeline = validate_candidate_timeline


__all__ = [
    "FRAME_MS", "FRAME_SECONDS", "FUSION_SCHEMA", "FusionError",
    "QUERY_FRAMES", "TIMELINE_SCHEMA", "TIMING_LABEL",
    "candidate_id_for_occurrence", "fuse_candidate_timeline",
    "fuse_qwen_nvasr_candidates", "validate_candidate_timeline",
    "validate_nvasr_candidate_timeline",
]
