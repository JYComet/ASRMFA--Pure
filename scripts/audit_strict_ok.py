#!/usr/bin/env python3
"""Independent disk auditor for the strict-ok v3.1 publication contract.

This program deliberately does not trust postprocess' self-report.  It rereads
the final TextGrid, source MFA alignment, CTC bundle, reference, and WAV from
disk.  Failed candidates are isolated into the run-local ``filtered`` folder;
the command is not a repair tool and never deletes an input result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import os
import re
import shutil
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (  # noqa: E402
    is_english_token, is_nvv_token, is_pinyin_syllable, is_punct,
    is_silence, is_unknown_token, is_english_phone, validate_ctc_transcript_bundle,
    PIPELINE_ACCOUNTING_SCHEMA, read_pipeline_accounting_receipt,
    validate_pipeline_accounting_receipt,
    stable_json_digest,
    CTC_RAW_MANIFEST_NAME, CTC_WORK_RECEIPT_NAME,
    validate_ctc_raw_manifest, validate_ctc_work_receipt,
    normalize_authority_reference_numerals,
    _axis_audio_metadata,
)
from english_units import (  # noqa: E402
    EnglishUnitError, canonicalize_english_token, is_english_fragment_token,
    parse_english_units,
    project_authority_semantics,
    resolve_processed_english_token,
    validate_processed_english_token_binding,
)
from postprocess_textgrids import parse_textgrid, tier_by_name  # noqa: E402

POLICY_VERSION = "strict-ok-v3.2"
EN_PROVENANCE_SCHEMA = "strict-en-mfa-v2"
HISTORICAL_EN_PROVENANCE_SCHEMA = "strict-en-mfa-v1"
CANONICAL_UNITS_SCHEMA = "canonical-english-units-v1"
SOS_PRONUNCIATION_POLICY_ID = "sos-exact-override-v1"
SOS_EXPECTED_PRONUNCIATION = ("EH2", "S", "OW2", "EH1", "S")
APP_EXPECTED_PRONUNCIATION = ("AE1", "P")
STRICT_REPLAY_SCHEMA = "strict-replay-import-v2.1"
STRICT_REPLAY_CANONICAL_SCHEMA = "mfa-quality-canonical-samples-v1"
STRICT_REPLAY_CANONICAL_SHA256 = "d88b9ac874283dbc67dc38003fb78d872b799597ce940175a8301f78aa2c5bcf"
TIER_NAMES = ["raw_text", "pinyin", "hanzi", "words", "pinyin_phones"]
EPS = 0.003
MFA_INPUT_AXIS_SCHEMA = "mfa-input-axis-receipt-v1"
MFA_ALIGNMENT_AXIS_SCHEMA = "mfa-alignment-axis-receipt-v1"
MFA_ALIGNMENT_AXIS_V2_SCHEMA = "mfa-alignment-axis-receipt-v2"
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_NVV = re.compile(r"<([A-Za-z][A-Za-z-]*)>")
_ENGLISH = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_PINYIN = re.compile(r"^[a-z]+[1-5]$")
_SP1 = re.compile(r"<sp1>", re.I)
_UNKNOWN_REPAIR_PROOF_SCHEMA = "mfa-unknown-recovery-proof-v1"
_EVIDENCE_REPAIR_SCHEMA = "evidence-constrained-repair-v1"
CTC_LIFECYCLE_SCHEMA = "ctc-processed-input-lifecycle-v1"
PROCESSED_GEOMETRY_SCHEMA = "processed-words-geometry-v1"
NVASR_CANDIDATE_SCHEMA_VERSION = 3
NVASR_CANDIDATE_PROVENANCE_SCHEMA = "nvasr-candidate-provenance-v1"
NVASR_MAPPING_BASIS = "raw_ctc_label_neighbors_forced_overlap-v2"
NVASR_MAPPING_AXIS = "non_nvv_compact_v1"
NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA = "nvasr-raw-timeline-neighbors-v1"
NVASR_SPIKE_ANCHOR_SCHEMA = "ctc_spike_anchor_v2"
NVASR_ANCHOR_COORDINATE_SYSTEM = \
    "speech_seconds_from_ctc_encoder_frames"
NVASR_ANCHOR_QUANTIZATION = \
    "half_open_60ms_frames_centered_30ms_round_half_up"
CTC_RAW_TOKEN_ROW_SCHEMA = "ctc_raw_token_row_v1"
NVASR_PRODUCER_AUTHORITY_SCHEMA = "nvasr-producer-authority-v1"
NVASR_IMMUTABLE_PROJECTION_SCHEMA = "nvasr-immutable-projection-v1"
NVASR_QUERY_FRAMES = 4
NVASR_FRAME_MS = 60

# A sidecar row may retain only its top-level schema after fields have been
# deleted.  Keep those current CTC/NVASR envelopes as evidence of the modern
# contract so that malformed rows cannot fall back to the legacy path.
_NVASR_MODERN_SCHEMA_MARKERS = frozenset({
    CTC_LIFECYCLE_SCHEMA,
    "nvasr-candidate-timeline-v1",
    NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA,
    NVASR_SPIKE_ANCHOR_SCHEMA,
    CTC_RAW_TOKEN_ROW_SCHEMA,
    NVASR_CANDIDATE_PROVENANCE_SCHEMA,
    NVASR_PRODUCER_AUTHORITY_SCHEMA,
    NVASR_IMMUTABLE_PROJECTION_SCHEMA,
    "ctc-frame-support-v1",
    "nvasr-owner-selection-v2",
})
_MODERN_REPORT_CONTRACT_KEYS = frozenset({
    "ctc_lifecycle", "nvasr_producer_authority",
    "nvasr_candidate_provenance", "nvasr_owner_selection",
    "nvasr_frame_support",
})


def _canonicalize_reference_hyphens(text: str) -> str:
    """Remove lexical ASCII hyphens without changing bracketed NVV labels."""
    result: list[str] = []
    in_angle_label = False
    for char in str(text):
        if char == "<":
            in_angle_label = True
        elif char == ">":
            in_angle_label = False
        if char != "-" or in_angle_label:
            result.append(char)
    return "".join(result)


def _evidence_digest(value: object) -> str:
    payload = ({key: item for key, item in value.items() if key != "digest"}
               if isinstance(value, dict) else value)
    canonical = json.loads(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _lexical_identity(value: object, *, ctc_item: dict | None = None) -> str:
    """Mirror postprocess lexical identity for independent disk auditing."""
    text = str(value or "").strip()
    if isinstance(ctc_item, dict):
        canonical = ctc_item.get("canonical_unit")
        if isinstance(canonical, dict):
            match_key = canonical.get("match_key")
            if isinstance(match_key, str) and match_key:
                return match_key.casefold()
    if is_unknown_token(text):
        return "<unknown>"
    if is_english_token(text):
        try:
            return canonicalize_english_token(text)
        except (EnglishUnitError, TypeError, ValueError):
            pass
    return text.casefold()
_SILENCE = re.compile(r"<sp[0-3]>", re.I)
_PUNCT_MAP = str.maketrans({",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："})


def _inside(root: Path, path: Path) -> bool:
    """True only when a resolved path stays beneath a resolved root."""
    try:
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError:
        return False
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _safe_file_under(root: Path, raw: object) -> Path:
    """Return an ordinary file below root, rejecting every child symlink."""
    if not isinstance(raw, str) or not raw:
        raise ValueError("missing path")
    # The configured root itself is the trusted boundary and may be a mount
    # symlink.  Every component supplied beneath it, however, is evidence and
    # must be a real directory/file rather than a link to mutable elsewhere.
    lexical_root = root.absolute()
    path = Path(raw)
    lexical_path = path.absolute() if path.is_absolute() else lexical_root / path
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError("path outside required root") from exc
    cursor = lexical_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("symlink evidence path")
    if not lexical_path.is_file() or not _inside(root, lexical_path):
        raise ValueError("path missing or escapes required root")
    return lexical_path.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_json(path: Path, label: str) -> dict:
    """Read a non-symlink JSON object from the explicitly bound workspace."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} missing or symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _nvasr_candidate_immutable_projection(row: dict) -> dict:
    """Independently project the exact producer-owned candidate fields."""
    projection = {
        "schema": NVASR_IMMUTABLE_PROJECTION_SCHEMA,
        "candidate_schema": {
            "nvasr_candidate_schema_version": row.get(
                "nvasr_candidate_schema_version"),
            "provenance_schema": row.get("provenance_schema"),
        },
        "candidate_identity": {
            "candidate_id": row.get("candidate_id"),
            "candidate_kind": row.get("candidate_kind"),
            "word": row.get("word"),
            "candidate_surface": row.get("candidate_surface"),
            "candidate_source": row.get("candidate_source"),
            "candidate_token_id": row.get("candidate_token_id"),
            "candidate_token_ids": row.get("candidate_token_ids"),
            "ctc_lexical_ordinal": row.get("ctc_lexical_ordinal"),
        },
        "frame_coordinates": {
            "raw_start_frame": row.get("raw_start_frame"),
            "raw_end_frame": row.get("raw_end_frame"),
            "raw_frame_count": row.get("raw_frame_count"),
            "raw_start_s": row.get("raw_start_s"),
            "raw_end_s": row.get("raw_end_s"),
            "raw_span": row.get("raw_span"),
            "speech_start_frame": row.get("speech_start_frame"),
            "speech_end_frame": row.get("speech_end_frame"),
            "speech_frame_count": row.get("speech_frame_count"),
            "speech_start_s": row.get("speech_start_s"),
            "speech_end_s": row.get("speech_end_s"),
            "speech_span": row.get("speech_span"),
            "query_frames": row.get("query_frames"),
            "frame_ms": row.get("frame_ms"),
        },
        "ctc_spike_anchor": row.get("ctc_spike_anchor"),
        "mapping": {
            "mapping_basis": row.get("mapping_basis"),
            "mapping_axis": row.get("mapping_axis"),
            # Final canonical coordinates and immutable raw coordinates are
            # distinct producer-owned contracts after the final rebase.
            "mapping_key": row.get("mapping_key"),
            "raw_timeline_mapping_key": row.get(
                "raw_timeline_mapping_key"),
            "ordered_semantic_neighbors": row.get(
                "ordered_semantic_neighbors"),
            "mapping_outcome": row.get("mapping_outcome"),
        },
        "raw_timeline": {
            "raw_timeline_neighbors_schema": row.get(
                "raw_timeline_neighbors_schema"),
            "raw_timeline_index": row.get("raw_timeline_index"),
            "raw_timeline_event_count": row.get(
                "raw_timeline_event_count"),
            "raw_timeline_neighbors": row.get("raw_timeline_neighbors"),
            "raw_timeline_evidence_sha256": row.get(
                "raw_timeline_evidence_sha256"),
        },
        "forced_correspondence": {
            "forced_span": row.get("forced_span"),
            "mapping_selection": row.get("mapping_selection"),
            "mapping_forced_speech_overlap_s": row.get(
                "mapping_forced_speech_overlap_s"),
            "mapping_forced_ctc_anchor_overlap_s": row.get(
                "mapping_forced_ctc_anchor_overlap_s"),
            "nvv_deduplication": row.get("nvv_deduplication"),
        },
        "producer_locator": row.get("ctc_raw_token_row"),
    }
    return deepcopy(projection)


def _nvasr_round_half_up(value: float) -> float:
    return float(Decimal(str(value)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _nvasr_expected_anchor(raw_start: int, raw_end: int) -> dict:
    half_frame_s = NVASR_FRAME_MS / 2000.0
    return {
        "schema": NVASR_SPIKE_ANCHOR_SCHEMA,
        "raw_start_frame": raw_start,
        "raw_end_frame": raw_end,
        "start": _nvasr_round_half_up(
            (raw_start - NVASR_QUERY_FRAMES)
            * NVASR_FRAME_MS / 1000.0 - half_frame_s),
        "end": _nvasr_round_half_up(
            (raw_end - NVASR_QUERY_FRAMES)
            * NVASR_FRAME_MS / 1000.0 - half_frame_s),
        "ordered_source_frame_ids": list(range(raw_start, raw_end)),
        "raw_frame_count": raw_end - raw_start,
        "query_frames": NVASR_QUERY_FRAMES,
        "frame_ms": NVASR_FRAME_MS,
        "speech_start_frame": raw_start - NVASR_QUERY_FRAMES,
        "speech_end_frame": raw_end - NVASR_QUERY_FRAMES,
        "coordinate_system": NVASR_ANCHOR_COORDINATE_SYSTEM,
        "quantization": NVASR_ANCHOR_QUANTIZATION,
    }


def _nvasr_raw_evidence_digest(row: dict) -> str:
    material = {
        "schema": row.get("raw_timeline_neighbors_schema"),
        "candidate_id": row.get("candidate_id"),
        "candidate_surface": row.get("candidate_surface"),
        "candidate_source": row.get("candidate_source"),
        "candidate_token_id": row.get("candidate_token_id"),
        "candidate_token_ids": row.get("candidate_token_ids"),
        "raw_start_frame": row.get("raw_start_frame"),
        "raw_end_frame": row.get("raw_end_frame"),
        "query_frames": row.get("query_frames"),
        "frame_ms": row.get("frame_ms"),
        "raw_timeline_index": row.get("raw_timeline_index"),
        "raw_timeline_event_count": row.get("raw_timeline_event_count"),
        "mapping_key": row.get("raw_timeline_mapping_key",
                                row.get("mapping_key")),
        "raw_timeline_neighbors": row.get("raw_timeline_neighbors"),
    }
    return stable_json_digest(material)


def _nvasr_raw_event_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
            "surface", "source", "token_id", "ordered_source_frame_ids"}:
        return False
    frames = value.get("ordered_source_frame_ids")
    token_id = value.get("token_id")
    return (
        isinstance(value.get("surface"), str)
        and value.get("source") in {"ctc", "blank_run"}
        and isinstance(token_id, int) and not isinstance(token_id, bool)
        and token_id >= 0 and isinstance(frames, list) and bool(frames)
        and all(isinstance(frame, int) and not isinstance(frame, bool)
                and frame >= 0 for frame in frames)
        and frames == list(range(frames[0], frames[-1] + 1))
    )


def _nvasr_authority_candidate_reasons(row: dict) -> list[str]:
    """Validate sealed authority without importing postprocess decisions."""
    reasons: list[str] = []
    if row.get("nvasr_candidate_schema_version") != 3:
        reasons.append("nvasr_candidate_schema_version_3_required")
    if row.get("provenance_schema") != NVASR_CANDIDATE_PROVENANCE_SCHEMA:
        reasons.append("nvasr_candidate_provenance_schema_mismatch")
    if row.get("candidate_kind") != "nvv":
        reasons.append("nvasr_candidate_kind_mismatch")
    if (row.get("mapping_basis") != NVASR_MAPPING_BASIS
            or row.get("mapping_axis") != NVASR_MAPPING_AXIS
            or row.get("mapping_outcome") != "unique"):
        reasons.append("nvasr_candidate_mapping_contract_invalid")
    mapping_key = row.get("mapping_key")
    if (not isinstance(mapping_key, dict)
            or set(mapping_key) != {
                "left_lexical_ordinal", "right_lexical_ordinal"}
            or any(value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
                or value < 0) for value in mapping_key.values())):
        reasons.append("nvasr_candidate_mapping_key_invalid")
        mapping_key = None
    if (row.get("nvasr_candidate_schema_version") == 3
            and "raw_timeline_mapping_key" not in row):
        reasons.append("raw_timeline_mapping_key_required")
    raw_mapping_key = row.get("raw_timeline_mapping_key")
    if (not isinstance(raw_mapping_key, dict)
            or set(raw_mapping_key) != {
                "left_lexical_ordinal", "right_lexical_ordinal"}
            or any(value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
                or value < 0) for value in raw_mapping_key.values())):
        reasons.append("raw_timeline_mapping_key_invalid")

    neighbors = row.get("ordered_semantic_neighbors")
    expected_sides = (["left"] if isinstance(mapping_key, dict)
                      and mapping_key.get("left_lexical_ordinal") is not None
                      else []) + (["right"] if isinstance(mapping_key, dict)
                                  and mapping_key.get(
                                      "right_lexical_ordinal") is not None
                                  else [])
    expected_neighbor_order = [
        (side, mapping_key.get(f"{side}_lexical_ordinal"))
        for side in ("left", "right")
        if isinstance(mapping_key, dict)
        and mapping_key.get(f"{side}_lexical_ordinal") is not None
    ]
    if (not isinstance(neighbors, list)
            or [item.get("side") for item in neighbors
                if isinstance(item, dict)] != expected_sides
            or [(item.get("side"), item.get("lexical_ordinal"))
                for item in neighbors if isinstance(item, dict)]
            != expected_neighbor_order
            or any(not isinstance(item, dict)
                   or set(item) != {"side", "lexical_ordinal",
                                    "occurrence_id", "surface",
                                    "surface_occurrence"}
                   or not isinstance(item.get("lexical_ordinal"), int)
                   or isinstance(item.get("lexical_ordinal"), bool)
                   or item.get("occurrence_id") !=
                   f"nvasr-lexical-{item.get('lexical_ordinal'):04d}"
                   or not isinstance(item.get("surface"), str)
                   or not isinstance(item.get("surface_occurrence"), int)
                   or isinstance(item.get("surface_occurrence"), bool)
                   for item in (neighbors if isinstance(neighbors, list)
                                else []))):
        reasons.append("nvasr_candidate_semantic_neighbors_invalid")

    raw_start = row.get("raw_start_frame")
    raw_end = row.get("raw_end_frame")
    speech_start = row.get("speech_start_frame")
    speech_end = row.get("speech_end_frame")
    frame_position_valid = (
        isinstance(raw_start, int) and not isinstance(raw_start, bool)
        and isinstance(raw_end, int) and not isinstance(raw_end, bool)
        and isinstance(speech_start, int) and not isinstance(speech_start, bool)
        and isinstance(speech_end, int) and not isinstance(speech_end, bool)
        and raw_start >= 0 and raw_end > raw_start)
    if frame_position_valid:
        frame_count = raw_end - raw_start
        if (row.get("raw_frame_count") != frame_count
                or row.get("speech_frame_count") != frame_count
                or speech_start != raw_start - NVASR_QUERY_FRAMES
                or speech_end != raw_end - NVASR_QUERY_FRAMES):
            reasons.append("nvasr_candidate_frame_count_or_offset_invalid")
        expected_raw = [raw_start * NVASR_FRAME_MS / 1000.0,
                        raw_end * NVASR_FRAME_MS / 1000.0]
        expected_speech = [speech_start * NVASR_FRAME_MS / 1000.0,
                           speech_end * NVASR_FRAME_MS / 1000.0]
        for values, expected, label in (
                ([row.get("raw_start_s"), row.get("raw_end_s")],
                 expected_raw, "raw_coordinates"),
                (row.get("raw_span"), expected_raw, "raw_span"),
                ([row.get("speech_start_s"), row.get("speech_end_s")],
                 expected_speech, "speech_coordinates"),
                (row.get("speech_span"), expected_speech, "speech_span")):
            if (not isinstance(values, (list, tuple)) or len(values) != 2
                    or any(not isinstance(value, (int, float))
                           or isinstance(value, bool)
                           or not math.isclose(float(value), expected[index],
                                               abs_tol=1e-9)
                           for index, value in enumerate(values))):
                reasons.append(
                    f"nvasr_candidate_{label}_binding_invalid")
    else:
        reasons.append("nvasr_candidate_frame_coordinates_invalid")
    token_id = row.get("candidate_token_id")
    if (not isinstance(row.get("candidate_id"), str)
            or not row.get("candidate_id")
            or not isinstance(row.get("candidate_surface"), str)
            or not row.get("candidate_surface")
            or row.get("candidate_source") not in {"ctc", "blank_run"}
            or not isinstance(token_id, int) or isinstance(token_id, bool)
            or token_id < 0 or not frame_position_valid
            or row.get("candidate_token_ids") != [token_id] * (
                raw_end - raw_start if frame_position_valid else 0)):
        reasons.append("raw_timeline_candidate_identity_invalid")
    if (not frame_position_valid
            or row.get("query_frames") != NVASR_QUERY_FRAMES
            or row.get("frame_ms") != NVASR_FRAME_MS
            or row.get("ctc_spike_anchor") != _nvasr_expected_anchor(
                raw_start if frame_position_valid else 0,
                raw_end if frame_position_valid else 1)):
        reasons.append("ctc_spike_anchor_binding_invalid")
    else:
        count = raw_end - raw_start
        if not math.isclose(
                float(row["ctc_spike_anchor"]["end"])
                - float(row["ctc_spike_anchor"]["start"]),
                count * NVASR_FRAME_MS / 1000.0, abs_tol=1e-9):
            reasons.append("ctc_spike_anchor_duration_invalid")

    index = row.get("raw_timeline_index")
    count = row.get("raw_timeline_event_count")
    raw_neighbors = row.get("raw_timeline_neighbors")
    raw_position_valid = (
        isinstance(index, int) and not isinstance(index, bool)
        and isinstance(count, int) and not isinstance(count, bool)
        and count > 0 and 0 <= index < count)
    if (row.get("raw_timeline_neighbors_schema") !=
            NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA
            or not raw_position_valid
            or not isinstance(raw_neighbors, dict)
            or set(raw_neighbors) != {"left", "right"}):
        reasons.append("raw_timeline_contract_invalid")
    else:
        for side, required in (
                ("left", index > 0), ("right", index + 1 < count)):
            value = raw_neighbors[side]
            if (value is None) != (not required) or (
                    value is not None and not _nvasr_raw_event_valid(value)):
                reasons.append("raw_timeline_contract_invalid")
        if (frame_position_valid
                and _nvasr_raw_event_valid(raw_neighbors["left"])
                and raw_neighbors["left"]["ordered_source_frame_ids"][-1]
                >= raw_start):
            reasons.append("raw_timeline_contract_invalid")
        if (frame_position_valid
                and _nvasr_raw_event_valid(raw_neighbors["right"])
                and raw_neighbors["right"]["ordered_source_frame_ids"][0]
                < raw_end):
            reasons.append("raw_timeline_contract_invalid")
    if row.get("raw_timeline_evidence_sha256") != \
            _nvasr_raw_evidence_digest(row):
        reasons.append("raw_timeline_evidence_digest_mismatch")
    forced = row.get("forced_span")
    if (not isinstance(forced, list) or len(forced) != 2
            or any(not isinstance(value, (int, float))
                   or isinstance(value, bool) or not math.isfinite(float(value))
                   for value in forced)
            or float(forced[0]) < 0 or float(forced[1]) <= float(forced[0])):
        reasons.append("nvasr_candidate_forced_correspondence_invalid")
    return list(dict.fromkeys(reasons))


def _nvasr_locator_reasons(
        row: dict, stem: str, sidecar: str, ordinal: int) -> list[str]:
    locator = row.get("ctc_raw_token_row")
    if not isinstance(locator, dict) or set(locator) != {
            "schema", "stem", "sidecar", "row_ordinal"}:
        return [f"ctc_raw_token_row_locator_malformed:{ordinal}"]
    reasons = []
    if locator.get("schema") != CTC_RAW_TOKEN_ROW_SCHEMA:
        reasons.append(f"ctc_raw_token_row_schema_mismatch:{ordinal}")
    if locator.get("stem") != stem:
        reasons.append(f"ctc_raw_token_row_stem_mismatch:{ordinal}")
    if locator.get("sidecar") != sidecar:
        reasons.append(f"ctc_raw_token_row_sidecar_mismatch:{ordinal}")
    if locator.get("row_ordinal") != ordinal:
        reasons.append(f"ctc_raw_token_row_ordinal_mismatch:{ordinal}")
    return reasons


def _nvasr_read_jsonl(path: Path, label: str) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} missing or symlink")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{label} empty or contains blank rows")
    rows = [json.loads(line) for line in lines]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} contains non-object rows")
    return rows


_NVASR_V3_SIDECAR_MARKER_KEYS = frozenset({
    "provenance_schema", "candidate_id", "candidate_kind",
    "candidate_surface", "candidate_source", "candidate_token_id",
    "candidate_token_ids", "raw_span", "speech_span", "raw_start_frame",
    "raw_end_frame", "speech_start_frame", "speech_end_frame",
    "raw_start_s", "raw_end_s", "speech_start_s", "speech_end_s",
    "raw_frame_count", "speech_frame_count", "query_frames", "frame_ms",
    "mapping_basis", "mapping_axis", "mapping_key",
    "raw_timeline_mapping_key", "mapping_outcome",
    "forced_span", "nvasr_candidate_schema_version",
    "ordered_semantic_neighbors", "raw_timeline_neighbors",
    "raw_timeline_neighbors_schema", "raw_timeline_index",
    "raw_timeline_event_count", "raw_timeline_evidence_sha256",
    "ctc_spike_anchor", "ctc_raw_token_row", "ctc_lexical_ordinal",
    "mapping_selection", "mapping_forced_speech_overlap_s",
    "mapping_forced_ctc_anchor_overlap_s", "nvv_deduplication",
})


def _nvasr_v3_sidecar_marker(row: dict) -> bool:
    """Recognize current NVASR evidence before lifecycle discovery.

    This is intentionally presence-based for v3-only fields.  A partially
    deleted anchor or raw-row locator is still evidence that must not be
    downgraded to the unbound legacy mode.
    """
    return (
        row.get("schema") in _NVASR_MODERN_SCHEMA_MARKERS
        or
        row.get("candidate_kind") == "nvv"
        or row.get("nvasr_candidate_schema_version") ==
        NVASR_CANDIDATE_SCHEMA_VERSION
        or row.get("provenance_schema") == NVASR_CANDIDATE_PROVENANCE_SCHEMA
        or bool(_NVASR_V3_SIDECAR_MARKER_KEYS.intersection(row))
    )


def _ctc_v3_sidecar_precheck(
        ctc_dir: Path, expected: set[str] | None) -> list[str]:
    """Reject unsafe/modern sidecars that lack an explicit lifecycle.

    A missing sidecar remains compatible with old fixtures; an existing but
    unsafe or malformed sidecar never becomes evidence of that compatibility.
    """
    stems = sorted(expected if expected is not None else {
        path.stem for path in ctc_dir.glob("*.lab")
    })
    reasons: list[str] = []
    for stem in stems:
        sidecar = ctc_dir / f"{stem}_tokens.jsonl"
        if not sidecar.exists() and not sidecar.is_symlink():
            continue
        if sidecar.is_symlink() or not sidecar.is_file():
            reasons.append(f"ctc_v3_token_sidecar_unreadable:{stem}")
            continue
        try:
            rows = _nvasr_read_jsonl(sidecar, "CTC token sidecar")
        except (OSError, UnicodeError, TypeError, ValueError,
                json.JSONDecodeError):
            reasons.append(f"ctc_v3_token_sidecar_unreadable:{stem}")
            continue
        if any(_nvasr_v3_sidecar_marker(row) for row in rows):
            reasons.append(f"ctc_lifecycle_missing_for_v3:{stem}")
    return sorted(set(reasons))


def _nvasr_authority_summary(
        raw_manifest: dict, work_receipt: dict, raw_dir: Path,
        work_dir: Path, stem: str) -> dict:
    """Rebuild the public authority summary from independently read bytes."""
    reasons: list[str] = []
    sidecar = f"{stem}_tokens.jsonl"
    if (not isinstance(stem, str) or not stem or Path(stem).name != stem
            or Path(sidecar).name != sidecar):
        reasons.append("nvasr_token_sidecar_stem_invalid")
    raw_entries = [item for item in raw_manifest.get("files", [])
                   if isinstance(item, dict) and item.get("stem") == stem
                   and item.get("suffix") == "_tokens.jsonl"]
    work_entries = [item for item in work_receipt.get("files", [])
                    if isinstance(item, dict) and item.get("stem") == stem
                    and item.get("suffix") == "_tokens.jsonl"]
    if len(raw_entries) != 1 or raw_entries[0].get("name") != sidecar:
        reasons.append("raw_manifest_token_sidecar_resolution_invalid")
    if len(work_entries) != 1 or work_entries[0].get("name") != sidecar:
        reasons.append("work_receipt_token_sidecar_resolution_invalid")
    raw_rows: list[dict] = []
    work_rows: list[dict] = []
    if not reasons:
        try:
            raw_rows = _nvasr_read_jsonl(
                raw_dir / sidecar, "manifest-sealed raw token sidecar")
            work_rows = _nvasr_read_jsonl(
                work_dir / sidecar, "receipt-bound work token sidecar")
        except (OSError, UnicodeError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            reasons.append(f"nvasr_token_sidecar_unreadable:{exc}")

    for ordinal, item in enumerate(raw_rows):
        reasons.extend(_nvasr_locator_reasons(
            item, stem, sidecar, ordinal))
    for ordinal, item in enumerate(work_rows):
        reasons.extend(_nvasr_locator_reasons(
            item, stem, sidecar, ordinal))
    raw_locators = [item.get("ctc_raw_token_row") for item in raw_rows]
    work_locators = [item.get("ctc_raw_token_row") for item in work_rows]
    if len(raw_rows) != len(work_rows) or raw_locators != work_locators:
        reasons.append("ctc_raw_token_row_sequence_mismatch")
    if len({stable_json_digest(value) for value in raw_locators}) != len(
            raw_locators):
        reasons.append("ctc_raw_token_row_locator_duplicate")

    raw_candidates = [item for item in raw_rows
                      if item.get("candidate_kind") == "nvv"]
    work_candidates = [item for item in work_rows
                       if item.get("candidate_kind") == "nvv"]
    if (any(is_nvv_token(str(item.get("word", "")).strip())
            and item.get("candidate_kind") != "nvv" for item in raw_rows)
            or any(is_nvv_token(str(item.get("word", "")).strip())
                   and item.get("candidate_kind") != "nvv"
                   for item in work_rows)):
        reasons.append("legacy_or_unprovenanced_nvasr_row")
    raw_ids = [item.get("candidate_id") for item in raw_candidates]
    work_ids = [item.get("candidate_id") for item in work_candidates]
    if (not all(isinstance(value, str) and value for value in raw_ids)
            or raw_ids != sorted(raw_ids)
            or len(raw_ids) != len(set(raw_ids))
            or work_ids != raw_ids):
        reasons.append("nvasr_candidate_identity_sequence_mismatch")

    projections = []
    for candidate_index, raw_row in enumerate(raw_candidates):
        locator = raw_row.get("ctc_raw_token_row")
        ordinal = (locator.get("row_ordinal")
                   if isinstance(locator, dict) else candidate_index)
        for reason in _nvasr_authority_candidate_reasons(raw_row):
            reasons.append(f"sealed_raw_candidate:{ordinal}:{reason}")
        projection = _nvasr_candidate_immutable_projection(raw_row)
        projections.append(projection)
        if candidate_index >= len(work_candidates):
            continue
        work_row = work_candidates[candidate_index]
        for reason in _nvasr_authority_candidate_reasons(work_row):
            reasons.append(f"work_candidate:{ordinal}:{reason}")
        if _nvasr_candidate_immutable_projection(work_row) != projection:
            reasons.append(
                f"sealed_raw_candidate_projection_mismatch:{ordinal}:"
                f"{raw_row.get('candidate_id')}")
    if len(raw_candidates) != len(work_candidates):
        reasons.append("nvasr_candidate_count_mismatch")

    projection_digest = stable_json_digest({
        "schema": NVASR_IMMUTABLE_PROJECTION_SCHEMA,
        "stem": stem,
        "candidates": projections,
    })
    return {
        "schema": NVASR_PRODUCER_AUTHORITY_SCHEMA,
        "status": "verified" if not reasons else "rejected",
        "raw_manifest_identity": raw_manifest.get("identity"),
        "raw_tokens_sha256": (raw_entries[0].get("sha256")
                              if len(raw_entries) == 1 else None),
        "work_receipt_identity": work_receipt.get("identity"),
        "candidate_count": len(raw_candidates),
        "ordered_projection_sha256": projection_digest,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _ctc_lifecycle_reasons(
        args: argparse.Namespace,
        expected: set[str] | None = None) -> tuple[list[str], dict | None]:
    """Independently bind immutable CTC raw bytes to mutable work bytes.

    The audit deliberately accepts a completely unbound legacy fixture.  The
    moment either lifecycle marker is present, however, all bindings are
    mandatory and a changed raw artifact invalidates the complete candidate
    set.  Equal content between raw and work is expected; equal inode identity
    is not, because work must be a physical copy.
    """
    ctc_dir = Path(getattr(args, "ctc_dir", ""))
    raw_arg = getattr(args, "ctc_raw_manifest", None)
    work_arg = getattr(args, "ctc_work_receipt", None)
    work_path = Path(work_arg) if work_arg is not None else ctc_dir / CTC_WORK_RECEIPT_NAME
    default_raw_path = ctc_dir / CTC_RAW_MANIFEST_NAME
    raw_path = (Path(raw_arg) if raw_arg is not None
                else (default_raw_path
                      if (default_raw_path.exists()
                          or default_raw_path.is_symlink()) else None))
    marker_present = (raw_path is not None or work_arg is not None
                      or work_path.exists()
                      or work_path.is_symlink()
                      or (raw_path is not None and
                          (raw_path.exists() or raw_path.is_symlink())))
    precheck_reasons = _ctc_v3_sidecar_precheck(ctc_dir, expected)
    if precheck_reasons and not marker_present:
        return precheck_reasons, None
    if not marker_present:
        return [], None

    errors: list[str] = []
    raw_manifest: dict | None = None
    work_receipt: dict | None = None
    if work_path.is_file() and not work_path.is_symlink():
        try:
            work_receipt = _regular_json(work_path, "CTC work receipt")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"ctc_work_receipt_unreadable:{exc}")
    if raw_path is None and isinstance(work_receipt, dict):
        binding = work_receipt.get("raw_manifest")
        if isinstance(binding, dict) and isinstance(binding.get("path"), str):
            raw_path = Path(binding["path"])
    if raw_path is None:
        errors.append("ctc_raw_manifest_binding_missing")
    elif raw_path.is_file() and not raw_path.is_symlink():
        try:
            raw_manifest = _regular_json(raw_path, "CTC raw manifest")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"ctc_raw_manifest_unreadable:{exc}")
    else:
        errors.append("ctc_raw_manifest_missing_or_symlink")
    if work_path.is_symlink() or not work_path.is_file():
        errors.append("ctc_work_receipt_missing_or_symlink")

    raw_dir = raw_path.parent if raw_path is not None else None
    work_dir = work_path.parent
    if raw_dir is not None and raw_dir.resolve() == work_dir.resolve():
        errors.append("ctc_raw_work_alias")
    if raw_manifest is not None and raw_dir is not None:
        errors.extend(validate_ctc_raw_manifest(raw_dir, raw_manifest))
        if raw_manifest.get("raw_root") != str(raw_dir.resolve()):
            errors.append("ctc_raw_manifest_root_binding_mismatch")
        producer = raw_manifest.get("producer_receipt")
        if isinstance(producer, dict):
            expected_producer = raw_dir / str(producer.get("name", ""))
            if producer.get("path") != str(expected_producer.resolve()):
                errors.append("ctc_producer_receipt_path_binding_mismatch")
    if work_receipt is not None and raw_path is not None:
        errors.extend(validate_ctc_work_receipt(work_dir, raw_path, work_receipt))
        binding = work_receipt.get("raw_manifest")
        if not isinstance(binding, dict):
            errors.append("ctc_work_raw_manifest_binding_missing")
        else:
            if binding.get("path") != str(raw_path.resolve()):
                errors.append("ctc_work_raw_manifest_path_mismatch")
            if raw_path.is_file() and not raw_path.is_symlink():
                if binding.get("sha256") != _sha256(raw_path):
                    errors.append("ctc_work_raw_manifest_digest_mismatch")
                if raw_manifest is not None and binding.get("identity") != raw_manifest.get("identity"):
                    errors.append("ctc_work_raw_manifest_identity_mismatch")
        if work_receipt.get("work_root") != str(ctc_dir.resolve()):
            errors.append("ctc_work_receipt_root_binding_mismatch")
        if not isinstance(work_receipt.get("transform_ledger"), list):
            errors.append("ctc_work_receipt_lineage_missing")

    # Validate that raw and work are independent physical trees.  Hashes may
    # intentionally match after a copy; inode identity may never match.
    if raw_manifest is not None and raw_dir is not None and raw_dir.resolve() != work_dir.resolve():
        for row in raw_manifest.get("files", []):
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                continue
            raw_file = raw_dir / row["name"]
            work_file = work_dir / row["name"]
            if raw_file.is_symlink() or work_file.is_symlink():
                errors.append(f"ctc_raw_work_symlink:{row['name']}")
                continue
            if raw_file.is_file() and work_file.is_file():
                try:
                    if os.path.samestat(raw_file.stat(), work_file.stat()):
                        errors.append(f"ctc_raw_work_alias:{row['name']}")
                except OSError:
                    errors.append(f"ctc_raw_work_stat_unreadable:{row['name']}")

    if any("hash mismatch" in item.lower() or "digest mismatch" in item.lower()
           or "identity mismatch" in item.lower() for item in errors):
        errors.append("ctc_raw_digest_mismatch")
    lifecycle = None
    if raw_manifest is not None and work_receipt is not None and raw_path is not None:
        ledger = work_receipt.get("transform_ledger")
        authority_by_stem = {}
        manifest_stems = raw_manifest.get("stems")
        authority_stems = (sorted(expected) if expected is not None
                           else (list(manifest_stems)
                                 if isinstance(manifest_stems, list) else []))
        for stem in authority_stems:
            try:
                summary = _nvasr_authority_summary(
                    raw_manifest, work_receipt, raw_path.parent,
                    work_path.parent, stem)
            except (OSError, UnicodeError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                summary = {
                    "schema": NVASR_PRODUCER_AUTHORITY_SCHEMA,
                    "status": "rejected",
                    "raw_manifest_identity": raw_manifest.get("identity"),
                    "raw_tokens_sha256": None,
                    "work_receipt_identity": work_receipt.get("identity"),
                    "candidate_count": 0,
                    "ordered_projection_sha256": None,
                    "reasons": [f"nvasr_authority_unreadable:{exc}"],
                }
            authority_by_stem[stem] = summary
            if summary.get("status") != "verified":
                summary_reasons = summary.get("reasons")
                if isinstance(summary_reasons, list) and summary_reasons:
                    errors.extend(
                        f"nvasr_producer_authority:{stem}:{reason}"
                        for reason in summary_reasons)
                else:
                    errors.append(
                        f"nvasr_producer_authority:{stem}:rejected")
        lifecycle = {
            "schema": CTC_LIFECYCLE_SCHEMA,
            "raw_manifest": {"path": str(raw_path.resolve()),
                             "sha256": _sha256(raw_path),
                             "identity": raw_manifest.get("identity")},
            "work_receipt": {"path": str(work_path.resolve()),
                              "sha256": _sha256(work_path),
                              "identity": work_receipt.get("identity"),
                              "lineage_entries": len(ledger) if isinstance(ledger, list) else 0},
            "_nvasr_producer_authority": authority_by_stem,
        }
    return sorted(set(errors)), lifecycle


def _processed_geometry_digest(tg) -> str:
    words = tier_by_name(tg, "words")
    # Match postprocess_textgrids.write_textgrid's six-decimal serializer.
    rows = [{"xmin": round(float(interval.xmin), 6),
             "xmax": round(float(interval.xmax), 6),
             "text": interval.text}
            for interval in (words.intervals if words is not None else [])]
    return _evidence_digest(rows)


def _postprocess_v3_claim(row: dict) -> bool:
    """Return whether a report asserts current CTC/NVASR v3 evidence."""
    # Presence itself is the modern-contract claim.  A deleted, empty, or
    # malformed value must remain fail-closed instead of becoming legacy.
    return bool(_MODERN_REPORT_CONTRACT_KEYS.intersection(row))


def _nvasr_report_subcontract_reasons(
        row: dict, *, expected_candidate_count: int | None = None
        ) -> list[str]:
    """Reject malformed, negative, or incomplete NVASR subreports.

    A producer-authority summary with candidates is the independent source of
    truth for coverage.  Merely publishing an empty ``verified`` or
    ``not_applicable`` report must never erase those candidates at audit time.
    """
    reasons: list[str] = []
    if expected_candidate_count is None:
        authority = row.get("nvasr_producer_authority")
        if isinstance(authority, dict):
            expected_candidate_count = authority.get("candidate_count")
    expected_count_valid = (
        isinstance(expected_candidate_count, int)
        and not isinstance(expected_candidate_count, bool)
        and expected_candidate_count >= 0)
    candidate_bearing = bool(
        expected_count_valid and expected_candidate_count > 0)
    for key in ("nvasr_owner_selection", "nvasr_frame_support",
                "nvasr_candidate_provenance"):
        if key not in row:
            if candidate_bearing:
                reasons.append(f"{key}_missing")
            continue
        report = row.get(key)
        if not isinstance(report, dict):
            reasons.append(f"{key}_malformed")
            continue
        status = report.get("status")
        allowed_statuses = ({"verified"} if candidate_bearing
                            else {"verified", "not_applicable"})
        if status not in allowed_statuses:
            reasons.append(f"{key}_not_verified")
        report_reasons = report.get("reasons")
        if report_reasons not in (None, []):
            reasons.append(f"{key}_has_reasons")
        candidates = report.get("candidates")
        if (expected_count_valid
                and (not isinstance(candidates, list)
                     or len(candidates) != expected_candidate_count)):
            reasons.append(f"{key}_candidate_count_mismatch")
        if (key == "nvasr_candidate_provenance"
                and expected_count_valid
                and report.get("candidate_count") != expected_candidate_count):
            reasons.append(f"{key}_declared_candidate_count_mismatch")
        if status == "not_applicable":
            candidate_count = report.get("candidate_count")
            if candidates not in (None, []) or candidate_count not in (None, 0):
                reasons.append(f"{key}_not_applicable_with_candidates")
    return reasons


def _postprocess_contract_reasons(
        row: dict | None, tg, lifecycle: dict | None) -> list[str]:
    """Check report identity and frozen final geometry against disk."""
    if not isinstance(row, dict):
        return ["postprocess_report_missing"] if lifecycle is not None else []
    v3_claim = _postprocess_v3_claim(row)
    fields_present = any(key in row for key in (
        "ctc_lifecycle", "processed_geometry", "processed_geometry_digest",
        "processed_operation_ledger", "nvasr_producer_authority",
        "nvasr_candidate_provenance", "nvasr_owner_selection",
        "nvasr_frame_support"))
    if lifecycle is None and not fields_present and not v3_claim:
        return []
    reasons: list[str] = []
    if lifecycle is None and v3_claim:
        reasons.append("postprocess_v3_claim_without_ctc_lifecycle")
    reported_lifecycle = row.get("ctc_lifecycle")
    if lifecycle is not None:
        if not isinstance(reported_lifecycle, dict):
            reasons.append("postprocess_raw_work_identity_missing")
        else:
            for section in ("raw_manifest", "work_receipt"):
                expected = lifecycle[section]
                actual = reported_lifecycle.get(section)
                if not isinstance(actual, dict):
                    reasons.append(f"postprocess_{section}_identity_missing")
                    continue
                for key in ("path", "sha256", "identity"):
                    if actual.get(key) != expected[key]:
                        reasons.append(f"postprocess_{section}_{key}_mismatch")
            if reported_lifecycle.get("stem") != row.get("stem"):
                reasons.append("postprocess_ctc_lifecycle_stem_mismatch")

        authority_by_stem = lifecycle.get("_nvasr_producer_authority")
        expected_authority = (
            authority_by_stem.get(row.get("stem"))
            if isinstance(authority_by_stem, dict) else None)
        reported_authority = row.get("nvasr_producer_authority")
        if not isinstance(expected_authority, dict):
            reasons.append("postprocess_nvasr_producer_authority_unavailable")
        elif not isinstance(reported_authority, dict):
            reasons.append("postprocess_nvasr_producer_authority_missing")
        elif reported_authority != expected_authority:
            reasons.append("postprocess_nvasr_producer_authority_mismatch")
        expected_candidate_count = (
            expected_authority.get("candidate_count")
            if isinstance(expected_authority, dict) else None)
        reasons.extend(
            f"postprocess_{reason}"
            for reason in _nvasr_report_subcontract_reasons(
                row, expected_candidate_count=expected_candidate_count))

    geometry = row.get("processed_geometry")
    digest = row.get("processed_geometry_digest")
    ledger = row.get("processed_operation_ledger")
    if not isinstance(geometry, dict):
        reasons.append("processed_geometry_missing")
        return sorted(set(reasons))
    if geometry.get("schema") != PROCESSED_GEOMETRY_SCHEMA:
        reasons.append("processed_geometry_schema_mismatch")
    if geometry.get("frozen") is not True:
        reasons.append("processed_geometry_not_frozen")
    if not isinstance(digest, str) or not digest:
        reasons.append("processed_geometry_digest_missing")
    if not isinstance(ledger, list) or not ledger:
        reasons.append("processed_operation_ledger_missing")
    if geometry.get("digest") != digest or geometry.get("ledger") != ledger:
        reasons.append("processed_geometry_report_binding_mismatch")
    geometry_contract = row.get("processed_geometry_contract")
    if isinstance(geometry_contract, dict) and geometry_contract.get("status") == "rejected":
        reasons.extend(f"processed_geometry_contract:{item}"
                       for item in geometry_contract.get("reasons", [])
                       if isinstance(item, str))
        if not geometry_contract.get("reasons"):
            reasons.append("processed_geometry_contract:rejected")
    if not any(isinstance(item, dict) and item.get("operation") == "boundary_freeze"
               for item in (ledger if isinstance(ledger, list) else [])):
        reasons.append("processed_geometry_freeze_operation_missing")
    publication = row.get("publication_contract")
    if isinstance(publication, dict):
        if publication.get("status") != "verified":
            reasons.append("publication_contract_not_verified")
        if publication.get("reasons"):
            reasons.append("publication_contract_has_reasons")
    if tg is not None:
        actual_digest = _processed_geometry_digest(tg)
        if digest != actual_digest:
            reasons.append("processed_geometry_digest_mismatch")

        # The publication proof is deliberately based on final ``word_span``
        # (or its explicit ``published_span`` alias).  ``resolved_span`` is
        # historical evidence only and is never accepted as authority.
        publication = row.get("publication_contract")
        details = (publication.get("details", {})
                   if isinstance(publication, dict) else {})
        proofs = details.get("ctc_lexical_evidence_proof", [])
        words = tier_by_name(tg, "words")
        lexical = [iv for iv in (words.intervals if words is not None else [])
                   if iv.text.strip() and not is_silence(iv.text) and not is_punct(iv.text)]
        if not isinstance(proofs, list) or len(proofs) != len(lexical):
            reasons.append("processed_published_span_binding_missing")
        else:
            for index, (proof, interval) in enumerate(zip(proofs, lexical)):
                if not isinstance(proof, dict):
                    reasons.append(f"processed_published_span_invalid:{index}")
                    continue
                span = proof.get("published_span", proof.get("word_span"))
                if "published_span" not in proof and "word_span" not in proof:
                    reasons.append(f"processed_published_span_missing:{index}")
                    continue
                try:
                    if (len(span) != 2 or not _same_number(interval.xmin, span[0])
                            or not _same_number(interval.xmax, span[1])):
                        reasons.append(f"processed_published_span_mismatch:{index}")
                except (TypeError, ValueError):
                    reasons.append(f"processed_published_span_invalid:{index}")
                if "published_span" not in proof and "resolved_span" in proof and "word_span" not in proof:
                    reasons.append(f"resolved_span_not_publication_authority:{index}")
    return sorted(set(reasons))


def _wav_duration(path: Path) -> float:
    return float(_axis_audio_metadata(path)["duration_s"])


def _axis_digest(value: object) -> str:
    return stable_json_digest(value)


def _axis_wav_meta(path: Path) -> dict:
    # Keep the auditor independent from postprocess while sharing the generic
    # immutable audio-header reader used to create the axis receipts.  It
    # preserves stdlib wave semantics for PCM and narrowly supports IEEE-float
    # WAV/WAVEX through libsndfile.
    return _axis_audio_metadata(path)


def _axis_contract_reasons(args: argparse.Namespace,
                           expected: set[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Independently validate MFA and TTS audio axes from explicit receipts."""
    errors: list[str] = []
    stem_reasons: dict[str, list[str]] = {}
    input_path = getattr(args, "mfa_input_axis_receipt", None)
    align_path = getattr(args, "mfa_alignment_axis_receipt", None)
    mfa_root = getattr(args, "mfa_axis_audio_root", None)
    tts_root = getattr(args, "tts_authoritative_audio_root", None)
    if not all(isinstance(value, Path) for value in (input_path, align_path, mfa_root, tts_root)):
        return ["axis_contract_receipts_missing"], stem_reasons
    try:
        for path, label, directory in ((input_path, "mfa_input_axis", False),
                                       (align_path, "mfa_alignment_axis", False),
                                       (mfa_root, "mfa_axis_audio_root", True),
                                       (tts_root, "tts_authoritative_audio_root", True)):
            if (not path.is_absolute() or ".." in path.parts or path.is_symlink()
                    or (not path.is_dir() if directory else not path.is_file())):
                raise ValueError(f"{label} path invalid")
        input_axis = json.loads(input_path.read_text(encoding="utf-8"))
        alignment_axis = json.loads(align_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return [f"axis_contract_receipt_unreadable:{exc}"], stem_reasons
    if input_axis.get("schema") != MFA_INPUT_AXIS_SCHEMA:
        errors.append("mfa_input_axis_schema_mismatch")
    if input_axis.get("source_role") != "mfa_axis_audio":
        errors.append("mfa_input_axis_source_role_mismatch")
    alignment_schema = alignment_axis.get("schema")
    if alignment_schema not in (MFA_ALIGNMENT_AXIS_SCHEMA, MFA_ALIGNMENT_AXIS_V2_SCHEMA):
        errors.append("mfa_alignment_axis_schema_mismatch")
    stems = input_axis.get("stems")
    stems_valid = (isinstance(stems, list)
                   and all(isinstance(stem, str) and stem for stem in stems))
    if (not stems_valid or stems != sorted(set(stems))
            or input_axis.get("stems_digest") != _axis_digest(stems)):
        errors.append("axis_stem_conservation_invalid")
        stems = []
    if Path(str(input_axis.get("axis_root", ""))).resolve() != mfa_root.resolve():
        errors.append("mfa_axis_audio_root_binding_mismatch")
    declared_tts_root = input_axis.get("tts_authoritative_audio_root")
    if declared_tts_root is not None and Path(str(declared_tts_root)).resolve() != tts_root.resolve():
        errors.append("tts_authoritative_audio_root_binding_mismatch")
    if Path(str(alignment_axis.get("alignment_root", ""))).resolve() != args.aligned_dir.resolve():
        errors.append("mfa_alignment_root_binding_mismatch")
    if (alignment_axis.get("input_axis_schema") != MFA_INPUT_AXIS_SCHEMA
            or alignment_axis.get("input_axis_digest") != _axis_digest(input_axis)
            or alignment_axis.get("stems") != stems
            or alignment_axis.get("stems_digest") != _axis_digest(stems)):
        errors.append("axis_receipt_digest_or_stem_mismatch")
    input_rows = input_axis.get("audio")
    align_rows = alignment_axis.get("alignments")
    rows_valid = (isinstance(input_rows, list) and all(isinstance(row, dict) for row in input_rows)
                  and isinstance(align_rows, list) and all(isinstance(row, dict) for row in align_rows))
    if not rows_valid or [row.get("stem") for row in input_rows] != stems:
        errors.append("axis_stem_conservation_invalid")
        return sorted(set(errors)), stem_reasons
    if alignment_axis.get("scale") != 1.0:
        errors.append("mfa_alignment_axis_scale_mismatch")
    if input_axis.get("scale") != 1.0:
        errors.append("mfa_input_axis_scale_mismatch")
    if alignment_schema == MFA_ALIGNMENT_AXIS_SCHEMA:
        if [row.get("stem") for row in align_rows] != stems:
            errors.append("axis_stem_conservation_invalid")
            return sorted(set(errors)), stem_reasons
        aligned_rows = align_rows
        missing_stems: set[str] = set()
    elif alignment_schema == MFA_ALIGNMENT_AXIS_V2_SCHEMA:
        if [row.get("stem") for row in align_rows] != stems:
            errors.append("axis_stem_conservation_invalid")
            return sorted(set(errors)), stem_reasons
        status_by_stem = {row.get("stem"): row.get("status") for row in align_rows}
        if any(status not in {"aligned", "missing_mfa_alignment"}
               for status in status_by_stem.values()):
            errors.append("mfa_alignment_axis_status_invalid")
            return sorted(set(errors)), stem_reasons
        missing_stems = {stem for stem, status in status_by_stem.items()
                         if status == "missing_mfa_alignment"}
        aligned_rows = [row for row in align_rows if row.get("status") == "aligned"]
        expected_counts = {"aligned": len(aligned_rows),
                           "missing_mfa_alignment": len(missing_stems)}
        if alignment_axis.get("status_counts") != expected_counts:
            errors.append("mfa_alignment_axis_status_counts_mismatch")
        actual_grids = {path.stem for path in args.aligned_dir.glob("*.TextGrid")}
        if actual_grids != set(stems) - missing_stems:
            errors.append("mfa_alignment_axis_status_partition_mismatch")
    else:
        return sorted(set(errors)), stem_reasons
    # The immutable MFA input axis covers the full source set.  A v2
    # alignment receipt may explicitly exclude stems for which MFA emitted no
    # TextGrid; those stems are accounted for as exclusions in the pipeline
    # receipt and must not make the axis look corrupt.  The strict candidate
    # set is therefore the aligned partition, while the axis set remains the
    # full source partition.
    expected_axis_stems = (set(stems) if alignment_schema == MFA_ALIGNMENT_AXIS_SCHEMA
                           else set(stems) - missing_stems)
    # Production postprocess may retain explicit missing-MFA placeholders in
    # the eligible/filtered partition, while replay receipts may model those
    # same stems as exclusions.  Both are valid only when the v2 receipt and
    # the missing-alignment ledger conserve the full axis; reject any other
    # subset.
    if set(expected) not in (set(stems), expected_axis_stems):
        errors.append("axis_stem_conservation_invalid")
    input_by_stem = {row["stem"]: row for row in input_rows if isinstance(row, dict) and "stem" in row}
    align_by_stem = {row["stem"]: row for row in aligned_rows if isinstance(row, dict) and "stem" in row}
    if len(input_by_stem) != len(stems) or len(align_by_stem) != len(stems) - len(missing_stems):
        errors.append("axis_stem_conservation_invalid")
        return sorted(set(errors)), stem_reasons

    transforms: dict[str, dict] = {}
    transform_paths = input_axis.get("transform_receipts", [])
    if not isinstance(transform_paths, list) or any(not isinstance(item, str) for item in transform_paths):
        errors.append("audio_transform_receipts_invalid")
    elif transform_paths:
        if len(transform_paths) != len(stems):
            errors.append("audio_transform_receipt_stem_conservation_invalid")
        for raw_path in transform_paths:
            try:
                receipt_path = Path(raw_path)
                if (not receipt_path.is_absolute() or ".." in receipt_path.parts
                        or receipt_path.is_symlink() or not receipt_path.is_file()):
                    raise ValueError("unsafe receipt path")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                input_row = receipt.get("input", {})
                stem = Path(str(input_row.get("path", ""))).stem
                if stem in transforms or stem not in stems:
                    raise ValueError("duplicate or unexpected transform stem")
                transforms[stem] = receipt
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                errors.append("audio_transform_receipt_invalid")
    for stem in stems:
        row = input_by_stem[stem]
        reasons = stem_reasons.setdefault(stem, [])
        try:
            audio = Path(str(row["path"]))
            if (audio.is_symlink() or not audio.is_file() or audio.resolve().parent != mfa_root.resolve()
                    or audio.name != f"{stem}.wav"):
                raise ValueError("mfa audio path binding")
            actual = _axis_wav_meta(audio)
            if (row.get("sha256") != actual["sha256"]
                    or row.get("sample_rate") != actual["sample_rate"]
                    or row.get("frames") != actual["frames"]
                    or abs(float(row.get("duration_s")) - actual["duration_s"]) > EPS):
                raise ValueError("mfa audio metadata/hash")
        except (OSError, ValueError, TypeError, KeyError):
            errors.append(f"mfa_axis_audio_receipt_invalid:{stem}")
            continue
        if stem in missing_stems:
            reasons.append("missing_mfa_alignment")
        else:
            alignment = align_by_stem[stem]
            aligned = args.aligned_dir / f"{stem}.TextGrid"
            try:
                tg = parse_textgrid(aligned)
                if (alignment.get("path") != str(aligned.resolve())
                        or alignment.get("sha256") != _sha256(aligned)
                        or alignment.get("audio_sha256") != row.get("sha256")
                        or abs(float(alignment.get("xmax")) - tg.xmax) > EPS
                        or abs(tg.xmax - actual["duration_s"]) > EPS):
                    reasons.append("mfa_alignment_axis_mismatch")
            except (OSError, ValueError, TypeError, KeyError):
                reasons.append("mfa_alignment_axis_mismatch")
        try:
            # TTS audio can be nested by speaker; the transform receipt binds
            # this stem to its actual source path while MFA keeps a flat axis.
            transform = transforms.get(stem)
            transform_input = (transform.get("input", {})
                               if isinstance(transform, dict) else {})
            bound_tts = Path(str(transform_input.get("path", "")))
            tts = (bound_tts if bound_tts.is_absolute()
                   else tts_root / f"{stem}.wav")
            tts_meta = _axis_wav_meta(tts)
            identity = all(tts_meta[key] == actual[key]
                           for key in ("sha256", "sample_rate", "frames", "channels", "sample_width")) and abs(
                               tts_meta["duration_s"] - actual["duration_s"]) <= EPS
            if transform is not None:
                inp, out = transform.get("input"), transform.get("output")
                valid_transform = (
                    transform.get("schema") == "audio-transform-receipt-v1"
                    and transform.get("scale") == 1.0
                    and all(transform.get(key) == 0.0 for key in
                            ("head_transform_s", "tail_transform_s", "shift_s"))
                    and isinstance(inp, dict) and isinstance(out, dict)
                    and inp.get("path") == str(tts.resolve())
                    and out.get("path") == str(audio.resolve())
                    and all(inp.get(key) == tts_meta[key] for key in tts_meta)
                    and all(out.get(key) == actual[key] for key in actual)
                    and abs(float(inp.get("duration_s")) - float(out.get("duration_s"))) <= EPS)
                if not valid_transform:
                    reasons.append("tts_audio_axis_mismatch")
            elif not identity:
                reasons.append("tts_audio_axis_mismatch")
        except (OSError, ValueError, TypeError):
            reasons.append("tts_audio_axis_mismatch")
    return sorted(set(errors)), {stem: sorted(set(reasons)) for stem, reasons in stem_reasons.items()}


def _semantic_tokens(text: str) -> list[tuple[str, str]]:
    """Return ordered CJK/NVV/punctuation/English tokens, excluding silence."""
    return [(item["kind"], item["surface"].casefold()
             if item["kind"] == "english" else item["surface"])
            for item in project_authority_semantics(text)]


def _semantic_sequence_compatible(reference: list[tuple[str, str]],
                                  observed: list[tuple[str, str]]) -> bool:
    """Allow missing reference punctuation, but reject all other drift."""
    cursor = 0
    for actual in observed:
        while (cursor < len(reference)
               and reference[cursor][0] == "punct"
               and reference[cursor] != actual):
            cursor += 1
        if cursor >= len(reference) or reference[cursor] != actual:
            return False
        cursor += 1
    return all(kind == "punct" for kind, _ in reference[cursor:])


def _tier_text(tier) -> str:
    return " ".join(iv.text for iv in tier.intervals if iv.text).strip()


def _strict_parse(path: Path):
    raw = path.read_text(encoding="utf-8")
    if 'File type = "ooTextFile"' not in raw or 'Object class = "TextGrid"' not in raw:
        raise ValueError("not a long-text TextGrid")
    # A valid five-tier result must declare exactly five numbered items.  This
    # catches inputs that the permissive legacy reader would otherwise accept.
    if len(re.findall(r"(?m)^\s*item \[\d+\]:\s*$", raw)) != 5:
        raise ValueError("declared item count is not exactly five")
    tg = parse_textgrid(path)
    if [tier.name for tier in tg.tiers] != TIER_NAMES:
        raise ValueError("tiers must be exactly raw_text,pinyin,hanzi,words,pinyin_phones")
    return tg


def _numeric_reasons(tg, duration: float) -> list[str]:
    reasons: list[str] = []
    if not all(math.isfinite(value) for value in (tg.xmin, tg.xmax, duration)):
        return ["non_finite_grid_or_wav_duration"]
    if tg.xmin < -EPS or tg.xmax <= tg.xmin:
        reasons.append("invalid_grid_bounds")
    if abs(tg.xmax - duration) > EPS:
        reasons.append("wav_duration_mismatch")
    for tier in tg.tiers:
        previous = -math.inf
        for interval in tier.intervals:
            if not all(math.isfinite(value) for value in (interval.xmin, interval.xmax)):
                reasons.append(f"non_finite:{tier.name}")
                continue
            if interval.xmin < -EPS or interval.xmax > duration + EPS:
                reasons.append(f"out_of_bounds:{tier.name}")
            if interval.xmax <= interval.xmin:
                reasons.append(f"non_positive_interval:{tier.name}")
            if interval.xmin < previous:
                reasons.append(f"overlap_or_nonmonotonic:{tier.name}")
            previous = max(previous, interval.xmax)
    return reasons


def _reference_index(reference_dir: Path, expected: set[str]) -> tuple[dict[str, Path], list[str]]:
    """Index exact authority basenames once; reject ambiguity, never guess.

    Corpus references can be nested by speaker.  A per-stem ``rglob`` would be
    both expensive and nondeterministic, so this one pass only retains exact
    ``{stem}.txt`` / ``{stem}_ref.txt`` names for the expected corpus set.
    """
    candidates: dict[str, list[tuple[int, Path]]] = {}
    errors: list[str] = []
    try:
        files = reference_dir.rglob("*.txt")
        for path in files:
            name = path.name
            if name.endswith("_ref.txt"):
                stem, priority = name[:-len("_ref.txt")], 1
            else:
                stem, priority = path.stem, 0
            if stem in expected:
                candidates.setdefault(stem, []).append((priority, path))
    except OSError as exc:
        return {}, [f"reference_index_unreadable:{exc}"]
    index: dict[str, Path] = {}
    for stem, entries in candidates.items():
        # Both source forms are authority candidates; priority may select one
        # only when it is unique.  Any duplicate basename or competing form is
        # a global failure rather than an arbitrary path choice.
        priorities = {priority for priority, _ in entries}
        if len(entries) > 1:
            kind = "reference_priority_conflict" if len(priorities) > 1 else "reference_basename_duplicate"
            errors.append(f"{kind}:{stem}")
            continue
        index[stem] = entries[0][1]
    return index, errors


def _report_index(path: Path) -> tuple[dict[str, dict], list[str]]:
    rows: dict[str, dict] = {}
    failures: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [f"report_unreadable:{exc}"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            stem = row["stem"]
            if not isinstance(stem, str) or stem in rows:
                raise ValueError("missing/duplicate stem")
            rows[stem] = row
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"invalid_report_row:{number}:{exc}")
    return rows, failures


def _fallback_report_bookkeeping_reasons(row: dict) -> list[str]:
    """Validate persisted fallback ledgers without treating them as authority.

    Final eligibility is still decided from disk by the independent semantic,
    CTC, geometry, and publication checks.  This validator only prevents a
    malformed or stale positive bookkeeping payload from being silently
    whitelisted.
    """
    invalid: list[str] = []

    def fail(key: str) -> None:
        if key not in invalid:
            invalid.append(key)

    def sealed(value: object, schema: str) -> bool:
        return bool(
            isinstance(value, dict)
            and value.get("schema") == schema
            and isinstance(value.get("digest"), str)
            and re.fullmatch(r"[0-9a-f]{64}", value["digest"])
            and value["digest"] == _evidence_digest(value)
        )

    fixes = row.get("fixes")
    if fixes is not None and (
            not isinstance(fixes, list)
            or any(not isinstance(item, dict)
                   or set(item) != {"rule", "word"}
                   or item.get("rule") not in {
                       "short_word_fix", "content_short_word_fix"}
                   or not isinstance(item.get("word"), str)
                   or not item["word"].strip()
                   for item in fixes)):
        fail("fixes")

    surface = row.get("fallback_punctuation_surface")
    punctuation: list[dict] = []
    if surface is not None:
        surface_valid = sealed(surface, "fallback-punctuation-surface-v1")
        source_text = surface.get("source_text") if isinstance(surface, dict) else None
        lexical_count = surface.get("lexical_count") if isinstance(surface, dict) else None
        punctuation_value = surface.get("punctuation") if isinstance(surface, dict) else None
        surface_valid = bool(
            surface_valid
            and isinstance(source_text, str)
            and surface.get("source_digest") == hashlib.sha256(
                source_text.encode()).hexdigest()
            and isinstance(lexical_count, int)
            and not isinstance(lexical_count, bool)
            and lexical_count >= 0
            and isinstance(punctuation_value, list)
            and all(isinstance(item, dict)
                    and type(item.get("source_index")) is int
                    and type(item.get("lexical_boundary")) is int
                    and 0 <= item["lexical_boundary"] <= lexical_count
                    and is_punct(str(item.get("label", "")))
                    for item in punctuation_value))
        if not surface_valid:
            fail("fallback_punctuation_surface")
        else:
            punctuation = punctuation_value

    projection = row.get("fallback_punctuation_projection")
    if projection is not None:
        projection_valid = sealed(
            projection, "fallback-punctuation-projection-v1")
        projection_valid = bool(
            projection_valid
            and isinstance(projection.get("entries"), list)
            and isinstance(projection.get("mapped"), list)
            and type(projection.get("source_lexical_count")) is int
            and type(projection.get("final_lexical_count")) is int)
        if isinstance(surface, dict):
            projection_valid = bool(
                projection_valid
                and projection.get("source_text") == surface.get("source_text")
                and projection.get("source_digest") == surface.get("source_digest")
                and projection.get("surface_ledger_digest") == surface.get("digest"))
        if punctuation:
            projection_valid = bool(
                projection_valid
                and projection.get("safe") is True
                and projection.get("status") == "verified"
                and projection.get("reasons") == []
                and len(projection.get("entries", [])) == len(punctuation))
        else:
            # No source mark means there is no punctuation authority to
            # project.  The producer's negative empty diagnostic is benign.
            projection_valid = bool(
                projection_valid
                and projection.get("safe") is False
                and projection.get("status") == "rejected"
                and projection.get("entries") == [])
        if not projection_valid:
            fail("fallback_punctuation_projection")

    commit = row.get("fallback_surface_final_commit")
    if commit is not None:
        commit_valid = bool(
            isinstance(commit, dict)
            and commit.get("schema") == "fallback-punctuation-surface-v1"
            and commit.get("status") == "verified"
            and commit.get("reasons") == []
            and isinstance(surface, dict)
            and commit.get("source_digest") == surface.get("source_digest")
            and commit.get("ledger_digest") == surface.get("digest"))
        if not commit_valid:
            fail("fallback_surface_final_commit")

    correspondence = row.get("fallback_correspondence")
    if correspondence is not None:
        entries = (correspondence.get("entries")
                   if isinstance(correspondence, dict) else None)
        counts = ([correspondence.get(key) for key in (
            "source_count", "ctc_count", "final_count")]
                  if isinstance(correspondence, dict) else [])
        correspondence_valid = bool(
            sealed(correspondence, "fallback-lexical-correspondence-v2")
            and correspondence.get("status") == "mapped"
            and correspondence.get("safe") is True
            and correspondence.get("reasons") in (None, [])
            and isinstance(entries, list)
            and all(type(value) is int and value >= 0 for value in counts)
            and len(entries) == counts[0]
            and counts[1] == counts[2]
            and all(isinstance(item, dict) for item in entries))
        if not correspondence_valid:
            fail("fallback_correspondence")

    unknown = row.get("fallback_unknown_projection")
    if unknown is not None:
        entries = unknown.get("entries") if isinstance(unknown, dict) else None
        unknown_valid = bool(
            sealed(unknown, "fallback-source-ctc-projection-v1")
            and unknown.get("status") in {"mapped", "omitted"}
            and unknown.get("safe") is True
            and unknown.get("solution_count") == 1
            and type(unknown.get("source_count")) is int
            and type(unknown.get("ctc_count")) is int
            and isinstance(entries, list)
            and len(entries) == unknown.get("source_count")
            and isinstance(unknown.get("recovered"), list))
        if not unknown_valid:
            fail("fallback_unknown_projection")

    bgm = row.get("bgm_ctc_gap_selection")
    if bgm is not None:
        validation = bgm.get("validation") if isinstance(bgm, dict) else None
        bgm_valid = bool(
            isinstance(bgm, dict)
            and bgm.get("schema") == "fallback-bgm-ctc-gap-selection-v1"
            and bgm.get("selection_mode") == "ctc_gap_supported"
            and isinstance(bgm.get("evaluated_intervals"), list)
            and isinstance(validation, dict)
            and validation.get("status") == "verified"
            and validation.get("reasons") == []
            and isinstance(correspondence, dict)
            and validation.get("digest") == correspondence.get("digest"))
        if not bgm_valid:
            fail("bgm_ctc_gap_selection")

    publication = row.get("publication_contract")
    details = (publication.get("details")
               if isinstance(publication, dict) else None)
    if isinstance(details, dict):
        surface_authority = details.get("fallback_surface_authority")
        if isinstance(surface, dict) and (
                not isinstance(surface_authority, dict)
                or surface_authority.get("status") != "verified"
                or surface_authority.get("source_digest")
                != surface.get("source_digest")
                or surface_authority.get("ledger_digest")
                != surface.get("digest")):
            fail("fallback_punctuation_surface")
        projection_authority = details.get(
            "fallback_punctuation_projection_authority")
        if punctuation and isinstance(projection, dict) and (
                not isinstance(projection_authority, dict)
                or projection_authority.get("status") != "verified"
                or projection_authority.get("ledger_digest")
                != projection.get("digest")):
            fail("fallback_punctuation_projection")
        correspondence_projection = details.get(
            "fallback_correspondence_projection")
        if isinstance(correspondence, dict) and (
                not isinstance(correspondence_projection, dict)
                or correspondence_projection.get("digest")
                != correspondence.get("digest")):
            fail("fallback_correspondence")
    return invalid


def _report_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != "ok":
        reasons.append("report_status_not_ok")
    for key in ("hard_integrity_reasons", "filter_reasons", "warnings", "alignment_issues"):
        if row.get(key):
            reasons.append(f"report_positive:{key}")
    reasons.extend(
        f"report_{reason}"
        for reason in _nvasr_report_subcontract_reasons(row))
    reasons.extend(
        f"report_positive:{key}"
        for key in _fallback_report_bookkeeping_reasons(row))
    sp3 = row.get("sp3")
    if sp3:
        benign_edge = False
        output = Path(str(row.get("output", "")))
        try:
            tg = parse_textgrid(output)
            words = next(tier for tier in tg.tiers if tier.name == "words")
            details = sp3.get("details", []) if isinstance(sp3, dict) else []
            sp3_indices = [int(item["index"]) for item in details
                           if isinstance(item, dict) and isinstance(item.get("index"), int)]
            benign_edge = bool(sp3_indices) and all(i in (0, len(words.intervals) - 1) for i in sp3_indices)
        except (OSError, StopIteration, ValueError, IndexError):
            benign_edge = False
        if not benign_edge:
            reasons.append("report_positive:sp3")
    # These fields are normal, independently rechecked transformations.  They
    # are not proof of correctness by themselves, but the auditor separately
    # verifies final tier geometry, reference sequence, phones and provenance.
    # Keep genuine positive QC fields above as vetoes while allowing benign
    # bookkeeping that postprocess emits for every corrected-but-valid stem.
    allowed = {
        "stem", "status", "output", "textgrid_duration", "reference_source",
        "reference_mode", "reference_text_authoritative", "fallback_transcript",
        "reference_coverage", "warnings",
        "hard_integrity_reasons", "filter_reasons", "alignment_issues",
        "english_provenance", "silence_merges", "pp_deoverlap_fixed",
        "text_corrected", "pinyin_displacement", "text_order",
        "sp3", "mfa_unknown_source_redeemed", "evidence_repairs",
        "fallback_lexical_alignment", "cjk_details",
        "ctc_lifecycle", "processed_geometry_contract",
        "nvasr_producer_authority", "nvasr_owner_selection",
        "nvasr_frame_support", "nvasr_candidate_provenance",
        "processed_geometry_digest", "processed_operation_ledger",
        "processed_geometry", "publication_contract",
        # Evidence/bookkeeping fields are positive provenance, not QC vetoes.
        # They are independently checked through the reference, CTC, English
        # ledger, and publication contracts below.
        "authority_compound_reconciliation", "english_surface_units_restored",
        "reference_numeral_normalization", "reference_text_normalized",
        "reference_text_original_raw", "reference_text_raw_sha256",
        "visual_reference_digest", "word_energy_audit", "swallowed_punct",
        "terminal_punctuation_tail_absorption", "punctuation_gap_restorations",
        "fixes", "bgm_ctc_gap_selection", "fallback_correspondence",
        "fallback_punctuation_projection", "fallback_punctuation_surface",
        "fallback_surface_final_commit", "fallback_unknown_projection",
    }
    coverage = row.get("reference_coverage") or {}
    displacement = row.get("pinyin_displacement") or {}
    order = row.get("text_order") or {}
    fallback_safe = (
        row.get("reference_mode") == "fallback"
        and coverage.get("reference_validation_applied") is False
        and (row.get("fallback_lexical_alignment") or {}).get("safe") is True
    )
    authority_text_corrected_ok = (
        coverage.get("exact_cjk_sequence") is True
        and displacement.get("mismatch_rate") == 0.0
        and displacement.get("displacement_runs") == 0
        and order.get("in_order") is True
    )
    fallback_text_corrected_ok = (
        fallback_safe
        and displacement.get("mismatch_rate", 0.0) == 0.0
        and displacement.get("displacement_runs", 0) == 0
        and (not order or order.get("in_order") is True)
    )
    if row.get("text_corrected") and not (
            authority_text_corrected_ok or fallback_text_corrected_ok):
        reasons.append("report_positive:text_corrected")
    if row.get("pinyin_displacement") and not (
            displacement.get("mismatch_rate") == 0.0
            and displacement.get("displacement_runs") == 0):
        reasons.append("report_positive:pinyin_displacement")
    if row.get("text_order") and not (
            order.get("in_order") is True
            and (order.get("ref_cjk_count") == order.get("hanzi_cjk_count")
                 or fallback_safe)):
        reasons.append("report_positive:text_order")
    for key, value in row.items():
        if key not in allowed and value:
            reasons.append(f"report_positive:{key}")
    return reasons


def _fallback_contract_reasons(stem: str, row: dict | None,
                               reference_path: Path | None,
                               ctc_dir: Path) -> list[str]:
    """Bind a no-reference stem to one immutable transcript source.

    The report is only a locator; the auditor reopens the expected source on
    disk and verifies its mode, exact path, regular-file status and digest.
    """
    if not isinstance(row, dict):
        return ["fallback_contract_report_missing"]
    reasons: list[str] = []
    report_mode = row.get("reference_mode")
    source = row.get("reference_source")
    authoritative = row.get("reference_text_authoritative")
    evidence = row.get("fallback_transcript")
    if reference_path is not None:
        reasons.append("fallback_mode_reference_conflict")
    if report_mode != "fallback":
        reasons.append("fallback_mode_report_mismatch")
    if authoritative is not False:
        reasons.append("fallback_authority_flag_mismatch")
    if source not in {"asr_fallback", "lab_fallback"}:
        reasons.append("fallback_source_invalid")
    if not isinstance(evidence, dict):
        return sorted(set(reasons + ["fallback_transcript_evidence_missing"]))
    if evidence.get("source") != source:
        reasons.append("fallback_source_evidence_mismatch")
    expected = (ctc_dir / f"{stem}_text_cn.txt" if source == "asr_fallback"
                else ctc_dir / f"{stem}.lab")
    raw_path = evidence.get("path")
    path = Path(raw_path) if isinstance(raw_path, str) else None
    if path is None or not path.is_absolute() or path != expected.absolute():
        reasons.append("fallback_transcript_path_mismatch")
    else:
        try:
            if path.is_symlink() or not path.is_file() or path.resolve() != expected.resolve():
                reasons.append("fallback_transcript_not_regular")
            elif not path.read_text(encoding="utf-8").strip():
                reasons.append("fallback_transcript_empty")
            if evidence.get("sha256") != _sha256(path):
                reasons.append("fallback_transcript_hash_mismatch")
        except (OSError, UnicodeError):
            reasons.append("fallback_transcript_unreadable")
    return sorted(set(reasons))


def _unknown_recovery_proof_reasons(
        stem: str, final_tg, reference: str, source_path: Path,
        ctc_dir: Path, row: dict, global_manifest: dict | None) -> list[str]:
    """Independently validate the structured initial-Mira unknown proof."""
    proof = row.get("mfa_unknown_source_redeemed") if isinstance(row, dict) else None
    if not isinstance(proof, dict):
        return ["unknown_recovery_proof_missing"]
    reasons: list[str] = []
    if (proof.get("schema") != _UNKNOWN_REPAIR_PROOF_SCHEMA
            or proof.get("scenario") != "initial_mira"
            or proof.get("stem") != stem):
        return ["unknown_recovery_proof_schema"]
    try:
        source_tg = parse_textgrid(source_path)
        source_tiers = [tier for tier in source_tg.tiers if tier.name == "words"]
        if len(source_tiers) != 1:
            return ["unknown_recovery_source_invalid"]
        source_intervals = source_tiers[0].intervals
        unknowns = [(index, iv) for index, iv in enumerate(source_intervals)
                    if is_unknown_token(iv.text.strip())]
        if len(unknowns) != 1:
            return ["unknown_recovery_source_unknown_count"]
        source_index, source_iv = unknowns[0]
        lexical_before = sum(1 for iv in source_intervals[:source_index]
                             if iv.text.strip() and not is_silence(iv.text.strip())
                             and not is_punct(iv.text.strip()))
        if (lexical_before != 0 or source_index == 0
                or source_intervals[source_index - 1].text.strip() != "<eps>"
                or source_index + 1 >= len(source_intervals)
                or source_intervals[source_index + 1].text.strip() != "<eps>"):
            return ["unknown_recovery_source_geometry"]
        source_value = {"ordinal": source_index, "start": float(source_iv.xmin),
                        "end": float(source_iv.xmax), "text": source_iv.text.strip()}
        source_proof = proof.get("source", {})
        if (source_proof.get("interval") != source_value
                or source_proof.get("interval_sha256") != _evidence_digest(source_value)
                or source_proof.get("lexical_ordinal") != 0
                or source_proof.get("neighbors") != ["<eps>", "<eps>"]):
            return ["unknown_recovery_source_binding"]

        token_path = ctc_dir / f"{stem}_tokens.jsonl"
        tokens = [json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        if not tokens or not isinstance(tokens[0], dict):
            return ["unknown_recovery_ctc_missing"]
        token = tokens[0]
        token_value = {"ordinal": 0, "word": token.get("word", ""),
                       "start_s": float(token["start_s"]),
                       "end_s": float(token["end_s"]),
                       "type": token.get("type", "word")}
        ctc_proof = proof.get("ctc", {})
        if (token_value["word"].strip().casefold() != "mira"
                or token_value["type"] != "word"
                or ctc_proof.get("token") != token_value
                or ctc_proof.get("token_sha256") != _evidence_digest(token_value)
                ):
            return ["unknown_recovery_ctc_binding"]
        if any(isinstance(item, dict)
               and int(item.get("ordinal", index)) != index
               for index, item in enumerate([token])):
            return ["unknown_recovery_ctc_binding"]
        source_sequence = [_lexical_identity(iv.text.strip()) for iv in source_intervals
                           if iv.text.strip() and not is_silence(iv.text.strip())
                           and not is_punct(iv.text.strip())]
        source_unknown_position = next((index for index, value in enumerate(source_sequence)
                                        if value == "<unknown>"), None)
        if source_unknown_position is None:
            return ["unknown_recovery_source_binding"]
        source_sequence[source_unknown_position] = "mira"
        ctc_sequence = [_lexical_identity(item.get("word", ""), ctc_item=item)
                        for item in tokens
                        if isinstance(item, dict) and item.get("type", "word") == "word"]
        correspondence = proof.get("ordered_correspondence", {})
        if (source_sequence != ctc_sequence
                or correspondence.get("source_tokens") != source_sequence
                or correspondence.get("ctc_tokens") != ctc_sequence
                or correspondence.get("sha256") != _evidence_digest(source_sequence)):
            return ["unknown_recovery_token_correspondence"]

        reference_tokens = _semantic_tokens(reference)
        reference_semantics = project_authority_semantics(reference)
        reference_matches = [item for item in reference_semantics
                             if item.get("kind") == "english"
                             and item.get("alignment_token", "").casefold() == "mira"
                             and item.get("reference_ordinal") == 0]
        reference_proof = proof.get("reference", {})
        if (len(reference_matches) != 1
                or reference_proof.get("ordinal") != 0
                or reference_proof.get("token") != str(reference_matches[0]["surface"]).casefold()):
            return ["unknown_recovery_reference_binding"]

        final_words = final_tg.tiers[3].intervals
        final_lexical = [iv for iv in final_words
                         if iv.text.strip() and not is_silence(iv.text)
                         and not is_punct(iv.text)]
        owners = [iv for index, iv in enumerate(final_lexical)
                  if index == 0 and _lexical_identity(iv.text) == "mira"]
        if len(owners) != 1:
            return ["unknown_recovery_final_owner"]
        owner = owners[0]
        owner_value = {"text": owner.text.strip(), "start": float(owner.xmin),
                       "end": float(owner.xmax)}
        final_proof = proof.get("final", {})
        if final_proof.get("owner") != owner_value:
            return ["unknown_recovery_final_binding"]
        final_sequence = _semantic_tokens(_tier_text(final_tg.tiers[2]))
        if final_sequence != reference_tokens:
            return ["unknown_recovery_final_semantic_mismatch"]
        if (final_proof.get("semantic_sequence_sha256")
                != _evidence_digest([[kind, value] for kind, value in final_sequence])
                or final_proof.get("semantic_token_count") != len(final_sequence)):
            return ["unknown_recovery_final_binding"]

        ledger_proof = proof.get("english_ledger", {})
        if not isinstance(global_manifest, dict):
            return ["unknown_recovery_ledger_missing"]
        ledger_info = global_manifest.get("_ledger_by_stem", {}).get(stem)
        if not ledger_info:
            return ["unknown_recovery_ledger_missing"]
        ledger_path = ledger_info["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if _sha256(ledger_path) != ledger_proof.get("ledger_sha256"):
            return ["unknown_recovery_ledger_hash"]
        records = [record for segment in ledger.get("segments", [])
                   for record in segment.get("words", [])
                   if isinstance(record, dict)
                   and record.get("word_id") == ledger_proof.get("word_id")]
        if len(records) != 1:
            return ["unknown_recovery_ledger_word"]
        record = records[0]
        if (record.get("status") != "verified"
                or record.get("provenance") != "english_mfa_textgrid"
                or str(record.get("ctc_text", "")).casefold() != "mira"
                or ledger_proof.get("ctc_ordinal") != record.get("ctc_ordinal")
                or ledger_proof.get("ctc_text", "").casefold() != "mira"
                or ledger_proof.get("word_sha256") != _evidence_digest(record)):
            return ["unknown_recovery_ledger_word"]
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return ["unknown_recovery_proof_invalid"]
    return reasons


def _evidence_repair_reasons(stem: str, final_tg, row: dict) -> list[str]:
    """Check repair records without trusting their report-only labels."""
    repairs = row.get("evidence_repairs") if isinstance(row, dict) else None
    if repairs is None:
        return []
    if not isinstance(repairs, list) or not repairs:
        return ["evidence_repair_record_invalid"]
    allowed_overlap = {"000240", "000314", "001776", "001802"}
    words = final_tg.tiers[3].intervals
    reasons: list[str] = []
    for repair in repairs:
        if (not isinstance(repair, dict)
                or repair.get("schema") != _EVIDENCE_REPAIR_SCHEMA
                or repair.get("stem") != stem
                or repair.get("proof") != "source_mfa_ctc_unique_monotone_boundary"):
            reasons.append("evidence_repair_record_invalid")
            continue
        indices = repair.get("word_indices")
        if (not isinstance(indices, list) or len(indices) != 2
                or any(type(index) is not int for index in indices)
                or indices[1] != indices[0] + 1
                or indices[0] < 0):
            reasons.append("evidence_repair_geometry_invalid")
            continue
        source = repair.get("source_words")
        ctc = repair.get("ctc_tokens")
        if (not isinstance(source, list) or len(source) != 2
                or not isinstance(ctc, list) or len(ctc) != 2):
            reasons.append("evidence_repair_evidence_missing")
            continue
        # ``word_indices`` belongs to the pre-rebuild visual words tier.  The
        # final tier can legitimately have a different number of intervals
        # after MFA/authority reconciliation (for example an <eps> interval
        # is removed).  Resolve the pair by its dual-evidence labels and the
        # committed boundary instead of treating that historical index as a
        # final-tier primary key.
        expected_left = str(source[0].get("text", "")).strip()
        expected_right = str(source[1].get("text", "")).strip()
        if not expected_left or not expected_right:
            expected_left = str(ctc[0].get("word", "")).strip()
            expected_right = str(ctc[1].get("word", "")).strip()
        boundary = repair.get("boundary_s")
        candidates = [
            (left, right) for left, right in zip(words, words[1:])
            if left.text.strip().casefold() == expected_left.casefold()
            and right.text.strip().casefold() == expected_right.casefold()
        ]
        if (not isinstance(boundary, (int, float)) or not candidates):
            reasons.append("evidence_repair_boundary_invalid")
            continue
        matching = [
            (left, right) for left, right in candidates
            if _same_number(left.xmax, boundary)
            and _same_number(right.xmin, boundary)
            and left.xmax <= right.xmin + EPS
        ]
        if len(matching) != 1:
            reasons.append("evidence_repair_boundary_invalid")
            continue
        left, right = matching[0]
        if (not _PINYIN.fullmatch(left.text.strip())
                or not _PINYIN.fullmatch(right.text.strip())):
            reasons.append("evidence_repair_owner_invalid")
        try:
            if (source[0].get("ordinal") >= source[1].get("ordinal")
                    or source[0].get("end") > source[1].get("start")
                    or source[0].get("end") - source[0].get("start") < 0.030 - 1e-9
                    or source[1].get("end") - source[1].get("start") < 0.030 - 1e-9
                    or ctc[0].get("ordinal") >= ctc[1].get("ordinal")
                    or ctc[0].get("end_s") > ctc[1].get("start_s") + 1e-3
                    or ctc[0].get("end_s") - ctc[0].get("start_s") < 0.030 - 1e-9
                    or ctc[1].get("end_s") - ctc[1].get("start_s") < 0.030 - 1e-9
                    or source[0].get("text", "").casefold() != left.text.strip().casefold()
                    or source[1].get("text", "").casefold() != right.text.strip().casefold()
                    or ctc[0].get("word", "").casefold() != left.text.strip().casefold()
                    or ctc[1].get("word", "").casefold() != right.text.strip().casefold()):
                reasons.append("evidence_repair_dual_evidence_invalid")
        except (AttributeError, TypeError):
            reasons.append("evidence_repair_dual_evidence_invalid")
        if repair.get("kind") == "overlap" and stem[:6] not in allowed_overlap:
            reasons.append("evidence_repair_overlap_not_allowlisted")
    return sorted(set(reasons))


def _sp1_reasons(tg) -> list[str]:
    reasons: list[str] = []
    # raw_text and pinyin are the two surface tiers whose single full-span
    # interval carries the preserved leading marker.  Derived tiers may start
    # at t=0 with their first lexical owner; requiring a synthetic <sp1>
    # interval in every derived tier falsely rejects valid head-silence output.
    for tier in tg.tiers:
        if tier.name not in {"raw_text", "pinyin"}:
            continue
        text = _tier_text(tier)
        if not text.startswith("<sp1>") or len(_SP1.findall(text)) != 1:
            reasons.append(f"sp1_contract:{tier.name}")
    for tier in tg.tiers:
        if tier.name in {"raw_text", "pinyin"}:
            continue
        for index, interval in enumerate(tier.intervals):
            if not _SP1.search(interval.text or ""):
                continue
            if index == 0:
                if interval.xmin > tier.xmin + EPS:
                    reasons.append(f"sp1_contract:{tier.name}")
                continue
            # A final endpoint marker can remain after the last lexical or
            # punctuation owner.  Only an interior marker with a later
            # lexical owner violates the merge contract.
            later_lexical = any(
                not is_silence(next_iv.text)
                and not is_punct(next_iv.text)
                for next_iv in tier.intervals[index + 1:]
            )
            if later_lexical:
                reasons.append(f"sp1_contract:{tier.name}")
    return reasons


def _publication_geometry_reasons(tg) -> list[str]:
    """Independently validate the final display-owner partition.

    ``pinyin_phones`` is intentionally sparse during true silence, so this
    check requires ownership for each phone that exists but does not require
    that tier to cover the axis.  The words and hanzi tiers are the published
    display partition and must cover the complete axis within the shared
    axis epsilon.
    """
    reasons: list[str] = []
    words = tg.tiers[3]
    hanzi = tg.tiers[2]

    def partition_reasons(tier) -> None:
        if not tier.intervals:
            reasons.append(f"{tier.name}_empty")
            return
        first = tier.intervals[0]
        if first.xmin > tg.xmin + EPS:
            reasons.append(f"{tier.name}_coverage_hole")
        previous = first
        for current in tier.intervals[1:]:
            delta = current.xmin - previous.xmax
            if delta > EPS:
                reasons.append(f"{tier.name}_coverage_hole")
            elif delta < -EPS:
                reasons.append(f"{tier.name}_overlap")
            previous = current
        if tier.intervals[-1].xmax < tg.xmax - EPS:
            reasons.append(f"{tier.name}_coverage_hole")
        if any(is_silence(interval.text)
               and interval.xmin > tg.xmin + EPS
               and interval.xmax < tg.xmax - EPS
               for interval in tier.intervals):
            reasons.append("strict_interior_sp")

    partition_reasons(words)
    partition_reasons(hanzi)
    if len(words.intervals) != len(hanzi.intervals):
        reasons.append("hanzi_words_count_mismatch")
    for word, label in zip(words.intervals, hanzi.intervals):
        if (abs(word.xmin - label.xmin) > EPS
                or abs(word.xmax - label.xmax) > EPS):
            reasons.append("hanzi_words_boundary_mismatch")
            continue
        word_text = word.text.strip()
        label_text = label.text.strip()
        if ((is_silence(word_text) or is_punct(word_text))
                and word_text != label_text):
            reasons.append("silence_or_punctuation_label_mismatch")
            if is_silence(word_text) and is_silence(label_text):
                reasons.append("silence_label_split")
        elif is_english_token(word_text) and word_text != label_text:
            reasons.append("english_label_mismatch")

    phones = tg.tiers[4]
    for phone in phones.intervals:
        owners = [word for word in words.intervals
                  if phone.xmin >= word.xmin
                  and phone.xmax <= word.xmax]
        if len(owners) != 1:
            reasons.append("phone_owner_mismatch")
            break
    return sorted(set(reasons))


def _content_reasons(tg, reference: str, *, reference_authoritative: bool = True) -> list[str]:
    reasons: list[str] = []
    raw, pinyin, hanzi, words, phones = tg.tiers
    reasons.extend(_publication_geometry_reasons(tg))
    if any(len(tier.intervals) != 1 for tier in (raw, pinyin)):
        reasons.append("raw_or_pinyin_not_single_full_interval")
    for tier in (raw, pinyin):
        if len(tier.intervals) == 1:
            interval = tier.intervals[0]
            if abs(interval.xmin - tg.xmin) > EPS or abs(interval.xmax - tg.xmax) > EPS:
                reasons.append(f"not_full_span:{tier.name}")
    if len(hanzi.intervals) != len(words.intervals):
        reasons.append("hanzi_words_count_mismatch")
    for h_iv, w_iv in zip(hanzi.intervals, words.intervals):
        if abs(h_iv.xmin - w_iv.xmin) > EPS or abs(h_iv.xmax - w_iv.xmax) > EPS:
            reasons.append("hanzi_words_boundary_mismatch")
            break

    reference_tokens = _semantic_tokens(reference) if reference_authoritative else []
    if reference_authoritative:
        final_tokens = _semantic_tokens(
            _canonicalize_reference_hyphens(_tier_text(hanzi)))
        if not _semantic_sequence_compatible(reference_tokens, final_tokens):
            reasons.append("reference_semantic_sequence_mismatch")
        if not _semantic_sequence_compatible(
                reference_tokens,
                _semantic_tokens(_canonicalize_reference_hyphens(_tier_text(raw)))):
            reasons.append("reference_raw_semantic_sequence_mismatch")
        # Pinyin syllables are the Chinese realization, not reference English
        # words.  Remove only fully toned pinyin tokens before comparing the
        # remaining NVV/punctuation/English sequence to authority.
        pinyin_semantic = _semantic_tokens(re.sub(
            r"(?<![A-Za-z])[a-z]+[1-5](?![A-Za-z0-9])", "",
            _canonicalize_reference_hyphens(_SP1.sub("", _tier_text(pinyin)))))
        reference_non_cjk = [token for token in reference_tokens if token[0] != "cjk"]
        if not _semantic_sequence_compatible(reference_non_cjk, pinyin_semantic):
            reasons.append("reference_pinyin_semantic_sequence_mismatch")
        reference_cjk = [value for kind, value in reference_tokens if kind == "cjk"]
        hanzi_cjk = [char for iv in hanzi.intervals for char in iv.text if _CJK.fullmatch(char)]
        if reference_cjk != hanzi_cjk:
            reasons.append("reference_hanzi_cjk_mismatch")
    else:
        # Fallback has no lexical authority, but CJK ownership remains a
        # common contract: toned-pinyin words and CJK hanzi intervals must
        # occupy the same word slots.  This deliberately does not compare to
        # source semantic text or source CJK counts.
        pinyin_indices = {i for i, iv in enumerate(words.intervals)
                          if _PINYIN.fullmatch(iv.text.strip())}
        cjk_indices = {i for i, iv in enumerate(hanzi.intervals)
                       if _CJK.fullmatch(iv.text.strip())}
        if pinyin_indices != cjk_indices:
            reasons.append("cjk_pinyin_ownership_mismatch")
        reference_cjk = []
    if any(_PINYIN.search(iv.text.strip()) for iv in hanzi.intervals):
        reasons.append("hanzi_contains_pinyin")

    pinyin_words = [iv for iv in words.intervals if _PINYIN.fullmatch(iv.text.strip())]
    if reference_authoritative and len(reference_cjk) != len(pinyin_words):
        reasons.append("cjk_pinyin_count_mismatch")
    cjk_word_indices = [i for i, iv in enumerate(hanzi.intervals) if _CJK.fullmatch(iv.text.strip())]
    if reference_authoritative and len(cjk_word_indices) != len(reference_cjk):
        reasons.append("hanzi_cjk_interval_count_mismatch")
    for index in cjk_word_indices:
        if not _PINYIN.fullmatch(words.intervals[index].text.strip()):
            reasons.append("cjk_without_toned_pinyin_word")
            break

    # Phones may only occupy their owning word; every English word requires
    # real en:-prefixed phones, never a self-referential lexical phone.
    for phone in phones.intervals:
        owners = [word for word in words.intervals
                  if phone.xmin >= word.xmin and phone.xmax <= word.xmax]
        if not owners:
            reasons.append("phone_outside_word")
            break
    for word in words.intervals:
        token = word.text.strip()
        owned = [phone.text.strip() for phone in phones.intervals
                 if phone.xmax > word.xmin + EPS and phone.xmin < word.xmax - EPS
                 and not is_silence(phone.text.strip())]
        if is_english_token(token):
            if not owned or any(not phone.startswith("en:") or not phone[3:] for phone in owned):
                reasons.append("english_missing_en_phones")
            # A valid ARPABET symbol can equal a short lexical token (e.g.
            # ``S`` is a real phone in the word ``S``).  Reject only a wholly
            # self-referential phone sequence, never a mixed sequence that
            # contains genuine English MFA evidence.
            en_owned = [phone[3:] for phone in owned if phone.startswith("en:")]
            if en_owned and all(phone.lower() == token.lower() for phone in en_owned):
                reasons.append("english_self_referential_phone")
        if is_unknown_token(token):
            reasons.append("final_unknown_token")
        elif token.lower() == "unk":
            # A literal English "unk" is only acceptable with matching
            # authoritative reference and genuine English phone evidence.
            ref_english = [value for kind, value in reference_tokens if kind == "english"]
            if (reference_authoritative and
                    ("unk" not in ref_english or not owned or not all(p.startswith("en:") for p in owned))):
                reasons.append("ambiguous_bare_unk")
    for tier in tg.tiers:
        for interval in tier.intervals:
            if is_unknown_token(interval.text):
                reasons.append("final_unknown_token")
            if interval.text.strip() == "spn":
                reasons.append("final_lexical_spn")
    return reasons


def _aligned_reasons(path: Path, reference: str) -> list[str]:
    try:
        tg = parse_textgrid(path)
    except Exception as exc:
        return [f"aligned_unreadable:{exc}"]
    ref_english = {value for kind, value in _semantic_tokens(reference) if kind == "english"}
    reasons: list[str] = []
    word_tier = next((tier for tier in tg.tiers if tier.name == "words"), None)
    phone_tier = next((tier for tier in tg.tiers if tier.name == "phones"), None)
    if word_tier is None or phone_tier is None:
        return ["aligned_missing_words_or_phones"]
    for interval in word_tier.intervals:
        label = interval.text.strip()
        if is_unknown_token(label):
            reasons.append("aligned_unknown_token")
        elif label.lower() == "unk":
            owned = [p.text.strip() for p in phone_tier.intervals
                     if p.xmax > interval.xmin + EPS and p.xmin < interval.xmax - EPS]
            if "unk" not in ref_english or not owned or all(p.lower() == "unk" for p in owned):
                reasons.append("aligned_ambiguous_bare_unk")
        if is_english_token(label):
            owned = [p.text.strip() for p in phone_tier.intervals
                     if p.xmax > interval.xmin + EPS and p.xmin < interval.xmax - EPS]
            if owned and all(phone.lower() == label.lower() for phone in owned):
                reasons.append("aligned_english_self_referential_phone")
    for interval in [*word_tier.intervals, *phone_tier.intervals]:
        label = interval.text.strip()
        if is_unknown_token(label):
            reasons.append("aligned_unknown_token")
    for phone in phone_tier.intervals:
        if phone.text.strip() != "spn":
            continue
        owners = [word.text.strip() for word in word_tier.intervals
                  if phone.xmax > word.xmin + EPS and phone.xmin < word.xmax - EPS]
        if not owners:
            reasons.append("aligned_spn_without_owner")
            continue
        # ``spn`` is a normal MFA placeholder for English/NVV/silence/punct
        # regions.  It is lexical failure only when owned by Chinese pinyin or
        # an explicit unknown placeholder (final English still needs en:).
        if any(is_pinyin_syllable(owner) or is_unknown_token(owner)
               or _CJK.search(owner) for owner in owners):
            reasons.append("aligned_lexical_spn")
    return reasons


def _named_source_tiers(path: Path):
    """Independently parse the one and only MFA words/phones tiers."""
    tg = parse_textgrid(path)
    words = [tier for tier in tg.tiers if tier.name == "words"]
    phones = [tier for tier in tg.tiers if tier.name == "phones"]
    if len(words) != 1 or len(phones) != 1:
        raise ValueError("source_interval_invalid")
    return words[0].intervals, phones[0].intervals


def _ctc_english_words(path: Path) -> dict[int, str]:
    """Read CTC named words with their full-tier ordinal; duplicates stay distinct."""
    tg = parse_textgrid(path)
    tiers = [tier for tier in tg.tiers if tier.name == "words"]
    if len(tiers) != 1:
        raise ValueError("ctc_words_tier_invalid")
    return {ordinal: iv.text.strip() for ordinal, iv in enumerate(tiers[0].intervals)
            if is_english_token(iv.text.strip())}


def _processed_ctc_tokens(ctc_dir: Path, stem: str) -> list[dict] | None:
    """Load the independently produced processed-token sidecar, if present."""
    path = ctc_dir / f"{stem}_tokens.jsonl"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return rows if all(isinstance(row, dict) for row in rows) else None


def _source_english_words(path: Path) -> list[dict]:
    """Validate source MFA phones directly; never trust a ledger phone list."""
    source_words, source_phones = _named_source_tiers(path)
    # Preserve the source words-tier ordinal.  MFA's English source TextGrid
    # includes leading ``sp``/silence intervals, so the ordinal of the second
    # lexical word is not necessarily zero-based within the filtered list.
    lexical_words = [(ordinal, iv) for ordinal, iv in enumerate(source_words)
                     if iv.text.strip() and not is_silence(iv.text.strip())]
    previous = -math.inf
    for _, iv in lexical_words:
        if (not all(math.isfinite(v) for v in (iv.xmin, iv.xmax))
                or iv.xmax <= iv.xmin or iv.xmin < previous):
            raise ValueError("source_interval_invalid")
        previous = max(previous, iv.xmax)
    lexical_phones = [(ordinal, iv) for ordinal, iv in enumerate(source_phones)
                      if not is_silence(iv.text.strip()) and iv.text.strip() != "sp"]
    owners: list[list] = [[] for _ in lexical_words]
    previous = -math.inf
    for source_ordinal, phone in lexical_phones:
        label = phone.text.strip()
        if (not all(math.isfinite(v) for v in (phone.xmin, phone.xmax))
                or phone.xmax <= phone.xmin or phone.xmin < previous):
            raise ValueError("source_interval_invalid")
        previous = max(previous, phone.xmax)
        if not is_english_phone(label):
            raise ValueError("english_phone_unknown")
        matches = [index for index, (_, word) in enumerate(lexical_words)
                   if phone.xmin >= word.xmin - EPS and phone.xmax <= word.xmax + EPS]
        if len(matches) != 1:
            raise ValueError("source_interval_invalid")
        owners[matches[0]].append((source_ordinal, phone))
    result: list[dict] = []
    for (ordinal, word), phones in zip(lexical_words, owners):
        if not phones:
            raise ValueError("english_phone_empty")
        if abs(phones[0][1].xmin - word.xmin) > EPS or abs(phones[-1][1].xmax - word.xmax) > EPS:
            raise ValueError("source_interval_invalid")
        for (_, left), (_, right) in zip(phones, phones[1:]):
            if right.xmin - left.xmax > EPS:
                raise ValueError("source_interval_invalid")
        result.append({"ordinal": ordinal, "text": word.text.strip(), "start": word.xmin,
                       "end": word.xmax,
                       "phones": [{"ordinal": phone_ordinal, "label": phone.text.strip(),
                                   "start": phone.xmin, "end": phone.xmax,
                                   "mfa_phone_ordinal": source_ordinal}
                                  for phone_ordinal, (source_ordinal, phone) in enumerate(phones)]})
    return result


def _same_number(left: object, right: object) -> bool:
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right) and abs(left - right) <= EPS)


def _pronunciation_consumer_reasons(record: dict, source_word: dict,
                                    ledger: dict) -> list[str]:
    """Independently validate W1 pronunciation policy at the strict audit."""
    token = str(record.get("alignment_token", "")).casefold()
    source_labels = tuple(str(phone.get("label", "")).strip()
                          for phone in source_word.get("phones", [])
                          if isinstance(phone, dict))
    ledger_labels = tuple(str(phone.get("label", "")).strip()
                          for phone in record.get("phones", [])
                          if isinstance(phone, dict))
    if token == "app":
        return ([] if ledger_labels == APP_EXPECTED_PRONUNCIATION
                and source_labels == APP_EXPECTED_PRONUNCIATION
                else ["app_expected_pronunciation_mismatch"])
    if token != "sos":
        return []
    policy = record.get("pronunciation_policy")
    dictionary = ledger.get("dictionary_provenance")
    errors: list[str] = []
    if (record.get("pronunciation_policy_id") != SOS_PRONUNCIATION_POLICY_ID
            or not isinstance(policy, dict)
            or policy.get("policy_id") != SOS_PRONUNCIATION_POLICY_ID
            or tuple(policy.get("expected_pronunciation", ())) != SOS_EXPECTED_PRONUNCIATION
            or tuple(policy.get("actual_source_sequence", ())) != source_labels
            or source_labels != SOS_EXPECTED_PRONUNCIATION
            or ledger_labels != source_labels
            or policy.get("dictionary_provenance") != dictionary
            or record.get("dictionary_provenance") != dictionary):
        errors.append("sos_pronunciation_policy_mismatch")
    if (not isinstance(dictionary, dict)
            or not isinstance(dictionary.get("path"), str)
            or not isinstance(dictionary.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", dictionary["sha256"])):
        errors.append("sos_dictionary_provenance_invalid")
    else:
        try:
            dictionary_path = Path(dictionary["path"])
            if (dictionary_path.is_symlink() or not dictionary_path.is_file()
                    or _sha256(dictionary_path) != dictionary["sha256"]):
                errors.append("sos_dictionary_hash_mismatch")
        except (OSError, ValueError, TypeError):
            errors.append("sos_dictionary_provenance_unreadable")
    return sorted(set(errors))


def _load_english_manifest(args: argparse.Namespace) -> tuple[dict | None, list[str]]:
    """Load and validate the global strict-en-mfa-v1 contract once."""
    try:
        manifest_path = _safe_file_under(args.en_phones_dir, str(args.en_manifest))
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, ["english_provenance_manifest_failed"]
    if (raw.get("schema") != EN_PROVENANCE_SCHEMA
            or raw.get("strict_provenance") is not True
            or raw.get("canonical_units") != CANONICAL_UNITS_SCHEMA):
        if raw.get("schema") == HISTORICAL_EN_PROVENANCE_SCHEMA:
            return None, ["english_provenance_legacy_schema"]
        return None, ["english_provenance_manifest_failed"]
    if raw.get("status") not in {"success", "partial", "no_english"}:
        return None, ["english_provenance_manifest_failed"]
    for key in ("expected_segments", "produced_segments", "rejected_segments", "stem_ledgers", "counts"):
        if key not in raw:
            return None, ["english_provenance_manifest_failed"]
    expected = raw.get("expected_segments")
    produced = raw.get("produced_segments")
    rejected = raw.get("rejected_segments")
    if (not isinstance(expected, list) or not isinstance(produced, list) or not isinstance(rejected, list)
            or not all(isinstance(item, str) for item in expected + produced)
            or len(expected) != len(set(expected)) or len(produced) != len(set(produced))):
        return None, ["english_provenance_manifest_failed"]
    rejected_ids: list[str] = []
    for item in rejected:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not isinstance(item.get("reason"), str) or not item.get("reason")):
            return None, ["english_provenance_manifest_failed"]
        rejected_ids.append(item["id"])
    if (len(rejected_ids) != len(set(rejected_ids)) or set(produced) & set(rejected_ids)
            or set(expected) != set(produced) | set(rejected_ids)):
        return None, ["english_provenance_manifest_failed"]
    if ((raw["status"] == "success" and rejected_ids)
            or (raw["status"] == "partial" and not rejected_ids)):
        return None, ["english_provenance_manifest_failed"]
    if raw["status"] == "no_english":
        counts = raw.get("counts")
        if (expected or produced or rejected or raw.get("stem_ledgers")
                or not isinstance(counts, dict)
                or set(counts) != {"english_stems", "english_segments", "english_words",
                                   "verified_words", "rejected_words"}
                or any(type(value) is not int or value != 0 for value in counts.values())):
            return None, ["english_provenance_manifest_failed"]
    ledgers = raw.get("stem_ledgers")
    if not isinstance(ledgers, list):
        return None, ["english_provenance_manifest_failed"]
    ledger_by_stem: dict[str, dict] = {}
    try:
        for entry in ledgers:
            stem = entry["stem"]
            if not _safe_stem(stem) or stem in ledger_by_stem:
                raise ValueError("invalid ledger stem")
            ledger_path = _safe_file_under(args.en_phones_dir, entry["path"])
            if _sha256(ledger_path) != entry["sha256"]:
                raise ValueError("ledger hash")
            ledger_by_stem[stem] = {"entry": entry, "path": ledger_path}
    except (KeyError, TypeError, OSError, ValueError):
        return None, ["english_provenance_hash_mismatch"]
    if raw["status"] in {"success", "partial"}:
        try:
            expected_stems = set()
            for item in expected:
                stem, ordinal = item.rsplit(":s", 1)
                if not _safe_stem(stem) or not ordinal.isdecimal() or item != f"{stem}:s{int(ordinal)}":
                    raise ValueError("invalid stable segment id")
                expected_stems.add(stem)
            mfa = raw.get("mfa")
            if (not isinstance(mfa, dict) or type(mfa.get("return_code")) is not int
                    or mfa.get("return_code") != 0 or mfa.get("timed_out") is not False
                    or mfa.get("exception") != ""
                    or not isinstance(mfa.get("command"), list)
                    or not isinstance(mfa.get("timeout_seconds"), (int, float))
                    or mfa.get("timeout_seconds") < 0
                    or any(not isinstance(mfa.get(key), str)
                           or not re.fullmatch(r"[0-9a-f]{64}", mfa[key])
                           for key in ("acoustic_model_sha256", "dictionary_sha256"))):
                raise ValueError("invalid successful MFA record")
        except (ValueError, AttributeError, TypeError):
            return None, ["english_provenance_manifest_failed"]
        counts = raw.get("counts", {})
        if (len(expected_stems) != len(ledger_by_stem) or set(ledger_by_stem) != expected_stems
                or not isinstance(counts, dict)
                or counts.get("english_stems") != len(expected_stems)
                or counts.get("english_segments") != len(expected)
                or not all(type(counts.get(key)) is int and counts[key] >= 0
                           for key in ("english_words", "verified_words", "rejected_words"))
                or counts.get("verified_words") + counts.get("rejected_words") != counts.get("english_words")):
            return None, ["english_provenance_manifest_failed"]
    raw["_ledger_by_stem"] = ledger_by_stem
    return raw, []


def _safe_stem(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and value not in {".", ".."}
            and "\x00" not in value and Path(value).name == value)


def _english_provenance_reasons(stem: str, final_tg, ctc_dir: Path,
                                args: argparse.Namespace, global_manifest: dict | None,
                                reference_text: str | None = None) -> tuple[list[str], dict | None]:
    """Cross-check source MFA TextGrids against final en: phones and ledger."""
    def _compact_english(value: object) -> str:
        # CTC canonicalizes hyphenated authority surfaces (e.g. V-Up) to
        # vup, while the final words tier preserves the authority spelling.
        # Compare lexical identity modulo separators at the provenance
        # boundary; hyphen ownership is checked separately by authority-unit
        # validation.
        return re.sub(r"[^a-z0-9]", "", str(value).casefold())

    final_words = [iv for iv in final_tg.tiers[3].intervals if is_english_token(iv.text.strip())]
    if not final_words:
        return [], None
    if global_manifest is None:
        return ["english_provenance_manifest_missing"], None
    if global_manifest.get("status") not in {"success", "partial"}:
        return ["english_provenance_manifest_failed"], None
    rejected_for_stem = {
        item["id"] for item in global_manifest.get("rejected_segments", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
        and item["id"].startswith(f"{stem}:s")
    }
    if rejected_for_stem:
        return ["english_segment_rejected"], None
    try:
        ledger_info = global_manifest["_ledger_by_stem"].get(stem)
        if ledger_info is None:
            raise KeyError("missing ledger")
        ledger_path = ledger_info["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (ledger.get("schema") != EN_PROVENANCE_SCHEMA
                or ledger.get("stem") != stem
                or ledger.get("canonical_units") != CANONICAL_UNITS_SCHEMA):
            if ledger.get("schema") == HISTORICAL_EN_PROVENANCE_SCHEMA:
                return ["english_provenance_legacy_schema"], None
            return ["english_provenance_manifest_failed"], None
        ctc_path = ctc_dir / f"{stem}.TextGrid"
        if not ctc_path.is_file():
            ctc_path = ctc_dir / stem / f"{stem}.TextGrid"
        if not ctc_path.is_file() or _sha256(ctc_path) != ledger.get("ctc_textgrid_sha256"):
            return ["english_provenance_hash_mismatch"], None
        ctc_english = _ctc_english_words(ctc_path)
        ctc_tg = parse_textgrid(ctc_path)
        ctc_word_tier = next(
            (tier for tier in ctc_tg.tiers if tier.name == "words"), None)
        if ctc_word_tier is None:
            return ["english_provenance_manifest_failed"], None
        processed_tokens = _processed_ctc_tokens(ctc_dir, stem)
        ctc_word_intervals = list(enumerate(ctc_word_tier.intervals))
        used_ctc_ordinals: set[int] = set()

        def _strict_span_valid(record: dict) -> bool:
            """Apply the shared span contract with independently loaded CTC evidence."""
            source = record.get("source_ctc_ordinals")
            if (not isinstance(source, list) or not source
                    or any(type(value) is not int or value < 0 for value in source)
                    or any(left >= right for left, right in zip(source, source[1:]))):
                return False
            contiguous = all(right - left == 1
                             for left, right in zip(source, source[1:]))
            if contiguous:
                return record.get("ctc_ordinal") == source[0]
            token = resolve_processed_english_token(processed_tokens, source)
            try:
                validate_processed_english_token_binding(record, token)
            except (EnglishUnitError, TypeError, ValueError):
                return False
            return True

        def _actual_ctc_ordinal(record: dict) -> int | None:
            """Resolve source-side ordinal to the actual named-tier ordinal.

            Canonical token sidecars retain their own source ordinal lineage.
            That ordinal can differ from the final words-tier ordinal when
            blank/pause intervals were materialized.  Audit by the immutable
            text/span evidence, then compare against the actual CTC tier.
            """
            text = _compact_english(record.get("ctc_text", ""))
            span = record.get("canonical_span")
            start = (float(span[0]) if isinstance(span, list) and len(span) == 2
                     and isinstance(span[0], (int, float)) else None)
            candidates = []
            for ordinal, interval in ctc_word_intervals:
                if ordinal in used_ctc_ordinals or ordinal not in ctc_english:
                    continue
                actual = _compact_english(interval.text)
                if not actual or not text or not (actual == text
                        or actual.startswith(text) or text.startswith(actual)):
                    continue
                if start is not None and abs(float(interval.xmin) - start) > 0.012:
                    continue
                candidates.append(ordinal)
            if len(candidates) == 1:
                used_ctc_ordinals.add(candidates[0])
                return candidates[0]
            return None
        segments = ledger.get("segments")
        if not isinstance(segments, list) or not segments:
            return ["english_provenance_manifest_failed"], None
        expected_ids = [entry for entry in global_manifest["expected_segments"]
                        if isinstance(entry, str) and entry.startswith(f"{stem}:s")]
        segment_ids = [seg.get("segment_id") for seg in segments if isinstance(seg, dict)]
        segment_ordinals = [seg.get("segment_ordinal") for seg in segments if isinstance(seg, dict)]
        if (len(segment_ids) != len(segments) or len(segment_ids) != len(set(segment_ids))
                or len(segment_ordinals) != len(segments) or len(segment_ordinals) != len(set(segment_ordinals))
                or any(type(ordinal) is not int or ordinal < 0
                       or sid != f"{stem}:s{ordinal}"
                       for sid, ordinal in zip(segment_ids, segment_ordinals))
                or set(segment_ids) != set(expected_ids)):
            return ["english_provenance_manifest_failed"], None
        verified_words: list[dict] = []
        seen_unit_ids: set[str] = set()
        sources: list[Path] = []
        used_source_paths: set[Path] = set()
        for segment in sorted(segments, key=lambda item: item.get("segment_ordinal", -1)):
            sid = segment.get("segment_id")
            if segment.get("status") != "verified" or sid not in global_manifest["produced_segments"]:
                return ["english_segment_rejected"], None
            source = segment.get("mfa_textgrid")
            if not isinstance(source, dict):
                return ["source_textgrid_missing"], None
            try:
                source_path = _safe_file_under(args.en_aligned_dir, source.get("path"))
            except ValueError:
                return ["source_textgrid_missing"], None
            ordinal = segment["segment_ordinal"]
            seg_name = f"{stem}_seg{ordinal}"
            allowed_source_paths = [
                args.en_aligned_dir / f"{seg_name}.TextGrid",
                args.en_aligned_dir / seg_name / f"{seg_name}.TextGrid",
            ]
            try:
                allowed_resolved = {_safe_file_under(args.en_aligned_dir, str(item))
                                    for item in allowed_source_paths if item.exists()}
            except ValueError:
                return ["source_textgrid_missing"], None
            if source_path not in allowed_resolved or len(allowed_resolved) != 1:
                return ["source_textgrid_missing"], None
            if source_path in used_source_paths:
                return ["source_textgrid_missing"], None
            if _sha256(source_path) != source.get("sha256"):
                return ["source_textgrid_hash_mismatch"], None
            source_words = _source_english_words(source_path)
            ledger_words = segment.get("words")
            if not isinstance(ledger_words, list) or len(ledger_words) != len(source_words):
                return ["english_word_unmatched"], None
            for index, (record, source_word) in enumerate(zip(ledger_words, source_words)):
                mfa_word = record.get("mfa_word")
                actual_ctc_ordinal = _actual_ctc_ordinal(record)
                expected_id = (f"{sid}:w{actual_ctc_ordinal}"
                               if actual_ctc_ordinal is not None
                               else f"{sid}:w{record.get('ctc_ordinal')}")
                if (record.get("status") != "verified" or record.get("word_id") != expected_id
                        or not isinstance(mfa_word, dict)
                        or actual_ctc_ordinal is None
                        or _compact_english(ctc_english[actual_ctc_ordinal]) != _compact_english(record.get("ctc_text", ""))
                        or _compact_english(record.get("ctc_text", "")) != _compact_english(source_word["text"])
                        or mfa_word.get("ordinal") != source_word["ordinal"]
                        or not _same_number(mfa_word.get("start"), source_word["start"])
                        or not _same_number(mfa_word.get("end"), source_word["end"])
                        or record.get("provenance") != "english_mfa_textgrid"
                        or record.get("canonical_binding") != CANONICAL_UNITS_SCHEMA
                        or not isinstance(record.get("unit_id"), str)
                        or not isinstance(record.get("alignment_token"), str)
                        or not isinstance(record.get("source_ctc_ordinals"), list)
                        or not isinstance(record.get("canonical_span"), list)
                        or len(record.get("canonical_span", [])) != 2):
                    return ["english_word_unmatched"], None
                source_ordinals = record["source_ctc_ordinals"]
                if not _strict_span_valid(record):
                    return ["english_provenance_manifest_failed"], None
                ledger_phones = record.get("phones")
                if not isinstance(ledger_phones, list) or len(ledger_phones) != len(source_word["phones"]):
                    return ["final_sequence_mismatch"], None
                for ledger_phone, source_phone in zip(ledger_phones, source_word["phones"]):
                    if (ledger_phone.get("label") != source_phone["label"]
                            or ledger_phone.get("ordinal") != source_phone["ordinal"]
                            or ledger_phone.get("mfa_phone_ordinal") != source_phone["mfa_phone_ordinal"]
                            or not _same_number(ledger_phone.get("start"), source_phone["start"])
                            or not _same_number(ledger_phone.get("end"), source_phone["end"])):
                        return ["english_provenance_hash_mismatch"], None
                pronunciation_reasons = _pronunciation_consumer_reasons(
                    record, source_word, ledger)
                if pronunciation_reasons:
                    return pronunciation_reasons, None
                verified_words.append({"ledger": record, "source": source_word,
                                       "actual_ctc_ordinal": actual_ctc_ordinal})
                seen_unit_ids.add(record["unit_id"])
            sources.append(source_path); used_source_paths.add(source_path)
        verified_words.sort(key=lambda item: item["ledger"].get("ctc_ordinal", -1))
        source_ordinals = [item.get("actual_ctc_ordinal") for item in verified_words]
        if source_ordinals != sorted(ctc_english) or len(source_ordinals) != len(ctc_english):
            return ["english_word_unmatched"], None
        try:
            authority_units = parse_english_units(reference_text) if reference_text else ()
        except (EnglishUnitError, TypeError, ValueError):
            authority_units = ()
        if reference_text:
            if len(authority_units) != len(final_words):
                return ["english_authoritative_compound_split"], None
            grouped: list[dict] = []
            record_cursor = 0
            for unit in authority_units:
                start_cursor = record_cursor
                compact = ""
                while record_cursor < len(verified_words):
                    evidence = verified_words[record_cursor]
                    source_text = str(evidence["source"].get("text", ""))
                    if not is_english_fragment_token(source_text):
                        break
                    compact += re.sub(r"[^a-z0-9]", "", source_text.casefold())
                    if compact != re.sub(r"[^a-z0-9]", "", unit.surface_text.casefold())[:len(compact)]:
                        return ["english_authoritative_compound_split"], None
                    record_cursor += 1
                    if compact == re.sub(r"[^a-z0-9]", "", unit.surface_text.casefold()):
                        break
                members = verified_words[start_cursor:record_cursor]
                if (not members
                        or compact != re.sub(r"[^a-z0-9]", "", unit.surface_text.casefold())):
                    return ["english_authoritative_compound_split"], None
                if any(item["ledger"].get("unit_id") != unit.unit_id
                       or item["ledger"].get("alignment_token") != unit.alignment_token
                       for item in members):
                    return ["english_unit_owner_mismatch"], None
                ordinals = [ordinal for item in members
                            for ordinal in item["ledger"].get("source_ctc_ordinals", [])]
                if not ordinals:
                    return ["english_authoritative_compound_split"], None
                combined = deepcopy(members[0]["ledger"])
                combined["ctc_text"] = unit.surface_text
                combined["unit_id"] = unit.unit_id
                combined["alignment_token"] = unit.alignment_token
                combined["source_ctc_ordinals"] = ordinals
                combined["ctc_ordinal"] = ordinals[0]
                combined["canonical_span"] = [
                    members[0]["ledger"]["canonical_span"][0],
                    members[-1]["ledger"]["canonical_span"][1]]
                if not _strict_span_valid(combined):
                    return ["english_authoritative_compound_split"], None
                combined["mfa_word"] = deepcopy(members[0]["ledger"]["mfa_word"])
                combined["mfa_word"]["text"] = unit.alignment_token
                combined["mfa_word"]["start"] = members[0]["ledger"]["mfa_word"]["start"]
                combined["mfa_word"]["end"] = members[-1]["ledger"]["mfa_word"]["end"]
                phones = []
                for member in members:
                    phones.extend(deepcopy(member["ledger"].get("phones", [])))
                for ordinal, phone in enumerate(phones):
                    phone["ordinal"] = ordinal
                    phone["mfa_phone_ordinal"] = ordinal
                combined["phones"] = phones
                source = deepcopy(members[0]["source"])
                source["text"] = unit.surface_text
                source["start"] = members[0]["source"]["start"]
                source["end"] = members[-1]["source"]["end"]
                source_phones = []
                for member in members:
                    source_phones.extend(deepcopy(member["source"].get("phones", [])))
                for ordinal, phone in enumerate(source_phones):
                    phone["ordinal"] = ordinal
                source["phones"] = source_phones
                grouped.append({"ledger": combined, "source": source})
            if record_cursor != len(verified_words):
                return ["english_authoritative_compound_split"], None
            verified_words = grouped
        elif len(verified_words) != len(final_words):
            return ["english_word_count_mismatch"], None
        final_phones = final_tg.tiers[4].intervals
        matched_en_phone_indices: set[int] = set()
        for position, (final_word, evidence) in enumerate(zip(final_words, verified_words)):
            record, source_word = evidence["ledger"], evidence["source"]
            ordinal = record.get("ctc_ordinal")
            if authority_units:
                unit = authority_units[position]
                if (record.get("unit_id") != unit.unit_id
                        or record.get("alignment_token") != unit.alignment_token
                        or final_word.text.strip() != unit.surface_text):
                    return ["english_unit_owner_mismatch"], None
                hanzi_owners = [iv for iv in final_tg.tiers[2].intervals
                                if is_english_token(iv.text.strip())
                                and abs(iv.xmin - final_word.xmin) <= EPS
                                and abs(iv.xmax - final_word.xmax) <= EPS]
                if len(hanzi_owners) != 1 or hanzi_owners[0].text.strip() != unit.surface_text:
                    return ["english_hanzi_owner_mismatch"], None
            if (re.sub(r"[^a-z0-9]", "", final_word.text.strip().casefold())
                    != re.sub(r"[^a-z0-9]", "", record.get("alignment_token", "").casefold())):
                return ["english_word_unmatched"], None
            # Every positive-overlap phone inside an English word is part of
            # its evidence sequence.  Silence cannot be smuggled into the
            # word, and no phone may cross the word boundary.
            matching = [(index, phone) for index, phone in enumerate(final_phones)
                        if phone.xmax > final_word.xmin and phone.xmin < final_word.xmax]
            if any(phone.xmin < final_word.xmin or phone.xmax > final_word.xmax
                   for _, phone in matching):
                return ["final_sequence_mismatch"], None
            expected = source_word["phones"]
            if not matching:
                return ["english_phone_empty"], None
            if len(matching) != len(expected):
                return ["final_sequence_mismatch"], None
            for index, phone_and_original in enumerate(zip(matching, expected)):
                phone_index, phone = phone_and_original[0]
                original = phone_and_original[1]
                if phone.text.strip() != f"en:{original['label']}":
                    return ["fallback_forbidden"], None
                matched_en_phone_indices.add(phone_index)
                denominator = source_word["end"] - source_word["start"]
                left = final_word.xmin + (original["start"] - source_word["start"]) / denominator * (final_word.xmax - final_word.xmin)
                right = final_word.xmin + (original["end"] - source_word["start"]) / denominator * (final_word.xmax - final_word.xmin)
                if abs(phone.xmin - left) > EPS or abs(phone.xmax - right) > EPS:
                    return ["final_timing_mismatch"], None
        all_en_phone_indices = {index for index, phone in enumerate(final_phones)
                                if phone.text.strip().startswith("en:")}
        if all_en_phone_indices != matched_en_phone_indices:
            return ["final_sequence_mismatch"], None
        return [], {"ledger": ledger_path, "sources": sources}
    except FileNotFoundError:
        return ["source_textgrid_missing"], None
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        text = str(exc)
        if "english_phone_empty" in text:
            return ["english_phone_empty"], None
        if "english_phone_unknown" in text:
            return ["english_phone_unknown"], None
        if "source_interval_invalid" in text:
            return ["source_interval_invalid"], None
        return ["english_provenance_manifest_failed"], None


def _copy_english_evidence(stem: str, evidence: dict, output_dir: Path,
                           staging_root: Path) -> dict:
    """Stage ordinary copies; only the caller may atomically publish the run."""
    final_root = output_dir / "_provenance" / "english"
    final_base = final_root / stem
    staged_base = staging_root / stem
    if final_base.exists() or staged_base.exists():
        raise ValueError("evidence collision")
    try:
        staged_base.mkdir()
        ledger_dest = staged_base / "ledger.json"
        shutil.copyfile(evidence["ledger"], ledger_dest)
        if ledger_dest.is_symlink() or not ledger_dest.is_file():
            raise ValueError("ledger evidence is not regular")
        source_entries: list[dict] = []
        sources_dir = staged_base / "sources"; sources_dir.mkdir()
        for ordinal, source in enumerate(evidence["sources"]):
            dest = sources_dir / f"{ordinal:03d}_{source.name}"
            shutil.copyfile(source, dest)
            if dest.is_symlink() or not dest.is_file():
                raise ValueError("source evidence is not regular")
            source_entries.append({"path": str((final_base / "sources" / dest.name).relative_to(output_dir)),
                                   "sha256": _sha256(dest)})
        copied = {"schema": EN_PROVENANCE_SCHEMA,
                  "ledger": {"path": str((final_base / "ledger.json").relative_to(output_dir)), "sha256": _sha256(ledger_dest)},
                  "source_textgrids": source_entries}
        if not copied["ledger"]["sha256"] or any(not item["sha256"] for item in source_entries):
            raise ValueError("evidence hash failure")
        return copied
    except Exception:
        shutil.rmtree(staged_base, ignore_errors=True)
        raise


def _evidence_recheck(manifest: dict, output_dir: Path) -> list[str]:
    """Re-hash copied evidence immediately before its manifest becomes visible."""
    failures: list[str] = []
    for entry in manifest.get("ok", []):
        evidence = entry.get("english_provenance")
        if evidence is None:
            continue
        records = [evidence.get("ledger")] + list(evidence.get("source_textgrids", []))
        for record in records:
            if not isinstance(record, dict):
                failures.append("english_provenance_evidence_invalid")
                continue
            try:
                relative = Path(record["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe evidence path")
                candidate = output_dir / relative
                if candidate.is_symlink():
                    raise ValueError("evidence symlink")
                path = _safe_file_under(output_dir, str(candidate))
                if _sha256(path) != record.get("sha256"):
                    failures.append("english_provenance_evidence_hash_mismatch")
            except (KeyError, OSError, ValueError):
                failures.append("english_provenance_evidence_missing")
    return sorted(set(failures))


def _cleanup_evidence_staging(path: Path) -> None:
    """Remove only this run's staging tree and an empty parent we created."""
    shutil.rmtree(path, ignore_errors=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _load_pipeline_receipt(path: Path) -> tuple[dict | None, list[str]]:
    """Read the frozen v2 source-denominator receipt for this strict run."""
    try:
        receipt = read_pipeline_accounting_receipt(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"pipeline_accounting_receipt_failed:{exc}"]
    errors = validate_pipeline_accounting_receipt(receipt)
    if errors:
        return None, [f"pipeline_accounting_receipt_invalid:{error}" for error in errors]
    if receipt.get("schema") != PIPELINE_ACCOUNTING_SCHEMA:
        return None, ["pipeline_accounting_receipt_schema_mismatch"]
    if receipt.get("mode") != "strict_replay":
        extra = receipt.get("extra", {})
        if (isinstance(extra, dict) and
                any(key in extra for key in ("strict_replay_receipt", "strict_replay_evidence"))):
            return None, ["production receipt carries strict_replay schema/bindings"]
    return receipt, []


def _replay_cli_binding_reasons(args: argparse.Namespace, receipt: dict,
                                output_dir: Path) -> list[str]:
    """Validate explicit replay evidence paths and their DAG bindings.

    Replay is the only route allowed to consume ``strict_replay_english_import``.
    Every path is supplied by the runner and must be an ordinary file at the
    exact role location; no sibling/derived path discovery is permitted.
    """
    errors: list[str] = []
    raw_eng = getattr(args, "strict_replay_english_import", None)
    raw_manifest = getattr(args, "strict_replay_english_manifest", None)
    raw_formal = getattr(args, "strict_replay_formal_receipt", None)
    raw_immutable = getattr(args, "strict_replay_immutable_import", None)
    raw_report = getattr(args, "strict_replay_postprocess_report", None)
    raw_subset = getattr(args, "strict_replay_english_subset", None)
    raw_subset_hash = getattr(args, "strict_replay_english_subset_sha256", None)
    raw_parent_hash = getattr(args, "strict_replay_parent_english_sha256", None)
    values = ((raw_eng, "English import"), (raw_manifest, "English manifest"),
              (raw_formal, "formal receipt"), (raw_immutable, "immutable import"),
              (raw_report, "postprocess report"), (raw_subset, "English subset"))
    if any(not isinstance(value, Path) for value, _ in values):
        return ["strict_replay explicit evidence paths missing"]
    def ordinary(value: Path, label: str) -> Path | None:
        if not value.is_absolute() or ".." in value.parts:
            errors.append(f"strict_replay {label} path is not normalized absolute")
            return None
        if value.is_symlink() or not value.is_file():
            errors.append(f"strict_replay {label} is missing/non-regular")
            return None
        try:
            resolved = value.resolve(strict=True)
        except OSError:
            errors.append(f"strict_replay {label} cannot resolve")
            return None
        if resolved != value:
            errors.append(f"strict_replay {label} lexical/real path alias")
        return resolved
    eng = ordinary(raw_eng, "English import")
    manifest = ordinary(raw_manifest, "English manifest")
    formal = ordinary(raw_formal, "formal receipt")
    immutable = ordinary(raw_immutable, "immutable import")
    report = ordinary(raw_report, "postprocess report")
    subset = ordinary(raw_subset, "English subset")
    expected_formal = output_dir / ".pipeline_run_receipt_v2.json"
    if formal is not None and formal != expected_formal.resolve():
        errors.append("strict_replay formal receipt path mismatch")
    if report is not None and report != (output_dir / "postprocess_report.jsonl").resolve():
        errors.append("strict_replay postprocess report path mismatch")
    if immutable is None or eng is None:
        return errors
    workspace = immutable.parent
    if immutable.name != "strict_replay_import.json":
        errors.append("strict_replay immutable import basename mismatch")
    if immutable != workspace / "strict_replay_import.json":
        errors.append("strict_replay immutable import parent mismatch")
    expected_eng = workspace / "strict_replay_english_import.json"
    if eng != expected_eng:
        errors.append("strict_replay English import exact workspace path mismatch")
    expected_subset = workspace / "strict_replay_english_alignment_subset.json"
    if subset != expected_subset:
        errors.append("strict_replay English subset exact workspace path mismatch")
    if not isinstance(raw_subset_hash, str) or subset is None or raw_subset_hash != _sha256(subset):
        errors.append("strict_replay English subset CLI hash mismatch")
    if not isinstance(raw_parent_hash, str) or manifest is None or raw_parent_hash != _sha256(manifest):
        errors.append("strict_replay parent English CLI hash mismatch")
    if manifest is not None and not (workspace in manifest.parents):
        errors.append("strict_replay English manifest escapes workspace")
    try:
        import_payload = json.loads(immutable.read_text(encoding="utf-8"))
        paths = import_payload.get("paths", {})
        if import_payload.get("schema") != STRICT_REPLAY_SCHEMA:
            errors.append("strict_replay immutable import schema mismatch")
        if paths.get("workspace") != str(workspace) or paths.get("immutable_import") != str(immutable):
            errors.append("strict_replay immutable CLI/import binding mismatch")
        if paths.get("output") != str(output_dir.resolve()):
            errors.append("strict_replay immutable output binding mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("strict_replay immutable import unreadable")
    try:
        formal_payload = json.loads(formal.read_text(encoding="utf-8")) if formal else {}
        extra = formal_payload.get("extra", {})
        evidence = extra.get("strict_replay_evidence", {})
        if formal_payload.get("mode") != "strict_replay":
            errors.append("strict_replay formal receipt mode mismatch")
        if extra.get("strict_replay_receipt") != str(immutable):
            errors.append("strict_replay formal/import binding mismatch")
        if evidence.get("import_manifest") != str(immutable):
            errors.append("strict_replay formal immutable evidence mismatch")
        if evidence.get("english_import") != str(eng):
            errors.append("strict_replay formal/English binding mismatch")
        if evidence.get("english_sha256") != _sha256(eng):
            errors.append("strict_replay formal English hash mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("strict_replay formal receipt unreadable")
    return errors


def _formal_post_accounting_reasons(receipt: dict, output_dir: Path,
                                    filtered_dir: Path,
                                    report_rows: dict[str, dict]) -> list[str]:
    """Check the pre-isolation post-stage conservation contract."""
    errors: list[str] = []
    eligible_raw = receipt.get("eligible", {})
    output_raw = receipt.get("output", {})
    filtered_raw = receipt.get("filtered", {})
    eligible = set(eligible_raw.get("stems", [])) if isinstance(eligible_raw, dict) else set()
    formal_output = set(output_raw.get("stems", [])) if isinstance(output_raw, dict) else set()
    formal_filtered = set(filtered_raw.get("stems", [])) if isinstance(filtered_raw, dict) else set()
    if receipt.get("paths", {}).get("output") != str(output_dir.resolve()):
        errors.append("formal receipt output path binding mismatch")
    if receipt.get("paths", {}).get("filtered") != str(filtered_dir.resolve()):
        errors.append("formal receipt filtered path binding mismatch")
    if receipt.get("mode") == "strict_replay" and receipt.get("paths", {}).get("report") != str(
            (output_dir / "postprocess_report.jsonl").resolve()):
        errors.append("formal receipt report path binding mismatch")
    if formal_output & formal_filtered or formal_output | formal_filtered != eligible:
        errors.append("formal eligible/output/filtered conservation mismatch")
    if set(report_rows) != eligible:
        errors.append("formal report membership mismatch")
    for stem, row in report_rows.items():
        status = row.get("status")
        raw_path = row.get("output")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
            errors.append(f"report output path unsafe:{stem}")
            continue
        expected_dir = output_dir if stem in formal_output else filtered_dir
        expected_path = expected_dir / f"{stem}.TextGrid"
        if Path(raw_path).resolve() != expected_path.resolve():
            errors.append(f"report output path/formal set mismatch:{stem}")
        if stem in formal_output and status != "ok":
            errors.append(f"report output status mismatch:{stem}")
        if stem in formal_filtered and status == "ok":
            errors.append(f"report filtered status mismatch:{stem}")
    return errors


def _strict_replay_receipt_reasons(receipt: dict, output_dir: Path,
                                   filtered_dir: Path,
                                   report_rows: dict[str, dict]) -> list[str]:
    """Validate replay-only slot/accounting invariants.

    This branch is deliberately gated by ``mode == strict_replay`` at the
    caller.  The normal strict-ok audit therefore retains its historical v2
    contract while replay receipts receive the stronger canonical-subset
    checks required by S0.1.
    """
    errors: list[str] = []
    if receipt.get("mode") != "strict_replay":
        return errors
    if receipt.get("schema") != PIPELINE_ACCOUNTING_SCHEMA:
        errors.append("strict_replay_accounting_schema_mismatch")
    binding = receipt.get("extra", {}).get("strict_replay_receipt")
    if not isinstance(binding, str) or not binding:
        errors.append("strict_replay_import_receipt_binding_missing")
        return errors
    import_path = Path(binding)
    if import_path.is_symlink() or not import_path.is_file():
        errors.append("strict_replay_import_receipt_missing")
        return errors
    try:
        import_payload_paths = None
        # The immutable import is owned by workspace; the formal receipt owns
        # output/.pipeline_run_receipt_v2.json.  Never treat output as the
        # import's parent (that was the obsolete accounting contract).
        sidecar = import_path.parent / "strict_replay_import.sha256"
        if sidecar.is_file() and sidecar.read_text(encoding="ascii").strip() != _sha256(import_path):
            errors.append("strict_replay_import_receipt_sidecar_hash_mismatch")
    except (OSError, UnicodeError):
        errors.append("strict_replay_import_receipt_sidecar_unreadable")
    try:
        import_payload = json.loads(import_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("strict_replay_import_receipt_unreadable")
        return errors
    if not isinstance(import_payload, dict) or import_payload.get("schema") != STRICT_REPLAY_SCHEMA:
        errors.append("strict_replay_import_receipt_schema_mismatch")
        return errors
    paths = import_payload.get("paths", {})
    if (not isinstance(paths, dict) or paths.get("immutable_import") != str(import_path)
            or paths.get("output") != str(output_dir.resolve())):
        errors.append("strict_replay_import_payload_path_binding_mismatch")
    elif paths.get("workspace") != str(import_path.parent):
        errors.append("strict_replay_import_payload_workspace_binding_mismatch")

    canonical = import_payload.get("canonical")
    if not isinstance(canonical, dict) or canonical.get("schema") != STRICT_REPLAY_CANONICAL_SCHEMA:
        errors.append("strict_replay_canonical_schema_mismatch")
    cpath_raw = canonical.get("path") if isinstance(canonical, dict) else None
    try:
        canonical_path = Path(cpath_raw) if isinstance(cpath_raw, str) else Path("")
        canonical_hash = canonical.get("sha256") if isinstance(canonical, dict) else None
        if (not canonical_path.is_file() or canonical_path.is_symlink()
                or canonical_hash != STRICT_REPLAY_CANONICAL_SHA256
                or _sha256(canonical_path) != STRICT_REPLAY_CANONICAL_SHA256):
            errors.append("strict_replay_canonical_hash_mismatch")
        cdata = json.loads(canonical_path.read_text(encoding="utf-8"))
        centries = cdata.get("entries", [])
        if cdata.get("count") != 96 or not isinstance(centries, list) or len(centries) != 96:
            errors.append("strict_replay_canonical_slot_count_mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("strict_replay_canonical_unreadable")
        centries = []

    # v2.1 exposes exactly one canonical slot identity vector.  Do not fall
    # back to the pre-v2.1 ``slots`` field: a slots-only receipt is a legacy
    # negative case, never an active strict-replay input.
    slots = import_payload.get("slot_stem_mapping")
    if (not isinstance(slots, list) or len(slots) != 96
            or any(not isinstance(row, dict)
                   or set(row) != {"slot", "stem"}
                   for row in slots)):
        errors.append("strict_replay_slots_incomplete")
        slots = []
    canonical_stems = [row.get("stem") for row in centries if isinstance(row, dict)]
    slot_stems = [row.get("stem") for row in slots if isinstance(row, dict)]
    if len(slot_stems) != 96 or slot_stems != canonical_stems:
        errors.append("strict_replay_slot_mapping_mismatch")
    slot_ids = [row.get("slot") for row in slots if isinstance(row, dict)]
    if any(not isinstance(slot, int) for slot in slot_ids) or len(slot_ids) != len(set(slot_ids)):
        errors.append("strict_replay_slot_duplicate")

    selected = import_payload.get("selection_slot_records")
    # A scalar selected_slots=24 is not a slot mapping and is explicitly
    # rejected; only the canonical 24-slot pilot (or explicit full 96 run) is
    # accepted in strict_replay scope.
    if not isinstance(selected, list) or len(selected) not in (24, 96):
        errors.append("strict_replay_selected_slots_not_canonical_pilot")
        selected = []
    selected_pairs = [(row.get("slot"), row.get("stem")) for row in selected
                      if isinstance(row, dict)]
    canonical_pairs = {(row.get("slot"), row.get("stem")) for row in slots
                       if isinstance(row, dict)}
    if len(selected_pairs) != len(set(selected_pairs)):
        errors.append("strict_replay_selected_slot_duplicate")
    if any(pair not in canonical_pairs for pair in selected_pairs):
        errors.append("strict_replay_selected_slot_outside_canonical")
    if import_payload.get("source_manifest_slots") != 96:
        if import_payload.get("source_count") != 21 or import_payload.get("eligible_count") != 18 or import_payload.get("excluded_count") != 3:
            errors.append("strict_replay_source_manifest_slots_not_96")
    if import_payload.get("selection_slot_count") != len(selected) or import_payload.get("selection_slot_digest") != stable_json_digest(selected):
        errors.append("strict_replay_selected_slot_count_mismatch")
    if len(selected) == 24:
        pilot = import_payload.get("pilot_selector", {})
        if (import_payload.get("pilot_selector_version") != "strict-replay-selector-v1"
                or pilot.get("pilot") is not True):
            errors.append("strict_replay_pilot_selector_missing")
        if len({(row.get("category"), row.get("range")) for row in selected
                if isinstance(row, dict)}) != 24:
            errors.append("strict_replay_pilot_category_range_duplicate")

    selected_stems = {stem for _, stem in selected_pairs}
    assets = import_payload.get("assets")
    asset_rows = assets if isinstance(assets, list) else []
    asset_stems = [row.get("stem") for row in asset_rows if isinstance(row, dict)]
    if not isinstance(assets, list) or len(asset_stems) != len(set(asset_stems)) or set(asset_stems) != selected_stems:
        errors.append("strict_replay_asset_membership_mismatch")
    # Taxonomy is advisory evidence only when present, but malformed or
    # contradictory primary/secondary reasons must fail closed.  In
    # particular, ``recovered`` can never stand in for missing MFA alignment.
    for reason_row in [*asset_rows, *report_rows.values()]:
        if not isinstance(reason_row, dict):
            continue
        primary = reason_row.get("primary_reason")
        secondary = reason_row.get("secondary_reasons", reason_row.get("secondary_reason", []))
        if primary is not None and (not isinstance(primary, str) or not primary):
            errors.append("strict_replay_primary_reason_invalid")
        if isinstance(secondary, str):
            secondary = [secondary]
        if secondary is not None and (not isinstance(secondary, list)
                                      or any(not isinstance(item, str) or not item for item in secondary)):
            errors.append("strict_replay_secondary_reasons_invalid")
        secondary_values = secondary if isinstance(secondary, list) else []
        if isinstance(primary, str) and primary in set(secondary_values):
            errors.append("strict_replay_primary_secondary_reason_overlap")
        if primary == "recovered" or "recovered" in set(secondary_values):
            errors.append("strict_replay_missing_marked_recovered")
    missing = import_payload.get("missing_mfa_alignment", [])
    missing_valid = (isinstance(missing, list)
                     and all(isinstance(item, str) for item in missing))
    if (not missing_valid or len(missing) != len(set(missing))
            or not set(missing) <= selected_stems):
        errors.append("strict_replay_missing_alignment_membership_mismatch")
        missing = []
    exclusions = receipt.get("exclusions", [])
    exclusion_stems = set()
    for row in exclusions if isinstance(exclusions, list) else []:
        if not isinstance(row, dict) or row.get("reason") == "recovered":
            errors.append("strict_replay_missing_marked_recovered")
            continue
        stem = row.get("stem")
        exclusion_stems.add(stem)
        if stem in set(missing) and row.get("reason") != "missing_mfa_alignment":
            errors.append("strict_replay_missing_reason_mismatch")
    if exclusion_stems != set(missing):
        errors.append("strict_replay_aligned_missing_accounting_mismatch")

    report = import_payload.get("report", {})
    if (not isinstance(report, dict)
            or report.get("source") != len(selected_stems)
            or report.get("eligible") != len(selected_stems) - len(set(missing))):
        errors.append("strict_replay_report_pre_summary_mismatch")
    # Import is pre-stage evidence: output/filtered must be scalar zeroes and
    # are never treated as a second post-stage conservation root.
    if (type(report.get("output")) is not int or type(report.get("filtered")) is not int
            or report.get("output") != 0 or report.get("filtered") != 0):
        errors.append("strict_replay_import_report_not_zero_pre_summary")
    stages = import_payload.get("stages", [])
    if not isinstance(stages, list) or any(
            not isinstance(stage, dict) or stage.get("return_code") != 0
            or stage.get("reasons") for stage in stages):
        errors.append("strict_replay_stage_reasons_nonempty")
    if import_payload.get("global_reasons"):
        errors.append("strict_replay_global_reasons_nonempty")
    # Report rows and output/filtered names are checked against the accounting
    # denominator here, without interpreting their reason taxonomy.
    if set(report_rows) != selected_stems - set(missing):
        errors.append("strict_replay_report_membership_mismatch")
    output_names = {path.stem for path in output_dir.glob("*.TextGrid")}
    filtered_names = {path.stem for path in filtered_dir.glob("*.TextGrid")}
    eligible = selected_stems - set(missing)
    if output_names & filtered_names or output_names | filtered_names != eligible:
        errors.append("strict_replay_output_filtered_not_conserved")
    return sorted(set(errors))


def audit(args: argparse.Namespace) -> tuple[dict, bool]:
    ctc_dir = args.ctc_dir
    reference_mode_policy = getattr(args, "reference_mode", "auto")
    receipt_path = getattr(args, "pipeline_receipt", None)
    if receipt_path is None:
        # Receipt discovery from ctc/workspace siblings is forbidden.  The
        # runner must pass the exact formal receipt path explicitly.
        receipt_path = Path("/") / ".missing_pipeline_run_receipt_v2.json"
    pipeline_receipt, receipt_reasons = _load_pipeline_receipt(Path(receipt_path))
    ctc_stems = {path.stem for path in ctc_dir.glob("*.lab")}
    expected = (set(pipeline_receipt["eligible"]["stems"])
                if pipeline_receipt is not None else set(ctc_stems))
    axis_global_reasons, axis_stem_reasons = _axis_contract_reasons(args, expected)
    global_reasons = list(axis_global_reasons)
    lifecycle_reasons, lifecycle = _ctc_lifecycle_reasons(args, expected)
    global_reasons.extend(lifecycle_reasons)
    if lifecycle_reasons:
        axis_stem_reasons = {
            stem: sorted(set(axis_stem_reasons.get(stem, [])
                             + ["ctc_lifecycle_invalid"]))
            for stem in expected
        }
    if axis_global_reasons:
        # Infrastructure-invalid receipts must not leave a publication
        # candidate behind.  Mark every expected stem for isolation below.
        axis_stem_reasons = {
            stem: sorted(set(axis_stem_reasons.get(stem, [])
                             + ["axis_contract_invalid"]))
            for stem in expected
        }
    output = {path.stem: path for path in args.output_dir.glob("*.TextGrid")}
    filtered = {path.stem: path for path in args.filtered_dir.glob("*.TextGrid")}
    report_rows, report_reasons = _report_index(args.report)
    global_reasons.extend(report_reasons)
    # A v2 receipt is the sole authority for a partial MFA partition.  Its
    # missing rows must be represented by the runner's filtered placeholder
    # and report ledger; otherwise a stem could vanish between MFA and audit.
    receipt_missing = {stem for stem, reasons in axis_stem_reasons.items()
                       if "missing_mfa_alignment" in reasons}
    for stem in sorted(receipt_missing):
        row = report_rows.get(stem, {})
        row_reasons = row.get("filter_reasons", []) if isinstance(row, dict) else []
        if (stem in output or stem not in filtered
                or not isinstance(row_reasons, list)
                or "missing_mfa_alignment" not in row_reasons
                or row.get("status") != "filtered_missing_mfa_alignment"):
            global_reasons.append(f"mfa_alignment_missing_ledger_mismatch:{stem}")
    reference_index, reference_errors = _reference_index(args.reference_dir, expected)
    if reference_mode_policy == "fallback":
        # The batch policy, not incidental files in reference_dir, decides
        # authority.  This keeps an ASR-only audit isolated from stale
        # reference TXT files copied into the same source tree.
        reference_index = {}
        reference_errors = []
    global_reasons.extend(reference_errors)
    global_reasons.extend(receipt_reasons)
    if pipeline_receipt is not None:
        global_reasons.extend(_formal_post_accounting_reasons(
            pipeline_receipt, args.output_dir, args.filtered_dir, report_rows))
    if pipeline_receipt is not None:
        for exclusion in pipeline_receipt.get("exclusions", []):
            if isinstance(exclusion, dict) and exclusion.get("reason") == "recovered":
                global_reasons.append("missing_alignment_marked_recovered")
    # Replay receipts carry a canonical 96-slot identity while this audit may
    # process only its selected 24-slot pilot.  Apply the stronger receipt
    # contract only to that explicit mode; ordinary strict-ok runs remain on
    # the production v2 accounting rules above.
    if pipeline_receipt is not None and pipeline_receipt.get("mode") == "strict_replay":
        global_reasons.extend(_replay_cli_binding_reasons(args, pipeline_receipt, args.output_dir))
        global_reasons.extend(_strict_replay_receipt_reasons(
            pipeline_receipt, args.output_dir, args.filtered_dir, report_rows))
        english_import = getattr(args, "strict_replay_english_import", None)
        if english_import is not None:
            try:
                from verify_strict_replay_english_subset import verify_english_import_active
                global_reasons.extend(verify_english_import_active(
                    Path(english_import),
                    replay_path=Path(getattr(args, "strict_replay_immutable_import")),
                    formal_path=Path(getattr(args, "strict_replay_formal_receipt")),
                    subset_path=Path(getattr(args, "strict_replay_english_subset")),
                    parent_path=Path(getattr(args, "en_manifest")),
                    require_final=False,
                    config_path=getattr(args, "config", None),
                    dictionary_path=getattr(args, "mfa_en_dictionary", None)))
            except (ImportError, OSError, ValueError, TypeError) as exc:
                global_reasons.append(f"strict_replay_english_import_verifier_failed:{exc}")
    if pipeline_receipt is not None:
        # CTC is produced on the full immutable MFA axis.  The accounting
        # receipt's eligible set can be smaller when MFA explicitly reports
        # missing alignments, so compare CTC against the axis set rather than
        # incorrectly requiring it to equal the post-MFA eligible subset.
        axis_stems: set[str] = set()
        try:
            axis_payload = json.loads(Path(args.mfa_input_axis_receipt).read_text(encoding="utf-8"))
            axis_stems = set(axis_payload.get("stems", []))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        if (axis_stems and ctc_stems != axis_stems) or (not axis_stems and ctc_stems != expected):
            global_reasons.append("ctc_eligible_membership_mismatch")
    _replay_mode = pipeline_receipt is not None and pipeline_receipt.get("mode") == "strict_replay"
    # Replay preserves the parent-global manifest byte-for-byte but uses the
    # separately verified selected-stem subset for denominator/provenance.
    # Never apply production global counts to the pilot subset.
    if _replay_mode:
        english_manifest, english_global_reasons = None, []
    else:
        english_manifest, english_global_reasons = _load_english_manifest(args)
    global_reasons.extend(english_global_reasons)
    evidence_final_root = args.output_dir / "_provenance" / "english"
    evidence_staging_root: Path | None = None
    # A strict run never merges evidence from an older run.  This applies to
    # pure-Chinese runs too: otherwise an old proof tree could silently travel
    # with a new result manifest.
    if evidence_final_root.exists() or evidence_final_root.is_symlink():
        global_reasons.append("english_provenance_evidence_collision")
    manifest: dict = {
        "policy_version": POLICY_VERSION,
        "english_provenance_policy": {"schema": EN_PROVENANCE_SCHEMA, "required": True,
                                        "evidence_root": "_provenance/english"},
        "checks": {"executed": ["textgrid", "wav_duration", "reference_authority", "ctc_bundle", "aligned_unknown_spn", "english_mfa_source_provenance", "report_veto", "set_conservation"],
                   "not_evaluated": ["subjective_acoustic_naturalness"]},
        "output_dir": str(args.output_dir.resolve()),
        "filtered_dir": str(args.filtered_dir.resolve()),
        "pipeline_accounting_receipt": {
            "path": str(Path(receipt_path).resolve()),
            "sha256": _sha256(Path(receipt_path)) if Path(receipt_path).is_file() else "",
            "schema": PIPELINE_ACCOUNTING_SCHEMA,
        },
        "pipeline_accounting": (pipeline_receipt.get("derived", {})
                                 if pipeline_receipt is not None else {}),
        "postprocess_report": {
            "path": str(args.report.resolve()),
            "sha256": _sha256(args.report) if args.report.is_file() else "",
        },
        "ctc_lifecycle": lifecycle,
        "expected_stems": sorted(expected),
        "reference_mode_policy": reference_mode_policy,
        "ok": [],
        "rejected": {},
    }
    if pipeline_receipt is not None and pipeline_receipt.get("mode") == "strict_replay":
        english_import_path = getattr(args, "strict_replay_english_import", None)
        if english_import_path is not None:
            english_import_path = Path(english_import_path)
            subset_arg = getattr(args, "strict_replay_english_subset", None)
            parent_arg = getattr(args, "en_manifest", None)
            manifest["strict_replay_evidence"] = {
                "formal_receipt": {"path": str(Path(receipt_path).resolve()),
                                   "sha256": (_sha256(Path(receipt_path))
                                              if Path(receipt_path).is_file() else "")},
                "english_import": {"path": str(english_import_path.resolve()),
                                    "sha256": (_sha256(english_import_path)
                                               if english_import_path.is_file() else "")},
                "english_subset": {"path": str(Path(subset_arg).resolve()) if isinstance(subset_arg, Path) else "",
                                    "sha256": (_sha256(subset_arg)
                                               if isinstance(subset_arg, Path) and subset_arg.is_file() else "")},
                "parent_english_manifest": {"path": str(Path(parent_arg).resolve()) if isinstance(parent_arg, Path) else "",
                                             "sha256": (_sha256(parent_arg)
                                                        if isinstance(parent_arg, Path) and parent_arg.is_file() else "")},
            }
    overlap = set(output) & set(filtered)
    if overlap:
        global_reasons.append("output_filtered_overlap")
    if set(output) | set(filtered) != expected:
        global_reasons.append("output_filtered_expected_not_conserved")
    if set(report_rows) != expected:
        global_reasons.append("report_expected_not_conserved")

    for stem in sorted(expected):
        reasons: list[str] = list(axis_stem_reasons.get(stem, ()))
        candidate = output.get(stem)
        if candidate is None:
            # Preserve manifest set accounting when an earlier stage already
            # isolated this expected stem; output must still equal manifest ok.
            manifest["rejected"][stem] = ["preexisting_filtered_candidate"]
            continue
        reference_path = reference_index.get(stem)
        reference_original = (reference_path.read_text(encoding="utf-8").strip()
                              if reference_path is not None else "")
        reference = reference_original
        if reference_path is not None:
            # The postprocessor projects authority references after the
            # target1/target2 -> target一/target二 normalization step.  The
            # independent disk audit must consume the same canonical
            # semantic stream; otherwise a correct final ``一``/``二`` is
            # compared against the raw ASCII suffix and rejected.
            if reference_mode_policy != "fallback":
                reference = normalize_authority_reference_numerals(reference)
            # Match postprocess's in-memory authority canonicalization.  The
            # source file remains untouched; only audit comparisons use the
            # hyphenless lexical projection.
            reference = _canonicalize_reference_hyphens(reference)
        row = report_rows.get(stem)
        reference_authoritative = (
            reference_mode_policy == "authority"
            or (reference_mode_policy == "auto" and reference_path is not None)
        )
        if reference_mode_policy == "authority" and reference_path is None:
            reasons.append("authority_reference_missing")
        if reference_authoritative:
            # A legacy authority report has no explicit mode fields.  New
            # reports must still agree with the disk-selected authority.
            if isinstance(row, dict):
                if row.get("reference_mode") not in (None, "authority"):
                    reasons.append("authority_mode_report_mismatch")
                if row.get("reference_source") in {"asr_fallback", "lab_fallback"}:
                    reasons.append("authority_source_report_mismatch")
                if row.get("reference_text_authoritative") not in (None, True):
                    reasons.append("authority_flag_report_mismatch")
                if row.get("fallback_transcript"):
                    reasons.append("authority_fallback_evidence_present")
        else:
            reasons.extend(_fallback_contract_reasons(stem, row, reference_path, ctc_dir))
        wav = args.wav_dir / f"{stem}.wav"
        aligned = args.aligned_dir / f"{stem}.TextGrid"
        if not wav.is_file():
            reasons.append("missing_wav")
        if not aligned.is_file():
            reasons.append("missing_aligned")
        tg = None
        provenance_evidence = None
        provenance_reasons: list[str] = []
        if not reasons:
            try:
                tg = _strict_parse(candidate)
                reasons.extend(_numeric_reasons(tg, _wav_duration(wav)))
                reasons.extend(_sp1_reasons(tg))
                reasons.extend(_content_reasons(
                    tg, reference, reference_authoritative=reference_authoritative))
                if _replay_mode:
                    provenance_reasons, provenance_evidence = [], None
                else:
                    provenance_reasons, provenance_evidence = _english_provenance_reasons(
                        stem, tg, ctc_dir, args, english_manifest,
                        # Use the normalized authority text, but preserve
                        # lexical hyphens for English surface identity.  The
                        # ``reference`` variable above is additionally
                        # hyphen-canonicalized for semantic comparison; using
                        # it here would turn ``V-Up`` into ``VUp`` and create
                        # a false English-unit owner mismatch.
                        (normalize_authority_reference_numerals(reference_original)
                         if reference_authoritative else None))
                reasons.extend(provenance_reasons)
            except Exception as exc:
                reasons.append(f"invalid_final_textgrid:{exc}")
            reasons.extend(f"ctc_bundle:{item}" for item in validate_ctc_transcript_bundle(args.ctc_dir, stem))
            aligned_rejection_reasons = _aligned_reasons(aligned, reference)
            # The main Chinese MFA TextGrid may carry a lexical English
            # placeholder phone.  Once the independent English MFA
            # provenance is verified, that source placeholder is not a
            # published phone and must not veto the final en: sequence.
            if not provenance_reasons:
                aligned_rejection_reasons = [
                    item for item in aligned_rejection_reasons
                    if item != "aligned_english_self_referential_phone"
                ]
            # MFA may emit ``unk``/``spn`` on the source Chinese alignment
            # while postprocess has independently redeemed that interval via
            # the authoritative reference, complete final geometry, and the
            # verified English ledger.  In that explicit case the source
            # placeholder is diagnostic evidence, not the published label;
            # final TextGrid checks above remain mandatory.
            if (isinstance(row, dict)
                    and isinstance(row.get("mfa_unknown_source_redeemed"), dict)):
                # ``reference`` is the hyphenless semantic projection used by
                # the general content checks.  Unknown recovery also binds
                # the final English surface, so preserve lexical hyphens for
                # its independent semantic-sequence verification.
                proof_reference = (
                    normalize_authority_reference_numerals(reference_original)
                    if reference_authoritative else reference)
                proof_reasons = _unknown_recovery_proof_reasons(
                    stem, tg, proof_reference, aligned, ctc_dir, row,
                    english_manifest)
                reasons.extend(proof_reasons)
                if not proof_reasons:
                    aligned_rejection_reasons = [
                        item for item in aligned_rejection_reasons
                        if item not in {"aligned_unknown_token", "aligned_lexical_spn"}
                    ]
            reasons.extend(aligned_rejection_reasons)
        if row is None:
            reasons.append("missing_report_row")
        else:
            reasons.extend(_report_reasons(row))
            if tg is not None:
                reasons.extend(_evidence_repair_reasons(stem, tg, row))
            reasons.extend(_postprocess_contract_reasons(row, tg, lifecycle))
            if lifecycle is None and _postprocess_v3_claim(row):
                global_reasons.append(
                    "postprocess_v3_claim_without_ctc_lifecycle")
        reasons = sorted(set(reasons))
        if reasons:
            manifest["rejected"][stem] = reasons
            if args.isolate:
                destination = args.filtered_dir / candidate.name
                if destination.exists():
                    global_reasons.append(f"filtered_collision:{stem}")
                else:
                    os.replace(candidate, destination)
        else:
            try:
                if provenance_evidence is not None:
                    if evidence_staging_root is None:
                        if evidence_final_root.exists():
                            raise ValueError("evidence collision")
                        evidence_staging_root = (args.output_dir / "_provenance"
                                                 / f".english_audit_{os.getpid()}_staging")
                        if evidence_staging_root.exists():
                            raise ValueError("evidence staging collision")
                        evidence_staging_root.mkdir(parents=True)
                    copied_evidence = _copy_english_evidence(
                        stem, provenance_evidence, args.output_dir, evidence_staging_root)
                else:
                    copied_evidence = None
            except Exception as exc:
                reasons = [f"english_provenance_manifest_failed:{exc}"]
                manifest["rejected"][stem] = reasons
                # Copying verified evidence is part of publication proof.  A
                # partial/failed copy must never coexist with a publishable
                # strict manifest, even if other stems happened to validate.
                global_reasons.append(f"english_provenance_evidence_copy_failed:{stem}")
                if args.isolate:
                    destination = args.filtered_dir / candidate.name
                    if destination.exists():
                        global_reasons.append(f"filtered_collision:{stem}")
                    else:
                        os.replace(candidate, destination)
                continue
            entry = {
                "stem": stem,
                "textgrid_sha256": _sha256(candidate),
                "mode": "authority" if reference_authoritative else "fallback",
            }
            if reference_authoritative:
                entry["reference"] = {
                    "path": str(reference_path.resolve()),
                    "sha256": _sha256(reference_path),
                }
            else:
                fallback = row.get("fallback_transcript", {}) if isinstance(row, dict) else {}
                entry["fallback_transcript"] = {
                    "source": fallback.get("source"),
                    "path": fallback.get("path"),
                    "sha256": fallback.get("sha256"),
                }
            if copied_evidence is not None:
                entry["english_provenance"] = copied_evidence
            manifest["ok"].append(entry)

    # Recalculate after isolation: a move must preserve the exact corpus set.
    final_output = {path.stem for path in args.output_dir.glob("*.TextGrid")}
    final_filtered = {path.stem for path in args.filtered_dir.glob("*.TextGrid")}
    if final_output & final_filtered or final_output | final_filtered != expected:
        global_reasons.append("post_audit_set_conservation_failed")
    ok_stems = {entry["stem"] for entry in manifest["ok"]}
    if final_output != ok_stems:
        global_reasons.append("output_not_exactly_manifest_ok")
    # Evidence is a run-level transaction: do not leave individual proof
    # directories behind if any later candidate or global conservation check
    # failed.  Existing evidence is never deleted; its presence is a closed
    # collision before staging begins.
    if evidence_staging_root is not None:
        if global_reasons:
            _cleanup_evidence_staging(evidence_staging_root)
            for entry in manifest["ok"]:
                entry.pop("english_provenance", None)
        else:
            try:
                # Recheck staged ordinary files before their single atomic
                # rename makes any evidence visible to a manifest reader.
                for entry in manifest["ok"]:
                    evidence = entry.get("english_provenance")
                    if evidence is None:
                        continue
                    records = [evidence["ledger"]] + evidence["source_textgrids"]
                    for record in records:
                        staged = evidence_staging_root / Path(record["path"]).relative_to(
                            Path("_provenance") / "english")
                        if staged.is_symlink() or not staged.is_file() or _sha256(staged) != record["sha256"]:
                            raise ValueError("staged evidence hash failure")
                os.replace(evidence_staging_root, evidence_final_root)
                manifest["_evidence_committed_this_run"] = True
            except Exception as exc:
                _cleanup_evidence_staging(evidence_staging_root)
                global_reasons.append(f"english_provenance_evidence_commit_failed:{exc}")
                for entry in manifest["ok"]:
                    entry.pop("english_provenance", None)
    manifest["safe_empty"] = not manifest["ok"]
    manifest["safe_empty_applied"] = bool(manifest["safe_empty"] and global_reasons)
    if global_reasons:
        manifest["primary_global_reason"] = sorted(set(global_reasons))[0]
    manifest["global_reasons"] = sorted(set(global_reasons))
    return manifest, not global_reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and isolate strict-ok v3.1 MFA output.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--ctc-dir", type=Path, required=True)
    parser.add_argument("--pipeline-receipt", type=Path, default=None,
                        help="pipeline-run-receipt-v2 (defaults to ctc-dir/.pipeline_run_receipt_v2.json)")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-mode", choices=("auto", "authority", "fallback"),
                        default="auto",
                        help="Transcript authority policy; fallback ignores reference-dir TXT files.")
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--ctc-raw-manifest", type=Path, default=None,
                        help="Explicit immutable CTC raw manifest; otherwise derive from work receipt.")
    parser.add_argument("--ctc-work-receipt", type=Path, default=None,
                        help="Explicit mutable CTC work receipt; otherwise use ctc-dir/.ctc_work_receipt.json.")
    parser.add_argument("--mfa-input-axis-receipt", type=Path, default=None)
    parser.add_argument("--mfa-alignment-axis-receipt", type=Path, default=None)
    parser.add_argument("--mfa-axis-audio-root", type=Path, default=None)
    parser.add_argument("--tts-authoritative-audio-root", type=Path, default=None)
    parser.add_argument("--en-phones-dir", type=Path, required=True,
                        help="strict-en-mfa-v1 per-stem ledger directory")
    parser.add_argument("--en-aligned-dir", type=Path, required=True,
                        help="retained English MFA TextGrid root")
    parser.add_argument("--en-manifest", type=Path, required=True,
                        help="strict-en-mfa-v1 global run manifest")
    parser.add_argument("--strict-replay-english-import", type=Path, default=None,
                        help="strict-replay-english-import-v1 producer manifest (strict_replay only)")
    parser.add_argument("--strict-replay-english-manifest", type=Path, default=None,
                        help="explicit replay English producer manifest path")
    parser.add_argument("--strict-replay-formal-receipt", type=Path, default=None,
                        help="explicit replay output/.pipeline_run_receipt_v2.json path")
    parser.add_argument("--strict-replay-immutable-import", type=Path, default=None,
                        help="explicit replay workspace/strict_replay_import.json path")
    parser.add_argument("--strict-replay-postprocess-report", type=Path, default=None,
                        help="explicit replay output/postprocess_report.jsonl path")
    parser.add_argument("--strict-replay-english-subset", type=Path, default=None,
                        help="explicit replay English alignment subset path")
    parser.add_argument("--strict-replay-english-subset-sha256", default=None,
                        help="hash binding for replay English alignment subset")
    parser.add_argument("--strict-replay-parent-english-sha256", default=None,
                        help="hash binding for copied parent-global English manifest")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--no-isolate", dest="isolate", action="store_false")
    parser.set_defaults(isolate=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_dir.mkdir(parents=True, exist_ok=True)
    manifest, clean = audit(args)
    path = args.manifest or args.output_dir / "strict_ok_manifest.json"
    evidence_failures = _evidence_recheck(manifest, args.output_dir)
    if evidence_failures:
        manifest["global_reasons"] = sorted(set(manifest["global_reasons"] + evidence_failures))
        if manifest.pop("_evidence_committed_this_run", False):
            # The marker can only be set after this run atomically created the
            # root, so this cannot delete a user/pre-existing evidence tree.
            shutil.rmtree(args.output_dir / "_provenance" / "english", ignore_errors=True)
            try:
                (args.output_dir / "_provenance").rmdir()
            except OSError:
                pass
            for entry in manifest.get("ok", []):
                entry.pop("english_provenance", None)
        clean = False
    else:
        manifest.pop("_evidence_committed_this_run", None)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(f"strict-ok: {len(manifest['ok'])} ok, {len(manifest['rejected'])} rejected; manifest={path}")
    if manifest["safe_empty"]:
        print("strict-ok safe_empty: no publication candidate exists")
    return 0 if clean and manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
