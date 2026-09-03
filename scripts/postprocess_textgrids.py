#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process MFA TextGrids for Chinese forced alignment (pinyin + tone numbers).

Builds 5-tier (or 6-tier) TextGrid:
  raw_text       — original Chinese sentence
  pinyin         — pinyin with tone numbers + punctuation
  words          — MFA-aligned pinyin words (with tone numbers)
  phones         — MFA-aligned phones (IPA notation)
  pinyin_phones  — IPA phones reverse-mapped to pinyin tone-number notation
  corrected_text — (optional) Chinese text with punctuation corrected against
                    actual silence gaps: deleted where silence is missing,
                    [sp] inserted where silence exists without punctuation

Also generates tone_mapping.json — bidirectional IPA↔pinyin tone reference table.
"""

import argparse
import array
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import shutil
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from pypinyin import lazy_pinyin, Style
except ModuleNotFoundError:
    raise SystemExit("pypinyin is not installed. Run: pip install pypinyin")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_utils import (
    SILENCE_LABELS, NVV_NAMES, CHINESE_INITIALS_SET,
    IPA_CONSONANT_MAP, IPA_TONE_TO_DIGIT, IPA_VOWEL_BASE_MAP,
    TONE_MARK_CHARS, FINAL_DECOMPOSE, FINAL_TONE_INDEX,
    CHINESE_SHORT_WORDS,
    is_cjk, is_nvv_token, is_english_token, is_pinyin_syllable,
    is_unknown_token, is_word_like, is_punct, extract_word_chars,
    normalize_authority_reference_numerals,
    REFERENCE_NUMERAL_NORMALIZATION_SCHEMA,
    is_english_phone, is_english_vowel_phone, is_english_consonant_phone,
    en_ipa_to_arpabet, apply_arpabet_stress, align_sequences,
    is_silence, EN_PHONE_PREFIX,
    validate_ctc_raw_manifest, validate_ctc_work_receipt,
)
from english_units import (
    EnglishUnitError,
    canonicalize_english_token,
    is_english_fragment_token,
    merge_authority_fragment_group,
    parse_english_units,
    project_authority_semantics,
    resolve_processed_english_token,
    validate_processed_english_token_binding,
)
SHORT_PAUSE_PUNCT = set("，、：；,")
LONG_PAUSE_PUNCT = set("。？！…!?.")
SHORT_PAUSE_TOKEN = "[PAUSE]"
LONG_PAUSE_TOKEN = "<PAUSE>"

# This schema is intentionally duplicated rather than imported from
# align_english_mfa: post-processing must be able to reject malformed or
# legacy producer output without creating an import cycle.
STRICT_EN_MFA_SCHEMA = "strict-en-mfa-v2"
HISTORICAL_STRICT_EN_MFA_SCHEMA = "strict-en-mfa-v1"
CANONICAL_UNITS_SCHEMA = "canonical-english-units-v1"
SOS_PRONUNCIATION_POLICY_ID = "sos-exact-override-v1"
SOS_EXPECTED_PRONUNCIATION = ("EH2", "S", "OW2", "EH1", "S")
APP_EXPECTED_PRONUNCIATION = ("AE1", "P")
_STRICT_EN_SILENCE = {"sil", "sp", "spn", "<eps>"}
MFA_INPUT_AXIS_SCHEMA = "mfa-input-axis-receipt-v1"
MFA_ALIGNMENT_AXIS_SCHEMA = "mfa-alignment-axis-receipt-v1"
MFA_ALIGNMENT_AXIS_V2_SCHEMA = "mfa-alignment-axis-receipt-v2"
AXIS_EPS = 0.003
CTC_FRAME_MS = 60
CTC_QUERY_FRAMES = 4
CTC_FRAME_SUPPORT_SCHEMA = "ctc-frame-support-v1"
_FRAME_SUPPORT_EPS = 1e-6
_FRAME_SUPPORT_DISPLAY_MIN_S = 1e-6
_EVIDENCE_REPAIR_FLOOR_S = 0.030
_EVIDENCE_REPAIR_SCHEMA = "evidence-constrained-repair-v1"
_UNKNOWN_REPAIR_PROOF_SCHEMA = "mfa-unknown-recovery-proof-v1"
FALLBACK_CORRESPONDENCE_SCHEMA = "fallback-lexical-correspondence-v2"
PUNCTUATION_EVIDENCE_SCHEMA = "ctc-punctuation-evidence-v2"
NVASR_CANDIDATE_PROVENANCE_SCHEMA = "nvasr-candidate-provenance-v1"
NVASR_MAPPING_BASIS = "raw_ctc_label_neighbors_forced_overlap-v2"
PUBLICATION_TRANSACTION_SCHEMA = "derived-publication-transaction-v2"
FALLBACK_SURFACE_SCHEMA = "fallback-punctuation-surface-v1"
FALLBACK_PUNCTUATION_PROJECTION_SCHEMA = "fallback-punctuation-projection-v1"
WORD_ENERGY_EVIDENCE_SCHEMA = "word-energy-evidence-v1"
_OVERLAP_EVIDENCE_STEMS = frozenset({"000240", "000314", "001776", "001802"})
_SEMANTIC_NVV = re.compile(r"<([A-Za-z][A-Za-z-]*)>")

# Keep the serialized operation names stable while allowing the word-energy
# audit to consume ledgers written by older and newer geometry passes.
_MERGE_OPERATION_POLICIES = {
    "energy_short_sp_merge": "energy_short_sp_merge",
    "forced_internal_sp1_merge": "forced_internal_sp1_forward",
    "forced_internal_sp1_forward": "forced_internal_sp1_forward",
    "forced_internal_sp1_forward_merge": "forced_internal_sp1_forward",
    "short_internal_pause_left": "short_internal_pause_left",
    "short_internal_pause_left_merge": "short_internal_pause_left",
    "valid_internal_sp0_merge": "valid_internal_sp0_forward",
    "valid_internal_sp0_forward": "valid_internal_sp0_forward",
    "valid_internal_sp0_forward_merge": "valid_internal_sp0_forward",
    "internal_sp0_forward_merge": "valid_internal_sp0_forward",
    "unknown_sp0_forward_merge": "unknown_sp0_forward",
    "nvv_adjacent_sp0_forward": "nvv_adjacent_sp0_forward",
    "nvv_adjacent_sp0_forward_merge": "nvv_adjacent_sp0_forward",
    "nvv_adjacent_sp1_ctc_merge": "nvv_adjacent_sp1_ctc_containing_owner",
    "nvv_adjacent_sp1_ctc_containing_owner": "nvv_adjacent_sp1_ctc_containing_owner",
    "ctc_containing_owner_merge": "ctc_containing_owner",
    "ctc_containing_owner": "ctc_containing_owner",
    "merged_left_fallback": "merged_left_fallback",
}


def _merge_operation_metadata(operation: object,
                              policy: object = None) -> tuple[str | None, str | None]:
    """Return recognized merge operation and a non-null effective policy."""
    name = str(operation).strip() if operation is not None else ""
    inferred = _MERGE_OPERATION_POLICIES.get(name)
    if inferred is None:
        return None, None
    explicit = str(policy).strip() if policy is not None else ""
    return name, explicit or inferred


def _merge_operation_for_policy(policy: object) -> str:
    """Select the canonical operation emitted for a committed merge."""
    value = str(policy).strip() if policy is not None else ""
    if value == "nvv_adjacent_sp0_forward":
        return "nvv_adjacent_sp0_forward_merge"
    if value == "nvv_adjacent_sp1_ctc_containing_owner":
        return "nvv_adjacent_sp1_ctc_merge"
    if value == "valid_internal_sp0_forward":
        return "valid_internal_sp0_forward_merge"
    if value in {"forced_internal_sp1", "forced_internal_sp1_forward"}:
        return "forced_internal_sp1_forward_merge"
    if value == "short_internal_pause_left":
        return "short_internal_pause_left"
    if value == "unknown_sp0_forward":
        return "unknown_sp0_forward_merge"
    if value == "ctc_containing_owner":
        return "ctc_containing_owner_merge"
    if value == "energy_owner":
        return "energy_short_sp_merge"
    if value == "merged_left_fallback":
        return "merged_left_fallback"
    return "energy_short_sp_merge"
_SEMANTIC_ENGLISH = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*[0-9]*")
_SEMANTIC_SP1 = re.compile(r"<sp1>", re.I)
_SEMANTIC_SILENCE = re.compile(r"<sp[0-3]>", re.I)
_SEMANTIC_PUNCT_MAP = str.maketrans({
    ",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "：",
})
_AUTHORITY_NUMERAL_PINYIN = {
    "零": "ling2", "一": "yi1", "二": "er4", "三": "san1",
    "四": "si4", "五": "wu3", "六": "liu4", "七": "qi1",
    "八": "ba1", "九": "jiu3", "十": "shi2",
}


def _axis_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _evidence_digest(value: object) -> str:
    """Digest a proof value without depending on whitespace or key order."""
    return _axis_digest(value)


def _axis_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_wav_meta(path: Path) -> dict:
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
    except wave.Error as pcm_error:
        # The stdlib wave module rejects WAVE_FORMAT_IEEE_FLOAT (tag 3).
        # soundfile is an existing runtime dependency and can inspect those
        # headers; keep this fallback narrow so PCM metadata semantics stay
        # on the historical wave.open path.
        try:
            import soundfile as _sf
            info = _sf.info(str(path))
        except Exception:
            raise pcm_error
        if info.format not in {"WAV", "WAVEX"} or info.subtype not in {"FLOAT", "DOUBLE"}:
            raise pcm_error
        frames = info.frames
        rate = info.samplerate
        channels = info.channels
        sample_width = {"FLOAT": 4, "DOUBLE": 8}[info.subtype]
    if rate <= 0 or frames < 0:
        raise ValueError(f"invalid WAV metadata: {path}")
    return {"sha256": _axis_sha(path), "duration_s": frames / rate,
            "sample_rate": rate, "frames": frames,
            "channels": channels, "sample_width": sample_width}


def _strict_semantic_tokens(text: str) -> list[tuple[str, str]]:
    """Mirror audit_strict_ok's final reference-sequence token contract.

    Postprocess cannot import the auditor without a cycle, so this is an
    intentional local copy.  In particular, tier labels are later joined by
    spaces: separate English labels ``all`` and ``in`` must remain separate,
    whereas real lexical corruption such as ``N`` becoming ``Noa`` remains a
    mismatch.  Canonical ``<sp0>``–``<sp3>`` labels are non-lexical and are
    removed before tokenization.
    """
    return [(item["kind"], item["surface"].casefold()
             if item["kind"] == "english" else item["surface"])
            for item in project_authority_semantics(text)]


def _lexical_identity(value: object, *, ctc_item: dict | None = None) -> str:
    """Return the comparison identity while preserving the display surface.

    CTC canonical units already carry the producer's identity (for example
    ``vtuber`` for both ``v-tuber`` and ``vtuber``).  Fall back to the shared
    English parser for source/final labels, and only then use case-folded text
    for pinyin/other lexical labels.  This function is deliberately used only
    for ordered evidence comparisons; TextGrid surfaces are never rewritten.
    """
    text = str(value or "").strip()
    if isinstance(ctc_item, dict):
        canonical = ctc_item.get("canonical_unit")
        if isinstance(canonical, dict):
            match_key = canonical.get("match_key")
            if isinstance(match_key, str) and match_key:
                nvv_key = _canonical_nvv_identity(match_key)
                return nvv_key if nvv_key is not None else match_key.casefold()
    nvv_key = _canonical_nvv_identity(text)
    if nvv_key is not None:
        return nvv_key
    if is_unknown_token(text):
        return "<unknown>"
    if is_english_token(text):
        try:
            return canonicalize_english_token(text)
        except (EnglishUnitError, TypeError, ValueError):
            pass
    return text.casefold()


def _canonical_nvv_identity(value: object) -> str | None:
    """Canonicalize only the known NVV labels for lexical evidence.

    Final display tiers conventionally use bracketed uppercase NVV labels,
    while source/CTC evidence may carry the same label bare and lowercase.
    Unknown angle-bracketed text is intentionally left alone so arbitrary
    English identity comparisons do not become more permissive.
    """
    text = str(value or "").strip()
    bare = text.strip("<>").strip()
    known = {str(name).strip("<>").casefold() for name in NVV_NAMES}
    if bare and bare.casefold() in known:
        return f"<{bare.upper()}>"
    return None


def _is_substantive_interior_silence(
        intervals: list["Interval"], index: int) -> bool:
    """Return whether a silence has lexical owners on both sides.

    Punctuation, empty labels, and other silence intervals do not own a
    lexical gap.  Scanning past them prevents a normal tail silence before
    terminal punctuation (and a leading punctuation/silence chain) from being
    mistaken for an unowned interior pause, while preserving a genuine
    lexical-silence-lexical pause.
    """
    if index < 0 or index >= len(intervals):
        return False
    if not is_silence(intervals[index].text):
        return False

    def _has_lexical_owner(step: int) -> bool:
        cursor = index + step
        while 0 <= cursor < len(intervals):
            label = (intervals[cursor].text or "").strip()
            if not label or is_silence(label) or is_punct(label):
                cursor += step
                continue
            return True
        return False

    return _has_lexical_owner(-1) and _has_lexical_owner(1)


def _final_unexpected_silence_reasons(words_tier: "Tier | None") -> list[str]:
    """Recompute unexpected-silence evidence from finalized word geometry."""
    if words_tier is None:
        return []
    for index, interval in enumerate(words_tier.intervals):
        if (interval.text.strip() in {"<sp1>", "<sp2>", "<sp3>"}
                and _is_substantive_interior_silence(words_tier.intervals, index)):
            return ["unexpected_silence"]
    return []


def _load_axis_contract(args) -> tuple[list[str], dict[str, list[str]]]:
    """Read explicit MFA/TTS axis receipts before any candidate is written."""
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
    if Path(str(alignment_axis.get("alignment_root", ""))).resolve() != args.textgrid_dir.resolve():
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
        actual_grids = {path.stem for path in args.textgrid_dir.glob("*.TextGrid")}
        if actual_grids != set(stems) - missing_stems:
            errors.append("mfa_alignment_axis_status_partition_mismatch")
    else:
        # Keep all downstream parsing closed while reporting the schema defect.
        return sorted(set(errors)), stem_reasons
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
                    or abs(float(row.get("duration_s")) - actual["duration_s"]) > AXIS_EPS):
                raise ValueError("mfa audio metadata/hash")
        except (OSError, ValueError, TypeError, KeyError):
            errors.append(f"mfa_axis_audio_receipt_invalid:{stem}")
            continue
        if stem in missing_stems:
            reasons.append("missing_mfa_alignment")
        else:
            alignment = align_by_stem[stem]
            aligned = args.textgrid_dir / f"{stem}.TextGrid"
            try:
                tg = parse_textgrid(aligned)
                xmax = tg.xmax
                if (alignment.get("path") != str(aligned.resolve())
                        or alignment.get("sha256") != _axis_sha(aligned)
                        or alignment.get("audio_sha256") != row.get("sha256")
                        or abs(float(alignment.get("xmax")) - xmax) > AXIS_EPS
                        or abs(xmax - actual["duration_s"]) > AXIS_EPS):
                    reasons.append("mfa_alignment_axis_mismatch")
            except (OSError, ValueError, TypeError, KeyError):
                reasons.append("mfa_alignment_axis_mismatch")
        try:
            # The TTS source namespace may be nested (speaker/voice), while
            # the MFA axis is intentionally flat.  The transform receipt is
            # the authoritative stem-to-physical-source binding; falling
            # back to ``tts_root/stem.wav`` incorrectly rejects every nested
            # source as an axis mismatch.
            transform = transforms.get(stem)
            transform_input = (transform.get("input", {})
                               if isinstance(transform, dict) else {})
            bound_tts = Path(str(transform_input.get("path", "")))
            tts = (bound_tts if bound_tts.is_absolute()
                   else tts_root / f"{stem}.wav")
            tts_meta = _axis_wav_meta(tts)
            identity = all(tts_meta[key] == actual[key]
                           for key in ("sha256", "sample_rate", "frames", "channels", "sample_width")) and abs(
                               tts_meta["duration_s"] - actual["duration_s"]) <= AXIS_EPS
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
                    and abs(float(inp.get("duration_s")) - float(out.get("duration_s"))) <= AXIS_EPS)
                if not valid_transform:
                    reasons.append("tts_audio_axis_mismatch")
            elif not identity:
                reasons.append("tts_audio_axis_mismatch")
        except (OSError, ValueError, TypeError):
            reasons.append("tts_audio_axis_mismatch")
    return sorted(set(errors)), {stem: sorted(set(reasons)) for stem, reasons in stem_reasons.items()}

@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str

    @property
    def duration(self) -> float:
        return self.xmax - self.xmin


@dataclass
class Tier:
    name: str
    xmin: float
    xmax: float
    intervals: list[Interval]


@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: list[Tier]


def _copy_tier_metadata(source: Tier, target: Tier) -> Tier:
    """Carry non-serialized provenance across an in-memory tier rebuild.

    Several post-processing stages construct a fresh ``Tier`` object.  CTC
    boundary authority is deliberately not serialized as a TextGrid label,
    but it must survive those rebuilds or a later stage can unknowingly
    overwrite a previously accepted CTC decision.
    """
    for name in ("_ctc_word_authority", "_phone_lineage_invalid",
                 "_processed_geometry_ledger", "_processed_geometry_frozen",
                 "_processed_geometry_digest", "_canonical_authority_units",
                 "_word_energy_premerge_spans", "_word_energy_merge_ledger",
                 "_fallback_unknown_projection", "_punctuation_evidence_ledger"):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    return target


def _load_ctc_lifecycle(txt_dir: Path, stem: str) -> dict | None:
    """Read the producer's raw/work contract without mutating either tree.

    The normal pipeline supplies both environment bindings.  Tiny historical
    fixtures intentionally do not, so absence of both bindings remains a
    compatibility mode.  Once either binding is present, every identity and
    stem/path relationship is mandatory: a partially bound run must fail
    closed instead of silently treating mutable work files as raw evidence.
    """
    raw_value = os.environ.get("CTC_RAW_MANIFEST")
    work_value = os.environ.get("CTC_WORK_RECEIPT")
    if not raw_value and not work_value:
        return None
    if not raw_value or not work_value:
        raise ValueError("CTC raw/work lifecycle bindings are incomplete")

    raw_path = Path(raw_value)
    work_receipt_path = Path(work_value)
    if (raw_path.is_symlink() or not raw_path.is_file()
            or work_receipt_path.is_symlink() or not work_receipt_path.is_file()):
        raise ValueError("CTC raw manifest or work receipt is missing/unsafe")
    raw_dir = raw_path.parent
    work_dir = work_receipt_path.parent
    errors = list(validate_ctc_raw_manifest(raw_dir))
    errors.extend(validate_ctc_work_receipt(work_dir, raw_path))
    if work_dir.resolve() != Path(txt_dir).resolve():
        errors.append("CTC work receipt root does not match txt_dir")
    try:
        raw_manifest = json.loads(raw_path.read_text(encoding="utf-8"))
        work_receipt = json.loads(work_receipt_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"CTC lifecycle metadata unreadable: {exc}") from exc
    stems = raw_manifest.get("stems") if isinstance(raw_manifest, dict) else None
    if not isinstance(stems, list) or stem not in stems:
        errors.append(f"CTC lifecycle stem missing from raw manifest:{stem}")
    if (not isinstance(work_receipt, dict)
            or work_receipt.get("work_root") != str(Path(txt_dir).resolve())):
        errors.append("CTC work receipt lineage root mismatch")
    ledger = work_receipt.get("transform_ledger") if isinstance(work_receipt, dict) else None
    if not isinstance(ledger, list):
        errors.append("CTC work receipt lineage missing")
    if errors:
        raise ValueError("invalid CTC raw/work lifecycle: " + "; ".join(errors))
    return {
        "schema": "ctc-processed-input-lifecycle-v1",
        "raw_manifest": {
            "path": str(raw_path.resolve()),
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "identity": raw_manifest.get("identity"),
        },
        "work_receipt": {
            "path": str(work_receipt_path.resolve()),
            "sha256": hashlib.sha256(work_receipt_path.read_bytes()).hexdigest(),
            "identity": work_receipt.get("identity"),
            "lineage_entries": len(ledger),
        },
        "stem": stem,
    }


def _processed_geometry_ledger(words_tier: Tier | None) -> list[dict]:
    ledger = getattr(words_tier, "_processed_geometry_ledger", None)
    return ledger if isinstance(ledger, list) else []


def _ensure_processed_geometry_state(words_tier: Tier | None) -> list[dict]:
    """Attach mutable publication provenance while preserving raw spans."""
    if words_tier is None:
        return []
    ledger = _processed_geometry_ledger(words_tier)
    if not hasattr(words_tier, "_processed_geometry_ledger"):
        ledger = []
        for entry in _ctc_authority_entries(words_tier) or []:
            if not isinstance(entry, dict):
                continue
            operations = entry.setdefault("operations", [])
            if entry.get("arbitration") == "ctc_neighbor_compensation":
                operation = {"operation": "ctc_neighbor_compensation",
                             "lexical_ordinal": entry.get("lexical_ordinal")}
                if operation not in operations:
                    operations.append(operation)
                if operation not in ledger:
                    ledger.append(operation)
            initial = entry.get("initial_resolved_span")
            if not isinstance(initial, list):
                resolved = entry.get("resolved_span")
                if isinstance(resolved, (list, tuple)) and len(resolved) == 2:
                    entry["initial_resolved_span"] = [float(resolved[0]), float(resolved[1])]
            entry.setdefault("published_span", None)
        words_tier._processed_geometry_ledger = ledger
    return ledger


def _record_processed_geometry_operation(words_tier: Tier | None,
                                         operation: str,
                                         **details: object) -> None:
    """Record a processed mutation; never rewrite ``ctc_span`` evidence."""
    if words_tier is None:
        return
    ledger = _ensure_processed_geometry_state(words_tier)
    record = {"operation": operation, **details}
    if record not in ledger:
        ledger.append(record)
    for entry in _ctc_authority_entries(words_tier) or []:
        if isinstance(entry, dict):
            entry.setdefault("operations", [])
            if record not in entry["operations"]:
                entry["operations"].append(record)


def _processed_geometry_digest(words_tier: Tier | None) -> str:
    rows = []
    for interval in (words_tier.intervals if words_tier is not None else []):
        # TextGrid serialization uses _fmt(...:.6f).  Bind the report to the
        # published, serialized geometry rather than pre-serialization float
        # noise; otherwise strict audit rejects otherwise identical files.
        rows.append({"xmin": round(float(interval.xmin), 6),
                     "xmax": round(float(interval.xmax), 6),
                     "text": interval.text})
    return _evidence_digest(rows)


def _freeze_processed_geometry(textgrid: TextGrid) -> tuple[Tier | None, list[str]]:
    """Freeze current words geometry; validation is deliberately non-repairing.

    ``resolved_span`` and ``ctc_span`` remain historical/raw evidence.  The
    interval list present at this point is the only publication authority.
    """
    words_tier = tier_by_name(textgrid, "words")
    if words_tier is None:
        return None, ["words_missing_at_boundary_freeze"]
    _ensure_processed_geometry_state(words_tier)
    reasons: list[str] = []
    previous_end = words_tier.xmin
    for index, interval in enumerate(words_tier.intervals):
        if (not math.isfinite(interval.xmin) or not math.isfinite(interval.xmax)
                or interval.xmax <= interval.xmin):
            reasons.append(f"invalid_words_interval:{index}")
        if interval.xmin < previous_end - AXIS_EPS:
            reasons.append(f"words_overlap_at_freeze:{index}")
        previous_end = max(previous_end, interval.xmax)
    lexical = [iv for iv in words_tier.intervals
               if iv.text.strip() and not is_silence(iv.text) and not is_punct(iv.text)]
    entries = _ctc_authority_entries(words_tier) or []
    if not entries:
        # Small in-memory geometry fixtures and no-CTC runs have no authority
        # ledger to publish; the visual interval list is still a valid freeze.
        pass
    elif len(entries) == len(lexical):
        for entry, interval in zip(entries, lexical):
            entry["published_span"] = [float(interval.xmin), float(interval.xmax)]
    else:
        reasons.append("ctc_processed_lexical_count_mismatch")
    digest = _processed_geometry_digest(words_tier)
    _record_processed_geometry_operation(words_tier, "boundary_freeze",
                                         geometry_digest=digest,
                                         interval_count=len(words_tier.intervals))
    words_tier._processed_geometry_frozen = True
    words_tier._processed_geometry_digest = digest
    textgrid._processed_geometry_frozen = True
    textgrid._processed_geometry_digest = digest
    textgrid._processed_geometry_ledger = list(_processed_geometry_ledger(words_tier))
    return words_tier, reasons


def _rebuild_derived_from_frozen_words(textgrid: TextGrid, ipa_to_pinyin: dict[str, str],
                                       pinyin_dict: dict[str, list[str]] | None,
                                       raw_text: str, en_mfa_windows=None,
                                       warnings: list[str] | None = None,
                                       reference_authoritative: bool = False,
                                       pinyin_text: str = "",
                                       fallback_surface_ledger: dict | None = None) -> None:
    """Atomically rebuild every words-derived publication tier.

    This is the only publication transaction after words ownership is frozen.
    In particular, raw_text and pinyin are rebuilt here instead of being left
    at a pre-owner-commit snapshot.
    """
    words_tier = tier_by_name(textgrid, "words")
    if words_tier is None or not getattr(textgrid, "_processed_geometry_frozen", False):
        raise RuntimeError("derived rebuild requires frozen processed words")
    phones_tier = tier_by_name(textgrid, "phones")
    lineage = getattr(textgrid, "_phone_lineage", None)
    if phones_tier is not None and isinstance(lineage, dict):
        rebuilt = _rebuild_phones_from_lineage(words_tier, phones_tier, lineage)
        if rebuilt is not None:
            phones_tier = rebuilt
            for index, tier in enumerate(textgrid.tiers):
                if tier.name == "phones":
                    textgrid.tiers[index] = phones_tier
                    break
    hanzi_tier = _build_hanzi_tier(
        words_tier, raw_text, warnings or [],
        reference_authoritative=reference_authoritative)
    if hanzi_tier is not None:
        for index, tier in enumerate(textgrid.tiers):
            if tier.name == "hanzi":
                textgrid.tiers[index] = hanzi_tier
                break
    if phones_tier is not None and pinyin_dict is not None:
        synced = build_pinyin_phones_tier(
            phones_tier, ipa_to_pinyin, words_tier, pinyin_dict,
            en_mfa_windows=en_mfa_windows)
        if synced is not None:
            for index, tier in enumerate(textgrid.tiers):
                if tier.name == "pinyin_phones":
                    textgrid.tiers[index] = synced
                    break

    source_surface = str(raw_text or "")
    surface_validation = {"status": "not_applicable", "reasons": []}
    if not reference_authoritative:
        supplied_surface = (fallback_surface_ledger
                             if fallback_surface_ledger is not None else
                             getattr(textgrid, "_fallback_punctuation_surface_ledger", None))
        if supplied_surface is None:
            supplied_surface = _fallback_punctuation_surface_ledger(source_surface)
        valid_surface, surface_validation = _validate_fallback_punctuation_surface_ledger(
            supplied_surface, source_surface)
        if valid_surface:
            source_surface = str(supplied_surface["source_text"])
        elif warnings is not None and "fallback_surface_ledger_invalid" not in warnings:
            warnings.append("fallback_surface_ledger_invalid")
        textgrid._fallback_punctuation_surface_ledger = supplied_surface
    hanzi_tier = tier_by_name(textgrid, "hanzi")
    punctuation = [iv.text.strip() for iv in words_tier.intervals
                   if is_punct(iv.text) and not is_silence(iv.text)]
    if reference_authoritative:
        rendered_raw = "<sp1>" + _canonicalize_surface_nvv_markup(
            str(raw_text or "")).replace("<sp1>", "")
        rendered_pinyin = _reference_pinyin_text(
            str(raw_text or ""), str(pinyin_text or ""))
    else:
        rendered_raw = "<sp1>" + _canonicalize_surface_nvv_markup(
            source_surface).replace("<sp1>", "")
        rendered_pinyin = "<sp1> " + " ".join(
            iv.text.strip() for iv in words_tier.intervals
            if iv.text.strip() and not is_silence(iv.text))
    for tier_name, value in (("raw_text", rendered_raw),
                             ("pinyin", rendered_pinyin)):
        tier = tier_by_name(textgrid, tier_name)
        if tier is not None and tier.intervals:
            tier.intervals[0].text = value
    textgrid._derived_publication_transaction = {
        "schema": PUBLICATION_TRANSACTION_SCHEMA,
        "words_digest": _processed_geometry_digest(words_tier),
        "punctuation_sequence": punctuation,
        "source_surface_digest": (
            supplied_surface.get("source_digest")
            if not reference_authoritative
            and isinstance(supplied_surface, dict) else None),
        "source_surface_ledger_digest": (
            supplied_surface.get("digest")
            if not reference_authoritative
            and isinstance(supplied_surface, dict) else None),
        "source_surface_validation": surface_validation,
        "tiers": ["hanzi", "pinyin_phones", "raw_text", "pinyin"],
    }


def _ctc_authority_entries(words_tier: Tier | None) -> list[dict] | None:
    entries = getattr(words_tier, "_ctc_word_authority", None)
    return entries if isinstance(entries, list) else None


def _nvasr_nvv_identity(value: object) -> str:
    return str(value or "").strip().strip("<>").strip("[]").upper().replace(
        " ", "-")


def _nvasr_valid_span(value: object) -> list[float] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(item, bool)
                   or not isinstance(item, (int, float))
                   or not math.isfinite(float(item)) for item in value)):
        return None
    start, end = float(value[0]), float(value[1])
    if start < 0 or end <= start:
        return None
    return [start, end]


def _nvasr_owner_requirements(
        row: dict, frame_support: list[float] | None,
        *, frame_ms: int = CTC_FRAME_MS) -> tuple[
            list[list[float]] | None, list[float] | None,
            list[list[float]] | None, list[str]]:
    """Validate the dedup ledger and derive the protected owner envelope."""
    errors: list[str] = []
    if frame_support is None:
        return None, None, None, errors
    ledger = row.get("nvv_deduplication")
    if ledger is None:
        forced = _nvasr_valid_span(row.get("forced_span"))
        if forced is None:
            errors.append("forced_span_malformed")
            return None, None, None, errors
        if (forced[0] < frame_support[0] - _FRAME_SUPPORT_EPS
                or forced[1] > frame_support[1] + _FRAME_SUPPORT_EPS):
            errors.append("forced_span_outside_selected_support")
            return None, None, None, errors
        return [list(frame_support)], list(frame_support), None, errors

    if not isinstance(ledger, dict):
        errors.append("dedup_ledger_malformed")
        return None, None, None, errors
    if ledger.get("schema") != "nvv-adjacent-deduplication-v1":
        errors.append("dedup_schema_mismatch")
    if _nvasr_nvv_identity(ledger.get("label")) != _nvasr_nvv_identity(
            row.get("word")):
        errors.append("dedup_label_mismatch")
    count = ledger.get("occurrence_count")
    spans_value = ledger.get("forced_occurrence_spans")
    if (not isinstance(count, int) or isinstance(count, bool) or count < 2):
        errors.append("dedup_occurrence_count_invalid")
    if not isinstance(spans_value, list):
        errors.append("dedup_occurrence_spans_malformed")
        spans_value = []
    if isinstance(count, int) and not isinstance(count, bool) \
            and len(spans_value) != count:
        errors.append("dedup_occurrence_count_mismatch")
    spans: list[list[float]] = []
    for index, value in enumerate(spans_value):
        span = _nvasr_valid_span(value)
        if span is None:
            errors.append(f"dedup_occurrence_span_malformed:{index}")
        else:
            spans.append(span)
    if len(spans) == len(spans_value):
        for index, (left, right) in enumerate(zip(spans, spans[1:])):
            if right[0] < left[0] - _FRAME_SUPPORT_EPS:
                errors.append(f"dedup_occurrence_order_invalid:{index}")
            if right[0] < left[1] - _FRAME_SUPPORT_EPS:
                errors.append(f"dedup_occurrence_overlap:{index}")
    forced = _nvasr_valid_span(row.get("forced_span"))
    if forced is None:
        errors.append("forced_span_malformed")
    elif spans:
        envelope = [spans[0][0], spans[-1][1]]
        if any(not math.isclose(forced[pos], envelope[pos], abs_tol=1e-9)
               for pos in range(2)):
            errors.append("dedup_forced_span_not_occurrence_envelope")
    containing = [span for span in spans
                  if span[0] <= frame_support[0] + _FRAME_SUPPORT_EPS
                  and span[1] >= frame_support[1] - _FRAME_SUPPORT_EPS]
    if len(containing) != 1:
        errors.append("dedup_frame_support_occurrence_binding_invalid")
    if errors:
        return None, None, spans or None, errors
    required = [list(frame_support), *[list(span) for span in spans]]
    owner_span = [min(span[0] for span in required),
                  max(span[1] for span in required)]
    return required, owner_span, spans, errors


def _nvasr_frame_support(
        row: dict, *, wav_duration_s: float | None = None,
        validate_forced: bool = False, require_mapping: bool = False
        ) -> tuple[list[float] | None, str | None, bool, list[str]]:
    """Derive immutable physical support from the persisted CTC frame row.

    ``raw_span`` lives on the encoder axis.  The speech-axis support is the
    speech frame range shifted left by half a frame; its unclamped duration is
    therefore exactly ``raw_frame_count * CTC_FRAME_MS``.  ``forced_span`` and
    ``adjusted_span`` are correspondence evidence only and never define this
    support.  A WAV duration is the sole permitted clamp authority.
    """
    reasons: list[str] = []

    def finite(value: object) -> bool:
        return (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)))

    def integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    raw_start_frame = row.get("raw_start_frame")
    raw_end_frame = row.get("raw_end_frame")
    speech_start_frame = row.get("speech_start_frame")
    speech_end_frame = row.get("speech_end_frame")
    raw_frame_count = row.get("raw_frame_count")
    speech_frame_count = row.get("speech_frame_count")
    frame_ms = row.get("frame_ms")
    raw_frame_fields = (raw_start_frame, raw_end_frame)
    if any(not integer(value) for value in raw_frame_fields):
        reasons.append("frame_coordinates_malformed")
    elif any(value < 0 for value in raw_frame_fields):
        reasons.append("raw_frame_coordinates_out_of_axis")
    elif raw_end_frame <= raw_start_frame:
        reasons.append("raw_frame_range_invalid")
    if not integer(raw_frame_count) or raw_frame_count <= 0:
        reasons.append("raw_frame_count_invalid")
    elif integer(raw_start_frame) and integer(raw_end_frame) \
            and raw_frame_count != raw_end_frame - raw_start_frame:
        reasons.append("raw_frame_count_mismatch")
    raw_span = row.get("raw_span")
    speech_span = row.get("speech_span")
    forced_span = row.get("forced_span")
    adjusted_span = row.get("adjusted_span")

    # Older in-memory fixtures omitted redundant speech frame fields.  They
    # are safely recoverable from the persisted speech span only when that
    # span lies exactly on the CTC frame grid; malformed supplied fields are
    # never repaired this way.
    if (speech_start_frame is None and speech_end_frame is None
            and isinstance(speech_span, (list, tuple)) and len(speech_span) == 2
            and all(finite(value) for value in speech_span)):
        derived_start = float(speech_span[0]) * 1000.0 / CTC_FRAME_MS
        derived_end = float(speech_span[1]) * 1000.0 / CTC_FRAME_MS
        if (derived_start.is_integer() and derived_end.is_integer()):
            speech_start_frame = int(derived_start)
            speech_end_frame = int(derived_end)
    if speech_frame_count is None and integer(speech_start_frame) \
            and integer(speech_end_frame):
        speech_frame_count = speech_end_frame - speech_start_frame
    if (not integer(speech_start_frame) or not integer(speech_end_frame)
            or speech_start_frame < 0 or speech_end_frame <= speech_start_frame):
        reasons.append("speech_frame_coordinates_malformed")
    if (not integer(speech_frame_count) or speech_frame_count <= 0
            or (integer(raw_frame_count)
                and speech_frame_count != raw_frame_count)):
        reasons.append("speech_frame_count_mismatch")
    if frame_ms != CTC_FRAME_MS:
        reasons.append("frame_ms_mismatch")
    if (integer(raw_start_frame) and integer(raw_end_frame)
            and integer(speech_start_frame) and integer(speech_end_frame)
            and (raw_start_frame - speech_start_frame != CTC_QUERY_FRAMES
                 or raw_end_frame - speech_end_frame != CTC_QUERY_FRAMES)):
        reasons.append("speech_frame_query_offset_mismatch")

    def valid_span(value: object) -> bool:
        return _nvasr_valid_span(value) is not None

    if not valid_span(raw_span):
        reasons.append("raw_span_malformed")
    if not valid_span(speech_span):
        reasons.append("speech_span_malformed")
    if not valid_span(forced_span):
        reasons.append("forced_span_malformed")
    if not valid_span(adjusted_span):
        reasons.append("adjusted_span_malformed")
    mapping_selection = row.get("mapping_selection")

    if (require_mapping
            and mapping_selection == "unique_max_forced_speech_overlap"
            and valid_span(forced_span) and valid_span(speech_span)):
        expected_overlap = max(
            0.0,
            min(float(forced_span[1]), float(speech_span[1]))
            - max(float(forced_span[0]), float(speech_span[0])),
        )
        overlap_score = row.get("mapping_forced_speech_overlap_s")
        if (not finite(overlap_score)
                or not math.isclose(float(overlap_score), expected_overlap,
                                     abs_tol=1e-6)):
            reasons.append("mapping_speech_overlap_score_mismatch")

    if valid_span(raw_span):
        if not integer(raw_start_frame) or not integer(raw_end_frame):
            reasons.append("raw_span_frame_binding_invalid")
        else:
            expected_raw = [float(raw_start_frame) * CTC_FRAME_MS / 1000.0,
                            float(raw_end_frame) * CTC_FRAME_MS / 1000.0]
            if any(not math.isclose(float(raw_span[index]), expected,
                                    abs_tol=1e-9)
                   for index, expected in enumerate(expected_raw)):
                reasons.append("raw_span_frame_binding_invalid")
            elif integer(raw_frame_count) and not math.isclose(
                    float(raw_span[1]) - float(raw_span[0]),
                    float(raw_frame_count) * CTC_FRAME_MS / 1000.0,
                    abs_tol=1e-9):
                reasons.append("raw_span_duration_mismatch")
    if valid_span(speech_span):
        if not integer(speech_start_frame) or not integer(speech_end_frame):
            reasons.append("speech_span_frame_binding_invalid")
        else:
            expected_speech = [float(speech_start_frame) * CTC_FRAME_MS / 1000.0,
                               float(speech_end_frame) * CTC_FRAME_MS / 1000.0]
            if any(not math.isclose(float(speech_span[index]), expected,
                                    abs_tol=1e-9)
                   for index, expected in enumerate(expected_speech)):
                reasons.append("speech_span_frame_binding_invalid")
            elif integer(speech_frame_count) and not math.isclose(
                    float(speech_span[1]) - float(speech_span[0]),
                    float(speech_frame_count) * CTC_FRAME_MS / 1000.0,
                    abs_tol=1e-9):
                reasons.append("speech_span_duration_mismatch")
    if (valid_span(adjusted_span) and isinstance(row.get("start_s"), (int, float))
            and isinstance(row.get("end_s"), (int, float))):
        if any(not math.isclose(float(adjusted_span[index]), float(value),
                                abs_tol=1e-9)
               for index, value in enumerate((row["start_s"], row["end_s"]))):
            reasons.append("adjusted_span_coordinate_binding_invalid")

    support: list[float] | None = None
    source: str | None = None
    frame_limited = raw_frame_count == 1
    valid_selections = {
        "label_neighbors", "unique_max_forced_speech_overlap",
        "unique_max_forced_raw_overlap", "unique_punctuation_topology_bound",
    }
    if mapping_selection not in valid_selections:
        reasons.append("mapping_selection_unknown")
    if require_mapping:
        if row.get("provenance_schema") != NVASR_CANDIDATE_PROVENANCE_SCHEMA:
            reasons.append("mapping_provenance_invalid")
        if (row.get("mapping_basis") != NVASR_MAPPING_BASIS
                or row.get("mapping_outcome") != "unique"):
            reasons.append("mapping_outcome_invalid")
        if mapping_selection == "unique_max_forced_speech_overlap":
            overlap_score = row.get("mapping_forced_speech_overlap_s")
            if (not finite(overlap_score) or float(overlap_score) <= 0):
                reasons.append("mapping_speech_overlap_score_invalid")

    if (mapping_selection == "unique_max_forced_raw_overlap"
            and valid_span(forced_span) and valid_span(raw_span)
            and valid_span(speech_span)):
        forced_raw_overlap = (
            min(float(forced_span[1]), float(raw_span[1]))
            - max(float(forced_span[0]), float(raw_span[0])))
        if forced_raw_overlap <= 1e-9:
            reasons.append(
                "mapping_raw_selection_forced_raw_overlap_nonpositive")
        forced_speech_overlap = (
            min(float(forced_span[1]), float(speech_span[1]))
            - max(float(forced_span[0]), float(speech_span[0])))
        if forced_speech_overlap > 1e-9:
            reasons.append(
                "mapping_raw_selection_forced_speech_overlap_positive")

    if (not reasons and valid_span(speech_span)
            and integer(raw_frame_count)):
        if mapping_selection == "unique_max_forced_raw_overlap":
            half_frame_s = CTC_FRAME_MS / 2000.0
            selected = [float(raw_span[0]) - half_frame_s,
                        float(raw_span[1]) - half_frame_s]
            source = "raw_ctc_frame_span_centered"
        elif mapping_selection == "unique_punctuation_topology_bound":
            selected = [float(forced_span[0]), float(forced_span[1])]
            source = "forced_span_punctuation_topology"
        else:
            half_frame_s = CTC_FRAME_MS / 2000.0
            selected = [float(speech_span[0]) - half_frame_s,
                        float(speech_span[1]) - half_frame_s]
            source = "raw_ctc_frames_shifted_to_speech_axis"
        expected_duration = raw_frame_count * CTC_FRAME_MS / 1000.0
        if not math.isclose(selected[1] - selected[0], expected_duration,
                            abs_tol=1e-9):
            reasons.append("frame_support_duration_mismatch")
        if mapping_selection == "unique_punctuation_topology_bound":
            if raw_frame_count != 1:
                reasons.append("topology_frame_count_invalid")
            if not math.isclose(float(forced_span[1]) - float(forced_span[0]),
                                CTC_FRAME_MS / 1000.0, abs_tol=1e-9):
                reasons.append("topology_duration_invalid")
        if require_mapping:
            if selected[0] < -_FRAME_SUPPORT_EPS:
                reasons.append("frame_support_out_of_axis")
            if wav_duration_s is not None:
                if (not finite(wav_duration_s)
                        or float(wav_duration_s) <= 0):
                    reasons.append("wav_axis_malformed")
                elif selected[1] > float(wav_duration_s) + _FRAME_SUPPORT_EPS:
                    reasons.append("frame_support_out_of_axis")
            if not reasons:
                support = selected
        elif wav_duration_s is not None:
            if (not finite(wav_duration_s) or float(wav_duration_s) <= 0):
                reasons.append("wav_axis_malformed")
            else:
                wav_end = float(wav_duration_s)
                support = [max(0.0, selected[0]), min(wav_end, selected[1])]
                source = (f"{source}_wav_axis_clamp" if support != selected
                          else source)
                if support[1] <= support[0] + AXIS_EPS:
                    reasons.append("frame_support_out_of_wav_axis")
                    support = None
                    source = None
        else:
            support = selected
            if support[0] < -AXIS_EPS:
                reasons.append("frame_support_before_wav_axis_without_clamp")
                support = None
                source = None
    if (validate_forced or require_mapping) and support is not None:
        _required, _owner_span, _dedup_spans, owner_reasons = (
            _nvasr_owner_requirements(row, support))
        reasons.extend(owner_reasons)
        if owner_reasons:
            support = None
            source = None
    return support, source, frame_limited, list(dict.fromkeys(reasons))


def _contain_nvasr_frame_support(
        words_tier: Tier | None, ctc_tokens: list[dict] | None, *,
        wav_duration_s: float | None = None) -> dict:
    """Contain every final NVV owner around its physical frame support.

    Persisted frame support outranks the neighbouring words-tier display
    geometry.  A support/display collision is reconciled transactionally by
    repartitioning only the affected ordered display block; labels, interval
    count, and the physical support itself never change.  Only malformed or
    ambiguous mapping, incompatible physical supports, or insufficient axis
    capacity reject publication.
    """
    rows = [row for row in (ctc_tokens or [])
            if isinstance(row, dict) and row.get("candidate_kind") == "nvv"]
    final_rows = [interval for interval in (words_tier.intervals
                                            if words_tier is not None else [])
                  if is_nvv_token(interval.text.strip())]
    result = {
        "schema": CTC_FRAME_SUPPORT_SCHEMA,
        "status": "not_applicable" if not rows else "rejected",
        "reasons": [], "changed": 0, "repartitioned_intervals": 0,
        "reconciliations": [], "candidates": [],
    }
    if not rows:
        if final_rows:
            result["reasons"].append("final_nvv_without_frame_support")
            result["candidates"] = [{
                "frame_support_span": None,
                "frame_support_source": "missing",
                "owner_required_segments": [],
                "owner_required_span": None,
                "dedup_forced_occurrence_spans": None,
                "final_contains_frame_support": False,
                "final_contains_owner_required_segments": False,
                "frame_limited": False,
            } for _ in final_rows]
            result["status"] = "rejected"
        return result
    if len(rows) != len(final_rows):
        result["reasons"].append("final_nvv_frame_support_count_mismatch")
        return result

    intervals = list(words_tier.intervals)
    final_indices = [index for index, interval in enumerate(intervals)
                     if is_nvv_token(interval.text.strip())]

    # Reconstruct the persisted neighbour mapping on the same compact
    # non-NVV lexical axis used by the producer.  Object identity is safe
    # here because ``rows`` is a filtered view of this exact in-memory list.
    lexical_ctc_rows: list[dict] = []
    for row in ctc_tokens or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("word", row.get("text", ""))).strip()
        if text and not is_silence(text) and not is_punct(text):
            lexical_ctc_rows.append(row)
    ordinary_count = sum(
        1 for row in lexical_ctc_rows
        if row.get("candidate_kind") != "nvv")
    ordinary_before = 0
    expected_neighbours: dict[int, tuple[int | None, int | None]] = {}
    ctc_ordinals: dict[int, int] = {}
    for ordinal, row in enumerate(lexical_ctc_rows):
        ctc_ordinals[id(row)] = ordinal
        if row.get("candidate_kind") == "nvv":
            expected_neighbours[id(row)] = (
                ordinary_before - 1 if ordinary_before else None,
                ordinary_before if ordinary_before < ordinary_count else None,
            )
        else:
            ordinary_before += 1

    lexical_indices = [
        index for index, interval in enumerate(intervals)
        if interval.text.strip() and not is_silence(interval.text)
        and not is_punct(interval.text)]
    lexical_ordinals = {index: ordinal
                        for ordinal, index in enumerate(lexical_indices)}
    authority_entries = _ctc_authority_entries(words_tier)
    candidate_ids: set[str] = set()
    supports: list[dict] = []
    for index, row in enumerate(rows):
        support, source, frame_limited, reasons = _nvasr_frame_support(
            row, wav_duration_s=wav_duration_s, require_mapping=True)
        required_segments, owner_required_span, dedup_spans, owner_reasons = (
            _nvasr_owner_requirements(row, support))
        reasons = list(reasons) + list(owner_reasons)
        item = {
            "candidate_id": row.get("candidate_id"),
            "label": row.get("word"),
            "mapping_selection": row.get("mapping_selection"),
            "frame_support_span": support,
            "frame_support_source": source or "rejected",
            "owner_required_segments": required_segments or [],
            "owner_required_span": owner_required_span,
            "dedup_forced_occurrence_spans": dedup_spans,
            "final_contains_frame_support": False,
            "final_contains_owner_required_segments": False,
            "frame_limited": frame_limited,
        }
        result["candidates"].append(item)
        for reason in reasons:
            result["reasons"].append(f"{reason}:{index}")
            if reason == "frame_support_out_of_axis":
                result["reasons"].append(
                    f"frame_support_out_of_textgrid_axis:{index}")

        candidate_id = row.get("candidate_id")
        if (not isinstance(candidate_id, str) or not candidate_id
                or candidate_id in candidate_ids):
            result["reasons"].append(
                f"frame_support_mapping_identity_invalid:{index}")
        else:
            candidate_ids.add(candidate_id)
        if row.get("provenance_schema") != NVASR_CANDIDATE_PROVENANCE_SCHEMA:
            result["reasons"].append(
                f"frame_support_mapping_schema_mismatch:{index}")
        if (row.get("mapping_basis") != NVASR_MAPPING_BASIS
                or row.get("mapping_outcome") != "unique"):
            result["reasons"].append(
                f"frame_support_mapping_not_unique:{index}")
        mapping_key = row.get("mapping_key")
        expected = expected_neighbours.get(id(row))
        if not isinstance(mapping_key, dict) or expected is None:
            result["reasons"].append(
                f"frame_support_mapping_key_malformed:{index}")
        else:
            neighbours = (mapping_key.get("left_lexical_ordinal"),
                          mapping_key.get("right_lexical_ordinal"))
            if (neighbours != expected or any(
                    value is not None
                    and (not isinstance(value, int) or isinstance(value, bool)
                         or value < 0)
                    for value in neighbours)):
                result["reasons"].append(
                    f"frame_support_mapping_key_mismatch:{index}")

        owner_index = final_indices[index]
        owner = intervals[owner_index]
        if _nvasr_nvv_identity(owner.text) != _nvasr_nvv_identity(
                row.get("word")):
            result["reasons"].append(
                f"frame_support_mapping_label_mismatch:{index}")
        ctc_ordinal = ctc_ordinals.get(id(row))
        owner_lexical_ordinal = lexical_ordinals.get(owner_index)
        if authority_entries is not None:
            authority = (
                authority_entries[owner_lexical_ordinal]
                if isinstance(owner_lexical_ordinal, int)
                and owner_lexical_ordinal < len(authority_entries)
                and isinstance(authority_entries[owner_lexical_ordinal], dict)
                else None)
            authority_ordinal = (
                authority.get("ctc_lexical_ordinal")
                if isinstance(authority, dict) else None)
            authority_identity = (
                _nvasr_nvv_identity(authority.get(
                    "text", authority.get("word", "")))
                if isinstance(authority, dict) else "")
            if (not isinstance(ctc_ordinal, int)
                    or authority_ordinal != ctc_ordinal
                    or authority_identity != _nvasr_nvv_identity(
                        row.get("word"))):
                result["reasons"].append(
                    f"frame_support_authority_mapping_conflict:{index}")
        if support is not None and not reasons:
            supports.append({
                "candidate_index": index,
                "owner_index": owner_index,
                "support": support,
                "source": source,
                "frame_limited": frame_limited,
                "owner_required_segments": required_segments,
                "owner_required_span": owner_required_span,
            })

    if result["reasons"]:
        result["reasons"] = list(dict.fromkeys(result["reasons"]))
        return result

    # Attributable supports are immutable and must retain the same lexical
    # order.  Positive overlap between two such supports cannot be repaired
    # by moving display ownership and is therefore a dedicated hard failure.
    for position, current in enumerate(supports):
        candidate_index = current["candidate_index"]
        support = current["support"]
        if (support[0] < words_tier.xmin - _FRAME_SUPPORT_EPS
                or support[1] > words_tier.xmax + _FRAME_SUPPORT_EPS):
            result["reasons"].append(
                f"frame_support_out_of_textgrid_axis:{candidate_index}")
        if position:
            previous = supports[position - 1]
            previous_support = previous["support"]
            if support[0] < previous_support[0] - _FRAME_SUPPORT_EPS:
                result["reasons"].append(
                    "frame_support_physical_order_conflict:"
                    f"{previous['candidate_index']}:{candidate_index}")
            overlap = (min(previous_support[1], support[1])
                       - max(previous_support[0], support[0]))
            if overlap > _FRAME_SUPPORT_EPS:
                result["reasons"].append(
                    "frame_support_physical_conflict:"
                    f"{previous['candidate_index']}:{candidate_index}")
    if result["reasons"]:
        result["reasons"] = list(dict.fromkeys(result["reasons"]))
        return result

    original = list(intervals)
    tentative = list(intervals)
    # Repartitioning must preserve the complete owner envelope.  Physical
    # ordering/conflict checks above intentionally remain frame-support-only.
    protected = {entry["owner_index"]: entry["owner_required_span"]
                 for entry in supports}
    expanded: set[int] = set()
    for entry in supports:
        candidate_index = entry["candidate_index"]
        owner_index = entry["owner_index"]
        support = entry["owner_required_span"]
        owner = tentative[owner_index]
        reconciled = Interval(min(owner.xmin, support[0]),
                              max(owner.xmax, support[1]), owner.text)
        if (not math.isclose(owner.xmin, reconciled.xmin, abs_tol=1e-12)
                or not math.isclose(owner.xmax, reconciled.xmax,
                                    abs_tol=1e-12)):
            expanded.add(owner_index)
            result["changed"] += 1
        tentative[owner_index] = reconciled
        result["candidates"][candidate_index][
            "final_contains_frame_support"] = True
        result["candidates"][candidate_index][
            "final_contains_owner_required_segments"] = True

    # A widened owner can cross more than its immediate neighbour.  Form
    # complete index ranges from every resulting collision so intervening
    # display labels can be squeezed/repositioned without being deleted.
    conflict_ranges: list[tuple[int, int]] = []
    for left in range(len(tentative)):
        for right in range(left + 1, len(tentative)):
            if left not in expanded and right not in expanded:
                continue
            overlap = (min(tentative[left].xmax, tentative[right].xmax)
                       - max(tentative[left].xmin, tentative[right].xmin))
            if overlap > _FRAME_SUPPORT_EPS:
                conflict_ranges.append((left, right))

    components: list[list[int]] = []
    for left, right in sorted(conflict_ranges):
        if not components or left > components[-1][1]:
            components.append([left, right])
        else:
            components[-1][1] = max(components[-1][1], right)

    for left, right in components:
        block = tentative[left:right + 1]
        outer_start = min(interval.xmin for interval in block)
        outer_end = max(interval.xmax for interval in block)
        interval_count = right - left + 1
        if (outer_start < words_tier.xmin - _FRAME_SUPPORT_EPS
                or outer_end > words_tier.xmax + _FRAME_SUPPORT_EPS
                or (outer_end - outer_start
                    < interval_count * _FRAME_SUPPORT_DISPLAY_MIN_S
                    - 1e-12)):
            result["reasons"].append(
                f"frame_support_axis_capacity_conflict:{left}:{right}")
            continue

        boundary_count = interval_count - 1
        desired: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for offset in range(boundary_count):
            global_index = left + offset
            desired.append((original[global_index].xmax
                            + original[global_index + 1].xmin) / 2.0)
            support_ends = [
                support[1] for owner_index, support in protected.items()
                if left <= owner_index <= global_index]
            support_starts = [
                support[0] for owner_index, support in protected.items()
                if global_index < owner_index <= right]
            lower.append(max(
                [outer_start
                 + (offset + 1) * _FRAME_SUPPORT_DISPLAY_MIN_S]
                + support_ends))
            upper.append(min(
                [outer_end
                 - (boundary_count - offset) * _FRAME_SUPPORT_DISPLAY_MIN_S]
                + support_starts))

        earliest: list[float] = []
        for offset, value in enumerate(lower):
            if offset:
                value = max(
                    value,
                    earliest[-1] + _FRAME_SUPPORT_DISPLAY_MIN_S)
            earliest.append(value)
        latest = [0.0] * boundary_count
        for offset in range(boundary_count - 1, -1, -1):
            value = upper[offset]
            if offset < boundary_count - 1:
                value = min(
                    value,
                    latest[offset + 1] - _FRAME_SUPPORT_DISPLAY_MIN_S)
            latest[offset] = value
        if any(earliest[offset] > latest[offset] + 1e-12
               for offset in range(boundary_count)):
            result["reasons"].append(
                f"frame_support_axis_capacity_conflict:{left}:{right}")
            continue

        boundaries: list[float] = []
        for offset in range(boundary_count):
            minimum = earliest[offset]
            if boundaries:
                minimum = max(
                    minimum,
                    boundaries[-1] + _FRAME_SUPPORT_DISPLAY_MIN_S)
            boundary = min(max(desired[offset], minimum), latest[offset])
            boundaries.append(boundary)
        points = [outer_start, *boundaries, outer_end]
        for offset, interval in enumerate(block):
            tentative[left + offset] = Interval(
                points[offset], points[offset + 1], interval.text)

    if result["reasons"]:
        result["reasons"] = list(dict.fromkeys(result["reasons"]))
        result["changed"] = 0
        for item in result["candidates"]:
            item["final_contains_frame_support"] = False
            item["final_contains_owner_required_segments"] = False
        return result

    for index, interval in enumerate(tentative):
        if (not math.isfinite(interval.xmin) or not math.isfinite(interval.xmax)
                or interval.xmax - interval.xmin
                < _FRAME_SUPPORT_DISPLAY_MIN_S - 1e-12):
            result["reasons"].append(
                f"frame_support_axis_capacity_conflict:{index}:{index}")
        if (index and tentative[index - 1].xmax
                > interval.xmin + _FRAME_SUPPORT_EPS):
            result["reasons"].append(
                f"frame_support_axis_capacity_conflict:{index - 1}:{index}")
    for entry in supports:
        owner = tentative[entry["owner_index"]]
        support = entry["support"]
        if (owner.xmin > support[0] + _FRAME_SUPPORT_EPS
                or owner.xmax < support[1] - _FRAME_SUPPORT_EPS):
            result["reasons"].append(
                "frame_support_axis_capacity_conflict:"
                f"{entry['owner_index']}:{entry['owner_index']}")
        required_span = entry["owner_required_span"]
        if (required_span is None
                or owner.xmin > required_span[0] + _FRAME_SUPPORT_EPS
                or owner.xmax < required_span[1] - _FRAME_SUPPORT_EPS):
            result["reasons"].append(
                "frame_support_owner_required_containment_conflict:"
                f"{entry['owner_index']}:{entry['owner_index']}")
    if result["reasons"]:
        result["reasons"] = list(dict.fromkeys(result["reasons"]))
        result["changed"] = 0
        for item in result["candidates"]:
            item["final_contains_frame_support"] = False
            item["final_contains_owner_required_segments"] = False
        return result

    for index, (before, after) in enumerate(zip(original, tentative)):
        if (math.isclose(before.xmin, after.xmin, abs_tol=1e-12)
                and math.isclose(before.xmax, after.xmax, abs_tol=1e-12)):
            continue
        result["reconciliations"].append({
            "index": index,
            "label": before.text.strip(),
            "before": [float(before.xmin), float(before.xmax)],
            "after": [float(after.xmin), float(after.xmax)],
            "source": "physical_frame_support_display_repartition",
        })
    result["repartitioned_intervals"] = len(result["reconciliations"])
    if result["reconciliations"]:
        contained = _copy_tier_metadata(
            words_tier, Tier(words_tier.name, words_tier.xmin,
                             words_tier.xmax, tentative))
        # The caller owns TextGrid replacement.  Return the tier in a private
        # field so this helper remains convenient for direct audit tests.
        result["_contained_tier"] = contained
    result["status"] = "verified"
    return result


def _nvasr_candidate_provenance_audit(
        ctc_tokens: list[dict] | None, words_tier: Tier | None, *,
        required: bool = True, wav_duration_s: float | None = None) -> dict:
    """Audit durable NVASR candidate spans against the final words owner.

    CTC rows are the only source of raw/forced/adjusted evidence.  The final
    words interval is reported as display ownership and is never substituted
    back into an acoustic span.  Any missing stage, duplicate identity, or
    final-vs-adjusted divergence is a hard rejection.
    """
    final_rows = [interval for interval in (words_tier.intervals
                                             if words_tier is not None else [])
                  if is_nvv_token(interval.text.strip())]
    ctc_nvv_rows = [row for row in (ctc_tokens or [])
                    if isinstance(row, dict)
                    and is_nvv_token(str(row.get("word", "")).strip())]
    rows = [row for row in (ctc_tokens or [])
            if isinstance(row, dict)
            and row.get("candidate_kind") == "nvv"]
    unprovenanced_ctc_nvv = [row for row in ctc_nvv_rows
                             if row.get("candidate_kind") != "nvv"]
    base = {
        "schema": NVASR_CANDIDATE_PROVENANCE_SCHEMA,
        "mapping_basis": NVASR_MAPPING_BASIS,
        "candidate_count": len(rows),
        "status": "not_applicable" if not rows else "rejected",
        "reasons": [],
        "candidates": [],
    }
    if not rows:
        if required and final_rows:
            base["reasons"].extend([
                "final_nvv_count_mismatch",
                "final_nvv_without_candidate_provenance",
            ])
        if required and unprovenanced_ctc_nvv:
            base["reasons"].append("ctc_nvv_without_candidate_provenance")
        if base["reasons"]:
            base["status"] = "rejected"
        return base

    reasons: list[str] = []
    if required and unprovenanced_ctc_nvv:
        reasons.append("ctc_nvv_without_candidate_provenance")
    ids: set[str] = set()

    def nvv_identity(value: object) -> str:
        return str(value or "").strip().strip("<>[]").upper().replace(" ", "-")

    def span(row: dict, key: str) -> list[float] | None:
        value = row.get(key)
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(float(item)) for item in value)):
            return None
        start, end = float(value[0]), float(value[1])
        return [start, end] if 0 <= start < end else None

    if len(final_rows) != len(rows):
        reasons.append("final_nvv_count_mismatch")

    # The ordinal used for a divergence exception is the ordinal in the full
    # lexical CTC stream, not the ordinal among NVV rows.  Keep object identity
    # as the normal binding and accept an explicit ordinal only when a caller
    # has serialized the row independently.
    full_ctc_ordinals: dict[int, int] = {}
    full_ctc_ordinal_values: list[int] = []
    for ctc_row in ctc_tokens or []:
        if not isinstance(ctc_row, dict):
            continue
        text = str(ctc_row.get("word", ctc_row.get("text", ""))).strip()
        if not text or is_silence(text) or is_punct(text):
            continue
        ordinal = len(full_ctc_ordinal_values)
        full_ctc_ordinals[id(ctc_row)] = ordinal
        full_ctc_ordinal_values.append(ordinal)

    authority_entries = _ctc_authority_entries(words_tier)
    def authority_matches(ctc_ordinal: int) -> list[dict]:
        if authority_entries is None:
            return []
        matches = []
        for authority_index, authority in enumerate(authority_entries):
            if not isinstance(authority, dict):
                continue
            if "ctc_lexical_ordinal" in authority:
                authority_ordinal = authority.get("ctc_lexical_ordinal")
                if (isinstance(authority_ordinal, int)
                        and not isinstance(authority_ordinal, bool)
                        and authority_ordinal == ctc_ordinal):
                    matches.append(authority)
            elif authority_index == ctc_ordinal:
                # Small fixtures may omit the redundant field; the frozen
                # list position is then the only available ordinal binding.
                matches.append(authority)
        return matches

    def candidate_ctc_ordinal(row: dict) -> int | None:
        explicit = row.get("ctc_lexical_ordinal")
        if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
            return explicit
        return full_ctc_ordinals.get(id(row))

    def spans_equal(left: list[float] | None, right: list[float] | None) -> bool:
        return (left is not None and right is not None
                and all(math.isclose(a, b, abs_tol=AXIS_EPS)
                        for a, b in zip(left, right)))

    candidate_ordinals = [candidate_ctc_ordinal(row) for row in rows]
    candidate_ordinal_counts = {
        ordinal: candidate_ordinals.count(ordinal)
        for ordinal in set(candidate_ordinals) if ordinal is not None}
    authority_ordinal_counts: dict[int, int] = {}
    if authority_entries is not None:
        for authority_index, authority in enumerate(authority_entries):
            if not isinstance(authority, dict):
                continue
            ordinal = authority.get("ctc_lexical_ordinal", authority_index)
            if isinstance(ordinal, int) and not isinstance(ordinal, bool):
                authority_ordinal_counts[ordinal] = (
                    authority_ordinal_counts.get(ordinal, 0) + 1)

    for index, row in enumerate(rows):
        candidate_id = row.get("candidate_id")
        if (not isinstance(candidate_id, str) or not candidate_id
                or candidate_id in ids):
            reasons.append(f"candidate_identity_invalid:{index}")
        if isinstance(candidate_id, str):
            ids.add(candidate_id)
        if row.get("provenance_schema") != NVASR_CANDIDATE_PROVENANCE_SCHEMA:
            reasons.append(f"provenance_schema_mismatch:{index}")
        if row.get("mapping_basis") != NVASR_MAPPING_BASIS:
            reasons.append(f"mapping_basis_mismatch:{index}")
        if row.get("mapping_outcome") != "unique":
            reasons.append(f"mapping_not_unique:{index}")
        mapping_selection = row.get("mapping_selection")
        if mapping_selection not in {
                "label_neighbors", "unique_max_forced_speech_overlap",
                "unique_max_forced_raw_overlap",
                "unique_punctuation_topology_bound"}:
            reasons.append(f"mapping_selection_unknown:{index}")
        mapping_key = row.get("mapping_key")
        if not isinstance(mapping_key, dict):
            reasons.append(f"mapping_key_malformed:{index}")
        else:
            neighbors = (mapping_key.get("left_lexical_ordinal"),
                         mapping_key.get("right_lexical_ordinal"))
            if (neighbors[0] is None and neighbors[1] is None) or any(
                    value is not None
                    and (not isinstance(value, int) or isinstance(value, bool)
                         or value < 0)
                    for value in neighbors):
                reasons.append(f"mapping_key_malformed:{index}")
        if "ctc_lexical_ordinal" in row:
            ordinal = row.get("ctc_lexical_ordinal")
            if (not isinstance(ordinal, int) or isinstance(ordinal, bool)
                    or ordinal < 0):
                reasons.append(f"candidate_ordinal_invalid:{index}")
        ctc_ordinal = candidate_ordinals[index]
        if (ctc_ordinal is not None
                and candidate_ordinal_counts.get(ctc_ordinal, 0) != 1):
            reasons.append(f"candidate_ordinal_non_unique:{index}")
        raw = span(row, "raw_span")
        speech = span(row, "speech_span")
        forced = span(row, "forced_span")
        adjusted = span(row, "adjusted_span")
        frame_support, frame_support_source, frame_limited, frame_reasons = (
            _nvasr_frame_support(row, wav_duration_s=wav_duration_s,
                                 require_mapping=True))
        reasons.extend(f"{reason}:{index}" for reason in frame_reasons)
        owner_required_segments, owner_required_span, dedup_spans, owner_reasons = (
            _nvasr_owner_requirements(row, frame_support))
        reasons.extend(f"{reason}:{index}" for reason in owner_reasons)
        if raw is None:
            reasons.append(f"raw_span_missing:{index}")
        if speech is None:
            reasons.append(f"speech_span_missing:{index}")
        if forced is None:
            reasons.append(f"forced_span_missing:{index}")
        if adjusted is None:
            reasons.append(f"adjusted_span_missing:{index}")
        if raw is not None:
            raw_coordinate_values = [row.get("raw_start_s"), row.get("raw_end_s")]
            if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in raw_coordinate_values)
                    or any(not math.isclose(raw[pos], float(value), abs_tol=1e-9)
                           for pos, value in enumerate(raw_coordinate_values))):
                reasons.append(f"raw_coordinate_binding_invalid:{index}")
        final = None
        if index < len(final_rows):
            final = [float(final_rows[index].xmin), float(final_rows[index].xmax)]
            if nvv_identity(final_rows[index].text) != nvv_identity(row.get("word")):
                reasons.append(f"final_nvv_label_mismatch:{index}")
        adjusted_differs = (adjusted is not None and final is not None
                            and not spans_equal(adjusted, final))
        if adjusted_differs:
            # A display-only final span may diverge from the adjusted span
            # only when the frozen full-CTC authority proves the exact same
            # lexical owner and both sides of the publication transaction.
            # Every component is checked independently so a convenient ordinal
            # or identity cannot redeem arbitrary geometry.
            ctc_ordinal = candidate_ordinals[index]
            authority_options = (
                authority_matches(ctc_ordinal)
                if isinstance(ctc_ordinal, int) else [])
            authority = authority_options[0] if len(authority_options) == 1 else None
            authorized = isinstance(authority, dict)
            if not authorized:
                reasons.append(f"unauthorized_final_ctc_divergence:{index}")
            else:
                authority_ordinal = authority.get(
                    "ctc_lexical_ordinal", ctc_ordinal)
                authority_identity = nvv_identity(
                    authority.get("text", authority.get("word", "")))
                row_identity = nvv_identity(row.get("word"))
                ctc_span = span(authority, "ctc_span")
                published_span = span(authority, "published_span")
                common_authority = (
                    isinstance(ctc_ordinal, int)
                    and authority_ordinal == ctc_ordinal
                    and authority_ordinal_counts.get(ctc_ordinal, 0) == 1
                    and authority_identity == row_identity
                    and spans_equal(ctc_span, adjusted)
                    and spans_equal(published_span, final)
                )
                ctc_authority_ok = (
                    common_authority
                    and authority.get("boundary_source") == "ctc")
                # Physical frame support is narrower authority than a CTC
                # display decision.  It may justify only the frozen display
                # divergence that contains this exact candidate's immutable
                # support; it never turns the wider display into evidence.
                frame_support_authority_ok = (
                    common_authority
                    and frame_support is not None
                    and final[0] <= frame_support[0] + _FRAME_SUPPORT_EPS
                    and final[1] >= frame_support[1] - _FRAME_SUPPORT_EPS)
                authority_ok = ctc_authority_ok or frame_support_authority_ok
                if not authority_ok:
                    reasons.append(f"unauthorized_final_ctc_divergence:{index}")
        base["candidates"].append({
            "candidate_id": candidate_id,
            "label": row.get("word"),
            "mapping_basis": row.get("mapping_basis"),
            "mapping_outcome": row.get("mapping_outcome"),
            "mapping_selection": mapping_selection,
            "mapping_key": row.get("mapping_key"),
            "raw_span": raw,
            "raw_frames": [row.get("raw_start_frame"), row.get("raw_end_frame")],
            "raw_frame_count": row.get("raw_frame_count"),
            "frame_ms": row.get("frame_ms"),
            "verified_one_frame": row.get("raw_frame_count") == 1,
            "speech_span": speech,
            "forced_span": forced,
            "adjusted_span": adjusted,
            "frame_support_span": frame_support,
            "frame_support_source": frame_support_source or "rejected",
            "owner_required_segments": owner_required_segments or [],
            "owner_required_span": owner_required_span,
            "dedup_forced_occurrence_spans": dedup_spans,
            "final_span": final,
            "display_span": final,
            "display_owner": "words_tier_final",
            "display_is_acoustic_evidence": False,
            "final_contains_frame_support": (
                frame_support is not None and final is not None
                and final[0] <= frame_support[0] + _FRAME_SUPPORT_EPS
                and final[1] >= frame_support[1] - _FRAME_SUPPORT_EPS),
            "final_contains_owner_required_segments": (
                owner_required_segments is not None and final is not None
                and all(final[0] <= segment[0] + _FRAME_SUPPORT_EPS
                        and final[1] >= segment[1] - _FRAME_SUPPORT_EPS
                        for segment in owner_required_segments)),
            "frame_limited": frame_limited,
        })
    base["reasons"] = list(dict.fromkeys(reasons))
    base["status"] = "verified" if not base["reasons"] else "rejected"
    return base


def _ctc_authoritative_ordinal(words_tier: Tier | None,
                               interval_index: int) -> int | None:
    """Return the lexical ordinal for a CTC-authoritative interval."""
    if words_tier is None or interval_index < 0:
        return None
    lexical_ordinal = 0
    for index, interval in enumerate(words_tier.intervals):
        if not interval.text.strip() or is_silence(interval.text) or is_punct(interval.text):
            continue
        if index == interval_index:
            entries = _ctc_authority_entries(words_tier)
            if entries is not None and lexical_ordinal < len(entries):
                entry = entries[lexical_ordinal]
                if (isinstance(entry, dict)
                        and entry.get("boundary_source") == "ctc"):
                    return lexical_ordinal
            return None
        lexical_ordinal += 1
    return None


def _ctc_authoritative_ordinals(words_tier: Tier | None) -> set[int]:
    entries = _ctc_authority_entries(words_tier)
    if entries is None:
        return set()
    return {index for index, entry in enumerate(entries)
            if isinstance(entry, dict)
            and entry.get("boundary_source") == "ctc"}


def _reassert_ctc_word_authority_tier(words_tier: Tier | None
                                      ) -> tuple[Tier | None, int]:
    """Restore the accepted word decision after a mutating stage.

    This is intentionally a narrow reapplication of an already accepted
    decision.  It does not invent anchors or repair CTC sequence errors.  A
    later punctuation pass may still clip the two immediate lexical owners,
    but generic energy, gap, or overlap code cannot silently replace the
    accepted CTC/MFA decision with a new interval.  CTC-owned entries use the
    CTC-derived resolved span; MFA-owned entries use the span produced by the
    original snap decision and therefore retain the original arbitration.
    """
    entries = _ctc_authority_entries(words_tier)
    if words_tier is None or entries is None:
        return words_tier, 0
    lexical_positions = [index for index, interval in enumerate(words_tier.intervals)
                          if interval.text.strip()
                          and not is_silence(interval.text)
                          and not is_punct(interval.text)]
    if len(lexical_positions) != len(entries):
        return words_tier, 0
    intervals = list(words_tier.intervals)
    changed = 0
    for lexical_ordinal, entry in enumerate(entries):
        if (not isinstance(entry, dict)
                or entry.get("boundary_source") not in {"ctc", "mfa"}):
            continue
        span = entry.get("resolved_span") or entry.get("ctc_span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        try:
            start, end = float(span[0]), float(span[1])
        except (TypeError, ValueError):
            continue
        start = max(words_tier.xmin, start)
        end = min(words_tier.xmax, end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start + AXIS_EPS:
            continue
        position = lexical_positions[lexical_ordinal]
        current = intervals[position]
        if abs(current.xmin - start) > AXIS_EPS or abs(current.xmax - end) > AXIS_EPS:
            intervals[position] = Interval(start, end, current.text)
            changed += 1
        if entry.get("boundary_source") == "mfa":
            mfa_span = entry.get("mfa_span")
            ctc_span = entry.get("ctc_span")
            if (isinstance(mfa_span, (list, tuple)) and len(mfa_span) == 2
                    and isinstance(ctc_span, (list, tuple)) and len(ctc_span) == 2):
                mfa_overlap = min(end, float(mfa_span[1])) - max(
                    start, float(mfa_span[0]))
                ctc_overlap = min(end, float(ctc_span[1])) - max(
                    start, float(ctc_span[0]))
                # A later CTC-boundary compensation may have happened in a
                # tier rebuild that preserved the original decision metadata.
                # Promote provenance here as the final publication contract
                # is evaluated after that rebuild, not only inside snap.
                if (mfa_overlap <= AXIS_EPS and ctc_overlap > AXIS_EPS):
                    entry["boundary_source"] = "ctc"
                    entry["arbitration"] = "ctc_neighbor_compensation"
    if not changed:
        return words_tier, 0
    return _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)), changed


# ---------------------------------------------------------------------------
# NVV bracket + sp1 normalization (runs BEFORE QC filtering)
# ---------------------------------------------------------------------------

_NVV_PATTERN = re.compile(
    r"(?<![A-Za-z<>-])("
    + "|".join(re.escape(name) for name in sorted(NVV_NAMES, key=len, reverse=True))
    + r")(?![A-Za-z<>-])",
    re.IGNORECASE
)

_SP_PREFIX_PATTERN = re.compile(r"^<sp[0-9]>")

# Chinese IPA phone markers: pinyin tone digits, common Chinese initials, tone chars
_CHINESE_PHONE_RE = re.compile(
    r"(?:[1-5]$)"                          # tone digit suffix (pinyin)
    r"|^(?:[pbpmfdtnlgkhjqxrzcsyw]|[zcs]h|[dt]h)$"  # Chinese pinyin initials
    r"|[" + re.escape("".join(TONE_MARK_CHARS)) + r"]"  # IPA tone marks
    # Chinese-specific IPA phones not in English MFA inventory.
    # Excludes ʰ ʲ ʷ (used in both), ŋ (English NG).
    # ɕ=tɕ initial (x/j/q), ʂ=retroflex (sh), ʐ=retroflex (r-),
    # ʈ=retroflex stop (zh/ch), ɤ=back unrounded vowel (Chinese e).
    r"|^[a-z]*[ɕʂʐʈɳɲɻɤ]+[a-z]*$"
)


def _looks_chinese_phone(phone: str) -> bool:
    """Return True if *phone* matches Chinese IPA/pinyin patterns.

    Used to distinguish Chinese phones from English MFA IPA phones when
    both may contain IPA characters (e.g. ə appears in both).
    """
    p = phone.strip()
    if not p:
        return False
    if p in ("sil", "sp", "spn", "<eps>"):
        return False
    if p.startswith(EN_PHONE_PREFIX):
        return False
    if is_english_phone(p):
        # ARPABET English phone — definitely not Chinese
        return False
    return bool(_CHINESE_PHONE_RE.search(p))



def _finalize_textgrid(tg: TextGrid) -> None:
    """Apply final normalizations **before** QC filtering.

    Transforms every tier *in-place*:
      1. Wrap bare NVV names with ``< >`` in all intervals (standalone
         AND embedded inside long single-interval text).
      2. Tier 1 (raw_text): prepend ``<sp1>`` if not already present.
      3. Tiers 2–5: rename the first ``<spN>`` to ``<sp1>``.
    """
    for t_idx, tier in enumerate(tg.tiers):
        for iv in tier.intervals:
            if not iv.text:
                continue
            iv.text = _NVV_PATTERN.sub(lambda m: f"<{m.group(1).upper()}>", iv.text)

        if t_idx == 0:
            first_iv = tier.intervals[0] if tier.intervals else None
            if first_iv and first_iv.text.strip() and not first_iv.text.startswith("<sp"):
                first_iv.text = f"<sp1>{first_iv.text}"
        elif t_idx <= 4:
            # Insert leading <sp1> when the first interval starts after 0
            # and no silence interval marks the opening gap.
            if (tier.intervals and tier.intervals[0].xmin > 0.005
                    and not tier.intervals[0].text.startswith("<sp")):
                tier.intervals.insert(0, Interval(0.0, tier.intervals[0].xmin, "<sp1>"))
            for iv in tier.intervals:
                if not iv.text:
                    continue
                if _SP_PREFIX_PATTERN.match(iv.text):
                    iv.text = _SP_PREFIX_PATTERN.sub("<sp1>", iv.text, count=1)
                    break
                if (iv.text.startswith("<sp") and iv.text.endswith(">")
                        and len(iv.text) == 5 and iv.text[3].isdigit()):
                    iv.text = "<sp1>"
                    break


# ---------------------------------------------------------------------------
# TextGrid I/O (same as before)
# ---------------------------------------------------------------------------

def parse_textgrid(path: Path) -> TextGrid:
    lines = path.read_text(encoding="utf-8").splitlines()
    xmin = xmax = 0.0
    tiers: list[Tier] = []
    current: Tier | None = None
    pending_xmin: float | None = None
    pending_xmax: float | None = None
    in_items = in_interval = False

    for raw_line in lines:
        line = raw_line.strip()
        if line == "item []:":
            in_items = True
            continue
        if not in_items:
            if line.startswith("xmin = "):
                xmin = float(line.split("=", 1)[1])
            elif line.startswith("xmax = "):
                xmax = float(line.split("=", 1)[1])
            continue
        if line.startswith("item ["):
            if current is not None:
                tiers.append(current)
            current = Tier(name="", xmin=xmin, xmax=xmax, intervals=[])
            pending_xmin = pending_xmax = None
            in_interval = False
        elif current is not None and line.startswith("name = "):
            current.name = _unquote(line.split("=", 1)[1].strip())
        elif current is not None and line.startswith("xmin = "):
            val = float(line.split("=", 1)[1])
            if in_interval:
                pending_xmin = val
            else:
                current.xmin = val
        elif current is not None and line.startswith("xmax = "):
            val = float(line.split("=", 1)[1])
            if in_interval:
                pending_xmax = val
            else:
                current.xmax = val
        elif current is not None and line.startswith("intervals ["):
            pending_xmin = pending_xmax = None
            in_interval = True
        elif current is not None and line.startswith("text = "):
            text = _unquote(line.split("=", 1)[1].strip())
            if pending_xmin is None or pending_xmax is None:
                raise ValueError(f"Malformed interval near: {raw_line}")
            current.intervals.append(Interval(pending_xmin, pending_xmax, text))
            pending_xmin = pending_xmax = None
            in_interval = False

    if current is not None:
        tiers.append(current)
    if not tiers:
        raise ValueError(f"No tiers found in {path}")
    return TextGrid(xmin=xmin, xmax=xmax, tiers=tiers)


def _unquote(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('""', '"')


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def write_textgrid(tg: TextGrid, path: Path) -> None:
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        f"xmin = {_fmt(tg.xmin)} ", f"xmax = {_fmt(tg.xmax)} ",
        "tiers? <exists> ", f"size = {len(tg.tiers)} ", "item []: ",
    ]
    for ti, tier in enumerate(tg.tiers, start=1):
        lines.extend([
            f"    item [{ti}]:", '        class = "IntervalTier" ',
            f"        name = {_quote(tier.name)} ",
            f"        xmin = {_fmt(tier.xmin)} ", f"        xmax = {_fmt(tier.xmax)} ",
            f"        intervals: size = {len(tier.intervals)} ",
        ])
        for ii, iv in enumerate(tier.intervals, start=1):
            lines.extend([
                f"        intervals [{ii}]:",
                f"            xmin = {_fmt(iv.xmin)} ",
                f"            xmax = {_fmt(iv.xmax)} ",
                f"            text = {_quote(iv.text)} ",
            ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# IPA ↔ Pinyin bidirectional mapping (built from dictionaries)
# ---------------------------------------------------------------------------

def load_dict(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load a pronunciation dictionary.

    Returns (dict, case_map) where dict maps token->[phones] and case_map
    maps lowercase->canonical form (so MFA's lowercase output can be fixed).
    """
    d = {}
    case_map = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                d[token] = parts[1:]
                lower = token.lower()
                if lower not in case_map:
                    case_map[lower] = token
    return d, case_map


def decompose_pinyin_phone(phone: str) -> list[str]:
    """Decompose a pinyin phone into individual components for 1:1 IPA alignment.

    E.g., 'ai1' -> ['a1', 'i'], 'ian3' -> ['i', 'e3', 'n'], 'b' -> ['b'].
    """
    m = re.match(r'^(.+?)([1-5])$', phone)
    if not m:
        return [phone]
    base, tone = m.group(1), m.group(2)
    if base not in FINAL_DECOMPOSE:
        return [phone]
    components = FINAL_DECOMPOSE[base]
    tone_idx = FINAL_TONE_INDEX.get(base, 0)
    result = []
    for i, comp in enumerate(components):
        if i == tone_idx:
            result.append(comp + tone)
        else:
            result.append(comp)
    return result


def is_vowel_phone(text: str) -> bool:
    """Chinese finals end with tone digit 1-5 or tone mark; initials don't."""
    t = text.strip().lower()
    if t in CHINESE_INITIALS_SET:
        return False
    return bool(re.search(r'[1-5]$', t) or any(c in TONE_MARK_CHARS for c in t))


def is_consonant_phone(text: str) -> bool:
    """Chinese initials: consonant phones without tone marks/digits."""
    t = text.strip().lower()
    return t in CHINESE_INITIALS_SET or (t and not is_vowel_phone(t))


def build_ipa_to_pinyin_map(pinyin_dict: dict[str, list[str]],
                            ipa_dict: dict[str, list[str]]) -> dict[str, str]:
    """
    Build IPA->pinyin phone mapping: static table + dict-based cross-referencing.
    """
    mapping: dict[str, str] = {}

    # 1. Fill from static consonant map
    for ipa_p, py_p in IPA_CONSONANT_MAP.items():
        if py_p:
            mapping[ipa_p] = py_p

    # 2. Fill from dict-based cross-referencing, decomposing compound finals
    #    so that IPA and pinyin phone sequences always align 1:1.
    for token, pinyin_phones in pinyin_dict.items():
        ipa_phones = ipa_dict.get(token)
        if not ipa_phones:
            continue
        decomposed_py: list[str] = []
        for phone in pinyin_phones:
            decomposed_py.extend(decompose_pinyin_phone(phone))
        if len(ipa_phones) == len(decomposed_py):
            for ipa_p, py_p in zip(ipa_phones, decomposed_py):
                if ipa_p not in mapping:
                    mapping[ipa_p] = py_p

    # 3. Generate vowel+tone mappings
    for base_ipa, base_py in IPA_VOWEL_BASE_MAP.items():
        for tone_ipa, tone_digit in IPA_TONE_TO_DIGIT.items():
            ipa_phone = base_ipa + tone_ipa
            py_phone = base_py + tone_digit
            if ipa_phone not in mapping:
                mapping[ipa_phone] = py_phone

    return mapping


def build_tone_reference_table(ipa_to_pinyin: dict[str, str]) -> dict[str, object]:
    """
    Build a structured tone reference: consonant mapping + vowel tone mapping.
    Returns a dict with 'consonants', 'vowel_tones', 'tone_marks' sections.
    """
    consonants = {}
    vowel_tones = {}
    tone_marks_set = set()

    for ipa_p, py_p in sorted(ipa_to_pinyin.items()):
        # Tone mark pattern: Chao tone letters ˥ ˧ ˨ ˩ ˦
        has_tone = bool(re.search(r'[˥˧˨˩˦]', ipa_p))
        if has_tone:
            # Extract base vowel and tone
            base = re.sub(r'[˥˧˨˩˦]+', '', ipa_p)
            tone_match = re.search(r'[˥˧˨˩˦]+', ipa_p)
            tone_ipa = tone_match.group(0) if tone_match else ''
            tone_digit = re.search(r'[1-5]$', py_p)
            tone_num = tone_digit.group(0) if tone_digit else '?'

            key = f"{base} -> {py_p}"
            if key not in vowel_tones:
                vowel_tones[key] = {"ipa_phone": ipa_p, "pinyin_phone": py_p,
                                    "base": base, "tone_ipa": tone_ipa, "tone_digit": tone_num}
            tone_marks_set.add((tone_ipa, tone_num))
        else:
            if ipa_p not in consonants:
                consonants[ipa_p] = py_p

    # Sort tone marks
    tone_list = sorted(tone_marks_set, key=lambda x: x[1])

    return {
        "description": "IPA ↔ Pinyin bidirectional phone mapping reference",
        "consonants": dict(sorted(consonants.items())),
        "vowel_with_tones": vowel_tones,
        "tone_marks_table": {ipa: digit for ipa, digit in tone_list},
        "tone_marks_table_reverse": {digit: ipa for ipa, digit in tone_list},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def silence_label(duration: float) -> str:
    if duration < 0.2:
        return "<sp0>"
    if duration < 0.5:
        return "<sp1>"
    if duration < 1.5:
        return "<sp2>"
    return "<sp3>"


def _silence_label_from_ticks(duration_ticks: int) -> str:
    """Classify a serialized interval using integer microsecond ticks."""
    if duration_ticks < 200_000:
        return "<sp0>"
    if duration_ticks < 500_000:
        return "<sp1>"
    if duration_ticks < 1_500_000:
        return "<sp2>"
    return "<sp3>"


def _pure_silence_label(text: object) -> str | None:
    """Return a canonical ``<spN>`` label, excluding attached text."""
    label = str(text).strip().casefold()
    if label in {"<sp0>", "<sp1>", "<sp2>", "<sp3>"}:
        return label
    return None


def _reconcile_publication_geometry(words_tier: Tier) -> Tier:
    """Canonicalize the display owner partition without enlarging words.

    Reference/CTC reconciliation may leave a small frame residual or a real
    pause between lexical owners.  Only residuals within ``AXIS_EPS`` snap to
    the adjacent owner; every larger gap becomes an explicit canonical
    silence.  Overlaps are trimmed at the already-established owner boundary,
    so a punctuation anchor can never extend a lexical word over a pause.
    """
    axis_start, axis_end = float(words_tier.xmin), float(words_tier.xmax)
    if not (math.isfinite(axis_start) and math.isfinite(axis_end)
            and axis_end > axis_start):
        return words_tier

    source = []
    for interval in words_tier.intervals:
        start, end = float(interval.xmin), float(interval.xmax)
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start = max(axis_start, start)
        end = min(axis_end, end)
        if end > start:
            source.append(Interval(start, end, interval.text))
    source.sort(key=lambda interval: (interval.xmin, interval.xmax))

    result: list[Interval] = []
    cursor = axis_start

    def append_silence(start: float, end: float) -> None:
        if end <= start:
            return
        duration = end - start
        label = silence_label(duration)
        result.append(Interval(start, end, label))

    for interval in source:
        start, end = interval.xmin, interval.xmax
        gap = start - cursor
        if gap > AXIS_EPS:
            append_silence(cursor, start)
        elif gap > 0:
            start = cursor
        if start < cursor:
            start = cursor
        if end <= start:
            continue
        text = interval.text
        if is_silence(text):
            # Compute the canonical label once for this owner.  The hanzi
            # tier mirrors this interval after the words tier is rebuilt.
            text = silence_label(end - start)
        result.append(Interval(start, end, text))
        cursor = end

    if cursor < axis_end - AXIS_EPS:
        append_silence(cursor, axis_end)
    elif cursor < axis_end:
        # The residual is an axis tick, not a substantive edge pause.
        if result:
            last = result[-1]
            result[-1] = Interval(last.xmin, axis_end, last.text)
        else:
            append_silence(axis_start, axis_end)
    return _copy_tier_metadata(
        words_tier, Tier(words_tier.name, axis_start, axis_end, result))


# TextGrid timestamps are decimal seconds, but they arrive here as binary
# floats after parsing.  QC boundaries such as 30 ms are policy ticks, not
# approximate floating-point tolerances: 30.000 ms is valid while 29.999 ms
# is not.  Convert through the decimal spelling emitted by the parser and
# compare integer microsecond ticks so the boundary is deterministic.
_TIME_TICK_HZ = 1_000_000


def _duration_ticks(xmin: float, xmax: float) -> int:
    """Return duration on the six-decimal serialized TextGrid axis."""
    try:
        # QC audits the artifact that will be written.  Internal arithmetic
        # can leave ``11.57`` as ``11.569999999...``; using ``str(float)``
        # directly then reports a false 29,999 us interval even though both
        # endpoints serialize to an exact 30 ms span.
        raw = (Decimal(_fmt(xmax)) - Decimal(_fmt(xmin))) * _TIME_TICK_HZ
        if not raw.is_finite():
            return 0
        return int(raw)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return 0


def _threshold_ticks(seconds: float) -> int:
    """Convert a seconds threshold to integer microsecond ticks."""
    return _duration_ticks(0.0, seconds)


def _phone_duration_qc_issues(
        phone: Interval, phone_idx: int, *, filter_short_phone: bool,
        short_phone_sec: float, long_consonant_sec: float,
        long_vowel_sec: float, english: bool = False) -> list[dict]:
    """Classify one phone on the serialized TextGrid time axis.

    Phone boundaries are repeatedly transformed before publication, so their
    binary-float subtraction can land infinitesimally above or below a policy
    threshold even when both endpoints serialize to the exact boundary.  QC
    must judge the artifact users receive: six-decimal endpoint ticks.
    """
    duration_ticks = _duration_ticks(phone.xmin, phone.xmax)
    short_ticks = _threshold_ticks(short_phone_sec)
    long_consonant_ticks = _threshold_ticks(long_consonant_sec)
    long_vowel_ticks = _threshold_ticks(long_vowel_sec)
    duration = round(duration_ticks / _TIME_TICK_HZ, 6)

    if english:
        clean = phone.text.replace(EN_PHONE_PREFIX, "")
        issues: list[dict] = []
        if filter_short_phone and duration_ticks < short_ticks:
            issues.append({
                "rule": "short_phone_en", "text": phone.text,
                "phone_idx": phone_idx, "duration": duration})
        if (is_english_vowel_phone(clean)
                and duration_ticks > long_vowel_ticks):
            issues.append({
                "rule": "long_vowel_en", "text": phone.text,
                "phone_idx": phone_idx, "duration": duration})
        if (is_english_consonant_phone(clean)
                and duration_ticks > long_consonant_ticks):
            issues.append({
                "rule": "long_consonant_en", "text": phone.text,
                "phone_idx": phone_idx, "duration": duration})
        return issues

    issues = []
    if filter_short_phone and duration_ticks < short_ticks:
        issues.append({
            "rule": "short_phone", "text": phone.text,
            "phone_idx": phone_idx, "duration": duration})
    if (is_consonant_phone(phone.text)
            and duration_ticks > long_consonant_ticks):
        issues.append({
            "rule": "long_consonant_phone", "text": phone.text,
            "phone_idx": phone_idx, "duration": duration})
    if is_vowel_phone(phone.text) and duration_ticks > long_vowel_ticks:
        issues.append({
            "rule": "long_vowel_phone", "text": phone.text,
            "phone_idx": phone_idx, "duration": duration})
    return issues


def tier_by_name(tg: TextGrid, name: str) -> Tier | None:
    for tier in tg.tiers:
        if tier.name.lower() == name.lower():
            return tier
    return None

# ---------------------------------------------------------------------------
# ── Per-initial duration ratios for the proportional-split fallback ──
# When MFA under-produces phones for a Chinese syllable (Regression Case 26),
# the word interval is split init:final according to these ratios.  Each value
# represents the typical fraction of the syllable occupied by the initial
# consonant.  Fallback default is 0.35 (affricate / general).
_INIT_FRAC: dict[str, float] = {
    # Stops — shortest, ~15-25% of syllable
    'b': 0.20, 'p': 0.20, 'd': 0.20, 't': 0.20, 'g': 0.20, 'k': 0.20,
    # Nasals / laterals — ~15-25%
    'm': 0.22, 'n': 0.22, 'l': 0.22,
    # Fricatives — ~20-35%
    'f': 0.28, 's': 0.28, 'sh': 0.28, 'x': 0.28, 'h': 0.28, 'r': 0.28,
    # Affricates — ~25-40% (also serves as the .get() default)
    'z': 0.35, 'c': 0.35, 'zh': 0.35, 'ch': 0.35, 'j': 0.35, 'q': 0.35,
}

# Regr. Case 44: maximum initial fraction per phone class.
# When MFA places the init→final boundary giving the initial MORE than
# this fraction of the word, the boundary is rejected and a proportional
# split is used instead.  Stops and nasals get tighter caps; fricatives
# and affricates (which have longer acoustic realisations) get more room.
_INIT_MAX_FRAC: dict[str, float] = {
    # Stops — shouldn't exceed 35% of syllable in normal speech
    'b': 0.35, 'd': 0.35, 'g': 0.35,
    # Aspirated stops — up to 40%
    'p': 0.40, 't': 0.40, 'k': 0.40,
    # Nasals / laterals — up to 40%
    'm': 0.40, 'n': 0.40, 'l': 0.40,
    # Fricatives — can be sustained, up to 50%
    'f': 0.50, 's': 0.50, 'sh': 0.50, 'x': 0.50, 'h': 0.50, 'r': 0.50,
    # Affricates — up to 45% (also default)
    'z': 0.45, 'c': 0.45, 'zh': 0.45, 'ch': 0.45, 'j': 0.45, 'q': 0.45,
}


def _proportional_initial_split(xmin: float, xmax: float,
                                initial_fraction: float) -> float:
    """Return a split that leaves a physically usable final segment.

    At least 30 ms is retained for each side of a normal word.  In the
    30--60 ms range the only safe floor is half the word, so a fallback cannot
    squeeze the final into a sub-15 ms fragment.  Sub-30 ms words remain on
    the midpoint path and are vetoed by the existing short-word contract.
    """
    word_duration = max(0.0, float(xmax) - float(xmin))
    if word_duration < 0.030:
        return float(xmin) + word_duration * 0.5
    segment_floor = 0.030 if word_duration >= 0.060 else word_duration * 0.5
    split = float(xmin) + max(segment_floor, word_duration * initial_fraction)
    return min(split, float(xmax) - segment_floor)

# IPA -> Pinyin reverse-mapped phone tier
# ---------------------------------------------------------------------------

def build_pinyin_phones_tier(phones_tier: Tier,
                              ipa_to_pinyin: dict[str, str],
                              words_tier: Tier | None = None,
                              pinyin_dict: dict[str, list[str]] | None = None,
                              en_mfa_windows: dict[tuple[str, float], tuple[float, float]] | None = None) -> Tier:
    """Build pinyin_phones tier using fullpinyin dict's initial+final format.

    For each word, look up the fullpinyin dict entry (e.g. pao4 -> [p, ao4]),
    then use MFA phone boundaries to split the word interval into the dict's
    phone segments.  Punctuation and silence pass through unchanged.

    When *en_mfa_windows* is provided, English word phones are filtered
    to only include those within the English MFA alignment time window,
    preventing neighbouring Chinese phones from leaking into English ranges.
    Keys are ``(word_text_lower, rounded_start_time)`` tuples to support
    duplicate English words within the same utterance (Regression Case 32).
    """
    if words_tier is None or pinyin_dict is None:
        # Fallback: 1:1 IPA->pinyin mapping
        return _build_pinyin_phones_1to1(phones_tier, ipa_to_pinyin)

    new_intervals = []
    phone_idx = 0
    mfa_phones = phones_tier.intervals

    for w_iv in words_tier.intervals:
        word = w_iv.text.strip().lower()
        if is_silence(w_iv.text) or not word or word in ("", "<eps>"):
            # Silence / empty: copy matching phone intervals
            dur_label = silence_label(w_iv.duration)
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dur_label))
            # Skip past phones in this silence range
            while phone_idx < len(mfa_phones) and mfa_phones[phone_idx].xmax <= w_iv.xmax + 0.001:
                phone_idx += 1
            continue

        # Collect MFA phones that fall within this word interval
        word_phones = []
        while phone_idx < len(mfa_phones) and mfa_phones[phone_idx].xmin < w_iv.xmax - 0.001:
            p = mfa_phones[phone_idx]
            if p.xmax > w_iv.xmin + 0.001:
                word_phones.append((max(p.xmin, w_iv.xmin), min(p.xmax, w_iv.xmax), p.text))
            phone_idx += 1

        # ── Filter out leaking phones from adjacent words ──
        # When MFA aligns a word as silence/spn (common for NVV tokens and
        # OOV English words), the only phones in its range are fragments of
        # the next word's first phone.  These fragments don't belong to this
        # word.  Detect: filter out all non-silence phones whose start is
        # more than 30% past the word's own start.
        if word_phones and not is_punct(w_iv.text):
            w_dur = w_iv.xmax - w_iv.xmin
            if w_dur > 0.06:
                # Find the first non-silence phone
                real_phones = [(s, e, t) for s, e, t in word_phones
                               if not is_silence(t)]
                if real_phones and real_phones[0][0] > w_iv.xmin + w_dur * 0.30:
                    # The first real phone starts well into the word — phones
                    # before it were all silence/spn.  Remove the leaking ones.
                    # Keep only silence labels that are fully within the word.
                    word_phones = [(s, e, t) for s, e, t in word_phones
                                   if is_silence(t) and s >= w_iv.xmin - 0.001
                                   and e <= w_iv.xmax + 0.001]

        # ── Look up dict entry for this word (before empty-phone check
        #     so we can fall back to a proportional split even when MFA
        #     produced zero or only one phone for a multi-phone syllable).
        #     Regression Case 26. ──
        dict_phones = None
        for key in pinyin_dict:
            if key.lower() == word:
                dict_phones = pinyin_dict[key]
                break

        if not word_phones:
            # No MFA phones in this word interval.
            # When the dict has initial+final, split the interval
            # proportionally instead of using the whole word as a
            # single phone (which would lose the initial–final split).
            # Regression Case 26 (FULL_WORD_AS_PHONE).
            if (dict_phones and len(dict_phones) >= 2
                    and not is_punct(w_iv.text)
                    and not is_nvv_token(w_iv.text)
                    and not is_english_token(w_iv.text)):
                word_dur = w_iv.xmax - w_iv.xmin
                _init_frac = _INIT_FRAC.get(dict_phones[0], 0.35)
                split = _proportional_initial_split(
                    w_iv.xmin, w_iv.xmax, _init_frac)
                new_intervals.append(Interval(w_iv.xmin, split, dict_phones[0]))
                final_label = " ".join(dict_phones[1:]) if len(dict_phones) > 2 else dict_phones[1]
                new_intervals.append(Interval(split, w_iv.xmax, final_label))
            else:
                new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, word))
            continue

        # Punctuation: pass through as-is
        if is_punct(w_iv.text):
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
            continue

        # NVV token: one self-referential phone — normalize to <UPPERCASE>
        if is_nvv_token(w_iv.text):
            nvv_text = f"<{w_iv.text.strip().strip('<>').upper()}>"
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, nvv_text))
            continue

        # English token: use phoneme intervals if available, else self-reference.
        # English token: all phones within an English word interval are
        # treated as English MFA IPA.  The word's language tag is the
        # authoritative signal — do NOT fall back to phone-level regex
        # heuristics (which misclassify e.g. "m" as Chinese pinyin).
        if is_english_token(w_iv.text):
            # ── Regr. Case 37: en_mfa_windows keyed by (word_text, start_time)
            #     so duplicate English words in the same utterance don't
            #     overwrite each other. ──
            # Separate en:-prefixed phones (injected by _apply_en_phones) from
            # raw IPA phones.  en:-prefixed phones are ALWAYS kept — they were
            # already vetted by _apply_en_phones and their boundaries are
            # proportionally scaled from English MFA alignment.
            en_prefixed = [(s, e, t) for s, e, t in word_phones
                           if t.startswith(EN_PHONE_PREFIX)]
            other_phones = [(s, e, t) for s, e, t in word_phones
                           if not t.startswith(EN_PHONE_PREFIX) and not is_silence(t)]
            sil_phones = [(s, e, t) for s, e, t in word_phones
                         if is_silence(t)]

            if en_prefixed:
                # en: phones are authoritative — they came from _apply_en_phones
                # which already scaled English MFA timing to the CTC-snapped
                # word boundaries.  Use them directly.
                word_phones = sil_phones + en_prefixed
            elif other_phones and en_mfa_windows:
                # Legacy path: no en: prefix, filter by MFA alignment window
                wl = w_iv.text.strip().lower()
                # Search time-qualified keys for a matching window
                matched_window = None
                w_start_rounded = round(w_iv.xmin, 2)
                for (key_wl, key_ts), (es, ee) in en_mfa_windows.items():
                    if key_wl == wl and abs(key_ts - w_start_rounded) < 0.5:
                        matched_window = (es, ee)
                        break
                # Fallback: try bare text key (backward compat with old data)
                if matched_window is None and wl in en_mfa_windows:
                    # Type guard: only unpack if it looks like a bare string key
                    val = en_mfa_windows.get(wl)  # type: ignore[arg-type]
                    if isinstance(val, tuple) and len(val) == 2:
                        matched_window = val

                if matched_window:
                    es, ee = matched_window
                    other_phones = [
                        (s, e, t) for s, e, t in other_phones
                        if s >= es - 0.3 and e <= ee + 0.3
                        and not _looks_chinese_phone(t)
                    ]
                else:
                    # No MFA window — keep only non-Chinese-looking phones
                    other_phones = [(s, e, t) for s, e, t in other_phones
                                    if not _looks_chinese_phone(t)]
                word_phones = sil_phones + other_phones
            elif other_phones:
                # No en_mfa_windows available — keep non-Chinese-looking phones
                other_phones = [(s, e, t) for s, e, t in other_phones
                                if not _looks_chinese_phone(t)]
                word_phones = sil_phones + other_phones
            else:
                # No phones at all — will fall through to self-reference
                word_phones = sil_phones

            if word_phones:
                for s, e, txt in word_phones:
                    if is_silence(txt):
                        new_intervals.append(Interval(s, e, txt))
                    elif txt.startswith(EN_PHONE_PREFIX):
                        new_intervals.append(Interval(s, e, en_ipa_to_arpabet(txt)))
                    else:
                        # English phone -> ARPABET with en: prefix
                        label = en_ipa_to_arpabet(f"{EN_PHONE_PREFIX}{txt}")
                        if label:  # skip empty mappings (glottal stop)
                            new_intervals.append(Interval(s, e, label))
                continue
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
            continue

        if dict_phones and len(dict_phones) >= 1:
            # Initial + final from fullpinyin dict
            if len(dict_phones) == 1:
                # Zero-initial (e.g. 'a5'): single dict phone for entire interval
                new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dict_phones[0]))
            else:
                # dict_phones >= 2: needs initial + final split
                word_dur = w_iv.xmax - w_iv.xmin

                # ── Try MFA phone boundary first ──
                use_mfa_split = False
                # Guard: when the leakage filter (line 516-528) stripped all
                # real phones and only silence/spn entries remain, do NOT use
                # silence boundaries for the initial/final split — that produces
                # garbage timing (e.g. 5ms "ch" + 355ms "ang4").  Fall back to
                # the proportional split below (Regr. Case 26/43).
                _real_phones = [(s, e, t) for s, e, t in word_phones
                                if not is_silence(t) and t != "spn"]
                if len(word_phones) >= 2 and _real_phones:
                    _init_end = word_phones[0][1]
                    _final_start = word_phones[1][0]
                    _init_frac_mfa = (_init_end - w_iv.xmin) / max(word_dur, 0.001)
                    # Regr. Case 44: phonetically-motivated upper bound on
                    # initial fraction.  MFA sometimes places the init→final
                    # boundary too far into the word (e.g. h→ao at 70%).
                    _init_max_frac = _INIT_MAX_FRAC.get(dict_phones[0], 0.55)
                    _init_min_dur = (
                        0.030 if word_dur >= 0.060
                        else word_dur * 0.5 if word_dur >= 0.030
                        else 0.0)
                    _candidate_init_dur = _init_end - w_iv.xmin
                    _candidate_final_dur = w_iv.xmax - _final_start
                    _candidate_contiguous = abs(_final_start - _init_end) <= AXIS_EPS
                    if ((_init_frac_mfa <= _init_max_frac or word_dur <= 0.060)
                            and _candidate_contiguous
                            and _candidate_init_dur >= _init_min_dur
                            and _candidate_final_dur >= _init_min_dur):
                        use_mfa_split = True
                        # Snap initial start to word start (Regression Case 7)
                        new_intervals.append(Interval(w_iv.xmin, _init_end, dict_phones[0]))
                        final_label = " ".join(dict_phones[1:]) if len(dict_phones) > 2 else dict_phones[1]
                        # A valid MFA candidate is contiguous within AXIS_EPS.
                        # Reuse one exact split for both output intervals so a
                        # sub-frame gap/overlap cannot survive serialization.
                        new_intervals.append(Interval(_init_end, w_iv.xmax, final_label))

                if not use_mfa_split:
                    # Proportional split fallback: dict_phones >= 2 but
                    # MFA under-produced or boundary was rejected.
                    # Regression Case 26 (MISSING_FINAL) + Case 43.
                    _init_frac = _INIT_FRAC.get(dict_phones[0], 0.35)
                    split = _proportional_initial_split(
                        w_iv.xmin, w_iv.xmax, _init_frac)
                    new_intervals.append(Interval(w_iv.xmin, split, dict_phones[0]))
                    final_label = " ".join(dict_phones[1:]) if len(dict_phones) > 2 else dict_phones[1]
                    new_intervals.append(Interval(split, w_iv.xmax, final_label))
        else:
            # Fallback: 1:1 IPA->pinyin
            for s, e, txt in word_phones:
                new_intervals.append(Interval(s, e, ipa_to_pinyin.get(txt, txt)))

    return Tier("pinyin_phones", phones_tier.xmin, phones_tier.xmax, new_intervals)


def _build_pinyin_phones_1to1(phones_tier: Tier, ipa_to_pinyin: dict[str, str]) -> Tier:
    """Fallback: 1:1 IPA->pinyin mapping when words_tier/pinyin_dict unavailable."""
    new_intervals = []
    for iv in phones_tier.intervals:
        txt = iv.text.strip()
        if is_silence(txt):
            new_intervals.append(Interval(iv.xmin, iv.xmax, silence_label(iv.duration)))
        else:
            new_intervals.append(Interval(iv.xmin, iv.xmax, ipa_to_pinyin.get(txt, txt)))
    return Tier("pinyin_phones", phones_tier.xmin, phones_tier.xmax, new_intervals)


def _register_suspicious_alignment(align_issues: list[dict],
                                   filter_reasons: list[str]) -> list[dict]:
    """Finalize phone-QC issues once all Phase 5 producers have run."""
    unique: list[dict] = []
    seen: set[str] = set()
    for issue in align_issues:
        if not isinstance(issue, dict):
            continue
        key = json.dumps(issue, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    align_issues[:] = unique
    if unique and "suspicious_alignment" not in filter_reasons:
        filter_reasons.append("suspicious_alignment")
    return unique


def _find_internal_pp_gaps(pp_tier: Tier | None, words_tier: Tier | None,
                           threshold_s: float = 0.010) -> list[dict]:
    """Return detailed pinyin-phone gaps that fall inside one content word."""
    if pp_tier is None or words_tier is None:
        return []

    content_ranges = [
        (iv.xmin, iv.xmax, iv.text.strip())
        for iv in words_tier.intervals
        if (iv.text.strip() and not is_silence(iv.text)
            and not is_english_token(iv.text.strip()))
    ]
    threshold_ticks = _threshold_ticks(threshold_s)
    gaps: list[dict] = []
    for left, right in zip(pp_tier.intervals, pp_tier.intervals[1:]):
        gap_ticks = _duration_ticks(left.xmax, right.xmin)
        if gap_ticks <= threshold_ticks:
            continue
        # A gap at a word boundary is legal sparse-tier coverage.  Require
        # both endpoints to be strictly inside the same content word; the
        # 1 ms tolerance prevents a boundary phone ending at 10.830 and the
        # next phone starting at 10.830/10.890 from being assigned to either
        # adjacent word (fix02).
        owner = next((row for row in content_ranges
                      if row[0] + 0.001 < left.xmax
                      and right.xmin < row[1] - 0.001), None)
        if owner is None:
            continue
        gaps.append({
            "start_s": round(left.xmax, 6),
            "end_s": round(right.xmin, 6),
            "duration_us": gap_ticks,
            "duration_ms": round(gap_ticks / 1000.0, 3),
            "left_phone": left.text.strip(),
            "right_phone": right.text.strip(),
            "word": owner[2],
        })
    return gaps


def _count_internal_pp_gaps(pp_tier: Tier | None, words_tier: Tier | None,
                            threshold_s: float = 0.010) -> int:
    """Count pinyin-phone gaps that fall inside one content-word interval.

    ``pinyin_phones`` is a sparse acoustic tier: a real pause between words
    may have no phone interval after later boundary caps.  That is not a tier
    discontinuity.  Only an uncovered gap inside one non-silence word means
    the word's phone reconstruction lost coverage.
    """
    return len(_find_internal_pp_gaps(pp_tier, words_tier, threshold_s))


def _collect_tier_discontinuities(textgrid: TextGrid,
                                  words_tier: Tier | None,
                                  threshold_s: float = 0.010) -> list[str]:
    """Return structural discontinuities in final, user-facing tiers.

    Raw text and pinyin are single full-span intervals.  ``phones`` is an
    internal MFA tier dropped from the final TextGrid.  ``pinyin_phones`` is
    intentionally sparse across natural pauses, so only gaps inside a content
    word are relevant there.  Treating all sparse-tier gaps as failures made
    normal pauses look like systemic alignment collapse.
    """
    discontinuities: list[str] = []
    for tier_name in ("hanzi", "words"):
        tier = tier_by_name(textgrid, tier_name)
        if tier is None or len(tier.intervals) < 5:
            continue
        # Match Case 35's semantic boundary rule: punctuation and explicit
        # silence own their adjacent timing space.  A structural hole exists
        # only when two neighboring content intervals leave time uncovered.
        gaps = sum(
            1
            for left, right in zip(tier.intervals, tier.intervals[1:])
            if (right.xmin - left.xmax > threshold_s
                and left.text.strip() and right.text.strip()
                and not is_silence(left.text) and not is_silence(right.text)
                and not is_punct(left.text.strip()) and not is_punct(right.text.strip()))
        )
        if gaps > len(tier.intervals) * 0.10:
            discontinuities.append(f"{tier.name}({gaps}/{len(tier.intervals)})")

    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if pp_tier is not None and len(pp_tier.intervals) >= 5:
        gaps = _count_internal_pp_gaps(pp_tier, words_tier, threshold_s)
        if gaps > len(pp_tier.intervals) * 0.10:
            discontinuities.append(f"{pp_tier.name}({gaps}/{len(pp_tier.intervals)})")
    return discontinuities


def _tier_desync_counts(hanzi_tier: Tier | None,
                         words_tier: Tier | None) -> tuple[int, int]:
    """Compare semantic hanzi/words units without splitting English words.

    CJK still requires exactly one tone-number pinyin word.  English source
    labels may be hyphenated (for example ``K-Pop``), while MFA/CTC can emit
    their contiguous fragments as ``kp`` and ``op``.  Compare that one
    semantic English unit to the normalized concatenation so it cannot shift
    every following CJK pair.  Punctuation and silence do not carry lexical
    identity and are intentionally absent from this check.
    """
    if hanzi_tier is None or words_tier is None:
        return 0, 0

    def content(tier: Tier) -> list[str]:
        return [iv.text.strip() for iv in tier.intervals
                if iv.text.strip() and not is_silence(iv.text)
                and not is_punct(iv.text.strip())]

    def english_key(text: str) -> str:
        return "".join(char.casefold() for char in text
                       if char.isascii() and char.isalpha())

    hanzi, words = content(hanzi_tier), content(words_tier)
    h_index = w_index = mismatches = semantic_words = 0
    while h_index < len(hanzi):
        h_text = hanzi[h_index]
        semantic_words += 1
        if is_cjk(h_text):
            if w_index >= len(words) or not re.fullmatch(r"[a-z]+[1-5]", words[w_index]):
                mismatches += 1
            else:
                w_index += 1
        elif is_english_token(h_text):
            expected = english_key(h_text)
            actual = ""
            while (w_index < len(words) and is_english_token(words[w_index])
                   and len(actual) < len(expected)):
                actual += english_key(words[w_index])
                w_index += 1
            if actual != expected:
                mismatches += 1
        else:
            if w_index >= len(words) or words[w_index].casefold() != h_text.casefold():
                mismatches += 1
            else:
                w_index += 1
        h_index += 1
    extra_words = len(words) - w_index
    mismatches += extra_words
    return mismatches, max(len(hanzi), semantic_words + extra_words)


def _align_reference_punctuation(ref_punct: list[dict],
                                 observed_punct: list[dict]) -> tuple[list[tuple[int, int]], list[dict], list[dict]]:
    """Align displayed punctuation without turning omission into misownership.

    Reference punctuation is allowed to be missing.  Therefore matching from
    each reference mark to the first later observed mark is incorrect when
    repeated labels are present: ``，！...！`` can leave only the final ``！``
    and that mark must be matched to the final reference occurrence.  This
    dynamic-programming alignment keeps both sequences ordered, prefers an
    exact lexical boundary, and only falls back to an observed ``extra`` mark
    when no ordered reference occurrence can explain it.
    """
    ref_count = len(ref_punct)
    obs_count = len(observed_punct)

    @lru_cache(maxsize=None)
    def solve(obs_index: int, ref_start: int):
        if obs_index >= obs_count:
            return (0, 0, 0, ())

        candidates = []
        observed = observed_punct[obs_index]
        for ref_index in range(ref_start, ref_count):
            reference = ref_punct[ref_index]
            if observed["label"] != reference["label"]:
                continue
            tail = solve(obs_index + 1, ref_index + 1)
            boundary_error = int(observed["boundary"] != reference["boundary"])
            distance = abs(observed["boundary"] - reference["boundary"])
            candidates.append((
                tail[0],
                tail[1] + boundary_error,
                tail[2] + distance,
                ((obs_index, ref_index),) + tail[3],
            ))

        # If this observed mark cannot be explained by a later reference
        # occurrence, it is genuinely extra.  The extra-count component is
        # deliberately ordered before boundary distance so an out-of-order
        # mark remains a boundary error rather than being hidden as omission.
        tail = solve(obs_index + 1, ref_start)
        candidates.append((tail[0] + 1, tail[1], tail[2],
                           ((obs_index, None),) + tail[3]))
        return min(candidates, key=lambda item: item[:3])

    mapping = solve(0, 0)[3]
    matched = [(ref_index, obs_index)
               for obs_index, ref_index in mapping
               if ref_index is not None]
    used_ref = {ref_index for ref_index, _ in matched}
    used_observed = {obs_index for _, obs_index in matched}
    missing = [{"index": ref_index, **ref_punct[ref_index]}
               for ref_index in range(ref_count)
               if ref_index not in used_ref]
    extra = [dict(index=obs_index, **observed_punct[obs_index])
             for obs_index in range(obs_count)
             if obs_index not in used_observed]
    matched.sort()
    return matched, missing, extra


def _publication_contract_audit(
        words_tier: Tier | None,
        hanzi_tier: Tier | None,
        pinyin_phones_tier: Tier | None,
        phones_tier: Tier | None,
        reference_text: str,
        source_words: list[dict] | None,
        ctc_tokens: list[dict] | None,
        reference_authoritative: bool,
        english_provenance: dict | None = None,
        unknown_recovery_proof: dict | None = None,
        fallback_correspondence: dict | None = None,
        reference_mode: str | None = None,
        raw_text_tier: Tier | None = None,
        pinyin_tier: Tier | None = None,
        fallback_surface_ledger: dict | None = None,
        fallback_punctuation_projection: dict | None = None
        ) -> tuple[list[str], dict]:
    """Audit final publication ownership without repairing uncertainty.

    This is deliberately a veto, not a normalizer.  Words/hanzi must be a
    complete display partition; every source/derived phone must have exactly
    one word owner; source CTC lexical order and spans must agree exactly; and
    reference punctuation/semantic ownership must be local.  The numeric
    ``AXIS_EPS`` is used only for serialized-axis ticks, never to turn an
    unowned interval into a publishable owner.
    """
    reasons: list[str] = []
    details: dict[str, object] = {}
    fallback_pause_gate = _fallback_pause_qualification(
        words_tier, reference_mode, fallback_correspondence,
        source_words, ctc_tokens)
    if fallback_pause_gate["pause_count"]:
        details["fallback_pause_qualification"] = fallback_pause_gate

    def _authority_evidence_projection(final_items, source_items, ctc_items):
        """Project ordered fragment evidence into authority semantic owners."""
        def value(item):
            if isinstance(item, dict):
                return item.get("text", item.get("word", ""))
            return item

        semantic = [item for item in project_authority_semantics(reference_text)
                    if item["kind"] != "punct"]
        if not any(item["kind"] == "english" for item in semantic):
            return None
        if len(final_items) != len(semantic):
            return None
        cursors = [0, 0]
        groups = []
        for expected, final in zip(semantic, final_items):
            observed = final.text.strip()
            if expected["kind"] == "english":
                if (re.sub(r"[^a-z0-9]", "", observed.casefold())
                        != re.sub(r"[^a-z0-9]", "", expected["surface"].casefold())):
                    return None
            elif expected["kind"] == "cjk":
                if not is_pinyin_syllable(observed):
                    return None
            for slot, items in enumerate((source_items, ctc_items)):
                while (cursors[slot] < len(items)
                       and is_punct(str(value(items[cursors[slot]])))):
                    cursors[slot] += 1
                start = cursors[slot]
                if expected["kind"] == "english":
                    compact = ""
                    while cursors[slot] < len(items):
                        item = items[cursors[slot]]
                        item_value = str(value(item))
                        if not is_english_fragment_token(item_value):
                            break
                        compact += re.sub(r"[^a-z0-9]", "", item_value.casefold())
                        cursors[slot] += 1
                        target = re.sub(r"[^a-z0-9]", "", expected["surface"].casefold())
                        if compact == target:
                            break
                    if compact != re.sub(r"[^a-z0-9]", "", expected["surface"].casefold()):
                        return None
                else:
                    if cursors[slot] >= len(items):
                        return None
                    item_value = str(value(items[cursors[slot]]))
                    if expected["kind"] == "cjk" and not is_pinyin_syllable(item_value):
                        return None
                    cursors[slot] += 1
                groups.append((expected["unit_id"], final.text.strip(), start,
                               cursors[slot], slot))
        if any(cursors[slot] < len(items)
               and not all(is_punct(str(value(item)))
                           for item in items[cursors[slot]:])
               for slot, items in enumerate((source_items, ctc_items))):
            return None
        return groups

    def add(reason: str, value: object) -> None:
        if reason not in reasons:
            reasons.append(reason)
        details[reason] = value

    def _partition(tier: Tier | None, name: str, *, full_axis: bool) -> None:
        if tier is None or not tier.intervals:
            add(f"{name}_missing", {"count": 0})
            return
        rows = tier.intervals
        local: list[dict] = []
        for index, iv in enumerate(rows):
            if (not math.isfinite(iv.xmin) or not math.isfinite(iv.xmax)
                    or iv.xmax <= iv.xmin):
                local.append({"index": index, "label": iv.text,
                              "start_s": iv.xmin, "end_s": iv.xmax})
                continue
            if iv.xmin < tier.xmin - AXIS_EPS or iv.xmax > tier.xmax + AXIS_EPS:
                local.append({"index": index, "label": iv.text,
                              "start_s": iv.xmin, "end_s": iv.xmax})
            if index:
                delta = iv.xmin - rows[index - 1].xmax
                # Derived phone tiers are intentionally sparse across
                # punctuation/inter-word gaps; only overlap is a structural
                # violation there.  Words/hanzi are the complete display
                # partition and must also reject holes.
                if delta < -AXIS_EPS or (full_axis and delta > AXIS_EPS):
                    local.append({"index": index, "gap_s": round(max(delta, 0.0), 6),
                                  "overlap_s": round(max(-delta, 0.0), 6)})
        if full_axis:
            if rows[0].xmin > tier.xmin + AXIS_EPS:
                local.append({"edge": "head", "gap_s": rows[0].xmin - tier.xmin})
            if rows[-1].xmax < tier.xmax - AXIS_EPS:
                local.append({"edge": "tail", "gap_s": tier.xmax - rows[-1].xmax})
            interior_silence = [
                {"index": index, "label": iv.text.strip(),
                 "start_s": iv.xmin, "end_s": iv.xmax}
                for index, iv in enumerate(rows)
                if _is_substantive_interior_silence(rows, index)
            ]
            if interior_silence:
                # This is evidence for strict filtering, not a publishable
                # owner.  Keep it separate from structural gaps so a
                # materialized silence can never be mistaken for repair.
                details["strict_interior_sp"] = interior_silence[:20]
                # Correspondence proves lexical neighbors only.  It can
                # never redeem a retained substantive silence.
                add("strict_interior_sp", interior_silence[:20])
        if local:
            add(f"{name}_owner_partition_mismatch", local[:20])

    _partition(words_tier, "words", full_axis=True)
    _partition(hanzi_tier, "hanzi", full_axis=True)
    _partition(pinyin_phones_tier, "pinyin_phones", full_axis=False)
    _partition(phones_tier, "phones", full_axis=False)

    # Surface tiers are derived publication artifacts, not an earlier
    # snapshot.  Audit their full-span shape and punctuation sequence against
    # the frozen words owner so stale raw_text/pinyin cannot pass silently.
    word_punctuation = [iv.text.strip() for iv in (words_tier.intervals
                       if words_tier is not None else [])
                       if is_punct(iv.text) and not is_silence(iv.text)]
    surface_punctuation = word_punctuation
    fallback_surface_valid = False
    if reference_mode == "fallback" and fallback_surface_ledger is not None:
        surface_valid, surface_details = (
            _validate_fallback_punctuation_surface_ledger(
                fallback_surface_ledger))
        fallback_surface_valid = surface_valid
        details["fallback_surface_authority"] = surface_details
        if not surface_valid:
            add("fallback_surface_ledger_invalid", surface_details)
        else:
            surface_punctuation = [
                str(item["label"]).strip()
                for item in fallback_surface_ledger.get("punctuation", [])]
    for tier, name in ((raw_text_tier, "raw_text"),
                       (pinyin_tier, "pinyin")):
        # Direct callers that predate the surface-tier audit may omit these
        # optional views.  The production call supplies both explicitly.
        if tier is None:
            continue
        if len(tier.intervals) != 1:
            add(f"{name}_publication_shape_mismatch", {
                "intervals": 0 if tier is None else len(tier.intervals)})
            continue
        interval = tier.intervals[0]
        if (abs(interval.xmin - tier.xmin) > AXIS_EPS
                or abs(interval.xmax - tier.xmax) > AXIS_EPS):
            add(f"{name}_publication_shape_mismatch", {
                "span": [interval.xmin, interval.xmax],
                "axis": [tier.xmin, tier.xmax]})
        observed = _surface_punctuation(tier)
        if observed != surface_punctuation:
            add(f"{name}_punctuation_sequence_mismatch", {
                "source": surface_punctuation, "words": word_punctuation,
                "observed": observed})

    lineage_invalid = getattr(phones_tier, "_phone_lineage_invalid", None)
    if lineage_invalid:
        add("phone_lineage_ambiguous", lineage_invalid)

    if words_tier is not None and hanzi_tier is not None:
        if len(words_tier.intervals) != len(hanzi_tier.intervals):
            add("words_hanzi_bounds_mismatch", {
                "words": len(words_tier.intervals),
                "hanzi": len(hanzi_tier.intervals),
            })
        else:
            mismatches: list[dict] = []
            for index, (word, label) in enumerate(
                    zip(words_tier.intervals, hanzi_tier.intervals)):
                if (abs(word.xmin - label.xmin) > AXIS_EPS
                        or abs(word.xmax - label.xmax) > AXIS_EPS):
                    mismatches.append({"index": index,
                                       "word": [word.xmin, word.xmax],
                                       "hanzi": [label.xmin, label.xmax]})
                    continue
                word_text, label_text = word.text.strip(), label.text.strip()
                if ((is_silence(word_text) or is_punct(word_text))
                        and word_text != label_text):
                    mismatches.append({"index": index, "word": word_text,
                                       "hanzi": label_text})
                elif (is_english_token(word_text)
                      and word_text.casefold() != label_text.casefold()):
                    mismatches.append({"index": index, "word": word_text,
                                       "hanzi": label_text})
            if mismatches:
                add("words_hanzi_bounds_mismatch", mismatches[:20])

    # Every source and derived phone must be wholly owned by one final words
    # interval.  A phone crossing a boundary is ambiguous even if its overlap
    # ratio happens to be high.
    if words_tier is not None:
        for tier_name, tier in (("phones", phones_tier),
                                ("pinyin_phones", pinyin_phones_tier)):
            if tier is None:
                continue
            bad: list[dict] = []
            for index, phone in enumerate(tier.intervals):
                owners = [word for word in words_tier.intervals
                          if phone.xmin >= word.xmin - AXIS_EPS
                          and phone.xmax <= word.xmax + AXIS_EPS]
                if len(owners) != 1:
                    overlaps = []
                    for owner_index, word in enumerate(words_tier.intervals):
                        overlap = min(phone.xmax, word.xmax) - max(
                            phone.xmin, word.xmin)
                        if overlap > AXIS_EPS:
                            overlaps.append({
                                "index": owner_index,
                                "label": word.text.strip(),
                                "overlap_s": round(overlap, 6),
                            })
                    overlaps.sort(key=lambda row: row["overlap_s"], reverse=True)
                    bad.append({"index": index, "label": phone.text,
                                "start_s": phone.xmin, "end_s": phone.xmax,
                                "owner_count": len(owners),
                                "candidate_owners": overlaps[:4]})
            if bad:
                add(f"{tier_name}_owner_mismatch", bad[:20])

        # A phone gap strictly inside one lexical word is not a natural
        # inter-word pause.  Preserve it as evidence for filtering.
        if pinyin_phones_tier is not None:
            internal: list[dict] = []
            for left, right in zip(pinyin_phones_tier.intervals,
                                   pinyin_phones_tier.intervals[1:]):
                owner = next((word for word in words_tier.intervals
                              if left.xmax > word.xmin + AXIS_EPS
                              and right.xmin < word.xmax - AXIS_EPS
                              and not is_silence(word.text)
                              and not is_punct(word.text)), None)
                if owner is not None and right.xmin - left.xmax > AXIS_EPS:
                    internal.append({"word": owner.text.strip(),
                                     "start_s": left.xmax,
                                     "end_s": right.xmin})
            if internal:
                add("pinyin_phones_internal_hole", internal[:20])

    if (reference_mode == "fallback" and words_tier is not None
            and isinstance(fallback_surface_ledger, dict)):
        projection_valid, projection_details = (
            _validate_fallback_punctuation_projection(
                fallback_punctuation_projection,
                fallback_surface_ledger.get("source_text"), words_tier,
                ctc_tokens)) if fallback_punctuation_projection is not None else (
                    False, {"status": "not_provided", "reasons": []})
        if projection_valid:
            expected = [{"label": str(item.get("label", "")).strip(),
                         "boundary": item.get("final_boundary"),
                        "source_boundary": item.get("source_boundary")}
                        for item in fallback_punctuation_projection.get(
                            "entries", [])]
            details["fallback_punctuation_projection_authority"] = (
                projection_details)
        else:
            expected = [{"label": str(item.get("label", "")).strip(),
                         "boundary": item.get("lexical_boundary")}
                        for item in fallback_surface_ledger.get("punctuation", [])]
            if fallback_punctuation_projection is not None:
                details["fallback_punctuation_projection_authority"] = (
                    projection_details)
        observed = []
        lexical_before = 0
        for interval in words_tier.intervals:
            label = interval.text.strip()
            if label and not is_silence(label) and not is_punct(label):
                lexical_before += 1
            elif is_punct(label) and not is_silence(label):
                observed.append({"label": label, "boundary": lexical_before,
                                 "interval": [interval.xmin, interval.xmax]})
        expected_pairs = [(item["label"], item["boundary"])
                          for item in expected]
        observed_pairs = [(item["label"], item["boundary"])
                          for item in observed]
        details["fallback_punctuation_projection"] = {
            "expected": expected,
            "observed": observed,
        }
        if observed_pairs != expected_pairs:
            add("fallback_punctuation_ownership_mismatch",
                details["fallback_punctuation_projection"])

    if (reference_mode == "fallback" and words_tier is not None
            and fallback_surface_valid
            and isinstance(fallback_surface_ledger, dict)):
        cross_kind_safe, cross_kind_details = (
            _fallback_cjk_cross_kind_owner_audit(
                fallback_surface_ledger["source_text"], words_tier,
                fallback_punctuation_projection, ctc_tokens))
        details["fallback_cjk_cross_kind_owner"] = cross_kind_details
        if not cross_kind_safe:
            add("fallback_cjk_cross_kind_owner_unproven", cross_kind_details)

    if reference_authoritative and words_tier is not None:
        ref_units = _extract_word_chars(reference_text)
        ref_punct = []
        ref_lexical_count = 0
        for unit in ref_units:
            if is_punct(unit):
                ref_punct.append({"label": unit.strip(),
                                  "boundary": ref_lexical_count})
            elif is_word_like(unit):
                ref_lexical_count += 1
        # A missing mark is not proof of an alignment error: punctuation is
        # often swallowed by a long pause or omitted by CTC.  Extra marks,
        # reordered marks, and a mark attached to the wrong lexical boundary
        # are errors because they change the displayed segmentation.
        observed_punct = []
        lexical = [iv for iv in words_tier.intervals
                   if iv.text.strip() and not is_silence(iv.text)
                   and not is_punct(iv.text)]
        for iv in words_tier.intervals:
            if not (is_punct(iv.text) and not is_silence(iv.text)):
                continue
            boundary = sum(1 for word in lexical
                           if word.xmax <= iv.xmin + AXIS_EPS)
            observed_punct.append({"label": iv.text.strip(),
                                   "boundary": boundary,
                                   "interval": [iv.xmin, iv.xmax]})

        matched, missing, extra = _align_reference_punctuation(
            ref_punct, observed_punct)
        boundary_errors = []
        for ref_index, obs_index in matched:
            expected = ref_punct[ref_index]
            actual = observed_punct[obs_index]
            if actual["boundary"] != expected["boundary"]:
                boundary_errors.append({
                    "reference_index": ref_index,
                    "observed_index": obs_index,
                    "label": actual["label"],
                    "expected_boundary": expected["boundary"],
                    "observed_boundary": actual["boundary"],
                    "interval": actual["interval"],
                })
        punct_projection = {
            "reference": [item["label"] for item in ref_punct],
            "observed": [item["label"] for item in observed_punct],
            "missing_allowed": missing,
            "extra": extra,
            "boundary_errors": boundary_errors,
        }
        details["reference_punctuation_projection"] = punct_projection
        if extra or boundary_errors:
            add("reference_punctuation_ownership_mismatch", punct_projection)
        punct_bad: list[dict] = []
        lexical = [iv for iv in words_tier.intervals
                   if iv.text.strip() and not is_silence(iv.text)
                   and not is_punct(iv.text)]
        for index, punct in enumerate(
                iv for iv in words_tier.intervals
                if is_punct(iv.text) and not is_silence(iv.text)):
            prev = next((word for word in reversed(lexical)
                         if word.xmax <= punct.xmin + AXIS_EPS), None)
            nxt = next((word for word in lexical
                        if word.xmin >= punct.xmax - AXIS_EPS), None)
            local_start = prev.xmax if prev is not None else words_tier.xmin
            local_end = nxt.xmin if nxt is not None else words_tier.xmax
            if (punct.xmin < local_start - AXIS_EPS
                    or punct.xmax > local_end + AXIS_EPS):
                punct_bad.append({"index": index, "label": punct.text,
                                  "interval": [punct.xmin, punct.xmax],
                                  "local_gap": [local_start, local_end]})
        if punct_bad:
            add("punctuation_local_owner_mismatch", punct_bad[:20])

    # CTC lexical identity is exact, but geometry is an evidence envelope:
    # final words may be reconstructed from the ordered source-MFA and CTC
    # owners, rather than being required to fit one raw CTC interval.  This
    # preserves provenance while allowing a boundary repair to use the
    # convex hull of its independent lexical evidence.
    current_lexical = [iv for iv in (words_tier.intervals if words_tier else [])
                       if iv.text.strip() and not is_silence(iv.text)
                       and not is_punct(iv.text)]
    # Keep the raw MFA snapshot immutable for diagnostics, but let a
    # separately verified recovery proof participate in the lexical contract.
    # Without this projection, a proven ``<unk> -> Mira`` recovery was still
    # compared as ``<unk>`` here and vetoed after all earlier processing.
    resolved_source_words = [dict(item) for item in (source_words or [])]
    if isinstance(unknown_recovery_proof, dict):
        source_interval = (unknown_recovery_proof.get("source", {})
                           .get("interval", {}))
        source_ordinal = source_interval.get("ordinal")
        resolved_word = (unknown_recovery_proof.get("ctc", {})
                         .get("token", {}).get("word", ""))
        if (type(source_ordinal) is int and isinstance(resolved_word, str)
                and resolved_word.strip()):
            for item in resolved_source_words:
                if item.get("ordinal") == source_ordinal:
                    item["raw_text"] = item.get("text", "")
                    item["text"] = resolved_word.strip()
                    break
            else:
                resolved_source_words = [dict(item) for item in (source_words or [])]
        details["ctc_lexical_source_resolution"] = {
            "schema": "ctc-lexical-source-resolution-v1",
            "raw_unknown": source_interval.get("text"),
            "resolved_text": resolved_word.strip() if isinstance(resolved_word, str) else "",
            "source_ordinal": source_ordinal,
            "proof_schema": unknown_recovery_proof.get("schema"),
        }
    source_lexical = [item for item in resolved_source_words
                      if str(item.get("text", "")).strip()
                      and not is_silence(str(item.get("text", "")))
                      and not is_punct(str(item.get("text", "")))]
    ctc_lexical: list[dict] = []
    for token in ctc_tokens or []:
        if not isinstance(token, dict) or str(token.get("type", "word")) != "word":
            continue
        text = str(token.get("word", "")).strip()
        if text and not is_silence(text) and not is_punct(text):
            ctc_lexical.append(token)
    lexical_evidence: list[dict] = []
    if current_lexical and (not source_lexical or not ctc_lexical):
        add("ctc_lexical_evidence_missing", {
            "current": len(current_lexical),
            "source": len(source_lexical), "ctc": len(ctc_lexical)})
    elif current_lexical:
        current_text = [_lexical_identity(iv.text) for iv in current_lexical]
        source_text = [_lexical_identity(item.get("text", ""))
                       for item in source_lexical]
        ctc_text = [_lexical_identity(item.get("word", ""), ctc_item=item)
                    for item in ctc_lexical]
        if (not reference_authoritative
                and isinstance(fallback_correspondence, dict)
                and fallback_correspondence.get("safe")):
            mapped_source = [entry for entry in
                             fallback_correspondence.get("entries", [])
                             if entry.get("status") == "mapped"]
            if len(mapped_source) == len(current_lexical):
                source_by_ordinal = {
                    item.get("ordinal"): item for item in (source_words or [])
                    if isinstance(item, dict)
                }
                source_lexical = [
                    {**source_by_ordinal.get(entry.get("source_ordinal"), {}),
                     "text": entry.get("resolved_text", "")}
                    for entry in mapped_source
                ]
                source_text = [_lexical_identity(entry.get("resolved_text", ""))
                               for entry in mapped_source]
                details["fallback_correspondence_projection"] = {
                    "digest": fallback_correspondence.get("digest"),
                    "mapped_count": len(mapped_source),
                }
        authority_projection = None
        if reference_authoritative:
            authority_projection = _authority_evidence_projection(
                current_lexical, source_lexical, ctc_lexical)
            if authority_projection is not None:
                details["ctc_lexical_evidence_projection"] = authority_projection
        if (authority_projection is None
                and (current_text != source_text or current_text != ctc_text)):
            add("ctc_lexical_sequence_mismatch", {
                "current": current_text[:20], "source": source_text[:20],
                "ctc": ctc_text[:20],
                "current_surface": [iv.text.strip() for iv in current_lexical[:20]],
                "source_surface": [str(item.get("text", "")).strip()
                                   for item in source_lexical[:20]],
                "ctc_surface": [str(item.get("word", "")).strip()
                                for item in ctc_lexical[:20]],
            })
        else:
            # Source MFA and CTC remain lexical identity/order evidence, but
            # neither span is a publication geometry fence.  The processed
            # words tier may deliberately move outside both raw spans after
            # CTC/MFA arbitration, energy refinement, punctuation ownership,
            # or silence compensation.  Geometry is retained below for
            # diagnostics only; it must not veto a valid processed owner.
            previous_word_end = -math.inf
            previous_source_start = -math.inf
            previous_ctc_start = -math.inf
            punctuation_intervals = [iv for iv in (words_tier.intervals if words_tier else [])
                                     if is_punct(iv.text) and not is_silence(iv.text)]
            authority_entries = _ctc_authority_entries(words_tier) or []
            for index, (word, source_item, token) in enumerate(
                    zip(current_lexical, source_lexical, ctc_lexical)):
                try:
                    ctc_start, ctc_end = float(token["start_s"]), float(token["end_s"])
                    source_start = float(source_item.get("start", source_item.get("xmin")))
                    source_end = float(source_item.get("end", source_item.get("xmax")))
                except (KeyError, TypeError, ValueError):
                    lexical_evidence.append({"index": index, "error": "invalid_span"})
                    continue
                envelope_start = min(source_start, ctc_start)
                envelope_end = max(source_end, ctc_end)
                source_overlap = min(word.xmax, source_end) - max(word.xmin, source_start)
                ctc_overlap = min(word.xmax, ctc_end) - max(word.xmin, ctc_start)
                final_word_order = word.xmin >= previous_word_end - AXIS_EPS
                source_order = source_start >= previous_source_start - AXIS_EPS
                ctc_order = ctc_start >= previous_ctc_start - AXIS_EPS
                punct_overlap = [punct.text.strip() for punct in punctuation_intervals
                                 if min(word.xmax, punct.xmax) - max(word.xmin, punct.xmin)
                                 > AXIS_EPS]
                authority = (authority_entries[index]
                             if index < len(authority_entries)
                             and isinstance(authority_entries[index], dict)
                             else {})
                ctc_authoritative = (
                    authority.get("boundary_source") == "ctc"
                    # Provenance can be lost when a late reference/geometry
                    # rebuild replaces the words tier.  If the final word
                    # has no MFA evidence at all but is positively contained
                    # in its CTC span, infer the same safe CTC ownership from
                    # geometry instead of rejecting the CTC reconstruction.
                    or (source_overlap <= AXIS_EPS
                        and ctc_overlap > AXIS_EPS))
                valid = (
                    all(math.isfinite(value) for value in
                        (ctc_start, ctc_end, source_start, source_end))
                    and ctc_end > ctc_start and source_end > source_start
                    # Raw CTC/MFA geometry is advisory.  The final processed
                    # interval is allowed to be compensated beyond either
                    # evidence span; only lexical order and punctuation
                    # ownership remain publication contracts here.
                    and final_word_order and source_order and ctc_order
                    and not punct_overlap)
                proof = {
                    "index": index,
                    "lexical_ordinal": index,
                    "word": word.text.strip(),
                    "word_span": [word.xmin, word.xmax],
                    "source_mfa_span": [source_start, source_end],
                    "ctc_span": [ctc_start, ctc_end],
                    "ctc_authoritative": ctc_authoritative,
                    "evidence_envelope": [envelope_start, envelope_end],
                    "source_overlap_s": max(0.0, source_overlap),
                    "ctc_overlap_s": max(0.0, ctc_overlap),
                    "punctuation_overlap": punct_overlap,
                    "monotone": final_word_order,
                    "final_word_order": final_word_order,
                    "source_order": source_order,
                    "ctc_order": ctc_order,
                    "accepted": valid,
                }
                previous_word_end = max(previous_word_end, word.xmax)
                previous_source_start = source_start
                previous_ctc_start = ctc_start
                if not valid:
                    lexical_evidence.append(proof)
                else:
                    lexical_evidence.append(proof)
            if lexical_evidence:
                # Geometry failures are diagnostic evidence only.  The
                # processed interval is allowed to be compensated outside
                # raw CTC/MFA spans; sequence/order and punctuation contracts
                # are enforced by the remaining checks above.
                # This is a publication binding, not a diagnostic preview.
                # The independent strict audit compares it one-for-one with
                # the final lexical words tier; truncating at 20 silently
                # rejected every longer utterance even when all spans were
                # valid.
                details["ctc_lexical_evidence_proof"] = lexical_evidence

    english_count = sum(1 for iv in current_lexical if is_english_token(iv.text))
    if current_lexical and english_count:
        if not isinstance(english_provenance, dict):
            add("english_provenance_missing", {"english_words": english_count})
        elif english_provenance.get("status") != "verified":
            add("english_provenance_rejected", english_provenance)
        else:
            if int(english_provenance.get("verified_words", 0)) < english_count:
                add("english_provenance_word_count_mismatch", {
                    "english_words": english_count,
                    "verified_words": english_provenance.get("verified_words")})
            if pinyin_phones_tier is not None:
                missing: list[str] = []
                for word in current_lexical:
                    if not is_english_token(word.text):
                        continue
                    owned = [phone.text.strip() for phone in pinyin_phones_tier.intervals
                             if phone.xmax > word.xmin + AXIS_EPS
                             and phone.xmin < word.xmax - AXIS_EPS
                             and not is_silence(phone.text)]
                    if not owned or any(not label.startswith(EN_PHONE_PREFIX)
                                        for label in owned):
                        missing.append(word.text.strip())
                if missing:
                    add("english_phone_owner_mismatch", missing[:20])

    return sorted(set(reasons)), details


def _record_filterable_qc(report: dict, filter_reasons: list[str],
                          enabled: bool, name: str, details) -> None:
    """Always retain diagnostics; filter only when quality filtering is on."""
    report[name] = details
    if enabled:
        filter_reasons.append(name)


def _resolve_spn(phone_iv: Interval, words_tier: Tier | None,
                 pinyin_dict: dict[str, list[str]] | None) -> str:
    """Find the word overlapping this spn phone interval and return its pinyin label."""
    if words_tier is None or pinyin_dict is None:
        return silence_label(phone_iv.duration)
    for w_iv in words_tier.intervals:
        if w_iv.xmin <= phone_iv.xmin < w_iv.xmax or phone_iv.xmin <= w_iv.xmin < phone_iv.xmax:
            word = w_iv.text.strip().lower()
            # Look up in pinyin dict (case-insensitive)
            for key in pinyin_dict:
                if key.lower() == word:
                    return ' '.join(pinyin_dict[key])
            break
    return silence_label(phone_iv.duration)


# ---------------------------------------------------------------------------
# Punctuation-silence cross-check: compare pinyin punctuation with actual
# silence gaps in the words tier, then produce a corrected Chinese text.
# ---------------------------------------------------------------------------

def handle_unexpected_silences(textgrid: TextGrid, pinyin_text: str) -> list[str]:
    """Diagnose unexpected inter-word silences without changing geometry.

    Silence ownership is resolved once, from the final visual words snapshot,
    by :func:`_resolve_visual_short_silence_merges`.  This historical helper
    remains part of the public processing API, but is intentionally
    diagnostic-only so an early ``<sp0>`` pass cannot consume the same gap a
    second time.
    """
    words_tier = tier_by_name(textgrid, "words")
    phones_tier = tier_by_name(textgrid, "phones")
    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if words_tier is None or phones_tier is None or pp_tier is None:
        return []

    pinyin_tokens = pinyin_text.split()
    word_items = [(iv.text.strip(), is_silence(iv.text)) for iv in words_tier.intervals]
    tg_word_idx = [i for i, (text, is_sil) in enumerate(word_items)
                   if not is_sil and not is_punct(text)]
    py_word_idx = [i for i, t in enumerate(pinyin_tokens) if is_word_like(t)]

    if len(tg_word_idx) != len(py_word_idx) or len(tg_word_idx) == 0:
        return []

    n = len(tg_word_idx)

    # Build gap_sil (only inter-word gaps, index 1..n-1 -> words k-1 -> k)
    gap_sil = [None] * n  # gap_sil[i] = silence label for gap BEFORE word i (i >= 1)
    for k in range(1, n):
        lo = tg_word_idx[k - 1] + 1
        hi = tg_word_idx[k]
        for j in range(lo, hi):
            if word_items[j][1]:
                gap_sil[k] = word_items[j][0]  # store the silence label
                break

    # Build gap_punct for same gaps
    gap_punct = [False] * n
    _punctuation_chars = '，。…！？、；：,.!?;:～'
    def _token_has_punctuation(token: str) -> bool:
        cleaned = re.sub(r"<sp[0-3]>", "", token, flags=re.IGNORECASE)
        return any(ch in _punctuation_chars for ch in cleaned)

    for k in range(1, n):
        lo = py_word_idx[k - 1] + 1
        hi = py_word_idx[k]
        gap_punct[k] = any(is_punct(pinyin_tokens[i]) or _token_has_punctuation(pinyin_tokens[i])
                            for i in range(lo, hi))

    filter_reasons = []
    for k in range(1, n):
        sil_label = gap_sil[k]
        has_punct = gap_punct[k]
        if sil_label is None or has_punct:
            continue
        if sil_label in ("<sp1>", "<sp2>", "<sp3>"):
            filter_reasons.append("unexpected_silence")
    return sorted(set(filter_reasons))


def absorb_nvv_trailing(textgrid: TextGrid) -> None:
    """NVV absorbs trailing punctuation + silence chain until next content word.

    MFA cannot acoustically model NVV tokens (LAUGHTER, BREATHING, …).
    Their boundaries are imprecise, and the audio between an NVV and the
    next real word — punctuation and silence — is actually part of the
    NVV (e.g. laughter tail).  This pass extends NVV ``xmax`` to absorb
    that chain, so ``mid_sp`` doesn't flag the orphaned intervals.

    Example::

        <LAUGHTER> [9.745-9.81]  ！ [9.81-9.815]  <sp2> [9.815-10.51]  bie2
        → <LAUGHTER> [9.745-10.51]  bie2

    Operates on words, phones, and pinyin_phones tiers in sync.
    """
    words_tier = tier_by_name(textgrid, "words")
    phones_tier = tier_by_name(textgrid, "phones")
    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if words_tier is None:
        return

    surface_ledger = getattr(
        textgrid, "_fallback_punctuation_surface_ledger", None)
    protected_boundaries = {
        item.get("lexical_boundary")
        for item in (surface_ledger.get("punctuation", [])
                     if isinstance(surface_ledger, dict) else [])
        if type(item.get("lexical_boundary")) is int
    }

    intervals = list(words_tier.intervals)
    to_delete_words: set[int] = set()
    to_delete_phones: set[int] = set()
    to_delete_pp: set[int] = set()
    for i in range(len(intervals)):
        if not is_nvv_token(intervals[i].text):
            continue
        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            # NVV may use CTC as its only valid acoustic boundary.  Do not
            # later absorb punctuation/silence and overwrite that anchor.
            continue

        # Absorb trailing punct + silence chain.
        j = i + 1
        nvv_boundary = sum(
            1 for interval in intervals[:i + 1]
            if interval.text.strip() and not is_silence(interval.text)
            and not is_punct(interval.text))
        absorbed_sil_ranges: list[tuple[float, float]] = []
        while j < len(intervals):
            text = intervals[j].text.strip()
            if is_punct(text):
                # A fallback source mark is a display owner, not disposable
                # NVV tail.  Leave it in place; the following D3 pass may
                # safely let that punctuation own adjacent silence.
                if nvv_boundary in protected_boundaries:
                    break
                j += 1
            elif is_silence(text) and text:
                absorbed_sil_ranges.append((intervals[j].xmin, intervals[j].xmax))
                j += 1
            else:
                break

        if j <= i + 1:
            continue  # Nothing to absorb.

        # Extend NVV to the start of the next content word.
        next_iv = intervals[j] if j < len(intervals) else None
        new_xmax = next_iv.xmin if next_iv else intervals[j - 1].xmax
        intervals[i] = Interval(intervals[i].xmin, new_xmax, intervals[i].text)

        # Mark punct + silence for deletion.
        for d in range(i + 1, j):
            to_delete_words.add(d)

        # Clean up matching silence from phones & pp tiers.
        for sil_xmin, sil_xmax in absorbed_sil_ranges:
            if phones_tier:
                for pi, p in enumerate(phones_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_xmin) < 0.01 \
                       and abs(p.xmax - sil_xmax) < 0.01:
                        to_delete_phones.add(pi)
                        break
            if pp_tier:
                for pi, p in enumerate(pp_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_xmin) < 0.01 \
                       and abs(p.xmax - sil_xmax) < 0.01:
                        to_delete_pp.add(pi)
                        break

    if not to_delete_words:
        return

    # Apply deletions.
    intervals = [iv for idx, iv in enumerate(intervals)
                 if idx not in to_delete_words]
    words_tier.intervals = intervals
    if phones_tier and to_delete_phones:
        phones_tier.intervals = [iv for idx, iv in enumerate(phones_tier.intervals)
                                 if idx not in to_delete_phones]
    if pp_tier and to_delete_pp:
        pp_tier.intervals = [iv for idx, iv in enumerate(pp_tier.intervals)
                             if idx not in to_delete_pp]

    # Clean up zero-duration remnants.
    for tier in (words_tier, phones_tier, pp_tier):
        if tier:
            tier.intervals = [iv for iv in tier.intervals
                              if iv.duration > 0.001 or not iv.text.strip()]


def _fix_overlapping_boundaries(words_tier) -> int:
    """Resolve overlaps between adjacent intervals.  Regr. Case 38, 52.

    Operates on *words_tier* intervals in-place.  Returns the number of
    overlaps that were fixed (so the caller can decide whether to re-sync
    derived tiers).

    Strategy
    --------
    * Two **content words** (or content + English) overlapping < 30 ms →
      split the overlap evenly.  English/NVV tokens are clipped to the
      content word's boundary (they lack MFA acoustic models, so their
      CTC boundaries are less precise).
    * Content word overlapping with **punctuation** → clip the punctuation
      side unconditionally (Regr. Case 52 — punct of any size leaking into
      content is always wrong).
    * Content-content overlaps ≥ 30 ms are **left untouched** — they will
      be caught by the downstream ``overlapping_words`` QC filter (Case 27-B).
    * Zero-duration remnants are removed after all fixes are applied.
    """
    intervals = list(words_tier.intervals)
    n = len(intervals)
    fixed = 0

    for i in range(n - 1):
        cur = intervals[i]
        nxt = intervals[i + 1]
        if cur.xmax is None or nxt.xmin is None:
            continue
        overlap = cur.xmax - nxt.xmin
        overlap_ticks = _duration_ticks(nxt.xmin, cur.xmax)
        if overlap_ticks <= 500:      # sub-0.5 ms — float noise, skip
            continue

        cur_text = cur.text.strip() if cur.text else ""
        nxt_text = nxt.text.strip() if nxt.text else ""

        cur_is_content = (cur_text and not is_punct(cur_text)
                          and not is_silence(cur_text))
        nxt_is_content = (nxt_text and not is_punct(nxt_text)
                          and not is_silence(nxt_text))
        cur_is_en_nvv = is_english_token(cur_text) or is_nvv_token(cur_text)
        nxt_is_en_nvv = is_english_token(nxt_text) or is_nvv_token(nxt_text)
        cur_ctc = _ctc_authoritative_ordinal(words_tier, i) is not None
        nxt_ctc = _ctc_authoritative_ordinal(words_tier, i + 1) is not None

        # ── Two content words with mild overlap (incl. English/NVV adjacent) ──
        # Regr. Case 38: when one side is English/NVV (no MFA acoustic model),
        # clip that side to the content word's boundary.
        if cur_is_content and nxt_is_content and overlap_ticks < 30_000:
            if cur_ctc and nxt_ctc:
                # Both owners came from CTC.  If a later punctuation or
                # geometry pass introduced an overlap, split only the CTC
                # overlap region; never use MFA duration or midpoint of the
                # mutated intervals as a new authority.
                entries = _ctc_authority_entries(words_tier) or []
                left_ord = _ctc_authoritative_ordinal(words_tier, i)
                right_ord = _ctc_authoritative_ordinal(words_tier, i + 1)
                left_span = entries[left_ord].get("resolved_span") if left_ord is not None and left_ord < len(entries) else None
                right_span = entries[right_ord].get("resolved_span") if right_ord is not None and right_ord < len(entries) else None
                if (isinstance(left_span, list) and len(left_span) == 2
                        and isinstance(right_span, list) and len(right_span) == 2
                        and left_span[1] > right_span[0]):
                    boundary = (float(left_span[1]) + float(right_span[0])) / 2.0
                else:
                    boundary = (cur.xmax + nxt.xmin) / 2.0
                intervals[i] = Interval(cur.xmin, boundary, cur.text)
                intervals[i + 1] = Interval(boundary, nxt.xmax, nxt.text)
            elif cur_ctc:
                # Preserve the CTC owner's end; move only the non-CTC word.
                intervals[i + 1] = Interval(cur.xmax, nxt.xmax, nxt.text)
            elif nxt_ctc:
                # Preserve the CTC owner's start; trim only the non-CTC word.
                intervals[i] = Interval(cur.xmin, nxt.xmin, cur.text)
            elif cur_is_en_nvv and not nxt_is_en_nvv:
                # English/NVV → content: clip English/NVV end
                intervals[i] = Interval(cur.xmin, nxt.xmin, cur.text)
            elif nxt_is_en_nvv and not cur_is_en_nvv:
                # content → English/NVV: push English/NVV start forward
                intervals[i + 1] = Interval(cur.xmax, nxt.xmax, nxt.text)
            else:
                # Both content or both English/NVV: split evenly
                mid = (cur.xmax + nxt.xmin) / 2.0
                intervals[i] = Interval(cur.xmin, mid, cur.text)
                intervals[i + 1] = Interval(mid, nxt.xmax, nxt.text)
            fixed += 1

        # ── Content word followed by punctuation that leaks into it ──
        # Regr. Case 52: removed 100ms threshold — punct-content overlaps
        # of any size are always wrong and should be clipped.
        elif cur_is_content and is_punct(nxt_text):
            intervals[i + 1] = Interval(cur.xmax, nxt.xmax, nxt.text)
            fixed += 1

        # ── Punctuation leaking into following content word ──
        elif is_punct(cur_text) and nxt_is_content:
            intervals[i] = Interval(cur.xmin, nxt.xmin, cur.text)
            fixed += 1

    # Remove zero-duration remnants
    intervals[:] = [iv for iv in intervals if iv.xmax - iv.xmin > 0.001]
    words_tier.intervals = intervals
    return fixed


def _fix_pp_phone_overlaps(pp_tier: Tier) -> int:
    """Resolve adjacent phone↔phone overlaps in pinyin_phones tier.

    MFA HMM alignment produces soft transitions where a final (rhyme)
    can overlap the next initial (onset) by 40-100ms.  These are not
    detected by _fix_overlapping_boundaries (which only fixes the words
    tier).  This pass clips all adjacent phone overlaps at the midpoint.

    Punctuation phones (,/。/！/？) and en: phones are clipped to favour
    the content phone: punct is trimmed, en: phones keep their start.
    """
    intervals = list(pp_tier.intervals)
    n = len(intervals)
    fixed = 0

    for i in range(n - 1):
        cur = intervals[i]
        nxt = intervals[i + 1]
        overlap = cur.xmax - nxt.xmin
        if overlap <= 0.001:       # sub-1ms — float noise, skip
            continue

        cur_text = cur.text.strip() if cur.text else ""
        nxt_text = nxt.text.strip() if nxt.text else ""
        cur_is_punct = cur_text in ('，', '。', '！', '？', '、', '：', '；', '…')
        nxt_is_punct = nxt_text in ('，', '。', '！', '？', '、', '：', '；', '…')
        cur_is_en = cur_text.startswith('en:')
        nxt_is_en = nxt_text.startswith('en:')

        # Punct overlapped by content phone → trim punct
        if cur_is_punct and not nxt_is_punct and not nxt_is_en:
            intervals[i] = Interval(cur.xmin, nxt.xmin, cur.text)
            fixed += 1
        elif nxt_is_punct and not cur_is_punct and not cur_is_en:
            intervals[i + 1] = Interval(cur.xmax, nxt.xmax, nxt.text)
            fixed += 1
        # en: phone overlapped by content phone → trim en: side
        elif cur_is_en and not nxt_is_en:
            intervals[i] = Interval(cur.xmin, nxt.xmin, cur.text)
            fixed += 1
        elif nxt_is_en and not cur_is_en:
            intervals[i + 1] = Interval(cur.xmax, nxt.xmax, nxt.text)
            fixed += 1
        # Two content phones → split at midpoint
        else:
            mid = round((cur.xmax + nxt.xmin) / 2.0, 4)
            intervals[i] = Interval(cur.xmin, mid, cur.text)
            intervals[i + 1] = Interval(mid, nxt.xmax, nxt.text)
            fixed += 1

    # Remove zero-duration remnants
    intervals[:] = [iv for iv in intervals if iv.xmax - iv.xmin > 0.001]
    pp_tier.intervals = intervals
    return fixed


def _build_gap_ownership_evidence(
        words_tier: Tier,
        source_words: list[dict] | None,
        ctc_tokens: list[dict] | None,
        reference_text: str = "",
        reference_authoritative: bool = False,
        max_gap_s: float = 0.030) -> dict[tuple[int, int], dict]:
    """Build an explicit proof map for mechanical word-boundary residuals.

    A duration is only a candidate here.  The proof also requires exact
    lexical order agreement across the current words tier, the source words
    snapshot, and CTC lexical tokens; contiguous/overlapping CTC spans; and
    no source silence, source punctuation, CTC punctuation, or authoritative
    reference punctuation at the boundary.  Missing or ambiguous evidence is
    represented by a false decision, so callers fail closed.

    Keys are interval-index pairs in ``words_tier``.  A direct content gap is
    keyed by ``(left_index, right_index)``.  An explicit silence interval
    between content owners uses the same outer-owner key.
    """
    result: dict[tuple[int, int], dict] = {}
    intervals = list(words_tier.intervals)
    current_lexical = [
        (index, iv.text.strip()) for index, iv in enumerate(intervals)
        if iv.text.strip() and not is_silence(iv.text) and not is_punct(iv.text)
    ]
    if not source_words or not ctc_tokens or len(current_lexical) < 2:
        return result

    source_lexical = [
        (index, str(item.get("text", "")).strip())
        for index, item in enumerate(source_words)
        if str(item.get("text", "")).strip()
        and not is_silence(str(item.get("text", "")))
        and not is_punct(str(item.get("text", "")))
    ]
    ctc_rows: list[tuple[int, dict]] = []
    for index, token in enumerate(ctc_tokens):
        if not isinstance(token, dict) or str(token.get("type", "word")) != "word":
            continue
        text = str(token.get("word", "")).strip()
        try:
            start, end = float(token["start_s"]), float(token["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if (not text or is_silence(text) or is_punct(text)
                or not math.isfinite(start) or not math.isfinite(end)
                or end <= start):
            continue
        ctc_rows.append((index, {"text": text, "start": start, "end": end}))

    def _same_sequence(left: list[tuple[int, str]], right: list[tuple[int, str]]) -> bool:
        return (len(left) == len(right)
                and all(a[1].casefold() == b[1].casefold()
                        for a, b in zip(left, right)))

    if (not _same_sequence(current_lexical, source_lexical)
            or not _same_sequence(current_lexical,
                                   [(index, row["text"]) for index, row in ctc_rows])):
        return result

    reference_punctuation_boundaries: set[int] = set()
    if reference_authoritative:
        reference_lexical_count = 0
        for unit in _extract_word_chars(reference_text):
            if is_punct(unit):
                reference_punctuation_boundaries.add(reference_lexical_count)
            elif is_word_like(unit):
                reference_lexical_count += 1
        if reference_lexical_count != len(current_lexical):
            return result

    max_gap_ticks = _threshold_ticks(max_gap_s)
    for lexical_index in range(len(current_lexical) - 1):
        left_index, _ = current_lexical[lexical_index]
        right_index, _ = current_lexical[lexical_index + 1]
        left_ctc_index, left_ctc = ctc_rows[lexical_index]
        right_ctc_index, right_ctc = ctc_rows[lexical_index + 1]
        current_left = intervals[left_index]
        current_right = intervals[right_index]
        gap_ticks = _duration_ticks(current_left.xmax, current_right.xmin)
        source_between = source_words[source_lexical[lexical_index][0] + 1:
                                     source_lexical[lexical_index + 1][0]]
        ctc_between = ctc_tokens[left_ctc_index + 1:right_ctc_index]
        source_owner = any(
            is_silence(str(item.get("text", "")))
            or is_punct(str(item.get("text", ""))) for item in source_between)
        ctc_owner = any(
            is_silence(str(item.get("word", "")))
            or is_punct(str(item.get("word", ""))) for item in ctc_between
            if isinstance(item, dict))
        reference_owner = lexical_index + 1 in reference_punctuation_boundaries
        ctc_contiguous = (left_ctc["end"] + AXIS_EPS >= right_ctc["start"])
        mechanical = (0 < gap_ticks < max_gap_ticks and ctc_contiguous
                      and not source_owner and not ctc_owner
                      and not reference_owner)
        result[(left_index, right_index)] = {
            "mechanical_frame_residual": mechanical,
            "source_ctc_contiguous": ctc_contiguous,
            "source_owner": source_owner,
            "ctc_owner": ctc_owner,
            "reference_owner": reference_owner,
            "gap_duration_us": gap_ticks,
        }
    return result


def _gap_evidence_allows_absorption(
        ownership_evidence: dict[tuple[int, int], dict] | None,
        left_index: int, right_index: int) -> bool:
    """Return true only for an explicit, positive ownership decision."""
    if not isinstance(ownership_evidence, dict):
        return False
    decision = ownership_evidence.get((left_index, right_index))
    return (isinstance(decision, dict)
            and decision.get("mechanical_frame_residual") is True
            and decision.get("source_owner") is False
            and decision.get("ctc_owner") is False
            and decision.get("reference_owner") is False
            and decision.get("source_ctc_contiguous") is True)


def _absorb_tiny_gaps(
        words_tier: Tier, max_gap_s: float = 0.030,
        ownership_evidence: dict[tuple[int, int], dict] | None = None) -> Tier:
    """Remove only serialization-scale residuals (``AXIS_EPS`` or less).

    Real positive-duration gaps, including old source-proven 10–30 ms gaps,
    belong to the final visual silence resolver.  Keeping this helper narrow
    prevents a derived-tier sync from consuming a gap before the immutable
    visual snapshot is taken.
    """
    intervals = list(words_tier.intervals)
    n = len(intervals)
    to_delete: set[int] = set()
    residual_ticks = _threshold_ticks(AXIS_EPS)

    for i in range(n - 1):
        if i in to_delete:
            continue
        cur = intervals[i]
        nxt = intervals[i + 1]
        cur_text = cur.text.strip() if cur.text else ""
        nxt_text = nxt.text.strip() if nxt.text else ""

        # Only an AXIS_EPS serialization residual may be removed here.
        if not cur_text or not nxt_text:
            continue
        if is_punct(cur_text) or is_punct(nxt_text):
            continue
        direct_gap_ticks = _duration_ticks(cur.xmax, nxt.xmin)
        if (not is_silence(cur_text) and not is_silence(nxt_text)
                and 0 < direct_gap_ticks <= residual_ticks):
            intervals[i] = Interval(cur.xmin, nxt.xmin, cur.text)
            continue
        if is_silence(cur_text):
            if (_duration_ticks(cur.xmin, cur.xmax) <= residual_ticks
                    and i > 0
                    and i + 1 < n
                    and not is_silence(intervals[i - 1].text)
                    and not is_silence(nxt.text)):
                intervals[i - 1] = Interval(
                    intervals[i - 1].xmin, nxt.xmin, intervals[i - 1].text)
                to_delete.add(i)
        elif is_silence(nxt_text):
            continue

    intervals = [iv for idx, iv in enumerate(intervals) if idx not in to_delete]
    # Remove zero-duration remnants
    intervals = [iv for iv in intervals if iv.duration > 0.001]
    return _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))


def _sync_derived_tiers(textgrid: TextGrid, ipa_to_pinyin: dict[str, str],
                        pinyin_dict: dict[str, list[str]] | None = None,
                        raw_text: str = "",
                        en_mfa_windows: dict[tuple[str, float], tuple[float, float]] | None = None,
                        report_warnings: list[str] | None = None,
                        reference_authoritative: bool = False,
                        gap_ownership_evidence: dict[tuple[int, int], dict] | None = None) -> None:
    """Rebuild hanzi and pinyin_phones from the current words + phones tiers.

    Call this after ANY in-place modification to words tier boundaries
    to keep all three boundary tiers (words, hanzi, pinyin_phones) in
    lockstep.  Without this, downstream code reads stale tier data.

    This is the SINGLE sync point for derived tiers — every words-tier
    mutation path must go through here.
    """
    words_tier = tier_by_name(textgrid, "words")
    phones_tier = tier_by_name(textgrid, "phones")
    if words_tier is None:
        return

    # Rebuild source phones from the pre-mutation lineage before deriving
    # pinyin_phones.  Ambiguous lineage is retained as a veto; it is never
    # repaired by strongest-overlap assignment.
    lineage = getattr(textgrid, "_phone_lineage", None)
    if phones_tier is not None and isinstance(lineage, dict):
        rebuilt_phones = _rebuild_phones_from_lineage(words_tier, phones_tier, lineage)
        if rebuilt_phones is None:
            invalid = lineage.get("reasons", ["phone_lineage_rebuild_failed"])
            textgrid._phone_lineage_invalid = invalid
            phones_tier._phone_lineage_invalid = invalid
        else:
            phones_tier = rebuilt_phones
            for index, tier in enumerate(textgrid.tiers):
                if tier.name == "phones":
                    textgrid.tiers[index] = phones_tier
                    break
            if lineage.get("status") in {"verified", "partial"}:
                # A partial lineage is safe after its nonlexical crossings
                # have been clipped; do not carry a stale all-tier veto from
                # an earlier failed rebuild into the publication contract.
                if hasattr(textgrid, "_phone_lineage_invalid"):
                    delattr(textgrid, "_phone_lineage_invalid")
                if hasattr(phones_tier, "_phone_lineage_invalid"):
                    delattr(phones_tier, "_phone_lineage_invalid")

    # 0. Absorb only source-proven frame residuals before rebuilding derived
    #    tiers.  An absent evidence map deliberately preserves every gap.
    words_tier = _absorb_tiny_gaps(
        words_tier, ownership_evidence=gap_ownership_evidence)
    words_tier = _reconcile_publication_geometry(words_tier)
    # Update the tier in-place in the textgrid
    for i, t in enumerate(textgrid.tiers):
        if t.name == "words":
            textgrid.tiers[i] = words_tier
            break

    # 1. Rebuild hanzi from updated words tier.
    # This is an invariant: words is authoritative, so stale derived tiers
    # must never survive a failed rebuild (Regression Case 66).
    if raw_text:
        try:
            hanzi_tier = _build_hanzi_tier(
                words_tier, raw_text, report_warnings or [],
                reference_authoritative=reference_authoritative)
            if hanzi_tier:
                found = False
                for i, t in enumerate(textgrid.tiers):
                    if t.name == "hanzi":
                        textgrid.tiers[i] = hanzi_tier
                        found = True
                        break
                if not found:
                    for i, t in enumerate(textgrid.tiers):
                        if t.name == "words":
                            textgrid.tiers.insert(i, hanzi_tier)
                            break
        except Exception as exc:
            raise RuntimeError(
                "failed to rebuild hanzi tier from authoritative words tier"
            ) from exc

    # 2. Rebuild pinyin_phones from updated phones + words tiers.
    if phones_tier is not None and pinyin_dict is not None:
        try:
            synced_pp = build_pinyin_phones_tier(
                phones_tier, ipa_to_pinyin, words_tier, pinyin_dict,
                en_mfa_windows=en_mfa_windows)
            if synced_pp:
                for i, t in enumerate(textgrid.tiers):
                    if t.name == "pinyin_phones":
                        textgrid.tiers[i] = synced_pp
                        break
        except Exception as exc:
            raise RuntimeError(
                "failed to rebuild pinyin_phones tier from words/phones tiers"
            ) from exc


def strip_edge_punctuation(textgrid: TextGrid) -> None:
    """Remove leading/trailing punctuation that sits at the edge before/after
    all real words, absorbing its time into the adjacent interval.

    Edge punctuation appears when NVASR strips NVV tags (e.g. ``<|HAPPY|>``)
    but leaves orphaned ellipsis/punct between the removed tag and the first
    word.  Without this cleanup, ``…`` can appear as the first word in the
    hanzi/words tiers.
    """
    from dataclasses import replace as _replace

    words_tier = tier_by_name(textgrid, "words")
    if words_tier is None:
        return
    intervals = list(words_tier.intervals)
    if len(intervals) < 2:
        return

    def _is_real_word(iv) -> bool:
        """True if this interval is a content word, not silence/NVV/punct."""
        return (
            not is_silence(iv.text)
            and not is_punct(iv.text)
            and iv.text.strip() not in ("", "<eps>")
        )
        # Note: NVV tokens are real content — they occupy time and can absorb punct

    # ── Find first and last real word ──
    first_real = None
    last_real = None
    for i, iv in enumerate(intervals):
        if _is_real_word(iv):
            first_real = i
            break
    for i in range(len(intervals) - 1, -1, -1):
        if _is_real_word(intervals[i]):
            last_real = i
            break

    if first_real is None or last_real is None:
        return

    # ── Strip leading punct: absorb into the preceding interval ──
    # Walk backwards from first_real-1 to 0; every punct gets absorbed into
    # its neighbour.  Silence intervals (<spN>) are NOT punct and must be
    # skipped — is_punct() returns True for some bracket-wrapped tokens.
    leading_punct_indices = []
    for i in range(first_real):
        if not is_silence(intervals[i].text) and is_punct(intervals[i].text):
            leading_punct_indices.append(i)

    for pi in sorted(leading_punct_indices, reverse=True):
        p_iv = intervals[pi]
        # Absorb into preceding interval (if any) by extending its xmax
        if pi > 0:
            intervals[pi - 1] = _replace(intervals[pi - 1], xmax=p_iv.xmax)
        elif pi + 1 < len(intervals):
            # First interval is punct — absorb into next interval
            intervals[pi + 1] = _replace(intervals[pi + 1], xmin=p_iv.xmin)
        intervals[pi] = _replace(intervals[pi], xmin=0, xmax=0, text="")

    # NOTE: There is intentionally NO trailing strip.
    # Trailing punctuation (。！？…) after the last real word is ALWAYS
    # legitimate — sentences naturally end with punctuation.  The "mirror"
    # design (stripping both edges) is a logical error because leading
    # and trailing edges are NOT symmetric:
    #   - Leading punct: always orphaned (tag-stripping artifact) → strip
    #   - Trailing punct: always legitimate (end-of-sentence) → keep
    # NVV-trailing punct+silence chains are already handled upstream by
    # absorb_nvv_trailing (Case 9 W1) and absorb_silence_into_punct (Case 9 W2).

    # ── Apply changes ──
    intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
    new_words = _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))
    for i, t in enumerate(textgrid.tiers):
        if t.name == "words":
            textgrid.tiers[i] = new_words
            break

    # Sync pinyin_phones: remove corresponding punct intervals (same time range)
    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if pp_tier is not None:
        pp_ivs = [iv for iv in pp_tier.intervals
                  if iv.duration > 0.001 and not is_punct(iv.text)]
        new_pp = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, pp_ivs)
        for i, t in enumerate(textgrid.tiers):
            if t.name == "pinyin_phones":
                textgrid.tiers[i] = new_pp
                break


def absorb_silence_into_punct(textgrid: TextGrid) -> None:
    """Absorb punctuation-adjacent ``<spN>`` intervals into punctuation.

    Punctuation is silent by nature — the silence immediately before or after
    it is its realised duration.  This is the **fallback** pass: it handles
    residual ``<spN>`` adjacent to punctuation that was not already absorbed
    by an NVV in :func:`absorb_nvv_trailing`.  Silence strictly between two
    lexical owners is deliberately untouched and remains a filterable pause.

    Without this step, a 5 ms ``！`` followed by a 695 ms ``<sp2>`` leaves
    an orphaned silence in the middle of the words tier, which the
    ``mid_sp`` filter would reject.

    Operates on words, phones, and pinyin_phones tiers in sync.
    """
    words_tier = tier_by_name(textgrid, "words")
    phones_tier = tier_by_name(textgrid, "phones")
    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if words_tier is None:
        return

    intervals = list(words_tier.intervals)
    to_delete_words: set[int] = set()
    to_delete_phones: set[int] = set()
    to_delete_pp: set[int] = set()
    absorbed_count = 0

    i = 0
    while i < len(intervals) - 1:
        cur_text = intervals[i].text.strip()
        next_text = intervals[i + 1].text.strip()
        if is_silence(cur_text) and is_punct(next_text) and cur_text:
            sil_iv = intervals[i]
            punct_iv = intervals[i + 1]
            # A pause before punctuation is punctuation-owned, just like a
            # pause after punctuation.  Keeping it as a separate word tier
            # interval is what used to make every reference sample fail the
            # strict interior-sp audit.
            intervals[i + 1] = Interval(sil_iv.xmin, punct_iv.xmax,
                                         punct_iv.text)
            to_delete_words.add(i)
            absorbed_count += 1
            if phones_tier:
                for pi, p in enumerate(phones_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_phones.add(pi)
                        break
            if pp_tier:
                for pi, p in enumerate(pp_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_pp.add(pi)
                        break
            # Keep the punctuation at i+1 as the current candidate so a
            # pause on both sides is collapsed in one pass.
            i += 1
        elif is_punct(cur_text) and is_silence(next_text) and next_text:
            sil_iv = intervals[i + 1]
            # Extend punctuation to absorb the silence duration.
            intervals[i] = Interval(intervals[i].xmin, sil_iv.xmax, intervals[i].text)
            to_delete_words.add(i + 1)
            absorbed_count += 1

            # Remove matching silence from phones & pp tiers.
            if phones_tier:
                for pi, p in enumerate(phones_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_phones.add(pi)
                        break
            if pp_tier:
                for pi, p in enumerate(pp_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_pp.add(pi)
                        break

            i += 2  # Skip the absorbed silence.
        else:
            i += 1

    if not to_delete_words:
        return

    # Apply deletions.
    intervals = [iv for idx, iv in enumerate(intervals)
                 if idx not in to_delete_words]
    words_tier.intervals = intervals
    if phones_tier and to_delete_phones:
        phones_tier.intervals = [iv for idx, iv in enumerate(phones_tier.intervals)
                                 if idx not in to_delete_phones]
    if pp_tier and to_delete_pp:
        pp_tier.intervals = [iv for idx, iv in enumerate(pp_tier.intervals)
                             if idx not in to_delete_pp]

    # Clean up zero-duration remnants.
    for tier in (words_tier, phones_tier, pp_tier):
        if tier:
            tier.intervals = [iv for iv in tier.intervals
                              if iv.duration > 0.001 or not iv.text.strip()]
    _record_processed_geometry_operation(
        words_tier, "punct_pause_absorption", count=absorbed_count)


def _finalise_textgrid(textgrid: TextGrid, raw_text: str, pinyin_text: str,
                       args, warnings: list | None = None,
                       *, reference_authoritative: bool = False) -> TextGrid:
    """Clean up corrected text and restructure tiers for final output.

    1. Remove ``[sp]`` markers from corrected_text (merged as sp0).
    2. Prefix ``<sp1>`` to mark leading silence.
    3. Replace raw_text tier with the final text.
    4. Sync pinyin tier punctuation + ``<sp1>`` prefix.
    5. Insert a hanzi tier (one CJK char per word interval).
    6. Reorder: raw_text, pinyin, hanzi, words, phones, pinyin_phones.

    *warnings* (when provided) is threaded through to
    :func:`_build_hanzi_tier` for defensive mismatch detection.
    """
    corrected_tier = tier_by_name(textgrid, "corrected_text")
    if corrected_tier is None:
        return textgrid
    corrected = corrected_tier.intervals[0].text

    # 1. Strip [sp] (already merged)
    final_text = corrected.replace('[sp]', '')
    # 2. Prefix <sp1>
    final_text = '<sp1>' + final_text

    # 3. Replace raw_text tier
    raw_tier = tier_by_name(textgrid, "raw_text")
    if raw_tier is not None:
        raw_tier.intervals[0].text = final_text

    # 4. Sync pinyin: strip punct not in final text, add <sp1> prefix
    pinyin_tier = tier_by_name(textgrid, "pinyin")
    if pinyin_tier is not None:
        py_final = _sync_pinyin_punctuation(pinyin_tier.intervals[0].text, raw_text, final_text)
        pinyin_tier.intervals[0].text = py_final

    # 5. Build hanzi tier — one CJK char per word interval
    words_tier = tier_by_name(textgrid, "words")
    hanzi_tier = (_build_hanzi_tier(
        words_tier, raw_text, warnings,
        reference_authoritative=reference_authoritative)
        if words_tier else None)

    # 6. Remove corrected_text, reorder tiers
    new_tiers = []
    for tier in textgrid.tiers:
        if tier.name == "corrected_text":
            continue
        elif tier.name == "words" and hanzi_tier is not None:
            new_tiers.append(hanzi_tier)
            new_tiers.append(tier)
        else:
            new_tiers.append(tier)

    return TextGrid(textgrid.xmin, textgrid.xmax, new_tiers)


def _sync_pinyin_punctuation(pinyin_text: str, raw_text: str, final_text: str) -> str:
    """Sync pinyin punctuation to match the final corrected Chinese text.

    Takes the pinyin-word sequence and re-inserts punctuation exactly where
    the final Chinese text has it (between the same word positions).  Punctuation
    that was deleted in the final text is dropped.
    """
    py_words = [t for t in pinyin_text.split() if is_word_like(t)]
    # Build final_text character sequence: word chars vs punct
    final_chars = list(final_text.replace('<sp1>', ''))
    result = []
    word_idx = 0
    for ch in final_chars:
        if is_word_like(ch):
            if word_idx < len(py_words):
                result.append(py_words[word_idx])
                word_idx += 1
        elif is_punct(ch):
            result.append(ch)
        else:
            result.append(ch)

    return '<sp1> ' + ' '.join(result)


def _extract_word_chars(text: str) -> list[str]:
    """Extract word-like chars from raw text, grouping consecutive non-CJK alpha chars
    and trailing digits (pinyin tone numbers).

    Angle brackets (``<``, ``>``) are grouped with the alpha buffer so that
    NVV tokens like ``<LAUGHTER>`` and ``<QUESTION-YI>`` stay as a single
    unit.  ``<`` flushes any pending buffer and opens a new group; ``>``
    closes the group and flushes immediately so the next word is separate.
    """
    result = []
    buf = ""
    for c in text:
        if is_cjk(c):
            if buf:
                result.append(buf)
                buf = ""
            result.append(c)
        elif c == '<':
            if buf:
                result.append(buf)
                buf = ""
            buf += c
        elif c == '>':
            buf += c
            result.append(buf)
            buf = ""
        elif c.isalpha() or c == '-':
            buf += c  # hyphen in NVV tokens like QUESTION-YI stays with alpha
        elif c.isdigit():
            buf += c  # pinyin tone number, keep with preceding alpha
        # punctuation: flush buffer, keep as separate entry; whitespace: flush & skip
        else:
            if buf:
                result.append(buf)
                buf = ""
            if not c.isspace():
                result.append(c)
    if buf:
        result.append(buf)
    return result


def _canonicalize_reference_hyphens(text: str) -> str:
    """Remove lexical ASCII hyphens from an in-memory processing string.

    Hyphens outside angle-bracket labels are lexical separators, so
    ``v-tuber`` and ``open-ai`` become ``vtuber`` and ``openai``.  Hyphens
    inside bracketed NVV labels are preserved (for example
    ``<QUESTION-YI>``), keeping the NVV label and its provenance unchanged.
    This helper never writes back to the source/reference file.
    """
    result: list[str] = []
    label_closers: list[str] = []
    for char in str(text):
        if char in "<[":
            label_closers.append(">" if char == "<" else "]")
        elif label_closers and char == label_closers[-1]:
            label_closers.pop()
        if char != "-" or label_closers:
            result.append(char)
    return "".join(result)


def _reference_pinyin_text(reference_text: str, source_pinyin: str) -> str:
    """Render the pinyin tier from the authoritative reference sequence.

    The lab is an acoustic alignment input and can contain tokenizer
    fragments (for example ``kp op`` for the reference spelling ``K-Pop``).
    Those fragments must not rewrite the user-facing lexical text.  Consume
    only the toned CJK syllables from the lab, while taking English/NVV and
    punctuation spellings from the reference itself.
    """
    source_cjk = [token for token in source_pinyin.split()
                  if is_pinyin_syllable(token)]
    cjk_index = 0
    rendered: list[str] = []
    for unit in _extract_word_chars(reference_text):
        if is_cjk(unit):
            if cjk_index >= len(source_cjk):
                # Numerals introduced by the authority normalizer have a
                # deterministic Mandarin reading even when the legacy lab
                # still exposes the ASCII suffix (target1/target2).
                # Other missing CJK tones remain fail-closed as before.
                rendered.append(_AUTHORITY_NUMERAL_PINYIN.get(unit, unit))
            else:
                rendered.append(source_cjk[cjk_index])
                cjk_index += 1
        elif is_nvv_token(unit):
            rendered.append(f"<{unit.strip('<>').upper()}>")
        else:
            rendered.append(unit)
    return "<sp1> " + " ".join(rendered)


def _canonicalize_surface_nvv_markup(text: str) -> str:
    """Canonicalize known NVV labels without rewriting lexical hyphens.

    NVV labels occur both as the canonical ``<NAME>`` spelling and as bare
    labels in ASR/reference surfaces.  Only names from ``NVV_NAMES`` are
    rewritten; an ASCII hyphen in an ordinary lexical surface such as
    ``open-ai`` remains untouched.
    """
    value = str(text or "")

    def bracketed(match: re.Match[str]) -> str:
        token = match.group(0)
        return (f"<{token[1:-1].upper()}>"
                if is_nvv_token(token) else token)

    names = sorted((str(name).strip("<>") for name in NVV_NAMES),
                   key=len, reverse=True)
    if not names:
        return value
    bare = re.compile(
        r"(?<![A-Za-z0-9-])(?:" + "|".join(re.escape(name)
        for name in names) + r")(?![A-Za-z0-9-])", re.IGNORECASE)
    chunks = re.split(r"(<[^>\r\n]*>)", value)
    for index, chunk in enumerate(chunks):
        if index % 2:
            match = re.fullmatch(r"<[A-Za-z][A-Za-z-]*>", chunk)
            chunks[index] = bracketed(match) if match is not None else chunk
        else:
            chunks[index] = bare.sub(
                lambda match: f"<{match.group(0).upper()}>", chunk)
    return "".join(chunks)


def _surface_punctuation(tier: Tier | None) -> list[str]:
    """Extract surface punctuation after removing known NVV markup.

    Bracketed NVV labels are removed as markup, while recognized bare labels
    are canonicalized first.  This prevents the hyphen in labels such as
    ``Surprise-wa`` from being mistaken for lexical punctuation.
    """
    if tier is None or not tier.intervals:
        return []
    surface = _canonicalize_surface_nvv_markup(
        str(tier.intervals[0].text or ""))
    surface = re.sub(r"<[^>\r\n]*>", "", surface)
    return [char for char in surface if is_punct(char)]


def _repair_authority_punctuation_geometry(
        words_tier: Tier, reference_text: str,
        punct_entries: list[dict] | None = None) -> int:
    """Repair punctuation timing only when an authority anchor owns silence.

    CTC punctuation labels are lexical anchors, not permission to consume an
    arbitrary neighbouring word.  A geometry repair is therefore allowed only
    when the matching reference anchor overlaps an explicit silence interval in
    ``words``.  The silence is folded into that punctuation interval; an
    unexplained mid-word silence (with no colocated authority anchor) is left
    untouched for the normal ``mid_sp`` QC path.
    """
    if not punct_entries:
        return 0
    allowed = '，。…！？、；：,.!?;:～'
    current = list(words_tier.intervals)
    punct_indices = [i for i, iv in enumerate(current)
                     if iv.text.strip() in allowed]
    if not punct_indices:
        return 0

    anchors_by_char: dict[str, list[tuple[float, float]]] = {}
    for entry in punct_entries:
        char = str(entry.get("word", "")).strip()
        if char not in allowed:
            continue
        try:
            start, end = float(entry["start_s"]), float(entry["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            anchors_by_char.setdefault(char, []).append((start, end))

    used: dict[str, int] = {}
    changed = 0
    absorbed: set[int] = set()
    owner_ranges: list[tuple[float, float]] = []
    for punct_index in punct_indices:
        punct = current[punct_index]
        char = punct.text.strip()
        occurrence = used.get(char, 0)
        used[char] = occurrence + 1
        anchors = anchors_by_char.get(char, [])
        if occurrence >= len(anchors):
            continue
        anchor_start, anchor_end = anchors[occurrence]

        # Establish the immediate lexical owner envelope.  The old MFA gap
        # may be zero or stale; only the two adjacent lexical owners may
        # overlap this authoritative anchor.
        prev = next((current[i] for i in range(punct_index - 1, -1, -1)
                     if current[i].text.strip() and not is_silence(current[i].text)
                     and not is_punct(current[i].text.strip())), None)
        nxt = next((current[i] for i in range(punct_index + 1, len(current))
                    if current[i].text.strip() and not is_silence(current[i].text)
                    and not is_punct(current[i].text.strip())), None)
        left_bound = prev.xmin if prev is not None else words_tier.xmin
        right_bound = nxt.xmax if nxt is not None else words_tier.xmax
        if (anchor_start < left_bound - AXIS_EPS
                or anchor_end > right_bound + AXIS_EPS):
            continue
        if any(
                iv is not prev and iv is not nxt
                and iv.text.strip() and not is_silence(iv.text)
                and not is_punct(iv.text)
                and iv.xmax > anchor_start + AXIS_EPS
                and iv.xmin < anchor_end - AXIS_EPS
                for iv in current):
            continue

        local_silence = []
        for index, iv in enumerate(current):
            if index in absorbed or not is_silence(iv.text) or not iv.text.strip():
                continue
            if iv.xmax > anchor_start and iv.xmin < anchor_end:
                local_silence.append((index, iv))
        if not local_silence:
            continue

        owner_start = max(words_tier.xmin, anchor_start)
        owner_end = min(words_tier.xmax, anchor_end)
        if owner_end <= owner_start + 0.001:
            continue
        owner_ranges.append((owner_start, owner_end))
        if abs(owner_start - punct.xmin) > 1e-9 or abs(owner_end - punct.xmax) > 1e-9:
            current[punct_index] = Interval(owner_start, owner_end, punct.text)
        # Removing a colocated silence is itself a geometry repair even when
        # the punctuation interval already exactly spans the authority anchor.
        # Without this marker the early return below leaves overlapping
        # punctuation/silence intervals in place, triggering ``mid_sp`` and
        # overlap diagnostics downstream.
        changed += 1
        for index, silence in local_silence:
            absorbed.add(index)

    if not changed:
        return 0
    rebuilt: list[Interval] = []
    for index, iv in enumerate(current):
        if index in absorbed:
            pieces = [(iv.xmin, iv.xmax)]
            for owner_start, owner_end in owner_ranges:
                next_pieces = []
                for piece_start, piece_end in pieces:
                    if piece_end <= owner_start or piece_start >= owner_end:
                        next_pieces.append((piece_start, piece_end))
                        continue
                    if piece_start < owner_start:
                        next_pieces.append((piece_start, owner_start))
                    if piece_end > owner_end:
                        next_pieces.append((owner_end, piece_end))
                pieces = next_pieces
            for piece_start, piece_end in pieces:
                if piece_end > piece_start + 0.001:
                    rebuilt.append(Interval(piece_start, piece_end, iv.text))
            continue
        rebuilt.append(iv)
    words_tier.intervals = sorted(rebuilt, key=lambda iv: (iv.xmin, iv.xmax, iv.text))
    return changed


def _repair_reference_authority_colocated_silence(
        words_tier: Tier, reference_text: str) -> int:
    """Remove only exact silence duplicated by restored reference punctuation.

    A reference mark absent from ``_punct.json`` has no CTC timing anchor, so
    :func:`_repair_authority_punctuation_geometry` deliberately cannot claim
    the coincident pause.  Once the lexical reference reconciliation has
    restored that mark, an exact (within one microsecond tick) punctuation /
    explicit-silence duplicate is nevertheless unambiguous.  Do not use this
    fallback for partial overlaps, non-authoritative punctuation, or a words
    tier whose punctuation projection is not exactly the reference sequence.
    """
    allowed = '，。…！？、；：,.!?;:～'
    reference_punct = [unit.strip() for unit in _extract_word_chars(reference_text)
                       if is_punct(unit) and unit.strip() in allowed]
    current = list(words_tier.intervals)
    current_punct = [iv.text.strip() for iv in current
                     if iv.text.strip() in allowed]
    if not reference_punct or current_punct != reference_punct:
        return 0

    def _within_tick(left: float, right: float) -> bool:
        try:
            delta = (Decimal(str(left)) - Decimal(str(right))) * _TIME_TICK_HZ
            return delta.is_finite() and abs(delta) <= 1
        except (InvalidOperation, ValueError, TypeError, OverflowError):
            return False

    remove: set[int] = set()
    silences = [(index, iv) for index, iv in enumerate(current)
                if is_silence(iv.text) and iv.text.strip()]
    for punct in (iv for iv in current if iv.text.strip() in allowed):
        for index, silence in silences:
            if (_within_tick(punct.xmin, silence.xmin)
                    and _within_tick(punct.xmax, silence.xmax)):
                remove.add(index)
    if not remove:
        return 0
    words_tier.intervals = [iv for index, iv in enumerate(current) if index not in remove]
    return len(remove)


def _restore_reference_punctuation_legacy(words_tier: Tier, reference_text: str,
                                          punct_entries: list[dict] | None = None) -> int:
    """Make the words tier's punctuation sequence equal the authority.

    CTC punctuation is an alignment anchor, not lexical authority.  When a
    broad CTC pause has swallowed a reference comma or produced an extra
    terminal full stop, rebuilding only from CTC punctuation can silently
    change the transcript.  This pass keeps lexical word intervals, removes
    the current punctuation projection, and restores the reference sequence.
    Anchor timing is bound to the reference lexical boundary and the
    immediate lexical owners.  It may cross an old MFA boundary when it
    overlaps only those two owners; it may never consume a non-adjacent word.
    A missing punctuation interval is restored only when its CTC anchor has
    explicit local silence evidence.  Voiced lexical audio is never carved
    solely to make room for a reference mark.
    """
    ref_puncts: list[tuple[str, int]] = []
    lexical_count = 0
    for unit in _extract_word_chars(reference_text):
        if is_punct(unit):
            if unit.strip() in '，。…！？、；：,.!?;:～':
                ref_puncts.append((unit, lexical_count))
        elif is_word_like(unit):
            lexical_count += 1

    current = list(words_tier.intervals)
    current_puncts = [iv.text.strip() for iv in current
                      if iv.text.strip() in '，。…！？、；：,.!?;:～']
    desired_puncts = [char for char, _ in ref_puncts]
    if current_puncts == desired_puncts:
        # When the punctuation label is already present, preserve it.  A CTC
        # anchor without a local silence is not permission to re-carve the
        # neighbouring lexical words.  With local silence, only repair the
        # punctuation-owned geometry.
        anchored = _repair_authority_punctuation_geometry(
            words_tier, reference_text, punct_entries)
        repaired = anchored + _repair_reference_authority_colocated_silence(
            words_tier, reference_text)
        canonical = _reconcile_publication_geometry(words_tier)
        words_tier.intervals = canonical.intervals
        return repaired

    lexical = [iv for iv in current
               if iv.text.strip() and not is_silence(iv.text)
               and iv.text.strip() not in '，。…！？、；：,.!?;:～']
    # Map reference lexical boundaries onto the current timed intervals.  A
    # hyphenated authority unit can occupy several strict-English intervals
    # (K-Pop -> kp/op), but reference punctuation belongs after the group.
    ref_lexical = [unit for unit in _extract_word_chars(reference_text)
                   if is_word_like(unit)]
    current_boundary = [0]
    current_index = 0
    for unit in ref_lexical:
        target = re.sub(r'[^a-z0-9]', '', unit.lower())
        if '-' in unit and target:
            compact = ''
            start = current_index
            while current_index < len(lexical):
                probe = lexical[current_index].text.strip()
                if not is_english_token(probe):
                    break
                compact += re.sub(r'[^a-z0-9]', '', probe.lower())
                current_index += 1
                if compact == target:
                    break
            if current_index == start and current_index < len(lexical):
                current_index += 1
        elif current_index < len(lexical):
            current_index += 1
        current_boundary.append(current_index)
    existing = [iv for iv in current if iv.text.strip() in desired_puncts]
    anchors_by_char: dict[str, list[tuple[float, float]]] = {}
    for entry in punct_entries or []:
        char = str(entry.get('word', '')).strip()
        if char in desired_puncts:
            try:
                start, end = float(entry['start_s']), float(entry['end_s'])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(start) and math.isfinite(end) and end > start:
                anchors_by_char.setdefault(char, []).append((start, end))

    used_anchor: dict[str, int] = {}
    punctuation: list[tuple[int, Interval]] = []
    for ref_index, (char, boundary) in enumerate(ref_puncts):
        mapped_boundary = (current_boundary[boundary]
                           if boundary < len(current_boundary)
                           else len(lexical))
        prev = (lexical[mapped_boundary - 1]
                if mapped_boundary > 0 and mapped_boundary <= len(lexical)
                else None)
        nxt = (lexical[mapped_boundary]
               if mapped_boundary < len(lexical) else None)
        gap_start = prev.xmax if prev is not None else words_tier.xmin
        gap_end = nxt.xmin if nxt is not None else words_tier.xmax
        if gap_end < gap_start:
            gap_start = gap_end = max(words_tier.xmin, min(words_tier.xmax, gap_start))

        # Reuse the corresponding authoritative CTC anchor.  Its duration is
        # allowed to be wider than the old MFA gap: punctuation commonly owns
        # a long realised pause, and CTC may continue that owner beyond the
        # next lexical boundary.  The *publication* owner is still local to
        # the two reference-adjacent lexical words.  In particular, a broad
        # anchor must not make the next spoken word part of punctuation.
        occurrence = sum(1 for c, _ in ref_puncts[:ref_index] if c == char)
        candidates = anchors_by_char.get(char, [])
        anchor = candidates[occurrence] if occurrence < len(candidates) else None
        anchor_accepted = False
        if anchor is not None:
            raw_start, raw_end = anchor
            # A reference mark may be restored from CTC timing only when the
            # timing has local silence evidence.  Otherwise placing the mark
            # would steal voiced audio from a lexical word merely because an
            # anchor numerically overlaps it.
            local_silences = [iv for iv in current
                              if is_silence(iv.text) and iv.text.strip()
                              and iv.xmax > raw_start + AXIS_EPS
                              and iv.xmin < raw_end - AXIS_EPS]
            if not local_silences:
                # An edge anchor immediately adjacent to an existing silence
                # may remain an anchor-only punctuation interval (e.g. a
                # terminal mark after a retained tail pause), but it still
                # must not clip a lexical owner.  A bare anchor in voiced
                # audio, or across a word boundary with no silence, is not
                # restored.
                adjacent_silence = any(
                    is_silence(iv.text) and iv.text.strip()
                    and iv.xmax >= raw_start - AXIS_EPS
                    and iv.xmin <= raw_end + AXIS_EPS
                    for iv in current)
                overlaps_lexical = any(
                    not is_silence(iv.text) and iv.text.strip()
                    and not is_punct(iv.text.strip())
                    and iv.xmax > raw_start + AXIS_EPS
                    and iv.xmin < raw_end - AXIS_EPS
                    for iv in current)
                if not adjacent_silence or overlaps_lexical:
                    continue
                start, end = raw_start, raw_end
            else:
                # Replace the supporting silence itself, not any voiced
                # prefix/suffix included in the broad CTC anchor.
                support = max(
                    local_silences,
                    key=lambda iv: min(iv.xmax, raw_end)
                    - max(iv.xmin, raw_start))
                start, end = support.xmin, support.xmax
                if end <= start + AXIS_EPS:
                    continue
            has_local_silence = any(
                is_silence(iv.text) and iv.text.strip()
                and iv.xmax >= start - AXIS_EPS
                and iv.xmin <= end + AXIS_EPS
                for iv in current)
            if not has_local_silence:
                continue
            left_bound = prev.xmin if prev is not None else words_tier.xmin
            right_bound = nxt.xmax if nxt is not None else words_tier.xmax
            if prev is not None and nxt is not None:
                # The reference sequence is authoritative.  A CTC/MFA
                # boundary adjustment may have extended either immediate
                # lexical owner across the punctuation anchor.  Such an
                # anchor is still valid: the repair below clips the owner
                # on the correct side and gives the interval back to the
                # punctuation.  Only an anchor outside the two adjacent
                # owners is unrelated and must be rejected.
                crosses_local_boundary = (
                    end > prev.xmin + AXIS_EPS
                    and start < nxt.xmax - AXIS_EPS)
            elif prev is not None:
                crosses_local_boundary = end > prev.xmin + AXIS_EPS
            elif nxt is not None:
                crosses_local_boundary = start < nxt.xmax - AXIS_EPS
            else:
                crosses_local_boundary = True
            anchor_accepted = (
                math.isfinite(start) and math.isfinite(end)
                and end > start + AXIS_EPS
                and end > left_bound + AXIS_EPS
                and start < right_bound - AXIS_EPS
                and crosses_local_boundary)
            if anchor_accepted:
                # Project the (possibly very wide) CTC owner onto the local
                # reference boundary.  If the anchor starts in the left
                # lexical owner, retain that partial overlap; otherwise the
                # preceding gap belongs to punctuation from the left owner's
                # end.  Symmetrically, if the anchor runs through the right
                # owner and beyond, stop at the right owner's start instead
                # of consuming the next word and all later speech.
                if prev is not None:
                    if start > prev.xmin + AXIS_EPS and start < prev.xmax - AXIS_EPS:
                        owner_start = start
                    elif start >= prev.xmax - AXIS_EPS and (
                            nxt is None or end <= nxt.xmax + AXIS_EPS):
                        # Preserve the actual CTC punctuation onset when it
                        # begins inside the local gap.  Only a broad anchor
                        # that runs through the next lexical owner is widened
                        # back to the preceding owner's end.
                        owner_start = start
                    else:
                        owner_start = prev.xmax
                else:
                    owner_start = words_tier.xmin
                if nxt is not None:
                    if end > nxt.xmin + AXIS_EPS and end < nxt.xmax - AXIS_EPS:
                        owner_end = end
                    elif end <= nxt.xmin + AXIS_EPS:
                        owner_end = end
                    else:
                        owner_end = nxt.xmin
                else:
                    # Terminal punctuation may have a wide but finite CTC
                    # duration.  Preserve that duration, bounded by the
                    # TextGrid axis, rather than extending it to the axis end
                    # when the anchor itself ends earlier.
                    owner_end = min(end, words_tier.xmax)

                # A broad anchor is useful only when it actually covers the
                # local boundary.  Do not synthesize punctuation from a
                # distant anchor when there is no local silence/overlap.
                if (owner_end > owner_start + AXIS_EPS
                        and end > owner_start + AXIS_EPS
                        and start < owner_end - AXIS_EPS):
                    punctuation.append((mapped_boundary,
                                        Interval(owner_start, owner_end, char)))
                    continue
                anchor_accepted = False
            # An authoritative-looking anchor that crosses a non-adjacent
            # owner with no local overlap is not replaceable by a guessed
            # interval.
            continue
        if gap_end - gap_start > 0.001:
            # Reference-only punctuation may reuse an exact existing local
            # nonlexical owner, but it cannot synthesize the whole lexical
            # gap merely because the gap is nonzero.
            local_nonlexical = [iv for iv in current
                                if (is_silence(iv.text) or is_punct(iv.text))
                                and iv.xmax > gap_start and iv.xmin < gap_end]
            if not local_nonlexical:
                continue
            start = max(gap_start, min(iv.xmin for iv in local_nonlexical))
            end = min(gap_end, max(iv.xmax for iv in local_nonlexical))
        else:
            # No local owner range means there is no evidence-backed time
            # for this reference mark.  Do not synthesize a centered interval
            # or claim audio from either lexical neighbour.
            continue
        start = max(words_tier.xmin, start)
        end = min(words_tier.xmax, end)
        if end <= start + 0.001:
            continue
        punctuation.append((mapped_boundary, Interval(start, end, char)))

    # Clip lexical intervals around the restored punctuation.  The labels and
    # English MFA word instances stay untouched; only their ownership ranges
    # are shortened where a punctuation anchor crosses a word boundary.
    for boundary, punct in punctuation:
        # Clip only the immediate left/right owners.  In particular, do not
        # shorten every word on one side of the boundary: that would turn a
        # local punctuation anchor into a global lexical extension.
        left_update = None
        right_update = None
        if boundary > 0:
            left = lexical[boundary - 1]
            left_end = min(left.xmax, punct.xmin)
            if left_end <= left.xmin + AXIS_EPS:
                continue
            left_update = Interval(left.xmin, left_end, left.text)
        if boundary < len(lexical):
            right = lexical[boundary]
            right_start = max(right.xmin, punct.xmax)
            if right_start >= right.xmax - AXIS_EPS:
                continue
            right_update = Interval(right_start, right.xmax, right.text)
        if left_update is not None:
            lexical[boundary - 1] = left_update
        if right_update is not None:
            lexical[boundary] = right_update
    lexical = [iv for iv in lexical if iv.xmax > iv.xmin + 0.001]

    # Preserve silence intervals and replace only punctuation/lexical content.
    # Accepted punctuation owns only its authoritative anchor.  Split any
    # explicit silence intersecting that anchor and preserve every uncovered
    # remainder as silence for the strict interior-SP audit.
    silences: list[Interval] = []
    for silence in (iv for iv in current if is_silence(iv.text) or not iv.text.strip()):
        pieces = [(silence.xmin, silence.xmax)]
        for _, punct in punctuation:
            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if piece_end <= punct.xmin + AXIS_EPS or piece_start >= punct.xmax - AXIS_EPS:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < punct.xmin - AXIS_EPS:
                    next_pieces.append((piece_start, punct.xmin))
                if piece_end > punct.xmax + AXIS_EPS:
                    next_pieces.append((punct.xmax, piece_end))
            pieces = next_pieces
        silences.extend(Interval(start, end, silence.text)
                        for start, end in pieces if end > start + AXIS_EPS)
    words_tier.intervals = sorted(lexical + silences + [iv for _, iv in punctuation],
                                  key=lambda iv: (iv.xmin, iv.xmax, iv.text))
    geometry_fixed = _repair_authority_punctuation_geometry(
        words_tier, reference_text, punct_entries)
    reference_fixed = _repair_reference_authority_colocated_silence(
        words_tier, reference_text)
    canonical = _reconcile_publication_geometry(words_tier)
    words_tier.intervals = canonical.intervals
    return len(punctuation) + geometry_fixed + reference_fixed


def _restore_reference_punctuation(words_tier: Tier, reference_text: str,
                                   punct_entries: list[dict] | None = None) -> int:
    """Incrementally reconcile reference punctuation ownership.

    The current words interval list is transactional input: every observed
    punctuation interval is carried into the output.  A partial projection is
    repaired by adding only missing marks, never by rebuilding lexical and
    silence intervals from scratch.  A new mark may own one explicit local
    silence when a matching CTC anchor overlaps it, or when that silence is
    the unique gap at the authoritative boundary.  Voiced lexical intervals
    are never carved by this resolver.
    """
    allowed = '，。…！？、；：,.!?;:～'
    ref_puncts: list[dict] = []
    ref_lexical: list[str] = []
    for unit in _extract_word_chars(reference_text):
        if is_punct(unit) and unit.strip() in allowed:
            ref_puncts.append({"label": unit.strip(),
                               "boundary": len(ref_lexical)})
        elif is_word_like(unit):
            ref_lexical.append(unit)

    current = list(words_tier.intervals)
    lexical = [iv for iv in current
               if iv.text.strip() and not is_silence(iv.text)
               and not is_punct(iv.text.strip())]
    desired = [item["label"] for item in ref_puncts]

    # Preserve the existing hyphenated-English grouping rule when projecting
    # reference lexical boundaries onto timed lexical intervals.
    current_boundary = [0]
    current_index = 0
    for unit in ref_lexical:
        target = re.sub(r'[^a-z0-9]', '', unit.lower())
        if '-' in unit and target:
            compact = ''
            start = current_index
            while current_index < len(lexical):
                probe = lexical[current_index].text.strip()
                if not is_english_token(probe):
                    break
                compact += re.sub(r'[^a-z0-9]', '', probe.lower())
                current_index += 1
                if compact == target:
                    break
            if current_index == start and current_index < len(lexical):
                current_index += 1
        elif current_index < len(lexical):
            current_index += 1
        current_boundary.append(current_index)

    def boundary_owners(boundary: int) -> tuple[Interval | None, Interval | None,
                                                  float, float]:
        mapped = (current_boundary[boundary]
                  if 0 <= boundary < len(current_boundary)
                  else len(lexical))
        prev = lexical[mapped - 1] if mapped > 0 else None
        nxt = lexical[mapped] if mapped < len(lexical) else None
        start = prev.xmax if prev is not None else words_tier.xmin
        end = nxt.xmin if nxt is not None else words_tier.xmax
        return prev, nxt, min(start, end), max(start, end)

    # Alignment is used only to identify missing occurrences.  All observed
    # punctuation, including extras and wrong-boundary marks, remains in the
    # transaction and stays available to the publication veto.
    observed: list[dict] = []
    lexical_before = 0
    for index, interval in enumerate(current):
        text = interval.text.strip()
        if text and not is_silence(text) and not is_punct(text):
            lexical_before += 1
        elif text in allowed:
            observed.append({"label": text, "boundary": lexical_before,
                             "interval_index": index})
    matched, missing, extra = _align_reference_punctuation(ref_puncts, observed)
    boundary_errors = {
        ref_index for ref_index, obs_index in matched
        if ref_puncts[ref_index]["boundary"] != observed[obs_index]["boundary"]
    }
    safe_projection = not extra and not boundary_errors
    missing_indices = {item["index"] for item in missing}

    # Parse anchors once in CTC order.  Matching an anchor to the missing
    # occurrence is local-gap based rather than same-character ordinal based,
    # which is important when the same mark occurs more than once.
    anchors: list[dict] = []
    for entry_index, entry in enumerate(punct_entries or []):
        char = str(entry.get('word', '')).strip()
        if char not in allowed:
            continue
        try:
            start, end = float(entry['start_s']), float(entry['end_s'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            anchors.append({"label": char, "start": start, "end": end,
                            "index": entry_index})
    anchors.sort(key=lambda item: (item["start"], item["end"], item["index"]))

    used_anchor_indices: set[int] = set()
    last_anchor_position = -1
    additions: list[tuple[Interval, int]] = []
    removed_silence_indices: set[int] = set()
    for ref_index, item in enumerate(ref_puncts):
        if ref_index not in missing_indices:
            continue
        char = item["label"]
        prev, nxt, gap_start, gap_end = boundary_owners(item["boundary"])
        if gap_end <= gap_start + AXIS_EPS:
            continue
        local_silences = [
            (index, interval) for index, interval in enumerate(current)
            if is_silence(interval.text) and interval.text.strip()
            and interval.xmin >= gap_start - AXIS_EPS
            and interval.xmax <= gap_end + AXIS_EPS
            and interval.xmax > interval.xmin + AXIS_EPS
        ]

        def supports(anchor: dict, silence: Interval) -> bool:
            return (anchor["label"] == char
                    and anchor["end"] > silence.xmin + AXIS_EPS
                    and anchor["start"] < silence.xmax - AXIS_EPS)

        candidates = [anchor for anchor in anchors
                      if anchor["index"] not in used_anchor_indices
                      and anchor["index"] > last_anchor_position
                      and any(supports(anchor, silence)
                              for _, silence in local_silences)]
        same_char_anchors = [anchor for anchor in anchors
                             if anchor["label"] == char
                             and anchor["index"] not in used_anchor_indices
                             and anchor["index"] > last_anchor_position]
        if candidates:
            def anchor_key(anchor: dict) -> tuple[float, float, int]:
                overlap = max(
                    min(anchor["end"], silence.xmax)
                    - max(anchor["start"], silence.xmin)
                    for _, silence in local_silences
                    if supports(anchor, silence))
                gap_center = (gap_start + gap_end) / 2.0
                anchor_center = (anchor["start"] + anchor["end"]) / 2.0
                return (-overlap, abs(anchor_center - gap_center),
                        anchor["index"])
            anchor = min(candidates, key=anchor_key)
            used_anchor_indices.add(anchor["index"])
            last_anchor_position = anchor["index"]
            supported = [(index, silence) for index, silence in local_silences
                         if supports(anchor, silence)]
            # Multiple disjoint local gaps are ambiguous; retain them as
            # explicit silence rather than assigning one by guesswork.
            if len(supported) != 1:
                continue
            silence_index, silence = supported[0]
        elif not same_char_anchors and len(local_silences) == 1:
            # Reference-only fallback: a unique local gap is evidence, but a
            # missing gap must never be synthesized from lexical duration.
            silence_index, silence = local_silences[0]
        else:
            continue

        additions.append((Interval(silence.xmin, silence.xmax, char),
                          silence_index))
        removed_silence_indices.add(silence_index)

    if additions:
        # Commit only after all missing occurrences have been resolved.  This
        # is the transactional boundary: existing punctuation is untouched.
        committed = [interval for index, interval in enumerate(current)
                     if index not in removed_silence_indices]
        committed.extend(interval for interval, _ in additions)
        committed.sort(key=lambda interval: (interval.xmin, interval.xmax,
                                             interval.text))
        words_tier.intervals = committed

    changed = len(additions)
    current_puncts = [iv.text.strip() for iv in words_tier.intervals
                      if iv.text.strip() in allowed]
    if current_puncts == desired and safe_projection:
        # Existing exact projections may still contain an authority anchor
        # duplicated by explicit silence.  This is a local repair only.
        changed += _repair_authority_punctuation_geometry(
            words_tier, reference_text, punct_entries)
        changed += _repair_reference_authority_colocated_silence(
            words_tier, reference_text)
        canonical = _reconcile_publication_geometry(words_tier)
        words_tier.intervals = canonical.intervals
    return changed


def _normalize_authority_short_interword_silence(
        textgrid: TextGrid, reference_text: str,
        punct_entries: list[dict] | None = None) -> int:
    """Compatibility no-op; final ownership belongs to the energy resolver.

    The former implementation performed a fixed-left authority merge and
    directly edited source/derived phones.  Keeping that second owner pass
    would make final words disagree with the immutable visual snapshot.
    """
    return 0

def _clip_pinyin_phones_to_words(pp_tier: Tier, words_tier: Tier) -> int:
    """Keep phones only when one semantic owner contains them uniquely.

    Strongest-overlap assignment is deliberately forbidden: a crossing phone
    remains unchanged so the publication audit can veto its ambiguous lineage.
    """
    words = [iv for iv in words_tier.intervals if iv.text.strip()]
    changed = 0
    clipped: list[Interval] = []
    for phone in pp_tier.intervals:
        if not phone.text.strip() or is_silence(phone.text):
            clipped.append(phone)
            continue
        owners = [word for word in words
                  if phone.xmin >= word.xmin - AXIS_EPS
                  and phone.xmax <= word.xmax + AXIS_EPS]
        if len(owners) != 1:
            clipped.append(phone)
            continue
        owner = owners[0]
        start = max(phone.xmin, owner.xmin)
        end = min(phone.xmax, owner.xmax)
        if end <= start + 0.001:
            clipped.append(phone)
            continue
        if start != phone.xmin or end != phone.xmax:
            changed += 1
        clipped.append(Interval(start, end, phone.text))
    pp_tier.intervals = [iv for iv in clipped if iv.xmax > iv.xmin + 0.001
                         or not iv.text.strip()]
    return changed


def _resolve_phone_owner_overlaps(
        phone_tier: Tier | None, words_tier: Tier | None) -> tuple[int, list[dict]]:
    """Resolve a crossing phone only when its semantic owner is clear.

    A phone that crosses a word boundary is not automatically an error.  MFA
    can place a transition a few frames on the wrong side of the word edge.
    The owner is inferred from the amount of acoustic time inside each display
    owner.  A strict majority with a meaningful margin is clipped to that
    owner's boundary; ties and silence phones remain untouched and are
    reported as ambiguous.

    This deliberately does not use ``strongest overlap`` as a universal
    assignment rule.  It is allowed only for a non-silence phone with a
    majority owner (>=55% of the phone and >=10% more than the runner-up).
    Otherwise the publication contract must reject the interval rather than
    silently moving a phoneme to a neighbouring word.
    """
    if phone_tier is None or words_tier is None:
        return 0, []

    owners = [iv for iv in words_tier.intervals if iv.xmax > iv.xmin + AXIS_EPS]
    fixed = 0
    ambiguous: list[dict] = []
    resolved: list[Interval] = []
    for index, phone in enumerate(phone_tier.intervals):
        if not phone.text.strip() or phone.duration <= AXIS_EPS:
            resolved.append(phone)
            continue

        containing = [owner for owner in owners
                      if phone.xmin >= owner.xmin - AXIS_EPS
                      and phone.xmax <= owner.xmax + AXIS_EPS]
        if len(containing) == 1:
            resolved.append(phone)
            continue

        candidates = []
        for owner_index, owner in enumerate(owners):
            overlap = min(phone.xmax, owner.xmax) - max(phone.xmin, owner.xmin)
            if overlap > AXIS_EPS:
                candidates.append((overlap, owner_index, owner))
        candidates.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        candidate_detail = [
            {"owner_index": owner_index, "label": owner.text.strip(),
             "overlap_s": round(overlap, 6)}
            for overlap, owner_index, owner in candidates[:4]
        ]

        # A silence phone is evidence of a pause, not a lexical phone.  Never
        # assign it to one of two neighbouring words by overlap strength.
        if is_silence(phone.text) or not candidates:
            ambiguous.append({"index": index, "label": phone.text.strip(),
                              "start_s": phone.xmin, "end_s": phone.xmax,
                              "reason": "silence_or_no_owner" if is_silence(phone.text)
                              else "no_owner",
                              "candidate_owners": candidate_detail})
            resolved.append(phone)
            continue

        duration = max(phone.duration, AXIS_EPS)
        top_overlap, _, top_owner = candidates[0]
        second_overlap = candidates[1][0] if len(candidates) > 1 else 0.0
        clear_majority = (
            top_overlap / duration >= 0.55
            and top_overlap - second_overlap >= max(AXIS_EPS, duration * 0.10)
        )
        if not clear_majority:
            ambiguous.append({"index": index, "label": phone.text.strip(),
                              "start_s": phone.xmin, "end_s": phone.xmax,
                              "reason": "owner_tie_or_weak_majority",
                              "candidate_owners": candidate_detail})
            resolved.append(phone)
            continue

        start = max(phone.xmin, top_owner.xmin)
        end = min(phone.xmax, top_owner.xmax)
        if end <= start + AXIS_EPS:
            ambiguous.append({"index": index, "label": phone.text.strip(),
                              "start_s": phone.xmin, "end_s": phone.xmax,
                              "reason": "owner_clip_empty",
                              "candidate_owners": candidate_detail})
            resolved.append(phone)
            continue
        resolved.append(Interval(start, end, phone.text))
        if start != phone.xmin or end != phone.xmax:
            fixed += 1

    phone_tier.intervals = [iv for iv in resolved
                            if iv.xmax > iv.xmin + AXIS_EPS
                            or not iv.text.strip()]
    return fixed, ambiguous


def _fix_non_english_pp_overlaps(pp_tier: Tier) -> int:
    """De-overlap Chinese/punctuation phones without rewriting strict English."""
    intervals = list(pp_tier.intervals)
    fixed = 0
    for index in range(len(intervals) - 1):
        cur, nxt = intervals[index], intervals[index + 1]
        if (cur.xmax <= nxt.xmin + 0.001
                or cur.text.strip().startswith(EN_PHONE_PREFIX)
                or nxt.text.strip().startswith(EN_PHONE_PREFIX)):
            continue
        midpoint = round((cur.xmax + nxt.xmin) / 2.0, 4)
        if midpoint <= cur.xmin + 0.001 or nxt.xmax <= midpoint + 0.001:
            continue
        intervals[index] = Interval(cur.xmin, midpoint, cur.text)
        intervals[index + 1] = Interval(midpoint, nxt.xmax, nxt.text)
        fixed += 1
    pp_tier.intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001
                         or not iv.text.strip()]
    return fixed


# ---------------------------------------------------------------------------
# Sequence alignment: CTC/MFA word tokens -> reference word units
# ---------------------------------------------------------------------------

def _word_matches(ctc_token: str, ref_unit: str) -> bool:
    """Check if a word-tier token plausibly matches a reference word unit.

    CJK units must match their pinyin reading exactly.
    Alpha-group units (English / NVV) use deterministic textual matching.
    CTC pinyin is never a substitute for a reference English word: raw CTC
    fragments remain provenance only and canonical reference projection owns
    the English MFA surface.
    """
    c = ctc_token.strip().lower()
    r = ref_unit.lower()

    if is_cjk(ref_unit):
        try:
            py = lazy_pinyin(ref_unit, style=Style.TONE3,
                            neutral_tone_with_five=True, errors="default")
            return py is not None and len(py) > 0 and py[0] == c
        except Exception:
            return False

    # Alpha group (English word or NVV tag)
    if not r.isascii():
        return False
    # A tone-bearing pinyin token is never an English candidate, even when
    # its spelling is a substring/equal match for an authority token.
    if is_pinyin_syllable(c) or re.fullmatch(r"[0-9]+", c):
        return False

    # Direct substring containment
    if c in r or r in c:
        return True

    # Single-letter CTC token -> fragment of the English word
    if len(c) == 1 and c.isalpha():
        return c in r

    # NVV token matching
    c_clean = c.strip('<>')
    r_clean = r.strip('<>')
    if c_clean in r_clean or r_clean in c_clean:
        return True

    return False


def _align_word_sequences(ctc_seq: list[str],
                          ref_seq: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch global alignment of CTC tokens to reference units.

    Returns a list of ``(ctc_idx, ref_idx)`` pairs.  *ctc_idx* may be
    ``None`` (reference-only gap) and *ref_idx* may be ``None`` (CTC-only
    gap — tokenizer fragment to be merged).

    Match cost is 0 when :func:`_word_matches` returns True, 1 otherwise.
    Gap cost is 1 on both axes.
    """
    n, m = len(ctc_seq), len(ref_seq)
    INF = n + m + 10

    # dp[i][j] = min cost for ctc_seq[:i] ↔ ref_seq[:j]
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i          # skip all CTC tokens
    for j in range(1, m + 1):
        dp[0][j] = j          # skip all ref units

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = 0 if _word_matches(ctc_seq[i - 1], ref_seq[j - 1]) else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # skip CTC token
                dp[i][j - 1] + 1,          # skip ref unit
                dp[i - 1][j - 1] + match_cost,  # align
            )

    # Backtrack — gap-first tie-breaking.
    # When a CTC gap and a fuzzy match have the same optimal cost,
    # prefer the gap so the *earlier* CTC token consumes the reference
    # unit and later tokens are gapped.  For exact matches (CJK pinyin,
    # NVV, English substring) the match path is always strictly cheaper,
    # so this order does not affect those cases.
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            pairs.append((None, j - 1))
            j -= 1
        else:
            # i > 0 and j > 0 — must be a match
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
    pairs.reverse()
    return pairs


def _alpha_text_matches(token: str, ref: str) -> bool:
    """Check if an alpha-group word token matches a reference word unit.

    Uses textual matching only.  It is deliberately not a pinyin-to-English
    transliterator: English lexical authority comes from the canonical
    reference projection, not CTC phonetic guesses.
    """
    c = token.strip().lower()
    r = ref.lower()

    # Never infer an English reference word from a pinyin syllable.  Named
    # variants such as rui4+ya4 -> ria are handled only by an explicit,
    # reference-bound canonicalization path before English MFA.
    if is_pinyin_syllable(c) or re.fullmatch(r"[0-9]+", c):
        return False

    c_compact = re.sub(r'[^a-z0-9]', '', c)
    r_compact = re.sub(r'[^a-z0-9]', '', r)
    if c_compact and r_compact and c_compact == r_compact:
        return True

    # Direct substring containment
    if c in r or r in c:
        return True

    # Single-letter CTC token -> fragment of the English word
    if len(c) == 1 and c.isalpha():
        return c in r

    # NVV token matching (strip angle brackets)
    c_clean = c.strip('<>')
    r_clean = r.strip('<>')
    if c_clean in r_clean or r_clean in c_clean:
        return True

    # Do not turn a pinyin token into an English reference word.
    if len(c) >= 2 and c[-1].isdigit() and c[:-1].isalpha():
        return False

    return False


def _fallback_cjk_alignment(raw_text: str, words_tier: Tier | None) -> dict:
    """Align fallback CJK source characters to realised pinyin words.

    A no-reference run has no lexical authority: the ASR transcript can have
    insertions/deletions while the MFA words tier still contains a valid,
    monotonic acoustic sequence.  The old positional projection treated one
    missing source character as a global pinyin shift.  This small global
    alignment keeps that shift local and returns only exact pinyin matches as
    safe projection evidence.  It never makes a non-matching token look
    correct.
    """
    source = [ch for ch in raw_text.replace("<sp1>", "") if is_cjk(ch)]
    actual: list[str] = []
    if words_tier is not None:
        actual = [iv.text.strip() for iv in words_tier.intervals
                  if is_pinyin_syllable(iv.text.strip())]

    def _base(value: str) -> str:
        return re.sub(r"\d+$", "", value).lower()

    expected = []
    for char in source:
        try:
            values = lazy_pinyin(char, style=Style.TONE3,
                                 neutral_tone_with_five=True)
        except Exception:
            values = []
        expected.append(_base(values[0]) if values else "")
    observed = [_base(token) for token in actual]

    n, m = len(expected), len(observed)
    if not n or not m:
        return {
            "source_count": n, "actual_count": m, "exact_matches": 0,
            "source_only": n, "actual_only": m, "exact_rate": 0.0,
            "actual_to_source": {}, "source_only_internal": False,
            "actual_only_internal": False, "safe": False,
        }

    # Exact matches are free.  A substitution is deliberately more expensive
    # than deleting/inserting one item, so a wrong pinyin cannot consume a
    # source character and create another global shift.
    inf = n + m + 20
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = 0 if expected[i - 1] and expected[i - 1] == observed[j - 1] else 3
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + match)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        # Prefer exact diagonal matches, then source/actual gaps, then a
        # substitution.  This is deterministic for repeated syllables.
        if (i and j and expected[i - 1] and expected[i - 1] == observed[j - 1]
                and dp[i][j] == dp[i - 1][j - 1]):
            pairs.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        elif j and dp[i][j] == dp[i][j - 1] + 1:
            pairs.append((None, j - 1))
            j -= 1
        else:
            pairs.append((i - 1, j - 1))
            i -= 1; j -= 1
    pairs.reverse()

    exact_pairs = [(si, ai) for si, ai in pairs
                   if si is not None and ai is not None
                   and expected[si] and expected[si] == observed[ai]]
    matched_source = {si for si, _ in exact_pairs}
    matched_actual = {ai for _, ai in exact_pairs}
    source_only = [idx for idx in range(n) if idx not in matched_source]
    actual_only = [idx for idx in range(m) if idx not in matched_actual]
    last_source = max(matched_source, default=-1)
    last_actual = max(matched_actual, default=-1)
    source_only_internal = any(idx < last_source for idx in source_only)
    actual_only_internal = any(idx < last_actual for idx in actual_only)
    exact_rate = len(exact_pairs) / m if m else 0.0
    # A safe fallback repair requires every realised pinyin token to map to a
    # source character.  Source-only tokens are tolerated because ASR can
    # over-transcribe, but only within a bounded amount.
    source_budget = max(3, int(math.ceil(n * 0.20)))
    safe = bool(matched_source) and not actual_only and exact_rate >= 0.85 \
        and len(source_only) <= source_budget
    return {
        "source_count": n,
        "actual_count": m,
        "exact_matches": len(exact_pairs),
        "source_only": len(source_only),
        "actual_only": len(actual_only),
        "exact_rate": round(exact_rate, 3),
        "source_only_internal": source_only_internal,
        "actual_only_internal": actual_only_internal,
        "safe": safe,
        "actual_to_source": {ai: si for si, ai in exact_pairs},
    }


def _fallback_punctuation_projection(
        raw_text: str, words_tier: Tier | None,
        ctc_tokens: list[dict] | None = None) -> dict:
    """Build a digest-bound source-to-final punctuation projection.

    The fallback surface ledger remains the immutable source of labels.  This
    additive proof only translates source lexical boundaries to the realised
    final lexical sequence.  CJK anchors come from the independently
    recomputed safe CJK alignment; all other anchors require exact lexical
    identity and unique occurrence on both surfaces.  A repeated exact
    identity may anchor only when already-safe neighboring anchors partition
    every source/final occurrence into corresponding singleton intervals.
    Unmatched final NVV tokens remain deliberately unclaimed.  A source
    boundary inside such an anchor interval is usable only when exactly one
    candidate final boundary has one positive local owner (existing
    punctuation, one explicit silence, or one exact positive CTC gap).
    """
    surface = _fallback_punctuation_surface_ledger(raw_text)
    alignment = _fallback_cjk_alignment(raw_text, words_tier)
    source_lexical = [
        unit for unit in _extract_word_chars(str(raw_text or ""))
        if is_word_like(unit)
    ]
    source_units = [
        {"ordinal": ordinal, "text": unit,
         "identity": _lexical_identity(unit),
         "is_cjk": is_cjk(unit)}
        for ordinal, unit in enumerate(source_lexical)
    ]
    final_units = []
    if words_tier is not None:
        final_units = [
            {"ordinal": final_ordinal, "text": interval.text.strip(),
             "identity": _lexical_identity(interval.text),
             "is_cjk": is_pinyin_syllable(interval.text.strip()),
             "interval": interval}
            for final_ordinal, interval in enumerate(
                interval for interval in words_tier.intervals
                if interval.text.strip() and not is_silence(interval.text)
                and not is_punct(interval.text))
        ]

    actual_to_source = alignment.get("actual_to_source", {})
    cjk_source_ordinals = [
        index for index, item in enumerate(source_units) if item["is_cjk"]
    ]
    anchors_by_source: dict[int, dict] = {}
    anchors_by_final: dict[int, dict] = {}
    actual_cjk_ordinal = 0
    mapping_errors: list[str] = []

    def add_anchor(source_ordinal: int, final_ordinal: int,
                   kind: str) -> None:
        existing_source = anchors_by_source.get(source_ordinal)
        existing_final = anchors_by_final.get(final_ordinal)
        if ((existing_source is not None
             and existing_source["final_lexical_ordinal"] != final_ordinal)
                or (existing_final is not None
                    and existing_final["source_lexical_ordinal"]
                    != source_ordinal)):
            mapping_errors.append("anchor_identity_conflict")
            return
        anchor = {
            "final_lexical_ordinal": final_ordinal,
            "source_lexical_ordinal": source_ordinal,
            "source_text": source_units[source_ordinal]["text"],
            "final_text": final_units[final_ordinal]["text"],
            "anchor_kind": kind,
        }
        anchors_by_source[source_ordinal] = anchor
        anchors_by_final[final_ordinal] = anchor

    for final_item in final_units:
        if not final_item["is_cjk"]:
            continue
        source_cjk_ordinal = actual_to_source.get(actual_cjk_ordinal)
        if type(source_cjk_ordinal) is not int or not (
                0 <= source_cjk_ordinal < len(cjk_source_ordinals)):
            mapping_errors.append("cjk_actual_to_source_missing")
        else:
            add_anchor(cjk_source_ordinals[source_cjk_ordinal],
                       final_item["ordinal"], "safe_cjk_alignment")
        actual_cjk_ordinal += 1
    if actual_cjk_ordinal != alignment.get("actual_count", -1):
        mapping_errors.append("cjk_actual_count_mismatch")

    # Globally unique exact non-CJK identity is intrinsically unambiguous.
    # Repeated labels require the bounded proof below; choosing their global
    # positions directly would turn lexical order into authority.  In
    # particular, an unmatched final RIA never acquires a CJK source identity
    # merely because it occupies an omission interval.
    source_non_cjk: dict[str, list[int]] = {}
    final_non_cjk: dict[str, list[int]] = {}
    for item in source_units:
        if not item["is_cjk"]:
            source_non_cjk.setdefault(item["identity"], []).append(
                item["ordinal"])
    for item in final_units:
        if not item["is_cjk"]:
            final_non_cjk.setdefault(item["identity"], []).append(
                item["ordinal"])
    for identity, source_ordinals in source_non_cjk.items():
        final_ordinals = final_non_cjk.get(identity, [])
        if len(source_ordinals) == 1 and len(final_ordinals) == 1:
            add_anchor(source_ordinals[0], final_ordinals[0],
                       "exact_non_cjk_identity")

    # Repeated exact identities can still be authoritative when existing
    # safe anchors separate every occurrence into its own corresponding
    # interval.  The interval vector is compared on both axes; each non-empty
    # bucket must be a singleton.  Thus two BREATHING tokens separated by CJK
    # anchors are pairable, while two indistinguishable occurrences inside
    # one anchor interval are never paired by position.
    base_mapped = sorted(
        anchors_by_source.values(),
        key=lambda item: item["source_lexical_ordinal"])
    base_monotonic = not any(
        left["final_lexical_ordinal"] >= right["final_lexical_ordinal"]
        for left, right in zip(base_mapped, base_mapped[1:]))
    if not base_monotonic:
        mapping_errors.append("anchors_not_strictly_monotonic")
    else:
        interval_count = len(base_mapped) + 1

        def anchor_interval(ordinal: int, key: str) -> int:
            return sum(anchor[key] < ordinal for anchor in base_mapped)

        for identity, source_ordinals in sorted(source_non_cjk.items()):
            final_ordinals = final_non_cjk.get(identity, [])
            if len(source_ordinals) <= 1 or len(source_ordinals) != len(
                    final_ordinals):
                continue
            source_buckets = [[] for _ in range(interval_count)]
            final_buckets = [[] for _ in range(interval_count)]
            for ordinal in source_ordinals:
                source_buckets[anchor_interval(
                    ordinal, "source_lexical_ordinal")].append(ordinal)
            for ordinal in final_ordinals:
                final_buckets[anchor_interval(
                    ordinal, "final_lexical_ordinal")].append(ordinal)
            source_counts = [len(bucket) for bucket in source_buckets]
            final_counts = [len(bucket) for bucket in final_buckets]
            if source_counts != final_counts:
                mapping_errors.append(
                    "bounded_repeated_non_cjk_identity_distribution_mismatch")
                continue
            if any(count > 1 for count in source_counts):
                mapping_errors.append(
                    "bounded_repeated_non_cjk_identity_ambiguous")
                continue
            for source_bucket, final_bucket in zip(
                    source_buckets, final_buckets):
                if source_bucket:
                    add_anchor(source_bucket[0], final_bucket[0],
                               "bounded_repeated_non_cjk_identity")

    mapped = sorted(anchors_by_source.values(),
                    key=lambda item: item["source_lexical_ordinal"])
    if any(left["final_lexical_ordinal"] >= right["final_lexical_ordinal"]
           for left, right in zip(mapped, mapped[1:])):
        mapping_errors.append("anchors_not_strictly_monotonic")
    mapped_source = {item["source_lexical_ordinal"] for item in mapped}
    mapped_final = {item["final_lexical_ordinal"] for item in mapped}
    omitted_source = [index for index in range(len(source_units))
                      if index not in mapped_source]
    unanchored_final = [index for index in range(len(final_units))
                        if index not in mapped_final]

    ctc_spans = []
    ctc_malformed = False
    for index, token in enumerate(ctc_tokens or []):
        if not isinstance(token, dict) or token.get("type", "word") != "word":
            continue
        try:
            start, end = float(token["start_s"]), float(token["end_s"])
        except (KeyError, TypeError, ValueError):
            ctc_malformed = True
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            ctc_malformed = True
            continue
        ctc_spans.append({"ordinal": len(ctc_spans), "start": start,
                          "end": end, "text": str(token.get("word", ""))})

    ctc_exact_final = (len(ctc_spans) == len(final_units)
                       and all(_lexical_identity(ctc_item["text"])
                               == final_item["identity"]
                               for ctc_item, final_item in
                               zip(ctc_spans, final_units)))

    def boundary_geometry(boundary: int) -> tuple[dict, str | None]:
        left = final_units[boundary - 1]["interval"] if boundary > 0 else None
        right = final_units[boundary]["interval"] if boundary < len(final_units) else None
        left_index = (next((index for index, interval in enumerate(
            words_tier.intervals) if interval is left), -1)
                      if words_tier is not None and left is not None else -1)
        right_index = (next((index for index, interval in enumerate(
            words_tier.intervals) if interval is right),
                            len(words_tier.intervals))
                       if words_tier is not None and right is not None
                       else (len(words_tier.intervals)
                             if words_tier is not None else 0))
        between = ([(index, interval) for index, interval in enumerate(
            words_tier.intervals) if left_index < index < right_index]
                   if words_tier is not None else [])
        existing = [(index, interval) for index, interval in between
                    if is_punct(interval.text) and not is_silence(interval.text)]
        silences = [(index, interval) for index, interval in between
                    if is_silence(interval.text)]
        ctc_gap = None
        if ctc_exact_final:
            ctc_left = (ctc_spans[boundary - 1]["end"] if boundary > 0
                        else (words_tier.xmin
                              if words_tier is not None else 0.0))
            ctc_right = (ctc_spans[boundary]["start"]
                         if boundary < len(ctc_spans)
                         else (words_tier.xmax
                               if words_tier is not None else ctc_left))
            ctc_gap = ([ctc_left, ctc_right]
                       if ctc_right > ctc_left else None)
        channels = []
        if len(existing) == 1 and existing[0][1].xmax > existing[0][1].xmin:
            channels.append("existing_punctuation")
        elif len(existing) > 1:
            channels.extend(["existing_punctuation"] * len(existing))
        elif len(silences) == 1 and silences[0][1].xmax > silences[0][1].xmin:
            channels.append("explicit_silence")
        elif len(silences) > 1:
            channels.extend(["explicit_silence"] * len(silences))
        elif ctc_gap is not None and ctc_gap[1] > ctc_gap[0] + AXIS_EPS:
            channels.append("positive_ctc_gap")
        geometry = {"owner_count": len(channels), "owners": channels,
                    "ctc_gap": ctc_gap}
        return geometry, (channels[0] if len(channels) == 1 else None)

    entries = []
    duplicate_boundaries: set[int] = set()
    seen_boundaries: set[int] = set()
    for item in surface["punctuation"]:
        source_boundary = item["lexical_boundary"]
        left_anchor = next((anchor for anchor in reversed(mapped)
                            if anchor["source_lexical_ordinal"]
                            < source_boundary), None)
        right_anchor = next((anchor for anchor in mapped
                             if anchor["source_lexical_ordinal"]
                             >= source_boundary), None)
        lower = (left_anchor["final_lexical_ordinal"] + 1
                 if left_anchor is not None else 0)
        upper = (right_anchor["final_lexical_ordinal"]
                 if right_anchor is not None else len(final_units))
        candidates = (list(range(max(0, lower),
                                 min(len(final_units), upper) + 1))
                      if lower <= upper else [])
        candidate_geometry = []
        for boundary in candidates:
            geometry, owner = boundary_geometry(boundary)
            candidate_geometry.append({"final_boundary": boundary,
                                       "owner_geometry": geometry,
                                       "owner": owner})
        positive = [candidate for candidate in candidate_geometry
                    if candidate["owner"] is not None]
        # Adjacent anchors already prove a single boundary.  Geometry is only
        # a disambiguator when unmatched source/final units leave more than
        # one boundary in the interval.  This permits an exact zero-gap NVV
        # boundary to proceed to the separately guarded frame-owner path.
        selected = (candidate_geometry[0] if len(candidate_geometry) == 1
                    else (positive[0] if len(positive) == 1 else None))
        final_boundary = (selected["final_boundary"]
                          if selected is not None else None)
        geometry = (selected["owner_geometry"] if selected is not None else
                    {"owner_count": 0, "owners": [], "ctc_gap": None})
        owner = selected["owner"] if selected is not None else None
        if final_boundary is not None:
            if final_boundary in seen_boundaries:
                duplicate_boundaries.add(final_boundary)
            seen_boundaries.add(final_boundary)
        entries.append({
            "source_index": item["source_index"],
            "source_boundary": source_boundary,
            "final_boundary": final_boundary,
            "lexical_boundary": final_boundary,
            "label": item["label"],
            "left_lexical_ordinal": final_boundary - 1
            if final_boundary is not None and final_boundary > 0 else None,
            "right_lexical_ordinal": final_boundary
            if final_boundary is not None
            and final_boundary < len(final_units) else None,
            "crosses_source_omission": (lower != upper
                                         or final_boundary != source_boundary),
            "candidate_final_boundaries": candidates,
            "positive_owner_candidates": [
                candidate["final_boundary"] for candidate in positive],
            "owner_geometry": geometry,
            "owner": owner,
        })

    reasons = list(mapping_errors)
    if not alignment.get("safe"):
        reasons.append("cjk_alignment_unsafe")
    if ctc_malformed:
        reasons.append("ctc_geometry_malformed")
    if any(not item["candidate_final_boundaries"] for item in entries):
        reasons.append("punctuation_anchor_interval_empty")
    if any(len(item["candidate_final_boundaries"]) > 1
           and len(item["positive_owner_candidates"]) != 1
           for item in entries):
        reasons.append("punctuation_boundary_owner_candidate_not_unique")
    if duplicate_boundaries:
        reasons.append("multiple_source_marks_same_final_boundary")
    if any(item["final_boundary"] is None for item in entries):
        reasons.append("projected_boundary_owner_not_unique_positive")
    safe = not reasons and bool(entries)
    projection = {
        "schema": FALLBACK_PUNCTUATION_PROJECTION_SCHEMA,
        "source_text": str(raw_text or ""),
        "source_digest": surface["source_digest"],
        "surface_ledger_digest": surface["digest"],
        "alignment": alignment,
        "source_lexical_count": len(source_units),
        "final_lexical_count": len(final_units),
        "mapped": mapped,
        "omitted_source_lexical_ordinals": omitted_source,
        "unanchored_final_lexical_ordinals": unanchored_final,
        "entries": entries,
        "safe": safe,
        "status": "verified" if safe else "rejected",
        "reasons": sorted(set(reasons)),
    }
    projection["digest"] = _evidence_digest(projection)
    return projection


def _fallback_cjk_cross_kind_owner_audit(
        raw_text: str, words_tier: Tier | None,
        projection: dict | None = None,
        ctc_tokens: list[dict] | None = None) -> tuple[bool, dict]:
    """Veto an omitted raw CJK owner replaced by an unanchored English/NVV.

    This is intentionally narrower than fallback lexical correspondence:
    only the immutable raw surface lexical units, final words, and the
    source/final anchors from the punctuation projection participate.  CTC,
    correspondence, and English ledgers are not independent redemption
    evidence.  A source CJK omission and a final English/NVV owner are unsafe
    only when they occupy the same bounded interval between mapped anchors;
    a native raw English token such as ``RIA`` is already an exact anchor and
    therefore remains publishable.
    """
    # Recompute the anchor projection at the publication boundary.  The
    # caller's cached projection is useful for punctuation authority, but it
    # must not become an independent or stale owner proof here.
    projection = _fallback_punctuation_projection(
        str(raw_text or ""), words_tier, ctc_tokens)

    source_units = [
        unit for unit in _extract_word_chars(str(raw_text or ""))
        if is_word_like(unit)
    ]
    final_units = []
    if words_tier is not None:
        final_units = [
            interval.text.strip()
            for interval in words_tier.intervals
            if interval.text.strip()
            and not is_silence(interval.text)
            and not is_punct(interval.text)
        ]

    anchors = []
    for item in projection.get("mapped", []) if isinstance(projection, dict) else []:
        if not isinstance(item, dict):
            continue
        source_ordinal = item.get("source_lexical_ordinal")
        final_ordinal = item.get("final_lexical_ordinal")
        if (type(source_ordinal) is int and type(final_ordinal) is int
                and 0 <= source_ordinal < len(source_units)
                and 0 <= final_ordinal < len(final_units)):
            anchors.append({
                "source_lexical_ordinal": source_ordinal,
                "final_lexical_ordinal": final_ordinal,
                "source_text": source_units[source_ordinal],
                "final_text": final_units[final_ordinal],
                "anchor_kind": item.get("anchor_kind"),
            })
    anchors.sort(key=lambda item: item["source_lexical_ordinal"])
    if any(left["source_lexical_ordinal"] >= right["source_lexical_ordinal"]
           or left["final_lexical_ordinal"] >= right["final_lexical_ordinal"]
           for left, right in zip(anchors, anchors[1:])):
        return False, {
            "status": "rejected",
            "reason": "mapped_anchors_not_strictly_monotonic",
            "mapped_anchor_count": len(anchors),
            "omitted_source_cjk_count": 0,
            "unanchored_final_english_nvv_count": 0,
            "independent_identity_proof": False,
            "buckets": [],
        }

    mapped_source = {item["source_lexical_ordinal"] for item in anchors}
    mapped_final = {item["final_lexical_ordinal"] for item in anchors}

    def bucket(ordinal: int, key: str) -> int:
        return sum(item[key] < ordinal for item in anchors)

    source_omissions = [
        {"source_lexical_ordinal": ordinal, "source_text": text,
         "bucket": bucket(ordinal, "source_lexical_ordinal")}
        for ordinal, text in enumerate(source_units)
        if is_cjk(text) and ordinal not in mapped_source
    ]
    final_candidates = [
        {"final_lexical_ordinal": ordinal, "final_text": text,
         "owner_kind": ("nvv" if is_nvv_token(text) else "english"),
         "bucket": bucket(ordinal, "final_lexical_ordinal")}
        for ordinal, text in enumerate(final_units)
        if ordinal not in mapped_final
        and (is_nvv_token(text) or is_english_token(text))
    ]

    source_by_bucket: dict[int, list[dict]] = {}
    final_by_bucket: dict[int, list[dict]] = {}
    for item in source_omissions:
        source_by_bucket.setdefault(item["bucket"], []).append(item)
    for item in final_candidates:
        final_by_bucket.setdefault(item["bucket"], []).append(item)

    findings = []
    for anchor_bucket in sorted(set(source_by_bucket) & set(final_by_bucket)):
        bounding_anchors = []
        if anchor_bucket > 0:
            bounding_anchors.append(dict(anchors[anchor_bucket - 1]))
        if anchor_bucket < len(anchors):
            bounding_anchors.append(dict(anchors[anchor_bucket]))
        findings.append({
            "bucket": anchor_bucket,
            "source_cjk_omitted": source_by_bucket[anchor_bucket],
            "final_unanchored_owners": final_by_bucket[anchor_bucket],
            "bounding_anchors": bounding_anchors,
            "independent_identity_proof": False,
        })
    details = {
        "status": "rejected" if findings else "verified",
        "mapped_anchor_count": len(anchors),
        "omitted_source_cjk_count": len(source_omissions),
        "unanchored_final_english_nvv_count": len(final_candidates),
        "buckets": findings,
    }
    return not findings, details


def _validate_fallback_punctuation_projection(
        projection: dict | None, raw_text: str | None = None,
        words_tier: Tier | None = None,
        ctc_tokens: list[dict] | None = None) -> tuple[bool, dict]:
    """Recompute and validate the additive fallback punctuation proof."""

    def canonical_mapped_anchors(value: object) -> list[dict] | None:
        """Compare anchor authority, not its mutable display spelling.

        NVV display normalization can wrap and uppercase a final label after
        this proof is built.  Ordinals, anchor kind, and both semantic
        identities remain exact authority and therefore stay in the
        comparison.  Malformed mapped rows fail closed as ``mapped_mismatch``.
        """
        if not isinstance(value, list):
            return None
        canonical = []
        for item in value:
            if not isinstance(item, dict):
                return None
            source_ordinal = item.get("source_lexical_ordinal")
            final_ordinal = item.get("final_lexical_ordinal")
            anchor_kind = item.get("anchor_kind")
            source_text = item.get("source_text")
            final_text = item.get("final_text")
            if (type(source_ordinal) is not int
                    or type(final_ordinal) is not int
                    or not isinstance(anchor_kind, str) or not anchor_kind
                    or not isinstance(source_text, str) or not source_text.strip()
                    or not isinstance(final_text, str) or not final_text.strip()):
                return None
            canonical.append({
                "source_lexical_ordinal": source_ordinal,
                "final_lexical_ordinal": final_ordinal,
                "anchor_kind": anchor_kind,
                "source_identity": _lexical_identity(source_text),
                "final_identity": _lexical_identity(final_text),
            })
        return canonical

    if not isinstance(projection, dict):
        return False, {"schema": FALLBACK_PUNCTUATION_PROJECTION_SCHEMA,
                       "status": "rejected", "reasons": ["missing"]}
    reasons = []
    if projection.get("schema") != FALLBACK_PUNCTUATION_PROJECTION_SCHEMA:
        reasons.append("schema_mismatch")
    digest = projection.get("digest")
    if not isinstance(digest, str) or digest != _evidence_digest(
            {key: value for key, value in projection.items() if key != "digest"}):
        reasons.append("digest_mismatch")
    if raw_text is not None and projection.get("source_text") != str(raw_text or ""):
        reasons.append("source_text_mismatch")
    if not reasons and words_tier is not None:
        expected = _fallback_punctuation_projection(
            str(projection.get("source_text", "")), words_tier, ctc_tokens)
        for key in ("source_digest", "surface_ledger_digest", "alignment",
                    "source_lexical_count", "final_lexical_count",
                    "omitted_source_lexical_ordinals",
                    "unanchored_final_lexical_ordinals"):
            if projection.get(key) != expected.get(key):
                reasons.append(f"{key}_mismatch")
        if (canonical_mapped_anchors(projection.get("mapped"))
                != canonical_mapped_anchors(expected.get("mapped"))):
            reasons.append("mapped_mismatch")
        projection_entries = [
            {key: value for key, value in item.items()
             if key not in {"owner_geometry", "owner",
                            "positive_owner_candidates"}}
            for item in projection.get("entries", [])
            if isinstance(item, dict)]
        expected_entries = [
            {key: value for key, value in item.items()
             if key not in {"owner_geometry", "owner",
                            "positive_owner_candidates"}}
            for item in expected.get("entries", [])
            if isinstance(item, dict)]
        if projection_entries != expected_entries:
            reasons.append("entries_mismatch")
    result = {
        "schema": FALLBACK_PUNCTUATION_PROJECTION_SCHEMA,
        "status": "verified" if not reasons and projection.get("safe") else "rejected",
        "reasons": sorted(set(reasons)) if reasons else list(
            projection.get("reasons", [])),
        "ledger_digest": projection.get("digest"),
    }
    return result["status"] == "verified", result


# Descriptive aliases keep the proof discoverable to audit/test callers.
_build_fallback_punctuation_projection = _fallback_punctuation_projection
_validate_fallback_punctuation_projection = _validate_fallback_punctuation_projection


def _build_authority_hanzi_tier(words_tier: Tier, raw_text: str,
                                warnings: list | None = None) -> Tier:
    """Project final words against the ordered authority semantic stream."""
    semantic = list(project_authority_semantics(raw_text))
    cursor = 0
    intervals: list[Interval] = []
    hidden_hyphen_fragments: set[int] = set()
    pinyin_count = 0
    cjk_count = sum(item["kind"] == "cjk" for item in semantic)

    def next_kind(kind: str) -> dict | None:
        nonlocal cursor
        # ``{target1}``/``{target2}`` are authority placeholders.  The braces
        # are syntax, not lexical units, and must not block the semantic
        # transition target -> 一/二 (or the following CJK word).  Treat only
        # these two wrapper marks as transparent; all other ``other`` units
        # remain visible and therefore continue to fail closed.
        while cursor < len(semantic) and (
                semantic[cursor]["kind"] == "punct"
                or (semantic[cursor]["kind"] == "other"
                    and semantic[cursor]["surface"] in {"{", "}"})):
            cursor += 1
        if cursor < len(semantic) and semantic[cursor]["kind"] == kind:
            item = semantic[cursor]
            cursor += 1
            return item
        return None

    for word_index, iv in enumerate(words_tier.intervals):
        token = iv.text.strip()
        if word_index in hidden_hyphen_fragments:
            intervals.append(Interval(iv.xmin, iv.xmax, ""))
            continue
        if is_silence(token) or not token:
            intervals.append(Interval(iv.xmin, iv.xmax, silence_label(iv.duration)))
            continue
        if is_punct(token):
            # Punctuation is optional in the publication projection.  Consume
            # only an exact next mark; a different/extra mark remains visible
            # for the audit rather than shifting later semantic units.
            expected = semantic[cursor] if cursor < len(semantic) else None
            if expected and expected["kind"] == "punct":
                if expected["surface"] == token:
                    cursor += 1
                intervals.append(Interval(iv.xmin, iv.xmax, token))
            else:
                intervals.append(Interval(iv.xmin, iv.xmax, token))
            continue
        if is_pinyin_syllable(token):
            pinyin_count += 1
            item = next_kind("cjk")
            intervals.append(Interval(iv.xmin, iv.xmax,
                                      item["surface"] if item else token))
            continue
        clean = token.strip("<>")
        if is_nvv_token(clean):
            item = next_kind("nvv")
            intervals.append(Interval(iv.xmin, iv.xmax,
                                      item["surface"] if item else clean))
            continue
        if is_english_token(clean) or is_english_fragment_token(clean):
            item = next_kind("english")
            if item is not None:
                observed = re.sub(r"[^a-z0-9]", "", clean.casefold())
                expected = re.sub(r"[^a-z0-9]", "", item["surface"].casefold())
                if "-" in item["surface"] and observed != expected:
                    compact = observed
                    group = [word_index]
                    for probe_index in range(word_index + 1,
                                             min(word_index + 6,
                                                 len(words_tier.intervals))):
                        probe = words_tier.intervals[probe_index].text.strip()
                        probe_clean = probe.strip("<>")
                        if not is_english_fragment_token(probe_clean):
                            break
                        compact += re.sub(r"[^a-z0-9]", "", probe_clean.casefold())
                        group.append(probe_index)
                        if compact == expected:
                            hidden_hyphen_fragments.update(group[1:])
                            break
                    if compact == expected:
                        label = item["surface"]
                        extra_index = group[-1] + 1
                        if (extra_index < len(words_tier.intervals)
                                and is_english_fragment_token(
                                    words_tier.intervals[extra_index].text.strip("<>"))
                                and (cursor >= len(semantic)
                                     or semantic[cursor]["kind"] != "english")):
                            if warnings is not None:
                                marker = "reference_hyphen_fragment_mismatch"
                                if marker not in warnings:
                                    warnings.append(marker)
                    else:
                        label = clean
                        if warnings is not None:
                            marker = "reference_hyphen_fragment_mismatch"
                            if marker not in warnings:
                                warnings.append(marker)
                else:
                    label = item["surface"] if observed == expected else clean
            else:
                label = clean
            intervals.append(Interval(iv.xmin, iv.xmax, label))
            continue
        item = next_kind("other")
        intervals.append(Interval(iv.xmin, iv.xmax,
                                  item["surface"] if item else clean))

    if warnings is not None and pinyin_count != cjk_count:
        warnings.append(
            f"hanzi tier mismatch: {pinyin_count} pinyin tokens vs "
            f"{cjk_count} reference CJK chars")
    return Tier("hanzi", words_tier.xmin, words_tier.xmax, intervals)


def _build_hanzi_tier(words_tier: Tier, raw_text: str,
                      warnings: list | None = None,
                      *, reference_authoritative: bool = False) -> Tier:
    """Build the *hanzi* tier by sequential mapping of word tokens to
    reference text units.

    **CJK characters**: authoritative references retain the historical
    positional projection.  Fallback transcripts use a monotonic pinyin/CJK
    alignment so one ASR insertion or deletion cannot shift every later label.

    **English / NVV tokens**: greedy substring matching against alpha
    reference units, handling tokenizer fragmentation (``li`` + ``ve``
    → ``live``) and MFA merging (``SURPRISE-OH`` → ``SURPRISE`` +
    ``OH``).

    **Punctuation**: passed through without consuming any cursor.

    **Silence**: silence label preserved.

    Emits warnings via *warnings* (when provided) if the number of
    pinyin-syllable tokens does not equal the number of reference CJK
    characters.
    """
    if reference_authoritative:
        return _build_authority_hanzi_tier(words_tier, raw_text, warnings)

    clean = raw_text.replace('<sp1>', '')
    char_units = _extract_word_chars(clean)

    # ── Separate reference units into CJK queue and alpha queue ──
    ref_cjk: list[str] = []     # CJK characters in reference order
    ref_alpha: list[str] = []   # English words / NVV tokens in reference order
    for u in char_units:
        if not is_word_like(u):
            continue            # skip punct in reference
        if is_cjk(u):
            ref_cjk.append(u)
        else:
            ref_alpha.append(u)

    # ── Build hanzi intervals ──
    intervals: list[Interval] = []
    cjk_idx = 0
    alpha_idx = 0
    hidden_hyphen_fragments: set[int] = set()
    failed_hyphen_fragments: set[int] = set()

    # Track pinyin-syllable count for defensive mismatch detection
    pinyin_count = 0
    fallback_projection = {}
    if not reference_authoritative:
        fallback_projection = _fallback_cjk_alignment(raw_text, words_tier).get(
            "actual_to_source", {})
    fallback_pinyin_ordinal = 0

    for word_index, iv in enumerate(words_tier.intervals):
        token = iv.text.strip()

        if word_index in hidden_hyphen_fragments:
            # Keep the strict-English word instance and its phone ledger, but
            # render a split hyphenated reference spelling only once.
            intervals.append(Interval(iv.xmin, iv.xmax, ""))
            continue

        if word_index in failed_hyphen_fragments:
            # A malformed reference-only group is deliberately left visible;
            # preserving the source fragment makes the later hard audit fail
            # closed instead of silently projecting a partial spelling.
            intervals.append(Interval(iv.xmin, iv.xmax, token.strip('<>')))
            continue

        # Silence → preserve the already-normalized words label exactly.
        # Reclassifying ``iv.duration`` here crosses the 200/500 ms boundary
        # through binary-float noise (for example 1.77 - 1.57 can be just
        # below 0.2), desynchronizing the derived hanzi tier from its frozen
        # owner tier.  Non-canonical legacy silence still uses integer ticks.
        if is_silence(iv.text) or not token:
            canonical_silence = _pure_silence_label(token)
            intervals.append(Interval(
                iv.xmin, iv.xmax,
                canonical_silence or _silence_label_from_ticks(
                    _duration_ticks(iv.xmin, iv.xmax))))
            continue

        # Punctuation → pass through, consume no cursor
        if is_punct(iv.text):
            intervals.append(Interval(iv.xmin, iv.xmax, iv.text))
            continue

        # ── Pinyin syllable → consume next CJK character ──
        if is_pinyin_syllable(token):
            pinyin_count += 1
            if not reference_authoritative and fallback_pinyin_ordinal in fallback_projection:
                label = ref_cjk[fallback_projection[fallback_pinyin_ordinal]]
                fallback_pinyin_ordinal += 1
            elif not reference_authoritative:
                label = token
                fallback_pinyin_ordinal += 1
            elif cjk_idx < len(ref_cjk):
                label = ref_cjk[cjk_idx]
                cjk_idx += 1
            else:
                # No more CJK chars — fall back to token text
                label = token
            intervals.append(Interval(iv.xmin, iv.xmax, label))
            continue

        # ── English / NVV token → greedy match against alpha refs ──
        # An MFA token may consume multiple reference alpha units
        # (merged case, e.g. SURPRISE-OH → "SURPRISE" + "OH").
        # Conversely, a reference unit may be split across multiple
        # MFA tokens (fragmented case, e.g. "li" + "ve" → "live").
        # Strip angle brackets: _finalize_textgrid may have already
        # wrapped NVV tokens with < > before we run.  Matching and
        # fallback labels must use the clean form to avoid bracket
        # pollution in the hanzi tier and misaligned cursors.
        clean_token = token.strip('<>')
        matched_refs: list[str] = []

        # A tokenizer can split one reference spelling (K-Pop) into adjacent
        # strict-English instances (kp/op).  This is projection-only: words
        # and strict phone provenance remain unchanged.
        if (alpha_idx < len(ref_alpha)
                and '-' in ref_alpha[alpha_idx]
                and is_english_token(clean_token)):
            target = ref_alpha[alpha_idx]
            target_compact = re.sub(r'[^a-z0-9]', '', target.lower())
            compact = ""
            group_indices: list[int] = []
            for probe_index in range(word_index,
                                     min(word_index + 6, len(words_tier.intervals))):
                probe = words_tier.intervals[probe_index].text.strip()
                if not is_english_token(probe):
                    break
                compact += re.sub(r'[^a-z0-9]', '', probe.lower())
                group_indices.append(probe_index)
                if compact == target_compact:
                    matched_refs.append(target)
                    alpha_idx += 1
                    hidden_hyphen_fragments.update(group_indices[1:])
                    if reference_authoritative:
                        # An immediately following English fragment is an
                        # extra only when it cannot begin the next reference
                        # unit.  Mark it so a trailing/foreign fragment does
                        # not pass as an unrelated lexical word.
                        extra_index = group_indices[-1] + 1
                        if extra_index < len(words_tier.intervals):
                            extra = words_tier.intervals[extra_index].text.strip()
                            next_ref = ref_alpha[alpha_idx] if alpha_idx < len(ref_alpha) else ""
                            if (is_english_token(extra)
                                    and (not next_ref
                                         or not _alpha_text_matches(extra, next_ref))):
                                failed_hyphen_fragments.add(extra_index)
                                if warnings is not None:
                                    marker = "reference_hyphen_fragment_mismatch"
                                    if marker not in warnings:
                                        warnings.append(marker)
                    break
                if reference_authoritative:
                    # Reference-only projection accepts only an exact ordered
                    # concatenation.  A subsequence/substring match can
                    # consume a partial, reordered, or foreign fragment.
                    if not target_compact.startswith(compact):
                        break
                else:
                    # Preserve legacy no-reference behaviour, including its
                    # permissive fragment matching.
                    probe_iter = iter(target_compact)
                    if not all(any(ch == target_ch for target_ch in probe_iter)
                               for ch in compact):
                        break

            if reference_authoritative and not matched_refs:
                failed_hyphen_fragments.update(group_indices)
                if warnings is not None:
                    marker = "reference_hyphen_fragment_mismatch"
                    if marker not in warnings:
                        warnings.append(marker)

        if matched_refs:
            intervals.append(Interval(iv.xmin, iv.xmax, matched_refs[0]))
            continue

        while alpha_idx < len(ref_alpha):
            ref_unit = ref_alpha[alpha_idx]
            if _alpha_text_matches(clean_token, ref_unit):
                matched_refs.append(ref_unit)
                alpha_idx += 1
                # Check if the token also consumes the NEXT ref unit
                # by seeing whether both refs are substrings of the token
                continue_match = False
                if alpha_idx < len(ref_alpha):
                    next_ref = ref_alpha[alpha_idx].lower()
                    if next_ref in clean_token.lower() or clean_token.lower() in next_ref:
                        continue_match = True
                if not continue_match:
                    break
            else:
                break

        if matched_refs:
            # Use the first consumed reference unit as the label
            # (for the common single-consumption case this is just
            # the matched ref unit)
            label = matched_refs[0]
        elif clean_token.isascii() and all(c.isalpha() or c == '-' for c in clean_token):
            # NVV / English token with no matching ref unit — use as-is
            label = clean_token
        else:
            label = clean_token

        intervals.append(Interval(iv.xmin, iv.xmax, label))

    # ── Defensive mismatch detection ──
    if warnings is not None and len(ref_cjk) > 0:
        n_cjk = len(ref_cjk)
        if pinyin_count > n_cjk:
            warnings.append(
                f"hanzi tier mismatch: {pinyin_count} pinyin tokens vs "
                f"{n_cjk} reference CJK chars — "
                f"{pinyin_count - n_cjk} pinyin token(s) fell back "
                f"(no more CJK chars to consume)"
            )
        elif pinyin_count < n_cjk:
            warnings.append(
                f"hanzi tier mismatch: {pinyin_count} pinyin tokens vs "
                f"{n_cjk} reference CJK chars — "
                f"{n_cjk - pinyin_count} reference CJK char(s) were not "
                f"assigned to any pinyin token"
            )

    return Tier("hanzi", words_tier.xmin, words_tier.xmax, intervals)


def assess_reference_coverage(
    reference_text: str,
    words_tier: Tier | None,
    hanzi_tier: Tier | None,
    *,
    reference_source: str,
    unknown_source_count: int = 0,
    reference_authoritative: bool | None = None,
) -> tuple[dict, list[str]]:
    """Assess hard lexical integrity independently of optional acoustic QC.

    ASR/lab fallback text is a source for rebuilding derived tiers, not an
    original/reference transcript.  Keep the source CJK sequence available
    for the independent fallback structural check, but do not apply the
    reference semantic contract to it.  ``reference_authoritative`` is
    passed by the production branch; the source-name inference keeps direct
    helper callers and older focused tests backwards compatible.
    """
    reference_cjk = "".join(ch for ch in reference_text if is_cjk(ch))
    word_intervals = words_tier.intervals if words_tier is not None else []
    pinyin_tokens = [
        iv.text.strip() for iv in word_intervals
        if is_pinyin_syllable(iv.text.strip())
    ]
    hanzi_intervals = hanzi_tier.intervals if hanzi_tier is not None else []
    hanzi_cjk = "".join(
        label for iv in hanzi_intervals
        if len((label := iv.text.strip())) == 1 and is_cjk(label)
    )
    if reference_authoritative is None:
        reference_authoritative = reference_source not in {
            "asr_fallback", "lab_fallback"
        }
    if not reference_authoritative:
        reasons = ["mfa_unknown_source"] if unknown_source_count else []
        return {
            "reference_source": reference_source,
            "reference_validation_applied": False,
            "reference_cjk_count": None,
            "pinyin_token_count": len(pinyin_tokens),
            "assigned_cjk_count": len(hanzi_cjk),
            "missing_cjk_count": None,
            "extra_pinyin_count": None,
            "unknown_source_count": unknown_source_count,
            "exact_cjk_sequence": None,
            "exact_semantic_sequence": None,
            "source_cjk": reference_cjk,
            "reference_cjk": "",
            "hanzi_cjk": hanzi_cjk,
        }, reasons

    reference_semantic = _strict_semantic_tokens(reference_text)
    hanzi_semantic = _strict_semantic_tokens(" ".join(
        iv.text for iv in hanzi_intervals if iv.text).strip())

    def _semantic_compatible_with_missing_punctuation(
            reference: list[tuple[str, str]],
            observed: list[tuple[str, str]]) -> bool:
        """Allow reference punctuation to disappear, but not mutate/order/add.

        Lexical and NVV tokens remain an exact ordered contract.  While
        walking the observed sequence, only unmatched reference punctuation
        may be skipped.  Consequently a missing comma is tolerated, whereas
        a period in its place, an extra mark, or a reordered mark fails.
        """
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

    semantic_compatible = _semantic_compatible_with_missing_punctuation(
        reference_semantic, hanzi_semantic)

    lexical_reference = re.sub(r"<sp\d+>", "", reference_text)
    has_lexical_reference = bool(
        reference_cjk
        or re.search(r"[A-Za-z]", lexical_reference)
        or any(is_nvv_token(token) for token in extract_word_chars(lexical_reference))
    )

    reasons: list[str] = []
    if not reference_text.strip():
        reasons.append("empty_reference")
    elif not has_lexical_reference:
        reasons.append("no_lexical_reference")

    if reference_cjk:
        if not pinyin_tokens:
            reasons.append("cjk_alignment_collapse")
        if len(reference_cjk) != len(pinyin_tokens):
            reasons.append("cjk_token_count_mismatch")
    elif pinyin_tokens:
        reasons.append("unexpected_pinyin_without_cjk")

    if reference_cjk != hanzi_cjk:
        reasons.append("cjk_mismatch")
    # CJK-only equality cannot prove that interleaved English/NVV units kept
    # their reference position (for example ``第N天，Noa``).  The strict disk
    # auditor remains the publication authority; this is an earlier,
    # deterministic postprocess veto for the same lexical corruption.
    if not semantic_compatible:
        reasons.append("reference_semantic_sequence_mismatch")
    if unknown_source_count:
        reasons.append("mfa_unknown_source")

    coverage = {
        "reference_source": reference_source,
        "reference_validation_applied": True,
        "reference_cjk_count": len(reference_cjk),
        "pinyin_token_count": len(pinyin_tokens),
        "assigned_cjk_count": len(hanzi_cjk),
        "missing_cjk_count": max(0, len(reference_cjk) - len(pinyin_tokens)),
        "extra_pinyin_count": max(0, len(pinyin_tokens) - len(reference_cjk)),
        "unknown_source_count": unknown_source_count,
        "exact_cjk_sequence": reference_cjk == hanzi_cjk,
        # The publication contract treats missing punctuation as a tolerated
        # projection loss.  Keep the historical field name for consumers,
        # but report the contract-compatible result rather than re-rejecting a
        # candidate solely because a reference punctuation mark disappeared.
        "exact_semantic_sequence": semantic_compatible,
        "semantic_sequence_missing_punctuation": [
            token for token in reference_semantic
            if token[0] == "punct" and token not in hanzi_semantic
        ],
        "reference_cjk": reference_cjk,
        "hanzi_cjk": hanzi_cjk,
    }
    return coverage, list(dict.fromkeys(reasons))


def _normalize_word_spellings(words_tier: Tier, raw_text: str,
                              *, authority_strict: bool = False) -> None:
    """Replace tokenizer-damaged English words with canonical reference spellings.

    Uses Needleman-Wunsch alignment (:func:`_align_word_sequences`) to
    map word-tier tokens to reference word units.  When a token is a
    fragment of an English word (e.g. "Cla" for "Claude"), the word-tier
    text is updated in-place to match the reference spelling so that all
    downstream tiers (words, pinyin_phones, hanzi) stay consistent.

    Three passes:
      1. Replace matched English tokens that differ from reference spelling.
      2. Merge orphan ASCII-alpha fragments (tokenizer remnants) into adjacent
         corrected English words by extending time ranges.
      3. For unmatched reference English words, find orphan ASCII-alpha tokens
         in the approximate region and replace them.

    Regression Case 62: NVASR tokenizer breaks English words into letter
    fragments (e.g. "Claude" → "Cla"+"ude").  normalize_english_tokens.py
    may fail to merge them when _text_cn.txt (ASR output) differs from the
    original reference .txt.  This function uses the original reference text
    (raw_text) as ground truth to correct all surviving errors.
    """
    clean = raw_text.replace('<sp1>', '')
    char_units = _extract_word_chars(clean)

    # Reference word units (punct filtered)
    ref_units: list[tuple[int, str]] = []
    for i, u in enumerate(char_units):
        if is_word_like(u):
            ref_units.append((i, u))

    # ── English reference positions (auto-detect from raw_text) ──
    # ASCII-alpha, len >= 2, excluding NVV tokens (which have no acoustic
    # model and must keep their canonical <>-wrapped form).
    en_ref_positions: dict[int, str] = {}   # ref_units index → english word
    for ri, (ci, u) in enumerate(ref_units):
        if (u.isascii() and (u.isalpha() or is_english_token(u))
                and len(re.sub(r"[^a-z0-9]", "", u)) >= 2
                and not is_nvv_token(u)):
            en_ref_positions[ri] = u

    # Word-tier tokens (silence & punct filtered)
    word_entries: list[tuple[int, str]] = []
    for i, iv in enumerate(words_tier.intervals):
        if is_silence(iv.text) or not iv.text.strip():
            continue
        if is_punct(iv.text):
            continue
        word_entries.append((i, iv.text.strip()))

    if not word_entries or not ref_units:
        return

    # Authority mode is transactional: do not rename one interval and leave
    # the remaining CTC fragments for a later heuristic pass.  The caller
    # performs one exact compound reconciliation against the immutable source
    # ledger before committing the final words owner.
    if authority_strict:
        return

    # Align
    ctc_texts = [t for _, t in word_entries]
    ref_texts = [u for _, u in ref_units]
    alignment = _align_word_sequences(ctc_texts, ref_texts)

    # Build lookup: ctc_i → ref_i and ref_i → [ctc_i...]
    ctc_to_ref: dict[int, int] = {}
    ref_to_ctc: dict[int, list[int]] = {}
    for ctc_i, ref_i in alignment:
        if ctc_i is not None and ref_i is not None:
            ctc_to_ref[ctc_i] = ref_i
            ref_to_ctc.setdefault(ref_i, []).append(ctc_i)

    # ── Pass 1: Replace matched English tokens with canonical spelling ──
    # For every matched pair where the reference is an English word and the
    # word-tier text differs, overwrite it with the reference spelling.
    # NVV tokens are NEVER replaced (Regression Case 17).
    fixed_ctc_indices: set[int] = set()   # word_entries indices fixed in Pass 1
    for ctc_i, ref_i in alignment:
        if ctc_i is None or ref_i is None:
            continue
        if ref_i not in en_ref_positions:
            continue
        ref_spelling = en_ref_positions[ref_i]
        wi, w_text = word_entries[ctc_i]
        if is_nvv_token(w_text):
            continue
        if ref_spelling != w_text:
            words_tier.intervals[wi].text = ref_spelling
            fixed_ctc_indices.add(ctc_i)

    if not en_ref_positions:
        return

    # ── Pass 2: Merge orphan ASCII-alpha fragments into corrected words ──
    # After Pass 1, unmatched CTC tokens (ref_i=None) that are ASCII-alpha
    # (e.g. tokenizer remnants like "ude" after "Cla"→"Claude") are merged
    # into the nearest corrected English word by extending its time range.
    # Safety: only ASCII-alpha (no digits, no CJK) — pinyin syllables like
    # "rui4" and CJK tokens like "的" are protected.
    merged_ctc: set[int] = set()
    for ctc_i, ref_i in alignment:
        if ref_i is not None:
            continue          # already matched — skip
        if ctc_i is None:
            continue
        wi, w_text = word_entries[ctc_i]
        if not (w_text.isascii() and w_text.isalpha()):
            continue          # not an English fragment (pinyin / CJK / NVV)
        if is_nvv_token(w_text):
            continue

        # Merge into the nearest fixed English word (look left, then right)
        merged = False
        # ── Left search: walk backward through alignment to find fixed neighbour ──
        for left_i in range(ctc_i - 1, -1, -1):
            if left_i in fixed_ctc_indices:
                left_wi = word_entries[left_i][0]
                left_iv = words_tier.intervals[left_wi]
                cur_iv = words_tier.intervals[wi]
                words_tier.intervals[left_wi] = Interval(
                    left_iv.xmin, max(left_iv.xmax, cur_iv.xmax), left_iv.text)
                # Zero out the merged fragment (cleaned up below)
                words_tier.intervals[wi] = Interval(cur_iv.xmin, cur_iv.xmin, "")
                merged_ctc.add(ctc_i)
                merged = True
                break
            # Only skip over other English fragments; stop at CJK/pinyin/NVV
            left_text = word_entries[left_i][1]
            if not (left_text.isascii() and left_text.isalpha() and not is_nvv_token(left_text)):
                break
        if merged:
            continue

        # ── Right search ──
        for right_i in range(ctc_i + 1, len(word_entries)):
            if right_i in fixed_ctc_indices:
                right_wi = word_entries[right_i][0]
                right_iv = words_tier.intervals[right_wi]
                cur_iv = words_tier.intervals[wi]
                words_tier.intervals[right_wi] = Interval(
                    min(right_iv.xmin, cur_iv.xmin), right_iv.xmax, right_iv.text)
                words_tier.intervals[wi] = Interval(cur_iv.xmin, cur_iv.xmin, "")
                merged_ctc.add(ctc_i)
                merged = True
                break
            right_text = word_entries[right_i][1]
            if not (right_text.isascii() and right_text.isalpha() and not is_nvv_token(right_text)):
                break

    # ── Pass 3: Unmatched reference English words → replace orphan CTC tokens ──
    # A reference English word may have no matched CTC token (e.g. when the
    # word-tier token is a wrong merge like "Cudude" that NW can't match to
    # "Claude").  For each unmatched English reference word, scan for orphan
    # ASCII-alpha CTC tokens in the approximate region and replace the first
    # one with the reference spelling.  Region is bounded by the neighbouring
    # matched CJK anchors on either side.
    for ref_i, en_word in en_ref_positions.items():
        if ref_i in ref_to_ctc:
            continue  # already matched — handled in Pass 1/2

        # Find left/right CTC boundaries from matched neighbouring ref units
        left_ctc_bound = 0
        for lr in range(ref_i - 1, -1, -1):
            if lr in ref_to_ctc:
                left_ctc_bound = max(ref_to_ctc[lr]) + 1
                break
        right_ctc_bound = len(word_entries)
        for rr in range(ref_i + 1, len(ref_units)):
            if rr in ref_to_ctc:
                right_ctc_bound = min(ref_to_ctc[rr])
                break

        # Scan for orphan ASCII-alpha tokens in [left_ctc_bound, right_ctc_bound)
        orphan_candidates: list[int] = []
        for ctc_i in range(left_ctc_bound, min(right_ctc_bound, len(word_entries))):
            if ctc_i in ctc_to_ref or ctc_i in merged_ctc:
                continue  # already matched or merged
            wi, w_text = word_entries[ctc_i]
            if w_text.isascii() and w_text.isalpha() and not is_nvv_token(w_text):
                orphan_candidates.append(ctc_i)

        if orphan_candidates:
            # Replace the first orphan with the reference word
            first_orphan = orphan_candidates[0]
            wi = word_entries[first_orphan][0]
            words_tier.intervals[wi].text = en_word
            fixed_ctc_indices.add(first_orphan)
            # Merge remaining orphans into this word
            for other in orphan_candidates[1:]:
                other_wi = word_entries[other][0]
                words_tier.intervals[wi] = Interval(
                    words_tier.intervals[wi].xmin,
                    max(words_tier.intervals[wi].xmax, words_tier.intervals[other_wi].xmax),
                    en_word)
                words_tier.intervals[other_wi] = Interval(
                    words_tier.intervals[other_wi].xmin,
                    words_tier.intervals[other_wi].xmin, "")

    # ── Clean up zero-duration placeholders ──
    words_tier.intervals = [iv for iv in words_tier.intervals
                           if iv.xmax - iv.xmin > 0.001]


def _merge_authority_alpha_digit_fragments(
        words_tier: Tier, reference_text: str,
        ctc_tokens: list[dict] | None = None,
        report: dict | None = None) -> list[str]:
    """Commit one exact authority compound projection.

    The historical name is retained for callers, but the transaction handles
    every reference English unit, not only alpha+digit spellings.  CTC rows
    are consumed in file order when the real ``tokens.jsonl`` has no ordinal;
    an explicitly malformed ordinal still fails closed.  All candidate groups
    are validated before the words tier is replaced, so a partial repair can
    never hide a CJK/NVV/punctuation boundary or a missing fragment.
    """
    try:
        units = parse_english_units(reference_text)
    except (EnglishUnitError, TypeError, ValueError):
        return []
    if not isinstance(ctc_tokens, list) or not ctc_tokens:
        return []
    # Authority numeral normalization turns ``target1`` into the semantic
    # stream ``target`` + ``一``.  Keep the ASCII suffix out of the English
    # fragment transaction, then bind it to the corresponding CJK owner.
    # Direct callers using the historical ``target1`` reference retain the
    # old alpha+digit transaction below.
    semantic = list(project_authority_semantics(reference_text))
    numeral_surfaces = [item["surface"] for item in semantic
                        if item.get("kind") == "cjk"
                        and item.get("surface") in {"一", "二"}]
    numeral_digit_to_surface = {"1": "一", "2": "二"}
    authority_has_cjk_numerals = bool(numeral_surfaces)

    ctc_rows: list[dict] = []
    all_rows: list[dict] = []
    for index, token in enumerate(ctc_tokens):
        if not isinstance(token, dict) or str(token.get("type", "word")) != "word":
            continue
        text = str(token.get("word", "")).strip().lstrip("▁")
        try:
            start = float(token["start_s"])
            end = float(token["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_ordinals = token.get("source_ctc_ordinals")
        explicit = raw_ordinals is not None or any(
            key in token for key in ("source_ctc_ordinal", "ordinal"))
        if raw_ordinals is None:
            raw_ordinal = token.get("source_ctc_ordinal", token.get("ordinal"))
            raw_ordinals = [raw_ordinal] if raw_ordinal is not None else None
        if explicit and (not isinstance(raw_ordinals, (list, tuple))
                         or len(raw_ordinals) != 1
                         or not isinstance(raw_ordinals[0], int)
                         or isinstance(raw_ordinals[0], bool)
                         or raw_ordinals[0] < 0):
            if report is not None:
                report.setdefault("authority_compound_reconciliation", {})[
                    "status"] = "rejected"
                report["authority_compound_reconciliation"]["reason"] = (
                    "invalid_source_ctc_ordinal")
            return []
        if (not text or not math.isfinite(start) or not math.isfinite(end)
                or end <= start):
            continue
        row = {"index": index, "text": text, "start": start, "end": end,
               "ordinal": (raw_ordinals[0] if raw_ordinals is not None
                           else index),
               "ordinal_source": "explicit" if explicit else "file_order"}
        all_rows.append(row)
        if (is_english_fragment_token(text)
                and not (authority_has_cjk_numerals
                         and text in numeral_digit_to_surface)):
            ctc_rows.append(row)
    if not ctc_rows:
        if report is not None:
            report["authority_compound_reconciliation"] = {
                "status": "rejected", "reason": "ctc_evidence_missing"}
        return []

    planned: list[dict] = []
    cursor = 0
    for unit in units:
        matched = None
        for end in range(cursor + 1, len(ctc_rows) + 1):
            try:
                merged = merge_authority_fragment_group(unit, ctc_rows[cursor:end])
            except EnglishUnitError:
                continue
            matched = (end, merged)
            break
        if matched is None:
            if report is not None:
                report.setdefault("authority_compound_reconciliation", {})[
                    "status"] = "rejected"
                report["authority_compound_reconciliation"]["reason"] = (
                    "fragment_group_not_exact")
            return []
        end, merged = matched
        group = ctc_rows[cursor:end]
        source_window = [row for row in all_rows
                         if group[0]["index"] <= row["index"] <= group[-1]["index"]]
        if len(source_window) != len(group):
            if report is not None:
                report["authority_compound_reconciliation"] = {
                    "status": "rejected", "reason": "cross_non_english_boundary",
                    "unit_id": unit.unit_id,
                }
            return []
        source_start, source_end = group[0]["start"], group[-1]["end"]
        if any(right["start"] - left["end"] > AXIS_EPS
               for left, right in zip(group, group[1:])):
            if report is not None:
                report["authority_compound_reconciliation"] = {
                    "status": "rejected", "reason": "source_span_gap",
                    "unit_id": unit.unit_id}
            return []
        owner_indices: set[int] = set()
        for row in group:
            row_compact = re.sub(r"[^a-z0-9]", "", row["text"].casefold())
            candidates = [index for index, interval in enumerate(words_tier.intervals)
                          if (is_english_fragment_token(interval.text.strip())
                              and interval.xmax > row["start"] - AXIS_EPS
                              and interval.xmin < row["end"] + AXIS_EPS
                              and (lambda compact: compact == row_compact
                                   or compact.startswith(row_compact)
                                   or row_compact.startswith(compact))(
                                       re.sub(r"[^a-z0-9]", "", interval.text.strip().casefold()))) ]
            if len(candidates) != 1:
                # A pre-merged owner is acceptable only if it covers the
                # complete source group and has the exact authority surface.
                candidates = [index for index, interval in enumerate(words_tier.intervals)
                              if (interval.xmin <= source_start + AXIS_EPS
                                  and interval.xmax >= source_end - AXIS_EPS
                                  and re.sub(r"[^a-z0-9]", "", interval.text.strip().casefold())
                                     == re.sub(r"[^a-z0-9]", "", unit.surface_text.casefold()))]
            if len(candidates) != 1:
                if report is not None:
                    report.setdefault("authority_compound_reconciliation", {})[
                        "status"] = "rejected"
                    report["authority_compound_reconciliation"]["reason"] = (
                        "word_owner_not_exact")
                return []
            owner_indices.add(candidates[0])
        first_index, last_index = min(owner_indices), max(owner_indices)
        for index in range(first_index, last_index + 1):
            text = words_tier.intervals[index].text.strip()
            if (not is_silence(text) and not is_english_fragment_token(text)):
                if report is not None:
                    report.setdefault("authority_compound_reconciliation", {})[
                        "status"] = "rejected"
                    report["authority_compound_reconciliation"]["reason"] = (
                        "cross_non_english_boundary")
                return []
        planned.append({"unit": unit, "merged": merged, "first": first_index,
                        "last": last_index, "indices": owner_indices,
                        "source_rows": group, "source_span": [source_start, source_end]})
        cursor = end
    if cursor != len(ctc_rows):
        if report is not None:
            report["authority_compound_reconciliation"] = {
                "status": "rejected", "reason": "extra_ctc_fragment"}
        return []

    numeral_plans: list[dict] = []
    if authority_has_cjk_numerals:
        numeral_rows = [row for row in all_rows
                        if row["text"] in numeral_digit_to_surface]
        if numeral_rows:
            expected = iter(numeral_surfaces)
            for row in numeral_rows:
                surface = numeral_digit_to_surface[row["text"]]
                try:
                    expected_surface = next(expected)
                except StopIteration:
                    expected_surface = None
                if expected_surface != surface:
                    if report is not None:
                        report["authority_compound_reconciliation"] = {
                            "status": "rejected",
                            "reason": "numeral_fragment_order_or_surface_mismatch",
                            "source_ctc_index": row["index"],
                        }
                    return []
                candidates = [index for index, interval in enumerate(words_tier.intervals)
                              if (interval.xmax > row["start"] - AXIS_EPS
                                  and interval.xmin < row["end"] + AXIS_EPS
                                  and (interval.text.strip() == row["text"]
                                       or interval.text.strip()
                                          == _AUTHORITY_NUMERAL_PINYIN.get(surface)))]
                if len(candidates) != 1:
                    if report is not None:
                        report["authority_compound_reconciliation"] = {
                            "status": "rejected",
                            "reason": "numeral_fragment_owner_not_exact",
                            "source_ctc_index": row["index"],
                        }
                    return []
                owner = words_tier.intervals[candidates[0]]
                if owner.text.strip() == row["text"]:
                    words_tier.intervals[candidates[0]] = Interval(
                        owner.xmin, owner.xmax,
                        _AUTHORITY_NUMERAL_PINYIN[surface])
                numeral_plans.append({
                    "surface": surface,
                    "pinyin": _AUTHORITY_NUMERAL_PINYIN[surface],
                    "source_row": row,
                    "owner_index": candidates[0],
                    "geometry_operation": "authority_numeral_fragment_projection",
                })
            try:
                next(expected)
            except StopIteration:
                pass
            else:
                if report is not None:
                    report["authority_compound_reconciliation"] = {
                        "status": "rejected",
                        "reason": "missing_numeral_fragment",
                    }
                return []

    replacements: dict[int, dict] = {}
    removed: set[int] = set()
    for item in planned:
        first, last = item["first"], item["last"]
        if any(index in replacements or index in removed
               for index in range(first, last + 1)):
            return []
        owner = words_tier.intervals[first]
        replacements[first] = {
            "interval": Interval(owner.xmin, words_tier.intervals[last].xmax,
                                  item["unit"].surface_text),
            "unit": item["unit"],
        }
        removed.update(range(first + 1, last + 1))
    changed_unit_ids = [item["unit"].unit_id for item in planned
                        if len(item["source_rows"]) > 1
                        or item["first"] != item["last"]
                        or words_tier.intervals[item["first"]].text.strip()
                           != item["unit"].surface_text]
    if planned:
        words_tier.intervals = [replacements[index]["interval"]
                                if index in replacements else interval
                                for index, interval in enumerate(words_tier.intervals)
                                if index not in removed]
        words_tier._canonical_authority_units = [
            {"unit_id": item["unit"].unit_id,
             "surface": item["unit"].surface_text,
             "alignment_token": item["unit"].alignment_token,
             "source_ctc_indices": [row["index"] for row in item["source_rows"]],
             "source_ctc_ordinals": list(item["merged"].source_ctc_ordinals),
             "source_spans": [[row["start"], row["end"]]
                              for row in item["source_rows"]],
             "geometry_operation": "authority_compound_reconciliation"}
            for item in planned]
    elif not hasattr(words_tier, "_canonical_authority_units"):
        words_tier._canonical_authority_units = []
    if numeral_plans:
        words_tier._canonical_authority_units.extend({
            "unit_id": None,
            "surface": item["surface"],
            "alignment_token": item["pinyin"],
            "source_ctc_indices": [item["source_row"]["index"]],
            "source_ctc_ordinals": [item["source_row"]["ordinal"]],
            "source_spans": [[item["source_row"]["start"],
                              item["source_row"]["end"]]],
            "geometry_operation": item["geometry_operation"],
        } for item in numeral_plans)
    restored = changed_unit_ids
    if report is not None:
        report["authority_compound_reconciliation"] = {
            "status": "accepted" if planned else "not_required",
            "units": [entry for entry in getattr(words_tier, "_canonical_authority_units", [])],
        }
        if numeral_plans:
            report["authority_compound_reconciliation"]["numeral_fragments"] = [
                {"surface": item["surface"], "pinyin": item["pinyin"],
                 "source_ctc_index": item["source_row"]["index"],
                 "source_span": [item["source_row"]["start"],
                                 item["source_row"]["end"]],
                 "geometry_operation": item["geometry_operation"]}
                for item in numeral_plans]
    return restored


def _restore_reference_surfaces(words_tier: Tier | None,
                                hanzi_tier: Tier | None,
                                reference_text: str) -> list[str]:
    """Restore canonical reference spelling on the two user-visible tiers.

    ``english_units`` owns the hyphenated surface, while CTC/MFA and the
    internal processing string use its hyphenless alignment token.  Surface
    restoration is therefore deliberately late and limited to ``words`` and
    ``hanzi``; raw/pinyin tiers remain the established processing projection.
    A surface is changed only for a single ordered owner whose compact text
    is exactly the canonical unit token.
    """
    if words_tier is None or hanzi_tier is None or not reference_text:
        return []
    try:
        units = parse_english_units(reference_text)
    except (EnglishUnitError, TypeError, ValueError):
        return []
    owners = [iv for iv in words_tier.intervals if is_english_token(iv.text.strip())]
    if len(owners) != len(units):
        return []
    restored: list[str] = []
    for unit, owner in zip(units, owners):
        compact = re.sub(r"[^a-z0-9]", "", owner.text.strip().casefold())
        if compact != unit.alignment_token:
            continue
        owner.text = unit.surface_text
        matching = [iv for iv in hanzi_tier.intervals
                    if is_english_token(iv.text.strip())
                    and abs(iv.xmin - owner.xmin) <= AXIS_EPS
                    and abs(iv.xmax - owner.xmax) <= AXIS_EPS
                    and re.sub(r"[^a-z0-9]", "", iv.text.strip().casefold()) == compact]
        if len(matching) == 1:
            matching[0].text = unit.surface_text
        restored.append(unit.unit_id)
    return restored


# ---------------------------------------------------------------------------
# Audio I/O (NumPy-based — shared with audio_energy.py)
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> tuple["np.ndarray", int]:
    """Load WAV as float32 mono numpy array.  Returns (audio, sample_rate)."""
    import numpy as _np
    import soundfile as _sf
    data, sr = _sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0].copy()
    return _np.ascontiguousarray(data, dtype=_np.float32), int(sr)


# ---------------------------------------------------------------------------
# Energy helpers (NumPy vectorised)
# ---------------------------------------------------------------------------

def _frame_rms_vec(audio, sr: int, frame_ms: float = 5.0
                   ) -> tuple["np.ndarray", float]:
    """RMS per frame (vectorised).  Returns (rms, frame_dur_s)."""
    import numpy as _np
    fs = max(1, int(frame_ms / 1000.0 * sr))
    n_frames = max(0, (len(audio) - fs) // fs + 1)
    if n_frames == 0 or n_frames * fs > len(audio):
        return _np.array([], dtype=_np.float32), 0.0
    frames = audio[:n_frames * fs].reshape(n_frames, fs)
    rms = _np.sqrt(_np.mean(frames.astype(_np.float64) ** 2, axis=1) + 1e-12)
    return rms.astype(_np.float32), fs / sr


def _rms_frames_in_span(audio, sr: int, xmin: float, xmax: float,
                        frame_ms: float = 10.0) -> tuple["np.ndarray", list[tuple[float, float]]]:
    """Return complete, globally aligned RMS frames in ``[xmin, xmax)``.

    Word-energy evidence is deliberately based on the same 10 ms RMS
    primitive everywhere.  Partial frames are excluded: treating a short
    boundary fragment as a complete acoustic observation was the source of
    the old mean-absolute/RMS unit mismatch.
    """
    import numpy as _np
    if audio is None or sr <= 0 or xmax <= xmin:
        return _np.array([], dtype=_np.float32), []
    frame_samples = max(1, int(round(frame_ms / 1000.0 * sr)))
    start_sample = max(0, int(math.ceil(float(xmin) * sr - 1e-9)))
    end_sample = min(len(audio), int(math.floor(float(xmax) * sr + 1e-9)))
    first = ((start_sample + frame_samples - 1) // frame_samples) * frame_samples
    last = end_sample - frame_samples
    if last < first:
        return _np.array([], dtype=_np.float32), []
    starts = list(range(first, last + 1, frame_samples))
    frame_array = _np.asarray(
        [audio[start:start + frame_samples] for start in starts],
        dtype=_np.float64)
    if frame_array.size == 0:
        return _np.array([], dtype=_np.float32), []
    values = _np.sqrt(_np.mean(frame_array ** 2, axis=1))
    spans = [(start / sr, (start + frame_samples) / sr)
             for start in starts]
    return values.astype(_np.float32), spans


def _word_rms(audio, sr: int, xmin: float, xmax: float) -> float:
    """Mean 10 ms frame RMS in time slice ``[xmin, xmax)``."""
    values, _ = _rms_frames_in_span(audio, sr, xmin, xmax, frame_ms=10.0)
    return float(np.mean(values)) if len(values) else 0.0


def _word_energy_enabled(args) -> bool:
    """Resolve the tri-state detector switch without strict-mode side effects."""
    explicit = getattr(args, "enable_word_in_silence_filter", None)
    if explicit is False:
        return False
    ratio = float(getattr(args, "filter_word_energy_ratio", 0.0) or 0.0)
    return bool(explicit is True or (explicit is None and ratio > 0.0))


def _word_energy_noise_model(audio, sr: int, words_tier: Tier | None,
                             args) -> dict:
    """Build the single noise model used by both energy-entry points."""
    ratio = float(getattr(args, "filter_word_energy_ratio", 0.0) or 0.0)
    silence_values: list[float] = []
    if words_tier is not None and audio is not None:
        for interval in words_tier.intervals:
            if not is_silence(interval.text):
                continue
            values, _ = _rms_frames_in_span(
                audio, sr, interval.xmin, interval.xmax, frame_ms=10.0)
            silence_values.extend(float(value) for value in values)
    if silence_values:
        values = np.asarray(silence_values, dtype=np.float32)
        source = "explicit_silence_owner"
    elif audio is not None and sr > 0:
        values, _ = _frame_rms_vec(audio, sr, frame_ms=10.0)
        source = "global_audio_percentile_15"
    else:
        values = np.asarray([], dtype=np.float32)
        source = "unavailable"
    noise_floor = float(np.percentile(values, 15)) if len(values) else 0.0
    return {
        "source": source,
        "frame_count": int(len(values)),
        "frame_ms": 10.0,
        "percentile": 15 if source == "global_audio_percentile_15" else None,
        "noise_floor": noise_floor,
        "ratio": ratio,
        "threshold": noise_floor * ratio,
    }


def _nearest_lexical_neighbors(words_tier: Tier | None) -> dict[int, dict]:
    """Map each word index to nearest lexical owners, crossing silence only."""
    if words_tier is None:
        return {}
    result: dict[int, dict] = {}
    for index, interval in enumerate(words_tier.intervals):
        if not interval.text.strip() or is_silence(interval.text):
            continue
        prev = index - 1
        while prev >= 0 and is_silence(words_tier.intervals[prev].text):
            prev -= 1
        next_index = index + 1
        while (next_index < len(words_tier.intervals)
               and is_silence(words_tier.intervals[next_index].text)):
            next_index += 1
        prev_iv = words_tier.intervals[prev] if prev >= 0 else None
        next_iv = (words_tier.intervals[next_index]
                   if next_index < len(words_tier.intervals) else None)
        result[index] = {
            "previous": (prev_iv.text.strip() if prev_iv is not None else None),
            "next": (next_iv.text.strip() if next_iv is not None else None),
            "previous_index": prev,
            "next_index": next_index if next_iv is not None else None,
            "crossed_silence_previous": prev >= 0 and prev < index - 1,
            "crossed_silence_next": next_iv is not None and next_index > index + 1,
        }
    return result


def _word_energy_audit(words_tier: Tier | None, args, audio=None, sr: int = 16000,
                       *, textgrid: TextGrid | None = None,
                       ctc_tokens: list[dict] | None = None) -> dict:
    """Audit lexical owners using immutable source/merge evidence.

    This is intentionally a pure report builder.  It never changes words or
    derived tiers and never treats rebuilt phones as fresh acoustic evidence.
    """
    model = _word_energy_noise_model(audio, sr, words_tier, args)
    threshold = float(model["threshold"])
    premerge = getattr(words_tier, "_word_energy_premerge_spans", {}) if words_tier else {}
    merge_ledger = getattr(words_tier, "_word_energy_merge_ledger", []) if words_tier else []
    lineage = getattr(textgrid, "_phone_lineage", None) if textgrid is not None else None
    lineage_invalid = getattr(textgrid, "_phone_lineage_invalid", None) if textgrid is not None else None
    ctc_entries = _ctc_authority_entries(words_tier) or []
    neighbours = _nearest_lexical_neighbors(words_tier)
    lexical_rows = []
    if words_tier is not None:
        for index, interval in enumerate(words_tier.intervals):
            label = interval.text.strip()
            if label and not is_silence(label) and not is_punct(label):
                lexical_rows.append((index, len(lexical_rows), interval))
    items: list[dict] = []

    def _span(value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _valid_span(value):
        span = _span(value)
        if (span is None or not all(math.isfinite(point) for point in span)
                or span[1] <= span[0] + AXIS_EPS):
            return None
        return span

    for index, ordinal, interval in lexical_rows:
        final_span = [float(interval.xmin), float(interval.xmax)]
        old_span = _span(premerge.get(str(ordinal), premerge.get(ordinal)))
        if old_span is None:
            old_span = list(final_span)
        ctc_span = None
        if ordinal < len(ctc_entries) and isinstance(ctc_entries[ordinal], dict):
            entry = ctc_entries[ordinal]
            # A malformed explicit ctc_span must not be upgraded by a stale
            # resolved_span.  The bounded correction below is enabled only by
            # an actually valid CTC boundary anchor.
            if entry.get("ctc_span") is not None:
                ctc_span = _valid_span(entry.get("ctc_span"))
            else:
                ctc_span = _valid_span(entry.get("resolved_span"))
        if ctc_span is None and ctc_tokens:
            # Keep ordinal binding positional.  This fallback is deliberately
            # not word-string matching, so repeated tokens cannot steal one
            # another's evidence.
            lexical_ctc = [token for token in ctc_tokens
                            if isinstance(token, dict)
                            and str(token.get("word", token.get("text", ""))).strip()
                            and not is_silence(str(token.get("word", token.get("text", ""))))
                            and not is_punct(str(token.get("word", token.get("text", ""))))]
            if ordinal < len(lexical_ctc):
                token = lexical_ctc[ordinal]
                ctc_span = _span([token.get("start_s", token.get("start")),
                                  token.get("end_s", token.get("end"))])
        source_phone_spans: list[list[float]] = []
        lineage_reason = None
        if not isinstance(lineage, dict):
            lineage_reason = "phone_lineage_missing"
        else:
            if lineage_invalid or lineage.get("status") == "rejected":
                lineage_reason = "phone_lineage_ambiguous"
            elif lineage.get("status") == "partial":
                for reason in lineage.get("reasons", []) or []:
                    if (isinstance(reason, dict)
                            and not is_silence(str(reason.get("label", "")))):
                        lineage_reason = "phone_lineage_ambiguous"
                        break
            for rows in (lineage.get("owners", {}) or {}).values():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if isinstance(row, dict) and row.get("lexical_ordinal") == ordinal:
                        span = _span([row.get("start"), row.get("end")])
                        if span is not None:
                            source_phone_spans.append(span)
            if not source_phone_spans and lineage_reason is None:
                lineage_reason = "phone_lineage_missing_owner"
        operation = None
        merge_policy = None
        for merge in merge_ledger if isinstance(merge_ledger, list) else []:
            if not isinstance(merge, dict):
                continue
            if ordinal in {merge.get("left_lexical_ordinal"),
                           merge.get("right_lexical_ordinal"),
                           merge.get("lexical_ordinal")}:
                operation, merge_policy = _merge_operation_metadata(
                    merge.get("operation"), merge.get("policy"))
                if operation is not None:
                    break
        core_values, core_frames = _rms_frames_in_span(
            audio, sr, old_span[0], old_span[1], frame_ms=10.0)
        final_values, final_frames = _rms_frames_in_span(
            audio, sr, final_span[0], final_span[1], frame_ms=10.0)
        active_floor = max(threshold, 1e-12)
        active = core_values > active_floor if len(core_values) else np.asarray([], dtype=bool)
        max_run = run = 0
        for flag in active:
            run = run + 1 if bool(flag) else 0
            max_run = max(max_run, run)
        adjacency = neighbours.get(index, {})
        english_nvv_adjacency = {
            "previous": adjacency.get("previous") if (
                adjacency.get("previous") and (
                    is_english_token(adjacency["previous"])
                    or is_nvv_token(adjacency["previous"]))) else None,
            "next": adjacency.get("next") if (
                adjacency.get("next") and (
                    is_english_token(adjacency["next"])
                    or is_nvv_token(adjacency["next"]))) else None,
            "crossed_silence": bool(adjacency.get("crossed_silence_previous")
                                     or adjacency.get("crossed_silence_next")),
        }
        item = {
            "lexical_ordinal": ordinal,
            "word": interval.text.strip(),
            "final_span": final_span,
            "premerge_span": old_span,
            "source_phone_spans": source_phone_spans,
            "ctc_span": ctc_span,
            "merge_operation": operation,
            "merge_policy": merge_policy,
            "english_nvv_adjacency": english_nvv_adjacency,
            "lineage": {"status": lineage.get("status") if isinstance(lineage, dict) else None,
                        "reason": lineage_reason,
                        "source_phone_count": len(source_phone_spans)},
            "rms": {
                "frame_ms": 10.0,
                "premerge_frame_count": int(len(core_values)),
                "final_frame_count": int(len(final_values)),
                "premerge_rms": float(np.mean(core_values)) if len(core_values) else 0.0,
                "final_rms": float(np.mean(final_values)) if len(final_values) else 0.0,
                "active_frame_count": int(np.sum(active)),
                "max_continuous_active_frames": int(max_run),
            },
            "classification": "not_applicable",
            "resulting_reason": None,
        }
        if lineage_reason is None and source_phone_spans:
            cursor = old_span[0]
            for start, end in sorted(source_phone_spans):
                if start > cursor + AXIS_EPS:
                    lineage_reason = "phone_hole"
                    break
                cursor = max(cursor, end)
            if cursor < old_span[1] - AXIS_EPS:
                lineage_reason = "phone_hole"
        if (audio is not None and sr > 0 and source_phone_spans
                and (min(span[0] for span in source_phone_spans) < -AXIS_EPS
                     or max(span[1] for span in source_phone_spans) > len(audio) / sr + AXIS_EPS)):
            lineage_reason = "audio_hole"
        ctc_boundary_authority = ctc_span is not None
        diagnostic_reasons: list[str] = []
        if (ctc_boundary_authority
                and lineage_reason in {"phone_hole", "audio_hole"}):
            diagnostic_reasons.append(lineage_reason)
            lineage_reason = None
        if ctc_boundary_authority and source_phone_spans:
            source_start = min(span[0] for span in source_phone_spans)
            source_end = max(span[1] for span in source_phone_spans)
            outside_anchor = (
                source_start < ctc_span[0] - AXIS_EPS
                or source_end > ctc_span[1] + AXIS_EPS)
            outside_final = (
                source_start < final_span[0] - AXIS_EPS
                or source_end > final_span[1] + AXIS_EPS)
            if outside_anchor or outside_final:
                diagnostic_reasons.append("source_phone_outside_ctc_or_final")
        if lineage_reason:
            item["lineage"]["reason"] = lineage_reason
        item["lineage"]["ctc_boundary_authority"] = ctc_boundary_authority
        item["lineage"]["diagnostic_reasons"] = diagnostic_reasons
        item["diagnostics"] = {
            "ctc_boundary_authority": ctc_boundary_authority,
            "phone_hole_or_audio_hole_suppressed": bool(diagnostic_reasons and any(
                reason in {"phone_hole", "audio_hole"}
                for reason in diagnostic_reasons)),
            "source_phone_outside_ctc_or_final": (
                "source_phone_outside_ctc_or_final" in diagnostic_reasons),
        }
        if (is_english_token(interval.text) or is_nvv_token(interval.text)
                or interval.xmin <= words_tier.xmin + AXIS_EPS
                or interval.xmax >= words_tier.xmax - AXIS_EPS):
            classification = "not_applicable"
        elif lineage_reason or audio is None or sr <= 0 or not len(core_values):
            classification = "word_energy_evidence_unresolved"
            item["resulting_reason"] = "word_energy_evidence_unresolved"
        else:
            # A source-phone owner with active material outside the lexical
            # core identifies a boundary defect, not a quiet word.  Require
            # three consecutive 10 ms frames to avoid AXIS_EPS noise.
            mismatch = False
            if source_phone_spans and not ctc_boundary_authority:
                source_start = min(span[0] for span in source_phone_spans)
                source_end = max(span[1] for span in source_phone_spans)
                excluded = []
                # Check both the immutable lexical core and the final
                # display span.  A late boundary trim must remain visible as
                # a boundary mismatch even when the premerge core was wider.
                for boundary_start, boundary_end in (old_span, final_span):
                    if source_start < boundary_start - AXIS_EPS:
                        excluded.append((source_start,
                                         min(source_end, boundary_start)))
                    if source_end > boundary_end + AXIS_EPS:
                        excluded.append((max(source_start, boundary_end),
                                         source_end))
                for start, end in excluded:
                    values, _ = _rms_frames_in_span(
                        audio, sr, start, end, frame_ms=10.0)
                    flags = values > max(threshold, 1e-12)
                    run = 0
                    for flag in flags:
                        run = run + 1 if bool(flag) else 0
                        if run >= 3:
                            mismatch = True
                            break
                    if mismatch:
                        break
            if mismatch:
                classification = "word_energy_boundary_mismatch"
                item["resulting_reason"] = "word_energy_boundary_mismatch"
            elif float(np.max(core_values)) <= 1e-12 or float(np.mean(core_values)) < threshold:
                classification = "true_low_energy"
                item["resulting_reason"] = "word_in_silence"
            elif (operation is not None
                  and float(np.mean(final_values)) < threshold):
                classification = "silence_merge_dilution"
                item["resulting_reason"] = None
            else:
                classification = "energetic"
        item["classification"] = classification
        items.append(item)
    return {
        "schema": WORD_ENERGY_EVIDENCE_SCHEMA,
        "noise_model": model,
        "enabled": _word_energy_enabled(args),
        "items": items,
    }


def _noise_floor(audio, sr: int, bottom_pct: float = 0.10) -> float:
    """Estimate noise floor from quietest *bottom_pct* of 5ms frames."""
    import numpy as _np
    rms, _ = _frame_rms_vec(audio, sr, frame_ms=5.0)
    if len(rms) == 0:
        return 0.0
    k = max(1, int(len(rms) * bottom_pct))
    return float(_np.partition(rms, k)[k])


def _visual_words_snapshot(words_tier: Tier | None) -> tuple[tuple[dict, ...], str]:
    """Copy the final visual words intervals and return their stable digest."""
    rows = tuple(
        {"index": index, "xmin": float(interval.xmin),
         "xmax": float(interval.xmax), "text": interval.text}
        for index, interval in enumerate(words_tier.intervals if words_tier else [])
    )
    digest_rows = [{key: value for key, value in row.items() if key != "index"}
                   for row in rows]
    return rows, _evidence_digest(digest_rows)


def _resolve_visual_short_silence_merges(
        textgrid: TextGrid, audio, sr: int, args,
        report: dict | None = None,
        ctc_tokens: list[dict] | None = None,
        reference_punct_entries: list[dict] | None = None,
        fallback_punctuation_projection: dict | None = None) -> list[dict]:
    """Resolve final visual ``<sp0>``/``<sp1>`` owners from local audio energy.

    The snapshot is immutable for the complete decision pass.  Raw CTC/MFA
    spans are used only to identify punctuation ownership; all gap coordinates
    and lexical owners come from the snapshot.  Actual source/derived phone
    synchronization is deliberately deferred to the existing freeze and
    lineage rebuild barrier.  Canonical internal SP0/SP1 eligibility is
    governed by semantic duration, not merge switches or configured max
    values; those values remain diagnostic ledger fields.
    """
    words_tier = tier_by_name(textgrid, "words")
    if words_tier is None:
        return []
    snapshot, visual_digest = _visual_words_snapshot(words_tier)
    decisions: list[dict] = []
    lexical_ordinals = {}
    _lexical_count = 0
    for _row in snapshot:
        _label = str(_row["text"]).strip()
        if _label and not is_silence(_label) and not is_punct(_label):
            lexical_ordinals[int(_row["index"])] = _lexical_count
            _lexical_count += 1
    projected_by_key = {}
    if isinstance(fallback_punctuation_projection, dict):
        for _entry in fallback_punctuation_projection.get("entries", []):
            if not isinstance(_entry, dict):
                continue
            projected_by_key[(_entry.get("left_lexical_ordinal"),
                              _entry.get("right_lexical_ordinal"))] = _entry
    # The premerge display spans are immutable evidence for the later word
    # energy audit.  Keep them on the tier so every copy/freeze boundary can
    # carry the ledger without matching repeated word labels.
    words_tier._word_energy_premerge_spans = {
        str(ordinal): [float(row["xmin"]), float(row["xmax"])]
        for index, ordinal in lexical_ordinals.items()
        for row in (snapshot[index],)
    }
    if report is not None:
        report.setdefault("silence_merges", [])
        report["visual_reference_digest"] = visual_digest

    configured_max_sil = getattr(
        args, "merge_max_sil_sec", getattr(args, "min_sil_merge_sec", 0.2))
    effective_max_sil = min(float(configured_max_sil), 0.5)
    merge_enabled = bool(getattr(args, "merge_silence", True))
    energy_threshold = float(getattr(args, "merge_energy_threshold", 0.5))
    phone_tier = tier_by_name(textgrid, "phones")
    lineage = getattr(textgrid, "_phone_lineage", None)
    lineage_invalid = getattr(textgrid, "_phone_lineage_invalid", None)
    lineage_ambiguous = bool(lineage_invalid)
    if isinstance(lineage, dict):
        lineage_ambiguous = lineage_ambiguous or lineage.get("status") in {
            "rejected", "partial"}
        lineage_ambiguous = lineage_ambiguous or bool(lineage.get("reasons"))

    def _lexical(row: dict) -> bool:
        text = str(row["text"]).strip()
        return bool(text) and not is_silence(text) and not is_punct(text)

    def _overlap(start: float, end: float, other_start: object,
                 other_end: object) -> bool:
        try:
            return (min(end, float(other_end)) > max(start, float(other_start))
                    + AXIS_EPS)
        except (TypeError, ValueError):
            return False

    def _punctuation_neighbor_key(
            start: float, end: float) -> tuple[int | None, int | None]:
        left = next((row for row in reversed(snapshot)
                     if _lexical(row) and row["xmax"] <= start + AXIS_EPS), None)
        right = next((row for row in snapshot
                      if _lexical(row) and row["xmin"] >= end - AXIS_EPS), None)
        return (lexical_ordinals.get(left["index"]) if left else None,
                lexical_ordinals.get(right["index"]) if right else None)

    def _projection_allows_punctuation(
            entry: dict, label: str,
            expected_key: tuple[int | None, int | None]) -> bool:
        """Require every fallback candidate channel to match projection."""
        if not isinstance(fallback_punctuation_projection, dict):
            return True
        projected = projected_by_key.get(expected_key)
        return bool(
            projected is not None
            and str(projected.get("label", "")).strip() == label
            and (entry.get("left_lexical_ordinal"),
                 entry.get("right_lexical_ordinal")) == expected_key)

    def _punctuation_candidate(
            start: float, end: float) -> tuple[str | None, dict | None]:
        """Select one owner/restoration candidate under one shared policy."""
        expected_key = _punctuation_neighbor_key(start, end)
        for entries, reference in ((reference_punct_entries, True),
                                   (ctc_tokens, False)):
            candidates = []
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("word", entry.get("text", ""))).strip()
                kind = str(entry.get("type", entry.get("kind", ""))).casefold()
                if not is_punct(label):
                    continue
                if (not reference and kind not in {"punct", "punctuation"}
                        and not is_punct(label)):
                    continue
                try:
                    entry_start = float(entry.get("start_s"))
                    entry_end = float(entry.get("end_s"))
                except (TypeError, ValueError):
                    continue
                if (not math.isfinite(entry_start)
                        or not math.isfinite(entry_end)
                        or entry_end <= entry_start
                        or not _overlap(start, end, entry_start, entry_end)
                        or not _projection_allows_punctuation(
                            entry, label, expected_key)):
                    continue
                # A broad punctuation span that crosses a lexical owner is
                # not local evidence for this gap.  Without this guard a
                # punctuation anchor around word N also vetoes the gap
                # between word N+1 and N+2.
                if any(
                        _lexical(row)
                        and _overlap(entry_start, entry_end,
                                     float(row["xmin"]), float(row["xmax"]))
                        for row in snapshot):
                    continue
                candidates.append((entry_end - entry_start, entry_start,
                                   label, entry_end))
            if candidates:
                _, entry_start, label, entry_end = min(
                    candidates, key=lambda item: (item[0], item[1]))
                return (
                    "reference_punctuation_owner" if reference
                    else "ctc_punctuation_owner",
                    {"label": label,
                     "source": "reference" if reference else "ctc",
                     "evidence_span": [entry_start, entry_end]},
                )
        return None, None

    def _ctc_word_spans_by_ordinal() -> dict[int, list[dict]]:
        """Return raw CTC word spans keyed by compact lexical ordinal.

        The resolver must not match by surface text: repeated words and bare
        NVV labels make that ambiguous.  Prefer an explicit lexical ordinal
        when the producer supplied one, otherwise use the ordered word-token
        position.  Authority metadata is only a fallback for callers that
        already bound a CTC span but did not pass the raw token list.
        """
        result: dict[int, list[dict]] = {}
        word_ordinal = 0
        for list_index, entry in enumerate(ctc_tokens or []):
            if not isinstance(entry, dict):
                continue
            if str(entry.get("type", "word")) != "word":
                continue
            label = str(entry.get("word", "")).strip()
            if not label or is_silence(label) or is_punct(label):
                continue
            try:
                start = float(entry["start_s"])
                end = float(entry["end_s"])
            except (KeyError, TypeError, ValueError):
                word_ordinal += 1
                continue
            if (not math.isfinite(start) or not math.isfinite(end)
                    or end <= start):
                word_ordinal += 1
                continue
            explicit = entry.get("lexical_ordinal")
            ordinal = explicit if type(explicit) is int else word_ordinal
            result.setdefault(ordinal, []).append({
                "ctc_lexical_ordinal": ordinal,
                "ctc_list_index": list_index,
                "ctc_span": [start, end],
                "source": "ctc_tokens",
            })
            word_ordinal += 1

        authority = _ctc_authority_entries(words_tier)
        if isinstance(authority, list):
            for ordinal, entry in enumerate(authority):
                if result.get(ordinal) or not isinstance(entry, dict):
                    continue
                span = entry.get("ctc_span")
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    continue
                try:
                    start, end = float(span[0]), float(span[1])
                except (TypeError, ValueError):
                    continue
                if (math.isfinite(start) and math.isfinite(end)
                        and end > start):
                    result[ordinal] = [{
                        "ctc_lexical_ordinal": ordinal,
                        "ctc_span": [start, end],
                        "source": "ctc_authority",
                    }]
        return result

    def _unique_ctc_containing_owner(
            left: dict | None, right: dict | None,
            gap_start: float, gap_end: float) -> dict | None:
        """Find one ordinal-unique CTC owner whose span fully contains gap."""
        if not ((left is not None and right is not None)
                and _lexical(left) and _lexical(right)):
            return None
        spans_by_ordinal = _ctc_word_spans_by_ordinal()
        candidates: list[dict] = []
        for side, row in (("left", left), ("right", right)):
            ordinal = lexical_ordinals.get(int(row["index"]))
            span_entries = spans_by_ordinal.get(ordinal, [])
            # Multiple spans bound to the same final ordinal are ambiguous,
            # even when only one happens to contain this particular gap.
            if len(span_entries) != 1:
                continue
            span_entry = span_entries[0]
            start, end = span_entry["ctc_span"]
            contains = (start <= gap_start + AXIS_EPS
                        and end >= gap_end - AXIS_EPS)
            if contains:
                candidates.append({
                    "owner_side": side,
                    "owner_lexical_ordinal": ordinal,
                    "ctc_span": [float(start), float(end)],
                    "ctc_lexical_ordinal": span_entry[
                        "ctc_lexical_ordinal"],
                    "source": span_entry["source"],
                })
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _phone_reason(start: float, end: float) -> str | None:
        if lineage_ambiguous:
            return "phone_lineage_ambiguous"
        if phone_tier is None:
            return None
        overlaps = [phone for phone in phone_tier.intervals
                     if _overlap(start, end, phone.xmin, phone.xmax)]
        if not overlaps:
            return "phone_hole"
        if any(not is_silence(phone.text) for phone in overlaps):
            return "phone_lineage_ambiguous"
        cursor = start
        for phone in sorted(overlaps, key=lambda iv: (iv.xmin, iv.xmax)):
            if phone.xmin > cursor + AXIS_EPS:
                return "phone_hole"
            cursor = max(cursor, phone.xmax)
        if cursor < end - AXIS_EPS:
            return "phone_hole"
        return None

    def _rms_segment(start: float, end: float):
        import numpy as _np
        if end <= start:
            return _np.array([], dtype=_np.float32)
        ss = max(0, int(round(start * sr)))
        ee = min(len(audio), int(round(end * sr)))
        if ee <= ss:
            return _np.array([], dtype=_np.float32)
        values, _ = _frame_rms_vec(audio[ss:ee], sr, frame_ms=5.0)
        return values

    audio_length = len(audio) / sr if audio is not None and sr > 0 else 0.0
    noise_floor = None
    active_floor = None
    all_audio_zero = False
    if audio is not None and sr > 0 and len(audio):
        noise_floor = float(_noise_floor(audio, sr))
        active_floor = max(3.0 * noise_floor, 0.001)
        all_audio_zero = bool(float(np.max(np.abs(audio))) <= 1e-6)

    # Gather every explicit silence run from the immutable visual list.  This
    # includes protected runs so report consumers can see why they survived.
    index = 0
    while index < len(snapshot):
        row = snapshot[index]
        label = str(row["text"]).strip().casefold()
        if not is_silence(label):
            index += 1
            continue
        run_start_index = index
        run_end_index = index
        while (run_end_index + 1 < len(snapshot)
               and is_silence(str(snapshot[run_end_index + 1]["text"]).strip())):
            run_end_index += 1
        left_index = run_start_index - 1
        right_index = run_end_index + 1
        gap_start = float(snapshot[run_start_index]["xmin"])
        gap_end = float(snapshot[run_end_index]["xmax"])
        gap_duration = gap_end - gap_start
        left = snapshot[left_index] if left_index >= 0 else None
        right = snapshot[right_index] if right_index < len(snapshot) else None
        decision = {
            "visual_reference_digest": visual_digest,
            "gap_span": [gap_start, gap_end],
            "run_start_index": run_start_index,
            "run_end_index": run_end_index,
            "gap_label": " ".join(str(snapshot[pos]["text"]).strip()
                                   for pos in range(run_start_index, run_end_index + 1)),
            "label": None,
            "gap_duration": round(gap_duration, 9),
            "configured_max_sil_sec": round(float(configured_max_sil), 9),
            "effective_max_sil_sec": round(float(effective_max_sil), 9),
            "label_matches_duration": False,
            "left_word": left["text"].strip() if left and _lexical(left) else None,
            "right_word": right["text"].strip() if right and _lexical(right) else None,
            "left_index": left_index if left and _lexical(left) else None,
            "right_index": right_index if right and _lexical(right) else None,
            "left_old_span": ([left["xmin"], left["xmax"]]
                              if left and _lexical(left) else None),
            "right_old_span": ([right["xmin"], right["xmax"]]
                               if right and _lexical(right) else None),
            "left_lexical_ordinal": lexical_ordinals.get(left_index),
            "right_lexical_ordinal": lexical_ordinals.get(right_index),
            "old_span": {
                "left": ([left["xmin"], left["xmax"]]
                          if left and _lexical(left) else None),
                "right": ([right["xmin"], right["xmax"]]
                           if right and _lexical(right) else None),
            },
            "new_span": None,
            "energy": {},
            "winner": None,
            "winner_share": None,
            "margin": None,
            "decision": "preserve",
            "reason": None,
            "after_span": None,
            "operation": None,
            "policy": None,
            "direction_source": None,
            "punctuation_owner": None,
            "punctuation_evidence": None,
            "punctuation_gap_restore": False,
            "phone_reason": None,
        }
        run_labels = [_pure_silence_label(snapshot[pos]["text"])
                      for pos in range(run_start_index, run_end_index + 1)]
        run_label = run_labels[0] if run_labels and len(set(run_labels)) == 1 else None
        decision["label"] = run_label
        # Boundary labels are defined on the serialized axis ticks.  Calling
        # ``silence_label`` on the raw binary float would classify 0.2 made
        # from 0.6 - 0.4 as ``<sp0>`` instead of the specified ``<sp1>``.
        gap_ticks = _duration_ticks(gap_start, gap_end)
        expected_label = silence_label(gap_ticks / 1_000_000.0)
        decision["expected_label"] = expected_label
        decision["label_matches_duration"] = (
            run_label is not None and run_label == expected_label)
        owner_reason, punctuation_evidence = _punctuation_candidate(
            gap_start, gap_end)
        phone_reason = _phone_reason(gap_start, gap_end)
        decision["punctuation_owner"] = owner_reason
        decision["punctuation_evidence"] = punctuation_evidence
        decision["phone_reason"] = phone_reason
        terminal_tail_candidate = (
            left is not None and is_punct(str(left["text"]).strip())
            and run_end_index == len(snapshot) - 1
            and bool(run_labels) and all(label is not None for label in run_labels)
            and abs(gap_start - float(left["xmax"])) <= AXIS_EPS
            and gap_end >= words_tier.xmax - AXIS_EPS)
        terminal_punctuation_head_candidate = (
            left is not None and _lexical(left)
            and right is not None and is_punct(str(right["text"]).strip())
            and run_end_index + 1 == len(snapshot) - 1
            and bool(run_labels) and all(label is not None for label in run_labels)
            and abs(gap_start - float(left["xmax"])) <= AXIS_EPS
            and abs(gap_end - float(right["xmin"])) <= AXIS_EPS
            and float(right["xmax"]) >= words_tier.xmax - AXIS_EPS)
        terminal_nvv_sp0_candidate = (
            left is not None and is_nvv_token(str(left["text"]).strip())
            and right is None
            and run_end_index == len(snapshot) - 1
            and run_label == "<sp0>"
            and decision["label_matches_duration"]
            and gap_start > words_tier.xmin + AXIS_EPS
            and gap_end >= words_tier.xmax - AXIS_EPS
            and abs(gap_start - float(left["xmax"])) <= AXIS_EPS
            and gap_ticks < _threshold_ticks(0.2)
            and owner_reason is None)
        internal_shape = (
            left is not None and right is not None
            and _lexical(left) and _lexical(right)
            and gap_start > words_tier.xmin + AXIS_EPS
            and gap_end < words_tier.xmax - AXIS_EPS
            and owner_reason is None)
        # ``silence_label`` classifies an exact 200 ms interval as sp1, but
        # CTC/MFA frame snapping can turn a sub-200 ms source sp0 into exactly
        # 200000 us.  The configured merge bound is inclusive: retaining that
        # one boundary value would publish an interior sp1 and filter an
        # otherwise exact utterance (LAria_00242).
        valid_internal_sp0 = (
            run_label == "<sp0>"
            and (decision["label_matches_duration"]
                 or gap_ticks == _threshold_ticks(0.2))
            and internal_shape
            and gap_ticks <= _threshold_ticks(0.2))
        left_is_nvv = bool(left is not None and is_nvv_token(left["text"]))
        right_is_nvv = bool(right is not None and is_nvv_token(right["text"]))
        nvv_adjacent_sp0 = valid_internal_sp0 and (left_is_nvv or right_is_nvv)
        forced_internal_sp1 = (
            run_label == "<sp1>"
            and decision["label_matches_duration"]
            and internal_shape
            and not left_is_nvv and not right_is_nvv
            and gap_ticks < _threshold_ticks(0.5))
        canonical_internal_sp1 = (
            run_label == "<sp1>"
            and decision["label_matches_duration"]
            and internal_shape
            and gap_ticks < _threshold_ticks(0.5))
        eligible_internal = valid_internal_sp0 or canonical_internal_sp1
        nvv_adjacent_sp1_candidate = (
            canonical_internal_sp1 and (left_is_nvv or right_is_nvv))
        ctc_containing_owner = (
            _unique_ctc_containing_owner(left, right, gap_start, gap_end)
            if eligible_internal else None)
        decision["left_is_nvv"] = left_is_nvv
        decision["right_is_nvv"] = right_is_nvv
        decision["valid_internal_sp0"] = valid_internal_sp0
        decision["nvv_adjacent_sp0"] = nvv_adjacent_sp0
        decision["forced_internal_sp1"] = forced_internal_sp1
        decision["terminal_punctuation_head_candidate"] = (
            terminal_punctuation_head_candidate)
        decision["terminal_nvv_sp0_candidate"] = terminal_nvv_sp0_candidate
        decision["canonical_internal_sp1"] = canonical_internal_sp1
        decision["eligible_internal"] = eligible_internal
        decision["ctc_containing_owner"] = ctc_containing_owner
        decision["nvv_adjacent_sp1_ctc_evidence"] = (
            ctc_containing_owner if nvv_adjacent_sp1_candidate else None)
        if ctc_containing_owner is not None:
            decision["ctc_owner_lexical_ordinal"] = (
                ctc_containing_owner["owner_lexical_ordinal"])
            decision["ctc_span"] = list(
                ctc_containing_owner["ctc_span"])
        if eligible_internal:
            forced_gate_reasons: list[str] = []
            if _duration_ticks(gap_start, gap_end) > _threshold_ticks(effective_max_sil):
                forced_gate_reasons.append("long_pause")
            if not merge_enabled:
                forced_gate_reasons.append("merge_disabled")
            if phone_reason:
                forced_gate_reasons.append(phone_reason)
            if audio is None or sr <= 0:
                forced_gate_reasons.append("missing_audio")
            elif gap_start < 0.0 or gap_end > audio_length + 1e-9:
                forced_gate_reasons.append("audio_not_covered")
            elif all_audio_zero:
                forced_gate_reasons.append("preserve_all_zero_audio")
            decision["forced_gate_reasons"] = forced_gate_reasons
        if terminal_nvv_sp0_candidate:
            decision["operation"] = "terminal_nvv_sp0_absorption"
            decision["policy"] = "terminal_nvv_sp0_absorption"
            decision["decision"] = "terminal_nvv_sp0_absorption"
            decision["reason"] = "terminal_nvv_sp0_absorption"
            decision["owner_index"] = left_index
            decision["left_owner"] = left["text"].strip()
            decision["tail_span"] = [gap_start, gap_end]
        elif terminal_punctuation_head_candidate:
            decision["operation"] = "terminal_punctuation_head_absorption"
            decision["policy"] = "terminal_punctuation_head_absorption"
            decision["decision"] = "terminal_punctuation_head_absorption"
            decision["reason"] = "terminal_punctuation_head_absorption"
            decision["owner_index"] = right_index
            decision["punctuation_index"] = right_index
            decision["punctuation_label"] = str(right["text"]).strip()
            decision["tail_span"] = [gap_start, gap_end]
        elif terminal_tail_candidate:
            decision["operation"] = "terminal_punctuation_tail_absorption"
            decision["policy"] = "terminal_punctuation_tail_absorption"
            decision["decision"] = "terminal_punctuation_tail_absorption"
            decision["reason"] = "terminal_punctuation_tail_absorption"
            decision["owner_index"] = left_index
            decision["left_owner"] = left["text"].strip()
            decision["tail_span"] = [gap_start, gap_end]
        elif (left is None or right is None or not _lexical(left)
                or not _lexical(right)
                or gap_start <= words_tier.xmin + AXIS_EPS
                or gap_end >= words_tier.xmax - AXIS_EPS):
            decision["reason"] = "edge"
        elif owner_reason:
            decision["reason"] = owner_reason
        elif run_label is None:
            decision["reason"] = "mixed_or_noncanonical_silence_labels"
        elif run_label not in {"<sp0>", "<sp1>"}:
            decision["reason"] = "unsupported_silence_label"
        elif not decision["label_matches_duration"] and not valid_internal_sp0:
            decision["reason"] = "silence_label_duration_mismatch"
        elif (run_label == "<sp0>"
              and gap_ticks > _threshold_ticks(0.2)):
            decision["reason"] = "long_pause"
        elif not eligible_internal:
            decision["reason"] = "silence_label_duration_mismatch"
        elif ctc_containing_owner is not None:
            direction = ctc_containing_owner["owner_side"]
            decision["decision"] = "merged_left" if direction == "left" else "merged_right"
            decision["reason"] = (
                "nvv_adjacent_sp1_ctc_containing_owner"
                if nvv_adjacent_sp1_candidate else "ctc_containing_owner")
            decision["policy"] = (
                "nvv_adjacent_sp1_ctc_containing_owner"
                if nvv_adjacent_sp1_candidate else "ctc_containing_owner")
            decision["direction_source"] = "ctc_containing_owner"
        else:
            # Missing/ambiguous audio or phone evidence has no direction.  It
            # therefore falls through to the canonical merged-left fallback;
            # an accepted energy owner is the only acoustic direction.
            energy_reason = None
            if phone_reason:
                energy_reason = phone_reason
            elif audio is None or sr <= 0:
                energy_reason = "missing_audio"
            elif gap_start < 0.0 or gap_end > audio_length + 1e-9:
                energy_reason = "audio_not_covered"
            elif all_audio_zero:
                energy_reason = "preserve_all_zero_audio"
            else:
                gap_rms = _rms_segment(gap_start, gap_end)
                left_context = _rms_segment(max(left["xmin"], gap_start - 0.05),
                                            gap_start)
                right_context = _rms_segment(gap_end,
                                             min(right["xmax"], gap_end + 0.05))
                active = gap_rms >= float(active_floor)
                max_run = run = 0
                for flag in active:
                    run = run + 1 if bool(flag) else 0
                    max_run = max(max_run, run)
                left_half = gap_rms[:len(gap_rms) // 2]
                right_half = gap_rms[len(gap_rms) // 2:]
                excess = np.maximum(gap_rms - float(active_floor), 0.0)
                left_mass = float(np.sum(excess[:len(excess) // 2]))
                right_mass = float(np.sum(excess[len(excess) // 2:]))
                total_mass = left_mass + right_mass
                winner = "merged_left" if left_mass >= right_mass else "merged_right"
                winner_mass = max(left_mass, right_mass)
                loser_mass = min(left_mass, right_mass)
                winner_share = winner_mass / total_mass if total_mass > 0 else 0.0
                margin = ((winner_mass - loser_mass) / total_mass
                          if total_mass > 0 else 0.0)
                left_gap_median = float(np.median(left_half)) if len(left_half) else 0.0
                right_gap_median = float(np.median(right_half)) if len(right_half) else 0.0
                left_context_median = (float(np.median(left_context))
                                       if len(left_context) else 0.0)
                right_context_median = (float(np.median(right_context))
                                        if len(right_context) else 0.0)
                left_ratio = left_gap_median / max(left_context_median, 1e-6)
                right_ratio = right_gap_median / max(right_context_median, 1e-6)
                context_ratio = left_ratio if winner == "merged_left" else right_ratio
                decision["energy"] = {
                    "noise_floor": round(float(noise_floor), 9),
                    "active_floor": round(float(active_floor), 9),
                    "frame_ms": 5.0,
                    "frame_count": int(len(gap_rms)),
                    "active_frame_count": int(np.sum(active)),
                    "max_continuous_active_frames": int(max_run),
                    "left_excess_energy_mass": round(left_mass, 9),
                    "right_excess_energy_mass": round(right_mass, 9),
                    "left_gap_median_rms": round(left_gap_median, 9),
                    "right_gap_median_rms": round(right_gap_median, 9),
                    "left_context_median_rms": round(left_context_median, 9),
                    "right_context_median_rms": round(right_context_median, 9),
                    "left_context_ratio": round(left_ratio, 9),
                    "right_context_ratio": round(right_ratio, 9),
                    "winner_context_ratio": round(context_ratio, 9),
                }
                decision["winner"] = winner
                decision["winner_share"] = round(winner_share, 9)
                decision["margin"] = round(margin, 9)
                if max_run < 3:
                    energy_reason = "preserve_no_continuous_active"
                elif total_mass <= 0 or winner_mass < float(active_floor):
                    energy_reason = "preserve_low_energy"
                elif winner_share < 0.55 or margin < 0.10:
                    energy_reason = "preserve_ambiguous_energy"
                elif context_ratio < energy_threshold:
                    energy_reason = "preserve_context_ratio"
                else:
                    decision["decision"] = winner
                    decision["reason"] = "energy_owner"
                    decision["policy"] = "energy_owner"
                    decision["direction_source"] = "energy_owner"
                    decision["energy_reason"] = "energy_owner"
            if decision["decision"] == "preserve":
                decision["energy_reason"] = energy_reason
                decision["forced_original_reason"] = energy_reason
                decision["decision"] = "merged_left"
                if nvv_adjacent_sp0:
                    decision["reason"] = "nvv_adjacent_sp0_forward"
                    decision["policy"] = "nvv_adjacent_sp0_forward"
                else:
                    decision["reason"] = "merged_left_fallback"
                    decision["policy"] = "merged_left_fallback"
                decision["direction_source"] = "forced_left_fallback"
        if owner_reason and punctuation_evidence is not None:
            decision["punctuation_gap_restore"] = True
            decision["operation"] = "punctuation_gap_restore"
            decision["punctuation_label"] = punctuation_evidence["label"]
        if decision["after_span"] is None:
            decision["after_span"] = {
                "left": decision["left_old_span"],
                "right": decision["right_old_span"],
            }
        decisions.append(decision)
        index = run_end_index + 1

    merges = [item for item in decisions if item["decision"] in {
        "merged_left", "merged_right"}]
    terminal_absorptions = [
        item for item in decisions
        if item.get("operation") in {
            "terminal_punctuation_head_absorption",
            "terminal_punctuation_tail_absorption",
            "terminal_nvv_sp0_absorption",
        }]
    punctuation_restorations = [
        item for item in decisions if item.get("punctuation_gap_restore")]
    if merges or terminal_absorptions or punctuation_restorations:
        updated = [Interval(row["xmin"], row["xmax"], row["text"])
                   for row in snapshot]
        removed: set[int] = set()
        for item in merges:
            left_index = int(item["left_index"])
            right_index = int(item["right_index"])
            gap_start, gap_end = item["gap_span"]
            if item["decision"] == "merged_left":
                old = updated[left_index]
                updated[left_index] = Interval(old.xmin, max(old.xmax, gap_end), old.text)
                item["after_span"] = [old.xmin, max(old.xmax, gap_end)]
            else:
                old = updated[right_index]
                updated[right_index] = Interval(min(old.xmin, gap_start), old.xmax, old.text)
                item["after_span"] = [min(old.xmin, gap_start), old.xmax]
            item["new_span"] = item["after_span"]
            for pos in range(left_index + 1, right_index):
                removed.add(pos)
            item["operation"] = _merge_operation_for_policy(
                item.get("policy"))
        for item in terminal_absorptions:
            owner_index = int(item["owner_index"])
            old = updated[owner_index]
            tail_end = float(item["tail_span"][1])
            if item.get("operation") == "terminal_punctuation_head_absorption":
                tail_start = float(item["tail_span"][0])
                updated[owner_index] = Interval(
                    min(old.xmin, tail_start), old.xmax, old.text)
            else:
                updated[owner_index] = Interval(old.xmin, max(old.xmax, tail_end), old.text)
            item["after_span"] = [updated[owner_index].xmin,
                                   updated[owner_index].xmax]
            item["new_span"] = item["after_span"]
            for pos in range(item["run_start_index"], item["run_end_index"] + 1):
                removed.add(pos)
        for item in punctuation_restorations:
            start = int(item["run_start_index"])
            gap_start, gap_end = item["gap_span"]
            updated[start] = Interval(
                float(gap_start), float(gap_end),
                str(item["punctuation_label"]))
            item["after_span"] = [float(gap_start), float(gap_end)]
            item["new_span"] = item["after_span"]
            for pos in range(start + 1, int(item["run_end_index"]) + 1):
                removed.add(pos)
        words_tier.intervals = [iv for pos, iv in enumerate(updated) if pos not in removed]
        words_tier._word_energy_merge_ledger = list(merges) + list(terminal_absorptions)
        for item in merges:
            _record_processed_geometry_operation(
                words_tier,
                _merge_operation_for_policy(item.get("policy")),
                decision=item)
        for item in terminal_absorptions:
            _record_processed_geometry_operation(
                words_tier, str(item["operation"]), decision=item,
                policy=item.get("policy"))
        for item in punctuation_restorations:
            _record_processed_geometry_operation(
                words_tier, "punctuation_gap_restore", decision=item)
    if report is not None:
        report["silence_merges"] = decisions
        report["terminal_punctuation_tail_absorption"] = [
            item for item in terminal_absorptions
            if item.get("operation") == "terminal_punctuation_tail_absorption"]
        report["terminal_punctuation_head_absorption"] = [
            item for item in terminal_absorptions
            if item.get("operation") == "terminal_punctuation_head_absorption"]
        report["terminal_nvv_sp0_absorption"] = [
            item for item in terminal_absorptions
            if item.get("operation") == "terminal_nvv_sp0_absorption"]
        report["punctuation_gap_restorations"] = punctuation_restorations
    if not hasattr(words_tier, "_word_energy_merge_ledger"):
        words_tier._word_energy_merge_ledger = list(merges) + list(terminal_absorptions)
    return decisions


# Descriptive aliases keep the helper discoverable for callers/tests while
# preserving the private naming convention used by this module.
_resolve_visual_words_silence_merges = _resolve_visual_short_silence_merges
resolve_visual_short_silence_merges = _resolve_visual_short_silence_merges


def _normalize_final_internal_silence_labels(
        textgrid: TextGrid, report: dict | None = None) -> list[dict]:
    """Normalize retained internal pure-silence labels after owner commit.

    The visual resolver must see the original labels: normalizing before that
    decision would turn a stale exact-200ms ``<sp0>`` into ``<sp1>`` early and
    lose the inclusive stale-SP0 exception.  This pass therefore runs only
    after all owner mutations, changes labels (never spans), and before the
    processed-geometry freeze.
    """
    words_tier = tier_by_name(textgrid, "words")
    if words_tier is None:
        return []

    intervals = words_tier.intervals
    normalized: list[dict] = []
    retained: dict[tuple[int, int], dict] = {}
    for index, interval in enumerate(intervals):
        # Leading/trailing silence, including the leading <sp1> convention,
        # is intentionally outside this final internal-only normalization.
        if index == 0 or index == len(intervals) - 1:
            continue
        current = _pure_silence_label(interval.text)
        if current is None:
            continue
        duration_us = _duration_ticks(interval.xmin, interval.xmax)
        expected = _silence_label_from_ticks(duration_us)
        status = "unchanged"
        if current != expected:
            interval.text = expected
            status = "relabelled"
            operation = {
                "operation": "final_internal_silence_label_normalization",
                "span": [float(interval.xmin), float(interval.xmax)],
                "duration_us": duration_us,
                "from_label": current,
                "to_label": expected,
            }
            _record_processed_geometry_operation(
                words_tier, operation["operation"],
                span=operation["span"], duration_us=operation["duration_us"],
                from_label=operation["from_label"],
                to_label=operation["to_label"])
            normalized.append(operation)
        retained[(_duration_ticks(0.0, interval.xmin),
                  _duration_ticks(0.0, interval.xmax))] = {
            "serialized_label": interval.text.strip(),
            "normalization_status": status,
            "span": [float(interval.xmin), float(interval.xmax)],
        }

    # Preserve the original owner decision evidence.  These fields describe
    # the pre-normalization decision; serialized_label/status describe only
    # the final retained interval that is about to be frozen.
    decisions = report.get("silence_merges", []) if isinstance(report, dict) else []
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            span = decision.get("gap_span")
            match = None
            if isinstance(span, (list, tuple)) and len(span) == 2:
                start_ticks = _duration_ticks(0.0, span[0])
                end_ticks = _duration_ticks(0.0, span[1])
                for item in retained.values():
                    item_start = _duration_ticks(0.0, item["span"][0])
                    item_end = _duration_ticks(0.0, item["span"][1])
                    if (item_start == start_ticks and item_end == end_ticks):
                        match = item
                        break
            if match is None:
                decision["serialized_label"] = None
                decision["normalization_status"] = "not_retained_or_non_internal"
            else:
                decision["serialized_label"] = match["serialized_label"]
                decision["normalization_status"] = match["normalization_status"]

    if isinstance(report, dict):
        report["silence_label_normalization"] = list(normalized)
    return normalized


def _is_alpha_group(s: str) -> bool:
    """True for ASCII strings whose characters are all alpha or hyphen (NVV tokens)."""
    return s.isascii() and bool(s) and all(c.isalpha() or c == '-' for c in s)


# ── Merge-words dictionary ────────────────────────────────────────────
def _remove_nth_char(text: str, char: str, n: int) -> str:
    """删除 text 中第 n 个 (1-indexed) char 字符."""
    idx = -1
    for _ in range(n):
        idx = text.find(char, idx + 1)
        if idx == -1:
            return text
    return text[:idx] + text[idx + 1:]


def build_corrected_text(words_tier: Tier, raw_text: str, pinyin_text: str) -> str:
    """Compare punctuation in pinyin text with actual silence gaps in words tier.

    Returns corrected Chinese text:
      - Delete punctuation where no corresponding silence exists
      - Insert ``[sp]`` where silence exists but no punctuation
    """
    # ---- tokenize both sides ----
    pinyin_tokens = pinyin_text.split()
    word_items = [(iv.text.strip(), is_silence(iv.text)) for iv in words_tier.intervals]

    # word indices: exclude NVV tokens (transparent — not in raw Chinese text)
    py_word_idx = [i for i, t in enumerate(pinyin_tokens)
                   if is_word_like(t) and not is_nvv_token(t)]
    tg_word_idx = [i for i, (text, is_sil) in enumerate(word_items)
                   if not is_sil and not is_nvv_token(text) and not is_punct(text)]

    n_py = len(py_word_idx)
    n_tg = len(tg_word_idx)

    if n_py == 0 or n_tg == 0 or n_py != n_tg:
        return raw_text   # cannot reliably cross-check — return original

    n = n_py  # number of words

    # ---- build gap_sil[0..n] from words tier ----
    gap_sil = [False] * (n + 1)

    # leading gap
    if tg_word_idx[0] > 0:
        gap_sil[0] = any(word_items[i][1] for i in range(0, tg_word_idx[0]))

    # between-word gaps (gaps 1 .. n-1)
    for k in range(n - 1):
        lo = tg_word_idx[k] + 1
        hi = tg_word_idx[k + 1]
        gap_sil[k + 1] = any(word_items[i][1] for i in range(lo, hi))

    # trailing gap
    if tg_word_idx[-1] < len(word_items) - 1:
        gap_sil[n] = any(word_items[i][1] for i in range(tg_word_idx[-1] + 1, len(word_items)))

    # ---- build gap_punct[0..n] from pinyin ----
    gap_punct = [False] * (n + 1)

    # leading punct
    if py_word_idx[0] > 0:
        gap_punct[0] = any(is_punct(pinyin_tokens[i]) for i in range(0, py_word_idx[0]))

    # between-word punct
    for k in range(n - 1):
        lo = py_word_idx[k] + 1
        hi = py_word_idx[k + 1]
        gap_punct[k + 1] = any(is_punct(pinyin_tokens[i]) for i in range(lo, hi))

    # trailing punct
    if py_word_idx[-1] < len(pinyin_tokens) - 1:
        gap_punct[n] = any(is_punct(pinyin_tokens[i])
                           for i in range(py_word_idx[-1] + 1, len(pinyin_tokens)))

    # ---- walk raw Chinese text and produce corrected version ----
    # Use _extract_word_chars to get proper word units (CJK chars, English word
    # groups, punctuation).  Character-level iteration miscounts English words
    # where a multi-letter token like "ria" is one word unit but 3 word-like
    # characters, causing word_idx to drift out of sync with gap_sil/gap_punct.
    char_units = _extract_word_chars(raw_text)
    if not char_units:
        return raw_text

    # Build a parallel pinyin-word iterator so we know how many pinyin tokens
    # each char_unit consumes.  We need this because English word groups (e.g.
    # "live") are one char_unit but may map to one or more pinyin Word tokens.
    py_words = [t for t in pinyin_tokens
                if is_word_like(t) and not is_nvv_token(t)]
    py_cursor = 0

    result = []
    word_idx = 0  # word position (aligned with py_words / tg_word_idx)

    for unit in char_units:
        if is_word_like(unit):
            # How many pinyin-word slots does this unit consume?
            if is_cjk(unit):
                consume = 1
            else:
                # English / alpha group: consume consecutive pinyin tokens that
                # are also English (no tone digit) until we hit a CJK-linked
                # pinyin token or an NVV token.
                consume = 0
                while py_cursor < len(py_words):
                    t = py_words[py_cursor]
                    if t.isascii() and t.isalpha() and not t.isdigit():
                        consume += 1
                        py_cursor += 1
                    else:
                        break
                if consume == 0:
                    consume = 1  # safety: at least one slot

            # Emit gap marker before this word (if needed)
            if word_idx > 0:
                gap_pos = word_idx
                if gap_pos < len(gap_sil) and gap_sil[gap_pos] and not gap_punct[gap_pos]:
                    result.append('[sp]')

            result.append(unit)
            word_idx += consume
        elif is_punct(unit):
            gap_pos = word_idx  # gap after the last word
            if gap_pos < len(gap_sil):
                if gap_sil[gap_pos]:
                    result.append(unit)
            else:
                result.append(unit)
        else:
            result.append(unit)  # whitespace, etc.

    return ''.join(result)


# ---------------------------------------------------------------------------
# Energy-based fix (unchanged)
# ---------------------------------------------------------------------------

# (load_audio / frame_rms / median replaced by NumPy vectorised versions above)


def _frame_rms_legacy(audio, frame_size: int, hop_size: int):
    """Compatibility wrapper — use _frame_rms_vec for new code."""
    import numpy as _np
    if len(audio) < frame_size:
        return []
    n_frames = (len(audio) - frame_size) // hop_size + 1
    if n_frames <= 0:
        return []
    # Build frame indices (non-vectorised but much faster than element-wise)
    idx = _np.arange(n_frames) * hop_size
    frames = _np.array([audio[i:i + frame_size] for i in idx])
    rms = _np.sqrt(_np.mean(frames.astype(_np.float64) ** 2, axis=1) + 1e-12)
    return rms.tolist()


def _median_legacy(values) -> float:
    """Compatibility wrapper — use np.median for new code."""
    import numpy as _np
    if not hasattr(values, '__len__') or len(values) == 0:  # type: ignore[arg-type]
        return 0.0
    return float(_np.median(_np.asarray(values, dtype=_np.float64)))

# Alias old names to legacy wrappers (all callers continue to work)
frame_rms = _frame_rms_legacy
median = _median_legacy


def find_speech_in_silence(
    audio, sr: int, sil_start: float, sil_end: float,
    search_sec: float, frame_ms: float, hop_ms: float,
    thresh_ratio: float, min_region_sec: float,
) -> tuple[float, float] | None:
    """Find speech burst inside a silence region (vectorised)."""
    import numpy as _np
    search_end = min(sil_end, sil_start + search_sec)
    ss = max(0, int(sil_start * sr))
    es = min(len(audio), int(search_end * sr))
    if es <= ss:
        return None
    rms, frame_dur = _frame_rms_vec(audio[ss:es], sr, frame_ms=hop_ms)
    if len(rms) == 0:
        return None
    tail = rms[max(0, int(len(rms) * 0.6)):]
    noise = float(_np.median(tail)) if len(tail) > 0 else float(_np.median(rms))
    peak = float(_np.max(rms))
    threshold = max(noise * thresh_ratio, peak * 0.15)
    min_f = max(1, int(min_region_sec / (hop_ms / 1000.0)))
    active = rms > threshold
    # Find first sustained active run
    first = None
    for i in range(len(active) - min_f + 1):
        if _np.all(active[i:i + min_f]):
            first = i
            break
    if first is None:
        return None
    # Find first sustained inactive run after 'first'
    last = None
    for i in range(first + min_f, len(active) - min_f + 1):
        if _np.all(~active[i:i + min_f]):
            last = i
            break
    if last is None:
        last = int(_np.max(_np.where(active)[0])) + 1
    sp_start = sil_start + first * frame_dur
    sp_end = sil_start + last * frame_dur + frame_ms / 1000.0
    sp_end = min(sp_end, sil_end)
    if sp_end - sp_start < min_region_sec or sp_start - sil_start > 0.35:
        return None
    return sp_start, sp_end


def nonzero_mean(segment) -> float:
    """Mean absolute amplitude, ignoring near-zero samples (vectorised)."""
    import numpy as _np
    seg = _np.asarray(segment, dtype=_np.float32)
    nz = _np.abs(seg)
    mask = nz > 1e-12
    if not mask.any():
        return 0.0
    return float(_np.mean(nz[mask]))


def merge_short_silences(textgrid: TextGrid, wav_path: Path | None, args,
                         audio: list[float] | None = None, sr: int = 16000) -> tuple[TextGrid, list[dict]]:
    """
    Merge short sil intervals into the previous phone when energy conditions are met.

    For each 'sil' interval in the phones tier:
    1. Duration must be < merge_max_sil_sec
    2. Non-zero energy mean > previous phone non-zero mean * merge_energy_threshold

    If both pass, the sil is merged into the previous phone (extend its xmax),
    and the matching <eps> in the words tier is merged into the previous word.
    """
    if audio is None and (wav_path is None or not wav_path.exists()):
        return textgrid, []
    if audio is None:
        audio, sr = load_audio(wav_path)
    words = tier_by_name(textgrid, "words")
    phones = tier_by_name(textgrid, "phones")
    if words is None or phones is None:
        return textgrid, []

    merges = []

    for pi, p_iv in enumerate(phones.intervals):
        if p_iv.text.strip() != "sil":
            continue
        if p_iv.duration >= args.merge_max_sil_sec:
            continue
        if pi == 0:
            continue

        prev_iv = phones.intervals[pi - 1]

        # Compute energy for sil and previous phone
        sil_ss = max(0, int(p_iv.xmin * sr))
        sil_es = min(len(audio), int(p_iv.xmax * sr))
        prev_ss = max(0, int(prev_iv.xmin * sr))
        prev_es = min(len(audio), int(prev_iv.xmax * sr))

        sil_energy = nonzero_mean(audio[sil_ss:sil_es])
        prev_energy = nonzero_mean(audio[prev_ss:prev_es])

        if sil_energy <= prev_energy * args.merge_energy_threshold:
            continue

        # Find matching <eps> in words tier
        word_idx = None
        for wi, w_iv in enumerate(words.intervals):
            if w_iv.text.strip() == "<eps>" and \
               abs(w_iv.xmin - p_iv.xmin) < 0.01 and abs(w_iv.xmax - p_iv.xmax) < 0.01:
                word_idx = wi
                break

        merges.append({
            "phone_idx": pi, "prev_phone_idx": pi - 1,
            "word_idx": word_idx,
            "sil_energy": round(sil_energy, 6),
            "prev_energy": round(prev_energy, 6),
        })

    if not merges:
        return textgrid, []

    # Apply merges (reverse order to preserve indices)
    new_phones = [Interval(iv.xmin, iv.xmax, iv.text) for iv in phones.intervals]
    new_words = [Interval(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]

    for m in sorted(merges, key=lambda x: x["phone_idx"], reverse=True):
        si = m["phone_idx"]
        pi = m["prev_phone_idx"]
        if si < len(new_phones) and pi < len(new_phones):
            new_phones[pi].xmax = new_phones[si].xmax
            del new_phones[si]

        wi = m["word_idx"]
        if wi is not None and 0 < wi < len(new_words):
            new_words[wi - 1].xmax = new_words[wi].xmax
            del new_words[wi]

    new_tiers = []
    for tier in textgrid.tiers:
        if tier.name.lower() == "phones":
            new_tiers.append(Tier(tier.name, tier.xmin, tier.xmax, new_phones))
        elif tier.name.lower() == "words":
            new_tiers.append(Tier(tier.name, tier.xmin, tier.xmax, new_words))
        else:
            new_tiers.append(tier)

    return TextGrid(textgrid.xmin, textgrid.xmax, new_tiers), merges


def fix_short_words(textgrid: TextGrid, wav_path: Path | None, args,
                    audio: list[float] | None = None, sr: int = 16000) -> tuple[TextGrid, list[dict]]:
    if audio is None and (wav_path is None or not wav_path.exists()):
        return textgrid, []
    if audio is None:
        audio, sr = load_audio(wav_path)
    words = tier_by_name(textgrid, "words")
    phones = tier_by_name(textgrid, "phones")
    if words is None or phones is None:
        return textgrid, []
    fixes = []
    candidates = []
    for idx, iv in enumerate(words.intervals[:-1]):
        next_iv = words.intervals[idx + 1]
        if (not is_english_token(iv.text)
                and iv.text.strip().lower().rstrip('12345') in {w.rstrip('12345') for w in CHINESE_SHORT_WORDS}
                and iv.duration < args.fix_short_word_sec
                and is_silence(next_iv.text)
                and next_iv.duration >= args.fix_min_silence_sec):
            candidates.append(idx)
    # Extension: very short content words (< 50 ms) between two non-short,
    # non-silence words.  These are MFA artifacts (squeezed words) or
    # incorrect splits — try to extend using energy-based boundary search.
    content_candidates = []
    for idx, iv in enumerate(words.intervals[1:-1], start=1):
        if (not is_silence(iv.text) and not is_punct(iv.text)
                and not is_nvv_token(iv.text)
                and iv.duration < 0.050
                and iv.text.strip().lower().rstrip('12345')
                not in {w.rstrip('12345') for w in CHINESE_SHORT_WORDS}):
            prev_iv = words.intervals[idx - 1]
            next_iv = words.intervals[idx + 1]
            if (not is_silence(prev_iv.text) and not is_silence(next_iv.text)
                    and prev_iv.duration >= 0.050 and next_iv.duration >= 0.050):
                content_candidates.append(idx)
    if not candidates and not content_candidates:
        return textgrid, fixes
    for word_idx in candidates:
        word_iv = words.intervals[word_idx]
        sil_iv = words.intervals[word_idx + 1]
        region = find_speech_in_silence(
            audio, sr, sil_iv.xmin, sil_iv.xmax,
            search_sec=args.fix_search_sec, frame_ms=args.fix_frame_ms,
            hop_ms=args.fix_hop_ms, thresh_ratio=args.fix_threshold_ratio,
            min_region_sec=args.fix_min_region_sec,
        )
        if region is None:
            continue
        sp_start, sp_end = region
        if sp_end <= word_iv.xmax or sp_end >= sil_iv.xmax:
            continue
        old_xmax = word_iv.xmax
        word_iv.xmax = sp_end
        sil_iv.xmin = sp_end
        # Only extend the phone that touches the original word end boundary
        # (the last phone of the word).  Extending all phones would make the
        # first phone span the whole word and zero out the second syllable
        # in downstream tiers like pinyin_phones.
        for pi in [i for i, p in enumerate(phones.intervals)
                   if not is_silence(p.text) and abs(p.xmax - old_xmax) < 0.02]:
            phones.intervals[pi].xmax = sp_end
            # Keep the phones tier contiguous — the next interval's xmin must
            # follow suit, otherwise the extended phone overlaps the silence.
            if pi + 1 < len(phones.intervals):
                phones.intervals[pi + 1].xmin = sp_end
        fixes.append({"rule": "short_word_fix", "word": word_iv.text})
    # ── Content word candidates: bidirectional energy search ──
    for word_idx in content_candidates:
        word_iv = words.intervals[word_idx]
        prev_iv = words.intervals[word_idx - 1]
        next_iv = words.intervals[word_idx + 1]
        # Search rightward: check if the short word + next word's onset
        # region has continuous speech energy.
        region = find_speech_in_silence(
            audio, sr, word_iv.xmin,
            min(next_iv.xmin + 0.10, next_iv.xmax),
            search_sec=0.15, frame_ms=args.fix_frame_ms,
            hop_ms=args.fix_hop_ms, thresh_ratio=args.fix_threshold_ratio,
            min_region_sec=0.015,
        )
        if region:
            sp_start, sp_end = region
            if sp_end > word_iv.xmax and sp_end <= next_iv.xmin + 0.01:
                old_xmax = word_iv.xmax
                word_iv.xmax = sp_end
                next_iv.xmin = sp_end
                for pi in [i for i, p in enumerate(phones.intervals)
                           if not is_silence(p.text) and abs(p.xmax - old_xmax) < 0.02]:
                    phones.intervals[pi].xmax = sp_end
                    if pi + 1 < len(phones.intervals):
                        phones.intervals[pi + 1].xmin = sp_end
                fixes.append({"rule": "content_short_word_fix", "word": word_iv.text})
    return textgrid, fixes


# ---------------------------------------------------------------------------
# BGM / noise detection (global noise floor + per-silence energy check)
# ---------------------------------------------------------------------------

def detect_bgm_suspect(textgrid: TextGrid, wav_path: Path | None, args,
                        audio: list[float] | None = None, sr: int = 16000) -> list[dict]:
    """
    Detect if silence intervals have abnormally high energy (BGM/noise residual).

    Uses global noise floor estimation (bottom 60% RMS median of entire audio),
    then checks each silence interval against it. Flags the file if too many
    silence intervals are above the noise floor.
    """
    if audio is None and (wav_path is None or not wav_path.exists()):
        return []
    if audio is None:
        audio, sr = load_audio(wav_path)

    phones = tier_by_name(textgrid, "phones")
    if phones is None:
        return []

    # Step 1: noise floor from silence-labeled frames only
    frame_size = max(1, int(args.bgm_frame_ms / 1000.0 * sr))
    hop_size = max(1, int(args.bgm_hop_ms / 1000.0 * sr))

    # Collect RMS from all frames that fall within silence intervals
    sil_rms_vals = []
    for p_iv in phones.intervals:
        if not is_silence(p_iv.text) and p_iv.text != 'spn':
            continue
        ss = max(0, int(p_iv.xmin * sr))
        es = min(len(audio), int(p_iv.xmax * sr))
        seg = audio[ss:es]
        if len(seg) < frame_size:
            continue
        # Vectorised frame RMS
        n_frames = max(0, (len(seg) - frame_size) // hop_size + 1)
        if n_frames <= 0 or n_frames * hop_size > len(seg):
            continue
        frames = seg[:n_frames * hop_size].reshape(n_frames, -1)[:, :frame_size]
        frms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
        sil_rms_vals.extend(frms.tolist())

    if sil_rms_vals:
        sorted_sil = sorted(sil_rms_vals)
        # Use bottom 10% median as noise floor — avoids circular pollution
        # where loud mislabeled silences inflate the median
        noise_floor = float(np.median(np.array(sorted_sil[:max(1, int(len(sorted_sil) * 0.1))], dtype=np.float64)))
    else:
        # Fallback: use bottom 60% of all frames
        all_rms = frame_rms(audio, frame_size, hop_size)
        noise_floor = median(sorted(all_rms)[:max(1, int(len(all_rms) * 0.6))]) if all_rms else 1e-6
    if noise_floor <= 0:
        noise_floor = 1e-6

    # Step 2: average speech energy (for secondary comparison)
    speech_rms = []
    for p_iv in phones.intervals:
        if is_silence(p_iv.text) or p_iv.text == 'spn':
            continue
        ss = max(0, int(p_iv.xmin * sr))
        es = min(len(audio), int(p_iv.xmax * sr))
        seg = audio[ss:es]
        if len(seg) > 0:
            speech_rms.append(float(np.median(np.abs(seg))))
    avg_speech_e = sum(speech_rms) / len(speech_rms) if speech_rms else noise_floor

    # Build set of time ranges covered by actual words (non-silence, non-pause)
    word_ranges = []
    words_t = tier_by_name(textgrid, "words")
    for w_iv in (words_t.intervals if words_t else []):
        if not is_silence(w_iv.text) and w_iv.text not in ('<eps>','<pause>','[pause]'):
            word_ranges.append((w_iv.xmin, w_iv.xmax))

    def is_covered_by_word(xmin, xmax):
        for ws, we in word_ranges:
            if xmin >= ws - 0.01 and xmax <= we + 0.01:
                return True
        return False

    # Step 3: check each silence interval
    suspect_intervals = []
    for p_iv in phones.intervals:
        if not is_silence(p_iv.text) and p_iv.text != 'spn':
            continue
        # Skip spn intervals that cover actual words (OOV/alignment failure, not BGM)
        if p_iv.text == 'spn' and is_covered_by_word(p_iv.xmin, p_iv.xmax):
            continue
        if p_iv.duration < args.bgm_min_sil_dur:
            continue

        ss = max(0, int(p_iv.xmin * sr))
        es = min(len(audio), int(p_iv.xmax * sr))
        seg = audio[ss:es]
        if len(seg) == 0:
            continue
        mask = np.abs(seg) > 0
        sil_energy = float(np.mean(np.abs(seg[mask]))) if mask.any() else 0.0

        # Three conditions: above absolute floor, above noise floor, at speech level
        if (sil_energy > args.bgm_min_energy and
            sil_energy > noise_floor * args.bgm_noise_floor_ratio and
            sil_energy > avg_speech_e * args.bgm_speech_ratio):
            suspect_intervals.append({
                "xmin": round(p_iv.xmin, 4), "xmax": round(p_iv.xmax, 4),
                "duration": round(p_iv.duration, 4),
                "energy": round(sil_energy, 6),
                "noise_floor": round(noise_floor, 6),
            })

    # Step 4: file-level decision — any suspect interval triggers filter
    if not suspect_intervals:
        return []

    total_sil_dur = sum(p_iv.duration for p_iv in phones.intervals
                        if is_silence(p_iv.text) or p_iv.text == 'spn')
    suspect_dur = sum(s["duration"] for s in suspect_intervals)
    suspect_ratio = suspect_dur / total_sil_dur if total_sil_dur > 0 else 0

    return [{
            "rule": "bgm_suspect",
            "noise_floor": round(noise_floor, 6),
            "avg_speech_energy": round(avg_speech_e, 6),
            "suspect_intervals": len(suspect_intervals),
            "suspect_ratio": round(suspect_ratio, 3),
            "total_sil_dur": round(total_sil_dur, 3),
            "suspect_dur": round(suspect_dur, 3),
        "details": suspect_intervals[:10],
        }]


def _fallback_bgm_ctc_gap_selection(
        words_tier: Tier | None,
        source_words: list[dict] | None,
        ctc_tokens: list[dict] | None,
        correspondence: dict | None,
        audio_axis: tuple[float, float],
        reference_mode: str | None = "fallback") -> dict:
    """Select BGM scan spans from exact fallback CTC lexical gaps.

    The final words tier is the only candidate source.  In fallback mode a
    complete, digest-bound correspondence is required before a final silence
    can be narrowed to the gap between its exact adjacent CTC owners.  Any
    missing or unsafe proof deliberately returns the legacy full-silence span
    for every interval, so this helper cannot turn incomplete evidence into a
    new filter shortcut.
    """
    axis_start, axis_end = audio_axis
    try:
        axis_start, axis_end = float(axis_start), float(axis_end)
    except (TypeError, ValueError):
        axis_start, axis_end = 0.0, 0.0
    if not math.isfinite(axis_start) or not math.isfinite(axis_end):
        axis_start, axis_end = 0.0, 0.0
    axis_end = max(axis_start, axis_end)

    silences = []
    if words_tier is not None:
        silences = [(index, interval) for index, interval in
                    enumerate(words_tier.intervals)
                    if is_silence(interval.text)]

    validation = {
        "schema": FALLBACK_CORRESPONDENCE_SCHEMA,
        "status": "not_applicable" if reference_mode != "fallback" else "rejected",
        "reasons": (["reference_mode_not_fallback"]
                    if reference_mode != "fallback" else ["missing"]),
        "final_interval_reindexed": False,
        "final_surface_normalized": False,
        "digest": None,
    }
    can_narrow = False
    if reference_mode == "fallback":
        _valid, validation = _validate_fallback_correspondence(
            correspondence, source_words, ctc_tokens, words_tier)
        can_narrow = bool(_valid)

    lexical_positions = [
        index for index, interval in enumerate(words_tier.intervals)
    ] if words_tier is not None else []
    lexical_positions = [
        index for index in lexical_positions
        if (words_tier.intervals[index].text.strip()
            and not is_silence(words_tier.intervals[index].text)
            and not is_punct(words_tier.intervals[index].text))
    ] if words_tier is not None else []
    lexical_ordinal = {index: ordinal for ordinal, index in
                       enumerate(lexical_positions)}

    # Build owner lookup only from the validated ledger and exact CTC
    # ordinals.  No text or positional fallback is permitted here.
    ctc_by_ordinal = {}
    for index, token in enumerate(ctc_tokens or []):
        if not isinstance(token, dict) or token.get("type", "word") != "word":
            continue
        ordinal = token.get("ordinal", index)
        if type(ordinal) is int and ordinal == index:
            ctc_by_ordinal[ordinal] = token
    ctc_by_final: dict[int, dict] = {}
    source_by_final: dict[int, tuple[float, float]] = {}
    if can_narrow and isinstance(correspondence, dict):
        entries = correspondence.get("entries", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("status") != "mapped":
                    continue
                final_ordinal = entry.get("final_lexical_ordinal")
                ctc_ordinal = entry.get("ctc_ordinal")
                if (type(final_ordinal) is int and type(ctc_ordinal) is int
                        and ctc_ordinal in ctc_by_ordinal
                        and final_ordinal not in ctc_by_final):
                    ctc_by_final[final_ordinal] = ctc_by_ordinal[ctc_ordinal]
                    source_span = entry.get("source_span")
                    if (isinstance(source_span, (list, tuple))
                            and len(source_span) == 2):
                        try:
                            source_start = float(source_span[0])
                            source_end = float(source_span[1])
                        except (TypeError, ValueError):
                            source_start = source_end = math.nan
                        if (math.isfinite(source_start)
                                and math.isfinite(source_end)
                                and source_end > source_start):
                            source_by_final[final_ordinal] = (
                                source_start, source_end)
        if set(ctc_by_final) != set(range(len(lexical_positions))):
            can_narrow = False
            validation = dict(validation)
            validation["status"] = "rejected"
            validation["reasons"] = list(validation.get("reasons", []))
            validation["reasons"].append("owner_mapping_incomplete")
        elif set(source_by_final) != set(range(len(lexical_positions))):
            can_narrow = False
            validation = dict(validation)
            validation["status"] = "rejected"
            validation["reasons"] = list(validation.get("reasons", []))
            validation["reasons"].append("source_owner_mapping_incomplete")

    evaluated_intervals = []
    for index, interval in silences:
        original_start = float(interval.xmin)
        original_end = float(interval.xmax)
        original_duration = max(0.0, original_end - original_start)
        left_index = next((item for item in reversed(lexical_positions)
                           if item < index), None)
        right_index = next((item for item in lexical_positions if item > index), None)
        left_ordinal = lexical_ordinal.get(left_index)
        right_ordinal = lexical_ordinal.get(right_index)
        left_ctc = ctc_by_final.get(left_ordinal) if can_narrow else None
        right_ctc = ctc_by_final.get(right_ordinal) if can_narrow else None
        left_ctc_ordinal = (left_ctc.get("ordinal") if left_ctc is not None
                            else None)
        right_ctc_ordinal = (right_ctc.get("ordinal") if right_ctc is not None
                             else None)

        ctc_gap = None
        lexical_evidence_gap = None
        lexical_exclusions: list[list[float]] = []
        narrowing_basis = "legacy_full_final_silence"
        selection_mode = "legacy_full_final_silence"
        reason = "fallback_correspondence_invalid"
        if can_narrow and (left_ctc is not None or right_ctc is not None):
            try:
                left_end = (float(left_ctc["end_s"])
                            if left_ctc is not None else None)
                right_start = (float(right_ctc["start_s"])
                               if right_ctc is not None else None)
                gap_start = axis_start if left_end is None else left_end
                gap_end = axis_end if right_start is None else right_start
                if (not math.isfinite(gap_start) or not math.isfinite(gap_end)):
                    raise ValueError("non_finite_ctc_gap")
                ctc_gap = [round(gap_start, 6), round(gap_end, 6)]
                selection_mode = "ctc_gap_supported"
                reason = "exact_adjacent_ctc_gap"
                left_source = (source_by_final.get(left_ordinal)
                               if left_ordinal is not None else None)
                right_source = (source_by_final.get(right_ordinal)
                                if right_ordinal is not None else None)
                source_gap_start = (axis_start if left_source is None
                                    else left_source[1])
                source_gap_end = (axis_end if right_source is None
                                  else right_source[0])
                lexical_start = max(gap_start, source_gap_start)
                lexical_end = min(gap_end, source_gap_end)
                if (math.isfinite(lexical_start)
                        and math.isfinite(lexical_end)
                        and lexical_end > lexical_start):
                    lexical_evidence_gap = [round(lexical_start, 6),
                                            round(lexical_end, 6)]
                    narrowing_basis = "lexical_evidence_gap"
                    if gap_start < lexical_start - AXIS_EPS:
                        lexical_exclusions.append(
                            [round(gap_start, 6), round(lexical_start, 6)])
                    if lexical_end < gap_end - AXIS_EPS:
                        lexical_exclusions.append(
                            [round(lexical_end, 6), round(gap_end, 6)])
                else:
                    reason = "lexical_evidence_gap_invalid"
            except (KeyError, TypeError, ValueError, OverflowError):
                ctc_gap = None
                reason = "ctc_gap_evidence_malformed"
        elif can_narrow:
            reason = "adjacent_ctc_owner_missing"

        if selection_mode == "ctc_gap_supported":
            narrowing_gap = lexical_evidence_gap or ctc_gap
            evaluated_start = max(original_start, narrowing_gap[0], axis_start)
            evaluated_end = min(original_end, narrowing_gap[1], axis_end)
            if evaluated_end <= evaluated_start:
                evaluated_span = None
                evaluated_duration = 0.0
                reason = "ctc_gap_does_not_intersect_silence"
            else:
                evaluated_span = [round(evaluated_start, 6),
                                  round(evaluated_end, 6)]
                evaluated_duration = evaluated_end - evaluated_start
        else:
            # Unsafe or incomplete correspondence deliberately keeps the
            # historical full-final-silence scan.  The audio reader still
            # clips sample indices at the point of measurement, exactly as
            # the legacy path did.
            evaluated_span = [round(original_start, 6), round(original_end, 6)]
            evaluated_duration = original_duration

        evaluated_intervals.append({
            "index": index,
            "original_silence_span": [round(original_start, 6),
                                       round(original_end, 6)],
            "ctc_gap": ctc_gap,
            "lexical_evidence_gap": lexical_evidence_gap,
            "lexical_exclusions": lexical_exclusions,
            "narrowing_basis": narrowing_basis,
            "evaluated_intersection": evaluated_span,
            "left_owner_ordinal": left_ordinal,
            "right_owner_ordinal": right_ordinal,
            "left_final_lexical_ordinal": left_ordinal,
            "right_final_lexical_ordinal": right_ordinal,
            "left_ctc_ordinal": left_ctc_ordinal,
            "right_ctc_ordinal": right_ctc_ordinal,
            "excluded_duration": round(
                max(0.0, original_duration - evaluated_duration), 6),
            "evaluated_duration": round(evaluated_duration, 6),
            "selection_mode": selection_mode,
            "reason": reason,
        })

    return {
        "schema": "fallback-bgm-ctc-gap-selection-v1",
        "selection_mode": ("ctc_gap_supported" if can_narrow
                            else "legacy_full_final_silence"),
        "audio_axis": [round(axis_start, 6), round(axis_end, 6)],
        "validation": {
            "status": validation.get("status"),
            "digest": validation.get("digest"),
            "reasons": list(validation.get("reasons", [])),
        },
        "evaluated_intervals": evaluated_intervals,
    }


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def overlapping_intervals(tier: Tier, start: float, end: float, eps: float = 1e-4) -> list[Interval]:
    return [iv for iv in tier.intervals if iv.xmax > start + eps and iv.xmin < end - eps]


def overlap_duration(iv: Interval, start: float, end: float) -> float:
    return max(0.0, min(iv.xmax, end) - max(iv.xmin, start))


def detect_issues(textgrid: TextGrid, args, wav_path: Path | None = None,
                  audio: list[float] | None = None, sr: int = 16000) -> list[dict]:
    issues = []
    words = tier_by_name(textgrid, "words")
    phones = tier_by_name(textgrid, "pinyin_phones")
    if phones is None:
        phones = tier_by_name(textgrid, "phones")  # fallback
    if words is None or phones is None:
        return [{"rule": "missing_tier"}]

    if audio is None and wav_path is not None and wav_path.exists():
        audio, sr = load_audio(wav_path)
    word_energy_audit = _word_energy_audit(
        words, args, audio, sr, textgrid=textgrid)

    for idx, w in enumerate(words.intervals):
        if not w.text.strip() or is_silence(w.text):
            continue
        # English/NVV: MFA cannot model acoustically, energy & phone checks
        # are unreliable.  CTC boundaries are authoritative.
        _is_en_nvv = is_english_token(w.text) or is_nvv_token(w.text)
        ph = [p for p in overlapping_intervals(phones, w.xmin, w.xmax) if not is_silence(p.text)]
        if not ph:
            issues.append({"rule": "word_without_phone", "text": w.text})
            continue
        cov = sum(overlap_duration(p, w.xmin, w.xmax) for p in ph) / max(w.duration, 1e-6)
        ps = min(p.xmin for p in ph)
        pe = max(p.xmax for p in ph)
        sg = max(0.0, ps - w.xmin)
        eg = max(0.0, w.xmax - pe)
        if w.duration < args.filter_min_word_dur_sec:
            issues.append({"rule": "word_too_short", "text": w.text, "duration": round(w.duration, 4)})
        # Regr. Case 41: detect abnormally long words (> 3 s for Chinese,
        # > 8 s for English/NVV).  CTC anchor inflation (e.g. le5 = 5.6 s)
        # is caught by _snap_to_ctc's CTC_MAX_DUR guard; this check catches
        # any that slip through.
        _max_dur = 8.0 if (_is_en_nvv) else 3.0
        if w.duration > _max_dur:
            issues.append({"rule": "word_too_long", "text": w.text, "duration": round(w.duration, 4)})
        if w.duration >= args.filter_min_word_sec and cov < args.filter_min_phone_coverage:
            issues.append({"rule": "low_phone_coverage", "text": w.text, "coverage": round(cov, 3)})
        if sg > args.filter_edge_gap_sec or eg > args.filter_edge_gap_sec:
            issues.append({"rule": "large_edge_gap", "text": w.text})
        if w.duration > args.filter_long_word_sec:
            issues.append({"rule": "long_word", "text": w.text, "duration": round(w.duration, 3)})
        prev_w = words.intervals[idx - 1] if idx > 0 else None
        next_w = words.intervals[idx + 1] if idx + 1 < len(words.intervals) else None
        if (not _is_en_nvv and w.text.strip() and w.duration < 0.12
                and prev_w and is_silence(prev_w.text) and next_w and is_silence(next_w.text)
                and prev_w.duration >= args.filter_flank_silence_sec
                and next_w.duration >= args.filter_flank_silence_sec):
            issues.append({"rule": "short_word_between_silences", "text": w.text})
    if _word_energy_enabled(args):
        for item in word_energy_audit.get("items", []):
            if item.get("resulting_reason"):
                issues.append({
                    "rule": item["resulting_reason"],
                    "text": item.get("word"),
                    "energy": round(item["rms"]["premerge_rms"], 6),
                    "noise_floor": round(word_energy_audit["noise_model"]["noise_floor"], 6),
                    "threshold": round(word_energy_audit["noise_model"]["threshold"], 6),
                    "classification": item.get("classification"),
                })
    # ── Phone-level checks ──
    # Build time ranges for English / NVV word intervals so phone checks
    # can skip them — MFA cannot model these words and produces artifact
    # durations (e.g. "r" = 0.01 s) that are not real quality issues.
    en_nvv_ranges: list[tuple[float, float]] = []
    for w in words.intervals:
        if not w.text.strip() or is_silence(w.text):
            continue
        if is_english_token(w.text) or is_nvv_token(w.text):
            en_nvv_ranges.append((w.xmin, w.xmax))

    def _in_en_nvv_range(xmin: float, xmax: float) -> bool:
        for ws, we in en_nvv_ranges:
            if xmin >= ws - 0.005 and xmax <= we + 0.005:
                return True
        return False

    for pi, p in enumerate(phones.intervals):
        if not p.text.strip() or is_silence(p.text):
            continue
        # spn = MFA unknown phone — always inside English/NVV or OOV regions
        if p.text.strip() == 'spn':
            continue
        if _in_en_nvv_range(p.xmin, p.xmax):
            continue
        issues.extend(_phone_duration_qc_issues(
            p, pi + 1,
            filter_short_phone=args.filter_short_phone,
            short_phone_sec=args.filter_short_phone_sec,
            long_consonant_sec=args.filter_long_consonant_sec,
            long_vowel_sec=args.filter_long_vowel_sec,
        ))
    return issues


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _build_original_text_index(raw_text_dir: Path | None) -> dict[str, Path]:
    """Build a one-shot basename index for recursive reference lookup.

    Post-processing invokes :func:`find_original_text` once per TextGrid.  A
    recursive ``rglob`` for every stem turns that otherwise linear discovery
    step into repeated directory walks.  Keep the first path yielded for each
    basename, matching the legacy search's first-candidate behaviour while
    allowing O(1) lookups in workers.
    """
    if not raw_text_dir or not raw_text_dir.exists():
        return {}
    index: dict[str, Path] = {}
    try:
        for path in raw_text_dir.rglob("*.txt"):
            index.setdefault(path.name, path)
    except OSError:
        return {}
    return index


def _build_wav_index(wav_dir: Path | None) -> dict[str, Path]:
    """Build a deterministic one-shot basename index for WAV lookup."""
    if not wav_dir or not wav_dir.exists():
        return {}
    try:
        candidates = sorted(
            (path for path in wav_dir.rglob("*.wav") if path.is_file()),
            key=lambda path: (len(path.relative_to(wav_dir).parts),
                              path.relative_to(wav_dir).as_posix()),
        )
    except OSError:
        return {}
    index: dict[str, Path] = {}
    for path in candidates:
        index.setdefault(path.stem, path)
    return index


def _find_wav(stem: str, wav_dir: Path | None,
              wav_index: dict[str, Path] | None = None) -> Path:
    """Resolve a WAV using an optional index, with safe recursive fallback."""
    if not wav_dir:
        return Path(f"{stem}.wav")
    if wav_index is not None:
        indexed = wav_index.get(stem)
        if indexed is not None and indexed.is_file():
            return indexed

    # Preserve the legacy precedence: an exact top-level path wins before a
    # recursive candidate.  This also handles an indexed path disappearing.
    top_level = wav_dir / f"{stem}.wav"
    if top_level.is_file():
        return top_level
    try:
        candidates = sorted(
            (path for path in wav_dir.rglob(f"{stem}.wav") if path.is_file()),
            key=lambda path: path.relative_to(wav_dir).as_posix(),
        )
    except OSError:
        return top_level
    return candidates[0] if candidates else top_level


def find_original_text(stem: str, raw_text_dir: Path | None,
                       text_index: dict[str, Path] | None = None) -> str:
    """Find the original Chinese text for a given output stem (searches recursively)."""
    if not raw_text_dir or not raw_text_dir.exists():
        return ""
    if text_index is not None:
        names = [f"{stem}.txt", f"{stem}_ref.txt"]
        names.extend(f"{stem}{suffix}.txt"
                     for suffix in ("_qwen3-api", "_qwen3", "_firered"))
        m = re.search(r"_(firered|qwen3|qwen3-api)$", stem)
        if m:
            base = stem[:m.start()]
            names.extend([f"{base}.txt", f"{base}_ref.txt"])
            names.extend(f"{base}{suffix}.txt"
                         for suffix in ("_qwen3-api", "_qwen3", "_firered"))
        for name in names:
            path = text_index.get(name)
            if path is not None:
                try:
                    return path.read_text(encoding="utf-8").strip()
                except OSError:
                    # Preserve legacy behaviour if a source disappears after
                    # indexing: retry the original recursive search rather
                    # than silently changing the reference fallback result.
                    return find_original_text(stem, raw_text_dir, None)
        return ""
    # Prefer the exact source transcript.  ``*_ref.txt`` is emitted by
    # ctc_prealign/step_link_ctc when the original text lives outside the
    # CTC directory; it is authoritative and must precede ASR fallbacks.
    for pattern in (f"{stem}.txt", f"{stem}_ref.txt"):
        candidates = list(raw_text_dir.rglob(pattern))
        if candidates:
            return candidates[0].read_text(encoding="utf-8").strip()
    # Try with engine suffix appended
    for suffix in ("_qwen3-api", "_qwen3", "_firered"):
        candidates = list(raw_text_dir.rglob(f"{stem}{suffix}.txt"))
        if candidates:
            return candidates[0].read_text(encoding="utf-8").strip()
    # Try stripping suffix from stem and re-adding
    m = re.search(r"_(firered|qwen3|qwen3-api)$", stem)
    if m:
        base = stem[:m.start()]
        for pattern in (f"{base}.txt", f"{base}_ref.txt"):
            candidates = list(raw_text_dir.rglob(pattern))
            if candidates:
                return candidates[0].read_text(encoding="utf-8").strip()
        for suffix in ("_qwen3-api", "_qwen3", "_firered"):
            candidates = list(raw_text_dir.rglob(f"{base}{suffix}.txt"))
            if candidates:
                return candidates[0].read_text(encoding="utf-8").strip()
    return ""


def find_original_text_path(stem: str, raw_text_dir: Path | None,
                            text_index: dict[str, Path] | None = None) -> Path | None:
    """Return the authority source path without normalizing its bytes."""
    if not raw_text_dir or not raw_text_dir.exists():
        return None
    names = [f"{stem}.txt", f"{stem}_ref.txt"]
    names.extend(f"{stem}{suffix}.txt"
                 for suffix in ("_qwen3-api", "_qwen3", "_firered"))
    match = re.search(r"_(firered|qwen3|qwen3-api)$", stem)
    if match:
        base = stem[:match.start()]
        names.extend([f"{base}.txt", f"{base}_ref.txt"])
        names.extend(f"{base}{suffix}.txt"
                     for suffix in ("_qwen3-api", "_qwen3", "_firered"))
    if text_index is not None:
        for name in names:
            path = text_index.get(name)
            if path is not None and path.is_file():
                return path
        return None
    for name in names:
        candidates = sorted(path for path in raw_text_dir.rglob(name)
                            if path.is_file())
        if candidates:
            return candidates[0]
    return None


def _inject_fallback_punctuation_gaps(
        words_tier: Tier, pp_tier: Tier | None,
        punct_entries: list[dict], *,
        source_surface_ledger: dict | None = None,
        ctc_tokens: list[dict] | None = None,
        punctuation_projection: dict | None = None
        ) -> tuple[Tier, Tier | None]:
    """Project fallback punctuation onto one exact ordinal-bound owner.

    Source text supplies only the label.  Geometry must already exist as an
    interval, as explicit silence, or as a positive CTC lexical gap.  The
    latter is projected over the complete raw gap, never over a CTC lexical
    span and never over an inferred/carved word remainder.
    """
    if not punct_entries and source_surface_ledger is None:
        return words_tier, pp_tier
    lexical = [iv for iv in words_tier.intervals
               if iv.text.strip() and not is_silence(iv.text)
               and not is_punct(iv.text)]
    source_entries: dict[int, dict] = {}
    source_validation = {"status": "not_provided", "reasons": []}
    projection_validation = {"status": "not_provided", "reasons": []}
    projection_active = False
    if source_surface_ledger is not None:
        source_valid, source_validation = _validate_fallback_punctuation_surface_ledger(
            source_surface_ledger)
        if source_valid:
            for item in source_surface_ledger.get("punctuation", []):
                boundary = item.get("lexical_boundary")
                if type(boundary) is not int or boundary in source_entries:
                    source_validation = {
                        "schema": FALLBACK_SURFACE_SCHEMA,
                        "status": "rejected",
                        "reasons": ["duplicate_or_malformed_boundary"],
                    }
                    source_valid = False
                    break
                source_entries[boundary] = item
        if not source_valid:
            source_entries = {}
        elif source_surface_ledger.get("lexical_count") != len(lexical):
            source_validation = dict(source_validation)
            source_validation["status"] = "rejected"
            source_validation["reasons"] = list(
                source_validation.get("reasons", []))
            source_validation["reasons"].append(
                "source_final_lexical_count_mismatch")
            source_entries = {}

            projection_valid, projection_validation = (
                _validate_fallback_punctuation_projection(
                    punctuation_projection,
                    source_surface_ledger.get("source_text"),
                    words_tier, ctc_tokens))
            if projection_valid:
                projection_active = True
                for item in punctuation_projection.get("entries", []):
                    boundary = item.get("final_boundary")
                    label = str(item.get("label", "")).strip()
                    if (type(boundary) is int and is_punct(label)
                            and boundary not in source_entries):
                        source_entries[boundary] = {
                            "source_index": item.get("source_index"),
                            "source_boundary": item.get("source_boundary"),
                            "lexical_boundary": boundary,
                            "label": label,
                            "projected": True,
                        }

    # Use lexical ordinal rather than surface text.  Repeated words and NVV
    # labels must never select a neighboring owner by string coincidence.
    ctc_by_ordinal: dict[int, tuple[float, float]] = {}
    ctc_identity_by_ordinal: dict[int, str] = {}
    ctc_errors: list[str] = []
    for token in ctc_tokens or []:
        if not isinstance(token, dict) or token.get("type", "word") != "word":
            continue
        if not isinstance(token.get("word"), str) or not token["word"].strip():
            ctc_errors.append("ctc_word_malformed")
            continue
        ordinal = len(ctc_by_ordinal)
        try:
            start = float(token["start_s"])
            end = float(token["end_s"])
        except (KeyError, TypeError, ValueError):
            ctc_errors.append(f"ctc_span_malformed:{ordinal}")
            continue
        if (not math.isfinite(start) or not math.isfinite(end)
                or end <= start):
            ctc_errors.append(f"ctc_span_invalid:{ordinal}")
            continue
        explicit = token.get("lexical_ordinal")
        if explicit is not None:
            if type(explicit) is not int or explicit != ordinal:
                ctc_errors.append("ctc_ordinal_ambiguous")
                continue
        ctc_by_ordinal[ordinal] = (start, end)
        ctc_identity_by_ordinal[ordinal] = _lexical_identity(token["word"])

    owners: list[tuple[float, float, str, dict, tuple[int | None, int | None]]] = []
    rejected: list[dict] = []
    used_boundaries: set[tuple[int | None, int | None]] = set()

    def reject(entry: object, reason: str) -> None:
        rejected.append({"candidate_id": entry.get("candidate_id")
                         if isinstance(entry, dict) else None,
                         "reason": reason})

    def boundary_intervals(left_ordinal: int | None,
                           right_ordinal: int | None) -> list[tuple[int, Interval]]:
        left_index = (-1 if left_ordinal is None else
                      next((index for index, iv in enumerate(words_tier.intervals)
                            if iv is lexical[left_ordinal]), -1))
        right_index = (len(words_tier.intervals) if right_ordinal is None else
                       next((index for index, iv in enumerate(words_tier.intervals)
                             if iv is lexical[right_ordinal]), len(words_tier.intervals)))
        return [(index, iv) for index, iv in enumerate(words_tier.intervals)
                if left_index < index < right_index]

    def candidate_key(entry: dict) -> tuple[int | None, int | None] | None:
        left = entry.get("left_lexical_ordinal")
        right = entry.get("right_lexical_ordinal")
        if type(left) is not int and left is not None:
            return None
        if type(right) is not int and right is not None:
            return None
        return left, right

    def ctc_gap_for_key(
            left_ordinal: int | None,
            right_ordinal: int | None,
    ) -> tuple[tuple[float, float] | None, str | None]:
        """Return one exact raw CTC gap or a stable fail-closed reason."""
        if ctc_tokens is None:
            return None, "positive_ctc_gap_evidence_missing"
        if ctc_errors:
            return None, ctc_errors[0]
        if len(ctc_by_ordinal) != len(lexical):
            return None, "ctc_final_lexical_count_mismatch"
        left_span = (ctc_by_ordinal.get(left_ordinal)
                     if left_ordinal is not None else None)
        right_span = (ctc_by_ordinal.get(right_ordinal)
                      if right_ordinal is not None else None)
        gap_start = left_span[1] if left_span is not None else words_tier.xmin
        gap_end = right_span[0] if right_span is not None else words_tier.xmax
        if not (math.isfinite(gap_start) and math.isfinite(gap_end)):
            return None, "ctc_gap_non_finite"
        if gap_end <= gap_start:
            return None, "punctuation_gap_zero_width"
        if (gap_start < words_tier.xmin - AXIS_EPS
                or gap_end > words_tier.xmax + AXIS_EPS):
            return None, "ctc_gap_out_of_axis"
        if any(other_start < gap_end - AXIS_EPS
               and other_end > gap_start + AXIS_EPS
               for ordinal, (other_start, other_end) in ctc_by_ordinal.items()
               if ordinal not in {left_ordinal, right_ordinal}):
            return None, "ctc_gap_intersects_lexical_span"
        return (gap_start, gap_end), None

    projected_by_key = {}
    if projection_active and isinstance(punctuation_projection, dict):
        for item in punctuation_projection.get("entries", []):
            if not isinstance(item, dict):
                continue
            key = (item.get("left_lexical_ordinal"),
                   item.get("right_lexical_ordinal"))
            projected_by_key[key] = item

    def adjacent_nvv_frame_owner(
            left_ordinal: int | None,
            right_ordinal: int | None,
    ) -> tuple[tuple[float, float, dict] | None, str | None]:
        """Allocate one display frame only at an exact source NVV boundary.

        A source punctuation mark has no acoustic duration of its own.  When
        CTC puts an NVV and its next lexical token on the same frame edge, the
        otherwise exact source mark would have a zero-width TextGrid owner.
        The only safe lexical span from which to allocate a display marker is
        the adjacent non-verbal event: ordinary spoken words remain
        uncarvable.  The allocation is one CTC frame (60 ms), keeps a positive
        NVV remainder, and is recorded separately from a raw punctuation
        anchor so the publication audit can distinguish the two cases.
        """
        if ctc_tokens is None:
            return None, "positive_ctc_gap_evidence_missing"
        if ctc_errors:
            return None, ctc_errors[0]
        if len(ctc_by_ordinal) != len(lexical):
            return None, "ctc_final_lexical_count_mismatch"

        candidates: list[tuple[str, int]] = []
        if (left_ordinal is not None
                and is_nvv_token(lexical[left_ordinal].text.strip())):
            candidates.append(("left", left_ordinal))
        if (right_ordinal is not None
                and is_nvv_token(lexical[right_ordinal].text.strip())):
            candidates.append(("right", right_ordinal))
        if not candidates:
            return None, "punctuation_gap_zero_width"
        if len(candidates) != 1:
            return None, "zero_width_nvv_owner_ambiguous"

        side, ordinal = candidates[0]
        interval = lexical[ordinal]
        ctc_start, ctc_end = ctc_by_ordinal[ordinal]
        left_span = (ctc_by_ordinal.get(left_ordinal)
                     if left_ordinal is not None else None)
        right_span = (ctc_by_ordinal.get(right_ordinal)
                      if right_ordinal is not None else None)
        edge_left = left_span[1] if left_span is not None else words_tier.xmin
        edge_right = right_span[0] if right_span is not None else words_tier.xmax
        if abs(edge_right - edge_left) > AXIS_EPS:
            return None, "nvv_frame_owner_requires_zero_ctc_gap"

        frame_s = 0.060
        min_nvv_remainder_s = max(AXIS_EPS * 2.0,
                                  _EVIDENCE_REPAIR_FLOOR_S)
        if side == "left":
            owner_end = min(interval.xmax, ctc_end, edge_left)
            available = owner_end - max(interval.xmin, ctc_start)
            owner_width = min(frame_s, available - min_nvv_remainder_s)
            owner_start = owner_end - owner_width
        else:
            owner_start = max(interval.xmin, ctc_start, edge_right)
            available = min(interval.xmax, ctc_end) - owner_start
            owner_width = min(frame_s, available - min_nvv_remainder_s)
            owner_end = owner_start + owner_width
        if (owner_width <= AXIS_EPS
                or owner_end <= owner_start + AXIS_EPS):
            return None, "zero_width_nvv_owner_too_short"
        return (owner_start, owner_end, {
            "source": "fallback_surface_adjacent_nvv_frame",
            "nvv_side": side,
            "nvv_lexical_ordinal": ordinal,
            "supporting_ctc_nvv_span": [ctc_start, ctc_end],
            "allocation_width_s": owner_width,
        }), None

    for entry in sorted(punct_entries, key=lambda row: (
            float(row.get("start_s", math.inf))
            if isinstance(row, dict) else math.inf)):
        if not isinstance(entry, dict):
            reject(entry, "malformed_candidate")
            continue
        if entry.get("schema") != PUNCTUATION_EVIDENCE_SCHEMA:
            reject(entry, "punctuation_evidence_schema_mismatch")
            continue
        key = candidate_key(entry)
        if key is None:
            reject(entry, "candidate_neighbor_ordinal_malformed")
            continue
        left_ordinal, right_ordinal = key
        if left_ordinal is None and right_ordinal != 0:
            reject(entry, "nonleading_left_neighbor_missing")
            continue
        if right_ordinal is None and left_ordinal != len(lexical) - 1:
            reject(entry, "nonterminal_right_neighbor_missing")
            continue
        if (left_ordinal is not None and right_ordinal is not None
                and right_ordinal != left_ordinal + 1):
            reject(entry, "neighbors_not_adjacent")
            continue
        if key in used_boundaries:
            reject(entry, "duplicate_neighbor_boundary")
            continue
        try:
            raw_start = float(entry.get("raw_start_s", entry["start_s"]))
            raw_end = float(entry.get("raw_end_s", entry["end_s"]))
        except (KeyError, TypeError, ValueError):
            reject(entry, "candidate_span_or_neighbor_missing")
            continue
        if (not is_punct(str(entry.get("word", "")).strip())
                or not math.isfinite(raw_start)
                or not math.isfinite(raw_end) or raw_end <= raw_start
                or raw_start < words_tier.xmin - AXIS_EPS
                or raw_end > words_tier.xmax + AXIS_EPS):
            reject(entry, "candidate_span_invalid")
            continue
        if left_ordinal is not None and not (0 <= left_ordinal < len(lexical)):
            reject(entry, "left_neighbor_out_of_range")
            continue
        if right_ordinal is not None and not (0 <= right_ordinal < len(lexical)):
            reject(entry, "right_neighbor_out_of_range")
            continue
        source_item = source_entries.get(right_ordinal if left_ordinal is None
                                         else left_ordinal + 1)
        if source_surface_ledger is not None and source_item is None:
            reject(entry, "source_punctuation_boundary_missing")
            continue
        label = (str(source_item.get("label", "")).strip()
                 if source_item is not None else
                 str(entry.get("word", "")).strip())
        if not is_punct(label):
            reject(entry, "source_punctuation_label_invalid")
            continue
        if projection_active:
            projected = projected_by_key.get(key)
            if projected is None:
                reject(entry, "projected_boundary_missing")
                continue
            if str(entry.get("word", "")).strip() != str(
                    projected.get("label", "")).strip():
                reject(entry, "projected_label_mismatch")
                continue

        existing = [(index, iv) for index, iv in boundary_intervals(
            left_ordinal, right_ordinal) if is_punct(iv.text)]
        if len(existing) > 1:
            reject(entry, "ambiguous_existing_punctuation_owner")
            continue
        if existing:
            index, interval = existing[0]
            owners.append((interval.xmin, interval.xmax, label, entry, key))
            used_boundaries.add(key)
            continue

        explicit_silence = [(index, iv) for index, iv in boundary_intervals(
            left_ordinal, right_ordinal) if is_silence(iv.text)]
        if len(explicit_silence) > 1:
            reject(entry, "ambiguous_explicit_silence_owner")
            continue
        if explicit_silence:
            if projection_active:
                try:
                    raw_start = float(entry["raw_start_s"])
                    raw_end = float(entry["raw_end_s"])
                except (KeyError, TypeError, ValueError):
                    reject(entry, "candidate_span_or_neighbor_missing")
                    continue
                if raw_end <= raw_start or not any(
                        raw_end > silence.xmin + AXIS_EPS
                        and raw_start < silence.xmax - AXIS_EPS
                        for _, silence in explicit_silence):
                    reject(entry, "candidate_does_not_overlap_silence")
                    continue
            interval = explicit_silence[0][1]
            owners.append((interval.xmin, interval.xmax, label, entry, key))
            used_boundaries.add(key)
            continue

        gap, gap_reason = ctc_gap_for_key(left_ordinal, right_ordinal)
        if gap is None:
            reject(entry, str(gap_reason))
            continue
        gap_start, gap_end = gap
        if raw_end <= gap_start or raw_start >= gap_end:
            reject(entry, "candidate_does_not_overlap_gap")
            continue
        owners.append((gap_start, gap_end, label, entry, key))
        used_boundaries.add(key)

    # An existing punctuation or explicit silence is an owner in its own
    # right.  It must be relabelable from the source surface even when the CTC
    # punctuation sidecar omitted that occurrence.  A missing sidecar cannot,
    # however, fabricate a positive-gap owner.
    for boundary, source_item in sorted(source_entries.items()):
        key = (boundary - 1 if boundary > 0 else None,
               boundary if boundary < len(lexical) else None)
        if key in used_boundaries:
            continue
        existing = [(index, iv) for index, iv in boundary_intervals(
            key[0], key[1]) if is_punct(iv.text)]
        if len(existing) > 1:
            reject(source_item, "ambiguous_existing_punctuation_owner")
            continue
        if existing:
            interval = existing[0][1]
            owners.append((interval.xmin, interval.xmax,
                           str(source_item["label"]),
                           {"candidate_id": None, "source": "fallback_surface",
                            "raw_start_s": interval.xmin,
                            "raw_end_s": interval.xmax}, key))
            used_boundaries.add(key)
            continue
        explicit_silence = [(index, iv) for index, iv in boundary_intervals(
            key[0], key[1]) if is_silence(iv.text)]
        if len(explicit_silence) > 1:
            reject(source_item, "ambiguous_explicit_silence_owner")
            continue
        if explicit_silence:
            interval = explicit_silence[0][1]
            owners.append((interval.xmin, interval.xmax,
                           str(source_item["label"]),
                           {"candidate_id": None, "source": "fallback_surface",
                            "raw_start_s": interval.xmin,
                            "raw_end_s": interval.xmax}, key))
            used_boundaries.add(key)
        else:
            gap, gap_reason = ctc_gap_for_key(key[0], key[1])
            if gap is None:
                if gap_reason != "punctuation_gap_zero_width":
                    reject(source_item, str(gap_reason))
                    continue
                nvv_owner, nvv_reason = adjacent_nvv_frame_owner(
                    key[0], key[1])
                if nvv_owner is None:
                    reject(source_item, str(nvv_reason))
                    continue
                gap_start, gap_end, nvv_evidence = nvv_owner
                owner_entry = {
                    "candidate_id": None,
                    "raw_start_s": gap_start,
                    "raw_end_s": gap_end,
                    **nvv_evidence,
                }
            else:
                gap_start, gap_end = gap
                owner_entry = {
                    "candidate_id": None,
                    "source": "fallback_surface_ctc_gap",
                    "raw_start_s": gap_start,
                    "raw_end_s": gap_end,
                }
            owners.append((gap_start, gap_end, str(source_item["label"]),
                           owner_entry, key))
            used_boundaries.add(key)

    if not owners:
        if rejected:
            words_tier._punctuation_evidence_ledger = {
                "schema": PUNCTUATION_EVIDENCE_SCHEMA,
                "status": "rejected", "rejected": rejected,
                "owners": [], "source_validation": source_validation,
                "projection_validation": projection_validation}
        return words_tier, pp_tier

    # Subtract each trusted display owner from existing intervals and insert
    # one punctuation interval for the complete gap.  No lexical interval is
    # enlarged or re-ordered by this operation.
    result: list[Interval] = []
    edge_repairs: list[dict] = []
    lexical_ordinals = {id(interval): ordinal
                        for ordinal, interval in enumerate(lexical)}
    trimmed_lexical_spans: dict[int, tuple[float, float]] = {}
    for ordinal, interval in enumerate(lexical):
        trimmed_start = interval.xmin
        trimmed_end = interval.xmax
        for start, end, _label, _entry, key in owners:
            overlaps = (end > interval.xmin + AXIS_EPS
                        and start < interval.xmax - AXIS_EPS)
            if key[0] == ordinal and overlaps:
                trimmed_end = min(trimmed_end, start)
            if key[1] == ordinal and overlaps:
                trimmed_start = max(trimmed_start, end)
        trimmed_lexical_spans[ordinal] = (trimmed_start, trimmed_end)

    def lexical_edge_has_blocker(ordinal: int, start: float, end: float) -> bool:
        """Return whether a non-empty peer already owns an edge-completion gap."""
        if end <= start + AXIS_EPS:
            return False
        peer_blocks = any(
            other_ordinal != ordinal
            and other_end > other_start + AXIS_EPS
            and other_end > start + AXIS_EPS
            and other_start < end - AXIS_EPS
            for other_ordinal, (other_start, other_end)
            in trimmed_lexical_spans.items()
        )
        owner_blocks = any(
            owner_end > start + AXIS_EPS
            and owner_start < end - AXIS_EPS
            for owner_start, owner_end, _label, _entry, _key in owners
        )
        return peer_blocks or owner_blocks

    for interval in words_tier.intervals:
        lexical_ordinal = lexical_ordinals.get(id(interval))
        if lexical_ordinal is not None:
            # A boundary owner may trim the tail of its left lexical item or
            # the head of its right lexical item.  It must never punch a hole
            # through a lexical interval and preserve both fragments: that
            # duplicated NVV labels and changed lexical ordinals in r4.
            kept_start, kept_end = trimmed_lexical_spans[lexical_ordinal]
            for start, end, _label, _entry, key in owners:
                overlaps = (end > interval.xmin + AXIS_EPS
                            and start < interval.xmax - AXIS_EPS)
                if (key[0] == lexical_ordinal and not overlaps
                      and not ctc_errors
                      and len(ctc_by_ordinal) == len(lexical)
                      and _lexical_identity(interval.text)
                      == ctc_identity_by_ordinal.get(lexical_ordinal)
                      and source_validation.get("status") == "verified"
                      and kept_end < start - AXIS_EPS
                      and abs(ctc_by_ordinal[lexical_ordinal][1] - start)
                      <= AXIS_EPS
                      and not lexical_edge_has_blocker(
                          lexical_ordinal, kept_end, start)):
                    # A prior NVV/punctuation geometry pass can shorten an
                    # otherwise exact lexical edge and leave a narrow ownerless
                    # hole (LAria_00053: zi5 ended one 60 ms frame before its
                    # sealed comma).  Complete only to that word's exact CTC
                    # edge, and only when no peer interval owns the gap.
                    old_end = kept_end
                    kept_end = start
                    edge_repairs.append({
                        "source": "fallback_surface_ctc_lexical_edge_completion",
                        "side": "right",
                        "lexical_ordinal": lexical_ordinal,
                        "old_span": [kept_start, old_end],
                        "new_span": [kept_start, kept_end],
                        "supporting_ctc_span": list(
                            ctc_by_ordinal[lexical_ordinal]),
                    })
                if (key[1] == lexical_ordinal and not overlaps
                      and not ctc_errors
                      and len(ctc_by_ordinal) == len(lexical)
                      and _lexical_identity(interval.text)
                      == ctc_identity_by_ordinal.get(lexical_ordinal)
                      and source_validation.get("status") == "verified"
                      and kept_start > end + AXIS_EPS
                      and abs(ctc_by_ordinal[lexical_ordinal][0] - end)
                      <= AXIS_EPS
                      and not lexical_edge_has_blocker(
                          lexical_ordinal, end, kept_start)):
                    old_start = kept_start
                    kept_start = end
                    edge_repairs.append({
                        "source": "fallback_surface_ctc_lexical_edge_completion",
                        "side": "left",
                        "lexical_ordinal": lexical_ordinal,
                        "old_span": [old_start, kept_end],
                        "new_span": [kept_start, kept_end],
                        "supporting_ctc_span": list(
                            ctc_by_ordinal[lexical_ordinal]),
                    })
            if kept_end > kept_start + AXIS_EPS:
                result.append(Interval(kept_start, kept_end, interval.text))
            continue
        pieces = [(interval.xmin, interval.xmax)]
        for start, end, _label, _entry, _key in owners:
            next_pieces = []
            for piece_start, piece_end in pieces:
                if piece_end <= start + AXIS_EPS or piece_start >= end - AXIS_EPS:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < start - AXIS_EPS:
                    next_pieces.append((piece_start, start))
                if piece_end > end + AXIS_EPS:
                    next_pieces.append((end, piece_end))
            pieces = next_pieces
        result.extend(Interval(start, end, interval.text)
                      for start, end in pieces if end > start + AXIS_EPS)
    result.extend(Interval(start, end, label)
                  for start, end, label, _entry, _key in owners)
    result.sort(key=lambda iv: (iv.xmin, iv.xmax))
    rebuilt = _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, result))
    ledger = {
        "schema": PUNCTUATION_EVIDENCE_SCHEMA,
        "status": "verified" if not rejected else "partial",
        "edge_repairs": edge_repairs,
        "owners": [{"candidate_id": entry.get("candidate_id"),
                    "source": entry.get("source", "ctc"),
                    "label": label,
                    "raw_span": [raw_start, raw_end],
                    "display_span": [start, end],
                    "left_lexical_ordinal": key[0],
                    "right_lexical_ordinal": key[1],
                    **({"nvv_side": entry["nvv_side"],
                        "nvv_lexical_ordinal": entry["nvv_lexical_ordinal"],
                        "supporting_ctc_nvv_span": entry[
                            "supporting_ctc_nvv_span"],
                        "allocation_width_s": entry["allocation_width_s"]}
                       if entry.get("source") ==
                       "fallback_surface_adjacent_nvv_frame" else {})}
                   for start, end, label, entry, key in owners
                   for raw_start, raw_end in [(
                       float(entry.get("raw_start_s", entry.get("start_s"))),
                       float(entry.get("raw_end_s", entry.get("end_s"))))]] ,
        "rejected": rejected,
        "source_validation": source_validation,
        "projection_validation": projection_validation,
    }
    ledger["digest"] = _evidence_digest(ledger)
    rebuilt._punctuation_evidence_ledger = ledger
    if pp_tier is not None:
        pp_result = [Interval(iv.xmin, iv.xmax, iv.text)
                     for iv in pp_tier.intervals]
        for start, end, label, _entry, _key in owners:
            pp_result.append(Interval(start, end, label))
        pp_result.sort(key=lambda iv: (iv.xmin, iv.xmax))
        pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, pp_result)
    return rebuilt, pp_tier


def _inject_punctuation(words_tier: Tier, pp_tier: Tier | None,
                         punct_entries: list[dict],
                         reference_text: str = "",
                         reference_authoritative: bool = False,
                         *, source_surface_ledger: dict | None = None,
                         ctc_tokens: list[dict] | None = None,
                         punctuation_projection: dict | None = None
                         ) -> tuple[Tier, Tier | None]:
    """Inject only reference-confirmed punctuation in a local word gap.

    CTC punctuation is evidence about timing, not lexical authority.  With
    no authoritative reference sequence this function is intentionally a
    no-op: CTC-only punctuation, including terminal punctuation, must not
    consume any lexical or tail time.  Confirmed punctuation is bound to the
    reference lexical boundary first; its CTC interval is then clipped to
    that local owner range.  An anchor can never extend over unrelated audio.
    """
    from dataclasses import replace as _replace

    if not punct_entries and source_surface_ledger is None:
        return words_tier, pp_tier
    if not reference_authoritative:
        return _inject_fallback_punctuation_gaps(words_tier, pp_tier,
                                                 punct_entries,
                                                 source_surface_ledger=source_surface_ledger,
                                                 ctc_tokens=ctc_tokens,
                                                 punctuation_projection=punctuation_projection)
    if not reference_text:
        return words_tier, pp_tier

    allowed = '，。…！？、；：,.!?;:～'
    reference_puncts: list[tuple[str, int]] = []
    reference_lexical_count = 0
    for unit in _extract_word_chars(reference_text):
        if is_punct(unit) and unit.strip() in allowed:
            reference_puncts.append((unit.strip(), reference_lexical_count))
        elif is_word_like(unit):
            reference_lexical_count += 1
    lexical = [iv for iv in words_tier.intervals
               if iv.text.strip() and not is_silence(iv.text)
               and not is_punct(iv.text)]
    if reference_lexical_count != len(lexical):
        # Exact semantic binding is unavailable (for example an English
        # surface is still split), so defer to the later authority restore.
        return words_tier, pp_tier

    # Match CTC punctuation to the authoritative semantic sequence in time
    # order.  This permits a missing CTC mark but rejects extra/reordered
    # marks; no timestamp-only occurrence is accepted.
    def _punct_entry_time(pair: tuple[int, dict]) -> float:
        if not isinstance(pair[1], dict):
            return math.inf
        try:
            value = float(pair[1].get("start_s", 0.0))
        except (TypeError, ValueError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    entries_by_time = sorted(enumerate(punct_entries), key=_punct_entry_time)
    matched: list[tuple[dict, int]] = []
    reference_cursor = 0
    for _, entry in entries_by_time:
        if not isinstance(entry, dict):
            continue
        char = str(entry.get("word", "")).strip()
        if char not in allowed:
            continue
        match = None
        for ref_index in range(reference_cursor, len(reference_puncts)):
            if reference_puncts[ref_index][0] == char:
                match = ref_index
                break
        if match is None:
            continue
        reference_cursor = match + 1
        boundary = reference_puncts[match][1]
        prev = lexical[boundary - 1] if boundary > 0 else None
        nxt = lexical[boundary] if boundary < len(lexical) else None
        try:
            anchor_start = float(entry["start_s"])
            anchor_end = float(entry["end_s"])
        except (KeyError, TypeError, ValueError):
            # Semantic authority alone is not timing ownership.  Without a
            # finite source occurrence, fail closed instead of guessing the
            # whole local gap.
            continue
        left_bound = prev.xmin if prev is not None else words_tier.xmin
        right_bound = nxt.xmax if nxt is not None else words_tier.xmax
        if (not math.isfinite(anchor_start) or not math.isfinite(anchor_end)
                or anchor_end <= anchor_start
                or anchor_start < left_bound - AXIS_EPS
                or anchor_end > right_bound + AXIS_EPS):
            # An authoritative punctuation character with no local source
            # span is retained as a missing/filtered occurrence; it may not
            # consume unrelated audio or an edge remainder.
            continue
        # A confirmed mark may replace a pause, not voiced lexical audio.
        # Require an explicit silence interval overlapping its CTC anchor
        # before allowing the resolver to clip either adjacent word.
        local_silences = [iv for iv in words_tier.intervals
                          if is_silence(iv.text) and iv.text.strip()
                          and iv.xmax > anchor_start + AXIS_EPS
                          and iv.xmin < anchor_end - AXIS_EPS]
        if not local_silences:
            continue
        support = max(
            local_silences,
            key=lambda iv: min(iv.xmax, anchor_end)
            - max(iv.xmin, anchor_start))
        anchor_start = max(anchor_start, support.xmin)
        anchor_end = min(anchor_end, support.xmax)
        if anchor_end <= anchor_start + AXIS_EPS:
            continue
        has_local_silence = any(
            is_silence(iv.text) and iv.text.strip()
            and iv.xmax >= anchor_start - AXIS_EPS
            and iv.xmin <= anchor_end + AXIS_EPS
            for iv in words_tier.intervals)
        if not has_local_silence:
            continue
        left_end = min(prev.xmax, anchor_start) if prev is not None else anchor_start
        right_start = max(nxt.xmin, anchor_end) if nxt is not None else anchor_end
        if ((prev is not None and left_end <= prev.xmin + AXIS_EPS)
                or (nxt is not None and right_start >= nxt.xmax - AXIS_EPS)):
            continue
        nonadjacent_overlap = any(
            index not in {boundary - 1, boundary}
            and iv.xmax > anchor_start + AXIS_EPS
            and iv.xmin < anchor_end - AXIS_EPS
            for index, iv in enumerate(lexical))
        if nonadjacent_overlap:
            continue
        bound = dict(entry)
        bound["start_s"] = anchor_start
        bound["end_s"] = anchor_end
        # Keep the semantic boundary alongside the timing anchor.  The
        # resolver must know which of the two adjacent lexical owners may be
        # clipped when a later CTC/MFA adjustment has made the anchor fall
        # inside a word.
        bound["_reference_boundary"] = boundary
        matched.append((bound, boundary))

    # Only the bound, reference-confirmed occurrences enter the historical
    # interval resolver below.
    punct_entries = [entry for entry, _ in matched]
    if not punct_entries:
        return words_tier, pp_tier

    # Build combined interval list: original words + punctuation.  A CTC
    # punctuation interval that is exactly the same owner as an explicit
    # silence is provisional evidence, not permission to replace the silence.
    # Reference reconciliation can later restore the mark if it is actually
    # authoritative; retaining the silence here is what preserves a CTC-only
    # terminal mark's edge remainder.
    explicit_silences = [iv for iv in words_tier.intervals
                         if is_silence(iv.text) and iv.text.strip()]
    combined = []
    for iv in words_tier.intervals:
        combined.append((iv.xmin, iv.xmax, iv.text, "word"))
    for p in punct_entries:
        if any(abs(float(p["start_s"]) - silence.xmin) <= AXIS_EPS
               and abs(float(p["end_s"]) - silence.xmax) <= AXIS_EPS
               for silence in explicit_silences):
            continue
        combined.append((p["start_s"], p["end_s"], p["word"], "punct"))

    combined.sort(key=lambda x: x[0])

    # Resolve overlaps: punctuation keeps its CTC time, words are trimmed
    # 两轮处理: 先插入所有, 再裁剪 word 与 punct 的重叠
    resolved = []
    for c in combined:
        s, e, text, kind = c
        if e > s:
            resolved.append((s, e, text, kind))

    def _punct_boundary(start: float, end: float, text: str) -> int | None:
        """Return the reference boundary for a resolved punctuation anchor."""
        for entry in punct_entries:
            if (entry.get("word", "") == text
                    and abs(float(entry.get("start_s", -math.inf)) - start)
                    <= AXIS_EPS
                    and abs(float(entry.get("end_s", -math.inf)) - end)
                    <= AXIS_EPS):
                boundary = entry.get("_reference_boundary")
                return int(boundary) if boundary is not None else None
        return None

    # 构建 phone 边界查找: word_text -> [(phone_start, phone_end), ...]
    phone_map: dict[str, list[tuple[float, float]]] = {}
    if pp_tier is not None:
        for iv in pp_tier.intervals:
            if iv.text.strip() and not is_silence(iv.text):
                phone_map.setdefault("", []).append((iv.xmin, iv.xmax))

    def _phone_snap_left(trim_to: float, word_start: float) -> float:
        """Snap left-trim point forward to next phone boundary."""
        if pp_tier is None:
            return trim_to
        next_boundary = trim_to
        for p_iv in pp_tier.intervals:
            if p_iv.xmin >= word_start and p_iv.xmin > trim_to:
                next_boundary = p_iv.xmin
                break
            if p_iv.xmax > trim_to:
                # trim_to falls inside this phone, snap to its end
                next_boundary = p_iv.xmax
        return next_boundary

    # 第二轮: word 优先, 标点裁剪到词边界
    # Regr. Case 52: use while loop so inserted punct fragments
    # are processed (for-range captures len(resolved) once and misses them).
    pi = 0
    while pi < len(resolved):
        ps, pe, ptext, pkind = resolved[pi]
        if pkind != "punct":
            pi += 1
            continue
        for wi in range(len(resolved)):
            ws, we, wtext, wkind = resolved[wi]
            if wkind != "word" or is_silence(wtext):
                continue
            if ws < pe and we > ps:  # overlap exists
                if ws <= ps and we >= pe:
                    # A reference-confirmed punctuation anchor has priority
                    # over an extended lexical owner.  The old code deleted
                    # the punctuation here, which is exactly how a mark that
                    # existed before MFA/CTC boundary compensation was
                    # permanently lost.  Clip only the immediate owner on
                    # the side dictated by the reference boundary.
                    boundary = _punct_boundary(ps, pe, ptext)
                    prev_owner = (lexical[boundary - 1]
                                  if boundary is not None and boundary > 0
                                  else None)
                    next_owner = (lexical[boundary]
                                  if boundary is not None
                                  and boundary < len(lexical) else None)
                    is_prev_owner = (
                        prev_owner is not None
                        and wtext == prev_owner.text
                        and abs(ws - prev_owner.xmin) <= AXIS_EPS)
                    is_next_owner = (
                        next_owner is not None
                        and wtext == next_owner.text
                        and abs(we - next_owner.xmax) <= AXIS_EPS)
                    if is_prev_owner and ps > ws + AXIS_EPS:
                        resolved[wi] = (ws, ps, wtext, wkind)
                    elif is_next_owner and pe < we - AXIS_EPS:
                        resolved[wi] = (pe, we, wtext, wkind)
                    else:
                        # A confirmed anchor should not be discarded merely
                        # because a duplicate/fragmented lexical interval
                        # prevents an exact owner identity match.  Leave the
                        # word untouched; the authoritative restore pass will
                        # perform the same local clipping with the reference
                        # lexical sequence.
                        pass
                    break
                elif ws <= ps:
                    # word overlaps left side of punct → trim punct start
                    resolved[pi] = (we, pe, ptext, pkind)
                    ps = we
                elif we >= pe:
                    # word overlaps right side of punct → trim punct end
                    resolved[pi] = (ps, ws, ptext, pkind)
                    pe = ws
                else:
                    # word inside punct (ws > ps and we < pe):
                    # split punct into left part + right part
                    # Regr. Case 24: preserve left/right parts instead of deleting
                    # Regr. Case 52: insert right_part at pi+1 so the while
                    # loop processes it; break to avoid stale ps/pe
                    left_part  = (ps, ws, ptext, pkind)
                    right_part = (we, pe, ptext, pkind)
                    resolved[pi] = left_part
                    if right_part[1] > right_part[0] + 0.001:
                        resolved.insert(pi + 1, right_part)
                    break
        pi += 1

    # 去掉零时长 interval
    resolved = [(s, e, t, k) for s, e, t, k in resolved if e > s + 0.001]
    resolved.sort(key=lambda x: x[0])

    # Merge adjacent same-text intervals
    merged = []
    for item in resolved:
        # Merge adjacent same-text intervals, but never merge two word intervals
        # (consecutive identical words like pu4 pu4 must stay separate)
        if merged and merged[-1][2] == item[2] and abs(merged[-1][1] - item[0]) < 0.001 \
           and not (merged[-1][3] == "word" and item[3] == "word"):
            merged[-1] = (merged[-1][0], max(merged[-1][1], item[1]), item[2], item[3])
        else:
            merged.append(item)

    # Trim silence gaps overlapped by punct (gap / punct overlap from mixed boundaries)
    for pi in range(len(merged)):
        ps, pe, ptext, pkind = merged[pi]
        if pkind != "punct":
            continue
        for gi in range(len(merged)):
            gs, ge, gtext, gkind = merged[gi]
            if gkind != "word" or not is_silence(gtext):
                continue
            if gs < pe and ge > ps:
                if gs < ps:
                    merged[gi] = (gs, ps, gtext, gkind)  # keep left part of gap
                else:
                    merged[gi] = (pe, ge, gtext, gkind)  # keep right part of gap

    # 去掉零时长
    merged = [(s, e, t, k) for s, e, t, k in merged if e > s + 0.001]

    # Internal-sp ownership is deliberately not guessed in this constructor.
    # The owner-aware passes run after punctuation injection: an accepted
    # punctuation anchor absorbs only its colocated silence, while an
    # unowned internal <spN> remains visible and is vetoed by the final QC.

    # 最后标点保留其 CTC/local-gap end.  The old implementation extended
    # every final mark to ``words_tier.xmax`` and therefore converted an
    # explicit tail silence into punctuation (and enlarged the preceding
    # lexical owner after a later reference restore).
    last_punct = None
    for m in reversed(merged):
        if m[3] == "punct":
            last_punct = m
            break
    if last_punct:
        punct_start = last_punct[0]
        punct_text = last_punct[2]
        if last_punct[1] > punct_start:
            # Keep all owners through the local punctuation end.  Any
            # remaining edge is retained as canonical silence below.
            new_merged = []
            for m in merged:
                if m is last_punct:
                    new_merged.append((punct_start, last_punct[1], punct_text,
                                       "punct"))
                else:
                    new_merged.append(m)
            merged = new_merged

    # Build new words tier (skip zero-duration intervals, ensure sorted)
    merged.sort(key=lambda x: x[0])
    new_words = [Interval(iv[0], iv[1], iv[2]) for iv in merged if iv[1] > iv[0]]
    new_words_tier = _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, new_words))
    new_words_tier = _reconcile_publication_geometry(new_words_tier)

    # Build new pinyin_phones tier (word -> phone, punct -> punct char)
    if pp_tier is not None:
        pp_intervals = []
        for iv in merged:
            if iv[3] == "punct":
                pp_intervals.append(Interval(iv[0], iv[1], iv[2]))
            elif is_silence(iv[2]):
                continue  # skip silence gaps in phone tier
            else:
                # Copy original phone intervals that overlap
                word_phones = []
                for p_iv in pp_tier.intervals:
                    if p_iv.xmax > iv[0] and p_iv.xmin < iv[1] \
                       and not is_silence(p_iv.text):
                        word_phones.append(Interval(
                            max(p_iv.xmin, iv[0]), min(p_iv.xmax, iv[1]),
                            p_iv.text))
                # Extend first phone to word start (unvoiced stop compensation)
                if word_phones and word_phones[0].xmin > iv[0] + 0.005:
                    word_phones[0] = Interval(iv[0], word_phones[0].xmax, word_phones[0].text)
                # If word end was extended past last phone, extend last phone
                if word_phones and iv[1] > word_phones[-1].xmax + 0.005:
                    word_phones[-1] = Interval(
                        word_phones[-1].xmin, iv[1], word_phones[-1].text)
                pp_intervals.extend(word_phones)

        # ── Resolve phone↔punct overlaps in pp tier (Regr. Case 46) ──
        # Punct and content phones can overlap when CTC punct anchors
        # fall within a word's time range.  Punct keeps ≥ 60 ms;
        # overlapping phones are clipped.  If a word's phones are
        # fully covered by punct, they are rebuilt with proportional
        # timing within the remaining non-punct space.
        pp_intervals.sort(key=lambda x: x.xmin)
        _pp_resolved: list[Interval] = []
        for _piv in pp_intervals:
            if not _pp_resolved:
                _pp_resolved.append(_piv)
                continue
            _prev = _pp_resolved[-1]
            _overlap = _prev.xmax - _piv.xmin
            if _overlap <= 0.002:
                _pp_resolved.append(_piv)
                continue

            _prev_is_punct = is_punct(_prev.text) and not is_silence(_prev.text)
            _cur_is_punct = is_punct(_piv.text) and not is_silence(_piv.text)

            if _prev_is_punct and not _cur_is_punct:
                # Punct → content: ensure punct keeps ≥ 60 ms
                _punct_min_end = _prev.xmin + 0.060
                if _prev.xmax < _punct_min_end:
                    _prev = Interval(_prev.xmin, _punct_min_end, _prev.text)
                _piv = Interval(_prev.xmax, _piv.xmax, _piv.text)
                _pp_resolved[-1] = _prev
                if _piv.xmax > _piv.xmin + 0.002:
                    _pp_resolved.append(_piv)
            elif _cur_is_punct and not _prev_is_punct:
                # Content → punct: clip content before punct
                _punct_min_end = _piv.xmin + 0.060
                _piv_end = max(_piv.xmax, _punct_min_end)
                _prev = Interval(_prev.xmin, _piv.xmin, _prev.text)
                _piv = Interval(_piv.xmin, _piv_end, _piv.text)
                if _prev.xmax > _prev.xmin + 0.002:
                    _pp_resolved[-1] = _prev
                else:
                    _pp_resolved.pop()
                _pp_resolved.append(_piv)
            elif _prev_is_punct and _cur_is_punct:
                # Two puncts overlap — keep both but non-overlapping
                _punct_min_end = _prev.xmin + 0.060
                if _prev.xmax < _punct_min_end:
                    _prev = Interval(_prev.xmin, _punct_min_end, _prev.text)
                _piv = Interval(_prev.xmax, max(_piv.xmax, _prev.xmax + 0.060), _piv.text)
                _pp_resolved[-1] = _prev
                _pp_resolved.append(_piv)
            else:
                # Two content phones overlap — clip at midpoint
                _mid = (_prev.xmax + _piv.xmin) / 2.0
                _pp_resolved[-1] = Interval(_prev.xmin, _mid, _prev.text)
                _pp_resolved.append(Interval(_mid, _piv.xmax, _piv.text))

        new_pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, _pp_resolved)
    else:
        new_pp_tier = None

    return new_words_tier, new_pp_tier


def _extend_word_into_ellipsis(words_tier: Tier, pp_tier: Tier | None,
                                audio: list[float] | None, sr: int = 16000,
                                max_extend_s: float = 0.6,
                                min_marker_s: float = 0.06) -> tuple[Tier, Tier | None]:
    """Content word + … — extend word end if ellipsis has audible prolongation energy."""
    if audio is None:
        return words_tier, pp_tier

    all_rms, frame_dur = _frame_rms_vec(audio, sr, frame_ms=10.0)
    k = max(1, int(len(all_rms) * 0.15))
    nf = float(np.partition(all_rms, k)[k]) if len(all_rms) > 0 else 1e-6
    threshold = max(nf * 2.5, 0.005)

    intervals = list(words_tier.intervals)
    n = len(intervals)

    for i in range(n - 1):
        iv_curr = intervals[i]
        iv_next = intervals[i + 1]

        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            continue

        if is_nvv_token(iv_curr.text) or is_punct(iv_curr.text):
            continue
        if iv_curr.text.strip() in SILENCE_LABELS:
            continue
        if not is_word_like(iv_curr.text):
            continue
        if iv_next.text.strip() != '…':
            continue
        if i + 2 >= n:
            continue

        ellipsis_start = iv_next.xmin
        ellipsis_end = iv_next.xmax
        dur = ellipsis_end - ellipsis_start
        if dur < 0.1:
            continue

        # ── Per-word energy reference ──
        # Compare ellipsis energy against the preceding word's tail energy,
        # not just the global noise floor.  This prevents extending into
        # genuinely silent (or near-silent) ellipsis gaps.
        ws = int(max(0, iv_curr.xmax - 0.15) * sr)
        we = int(iv_curr.xmax * sr)
        word_tail = audio[ws:we] if we > ws else None
        if word_tail is not None and len(word_tail) > 0:
            wt_rms, _ = _frame_rms_vec(word_tail, sr, frame_ms=5.0)
            word_tail_rms = float(np.mean(wt_rms)) if len(wt_rms) > 0 else 0.0
        else:
            word_tail_rms = 0.0
        word_ref = max(word_tail_rms, threshold)

        ss = int(ellipsis_start * sr)
        ee = int(ellipsis_end * sr)
        seg = audio[ss:ee]

        seg_rms, _ = _frame_rms_vec(seg, sr, frame_ms=5.0)
        if len(seg_rms) == 0:
            continue

        # Energy in the first ~40 ms of the ellipsis (the prolongation zone).
        n_probe = max(1, int(0.04 / 0.005))
        probe_rms = seg_rms[:n_probe]
        probe_energy = float(np.mean(probe_rms))

        # Require the early ellipsis energy to be at least 30% of the word's
        # tail energy — otherwise it's just silence, not prolongation.
        if probe_energy < word_ref * 0.30:
            continue

        # Find energy decay: ≥2 consecutive frames below threshold (vectorised)
        below_mask = seg_rms < max(threshold, word_ref * 0.20)
        decay_idx = len(seg_rms)
        for j in range(len(below_mask) - 1):
            if below_mask[j] and below_mask[j + 1]:
                decay_idx = j
                break

        if decay_idx <= 0:
            # No clear decay — extend to cover the leading-energy portion
            n_above = 0
            for j in range(len(seg_rms)):
                if seg_rms[j] >= word_ref * 0.25:
                    n_above += 1
                else:
                    break
            extend_target = ellipsis_start + max(n_above * 0.005, dur * 0.10)
        elif decay_idx >= len(seg_rms):
            continue
        else:
            decay_time = max(0.0, ellipsis_start + decay_idx * 0.005)
            extend_target = min(decay_time, ellipsis_start + dur * 0.6)

        max_extend = min(max_extend_s, dur * 0.6)
        new_word_end = min(extend_target, iv_curr.xmax + max_extend)
        new_word_end = min(new_word_end, intervals[i + 2].xmin - 0.02)

        if ellipsis_end - new_word_end < min_marker_s:
            new_word_end = ellipsis_end - min_marker_s

        if new_word_end <= iv_curr.xmax + 0.015:
            continue

        intervals[i] = Interval(iv_curr.xmin, new_word_end, iv_curr.text)
        intervals[i + 1] = Interval(new_word_end, ellipsis_end, '…')

    intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
    new_words = _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))

    if pp_tier is not None:
        pp_ivs = list(pp_tier.intervals)
        for i in range(len(pp_ivs) - 1):
            pp_cur = pp_ivs[i]
            pp_next = pp_ivs[i + 1]
            if pp_next.text.strip() != '…':
                continue
            if is_nvv_token(pp_cur.text) or is_punct(pp_cur.text):
                continue
            if pp_cur.text.strip() in SILENCE_LABELS:
                continue
            # Find matching extended word in words tier
            for w_iv in intervals:
                if w_iv.text.strip() == '…':
                    continue
                if abs(w_iv.xmin - pp_cur.xmin) < 0.1:
                    pp_ivs[i] = Interval(pp_cur.xmin, w_iv.xmax, pp_cur.text)
                    pp_ivs[i + 1] = Interval(w_iv.xmax, pp_next.xmax, '…')
                    break
        pp_ivs = [iv for iv in pp_ivs if iv.xmax > iv.xmin + 0.001]
        new_pp = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, pp_ivs)
    else:
        new_pp = None

    return new_words, new_pp


def _merge_nvv_ellipsis(words_tier: Tier, pp_tier: Tier | None,
                         audio: list[float] | None, sr: int = 16000,
                         marker_ms: float = 60.0) -> tuple[Tier, Tier | None]:
    """NVV 后的省略号如果包含可听能量, 合并到 NVV, 只留 marker_ms 的标点."""
    if audio is None:
        return words_tier, pp_tier

    all_rms, _ = _frame_rms_vec(audio, sr, frame_ms=10.0)
    k = max(1, int(len(all_rms) * 0.15))
    nf = float(np.partition(all_rms, k)[k]) if len(all_rms) > 0 else 1e-6
    threshold = max(nf * 3.0, 0.005)

    intervals = list(words_tier.intervals)
    n = len(intervals)

    for i in range(n - 1):
        iv_curr = intervals[i]
        iv_next = intervals[i + 1]
        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            continue
        if not is_nvv_token(iv_curr.text):
            continue
        if iv_next.text.strip() != '…':
            continue

        ellipsis_start = iv_next.xmin
        ellipsis_end = iv_next.xmax
        ss = int(ellipsis_start * sr)
        ee = int(ellipsis_end * sr)
        if ee <= ss:
            continue
        seg = audio[ss:ee]
        seg_rms, _ = _frame_rms_vec(seg, sr, frame_ms=5.0)
        if len(seg_rms) == 0:
            continue
        energy_ratio = float(np.mean(seg_rms > threshold))

        # ≥30% 帧有能量 -> 合并; NVV 后极短省略号 (<100ms) 无条件合并
        ellipsis_dur = ellipsis_end - ellipsis_start
        if energy_ratio < 0.3 and ellipsis_dur >= 0.1:
            continue

        # 合并: NVV 延伸到省略号结束前 marker_ms
        marker_s = marker_ms / 1000.0
        new_nvv_end = max(ellipsis_end - marker_s, iv_curr.xmax)
        new_ellipsis_start = new_nvv_end
        new_ellipsis_end = ellipsis_end

        if new_ellipsis_end - new_ellipsis_start < 0.02:
            # 剩余太短, 删除省略号
            intervals[i] = Interval(iv_curr.xmin, ellipsis_end, iv_curr.text)
            intervals[i + 1] = Interval(0, 0, '')
        else:
            intervals[i] = Interval(iv_curr.xmin, new_nvv_end, iv_curr.text)
            intervals[i + 1] = Interval(new_ellipsis_start, new_ellipsis_end, '…')

    # 去零时长
    intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
    new_words = _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))

    # pinyin_phones: NVV 延伸到新边界
    if pp_tier is not None:
        pp_intervals = list(pp_tier.intervals)
        for i in range(len(pp_intervals)):
            if is_nvv_token(pp_intervals[i].text):
                for w_iv in intervals:
                    if w_iv.text == pp_intervals[i].text:
                        pp_intervals[i] = Interval(
                            max(pp_intervals[i].xmin, w_iv.xmin),
                            w_iv.xmax, pp_intervals[i].text)
                        break
            elif pp_intervals[i].text.strip() == '…':
                for w_iv in intervals:
                    if w_iv.text.strip() == '…':
                        pp_intervals[i] = Interval(w_iv.xmin, w_iv.xmax, '…')
                        break
        pp_intervals = [iv for iv in pp_intervals if iv.xmax > iv.xmin + 0.001]
        new_pp = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, pp_intervals)
    else:
        new_pp = None

    return new_words, new_pp


def _refine_boundaries_by_energy(words_tier: Tier, audio, sr: int,
                                  search_window: float = 0.2,
                                  min_word_dur: float = 0.03,
                                  punct_entries: list | None = None,
                                  _punct_boundary_hits: list | None = None) -> Tier:
    """词落在静音段时向后搜索语音起点, 整体后移 (不越过后词).  Vectorised."""
    import numpy as _np
    if _punct_boundary_hits is None:
        _punct_boundary_hits = []
    all_rms, _ = _frame_rms_vec(audio, sr, frame_ms=10.0)
    if len(all_rms) == 0:
        return words_tier
    k = max(1, int(len(all_rms) * 0.15))
    nf = float(_np.partition(all_rms, k)[k])

    intervals = list(words_tier.intervals)
    n = len(intervals)

    threshold = max(nf * 3.0, 0.001)

    # 从右往左处理: 后面的词先移, 给前面的词腾空间
    for i in range(n - 1, -1, -1):
        iv = intervals[i]
        if is_silence(iv.text) or not iv.text.strip():
            continue
        # Explicit visual silence is reserved for the final owner resolver;
        # boundary refinement may not resize or consume that interval.
        if ((i > 0 and is_silence(intervals[i - 1].text))
                or (i + 1 < n and is_silence(intervals[i + 1].text))):
            continue
        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            # A valid CTC anchor is the word-level authority.  Energy may
            # diagnose the interval later, but it must not move this word
            # outside the accepted CTC span.
            continue
        # Skip English/NVV: MFA cannot model their phones, so energy checks
        # are unreliable.  CTC boundaries (from _snap_to_ctc) are authoritative.
        if is_english_token(iv.text) or is_nvv_token(iv.text):
            continue
        word_start = iv.xmin
        word_end = iv.xmax
        dur = word_end - word_start

        # 检查整词能量: 是否完全在静音中
        w_ss = max(0, int(word_start * sr))
        w_ee = min(len(audio), int(word_end * sr))
        if w_ee <= w_ss:
            continue
        word_rms = float(_np.mean(_np.abs(audio[w_ss:w_ee])))

        if word_rms >= threshold:
            continue  # 词有能量, 不需要整体移动

        # 词在静音中 -> 搜索后方的语音起点
        search_end = min(word_start + search_window, len(audio) / sr)
        if i + 1 < n:
            next_iv = intervals[i + 1]
            if next_iv.xmax > next_iv.xmin:
                # 允许稍微越过 silence 间隔, 但不能越过下一个实词
                search_end = min(search_end, next_iv.xmax - min_word_dur)

        s_sample = int(word_start * sr)
        e_sample = int(search_end * sr)
        if e_sample <= s_sample:
            continue

        frame_s = max(1, int(0.005 * sr))
        n_frames = (e_sample - s_sample) // frame_s
        if n_frames <= 0:
            continue
        end_s = s_sample + n_frames * frame_s
        if end_s > len(audio):
            continue
        frames = audio[s_sample:end_s].reshape(n_frames, frame_s)
        frame_rms_arr = _np.mean(_np.abs(frames), axis=1)
        above = _np.where(frame_rms_arr > threshold)[0]
        if len(above) == 0:
            continue
        onset = (s_sample + above[0] * frame_s) / sr

        if onset is None or onset <= word_start:
            continue

        # 整体后移: 不越过后词, 空间不够则放弃
        dur = word_end - word_start
        new_start = onset
        new_end = onset + dur
        if i + 1 < n:
            next_iv = intervals[i + 1]
            if next_iv.xmax > next_iv.xmin and not is_silence(next_iv.text):
                new_end = min(new_end, next_iv.xmin - 0.005)
        if new_end - new_start < min_word_dur:
            continue

        # 前一个间隔如果是静音, 延伸覆盖空出的间隙
        if i > 0 and is_silence(intervals[i - 1].text):
            intervals[i - 1] = Interval(intervals[i - 1].xmin, new_start,
                                        intervals[i - 1].text)
        # 如果下一个是静音, 调整它的起点
        if i + 1 < n and is_silence(intervals[i + 1].text):
            intervals[i + 1] = Interval(new_end, intervals[i + 1].xmax,
                                        intervals[i + 1].text)
        intervals[i] = Interval(new_start, new_end, iv.text)

    # ── Silence-adjacent word start pull-back ──
    # When a word follows a silence gap (or another word but the
    # boundary region is all silence), and its energy onset is
    # clearly before the word start, pull the start back to the onset.
    # ── Silence-adjacent word start pull-back ──
    # When a word follows a SILENCE gap and its energy onset is
    # clearly before the word start, pull the start back to the onset.
    # Only silence-to-word (not word-to-word, which is handled by the
    # start pull-back below and is more prone to false positives).
    for i in range(1, n):
        iv = intervals[i]
        if is_silence(iv.text) or not iv.text.strip():
            continue
        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            continue
        if is_english_token(iv.text) or is_nvv_token(iv.text):
            continue
        prev_iv = intervals[i - 1]
        if not is_silence(prev_iv.text):
            continue
        word_start = iv.xmin
        search_back = min(0.150, word_start - prev_iv.xmin)
        if search_back < 0.030:
            continue
        s_sample = int((word_start - search_back) * sr)
        e_sample = int(word_start * sr)
        win3 = max(1, int(0.010 * sr))
        n_wins3 = (e_sample - s_sample) // win3
        if n_wins3 < 5:
            continue
        rms_vals3 = []
        for j in range(n_wins3):
            chunk = audio[s_sample + j*win3 : s_sample + (j+1)*win3]
            rms_vals3.append(float(_np.mean(_np.abs(chunk))))
        onset_win3 = None
        for j in range(1, n_wins3):
            if rms_vals3[j] > rms_vals3[j-1] * 5.0 and rms_vals3[j] > 0.0005:
                onset_win3 = j
                break
        if onset_win3 is None or onset_win3 < 2:
            continue
        onset_time = word_start - search_back + (onset_win3 - 0.5) * win3 / sr
        pull = word_start - onset_time
        if pull < 0.020 or pull > 0.120:
            continue
        # Verify onset area has real energy (check 3 frames around onset)
        onset_peak = max(rms_vals3[onset_win3:min(onset_win3+3, n_wins3)])
        if onset_peak < 0.002:
            continue
        new_boundary = round(onset_time, 3)
        intervals[i - 1] = Interval(prev_iv.xmin, new_boundary, prev_iv.text)
        intervals[i] = Interval(new_boundary, iv.xmax, iv.text)

    # ── Start pull-back: MFA boundary placed too late ──
    # When energy shows a deep dip followed by a clear syllable onset
    # before the word start, pull the start back to the dip.
    for i in range(1, n):
        iv = intervals[i]
        if is_silence(iv.text) or not iv.text.strip():
            continue
        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            continue
        if is_english_token(iv.text) or is_nvv_token(iv.text):
            continue
        prev_iv = intervals[i - 1]
        if prev_iv.xmax <= prev_iv.xmin:
            continue
        # Only adjust if previous interval is a real word (not silence)
        if is_silence(prev_iv.text):
            continue

        word_start = iv.xmin

        # Search up to 80ms backward.  Window must be short enough
        # that max_rms reflects the LOCAL neighbourhood, not a distant
        # peak from syllables 50ms away (which would make shallow vowel
        # decays appear as "deep valleys").
        search_back = min(0.08, word_start - prev_iv.xmin)
        if search_back < 0.030:
            continue

        s_sample = int((word_start - search_back) * sr)
        e_sample = int(word_start * sr)
        if e_sample <= s_sample:
            continue

        win = max(1, int(0.010 * sr))
        n_wins = (e_sample - s_sample) // win
        if n_wins < 5:
            continue

        rms_vals = []
        for j in range(n_wins):
            chunk = audio[s_sample + j*win : s_sample + (j+1)*win]
            rms_vals.append(float(_np.mean(_np.abs(chunk))))

        max_rms = max(rms_vals) if rms_vals else 1.0
        if max_rms < 0.003:
            continue  # too quiet to be meaningful

        # Find the deepest valley that satisfies:
        # 1. Below 50% of max energy in window (clear dip)
        # 2. Local minimum
        # 3. At least 25ms before word_start
        best_valley = None
        for j in range(2, n_wins - 2):
            r = rms_vals[j]
            if r >= max_rms * 0.50 or r < 0.003:
                continue
            # Local minimum check
            if r > rms_vals[j-1] or r > rms_vals[j+1]:
                continue
            valley_time = word_start - search_back + (j + 0.5) * win / sr
            pull = word_start - valley_time
            if pull < 0.025 or pull > 0.080:
                continue
            # Energy should be rising after the valley
            post_valley = rms_vals[j+1:min(j+4, n_wins)]
            if len(post_valley) >= 2 and _np.mean(post_valley) <= r * 1.2:
                continue  # no clear rise after valley
            # Don't make previous word shorter than 80ms
            new_prev_dur = valley_time - prev_iv.xmin
            if new_prev_dur < 0.080:
                continue
            best_valley = valley_time
            break  # take the earliest qualifying valley

        if best_valley is None:
            continue

        new_boundary = round(best_valley, 3)
        intervals[i - 1] = Interval(prev_iv.xmin, new_boundary, prev_iv.text)
        intervals[i] = Interval(new_boundary, iv.xmax, iv.text)

    # ── End extension: MFA boundary cut off vowel tail ──
    # When a word's energy continues past its MFA end into a silence
    # or NVV interval (i.e. the decay was mislabeled), extend the word
    # end to the true energy drop point.  Process left→right so
    # extensions chain correctly.
    _extended_indices: set[int] = set()  # track which words were extended
    for i in range(n):
        iv = intervals[i]
        if is_silence(iv.text) or not iv.text.strip():
            continue
        if _ctc_authoritative_ordinal(words_tier, i) is not None:
            continue
        if is_english_token(iv.text) or is_nvv_token(iv.text):
            continue
        if i + 1 >= n:
            continue
        next_iv = intervals[i + 1]
        if next_iv.xmax <= next_iv.xmin:
            continue
        # Explicit words-tier silence is reserved for the final visual
        # resolver; this stage may not consume or resize it.
        if is_silence(next_iv.text):
            continue
        if is_punct(next_iv.text) and not is_silence(next_iv.text):
            continue

        extend_into_word = False
        if not is_nvv_token(next_iv.text):
            # Regular word: check if the leading portion is dead silence.
            check_s = int(iv.xmax * sr)
            check_e = int(min(iv.xmax + 0.300, next_iv.xmax) * sr)
            if check_e - check_s < int(0.080 * sr):
                continue
            win_s = max(1, int(0.010 * sr))
            n2 = (check_e - check_s) // win_s
            if n2 < 10:
                continue
            max_silent_run = 0
            silent_run = 0
            for j2 in range(n2):
                chunk = audio[check_s + j2*win_s : check_s + (j2+1)*win_s]
                if float(_np.mean(_np.abs(chunk))) < 0.002:
                    silent_run += 1
                    max_silent_run = max(max_silent_run, silent_run)
                else:
                    silent_run = 0
            if max_silent_run < 8:
                continue
            # Check for punctuation in the gap: if punct exists, let
            # _inject_punctuation handle the silence placement.
            # Search through the FULL next interval(s), not just the
            # silent run — punct may sit past where energy rises.
            gap_end_full = next_iv.xmax
            # Also check the word after silence, if any
            if is_silence(next_iv.text) and i + 2 < n and not is_silence(intervals[i+2].text):
                gap_end_full = intervals[i+2].xmax
            has_punct_in_gap = False
            _punct_boundary_detail = None
            if punct_entries:
                for p in punct_entries:
                    if iv.xmax <= p["start_s"] <= gap_end_full:
                        has_punct_in_gap = True
                        break
                    # Also detect punct starting near the word boundary:
                    # when CTC punct starts just before MFA word end
                    # (within 100ms), but its body extends well past the
                    # word end, it's a separate pause marker — not a
                    # prolongation of the current word.  Regression Case 25-G.
                    _near = abs(p["start_s"] - iv.xmax) < 0.100
                    _body_past = p["end_s"] > iv.xmax + 0.060
                    if _near and _body_past:
                        has_punct_in_gap = True
                        _punct_boundary_detail = {
                            "word": iv.text.strip(),
                            "word_xmax": round(iv.xmax, 3),
                            "punct": p["word"],
                            "punct_start": round(p["start_s"], 3),
                            "punct_end": round(p["end_s"], 3),
                            "offset_ms": round((iv.xmax - p["start_s"]) * 1000, 1),
                        }
                        break
            if has_punct_in_gap:
                if _punct_boundary_detail:
                    _punct_boundary_hits.append(_punct_boundary_detail)
                continue  # punct will absorb the silence
            extend_into_word = True

        word_end = iv.xmax
        next_end = next_iv.xmax

        # When next interval is silence, look past it to the following word
        # for onset detection (silence itself has no energy to detect).
        onset_next = next_iv
        onset_end = next_end
        if is_silence(next_iv.text) and i + 2 < n:
            onset_next = intervals[i + 2]
            if not is_silence(onset_next.text) and not is_punct(onset_next.text):
                onset_end = onset_next.xmax

        if extend_into_word:
            # Dead silence after current word — skip the silence and
            # extend current word's end to where energy returns in the
            # following word (or silence gap end if no following word).
            search_s = int(word_end * sr)
            search_e = int(onset_end * sr)
            win_s = max(1, int(0.010 * sr))
            # Measure silent baseline from first 5 windows
            baseline_rms = 0.001
            count = 0
            for j in range(min(10, (search_e - search_s) // win_s)):
                chunk = audio[search_s + j*win_s : search_s + (j+1)*win_s]
                r = float(_np.mean(_np.abs(chunk)))
                if r < 0.003:
                    baseline_rms += r
                    count += 1
            if count > 0:
                baseline_rms /= count
            onset_threshold = max(baseline_rms * 3.0, 0.0015)
            onset_idx = None
            for j in range(0, (search_e - search_s) // win_s):
                chunk = audio[search_s + j*win_s : search_s + (j+1)*win_s]
                if float(_np.mean(_np.abs(chunk))) >= onset_threshold:
                    onset_idx = j
                    break
            if onset_idx is None or onset_idx < 10:
                continue
            # Leave at least 60ms for the word after the silence gap
            new_end_raw = word_end + onset_idx * win_s / sr
            onset_word_min_start = onset_end - 0.060
            new_end = min(new_end_raw, onset_word_min_start)
            if new_end - word_end < 0.050:
                continue
            ext_limit = new_end
        else:
            # NVV path: Check up to 250ms past word_end
            ext_limit = min(word_end + 0.25, next_end - 0.015)
        if ext_limit <= word_end + 0.015:
            continue

        if extend_into_word:
            # ext_limit already computed above; skip RMS vowel-tail analysis
            new_end = ext_limit
        else:
            s_sample = int(word_end * sr)
            e_sample = int(ext_limit * sr)
            if e_sample <= s_sample:
                continue

            win = max(1, int(0.010 * sr))
            n_wins = (e_sample - s_sample) // win
            if n_wins < 3:
                continue

            rms_vals = []
            for j in range(n_wins):
                chunk = audio[s_sample + j*win : s_sample + (j+1)*win]
                rms_vals.append(float(_np.mean(_np.abs(chunk))))

            first_half = _np.mean(rms_vals[:max(1, n_wins//2)])
            second_half = _np.mean(rms_vals[max(1, n_wins//2):])
            if second_half > first_half * 1.3:
                continue

            below_run = 0
            cutoff_win = n_wins
            for j, r in enumerate(rms_vals):
                if r < threshold:
                    below_run += 1
                    if below_run >= 3:
                        cutoff_win = j - below_run + 1
                        break
                else:
                    below_run = 0

            if cutoff_win < 2:
                continue

            new_end = word_end + (cutoff_win * win) / sr
            new_end = min(new_end, next_end - 0.005)

        if new_end - word_end < 0.020:
            continue  # too small to matter

        # Extend word, shorten next interval(s).
        min_next_dur = 0.040  # unified minimum for next word
        if onset_end - new_end < min_next_dur:
            new_end = onset_end - min_next_dur
            if new_end - word_end < 0.020:
                continue
        intervals[i] = Interval(iv.xmin, new_end, iv.text)
        _extended_indices.add(i)
        if is_silence(next_iv.text) and new_end >= next_iv.xmax - 0.001:
            # Silence fully absorbed: remove it, shift the following word.
            # Preserve the original end of the shifted word (don't shrink it).
            shifted_end = max(onset_end, onset_next.xmax)
            intervals[i + 1] = Interval(new_end, shifted_end, onset_next.text)
            if onset_next is not next_iv and i + 2 < n:
                intervals[i + 2] = Interval(0, 0, '')
        elif new_end < next_end:
            intervals[i + 1] = Interval(new_end, next_end, next_iv.text)

    # ── NVV forward extension: breath/paralinguistic energy often
    # continues past the MFA/CTC NVV boundary into the following
    # silence.  Extend NVV end to where energy truly drops to noise.
    for i in range(n):
        iv = intervals[i]
        if not is_nvv_token(iv.text):
            continue
        if i + 1 >= n:
            continue
        next_iv = intervals[i + 1]
        if not is_silence(next_iv.text):
            continue
        # NVV is a protected semantic owner; its explicit following silence
        # is still left for the final visual snapshot, not consumed here.
        continue
        if next_iv.xmax <= next_iv.xmin:
            continue

        nvv_end = iv.xmax
        # Look up to 400ms into following silence
        ext_limit = min(nvv_end + 0.4, next_iv.xmax)
        if ext_limit <= nvv_end + 0.015:
            continue

        s_sample = int(nvv_end * sr)
        e_sample = int(ext_limit * sr)
        win = max(1, int(0.010 * sr))
        n_wins = (e_sample - s_sample) // win
        if n_wins < 5:
            continue

        rms_vals = []
        for j in range(n_wins):
            chunk = audio[s_sample + j*win : s_sample + (j+1)*win]
            rms_vals.append(float(_np.mean(_np.abs(chunk))))

        # A breath-level energy floor: above absolute silence but
        # below speech.  Use max(nf * 1.5, 0.0003) so we catch
        # quiet breathing but not dead silence.
        breath_floor = max(float(nf) * 1.5, 0.0003)

        # Find sustained silence (3 frames = 30ms below breath_floor)
        below_run = 0
        cutoff_win = n_wins
        for j, r in enumerate(rms_vals):
            if r < breath_floor:
                below_run += 1
                if below_run >= 3:
                    cutoff_win = j - below_run + 1
                    break
            else:
                below_run = 0

        if cutoff_win < 5:
            continue  # less than 50ms extension — not worth it

        new_end = nvv_end + (cutoff_win * win) / sr
        new_end = min(new_end, next_iv.xmax - 0.005)

        if new_end - nvv_end < 0.050:
            continue

        intervals[i] = Interval(iv.xmin, new_end, iv.text)
        if new_end < next_iv.xmax:
            intervals[i + 1] = Interval(new_end, next_iv.xmax, next_iv.text)

    # ── End trimming: word tails that decay into silence ──
    # Sentence-final words often have their tail silence absorbed
    # into the word boundary.  Trim the end to the last frame above
    # threshold.  Applies to ALL word types including English (e.g. "bug"
    # at sentence end with 900ms trailing silence).
    # Skip words that were intentionally extended by end-extension above.
    for i in range(n - 1, -1, -1):
        iv = intervals[i]
        if i in _extended_indices:
            continue
        if is_silence(iv.text) or not iv.text.strip():
            continue
        if is_nvv_token(iv.text):
            continue  # NVV: no acoustic model for energy checks
        if is_punct(iv.text):
            continue
        dur = iv.xmax - iv.xmin
        if dur < 0.15:
            continue  # already short, don't trim further

        # Check tail region: last 30% of the word (min 80ms)
        tail_start_s = max(iv.xmin + dur * 0.7, iv.xmax - 0.300)
        tail_start = int(tail_start_s * sr)
        tail_end = int(iv.xmax * sr)
        if tail_end - tail_start < int(0.040 * sr):
            continue  # tail too short to analyze

        tail_seg = audio[tail_start:tail_end]
        tail_rms = float(_np.mean(_np.abs(tail_seg)))
        if tail_rms >= threshold * 0.8:
            continue  # tail has meaningful energy, keep boundary

        # Search backward from word end to find last frame above threshold
        w_start_s = int(iv.xmin * sr)
        w_end_s = int(iv.xmax * sr)
        frame_s = max(1, int(0.010 * sr))
        n_frames = (w_end_s - w_start_s) // frame_s
        if n_frames <= 0:
            continue
        end_s = w_start_s + n_frames * frame_s
        if end_s > len(audio):
            continue
        frames = audio[w_start_s:end_s].reshape(n_frames, frame_s)
        frame_rms_arr = _np.mean(_np.abs(frames), axis=1)
        last_above = -1
        for fi in range(n_frames - 1, -1, -1):
            if frame_rms_arr[fi] > threshold:
                last_above = fi
                break
        if last_above < 0:
            continue  # entire word below threshold, leave as-is

        new_end_s = (w_start_s + (last_above + 1) * frame_s) / sr
        trimmed = iv.xmax - new_end_s
        if trimmed < 0.030:
            continue  # trim too small, not worth creating a gap

        # Trim: word ends at new_end_s, remainder becomes silence gap
        intervals[i] = Interval(iv.xmin, min(new_end_s, iv.xmax), iv.text)
        gap_label = silence_label(trimmed)
        if i + 1 < len(intervals) and is_silence(intervals[i + 1].text):
            # Merge into existing trailing silence gap
            next_iv = intervals[i + 1]
            intervals[i + 1] = Interval(new_end_s, next_iv.xmax, next_iv.text)
        else:
            intervals.insert(i + 1, Interval(new_end_s, iv.xmax, gap_label))

    intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
    return _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))



def _snap_to_ctc(words_tier: Tier, pp_tier: Tier | None,
                  ctc_tokens: list[dict],
                  snap_threshold: float = 0.3,
                  punct_entries: list[dict] | None = None,
                  audio=None, sr: int = 16000,
                  _punct_boundary_hits: list | None = None) -> tuple[Tier, Tier | None]:
    """Resolve word spans with CTC as the cross-word anchor.

    MFA remains useful for deciding whether a close CTC span contains speech,
    and its phone timing is retained as the within-word proportional source.
    It must not move a word through the neighbouring CTC anchor: adjacent
    conflicts are resolved at the CTC boundary (or the midpoint of
    overlapping CTC spans), with any remaining gap represented explicitly.

    When keeping MFA boundaries, silence gaps use CTC gap positions to
    correctly place punctuation between words.
    """
    if _punct_boundary_hits is None:
        _punct_boundary_hits = []

    mfa_words = [(i, iv) for i, iv in enumerate(words_tier.intervals)
                 if not is_silence(iv.text) and iv.text.strip() not in ("", "<eps>")
                 and not is_punct(iv.text)]

    # Build alignment between MFA and CTC token sequences.
    # When counts differ (common with NVV/English tokens), use
    # Needleman-Wunsch to find matching pairs instead of skipping.
    ctc_aligned: list[dict | None] = list(ctc_tokens)  # 1:1 with mfa_words after alignment
    ctc_word_lexical_ordinal = {
        id(token): ordinal for ordinal, token in enumerate(ctc_tokens)}

    if len(mfa_words) != len(ctc_tokens):
        # Needleman-Wunsch alignment on token text
        mfa_texts = [iv.text.strip().lower() for _, iv in mfa_words]
        ctc_texts = [t.get("word", "").strip().lower() for t in ctc_tokens]
        matched_pairs = align_sequences(mfa_texts, ctc_texts)

        # Build aligned CTC list: None for unmatched MFA positions
        ctc_aligned = [None] * len(mfa_words)
        for mi, ci in matched_pairs:
            ctc_aligned[mi] = ctc_tokens[ci]
        import sys
        n_matched = sum(1 for x in ctc_aligned if x is not None)
        print(f"  _snap_to_ctc: token count mismatch (MFA={len(mfa_words)}, CTC={len(ctc_tokens)}) — "
              f"NW aligned {n_matched}/{len(mfa_words)} tokens", file=sys.stderr)

    new_word_ivs = []        # (xmin, xmax, text, source)
    new_phone_ivs = []       # (xmin, xmax, text)
    ctc_authority: list[dict] = []

    # Pass 0: detect NVV/English overlap with previous word's CTC.
    # When an NVV's CTC start falls before the previous word's CTC end,
    # the previous word's CTC boundary is inflated by the NVV's energy.
    # Clip the previous word's effective CTC end to the NVV's CTC start.
    ctc_end_clip = [None] * len(mfa_words)  # per-word CTC end ceiling
    for idx in range(1, len(mfa_words)):
        _, prev_mfa = mfa_words[idx - 1]
        _, cur_mfa = mfa_words[idx]
        prev_ctc = ctc_aligned[idx - 1]
        cur_ctc = ctc_aligned[idx]
        if prev_ctc is None or cur_ctc is None:
            continue
        if (is_nvv_token(cur_mfa.text) or is_english_token(cur_mfa.text)):
            if cur_ctc["start_s"] < prev_ctc["end_s"] - 0.010:
                # NVV overlaps previous word's CTC -> cap prev CTC end
                ctc_end_clip[idx - 1] = min(
                    ctc_end_clip[idx - 1] if ctc_end_clip[idx - 1] is not None else float('inf'),
                    cur_ctc["start_s"])

    prev_end = 0.0
    prev_ctc_start = 0.0
    prev_ctc_end = 0.0

    for idx, (wi, mfa_iv) in enumerate(mfa_words):
        ctc = ctc_aligned[idx]
        if ctc is None:
            # Unmatched token — keep MFA boundaries unchanged
            word_start = mfa_iv.xmin
            word_end = mfa_iv.xmax
            ctc_authority.append({
                "lexical_ordinal": len(ctc_authority),
                "text": mfa_iv.text.strip(),
                "boundary_source": "mfa",
                "ctc_lexical_ordinal": None,
                "ctc_span": None,
                "mfa_span": [float(mfa_iv.xmin), float(mfa_iv.xmax)],
                "resolved_span": [float(word_start), float(word_end)],
                "initial_resolved_span": [float(word_start), float(word_end)],
                "published_span": None,
                "operations": [],
            })
            for p_iv in (pp_tier.intervals if pp_tier else []):
                if p_iv.xmax > mfa_iv.xmin and p_iv.xmin < mfa_iv.xmax:
                    new_phone_ivs.append((p_iv.xmin, p_iv.xmax, p_iv.text))
            prev_end = word_end
            continue
        ctc_start = ctc["start_s"]
        ctc_end_raw = ctc["end_s"]
        # Apply NVV-overlap clip: when next word is NVV that overlaps,
        # cap this word's CTC end to NVV's CTC start (CTC inflated by NVV).
        ctc_end = min(ctc_end_raw, ctc_end_clip[idx]) if ctc_end_clip[idx] is not None else ctc_end_raw
        mfa_start = mfa_iv.xmin
        mfa_end = mfa_iv.xmax
        mfa_dur = mfa_end - mfa_start if mfa_end > mfa_start else 0.001

        start_diff = abs(mfa_start - ctc_start)
        end_diff = abs(mfa_end - ctc_end)
        # ── Boundary trust decision (ORDER CRITICAL) ──
        # Checks are evaluated in priority order; later checks override
        # earlier ones only when use_mfa is still True.
        use_mfa = (start_diff <= snap_threshold and end_diff <= snap_threshold)
        # Rule 0: MFA produced <unk> — alignment failed; restore CTC token text
        # and use CTC boundaries (same as Rule 1).
        if is_unknown_token(mfa_iv.text):
            use_mfa = False
            mfa_iv.text = ctc.get('word', mfa_iv.text)
        # Rule 1: NVV / English — no MFA acoustic model, normally use CTC.
        # A short NVV without corroborating punctuation may still be a CTC
        # noise artifact, so retain the historical MFA fallback in that case.
        # If a schema-valid punctuation candidate is ordinal-adjacent, its
        # gap geometry and the NVV geometry share the same CTC coordinate
        # system.  Keeping the displaced MFA span lets the punctuation owner
        # erase the short NVV (LAria_00285), so CTC is mandatory there.
        if is_nvv_token(mfa_iv.text):
            nvv_has_adjacent_punctuation = any(
                isinstance(entry, dict)
                and entry.get("schema") == PUNCTUATION_EVIDENCE_SCHEMA
                and entry.get("source") == "ctc"
                and (entry.get("left_lexical_ordinal") == idx
                     or entry.get("right_lexical_ordinal") == idx)
                for entry in (punct_entries or [])
            )
            use_mfa = ((ctc_end - ctc_start) < 0.10
                       and not nvv_has_adjacent_punctuation)
        elif is_english_token(mfa_iv.text):
            use_mfa = False
        # Rule 2a: MFA phone evidence arbitration.
        # When MFA placed phones in the disputed region between CTC end
        # and MFA end, AND those phones are within this word's range
        # (not the neighbour's), they ARE acoustic evidence for THIS word.
        # This overrides duration-ratio rules below.
        has_mfa_phone_evidence = False
        if end_diff > 0.010 and pp_tier is not None:
            early = min(mfa_end, ctc_end)
            later = max(mfa_end, ctc_end)
            # Only count phones that start before this word's MFA end.
            # Phones starting at/after MFA end belong to the next word.
            disputed_phones = [
                p for p in pp_tier.intervals
                if p.xmax > early and p.xmin < later
                and not is_silence(p.text)
                and p.xmin < mfa_end  # starts before this word's MFA end
            ]
            has_mfa_phone_evidence = len(disputed_phones) > 0
            if has_mfa_phone_evidence:
                pass  # MFA phones in disputed region → speech evidence → keep MFA

        # Rule 2b: MFA severely compressed a short word -> trust CTC
        # (skip if MFA phone evidence exists in disputed region)
        ctc_dur = ctc_end - ctc_start
        if use_mfa and not has_mfa_phone_evidence and mfa_dur < 0.06 and ctc_dur > 0.15:
            use_mfa = False
        # Rule 3: MFA stretched or compressed beyond 2x ratio -> trust CTC
        # (skip if MFA phone evidence exists in disputed region)
        # ALSO skip when MFA's shorter duration is due to trailing <eps>
        # (silence) that CTC assigned to this word. Two patterns:
        #   a) trailing silence before punctuation (jie2 case)
        #   b) preceding word's trailing <eps> absorbed into this word's CTC span (er4 case)
        ratio_skip = False
        if use_mfa and not has_mfa_phone_evidence \
           and ctc_dur > mfa_dur * 2.0:
            # Check for trailing <eps> after this word's MFA end
            has_trailing_sil = any(
                is_silence(iv.text)
                and iv.xmin >= mfa_end - 0.01
                and iv.xmin < ctc_end + 0.05
                for iv in words_tier.intervals
            )
            if has_trailing_sil:
                # Pattern (a): trailing silence + punct
                if punct_entries and mfa_end < ctc_end:
                    for p in punct_entries:
                        if mfa_end <= p["start_s"] <= mfa_end + 0.5:
                            ratio_skip = True
                            break
            # Pattern (b): CTC assigned preceding word's <eps> to this word.
            # This happens when Phase 1 (merge_short_silences) already merged
            # the <eps> into the previous word AND when the <eps> is still
            # visible between the two words.  In both cases the inflated CTC
            # duration is from absorbed silence, not actual speech compression.
            if not ratio_skip and ctc_start < mfa_start - 0.02:
                # Case 1: <eps> still visible between prev word and this word
                gap_sil = any(
                    is_silence(iv.text)
                    and iv.xmin >= prev_end - 0.01
                    and iv.xmax <= mfa_start + 0.02
                    and iv.xmax - iv.xmin > 0.03
                    for iv in words_tier.intervals
                )
                if gap_sil:
                    ratio_skip = True
        # Regr. Case 41: absolute duration guard against CTC anchor inflation.
        # CTC anchors can span large unlabeled silences (e.g. 5.6 s le5).
        # When CTC duration is > 3 s but MFA duration is < 1 s for a Chinese
        # word, the CTC anchor is clearly inflated — trust MFA boundaries.
        # Also extend ratio_skip: when CTC end is > 500 ms past MFA end,
        # the excess is almost certainly silence, not speech.
        if use_mfa and not ratio_skip and ctc_dur > 3.0 and mfa_dur < 1.0 \
           and not is_english_token(mfa_iv.text) and not is_nvv_token(mfa_iv.text):
            ratio_skip = True
        if use_mfa and not ratio_skip and ctc_end > mfa_end + 0.5 \
           and mfa_dur < 1.0 \
           and not is_english_token(mfa_iv.text) and not is_nvv_token(mfa_iv.text):
            ratio_skip = True

        if use_mfa and not has_mfa_phone_evidence and not ratio_skip \
           and (mfa_dur > ctc_dur * 2.0 or ctc_dur > mfa_dur * 2.0):
            use_mfa = False

        if use_mfa:
            word_start = mfa_start
            word_end = mfa_end
            # 差异较大时用中间点: 前半间隙归前词, 后半间隙归当前词
            if start_diff > 0.15:
                word_start = round((ctc_start + mfa_start) / 2, 3)
            if end_diff > 0.15:
                # MFA thinks word ends sooner than CTC (trailing <eps>/silence).
                # If MFA's trailing silence is followed by punctuation within
                # 500ms, keep MFA's end so the gap can be absorbed by the punct
                # instead of being snapped back into the word via midpoint.
                keep_mfa_end = False
                if mfa_end < ctc_end and punct_entries:
                    has_trailing_sil = any(
                        is_silence(iv.text)
                        and iv.xmin >= mfa_end - 0.01
                        and iv.xmin < ctc_end + 0.05
                        for iv in words_tier.intervals
                    )
                    if has_trailing_sil:
                        for p in punct_entries:
                            if mfa_end <= p["start_s"] <= mfa_end + 0.5:
                                keep_mfa_end = True
                                break
                if not keep_mfa_end:
                    word_end = round((ctc_end + mfa_end) / 2, 3)
            # MFA 把词放在长静音之后, CTC 说更早 -> 取标点之后的纯静音间隙
            # 如果纯静音间隙 > 100ms, 优先用 CTC 起点。
            # 但如果间隙中有标点，静音应归标点处理（_inject_punctuation），
            # 不应通过 SILENCE_GAP_SNAP_THRESH 把词首拉到 CTC。
            SILENCE_GAP_SNAP_THRESH = 0.10
            if mfa_start > ctc_start and start_diff <= snap_threshold:
                gap_start = prev_end
                has_punct_in_gap = False
                if punct_entries:
                    for p in punct_entries:
                        if p["start_s"] < mfa_start and p["end_s"] > prev_end:
                            gap_start = max(gap_start, p["end_s"])
                            has_punct_in_gap = True
                if has_punct_in_gap:
                    pass  # punct handles silence placement
                else:
                    pure_silence_gap = mfa_start - gap_start
                    if pure_silence_gap > SILENCE_GAP_SNAP_THRESH:
                        word_start = max(ctc_start, gap_start)
        else:
            word_start = ctc_start
            word_end = ctc_end

        # Reconcile the selected MFA spans against the neighbouring CTC
        # boundary even when they do not literally overlap.  A previous MFA
        # end can be later than the CTC end while the current MFA start is
        # exactly that late end; checking only ``word_start < prev_end`` would
        # miss this case and leave the CTC word squeezed or shifted right.
        if (ctc is not None and prev_ctc_end > prev_ctc_start + AXIS_EPS
                and len(new_word_ivs) >= 1
                and new_word_ivs[-1][3] == "word"):
            previous_entry = new_word_ivs[-1]
            if prev_ctc_end <= ctc_start + AXIS_EPS:
                ctc_prev_end = prev_ctc_end
                ctc_current_start = ctc_start
            else:
                ctc_boundary = (prev_ctc_end + ctc_start) / 2.0
                ctc_prev_end = ctc_boundary
                ctc_current_start = ctc_boundary
            needs_ctc_boundary = (
                previous_entry[1] > ctc_prev_end + AXIS_EPS
                or word_start > ctc_current_start + AXIS_EPS
                or (prev_ctc_end > ctc_start + AXIS_EPS
                    and word_start < ctc_current_start - AXIS_EPS))
            if needs_ctc_boundary and ctc_prev_end > previous_entry[0] + AXIS_EPS:
                new_word_ivs[-1] = (
                    previous_entry[0], ctc_prev_end,
                    previous_entry[2], previous_entry[3])
                prev_end = ctc_prev_end
                if ctc_authority:
                    ctc_authority[-1]["resolved_span"][1] = float(ctc_prev_end)
                word_start = ctc_current_start
                # Once the neighbouring boundary has required CTC
                # compensation, do not let the current MFA tail re-expand
                # through the same CTC-owned word span.
                if word_end > ctc_end + AXIS_EPS:
                    word_end = ctc_end

        # 防止词间重叠: start 不能在前一词 end 之前
        # NVV: 缩短前词尾让路（NVV 无 MFA 声学模型，CTC 是唯一依据）
        # English/Chinese normally push forward (MFA boundaries have acoustic
        # evidence), BUT when the previous word was extended beyond its CTC end
        # by a silence merge (prev_end > prev_ctc_end AND a real CTC gap),
        # the extra length is silence — shorten the previous word instead of
        # squeezing the current one.
        # Regr. Case 38: zero-tolerance for overlaps — MFA/CTC boundary
        # resolution must produce contiguous intervals.  Any overlap,
        # even sub-frame (≤ 2 ms), is resolved by the same logic that
        # handles larger overlaps: NVV pushes into prev word, English
        # and Chinese snap to the prev word's end.
        if word_start < prev_end:
            # CTC is the word-level anchor.  The old fallback simply assigned
            # ``word_start = prev_end`` whenever MFA and CTC overlapped.  If
            # the previous word retained an MFA end, that silently moved the
            # current CTC word to the right and could squeeze it to a few
            # milliseconds (for example wan3/fan4).  Resolve the pair at the
            # CTC boundary instead: an ordered pair preserves its CTC gap,
            # while overlapping CTC spans are split at their midpoint.
            ctc_pair_resolved = False
            if (ctc is not None and prev_ctc_end > prev_ctc_start + AXIS_EPS
                    and len(new_word_ivs) >= 1
                    and new_word_ivs[-1][3] == "word"):
                if prev_ctc_end <= ctc_start + AXIS_EPS:
                    ctc_prev_end = prev_ctc_end
                    ctc_current_start = ctc_start
                else:
                    ctc_boundary = (prev_ctc_end + ctc_start) / 2.0
                    ctc_prev_end = ctc_boundary
                    ctc_current_start = ctc_boundary
                previous_entry = new_word_ivs[-1]
                if ctc_prev_end > previous_entry[0] + AXIS_EPS:
                    new_word_ivs[-1] = (
                        previous_entry[0], ctc_prev_end,
                        previous_entry[2], previous_entry[3])
                    prev_end = ctc_prev_end
                    if ctc_authority:
                        ctc_authority[-1]["resolved_span"][1] = float(ctc_prev_end)
                word_start = max(word_start, ctc_current_start)
                ctc_pair_resolved = True

            if not ctc_pair_resolved:
                prev_was_silence_extended = (
                    prev_end > prev_ctc_end + 0.10  # >100ms silence extension
                    and not is_nvv_token(mfa_iv.text)
                    and not is_english_token(mfa_iv.text)
                )
                if is_nvv_token(mfa_iv.text) or prev_was_silence_extended:
                    if len(new_word_ivs) >= 1 and new_word_ivs[-1][3] == "word":
                        prev_entry = new_word_ivs[-1]
                        new_prev_end = max(word_start - 0.005, prev_entry[0] + 0.010)
                        if new_prev_end > prev_entry[0]:
                            new_word_ivs[-1] = (prev_entry[0], new_prev_end, prev_entry[2], prev_entry[3])
                            prev_end = new_prev_end
                            if ctc_authority:
                                ctc_authority[-1]["resolved_span"][1] = float(new_prev_end)
                        else:
                            word_start = prev_end
                    else:
                        word_start = prev_end
                else:
                    word_start = prev_end

        # Guard against inverted intervals: when overlap fix pushes word_start
        # past word_end (prev word CTC-snapped longer than current word's MFA
        # end), extend word_end to preserve the word with at least its MFA
        # duration or a 30 ms floor.
        if word_end <= word_start:
            word_end = word_start + max(mfa_dur, 0.030)

        # ── Gap absorption (ORDER CRITICAL — do not reorder) ──
        # 1. NVV absorption into preceding gap (paralinguistic)
        # 2. CTC-snap gap fill (boundary artifact from duration-ratio fix)
        # 3. Remaining gap -> silence label <spN>
        # NVV absorption MUST run first: it uses the original gap before
        # CTC-snap fill modifies prev_end.
        nvv_extended = False
        nvv_gap = word_start - prev_end
        nvv_has_punct = False
        if nvv_gap > 0.005 and punct_entries:
            for p in punct_entries:
                if prev_end <= p["start_s"] < word_start:
                    nvv_has_punct = True
                    break
        if is_nvv_token(mfa_iv.text) and prev_end > 0.01 \
           and 0.005 < nvv_gap <= 0.2 and not nvv_has_punct:
            nvv_extended = True
            word_start = prev_end

        # CTC-snap 间隙吸收: 当前词被 CTC snap (use_mfa=False) 时,
        # 前词 (MFA 信任) 与当前词之间的小间隙吸收到前词尾。
        # 场景: MFA 压缩了前词、拉伸了当前词, duration-ratio 只修正了当前词,
        #       留下的小间隙应归前词 (而非插入静音 <sp0>)。
        if (not use_mfa and not nvv_extended
              and len(new_word_ivs) >= 1
              and new_word_ivs[-1][3] == "word"
              and word_start > prev_end + 0.005):
            gap_dur = word_start - prev_end
            prev_ctc_dur = prev_ctc_end - prev_ctc_start if prev_ctc_end > prev_ctc_start else 0.001
            if gap_dur <= 0.2 and not nvv_has_punct:
                # Extend previous word's end to absorb the gap
                prev_entry = new_word_ivs[-1]
                new_prev_end = word_start
                extended_dur = new_prev_end - prev_entry[0]
                if extended_dur <= prev_ctc_dur * 2.0:
                    new_word_ivs[-1] = (prev_entry[0], new_prev_end, prev_entry[2], prev_entry[3])
                    prev_end = new_prev_end

        # Silence gap: use actual boundary gap (not CTC gap)
        actual_gap = word_start - prev_end
        if actual_gap > 0.005:
            dur_label = silence_label(actual_gap)
            new_word_ivs.append((prev_end, word_start, dur_label, "gap"))
            if pp_tier is not None:
                for p_iv in pp_tier.intervals:
                    if p_iv.xmax > prev_end and p_iv.xmin < word_start \
                       and is_silence(p_iv.text):
                        new_phone_ivs.append((
                            max(p_iv.xmin, prev_end),
                            min(p_iv.xmax, word_start),
                            p_iv.text))

        # Word
        new_word_ivs.append((word_start, word_end, mfa_iv.text, "word"))

        # Keep the boundary decision alive after this function returns.  The
        # interval geometry is later rebuilt several times; without this
        # provenance, energy/punctuation/normalisation passes cannot tell a
        # CTC-authoritative word from an MFA-authoritative one.
        ctc_authority.append({
            "lexical_ordinal": len(ctc_authority),
            "text": mfa_iv.text.strip(),
            "boundary_source": "mfa" if use_mfa else "ctc",
            "ctc_lexical_ordinal": ctc_word_lexical_ordinal.get(id(ctc)),
            "ctc_span": [float(ctc_start), float(ctc_end)],
            "raw_ctc_span": [float(ctc_start), float(ctc_end_raw)],
            "mfa_span": [float(mfa_start), float(mfa_end)],
            "resolved_span": [float(word_start), float(word_end)],
            "initial_resolved_span": [float(word_start), float(word_end)],
            "published_span": None,
            "operations": [],
        })

        # Phones: NVV 被扩展时同步扩展首音素; snap 到 CTC 时等比映射; 否则保留 MFA
        if pp_tier is not None:
            if nvv_extended:
                # NVV 词 start 被延伸, 首音素也延伸到 word_start
                first_phone = True
                for p_iv in pp_tier.intervals:
                    if p_iv.xmax > mfa_start and p_iv.xmin < mfa_end:
                        if first_phone:
                            new_phone_ivs.append((word_start, p_iv.xmax, p_iv.text))
                            first_phone = False
                        else:
                            new_phone_ivs.append((p_iv.xmin, p_iv.xmax, p_iv.text))
            elif not use_mfa and mfa_dur > 0:
                for p_iv in pp_tier.intervals:
                    if p_iv.xmax > mfa_start and p_iv.xmin < mfa_end:
                        rel_start = (max(p_iv.xmin, mfa_start) - mfa_start) / mfa_dur
                        rel_end = (min(p_iv.xmax, mfa_end) - mfa_start) / mfa_dur
                        new_phone_ivs.append((
                            ctc_start + rel_start * (ctc_end - ctc_start),
                            ctc_start + rel_end * (ctc_end - ctc_start),
                            p_iv.text))
            else:
                for p_iv in pp_tier.intervals:
                    if p_iv.xmax > mfa_start and p_iv.xmin < mfa_end:
                        new_phone_ivs.append((p_iv.xmin, p_iv.xmax, p_iv.text))

        prev_end = word_end
        prev_ctc_start = ctc_start
        prev_ctc_end = ctc_end

    # ── Post-loop contiguity pass ──
    # Adjacent words may independently choose MFA vs CTC boundaries.
    # When word N trusts MFA (xmax = mfa_end) and word N+1 snaps to CTC
    # (xmin = ctc_start), a gap > 20 ms can open.  This pass catches
    # remaining gaps between adjacent content words that the per-word
    # gap absorption (above) missed.  Threshold matches the QC filter
    # at _WT_GAP_THRESHOLD_S = 0.020.
    _WT_GAP_LIMIT = 0.020
    for _gi in range(len(new_word_ivs) - 1):
        cur = new_word_ivs[_gi]
        nxt = new_word_ivs[_gi + 1]
        if cur[3] != "word" or nxt[3] != "word":
            continue
        _gap = nxt[0] - cur[1]
        if _gap > _WT_GAP_LIMIT:
            # Absorb into the longer word
            if cur[1] - cur[0] >= nxt[1] - nxt[0]:
                new_word_ivs[_gi] = (cur[0], nxt[0], cur[2], cur[3])
            else:
                new_word_ivs[_gi + 1] = (cur[1], nxt[1], nxt[2], nxt[3])
        elif _gap < 0 and _gap > -0.005:
            # Tiny overlap: split at midpoint
            mid = (cur[1] + nxt[0]) / 2.0
            new_word_ivs[_gi] = (cur[0], mid, cur[2], cur[3])
            new_word_ivs[_gi + 1] = (mid, nxt[1], nxt[2], nxt[3])

    # Leading silence — from 0 to first word start (mirrors trailing silence)
    if new_word_ivs and new_word_ivs[0][0] > 0.005:
        dur_label = silence_label(new_word_ivs[0][0])
        new_word_ivs.insert(0, (0.0, new_word_ivs[0][0], dur_label, "gap"))

    # Trailing silence — from last word end to total duration
    total_dur = words_tier.xmax
    if total_dur > prev_end + 0.005:
        dur_label = silence_label(total_dur - prev_end)
        new_word_ivs.append((prev_end, total_dur, dur_label, "gap"))
        if pp_tier is not None:
            for p_iv in pp_tier.intervals:
                if p_iv.xmin >= prev_end and is_silence(p_iv.text):
                    new_phone_ivs.append((p_iv.xmin, p_iv.xmax, p_iv.text))

    # Merge adjacent same-text phone intervals (MFA bleed across boundaries)
    merged_pp = []
    for item in sorted(new_phone_ivs):
        if merged_pp and merged_pp[-1][2] == item[2] and abs(merged_pp[-1][1] - item[0]) < 0.002:
            merged_pp[-1] = (merged_pp[-1][0], item[1], item[2])
        else:
            merged_pp.append(item)
    new_phone_ivs = merged_pp

    # Preserve positive word gaps here.  ``_snap_to_ctc`` has no source
    # ownership proof, so a 10--30 ms residual must reach the explicit
    # evidence decision in ``_sync_derived_tiers`` (or strict QC).  Only
    # tiny overlaps are structurally repaired at this stage.
    for k in range(len(new_word_ivs) - 1, 0, -1):
        cur = new_word_ivs[k]
        prev = new_word_ivs[k - 1]
        gap = cur[0] - prev[1]
        if gap < 0 and gap >= -0.005 and prev[3] == "word":
            # Tiny overlap — split at midpoint (only word-word pairs)
            mid = (prev[1] + cur[0]) / 2.0
            new_word_ivs[k - 1] = (prev[0], mid, prev[2], prev[3])
            new_word_ivs[k] = (mid, cur[1], cur[2], cur[3])

    # Build new tiers
    new_words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax,
                          [Interval(s, e, t) for s, e, t, _ in new_word_ivs])
    if hasattr(words_tier, "_fallback_unknown_projection"):
        new_words_tier._fallback_unknown_projection = (
            words_tier._fallback_unknown_projection)
    # The contiguity passes above may adjust a word after its initial
    # decision was recorded.  Make the accepted post-snap geometry the
    # authority snapshot; otherwise the final barrier would resurrect the
    # pre-contiguity span and undo this CTC compensation.
    _final_lexical = [iv for iv in new_words_tier.intervals
                      if iv.text.strip() and not is_silence(iv.text)
                      and not is_punct(iv.text)]
    if len(_final_lexical) == len(ctc_authority):
        for _entry, _iv in zip(ctc_authority, _final_lexical):
            _entry["resolved_span"] = [float(_iv.xmin), float(_iv.xmax)]
            # Neighbour compensation can force an originally MFA-selected
            # word back onto its CTC span.  Keep provenance truthful: a final
            # interval with no MFA overlap but positive CTC overlap is now a
            # CTC-owned decision, so later audits and barriers must treat it
            # as such instead of rejecting the already-correct CTC geometry.
            if _entry.get("boundary_source") == "mfa":
                mfa_span = _entry.get("mfa_span")
                ctc_span = _entry.get("ctc_span")
                if (isinstance(mfa_span, (list, tuple)) and len(mfa_span) == 2
                        and isinstance(ctc_span, (list, tuple)) and len(ctc_span) == 2):
                    mfa_overlap = min(_iv.xmax, float(mfa_span[1])) - max(
                        _iv.xmin, float(mfa_span[0]))
                    ctc_overlap = min(_iv.xmax, float(ctc_span[1])) - max(
                        _iv.xmin, float(ctc_span[0]))
                    ctc_compensated = (
                        ctc_overlap > AXIS_EPS
                        and (abs(_iv.xmin - float(mfa_span[0])) > AXIS_EPS
                             or abs(_iv.xmax - float(mfa_span[1])) > AXIS_EPS))
                    if ctc_compensated and (
                            mfa_overlap <= AXIS_EPS
                            or _iv.xmin < float(mfa_span[0]) - AXIS_EPS
                            or _iv.xmax < float(mfa_span[1]) - AXIS_EPS):
                        _entry["boundary_source"] = "ctc"
                        _entry["arbitration"] = "ctc_neighbor_compensation"
    new_words_tier._ctc_word_authority = ctc_authority

    new_pp_tier = None
    if pp_tier is not None and new_phone_ivs:
        new_pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax,
                           [Interval(s, e, t) for s, e, t in new_phone_ivs])

    return new_words_tier, new_pp_tier


# ---------------------------------------------------------------------------
# English MFA phone integration
# ---------------------------------------------------------------------------

def load_en_phones(stem: str, en_phones_dir: Path | None) -> list[dict] | None:
    """Load English MFA phone alignment data for *stem*.

    Returns None when no data is available (file missing, empty, or dir unset).
    The caller must handle None gracefully: skip English phone injection entirely.
    """
    if en_phones_dir is None or not en_phones_dir.exists():
        return None
    path = en_phones_dir / f"{stem}_en_phones.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data:
            return None
        # New English MFA runs persist a strict-en-mfa-v1 ledger object,
        # while the legacy injection path consumes a flat list.  Normalize
        # the ledger here so disabling the independent strict-ok audit (which
        # is required for no-reference ASR jobs) does not disable English
        # phone injection or iterate over dictionary keys as entries.
        if isinstance(data, dict) and data.get("schema") == "strict-en-mfa-v1":
            normalized: list[dict] = []
            for segment in data.get("segments", []):
                if not isinstance(segment, dict):
                    continue
                words = segment.get("words", [])
                if not isinstance(words, list):
                    continue
                valid_words = [word for word in words
                               if isinstance(word, dict)
                               and isinstance(word.get("start"), (int, float))
                               and isinstance(word.get("mfa_word"), dict)
                               and isinstance(word["mfa_word"].get("start"), (int, float))]
                if not valid_words:
                    continue
                offset = float(valid_words[0]["start"]) - float(
                    valid_words[0]["mfa_word"]["start"])
                for word in words:
                    if not isinstance(word, dict):
                        continue
                    text = str(word.get("ctc_text", "")).strip()
                    mfa_word = word.get("mfa_word")
                    if (not text or not isinstance(mfa_word, dict)
                            or not isinstance(word.get("start"), (int, float))
                            or not isinstance(word.get("end"), (int, float))):
                        continue
                    en_start = offset + float(mfa_word.get("start", 0.0))
                    en_end = offset + float(mfa_word.get("end", 0.0))
                    phones = []
                    for phone in word.get("phones", []):
                        if not isinstance(phone, dict):
                            continue
                        label = str(phone.get("label", phone.get("phone", ""))).strip()
                        if not label:
                            continue
                        phones.append({
                            "phone": label,
                            "start": offset + float(phone.get("start", 0.0)),
                            "end": offset + float(phone.get("end", 0.0)),
                        })
                    normalized.append({
                        "seg_idx": segment.get("segment_ordinal", 0),
                        "offset": offset,
                        "word_text": text,
                        "word_start": float(word["start"]),
                        "word_end": float(word["end"]),
                        "en_word_start": en_start,
                        "en_word_end": en_end,
                        "phones": phones,
                    })
            return normalized or None
        if not isinstance(data, list):
            return None
        return data
    except Exception:
        return None


def _strict_en_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_en_report(status: str, required_words: int = 0, verified_words: int = 0,
                      failed_word_ids: list[str] | None = None, ledger_sha256: str = "",
                      reason: str = "") -> dict:
    """Return the fixed strict-English report shape on every outcome."""
    result = {"status": status, "required_words": int(required_words),
              "verified_words": int(verified_words),
              "failed_word_ids": list(failed_word_ids or []),
              "ledger_sha256": ledger_sha256}
    if reason:
        result["reason"] = reason
    return result


def _strict_en_fail(required_words: int, reason: str, *, ledger_sha256: str = "",
                    failed_word_ids: list[str] | None = None) -> tuple[dict, list[tuple[Interval, dict]]]:
    return (_strict_en_report("rejected", required_words, 0, failed_word_ids,
                              ledger_sha256, reason), [])


def _source_unknown_context(words_tier: Tier) -> list[dict]:
    """Snapshot source words needed by the narrow initial-Mira proof."""
    return [{"ordinal": ordinal, "start": float(iv.xmin),
             "end": float(iv.xmax), "text": iv.text.strip()}
            for ordinal, iv in enumerate(words_tier.intervals)]


def _fallback_punctuation_surface_ledger(raw_text: str) -> dict:
    """Seal fallback punctuation as an independent, semantic surface proof.

    The fallback transcript is allowed to provide display punctuation only.
    Its lexical sequence and its timestamp-free punctuation boundaries are
    therefore recorded before any derived tier is rebuilt.  The words/hanzi
    tiers are deliberately not inputs to this ledger.
    """
    source_text = str(raw_text or "")
    lexical_boundary = 0
    punctuation: list[dict] = []
    for source_index, unit in enumerate(_extract_word_chars(source_text)):
        if is_word_like(unit):
            lexical_boundary += 1
        elif is_punct(unit):
            punctuation.append({
                "source_index": source_index,
                "lexical_boundary": lexical_boundary,
                "label": unit,
            })
    payload = {
        "schema": FALLBACK_SURFACE_SCHEMA,
        "source_text": source_text,
        "source_digest": hashlib.sha256(
            source_text.encode("utf-8")).hexdigest(),
        "lexical_count": lexical_boundary,
        "punctuation": punctuation,
    }
    payload["digest"] = _evidence_digest(payload)
    return payload


def _validate_fallback_punctuation_surface_ledger(
        ledger: dict | None, raw_text: str | None = None) -> tuple[bool, dict]:
    """Validate the sealed fallback surface ledger without trusting its flags."""
    reasons: list[str] = []
    expected: dict | None = None
    if not isinstance(ledger, dict):
        reasons.append("missing")
    else:
        if ledger.get("schema") != FALLBACK_SURFACE_SCHEMA:
            reasons.append("schema_mismatch")
        digest = ledger.get("digest")
        if not isinstance(digest, str) or not digest:
            reasons.append("digest_missing")
        elif digest != _evidence_digest({
                key: value for key, value in ledger.items() if key != "digest"}):
            reasons.append("digest_mismatch")
        source_text = ledger.get("source_text")
        if not isinstance(source_text, str):
            reasons.append("source_text_missing")
        else:
            expected = _fallback_punctuation_surface_ledger(source_text)
            for key in ("source_text", "source_digest", "lexical_count",
                        "punctuation"):
                if ledger.get(key) != expected.get(key):
                    reasons.append(f"{key}_mismatch")
            if raw_text is not None and source_text != str(raw_text or ""):
                reasons.append("source_text_mismatch")
    return not reasons, {
        "schema": FALLBACK_SURFACE_SCHEMA,
        "status": "verified" if not reasons else "rejected",
        "reasons": reasons,
        "source_digest": ledger.get("source_digest")
        if isinstance(ledger, dict) else None,
        "ledger_digest": ledger.get("digest")
        if isinstance(ledger, dict) else None,
    }


# Short aliases keep the source proof easy to discover for audit/test callers.
_build_fallback_punctuation_surface_ledger = _fallback_punctuation_surface_ledger
_validate_fallback_surface_ledger = _validate_fallback_punctuation_surface_ledger


def _fallback_lexical_items(source_words: list[dict] | None,
                            ctc_tokens: list[dict] | None,
                            final_words: Tier | None) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Normalize the three fallback lexical streams for correspondence.

    This deliberately keeps raw source ordinals and final lexical ordinals in
    every item.  The fallback transcript is not included: it can describe a
    useful display surface, but it is never evidence for ownership.
    """
    errors: list[str] = []
    source: list[dict] = []
    for index, item in enumerate(source_words or []):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            errors.append(f"source_malformed:{index}")
            continue
        text = item["text"].strip()
        if not text or is_silence(text) or is_punct(text):
            continue
        try:
            ordinal = int(item.get("ordinal", index))
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"source_malformed:{index}")
            continue
        if ordinal != index or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            errors.append(f"source_malformed:{index}")
            continue
        source.append({"raw_ordinal": ordinal, "lexical_ordinal": len(source),
                       "text": text, "identity": _lexical_identity(text),
                       "start": start, "end": end})

    ctc: list[dict] = []
    for index, item in enumerate(ctc_tokens or []):
        if not isinstance(item, dict):
            errors.append(f"ctc_malformed:{index}")
            continue
        item_type = item.get("type", "word")
        if item_type != "word":
            continue
        if not isinstance(item.get("word"), str) or not item["word"].strip():
            errors.append(f"ctc_malformed:{index}")
            continue
        try:
            ordinal = int(item.get("ordinal", index))
            start, end = float(item["start_s"]), float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"ctc_malformed:{index}")
            continue
        if (ordinal != index or not math.isfinite(start) or not math.isfinite(end)
                or end <= start):
            errors.append(f"ctc_malformed:{index}")
            continue
        text = item["word"].strip()
        ctc.append({"list_ordinal": index, "ordinal": ordinal,
                    "lexical_ordinal": len(ctc), "text": text,
                    "identity": _lexical_identity(text, ctc_item=item),
                    "start": start, "end": end})

    final: list[dict] = []
    if final_words is None:
        errors.append("final_missing")
    else:
        for index, interval in enumerate(final_words.intervals):
            text = (interval.text or "").strip()
            if not text or is_silence(text) or is_punct(text):
                continue
            if not math.isfinite(interval.xmin) or not math.isfinite(interval.xmax) \
                    or interval.xmax <= interval.xmin:
                errors.append(f"final_malformed:{index}")
                continue
            final.append({"interval_ordinal": index,
                          "lexical_ordinal": len(final), "text": text,
                          "identity": _lexical_identity(text),
                          "start": float(interval.xmin),
                          "end": float(interval.xmax)})
    return source, ctc, final, errors


def _fallback_exact_source_ctc_projection(
        source: list[dict], ctc: list[dict]) -> dict:
    """Project source lexical owners onto CTC words exactly once, in order.

    A known source word may be explicitly omitted, but an unknown placeholder
    may only consume a known NVV/English CTC word.  The dynamic program keeps
    both choices for known words and counts complete monotone solutions; a
    projection is usable only when that count is exactly one.  Time spans are
    carried as evidence after the lexical solution and never participate in
    selecting an owner.
    """
    @lru_cache(maxsize=None)
    def _solve(source_index: int, ctc_index: int) -> tuple[int, tuple[int | None, ...] | None]:
        if source_index == len(source):
            return ((1, ()) if ctc_index == len(ctc) else (0, None))

        source_item = source[source_index]
        unknown = is_unknown_token(source_item["text"])
        choices: list[tuple[int, tuple[int | None, ...] | None]] = []
        if ctc_index < len(ctc):
            ctc_item = ctc[ctc_index]
            compatible = (
                (ctc_item["identity"] == source_item["identity"])
                if not unknown else
                (_canonical_nvv_identity(ctc_item["text"]) is not None
                 or is_english_token(ctc_item["text"])))
            if compatible:
                count, path = _solve(source_index + 1, ctc_index + 1)
                if count:
                    choices.append((count, (ctc_index,) + path))
        if not unknown:
            count, path = _solve(source_index + 1, ctc_index)
            if count:
                choices.append((count, (None,) + path))

        total = min(2, sum(count for count, _ in choices))
        if total != 1:
            return total, None
        return 1, next(path for count, path in choices if count == 1)

    solution_count, path = _solve(0, 0)
    projection = {
        "schema": "fallback-source-ctc-projection-v1",
        "status": "rejected" if solution_count != 1 else (
            "omitted" if any(item is None for item in path or ()) else "mapped"),
        "safe": solution_count == 1,
        "solution_count": solution_count,
        "source_count": len(source),
        "ctc_count": len(ctc),
        "entries": [],
        "first_mismatch": None,
    }

    if solution_count != 1:
        projection["first_mismatch"] = {
            "reason": ("non_unique_projection" if solution_count > 1
                       else "no_complete_projection"),
        }
        if solution_count == 0:
            # Add a useful fail-closed diagnostic without using timing to
            # repair the failed lexical projection.
            for source_index, source_item in enumerate(source):
                if is_unknown_token(source_item["text"]):
                    remaining = ctc[source_index:] if source_index < len(ctc) else []
                    if not any(_canonical_nvv_identity(item["text"]) is not None
                               or is_english_token(item["text"])
                               for item in remaining):
                        projection["first_mismatch"] = {
                            "reason": "unknown_target_not_known_nvv_or_english",
                            "source_ordinal": source_item["raw_ordinal"],
                        }
                        break
        path = tuple(None for _ in source)

    for source_item, ctc_index in zip(source, path):
        ctc_item = ctc[ctc_index] if ctc_index is not None else None
        entry = {
            "source_ordinal": source_item["raw_ordinal"],
            "source_lexical_ordinal": source_item["lexical_ordinal"],
            "source_text": source_item["text"],
            "source_identity": source_item["identity"],
            "source_span": [source_item["start"], source_item["end"]],
            "ctc_ordinal": ctc_item["ordinal"] if ctc_item else None,
            "ctc_lexical_ordinal": ctc_item["lexical_ordinal"] if ctc_item else None,
            "ctc_text": ctc_item["text"] if ctc_item else None,
            "ctc_identity": ctc_item["identity"] if ctc_item else None,
            "ctc_span": ([ctc_item["start"], ctc_item["end"]]
                         if ctc_item else None),
            "status": "mapped" if ctc_item is not None and solution_count == 1
                      else "omitted" if ctc_item is None and not is_unknown_token(source_item["text"])
                      else "mismatch",
            "reason": ("exact_ordered_correspondence" if ctc_item is not None
                       and solution_count == 1 else "source_omitted"
                       if ctc_item is None and not is_unknown_token(source_item["text"])
                       else projection["first_mismatch"]["reason"]),
        }
        if entry["status"] == "omitted":
            entry["omission_evidence"] = {
                "source_identity": source_item["identity"],
                "reason": "known_source_absent_from_unique_projection",
            }
        projection["entries"].append(entry)

    projection["digest"] = _evidence_digest(projection)
    return projection


def _fallback_lexical_correspondence_ledger(
        source_words: list[dict] | None,
        ctc_tokens: list[dict] | None,
        final_words: Tier | None) -> dict:
    """Build the complete fail-closed source→CTC→final lexical ledger.

    Known lexical source omissions are represented explicitly and do not
    consume a later CTC/final owner.  Every other deviation is retained in
    the ledger and makes it unsafe: reorder, substitution, final-only/CTC-only
    evidence, malformed evidence, and ambiguous ownership are never repaired
    by position.
    """
    source, ctc, final, malformed = _fallback_lexical_items(
        source_words, ctc_tokens, final_words)
    entries: list[dict] = []
    omissions: list[dict] = []
    first_mismatch: dict | None = None

    def reject(reason: str, **details: object) -> None:
        nonlocal first_mismatch
        if first_mismatch is None:
            first_mismatch = {"reason": reason, **details}

    status = "mapped"
    safe = not malformed
    if malformed:
        status = "rejected"
        reject("malformed_evidence", details=list(malformed))

    # CTC and final are both post-MFA evidence.  They must be an exact ordered
    # stream before source omissions or unknown recovery can be considered.
    if len(ctc) != len(final):
        status = "rejected"
        safe = False
        reject("final_only" if len(final) > len(ctc) else "ctc_only",
               ctc_index=min(len(ctc), len(final)),
               ctc_remaining=max(0, len(ctc) - len(final)),
               final_remaining=max(0, len(final) - len(ctc)))
    elif any(left["identity"] != right["identity"]
             for left, right in zip(ctc, final)):
        status = "rejected"
        safe = False
        for index, (left, right) in enumerate(zip(ctc, final)):
            if left["identity"] != right["identity"]:
                reject("substitution", ctc_ordinal=left["ordinal"],
                       final_lexical_ordinal=right["lexical_ordinal"],
                       ctc_text=left["text"], final_text=right["text"])
                break
    for left, right in zip(final, final[1:]):
        if left["end"] > right["start"] + AXIS_EPS:
            status = "rejected"
            safe = False
            reject("non_unique_ownership",
                   final_lexical_ordinal=right["lexical_ordinal"],
                   overlapping_with=left["lexical_ordinal"])
            break

    stream_exact = (len(ctc) == len(final)
                    and not any(left["identity"] != right["identity"]
                                for left, right in zip(ctc, final)))
    projection = _fallback_exact_source_ctc_projection(source, ctc)
    ctc_cursor = len(ctc) if stream_exact else 0
    if stream_exact:
        for projection_entry in projection["entries"]:
            entry = dict(projection_entry)
            ctc_ordinal = entry.get("ctc_lexical_ordinal")
            final_item = (final[ctc_ordinal] if isinstance(ctc_ordinal, int)
                          and ctc_ordinal < len(final) else None)
            entry.update({
                "final_lexical_ordinal": final_item["lexical_ordinal"] if final_item else None,
                "final_interval_ordinal": final_item["interval_ordinal"] if final_item else None,
                "final_text": final_item["text"] if final_item else None,
                "final_span": ([final_item["start"], final_item["end"]]
                               if final_item else None),
                "resolved_text": final_item["text"] if final_item else None,
            })
            if entry["status"] == "omitted":
                omissions.append(entry.copy())
            entries.append(entry)
        if not projection["safe"]:
            status = "rejected"
            safe = False
            reject(projection["first_mismatch"]["reason"],
                   **{key: value for key, value in projection["first_mismatch"].items()
                      if key != "reason"})
    else:
        # Keep detailed mismatch entries for the independent CTC/final gate;
        # source projection is not allowed to redeem a partial or substituted
        # final stream.
        ctc_cursor = 0
        for source_item in source:
            ctc_item = ctc[ctc_cursor] if ctc_cursor < len(ctc) else None
            final_item = final[ctc_cursor] if ctc_cursor < len(final) else None
            entry = {
                "source_ordinal": source_item["raw_ordinal"],
                "source_lexical_ordinal": source_item["lexical_ordinal"],
                "source_text": source_item["text"],
                "ctc_ordinal": ctc_item["ordinal"] if ctc_item else None,
                "ctc_lexical_ordinal": ctc_item["lexical_ordinal"] if ctc_item else None,
                "ctc_text": ctc_item["text"] if ctc_item else None,
                "final_lexical_ordinal": final_item["lexical_ordinal"] if final_item else None,
                "final_interval_ordinal": final_item["interval_ordinal"] if final_item else None,
                "final_text": final_item["text"] if final_item else None,
                "resolved_text": None,
                "status": "rejected",
                "reason": None,
            }
            if ctc_item is not None and final_item is not None:
                same_evidence = ctc_item["identity"] == final_item["identity"]
                if (source_item["identity"] == ctc_item["identity"]
                        and same_evidence):
                    entry.update(status="mapped", resolved_text=final_item["text"],
                                 reason="exact_ordered_correspondence")
                    ctc_cursor += 1
                elif (is_unknown_token(source_item["text"])
                      and same_evidence
                      and (_canonical_nvv_identity(ctc_item["text"]) is not None
                           or is_english_token(ctc_item["text"]))):
                    entry.update(status="mapped", resolved_text=final_item["text"],
                                 reason="known_nvv_or_english_unknown_recovery")
                    ctc_cursor += 1
                else:
                    safe = False
                    entry.update(status="mismatch", reason="unknown_correspondence"
                                 if is_unknown_token(source_item["text"])
                                 else "mismatch")
                    reject(entry["reason"], source_ordinal=source_item["raw_ordinal"],
                           ctc_ordinal=ctc_item["ordinal"], ctc_text=ctc_item["text"],
                           final_text=final_item["text"])
                    ctc_cursor += 1
            else:
                safe = False if is_unknown_token(source_item["text"]) else safe
                entry.update(status="mismatch" if is_unknown_token(source_item["text"])
                             else "omitted",
                             reason="unknown_unproved" if is_unknown_token(source_item["text"])
                             else "source_omitted")
                if entry["status"] == "omitted":
                    omissions.append(entry.copy())
                if is_unknown_token(source_item["text"]):
                    reject("unknown_unproved", source_ordinal=source_item["raw_ordinal"])
            entries.append(entry)

    if ctc_cursor < len(ctc):
        status = "rejected"
        safe = False
        reject("ctc_only", ctc_ordinal=ctc[ctc_cursor]["ordinal"],
               ctc_text=ctc[ctc_cursor]["text"])
        for item in ctc[ctc_cursor:]:
            entries.append({"source_ordinal": None,
                            "source_lexical_ordinal": None,
                            "source_text": None,
                            "ctc_ordinal": item["ordinal"],
                            "ctc_lexical_ordinal": item["lexical_ordinal"],
                            "ctc_text": item["text"],
                            "final_lexical_ordinal": item["lexical_ordinal"] if item["lexical_ordinal"] < len(final) else None,
                            "final_interval_ordinal": final[item["lexical_ordinal"]]["interval_ordinal"] if item["lexical_ordinal"] < len(final) else None,
                            "final_text": final[item["lexical_ordinal"]]["text"] if item["lexical_ordinal"] < len(final) else None,
                            "resolved_text": None, "status": "rejected",
                            "reason": "ctc_only"})

    if len(final) > len(ctc):
        status = "rejected"
        safe = False
        for item in final[len(ctc):]:
            entries.append({"source_ordinal": None,
                            "source_lexical_ordinal": None,
                            "source_text": None,
                            "ctc_ordinal": None,
                            "ctc_lexical_ordinal": None,
                            "ctc_text": None,
                            "final_lexical_ordinal": item["lexical_ordinal"],
                            "final_interval_ordinal": item["interval_ordinal"],
                            "final_text": item["text"],
                            "resolved_text": None, "status": "rejected",
                            "reason": "final_only"})

    # A repeated final owner can only be accepted when its CTC ordinal is the
    # unique correspondence key.  Duplicate candidates at a single key are
    # structurally impossible in the normalized stream; malformed duplicate
    # ordinals are rejected above rather than silently selecting one.
    source_to_final = {
        str(entry["source_ordinal"]): entry["final_lexical_ordinal"]
        for entry in entries if entry.get("status") == "mapped"
        and entry.get("source_ordinal") is not None
        and entry.get("final_lexical_ordinal") is not None}
    ctc_to_final = {
        str(entry["ctc_ordinal"]): entry["final_lexical_ordinal"]
        for entry in entries if entry.get("status") == "mapped"
        and entry.get("ctc_ordinal") is not None
        and entry.get("final_lexical_ordinal") is not None}
    if len(source_to_final) != len([e for e in entries if e.get("status") == "mapped"
                                    and e.get("source_ordinal") is not None]):
        status = "rejected"
        safe = False
        reject("non_unique_ownership")
    if first_mismatch is not None:
        safe = False
        status = "rejected"
    payload = {
        "schema": FALLBACK_CORRESPONDENCE_SCHEMA,
        "status": status,
        "source_count": len(source),
        "ctc_count": len(ctc),
        "final_count": len(final),
        "entries": entries,
        "omissions": omissions,
        "first_mismatch": first_mismatch,
        "mapping": {"source_to_final": source_to_final,
                    "ctc_to_final": ctc_to_final},
        "safe": bool(safe and not first_mismatch),
    }
    payload["digest"] = _evidence_digest(payload)
    return payload


_FALLBACK_QUALIFIED_PAUSE_LABELS = frozenset(
    {"<sp0>", "<sp1>", "<sp2>", "<sp3>"})


def _validate_fallback_correspondence(
        correspondence: dict | None,
        source_words: list[dict] | None,
        ctc_tokens: list[dict] | None,
        final_words: Tier | None) -> tuple[bool, dict]:
    """Validate the exact fallback ledger before it can redeem pause QC.

    ``safe`` is producer output, not an authority decision.  Recompute the
    ledger from the immutable source/CTC/final streams and bind the supplied
    payload to its digest, counts, mappings, entries, and first-mismatch state.
    Any missing or malformed field fails closed.
    """
    errors: list[str] = []
    final_interval_reindexed = False
    final_surface_normalized = False
    expected: dict | None = None
    if not isinstance(correspondence, dict):
        errors.append("missing")
    else:
        if correspondence.get("schema") != FALLBACK_CORRESPONDENCE_SCHEMA:
            errors.append("schema_mismatch")
        if correspondence.get("safe") is not True:
            errors.append("safe_false")
        if correspondence.get("first_mismatch") is not None:
            errors.append("first_mismatch_present")
        digest = correspondence.get("digest")
        if not isinstance(digest, str) or not digest:
            errors.append("digest_missing")
        elif digest != _evidence_digest(
                {key: value for key, value in correspondence.items()
                 if key != "digest"}):
            errors.append("digest_mismatch")

        try:
            expected = _fallback_lexical_correspondence_ledger(
                source_words, ctc_tokens, final_words)
        except (AttributeError, KeyError, TypeError, ValueError):
            errors.append("recompute_failed")
        if expected is not None:
            if expected.get("safe") is not True:
                errors.append("recomputed_unsafe")
            for key in ("status", "source_count", "ctc_count", "final_count",
                        "omissions", "first_mismatch", "mapping", "safe"):
                if correspondence.get(key) != expected.get(key):
                    errors.append(f"{key}_mismatch")
            supplied_entries = correspondence.get("entries")
            expected_entries = expected.get("entries")
            if (not isinstance(supplied_entries, list)
                    or not isinstance(expected_entries, list)
                    or len(supplied_entries) != len(expected_entries)):
                errors.append("entries_mismatch")
            else:
                for supplied, actual in zip(supplied_entries, expected_entries):
                    if not isinstance(supplied, dict) or not isinstance(actual, dict):
                        errors.append("entries_mismatch")
                        break
                    # Final serialization can insert a display-only leading
                    # `<sp1>`.  Its physical interval reindex is diagnostic;
                    # all lexical identities, ordinals and mappings remain
                    # mandatory and are compared below.
                    supplied_owner = {
                        key: value for key, value in supplied.items()
                        if key not in {"final_interval_ordinal", "final_text",
                                       "resolved_text"}}
                    actual_owner = {
                        key: value for key, value in actual.items()
                        if key not in {"final_interval_ordinal", "final_text",
                                       "resolved_text"}}
                    if supplied_owner != actual_owner:
                        errors.append("entries_mismatch")
                        break
                    for surface_key in ("final_text", "resolved_text"):
                        supplied_surface = supplied.get(surface_key)
                        actual_surface = actual.get(surface_key)
                        if supplied_surface == actual_surface:
                            continue
                        if (_lexical_identity(supplied_surface or "")
                                != _lexical_identity(actual_surface or "")):
                            errors.append("entries_mismatch")
                            break
                        final_surface_normalized = True
                    if errors and errors[-1] == "entries_mismatch":
                        break
                    if (supplied.get("final_interval_ordinal")
                            != actual.get("final_interval_ordinal")):
                        final_interval_reindexed = True
    return (not errors, {
        "schema": FALLBACK_CORRESPONDENCE_SCHEMA,
        "status": "verified" if not errors else "rejected",
        "reasons": errors,
        "final_interval_reindexed": final_interval_reindexed,
        "final_surface_normalized": final_surface_normalized,
        "digest": correspondence.get("digest")
        if isinstance(correspondence, dict) else None,
    })


def _fallback_pause_qualification(
        words_tier: Tier | None,
        reference_mode: str | None,
        correspondence: dict | None,
        source_words: list[dict] | None,
        ctc_tokens: list[dict] | None) -> dict:
    """Qualify each retained internal pause against its exact lexical gap.

    A valid ledger is necessary but not sufficient: the pause label must be
    canonical and both neighboring final lexical owners must be present in
    the ledger's exact CTC→final mapping.  Long ``<sp3>`` pauses additionally
    require the mapped neighboring CTC anchors to expose the same blank span;
    this distinguishes a real acoustic/CTC pause from missing ownership.
    Thus one bad pause remains a veto even when another pause qualifies.
    """
    valid_ledger, validation = _validate_fallback_correspondence(
        correspondence, source_words, ctc_tokens, words_tier)
    lexical_positions = [
        index for index, interval in enumerate(words_tier.intervals)
    ] if words_tier is not None else []
    lexical_positions = [
        index for index in lexical_positions
        if (words_tier.intervals[index].text.strip()
            and not is_silence(words_tier.intervals[index].text)
            and not is_punct(words_tier.intervals[index].text))
    ] if words_tier is not None else []
    lexical_ordinal = {index: ordinal for ordinal, index in
                       enumerate(lexical_positions)}
    mapped_final: set[int] = set()
    ctc_by_final: dict[int, dict] = {}
    if valid_ledger and isinstance(correspondence, dict):
        mapping = correspondence.get("mapping", {})
        ctc_to_final = mapping.get("ctc_to_final", {}) if isinstance(mapping, dict) else {}
        if isinstance(ctc_to_final, dict):
            mapped_final = {
                value for value in ctc_to_final.values()
                if type(value) is int
            }
            lexical_ctc = [
                item for item in (ctc_tokens or [])
                if isinstance(item, dict)
                and str(item.get("word", item.get("text", ""))).strip()
                and not is_silence(str(item.get("word", item.get("text", ""))))
                and not is_punct(str(item.get("word", item.get("text", ""))))
            ]
            for ctc_ordinal, final_ordinal in ctc_to_final.items():
                try:
                    ctc_index = int(ctc_ordinal)
                except (TypeError, ValueError):
                    continue
                if (type(final_ordinal) is int
                        and 0 <= ctc_index < len(lexical_ctc)):
                    ctc_by_final[final_ordinal] = lexical_ctc[ctc_index]

    pauses: list[dict] = []
    if words_tier is not None:
        for index, interval in enumerate(words_tier.intervals):
            if not _is_substantive_interior_silence(words_tier.intervals, index):
                continue
            left = next((item for item in reversed(lexical_positions)
                         if item < index), None)
            right = next((item for item in lexical_positions if item > index), None)
            left_ordinal = lexical_ordinal.get(left)
            right_ordinal = lexical_ordinal.get(right)
            label = interval.text.strip()
            reasons: list[str] = []
            if reference_mode != "fallback":
                reasons.append("reference_mode_not_fallback")
            if not valid_ledger:
                reasons.append("fallback_correspondence_invalid")
            if label not in _FALLBACK_QUALIFIED_PAUSE_LABELS:
                reasons.append("pause_label_not_qualified")
            if (left_ordinal is None or right_ordinal is None
                    or left_ordinal not in mapped_final
                    or right_ordinal not in mapped_final):
                reasons.append("pause_owner_mapping_missing")
            ctc_gap_evidence = None
            if label == "<sp3>" and not reasons:
                left_ctc = ctc_by_final.get(left_ordinal)
                right_ctc = ctc_by_final.get(right_ordinal)
                try:
                    ctc_gap_start = float(
                        left_ctc["end_s"] if "end_s" in left_ctc
                        else float(left_ctc["end_ms"]) / 1000.0)
                    ctc_gap_end = float(
                        right_ctc["start_s"] if "start_s" in right_ctc
                        else float(right_ctc["start_ms"]) / 1000.0)
                    pause_duration = max(interval.xmax - interval.xmin, 1e-9)
                    overlap = max(0.0, min(interval.xmax, ctc_gap_end)
                                  - max(interval.xmin, ctc_gap_start))
                    coverage = overlap / pause_duration
                    ctc_gap_evidence = {
                        "start_s": round(ctc_gap_start, 6),
                        "end_s": round(ctc_gap_end, 6),
                        "overlap_s": round(overlap, 6),
                        "pause_coverage": round(coverage, 6),
                    }
                    if (ctc_gap_end <= ctc_gap_start
                            or coverage < 0.90):
                        reasons.append("sp3_ctc_gap_not_supported")
                except (AttributeError, KeyError, TypeError, ValueError):
                    reasons.append("sp3_ctc_gap_evidence_missing")
            pauses.append({
                "index": index,
                "label": label,
                "start_s": round(interval.xmin, 6),
                "end_s": round(interval.xmax, 6),
                "duration_us": _duration_ticks(interval.xmin, interval.xmax),
                "left_lexical_ordinal": left_ordinal,
                "right_lexical_ordinal": right_ordinal,
                "ctc_gap_evidence": ctc_gap_evidence,
                "qualified": not reasons,
                "qualification_reasons": reasons,
            })
    qualified = [item["index"] for item in pauses if item["qualified"]]
    unqualified = [item["index"] for item in pauses if not item["qualified"]]
    unexpected_candidates = [
        item for item in pauses
        if item["label"] in {"<sp0>", "<sp1>", "<sp2>", "<sp3>"}
    ]
    reason_qualification = {
        "mid_sp": bool(pauses) and not unqualified,
        "strict_interior_sp": bool(pauses) and not unqualified,
        "unexpected_silence": bool(unexpected_candidates) and all(
            item["qualified"] for item in unexpected_candidates),
        "sp3": bool([item for item in pauses if item["label"] == "<sp3>"])
        and all(item["qualified"] for item in pauses
                if item["label"] == "<sp3>"),
    }
    return {
        "schema": "fallback-pause-qualification-v1",
        "reference_mode": reference_mode,
        "ledger": validation,
        "pause_count": len(pauses),
        "qualified_indices": qualified,
        "unqualified_indices": unqualified,
        "all_qualified": bool(pauses) and not unqualified,
        "reason_qualification": reason_qualification,
        "details": pauses,
    }


def _apply_fallback_pause_veto_qualification(
        filter_reasons: list[str], pause_gate: dict) -> list[str]:
    """Preserve pause vetoes; correspondence is diagnostic only.

    The old implementation globally removed pause reasons after lexical
    correspondence succeeded.  That allowed substantive SP intervals to
    pass while their evidence remained in the report.  Qualification remains
    useful diagnostics, but no fallback evidence can redeem a veto.
    """
    return list(filter_reasons)


def _terminal_punctuation_evidence_missing(
        words_tier: Tier | None, *, reference_authoritative: bool) -> dict | None:
    """Report an unowned terminal silence in no-reference publication.

    A terminal pause is publishable only when an exact punctuation owner has
    already absorbed it.  This check deliberately does not invent a mark or
    use the fallback transcript as punctuation authority.
    """
    if reference_authoritative or words_tier is None:
        return None
    nonempty = [(index, iv) for index, iv in enumerate(words_tier.intervals)
                if iv.text.strip()]
    if not nonempty:
        return None
    index, terminal = nonempty[-1]
    if not is_silence(terminal.text):
        return None
    previous = next((iv for _, iv in reversed(nonempty[:-1])
                     if not is_silence(iv.text)), None)
    if previous is None:
        return None
    return {
        "index": index,
        "label": terminal.text.strip(),
        "start_s": round(terminal.xmin, 6),
        "end_s": round(terminal.xmax, 6),
        "duration_us": _duration_ticks(terminal.xmin, terminal.xmax),
        "preceding_owner": previous.text.strip(),
        "reason": "terminal_punctuation_evidence_missing",
    }


def _published_nonleading_silence_details(
        words_tier: Tier | None) -> list[dict]:
    """Return every pure silence that is illegal in a publishable words tier.

    The leading ``<sp1>`` is a display convention.  All other pure
    ``<spN>`` intervals are unresolved ownership evidence and must never be
    classified as ``ok``.  This check is intentionally performed after the
    last owner transaction and final label normalization, so no later stage
    can reintroduce a trailing/internal pause behind the QC gate.
    """
    if words_tier is None:
        return []
    details: list[dict] = []
    for index, interval in enumerate(words_tier.intervals):
        label = interval.text.strip()
        if not is_silence(label):
            continue
        leading_allowed = (
            index == 0
            and label.casefold() == "<sp1>"
            and interval.xmin <= words_tier.xmin + AXIS_EPS)
        if leading_allowed:
            continue
        details.append({
            "index": index,
            "label": label,
            "start_s": round(float(interval.xmin), 6),
            "end_s": round(float(interval.xmax), 6),
            "duration_us": _duration_ticks(interval.xmin, interval.xmax),
            "reason": "nonleading_pure_silence_owner",
        })
    return details


# Descriptive aliases keep the contract discoverable to callers/tests while
# retaining one implementation and one digest definition.
_build_fallback_correspondence_ledger = _fallback_lexical_correspondence_ledger
_build_lexical_correspondence_ledger = _fallback_lexical_correspondence_ledger


def _apply_fallback_correspondence_to_lineage(lineage: dict,
                                              correspondence: dict) -> None:
    """Bind source phone owners to final ordinals without positional shifts."""
    if not isinstance(lineage, dict) or not isinstance(correspondence, dict):
        return
    by_source = {int(key): value for key, value in
                 correspondence.get("mapping", {}).get("source_to_final", {}).items()
                 if str(key).lstrip("-").isdigit() and isinstance(value, int)}
    for source_key, rows in lineage.get("owners", {}).items():
        try:
            source_ordinal = int(source_key)
        except (TypeError, ValueError):
            continue
        final_ordinal = by_source.get(source_ordinal)
        for row in rows if isinstance(rows, list) else []:
            row["final_lexical_ordinal"] = final_ordinal
            if final_ordinal is None and row.get("lexical_ordinal") is not None:
                row["omitted_source_owner"] = True


def _bind_source_phone_lineage(words_tier: Tier,
                               phones_tier: Tier | None,
                               correspondence: dict | None = None) -> dict:
    """Bind original MFA phones to one source word by ordered containment.

    This snapshot must be taken before any boundary mutation.  A lexical
    phone that crosses two source owners remains a hard structural failure.
    Ambiguous silence phones are retained as partial diagnostics: they have
    no lexical owner to preserve, and the rebuild stage will only materialize
    portions that remain inside final silence/punctuation owners.
    """
    result = {"schema": "source-phone-lineage-v1", "status": "verified",
              "owners": {}, "reasons": [],
              "source_intervals": [
                  {"ordinal": ordinal, "start": float(iv.xmin),
                   "end": float(iv.xmax), "text": iv.text.strip()}
                  for ordinal, iv in enumerate(words_tier.intervals)
              ]}
    if phones_tier is None:
        result["status"] = "missing"
        result["reasons"] = ["source_phones_missing"]
        return result
    source = list(words_tier.intervals)
    lexical_ordinals: dict[int, int] = {}
    lexical_count = 0
    for ordinal, word in enumerate(source):
        if (word.text.strip() and not is_silence(word.text)
                and not is_punct(word.text)):
            lexical_ordinals[ordinal] = lexical_count
            lexical_count += 1
    for phone_ordinal, phone in enumerate(phones_tier.intervals):
        if not phone.text.strip():
            continue
        owners = [ordinal for ordinal, word in enumerate(source)
                  if phone.xmin >= word.xmin - AXIS_EPS
                  and phone.xmax <= word.xmax + AXIS_EPS]
        if len(owners) != 1:
            if not is_silence(phone.text):
                result["status"] = "rejected"
            elif result["status"] == "verified":
                result["status"] = "partial"
            result["reasons"].append({
                "phone_ordinal": phone_ordinal,
                "label": phone.text,
                "span": [phone.xmin, phone.xmax],
                "owner_count": len(owners),
                "reason": "phone_lineage_ambiguous",
            })
            continue
        owner = owners[0]
        result["owners"].setdefault(str(owner), []).append({
            "phone_ordinal": phone_ordinal,
            "lexical_ordinal": lexical_ordinals.get(owner),
            "source_lexical_ordinal": lexical_ordinals.get(owner),
            "label": phone.text,
            "start": float(phone.xmin),
            "end": float(phone.xmax),
        })
    if result["status"] in {"verified", "partial"}:
        for rows in result["owners"].values():
            ordinals = [row["phone_ordinal"] for row in rows]
            if ordinals != sorted(ordinals):
                result["status"] = "rejected"
                result["reasons"].append("phone_lineage_order_invalid")
                break
    if isinstance(correspondence, dict):
        _apply_fallback_correspondence_to_lineage(result, correspondence)
    return result


def _rebuild_phones_from_lineage(words_tier: Tier,
                                 phones_tier: Tier,
                                 lineage: dict) -> Tier | None:
    """Affine-remap each source owner's original phone sequence."""
    if (not isinstance(lineage, dict)
            or lineage.get("status") not in {"verified", "partial"}):
        return None
    final_lexical = [iv for iv in words_tier.intervals
                     if iv.text.strip() and not is_silence(iv.text)
                     and not is_punct(iv.text)]
    source_intervals = lineage.get("source_intervals")
    if not isinstance(source_intervals, list):
        return None
    rebuilt: list[Interval] = []
    for owner_key, rows in lineage.get("owners", {}).items():
        try:
            owner_ordinal = int(owner_key)
        except (TypeError, ValueError):
            return None
        if not isinstance(rows, list) or not rows:
            continue
        source = next((item for item in source_intervals
                       if item.get("ordinal") == owner_ordinal), None)
        # Fallback correspondence may omit a source owner.  In that case the
        # row is intentionally skipped; using the old source lexical ordinal
        # here would shift every later phone/English owner left by one.
        lexical_ordinal = rows[0].get("final_lexical_ordinal",
                                     rows[0].get("lexical_ordinal"))
        if rows[0].get("omitted_source_owner"):
            continue
        if source is None or lexical_ordinal is None:
            # Non-lexical source phones retain their original label and order,
            # but their display ownership can be split by publication
            # geometry.  A source silence/punctuation row must be fully
            # covered by final non-lexical owners; strongest-overlap repair
            # would silently assign its crossing part to a lexical word.
            if source is None:
                return None
            source_start, source_end = float(source["start"]), float(source["end"])
            if source_end <= source_start:
                return None
            final_display = [iv for iv in words_tier.intervals
                             if iv.xmax > iv.xmin + AXIS_EPS]
            for row in rows:
                row_start, row_end = float(row["start"]), float(row["end"])
                if row_end <= row_start:
                    return None
                # A source silence can straddle a word and a pause/punctuation
                # after CTC compensation.  Never assign that part to the
                # lexical word by overlap strength; retain only display-owned
                # nonlexical pieces.  If no such piece remains, the source
                # silence is intentionally omitted from the derived tier.
                for owner in final_display:
                    if (not is_silence(owner.text)
                            and not is_punct(owner.text)):
                        continue
                    piece_start = max(row_start, owner.xmin)
                    piece_end = min(row_end, owner.xmax)
                    if piece_end > piece_start + AXIS_EPS:
                        rebuilt.append(Interval(
                            piece_start, piece_end, str(row["label"])))
            continue
        if not isinstance(lexical_ordinal, int) or lexical_ordinal >= len(final_lexical):
            return None
        target = final_lexical[lexical_ordinal]
        source_start, source_end = float(source["start"]), float(source["end"])
        source_duration = source_end - source_start
        if source_duration <= 0:
            return None
        target_duration = target.xmax - target.xmin
        if target_duration <= 0:
            return None
        for row in rows:
            ratio_start = (float(row["start"]) - source_start) / source_duration
            ratio_end = (float(row["end"]) - source_start) / source_duration
            start = target.xmin + ratio_start * target_duration
            end = target.xmin + ratio_end * target_duration
            if end <= start:
                return None
            rebuilt.append(Interval(start, end, str(row["label"])))
    rebuilt.sort(key=lambda iv: (iv.xmin, iv.xmax))
    # A non-lexical source phone may become several display-owned pieces, or
    # disappear when its final visual silence owner was merged into a lexical
    # word.  Lexical source rows must remain represented; non-lexical rows do
    # not, because retaining them would reintroduce a stale silence phone.
    expected_lexical = sum(
        len(rows) for rows in lineage.get("owners", {}).values()
        if rows and rows[0].get("final_lexical_ordinal") is not None
        and not rows[0].get("omitted_source_owner"))
    if len(rebuilt) < expected_lexical:
        return None
    return Tier(phones_tier.name, phones_tier.xmin, phones_tier.xmax, rebuilt)


def _lexical_ordinal(source_words: list[dict], ordinal: int) -> int:
    """Return an unknown's ordinal among non-silence/non-punctuation words."""
    return sum(1 for item in source_words[:ordinal]
               if item["text"] and not is_silence(item["text"])
               and not is_punct(item["text"]))


def _semantic_sequence_digest(sequence: list[tuple[str, str]]) -> str:
    return _evidence_digest([[kind, value] for kind, value in sequence])


def _build_mfa_unknown_recovery_proof(
        stem: str,
        source_words: list[dict],
        ctc_tokens: list[dict],
        reference_text: str,
        final_words: Tier | None,
        final_hanzi: Tier | None,
        strict_report: dict | None,
        strict_pairs: list[tuple[Interval, dict]]) -> dict | None:
    """Build proof for exactly the observed source ``<eps>,<unk>,<eps>`` case.

    Every returned field is bound to a concrete source interval, CTC token,
    reference semantic ordinal, ledger record, and finalized semantic
    sequence.  A report marker alone can never create this proof.
    """
    unknowns = [(index, item) for index, item in enumerate(source_words)
                if is_unknown_token(item["text"])]
    if len(unknowns) != 1 or not ctc_tokens or final_words is None:
        return None
    source_ordinal, source = unknowns[0]
    if (_lexical_ordinal(source_words, source_ordinal) != 0
            or source_ordinal == 0
            or source_words[source_ordinal - 1]["text"] != "<eps>"
            or source_ordinal + 1 >= len(source_words)
            or source_words[source_ordinal + 1]["text"] != "<eps>"):
        return None
    token = ctc_tokens[0]
    if (not isinstance(token, dict)
            or token.get("word", "").strip().casefold() != "mira"
            or token.get("type", "word") != "word"):
        return None
    try:
        token_start, token_end = float(token["start_s"]), float(token["end_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if (not math.isfinite(token_start) or not math.isfinite(token_end)
            or token_end <= token_start):
        return None
    source_sequence = [_lexical_identity(item["text"]) for item in source_words
                       if item["text"] and not is_silence(item["text"])
                       and not is_punct(item["text"])]
    source_unknown_position = next(
        (index for index, value in enumerate(source_sequence)
         if value == "<unknown>"), None)
    if source_unknown_position is None:
        return None
    source_sequence[source_unknown_position] = "mira"
    ctc_sequence = [_lexical_identity(item.get("word", ""), ctc_item=item)
                    for item in ctc_tokens
                    if isinstance(item, dict) and item.get("type", "word") == "word"]
    if source_sequence != ctc_sequence:
        return None

    reference_semantics = project_authority_semantics(reference_text)
    reference_matches = [item for item in reference_semantics
                         if item.get("kind") == "english"
                         and item.get("alignment_token", "").casefold() == "mira"
                         and item.get("reference_ordinal") == 0]
    if len(reference_matches) != 1:
        return None
    reference_item = reference_matches[0]
    reference_ordinal = int(reference_item["reference_ordinal"])
    reference_token = str(reference_item["surface"]).casefold()
    reference_sequence = _strict_semantic_tokens(reference_text)
    final_lexical = [iv for iv in final_words.intervals
                     if iv.text.strip() and not is_silence(iv.text)
                     and not is_punct(iv.text)]
    # The final words tier is ordered by the processed CTC/authority stream.
    # Bind the owner by lexical ordinal, not by raw MFA/CTC time overlap: the
    # latter is exactly what fails for the five observed leading-offset cases.
    owners = [iv for index, iv in enumerate(final_lexical)
              if index == 0 and _lexical_identity(iv.text) == "mira"]
    if len(owners) != 1:
        return None
    owner = owners[0]
    pair_matches = [(word, record) for word, record in strict_pairs
                    if word is owner
                    and str(record.get("ctc_text", "")).casefold() == "mira"]
    if len(pair_matches) != 1 or not isinstance(strict_report, dict):
        return None
    record = pair_matches[0][1]
    ledger_sha = strict_report.get("ledger_sha256", "")
    if (strict_report.get("status") != "verified"
            or not isinstance(ledger_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", ledger_sha)
            or record.get("provenance") != "english_mfa_textgrid"):
        return None
    final_sequence = (_strict_semantic_tokens(" ".join(iv.text for iv in final_hanzi.intervals
                                                    if iv.text))
                      if final_hanzi is not None else [])
    if final_sequence != reference_sequence:
        return None
    source_interval = {"ordinal": source["ordinal"], "start": source["start"],
                       "end": source["end"], "text": source["text"]}
    ctc_value = {"ordinal": 0, "word": token.get("word", ""),
                 "start_s": token_start, "end_s": token_end,
                 "type": token.get("type", "word")}
    owner_value = {"text": owner.text.strip(), "start": float(owner.xmin),
                   "end": float(owner.xmax)}
    return {
        "schema": _UNKNOWN_REPAIR_PROOF_SCHEMA,
        "scenario": "initial_mira",
        "stem": stem,
        "source": {"interval": source_interval,
                    "interval_sha256": _evidence_digest(source_interval),
                    "lexical_ordinal": 0,
                    "neighbors": ["<eps>", "<eps>"]},
        "ctc": {"token": ctc_value, "token_sha256": _evidence_digest(ctc_value)},
        "ordered_correspondence": {
            "source_tokens": source_sequence,
            "ctc_tokens": ctc_sequence,
            "sha256": _evidence_digest(source_sequence),
        },
        "reference": {"ordinal": reference_ordinal, "token": reference_token},
        "binding": {
            "source_lexical_ordinal": source_unknown_position,
            "ctc_lexical_ordinal": 0,
            "reference_ordinal": reference_ordinal,
            "temporal_overlap_s": max(
                0.0, min(source["end"], token_end)
                - max(source["start"], token_start)),
            "temporal_overlap_required": False,
        },
        "english_ledger": {
            "ledger_sha256": ledger_sha,
            "word_id": record.get("word_id"),
            "ctc_ordinal": record.get("ctc_ordinal"),
            "ctc_text": record.get("ctc_text"),
            "word_sha256": _evidence_digest(record),
        },
        "final": {"owner": owner_value,
                  "semantic_sequence_sha256": _semantic_sequence_digest(final_sequence),
                  "semantic_token_count": len(final_sequence)},
    }


def _dual_evidence_boundary_solution(
        source_words: list[dict], ctc_tokens: list[dict],
        left: Interval, right: Interval,
        *, allow_overlap: bool) -> dict | None:
    """Return the unique CTC boundary for one local Chinese word pair.

    The source and CTC sequences must independently agree on order and word
    identity.  Candidate matching is local to the final pair and therefore
    repeated words elsewhere cannot silently become owners.  English/NVV,
    punctuation, and silence are intentionally excluded by the caller.
    """
    left_text, right_text = left.text.strip().casefold(), right.text.strip().casefold()
    if not left_text or not right_text:
        return None
    local_start = min(left.xmin, right.xmin) - 0.5
    local_end = max(left.xmax, right.xmax) + 0.5
    source_left = [item for item in source_words
                   if item["text"].casefold() == left_text
                   and item["end"] > local_start and item["start"] < local_end]
    source_right = [item for item in source_words
                    if item["text"].casefold() == right_text
                    and item["end"] > local_start and item["start"] < local_end]
    ctc_words = [(index, item) for index, item in enumerate(ctc_tokens)
                 if isinstance(item, dict) and item.get("type", "word") == "word"
                 and item.get("word", "").strip()
                 and float(item.get("end_s", -1)) > local_start
                 and float(item.get("start_s", -1)) < local_end]
    ctc_left = [(index, item) for index, item in ctc_words
                if str(item["word"]).strip().casefold() == left_text]
    ctc_right = [(index, item) for index, item in ctc_words
                 if str(item["word"]).strip().casefold() == right_text]
    solutions: list[dict] = []
    for source_l in source_left:
        for source_r in source_right:
            if (source_l["ordinal"] >= source_r["ordinal"]
                    or source_l["end"] > source_r["start"]
                    or source_l["end"] - source_l["start"] < _EVIDENCE_REPAIR_FLOOR_S - 1e-9
                    or source_r["end"] - source_r["start"] < _EVIDENCE_REPAIR_FLOOR_S - 1e-9):
                continue
            for ctc_l_index, ctc_l in ctc_left:
                for ctc_r_index, ctc_r in ctc_right:
                    if ctc_l_index >= ctc_r_index:
                        continue
                    try:
                        ctc_l_start, ctc_l_end = float(ctc_l["start_s"]), float(ctc_l["end_s"])
                        ctc_r_start, ctc_r_end = float(ctc_r["start_s"]), float(ctc_r["end_s"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (ctc_l_end - ctc_l_start < _EVIDENCE_REPAIR_FLOOR_S - 1e-9
                            or ctc_r_end - ctc_r_start < _EVIDENCE_REPAIR_FLOOR_S - 1e-9
                            or ctc_l_end > ctc_r_start + 1e-6
                            or abs(ctc_l_end - ctc_r_start) > 1e-3):
                        continue
                    final_overlap = left.xmax - right.xmin
                    final_gap = right.xmin - left.xmax
                    final_short = (left.xmax - left.xmin < _EVIDENCE_REPAIR_FLOOR_S
                                   or right.xmax - right.xmin < _EVIDENCE_REPAIR_FLOOR_S)
                    if final_overlap > 0.005 and not allow_overlap:
                        continue
                    if final_overlap <= 0.005 and final_gap <= 0.020 and not final_short:
                        continue
                    # CTC supplies the local boundary.  When a preceding
                    # punctuation/authority interval makes that anchor leave
                    # a word below the immutable 30 ms floor, the only
                    # admissible adjustment is the deterministic floor clamp;
                    # midpoint heuristics are never used.
                    lower = left.xmin + _EVIDENCE_REPAIR_FLOOR_S
                    upper = right.xmax - _EVIDENCE_REPAIR_FLOOR_S
                    boundary = min(max(ctc_l_end, lower), upper)
                    if boundary < lower or boundary > upper:
                        continue
                    solutions.append({
                        "source": [source_l, source_r],
                        "ctc": [{"ordinal": ctc_l_index, **ctc_l},
                                 {"ordinal": ctc_r_index, **ctc_r}],
                        "boundary_s": boundary,
                        "ctc_boundary_s": ctc_l_end,
                        "reason": ("overlap" if final_overlap > 0.005
                                   else "short_word" if final_short else "words_tier_gap"),
                    })
    if len(solutions) != 1:
        return None
    return solutions[0]


def _apply_evidence_constrained_repairs(
        stem: str, source_words: list[dict], ctc_tokens: list[dict],
        textgrid: TextGrid) -> list[dict]:
    """Apply only unique source/CTC repairs; return structured evidence."""
    words = tier_by_name(textgrid, "words")
    if words is None:
        return []
    stem_id = stem[:6]
    pp = tier_by_name(textgrid, "pinyin_phones")
    hanzi = tier_by_name(textgrid, "hanzi")
    repairs: list[dict] = []
    for index in range(len(words.intervals) - 1):
        left, right = words.intervals[index:index + 2]
        labels = [left.text.strip(), right.text.strip()]
        if any(not is_pinyin_syllable(label) for label in labels):
            continue
        overlap_allowed = stem_id in _OVERLAP_EVIDENCE_STEMS
        solution = _dual_evidence_boundary_solution(
            source_words, ctc_tokens, left, right, allow_overlap=overlap_allowed)
        if solution is None:
            continue
        old_left, old_right = left.xmax, right.xmin
        boundary = solution["boundary_s"]
        if not (left.xmin + _EVIDENCE_REPAIR_FLOOR_S <= boundary
                <= right.xmax - _EVIDENCE_REPAIR_FLOOR_S):
            continue
        left.xmax = boundary
        right.xmin = boundary
        if hanzi is not None and index + 1 < len(hanzi.intervals):
            # Finalization keeps hanzi and words positionally aligned.
            hanzi.intervals[index].xmax = boundary
            hanzi.intervals[index + 1].xmin = boundary
        if pp is not None:
            # Never clip or move verified English phones.  Chinese phones
            # remain owned by the repaired word and are clipped only if they
            # crossed the new boundary.
            for phone in pp.intervals:
                if phone.text.strip().startswith(EN_PHONE_PREFIX):
                    continue
                if phone.xmax > left.xmin and phone.xmin < right.xmax:
                    if phone.xmin < boundary < phone.xmax:
                        if phone.xmin < left.xmax:
                            phone.xmax = left.xmax
                        elif phone.xmax > right.xmin:
                            phone.xmin = right.xmin
        repairs.append({
            "schema": _EVIDENCE_REPAIR_SCHEMA,
            "kind": solution["reason"],
            "stem": stem,
            "word_indices": [index, index + 1],
            "source_words": solution["source"],
            "ctc_tokens": [{key: value for key, value in token.items()
                            if key in {"ordinal", "word", "start_s", "end_s", "type"}}
                           for token in solution["ctc"]],
            "boundary_s": boundary,
            "previous_boundary": {"left_end": old_left, "right_start": old_right},
            "proof": "source_mfa_ctc_unique_monotone_boundary",
        })
    return repairs


def _strict_en_lexical_words(words_tier: Tier | None) -> list[Interval]:
    if words_tier is None:
        return []
    return [iv for iv in words_tier.intervals if is_english_token(iv.text.strip())]


def _restore_fallback_unknown_surfaces(
        words_tier: Tier, ctc_tokens: list[dict]) -> Tier:
    """Restore explicit MFA unknowns from one exact ordered projection.

    The lexical projection is solved before any interval is changed.  A
    known source item may be omitted, while each unknown must consume the next
    known NVV/English CTC item.  Ambiguous, partial, malformed, reordered, or
    substituted evidence returns an unchanged tier with a rejected proof.
    """
    source_words = _source_unknown_context(words_tier)
    source, ctc, _final, malformed = _fallback_lexical_items(
        source_words, ctc_tokens,
        Tier("words", words_tier.xmin, words_tier.xmax, []))
    projection = _fallback_exact_source_ctc_projection(source, ctc)
    if malformed:
        projection["safe"] = False
        projection["status"] = "rejected"
        projection["first_mismatch"] = {
            "reason": "malformed_evidence", "details": list(malformed)}
        projection["digest"] = _evidence_digest(projection)

    # Recovery is transactional: no surface, merge, or geometry operation is
    # performed until the complete source→CTC solution is unique and safe.
    if not projection.get("safe"):
        unchanged = _copy_tier_metadata(
            words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax,
                             list(words_tier.intervals)))
        unchanged._fallback_unknown_projection = projection
        return unchanged

    intervals = list(words_tier.intervals)
    recovered: list[dict] = []
    for entry in projection["entries"]:
        if (entry.get("status") != "mapped"
                or not is_unknown_token(entry.get("source_text", ""))):
            continue
        source_ordinal = entry.get("source_ordinal")
        if not isinstance(source_ordinal, int) or not (0 <= source_ordinal < len(intervals)):
            projection["safe"] = False
            projection["status"] = "rejected"
            projection["first_mismatch"] = {
                "reason": "source_ordinal_invalid",
                "source_ordinal": source_ordinal,
            }
            break
        ctc_text = str(entry.get("ctc_text", "")).strip().strip("<>")
        if (_canonical_nvv_identity(ctc_text) is None
                and not is_english_token(ctc_text)):
            projection["safe"] = False
            projection["status"] = "rejected"
            projection["first_mismatch"] = {
                "reason": "unknown_target_not_known_nvv_or_english",
                "source_ordinal": source_ordinal,
            }
            break
        intervals[source_ordinal] = Interval(
            intervals[source_ordinal].xmin, intervals[source_ordinal].xmax, ctc_text)
        entry["reason"] = "known_nvv_or_english_unknown_recovery"
        entry["recovery_evidence"] = {
            "source_ordinal": source_ordinal,
            "source_identity": entry.get("source_identity"),
            "ctc_ordinal": entry.get("ctc_ordinal"),
            "ctc_identity": entry.get("ctc_identity"),
            "ctc_span": entry.get("ctc_span"),
        }
        recovered.append(entry["recovery_evidence"])

    if not projection.get("safe"):
        unchanged = _copy_tier_metadata(
            words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax,
                             list(words_tier.intervals)))
        projection["digest"] = _evidence_digest(projection)
        unchanged._fallback_unknown_projection = projection
        return unchanged

    projection["recovered"] = recovered
    projection["digest"] = _evidence_digest(projection)
    restored = _copy_tier_metadata(
        words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax,
                         intervals))
    restored._fallback_unknown_projection = projection
    return restored


def _fallback_redeemed_unknown_entries(
        correspondence: dict | None,
        english_provenance: dict | None = None) -> list[dict]:
    """Return source unknowns backed by exact NVV or strict-English owners.

    The source→CTC→final correspondence already proves unique ordered
    ownership.  NVV identities are self-authenticating lexical tags; English
    targets additionally require the strict English phone ledger to be fully
    verified.  Chinese/pinyin targets and partial English provenance never
    redeem a source ``<unk>``.
    """
    if not isinstance(correspondence, dict) or not correspondence.get("safe"):
        return []
    english_verified = (
        isinstance(english_provenance, dict)
        and english_provenance.get("status") == "verified"
    )
    recovered: list[dict] = []
    for original in correspondence.get("entries", []):
        if (not isinstance(original, dict)
                or original.get("status") != "mapped"
                or not is_unknown_token(original.get("source_text", ""))):
            continue
        resolved = str(original.get("resolved_text", "")).strip()
        if _canonical_nvv_identity(resolved) is not None:
            target_kind = "known_nvv"
        elif is_english_token(resolved) and english_verified:
            target_kind = "strict_english"
        else:
            continue
        entry = dict(original)
        entry["recovery_target_kind"] = target_kind
        entry["recovery_provenance"] = (
            "exact_correspondence"
            if target_kind == "known_nvv"
            else "exact_correspondence+strict_english_ledger"
        )
        recovered.append(entry)
    return recovered


def _strict_en_join_key(text: str, *, hyphenated: bool = False) -> str:
    """Return the exact ledger-join key for one English spelling.

    Hyphenated reference words may be emitted by the Chinese tokenizer as
    contiguous alpha fragments (``kp`` + ``op`` for ``K-Pop``).  Only the
    reference hyphen is ignored for that one explicit join; all other text
    remains case-folded and order-sensitive.
    """
    value = str(text).strip().casefold()
    if hyphenated:
        return re.sub(r"[^a-z0-9]", "", value)
    return value


def _strict_en_phone_is_valid(phone: dict, mfa_word: dict) -> bool:
    """Validate one immutable producer phone before affine mapping it."""
    try:
        label = str(phone["label"]).strip()
        start, end = float(phone["start"]), float(phone["end"])
        word_start, word_end = float(mfa_word["start"]), float(mfa_word["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return (label not in _STRICT_EN_SILENCE and is_english_phone(label)
            and math.isfinite(start) and math.isfinite(end)
            and math.isfinite(word_start) and math.isfinite(word_end)
            and word_end > word_start and end > start
            and start >= word_start - 0.003 and end <= word_end + 0.003)


def _strict_en_pronunciation_reason(record: dict, ledger: dict) -> str | None:
    """Validate producer pronunciation policy at the English consumer.

    SOS is a run-local dictionary override.  A consumer must bind the policy,
    exact source sequence, dictionary path/hash, and phone ordinals already
    checked by the surrounding ledger validator; it may not infer SOS from
    labels alone.  APP is retained as a negative canary for a fabricated
    second ``P``.
    """
    token = str(record.get("alignment_token", "")).casefold()
    labels = tuple(str(phone.get("label", "")).strip()
                   for phone in record.get("phones", [])
                   if isinstance(phone, dict))
    if token == "app":
        return (None if labels == APP_EXPECTED_PRONUNCIATION
                else "app_expected_pronunciation_mismatch")
    if token != "sos":
        return None
    policy = record.get("pronunciation_policy")
    dictionary = ledger.get("dictionary_provenance")
    if (record.get("pronunciation_policy_id") != SOS_PRONUNCIATION_POLICY_ID
            or not isinstance(policy, dict)
            or policy.get("policy_id") != SOS_PRONUNCIATION_POLICY_ID
            or tuple(policy.get("expected_pronunciation", ())) != SOS_EXPECTED_PRONUNCIATION
            or tuple(policy.get("actual_source_sequence", ())) != labels
            or labels != SOS_EXPECTED_PRONUNCIATION
            or policy.get("dictionary_provenance") != dictionary
            or record.get("dictionary_provenance") != dictionary):
        return "sos_pronunciation_policy_mismatch"
    if (not isinstance(dictionary, dict)
            or not isinstance(dictionary.get("path"), str)
            or not isinstance(dictionary.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", dictionary["sha256"])):
        return "sos_dictionary_provenance_invalid"
    try:
        dictionary_path = Path(dictionary["path"])
        if (dictionary_path.is_symlink() or not dictionary_path.is_file()
                or _strict_en_sha256(dictionary_path) != dictionary["sha256"]):
            return "sos_dictionary_hash_mismatch"
    except (OSError, ValueError, TypeError):
        return "sos_dictionary_provenance_unreadable"
    return None


def load_strict_en_provenance(stem: str, words_tier: Tier | None,
                              en_phones_dir: Path | None, *,
                              hanzi_tier: Tier | None = None,
                              reference_text: str | None = None,
                              ctc_tokens: list[dict] | None = None,
                              correspondence: dict | None = None,
                              processed_ctc_words_tier: Tier | None = None,
                              processed_ctc_textgrid_path: Path | None = None,
                              ) -> tuple[dict, list[tuple[Interval, dict]]]:
    """Load a strict-en-mfa-v2 ledger and bind one owner to one unit.

    The old JSON list is deliberately not accepted here.  In particular, no
    text/time lookup is used: repeated words and English separated by Chinese
    are matched solely by their ordered instances in the full words tier.
    """
    english_words = _strict_en_lexical_words(words_tier)
    required = len(english_words)
    if en_phones_dir is None:
        return _strict_en_fail(required, "strict_en_manifest_missing")
    manifest_path = en_phones_dir / "en_alignment_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return _strict_en_fail(required, "strict_en_manifest_missing_or_corrupt")
    if (not isinstance(manifest, dict) or manifest.get("schema") != STRICT_EN_MFA_SCHEMA
            or manifest.get("strict_provenance") is not True
            or manifest.get("canonical_units") != CANONICAL_UNITS_SCHEMA
            or manifest.get("status") not in {"success", "partial", "no_english"}):
        reason = ("strict_en_manifest_legacy_schema"
                  if isinstance(manifest, dict)
                  and manifest.get("schema") == HISTORICAL_STRICT_EN_MFA_SCHEMA
                  else "strict_en_manifest_invalid")
        return _strict_en_fail(required, reason)
    expected_segments = manifest.get("expected_segments")
    produced_segments = manifest.get("produced_segments")
    rejected_segments = manifest.get("rejected_segments")
    if (not isinstance(expected_segments, list) or not isinstance(produced_segments, list)
            or not isinstance(rejected_segments, list)
            or not all(isinstance(item, str) for item in expected_segments)
            or not all(isinstance(item, str) for item in produced_segments)):
        return _strict_en_fail(required, "strict_en_manifest_partition_invalid")
    rejected_ids = [item.get("id") for item in rejected_segments if isinstance(item, dict)]
    if (len(rejected_ids) != len(rejected_segments)
            or not all(isinstance(item, str) for item in rejected_ids)
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("reason"), str)
                   or not item.get("reason")
                   for item in rejected_segments)
            or len(expected_segments) != len(set(expected_segments))
            or len(produced_segments) != len(set(produced_segments))
            or len(rejected_ids) != len(set(rejected_ids))
            or set(expected_segments) != set(produced_segments) | set(rejected_ids)
            or set(produced_segments) & set(rejected_ids)
            or (manifest.get("status") == "success" and rejected_ids)
            or (manifest.get("status") == "partial" and not rejected_ids)
            or (manifest.get("status") == "no_english" and expected_segments)):
        return _strict_en_fail(required, "strict_en_manifest_partition_invalid")
    if not english_words:
        return _strict_en_report("not_required"), []
    if manifest.get("status") not in {"success", "partial"}:
        return _strict_en_fail(required, "strict_en_manifest_has_no_english")

    prefix = f"{stem}:s"
    expected_for_stem = {item for item in expected_segments
                         if isinstance(item, str) and item.startswith(prefix)}
    produced_for_stem = {item for item in produced_segments
                         if isinstance(item, str) and item.startswith(prefix)}
    rejected_for_stem = {item.get("id") for item in rejected_segments
                         if isinstance(item, dict) and isinstance(item.get("id"), str)
                         and item["id"].startswith(prefix)}
    if (not expected_for_stem or expected_for_stem != produced_for_stem | rejected_for_stem
            or produced_for_stem & rejected_for_stem):
        return _strict_en_fail(required, "strict_en_manifest_segment_rejected_or_incomplete")

    entries = manifest.get("stem_ledgers")
    if not isinstance(entries, list):
        return _strict_en_fail(required, "strict_en_ledger_missing")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("stem") == stem]
    if len(matches) != 1:
        return _strict_en_fail(required, "strict_en_ledger_missing_or_ambiguous")
    entry = matches[0]
    expected_hash = entry.get("sha256")
    try:
        ledger_path = Path(entry["path"])
        actual_hash = _strict_en_sha256(ledger_path)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        return _strict_en_fail(required, "strict_en_ledger_missing_or_corrupt")
    if not isinstance(expected_hash, str) or not expected_hash or actual_hash != expected_hash:
        return _strict_en_fail(required, "strict_en_ledger_hash_mismatch", ledger_sha256=actual_hash)
    if (not isinstance(ledger, dict) or ledger.get("schema") != STRICT_EN_MFA_SCHEMA
            or ledger.get("stem") != stem
            or ledger.get("canonical_units") != CANONICAL_UNITS_SCHEMA):
        reason = ("strict_en_ledger_legacy_schema"
                  if isinstance(ledger, dict)
                  and ledger.get("schema") == HISTORICAL_STRICT_EN_MFA_SCHEMA
                  else "strict_en_ledger_schema_or_stem_mismatch")
        return _strict_en_fail(required, reason, ledger_sha256=actual_hash)
    if processed_ctc_textgrid_path is not None:
        try:
            ctc_textgrid_sha256 = ledger["ctc_textgrid_sha256"]
            if (not isinstance(ctc_textgrid_sha256, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", ctc_textgrid_sha256)
                    or _strict_en_sha256(processed_ctc_textgrid_path)
                    != ctc_textgrid_sha256):
                raise ValueError("processed CTC TextGrid hash mismatch")
        except (KeyError, OSError, TypeError, ValueError):
            return _strict_en_fail(
                required, "strict_en_ctc_textgrid_hash_mismatch",
                ledger_sha256=actual_hash)

    records: list[dict] = []
    seen_segment_ids: set[str] = set()
    segments = ledger.get("segments")
    if not isinstance(segments, list):
        return _strict_en_fail(required, "strict_en_segments_missing", ledger_sha256=actual_hash)
    for segment in sorted(segments, key=lambda item: item.get("segment_ordinal", -1)
                          if isinstance(item, dict) else -1):
        if not isinstance(segment, dict):
            return _strict_en_fail(required, "strict_en_segment_invalid", ledger_sha256=actual_hash)
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or segment_id in seen_segment_ids:
            return _strict_en_fail(required, "strict_en_segment_id_invalid", ledger_sha256=actual_hash)
        seen_segment_ids.add(segment_id)
        if segment_id not in expected_for_stem:
            return _strict_en_fail(required, "strict_en_segment_not_in_manifest", ledger_sha256=actual_hash)
        if segment.get("status") != "verified":
            failed = [str(word.get("word_id", "")) for word in segment.get("words", [])
                      if isinstance(word, dict)]
            return _strict_en_fail(required, "strict_en_segment_rejected", ledger_sha256=actual_hash,
                                   failed_word_ids=failed)
        source = segment.get("mfa_textgrid")
        try:
            source_path = Path(source["path"])
            if not isinstance(source.get("sha256"), str) or not source["sha256"]:
                raise ValueError("hash_missing")
            if _strict_en_sha256(source_path) != source["sha256"]:
                raise ValueError("hash_mismatch")
        except Exception:
            return _strict_en_fail(required, "strict_en_source_evidence_invalid",
                                   ledger_sha256=actual_hash)
        words = segment.get("words")
        if not isinstance(words, list):
            return _strict_en_fail(required, "strict_en_words_missing", ledger_sha256=actual_hash)
        records.extend(words)

    if seen_segment_ids != expected_for_stem:
        return _strict_en_fail(required, "strict_en_ledger_segment_partition_invalid",
                               ledger_sha256=actual_hash)

    # Validate the immutable record envelope before any projection/grouping.
    # This prevents a malformed, duplicated, reordered, or extra record from
    # disappearing when several fragments are represented by one final word.
    record_ids: set[str] = set()
    prior_ctc_ordinal = -1
    for record in records:
        if not isinstance(record, dict):
            return _strict_en_fail(required, "strict_en_word_invalid", ledger_sha256=actual_hash)
        word_id = record.get("word_id")
        try:
            ctc_ordinal = int(record["ctc_ordinal"])
        except (KeyError, TypeError, ValueError):
            return _strict_en_fail(required, "strict_en_word_identity_or_evidence_invalid",
                                   ledger_sha256=actual_hash)
        if (not isinstance(word_id, str) or not word_id or word_id in record_ids
                or ctc_ordinal <= prior_ctc_ordinal
                or record.get("status") != "verified"
                or record.get("provenance") != "english_mfa_textgrid"
                or record.get("canonical_binding") != CANONICAL_UNITS_SCHEMA
                or not isinstance(record.get("unit_id"), str)
                or not isinstance(record.get("alignment_token"), str)
                or not isinstance(record.get("source_ctc_ordinals"), list)
                or not isinstance(record.get("canonical_span"), list)
                or len(record["canonical_span"]) != 2
                or not isinstance(record.get("mfa_word"), dict)
                or not isinstance(record.get("phones"), list)
                or not record["phones"]):
            return _strict_en_fail(required, "strict_en_word_identity_or_evidence_invalid",
                                   ledger_sha256=actual_hash, failed_word_ids=[str(word_id or "")])
        if any(not isinstance(phone, dict) for phone in record["phones"]):
            return _strict_en_fail(required, "strict_en_phone_invalid",
                                   ledger_sha256=actual_hash,
                                   failed_word_ids=[str(word_id)])
        mfa_word = record["mfa_word"]
        try:
            mfa_start, mfa_end = float(mfa_word["start"]), float(mfa_word["end"])
        except (KeyError, TypeError, ValueError):
            return _strict_en_fail(required, "strict_en_mfa_word_invalid",
                                   ledger_sha256=actual_hash,
                                   failed_word_ids=[str(word_id)])
        if (not isinstance(mfa_word.get("ordinal"), int)
                or mfa_word["ordinal"] < 0
                or not math.isfinite(mfa_start) or not math.isfinite(mfa_end)
                or mfa_end <= mfa_start):
            return _strict_en_fail(required, "strict_en_mfa_word_invalid",
                                   ledger_sha256=actual_hash,
                                   failed_word_ids=[str(word_id)])
        source_ordinals = record["source_ctc_ordinals"]
        canonical_span = record["canonical_span"]
        if ((canonical_span[0] is None) != (canonical_span[1] is None)
                or (canonical_span[0] is not None and (
                    not isinstance(canonical_span[0], (int, float))
                    or isinstance(canonical_span[0], bool)
                    or not isinstance(canonical_span[1], (int, float))
                    or isinstance(canonical_span[1], bool)
                    or not all(math.isfinite(float(value)) for value in canonical_span)
                    or canonical_span[1] < canonical_span[0]))):
            return _strict_en_fail(required, "strict_en_unit_span_invalid",
                                   ledger_sha256=actual_hash,
                                   failed_word_ids=[str(word_id)])
        if (not source_ordinals
                or any(type(value) is not int or value < 0 for value in source_ordinals)
                or source_ordinals != list(range(source_ordinals[0], source_ordinals[-1] + 1))
                or ctc_ordinal != source_ordinals[0]):
            # A missing source ordinal is always invalid.  A non-contiguous
            # span is valid only when the exact processed token independently
            # carries the producer's explicit dropped-hyphen marker.
            try:
                if (not source_ordinals
                        or any(type(value) is not int or value < 0
                               for value in source_ordinals)
                        or any(left >= right for left, right in
                               zip(source_ordinals, source_ordinals[1:]))):
                    raise EnglishUnitError("strict_english_source_ordinals_invalid")
                token = resolve_processed_english_token(
                    ctc_tokens, source_ordinals)
                validate_processed_english_token_binding(record, token)
            except (EnglishUnitError, TypeError, ValueError):
                return _strict_en_fail(required, "strict_en_unit_span_invalid",
                                       ledger_sha256=actual_hash,
                                       failed_word_ids=[str(word_id)])
        prior_phone_end = -math.inf
        for phone_ordinal, phone in enumerate(record["phones"]):
            if (phone.get("ordinal") != phone_ordinal
                    or not _strict_en_phone_is_valid(phone, mfa_word)):
                return _strict_en_fail(required, "strict_en_phone_invalid",
                                       ledger_sha256=actual_hash,
                                       failed_word_ids=[str(word_id)])
            phone_start, phone_end = float(phone["start"]), float(phone["end"])
            if phone_start < prior_phone_end:
                return _strict_en_fail(required, "strict_en_phone_unordered",
                                       ledger_sha256=actual_hash,
                                       failed_word_ids=[str(word_id)])
            prior_phone_end = phone_end
        first_phone = record["phones"][0]
        last_phone = record["phones"][-1]
        if (abs(float(first_phone["start"]) - mfa_start) > AXIS_EPS
                or abs(float(last_phone["end"]) - mfa_end) > AXIS_EPS
                or any(float(right["start"]) - float(left["end"]) > AXIS_EPS
                       for left, right in zip(record["phones"], record["phones"][1:]))):
            return _strict_en_fail(required, "strict_en_phone_span_mismatch",
                                   ledger_sha256=actual_hash,
                                   failed_word_ids=[str(word_id)])
        pronunciation_reason = _strict_en_pronunciation_reason(record, ledger)
        if pronunciation_reason:
            return _strict_en_fail(required, pronunciation_reason,
                                   ledger_sha256=actual_hash,
                                   failed_word_ids=[str(word_id)])
        record_ids.add(word_id)
        prior_ctc_ordinal = ctc_ordinal

    # The authoritative transcript may keep a contiguous English spelling as
    # one word (e.g. ``Sila``), while CTC/MFA tokenization can split the same
    # acoustic span into verified records (``S`` + ``il`` + ``a``).  Reconcile
    # only exact, ordered concatenations; never drop a record or synthesize a
    # phone.  This preserves every MFA phone as provenance for the final word.
    if reference_text:
        try:
            authority_units = parse_english_units(reference_text)
        except (EnglishUnitError, TypeError, ValueError):
            authority_units = ()
        if len(authority_units) != required and required > len(authority_units):
            return _strict_en_fail(required, "strict_en_authoritative_compound_split",
                                   ledger_sha256=actual_hash)
    else:
        authority_units = ()

    if len(records) != required:
        # v2 canonical units are already merged by the producer.  A final
        # split cannot be repaired by text concatenation: it would create two
        # visible owners for one immutable unit and could enlarge ownership
        # over punctuation or a gap.
        return _strict_en_fail(
            required, "strict_en_authoritative_compound_split"
            if any("-" in str(record.get("ctc_text", ""))
                   or record.get("unit_id") in {
                       other.get("unit_id") for other in records
                       if isinstance(other, dict) and other is not record}
                   for record in records if isinstance(record, dict))
            else "strict_en_word_count_mismatch",
            ledger_sha256=actual_hash,
            failed_word_ids=[str(item.get("word_id", "")) for item in records
                             if isinstance(item, dict)])
    if reference_text and len(authority_units) != required:
        return _strict_en_fail(required, "strict_en_authority_unit_count_mismatch",
                               ledger_sha256=actual_hash)
    if not reference_text:
        grouped_records: list[dict] = []
        record_cursor = 0
        for final_word in english_words:
            target = final_word.text.strip().casefold()
            hyphenated = "-" in target
            target_key = _strict_en_join_key(target, hyphenated=hyphenated)
            joined = ""
            matched_end = None
            for candidate_end in range(record_cursor + 1, len(records) + 1):
                candidate = records[candidate_end - 1]
                if not isinstance(candidate, dict):
                    break
                joined += str(candidate.get("ctc_text", "")).strip()
                joined_key = _strict_en_join_key(joined, hyphenated=hyphenated)
                if joined_key == target_key:
                    matched_end = candidate_end
                    break
                if not target_key.startswith(joined_key):
                    break
            if matched_end is None:
                return _strict_en_fail(
                    required, "strict_en_word_count_mismatch", ledger_sha256=actual_hash,
                    failed_word_ids=[str(item.get("word_id", "")) for item in records
                                     if isinstance(item, dict)])
            chunk = records[record_cursor:matched_end]
            if len(chunk) == 1:
                grouped_records.append(chunk[0])
            else:
                first = chunk[0]
                last = chunk[-1]
                first_word = first.get("mfa_word")
                last_word = last.get("mfa_word")
                if (not isinstance(first_word, dict) or not isinstance(last_word, dict)
                        or any(item.get("word_id", "").rsplit(":w", 1)[0]
                               != first.get("word_id", "").rsplit(":w", 1)[0]
                               for item in chunk if isinstance(item, dict))):
                    return _strict_en_fail(
                        required, "strict_en_word_count_mismatch", ledger_sha256=actual_hash,
                        failed_word_ids=[str(item.get("word_id", "")) for item in chunk
                                         if isinstance(item, dict)])
                combined = dict(first)
                combined["ctc_text"] = final_word.text.strip()
                combined["ctc_ordinal"] = last.get("ctc_ordinal")
                combined["source_records"] = [dict(item) for item in chunk]
                combined["source_word_ids"] = [item.get("word_id") for item in chunk]
                combined["mfa_word"] = {
                    "ordinal": first_word.get("ordinal"),
                    "text": final_word.text.strip(),
                    "start": first_word.get("start"),
                    "end": last_word.get("end"),
                }
                combined_phones: list[dict] = []
                for item in chunk:
                    for phone in item.get("phones", []):
                        if not isinstance(phone, dict):
                            continue
                        copied = dict(phone)
                        # ``ordinal`` is local to the newly combined final word,
                        # but ``mfa_phone_ordinal`` is immutable provenance in
                        # the source segment.  Re-numbering both makes a later
                        # grouped word collide with earlier phones in the same
                        # segment (for example ``ria`` + ``Mil``/``ive``).
                        copied["ordinal"] = len(combined_phones)
                        combined_phones.append(copied)
                combined["phones"] = combined_phones
                grouped_records.append(combined)
            record_cursor = matched_end
        if record_cursor != len(records):
            return _strict_en_fail(
                required, "strict_en_word_count_mismatch", ledger_sha256=actual_hash,
                failed_word_ids=[str(item.get("word_id", "")) for item in records[record_cursor:]
                                 if isinstance(item, dict)])
        records = grouped_records
    ordered_english_words = english_words
    if isinstance(correspondence, dict):
        if not correspondence.get("safe"):
            return _strict_en_fail(required, "fallback_correspondence_rejected",
                                   ledger_sha256=actual_hash)
        # ``source_ctc_ordinals`` are raw interval ordinals in the *processed
        # CTC* TextGrid consumed by the English producer.  They are neither
        # the raw interval ordinals of the MFA source words tier
        # (``source_to_final``) nor compact lexical ordinals
        # (``ctc_to_final``).  Energy adjustment may insert blank intervals,
        # so the three axes can differ even for a one-word English unit.
        # Project the producer's raw processed-CTC owner onto the compact CTC
        # axis first, then consume the correspondence ledger's CTC mapping.
        if processed_ctc_words_tier is None:
            return _strict_en_fail(
                required, "fallback_processed_ctc_owner_axis_missing",
                ledger_sha256=actual_hash)
        processed_ctc_to_lexical: dict[int, int] = {}
        lexical_ordinal = 0
        for interval_ordinal, interval in enumerate(
                processed_ctc_words_tier.intervals):
            text = interval.text.strip()
            if not text or is_silence(text) or is_punct(text):
                continue
            processed_ctc_to_lexical[interval_ordinal] = lexical_ordinal
            lexical_ordinal += 1
        ctc_to_final = correspondence.get("mapping", {}).get(
            "ctc_to_final", {})
        if (not isinstance(ctc_to_final, dict)
                or lexical_ordinal != len(ctc_tokens or ())):
            return _strict_en_fail(
                required, "fallback_processed_ctc_owner_axis_invalid",
                ledger_sha256=actual_hash)
        final_lexical = [iv for iv in (words_tier.intervals if words_tier else [])
                         if iv.text.strip() and not is_silence(iv.text)
                         and not is_punct(iv.text)]
        bound: list[Interval] = []
        for record in records:
            try:
                source_ordinals = record["source_ctc_ordinals"]
                if (not isinstance(source_ordinals, list) or not source_ordinals
                        or any(type(value) is not int or value < 0
                               for value in source_ordinals)):
                    raise ValueError("invalid source CTC ordinals")
                owner_ordinals = {
                    int(ctc_to_final[str(processed_ctc_to_lexical[value])])
                    for value in source_ordinals
                }
            except (KeyError, TypeError, ValueError):
                return _strict_en_fail(required,
                                       "fallback_correspondence_owner_missing",
                                       ledger_sha256=actual_hash)
            if len(owner_ordinals) != 1:
                return _strict_en_fail(
                    required, "fallback_correspondence_non_unique_owner",
                    ledger_sha256=actual_hash)
            final_ordinal = next(iter(owner_ordinals))
            if final_ordinal < 0 or final_ordinal >= len(final_lexical):
                return _strict_en_fail(required,
                                       "fallback_correspondence_owner_missing",
                                       ledger_sha256=actual_hash)
            owner = final_lexical[final_ordinal]
            if not is_english_token(owner.text.strip()):
                return _strict_en_fail(required,
                                       "fallback_correspondence_owner_mismatch",
                                       ledger_sha256=actual_hash)
            bound.append(owner)
        if len({id(owner) for owner in bound}) != len(bound):
            return _strict_en_fail(required, "fallback_correspondence_non_unique_owner",
                                   ledger_sha256=actual_hash)
        ordered_english_words = bound

    pairs: list[tuple[Interval, dict]] = []
    previous_ctc_ordinal = -1
    seen_word_ids: set[str] = set()
    # MFA phone ordinals restart at zero for each English segment.  Scope the
    # uniqueness check by segment; treating them as stem-global rejects every
    # stem containing more than one English segment even when the ledger and
    # source TextGrids are valid.
    seen_mfa_phone_ordinals: set[tuple[str, int]] = set()
    for final_word, record in zip(ordered_english_words, records):
        if not isinstance(record, dict):
            return _strict_en_fail(required, "strict_en_word_invalid", ledger_sha256=actual_hash)
        word_id = record.get("word_id")
        mfa_word = record.get("mfa_word")
        phones = record.get("phones")
        try:
            ordinal = int(record["ctc_ordinal"])
        except (KeyError, TypeError, ValueError):
            ordinal = -1
        if (record.get("status") != "verified" or record.get("provenance") != "english_mfa_textgrid"
                or not isinstance(word_id, str) or not word_id or word_id in seen_word_ids
                or ordinal <= previous_ctc_ordinal
            or re.sub(r"[^a-z0-9]", "", str(record.get("alignment_token", "")).casefold())
               != re.sub(r"[^a-z0-9]", "", final_word.text.strip().casefold())
                or not isinstance(mfa_word, dict) or not isinstance(phones, list) or not phones):
            return _strict_en_fail(required, "strict_en_word_identity_or_evidence_invalid",
                                   ledger_sha256=actual_hash, failed_word_ids=[str(word_id or "")])
        seen_word_ids.add(word_id); previous_ctc_ordinal = ordinal
        segment_key = word_id.rsplit(":w", 1)[0]
        try:
            if (not isinstance(mfa_word.get("ordinal"), int)
                    or mfa_word["ordinal"] < 0
                    or re.sub(r"[^a-z0-9]", "", str(mfa_word.get("text", "")).casefold())
                       != re.sub(r"[^a-z0-9]", "", str(record.get("alignment_token", "")).casefold())):
                raise ValueError("mfa_word_identity")
        except Exception:
            return _strict_en_fail(required, "strict_en_mfa_word_invalid", ledger_sha256=actual_hash,
                                   failed_word_ids=[word_id])
        prior_end = -math.inf
        for phone_ordinal, phone in enumerate(phones):
            if (not isinstance(phone, dict) or phone.get("ordinal") != phone_ordinal
                    or not _strict_en_phone_is_valid(phone, mfa_word)):
                return _strict_en_fail(required, "strict_en_phone_invalid", ledger_sha256=actual_hash,
                                       failed_word_ids=[word_id])
            if float(phone["start"]) < prior_end:
                return _strict_en_fail(required, "strict_en_phone_unordered", ledger_sha256=actual_hash,
                                       failed_word_ids=[word_id])
            prior_end = float(phone["end"])
            phone_key = (segment_key, phone.get("mfa_phone_ordinal"))
            if (not isinstance(phone.get("mfa_phone_ordinal"), int)
                    or phone["mfa_phone_ordinal"] < 0
                    or phone_key in seen_mfa_phone_ordinals):
                return _strict_en_fail(required, "strict_en_mfa_phone_ordinal_invalid",
                                       ledger_sha256=actual_hash, failed_word_ids=[word_id])
            seen_mfa_phone_ordinals.add(phone_key)
        if authority_units:
            unit = authority_units[len(pairs)]
            if (record.get("unit_id") != unit.unit_id
                    or record.get("alignment_token") != unit.alignment_token
                    or final_word.text.strip() != unit.surface_text):
                return _strict_en_fail(required, "strict_en_unit_owner_mismatch",
                                       ledger_sha256=actual_hash,
                                       failed_word_ids=[word_id])
        if hanzi_tier is not None:
            owners = [iv for iv in hanzi_tier.intervals
                      if is_english_token(iv.text.strip())
                      and abs(iv.xmin - final_word.xmin) <= AXIS_EPS
                      and abs(iv.xmax - final_word.xmax) <= AXIS_EPS]
            if len(owners) != 1:
                return _strict_en_fail(required, "strict_en_hanzi_owner_mismatch",
                                       ledger_sha256=actual_hash,
                                       failed_word_ids=[word_id])
        pairs.append((final_word, record))
    return _strict_en_report("verified", required, required, [], actual_hash), pairs


def _strip_english_phone_intervals(pp_tier: Tier | None, words_tier: Tier | None) -> Tier | None:
    """Remove non-provenance English phone candidates from a filtered output."""
    if pp_tier is None:
        return None
    english = _strict_en_lexical_words(words_tier)
    if not english:
        return pp_tier
    retained = [phone for phone in pp_tier.intervals if not any(
        phone.xmax > word.xmin + 0.001 and phone.xmin < word.xmax - 0.001
        for word in english)]
    return Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, retained)


def inject_strict_en_phones(pp_tier: Tier | None, words_tier: Tier | None,
                            pairs: list[tuple[Interval, dict]]) -> Tier | None:
    """Affine-map exact MFA ARPABET evidence without snapping or relabelling."""
    base = _strip_english_phone_intervals(pp_tier, words_tier)
    if base is None:
        return None
    injected = list(base.intervals)
    for final_word, record in pairs:
        mfa_word = record["mfa_word"]
        source_start, source_end = float(mfa_word["start"]), float(mfa_word["end"])
        final_duration = final_word.xmax - final_word.xmin
        if final_duration <= 0:
            raise ValueError("strict_en_final_word_invalid")
        for phone in record["phones"]:
            start = final_word.xmin + ((float(phone["start"]) - source_start)
                                       / (source_end - source_start)) * final_duration
            end = final_word.xmin + ((float(phone["end"]) - source_start)
                                     / (source_end - source_start)) * final_duration
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("strict_en_affine_invalid")
            injected.append(Interval(start, end, f"{EN_PHONE_PREFIX}{phone['label']}"))
    injected.sort(key=lambda iv: (iv.xmin, iv.xmax, iv.text))
    return Tier(base.name, base.xmin, base.xmax, injected)


def _apply_en_phones(words_tier: Tier, pp_tier: Tier | None,
                     en_data: list[dict],
                     phone_prefix: str = "") -> tuple[Tier, Tier | None]:
    """Inject English MFA phonemes into phone tier, replacing self-referencing intervals.

    Strategy (avoids fragile phone-pool index tracking that conflicts with
    _snap_to_ctc's internal merging):

      1. Identify English word time ranges from the words tier.
      2. Filter the pp_tier: keep all intervals that do NOT fall inside an
         English word range.
      3. For each English word, look up its English MFA phonemes from
         *en_data* and inject them with proportional scaling to fit the
         CTC-snapped word boundaries.  When *phone_prefix* is set (e.g.
         ``"en:"``), every injected phone label is prefixed.
      4. Sort by xmin.

    Non-English and silence intervals pass through untouched.
    When *en_data* is None or empty, the function is a no-op.
    """
    if not en_data or pp_tier is None:
        return words_tier, pp_tier

    # Build time-ordered English word ranges from the words tier
    en_ranges: list[tuple[float, float, str]] = []
    for w_iv in words_tier.intervals:
        text = w_iv.text.strip()
        if is_english_token(text):
            en_ranges.append((w_iv.xmin, w_iv.xmax, text.lower()))

    if not en_ranges:
        return words_tier, pp_tier

    # Build English phone lookup: (text, rounded_start_0.5s) -> entry.
    # Coarse 0.5s rounding handles MFA-compressed word starts.
    en_lookup: dict[tuple[str, float], dict] = {}
    for entry in en_data:
        key = (entry["word_text"].strip().lower(), round(entry["word_start"] * 2) / 2)
        en_lookup[key] = entry

    # Step 1: Remove ALL phones that fall completely inside any English word's
    # time range (with 0.05s margin to catch spn/sil that Chinese MFA placed
    # slightly before/after the CTC-snapped word boundary).
    # Chinese MFA may assign spn, sil, or other non-matching labels to
    # self-referencing English tokens — a text-based match misses those.
    new_phone_ivs: list[Interval] = []
    _margin = 0.05
    for p_iv in pp_tier.intervals:
        removed = False
        for es, ee, _ in en_ranges:
            if es - _margin <= p_iv.xmin and p_iv.xmax <= ee + _margin:
                removed = True
                break
        if not removed:
            new_phone_ivs.append(p_iv)

    # Step 2: Inject canonical ARPABET phonemes for each English word.
    # When English MFA alignment is available, use its real timing
    # proportions — only the phone LABELS come from CMUdict.  When the
    # phone counts differ (e.g. CMUdict has an extra Y glide), the
    # closest IPA slot is split evenly.  Without English MFA data,
    # phones are distributed evenly across the word.
    from pipeline_utils import _load_cmudict, en_ipa_to_arpabet
    cmu = _load_cmudict()
    for w_start, w_end, w_text in en_ranges:
        word_dur = w_end - w_start if w_end > w_start else 0.001

        # ── Resolve English MFA timing ──
        key = (w_text, round(w_start * 2) / 2)
        en_entry = en_lookup.get(key)
        if en_entry is None:
            for entry in en_data:
                if entry["word_text"].strip().lower() == w_text:
                    if abs(entry["word_start"] - w_start) < 1.0:
                        en_entry = entry
                        break

        # ── Try CMUdict for canonical labels ──
        cmu_phones = cmu.get(w_text) if cmu else None

        if cmu_phones and en_entry and en_entry.get("phones"):
            # Build relative time slices from English MFA IPA phones.
            # Each slice maps to its ARPABET equivalent via en_ipa_to_arpabet.
            en_start = en_entry.get("en_word_start", en_entry["word_start"])
            en_end = en_entry.get("en_word_end", en_entry["word_end"])
            en_dur = en_end - en_start if en_end > en_start else word_dur
            ipa_slices: list[tuple[float, float, str]] = []
            for p in en_entry["phones"]:
                rs = max(0.0, min(1.0, (p["start"] - en_start) / en_dur))
                re = max(0.0, min(1.0, (p["end"] - en_start) / en_dur))
                arpa = en_ipa_to_arpabet(phone_prefix + p["phone"])
                arpa_clean = arpa[len(phone_prefix):] if arpa.startswith(phone_prefix) else arpa
                ipa_slices.append((rs, re, arpa_clean))

            # Distribute CMUdict phones across IPA time slices.
            # Greedy: for each CMUdict phone, consume IPA slices until
            # the slice's ARPABET class matches.  Unmatched slices are
            # merged into the closest matching CMUdict phone.
            n_cmu = len(cmu_phones)
            n_ipa = len(ipa_slices)
            if n_cmu == n_ipa:
                # 1:1 — use IPA timings directly with CMUdict labels
                for i in range(n_cmu):
                    rs, re, _ = ipa_slices[i]
                    s = round(w_start + rs * word_dur, 4)
                    e = round(w_start + re * word_dur, 4)
                    if e > s + 0.010:
                        label = f"{phone_prefix}{cmu_phones[i]}"
                        new_phone_ivs.append(Interval(s, e, label))
            elif n_cmu > n_ipa:
                # More CMUdict phones than IPA slices — split the longest slice(s)
                # to make room.  Build target relative cuts from IPA boundaries,
                # then assign CMUdict phones proportionally.
                cuts = [0.0]
                for _, re, _ in ipa_slices:
                    cuts.append(re)
                # Split the widest slice until we have enough segments
                while len(cuts) - 1 < n_cmu:
                    widest_i = max(range(len(cuts) - 1), key=lambda i: cuts[i + 1] - cuts[i])
                    mid = (cuts[widest_i] + cuts[widest_i + 1]) / 2.0
                    cuts.insert(widest_i + 1, mid)
                for i in range(n_cmu):
                    s = round(w_start + cuts[i] * word_dur, 4)
                    e = round(w_start + cuts[i + 1] * word_dur, 4)
                    new_phone_ivs.append(Interval(s, e, f"{phone_prefix}{cmu_phones[i]}"))
            else:
                # Fewer CMUdict phones than IPA slices — merge smallest gaps
                cuts = [0.0]
                for _, re, _ in ipa_slices:
                    cuts.append(re)
                while len(cuts) - 1 > n_cmu:
                    narrowest_i = min(range(len(cuts) - 1), key=lambda i: cuts[i + 1] - cuts[i])
                    del cuts[narrowest_i + 1]
                for i in range(n_cmu):
                    s = round(w_start + cuts[i] * word_dur, 4)
                    e = round(w_start + cuts[i + 1] * word_dur, 4)
                    new_phone_ivs.append(Interval(s, e, f"{phone_prefix}{cmu_phones[i]}"))
            continue

        if cmu_phones and len(cmu_phones) >= 1:
            # CMUdict available but no English MFA timing — even distribution
            n = len(cmu_phones)
            for i, arpa in enumerate(cmu_phones):
                s = round(w_start + (i / n) * word_dur, 4)
                e = round(w_start + ((i + 1) / n) * word_dur, 4)
                label = f"{phone_prefix}{arpa}"
                new_phone_ivs.append(Interval(s, e, label))
            continue

        # ── Fallback: use English MFA-aligned IPA phones ──
        key = (w_text, round(w_start * 2) / 2)
        en_entry = en_lookup.get(key)

        if en_entry is None:
            for entry in en_data:
                if entry["word_text"].strip().lower() == w_text:
                    if abs(entry["word_start"] - w_start) < 1.0:
                        en_entry = entry
                        break

        if en_entry and en_entry.get("phones"):
            en_start = en_entry.get("en_word_start", en_entry["word_start"])
            en_end = en_entry.get("en_word_end", en_entry["word_end"])
            en_dur = en_end - en_start if en_end > en_start else word_dur

            for p in en_entry["phones"]:
                rel_start = (p["start"] - en_start) / en_dur if en_dur > 0 else 0.0
                rel_end = (p["end"] - en_start) / en_dur if en_dur > 0 else 1.0
                rel_start = max(0.0, min(1.0, rel_start))
                rel_end = max(0.0, min(1.0, rel_end))
                mapped_start = round(w_start + rel_start * word_dur, 4)
                mapped_end = round(w_start + rel_end * word_dur, 4)
                if mapped_end > mapped_start + 0.010:
                    label = f"{phone_prefix}{p['phone']}"
                    new_phone_ivs.append(Interval(mapped_start, mapped_end, label))
        else:
            # No data at all — keep self-referencing as fallback
            label = f"{phone_prefix}{w_text}" if phone_prefix else w_text
            new_phone_ivs.append(Interval(w_start, w_end, label))

    # ── Snap English phone edges to word boundaries (Regr. Case 40) ──
    # After English MFA phones are injected and proportionally scaled,
    # snap the first phone's start and last phone's end to the word
    # boundaries.  This prevents boundary offsets caused by the linear
    # scaling from English MFA's padded segments to CTC-snapped words.
    for w_start, w_end, w_text in en_ranges:
        en_phones_for_word = [(idx, iv) for idx, iv in enumerate(new_phone_ivs)
                              if w_start <= iv.xmin and iv.xmax <= w_end + 0.005
                              and not is_silence(iv.text)]
        if not en_phones_for_word:
            continue
        # Snap first phone start to word start
        first_idx, first_iv = en_phones_for_word[0]
        if first_iv.xmin > w_start + 0.002:
            new_phone_ivs[first_idx] = Interval(w_start, first_iv.xmax, first_iv.text)
        # Snap last phone end to word end
        last_idx, last_iv = en_phones_for_word[-1]
        if w_end > last_iv.xmax + 0.002:
            new_phone_ivs[last_idx] = Interval(last_iv.xmin, w_end, last_iv.text)

    # Sort and merge same-text intervals
    new_phone_ivs.sort(key=lambda iv: iv.xmin)
    merged: list[Interval] = []
    for iv in new_phone_ivs:
        if (merged
                and merged[-1].text == iv.text
                and merged[-1].xmax >= iv.xmin - 0.001):
            merged[-1] = Interval(merged[-1].xmin,
                                  max(merged[-1].xmax, iv.xmax),
                                  merged[-1].text)
        else:
            merged.append(iv)

    # Deconflict: resolve overlapping intervals with different texts.
    # English phones take priority over silence; for non-silence overlaps
    # the later interval is clipped to start after the earlier one ends.
    resolved: list[Interval] = []
    for iv in merged:
        if not resolved:
            resolved.append(iv)
            continue
        prev = resolved[-1]
        if iv.xmin >= prev.xmax - 0.002:
            resolved.append(iv)
        elif is_silence(prev.text) and not is_silence(iv.text):
            # Silence before speech: trim silence
            new_end = iv.xmin
            if new_end > prev.xmin + 0.002:
                resolved[-1] = Interval(prev.xmin, new_end, prev.text)
            else:
                resolved.pop()  # silence reduced to zero — drop it
            resolved.append(iv)
        elif not is_silence(prev.text) and is_silence(iv.text):
            # Speech before silence: clip silence start
            new_start = max(iv.xmin, prev.xmax)
            if iv.xmax > new_start + 0.002:
                resolved.append(Interval(new_start, iv.xmax, iv.text))
            # else: silence fully covered by speech — drop
        elif iv.xmin < prev.xmax:
            # Two non-silence intervals overlap.
            if iv.xmax > prev.xmax + 0.002:
                # Later extends beyond earlier — clip to start after earlier
                resolved.append(Interval(prev.xmax, iv.xmax, iv.text))
            # else: later is fully inside earlier — keep it (don't drop);
            #       the merge step will handle same-text consolidation.
            else:
                resolved.append(iv)
        else:
            resolved.append(iv)

    new_pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, resolved)
    return words_tier, new_pp_tier


def _apply_en_stress(words_tier: Tier, pp_intervals: list[Interval]) -> None:
    """Apply CMUdict stress markers to ARPABET phones in-place.

    For each English word in *words_tier*, collects the corresponding
    ``en:`` phones from *pp_intervals* and applies stress markers via
    :func:`apply_arpabet_stress`.  Phones without stress data are left
    unchanged (unstressed-0 by default).
    """
    if not pp_intervals:
        return

    for w_iv in words_tier.intervals:
        text = w_iv.text.strip()
        if not is_english_token(text):
            continue
        # Collect en: phones for this word
        indices = []
        phones = []
        for i, iv in enumerate(pp_intervals):
            if iv.xmin >= w_iv.xmin - 0.002 and iv.xmax <= w_iv.xmax + 0.002:
                if iv.text.startswith(EN_PHONE_PREFIX):
                    indices.append(i)
                    phones.append(iv.text[len(EN_PHONE_PREFIX):])
        if not indices:
            continue

        # Apply stress
        stressed = apply_arpabet_stress(phones, text)
        if stressed == phones:
            continue  # no change

        for idx, new_phone in zip(indices, stressed):
            pp_intervals[idx] = Interval(
                pp_intervals[idx].xmin,
                pp_intervals[idx].xmax,
                f"{EN_PHONE_PREFIX}{new_phone}",
            )


def _strict_en_provenance_enabled(args) -> bool:
    """Require strict English ledgers independently of global strict-ok QC."""
    return bool(getattr(args, "strict_ok", False)
                or getattr(args, "strict_en_provenance", False))


def process_one(tg_path: Path, txt_dir: Path, wav_dir: Path,
                output_dir: Path, filtered_dir: Path, args,
                ipa_to_pinyin: dict[str, str],
                pinyin_dict: dict[str, list[str]],
                pinyin_case: dict[str, str] | None = None,
                raw_text_index: dict[str, Path] | None = None,
                wav_index: dict[str, Path] | None = None) -> dict:
    """Post-process a single MFA-aligned TextGrid into 5-tier output.

    PROCESSING ORDER IS CRITICAL.  The function is organised in 5 phases:
      Phase 1 — Acoustic preprocessing (silence merge, short-word fix)
      Phase 2 — Text correction & tier finalisation (hanzi, corrected_text)
      Phase 3 — Boundary adjustments (snap->CTC, energy refine, punct inject)
      Phase 4 — Post-boundary processing (unexpected sil, NVV/ellipsis merges)
      Phase 5 — Final text sync & QC

    English MFA phoneme injection runs BETWEEN Phase 3 and Phase 4:
      Phase 3.5 — _apply_en_phones: inject English MFA phonemes into
                 the words and phones tiers (only when en_data is available).

    DO NOT REORDER steps within or across phases without understanding
    the dependency chain documented at each phase boundary.
    """
    stem = tg_path.stem

    # Load English MFA phone data.
    # Auto-detect en_phones dir from workspace if not explicitly provided.
    strict_en_mode = _strict_en_provenance_enabled(args)
    en_phones_dir = getattr(args, 'en_phones_dir', None)
    if en_phones_dir is None:
        auto_dir = output_dir.parent / "en_phones"
        if auto_dir.exists():
            en_phones_dir = auto_dir
    # A strict run must never deserialize the historical list JSON: it enables
    # CMUdict/equal-split recovery in the legacy branch below.  Strict evidence
    # is loaded only after the final words tier is settled.
    en_data = None if strict_en_mode else load_en_phones(stem, en_phones_dir)
    report: dict = {"stem": stem, "status": "ok", "warnings": []}
    ctc_lifecycle = _load_ctc_lifecycle(txt_dir, stem)
    if ctc_lifecycle is None:
        report["ctc_lifecycle"] = {
            "schema": "ctc-processed-input-lifecycle-v1",
            "status": "legacy_single_directory_fixture",
        }
    else:
        report["ctc_lifecycle"] = ctc_lifecycle
    txt_path = txt_dir / f"{stem}.txt"
    lab_path = txt_dir / f"{stem}.lab"
    reference_mode_policy = getattr(args, "reference_mode", "auto")
    # Bind lab_fallback to the actual .lab artifact when the optional .txt is
    # empty.  A non-empty .txt remains the normal pinyin source.
    if (not txt_path.exists() or
            (not txt_path.read_text(encoding="utf-8").strip()
             and lab_path.is_file() and lab_path.read_text(encoding="utf-8").strip())):
        txt_path = lab_path
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing txt/lab: {txt_dir}/{stem}")
    tg = parse_textgrid(tg_path)
    if len(tg.tiers) < 2:
        raise ValueError(f"Need at least 2 tiers in {tg_path}")
    words_tier = tg.tiers[0]
    phones_tier = tg.tiers[1]
    source_words_snapshot = _source_unknown_context(words_tier)
    source_phone_lineage = _bind_source_phone_lineage(words_tier, phones_tier)
    mfa_unknown_before_snap = [
        iv.text.strip() for iv in words_tier.intervals
        if is_unknown_token(iv.text)
    ]
    mfa_unknown_intervals = [
        (float(iv.xmin), float(iv.xmax), iv.text.strip())
        for iv in words_tier.intervals if is_unknown_token(iv.text)
    ]

    # Fix MFA's forced lowercase: use dictionary's canonical form
    if pinyin_case:
        for iv in words_tier.intervals:
            word = iv.text.strip()
            if word and not is_silence(word):
                canonical = pinyin_case.get(word.lower())
                if canonical is not None and canonical != word:
                    iv.text = canonical

    # Tier 1: original/reference Chinese text.  This flag is intentionally
    # captured before ASR fallback: CTC may provide boundaries and language
    # hints, but it must not replace a supplied reference transcript.
    # An explicit fallback policy does not even inspect raw_text_dir, so a
    # mixed directory or stale reference sidecar cannot upgrade one stem to
    # authority while the rest of the batch remains ASR-only.
    reference_text_original_raw = ("" if reference_mode_policy == "fallback" else
                                   find_original_text(stem, args.raw_text_dir, raw_text_index))
    reference_source_path = (find_original_text_path(
        stem, args.raw_text_dir, raw_text_index)
        if reference_text_original_raw else None)
    reference_text_original = reference_text_original_raw
    raw_text = reference_text_original
    reference_text_authoritative = bool(reference_text_original_raw)
    if reference_mode_policy == "authority" and not reference_text_authoritative:
        raise FileNotFoundError(
            f"reference-mode=authority requires reference text for {stem}")
    reference_source = "original_or_ref" if reference_text_authoritative else ""
    if reference_text_authoritative:
        # Keep the loaded reference immutable for provenance.  Only the
        # authority projection receives numeral normalization; CTC labs,
        # tokens, and the pinyin source remain untouched.
        try:
            import cn2an as _cn2an
            _numeral_transform = _cn2an.transform
        except ImportError:
            _numeral_transform = None
        reference_text_original, _numeral_report = (
            normalize_authority_reference_numerals(
                reference_text_original_raw, _numeral_transform,
                return_report=True))
        raw_reference_bytes = (reference_source_path.read_bytes()
                               if reference_source_path is not None
                               else reference_text_original_raw.encode("utf-8"))
        _numeral_report["raw_sha256"] = hashlib.sha256(
            raw_reference_bytes).hexdigest()
        if reference_source_path is not None:
            _numeral_report["raw_path"] = str(reference_source_path.resolve())
        _numeral_report["normalized_sha256"] = hashlib.sha256(
            reference_text_original.encode("utf-8")).hexdigest()
        report["reference_text_original_raw"] = reference_text_original_raw
        report["reference_text_normalized"] = reference_text_original
        report["reference_text_raw_sha256"] = _numeral_report["raw_sha256"]
        report["reference_numeral_normalization"] = _numeral_report
        # Canonicalize only the in-memory authority string.  The original
        # source remains byte-for-byte untouched on disk.
        raw_text = _canonicalize_reference_hyphens(reference_text_original)
    if not reference_text_authoritative:
        # Try NVASR Chinese ASR output
        cn_path = txt_dir / f"{stem}_text_cn.txt"
        if cn_path.exists():
            raw_text = cn_path.read_text(encoding="utf-8").strip()
            reference_source = "asr_fallback"
    if not raw_text:
        # Fallback: use the pinyin txt content
        raw_text = txt_path.read_text(encoding="utf-8").strip()
        reference_source = "lab_fallback"
    if not reference_text_authoritative:
        reference_text_original = raw_text
    fallback_surface_ledger = (
        _fallback_punctuation_surface_ledger(raw_text)
        if not reference_text_authoritative else None)
    fallback_punctuation_projection = None
    if fallback_surface_ledger is not None:
        report["fallback_punctuation_surface"] = fallback_surface_ledger
    report["reference_mode"] = (
        "authority" if reference_text_authoritative else "fallback")
    report["reference_source"] = reference_source
    report["reference_text_authoritative"] = reference_text_authoritative
    if not reference_text_authoritative:
        report["reference_numeral_normalization"] = {
            "schema": REFERENCE_NUMERAL_NORMALIZATION_SCHEMA,
            "status": "not_applicable",
        }
    if not reference_text_authoritative:
        fallback_path = (cn_path if reference_source == "asr_fallback" else txt_path)
        report["fallback_transcript"] = {
            "source": reference_source,
            "path": str(fallback_path.resolve()),
            "sha256": hashlib.sha256(fallback_path.read_bytes()).hexdigest(),
        }

    # Tier 2: pinyin with punctuation (from corpus txt)
    pinyin_text = txt_path.read_text(encoding="utf-8").strip()
    if reference_text_authoritative:
        # CTC/lab text is normalized with the same lexical policy so a lab
        # containing ``v-tuber`` aligns identically to ``v tu ber``.
        pinyin_text = _canonicalize_reference_hyphens(pinyin_text)
    pinyin_text_original = pinyin_text

    # Fix <unk>/[bracketed] from MFA: self-referential NVV / English tokens
    # (BREATHING, li, ve etc.).  MFA replaces unknown tokens with <unk> or
    # [bracketed]; restore them from .lab tokens using CTC timestamps.
    lab_tokens = pinyin_text.split()
    # Load CTC token timestamps for time-based matching
    ctc_token_list: list[dict] = []
    tokens_path = txt_dir / f"{stem}_tokens.jsonl"
    if tokens_path.exists():
        for line in tokens_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                token = json.loads(line)
                if (reference_text_authoritative and isinstance(token, dict)
                        and isinstance(token.get("word"), str)):
                    token = dict(token)
                    token["word"] = _canonicalize_reference_hyphens(token["word"])
                ctc_token_list.append(token)

    # Restore only explicit MFA unknown placeholders.  A no-reference CTC
    # English anchor is not authority to relabel a neighbouring pinyin word.
    if ctc_token_list and not reference_text_authoritative:
        words_tier = _restore_fallback_unknown_surfaces(
            words_tier, ctc_token_list)
        fallback_unknown_projection = getattr(
            words_tier, "_fallback_unknown_projection", None)
        if isinstance(fallback_unknown_projection, dict):
            report["fallback_unknown_projection"] = fallback_unknown_projection

    raw_tier = Tier("raw_text", tg.xmin, tg.xmax,
                    [Interval(tg.xmin, tg.xmax, raw_text)])
    pinyin_tier = Tier("pinyin", tg.xmin, tg.xmax,
                       [Interval(tg.xmin, tg.xmax, pinyin_text)])
    pinyin_phones_tier = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin,
                                                   words_tier, pinyin_dict)

    # Build 5 tiers
    tiers = [raw_tier, pinyin_tier, words_tier, phones_tier, pinyin_phones_tier]
    new_tg = TextGrid(tg.xmin, tg.xmax, tiers)
    new_tg._phone_lineage = source_phone_lineage
    if fallback_surface_ledger is not None:
        new_tg._fallback_punctuation_surface_ledger = fallback_surface_ledger

    # Find WAV once from the batch index (may be in a subdirectory).
    wav_path = _find_wav(stem, wav_dir, wav_index)
    # Load WAV once for all audio-dependent steps
    wav_audio = wav_sr = None
    if wav_path.exists():
        try:
            wav_audio, wav_sr = load_audio(wav_path)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # Phase 1 — Acoustic preprocessing.
    # These must run BEFORE boundary adjustments (Phase 3) because
    # they operate on raw MFA phone/word boundaries.
    # ═══════════════════════════════════════════════════════════════

    # Short inter-word silence is owned only by the final visual resolver.
    # Keep the legacy option/API for compatibility, but do not let Phase 1
    # mutate words or phones before the final authority snapshot exists.
    merge_report = []
    report["silence_merges"] = merge_report

    if args.fix_short_word:
        new_tg, fixes = fix_short_words(new_tg, wav_path if wav_path.exists() else None, args,
                                        wav_audio, wav_sr)
        report["fixes"] = fixes

    # Rebuild pinyin_phones after merge/fix may have changed phone boundaries
    if merge_report or (args.fix_short_word and fixes):
        phones_tier = tier_by_name(new_tg, "phones")
        cur_words_tier = tier_by_name(new_tg, "words")
        if phones_tier is not None:
            rebuilt = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin, cur_words_tier, pinyin_dict)
            for i, tier in enumerate(new_tg.tiers):
                if tier.name.lower() == "pinyin_phones":
                    new_tg.tiers[i] = rebuilt
                    break

    # BGM/noise detection — moved to final check after all processing
    bgm_issues = []

    # Phone-level QC (short_phone, long_consonant, long_vowel) was
    # previously called here on raw MFA phones.  MFA boundaries near
    # English/NVV words are often too short, but the Phase-3 boundary
    # adjustments (_snap_to_ctc, _refine_boundaries_by_energy) stretch
    # them to realistic durations.  The phone checks now run in Phase 5
    # with the corrected boundaries.
    align_issues = []

    # ═══════════════════════════════════════════════════════════════
    # Phase 2 — Text correction & tier finalisation.
    # Must run AFTER Phase 1 (needs merged silences) and BEFORE
    # Phase 3 boundary adjustments (boundary changes invalidate
    # corrected_text's punctuation-silence cross-check).
    # ═══════════════════════════════════════════════════════════════

    # Relabel all silences
    new_tiers = []
    for tier in new_tg.tiers:
        relabeled = [Interval(iv.xmin, iv.xmax,
                              silence_label(iv.duration) if is_silence(iv.text) else iv.text)
                     for iv in tier.intervals]
        new_tiers.append(Tier(tier.name, tier.xmin, tier.xmax, relabeled))
    new_tg = TextGrid(new_tg.xmin, new_tg.xmax, new_tiers)
    new_tg._phone_lineage = source_phone_lineage
    if fallback_surface_ledger is not None:
        new_tg._fallback_punctuation_surface_ledger = fallback_surface_ledger

    # Tier 6: corrected Chinese text (punctuation ↔ silence cross-check)
    if args.enable_text_correction:
        words_tier = tier_by_name(new_tg, "words")
        if words_tier is not None:
            try:
                corrected = build_corrected_text(words_tier, raw_text, pinyin_text)
            except Exception:
                corrected = raw_text
            if corrected != raw_text:
                report["text_corrected"] = True
            corrected_tier = Tier("corrected_text", new_tg.xmin, new_tg.xmax,
                                  [Interval(new_tg.xmin, new_tg.xmax, corrected)])
            new_tg.tiers.append(corrected_tier)

    # Finalise: strip [sp] markers (merged), add <sp1> prefix,
    # sync pinyin, insert hanzi tier, reorder everything.
    # NOTE: warnings are NOT passed here — the hanzi tier built by
    # _finalise_textgrid is a throwaway (replaced in Phase 5).
    # Passing warnings would duplicate every mismatch message.
    if args.enable_text_correction:
        new_tg = _finalise_textgrid(
            new_tg, raw_text, pinyin_text, args,
            reference_authoritative=reference_text_authoritative)
        new_tg._phone_lineage = source_phone_lineage

    # ═══════════════════════════════════════════════════════════════
    # Phase 3 — Boundary adjustments (ORDER CRITICAL — DO NOT REORDER).
    #
    #   A. _snap_to_ctc          — authoritative word boundaries (CTC anchors)
    #   B. _refine_boundaries_by_energy — energy-based fine-tuning
    #   C. _inject_punctuation   — inject CTC punct anchors into words tier
    #
    # Rationale:
    #   A must be first: establishes the ground-truth word boundaries.
    #   B must be after A: needs snapped boundaries for RMS comparison.
    #   C must be after A+B: punct injection needs final word positions
    #     to correctly resolve word-punct overlaps.
    # ═══════════════════════════════════════════════════════════════

    # 输出路径先默认 output, 最终检查时再决定是否重定向到 filtered
    out_path = output_dir / tg_path.name
    stale = filtered_dir / tg_path.name

    # --- A. Snap MFA word boundaries to CTC anchors ---
    tokens_path = txt_dir / f"{stem}_tokens.jsonl"
    punct_path = txt_dir / f"{stem}_punct.json"
    _punct_boundary_hits: list[dict] = []
    punct_entries = []
    punctuation_evidence_schema_valid = True
    if punct_path.exists():
        punct_entries = json.loads(punct_path.read_text(encoding="utf-8"))
        punctuation_evidence_schema_valid = all(
            isinstance(entry, dict)
            and entry.get("schema") == PUNCTUATION_EVIDENCE_SCHEMA
            and isinstance(entry.get("candidate_id"), str)
            and entry.get("source") == "ctc"
            and isinstance(entry.get("left_lexical_ordinal"), (int, type(None)))
            and isinstance(entry.get("right_lexical_ordinal"), (int, type(None)))
            and isinstance(entry.get("raw_start_s"), (int, float))
            and isinstance(entry.get("raw_end_s"), (int, float))
            for entry in punct_entries
        )
        if not punctuation_evidence_schema_valid:
            report["punctuation_evidence_schema"] = {
                "schema": PUNCTUATION_EVIDENCE_SCHEMA,
                "status": "rejected",
                "reason": "punctuation_evidence_schema_mismatch",
            }
    if tokens_path.exists():
        ctc_tokens = []
        for line in tokens_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                ctc_tokens.append(json.loads(line))
        words_tier = tier_by_name(new_tg, "words")
        pp_tier = tier_by_name(new_tg, "pinyin_phones")
        if words_tier and ctc_tokens:
            words_tier, pp_tier = _snap_to_ctc(words_tier, pp_tier, ctc_tokens,
                                                   punct_entries=punct_entries,
                                                   audio=wav_audio, sr=wav_sr or 16000,
                                                   _punct_boundary_hits=_punct_boundary_hits)
            if _punct_boundary_hits:
                report.setdefault("punct_boundary_guard", [])
                report["punct_boundary_guard"] = _punct_boundary_hits
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                elif t.name == "pinyin_phones" and pp_tier is not None:
                    new_tg.tiers[i] = pp_tier
            new_tg._ctc_word_authority = getattr(
                words_tier, "_ctc_word_authority", None)
            _ensure_processed_geometry_state(words_tier)
            new_tg._processed_geometry_ledger = list(
                _processed_geometry_ledger(words_tier))

    # --- B. Energy-based boundary refinement ---
    if wav_audio is not None:
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            words_tier = _refine_boundaries_by_energy(words_tier, wav_audio, wav_sr,
                                                         punct_entries=punct_entries,
                                                         _punct_boundary_hits=_punct_boundary_hits)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                    break
            # Re-sync pinyin_phones after energy refinement.
            # _refine_boundaries_by_energy only adjusts words boundaries;
            # pinyin_phones still reflects the pre-refinement positions.
            # Rebuild from the current phones tier + updated words so all
            # three boundary tiers stay in lockstep.
            cur_phones_tier = tier_by_name(new_tg, "phones")
            if cur_phones_tier is not None:
                synced_pp = build_pinyin_phones_tier(cur_phones_tier, ipa_to_pinyin,
                                                      words_tier, pinyin_dict)
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "pinyin_phones":
                        new_tg.tiers[i] = synced_pp
                        break

    # --- C. Inject punctuation from CTC anchors ---
    words_tier = tier_by_name(new_tg, "words")
    pp_tier = tier_by_name(new_tg, "pinyin_phones")
    if fallback_surface_ledger is not None and words_tier is not None:
        _fallback_punctuation_projection_candidate = (
            _fallback_punctuation_projection(
                fallback_surface_ledger["source_text"], words_tier,
                ctc_token_list))
        report["fallback_punctuation_projection"] = (
            _fallback_punctuation_projection_candidate)
        # Rejected projection evidence remains diagnostic only; no resolver or
        # injection path may treat its mutable flags as authority.
        if _fallback_punctuation_projection_candidate.get("safe"):
            fallback_punctuation_projection = (
                _fallback_punctuation_projection_candidate)
            new_tg._fallback_punctuation_projection = (
                fallback_punctuation_projection)
    if (punct_entries or fallback_surface_ledger is not None) and words_tier:
            words_tier, pp_tier = _inject_punctuation(
                words_tier, pp_tier, punct_entries,
                reference_text=reference_text_original,
                reference_authoritative=reference_text_authoritative,
                source_surface_ledger=fallback_surface_ledger,
                ctc_tokens=ctc_token_list,
                punctuation_projection=fallback_punctuation_projection)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                elif t.name == "pinyin_phones" and pp_tier is not None:
                    new_tg.tiers[i] = pp_tier

    # ── Build en_mfa_windows early (needed by _sync_derived_tiers throughout Phases 3.5–5) ──
    # Regr. Case 37: key is (word_text, start_time_rounded) so duplicate English
    # words in the same utterance do not overwrite each other.
    en_mfa_windows: dict[tuple[str, float], tuple[float, float]] = {}
    if en_data:
        for entry in en_data:
            es = entry.get("en_word_start", entry["word_start"])
            ee = entry.get("en_word_end", entry["word_end"])
            key = (entry["word_text"].strip().lower(), round(es, 2))
            en_mfa_windows[key] = (es, ee)

    # ═══════════════════════════════════════════════════════════════
    # Phase 3.5 — English MFA phoneme injection.
    #
    # Runs AFTER boundary adjustments (snap->CTC, energy refine, punct
    # inject) so English words have their final CTC-snapped boundaries.
    # English MFA phonemes are proportionally scaled to fit within
    # those final word boundaries.
    #
    # This is a NO-OP when en_data is None (no English words in this
    # utterance, or English MFA step was skipped).
    # ═══════════════════════════════════════════════════════════════

    if en_data:
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            # Inject English MFA phones into phones tier.
            # Phase 5 build_pinyin_phones_tier detects these, converts
            # to ARPABET with en: prefix (no-op for ARPA model), and applies stress via
            # en_mfa_windows filtering to avoid boundary overlaps.
            phones_tier = tier_by_name(new_tg, "phones")
            if phones_tier is not None:
                _, phones_tier = _apply_en_phones(words_tier, phones_tier, en_data)
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "phones":
                        new_tg.tiers[i] = phones_tier
                        break
                # Re-sync pinyin_phones after English phone injection.
                # _apply_en_phones rewrites phone intervals for English words;
                # pinyin_phones must reflect the updated phones.
                # Regr. Case 37: pass en_mfa_windows so English phones are
                # correctly identified even in Phase 3.5 (before Phase 4
                # boundary changes).
                synced_pp = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin,
                                                      words_tier, pinyin_dict,
                                                      en_mfa_windows=en_mfa_windows)
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "pinyin_phones":
                        new_tg.tiers[i] = synced_pp
                        break
    elif not strict_en_mode:
        # No en_data — check if there are English tokens that need it
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            en_tokens = [iv.text for iv in words_tier.intervals
                         if is_english_token(iv.text)]
            if en_tokens:
                report.setdefault("warnings", []).append(
                    f"English tokens {en_tokens} found but no en_phones data. "
                    f"Pass --en-phones-dir or place en_phones/ next to output/.")

    # ═══════════════════════════════════════════════════════════════
    # Phase 4 — Post-boundary processing (ORDER CRITICAL).
    #
    #   D. handle_unexpected_silences — MUST be after _inject_punctuation:
    #      long silences are now '…' ellipsis, not <spN> gaps.
    #      Running before C would flag gaps that no longer exist.
    #   D2. absorb_nvv_trailing — NVV absorbs trailing punct+silence
    #      chain, extending NVV xmax to next content word.
    #   D3. absorb_silence_into_punct — fallback: punct absorbs trailing
    #      <spN> not already absorbed by an NVV.
    #   E. NVV+ellipsis unconditional merge — MUST be after C:
    #      needs '…' from punct injection.
    #   F. _merge_nvv_ellipsis (energy-based)
    #   G. _extend_word_into_ellipsis (energy-based)
    #
    # E–G all operate on NVV/ellipsis pairs and are order-independent
    # among themselves, but all depend on C having run first.
    # ═══════════════════════════════════════════════════════════════

    # ── Phase 4 前快照: 记录当前 words tier 中已有的 CTC 标点 ──
    # 用于 Phase 4 结束后比对哪些标点被融合/吸收了 (Regression Case 22).
    _punct_before: list[dict] = []
    if punct_entries:
        _wt_before = tier_by_name(new_tg, "words")
        if _wt_before:
            for p in punct_entries:
                if p["word"] not in '，。！？…、；：':
                    continue
                for iv in _wt_before.intervals:
                    if (not is_silence(iv.text) and iv.text.strip() == p["word"]
                            and abs(iv.xmin - p["start_s"]) < 0.5):
                        _punct_before.append(dict(p))
                        break

    # --- D. Handle unexpected silences ---
    # Keep the historical pinyin cross-check as pre-final diagnostics only.
    # The authoritative filter reason is recomputed after all punctuation and
    # ownership geometry mutations below.
    sil_filter_reasons = []
    if args.handle_unexpected_sil:
        handle_unexpected_silences(new_tg, pinyin_text)

    # --- D2. NVV absorbs trailing punctuation + silence chain ---
    # MFA cannot model NVV acoustically; the audio between an NVV and
    # the next real word is part of the NVV (e.g. laughter tail).
    absorb_nvv_trailing(new_tg)

    # --- D3. Absorb residual trailing silence into punctuation ---
    # Fallback: any <spN> still orphaned after punctuation (not already
    # absorbed by an NVV) is absorbed here so mid_sp won't flag it.
    absorb_silence_into_punct(new_tg)

    # --- D4. Strip edge punctuation (leading/trailing) ---
    # Punctuation sitting before the first real word or after the last
    # real word is absorbed into adjacent intervals.  Fixes orphaned
    # ellipsis left behind when NVASR strips NVV tags.  See Regression Case 17.
    strip_edge_punctuation(new_tg)

    # --- D5. Fix mild overlapping boundaries in words tier ---
    # Boundary adjustments (snap, refine, inject, absorb) can leave
    # adjacent word intervals with small overlaps.  Resolve the ones
    # that are clearly mechanical errors (< 30 ms between content words,
    # punct leaking into a neighbouring word).  Regression Case 27.
    _wt = tier_by_name(new_tg, "words")
    if _wt is not None:
        _overlaps_fixed = _fix_overlapping_boundaries(_wt)
        if _overlaps_fixed:
            # Sync derived tiers so hanzi + pinyin_phones reflect the fixes
            _sync_derived_tiers(new_tg, ipa_to_pinyin, pinyin_dict,
                                raw_text=raw_text,
                                en_mfa_windows=en_mfa_windows,
                                report_warnings=report.get("warnings", []),
                                reference_authoritative=reference_text_authoritative,
                                gap_ownership_evidence=_build_gap_ownership_evidence(
                                    _wt, source_words_snapshot, ctc_token_list,
                                    reference_text_original,
                                    reference_text_authoritative))

    # ── Phase 4 后比对: 哪些标点在 Phase 4 中被吞了 ──
    _swallowed_puncts: list[dict] = []
    if _punct_before:
        _wt_after = tier_by_name(new_tg, "words")
        if _wt_after:
            for p in _punct_before:
                _still_exists = False
                for iv in _wt_after.intervals:
                    if (not is_silence(iv.text) and iv.text.strip() == p["word"]
                            and abs(iv.xmin - p["start_s"]) < 0.5):
                        _still_exists = True
                        break
                if not _still_exists:
                    _swallowed_puncts.append(p)
            if _swallowed_puncts:
                report.setdefault("swallowed_punct", [])
                report["swallowed_punct"] = [p["word"] for p in _swallowed_puncts]

    # ── SYNC: D2/D3/D4 modified words tier in-place → rebuild derived tiers ──
    _sync_derived_tiers(new_tg, ipa_to_pinyin, pinyin_dict,
                         raw_text, en_mfa_windows, report.get("warnings", []),
                         reference_text_authoritative,
                         _build_gap_ownership_evidence(
                             tier_by_name(new_tg, "words"), source_words_snapshot,
                             ctc_token_list, reference_text_original,
                             reference_text_authoritative))

    # --- E. NVV + ellipsis unconditional merge ---
    words_tier = tier_by_name(new_tg, "words")
    pp_tier = tier_by_name(new_tg, "pinyin_phones")
    if words_tier:
        intervals = list(words_tier.intervals)
        for i in range(len(intervals) - 1):
            if _ctc_authoritative_ordinal(words_tier, i) is not None:
                continue
            if is_nvv_token(intervals[i].text) and intervals[i + 1].text.strip() == '…':
                gap = intervals[i + 1].xmin - intervals[i].xmax
                if gap < 0.02:
                    intervals[i] = Interval(intervals[i].xmin, intervals[i + 1].xmax,
                                            intervals[i].text)
                    intervals[i + 1] = Interval(0, 0, '')
        intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
        words_tier = _copy_tier_metadata(
            words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))
        for i, t in enumerate(new_tg.tiers):
            if t.name == "words":
                new_tg.tiers[i] = words_tier
                break
        if pp_tier:
            pp_ivs = list(pp_tier.intervals)
            for i in range(len(pp_ivs) - 1):
                if is_nvv_token(pp_ivs[i].text) and pp_ivs[i + 1].text.strip() == '…':
                    pp_ivs[i] = Interval(pp_ivs[i].xmin, pp_ivs[i + 1].xmax, pp_ivs[i].text)
                    pp_ivs[i + 1] = Interval(0, 0, '')
            pp_ivs = [iv for iv in pp_ivs if iv.xmax > iv.xmin + 0.001]
            pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, pp_ivs)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "pinyin_phones":
                    new_tg.tiers[i] = pp_tier
                    break

    # --- F. Energy-based NVV+ellipsis merge ---
    if wav_audio is not None:
        try:
            words_tier = tier_by_name(new_tg, "words")
            pp_tier = tier_by_name(new_tg, "pinyin_phones")
            if words_tier:
                words_tier, pp_tier = _merge_nvv_ellipsis(
                    words_tier, pp_tier, wav_audio, wav_sr)
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "words":
                        new_tg.tiers[i] = words_tier
                    elif t.name == "pinyin_phones" and pp_tier is not None:
                        new_tg.tiers[i] = pp_tier
        except Exception:
            pass

    # --- G. Energy-based word extension into ellipsis ---
    if wav_audio is not None:
        try:
            words_tier = tier_by_name(new_tg, "words")
            pp_tier = tier_by_name(new_tg, "pinyin_phones")
            if words_tier:
                words_tier, pp_tier = _extend_word_into_ellipsis(
                    words_tier, pp_tier, wav_audio, wav_sr)
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "words":
                        new_tg.tiers[i] = words_tier
                    elif t.name == "pinyin_phones" and pp_tier is not None:
                        new_tg.tiers[i] = pp_tier
        except Exception:
            pass

    # ── Final Phase 4 sync: ensure all derived tiers are current ──
    _sync_derived_tiers(new_tg, ipa_to_pinyin, pinyin_dict,
                         raw_text, en_mfa_windows, report.get("warnings", []),
                         reference_text_authoritative,
                         _build_gap_ownership_evidence(
                             tier_by_name(new_tg, "words"), source_words_snapshot,
                             ctc_token_list, reference_text_original,
                             reference_text_authoritative))

    # ═══════════════════════════════════════════════════════════════
    # Phase 5 — Final text sync & QC.
    # These steps rebuild tiers from the final word boundaries and
    # run quality checks.  Order among these is non-critical.
    # ═══════════════════════════════════════════════════════════════

    # ── Restore NVV word boundaries to CTC anchors ──
    # MFA compresses self-referencing NVV tokens; snap them back
    # and push the following word forward to avoid overlap.
    if tokens_path and tokens_path.exists():
        ctc_data = []
        for line in tokens_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                ctc_data.append(json.loads(line))
        words_tier = tier_by_name(new_tg, "words")
        if words_tier and ctc_data:
            intervals = list(words_tier.intervals)
            fallback_projection = getattr(
                words_tier, "_fallback_unknown_projection", None)
            ordinal_ctc = {}
            ctc_word_data = [item for item in ctc_data
                             if isinstance(item, dict)
                             and item.get("type", "word") == "word"]
            if isinstance(fallback_projection, dict) \
                    and fallback_projection.get("safe"):
                ordinal_ctc = {
                    entry.get("ctc_lexical_ordinal"): ctc_word_data[
                        entry["ctc_lexical_ordinal"]]
                    for entry in fallback_projection.get("entries", [])
                    if entry.get("status") == "mapped"
                    and isinstance(entry.get("ctc_lexical_ordinal"), int)
                    and 0 <= entry["ctc_lexical_ordinal"] < len(ctc_word_data)}
            elif fallback_projection is None:
                # Authority-mode compatibility uses the ordinal established
                # by _snap_to_ctc itself.  It is still lexical evidence, not
                # a second time-overlap owner search.
                for authority in getattr(words_tier, "_ctc_word_authority", []):
                    ctc_ordinal = authority.get("ctc_lexical_ordinal")
                    lexical = authority.get("lexical_ordinal")
                    if (isinstance(lexical, int)
                            and isinstance(ctc_ordinal, int)
                            and 0 <= ctc_ordinal < len(ctc_word_data)):
                        ordinal_ctc[lexical] = ctc_word_data[ctc_ordinal]
            lexical_ordinal = 0
            for i, iv in enumerate(intervals):
                if (not iv.text.strip() or is_silence(iv.text)
                        or is_punct(iv.text)):
                    continue
                current_lexical_ordinal = lexical_ordinal
                lexical_ordinal += 1
                if not is_nvv_token(iv.text.strip()):
                    continue
                # Presence of fallback projection metadata is an authority
                # decision: safe projections use the exact CTC ordinal, while
                # rejected projections never fall back to a guessed owner.
                best_ctc = ordinal_ctc.get(current_lexical_ordinal)
                if best_ctc and best_ctc["end_s"] > iv.xmax + 0.01:
                    new_end = best_ctc["end_s"]
                    # Set the next non-silence word to start at NVV's CTC end
                    for j in range(i + 1, len(intervals)):
                        nj = intervals[j]
                        if is_silence(nj.text):
                            continue
                        if nj.xmin < new_end:
                            nj.xmin = new_end
                        break
                    iv.xmax = new_end
            words_tier = _copy_tier_metadata(
                words_tier, Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals))
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                    break

    # 检测被吞掉的标点: CTC punct 条目在 words tier 中时间匹配不到 -> 从文本删除
    if (punct_entries and not reference_text_authoritative
            and fallback_surface_ledger is None):
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            # 收集 words tier 中所有标点 interval (按时间索引)
            punct_ivs_in_tier = []
            for iv in words_tier.intervals:
                c = iv.text.strip()
                if c in '，。…！？、；：':
                    punct_ivs_in_tier.append((iv.xmin, iv.xmax, c))
            # 标记已匹配的标点 interval
            matched = [False] * len(punct_ivs_in_tier)
            # 追踪每种标点在 raw_text 中的当前出现序号 (1-indexed)
            char_seq: dict[str, int] = {}
            for p in punct_entries:
                p_char = p["word"]
                p_start = p["start_s"]
                p_end = p["end_s"]
                # 当前是 raw_text 中第几个 p_char
                seq = char_seq.get(p_char, 0) + 1
                char_seq[p_char] = seq
                # 时间窗匹配: 查找 words tier 中时间重叠的标点
                found = False
                for j, (ps_iv, pe_iv, c_iv) in enumerate(punct_ivs_in_tier):
                    if matched[j]:
                        continue
                    if c_iv == p_char and ps_iv < p_end and pe_iv > p_start:
                        matched[j] = True
                        found = True
                        break
                if found:
                    continue
                # 标点没对应 -> 检查是否有 … 在同一位置 (CTC 长停顿替换了原标点)
                replaced = False
                if p_char in '，。！？':
                    for wi, iv in enumerate(words_tier.intervals):
                        if iv.text.strip() == '…' and abs(iv.xmin - p_start) < 0.3:
                            iv.text = p_char  # 用原标点替换省略号
                            replaced = True
                            break
                if not replaced and not reference_text_authoritative:
                    # 删除 raw_text 中第 seq 个 p_char, 不是第一个
                    raw_text = _remove_nth_char(raw_text, p_char, seq)
                    pinyin_text = _remove_nth_char(pinyin_text, p_char, seq)
                    # 删掉后序号不递增, 因为后面的字符前移了一位
                    char_seq[p_char] = seq - 1

            # 第二轮: 更新 text tiers (只在 words tier 实际变更后)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "raw_text":
                    t.intervals[0].text = raw_text
                elif t.name == "pinyin":
                    t.intervals[0].text = pinyin_text

    # Phase 5 — Rebuild derived tiers from final words tier.
    # ORDER IS CRITICAL: normalise spellings first, then build hanzi
    # from the normalised words.  Otherwise hanzi and raw_text freeze
    # stale (pre-normalisation) labels while words/pinyin advance.
    final_words_tier = tier_by_name(new_tg, "words")
    if final_words_tier:
        # 1. Normalise English words against original reference text (.txt).
        #    NVASR tokenizer (Chinese-centric) breaks English words into letter
        #    fragments (e.g. "Claude"→"Cla"+"ude") which may survive
        #    normalize_english_tokens.py when _text_cn.txt (ASR) differs from
        #    the reference.  raw_text from the original .txt is ground truth.
        #    Regression Case 62.
        _normalize_word_spellings(
            final_words_tier,
            reference_text_original if reference_text_authoritative else raw_text,
            authority_strict=reference_text_authoritative)

        # Reference punctuation is committed once, after all mutable
        # boundary/energy stages.  Its derived tiers are rebuilt at the final
        # authority commit below.
        if reference_text_authoritative:
            raw_text = reference_text_original
            pinyin_text = pinyin_text_original
        # Phase 5 spelling/punctuation authority can itself change interval
        # geometry after the earlier D5 cleanup.  Re-apply the same narrow
        # mechanical repair here so sub-30-ms overlaps introduced late do not
        # survive solely because of operation ordering.  Larger overlaps keep
        # flowing to the hard QC filter.
        _late_overlaps_fixed = _fix_overlapping_boundaries(final_words_tier)
        if _late_overlaps_fixed:
            _sync_derived_tiers(
                new_tg, ipa_to_pinyin, pinyin_dict,
                raw_text=(reference_text_original if reference_text_authoritative else raw_text),
                en_mfa_windows=en_mfa_windows,
                report_warnings=report.get("warnings", []),
                reference_authoritative=reference_text_authoritative,
                gap_ownership_evidence=_build_gap_ownership_evidence(
                    final_words_tier, source_words_snapshot, ctc_token_list,
                    reference_text_original, reference_text_authoritative))
            final_words_tier = tier_by_name(new_tg, "words")
        # 2. Rebuild hanzi from normalised words.
        hanzi_tier = _build_hanzi_tier(
            final_words_tier, raw_text, report.get("warnings", []),
            reference_authoritative=reference_text_authoritative)
        if hanzi_tier:
            found = False
            for i, t in enumerate(new_tg.tiers):
                if t.name == "hanzi":
                    new_tg.tiers[i] = hanzi_tier
                    found = True
                    break
            if not found:
                # Insert hanzi before words tier
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "words":
                        new_tg.tiers.insert(i, hanzi_tier)
                        break
        # Rebuild pinyin_phones from phones_tier with final word boundaries.
        # For English words, only phones within the English MFA alignment
        # window are used (filtered by build_pinyin_phones_tier via en_mfa_windows).
        final_phones_tier = tier_by_name(new_tg, "phones")
        if final_phones_tier and final_words_tier:
            synced_pp = build_pinyin_phones_tier(final_phones_tier, ipa_to_pinyin,
                                                  final_words_tier, pinyin_dict,
                                                  en_mfa_windows=en_mfa_windows)
            if synced_pp:
                w_idx = 0
                new_pp_ivs = list(synced_pp.intervals)
                for w_iv in final_words_tier.intervals:
                    if is_silence(w_iv.text) or not w_iv.text.strip():
                        continue
                    is_en = is_english_token(w_iv.text.strip())
                    while w_idx < len(new_pp_ivs) and new_pp_ivs[w_idx].xmax <= w_iv.xmin + 0.005:
                        w_idx += 1
                    word_pps = []
                    while w_idx < len(new_pp_ivs) and new_pp_ivs[w_idx].xmin < w_iv.xmax - 0.005:
                        word_pps.append(w_idx)
                        w_idx += 1
                    if word_pps:
                        first = word_pps[0]
                        last = word_pps[-1]
                        # Snap first phone to word start for ALL words.
                        # Regr. Case 40: English MFA phones may have
                        # residual offset after Phase 4 boundary changes.
                        if new_pp_ivs[first].xmin > w_iv.xmin + 0.005:
                            new_pp_ivs[first] = Interval(w_iv.xmin, new_pp_ivs[first].xmax, new_pp_ivs[first].text)
                            # ── Symmetric extension: when the first phone is
                            #     snapped backward, extend the previous word's
                            #     last phone forward to close the gap.  Mirrors
                            #     the last-phone extension below but in reverse.
                            _prev_word_idx = None
                            for __wi in range(len(final_words_tier.intervals) - 1, -1, -1):
                                _pw = final_words_tier.intervals[__wi]
                                if _pw.xmax <= w_iv.xmin - 0.005 and not is_silence(_pw.text) and _pw.text.strip():
                                    _prev_word_idx = __wi
                                    break
                            if _prev_word_idx is not None:
                                _pw_iv = final_words_tier.intervals[_prev_word_idx]
                                for __pi in range(len(new_pp_ivs) - 1, -1, -1):
                                    _pp = new_pp_ivs[__pi]
                                    if (_pp.xmin >= _pw_iv.xmin - 0.005
                                            and _pp.xmax <= _pw_iv.xmax + 0.005
                                            and not is_silence(_pp.text)):
                                        if _pp.xmax < w_iv.xmin - 0.002:
                                            _last_orig_dur = _pp.xmax - _pp.xmin
                                            _is_vowel = not bool(re.match(
                                                r'^[bpmfdtnlgkhjqxrzcs]$|^[zcs]h$', _pp.text))
                                            _max_dur = 0.400 if _is_vowel else 0.200
                                            _capped_dur = min(_max_dur, _last_orig_dur * 1.5)
                                            _extend_to = min(w_iv.xmin, _pp.xmin + max(_capped_dur, _last_orig_dur))
                                            if _extend_to > _pp.xmax:
                                                new_pp_ivs[__pi] = Interval(_pp.xmin, _extend_to, _pp.text)
                                        break
                        # Extend last phone to word end (Regr. Case 45).
                        # Apply a phonetically-motivated maximum duration
                        # so the tail phone is not inflated when the word
                        # boundary was stretched by CTC snap / silence
                        # absorption.  The cap is computed from the phone's
                        # own pre-extension duration:
                        #   - vowel / final:   max(400ms, 1.5× orig)
                        #   - consonant/init:   max(200ms, 1.5× orig)
                        #   - single-phone word: max(500ms, 1.5× orig)
                        # Excess time beyond the cap is NOT filled — it
                        # remains as a natural silence gap.
                        if w_iv.xmax > new_pp_ivs[last].xmax + 0.005:
                            extend_to = w_iv.xmax
                            # Find next word's first phone — may need shifting
                            next_first = None
                            for _wi in range(len(final_words_tier.intervals)):
                                _niv = final_words_tier.intervals[_wi]
                                if _niv.xmin > w_iv.xmax - 0.005 and not is_silence(_niv.text) and _niv.text.strip():
                                    for _npi in range(len(new_pp_ivs)):
                                        if new_pp_ivs[_npi].xmin >= _niv.xmin - 0.005 and not is_silence(new_pp_ivs[_npi].text):
                                            next_first = _npi
                                            break
                                    break
                            if next_first is not None:
                                extend_to = min(extend_to, new_pp_ivs[next_first].xmin)

                            # ── Duration cap (Regr. Case 45) ──
                            _last_text = new_pp_ivs[last].text
                            _last_orig_dur = new_pp_ivs[last].xmax - new_pp_ivs[last].xmin
                            _is_single = (first == last)
                            _is_vowel = not bool(re.match(
                                r'^[bpmfdtnlgkhjqxrzcs]$|^[zcs]h$', _last_text))
                            if _is_single:
                                _max_dur = 0.500
                            elif _is_vowel:
                                _max_dur = 0.400
                            else:
                                _max_dur = 0.200
                            # Allow phone to stretch up to 1.5× its original
                            # duration, capped by the absolute max.
                            _capped_dur = min(_max_dur, _last_orig_dur * 1.5)
                            if _capped_dur > _last_orig_dur:
                                extend_to = min(extend_to,
                                                new_pp_ivs[last].xmin + _capped_dur)
                            new_pp_ivs[last] = Interval(new_pp_ivs[last].xmin, extend_to, new_pp_ivs[last].text)
                synced_pp = Tier(synced_pp.name, synced_pp.xmin, synced_pp.xmax, new_pp_ivs)
                for i, t in enumerate(new_tg.tiers):
                    if t.name == "pinyin_phones":
                        new_tg.tiers[i] = synced_pp
                        break

            # Apply CMUdict stress to English ARPABET phones
            if en_data and not strict_en_mode:
                pp_tier_final = tier_by_name(new_tg, "pinyin_phones")
                if pp_tier_final and final_words_tier:
                    pp_intervals_final = list(pp_tier_final.intervals)
                    _apply_en_stress(final_words_tier, pp_intervals_final)
                    pp_tier_final = Tier(pp_tier_final.name, pp_tier_final.xmin,
                                         pp_tier_final.xmax, pp_intervals_final)
                    for i, t in enumerate(new_tg.tiers):
                        if t.name == "pinyin_phones":
                            new_tg.tiers[i] = pp_tier_final
                            break

            # ── Regr. Case 39: absorb residual pp tier micro-gaps ──
            # After all snaps and stretches, absorb any remaining gaps
            # ≤ 10 ms between consecutive non-punct phone intervals.
            # These are boundary residuals from the MFA↔CTC mismatch,
            # not real pauses.  Only silence intervals are absorbed;
            # content-to-content gaps are merged by extending the
            # preceding phone.
            _pp_t = tier_by_name(new_tg, "pinyin_phones")
            if _pp_t is not None:
                _pp_ivs = list(_pp_t.intervals)
                _pp_merged: list[Interval] = []
                for _piv in _pp_ivs:
                    if not _pp_merged:
                        _pp_merged.append(_piv)
                        continue
                    _prev = _pp_merged[-1]
                    _gap = _piv.xmin - _prev.xmax
                    if 0 < _gap <= 0.010:
                        # Tiny gap — extend previous phone to close it
                        _pp_merged[-1] = Interval(_prev.xmin, _piv.xmin, _prev.text)
                        if _piv.xmin > _prev.xmax:
                            _pp_merged.append(_piv)
                        # else: absorbed completely
                    elif _gap < 0 and _gap >= -0.003:
                        # Tiny overlap — clip previous phone
                        mid = (_prev.xmax + _piv.xmin) / 2.0
                        _pp_merged[-1] = Interval(_prev.xmin, mid, _prev.text)
                        _pp_merged.append(Interval(mid, _piv.xmax, _piv.text))
                    else:
                        _pp_merged.append(_piv)
                _pp_tier_new = Tier(_pp_t.name, _pp_t.xmin, _pp_t.xmax, _pp_merged)
                for _i, _t in enumerate(new_tg.tiers):
                    if _t.name == "pinyin_phones":
                        new_tg.tiers[_i] = _pp_tier_new
                        break

        # Rebuild pinyin tier from words (keeps punct in sync)
        pinyin_tier = tier_by_name(new_tg, "pinyin")
        if pinyin_tier:
            spaced = []
            prev_end = 0.0
            for iv in final_words_tier.intervals:
                gap = iv.xmin - prev_end
                if gap > 0.05:
                    spaced.append(" " * max(1, int(gap / 0.03)))
                if not is_silence(iv.text) and iv.text.strip():
                    spaced.append(iv.text)
                elif iv.text.strip():
                    spaced.append(iv.text)
                prev_end = iv.xmax
            pinyin_tier.intervals[0].text = " ".join(spaced) if spaced else pinyin_tier.intervals[0].text
        # Fallback source text is a display surface authority.  It must not be
        # reconstructed from hanzi/words: doing so loses source punctuation
        # and makes the later audit self-referential.
        raw_tier = tier_by_name(new_tg, "raw_text")
        hanzi_after = tier_by_name(new_tg, "hanzi")
        if raw_tier and hanzi_after:
            if reference_text_authoritative:
                # Keep supplied lexical content authoritative.  Rebuilding raw
                # from a collapsed hanzi tier created the former "empty==empty"
                # false pass; rendered punctuation edits remain in raw_text.
                raw_tier.intervals[0].text = raw_text
            else:
                raw_tier.intervals[0].text = (
                    "<sp1>" + _canonicalize_surface_nvv_markup(
                        str(raw_text or "")).replace("<sp1>", ""))

    # 最终恢复: CTC 长停顿注入 … 覆盖了原标点, 用 CTC punct 替换回去
    if (punct_entries and not reference_text_authoritative
            and fallback_surface_ledger is None):
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            for p in punct_entries:
                if p["word"] not in '，。！？…、；：':
                    continue
                # 检查 words tier 中是否有 …, 且位置接近 CTC punct
                for iv in words_tier.intervals:
                    if iv.text.strip() == '…' and abs(iv.xmin - p["start_s"]) < 0.3:
                        iv.text = p["word"]
                        break

    # ── 被吞标点恢复 ───────────────────────────────────────────────
    # 前提: ① CTC 标点在 _inject_punctuation 后存在于 words tier,
    #        ② Phase 4 (D/D2/D3/D4) 中该标点被融合/吸收 → 消失,
    #        ③ 该位置现在是 <spN> (MFA 对齐偏差经 snap 修正, 标点被吞后
    #           间隙重新暴露出来成为裸 <spN>)。
    # 不是泛泛地"标点缺失就补"——必须先确认标点确实经历过"存在→被吞"
    # 的过程, 且被吞后间隙以 <spN> 形态重新出现, 才替换恢复。
    # Regression Case 22.
    #
    # 匹配策略 (Case 24 修复): 按CTC序列顺序匹配, 而非时间重叠.
    # 标点在CTC序列中的前后邻词决定了其顺序位置;
    # 在words tier中找到同一个前词→<spN>→后词的三元组, 即为恢复目标.
    if (_swallowed_puncts and not reference_text_authoritative
            and fallback_surface_ledger is None):
        _words_t = tier_by_name(new_tg, "words")
        if _words_t:
            # Build CTC timeline: all items (tokens + puncts) sorted by start time
            _ctc_timeline = []
            # Re-read tokens (they may not be in scope at this point — loaded inside
            # a conditional block earlier in process_one)
            _tokens_path = txt_dir / f"{stem}_tokens.jsonl"
            if _tokens_path.exists():
                for line in _tokens_path.read_text(encoding="utf-8").strip().split("\n"):
                    if line:
                        t = json.loads(line)
                        _ctc_timeline.append(('token', t['word'], t['start_s']))
            if punct_entries:
                for p in punct_entries:
                    _ctc_timeline.append(('punct', p['word'], p['start_s']))
            _ctc_timeline.sort(key=lambda x: x[2])

            _restored = 0
            for p in _swallowed_puncts:
                p_s = p['start_s']
                # Find swallowed punct's sequential neighbors in CTC timeline
                prev_word = next_word = None
                for idx, (kind, word, ts) in enumerate(_ctc_timeline):
                    if (kind == 'punct' and word == p['word']
                            and abs(ts - p_s) < 0.01):
                        # Find previous content word
                        for j in range(idx - 1, -1, -1):
                            if _ctc_timeline[j][0] == 'token':
                                prev_word = _ctc_timeline[j][1]
                                break
                        # Find next content word
                        for j in range(idx + 1, len(_ctc_timeline)):
                            if _ctc_timeline[j][0] == 'token':
                                next_word = _ctc_timeline[j][1]
                                break
                        break

                if prev_word is None or next_word is None:
                    continue

                # Walk words tier sequentially:
                # find <spN> whose neighbors match prev_word / next_word
                _word_ivs = list(_words_t.intervals)
                for i in range(1, len(_word_ivs) - 1):
                    iv = _word_ivs[i]
                    if not is_silence(iv.text) or not iv.text.strip():
                        continue
                    left_txt = _word_ivs[i - 1].text.strip()
                    if is_silence(left_txt) or not left_txt:
                        continue
                    if left_txt != prev_word:
                        continue
                    right_txt = _word_ivs[i + 1].text.strip()
                    if is_silence(right_txt) or not right_txt:
                        continue
                    if right_txt != next_word:
                        continue
                    # Sequential match confirmed: <spN> sits between the
                    # same two content words as the swallowed punct in CTC
                    iv.text = p['word']
                    _restored += 1
                    break
            if _restored:
                # 同步 hanzi: 从更新后的 words 重建
                _hanzi_t = tier_by_name(new_tg, "hanzi")
                if _hanzi_t:
                    _new_hanzi = _build_hanzi_tier(
                        _words_t,
                        raw_text if raw_text else "",
                        report.get("warnings", []),
                        reference_authoritative=reference_text_authoritative)
                    if _new_hanzi:
                        for _i, _t in enumerate(new_tg.tiers):
                            if _t.name == "hanzi":
                                new_tg.tiers[_i] = _new_hanzi
                                break
                # 同步 pinyin_phones
                _phones_t = tier_by_name(new_tg, "phones")
                if _phones_t:
                    _new_pp = build_pinyin_phones_tier(
                        _phones_t, ipa_to_pinyin, _words_t, pinyin_dict,
                        en_mfa_windows=en_mfa_windows)
                    if _new_pp:
                        for _i, _t in enumerate(new_tg.tiers):
                            if _t.name == "pinyin_phones":
                                new_tg.tiers[_i] = _new_pp
                                break
                # Keep fallback raw_text bound to the sealed source surface;
                # swallowed punctuation must not trigger a derived rewrite.
                _raw_t = tier_by_name(new_tg, "raw_text")
                if _raw_t and not reference_text_authoritative:
                    _raw_t.intervals[0].text = (
                        "<sp1>" + _canonicalize_surface_nvv_markup(
                            str(raw_text or "")).replace("<sp1>", ""))
                report.setdefault("restored_punct", 0)
                report["restored_punct"] = _restored

    # ── 末尾标点恢复（仅替换显式尾部静音） ─────────────────────────
    # CTC 最后一个标点如果被前词吸收，只能从同一 CTC anchor 覆盖的显式
    # 尾部静音中恢复。没有显式静音时，不能从 voiced lexical word 裁剪
    # 音频/音素来强行保留标点；标点缺失由 publication projection 记为
    # missing_allowed，不是过滤理由。
    # Regression Case 25 follow-up: terminal punct recovery.
    if (punct_entries and not reference_text_authoritative
            and fallback_surface_ledger is None):
        _words_t = tier_by_name(new_tg, "words")
        if _words_t:
            # Find the last (rightmost) CTC punct
            _last_punct = max(punct_entries, key=lambda p: p["end_s"])
            _last_punct_word = _last_punct["word"]
            _last_punct_end = _last_punct["end_s"]
            try:
                _last_punct_start = float(_last_punct["start_s"])
            except (KeyError, TypeError, ValueError):
                _last_punct_start = -math.inf
            _has_explicit_tail_silence = any(
                is_silence(iv.text) and iv.text.strip()
                and iv.xmax > _last_punct_start + AXIS_EPS
                and iv.xmin < _last_punct_end - AXIS_EPS
                for iv in _words_t.intervals)

            # Check if this punct already exists as the last item in words tier
            _word_ivs = list(_words_t.intervals)
            _last_word_iv = None
            for iv in reversed(_word_ivs):
                if iv.text.strip():
                    _last_word_iv = iv
                    break

            _punct_at_end = (_last_word_iv is not None
                             and _last_word_iv.text.strip() == _last_punct_word)

            if not _punct_at_end and _last_word_iv is not None \
                    and _has_explicit_tail_silence:
                _last_idx = len(_word_ivs) - 1
                for _i in range(len(_word_ivs) - 1, -1, -1):
                    if _word_ivs[_i] is _last_word_iv:
                        _last_idx = _i
                        break

                # Carve at least 60ms: use CTC punct's original duration if longer
                _carve_s = max(0.060, (_last_punct_end - _last_punct["start_s"]))

                if _carve_s < _last_word_iv.xmax - _last_word_iv.xmin:
                    _punct_start = _last_word_iv.xmax - _carve_s
                    from dataclasses import replace as _replace
                    # Build new interval list: trim last word, append punct
                    _new_ivs = [_replace(iv) for iv in _word_ivs]
                    _new_ivs[_last_idx] = _replace(_last_word_iv,
                                                   xmax=_punct_start)
                    _new_ivs.append(_replace(_last_word_iv,
                                             xmin=_punct_start,
                                             xmax=_punct_start + _carve_s,
                                             text=_last_punct_word))
                    _words_t = _copy_tier_metadata(
                        _words_t, Tier(_words_t.name, _words_t.xmin,
                                       _words_t.xmax, _new_ivs))
                    for _i, _t in enumerate(new_tg.tiers):
                        if _t.name == "words":
                            new_tg.tiers[_i] = _words_t
                            break
                    report.setdefault("final_punct_restored", {})
                    report["final_punct_restored"] = {
                        "punct": _last_punct_word,
                        "carved_from": _last_word_iv.text.strip(),
                        "carved_s": round(_carve_s, 3)}

                    # Sync hanzi & pinyin_phones: trim last interval, append punct
                    for _tier_name in ("hanzi", "pinyin_phones"):
                        _t = tier_by_name(new_tg, _tier_name)
                        if _t is None:
                            continue
                        _t_ivs = list(_t.intervals)
                        # Trim the last non-empty interval to _punct_start
                        for _j in range(len(_t_ivs) - 1, -1, -1):
                            if _t_ivs[_j].text.strip():
                                _t_ivs[_j] = _replace(_t_ivs[_j], xmax=_punct_start)
                                break
                        # Append the restored punct
                        _t_ivs.append(_replace(_t_ivs[-1],
                                               xmin=_punct_start,
                                               xmax=_punct_start + _carve_s,
                                               text=_last_punct_word))
                        _t_new = Tier(_t.name, _t.xmin, _t.xmax, _t_ivs)
                        for _i, _tt in enumerate(new_tg.tiers):
                            if _tt.name == _tier_name:
                                new_tg.tiers[_i] = _t_new
                                break

    # Evidence-constrained recovery is the last non-authority boundary repair.
    # It must run before the single reference owner commit so that the commit
    # below is stable and every later operation is derived-only.
    evidence_repairs = _apply_evidence_constrained_repairs(
        stem, source_words_snapshot, ctc_token_list, new_tg)
    if evidence_repairs:
        report["evidence_repairs"] = evidence_repairs
    if reference_text_authoritative:
        # This pass is still a words-tier mutation, so it belongs before the
        # single authority commit.  It absorbs only punctuation-owned silence.
        absorb_silence_into_punct(new_tg)

    # Final authoritative reconciliation: commit the reference punctuation
    # owner once, immediately before derived-tier publication.
    if reference_text_authoritative:
        reference_output_text = _canonicalize_reference_hyphens(
            reference_text_original)
        _final_words = tier_by_name(new_tg, "words")
        if _final_words is not None:
            _restore_reference_punctuation(
                _final_words, reference_output_text, punct_entries)
            # One exact compound transaction owns all reference English
            # fragments.  It runs after punctuation restore and before the
            # final derived-tier rebuild/freeze; failed validation leaves the
            # split geometry untouched for the strict auditors to reject.
            _merge_authority_alpha_digit_fragments(
                _final_words, reference_output_text, ctc_token_list,
                report=report)
            # Derived-only rebuild: unlike _sync_derived_tiers this does not
            # reopen gap ownership or canonicalize the frozen owner list.
            _final_hanzi = _build_hanzi_tier(
                _final_words, reference_output_text,
                report.get("warnings", []), reference_authoritative=True)
            if _final_hanzi is not None:
                for _index, _tier in enumerate(new_tg.tiers):
                    if _tier.name == "hanzi":
                        new_tg.tiers[_index] = _final_hanzi
                        break
            _final_phones = tier_by_name(new_tg, "phones")
            if _final_phones is not None and pinyin_dict is not None:
                _final_pp = build_pinyin_phones_tier(
                    _final_phones, ipa_to_pinyin, _final_words,
                    pinyin_dict, en_mfa_windows=en_mfa_windows)
                if _final_pp is not None:
                    for _index, _tier in enumerate(new_tg.tiers):
                        if _tier.name == "pinyin_phones":
                            new_tg.tiers[_index] = _final_pp
                            break
        _raw_authoritative = tier_by_name(new_tg, "raw_text")
        if _raw_authoritative and _raw_authoritative.intervals:
            _raw_authoritative.intervals[0].text = (
                "<sp1>" + _canonicalize_surface_nvv_markup(
                    reference_output_text).replace("<sp1>", ""))
        _pinyin_authoritative = tier_by_name(new_tg, "pinyin")
        if _pinyin_authoritative and _pinyin_authoritative.intervals:
            _pinyin_authoritative.intervals[0].text = _reference_pinyin_text(
                reference_output_text, pinyin_text_original)

    # Fallback punctuation is a words-tier owner mutation.  Commit it before
    # visual silence arbitration so punctuation has its declared precedence
    # and no later pass can invalidate the final owner ordering.
    if not reference_text_authoritative and fallback_surface_ledger is not None:
        _surface_valid, _surface_details = (
            _validate_fallback_punctuation_surface_ledger(
                fallback_surface_ledger))
        report["fallback_surface_final_commit"] = _surface_details
        if _surface_valid:
            raw_text = str(fallback_surface_ledger["source_text"])
            _final_words = tier_by_name(new_tg, "words")
            if _final_words is not None:
                _final_words, _ = _inject_fallback_punctuation_gaps(
                    _final_words, None, punct_entries,
                    source_surface_ledger=fallback_surface_ledger,
                    ctc_tokens=ctc_token_list,
                    punctuation_projection=fallback_punctuation_projection)
                for _index, _tier in enumerate(new_tg.tiers):
                    if _tier.name == "words":
                        new_tg.tiers[_index] = _final_words
                        break

    # Final visual silence owner commit.  The resolver snapshots the current
    # words list after punctuation restore/absorption, computes every decision
    # against that immutable list, and rebuilds words once.  Phones and all
    # derived tiers are synchronized only by the freeze/lineage barrier below.
    _resolve_visual_short_silence_merges(
        new_tg, wav_audio, wav_sr or 16000, args, report=report,
        ctc_tokens=ctc_token_list, reference_punct_entries=punct_entries,
        fallback_punctuation_projection=fallback_punctuation_projection)
    # Owner arbitration is complete.  Only now normalize labels on retained
    # internal pure-silence intervals, immediately before geometry freeze;
    # this cannot reopen or alter the already-committed owner decision.
    _normalize_final_internal_silence_labels(new_tg, report=report)

    # ── Final de-overlap: pinyin_phones tier (must be after ALL
    #     Phase 5 phone modifications including _apply_en_stress) ──
    _pp = tier_by_name(new_tg, "pinyin_phones")
    if _pp is not None:
        _pp_fixed = _fix_pp_phone_overlaps(_pp)
        if _pp_fixed:
            report["pp_deoverlap_fixed"] = _pp_fixed

    # Strict MFA provenance is intentionally injected last.  Phase 4/5 may
    # normalise, stretch, merge, de-overlap, or apply CMU stress to legacy
    # English phones.  None of those transformations are admissible for a
    # strict result: the final pinyin_phones tier must be the exact ledger
    # sequence and affine timings, with only the ``en:`` namespace added.
    if reference_text_authoritative:
        restored_units = _restore_reference_surfaces(
            tier_by_name(new_tg, "words"), tier_by_name(new_tg, "hanzi"),
            reference_text_original)
        if restored_units:
            report["english_surface_units_restored"] = restored_units

    # Final geometry barrier.  All earlier stages may inspect or reconcile
    # CTC/MFA evidence, but the *current processed words* are now the only
    # publication authority.  In particular, never reassert ``resolved_span``
    # or a raw CTC span here: doing so resurrects the pre-processed boundary
    # after ``handle_unexpected_silences`` merged an internal <sp0>.
    # NVV frame support is the narrow exception: it is immutable physical
    # evidence and may locally repartition conflicting display ownership,
    # while remaining distinct from the wider non-acoustic display span.
    _wav_axis_duration = (len(wav_audio) / float(wav_sr)
                          if wav_audio is not None and wav_sr else None)
    _frame_support_result = _contain_nvasr_frame_support(
        tier_by_name(new_tg, "words"), ctc_token_list,
        wav_duration_s=_wav_axis_duration)
    report["nvasr_frame_support"] = _frame_support_result
    _frame_support_rejected = _frame_support_result.get("status") == "rejected"
    _contained_words = _frame_support_result.pop("_contained_tier", None)
    if _contained_words is not None:
        for _index, _tier in enumerate(new_tg.tiers):
            if _tier.name == "words":
                new_tg.tiers[_index] = _contained_words
                break
    _final_words_before_ctc = tier_by_name(new_tg, "words")
    _final_words_after_ctc = _final_words_before_ctc
    _ctc_reasserted = 0
    if _ctc_reasserted and _final_words_after_ctc is not None:
        # Reference-confirmed punctuation is the one intentional local
        # exception: it may carve the immediate CTC lexical owners, but no
        # generic MFA/energy pass is allowed to do so.
        if reference_text_authoritative:
            # The authority commit above is the single owner mutation.  This
            # barrier is intentionally derived-only; do not re-run restore.
            pass
        report["boundary_decision_reassertions"] = _ctc_reasserted
        for _index, _tier in enumerate(new_tg.tiers):
            if _tier.name == "words":
                new_tg.tiers[_index] = _final_words_after_ctc
                break
        _final_words_before_ctc = _final_words_after_ctc
        _lineage = getattr(new_tg, "_phone_lineage", None)
        _final_phones = tier_by_name(new_tg, "phones")
        if _final_phones is not None and isinstance(_lineage, dict):
            _rebuilt_final_phones = _rebuild_phones_from_lineage(
                _final_words_after_ctc, _final_phones, _lineage)
            if _rebuilt_final_phones is not None:
                for _index, _tier in enumerate(new_tg.tiers):
                    if _tier.name == "phones":
                        new_tg.tiers[_index] = _rebuilt_final_phones
                        _final_phones = _rebuilt_final_phones
                        break
                if _lineage.get("status") in {"verified", "partial"}:
                    if hasattr(new_tg, "_phone_lineage_invalid"):
                        delattr(new_tg, "_phone_lineage_invalid")
                    if hasattr(_final_phones, "_phone_lineage_invalid"):
                        delattr(_final_phones, "_phone_lineage_invalid")
        if _final_phones is not None:
            _final_hanzi = _build_hanzi_tier(
                _final_words_after_ctc,
                reference_text_original if reference_text_authoritative else raw_text,
                report.get("warnings", []),
                reference_authoritative=reference_text_authoritative)
            if _final_hanzi is not None:
                for _index, _tier in enumerate(new_tg.tiers):
                    if _tier.name == "hanzi":
                        new_tg.tiers[_index] = _final_hanzi
                        break
            if pinyin_dict is not None:
                _final_pp = build_pinyin_phones_tier(
                    _final_phones, ipa_to_pinyin, _final_words_after_ctc,
                    pinyin_dict, en_mfa_windows=en_mfa_windows)
                if _final_pp is not None:
                    for _index, _tier in enumerate(new_tg.tiers):
                        if _tier.name == "pinyin_phones":
                            new_tg.tiers[_index] = _final_pp
                            break

    # Fallback lexical correspondence is the only source of a no-reference
    # lexical owner mapping.  Build it after the final words mutation and
    # before freezing so phone/English lineage can consume the same immutable
    # source→CTC→final ordinals.  Authority runs deliberately retain their
    # existing proof path and do not consult fallback evidence.
    fallback_correspondence = None
    if not reference_text_authoritative:
        fallback_correspondence = _fallback_lexical_correspondence_ledger(
            source_words_snapshot, ctc_token_list, tier_by_name(new_tg, "words"))
        report["fallback_correspondence"] = fallback_correspondence
        if fallback_correspondence.get("safe"):
            _apply_fallback_correspondence_to_lineage(
                source_phone_lineage, fallback_correspondence)
            new_tg._phone_lineage = source_phone_lineage

    # Freeze only after the final authority/silence mutation above.  All
    # derived tiers below are rebuilt from this frozen interval list; no later
    # sync is allowed to reopen CTC/MFA arbitration.
    _frozen_words, _freeze_reasons = _freeze_processed_geometry(new_tg)
    if _freeze_reasons:
        report["processed_geometry_contract"] = {
            "status": "rejected", "reasons": _freeze_reasons}
    if _frozen_words is not None:
            _rebuild_derived_from_frozen_words(
            new_tg, ipa_to_pinyin, pinyin_dict,
            reference_text_original if reference_text_authoritative else raw_text,
            en_mfa_windows=en_mfa_windows,
            warnings=report.get("warnings", []),
            reference_authoritative=reference_text_authoritative,
            pinyin_text=pinyin_text_original,
            fallback_surface_ledger=fallback_surface_ledger)
    strict_en_rejected = False
    unknown_source_redeemed = False
    unknown_recovery_proof = None
    if strict_en_mode:
        final_words_tier = tier_by_name(new_tg, "words")
        processed_ctc_textgrid_path = txt_dir / f"{stem}.TextGrid"
        if not processed_ctc_textgrid_path.is_file():
            processed_ctc_textgrid_path = txt_dir / stem / f"{stem}.TextGrid"
        processed_ctc_words_tier = None
        if processed_ctc_textgrid_path.is_file():
            try:
                processed_ctc_words_tier = tier_by_name(
                    parse_textgrid(processed_ctc_textgrid_path), "words")
            except (OSError, UnicodeError, ValueError):
                processed_ctc_words_tier = None
        strict_en, strict_pairs = load_strict_en_provenance(
            stem, final_words_tier, en_phones_dir,
            hanzi_tier=tier_by_name(new_tg, "hanzi"),
            reference_text=(reference_text_original if reference_text_authoritative else None),
            ctc_tokens=ctc_token_list,
            correspondence=(fallback_correspondence
                            if not reference_text_authoritative else None),
            processed_ctc_words_tier=processed_ctc_words_tier,
            processed_ctc_textgrid_path=(processed_ctc_textgrid_path
                                         if processed_ctc_textgrid_path.is_file()
                                         else None))
        report["english_provenance"] = strict_en
        unknown_recovery_proof = _build_mfa_unknown_recovery_proof(
            stem, source_words_snapshot, ctc_token_list, reference_text_original,
            final_words_tier, tier_by_name(new_tg, "hanzi"), strict_en, strict_pairs)
        unknown_source_redeemed = bool(
            reference_text_authoritative and unknown_recovery_proof is not None)
        if unknown_source_redeemed:
            report["mfa_unknown_source_redeemed"] = unknown_recovery_proof
        if strict_en["status"] == "verified":
            try:
                strict_pp = inject_strict_en_phones(
                    tier_by_name(new_tg, "pinyin_phones"), final_words_tier, strict_pairs)
                if strict_pp is None:
                    raise ValueError("strict_en_pinyin_phones_missing")
                for _index, _tier in enumerate(new_tg.tiers):
                    if _tier.name == "pinyin_phones":
                        new_tg.tiers[_index] = strict_pp
                        break
            except Exception as exc:
                unknown_source_redeemed = False
                unknown_recovery_proof = None
                report.pop("mfa_unknown_source_redeemed", None)
                strict_en_rejected = True
                report["english_provenance"] = _strict_en_report(
                    "rejected", strict_en["required_words"], 0,
                    strict_en.get("failed_word_ids", []), strict_en.get("ledger_sha256", ""),
                    f"strict_en_injection_failed:{exc}")
                stripped = _strip_english_phone_intervals(
                    tier_by_name(new_tg, "pinyin_phones"), final_words_tier)
                if stripped is not None:
                    for _index, _tier in enumerate(new_tg.tiers):
                        if _tier.name == "pinyin_phones":
                            new_tg.tiers[_index] = stripped
                            break

        elif strict_en["status"] == "rejected":
            strict_en_rejected = True
            # A filtered TextGrid must not contain a fabricated English phone
            # sequence left by Chinese MFA or the legacy recovery path.
            stripped = _strip_english_phone_intervals(
                tier_by_name(new_tg, "pinyin_phones"), final_words_tier)
            if stripped is not None:
                for _index, _tier in enumerate(new_tg.tiers):
                    if _tier.name == "pinyin_phones":
                        new_tg.tiers[_index] = stripped
                        break

    # In no-reference mode, only a complete safe correspondence may redeem a
    # source <unk>.  The fallback transcript is intentionally absent from this
    # decision.  Each recovered item is already bound to the exact CTC/final
    # ordinal by the ledger, including multiple NVV items.
    fallback_unknown_recovered = []
    if (not reference_text_authoritative
            and isinstance(fallback_correspondence, dict)
            and fallback_correspondence.get("safe")):
        fallback_unknown_recovered = _fallback_redeemed_unknown_entries(
            fallback_correspondence,
            report.get("english_provenance"))
        if fallback_unknown_recovered:
            report["mfa_unknown_source_redeemed"] = {
                "schema": "fallback-unknown-recovery-v2",
                "entries": fallback_unknown_recovered,
                "ledger_digest": fallback_correspondence.get("digest"),
            }
            unknown_source_redeemed = len(fallback_unknown_recovered) == len(
                [item for item in source_words_snapshot
                 if is_unknown_token(item.get("text", ""))])

    # Strict English injection is intentionally last for English intervals;
    # clean any remaining Chinese/punctuation overlap without touching the
    # immutable ``en:`` phone geometry.
    _post_strict_pp = tier_by_name(new_tg, "pinyin_phones")
    if _post_strict_pp is not None:
        _post_fixed = _fix_non_english_pp_overlaps(_post_strict_pp)
        if _post_fixed:
            report["pp_deoverlap_fixed"] = int(
                report.get("pp_deoverlap_fixed", 0)) + _post_fixed

    # A final owner-aware pass repairs only clear MFA boundary leakage.  It
    # runs after strict English injection so the immutable English ledger is
    # never rewritten by a generic midpoint split.  Ambiguous phones remain
    # visible in the report and are vetoed by the publication contract.
    _publication_phones_tier = tier_by_name(new_tg, "phones")
    _final_words_for_phone_owner = tier_by_name(new_tg, "words")
    if _final_words_for_phone_owner is not None:
        _source_phone_fixed, _source_phone_ambiguous = (
            _resolve_phone_owner_overlaps(
                _publication_phones_tier, _final_words_for_phone_owner))
        _pp_phone_fixed, _pp_phone_ambiguous = (
            _resolve_phone_owner_overlaps(
                _post_strict_pp, _final_words_for_phone_owner))
        if _source_phone_fixed or _pp_phone_fixed:
            report["phone_owner_repairs"] = {
                "phones": _source_phone_fixed,
                "pinyin_phones": _pp_phone_fixed,
            }
        if _source_phone_ambiguous or _pp_phone_ambiguous:
            report["phone_owner_ambiguity"] = {
                "phones": _source_phone_ambiguous[:20],
                "pinyin_phones": _pp_phone_ambiguous[:20],
            }

    # The one final word-energy audit runs after visual ownership commit,
    # freeze/lineage rebuild, and strict English/phone ownership.  It is
    # diagnostic-first: a silence merge is evaluated on the immutable lexical
    # core and never on its display-expanded span.
    _word_energy_report = _word_energy_audit(
        tier_by_name(new_tg, "words"), args, wav_audio, wav_sr or 16000,
        textgrid=new_tg, ctc_tokens=ctc_token_list)
    report["word_energy_audit"] = _word_energy_report
    if (getattr(args, "filter_suspicious", True)
            and _word_energy_enabled(args)):
        for _energy_item in _word_energy_report.get("items", []):
            _energy_reason = _energy_item.get("resulting_reason")
            if _energy_reason in {
                    "word_in_silence",
                    "word_energy_boundary_mismatch",
                    "word_energy_evidence_unresolved"}:
                align_issues.append({
                    "rule": _energy_reason,
                    "text": _energy_item.get("word"),
                    "lexical_ordinal": _energy_item.get("lexical_ordinal"),
                    "energy": round(_energy_item["rms"]["premerge_rms"], 9),
                    "noise_floor": round(
                        _word_energy_report["noise_model"]["noise_floor"], 9),
                    "threshold": round(
                        _word_energy_report["noise_model"]["threshold"], 9),
                    "classification": _energy_item.get("classification"),
                })

    # ================================================================
    # 最终筛选: 所有处理完成后再统一判断 (用最终的边界和静音结构)
    # ================================================================
    filter_reasons = []
    if _frame_support_rejected:
        filter_reasons.append("nvasr_frame_support_rejected")
    if (getattr(args, "filter_suspicious", True)
            and _word_energy_enabled(args)):
        for _energy_reason in {
                item.get("resulting_reason")
                for item in _word_energy_report.get("items", [])}:
            if _energy_reason in {
                    "word_in_silence",
                    "word_energy_boundary_mismatch",
                    "word_energy_evidence_unresolved"}:
                filter_reasons.append(_energy_reason)
    axis_reasons = list(getattr(args, "_axis_stem_reasons", {}).get(stem, ()))
    if axis_reasons:
        filter_reasons.extend(axis_reasons)
        report["axis_reasons"] = axis_reasons
    if strict_en_rejected:
        filter_reasons.append("english_provenance_rejected")
    if punct_entries and not punctuation_evidence_schema_valid:
        filter_reasons.append("punctuation_evidence_schema_mismatch")
    if (not reference_text_authoritative
            and isinstance(fallback_correspondence, dict)
            and not fallback_correspondence.get("safe")):
        filter_reasons.append("fallback_correspondence_rejected")

    _terminal_punctuation_missing = _terminal_punctuation_evidence_missing(
        tier_by_name(new_tg, "words"),
        reference_authoritative=reference_text_authoritative)
    if _terminal_punctuation_missing is not None:
        report["terminal_punctuation_evidence_missing"] = (
            _terminal_punctuation_missing)
        filter_reasons.append("terminal_punctuation_evidence_missing")

    # Hard lexical integrity is independent of optional acoustic filtering.
    # NVV, punctuation and sentence-initial <sp1> are intentionally excluded
    # from the CJK/pinyin denominator.
    # Retain this pre-finalization snapshot only for the downstream CJK/pinyin
    # diagnostic below.  It is not publication evidence: `_finalize_textgrid`
    # still normalizes labels such as a leading `<sp2>` to `<sp1>`.
    _fallback_recovered_count = len(locals().get("fallback_unknown_recovered", []))
    unknown_source_count = (0 if unknown_source_redeemed else
                            max(0, len(mfa_unknown_before_snap)
                                - _fallback_recovered_count))
    _prewrite_coverage, _ = assess_reference_coverage(
        reference_text_original,
        tier_by_name(new_tg, "words"),
        tier_by_name(new_tg, "hanzi"),
        reference_source=reference_source,
        unknown_source_count=unknown_source_count,
        reference_authoritative=reference_text_authoritative,
    )
    if mfa_unknown_before_snap and not unknown_source_redeemed:
        report["mfa_unknown_source"] = {
            "count": unknown_source_count,
            "examples": mfa_unknown_before_snap[:20],
        }

    # ── Load CMUdict for English word QC (Regr. Case 48) ──
    # Case 32 (english_single_phone) and Case 33 (english_phone_deficit)
    # must use CMUdict — NOT pinyin_dict — to determine the expected
    # phone count for English words.  pinyin_dict is the CHINESE pinyin
    # decomposition dict and is semantically wrong for English tokens.
    from pipeline_utils import _load_cmudict as _load_cmu
    _cmu = _load_cmu()

    # Pinyin leakage: the Chinese text (raw_text tier) must not contain
    # pinyin syllables like "yan1" or "li3".  If found, the alignment has
    # failed to convert pinyin back to Chinese characters.
    import re as _re
    _raw_tier = tier_by_name(new_tg, "raw_text")
    _pinyin_hits: list[str] = []
    if _raw_tier is not None:
        for _iv in _raw_tier.intervals:
            # Exclude <spN> silence markers (sp1, sp2, sp3) — they are not
            # pinyin leakage but legitimate silence interval labels embedded
            # in the raw_text tier to mark sentence-initial pauses.
            _raw_text = _re.sub(r'<sp\d+>', '', _iv.text)
            _pinyin_hits.extend(
                token for token in _re.findall(r'\b(?!sp\d\b)[a-z]+[1-5]\b', _raw_text)
                if is_pinyin_syllable(token)
            )
    if _pinyin_hits:
        filter_reasons.append("pinyin_in_text")
        report["pinyin_in_text"] = sorted(set(_pinyin_hits))

    # ── Tier completeness: all 5 expected tiers must have content ──
    _expected_tiers = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")
    _missing_tiers: list[str] = []
    for _name in _expected_tiers:
        _t = tier_by_name(new_tg, _name)
        if _t is None or len(_t.intervals) == 0:
            _missing_tiers.append(_name)
    if _missing_tiers:
        filter_reasons.append("incomplete_tiers")
        report["incomplete_tiers"] = _missing_tiers

    # ── Inter-tier sync: hanzi ↔ words tier must agree on word identity ──
    # Each non-silence hanzi interval should map to the same word token in
    # the words tier at the same position (CJK→pinyin for Chinese, same text
    # for English).  A mismatch means the tiers have drifted apart.
    _hanzi_t = tier_by_name(new_tg, "hanzi")
    words_tier = tier_by_name(new_tg, "words")
    pp_tier = tier_by_name(new_tg, "pinyin_phones")
    _tier_mismatches, _tier_total = _tier_desync_counts(_hanzi_t, words_tier)
    if _tier_mismatches > 0:
        filter_reasons.append("tier_desync")
        report["tier_desync"] = f"hanzi↔words mismatches: {_tier_mismatches}/{_tier_total}"

    # ── Phone-word alignment: phones must live inside their word intervals ──
    _misaligned_phones = 0
    _total_phones = 0
    if words_tier is not None and pp_tier is not None:
        _word_ranges = [(iv.xmin, iv.xmax) for iv in words_tier.intervals
                        if not is_silence(iv.text) and iv.text.strip()]
        _tolerance = 0.05
        for _pi in pp_tier.intervals:
            if is_silence(_pi.text) or not _pi.text.strip():
                continue
            _total_phones += 1
            _inside = any(_ws - _tolerance <= _pi.xmin
                          and _pi.xmax <= _we + _tolerance
                          for _ws, _we in _word_ranges)
            if not _inside:
                _misaligned_phones += 1
        if _misaligned_phones > 0:
            filter_reasons.append("misaligned_phones")
            report["misaligned_phones"] = f"{_misaligned_phones}/{_total_phones}"

    # sp3 / mid_sp: 检查最终 words 层的静音结构。  Edge-ness is
    # semantic: punctuation and other non-lexical labels do not own a gap,
    # so a tail pause before terminal punctuation is still a normal edge.
    if words_tier:
        _sp3_details: list[dict] = []
        for i, iv in enumerate(words_tier.intervals):
            if iv.text.strip() == "<sp3>":
                _sp3_details.append({"index": i, "start_s": round(iv.xmin, 6),
                                     "end_s": round(iv.xmax, 6),
                                     "duration_us": _duration_ticks(iv.xmin, iv.xmax)})
                if _is_substantive_interior_silence(
                        words_tier.intervals, i):
                    filter_reasons.append("sp3")
        _mid_sp_details: list[dict] = []
        for i, iv in enumerate(words_tier.intervals):
                if (is_silence(iv.text) and iv.text.strip()
                        and _is_substantive_interior_silence(
                            words_tier.intervals, i)):
                    _mid_sp_details.append({"index": i, "label": iv.text.strip(),
                                        "start_s": round(iv.xmin, 6),
                                        "end_s": round(iv.xmax, 6),
                                        "duration_us": _duration_ticks(iv.xmin, iv.xmax),
                                        "prev": words_tier.intervals[i - 1].text.strip(),
                                        "next": words_tier.intervals[i + 1].text.strip()})
        if _sp3_details:
            report["sp3"] = {"count": len(_sp3_details), "details": _sp3_details[:10]}
        if _mid_sp_details:
            report["mid_sp"] = {"count": len(_mid_sp_details),
                                "details": _mid_sp_details[:10]}

    # BGM + word_in_silence: 用处理后的最终边界检测
    if wav_audio is not None and words_tier is not None:
        if args.detect_bgm:
            fs = max(1, int(args.bgm_frame_ms / 1000.0 * wav_sr))
            hs = max(1, int(args.bgm_hop_ms / 1000.0 * wav_sr))
            all_rms, _ = _frame_rms_vec(wav_audio, wav_sr, frame_ms=args.bgm_frame_ms)
            k = max(1, int(len(all_rms) * 0.6))
            nf_bgm = float(np.partition(all_rms, k)[k]) if len(all_rms) > 0 else 1e-6
            nf_bgm = max(nf_bgm, 1e-6)
            bgm_threshold = max(nf_bgm * args.bgm_noise_floor_ratio, 0.005)
            # Hard ceiling: when the 60th-percentile "noise floor" is
            # poisoned by loud content in silence regions, the threshold
            # must not exceed a value that clearly indicates non-silence.
            # Regression Case 25 follow-up.
            bgm_threshold = min(bgm_threshold, args.bgm_max_threshold)
            speech_energies = []
            suspect_intervals = []
            total_sil_dur = 0.0
            suspect_dur = 0.0
            _fallback_bgm = (
                not reference_text_authoritative
                and reference_mode_policy == "fallback")
            if _fallback_bgm:
                _bgm_selection = _fallback_bgm_ctc_gap_selection(
                    words_tier, source_words_snapshot, ctc_token_list,
                    fallback_correspondence,
                    (0.0, len(wav_audio) / float(wav_sr or 16000)),
                    reference_mode=reference_mode_policy)
                report["bgm_ctc_gap_selection"] = _bgm_selection
                _bgm_intervals = {
                    item["index"]: item
                    for item in _bgm_selection["evaluated_intervals"]}
            else:
                _bgm_selection = None
                _bgm_intervals = {}
            for _word_index, iv in enumerate(words_tier.intervals):
                if not is_silence(iv.text):
                    if iv.text.strip():
                        e = _word_rms(wav_audio, wav_sr, iv.xmin, iv.xmax)
                        if e > 0:
                            speech_energies.append(e)
                    continue
                _selection = _bgm_intervals.get(_word_index)
                if _selection is not None:
                    _evaluated = _selection.get("evaluated_intersection")
                    if not _evaluated:
                        continue
                    _scan_start, _scan_end = _evaluated
                    _scan_duration = max(0.0, _scan_end - _scan_start)
                else:
                    _scan_start, _scan_end = iv.xmin, iv.xmax
                    _scan_duration = max(0.0, _scan_end - _scan_start)
                # The minimum-duration gate applies after CTC-gap
                # intersection.  All other BGM thresholds remain unchanged.
                if _scan_duration < args.bgm_min_sil_dur:
                    continue
                total_sil_dur += _scan_duration
                # ── Frame-level energy check within silence interval ──
                # _word_rms() averages over the entire interval, which can
                # hide short bursts of loud content inside long silences.
                # Instead, scan 50ms frames and flag intervals where a
                # significant fraction of frames exceed the threshold.
                # Regression Case 25 follow-up.
                _frame_ms = 50.0
                _frame_samp = max(1, int(_frame_ms / 1000.0 * wav_sr))
                _s0 = int(_scan_start * wav_sr)
                _s1 = int(_scan_end * wav_sr)
                _n_frames = max(0, (_s1 - _s0 - _frame_samp) // max(1, _frame_samp // 2) + 1)
                if _n_frames <= 0:
                    _n_frames = 1
                _high_frames = 0
                _max_frame_e = 0.0
                _hop = max(1, _frame_samp // 2)
                for _fi in range(_n_frames):
                    _fs = _s0 + _fi * _hop
                    _fe = min(_fs + _frame_samp, _s1)
                    if _fe <= _fs:
                        continue
                    _fe_val = float(np.mean(np.abs(wav_audio[_fs:_fe])))
                    _max_frame_e = max(_max_frame_e, _fe_val)
                    if _fe_val > bgm_threshold:
                        _high_frames += 1
                _high_ratio = _high_frames / max(_n_frames, 1)
                # Flag if >= 20% of frames are above threshold (sustained
                # high energy, not just a transient click)
                if _high_ratio >= 0.20:
                    _suspect = {"xmin": round(_scan_start, 3),
                                "xmax": round(_scan_end, 3),
                                "duration": round(_scan_duration, 3),
                                "energy": round(_max_frame_e, 6),
                                "high_ratio": round(_high_ratio, 3),
                                "noise_floor": round(nf_bgm, 6)}
                    if _selection is not None:
                        for _evidence_key in (
                                "original_silence_span", "ctc_gap",
                                "evaluated_intersection",
                                "left_owner_ordinal", "right_owner_ordinal",
                                "left_ctc_ordinal", "right_ctc_ordinal",
                                "excluded_duration"):
                            _suspect[_evidence_key] = _selection.get(
                                _evidence_key)
                    suspect_intervals.append(_suspect)
                    suspect_dur += _scan_duration * _high_ratio
            if suspect_intervals:
                avg_speech = sum(speech_energies) / len(speech_energies) if speech_energies else 0
                suspect_ratio = suspect_dur / total_sil_dur if total_sil_dur > 0 else 0
                if suspect_ratio > args.bgm_speech_ratio * 0.1:
                    bgm_issues.append({"rule": "bgm_suspect",
                                       "noise_floor": round(nf_bgm, 6),
                                       "avg_speech_energy": round(avg_speech, 6),
                                       "suspect_intervals": len(suspect_intervals),
                                       "suspect_ratio": round(suspect_ratio, 3),
                                       "total_sil_dur": round(total_sil_dur, 3),
                                       "suspect_dur": round(suspect_dur, 3),
                                       "details": suspect_intervals})
                    if bgm_issues:
                        report["bgm_issues"] = bgm_issues
        # Phone-level QC — runs on POST-adjustment boundaries.
        if args.filter_suspicious:
            words_tier = tier_by_name(new_tg, "words")
            pp_tier2 = tier_by_name(new_tg, "pinyin_phones")
            if words_tier is not None and pp_tier2 is not None:
                # Build English/NVV ranges for targeted QC
                en_ranges: list[tuple[float, float]] = []
                nvv_ranges: list[tuple[float, float]] = []
                for w in words_tier.intervals:
                    if not w.text.strip() or is_silence(w.text):
                        continue
                    if is_english_token(w.text):
                        en_ranges.append((w.xmin, w.xmax))
                    elif is_nvv_token(w.text):
                        nvv_ranges.append((w.xmin, w.xmax))

                def _in_range(xmin: float, xmax: float,
                              ranges: list[tuple[float, float]]) -> bool:
                    for ws, we in ranges:
                        if xmin >= ws - 0.005 and xmax <= we + 0.005:
                            return True
                    return False

                short_phone_en = getattr(args, 'filter_short_phone_en_sec', 0.010)
                long_vowel_en = getattr(args, 'filter_long_vowel_en_sec', 0.500)
                long_cons_en = getattr(args, 'filter_long_consonant_en_sec', 1.000)

                for pi, p in enumerate(pp_tier2.intervals):
                    if not p.text.strip() or is_silence(p.text):
                        continue
                    if p.text.strip() == 'spn':
                        continue
                    # NVV: skip QC entirely (no acoustic model)
                    if _in_range(p.xmin, p.xmax, nvv_ranges):
                        continue
                    # English phone: use English-specific thresholds
                    if _in_range(p.xmin, p.xmax, en_ranges):
                        align_issues.extend(_phone_duration_qc_issues(
                            p, pi + 1,
                            filter_short_phone=args.filter_short_phone,
                            short_phone_sec=short_phone_en,
                            long_consonant_sec=long_cons_en,
                            long_vowel_sec=long_vowel_en,
                            english=True,
                        ))
                        continue
                    # Chinese phone: use standard thresholds
                    align_issues.extend(_phone_duration_qc_issues(
                        p, pi + 1,
                        filter_short_phone=args.filter_short_phone,
                        short_phone_sec=args.filter_short_phone_sec,
                        long_consonant_sec=args.filter_long_consonant_sec,
                        long_vowel_sec=args.filter_long_vowel_sec,
                    ))

        # All Phase 5 phone-QC branches have now contributed their issues.
        # Register the aggregate once so late short-phone findings cannot
        # leave an otherwise suspicious alignment marked ``ok``.
        _register_suspicious_alignment(align_issues, filter_reasons)

        # ── English phone coverage QC ──
        en_coverage_issues = []
        if en_data and args.filter_suspicious:
            min_en_cov = getattr(args, 'filter_min_en_phone_coverage', 0.25)
            words_tier = tier_by_name(new_tg, "words")
            pp_tier2 = tier_by_name(new_tg, "pinyin_phones")
            if words_tier and pp_tier2:
                for w_iv in words_tier.intervals:
                    if not is_english_token(w_iv.text.strip()):
                        continue
                    w_dur = w_iv.duration
                    if w_dur < 0.02:
                        continue
                    phone_dur = sum(
                        p.duration for p in pp_tier2.intervals
                        if p.xmin >= w_iv.xmin - 0.002
                        and p.xmax <= w_iv.xmax + 0.002
                        and p.text.startswith(EN_PHONE_PREFIX)
                    )
                    coverage = phone_dur / w_dur if w_dur > 0 else 0
                    if coverage < min_en_cov:
                        en_coverage_issues.append({
                            "word": w_iv.text.strip(),
                            "duration": round(w_dur, 4),
                            "phone_coverage": round(coverage, 3),
                        })
            if en_coverage_issues:
                report["en_low_coverage"] = en_coverage_issues

        # 更新 BGM + word_in_silence 到过滤原因
        if bgm_issues and "bgm_suspect" not in filter_reasons:
            filter_reasons.append("bgm_suspect")
            report["bgm_issues"] = bgm_issues
        if any(i["rule"] == "word_in_silence" for i in align_issues):
            if "word_in_silence" not in filter_reasons:
                filter_reasons.append("word_in_silence")

    # ── Hanzi tier integrity checks (BEFORE path decision) ──
    # These detect pinyin residue / CJK misalignment in the final
    # hanzi tier.  Must run here so filter_reasons is complete when
    # the output path is chosen below.
    raw_tier = tier_by_name(new_tg, "raw_text")
    hanzi_tier_final = tier_by_name(new_tg, "hanzi")
    if raw_tier and hanzi_tier_final:
        # (a) Direct pinyin residue scan — any pinyin syllable left in
        #     the hanzi tier is a hard alignment error.
        pinyin_labels: list[str] = []
        for iv in hanzi_tier_final.intervals:
            label = iv.text.strip()
            if label and is_pinyin_syllable(label):
                pinyin_labels.append(label)
        if pinyin_labels:
            filter_reasons.append("hanzi_pinyin")
            report.setdefault("hanzi_pinyin", {})["count"] = len(pinyin_labels)
            report["hanzi_pinyin"]["labels"] = pinyin_labels[:20]  # cap for report size

        # (b) CJK character coverage — compare raw_text CJK sequence
        #     against hanzi tier CJK sequence.  Missing or out-of-order
        #     CJK chars indicate the alignment dropped or misassigned them.
        # Compare against the immutable lexical source, never against the
        # rendered raw tier (which may have been rebuilt or punctuation-edited).
        # For reference mode this is the authority transcript; for fallback
        # mode it remains an independent ASR-source/hanzi structural check.
        raw_cjk = _prewrite_coverage.get(
            "source_cjk", _prewrite_coverage["reference_cjk"])
        hanzi_cjk = "".join(iv.text.strip() for iv in hanzi_tier_final.intervals
                           if iv.text.strip()
                           and ("一" <= iv.text.strip() <= "鿿"
                                or "㐀" <= iv.text.strip() <= "䶿"))
        _fallback_alignment = None
        if not reference_text_authoritative:
            _fallback_alignment = _fallback_cjk_alignment(
                reference_text_original, tier_by_name(new_tg, "words"))
            report["fallback_lexical_alignment"] = dict(_fallback_alignment)
        if raw_cjk != hanzi_cjk:
            # A fallback transcript is not lexical authority.  When every
            # realised pinyin token has a high-confidence monotonic source
            # match, source-only ASR insertions/deletions must not shift the
            # entire hanzi/pinyin projection or veto an otherwise coherent
            # acoustic alignment.  Authoritative reference mode retains the
            # exact historical CJK contract.
            _fallback_safe = bool(
                _fallback_alignment and _fallback_alignment.get("safe"))
            if reference_text_authoritative or not _fallback_safe:
                if "cjk_mismatch" not in filter_reasons:
                    filter_reasons.append("cjk_mismatch")
            else:
                report.setdefault("fallback_lexical_alignment", {})[
                    "cjk_mismatch_recovered"] = True
            report.setdefault("cjk_details", {})["raw_count"] = len(raw_cjk)
            report["cjk_details"]["hanzi_count"] = len(hanzi_cjk)
            report["cjk_details"]["delta"] = len(raw_cjk) - len(hanzi_cjk)

        # (c) Pinyin displacement detection — for each Chinese character in the
        #     hanzi tier, compare its expected pinyin (from pypinyin) against
        #     the actual pinyin in the words tier.  Consecutive mismatches
        #     indicate a displacement cascade caused by upstream STT errors
        #     propagating through the pinyin converter.
        #     Regression Case 52.
        _words_tier_qc = tier_by_name(new_tg, "words")
        if _words_tier_qc is not None and hanzi_tier_final is not None:
            _mismatch_count = 0
            _total_cjk = 0
            _consecutive_runs: list[dict] = []
            _current_run: list[dict] = []
            _run_start: int | None = None

            try:
                from pypinyin import lazy_pinyin, Style as _PyStyle
            except ImportError:
                lazy_pinyin = None  # type: ignore[assignment]

            if lazy_pinyin is not None and len(hanzi_tier_final.intervals) == len(_words_tier_qc.intervals):
                for _idx, (_h_iv, _w_iv) in enumerate(
                    zip(hanzi_tier_final.intervals, _words_tier_qc.intervals)
                ):
                    _h_text = _h_iv.text.strip()
                    _w_text = _w_iv.text.strip()

                    # Only check Chinese characters (single CJK char per interval)
                    if not (len(_h_text) == 1 and is_cjk(_h_text)):
                        # End current run if any
                        if _current_run and len(_current_run) >= 3:
                            _consecutive_runs.append({
                                "start": _run_start,
                                "end": _idx - 1,
                                "length": len(_current_run),
                                "sample": _current_run[:5],
                            })
                        _current_run = []
                        _run_start = None
                        continue

                    _total_cjk += 1

                    # Get expected pinyin (without tone)
                    try:
                        _expected = lazy_pinyin(_h_text, style=_PyStyle.TONE3,
                                                neutral_tone_with_five=True)
                        _exp_norm = _re.sub(r'\d+$', '', _expected[0]).lower() if _expected else ""
                    except Exception:
                        _exp_norm = ""

                    # Get actual pinyin from words tier (without tone)
                    _act_norm = _re.sub(r'\d+$', '', _w_text).lower()

                    if _exp_norm and _act_norm and _exp_norm != _act_norm:
                        _mismatch_count += 1
                        if _run_start is None:
                            _run_start = _idx
                        _current_run.append({
                            "idx": _idx, "hanzi": _h_text,
                            "expected": _exp_norm, "actual": _act_norm,
                        })
                    else:
                        # End current run
                        if _current_run and len(_current_run) >= 3:
                            _consecutive_runs.append({
                                "start": _run_start,
                                "end": _idx - 1,
                                "length": len(_current_run),
                                "sample": _current_run[:5],
                            })
                        _current_run = []
                        _run_start = None

                # Flush final run
                if _current_run and len(_current_run) >= 3:
                    _consecutive_runs.append({
                        "start": _run_start,
                        "end": len(hanzi_tier_final.intervals) - 1,
                        "length": len(_current_run),
                        "sample": _current_run[:5],
                    })

                if _total_cjk > 0:
                    _mismatch_rate = _mismatch_count / _total_cjk
                    _has_displacement = len(_consecutive_runs) > 0 and (
                        _mismatch_rate >= 0.25 or
                        any(r["length"] >= 6 for r in _consecutive_runs)
                    )

                    report.setdefault("pinyin_displacement", {})["mismatch_rate"] = round(_mismatch_rate, 3)
                    report["pinyin_displacement"]["total_cjk"] = _total_cjk
                    report["pinyin_displacement"]["mismatches"] = _mismatch_count
                    report["pinyin_displacement"]["displacement_runs"] = len(_consecutive_runs)

                    if _consecutive_runs:
                        report["pinyin_displacement"]["runs"] = [
                            {"start": r["start"], "end": r["end"],
                             "length": r["length"],
                             "sample_hanzi": "".join(s["hanzi"] for s in r["sample"]),
                             "sample_expected": "/".join(s["expected"] for s in r["sample"]),
                             "sample_actual": "/".join(s["actual"] for s in r["sample"])}
                            for r in _consecutive_runs[:5]
                        ]

                    if _has_displacement:
                        filter_reasons.append("pinyin_displacement")

                # ── text_order_mismatch: verify hanzi CJK char sequence
                #     is a subsequence of the ORIGINAL reference text (.txt).
                #     If not, CTC anchors have rearranged the character order
                #     which is a hard error — no ratio, no threshold. ──
                _orig_txt = raw_text  # fallback: may be CTC-normalized
                if (reference_text_authoritative
                        and getattr(args, 'original_txt_dir', None)):
                    _orig_path = Path(args.original_txt_dir) / f"{stem}.txt"
                    if _orig_path.exists():
                        _orig_txt = _orig_path.read_text(encoding="utf-8").strip()
                _ref_cjk = [c for c in re.sub(r'<sp\d+>', '', _orig_txt)
                            if '一' <= c <= '鿿']
                _hanzi_cjk = []
                for _h_iv in hanzi_tier_final.intervals:
                    _ht = _h_iv.text.strip()
                    if _ht and not _ht.startswith('<sp') and not _ht.startswith('<'):
                        for _c in _ht:
                            if '一' <= _c <= '鿿':
                                _hanzi_cjk.append(_c)
                if len(_ref_cjk) >= 6 and len(_hanzi_cjk) >= 6:
                    # Subsequence check: every char in hanzi must appear
                    # in ref in the same relative order
                    _ri = 0
                    _in_order = True
                    for _hc in _hanzi_cjk:
                        while _ri < len(_ref_cjk) and _ref_cjk[_ri] != _hc:
                            _ri += 1
                        if _ri >= len(_ref_cjk):
                            _in_order = False
                            break
                        _ri += 1
                    report["text_order"] = {
                        "ref_cjk_count": len(_ref_cjk),
                        "hanzi_cjk_count": len(_hanzi_cjk),
                        "in_order": _in_order,
                    }
                    if not _in_order:
                        # Find first 5 out-of-order positions for diagnostics
                        _samples = []
                        _ri = 0
                        _sample_count = 0
                        for _hi, _hc in enumerate(_hanzi_cjk):
                            while _ri < len(_ref_cjk) and _ref_cjk[_ri] != _hc:
                                _ri += 1
                            if _ri >= len(_ref_cjk):
                                _samples.append(
                                    f"hanzi[{_hi}]={_hc} not found after pos "
                                    f"{_ri if _ri < len(_ref_cjk) else 'end'} "
                                    f"in ref")
                                _sample_count += 1
                            elif _ri > _hi + 3:
                                _samples.append(
                                    f"hanzi[{_hi}]={_hc} found at "
                                    f"ref[{_ri}] (gap={_ri - _hi})")
                                _sample_count += 1
                            _ri += 1
                            if _sample_count >= 5:
                                break
                        report["text_order"]["samples"] = _samples
                        filter_reasons.append("text_order_mismatch")

    # ── Case 26-D / Regr. Case 47: init_only_phone + single_phone audit ──
    # After the proportional-split fix (Case 26+43), a multi-phone dict
    # word must never appear as its initial-only phone in pinyin_phones.
    # This check distinguishes three scenarios for single-phone pinyin words:
    #   A. True init_only: 1 phone = dict initial → FINAL IS MISSING (bug)
    #   B. Zero-initial:   1 phone = final, dict[0] is the initial (correct)
    #   C. Self-reference: 1 phone = word text itself (English fallback)
    _init_only_count = 0
    _init_only_examples: list[str] = []
    _zero_initial_count = 0       # correct single-phone zero-initial syllables
    _self_ref_count = 0           # self-referencing fallback labels
    if words_tier is not None and pp_tier is not None and pinyin_dict is not None:
        for _wi, _w_iv in enumerate(words_tier.intervals):
            _wt = _w_iv.text.strip()
            if not re.match(r'^[a-z]+[1-5]$', _wt) or len(_wt) <= 2:
                continue
            _dict_phones = pinyin_dict.get(_wt) or pinyin_dict.get(_wt.lower())
            if not _dict_phones or len(_dict_phones) < 2:
                continue  # zero-initial or not in dict — skipped by design
            _w_phones = [p for p in pp_tier.intervals
                         if p.xmax > _w_iv.xmin + 0.001
                         and p.xmin < _w_iv.xmax - 0.001
                         and not is_silence(p.text)]
            _phone_texts = [p.text.strip() for p in _w_phones]
            if len(_phone_texts) == 1:
                if _phone_texts[0] == _dict_phones[0]:
                    # Type A: single phone IS the initial → final missing
                    _init_only_count += 1
                    if len(_init_only_examples) < 5:
                        _init_only_examples.append(f"{_wt}→{_phone_texts}")
                elif _phone_texts[0] == _wt.lower():
                    # Type C: self-reference fallback
                    _self_ref_count += 1
                else:
                    # Type B: single phone is the final → zero-initial (correct)
                    _zero_initial_count += 1
    # Only flag true init_only (Type A).  Zero-initial (Type B) and
    # self-reference (Type C) are legitimate and not errors.
    if _init_only_count > 0:
        filter_reasons.append("init_only_phone")
        report["init_only_phone"] = {"count": _init_only_count,
                                      "examples": _init_only_examples}
    # Add diagnostic breakdown even when no errors (zero-initial is expected)
    if _zero_initial_count > 0 or _self_ref_count > 0:
        report["single_phone_breakdown"] = {
            "init_only_error": _init_only_count,
            "zero_initial_ok": _zero_initial_count,
            "self_ref_ok": _self_ref_count,
        }

    # ── Case 26-E: silence_boundary_split — initial-final boundary from silence ──
    # When the leakage filter (line 490) strips all real phones from word_phones,
    # only silence/spn entries may remain.  If >= 2 silence entries survive, the
    # MFA-precise branch (len(word_phones) >= 2) fires and uses silence boundaries
    # for the initial–final split, producing garbage timing (e.g. a 5 ms "ch"
    # followed by a 355 ms "ang4" for a 360 ms word).
    #
    # Detect from output: a multi-phone dict word whose first pinyin_phones
    # interval is shorter than 10 ms.  This is below the shortest physically
    # possible Chinese initial (~15–20 ms for stop consonants) and indicates
    # the split point came from a silence fragment rather than a real phone
    # boundary.  The 10 ms floor is deliberately conservative to avoid
    # flagging genuinely short initials in fast speech.
    _silence_split_count = 0
    _silence_split_examples: list[str] = []
    _SILENCE_SPLIT_FLOOR_S = 0.010  # seconds — physically impossible for any initial
    if words_tier is not None and pp_tier is not None and pinyin_dict is not None:
        for _wi, _w_iv in enumerate(words_tier.intervals):
            _wt = _w_iv.text.strip()
            if not re.match(r'^[a-z]+[1-5]$', _wt) or len(_wt) <= 2:
                continue
            _dict_phones = pinyin_dict.get(_wt) or pinyin_dict.get(_wt.lower())
            if not _dict_phones or len(_dict_phones) < 2:
                continue
            # Collect non-silence phones in this word's range, in order
            _w_phones = sorted(
                [p for p in pp_tier.intervals
                 if p.xmax > _w_iv.xmin + 0.001
                 and p.xmin < _w_iv.xmax - 0.001
                 and not is_silence(p.text)],
                key=lambda p: p.xmin)
            if len(_w_phones) >= 2:
                _first_dur = _w_phones[0].xmax - _w_phones[0].xmin
                # Regr. Case 48: only flag when the first "phone" is
                # actually a silence/spn label.  MFA can produce very
                # short real phones (e.g. ɕ at 10 ms between consecutive
                # identical words) — those are legitimate alignments,
                # not garbage splits from silence fragments.
                _first_label = _w_phones[0].text.strip()
                if _first_dur < _SILENCE_SPLIT_FLOOR_S and is_silence(_first_label):
                    _silence_split_count += 1
                    if len(_silence_split_examples) < 5:
                        _silence_split_examples.append(
                            f"{_wt}→{_w_phones[0].text.strip()}[{_first_dur*1000:.0f}ms]"
                            f" +{_w_phones[1].text.strip()}")
    if _silence_split_count > 0:
        filter_reasons.append("silence_boundary_split")
        report["silence_boundary_split"] = {"count": _silence_split_count,
                                             "examples": _silence_split_examples}

    # ── Case 27-B: overlapping_words — unresolved interval overlaps ──
    _overlap_count = 0
    _overlap_examples: list[str] = []
    _overlap_details: list[dict] = []
    if words_tier is not None:
        for _i in range(len(words_tier.intervals) - 1):
            _ov = words_tier.intervals[_i].xmax - words_tier.intervals[_i + 1].xmin
            if _ov > 0.005:
                _overlap_count += 1
                if len(_overlap_examples) < 5:
                    _overlap_examples.append(
                        f"{words_tier.intervals[_i].text.strip()}"
                        f"↔{words_tier.intervals[_i+1].text.strip()}"
                        f"({_ov*1000:.0f}ms)")
                if len(_overlap_details) < 10:
                    _left, _right = words_tier.intervals[_i], words_tier.intervals[_i + 1]
                    _overlap_details.append({
                        "left": _left.text.strip(), "right": _right.text.strip(),
                        "start_s": round(_right.xmin, 6),
                        "end_s": round(_left.xmax, 6),
                        "duration_us": _duration_ticks(_right.xmin, _left.xmax),
                    })
    if _overlap_count > 0:
        filter_reasons.append("overlapping_words")
        report["overlapping_words"] = {"count": _overlap_count,
                                        "examples": _overlap_examples,
                                        "details": _overlap_details}

    # ── Case 28: inverted_interval — xmin > xmax ──
    _inverted_count = 0
    _inverted_examples: list[str] = []
    for _tier in new_tg.tiers:
        for _iv in _tier.intervals:
            if _iv.xmin > _iv.xmax + 0.001:
                _inverted_count += 1
                if len(_inverted_examples) < 5:
                    _inverted_examples.append(
                        f"{_tier.name}:{_iv.text.strip()}"
                        f"[{_iv.xmin:.3f}>{_iv.xmax:.3f}]")
    if _inverted_count > 0:
        filter_reasons.append("inverted_interval")
        report["inverted_interval"] = {"count": _inverted_count,
                                        "examples": _inverted_examples}

    # ── Case 29: short_word — content word < 30 ms (physically impossible) ──
    _short_count = 0
    _short_examples: list[str] = []
    _short_details: list[dict] = []
    _SHORT_FLOOR_TICKS = 30_000  # integer microseconds; strict < 30 ms
    if words_tier is not None:
        for _iv in words_tier.intervals:
            _text = _iv.text.strip()
            if (not _text or is_silence(_iv.text) or is_punct(_text)
                    or is_nvv_token(_text)):
                continue
            _duration_us = _duration_ticks(_iv.xmin, _iv.xmax)
            if _duration_us < _SHORT_FLOOR_TICKS:
                _short_count += 1
                if len(_short_examples) < 8:
                    _short_examples.append(
                        f"{_text}[{_duration_us / 1000.0:.3f}ms]")
                if len(_short_details) < 10:
                    _short_details.append({"text": _text,
                                           "start_s": round(_iv.xmin, 6),
                                           "end_s": round(_iv.xmax, 6),
                                           "duration_us": _duration_us})
    if _short_count > 0:
        filter_reasons.append("short_word")
        report["short_word"] = {"count": _short_count,
                                 "examples": _short_examples,
                                 "details": _short_details}


    # ── Case 32: english_single_phone — English word (not NVV, not punct)
    #     whose pinyin_phones has only 1 self-referencing phone instead of
    #     proper en:-prefixed ARPABET phonemes.  English-path equivalent of
    #     Case 26 FULL_WORD_AS_PHONE. ──
    # Regr. Case 48: use CMUdict (not pinyin_dict) for expected-phone
    #     diagnostics.  pinyin_dict is the CHINESE syllable decomposition
    #     dict — looking up English words in it is semantically wrong.
    _en_single_count = 0
    _en_single_examples: list[str] = []
    if words_tier is not None and pp_tier is not None:
        _pp_idx = 0
        for _wi, _w_iv in enumerate(words_tier.intervals):
            _wt = _w_iv.text.strip()
            _ws, _we = _w_iv.xmin, _w_iv.xmax
            # Skip silence, punct, NVV — only check English tokens
            if not _wt or is_silence(_wt) or is_punct(_wt) or is_nvv_token(_wt):
                while (_pp_idx < len(pp_tier.intervals)
                       and pp_tier.intervals[_pp_idx].xmax <= _we + 0.002):
                    _pp_idx += 1
                continue
            if not is_english_token(_wt):
                while (_pp_idx < len(pp_tier.intervals)
                       and pp_tier.intervals[_pp_idx].xmax <= _we + 0.002):
                    _pp_idx += 1
                continue
            # Collect non-silence phones for this English word
            _w_phones = []
            __pi = _pp_idx
            while __pi < len(pp_tier.intervals) and pp_tier.intervals[__pi].xmin < _we - 0.001:
                _p = pp_tier.intervals[__pi]
                if (_p.xmax > _ws + 0.001 and _p.text
                        and not is_silence(_p.text.strip())):
                    _w_phones.append(_p.text.strip())
                __pi += 1
            # Self-reference: only 1 phone AND it equals the word itself.
            # Proper English phones have en: prefix; self-reference does not.
            if len(_w_phones) == 1 and _w_phones[0] in (_wt, _wt.lower(), _wt.upper()):
                _en_single_count += 1
                if len(_en_single_examples) < 5:
                    # Regr. Case 48: use _cmu (CMUdict) for English diagnostics.
                    _cmu_entry = _cmu.get(_wt.lower())
                    _en_single_examples.append(
                        f"{_wt}→{_w_phones[0]!r}" +
                        (f" (cmu:{_cmu_entry})" if _cmu_entry else " (not in CMUdict)"))
    if _en_single_count > 0:
        filter_reasons.append("english_single_phone")
        report["english_single_phone"] = {"count": _en_single_count,
                                           "examples": _en_single_examples}

    # ── Case 33: english_phone_deficit — English word has fewer phones
    #     than the dict expects (but > 1, so not caught by Case 32).
    #     English MFA under-produced phones for this word. ──
    # Regr. Case 48: use CMUdict (not pinyin_dict) for expected-phone
    #     count.  pinyin_dict entries have at most 2 phones (initial+final),
    #     so the old condition ``_n_got >= 2 and _n_got < 2`` could never
    #     fire — the check was effectively dead code.  CMUdict entries
    #     have 2-15 phones, so the deficit detection is now meaningful.
    _en_deficit_count = 0
    _en_deficit_examples: list[str] = []
    if words_tier is not None and pp_tier is not None:
        _pp_idx = 0
        for _wi, _w_iv in enumerate(words_tier.intervals):
            _wt = _w_iv.text.strip()
            _ws, _we = _w_iv.xmin, _w_iv.xmax
            if not _wt or not is_english_token(_wt):
                while (_pp_idx < len(pp_tier.intervals)
                       and pp_tier.intervals[_pp_idx].xmax <= _we + 0.002):
                    _pp_idx += 1
                continue
            # Regr. Case 48: CMUdict lookup for English words.
            _dp = _cmu.get(_wt.lower())
            if not _dp or len(_dp) < 2:
                while (_pp_idx < len(pp_tier.intervals)
                       and pp_tier.intervals[_pp_idx].xmax <= _we + 0.002):
                    _pp_idx += 1
                continue
            _w_phones = []
            __pi = _pp_idx
            while __pi < len(pp_tier.intervals) and pp_tier.intervals[__pi].xmin < _we - 0.001:
                _p = pp_tier.intervals[__pi]
                # Interval stores its label in ``text``; ``mark`` is not
                # part of this project's interval API (Case 64).
                if (_p.xmax > _ws + 0.001 and _p.text
                        and not is_silence(_p.text.strip())):
                    _w_phones.append(_p.text.strip())
                __pi += 1
            _n_got = len(_w_phones)
            _n_exp = len(_dp)
            if _n_got >= 2 and _n_got < _n_exp:
                if not all(ph in (_wt, _wt.lower(), _wt.upper()) for ph in _w_phones):
                    _en_deficit_count += 1
                    if len(_en_deficit_examples) < 5:
                        # Regr. Case 48: CMUdict entry shown for diagnostics.
                        _en_deficit_examples.append(
                            f"{_wt}→got {_n_got} phones, cmu:{_dp} ({_n_exp})")
    # A verified English MFA ledger is authoritative for both word identity
    # and phone sequence.  CMUdict is retained as a diagnostic fallback, but
    # must not reject a result whose exact provenance was already verified.
    _en_provenance_verified = (
        isinstance(report.get("english_provenance"), dict)
        and report["english_provenance"].get("status") == "verified"
    )
    if _en_deficit_count > 0 and not _en_provenance_verified:
        filter_reasons.append("english_phone_deficit")
        report["english_phone_deficit"] = {"count": _en_deficit_count,
                                            "examples": _en_deficit_examples}

    # ── Case 34: pp_tier_gaps — pinyin_phones has uncovered gaps *inside*
    #     one content word.  This tier is intentionally sparse across
    #     natural pauses, punctuation, and English words whose phones live in
    #     the provenance-backed English alignment.
    _PP_GAP_THRESHOLD_S = 0.010
    _pp_gap_details = _find_internal_pp_gaps(
        pp_tier, words_tier, _PP_GAP_THRESHOLD_S)
    _pp_gap_count = len(_pp_gap_details)
    if _pp_gap_count > 0:
        _record_filterable_qc(
            report, filter_reasons, args.filter_suspicious,
            "pp_tier_gaps", {"count": _pp_gap_count,
                              "details": _pp_gap_details[:10]}
        )

    # ── Case 35: words_tier_gaps — direct gaps between content words.
    #     Punctuation and explicit silence intervals own their boundary gaps;
    #     those are not alignment holes and must remain preserved.
    _wt_gap_count = 0
    _wt_gap_examples: list[str] = []
    _wt_gap_details: list[dict] = []
    _WT_GAP_THRESHOLD_S = 0.020
    if words_tier is not None and len(words_tier.intervals) >= 2:
        for _i in range(len(words_tier.intervals) - 1):
            _cur = words_tier.intervals[_i]
            _nxt = words_tier.intervals[_i + 1]
            _cl = _cur.text.strip() if _cur.text else ""
            _nl = _nxt.text.strip() if _nxt.text else ""
            if (not _cl or not _nl or is_silence(_cl) or is_silence(_nl)
                    or is_punct(_cl) or is_punct(_nl)):
                continue
            _gap_ticks = _duration_ticks(_cur.xmax, _nxt.xmin)
            if _gap_ticks > _threshold_ticks(_WT_GAP_THRESHOLD_S):
                _wt_gap_count += 1
                if len(_wt_gap_examples) < 5:
                    _wt_gap_examples.append(
                        f"{_cl!r}→{_nl!r}[{_gap_ticks / 1000.0:.3f}ms]")
                if len(_wt_gap_details) < 10:
                    _wt_gap_details.append({"left": _cl, "right": _nl,
                                            "start_s": round(_cur.xmax, 6),
                                            "end_s": round(_nxt.xmin, 6),
                                            "duration_us": _gap_ticks})
    if _wt_gap_count > 0:
        _record_filterable_qc(
            report, filter_reasons, args.filter_suspicious,
            "words_tier_gaps", {"count": _wt_gap_count,
                                 "examples": _wt_gap_examples,
                                 "details": _wt_gap_details}
        )

    # ── Case 36: tier_discontinuity — a tier has too many gaps
    #     (> 10% of intervals), indicating systematic alignment failure. ──
    _discon_tiers = _collect_tier_discontinuities(new_tg, words_tier)
    if _discon_tiers:
        _record_filterable_qc(
            report, filter_reasons, args.filter_suspicious,
            "tier_discontinuity", {"tiers": _discon_tiers}
        )


    # 统一设置过滤状态和输出路径
    # Finalize before the final classification.  Strict reviewers must inspect
    # exactly the labels that are written, never an earlier pre-finalized view.
    # ``phones`` is an internal source tier in the published five-tier
    # artifact, but it remains mandatory evidence for the final owner gate.
    # Capture it before the legacy output-shape reduction; never publish a
    # result whose source phones cross or escape a words owner.
    _publication_phones_tier = tier_by_name(new_tg, "phones")
    new_tg.tiers = [t for t in new_tg.tiers if t.name != "phones"]
    _finalize_textgrid(new_tg)

    # Final publication veto: a substantive interior silence must remain
    # visible for diagnosis and cannot be emitted as a strict candidate.  Do
    # this after finalization so leading ``<spN>`` canonicalization cannot
    # turn a provisional edge marker into a false interior hit.
    _final_words = tier_by_name(new_tg, "words")
    _fallback_pause_gate = _fallback_pause_qualification(
        _final_words, reference_mode_policy, fallback_correspondence,
        source_words_snapshot, ctc_token_list)
    if _fallback_pause_gate["pause_count"]:
        report["fallback_pause_qualification"] = _fallback_pause_gate
        _final_pause_details = _fallback_pause_gate["details"]
        report["mid_sp"] = {
            "count": len(_final_pause_details),
            "details": _final_pause_details[:10],
        }
        filter_reasons.append("mid_sp")
    else:
        report.pop("mid_sp", None)
    if args.handle_unexpected_sil:
        sil_filter_reasons = _final_unexpected_silence_reasons(_final_words)
        if sil_filter_reasons:
            report["unexpected_silence"] = list(sil_filter_reasons)
            report["unexpected_silence_evidence"] = {
                "count": len(_fallback_pause_gate["details"]),
                "details": _fallback_pause_gate["details"][:10],
            }
        else:
            # Do not carry a pre-final pinyin/geometry observation into the
            # published classification after punctuation ownership absorbed
            # or moved that silence.
            report.pop("unexpected_silence", None)
            report.pop("unexpected_silence_evidence", None)
        if sil_filter_reasons:
            filter_reasons.extend(sil_filter_reasons)
    _processed_digest = _processed_geometry_digest(_final_words)
    _processed_ledger = list(_processed_geometry_ledger(_final_words))
    report["processed_geometry_digest"] = _processed_digest
    report["processed_operation_ledger"] = _processed_ledger
    report["processed_geometry"] = {
        "schema": "processed-words-geometry-v1",
        "frozen": bool(getattr(_final_words, "_processed_geometry_frozen", False)),
        "digest": _processed_digest,
        "ledger": _processed_ledger,
    }
    _interior_sp_details = []
    if _final_words is not None:
        for _index, _interval in enumerate(_final_words.intervals):
            if _is_substantive_interior_silence(
                    _final_words.intervals, _index):
                _interior_sp_details.append({
                    "index": _index,
                    "label": _interval.text.strip(),
                    "start_s": round(_interval.xmin, 6),
                    "end_s": round(_interval.xmax, 6),
                    "duration_us": _duration_ticks(_interval.xmin, _interval.xmax),
                })
    if _interior_sp_details:
        report["strict_interior_sp"] = {
            "count": len(_interior_sp_details),
            "details": _interior_sp_details[:10],
        }
        if getattr(args, "filter_suspicious", True):
            filter_reasons.append("strict_interior_sp")

    # Final publication invariant: no later derived-tier or label-normalizing
    # step may turn an unresolved terminal/internal SP into an ``ok`` output.
    # Keep the full evidence in the report; the reason is a hard veto.
    _nonleading_silence = _published_nonleading_silence_details(_final_words)
    if _nonleading_silence:
        report["nonleading_pure_silence"] = {
            "count": len(_nonleading_silence),
            "details": _nonleading_silence[:20],
        }
        filter_reasons.append("nonleading_pure_silence")

    # NVASR candidate provenance is an independent acoustic contract.  Final
    # display ownership is recorded separately and cannot silently replace a
    # missing/ambiguous raw, forced, or adjusted candidate span.
    _nvasr_provenance = _nvasr_candidate_provenance_audit(
        ctc_token_list, _final_words,
        required=not reference_text_authoritative,
        wav_duration_s=_wav_axis_duration)
    report["nvasr_candidate_provenance"] = _nvasr_provenance
    if _nvasr_provenance["status"] == "rejected":
        filter_reasons.append("nvasr_candidate_provenance_rejected")

    _publication_contract_reasons, _publication_contract_details = (
        _publication_contract_audit(
            tier_by_name(new_tg, "words"),
            tier_by_name(new_tg, "hanzi"),
            tier_by_name(new_tg, "pinyin_phones"),
            _publication_phones_tier,
            reference_text_original,
            source_words_snapshot,
            ctc_token_list,
            reference_text_authoritative,
            report.get("english_provenance"),
            unknown_recovery_proof=unknown_recovery_proof,
            fallback_correspondence=fallback_correspondence,
            reference_mode=reference_mode_policy,
            raw_text_tier=tier_by_name(new_tg, "raw_text"),
            pinyin_tier=tier_by_name(new_tg, "pinyin"),
            fallback_surface_ledger=fallback_surface_ledger,
            fallback_punctuation_projection=fallback_punctuation_projection))
    report["publication_contract"] = {
        "schema": "publication-owner-contract-v1",
        "status": "rejected" if _publication_contract_reasons else "verified",
        "reasons": _publication_contract_reasons,
        "details": _publication_contract_details,
    }
    for _pause_evidence_name in (
            "mid_sp", "strict_interior_sp", "unexpected_silence",
            "unexpected_silence_evidence"):
        if _pause_evidence_name in report:
            _publication_contract_details[_pause_evidence_name] = report[
                _pause_evidence_name]
    filter_reasons.extend(_freeze_reasons)
    filter_reasons.extend(_publication_contract_reasons)

    # Hard lexical publication evidence must describe exactly the finalized
    # labels that are about to be written.  In particular, `_finalize_textgrid`
    # converts leading `<sp2>`/`<sp3>` to `<sp1>`; validating earlier would
    # interpret that provisional marker as literal semantic content and retain
    # a stale false veto in the report.
    _coverage, _coverage_reasons = assess_reference_coverage(
        reference_text_original,
        tier_by_name(new_tg, "words"),
        tier_by_name(new_tg, "hanzi"),
        reference_source=reference_source,
        unknown_source_count=unknown_source_count,
        reference_authoritative=reference_text_authoritative,
    )
    report["reference_coverage"] = _coverage
    report["hard_integrity_reasons"] = _coverage_reasons
    if report.get("warnings"):
        report["warning_evidence"] = {
            "count": len(report["warnings"]),
            "messages": list(report["warnings"][:20]),
        }
    if _coverage.get("exact_semantic_sequence") is False:
        _semantic_ref = _strict_semantic_tokens(reference_text_original)
        _semantic_obs = _strict_semantic_tokens(" ".join(
            iv.text for iv in (tier_by_name(new_tg, "hanzi").intervals
                               if tier_by_name(new_tg, "hanzi") else [])
            if iv.text).strip())
        _semantic_mismatches = []
        for _si, (_expected, _actual) in enumerate(
                zip(_semantic_ref, _semantic_obs)):
            if _expected != _actual:
                _semantic_mismatches.append({"index": _si,
                                              "expected": _expected,
                                              "actual": _actual})
            if len(_semantic_mismatches) >= 10:
                break
        report["semantic_evidence"] = {
            "reference_tokens": len(_semantic_ref),
            "observed_tokens": len(_semantic_obs),
            "mismatch_count_lower_bound": len(_semantic_mismatches),
            "mismatches": _semantic_mismatches,
        }
    filter_reasons.extend(_coverage_reasons)

    # strict-ok is intentionally stricter than the legacy best-effort mode:
    # every already-computed diagnostic is a veto.  It does not invent a new
    # acoustic judgement; the independent disk auditor records that judgement
    # as not evaluated.
    if getattr(args, "strict_ok", False) and report.get("warnings"):
        filter_reasons.append("warnings")
    filter_reasons = _apply_fallback_pause_veto_qualification(
        filter_reasons, _fallback_pause_gate)
    filter_reasons = list(dict.fromkeys(filter_reasons))
    if filter_reasons:
        report["status"] = "filtered_" + "_".join(filter_reasons)
        report["filter_reasons"] = filter_reasons
        if align_issues:
            report["alignment_issues"] = align_issues
        out_path = filtered_dir / tg_path.name
        stale = output_dir / tg_path.name
    else:
        report["status"] = "ok"
        out_path = output_dir / tg_path.name
        stale = filtered_dir / tg_path.name

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path}")

    if stale.exists() and args.overwrite:
        stale.unlink()
    write_textgrid(new_tg, out_path)
    report["output"] = str(out_path)
    report["textgrid_duration"] = round(tg.xmax - tg.xmin, 3)
    return report


# ── Module-level worker for multiprocessing (must be picklable) ──
_W = None


def _worker_init(_ipa, _py_dict, _py_case, _a, _txt_d, _wav_d, _out_d,
                 _filt_d, _raw_text_index, _wav_index=None):
    import os as _os
    for ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ[ev] = "1"
    global _W
    _W = (_ipa, _py_dict, _py_case, _a, _txt_d, _wav_d, _out_d, _filt_d,
          _raw_text_index, _wav_index)


def _worker_fn(tgp):
    (_ipa, _py_dict, _py_case, _a, _txt_d, _wav_d, _out_d, _filt_d,
     _raw_text_index, _wav_index) = _W
    return process_one(tgp, _txt_d, _wav_d, _out_d, _filt_d, _a,
                       _ipa, _py_dict, _py_case, _raw_text_index, _wav_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process MFA TextGrids for Chinese alignment.")
    parser.add_argument("--txt-dir", type=Path, default=PROJECT_ROOT / "corpus_clean" / "txt")
    parser.add_argument("--textgrid-dir", type=Path, default=PROJECT_ROOT / "aligned")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--filtered-dir", type=Path, default=PROJECT_ROOT / "filtered")
    parser.add_argument("--wav-dir", type=Path, default=PROJECT_ROOT / "corpus_clean" / "wav")
    parser.add_argument("--mfa-input-axis-receipt", type=Path, default=None)
    parser.add_argument("--mfa-alignment-axis-receipt", type=Path, default=None)
    parser.add_argument("--mfa-axis-audio-root", type=Path, default=None)
    parser.add_argument("--tts-authoritative-audio-root", type=Path, default=None)
    parser.add_argument("--raw-text-dir", type=Path, default=None,
                        help="Directory with original Chinese text files")
    parser.add_argument("--reference-mode", choices=("auto", "authority", "fallback"),
                        default="auto",
                        help="Transcript authority policy; fallback ignores raw reference files.")
    parser.add_argument("--original-txt-dir", type=Path, default=None,
                        help="Directory with original {stem}.txt reference texts (for text_order check)")
    parser.add_argument("--pinyin-dict", type=Path, default=PROJECT_ROOT / "dict" / "fullpinyin_enword.dict")
    parser.add_argument("--ipa-dict", type=Path, default=PROJECT_ROOT / "dict" / "mfa_ipa.dict")
    parser.add_argument("--en-phones-dir", type=Path, default=None,
                        help="Directory with English MFA phone JSON files ({stem}_en_phones.json).")
    parser.add_argument("--tone-ref", type=Path, default=PROJECT_ROOT / "output" / "tone_mapping.json",
                        help="Output path for tone reference table")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers for postprocessing (0=auto: cpu_count, 1=serial).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge-silence", action=argparse.BooleanOptionalAction, default=True,
                        help="Resolve final visual-word <sp0>/<sp1> gaps by bidirectional energy ownership.")
    parser.add_argument("--merge-max-sil-sec", type=float, default=0.2,
                        help="Final visual-word bidirectional energy-owner bound; hard-capped at 0.5s. Configure above 0.2s to admit ordinary <sp1> (default: 0.2s).")
    parser.add_argument("--merge-energy-threshold", type=float, default=0.5,
                        help="Merge when sil_nonzero_mean > prev_nonzero_mean * threshold (default: 0.5).")
    parser.add_argument("--fix-short-word", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-short-word-sec", type=float, default=0.25)
    parser.add_argument("--fix-min-silence-sec", type=float, default=0.4)
    parser.add_argument("--fix-search-sec", type=float, default=0.5)
    parser.add_argument("--fix-frame-ms", type=float, default=10.0)
    parser.add_argument("--fix-hop-ms", type=float, default=5.0)
    parser.add_argument("--fix-threshold-ratio", type=float, default=2.5)
    parser.add_argument("--fix-min-region-sec", type=float, default=0.04)
    parser.add_argument("--detect-bgm", action=argparse.BooleanOptionalAction, default=True,
                        help="Detect BGM/noise in silence intervals using global noise floor.")
    parser.add_argument("--bgm-frame-ms", type=float, default=10.0,
                        help="Frame size for noise floor estimation (ms).")
    parser.add_argument("--bgm-hop-ms", type=float, default=5.0,
                        help="Hop size for noise floor estimation (ms).")
    parser.add_argument("--bgm-noise-floor-ratio", type=float, default=2.0,
                        help="Silence energy > noise_floor * N triggers suspect.")
    parser.add_argument("--bgm-min-sil-dur", type=float, default=0.3,
                        help="Minimum silence duration to check (seconds).")
    parser.add_argument("--bgm-speech-ratio", type=float, default=1.0,
                        help="Silence energy > avg_speech * N triggers suspect (1.0 = at speech level).")
    parser.add_argument("--bgm-min-energy", type=float, default=0.01,
                        help="Absolute minimum RMS to trigger (filters out breathing/noise floor).")
    parser.add_argument("--bgm-max-threshold", type=float, default=0.05,
                        help="Hard ceiling on bgm_threshold. When 60th-percentile noise floor is "
                             "contaminated by loud content, the threshold is capped here so "
                             "abnormal silences are still detected.")
    parser.add_argument("--filter-suspicious", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-long-word-sec", type=float, default=1.0)
    parser.add_argument("--filter-flank-silence-sec", type=float, default=0.4)
    parser.add_argument("--filter-short-phone", action=argparse.BooleanOptionalAction, default=True,
                        help="Detect abnormally short phones (default: enabled).")
    parser.add_argument("--filter-short-phone-sec", type=float, default=0.015)
    parser.add_argument("--filter-long-consonant-sec", type=float, default=999.0,
                        help="Max consonant phone duration (default: disabled).")
    parser.add_argument("--filter-long-vowel-sec", type=float, default=999.0,
                        help="Max vowel phone duration (default: disabled).")
    parser.add_argument("--filter-short-phone-en-sec", type=float, default=0.010,
                        help="Min English phone duration (default: 0.010s).")
    parser.add_argument("--filter-long-vowel-en-sec", type=float, default=0.500,
                        help="Max English vowel duration (default: 0.500s).")
    parser.add_argument("--filter-long-consonant-en-sec", type=float, default=1.000,
                        help="Max English consonant duration (default: 1.000s).")
    parser.add_argument("--filter-min-en-phone-coverage", type=float, default=0.25,
                        help="Min phone coverage ratio for English words (default: 0.25).")
    parser.add_argument("--filter-min-word-sec", type=float, default=0.15)
    parser.add_argument("--filter-min-word-dur-sec", type=float, default=0.02,
                        help="Absolute minimum word duration (below = misaligned).")
    parser.add_argument("--filter-word-energy-ratio", type=float, default=2.0,
                        help="Flag lexical word core if 10ms-frame RMS < noise_floor * N; no extra floor.")
    parser.add_argument("--enable-word-in-silence-filter",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Explicitly enable the word-in-silence detector. "
                             "When omitted, a positive energy ratio preserves "
                             "legacy behaviour.")
    parser.add_argument("--filter-min-phone-coverage", type=float, default=0.35)
    parser.add_argument("--filter-edge-gap-sec", type=float, default=0.25)
    parser.add_argument("--copy-errors", action="store_true")
    parser.add_argument("--allow-filtered-integrity-failures", action="store_true",
                        help="Continue when mandatory-integrity failures were isolated in filtered/.")
    parser.add_argument("--strict-ok", action="store_true",
                        help="Treat every executed QC positive and warning as filterable.")
    parser.add_argument("--strict-en-provenance", action="store_true",
                        help="Require strict-en-mfa-v2 ledgers without enabling global strict-ok QC.")
    parser.add_argument("--enable-text-correction", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-check punctuation against silence gaps and emit corrected_text tier.")
    parser.add_argument("--handle-unexpected-sil", action=argparse.BooleanOptionalAction, default=True,
                        help="Diagnose unowned silence; final visual-word energy ownership is resolved once.")
    args = parser.parse_args()

    axis_errors, axis_stem_reasons = _load_axis_contract(args)
    if axis_errors:
        for error in axis_errors:
            print(f"ERROR: {error}")
        return 1
    args._axis_stem_reasons = axis_stem_reasons

    if args.strict_ok:
        # _record_filterable_qc honours this flag.  A configured legacy
        # --no-filter-suspicious must never weaken strict-ok.
        args.filter_suspicious = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_dir.mkdir(parents=True, exist_ok=True)
    # Discover reference transcripts once and share the immutable index with
    # serial/parallel workers.  This avoids one recursive directory walk per
    # TextGrid while preserving deterministic first-match precedence.
    raw_text_index = _build_original_text_index(args.raw_text_dir)
    wav_index = _build_wav_index(args.wav_dir)

    # Load dictionaries and build IPA->pinyin mapping
    print("Loading dictionaries...")
    pinyin_dict, pinyin_case = load_dict(args.pinyin_dict)
    ipa_dict, _ = load_dict(args.ipa_dict)
    print(f"  Pinyin dict: {len(pinyin_dict)} entries")
    print(f"  IPA dict: {len(ipa_dict)} entries")

    ipa_to_pinyin = build_ipa_to_pinyin_map(pinyin_dict, ipa_dict)
    print(f"  IPA->Pinyin phone mappings: {len(ipa_to_pinyin)}")

    # Build and export tone reference table
    tone_ref = build_tone_reference_table(ipa_to_pinyin)
    args.tone_ref.parent.mkdir(parents=True, exist_ok=True)
    with open(args.tone_ref, 'w', encoding='utf-8') as f:
        json.dump(tone_ref, f, ensure_ascii=False, indent=2)
    print(f"  Tone reference table: {args.tone_ref}")
    # Print tone marks safely (avoid gbk encoding issues on Windows)
    tm = tone_ref['tone_marks_table']
    tm_str = ", ".join(f"{k}->{v}" for k, v in tm.items())
    try:
        print(f"  Tone marks: {tm_str}")
    except UnicodeEncodeError:
        print(f"  Tone marks: {json.dumps(tm)}")

    tg_paths = sorted(args.textgrid_dir.glob("*.TextGrid"))
    if not tg_paths:
        print(f"No TextGrid files in {args.textgrid_dir}")
        return 1

    # Resolve worker count
    import multiprocessing as mp
    import platform as _plat
    n_workers = args.workers
    if n_workers <= 0:
        n_workers = min(32, len(tg_paths))  # cap at 32 — 384 forks on EPYC is wasteful
    n_workers = min(n_workers, len(tg_paths))

    reports = []
    if n_workers <= 1 or len(tg_paths) <= 2:
        # Serial path
        for tgp in tg_paths:
            try:
                reports.append(process_one(tgp, args.txt_dir, args.wav_dir,
                                           args.output_dir, args.filtered_dir, args,
                                           ipa_to_pinyin, pinyin_dict, pinyin_case,
                                           raw_text_index, wav_index))
            except Exception as exc:
                reports.append({"stem": tgp.stem, "status": "error", "error": str(exc)})
                if args.copy_errors:
                    shutil.copy2(tgp, args.filtered_dir / tgp.name)
    else:
        # ── Executor selection ──
        # Linux/macOS: ProcessPoolExecutor with fork — COW sharing of ~2200-entry
        #               dicts, true CPU parallelism via BLAS=1 per worker.
        # Windows:      ThreadPoolExecutor — avoids per-worker spawn overhead
        #               (each worker re-imports numpy/scipy/soundfile, ~2-5 s).
        #               NumPy energy analysis releases the GIL, so threads work.
        _is_win = _plat.system() == "Windows"
        if _is_win:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _exec_label = "ThreadPool"
        else:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor, as_completed
            _exec_label = "ProcessPool"
            _mp_ctx = _mp.get_context("fork")  # force fork — avoids pickle errors

        print(f"  Postprocess parallel: {n_workers} workers for {len(tg_paths)} files ({_exec_label})")
        if _is_win:
            # ThreadPool: set globals once, then all threads see them
            _worker_init(ipa_to_pinyin, pinyin_dict, pinyin_case,
                         args, args.txt_dir, args.wav_dir,
                         args.output_dir, args.filtered_dir, raw_text_index,
                         wav_index)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker_fn, tgp): tgp for tgp in tg_paths}
                for fut in as_completed(futures):
                    tgp = futures[fut]
                    try:
                        reports.append(fut.result())
                    except Exception as exc:
                        reports.append({"stem": tgp.stem, "status": "error", "error": str(exc)})
                        if args.copy_errors:
                            shutil.copy2(tgp, args.filtered_dir / tgp.name)
        else:
            # ProcessPool: initializer passes dicts once (COW after fork)
            with ProcessPoolExecutor(max_workers=n_workers,
                                     mp_context=_mp_ctx,
                                     initializer=_worker_init,
                                     initargs=(ipa_to_pinyin, pinyin_dict, pinyin_case,
                                               args, args.txt_dir, args.wav_dir,
                                               args.output_dir, args.filtered_dir,
                                               raw_text_index, wav_index)) as pool:
                futures = {pool.submit(_worker_fn, tgp): tgp for tgp in tg_paths}
                for fut in as_completed(futures):
                    tgp = futures[fut]
                    try:
                        reports.append(fut.result())
                    except Exception as exc:
                        reports.append({"stem": tgp.stem, "status": "error", "error": str(exc)})
                        if args.copy_errors:
                            shutil.copy2(tgp, args.filtered_dir / tgp.name)

    rp = args.output_dir / "postprocess_report.jsonl"
    with rp.open("w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Done. {counts}. report={rp}")
    error_count = counts.get("error", 0)
    if error_count:
        print(f"ERROR: {error_count} file(s) failed during post-processing; see {rp}")
        return 1
    hard_integrity_count = sum(1 for row in reports if row.get("hard_integrity_reasons"))
    if hard_integrity_count:
        print(f"  {hard_integrity_count} mandatory-integrity failures isolated in filtered/")
    if hard_integrity_count and not args.allow_filtered_integrity_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
