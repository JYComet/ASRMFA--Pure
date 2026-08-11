#!/usr/bin/env python3
"""Independent disk auditor for the strict-ok v3.1 publication contract.

This program deliberately does not trust postprocess' self-report.  It rereads
the final TextGrid, source MFA alignment, CTC bundle, reference, and WAV from
disk.  Failed candidates are isolated into the run-local ``filtered`` folder;
the command is not a repair tool and never deletes an input result.
"""

from __future__ import annotations

import argparse
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
)
from postprocess_textgrids import parse_textgrid  # noqa: E402

POLICY_VERSION = "strict-ok-v3.2"
EN_PROVENANCE_SCHEMA = "strict-en-mfa-v1"
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
_ENGLISH = re.compile(r"[A-Za-z]+")
_PINYIN = re.compile(r"^[a-z]+[1-5]$")
_SP1 = re.compile(r"<sp1>", re.I)
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


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError("non-positive WAV sample rate")
        return handle.getnframes() / rate


def _axis_digest(value: object) -> str:
    return stable_json_digest(value)


def _axis_wav_meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        if rate <= 0:
            raise ValueError("invalid WAV sample rate")
        return {"sha256": _sha256(path), "duration_s": frames / rate,
                "sample_rate": rate, "frames": frames,
                "channels": handle.getnchannels(), "sample_width": handle.getsampwidth()}


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
            or input_axis.get("stems_digest") != _axis_digest(stems)
            or set(stems) != set(expected)):
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
            tts = tts_root / f"{stem}.wav"
            tts_meta = _axis_wav_meta(tts)
            identity = all(tts_meta[key] == actual[key]
                           for key in ("sha256", "sample_rate", "frames", "channels", "sample_width")) and abs(
                               tts_meta["duration_s"] - actual["duration_s"]) <= EPS
            transform = transforms.get(stem)
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
    """Return ordered CJK/NVV/punctuation/English tokens, excluding sp1."""
    text = _SP1.sub("", text).translate(_PUNCT_MAP)
    result: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        nvv = _NVV.match(text, index)
        if nvv:
            label = nvv.group(1).upper()
            if is_nvv_token(label):
                result.append(("nvv", label))
            else:
                result.append(("other", nvv.group(0)))
            index = nvv.end()
            continue
        english = _ENGLISH.match(text, index)
        if english:
            word = english.group(0)
            if not is_nvv_token(word) and word.lower() != "sp1":
                result.append(("english", word.lower()))
            index = english.end()
            continue
        if _CJK.fullmatch(char):
            result.append(("cjk", char))
        elif is_punct(char):
            result.append(("punct", char))
        elif char not in "<>[]":
            result.append(("other", char))
        index += 1
    return result


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


def _report_reasons(row: dict) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != "ok":
        reasons.append("report_status_not_ok")
    for key in ("hard_integrity_reasons", "filter_reasons", "warnings", "alignment_issues"):
        if row.get(key):
            reasons.append(f"report_positive:{key}")
    # These fields are normal, independently rechecked transformations.  They
    # are not proof of correctness by themselves, but the auditor separately
    # verifies final tier geometry, reference sequence, phones and provenance.
    # Keep genuine positive QC fields above as vetoes while allowing benign
    # bookkeeping that postprocess emits for every corrected-but-valid stem.
    allowed = {
        "stem", "status", "output", "textgrid_duration", "reference_source",
        "reference_text_authoritative", "reference_coverage", "warnings",
        "hard_integrity_reasons", "filter_reasons", "alignment_issues",
        "english_provenance", "silence_merges", "pp_deoverlap_fixed",
        "text_corrected", "pinyin_displacement", "text_order",
    }
    coverage = row.get("reference_coverage") or {}
    displacement = row.get("pinyin_displacement") or {}
    order = row.get("text_order") or {}
    if row.get("text_corrected") and not (
            coverage.get("exact_cjk_sequence") is True
            and displacement.get("mismatch_rate") == 0.0
            and displacement.get("displacement_runs") == 0
            and order.get("in_order") is True):
        reasons.append("report_positive:text_corrected")
    if row.get("pinyin_displacement") and not (
            displacement.get("mismatch_rate") == 0.0
            and displacement.get("displacement_runs") == 0):
        reasons.append("report_positive:pinyin_displacement")
    if row.get("text_order") and not (
            order.get("in_order") is True
            and order.get("ref_cjk_count") == order.get("hanzi_cjk_count")):
        reasons.append("report_positive:text_order")
    for key, value in row.items():
        if key not in allowed and value:
            reasons.append(f"report_positive:{key}")
    return reasons


def _sp1_reasons(tg) -> list[str]:
    reasons: list[str] = []
    for tier in tg.tiers[:3]:
        text = _tier_text(tier)
        if not text.startswith("<sp1>") or len(_SP1.findall(text)) != 1:
            reasons.append(f"sp1_contract:{tier.name}")
    for tier in tg.tiers[3:]:
        if not tier.intervals or tier.intervals[0].text.strip() != "<sp1>":
            reasons.append(f"sp1_contract:{tier.name}")
    return reasons


def _content_reasons(tg, reference: str) -> list[str]:
    reasons: list[str] = []
    raw, pinyin, hanzi, words, phones = tg.tiers
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

    reference_tokens = _semantic_tokens(reference)
    final_tokens = _semantic_tokens(_tier_text(hanzi))
    if reference_tokens != final_tokens:
        reasons.append("reference_semantic_sequence_mismatch")
    if _semantic_tokens(_tier_text(raw)) != reference_tokens:
        reasons.append("reference_raw_semantic_sequence_mismatch")
    # Pinyin syllables are the Chinese realization, not reference English
    # words.  Remove only fully toned pinyin tokens before comparing the
    # remaining NVV/punctuation/English sequence to authority.
    pinyin_semantic = _semantic_tokens(re.sub(
        r"(?<![A-Za-z])[a-z]+[1-5](?![A-Za-z0-9])", "",
        _SP1.sub("", _tier_text(pinyin))))
    reference_non_cjk = [token for token in reference_tokens if token[0] != "cjk"]
    if pinyin_semantic != reference_non_cjk:
        reasons.append("reference_pinyin_semantic_sequence_mismatch")
    reference_cjk = [value for kind, value in reference_tokens if kind == "cjk"]
    hanzi_cjk = [char for iv in hanzi.intervals for char in iv.text if _CJK.fullmatch(char)]
    if reference_cjk != hanzi_cjk:
        reasons.append("reference_hanzi_cjk_mismatch")
    if any(_PINYIN.search(iv.text.strip()) for iv in hanzi.intervals):
        reasons.append("hanzi_contains_pinyin")

    pinyin_words = [iv for iv in words.intervals if _PINYIN.fullmatch(iv.text.strip())]
    if len(reference_cjk) != len(pinyin_words):
        reasons.append("cjk_pinyin_count_mismatch")
    cjk_word_indices = [i for i, iv in enumerate(hanzi.intervals) if _CJK.fullmatch(iv.text.strip())]
    if len(cjk_word_indices) != len(reference_cjk):
        reasons.append("hanzi_cjk_interval_count_mismatch")
    for index in cjk_word_indices:
        if not _PINYIN.fullmatch(words.intervals[index].text.strip()):
            reasons.append("cjk_without_toned_pinyin_word")
            break

    # Phones may only occupy their owning word; every English word requires
    # real en:-prefixed phones, never a self-referential lexical phone.
    for phone in phones.intervals:
        owners = [word for word in words.intervals
                  if phone.xmin >= word.xmin - EPS and phone.xmax <= word.xmax + EPS]
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
            if "unk" not in ref_english or not owned or not all(p.startswith("en:") for p in owned):
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


def _load_english_manifest(args: argparse.Namespace) -> tuple[dict | None, list[str]]:
    """Load and validate the global strict-en-mfa-v1 contract once."""
    try:
        manifest_path = _safe_file_under(args.en_phones_dir, str(args.en_manifest))
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, ["english_provenance_manifest_failed"]
    if raw.get("schema") != EN_PROVENANCE_SCHEMA or raw.get("strict_provenance") is not True:
        return None, ["english_provenance_legacy_schema"]
    if raw.get("status") not in {"success", "no_english"}:
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
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return None, ["english_provenance_manifest_failed"]
        rejected_ids.append(item["id"])
    if (len(rejected_ids) != len(set(rejected_ids)) or set(produced) & set(rejected_ids)
            or set(expected) != set(produced) | set(rejected_ids)):
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
    if raw["status"] == "success":
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
                                args: argparse.Namespace, global_manifest: dict | None) -> tuple[list[str], dict | None]:
    """Cross-check source MFA TextGrids against final en: phones and ledger."""
    final_words = [iv for iv in final_tg.tiers[3].intervals if is_english_token(iv.text.strip())]
    if not final_words:
        return [], None
    if global_manifest is None:
        return ["english_provenance_manifest_missing"], None
    if global_manifest.get("status") != "success":
        return ["english_provenance_manifest_failed"], None
    try:
        ledger_info = global_manifest["_ledger_by_stem"].get(stem)
        if ledger_info is None:
            raise KeyError("missing ledger")
        ledger_path = ledger_info["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("schema") != EN_PROVENANCE_SCHEMA or ledger.get("stem") != stem:
            return ["english_provenance_legacy_schema"], None
        ctc_path = ctc_dir / f"{stem}.TextGrid"
        if not ctc_path.is_file():
            ctc_path = ctc_dir / stem / f"{stem}.TextGrid"
        if not ctc_path.is_file() or _sha256(ctc_path) != ledger.get("ctc_textgrid_sha256"):
            return ["english_provenance_hash_mismatch"], None
        ctc_english = _ctc_english_words(ctc_path)
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
                expected_id = f"{sid}:w{record.get('ctc_ordinal')}"
                mfa_word = record.get("mfa_word")
                if (record.get("status") != "verified" or record.get("word_id") != expected_id
                        or not isinstance(mfa_word, dict)
                        or record.get("ctc_ordinal") not in ctc_english
                        or ctc_english[record.get("ctc_ordinal")].casefold() != record.get("ctc_text", "").casefold()
                        or record.get("ctc_text", "").casefold() != source_word["text"].casefold()
                        or mfa_word.get("ordinal") != source_word["ordinal"]
                        or not _same_number(mfa_word.get("start"), source_word["start"])
                        or not _same_number(mfa_word.get("end"), source_word["end"])
                        or record.get("provenance") != "english_mfa_textgrid"):
                    return ["english_word_unmatched"], None
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
                verified_words.append({"ledger": record, "source": source_word})
            sources.append(source_path); used_source_paths.add(source_path)
        verified_words.sort(key=lambda item: item["ledger"].get("ctc_ordinal", -1))
        source_ordinals = [item["ledger"].get("ctc_ordinal") for item in verified_words]
        if source_ordinals != sorted(ctc_english) or len(source_ordinals) != len(ctc_english):
            return ["english_word_unmatched"], None
        # The reference transcript can preserve a contiguous English spelling
        # as one word while CTC/MFA tokenization splits it into several
        # verified words (for example ``Sila`` -> ``S`` + ``il`` + ``a``).
        # Group only exact ordered concatenations so every source word and
        # phone remains accounted for in the independent audit.
        grouped_verified: list[dict] = []
        cursor = 0
        for final_word in final_words:
            target = final_word.text.strip().casefold()
            joined = ""
            matched_end = None
            for candidate_end in range(cursor + 1, len(verified_words) + 1):
                item = verified_words[candidate_end - 1]
                ordinal = item["ledger"].get("ctc_ordinal")
                joined += str(ctc_english.get(ordinal, ""))
                if joined.casefold() == target:
                    matched_end = candidate_end
                    break
                if not target.startswith(joined.casefold()):
                    break
            if matched_end is None:
                return ["english_word_unmatched"], None
            chunk = verified_words[cursor:matched_end]
            if len(chunk) == 1:
                grouped_verified.append(chunk[0])
            else:
                first = chunk[0]
                last = chunk[-1]
                combined_ledger = dict(first["ledger"])
                combined_ledger["ctc_text"] = final_word.text.strip()
                combined_source = dict(first["source"])
                combined_source["text"] = final_word.text.strip()
                combined_source["start"] = first["source"]["start"]
                combined_source["end"] = last["source"]["end"]
                combined_source["phones"] = [
                    phone
                    for item in chunk
                    for phone in item["source"]["phones"]
                ]
                grouped_verified.append({"ledger": combined_ledger,
                                         "source": combined_source})
            cursor = matched_end
        if cursor != len(verified_words):
            return ["english_word_unmatched"], None
        verified_words = grouped_verified
        final_phones = final_tg.tiers[4].intervals
        matched_en_phone_indices: set[int] = set()
        for final_word, evidence in zip(final_words, verified_words):
            record, source_word = evidence["ledger"], evidence["source"]
            ordinal = record.get("ctc_ordinal")
            if final_word.text.strip().casefold() != record.get("ctc_text", "").casefold():
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
    if pipeline_receipt is not None and ctc_stems != expected:
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
        "expected_stems": sorted(expected),
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
        reference = (reference_path.read_text(encoding="utf-8").strip()
                     if reference_path is not None else "")
        if reference_path is None:
            reasons.append("non_authoritative_reference")
        wav = args.wav_dir / f"{stem}.wav"
        aligned = args.aligned_dir / f"{stem}.TextGrid"
        if not wav.is_file():
            reasons.append("missing_wav")
        if not aligned.is_file():
            reasons.append("missing_aligned")
        provenance_evidence = None
        provenance_reasons: list[str] = []
        if not reasons:
            try:
                tg = _strict_parse(candidate)
                reasons.extend(_numeric_reasons(tg, _wav_duration(wav)))
                reasons.extend(_sp1_reasons(tg))
                reasons.extend(_content_reasons(tg, reference))
                if _replay_mode:
                    provenance_reasons, provenance_evidence = [], None
                else:
                    provenance_reasons, provenance_evidence = _english_provenance_reasons(
                        stem, tg, ctc_dir, args, english_manifest)
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
            reasons.extend(aligned_rejection_reasons)
        row = report_rows.get(stem)
        if row is None:
            reasons.append("missing_report_row")
        else:
            reasons.extend(_report_reasons(row))
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
                "reference": {"path": str(reference_path.resolve()), "sha256": _sha256(reference_path)},
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
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
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
