#!/usr/bin/env python3
"""
English MFA alignment — extract English segments from CTC TextGrids and align
with the english_us_arpa acoustic model (ARPABET phone set) for phoneme-level
boundaries.

Inputs:
  workspace/ctc_pretg_adj/  (or ctc_pretg)  — CTC TextGrids + .lab files
  workspace/audio_16k/                       — 16kHz mono audio
  pretrained_models/acoustic/english_us_arpa.zip
  dict/cmudict.dict
  pretrained_models/g2p/english_us_arpa.zip

Outputs:
  workspace/en_phones/{stem}_en_phones.json  — English phoneme alignments

Usage:
  python scripts/align_english_mfa.py \
      --ctc-dir workspace/ctc_pretg_adj \
      --audio-dir workspace/audio_16k \
      --output-dir workspace/en_phones \
      --acoustic-model pretrained_models/acoustic/english_us_arpa.zip \
      --dictionary dict/cmudict.dict \
      --g2p-model pretrained_models/g2p/english_us_arpa.zip \
      --temp-dir workspace/temp_en \
      --num-jobs 4
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (
    is_english_token, is_nvv_token, is_silence, SILENCE_LABELS,
    find_mfa_python, get_mfa_env, resolve_mfa_dither,
    is_english_phone as is_arpabet_phone,
    report_en_ipa_mappings, load_ctc_token_entries,
    CTC_PROCESSED_BOUNDARY_SOURCES,
)
from english_units import (
    EnglishUnit,
    EnglishUnitError,
    canonicalize_english_token,
    is_english_fragment_token,
    merge_authority_fragment_group,
    parse_english_units,
)

# English MFA phone inventory — vowels, consonants, and stress markers
_ENGLISH_VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
    'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX',
    'AA0', 'AE0', 'AH0', 'AO0', 'AW0', 'AX0', 'AY0',
    'EH0', 'ER0', 'EY0', 'IH0', 'IX0', 'IY0', 'OW0', 'OY0', 'UH0', 'UW0',
    'AA1', 'AE1', 'AH1', 'AO1', 'AW1', 'AY1',
    'EH1', 'ER1', 'EY1', 'IH1', 'IY1', 'OW1', 'OY1', 'UH1', 'UW1',
    'AA2', 'AE2', 'AH2', 'AO2', 'AW2', 'AY2',
    'EH2', 'ER2', 'EY2', 'IH2', 'IY2', 'OW2', 'OY2', 'UH2', 'UW2',
}
_ENGLISH_CONSONANTS = {
    'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'ENG', 'F', 'G',
    'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 'S',
    'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH',
}
_ENGLISH_SILENCE = {'sil', 'sp', 'spn', '<eps>'}

STRICT_SCHEMA = "strict-en-mfa-v2"
HISTORICAL_STRICT_SCHEMA = "strict-en-mfa-v1"
CANONICAL_UNITS_SCHEMA = "canonical-english-units-v1"
SOS_PRONUNCIATION_POLICY_ID = "sos-exact-override-v1"
SOS_ALIGNMENT_TOKEN = "sos"
SOS_SURFACE_TEXT = "SOS"
SOS_EXPECTED_PRONUNCIATION = ("EH2", "S", "OW2", "EH1", "S")
SOS_BASE_PRONUNCIATION = ("EH2", "OW2", "EH1", "S")
APP_ALIGNMENT_TOKEN = "app"
APP_EXPECTED_PRONUNCIATION = ("AE1", "P")
STRICT_COUNT_KEYS = (
    "english_stems", "english_segments", "english_words",
    "verified_words", "rejected_words",
)
# MFA writes times in seconds.  This is deliberately small enough that an
# unowned/cross-word phone cannot be hidden by a generous boundary allowance.
STRICT_BOUNDARY_TOLERANCE_S = 0.003


class StrictG2PError(RuntimeError):
    """Raised when strict provenance cannot construct a complete dictionary."""


def _dictionary_entries(text: str, token: str) -> list[tuple[str, ...]]:
    """Return pronunciations for one exact CMUdict token.

    CMU pronunciation variants (``WORD(1)``) are grouped with their base
    token, while words that merely contain the token are never considered.
    """
    entries: list[tuple[str, ...]] = []
    wanted = token.casefold()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].split("(", 1)[0].casefold()
        if key == wanted:
            entries.append(tuple(parts[1:]))
    return entries


def _replace_exact_dictionary_entry(text: str, token: str,
                                    pronunciation: tuple[str, ...]) -> str:
    """Replace all exact variants of *token* in a run-local dictionary."""
    wanted = token.casefold()
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].split("(", 1)[0].casefold() == wanted:
            continue
        lines.append(line.rstrip())
    lines.append(f"{token.upper()} {' '.join(pronunciation)}")
    return "\n".join(lines) + "\n"


def _validate_exact_dictionary_word(text: str, token: str,
                                    expected: tuple[str, ...], *,
                                    error_code: str) -> None:
    entries = _dictionary_entries(text, token)
    if len(entries) != 1 or entries[0] != expected:
        raise StrictG2PError(error_code)


def _validate_dictionary_provenance(provenance: dict, *,
                                    expected_sos: bool = True) -> None:
    """Validate the existing dictionary path/hash before SOS is published."""
    if not isinstance(provenance, dict):
        raise ValueError("sos_dictionary_provenance_missing")
    path_text = provenance.get("path")
    expected_hash = provenance.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        raise ValueError("sos_dictionary_provenance_missing")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError("sos_dictionary_hash_missing")
    path = Path(path_text)
    try:
        actual_hash = _provenance_sha256(path)
    except Exception as exc:
        raise ValueError("sos_dictionary_provenance_unreadable") from exc
    if actual_hash != expected_hash:
        raise ValueError("sos_dictionary_hash_mismatch")
    if expected_sos:
        try:
            dictionary_text = _read_dict(path)
        except Exception as exc:
            raise ValueError("sos_dictionary_provenance_unreadable") from exc
        if _dictionary_entries(dictionary_text, SOS_ALIGNMENT_TOKEN) != [SOS_EXPECTED_PRONUNCIATION]:
            raise ValueError("sos_dictionary_override_missing_or_tampered")


def validate_sos_pronunciation_record(
    actual_source_sequence: list[str] | tuple[str, ...],
    policy: dict,
    dictionary_provenance: dict,
) -> dict:
    """Validate and return one exact SOS override provenance record."""
    actual = tuple(actual_source_sequence)
    if actual != SOS_EXPECTED_PRONUNCIATION:
        raise ValueError("sos_expected_pronunciation_mismatch")
    if not isinstance(policy, dict):
        raise ValueError("sos_pronunciation_policy_missing")
    if policy.get("policy_id") != SOS_PRONUNCIATION_POLICY_ID:
        raise ValueError("sos_pronunciation_policy_id_mismatch")
    if tuple(policy.get("expected_pronunciation", ())) != SOS_EXPECTED_PRONUNCIATION:
        raise ValueError("sos_expected_pronunciation_mismatch")
    if tuple(policy.get("actual_source_sequence", ())) != actual:
        raise ValueError("sos_actual_source_sequence_mismatch")
    if policy.get("dictionary_provenance") != dictionary_provenance:
        raise ValueError("sos_dictionary_provenance_mismatch")
    _validate_dictionary_provenance(dictionary_provenance)
    return {
        "policy_id": SOS_PRONUNCIATION_POLICY_ID,
        "expected_pronunciation": list(SOS_EXPECTED_PRONUNCIATION),
        "actual_source_sequence": list(actual),
        "dictionary_provenance": dict(dictionary_provenance),
    }


def _sos_policy_record(actual_source_sequence: list[str] | tuple[str, ...],
                       dictionary_provenance: dict) -> dict:
    """Construct a validated SOS policy record for a strict ledger word."""
    policy = {
        "policy_id": SOS_PRONUNCIATION_POLICY_ID,
        "expected_pronunciation": list(SOS_EXPECTED_PRONUNCIATION),
        "actual_source_sequence": list(actual_source_sequence),
        "dictionary_provenance": dict(dictionary_provenance),
    }
    return validate_sos_pronunciation_record(
        actual_source_sequence, policy, dictionary_provenance)


def _unit_dict(unit: EnglishUnit, *, reference_span=None) -> dict:
    """Serialize one immutable Wave 1 unit and its binding metadata."""
    data = unit.to_dict()
    data["canonical_binding"] = CANONICAL_UNITS_SCHEMA
    if reference_span is not None:
        data["reference_span"] = list(reference_span)
    return data


def _validated_unit(word: dict) -> EnglishUnit:
    """Validate the canonical unit attached to a corpus/ledger word.

    The producer never reconstructs a unit from a display word.  This is the
    tamper boundary: a changed surface, token, ID, span, or source ordinal is
    rejected before dictionary, corpus, MFA, or ledger work.
    """
    raw = word.get("canonical_unit")
    if not isinstance(raw, dict) or raw.get("canonical_binding") != CANONICAL_UNITS_SCHEMA:
        raise EnglishUnitError("canonical_unit_binding_missing")
    try:
        surface = raw["surface_text"]
        # ``parse_english_units`` parses an isolated surface, so its local
        # ordinal necessarily starts at zero.  It is semantic evidence only:
        # validate surface/token/match_key from that parse, while identity is
        # bound to the raw reference ordinal carried by the producer record.
        parsed_units = parse_english_units(surface)
        ordinal = raw["reference_ordinal"]
        if (not isinstance(ordinal, int) or isinstance(ordinal, bool)
                or ordinal < 0 or len(parsed_units) != 1):
            raise EnglishUnitError("canonical_unit_identity_invalid")
        parsed = parsed_units[0]
        expected_unit_id = f"en-u{ordinal:04d}"
        if (raw.get("unit_id") != expected_unit_id
                or raw.get("alignment_token") != parsed.alignment_token
                or raw.get("match_key") != parsed.match_key):
            raise EnglishUnitError("canonical_unit_identity_tampered")
        source = raw.get("source_ctc_ordinals")
        if (not isinstance(source, list)
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                       for value in source)
                or tuple(source) != tuple(sorted(set(source)))):
            raise EnglishUnitError("canonical_unit_source_ordinals_invalid")
        span = raw.get("canonical_span")
        if not isinstance(span, list) or len(span) != 2:
            raise EnglishUnitError("canonical_unit_span_invalid")
        start, end = span
        if ((start is None) != (end is None)
                or (start is not None and (
                    not isinstance(start, (int, float)) or isinstance(start, bool)
                    or not isinstance(end, (int, float)) or isinstance(end, bool)
                    or end < start))):
            raise EnglishUnitError("canonical_unit_span_invalid")
        if (word.get("unit_id") != expected_unit_id
                or word.get("alignment_token") != parsed.alignment_token):
            raise EnglishUnitError("canonical_word_identity_tampered")
        if tuple(word.get("source_ctc_ordinals", ())) != tuple(source):
            raise EnglishUnitError("canonical_word_source_ordinals_tampered")
        if word.get("canonical_span") != span:
            raise EnglishUnitError("canonical_word_span_tampered")
        if word.get("text") != parsed.surface_text:
            raise EnglishUnitError("canonical_word_surface_tampered")
        return EnglishUnit(
            surface_text=parsed.surface_text,
            alignment_token=parsed.alignment_token,
            match_key=parsed.match_key,
            unit_id=expected_unit_id,
            reference_ordinal=ordinal,
            source_ctc_ordinals=tuple(source),
            merge_kind=raw.get("merge_kind", parsed.merge_kind),
            canonical_start=start,
            canonical_end=end,
        )
    except (KeyError, TypeError, ValueError, EnglishUnitError) as exc:
        if isinstance(exc, EnglishUnitError):
            raise
        raise EnglishUnitError("canonical_unit_malformed") from exc


def _reference_text_for_stem(ctc_dir: Path, stem: str, reference_dir: Path | None = None) -> str | None:
    """Read the optional CTC reference sidecar without broad file guessing."""
    roots = [reference_dir] if reference_dir is not None else [ctc_dir]
    candidates = [root / f"{stem}_ref.txt" for root in roots]
    candidates += [root / f"{stem}.ref.txt" for root in roots]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def _source_english_fragments(intervals: list[dict]) -> list[dict]:
    """Return English source intervals, retaining full words-tier ordinals."""
    result = []
    for index, interval in enumerate(intervals):
        text = str(interval.get("text", "")).strip()
        if not text or text in SILENCE_LABELS or text == "<eps>":
            continue
        if not is_english_fragment_token(text):
            continue
        result.append({"text": text, "ordinal": int(interval.get("ordinal", index)),
                       "start": float(interval["xmin"]), "end": float(interval["xmax"])})
    return result


def _processed_geometry_reason(word: dict, *, require: bool = False) -> str | None:
    """Return a stable fail-closed reason for the processed/canonical axes."""
    if "processed_ctc_span" not in word:
        return "processed_geometry_missing" if require else None
    canonical_span = word.get("canonical_span")
    processed_span = word.get("processed_ctc_span")
    if (not isinstance(canonical_span, (list, tuple))
            or len(canonical_span) != 2
            or not all(isinstance(value, (int, float))
                       and not isinstance(value, bool)
                       and math.isfinite(float(value))
                       for value in canonical_span)
            or float(canonical_span[1]) <= float(canonical_span[0])):
        return "processed_geometry_canonical_invalid"
    if (not isinstance(processed_span, (list, tuple))
            or len(processed_span) != 2):
        return "processed_geometry_missing"
    if (not all(isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in processed_span)):
        return "processed_geometry_non_numeric"
    processed_start, processed_end = map(float, processed_span)
    canonical_start, canonical_end = map(float, canonical_span)
    if processed_end <= processed_start:
        return "processed_geometry_non_positive"
    if abs(processed_start - canonical_start) > STRICT_BOUNDARY_TOLERANCE_S + 1e-9:
        return "processed_geometry_start_mismatch"
    if processed_end + STRICT_BOUNDARY_TOLERANCE_S < canonical_end:
        return "processed_geometry_end_before_canonical_end"
    return None


def _canonicalize_source_units(intervals: list[dict], reference_text: str | None) -> tuple[dict, ...]:
    """Bind CTC English fragments to exact canonical authority units.

    A missing reference is supported by making each valid source token its
    own direct authority unit.  When a reference exists, every lexical source
    token must be consumed by exactly one ordered authority unit; partial,
    extra, split, and reordered inputs raise instead of being guessed.
    """
    fragments = _source_english_fragments(intervals)
    if not fragments:
        return ()
    if not reference_text:
        units = []
        for fragment in fragments:
            authority = parse_english_units(fragment["text"])
            if len(authority) != 1:
                raise EnglishUnitError("source_unit_not_canonical")
            units.append({"unit": merge_authority_fragment_group(authority[0], [fragment]),
                          "reference_span": authority[0].canonical_span,
                          "fragments": [fragment]})
        return tuple(units)

    authorities = parse_english_units(reference_text)
    if not authorities:
        raise EnglishUnitError("reference_has_no_english_units")
    groups = []
    cursor = 0
    for authority in authorities:
        if cursor >= len(fragments):
            raise EnglishUnitError("source_unit_missing")
        matched = None
        for end in range(cursor + 1, len(fragments) + 1):
            group = fragments[cursor:end]
            try:
                merged = merge_authority_fragment_group(authority, group)
            except EnglishUnitError:
                continue
            matched = (merged, group, end)
            break
        if matched is None:
            raise EnglishUnitError("source_unit_not_exact")
        merged, group, cursor = matched
        groups.append((merged, group, authority.canonical_span))
    if cursor != len(fragments):
        raise EnglishUnitError("source_unit_extra")
    return tuple(
        {"unit": merged, "reference_span": reference_span, "fragments": group}
        for merged, group, reference_span in groups
    )


def _canonical_units_from_tokens(
        intervals: list[dict], token_rows: list[dict],
        reference_text: str | None) -> tuple[dict, ...] | None:
    """Bind canonical English identity and geometry to the token authority.

    A token sidecar containing canonical metadata is authoritative.  The
    TextGrid is accepted only as a geometry mirror; it cannot recreate a
    missing or inconsistent processed span from its shorter raw interval.
    ``None`` means the sidecar is legacy/non-authority and the historical
    TextGrid path remains in effect.
    """
    lexical_intervals = [item for item in intervals
                         if str(item.get("text", "")).strip()]
    canonical_rows = [row for row in token_rows
                      if isinstance(row.get("canonical_unit"), dict)]
    if not canonical_rows:
        return None
    if len(lexical_intervals) != len(token_rows):
        raise EnglishUnitError("canonical_token_textgrid_owner_mismatch")

    reference_identity = (
        hashlib.sha256(reference_text.encode("utf-8")).hexdigest()
        if isinstance(reference_text, str) else None)
    result = []
    for interval, token in zip(lexical_intervals, token_rows):
        unit = token.get("canonical_unit")
        if not isinstance(unit, dict):
            continue
        if "processed_ctc_span" in unit or "processed_ctc_boundary_source" in unit:
            raise EnglishUnitError("canonical_processed_geometry_owner_conflict")
        surface = token.get("surface_text")
        if not isinstance(surface, str) or not surface:
            raise EnglishUnitError("canonical_token_surface_missing")
        if str(interval.get("text", "")).strip() != str(token.get("word", "")).strip():
            raise EnglishUnitError("canonical_token_textgrid_text_mismatch")
        raw_span = token.get("canonical_span")
        unit_span = unit.get("canonical_span")
        processed_span = token.get("processed_ctc_span")
        if (not isinstance(raw_span, list) or len(raw_span) != 2
                or raw_span != unit_span):
            raise EnglishUnitError("canonical_token_raw_span_mismatch")
        geometry_reason = _processed_geometry_reason(
            {"canonical_span": raw_span, "processed_ctc_span": processed_span},
            require=True)
        if geometry_reason is not None:
            raise EnglishUnitError(geometry_reason)
        source = token.get("processed_ctc_boundary_source")
        if not isinstance(source, str) or source not in CTC_PROCESSED_BOUNDARY_SOURCES:
            raise EnglishUnitError("processed_ctc_boundary_source_missing")
        encoded = json.dumps(unit, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if token.get("canonical_unit_sha256") != hashlib.sha256(encoded).hexdigest():
            raise EnglishUnitError("canonical_token_hash_mismatch")
        if reference_identity is not None and token.get("reference_identity") != reference_identity:
            raise EnglishUnitError("canonical_token_reference_owner_mismatch")

        word = {
            "text": surface,
            "alignment_token": unit.get("alignment_token"),
            "unit_id": unit.get("unit_id"),
            "source_ctc_ordinals": token.get("source_ctc_ordinals"),
            "canonical_span": list(raw_span),
            "canonical_unit": dict(unit),
            "processed_ctc_span": list(processed_span),
            "processed_ctc_boundary_source": source,
            "start": float(processed_span[0]),
            "end": float(processed_span[1]),
            "ordinal": int(interval.get("ordinal", 0)),
        }
        _validated_unit(word)
        if (abs(float(interval["xmin"]) - word["start"]) > STRICT_BOUNDARY_TOLERANCE_S
                or abs(float(interval["xmax"]) - word["end"]) > STRICT_BOUNDARY_TOLERANCE_S):
            raise EnglishUnitError("canonical_token_textgrid_geometry_mismatch")
        result.append({"unit": _validated_unit(word), "reference_span": unit.get("reference_span"),
                       "fragments": [{"ordinal": word["ordinal"],
                                      "start": word["start"], "end": word["end"],
                                      "text": token.get("word", "")}],
                       "token_word": word})
    return tuple(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_sha256(path: Path) -> str:
    """Return a content hash for a file or a deterministic model directory tree."""
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(f"provenance path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_acoustic_model(acoustic_model: str, models_dir: Path) -> Path:
    """Resolve the exact model argument MFA will receive."""
    extracted = models_dir / "extracted_models" / "acoustic" / "english_us_arpa_acoustic"
    return extracted if extracted.is_dir() else Path(acoustic_model)


def _strict_expected_snapshot(en_segments: dict[str, list[dict]]) -> tuple[list[str], dict]:
    """Freeze the English denominator before any corpus-side mutation."""
    expected: list[str] = []
    english_words = 0
    for stem, segments in sorted(en_segments.items()):
        for seg in segments:
            ordinal = int(seg.get("segment_ordinal", seg.get("seg_idx", 0)))
            expected.append(f"{stem}:s{ordinal}")
            english_words += len(seg["words"])
    return expected, {"english_stems": len(en_segments), "english_segments": len(expected),
                      "english_words": english_words, "verified_words": 0,
                      "rejected_words": 0}


def _strict_counts(counts: Optional[dict] = None) -> dict:
    """Keep the manifest count contract stable in every success/failure path."""
    supplied = counts or {}
    return {key: int(supplied.get(key, 0)) for key in STRICT_COUNT_KEYS}


def _strict_failed_manifest(output_dir: Path, *, mfa: dict,
                            expected_segments: list[str], expected_counts: dict,
                            en_segments: dict[str, list[dict]] | None = None,
                            reason: str = "strict_mfa_failed") -> Path:
    """Write a complete denominator when strict processing stops before ledgers."""
    reasons: dict[str, str] = {}
    if en_segments:
        for stem, segments in en_segments.items():
            for seg in segments:
                sid = f"{stem}:s{int(seg.get('segment_ordinal', seg.get('seg_idx', 0)))}"
                if seg.get("reject_reason"):
                    reasons[sid] = str(seg["reject_reason"])
    rejected = [{"id": sid, "reason": reasons.get(sid, reason)} for sid in expected_segments]
    counts = dict(expected_counts)
    counts["verified_words"] = 0
    counts["rejected_words"] = counts["english_words"]
    return write_strict_manifest(output_dir, "failed", mfa=mfa,
                                 expected_segments=expected_segments,
                                 produced_segments=[], rejected_segments=rejected,
                                 stem_ledgers=[], counts=counts, reason=reason)


def _strict_mfa_record(outcome: dict, acoustic_model: Path, dictionary: Path) -> dict:
    """Fill the immutable MFA provenance fields on every strict outcome."""
    record = dict(outcome)
    record.setdefault("command", [])
    record.setdefault("return_code", "not_run")
    record.setdefault("timed_out", False)
    record.setdefault("timeout_seconds", 0)
    record.setdefault("exception", "")
    if "acoustic_model_sha256" not in record:
        try:
            record["acoustic_model_sha256"] = _provenance_sha256(acoustic_model)
        except Exception as exc:
            record["acoustic_model_sha256"] = ""
            record["exception"] = (record["exception"] + "; " if record["exception"] else "") + \
                                  f"acoustic model hashing failed: {exc}"
    if "dictionary_sha256" not in record:
        try:
            record["dictionary_sha256"] = _provenance_sha256(dictionary)
        except Exception as exc:
            record["dictionary_sha256"] = ""
            record["exception"] = (record["exception"] + "; " if record["exception"] else "") + \
                                  f"dictionary hashing failed: {exc}"
    record.setdefault("dictionary", str(dictionary))
    return record


def _strict_tiers(path: Path) -> tuple[list[dict], list[dict]]:
    """Read named words/phones tiers; positional tier parsing is forbidden."""
    from postprocess_textgrids import parse_textgrid
    tg = parse_textgrid(path)
    words_tiers = [tier for tier in tg.tiers if tier.name == "words"]
    phones_tiers = [tier for tier in tg.tiers if tier.name == "phones"]
    if len(words_tiers) != 1 or len(phones_tiers) != 1:
        raise ValueError("words_phones_tier_missing_or_duplicate")

    def conv(iv, ordinal: int) -> dict:
        return {"ordinal": ordinal, "text": iv.text.strip(), "start": iv.xmin, "end": iv.xmax}

    return ([conv(iv, ordinal) for ordinal, iv in enumerate(words_tiers[0].intervals)],
            [conv(iv, ordinal) for ordinal, iv in enumerate(phones_tiers[0].intervals)])


def _strict_rejected_words(sid: str, segment: dict, reason: str) -> list[dict]:
    """Keep stable CTC-derived IDs even when source evidence is unusable."""
    records = []
    for position, word in enumerate(segment["words"]):
        source_ordinals = list(word.get("source_ctc_ordinals", [word.get("ordinal", position)]))
        unit_id = word.get("unit_id")
        records.append({
            "word_id": f"{sid}:w{position}",
            "unit_id": unit_id,
            "ctc_ordinal": source_ordinals[0] if source_ordinals else word.get("ordinal", position),
            "source_ctc_ordinals": source_ordinals,
            "ctc_text": word.get("text", ""),
            "alignment_token": word.get("alignment_token"),
            "canonical_span": word.get("canonical_span"),
            "canonical_binding": CANONICAL_UNITS_SCHEMA,
            "start": word.get("start"), "end": word.get("end"),
            "processed_ctc_span": word.get("processed_ctc_span"),
            "processed_ctc_boundary_source": word.get(
                "processed_ctc_boundary_source"),
            "status": "rejected", "reason": reason, "mfa_word": None,
            "phones": [], "provenance": None,
        })
    return records
def _strict_source_words(words: list[dict]) -> list[dict]:
    """Return lexical MFA words after proving their temporal ordering."""
    lexical = [word for word in words if word["text"] and not is_silence(word["text"])]
    last_end = -math.inf
    for word in lexical:
        start, end = word["start"], word["end"]
        if not all(math.isfinite(value) for value in (start, end)) or end <= start:
            raise ValueError("word_invalid")
        if start < last_end:
            raise ValueError("word_overlap_or_unordered")
        last_end = end
    return lexical


def _strict_phone_silence(text: str) -> bool:
    """MFA's bare ``sp`` is silence even though CTC does not use that label."""
    return is_silence(text) or text in _ENGLISH_SILENCE


def _strict_verified_words(sid: str, segment: dict, words: list[dict], phones: list[dict],
                           dictionary_provenance: dict | None = None) -> list[dict]:
    """Validate the complete source hierarchy and construct verified evidence."""
    source_words = _strict_source_words(words)
    ctc_words = segment["words"]
    if len(source_words) != len(ctc_words):
        raise ValueError("word_count_mismatch")
    canonical_units = []
    for ctc_word in ctc_words:
        canonical_units.append(_validated_unit(ctc_word))
        geometry_reason = _processed_geometry_reason(ctc_word)
        if geometry_reason is not None:
            raise ValueError(geometry_reason)
    for ctc_word, source_word, unit in zip(ctc_words, source_words, canonical_units):
        try:
            source_token = canonicalize_english_token(source_word["text"])
        except EnglishUnitError as exc:
            raise ValueError("mfa_word_not_canonical") from exc
        if source_token != unit.alignment_token:
            raise ValueError("word_text_order_mismatch")

    lexical_phones = [phone for phone in phones if not _strict_phone_silence(phone["text"])]
    last_end = -math.inf
    owned: list[list[dict]] = [[] for _ in source_words]
    for phone in lexical_phones:
        start, end = phone["start"], phone["end"]
        if not all(math.isfinite(value) for value in (start, end)) or end <= start:
            raise ValueError("phone_invalid")
        if start < last_end:
            raise ValueError("phone_overlap_or_unordered")
        last_end = end
        if not is_arpabet_phone(phone["text"]) or phone["text"] in _ENGLISH_SILENCE:
            raise ValueError("phone_unknown")
        owners = [index for index, word in enumerate(source_words)
                  if start >= word["start"] - STRICT_BOUNDARY_TOLERANCE_S
                  and end <= word["end"] + STRICT_BOUNDARY_TOLERANCE_S]
        if len(owners) != 1:
            raise ValueError("phone_outside_or_cross_word")
        owned[owners[0]].append(phone)

    evidence: list[dict] = []
    for ctc_word, source_word, word_phones, unit in zip(ctc_words, source_words, owned, canonical_units):
        if not word_phones:
            raise ValueError("phone_empty")
        if abs(word_phones[0]["start"] - source_word["start"]) > STRICT_BOUNDARY_TOLERANCE_S:
            raise ValueError("phone_start_coverage")
        if abs(word_phones[-1]["end"] - source_word["end"]) > STRICT_BOUNDARY_TOLERANCE_S:
            raise ValueError("phone_end_coverage")
        for previous, current in zip(word_phones, word_phones[1:]):
            if current["start"] - previous["end"] > STRICT_BOUNDARY_TOLERANCE_S:
                raise ValueError("phone_gap")
        dictionary = dictionary_provenance or {}
        actual_source_sequence = tuple(phone["text"] for phone in word_phones)
        if unit.alignment_token == APP_ALIGNMENT_TOKEN and actual_source_sequence != APP_EXPECTED_PRONUNCIATION:
            raise ValueError("app_expected_pronunciation_mismatch")
        pronunciation_policy = None
        if unit.alignment_token == SOS_ALIGNMENT_TOKEN:
            pronunciation_policy = _sos_policy_record(
                actual_source_sequence, dictionary)
        item = {"word_id": f"{sid}:w{ctc_word['ordinal']}",
                "unit_id": unit.unit_id,
                "ctc_ordinal": ctc_word["source_ctc_ordinals"][0],
                "source_ctc_ordinals": list(unit.source_ctc_ordinals),
                "ctc_text": ctc_word["text"],
                "alignment_token": unit.alignment_token,
                "canonical_span": list(unit.canonical_span),
                "canonical_binding": CANONICAL_UNITS_SCHEMA,
                "start": ctc_word["start"], "end": ctc_word["end"],
                "processed_ctc_span": ctc_word.get("processed_ctc_span",
                                                     [ctc_word["start"], ctc_word["end"]]),
                "processed_ctc_boundary_source": ctc_word.get(
                    "processed_ctc_boundary_source", "legacy_ctc"),
                "status": "verified", "reason": "",
                "mfa_word": {"ordinal": source_word["ordinal"], "text": source_word["text"],
                             "start": source_word["start"], "end": source_word["end"]},
                "phones": [{"ordinal": position, "label": phone["text"], "start": phone["start"],
                            "end": phone["end"], "mfa_phone_ordinal": phone["ordinal"]}
                           for position, phone in enumerate(word_phones)],
                "provenance": "english_mfa_textgrid",
                "dictionary_provenance": dictionary}
        if pronunciation_policy is not None:
            item["pronunciation_policy"] = pronunciation_policy
            item["pronunciation_policy_id"] = SOS_PRONUNCIATION_POLICY_ID
        evidence.append(item)
    return evidence


def _strict_source_path(aligned_root: Path, seg_name: str) -> Path:
    """Resolve the only permitted MFA output layout without following escapes."""
    candidates = [aligned_root / f"{seg_name}.TextGrid",
                  aligned_root / seg_name / f"{seg_name}.TextGrid"]
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise ValueError("source_tg_missing_or_ambiguous")
    resolved = existing[0].resolve(strict=True)
    if resolved != aligned_root and aligned_root not in resolved.parents:
        raise ValueError("source_path_escapes_aligned_root")
    return resolved


def _strict_manifest_consistent(expected: list[str], produced: list[str], rejected: list[dict],
                                ledgers: list[dict], counts: dict) -> bool:
    """Check the global denominator before a producer may claim success/partial."""
    rejected_ids = [item.get("id") for item in rejected]
    if (len(expected) != len(set(expected))
            or len(produced) != len(set(produced))
            or len(rejected_ids) != len(set(rejected_ids))
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("id"), str)
                   or not isinstance(item.get("reason"), str)
                   or not item.get("reason")
                   for item in rejected)):
        return False
    if set(produced) & set(rejected_ids):
        return False
    if set(expected) != set(produced) | set(rejected_ids):
        return False
    expected_stems = {sid.split(":s", 1)[0] for sid in expected}
    if len(ledgers) != len(expected_stems) or {item.get("stem") for item in ledgers} != expected_stems:
        return False
    for item in ledgers:
        try:
            path = Path(item["path"])
            if not path.is_file() or _sha256(path) != item.get("sha256"):
                return False
        except Exception:
            return False
    return (counts["english_stems"] == len(expected_stems)
            and counts["english_segments"] == len(expected)
            and counts["verified_words"] + counts["rejected_words"] == counts["english_words"])


def write_strict_manifest(output_dir: Path, status: str, *, mfa: dict,
                          expected_segments: list[str], produced_segments: list[str],
                          rejected_segments: list[dict], stem_ledgers: list[dict],
                          counts: dict, reason: str = "") -> Path:
    payload = {"schema": STRICT_SCHEMA, "historical_schema": HISTORICAL_STRICT_SCHEMA,
               "canonical_units": CANONICAL_UNITS_SCHEMA,
               "status": status, "strict_provenance": True,
               "mfa": mfa, "expected_segments": expected_segments,
               "produced_segments": produced_segments, "rejected_segments": rejected_segments,
               "stem_ledgers": stem_ledgers, "counts": _strict_counts(counts), "reason": reason}
    path = output_dir / "en_alignment_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _strict_manifest_succeeded(path: Path) -> bool:
    """Accept a complete strict manifest, including explicit partial output."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if (payload.get("schema") != STRICT_SCHEMA
            or payload.get("status") not in {"success", "partial"}
            or payload.get("strict_provenance") is not True):
        return False
    expected = payload.get("expected_segments")
    produced = payload.get("produced_segments")
    rejected = payload.get("rejected_segments")
    ledgers = payload.get("stem_ledgers")
    counts = payload.get("counts")
    if (not isinstance(expected, list) or not isinstance(produced, list)
            or not isinstance(rejected, list) or not isinstance(ledgers, list)
            or not isinstance(counts, dict)):
        return False
    rejected_ids = [item.get("id") for item in rejected
                    if isinstance(item, dict)]
    if (len(rejected_ids) != len(rejected)
            or any(not isinstance(item.get("reason"), str) or not item["reason"]
                   for item in rejected if isinstance(item, dict))
            or len(expected) != len(set(expected))
            or len(produced) != len(set(produced))
            or len(rejected_ids) != len(set(rejected_ids))
            or set(produced) & set(rejected_ids)
            or set(expected) != set(produced) | set(rejected_ids)):
        return False
    expected_stems = {item.split(":s", 1)[0] for item in expected
                      if isinstance(item, str) and ":s" in item}
    if (len(ledgers) != len(expected_stems)
            or {item.get("stem") for item in ledgers if isinstance(item, dict)} != expected_stems):
        return False
    for item in ledgers:
        try:
            ledger_path = Path(item["path"])
            if (not ledger_path.is_file()
                    or _sha256(ledger_path) != item.get("sha256")):
                return False
        except Exception:
            return False
    if payload.get("status") == "success" and rejected_ids:
        return False
    if payload.get("status") == "partial" and not rejected_ids:
        return False
    return (counts.get("english_segments") == len(expected)
            and counts.get("verified_words", 0) + counts.get("rejected_words", 0)
            == counts.get("english_words", -1))


def produce_strict_ledgers(en_segments: dict[str, list[dict]], ctc_dir: Path,
                           aligned_dir: Path, output_dir: Path, mfa: dict,
                           expected_segments: Optional[list[str]] = None,
                           expected_counts: Optional[dict] = None) -> Path:
    """Produce fail-closed per-stem evidence from successful English MFA TGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered_expected: list[str] = []; produced: list[str] = []; rejected: list[dict] = []; ledgers: list[dict] = []
    verified = rejected_words = words_total = 0
    aligned_root = aligned_dir.resolve()
    for stem, segments in sorted(en_segments.items()):
        ctc_path = ctc_dir / f"{stem}.TextGrid"
        if not ctc_path.exists():
            ctc_path = ctc_dir / stem / f"{stem}.TextGrid"
        ctc_hash = ""
        ctc_error = ""
        try:
            ctc_hash = _sha256(ctc_path)
        except Exception as exc:
            ctc_error = f"ctc_textgrid_hash_failed: {exc}"
        ledger = {"schema": STRICT_SCHEMA, "stem": stem,
                  "canonical_units": CANONICAL_UNITS_SCHEMA,
                  "ctc_textgrid_sha256": ctc_hash,
                  "dictionary_provenance": {
                      "path": str(mfa.get("dictionary", "")),
                      "sha256": str(mfa.get("dictionary_sha256", "")),
                  },
                  "pronunciation_policies": [],
                  "segments": []}
        for seg in segments:
            ordinal = int(seg.get("segment_ordinal", seg.get("seg_idx", 0)))
            sid = f"{stem}:s{ordinal}"; discovered_expected.append(sid)
            record = {"segment_id": sid, "segment_ordinal": ordinal, "status": "verified",
                      "canonical_units": CANONICAL_UNITS_SCHEMA,
                      "reason": "", "mfa_textgrid": None, "words": []}
            words_total += len(seg["words"])
            if ctc_error:
                record["status"] = "rejected"; record["reason"] = ctc_error
            elif seg.get("skipped"):
                record["status"] = "rejected"
                record["reason"] = str(seg.get("reject_reason", "short_or_missing_source_tg"))
            else:
                try:
                    seg_name = seg.get("seg_name", stem + "_seg" + str(seg.get("seg_idx", 0)))
                    resolved = _strict_source_path(aligned_root, seg_name)
                    try:
                        source_hash = _sha256(resolved)
                    except Exception as exc:
                        raise ValueError(f"source_textgrid_hash_failed: {exc}") from exc
                    mw, mp = _strict_tiers(resolved)
                    record["words"] = _strict_verified_words(
                        sid, seg, mw, mp, ledger["dictionary_provenance"])
                    ledger["pronunciation_policies"].extend(
                        word["pronunciation_policy"] for word in record["words"]
                        if "pronunciation_policy" in word)
                    record["mfa_textgrid"] = {"path": str(resolved), "sha256": source_hash}
                    verified += len(record["words"])
                except Exception as exc:
                    record["status"] = "rejected"; record["reason"] = str(exc)
            if record["status"] == "rejected":
                record["words"] = _strict_rejected_words(sid, seg, record["reason"])
                rejected.append({"id": sid, "reason": record["reason"]}); rejected_words += len(seg["words"])
            else:
                produced.append(sid)
            ledger["segments"].append(record)
        ledger_path = output_dir / f"{stem}_en_phones.json"
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            ledgers.append({"stem": stem, "path": str(ledger_path), "sha256": _sha256(ledger_path)})
        except Exception:
            # Let the final global contract turn this into a failed manifest.
            ledgers.append({"stem": stem, "path": str(ledger_path), "sha256": ""})
    expected = expected_segments if expected_segments is not None else discovered_expected
    counts = _strict_counts(expected_counts)
    if expected_counts is None:
        counts.update({"english_stems": len(en_segments), "english_segments": len(expected),
                       "english_words": words_total})
    counts.update({"verified_words": verified, "rejected_words": rejected_words})
    consistent = (expected == discovered_expected and counts["english_words"] == words_total
                  and _strict_manifest_consistent(expected, produced, rejected, ledgers, counts))
    status = ("success" if consistent and not rejected
              else "partial" if consistent else "failed")
    reason = ("" if status == "success"
              else "partial_segments_rejected" if status == "partial"
              else "strict_manifest_inconsistent")
    return write_strict_manifest(output_dir, status, mfa=mfa, expected_segments=expected,
        produced_segments=produced, rejected_segments=rejected, stem_ledgers=ledgers,
        counts=counts, reason=reason)


def _read_dict(path: Path) -> str:
    """Read a pronunciation dictionary with encoding tolerance.

    Tries UTF-8 first (MFA default), falls back to latin-1 (CMUdict).
    Returns only valid dictionary lines (filters comments / blank lines).
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    # Filter out CMUdict comment/header lines — MFA chokes on them
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith((";;;", "#")):
            lines.append(stripped)
    return "\n".join(lines)


def is_english_phone(phone: str) -> bool:
    """Check if *phone* is an MFA English phone (ARPABET-based)."""
    p = phone.strip()
    return p in _ENGLISH_VOWELS or p in _ENGLISH_CONSONANTS or p in _ENGLISH_SILENCE


def parse_textgrid_simple(path: Path) -> list[dict]:
    """Parse only the named CTC ``words`` tier, preserving full ordinals."""
    lines = path.read_text(encoding="utf-8").splitlines()
    intervals = []
    current_name = ""
    in_interval = False
    pending_xmin = pending_xmax = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("name = "):
            current_name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("intervals ["):
            in_interval = True
            pending_xmin = pending_xmax = None
        elif in_interval and line.startswith("xmin = "):
            pending_xmin = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("xmax = "):
            pending_xmax = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("text = "):
            text = line.split("=", 1)[1].strip().strip('"')
            if current_name == "words" and pending_xmin is not None and pending_xmax is not None:
                intervals.append({"ordinal": len(intervals), "xmin": pending_xmin,
                                  "xmax": pending_xmax, "text": text})
            pending_xmin = pending_xmax = None
            in_interval = False

    return intervals


def find_english_segments(ctc_dir: Path, stems: list[str],
                          max_gap_s: float = 0.35,
                          reference_dir: Path | None = None) -> dict[str, list[dict]]:
    """Scan CTC TextGrids and .lab files; return English-word segments per stem.

    *max_gap_s* controls how far apart consecutive English words can be
    before they are split into separate segments (default 0.35 s).

    Returns: {stem: [{"seg_idx": 0, "words": [{"text": "hello", "start": 1.2, "end": 1.8}, ...],
                       "seg_start": 1.15, "seg_end": 1.85}]}
    """
    result: dict[str, list[dict]] = {}

    for stem in stems:
        tg_path = ctc_dir / f"{stem}.TextGrid"
        if not tg_path.exists():
            tg_path = ctc_dir / stem / f"{stem}.TextGrid"
        if not tg_path.exists():
            continue

        intervals = parse_textgrid_simple(tg_path)
        if not intervals:
            continue

        # Preserve the complete words-tier ordinal.  The optional reference
        # sidecar is the authority for compounds; absent a sidecar, each
        # strict source token is a direct canonical unit.
        reference_text = _reference_text_for_stem(ctc_dir, stem, reference_dir)
        en_words = []
        try:
            token_path = ctc_dir / f"{stem}_tokens.jsonl"
            token_rows = load_ctc_token_entries(token_path) if token_path.exists() else None
            canonical_groups = (
                _canonical_units_from_tokens(intervals, token_rows, reference_text)
                if token_rows is not None else None)
            if canonical_groups is None:
                canonical_groups = _canonicalize_source_units(intervals, reference_text)
        except EnglishUnitError as exc:
            # Keep the English denominator while preventing a malformed or
            # split source unit from reaching MFA or fabricating phones.
            source = _source_english_fragments(intervals)
            if not source:
                continue
            rejected_words = [{"text": item["text"], "start": item["start"],
                               "end": item["end"], "ordinal": item["ordinal"]}
                              for item in source]
            result[stem] = [{"seg_idx": 0, "segment_ordinal": 0,
                             "words": rejected_words,
                             "seg_start": rejected_words[0]["start"],
                             "seg_end": rejected_words[-1]["end"],
                             "canonical_reject_reason": exc.code}]
            continue

        for group in canonical_groups:
            unit = group["unit"]
            fragments = group["fragments"]
            token_word = group.get("token_word")
            if token_word is not None:
                start, end = token_word["start"], token_word["end"]
            else:
                start, end = unit.canonical_span
            if start is None or end is None:
                raise ValueError("canonical_unit_timing_missing")
            word = {
                "text": unit.surface_text,
                "alignment_token": unit.alignment_token,
                "unit_id": unit.unit_id,
                "source_ctc_ordinals": list(unit.source_ctc_ordinals),
                "canonical_span": list(unit.canonical_span),
                "canonical_unit": _unit_dict(unit, reference_span=group["reference_span"]),
                "start": start,
                "end": end,
                "ordinal": fragments[0]["ordinal"],
            }
            if token_word is not None:
                word["processed_ctc_span"] = token_word["processed_ctc_span"]
                word["processed_ctc_boundary_source"] = token_word[
                    "processed_ctc_boundary_source"]
            en_words.append(word)

        if not en_words:
            continue

        # Merge consecutive canonical units into segments.  A non-English
        # words-tier interval remains a hard separator.
        segments = []
        seg_words = [en_words[0]]
        seg_start = en_words[0]["start"]
        seg_end = en_words[0]["end"]

        for w in en_words[1:]:
            gap = w["start"] - seg_end
            previous_ordinals = seg_words[-1]["source_ctc_ordinals"]
            if (w["ordinal"] == previous_ordinals[-1] + 1
                    and gap < max_gap_s):
                seg_words.append(w)
                seg_end = w["end"]
            else:
                segments.append({
                    "words": seg_words,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                })
                seg_words = [w]
                seg_start = w["start"]
                seg_end = w["end"]

        if seg_words:
            segments.append({
                "words": seg_words,
                "seg_start": seg_start,
                "seg_end": seg_end,
            })

        # Assign segment indices
        for idx, seg in enumerate(segments):
            seg["seg_idx"] = idx
            seg["segment_ordinal"] = idx

        result[stem] = segments

    return result


def _build_corpus_stem(stem: str, segments: list[dict],
                       audio_dir: Path, corpus_dir: Path,
                       padding_s: float, min_dur_s: float,
                       strict: bool = False) -> tuple[str, list[dict] | None]:
    """Process a single stem's English segments for MFA corpus.

    Returns (stem, updated_segments) or (stem, None) if the stem should be skipped.
    Module-level so it is picklable for ProcessPoolExecutor.
    """
    import numpy as np

    def reject_all(reason: str) -> tuple[str, list[dict] | None]:
        if strict:
            for segment in segments:
                segment["skipped"] = True
                segment["reject_reason"] = reason
                segment.setdefault("offset", segment["seg_start"])
            return stem, segments
        return stem, None

    # Find audio file
    wav_path = audio_dir / f"{stem}.wav"
    if not wav_path.exists():
        candidates = list(audio_dir.rglob(f"{stem}.wav"))
        if candidates:
            wav_path = candidates[0]
        else:
            return reject_all("audio_missing")

    # Read audio (scipy handles PCM float and int, mono and multi-channel)
    try:
        from scipy.io import wavfile as _wavfile
        sr, audio = _wavfile.read(str(wav_path))
    except Exception:
        return reject_all("audio_unreadable")

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = audio.astype(np.float32) / 128.0 - 1.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    total_dur = len(audio) / sr
    valid_segments = []

    for seg in segments:
        if seg.get("canonical_reject_reason"):
            seg["skipped"] = True
            seg["reject_reason"] = str(seg["canonical_reject_reason"])
            seg["offset"] = seg["seg_start"]
            valid_segments.append(seg)
            continue
        try:
            for word in seg["words"]:
                _validated_unit(word)
        except EnglishUnitError as exc:
            seg["skipped"] = True
            seg["reject_reason"] = exc.code
            seg["offset"] = seg["seg_start"]
            valid_segments.append(seg)
            continue
        seg_start_raw = seg["seg_start"]
        seg_end_raw = seg["seg_end"]
        raw_duration_s = seg_end_raw - seg_start_raw

        # Add padding
        seg_start_padded = max(0.0, seg_start_raw - padding_s)
        seg_end_padded = min(total_dur, seg_end_raw + padding_s)
        padded_duration_s = seg_end_padded - seg_start_padded
        seg["raw_duration_s"] = raw_duration_s
        seg["padded_duration_s"] = padded_duration_s
        seg["padding_s"] = {
            "requested": padding_s,
            "left": seg_start_raw - seg_start_padded,
            "right": seg_end_padded - seg_end_raw,
        }

        # Eligibility is measured on the exact sample interval MFA will see.
        # Comparing binary floats made a serialized 150 ms clip occasionally
        # evaluate just below 0.150 and become a false short-segment reject.
        # The CTC/TextGrid axis is decimal seconds.  Convert it to the nearest
        # sample instead of flooring two independent binary-float products;
        # otherwise an exact threshold clip can lose one sample at one edge.
        start_sample = int(round(seg_start_padded * sr))
        end_sample = int(round(seg_end_padded * sr))
        minimum_samples = int(math.ceil(min_dur_s * sr - 1e-9))
        if end_sample - start_sample < minimum_samples:
            seg["skipped"] = True
            if strict:
                seg["reject_reason"] = "segment_too_short"
            seg["offset"] = seg_start_raw
            valid_segments.append(seg)
            continue

        if end_sample <= start_sample:
            seg["skipped"] = True
            if strict:
                seg["reject_reason"] = "segment_too_short"
            seg["offset"] = seg_start_raw
            valid_segments.append(seg)
            continue

        seg_audio = audio[start_sample:end_sample]
        seg_audio_int16 = (seg_audio * 32767).clip(-32768, 32767).astype(np.int16)

        seg_name = f"{stem}_seg{seg['seg_idx']}"
        seg_wav = corpus_dir / f"{seg_name}.wav"
        seg_lab = corpus_dir / f"{seg_name}.lab"

        from scipy.io import wavfile as _wavfile2
        _wavfile2.write(str(seg_wav), sr, seg_audio_int16)

        # .lab: English word sequence
        # MFA receives only the canonical, hyphenless alignment tokens.  The
        # surface spelling remains in the attached unit for provenance.
        lab_text = " ".join(w["alignment_token"] for w in seg["words"])
        seg_lab.write_text(lab_text + "\n", encoding="utf-8")

        seg["skipped"] = False
        seg["offset"] = seg_start_padded
        seg["seg_name"] = seg_name
        valid_segments.append(seg)

    if valid_segments:
        return (stem, valid_segments)
    else:
        return (stem, None)


def build_en_corpus(en_segments: dict[str, list[dict]],
                    audio_dir: Path, corpus_dir: Path,
                    padding_ms: float = 50.0,
                    min_segment_dur_ms: float = 150.0,
                    corpus_workers: int = 0,
                    strict: bool = False) -> dict[str, list[dict]]:
    """Extract English audio segments and build MFA corpus.

    Writes {stem}_seg{idx}.wav and {stem}_seg{idx}.lab to corpus_dir.
    Returns updated en_segments with offset info.

    Uses parallel workers when processing more than 4 stems —
    each stem's WAV read + segment extraction is independent.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    padding_s = padding_ms / 1000.0
    min_dur_s = min_segment_dur_ms / 1000.0

    stem_items = list(en_segments.items())
    if len(stem_items) <= 4:
        # Serial: too few stems to justify process overhead
        for stem, segments in stem_items:
            try:
                _, result = _build_corpus_stem(stem, segments, audio_dir,
                                               corpus_dir, padding_s, min_dur_s,
                                               strict)
            except Exception as exc:
                print(f"  [ERROR] corpus build {stem}: {exc}")
                if strict:
                    for segment in segments:
                        segment["skipped"] = True
                        segment["reject_reason"] = "corpus_worker_error"
                        segment.setdefault("offset", segment["seg_start"])
                    result = segments
                else:
                    result = None
            if result is not None:
                en_segments[stem] = result
            else:
                del en_segments[stem]
    else:
        _max_w = min(16, os.cpu_count(), len(stem_items))
        n_workers = corpus_workers if corpus_workers > 0 else _max_w
        n_workers = min(n_workers, len(stem_items))
        print(f"  Building English corpus: {len(stem_items)} stems,"
              f" {n_workers} workers")
        ctx = __import__('multiprocessing').get_context("fork")
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(_build_corpus_stem, stem, segments, audio_dir,
                                corpus_dir, padding_s, min_dur_s, strict): stem
                for stem, segments in stem_items
            }
            for future in as_completed(futures):
                stem = futures[future]
                done += 1
                try:
                    s, result = future.result()
                    if result is not None:
                        en_segments[s] = result
                    else:
                        en_segments.pop(s, None)
                except Exception as e:
                    print(f"  [ERROR] corpus build {stem}: {e}")
                    if strict:
                        for segment in en_segments[stem]:
                            segment["skipped"] = True
                            segment["reject_reason"] = "corpus_worker_error"
                            segment.setdefault("offset", segment["seg_start"])
                    else:
                        en_segments.pop(stem, None)
                if done % 200 == 0 or done == len(stem_items):
                    print(f"  [{done}/{len(stem_items)}] stems processed")

    return en_segments


def build_en_dict(en_segments: dict[str, list[dict]],
                  base_dict: Path, g2p_model: Path,
                  mfa_python: Path, models_dir: Path,
                  temp_dir: Path, g2p_timeout: int = 300,
                  strict: bool = False) -> Path:
    """Build English pronunciation dictionary for the corpus.

    Checks all English words against base_dict; runs G2P for OOV words.
    Always returns a clean dictionary (comments stripped) — never the
    raw base_dict path because MFA can't parse CMUdict comment lines.
    """
    # Collect all unique English words
    all_words: set[str] = set()
    for stem, segments in en_segments.items():
        for seg in segments:
            for w in seg["words"]:
                try:
                    unit = _validated_unit(w)
                except EnglishUnitError as exc:
                    if strict:
                        raise StrictG2PError(f"canonical unit rejected: {exc.code}") from exc
                    continue
                all_words.add(unit.alignment_token)

    if not all_words:
        return base_dict

    # Load base dictionary entries
    base_words: set[str] = set()
    base_dict_text = ""
    if base_dict.exists():
        base_dict_text = _read_dict(base_dict)
        for line in base_dict_text.splitlines():
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if parts:
                    word = parts[0].split("(")[0].lower()
                    try:
                        base_words.add(canonicalize_english_token(word))
                    except EnglishUnitError:
                        # Dictionary inventory may contain non-English rows;
                        # they are not eligible for canonical English lookup.
                        continue

    oov_words = sorted(all_words - base_words)
    # Always start with a clean (comment-free) copy of the base dictionary.
    # SOS is deliberately in-vocabulary, so it must be replaced before the
    # OOV decision rather than sent through G2P or copied from CMUdict.
    dictionary_text = base_dict_text
    if SOS_ALIGNMENT_TOKEN in all_words:
        dictionary_text = _replace_exact_dictionary_entry(
            dictionary_text, SOS_ALIGNMENT_TOKEN, SOS_EXPECTED_PRONUNCIATION)
    if APP_ALIGNMENT_TOKEN in all_words:
        _validate_exact_dictionary_word(
            dictionary_text, APP_ALIGNMENT_TOKEN, APP_EXPECTED_PRONUNCIATION,
            error_code="app_pronunciation_missing_or_tampered")

    combined = temp_dir / "en_combined.dict"
    if not oov_words:
        combined.parent.mkdir(parents=True, exist_ok=True)
        combined.write_text(dictionary_text, encoding="utf-8")
        if SOS_ALIGNMENT_TOKEN in all_words:
            _validate_exact_dictionary_word(
                dictionary_text, SOS_ALIGNMENT_TOKEN, SOS_EXPECTED_PRONUNCIATION,
                error_code="sos_dictionary_override_missing_or_tampered")
        return combined

    def merge_dictionary(g2p_text: str = "") -> Path:
        with combined.open("w", encoding="utf-8") as outf:
            if dictionary_text:
                outf.write(dictionary_text)
                if g2p_text:
                    outf.write("\n")
            outf.write(g2p_text)
        return combined

    def g2p_coverage(text: str) -> tuple[set[str], bool]:
        entries: set[str] = set()
        has_pronunciation = False
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                entries.add(parts[0].split("(")[0].lower())
                has_pronunciation = True
        return entries, has_pronunciation

    # Check dictionary cache (keyed by hash of sorted OOV word list)
    import hashlib
    cache_key = hashlib.sha1(",".join(oov_words).encode()).hexdigest()[:12]
    cache_dir = temp_dir / "en_dict_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_dict = cache_dir / f"{cache_key}.dict"
    if cached_dict.exists():
        cached_text = cached_dict.read_text(encoding="utf-8")
        entries, has_pronunciation = g2p_coverage(cached_text)
        if strict and (not has_pronunciation or not set(oov_words).issubset(entries)):
            missing = sorted(set(oov_words) - entries)
            raise StrictG2PError("strict G2P cache is empty or does not cover all OOV words"
                                 f" (missing: {', '.join(missing[:20])})")
        result = merge_dictionary(cached_text)
        final_text = result.read_text(encoding="utf-8")
        if SOS_ALIGNMENT_TOKEN in all_words:
            _validate_exact_dictionary_word(
                final_text, SOS_ALIGNMENT_TOKEN, SOS_EXPECTED_PRONUNCIATION,
                error_code="sos_dictionary_override_missing_or_tampered")
        if APP_ALIGNMENT_TOKEN in all_words:
            _validate_exact_dictionary_word(
                final_text, APP_ALIGNMENT_TOKEN, APP_EXPECTED_PRONUNCIATION,
                error_code="app_pronunciation_missing_or_tampered")
        return result

    # Run G2P for OOV words
    oov_file = temp_dir / "en_oov_words.txt"
    oov_file.write_text("\n".join(oov_words) + "\n", encoding="utf-8")
    g2p_output = temp_dir / "en_oov_dict.txt"

    g2p_model_path = str(g2p_model)
    if strict and g2p_model_path in ("", "."):
        raise StrictG2PError("strict G2P model missing")
    if not Path(g2p_model_path).exists():
        # Try zip extension
        g2p_zip = Path(str(g2p_model) + ".zip")
        if g2p_zip.exists():
            g2p_model_path = str(g2p_zip)
        else:
            message = f"strict G2P model missing: {g2p_model}"
            if strict:
                raise StrictG2PError(message)
            print(f"  WARNING: {message}; skipping OOV generation")
            return merge_dictionary()

    print(f"  Running G2P for {len(oov_words)} OOV English words...")
    try:
        # Use local temp dir — SQLite on CIFS/SMB fails with "database is locked"
        g2p_temp = temp_dir / "g2p_work"
        g2p_temp.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa",
             "g2p", str(oov_file), g2p_model_path, str(g2p_output),
             "--num_pronunciations", "1", "--clean",
             "--temporary_directory", str(g2p_temp)],
            env=get_mfa_env(mfa_python, models_dir),
            timeout=g2p_timeout, capture_output=True, text=True,
        )
        if rc.returncode != 0:
            detail = rc.stderr[-500:] if rc.stderr else "unknown"
            if strict:
                raise StrictG2PError(f"strict G2P returned nonzero exit code {rc.returncode}: {detail}")
            print(f"  WARNING: G2P failed: {detail}")
    except StrictG2PError:
        raise
    except subprocess.TimeoutExpired:
        if strict:
            raise StrictG2PError(f"strict G2P timed out after {g2p_timeout} seconds")
        print("  WARNING: G2P timed out")
    except Exception as e:
        if strict:
            raise StrictG2PError(f"strict G2P launch failed: {e}") from e
        print(f"  WARNING: G2P error: {e}")

    if not g2p_output.exists():
        if strict:
            raise StrictG2PError("strict G2P did not produce an output dictionary")
        # G2P failed — write base dict only so MFA can still run on
        # words that ARE in the dictionary (OOV words will fall back to
        # equal split in postprocessing)
        return merge_dictionary()

    g2p_text = g2p_output.read_text(encoding="utf-8")
    entries, has_pronunciation = g2p_coverage(g2p_text)
    if strict and not has_pronunciation:
        raise StrictG2PError("strict G2P produced an empty pronunciation dictionary")
    if strict and not set(oov_words).issubset(entries):
        missing = sorted(set(oov_words) - entries)
        raise StrictG2PError("strict G2P output does not cover all OOV words"
                             f" (missing: {', '.join(missing[:20])})")

    # Merge base dict + G2P output
    result = merge_dictionary(g2p_text)

    final_text = result.read_text(encoding="utf-8")
    if SOS_ALIGNMENT_TOKEN in all_words:
        _validate_exact_dictionary_word(
            final_text, SOS_ALIGNMENT_TOKEN, SOS_EXPECTED_PRONUNCIATION,
            error_code="sos_dictionary_override_missing_or_tampered")
    if APP_ALIGNMENT_TOKEN in all_words:
        _validate_exact_dictionary_word(
            final_text, APP_ALIGNMENT_TOKEN, APP_EXPECTED_PRONUNCIATION,
            error_code="app_pronunciation_missing_or_tampered")

    # Save to cache for future runs
    import shutil
    shutil.copy(g2p_output, cached_dict)

    print(f"  Combined dictionary: {combined} ({len(all_words)} words)")
    return result


def run_en_mfa(corpus_dir: Path, dict_path: Path, acoustic_model: str,
               output_dir: Path, temp_dir: Path, mfa_python: Path,
               models_dir: Path, num_jobs: int = 4,
               beam: int = 10, retry_beam: int = 40,
               fine_tune: bool = False, timeout: int = 1800,
               strict: bool = False, dither: float = 0.0) -> dict:
    """Run MFA align and retain an exception-safe, auditable outcome."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use extracted model if available
    acoustic_arg = str(_resolve_acoustic_model(acoustic_model, models_dir))

    dither = resolve_mfa_dither(dither)
    mfa_args = [
        "align", str(corpus_dir), str(dict_path),
        acoustic_arg, str(output_dir),
        "--temporary_directory", str(temp_dir),
        "--output_format", "long_textgrid",
        "--num_jobs", str(num_jobs),
        "--single_speaker",
        "--no_tokenization",
        "--beam", str(beam),
        "--retry_beam", str(retry_beam),
        "--dither", str(dither),
    ]
    if not strict:
        mfa_args.append("--clean")
    mfa_args.append("--overwrite")
    if fine_tune:
        mfa_args.append("--fine_tune")

    print(f"  Running English MFA align ({len(list(corpus_dir.glob('*.wav')))} segments)...")
    command = [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa"] + mfa_args
    mfa_root = temp_dir / "mfa_root"
    numba_cache = temp_dir / "numba_cache"
    mfa_root.mkdir(parents=True, exist_ok=True)
    numba_cache.mkdir(parents=True, exist_ok=True)
    mfa_env = get_mfa_env(mfa_python, models_dir)
    # The parent pipeline may itself inherit a shared MFA_ROOT_DIR.  Override
    # it here: MFA writes command_history.yaml and Numba cache files, and
    # concurrent English runs must never race in models/mfa or site-packages.
    mfa_env["MFA_ROOT_DIR"] = str(mfa_root)
    mfa_env["NUMBA_CACHE_DIR"] = str(numba_cache)
    result = {"return_code": None, "timed_out": False, "timeout_seconds": timeout,
              "command": command, "acoustic_model": acoustic_arg, "exception": "",
              "environment": {"MFA_ROOT_DIR": str(mfa_root),
                              "NUMBA_CACHE_DIR": str(numba_cache)}}
    log_path = temp_dir / "english_mfa.log"
    if strict:
        result["log_path"] = str(log_path)

    def append_log_footer() -> None:
        if strict:
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\n[outcome]\n")
                log.write(f"return_code: {result['return_code']}\n")
                log.write(f"timed_out: {result['timed_out']}\n")
                log.write(f"exception: {result['exception']}\n")

    try:
        run_kwargs = {"env": mfa_env, "timeout": timeout}
        if strict:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("command: " + " ".join(command) + "\n\n[output]\n")
                log.flush()
                run_kwargs.update({"stdout": log, "stderr": subprocess.STDOUT})
                rc = subprocess.run(command, **run_kwargs)
        else:
            rc = subprocess.run(command, **run_kwargs)
        if rc.returncode != 0:
            print(f"  WARNING: English MFA returned code {rc.returncode}")
            result["return_code"] = rc.returncode
            append_log_footer()
            return result
        result["return_code"] = 0
    except subprocess.TimeoutExpired as exc:
        print("  WARNING: English MFA timed out")
        result["timed_out"] = True; result["return_code"] = "timeout"
        append_log_footer()
        return result
    except Exception as e:
        print(f"  WARNING: English MFA error: {e}")
        result["return_code"] = "exception"; result["exception"] = str(e)
        append_log_footer()
        return result
    append_log_footer()
    return result


def retry_missing_en_segments(
        en_segments: dict[str, list[dict]], corpus_dir: Path,
        aligned_dir: Path, dict_path: Path, acoustic_model: str,
        temp_dir: Path, mfa_python: Path, models_dir: Path, *,
        beam: int = 100, retry_beam: int = 1000,
        timeout: int = 600, limit: int = 16,
        dither: float = 0.0) -> list[dict]:
    """Retry bounded utterance-level decoder misses in isolated MFA roots.

    MFA can return process success while omitting one or more utterances.  A
    missing TextGrid is not provenance, so each retry receives only the exact
    original clip/lab and publishes back only after strict tier validation.
    """
    missing: list[str] = []
    aligned_root = aligned_dir.resolve()
    for stem, segments in sorted(en_segments.items()):
        for segment in segments:
            if segment.get("skipped"):
                continue
            seg_name = str(segment.get(
                "seg_name", f"{stem}_seg{int(segment.get('seg_idx', 0))}"))
            try:
                _strict_source_path(aligned_root, seg_name)
            except ValueError:
                missing.append(seg_name)

    if not missing:
        return []

    retry_root = temp_dir / "en_singleton_retry"
    if retry_root.is_symlink():
        raise ValueError("singleton retry root must not be a symlink")
    if retry_root.exists():
        shutil.rmtree(retry_root)
    retry_root.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    for ordinal, seg_name in enumerate(missing):
        record = {"segment_name": seg_name, "status": "not_run"}
        if ordinal >= max(0, int(limit)):
            record.update({"status": "retry_limit_exceeded",
                           "reason": "singleton_retry_limit_exceeded"})
            records.append(record)
            continue
        wav = corpus_dir / f"{seg_name}.wav"
        lab = corpus_dir / f"{seg_name}.lab"
        try:
            if (wav.is_symlink() or lab.is_symlink()
                    or not wav.is_file() or not lab.is_file()):
                raise ValueError("singleton retry corpus artifact missing or aliased")
            unit_root = retry_root / f"r{ordinal:04d}_{hashlib.sha256(seg_name.encode()).hexdigest()[:12]}"
            retry_corpus = unit_root / "corpus"
            retry_aligned = unit_root / "aligned"
            retry_work = unit_root / "work"
            retry_corpus.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(wav, retry_corpus / wav.name)
            shutil.copyfile(lab, retry_corpus / lab.name)
            record["inputs"] = {
                "wav_sha256": _sha256(wav), "lab_sha256": _sha256(lab)}
            result = run_en_mfa(
                retry_corpus, dict_path, acoustic_model,
                retry_aligned, retry_work, mfa_python, models_dir, 1,
                beam=beam, retry_beam=retry_beam, fine_tune=False,
                timeout=timeout, strict=True, dither=dither)
            record["mfa"] = result
            if result.get("return_code") != 0:
                record.update({"status": "mfa_failed",
                               "reason": "singleton_retry_mfa_failed"})
                records.append(record)
                continue
            source = _strict_source_path(retry_aligned.resolve(), seg_name)
            _strict_tiers(source)
            destination = aligned_dir / f"{seg_name}.TextGrid"
            staging = aligned_dir / f".{seg_name}.singleton-retry.tmp"
            if staging.exists() or staging.is_symlink():
                raise ValueError("singleton retry staging collision")
            shutil.copyfile(source, staging)
            if _sha256(source) != _sha256(staging):
                raise ValueError("singleton retry publication hash mismatch")
            os.replace(staging, destination)
            record.update({"status": "recovered",
                           "textgrid_sha256": _sha256(destination)})
        except Exception as exc:
            record.update({"status": "validation_failed", "reason": str(exc)})
        records.append(record)
    recovered = sum(record["status"] == "recovered" for record in records)
    print(f"  Singleton English MFA retry: {recovered}/{len(missing)} recovered")
    return records


def parse_en_textgrid(tg_path: Path) -> dict:
    """Parse English MFA TextGrid into a simple structure.

    Returns {words: [{text, start, end, phones: [{phone, start, end}]}]}
    Assumes tier 0 = words, tier 1 = phones.
    """
    lines = tg_path.read_text(encoding="utf-8").splitlines()
    tiers: dict[str, list[dict]] = {}
    current_name = None
    current_tier: list[dict] | None = None
    in_interval = False
    pending_xmin = pending_xmax = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("item ["):
            current_name = None
            current_tier = None
            in_interval = False
        elif line.startswith("name = ") and current_tier is None:
            current_name = line.split("=", 1)[1].strip().strip('"')
            current_tier = tiers.setdefault(current_name, [])
        elif line.startswith("intervals [") and current_tier is not None:
            in_interval = True
            pending_xmin = pending_xmax = None
        elif in_interval and line.startswith("xmin = "):
            pending_xmin = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("xmax = "):
            pending_xmax = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("text = "):
            text = line.split("=", 1)[1].strip().strip('"')
            if pending_xmin is not None and pending_xmax is not None and current_tier is not None:
                current_tier.append({"ordinal": len(current_tier), "xmin": pending_xmin,
                                     "xmax": pending_xmax, "text": text})
            pending_xmin = pending_xmax = None
            in_interval = False

    words_tier = tiers.get("words", [])
    phones_tier = tiers.get("phones", [])
    if not words_tier or not phones_tier:
        return {"words": []}

    # Build word list with nested phones
    words = []
    phone_idx = 0
    for w in words_tier:
        text = w["text"].strip()
        if not text or text in SILENCE_LABELS or text == "<eps>":
            continue
        w_start = w["xmin"]
        w_end = w["xmax"]

        # Collect phones within this word interval
        word_phones = []
        while phone_idx < len(phones_tier):
            p = phones_tier[phone_idx]
            if p["xmin"] >= w_end - 0.001:
                break
            if p["xmax"] > w_start + 0.001:
                p_text = p["text"].strip()
                if p_text and p_text not in ("sil", "sp", "spn", "<eps>"):
                    word_phones.append({
                        "ordinal": p["ordinal"],
                        "phone": p_text,
                        "start": max(p["xmin"], w_start),
                        "end": min(p["xmax"], w_end),
                    })
            phone_idx += 1

        words.append({
            "text": text,
            "start": w_start,
            "end": w_end,
            "phones": word_phones,
        })

    return {"words": words}


# Track ARPABET phones from MFA output that fail validation.
# These are reported at the end and indicate model/dictionary mismatches.
_unknown_arpabet_phones: set[str] = set()


def collect_en_phones(en_segments: dict[str, list[dict]],
                      en_aligned_dir: Path,
                      output_dir: Path) -> int:
    """Parse English MFA TextGrids and write per-stem JSON files.

    Returns number of stems processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_processed = 0

    for stem, segments in en_segments.items():
        en_data = []

        def _word_record(unit: EnglishUnit, word: dict, *, seg_idx: int,
                         offset: float = 0.0, phones: list | None = None,
                         status: str | None = None, reason: str = "") -> dict:
            processed = word.get("processed_ctc_span")
            record = {
                "seg_idx": seg_idx,
                "offset": round(offset, 4),
                "word_text": unit.surface_text,
                "alignment_token": unit.alignment_token,
                "unit_id": unit.unit_id,
                "source_ctc_ordinals": list(unit.source_ctc_ordinals),
                "canonical_span": list(unit.canonical_span),
                "word_start": word.get("start"),
                "word_end": word.get("end"),
                "phones": phones if phones is not None else [],
            }
            if isinstance(processed, (list, tuple)) and len(processed) == 2:
                record["processed_ctc_span"] = list(processed)
                record["processed_ctc_boundary_source"] = word.get(
                    "processed_ctc_boundary_source")
            if status is not None:
                record["status"] = status
                record["reason"] = reason
            return record

        for seg in segments:
            seg_idx = seg["seg_idx"]

            if seg.get("skipped"):
                # G2P fallback: equal-duration split for CTC word interval
                for w in seg["words"]:
                    try:
                        unit = _validated_unit(w)
                    except EnglishUnitError:
                        continue
                    if "processed_ctc_span" in w:
                        en_data.append(_word_record(
                            unit, w, seg_idx=seg_idx, status="rejected",
                            reason=str(seg.get("canonical_reject_reason",
                                               seg.get("reject_reason",
                                                      "mfa_segment_unavailable")))))
                    else:
                        en_data.append(_word_record(unit, w, seg_idx=seg_idx))
                continue

            seg_name = seg.get("seg_name", f"{stem}_seg{seg_idx}")
            tg_path = en_aligned_dir / f"{seg_name}.TextGrid"
            if not tg_path.exists():
                # Try nested
                nested = en_aligned_dir / seg_name / f"{seg_name}.TextGrid"
                if nested.exists():
                    tg_path = nested
                else:
                    # MFA didn't produce output — fall back
                    for w in seg["words"]:
                        try:
                            unit = _validated_unit(w)
                        except EnglishUnitError:
                            continue
                        if "processed_ctc_span" in w:
                            en_data.append(_word_record(
                                unit, w, seg_idx=seg_idx, status="rejected",
                                reason="mfa_textgrid_missing"))
                        else:
                            en_data.append(_word_record(unit, w, seg_idx=seg_idx))
                    continue

            parsed = parse_en_textgrid(tg_path)
            offset = seg["offset"]

            # Match English MFA words to CTC words by text and sequence position
            mfa_words = parsed.get("words", [])
            ctc_words = seg["words"]

            # Simple positional matching: MFA words should align 1:1 with CTC words
            # (same text, same order)
            mfa_idx = 0
            for ctc_w in ctc_words:
                try:
                    unit = _validated_unit(ctc_w)
                except EnglishUnitError:
                    continue
                matched = None
                if mfa_idx < len(mfa_words):
                    mw = mfa_words[mfa_idx]
                    try:
                        matched_token = canonicalize_english_token(mw["text"].strip())
                    except EnglishUnitError:
                        matched_token = ""
                    if matched_token == unit.alignment_token:
                        matched = mw
                        mfa_idx += 1

                if matched:
                    # Map MFA phone times (relative to segment) to absolute times
                    phones_abs = []
                    for p in matched["phones"]:
                        ph = p["phone"].strip()
                        # Validate against known ARPABET phone set
                        if ph and not is_arpabet_phone(ph):
                            _unknown_arpabet_phones.add(ph)
                        phones_abs.append({
                            "phone": ph,
                            "start": round(offset + p["start"], 4),
                            "end": round(offset + p["end"], 4),
                        })

                    item = _word_record(unit, ctc_w, seg_idx=seg_idx,
                                        offset=offset, phones=phones_abs)
                    item.update({
                        "en_word_start": round(offset + matched["start"], 4),
                        "en_word_end": round(offset + matched["end"], 4),
                    })
                    en_data.append(item)
                else:
                    # A canonical mismatch is a rejection, never a short
                    # TextGrid/equal-duration phone fallback.
                    if "processed_ctc_span" in ctc_w:
                        en_data.append(_word_record(
                            unit, ctc_w, seg_idx=seg_idx, offset=offset,
                            status="rejected", reason="mfa_word_mismatch"))
                    else:
                        en_data.append(_word_record(unit, ctc_w,
                                                    seg_idx=seg_idx,
                                                    offset=offset))

        if en_data:
            out_path = output_dir / f"{stem}_en_phones.json"
            out_path.write_text(
                json.dumps(en_data, ensure_ascii=False, indent=2),
                encoding="utf-8")
            n_processed += 1

    # Report diagnostics
    n_mappings = report_en_ipa_mappings()
    if _unknown_arpabet_phones:
        print(f"  WARNING: Unknown ARPABET phones from MFA output "
              f"({len(_unknown_arpabet_phones)}): "
              f"{', '.join(sorted(_unknown_arpabet_phones)[:30])}"
              f"{'…' if len(_unknown_arpabet_phones) > 30 else ''}")
    elif n_mappings == 0:
        print(f"  All English phones validated (ARPABET native, no IPA mapping triggered)")

    return n_processed


def main():
    parser = argparse.ArgumentParser(description="English MFA alignment for English segments")
    parser.add_argument("--ctc-dir", type=Path, required=True,
                        help="CTC prealignment directory (with .TextGrid + .lab files)")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="16kHz mono audio directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for English phone JSON files")
    parser.add_argument("--acoustic-model", type=str, required=True,
                        help="Path to english_us_arpa acoustic model (.zip or extracted dir)")
    parser.add_argument("--dictionary", type=str, required=True,
                        help="Path to pronunciation dictionary (e.g. dict/cmudict.dict)")
    parser.add_argument("--g2p-model", type=str, default="",
                        help="Path to english_us_arpa G2P model (.zip)")
    parser.add_argument("--temp-dir", type=Path, default=None,
                        help="Temporary directory for MFA working files")
    parser.add_argument("--num-jobs", type=int, default=4,
                        help="Number of parallel MFA jobs")
    parser.add_argument("--dither", type=float, default=0.0,
                        help="MFCC dither; zero gives deterministic alignment")
    parser.add_argument("--padding-ms", type=float, default=75.0,
                        help="Padding around English segments (ms)")
    parser.add_argument("--min-segment-dur-ms", type=float, default=150.0,
                        help="Minimum segment duration for MFA (ms)")
    parser.add_argument("--max-gap-merge-s", type=float, default=0.35,
                        help="Max gap between consecutive English words to merge (s)")
    parser.add_argument("--beam", type=int, default=10,
                        help="MFA Viterbi beam width for English alignment")
    parser.add_argument("--retry-beam", type=int, default=40,
                        help="MFA retry beam width for English alignment")
    parser.add_argument("--singleton-retry-beam", type=int, default=100,
                        help="Beam for isolated retries of omitted utterances")
    parser.add_argument("--singleton-retry-retry-beam", type=int, default=1000,
                        help="Retry beam for isolated omitted utterances")
    parser.add_argument("--singleton-retry-timeout", type=int, default=600)
    parser.add_argument("--singleton-retry-limit", type=int, default=16)
    parser.add_argument("--fine-tune", action="store_true",
                        help="Enable MFA extra fine-tuning pass (default: disabled)")
    parser.add_argument("--corpus-workers", type=int, default=0,
                        help="Parallel workers for English corpus building (0=auto)")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--g2p-timeout", type=int, default=300)
    parser.add_argument("--strict-provenance", action="store_true")
    parser.add_argument("--python", type=str, default=None,
                        help="Python interpreter with MFA installed")
    args = parser.parse_args()

    # Resolve paths
    ctc_dir = args.ctc_dir
    audio_dir = args.audio_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir or (output_dir / "temp_en_mfa")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = PROJECT_ROOT / "models" / "mfa"

    # Discover stems from CTC directory
    stems = []
    for f in sorted(ctc_dir.glob("*.lab")):
        stems.append(f.stem)
    if not stems:
        # Try nested
        for d in sorted(ctc_dir.iterdir()):
            if d.is_dir():
                lab = d / f"{d.name}.lab"
                if lab.exists():
                    stems.append(d.name)
    if not stems:
        print(f"No .lab files found in {ctc_dir}")
        if args.strict_provenance:
            mfa = _strict_mfa_record(
                {"return_code": "not_run", "timed_out": False, "timeout_seconds": args.timeout,
                 "command": [], "exception": "no_ctc_stems", "reason": "no_ctc_stems"},
                Path(args.acoustic_model), Path(args.dictionary))
            write_strict_manifest(output_dir, "failed", mfa=mfa,
                                  expected_segments=[], produced_segments=[], rejected_segments=[], stem_ledgers=[],
                                  counts={}, reason="no_ctc_stems")
            return 1
        return 0

    print(f"Found {len(stems)} stems with CTC output")

    # Step 1: Find English segments
    print("Scanning for English word segments...")
    en_segments = find_english_segments(ctc_dir, stems, max_gap_s=args.max_gap_merge_s)
    n_with_en = len(en_segments)
    print(f"  {n_with_en} stems contain English words")

    if n_with_en == 0:
        print("No English words found — nothing to do.")
        if args.strict_provenance:
            # An all-Chinese corpus does not consume English MFA artifacts.
            # Do not resolve or hash an unrelated English model/dictionary.
            write_strict_manifest(output_dir, "no_english", mfa={
                "return_code": 0, "timed_out": False, "timeout_seconds": args.timeout,
                "command": [], "exception": "", "acoustic_model": "",
                "acoustic_model_sha256": "", "dictionary_sha256": ""},
                expected_segments=[], produced_segments=[], rejected_segments=[], stem_ledgers=[],
                counts={"english_stems": 0, "english_segments": 0, "english_words": 0, "verified_words": 0, "rejected_words": 0})
        return 0

    # An all-Chinese input needs no MFA environment; defer this check until we
    # know English alignment is actually required.
    if args.python:
        mfa_python = Path(args.python)
    else:
        mfa_python = find_mfa_python("")
    if not mfa_python or not mfa_python.exists():
        print("ERROR: Cannot find Python with MFA installed.")
        return 1

    total_en_words = sum(
        sum(len(seg["words"]) for seg in segs)
        for segs in en_segments.values()
    )
    print(f"  {total_en_words} total English words")

    # Strict manifests are denominated from discovery, never from the mutable
    # corpus result.  In particular, a missing WAV must not erase a segment.
    strict_expected, strict_counts = _strict_expected_snapshot(en_segments)

    # Step 2: Build English corpus
    en_corpus_dir = temp_dir / "en_corpus"
    en_aligned_dir = temp_dir / "en_aligned"
    en_work_dir = temp_dir / "en_mfa_work"
    if args.strict_provenance:
        # A strict run owns these exact child directories.  Clear them before
        # rebuilding the corpus so an old MFA export cannot satisfy the new
        # segment ledger (the historical fixed /tmp/mfa_temp layout caused
        # source_tg_missing_or_ambiguous to be reported for current data).
        for stale_dir in (en_corpus_dir, en_aligned_dir, en_work_dir, temp_dir / "log"):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        # These are evidence artifacts, including in pre-MFA failures.
        for artifact_dir in (en_corpus_dir, en_aligned_dir, en_work_dir, temp_dir / "log"):
            artifact_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting English audio segments...")
    en_segments = build_en_corpus(
        en_segments, audio_dir, en_corpus_dir,
        padding_ms=args.padding_ms,
        min_segment_dur_ms=args.min_segment_dur_ms,
        corpus_workers=args.corpus_workers,
        strict=args.strict_provenance,
    )

    n_segments = sum(
        sum(1 for seg in segs if not seg.get("skipped"))
        for segs in en_segments.values()
    )
    n_skipped = sum(
        sum(1 for seg in segs if seg.get("skipped"))
        for segs in en_segments.values()
    )
    print(f"  {n_segments} English segments for MFA, {n_skipped} skipped (too short)")

    if n_segments == 0:
        if args.strict_provenance:
            # Local corpus rejections do not waive global provenance.  No MFA
            # subprocess is needed, but the exact configured launch inputs
            # still have to be resolvable and hashable before a successful
            # strict ledger may be emitted.
            actual_acoustic = _resolve_acoustic_model(args.acoustic_model, models_dir)
            strict_preflight = _strict_mfa_record(
                {"return_code": "not_run", "timed_out": False,
                 "timeout_seconds": args.timeout, "command": [],
                 "acoustic_model": str(actual_acoustic), "exception": ""},
                actual_acoustic, Path(args.dictionary))
            if (not strict_preflight["acoustic_model_sha256"]
                    or not strict_preflight["dictionary_sha256"]):
                _strict_failed_manifest(output_dir, mfa=strict_preflight,
                                        expected_segments=strict_expected,
                                        expected_counts=strict_counts,
                                        en_segments=en_segments,
                                        reason="mfa_input_hash_failed")
                print("ERROR: strict MFA input hashing failed; MFA was not started")
                return 1
            manifest_path = produce_strict_ledgers(en_segments, ctc_dir, temp_dir / "en_aligned", output_dir,
                strict_preflight,
                expected_segments=strict_expected, expected_counts=strict_counts)
            if not _strict_manifest_succeeded(manifest_path):
                print("ERROR: strict English ledger manifest is missing, unreadable, or not successful")
                return 1
            print("  Wrote strict rejected ledgers (no alignable English segments)")
            return 0
        # Still produce output for G2P fallback
        n_done = collect_en_phones(en_segments, temp_dir / "en_aligned", output_dir)
        print(f"  Wrote fallback phone data for {n_done} stems (no MFA segments)")
        return 0

    # Step 3: Build dictionary
    print("Building English dictionary...")
    try:
        dict_path = build_en_dict(
            en_segments,
            Path(args.dictionary),
            Path(args.g2p_model) if args.g2p_model else Path(""),
            mfa_python, models_dir, temp_dir, args.g2p_timeout,
            strict=args.strict_provenance,
        )
    except StrictG2PError as exc:
        mfa = _strict_mfa_record(
            {"return_code": "not_run", "timed_out": False, "timeout_seconds": args.timeout,
             "command": [], "exception": str(exc)}, Path(args.acoustic_model), Path(args.dictionary))
        _strict_failed_manifest(output_dir, mfa=mfa, expected_segments=strict_expected,
                                expected_counts=strict_counts, en_segments=en_segments,
                                reason="g2p_failed")
        print("ERROR: strict English dictionary construction failed; MFA was not started")
        return 1

    # Resolve the *actual* MFA model (including a preferred extracted model)
    # and hash both launch inputs before any MFA subprocess can be started.
    strict_preflight: dict = {}
    if args.strict_provenance:
        actual_acoustic = _resolve_acoustic_model(args.acoustic_model, models_dir)
        strict_preflight = _strict_mfa_record(
            {"return_code": "not_run", "timed_out": False, "timeout_seconds": args.timeout,
             "command": [], "acoustic_model": str(actual_acoustic), "exception": ""},
            actual_acoustic, dict_path)
        if not strict_preflight["acoustic_model_sha256"] or not strict_preflight["dictionary_sha256"]:
            _strict_failed_manifest(output_dir, mfa=strict_preflight, expected_segments=strict_expected,
                                    expected_counts=strict_counts, en_segments=en_segments,
                                    reason="mfa_input_hash_failed")
            print("ERROR: strict MFA input hashing failed; MFA was not started")
            return 1

    # Step 4: Run English MFA
    outcome = run_en_mfa(
        en_corpus_dir, dict_path, args.acoustic_model,
        en_aligned_dir, en_work_dir,
        mfa_python, models_dir, args.num_jobs,
        beam=args.beam, retry_beam=args.retry_beam,
        fine_tune=args.fine_tune, timeout=args.timeout,
        strict=args.strict_provenance, dither=args.dither,
    )
    if args.strict_provenance:
        outcome["acoustic_model_sha256"] = strict_preflight["acoustic_model_sha256"]
        outcome["dictionary_sha256"] = strict_preflight["dictionary_sha256"]

    if outcome["return_code"] != 0:
        if args.strict_provenance:
            mfa = _strict_mfa_record(outcome, Path(outcome["acoustic_model"]), dict_path)
            _strict_failed_manifest(output_dir, mfa=mfa, expected_segments=strict_expected,
                                    expected_counts=strict_counts, en_segments=en_segments)
            print("ERROR: strict English MFA failed; no fallback artifacts were generated")
            return 1
        print("  WARNING: English MFA had issues — will use fallback for affected segments")

    if args.strict_provenance and outcome["return_code"] == 0:
        outcome["singleton_retries"] = retry_missing_en_segments(
            en_segments, en_corpus_dir, en_aligned_dir, dict_path,
            args.acoustic_model, temp_dir, mfa_python, models_dir,
            beam=args.singleton_retry_beam,
            retry_beam=args.singleton_retry_retry_beam,
            timeout=args.singleton_retry_timeout,
            limit=args.singleton_retry_limit, dither=args.dither)

    # Step 5: Collect results
    if args.strict_provenance:
        manifest_path = produce_strict_ledgers(en_segments, ctc_dir, en_aligned_dir, output_dir,
            _strict_mfa_record(outcome, Path(outcome["acoustic_model"]), dict_path),
            expected_segments=strict_expected, expected_counts=strict_counts)
        if not _strict_manifest_succeeded(manifest_path):
            print("ERROR: strict English ledger manifest is missing, unreadable, or not successful")
            return 1
        n_done = len(en_segments)
    else:
        n_done = collect_en_phones(en_segments, en_aligned_dir, output_dir)
    print(f"  Wrote English phone data for {n_done} stems")

    # Cleanup temp corpus (keep JSON output)
    if not args.strict_provenance:
        if en_corpus_dir.exists():
            shutil.rmtree(en_corpus_dir, ignore_errors=True)
        if en_work_dir.exists():
            shutil.rmtree(en_work_dir, ignore_errors=True)

    print(f"Done: English MFA alignment complete ({n_done} stems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
