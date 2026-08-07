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
)
from postprocess_textgrids import parse_textgrid  # noqa: E402

POLICY_VERSION = "strict-ok-v3.1"
EN_PROVENANCE_SCHEMA = "strict-en-mfa-v1"
TIER_NAMES = ["raw_text", "pinyin", "hanzi", "words", "pinyin_phones"]
EPS = 0.003
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
    # Fields below are normal provenance/count diagnostics.  Any other truthy
    # field is an existing QC positive and vetoes strict-ok without treating
    # routine timing/count metadata as a warning.
    allowed = {
        "stem", "status", "output", "textgrid_duration", "reference_source",
        "reference_text_authoritative", "reference_coverage", "warnings",
        "hard_integrity_reasons", "filter_reasons", "alignment_issues",
        "english_provenance",
    }
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
            if any(phone[3:].lower() == token.lower() for phone in owned if phone.startswith("en:")):
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
            if any(phone.lower() == label.lower() for phone in owned):
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
    lexical_words = [iv for iv in source_words if iv.text.strip() and not is_silence(iv.text.strip())]
    previous = -math.inf
    for iv in lexical_words:
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
        matches = [index for index, word in enumerate(lexical_words)
                   if phone.xmin >= word.xmin - EPS and phone.xmax <= word.xmax + EPS]
        if len(matches) != 1:
            raise ValueError("source_interval_invalid")
        owners[matches[0]].append((source_ordinal, phone))
    result: list[dict] = []
    for ordinal, (word, phones) in enumerate(zip(lexical_words, owners)):
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
        if (len(verified_words) != len(final_words)
                or [item["ledger"].get("ctc_ordinal") for item in verified_words] != sorted(ctc_english)
                or len(ctc_english) != len(final_words)):
            return ["english_word_unmatched"], None
        final_phones = final_tg.tiers[4].intervals
        matched_en_phone_indices: set[int] = set()
        for final_word, evidence in zip(final_words, verified_words):
            record, source_word = evidence["ledger"], evidence["source"]
            ordinal = record.get("ctc_ordinal")
            if (final_word.text.strip().casefold() != record.get("ctc_text", "").casefold()
                    or final_word.text.strip().casefold() != ctc_english[ordinal].casefold()):
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


def audit(args: argparse.Namespace) -> tuple[dict, bool]:
    ctc_dir = args.ctc_dir
    expected = {path.stem for path in ctc_dir.glob("*.lab")}
    output = {path.stem: path for path in args.output_dir.glob("*.TextGrid")}
    filtered = {path.stem: path for path in args.filtered_dir.glob("*.TextGrid")}
    report_rows, global_reasons = _report_index(args.report)
    reference_index, reference_errors = _reference_index(args.reference_dir, expected)
    global_reasons.extend(reference_errors)
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
        "expected_stems": sorted(expected),
        "ok": [],
        "rejected": {},
    }
    overlap = set(output) & set(filtered)
    if overlap:
        global_reasons.append("output_filtered_overlap")
    if set(output) | set(filtered) != expected:
        global_reasons.append("output_filtered_expected_not_conserved")
    if set(report_rows) != expected:
        global_reasons.append("report_expected_not_conserved")

    for stem in sorted(expected):
        reasons: list[str] = []
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
        if not reasons:
            try:
                tg = _strict_parse(candidate)
                reasons.extend(_numeric_reasons(tg, _wav_duration(wav)))
                reasons.extend(_sp1_reasons(tg))
                reasons.extend(_content_reasons(tg, reference))
                provenance_reasons, provenance_evidence = _english_provenance_reasons(
                    stem, tg, ctc_dir, args, english_manifest)
                reasons.extend(provenance_reasons)
            except Exception as exc:
                reasons.append(f"invalid_final_textgrid:{exc}")
            reasons.extend(f"ctc_bundle:{item}" for item in validate_ctc_transcript_bundle(args.ctc_dir, stem))
            reasons.extend(_aligned_reasons(aligned, reference))
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
    manifest["global_reasons"] = sorted(set(global_reasons))
    return manifest, not global_reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and isolate strict-ok v3.1 MFA output.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--ctc-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--aligned-dir", type=Path, required=True)
    parser.add_argument("--en-phones-dir", type=Path, required=True,
                        help="strict-en-mfa-v1 per-stem ledger directory")
    parser.add_argument("--en-aligned-dir", type=Path, required=True,
                        help="retained English MFA TextGrid root")
    parser.add_argument("--en-manifest", type=Path, required=True,
                        help="strict-en-mfa-v1 global run manifest")
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
