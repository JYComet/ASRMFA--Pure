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
    find_mfa_python, get_mfa_env,
    is_english_phone as is_arpabet_phone,
    report_en_ipa_mappings,
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

STRICT_SCHEMA = "strict-en-mfa-v1"
STRICT_COUNT_KEYS = (
    "english_stems", "english_segments", "english_words",
    "verified_words", "rejected_words",
)
# MFA writes times in seconds.  This is deliberately small enough that an
# unowned/cross-word phone cannot be hidden by a generous boundary allowance.
STRICT_BOUNDARY_TOLERANCE_S = 0.003


class StrictG2PError(RuntimeError):
    """Raised when strict provenance cannot construct a complete dictionary."""


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
    return [{"word_id": f"{sid}:w{word['ordinal']}", "ctc_ordinal": word["ordinal"],
             "ctc_text": word["text"], "start": word["start"], "end": word["end"],
             "status": "rejected", "reason": reason, "mfa_word": None,
             "phones": [], "provenance": None}
            for word in segment["words"]]


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


def _strict_verified_words(sid: str, segment: dict, words: list[dict], phones: list[dict]) -> list[dict]:
    """Validate the complete source hierarchy and construct verified evidence."""
    source_words = _strict_source_words(words)
    ctc_words = segment["words"]
    if len(source_words) != len(ctc_words):
        raise ValueError("word_count_mismatch")
    for ctc_word, source_word in zip(ctc_words, source_words):
        if source_word["text"].casefold() != ctc_word["text"].casefold():
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
    for ctc_word, source_word, word_phones in zip(ctc_words, source_words, owned):
        if not word_phones:
            raise ValueError("phone_empty")
        if abs(word_phones[0]["start"] - source_word["start"]) > STRICT_BOUNDARY_TOLERANCE_S:
            raise ValueError("phone_start_coverage")
        if abs(word_phones[-1]["end"] - source_word["end"]) > STRICT_BOUNDARY_TOLERANCE_S:
            raise ValueError("phone_end_coverage")
        for previous, current in zip(word_phones, word_phones[1:]):
            if current["start"] - previous["end"] > STRICT_BOUNDARY_TOLERANCE_S:
                raise ValueError("phone_gap")
        item = {"word_id": f"{sid}:w{ctc_word['ordinal']}", "ctc_ordinal": ctc_word["ordinal"],
                "ctc_text": ctc_word["text"], "start": ctc_word["start"], "end": ctc_word["end"],
                "status": "verified", "reason": "",
                "mfa_word": {"ordinal": source_word["ordinal"], "text": source_word["text"],
                             "start": source_word["start"], "end": source_word["end"]},
                "phones": [{"ordinal": position, "label": phone["text"], "start": phone["start"],
                            "end": phone["end"], "mfa_phone_ordinal": phone["ordinal"]}
                           for position, phone in enumerate(word_phones)],
                "provenance": "english_mfa_textgrid"}
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
    """Check the global denominator before a producer may claim success."""
    rejected_ids = [item.get("id") for item in rejected]
    if len(expected) != len(set(expected)) or len(produced) != len(set(produced)):
        return False
    if len(rejected_ids) != len(set(rejected_ids)) or set(produced) & set(rejected_ids):
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
    payload = {"schema": STRICT_SCHEMA, "status": status, "strict_provenance": True,
               "mfa": mfa, "expected_segments": expected_segments,
               "produced_segments": produced_segments, "rejected_segments": rejected_segments,
               "stem_ledgers": stem_ledgers, "counts": _strict_counts(counts), "reason": reason}
    path = output_dir / "en_alignment_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _strict_manifest_succeeded(path: Path) -> bool:
    """Accept only the producer's complete, known-success strict manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("schema") == STRICT_SCHEMA and payload.get("status") == "success"


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
                  "ctc_textgrid_sha256": ctc_hash,
                  "segments": []}
        for seg in segments:
            ordinal = int(seg.get("segment_ordinal", seg.get("seg_idx", 0)))
            sid = f"{stem}:s{ordinal}"; discovered_expected.append(sid)
            record = {"segment_id": sid, "segment_ordinal": ordinal, "status": "verified",
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
                    record["words"] = _strict_verified_words(sid, seg, mw, mp)
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
    return write_strict_manifest(output_dir, "success" if consistent else "failed", mfa=mfa, expected_segments=expected,
        produced_segments=produced, rejected_segments=rejected, stem_ledgers=ledgers,
        counts=counts, reason="" if consistent else "strict_manifest_inconsistent")


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
                intervals.append({"xmin": pending_xmin, "xmax": pending_xmax, "text": text})
            pending_xmin = pending_xmax = None
            in_interval = False

    return intervals


def find_english_segments(ctc_dir: Path, stems: list[str],
                          max_gap_s: float = 0.35) -> dict[str, list[dict]]:
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

        # Preserve the complete words-tier ordinal.  English spans may only
        # merge when adjacent ordinal entries are English; a Chinese/NVV/punct
        # interval is a hard separator regardless of acoustic gap.
        en_words = []
        for ordinal, iv in enumerate(intervals):
            text = iv["text"].strip()
            if not text or text in SILENCE_LABELS or text in ("", "<eps>"):
                continue
            if is_english_token(text):
                en_words.append({"text": text, "start": iv["xmin"], "end": iv["xmax"], "ordinal": ordinal})

        if not en_words:
            continue

        # Merge consecutive English words into segments
        segments = []
        seg_words = [en_words[0]]
        seg_start = en_words[0]["start"]
        seg_end = en_words[0]["end"]

        for w in en_words[1:]:
            gap = w["start"] - seg_end
            if w["ordinal"] == seg_words[-1]["ordinal"] + 1 and gap < max_gap_s:
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
        seg_start_raw = seg["seg_start"]
        seg_end_raw = seg["seg_end"]
        seg_dur = seg_end_raw - seg_start_raw

        # Skip segments that are too short for MFA
        if seg_dur < min_dur_s:
            seg["skipped"] = True
            if strict:
                seg["reject_reason"] = "segment_too_short"
            seg["offset"] = seg_start_raw
            valid_segments.append(seg)
            continue

        # Add padding
        seg_start_padded = max(0.0, seg_start_raw - padding_s)
        seg_end_padded = min(total_dur, seg_end_raw + padding_s)

        start_sample = int(seg_start_padded * sr)
        end_sample = int(seg_end_padded * sr)

        if end_sample <= start_sample + int(0.05 * sr):
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
        lab_text = " ".join(w["text"] for w in seg["words"])
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
                word = w["text"].strip().lower()
                if word and word.isalpha():
                    all_words.add(word)

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
                    base_words.add(word)

    oov_words = sorted(all_words - base_words)
    # Always start with a clean (comment-free) copy of the base dictionary
    combined = temp_dir / "en_combined.dict"
    if not oov_words:
        with open(combined, 'w', encoding='utf-8') as outf:
            outf.write(base_dict_text)
        return combined

    def merge_dictionary(g2p_text: str = "") -> Path:
        with combined.open("w", encoding="utf-8") as outf:
            if base_dict_text:
                outf.write(base_dict_text)
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
        return merge_dictionary(cached_text)

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
    merge_dictionary(g2p_text)

    # Save to cache for future runs
    import shutil
    shutil.copy(g2p_output, cached_dict)

    print(f"  Combined dictionary: {combined} ({len(all_words)} words)")
    return combined


def run_en_mfa(corpus_dir: Path, dict_path: Path, acoustic_model: str,
               output_dir: Path, temp_dir: Path, mfa_python: Path,
               models_dir: Path, num_jobs: int = 4,
               beam: int = 10, retry_beam: int = 40,
               fine_tune: bool = False, timeout: int = 1800,
               strict: bool = False) -> dict:
    """Run MFA align and retain an exception-safe, auditable outcome."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use extracted model if available
    acoustic_arg = str(_resolve_acoustic_model(acoustic_model, models_dir))

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
    ]
    if not strict:
        mfa_args.append("--clean")
    mfa_args.append("--overwrite")
    if fine_tune:
        mfa_args.append("--fine_tune")

    print(f"  Running English MFA align ({len(list(corpus_dir.glob('*.wav')))} segments)...")
    command = [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa"] + mfa_args
    result = {"return_code": None, "timed_out": False, "timeout_seconds": timeout,
              "command": command, "acoustic_model": acoustic_arg, "exception": ""}
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
        run_kwargs = {"env": get_mfa_env(mfa_python, models_dir), "timeout": timeout}
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


def parse_en_textgrid(tg_path: Path) -> dict:
    """Parse English MFA TextGrid into a simple structure.

    Returns {words: [{text, start, end, phones: [{phone, start, end}]}]}
    Assumes tier 0 = words, tier 1 = phones.
    """
    lines = tg_path.read_text(encoding="utf-8").splitlines()
    tiers_data: list[list[dict]] = []  # each tier = list of {xmin, xmax, text}
    current_tier: list[dict] = []
    in_interval = False
    pending_xmin = pending_xmax = None
    in_items = False

    for raw in lines:
        line = raw.strip()
        if line == "item []:":
            in_items = True
        elif in_items and line.startswith("item ["):
            if current_tier:
                tiers_data.append(current_tier)
                current_tier = []
            in_interval = False
        elif in_items and line.startswith("intervals ["):
            in_interval = True
            pending_xmin = pending_xmax = None
        elif in_interval and line.startswith("xmin = "):
            pending_xmin = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("xmax = "):
            pending_xmax = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("text = "):
            text = line.split("=", 1)[1].strip().strip('"')
            if pending_xmin is not None and pending_xmax is not None:
                current_tier.append({"xmin": pending_xmin, "xmax": pending_xmax, "text": text})
            pending_xmin = pending_xmax = None
            in_interval = False

    if current_tier:
        tiers_data.append(current_tier)

    if len(tiers_data) < 2:
        return {"words": []}

    words_tier = tiers_data[0]
    phones_tier = tiers_data[1]

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

        for seg in segments:
            seg_idx = seg["seg_idx"]

            if seg.get("skipped"):
                # G2P fallback: equal-duration split for CTC word interval
                for w in seg["words"]:
                    en_data.append({
                        "seg_idx": seg_idx,
                        "offset": 0.0,
                        "word_text": w["text"],
                        "word_start": w["start"],
                        "word_end": w["end"],
                        "phones": [],  # empty -> postprocessing uses equal split
                    })
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
                        en_data.append({
                            "seg_idx": seg_idx,
                            "offset": 0.0,
                            "word_text": w["text"],
                            "word_start": w["start"],
                            "word_end": w["end"],
                            "phones": [],
                        })
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
                ctc_text_lower = ctc_w["text"].strip().lower()
                # Find matching MFA word
                matched = None
                while mfa_idx < len(mfa_words):
                    mw = mfa_words[mfa_idx]
                    mfa_idx += 1
                    if mw["text"].strip().lower().rstrip('012') == ctc_text_lower.rstrip('012'):
                        matched = mw
                        break
                    # If MFA word doesn't match, check next (may have been merged/skipped)

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

                    en_data.append({
                        "seg_idx": seg_idx,
                        "offset": round(offset, 4),
                        "word_text": ctc_w["text"],
                        "word_start": ctc_w["start"],  # CTC word boundary (original time)
                        "word_end": ctc_w["end"],
                        "en_word_start": round(offset + matched["start"], 4),  # English MFA word boundary
                        "en_word_end": round(offset + matched["end"], 4),
                        "phones": phones_abs,
                    })
                else:
                    # No match found — fall back
                    en_data.append({
                        "seg_idx": seg_idx,
                        "offset": round(offset, 4),
                        "word_text": ctc_w["text"],
                        "word_start": ctc_w["start"],
                        "word_end": ctc_w["end"],
                        "phones": [],
                    })

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
    parser.add_argument("--padding-ms", type=float, default=50.0,
                        help="Padding around English segments (ms)")
    parser.add_argument("--min-segment-dur-ms", type=float, default=150.0,
                        help="Minimum segment duration for MFA (ms)")
    parser.add_argument("--max-gap-merge-s", type=float, default=0.35,
                        help="Max gap between consecutive English words to merge (s)")
    parser.add_argument("--beam", type=int, default=10,
                        help="MFA Viterbi beam width for English alignment")
    parser.add_argument("--retry-beam", type=int, default=40,
                        help="MFA retry beam width for English alignment")
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
        fine_tune=args.fine_tune, timeout=args.timeout, strict=args.strict_provenance,
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
