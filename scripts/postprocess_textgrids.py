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
import hashlib
import json
import math
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
    is_english_phone, is_english_vowel_phone, is_english_consonant_phone,
    en_ipa_to_arpabet, apply_arpabet_stress, align_sequences,
    is_silence, EN_PHONE_PREFIX,
)

SHORT_PAUSE_PUNCT = set("，、：；,")
LONG_PAUSE_PUNCT = set("。？！…!?.")
SHORT_PAUSE_TOKEN = "[PAUSE]"
LONG_PAUSE_TOKEN = "<PAUSE>"

# This schema is intentionally duplicated rather than imported from
# align_english_mfa: post-processing must be able to reject malformed or
# legacy producer output without creating an import cycle.
STRICT_EN_MFA_SCHEMA = "strict-en-mfa-v1"
_STRICT_EN_SILENCE = {"sil", "sp", "spn", "<eps>"}

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
                _min_seg = 0.030        # floor per segment
                if word_dur >= _min_seg * 2:
                    split = w_iv.xmin + max(_min_seg, word_dur * _init_frac)
                    split = min(split, w_iv.xmax - _min_seg)
                else:
                    split = w_iv.xmin + word_dur * 0.5
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
                    _init_frac_mfa = (_init_end - w_iv.xmin) / max(word_dur, 0.001)
                    # Regr. Case 44: phonetically-motivated upper bound on
                    # initial fraction.  MFA sometimes places the init→final
                    # boundary too far into the word (e.g. h→ao at 70%).
                    _init_max_frac = _INIT_MAX_FRAC.get(dict_phones[0], 0.55)
                    if _init_frac_mfa <= _init_max_frac or word_dur <= 0.060:
                        use_mfa_split = True
                        # Snap initial start to word start (Regression Case 7)
                        new_intervals.append(Interval(w_iv.xmin, _init_end, dict_phones[0]))
                        final_start = word_phones[1][0]
                        final_label = " ".join(dict_phones[1:]) if len(dict_phones) > 2 else dict_phones[1]
                        new_intervals.append(Interval(final_start, w_iv.xmax, final_label))

                if not use_mfa_split:
                    # Proportional split fallback: dict_phones >= 2 but
                    # MFA under-produced or boundary was rejected.
                    # Regression Case 26 (MISSING_FINAL) + Case 43.
                    _init_frac = _INIT_FRAC.get(dict_phones[0], 0.35)
                    _min_seg = 0.030        # floor per segment
                    if word_dur >= _min_seg * 2:
                        split = w_iv.xmin + max(_min_seg, word_dur * _init_frac)
                        split = min(split, w_iv.xmax - _min_seg)
                    else:
                        split = w_iv.xmin + word_dur * 0.5
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


def _count_internal_pp_gaps(pp_tier: Tier | None, words_tier: Tier | None,
                            threshold_s: float = 0.010) -> int:
    """Count pinyin-phone gaps that fall inside one content-word interval.

    ``pinyin_phones`` is a sparse acoustic tier: a real pause between words
    may have no phone interval after later boundary caps.  That is not a tier
    discontinuity.  Only an uncovered gap inside one non-silence word means
    the word's phone reconstruction lost coverage.
    """
    if pp_tier is None or words_tier is None:
        return 0

    content_ranges = [
        (iv.xmin, iv.xmax)
        for iv in words_tier.intervals
        if (iv.text.strip() and not is_silence(iv.text)
            and not is_english_token(iv.text.strip()))
    ]
    gaps = 0
    for left, right in zip(pp_tier.intervals, pp_tier.intervals[1:]):
        gap_start, gap_end = left.xmax, right.xmin
        if gap_end - gap_start <= threshold_s:
            continue
        if any(
            word_start <= gap_start + 0.001
            and gap_end <= word_end + 0.001
            for word_start, word_end in content_ranges
        ):
            gaps += 1
    return gaps


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
        gaps = sum(
            1
            for left, right in zip(tier.intervals, tier.intervals[1:])
            if right.xmin - left.xmax > threshold_s
        )
        if gaps > len(tier.intervals) * 0.10:
            discontinuities.append(f"{tier.name}({gaps}/{len(tier.intervals)})")

    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if pp_tier is not None and len(pp_tier.intervals) >= 5:
        gaps = _count_internal_pp_gaps(pp_tier, words_tier, threshold_s)
        if gaps > len(pp_tier.intervals) * 0.10:
            discontinuities.append(f"{pp_tier.name}({gaps}/{len(pp_tier.intervals)})")
    return discontinuities


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
    """Merge sp0 gaps unconditionally; flag sp1-3 gaps for filtering.

    After the punctuation–silence cross-check, any silence between words that
    has *no* corresponding punctuation is an unexpected pause:
      - ``<sp0>`` (< 0.2 s)  -> merge unconditionally (into adjacent
        punctuation when present, otherwise into the previous word).
        Short gaps have no semantic meaning in any context.
      - ``<sp1-3>`` (≥ 0.2 s) -> return as filter reasons (when no punct
        is present; <sp1-3> after punct is handled by the absorb pass).
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
    for k in range(1, n):
        lo = py_word_idx[k - 1] + 1
        hi = py_word_idx[k]
        gap_punct[k] = any(is_punct(pinyin_tokens[i]) for i in range(lo, hi))

    filter_reasons = []

    # Build delete markers for sp0 merges (avoid O(n²) list deletion)
    to_delete_words: set[int] = set()
    to_delete_phones: set[int] = set()
    to_delete_pp: set[int] = set()
    merge_ops: list[tuple[int, int, float]] = []  # (word_idx, sil_idx, sil_xmax)

    for k in range(1, n):
        sil_label = gap_sil[k]
        has_punct = gap_punct[k]
        if sil_label is None:
            continue
        if sil_label == "<sp0>":
            pass  # Always merge <sp0> regardless of punctuation — 15 ms
            # gaps have no semantic meaning in any context.
        elif has_punct:
            continue  # <sp1-3> + punct: skip (handled by absorb phase later)
        elif sil_label in ("<sp1>", "<sp2>", "<sp3>"):
            prev_text = word_items[tg_word_idx[k - 1]][0]
            next_text = word_items[tg_word_idx[k]][0]
            if not (is_english_token(prev_text) or is_english_token(next_text)
                    or is_nvv_token(prev_text) or is_nvv_token(next_text)):
                # Regular Chinese words: flag the unexpected silence
                # but still merge it into the previous word.  The silence
                # IS unexpected (hence the filter flag), but leaving it
                # in the words tier creates a mid_sp hit downstream.
                filter_reasons.append("unexpected_silence")
                # Fall through to the merge block below.
            else:
                # English/NVV-adjacent gaps: MFA artifacts (MFA can't
                # model English phones, inserts spn).  Skip.
                continue

        # <sp0>: merge into previous word (or adjacent punctuation).
        prev_word_idx = tg_word_idx[k - 1]
        sil_idx = None
        for j in range(prev_word_idx + 1, tg_word_idx[k]):
            if word_items[j][1]:
                sil_idx = j
                break
        if sil_idx is None:
            continue

        sil_iv = words_tier.intervals[sil_idx]

        # When punct exists, absorb <sp0> into the adjacent punctuation
        # rather than the previous word (extending the word over punct
        # would create an overlap).
        if has_punct:
            # Find the punctuation interval nearest to the <sp0>.
            punct_idx = sil_idx - 1
            while punct_idx > prev_word_idx:
                if is_punct(word_items[punct_idx][0]):
                    break
                punct_idx -= 1
            if punct_idx > prev_word_idx and is_punct(word_items[punct_idx][0]):
                # Extend punctuation to absorb the <sp0>.
                words_tier.intervals[punct_idx].xmax = sil_iv.xmax
                to_delete_words.add(sil_idx)
                # Clean up matching silence in phones & pp tiers
                # (punct has no phone entries, just delete the sil).
                for pi, p in enumerate(phones_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_phones.add(pi)
                        break
                for pi, p in enumerate(pp_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_pp.add(pi)
                        break
            else:
                # Fallback: merge into previous word (no punct found adjacent).
                merge_ops.append((prev_word_idx, sil_idx, sil_iv.xmax))
                to_delete_words.add(sil_idx)
                for pi, p in enumerate(phones_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_phones.add(pi)
                        break
                for pi, p in enumerate(pp_tier.intervals):
                    if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                       and abs(p.xmax - sil_iv.xmax) < 0.01:
                        to_delete_pp.add(pi)
                        break
        else:
            # No punct — original behaviour: merge into previous word.
            merge_ops.append((prev_word_idx, sil_idx, sil_iv.xmax))
            to_delete_words.add(sil_idx)

            # Find matching silence in phones & pp tiers
            for pi, p in enumerate(phones_tier.intervals):
                if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                   and abs(p.xmax - sil_iv.xmax) < 0.01:
                    to_delete_phones.add(pi)
                    break
            for pi, p in enumerate(pp_tier.intervals):
                if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 \
                   and abs(p.xmax - sil_iv.xmax) < 0.01:
                    to_delete_pp.add(pi)
                    break

    # Apply merge ops (extend word + last phone)
    for prev_wi, sil_idx, sil_xmax in merge_ops:
        prev_w = words_tier.intervals[prev_wi]
        prev_w.xmax = sil_xmax
        # Extend last phone of previous word
        for pi in range(len(phones_tier.intervals) - 1, -1, -1):
            p = phones_tier.intervals[pi]
            if not is_silence(p.text) and p.text != 'spn' \
               and abs(p.xmax - words_tier.intervals[sil_idx].xmin) < 0.01:
                p.xmax = sil_xmax
                if pi + 1 < len(phones_tier.intervals):
                    phones_tier.intervals[pi + 1].xmin = sil_xmax
                break

    # One-pass filter: keep non-deleted intervals (O(n) instead of O(n²))
    if to_delete_words:
        words_tier.intervals = [iv for i, iv in enumerate(words_tier.intervals)
                                if i not in to_delete_words]
    if to_delete_phones:
        phones_tier.intervals = [iv for i, iv in enumerate(phones_tier.intervals)
                                 if i not in to_delete_phones]
    if to_delete_pp:
        pp_tier.intervals = [iv for i, iv in enumerate(pp_tier.intervals)
                             if i not in to_delete_pp]

    # Clean up zero-duration remnants in all tiers
    for tier in (words_tier, phones_tier, pp_tier):
        tier.intervals = [iv for iv in tier.intervals
                          if iv.duration > 0.001 or not iv.text.strip()]

    return filter_reasons


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

    intervals = list(words_tier.intervals)
    to_delete_words: set[int] = set()
    to_delete_phones: set[int] = set()
    to_delete_pp: set[int] = set()

    for i in range(len(intervals)):
        if not is_nvv_token(intervals[i].text):
            continue

        # Absorb trailing punct + silence chain.
        j = i + 1
        absorbed_sil_ranges: list[tuple[float, float]] = []
        while j < len(intervals):
            text = intervals[j].text.strip()
            if is_punct(text):
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
        if overlap <= 0.0005:         # sub-0.5 ms — float noise, skip
            continue

        cur_text = cur.text.strip() if cur.text else ""
        nxt_text = nxt.text.strip() if nxt.text else ""

        cur_is_content = (cur_text and not is_punct(cur_text)
                          and not is_silence(cur_text))
        nxt_is_content = (nxt_text and not is_punct(nxt_text)
                          and not is_silence(nxt_text))
        cur_is_en_nvv = is_english_token(cur_text) or is_nvv_token(cur_text)
        nxt_is_en_nvv = is_english_token(nxt_text) or is_nvv_token(nxt_text)

        # ── Two content words with mild overlap (incl. English/NVV adjacent) ──
        # Regr. Case 38: when one side is English/NVV (no MFA acoustic model),
        # clip that side to the content word's boundary.
        if cur_is_content and nxt_is_content and overlap < 0.030:
            if cur_is_en_nvv and not nxt_is_en_nvv:
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


def _absorb_tiny_gaps(words_tier: Tier, max_gap_s: float = 0.030) -> Tier:
    """Absorb sub-frame gaps (< 30 ms) between consecutive content words.

    Regr. Case 39: MFA frame-level precision gaps (5-30 ms) that survive
    _snap_to_ctc's gap absorption pass (e.g. because they were introduced
    later by _inject_punctuation or Phase 4 operations) are absorbed into
    adjacent words.  Only targets gaps between two non-punct, non-silence
    content intervals.
    """
    intervals = list(words_tier.intervals)
    n = len(intervals)
    to_delete: set[int] = set()

    for i in range(n - 1):
        if i in to_delete:
            continue
        cur = intervals[i]
        nxt = intervals[i + 1]
        cur_text = cur.text.strip() if cur.text else ""
        nxt_text = nxt.text.strip() if nxt.text else ""

        # Only absorb gaps between two content words (not punct, not silence)
        if not cur_text or not nxt_text:
            continue
        if is_punct(cur_text) or is_punct(nxt_text):
            continue
        if is_silence(cur_text):
            # Silence gap between two content words — absorb if tiny
            if cur.duration < max_gap_s:
                # Absorb into the longer neighbouring word
                prev_word = intervals[i - 1] if i > 0 else None
                if (prev_word and not is_silence(prev_word.text)
                        and prev_word.duration >= nxt.duration):
                    intervals[i - 1] = Interval(prev_word.xmin, nxt.xmin, prev_word.text)
                else:
                    intervals[i + 1] = Interval(cur.xmin, nxt.xmax, nxt.text)
                to_delete.add(i)
        elif is_silence(nxt_text):
            continue  # word→silence: not a gap, silence is intentional

    intervals = [iv for idx, iv in enumerate(intervals) if idx not in to_delete]
    # Remove zero-duration remnants
    intervals = [iv for iv in intervals if iv.duration > 0.001]
    return Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)


def _sync_derived_tiers(textgrid: TextGrid, ipa_to_pinyin: dict[str, str],
                        pinyin_dict: dict[str, list[str]] | None = None,
                        raw_text: str = "",
                        en_mfa_windows: dict[tuple[str, float], tuple[float, float]] | None = None,
                        report_warnings: list[str] | None = None) -> None:
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

    # 0. Absorb frame-precision gaps before rebuilding derived tiers.
    #    Regr. Case 39: gaps < 30 ms are MFA alignment residuals, not
    #    real silences.  Absorb them now so hanzi + pinyin_phones don't
    #    inherit unnecessary gaps.
    words_tier = _absorb_tiny_gaps(words_tier)
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
            hanzi_tier = _build_hanzi_tier(words_tier, raw_text,
                                            report_warnings or [])
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
    new_words = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)
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
    """Absorb trailing ``<spN>`` silence intervals into preceding punctuation.

    Punctuation is silent by nature — the silence that follows it is its
    realised duration.  This is the **fallback** pass: it handles residual
    ``<spN>`` after punctuation that was not already absorbed by an NVV
    in :func:`absorb_nvv_trailing`.

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

    i = 0
    while i < len(intervals) - 1:
        cur_text = intervals[i].text.strip()
        next_text = intervals[i + 1].text.strip()
        if is_punct(cur_text) and is_silence(next_text) and next_text:
            sil_iv = intervals[i + 1]
            # Extend punctuation to absorb the silence duration.
            intervals[i] = Interval(intervals[i].xmin, sil_iv.xmax, intervals[i].text)
            to_delete_words.add(i + 1)

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


def _finalise_textgrid(textgrid: TextGrid, raw_text: str, pinyin_text: str,
                       args, warnings: list | None = None) -> TextGrid:
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
    hanzi_tier = _build_hanzi_tier(words_tier, raw_text, warnings) if words_tier else None

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
                # The later coverage audit will reject a genuinely truncated
                # lab; do not invent a tone here.
                rendered.append(unit)
            else:
                rendered.append(source_cjk[cjk_index])
                cjk_index += 1
        elif is_nvv_token(unit):
            rendered.append(f"<{unit.strip('<>').upper()}>")
        else:
            rendered.append(unit)
    return "<sp1> " + " ".join(rendered)


def _restore_reference_punctuation(words_tier: Tier, reference_text: str,
                                   punct_entries: list[dict] | None = None) -> int:
    """Make the words tier's punctuation sequence equal the authority.

    CTC punctuation is an alignment anchor, not lexical authority.  When a
    broad CTC pause has swallowed a reference comma or produced an extra
    terminal full stop, rebuilding only from CTC punctuation can silently
    change the transcript.  This pass keeps lexical word intervals, removes
    the current punctuation projection, and restores the reference sequence.
    Anchor timing is used only when it lies in the local word gap; otherwise
    the local gap/boundary is used so a long pause cannot erase neighbouring
    words.
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
        return 0

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

        # Reuse an existing interval of the same ordinal first.  Otherwise
        # use the corresponding CTC anchor only if it is local to the gap.
        occurrence = sum(1 for c, _ in ref_puncts[:ref_index] if c == char)
        candidates = anchors_by_char.get(char, [])
        anchor = candidates[occurrence] if occurrence < len(candidates) else None
        if anchor is not None and anchor[0] >= gap_start - 0.01 and anchor[1] <= gap_end + 0.01:
            start, end = anchor
        elif gap_end - gap_start > 0.001:
            start, end = gap_start, gap_end
        else:
            center = gap_start
            width = min(0.060, max(0.010, words_tier.xmax - words_tier.xmin))
            start, end = center - width / 2.0, center + width / 2.0
            if prev is not None:
                start = max(start, prev.xmin + 0.001)
            if nxt is not None:
                end = min(end, nxt.xmax - 0.001)
        start = max(words_tier.xmin, start)
        end = min(words_tier.xmax, end)
        if end <= start + 0.001:
            continue
        punctuation.append((mapped_boundary, Interval(start, end, char)))

    # Clip lexical intervals around the restored punctuation.  The labels and
    # English MFA word instances stay untouched; only their ownership ranges
    # are shortened where a punctuation anchor crosses a word boundary.
    for boundary, punct in punctuation:
        for index, iv in enumerate(lexical):
            if index < boundary and iv.xmax > punct.xmin:
                lexical[index] = Interval(iv.xmin, min(iv.xmax, punct.xmin), iv.text)
            elif index >= boundary and iv.xmin < punct.xmax:
                lexical[index] = Interval(max(iv.xmin, punct.xmax), iv.xmax, iv.text)
    lexical = [iv for iv in lexical if iv.xmax > iv.xmin + 0.001]

    # Preserve silence intervals and replace only punctuation/lexical content.
    silences = [iv for iv in current if is_silence(iv.text) or not iv.text.strip()]
    words_tier.intervals = sorted(lexical + silences + [iv for _, iv in punctuation],
                                  key=lambda iv: (iv.xmin, iv.xmax, iv.text))
    return len(punctuation)


def _clip_pinyin_phones_to_words(pp_tier: Tier, words_tier: Tier) -> int:
    """Keep each derived phone inside its strongest-overlap word owner."""
    words = [iv for iv in words_tier.intervals if iv.text.strip()]
    changed = 0
    clipped: list[Interval] = []
    for phone in pp_tier.intervals:
        if not phone.text.strip() or is_silence(phone.text):
            clipped.append(phone)
            continue
        owners = [word for word in words
                  if phone.xmax > word.xmin and phone.xmin < word.xmax]
        if not owners:
            clipped.append(phone)
            continue
        owner = max(owners, key=lambda word:
                    max(0.0, min(phone.xmax, word.xmax)
                        - max(phone.xmin, word.xmin)))
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

    # Never infer an English reference word from a pinyin syllable.  Named
    # variants such as rui4+ya4 -> ria are handled only by the explicit,
    # reference-bound RIA canonicalization path before English MFA.
    if len(c) >= 2 and c[-1].isdigit() and c[:-1].isalpha():
        return False

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


def _build_hanzi_tier(words_tier: Tier, raw_text: str,
                      warnings: list | None = None) -> Tier:
    """Build the *hanzi* tier by sequential mapping of word tokens to
    reference text units.

    **CJK characters**: each pinyin-syllable token in *words_tier*
    consumes the next unused CJK character from the reference text
    in order.  This mapping is purely positional — it does not depend
    on pypinyin tone accuracy or any dictionary.

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

    # Track pinyin-syllable count for defensive mismatch detection
    pinyin_count = 0

    for word_index, iv in enumerate(words_tier.intervals):
        token = iv.text.strip()

        if word_index in hidden_hyphen_fragments:
            # Keep the strict-English word instance and its phone ledger, but
            # render a split hyphenated reference spelling only once.
            intervals.append(Interval(iv.xmin, iv.xmax, ""))
            continue

        # Silence → keep silence label
        if is_silence(iv.text) or not token:
            intervals.append(Interval(iv.xmin, iv.xmax,
                                      silence_label(iv.duration)))
            continue

        # Punctuation → pass through, consume no cursor
        if is_punct(iv.text):
            intervals.append(Interval(iv.xmin, iv.xmax, iv.text))
            continue

        # ── Pinyin syllable → consume next CJK character ──
        if is_pinyin_syllable(token):
            pinyin_count += 1
            if cjk_idx < len(ref_cjk):
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
                    break
                # CTC fragments need not be contiguous inside a hyphenated
                # spelling (``kp`` is a subsequence of ``kpop``).
                probe_iter = iter(target_compact)
                if not all(any(ch == target_ch for target_ch in probe_iter)
                           for ch in compact):
                    break

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
) -> tuple[dict, list[str]]:
    """Assess hard lexical integrity independently of optional acoustic QC."""
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
    if unknown_source_count:
        reasons.append("mfa_unknown_source")

    coverage = {
        "reference_source": reference_source,
        "reference_cjk_count": len(reference_cjk),
        "pinyin_token_count": len(pinyin_tokens),
        "assigned_cjk_count": len(hanzi_cjk),
        "missing_cjk_count": max(0, len(reference_cjk) - len(pinyin_tokens)),
        "extra_pinyin_count": max(0, len(pinyin_tokens) - len(reference_cjk)),
        "unknown_source_count": unknown_source_count,
        "exact_cjk_sequence": reference_cjk == hanzi_cjk,
        "reference_cjk": reference_cjk,
        "hanzi_cjk": hanzi_cjk,
    }
    return coverage, list(dict.fromkeys(reasons))


def _normalize_word_spellings(words_tier: Tier, raw_text: str) -> None:
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
        if u.isascii() and u.isalpha() and len(u) >= 2 and not is_nvv_token(u):
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


def _word_rms(audio, sr: int, xmin: float, xmax: float) -> float:
    """Mean absolute amplitude in time slice [xmin, xmax)."""
    import numpy as _np
    s = max(0, int(xmin * sr))
    e = min(len(audio), int(xmax * sr))
    if e <= s:
        return 0.0
    return float(_np.mean(_np.abs(audio[s:e])))


def _noise_floor(audio, sr: int, bottom_pct: float = 0.10) -> float:
    """Estimate noise floor from quietest *bottom_pct* of 5ms frames."""
    import numpy as _np
    rms, _ = _frame_rms_vec(audio, sr, frame_ms=5.0)
    if len(rms) == 0:
        return 0.0
    k = max(1, int(len(rms) * bottom_pct))
    return float(_np.partition(rms, k)[k])


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

    noise_floor = 1e-6
    has_audio = audio is not None or (wav_path and wav_path.exists())
    if has_audio:
        if audio is None:
            audio, sr = load_audio(wav_path)
        try:
            sil_energies = []
            for p_iv in phones.intervals:
                if not is_silence(p_iv.text) and p_iv.text != 'spn':
                    continue
                ss = max(0, int(p_iv.xmin * sr))
                es = min(len(audio), int(p_iv.xmax * sr))
                if es - ss > 0:
                    seg = [abs(v) for v in audio[ss:es]]
                    if seg:
                        sil_energies.append(sum(seg) / len(seg))
            if sil_energies:
                noise_floor = sorted(sil_energies)[max(0, int(len(sil_energies) * 0.1))]
        except Exception:
            pass

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
        # Word energy at silence level -> likely misaligned into a silence gap.
        # Skip when the word is adjacent to an English / NVV token — MFA
        # cannot model those, so their boundaries bleed into neighbours.
        _prev_w = words.intervals[idx - 1] if idx > 0 else None
        _next_w = words.intervals[idx + 1] if idx + 1 < len(words.intervals) else None
        _near_en_nvv = (
            (_prev_w and (is_english_token(_prev_w.text) or is_nvv_token(_prev_w.text)))
            or (_next_w and (is_english_token(_next_w.text) or is_nvv_token(_next_w.text)))
        )
        if (not _is_en_nvv and not is_punct(w.text) and not _near_en_nvv
                and args.filter_word_energy_ratio > 0 and noise_floor > 1e-8):
            w_energy = _word_rms(audio, sr, w.xmin, w.xmax) if (wav_path and wav_path.exists()) else 999
            if 0 < w_energy < noise_floor * args.filter_word_energy_ratio:
                issues.append({"rule": "word_in_silence", "text": w.text,
                               "energy": round(w_energy, 6), "noise_floor": round(noise_floor, 6)})
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
        if args.filter_short_phone and p.duration < args.filter_short_phone_sec:
            issues.append({"rule": "short_phone", "text": p.text, "phone_idx": pi + 1,
                           "duration": round(p.duration, 6)})
        if is_consonant_phone(p.text) and p.duration > args.filter_long_consonant_sec:
            issues.append({"rule": "long_consonant_phone", "text": p.text, "phone_idx": pi + 1,
                           "duration": round(p.duration, 6)})
        if is_vowel_phone(p.text) and p.duration > args.filter_long_vowel_sec:
            issues.append({"rule": "long_vowel_phone", "text": p.text, "phone_idx": pi + 1,
                           "duration": round(p.duration, 6)})
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


def _inject_punctuation(words_tier: Tier, pp_tier: Tier | None,
                         punct_entries: list[dict]) -> tuple[Tier, Tier | None]:
    """Inject punctuation intervals from CTC anchors into words tier.

    Punctuation has no acoustic realization but has precise CTC anchor
    timestamps.  Each entry is inserted at its CTC time, splitting or
    trimming adjacent intervals as needed.  Corresponding silence is
    inserted in pinyin_phones.
    """
    from dataclasses import replace as _replace

    # Build combined interval list: original words + punctuation
    combined = []
    for iv in words_tier.intervals:
        combined.append((iv.xmin, iv.xmax, iv.text, "word"))
    for p in punct_entries:
        combined.append((p["start_s"], p["end_s"], p["word"], "punct"))

    combined.sort(key=lambda x: x[0])

    # Resolve overlaps: punctuation keeps its CTC time, words are trimmed
    # 两轮处理: 先插入所有, 再裁剪 word 与 punct 的重叠
    resolved = []
    for c in combined:
        s, e, text, kind = c
        if e > s:
            resolved.append((s, e, text, kind))

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
                    # word contains punct → delete punct
                    resolved[pi] = (0, 0, "", pkind)
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

    # 微小静音间隙合并到后续标点或 NVV (<sp> -> 吸收进标点/NVV)
    for gi in range(len(merged)):
        gs, ge, gtext, gkind = merged[gi]
        if not (gkind in ("word", "gap") and is_silence(gtext)):
            continue
        # 找后面紧接的标点或 NVV
        for pi in range(len(merged)):
            target = merged[pi]
            is_target = (target[3] == "punct" or is_nvv_token(target[2]))
            if is_target and abs(target[0] - ge) < 0.01:
                # NVV 前间隙无条件合并 (NVV 天然含静音, 但句首不合并)
                # 标点前间隙合并 ≤500ms
                gap_dur = ge - gs
                if gs < 0.01:
                    pass  # 句首间隙不合并
                elif is_nvv_token(target[2]) or gap_dur <= 0.5:
                    merged[pi] = (gs, target[1], target[2], target[3])
                    merged[gi] = (0, 0, "", "word")
                break

    merged = [(s, e, t, k) for s, e, t, k in merged if e > s + 0.001]

    # 标点右边界延伸到下个词的 start (消除标点与词之间的微间隙)
    for pi in range(len(merged)):
        ps, pe, ptext, pkind = merged[pi]
        if pkind != "punct":
            continue
        for wi in range(len(merged)):
            ws, we, wtext, wkind = merged[wi]
            if wkind == "word" and not is_silence(wtext) and ws >= pe:
                gap = ws - pe
                if 0 < gap < 0.5:
                    merged[pi] = (ps, ws, ptext, pkind)
                break

    # 标点延展后清理被覆盖的间隙
    for gi in range(len(merged)):
        gs, ge, gtext, gkind = merged[gi]
        if not (gkind in ("word", "gap") and is_silence(gtext)):
            continue
        for pi in range(len(merged)):
            ps, pe, ptext, pkind = merged[pi]
            if pkind != "punct":
                continue
            if gs < pe and ge > ps:
                if gs < ps:
                    merged[gi] = (gs, ps, gtext, gkind)
                else:
                    merged[gi] = (pe, ge, gtext, gkind)

    merged = [(s, e, t, k) for s, e, t, k in merged if e > s + 0.001]

    # 残余微小 <sp> 合并到前一词 (词间微间隙吸收)
    for gi in range(len(merged)):
        gs, ge, gtext, gkind = merged[gi]
        if not (gkind in ("word", "gap") and is_silence(gtext)):
            continue
        if ge - gs > 0.5:
            continue
        # 句首间隙不合并 (保留 <sp1> 标记)
        if gs < 0.01:
            continue
        # 优先合并到后一词 (延伸后词 start), 不成再合并到前一词
        merged_to_next = False
        for wi in range(len(merged)):
            ws, we, wtext, wkind = merged[wi]
            if wkind == "word" and not is_silence(wtext) and abs(ws - ge) < 0.01:
                merged[wi] = (gs, we, wtext, wkind)
                merged[gi] = (0, 0, "", "word")
                merged_to_next = True
                break
        if merged_to_next:
            continue
        for wi in range(len(merged)):
            ws, we, wtext, wkind = merged[wi]
            if wkind == "word" and not is_silence(wtext) and abs(we - gs) < 0.01:
                merged[wi] = (ws, ge, wtext, wkind)
                merged[gi] = (0, 0, "", "word")
                break

    merged = [(s, e, t, k) for s, e, t, k in merged if e > s + 0.001]

    # 最后标点: 吸收前后静音, 延伸到音频结束
    last_punct = None
    for m in reversed(merged):
        if m[3] == "punct":
            last_punct = m
            break
    if last_punct:
        punct_start = last_punct[0]
        punct_text = last_punct[2]
        # 反向找前一个非 silence 词, 标点从词的 end 开始
        for m in reversed(merged):
            if m[3] == "word" and not is_silence(m[2]):
                punct_start = m[1]
                break
        # 确保最后标点至少 30ms: 当末词 xmax == words_tier.xmax 时
        # (如音频恰好在词边界结束), 标点不会变成零时长被丢弃.
        punct_end = max(punct_start + 0.030, words_tier.xmax)
        # 重建: 保留非静音 + 最后标点(延伸)
        new_merged = []
        for m in merged:
            if m is last_punct:
                new_merged.append((punct_start, punct_end, punct_text, "punct"))
            elif m[1] <= punct_start + 0.001:
                new_merged.append(m)
        merged = new_merged

    # Build new words tier (skip zero-duration intervals, ensure sorted)
    merged.sort(key=lambda x: x[0])
    new_words = [Interval(iv[0], iv[1], iv[2]) for iv in merged if iv[1] > iv[0]]
    new_words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, new_words)

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
    new_words = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)

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
    new_words = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)

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
        if is_english_token(iv.text) or is_nvv_token(iv.text):
            continue
        if i + 1 >= n:
            continue
        next_iv = intervals[i + 1]
        if next_iv.xmax <= next_iv.xmin:
            continue
        # Extend into: NVV always; silence gaps when no punctuation follows;
        # regular words when leading portion is dead silence.
        # Only skip real punctuation (not silence tokens like <eps>/<spN>).
        # Silence gaps adjacent to words without punctuation are absorbable.
        if is_punct(next_iv.text) and not is_silence(next_iv.text):
            continue

        extend_into_word = False
        if not is_nvv_token(next_iv.text):
            # Silence or word: check if there's dead silence worth absorbing
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
    return Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)



def _snap_to_ctc(words_tier: Tier, pp_tier: Tier | None,
                  ctc_tokens: list[dict],
                  snap_threshold: float = 0.3,
                  punct_entries: list[dict] | None = None,
                  audio=None, sr: int = 16000,
                  _punct_boundary_hits: list | None = None) -> tuple[Tier, Tier | None]:
    """Snap MFA word boundaries to CTC anchors only when they differ too much.

    If |MFA - CTC| <= snap_threshold: trust MFA, keep MFA boundaries.
    If |MFA - CTC| >  snap_threshold: MFA likely misaligned, snap to CTC.
    When MFA and CTC disagree within the threshold, energy analysis on
    the disputed region decides: energy above noise → MFA wins (speech
    continues), energy at noise level → CTC wins (silence after CTC end).

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
        # Rule 1: NVV / English — no MFA acoustic model, always CTC.
        # Exception: NVV with CTC duration < 100ms — CTC detection may be
        # a noise artifact; keep MFA boundaries to avoid squeezing adjacent
        # words (e.g. BREATHING 60ms detection eating into ti2 tail).
        if is_nvv_token(mfa_iv.text):
            use_mfa = (ctc_end - ctc_start) < 0.10
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
        if word_end < word_start:
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

    # Eliminate tiny gaps between consecutive word intervals.
    # Regr. Case 39: MFA frame precision is 10 ms; gaps up to 30 ms
    # (3 frames) are alignment residuals, not real silences.  Absorb
    # them into the preceding word so the words tier is always
    # contiguous — downstream tiers (hanzi, pinyin_phones) depend on
    # this invariant.
    # Also eliminate tiny overlaps (≤ 3 ms) by splitting at the midpoint —
    # these are boundary artifacts from CTC/MFA precision mismatch.
    for k in range(len(new_word_ivs) - 1, 0, -1):
        cur = new_word_ivs[k]
        prev = new_word_ivs[k - 1]
        gap = cur[0] - prev[1]
        if 0 < gap <= 0.030 and prev[3] == "word":
            # Tiny gap — absorb into previous word
            new_word_ivs[k - 1] = (prev[0], cur[0], prev[2], prev[3])
        elif gap < 0 and gap >= -0.005 and prev[3] == "word":
            # Tiny overlap — split at midpoint (only word-word pairs)
            mid = (prev[1] + cur[0]) / 2.0
            new_word_ivs[k - 1] = (prev[0], mid, prev[2], prev[3])
            new_word_ivs[k] = (mid, cur[1], cur[2], cur[3])

    # Build new tiers
    new_words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax,
                          [Interval(s, e, t) for s, e, t, _ in new_word_ivs])

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


def _strict_en_lexical_words(words_tier: Tier | None) -> list[Interval]:
    if words_tier is None:
        return []
    return [iv for iv in words_tier.intervals if is_english_token(iv.text.strip())]


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


def load_strict_en_provenance(stem: str, words_tier: Tier | None,
                              en_phones_dir: Path | None) -> tuple[dict, list[tuple[Interval, dict]]]:
    """Load a strict-en-mfa-v1 ledger and bind it by ordered word instance.

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
            or manifest.get("status") not in {"success", "no_english"}):
        return _strict_en_fail(required, "strict_en_manifest_invalid")
    expected_segments = manifest.get("expected_segments")
    produced_segments = manifest.get("produced_segments")
    rejected_segments = manifest.get("rejected_segments")
    if (not isinstance(expected_segments, list) or not isinstance(produced_segments, list)
            or not isinstance(rejected_segments, list)
            or not all(isinstance(item, str) for item in expected_segments)
            or not all(isinstance(item, str) for item in produced_segments)):
        return _strict_en_fail(required, "strict_en_manifest_partition_invalid")
    rejected_ids = [item.get("id") for item in rejected_segments if isinstance(item, dict)]
    if (len(rejected_ids) != len(rejected_segments) or not all(isinstance(item, str) for item in rejected_ids)
            or len(expected_segments) != len(set(expected_segments))
            or len(produced_segments) != len(set(produced_segments))
            or len(rejected_ids) != len(set(rejected_ids))
            or set(expected_segments) != set(produced_segments) | set(rejected_ids)
            or set(produced_segments) & set(rejected_ids)
            or (manifest.get("status") == "no_english" and expected_segments)):
        return _strict_en_fail(required, "strict_en_manifest_partition_invalid")
    if not english_words:
        return _strict_en_report("not_required"), []
    if manifest.get("status") != "success":
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
            or produced_for_stem & rejected_for_stem or rejected_for_stem):
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
    if not isinstance(ledger, dict) or ledger.get("schema") != STRICT_EN_MFA_SCHEMA or ledger.get("stem") != stem:
        return _strict_en_fail(required, "strict_en_ledger_schema_or_stem_mismatch", ledger_sha256=actual_hash)

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

    # The authoritative transcript may keep a contiguous English spelling as
    # one word (e.g. ``Sila``), while CTC/MFA tokenization can split the same
    # acoustic span into verified records (``S`` + ``il`` + ``a``).  Reconcile
    # only exact, ordered concatenations; never drop a record or synthesize a
    # phone.  This preserves every MFA phone as provenance for the final word.
    if len(records) != required:
        grouped_records: list[dict] = []
        record_cursor = 0
        for final_word in english_words:
            target = final_word.text.strip().casefold()
            joined = ""
            matched_end = None
            for candidate_end in range(record_cursor + 1, len(records) + 1):
                candidate = records[candidate_end - 1]
                if not isinstance(candidate, dict):
                    break
                joined += str(candidate.get("ctc_text", "")).strip()
                if joined.casefold() == target:
                    matched_end = candidate_end
                    break
                if not target.startswith(joined.casefold()):
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
                        ordinal = len(combined_phones)
                        copied["ordinal"] = ordinal
                        copied["mfa_phone_ordinal"] = ordinal
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
    pairs: list[tuple[Interval, dict]] = []
    previous_ctc_ordinal = -1
    seen_word_ids: set[str] = set()
    # MFA phone ordinals restart at zero for each English segment.  Scope the
    # uniqueness check by segment; treating them as stem-global rejects every
    # stem containing more than one English segment even when the ledger and
    # source TextGrids are valid.
    seen_mfa_phone_ordinals: set[tuple[str, int]] = set()
    for final_word, record in zip(english_words, records):
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
                or str(record.get("ctc_text", "")).casefold() != final_word.text.strip().casefold()
                or not isinstance(mfa_word, dict) or not isinstance(phones, list) or not phones):
            return _strict_en_fail(required, "strict_en_word_identity_or_evidence_invalid",
                                   ledger_sha256=actual_hash, failed_word_ids=[str(word_id or "")])
        seen_word_ids.add(word_id); previous_ctc_ordinal = ordinal
        segment_key = word_id.rsplit(":w", 1)[0]
        try:
            if (not isinstance(mfa_word.get("ordinal"), int)
                    or mfa_word["ordinal"] < 0
                    or str(mfa_word.get("text", "")).casefold() != final_word.text.strip().casefold()):
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


def process_one(tg_path: Path, txt_dir: Path, wav_dir: Path,
                output_dir: Path, filtered_dir: Path, args,
                ipa_to_pinyin: dict[str, str],
                pinyin_dict: dict[str, list[str]],
                pinyin_case: dict[str, str] | None = None,
                raw_text_index: dict[str, Path] | None = None) -> dict:
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
    strict_en_mode = bool(getattr(args, "strict_ok", False))
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
    txt_path = txt_dir / f"{stem}.txt"
    if not txt_path.exists():
        txt_path = txt_dir / f"{stem}.lab"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing txt/lab: {txt_dir}/{stem}")
    tg = parse_textgrid(tg_path)
    if len(tg.tiers) < 2:
        raise ValueError(f"Need at least 2 tiers in {tg_path}")
    words_tier = tg.tiers[0]
    phones_tier = tg.tiers[1]
    mfa_unknown_before_snap = [
        iv.text.strip() for iv in words_tier.intervals
        if is_unknown_token(iv.text)
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
    raw_text = find_original_text(stem, args.raw_text_dir, raw_text_index)
    reference_text_authoritative = bool(raw_text)
    reference_source = "original_or_ref" if reference_text_authoritative else ""
    if not raw_text:
        # Try NVASR Chinese ASR output
        cn_path = txt_dir / f"{stem}_text_cn.txt"
        if cn_path.exists():
            raw_text = cn_path.read_text(encoding="utf-8").strip()
            reference_source = "asr_fallback"
    if not raw_text:
        # Fallback: use the pinyin txt content
        raw_text = txt_path.read_text(encoding="utf-8").strip()
        reference_source = "lab_fallback"
    reference_text_original = raw_text

    # Tier 2: pinyin with punctuation (from corpus txt)
    pinyin_text = txt_path.read_text(encoding="utf-8").strip()
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
                ctc_token_list.append(json.loads(line))

    # For each MFA word that is <unk>/[bracketed], or is a pinyin syllable
    # (e.g. rui4) where the CTC anchor says it should be English (e.g. ria),
    # restore the correct word text from CTC anchors by time overlap.
    if ctc_token_list and not reference_text_authoritative:
        for iv in words_tier.intervals:
            if is_silence(iv.text) or iv.text.strip() in ("", "<eps>"):
                continue
            is_bracket = iv.text.strip() in ("<unk>", "[bracketed]")
            is_pinyin = is_pinyin_syllable(iv.text.strip())
            if not is_bracket and not is_pinyin:
                continue
            best_ctc = None
            best_overlap = 0.0
            for ct in ctc_token_list:
                overlap = min(iv.xmax, ct["end_s"]) - max(iv.xmin, ct["start_s"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ctc = ct
            if best_ctc:
                ctc_word = best_ctc["word"].strip().strip("<>")
                if is_bracket and (is_nvv_token(ctc_word) or is_english_token(ctc_word)):
                    iv.text = ctc_word
                elif is_pinyin and is_english_token(ctc_word):
                    # MFA split an English word into pinyin fragments
                    iv.text = ctc_word

    # Merge consecutive intervals that belong to the same CTC English token.
    # When MFA splits e.g. "ria" into "rui4"+"ya4", both fragments fall
    # within the CTC token's time range and should be merged.
    # Guard: both words must be English tokens (not Chinese pinyin like "yi1")
    # to avoid swallowing real Chinese words that happen to overlap the
    # English token boundary.
    if ctc_token_list and not reference_text_authoritative:
        merged_intervals = []
        for iv in words_tier.intervals:
            if (merged_intervals
                    and is_english_token(merged_intervals[-1].text.strip())
                    and (is_english_token(iv.text.strip())
                         or is_pinyin_syllable(iv.text.strip()))):
                prev = merged_intervals[-1]
                for ct in ctc_token_list:
                    ct_word = ct["word"].strip().strip("<>")
                    if not is_english_token(ct_word):
                        continue
                    prev_ov = min(prev.xmax, ct["end_s"]) - max(prev.xmin, ct["start_s"])
                    cur_ov = min(iv.xmax, ct["end_s"]) - max(iv.xmin, ct["start_s"])
                    if prev_ov > 0 and cur_ov > 0 and prev.text.strip().lower() == ct_word.lower():
                        merged_intervals[-1] = Interval(prev.xmin, iv.xmax, prev.text)
                        break
                else:
                    merged_intervals.append(Interval(iv.xmin, iv.xmax, iv.text))
            else:
                merged_intervals.append(Interval(iv.xmin, iv.xmax, iv.text))
        words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, merged_intervals)

    raw_tier = Tier("raw_text", tg.xmin, tg.xmax,
                    [Interval(tg.xmin, tg.xmax, raw_text)])
    pinyin_tier = Tier("pinyin", tg.xmin, tg.xmax,
                       [Interval(tg.xmin, tg.xmax, pinyin_text)])
    pinyin_phones_tier = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin,
                                                   words_tier, pinyin_dict)

    # Build 5 tiers
    tiers = [raw_tier, pinyin_tier, words_tier, phones_tier, pinyin_phones_tier]
    new_tg = TextGrid(tg.xmin, tg.xmax, tiers)

    # Find WAV recursively (may be in subdirectory)
    wav_path = wav_dir / f"{stem}.wav"
    if not wav_path.exists():
        candidates = list(wav_dir.rglob(f"{stem}.wav"))
        if candidates:
            wav_path = candidates[0]
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

    merge_report = []
    if args.merge_silence:
        new_tg, merge_report = merge_short_silences(
            new_tg, wav_path if wav_path.exists() else None, args, wav_audio, wav_sr)
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
        new_tg = _finalise_textgrid(new_tg, raw_text, pinyin_text, args)

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
    if punct_path.exists():
        punct_entries = json.loads(punct_path.read_text(encoding="utf-8"))
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
    if punct_entries and words_tier:
            words_tier, pp_tier = _inject_punctuation(
                words_tier, pp_tier, punct_entries)
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
    sil_filter_reasons = []
    if args.handle_unexpected_sil:
        sil_filter_reasons = handle_unexpected_silences(new_tg, pinyin_text)
        if sil_filter_reasons:
            report["unexpected_silence"] = sil_filter_reasons

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
                                report_warnings=report.get("warnings", []))

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
                         raw_text, en_mfa_windows, report.get("warnings", []))

    # --- E. NVV + ellipsis unconditional merge ---
    words_tier = tier_by_name(new_tg, "words")
    pp_tier = tier_by_name(new_tg, "pinyin_phones")
    if words_tier:
        intervals = list(words_tier.intervals)
        for i in range(len(intervals) - 1):
            if is_nvv_token(intervals[i].text) and intervals[i + 1].text.strip() == '…':
                gap = intervals[i + 1].xmin - intervals[i].xmax
                if gap < 0.02:
                    intervals[i] = Interval(intervals[i].xmin, intervals[i + 1].xmax,
                                            intervals[i].text)
                    intervals[i + 1] = Interval(0, 0, '')
        intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
        words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)
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
                         raw_text, en_mfa_windows, report.get("warnings", []))

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
            for i, iv in enumerate(intervals):
                if not is_nvv_token(iv.text.strip()):
                    continue
                best_ctc = None
                best_overlap = 0.0
                for ct in ctc_data:
                    overlap = min(iv.xmax, ct["end_s"]) - max(iv.xmin, ct["start_s"])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_ctc = ct
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
            words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                    break

    # 检测被吞掉的标点: CTC punct 条目在 words tier 中时间匹配不到 -> 从文本删除
    if punct_entries:
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
        _normalize_word_spellings(final_words_tier, raw_text)

        # Reference punctuation is lexical authority.  CTC punctuation may
        # be missing, swallowed by a broad pause, or spuriously appended at
        # the end; repair the words tier before rebuilding all derived tiers.
        if reference_text_authoritative:
            _restore_reference_punctuation(
                final_words_tier, reference_text_original, punct_entries)
            _sync_derived_tiers(
                new_tg, ipa_to_pinyin, pinyin_dict,
                raw_text=reference_text_original,
                en_mfa_windows=en_mfa_windows,
                report_warnings=report.get("warnings", []))
            final_words_tier = tier_by_name(new_tg, "words")
            raw_text = reference_text_original
            pinyin_text = pinyin_text_original
        # 2. Rebuild hanzi from normalised words.
        hanzi_tier = _build_hanzi_tier(final_words_tier, raw_text,
                                        report.get("warnings", []))
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
        # Rebuild raw_text from hanzi tier (Chinese chars), not from words (pinyin)
        raw_tier = tier_by_name(new_tg, "raw_text")
        hanzi_after = tier_by_name(new_tg, "hanzi")
        if raw_tier and hanzi_after:
            if reference_text_authoritative:
                # Keep supplied lexical content authoritative.  Rebuilding raw
                # from a collapsed hanzi tier created the former "empty==empty"
                # false pass; rendered punctuation edits remain in raw_text.
                raw_tier.intervals[0].text = raw_text
            else:
                raw_tokens = [iv.text for iv in hanzi_after.intervals
                              if not is_silence(iv.text) and iv.text.strip()]
                if raw_tokens:
                    raw_tier.intervals[0].text = "".join(raw_tokens)

    # 最终恢复: CTC 长停顿注入 … 覆盖了原标点, 用 CTC punct 替换回去
    if punct_entries:
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
    if _swallowed_puncts:
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
                        report.get("warnings", []))
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
                # 同步 raw_text: 从更新后的 hanzi 重建
                _raw_t = tier_by_name(new_tg, "raw_text")
                _hanzi_t2 = tier_by_name(new_tg, "hanzi")
                if _raw_t and _hanzi_t2:
                    _raw_tokens = [iv.text for iv in _hanzi_t2.intervals
                                   if not is_silence(iv.text) and iv.text.strip()]
                    if _raw_tokens:
                        _raw_t.intervals[0].text = "".join(_raw_tokens)
                report.setdefault("restored_punct", 0)
                report["restored_punct"] = _restored

    # ── 末尾标点强制保留 ───────────────────────────────────────────
    # CTC 最后一个标点如果被前词 (通常是 NVV) 吸收, 从前词末尾截取至少
    # 60ms 还给标点。保证每个音频的句末标点不丢失。
    # Regression Case 25 follow-up: terminal punct recovery.
    if punct_entries:
        _words_t = tier_by_name(new_tg, "words")
        if _words_t:
            # Find the last (rightmost) CTC punct
            _last_punct = max(punct_entries, key=lambda p: p["end_s"])
            _last_punct_word = _last_punct["word"]
            _last_punct_end = _last_punct["end_s"]

            # Check if this punct already exists as the last item in words tier
            _word_ivs = list(_words_t.intervals)
            _last_word_iv = None
            for iv in reversed(_word_ivs):
                if iv.text.strip():
                    _last_word_iv = iv
                    break

            _punct_at_end = (_last_word_iv is not None
                             and _last_word_iv.text.strip() == _last_punct_word)

            if not _punct_at_end and _last_word_iv is not None:
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
                    _words_t = Tier(_words_t.name, _words_t.xmin,
                                    _words_t.xmax, _new_ivs)
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

    # Final authoritative reconciliation: terminal-punctuation recovery above
    # may have reintroduced a CTC-only mark.  Reapply the reference contract
    # immediately before strict English injection and publication.
    if reference_text_authoritative:
        _final_words = tier_by_name(new_tg, "words")
        if _final_words is not None:
            _restore_reference_punctuation(
                _final_words, reference_text_original, punct_entries)
            _sync_derived_tiers(
                new_tg, ipa_to_pinyin, pinyin_dict,
                raw_text=reference_text_original,
                en_mfa_windows=en_mfa_windows,
                report_warnings=report.get("warnings", []))
            _derived_pp = tier_by_name(new_tg, "pinyin_phones")
            _derived_words = tier_by_name(new_tg, "words")
            if _derived_pp is not None and _derived_words is not None:
                _clip_pinyin_phones_to_words(_derived_pp, _derived_words)
        _raw_authoritative = tier_by_name(new_tg, "raw_text")
        if _raw_authoritative and _raw_authoritative.intervals:
            _raw_authoritative.intervals[0].text = (
                "<sp1>" + reference_text_original.replace("<sp1>", ""))
        _pinyin_authoritative = tier_by_name(new_tg, "pinyin")
        if _pinyin_authoritative and _pinyin_authoritative.intervals:
            _pinyin_authoritative.intervals[0].text = _reference_pinyin_text(
                reference_text_original, pinyin_text_original)

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
    strict_en_rejected = False
    if strict_en_mode:
        final_words_tier = tier_by_name(new_tg, "words")
        strict_en, strict_pairs = load_strict_en_provenance(
            stem, final_words_tier, en_phones_dir)
        report["english_provenance"] = strict_en
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

    # Strict English injection is intentionally last for English intervals;
    # clean any remaining Chinese/punctuation overlap without touching the
    # immutable ``en:`` phone geometry.
    _post_strict_pp = tier_by_name(new_tg, "pinyin_phones")
    if _post_strict_pp is not None:
        _post_fixed = _fix_non_english_pp_overlaps(_post_strict_pp)
        if _post_fixed:
            report["pp_deoverlap_fixed"] = int(
                report.get("pp_deoverlap_fixed", 0)) + _post_fixed

    # ================================================================
    # 最终筛选: 所有处理完成后再统一判断 (用最终的边界和静音结构)
    # ================================================================
    filter_reasons = []
    if strict_en_rejected:
        filter_reasons.append("english_provenance_rejected")

    # Hard lexical integrity is independent of optional acoustic filtering.
    # NVV, punctuation and sentence-initial <sp1> are intentionally excluded
    # from the CJK/pinyin denominator.
    _coverage, _coverage_reasons = assess_reference_coverage(
        reference_text_original,
        tier_by_name(new_tg, "words"),
        tier_by_name(new_tg, "hanzi"),
        reference_source=reference_source,
        unknown_source_count=len(mfa_unknown_before_snap),
    )
    report["reference_coverage"] = _coverage
    report["hard_integrity_reasons"] = _coverage_reasons
    filter_reasons.extend(_coverage_reasons)
    if mfa_unknown_before_snap:
        report["mfa_unknown_source"] = {
            "count": len(mfa_unknown_before_snap),
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
            _pinyin_hits.extend(_re.findall(r'\b(?!sp\d\b)[a-z]+[1-5]\b', _raw_text))
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
    _tier_mismatches = 0
    _tier_total = 0
    if _hanzi_t is not None and words_tier is not None:
        _h_seq = [(iv.xmin, iv.xmax, iv.text.strip()) for iv in _hanzi_t.intervals
                  if not is_silence(iv.text) and iv.text.strip()]
        _w_seq = [(iv.xmin, iv.xmax, iv.text.strip()) for iv in words_tier.intervals
                  if not is_silence(iv.text) and iv.text.strip()]
        _n = min(len(_h_seq), len(_w_seq))
        _tier_total = max(len(_h_seq), len(_w_seq))
        if _n > 0:
            import re as _re2
            for _i in range(_n):
                _ht = _h_seq[_i][2]
                _wt = _w_seq[_i][2]
                # CJK hanzi → words should be pinyin reading
                if is_cjk(_ht):
                    if not _re2.match(r'^[a-z]+[1-5]$', _wt):
                        _tier_mismatches += 1
                # ASCII/English hanzi → words should be same stem
                elif _ht.isascii() and _ht.isalpha():
                    if _wt.lower().rstrip('012') != _ht.lower().rstrip('012'):
                        _tier_mismatches += 1
                # Punct/symbol → skip
        # Also flag if counts differ (extra or missing intervals in one tier)
        if len(_h_seq) != len(_w_seq):
            _tier_mismatches += abs(len(_h_seq) - len(_w_seq))
        if _tier_total > 0 and _tier_mismatches / _tier_total > 0.10:
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
        if _total_phones > 0 and _misaligned_phones / _total_phones > 0.15:
            filter_reasons.append("misaligned_phones")
            report["misaligned_phones"] = f"{_misaligned_phones}/{_total_phones}"

    # sp3 / mid_sp: 检查最终 words 层的静音结构
    # 首尾静音 (<spN> 在开头/结尾) 是正常的音频裁剪结果, 不算异常。
    # 只有中间 (非首非尾) 的长静音才是问题。
    if words_tier:
        n_ivs = len(words_tier.intervals)
        for i, iv in enumerate(words_tier.intervals):
            if iv.text.strip() == "<sp3>":
                if i == 0 or i == n_ivs - 1:
                    continue  # leading / trailing silence is normal
                filter_reasons.append("sp3")
        sp_in_mid = False
        for i, iv in enumerate(words_tier.intervals):
            if i == 0 or i == n_ivs - 1:
                continue  # leading / trailing silence is normal
            if is_silence(iv.text) and iv.text.strip():
                sp_in_mid = True
                break
        if sp_in_mid:
            filter_reasons.append("mid_sp")

    # suspicious_alignment (from phone-level QC in Phase 5)
    if align_issues:
        filter_reasons.append("suspicious_alignment")

    # unexpected_silence
    if sil_filter_reasons:
        filter_reasons.extend(sil_filter_reasons)

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
            for iv in words_tier.intervals:
                if not is_silence(iv.text):
                    if iv.text.strip():
                        e = _word_rms(wav_audio, wav_sr, iv.xmin, iv.xmax)
                        if e > 0:
                            speech_energies.append(e)
                    continue
                if iv.xmax - iv.xmin < args.bgm_min_sil_dur:
                    continue
                total_sil_dur += iv.xmax - iv.xmin
                # ── Frame-level energy check within silence interval ──
                # _word_rms() averages over the entire interval, which can
                # hide short bursts of loud content inside long silences.
                # Instead, scan 50ms frames and flag intervals where a
                # significant fraction of frames exceed the threshold.
                # Regression Case 25 follow-up.
                _frame_ms = 50.0
                _frame_samp = max(1, int(_frame_ms / 1000.0 * wav_sr))
                _s0 = int(iv.xmin * wav_sr)
                _s1 = int(iv.xmax * wav_sr)
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
                    suspect_intervals.append({"xmin": round(iv.xmin, 3), "xmax": round(iv.xmax, 3),
                                              "duration": round(iv.xmax - iv.xmin, 3),
                                              "energy": round(_max_frame_e, 6),
                                              "high_ratio": round(_high_ratio, 3),
                                              "noise_floor": round(nf_bgm, 6)})
                    suspect_dur += (iv.xmax - iv.xmin) * _high_ratio
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
        # word_in_silence
        if args.filter_suspicious and args.filter_word_energy_ratio > 0:
            all_rms, _ = _frame_rms_vec(wav_audio, wav_sr, frame_ms=10.0)
            k = max(1, int(len(all_rms) * 0.15))
            nf = float(np.partition(all_rms, k)[k]) if len(all_rms) > 0 else 1e-6
            threshold = max(nf * args.filter_word_energy_ratio, nf * 10.0)
            # Build English/NVV adjacency set
            en_nvv_neighbors: set[int] = set()
            for idx, iv in enumerate(words_tier.intervals):
                if not iv.text.strip() or is_silence(iv.text):
                    continue
                if is_english_token(iv.text) or is_nvv_token(iv.text):
                    if idx > 0:
                        en_nvv_neighbors.add(idx - 1)
                    en_nvv_neighbors.add(idx + 1)
            for idx, iv in enumerate(words_tier.intervals):
                if is_silence(iv.text) or not iv.text.strip():
                    continue
                if is_punct(iv.text):
                    continue
                if is_english_token(iv.text) or is_nvv_token(iv.text):
                    continue  # MFA can't model acoustically, CTC boundaries authoritative
                if idx in en_nvv_neighbors:
                    continue  # adjacent to English/NVV — boundaries unreliable
                w_energy = _word_rms(wav_audio, wav_sr, iv.xmin, iv.xmax)
                if 0 < w_energy < threshold:
                    align_issues.append({"rule": "word_in_silence", "text": iv.text,
                                         "energy": round(w_energy, 6),
                                         "noise_floor": round(nf, 6)})
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
                        clean = p.text.replace(EN_PHONE_PREFIX, "")
                        if args.filter_short_phone and p.duration < short_phone_en:
                            align_issues.append({
                                "rule": "short_phone_en", "text": p.text,
                                "phone_idx": pi + 1,
                                "duration": round(p.duration, 6)})
                        if is_english_vowel_phone(clean) and p.duration > long_vowel_en:
                            align_issues.append({
                                "rule": "long_vowel_en", "text": p.text,
                                "phone_idx": pi + 1,
                                "duration": round(p.duration, 6)})
                        if is_english_consonant_phone(clean) and p.duration > long_cons_en:
                            align_issues.append({
                                "rule": "long_consonant_en", "text": p.text,
                                "phone_idx": pi + 1,
                                "duration": round(p.duration, 6)})
                        continue
                    # Chinese phone: use standard thresholds
                    if args.filter_short_phone and p.duration < args.filter_short_phone_sec:
                        align_issues.append({
                            "rule": "short_phone", "text": p.text,
                            "phone_idx": pi + 1,
                            "duration": round(p.duration, 6)})
                    if is_consonant_phone(p.text) and p.duration > args.filter_long_consonant_sec:
                        align_issues.append({
                            "rule": "long_consonant_phone", "text": p.text,
                            "phone_idx": pi + 1,
                            "duration": round(p.duration, 6)})
                    if is_vowel_phone(p.text) and p.duration > args.filter_long_vowel_sec:
                        align_issues.append({
                            "rule": "long_vowel_phone", "text": p.text,
                            "phone_idx": pi + 1,
                            "duration": round(p.duration, 6)})

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
        # Compare against the immutable source reference, never against the
        # rendered raw tier (which may have been rebuilt or punctuation-edited).
        raw_cjk = _coverage["reference_cjk"]
        hanzi_cjk = "".join(iv.text.strip() for iv in hanzi_tier_final.intervals
                           if iv.text.strip()
                           and ("一" <= iv.text.strip() <= "鿿"
                                or "㐀" <= iv.text.strip() <= "䶿"))
        if raw_cjk != hanzi_cjk:
            if "cjk_mismatch" not in filter_reasons:
                filter_reasons.append("cjk_mismatch")
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
                if getattr(args, 'original_txt_dir', None):
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
    if _overlap_count > 0:
        filter_reasons.append("overlapping_words")
        report["overlapping_words"] = {"count": _overlap_count,
                                        "examples": _overlap_examples}

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
    _SHORT_FLOOR = 0.030  # seconds
    if words_tier is not None:
        for _iv in words_tier.intervals:
            _text = _iv.text.strip()
            if (not _text or is_silence(_iv.text) or is_punct(_text)
                    or is_nvv_token(_text)):
                continue
            if _iv.xmax - _iv.xmin < _SHORT_FLOOR:
                _short_count += 1
                if len(_short_examples) < 8:
                    _short_examples.append(
                        f"{_text}[{(_iv.xmax-_iv.xmin)*1000:.0f}ms]")
    if _short_count > 0:
        filter_reasons.append("short_word")
        report["short_word"] = {"count": _short_count,
                                 "examples": _short_examples}


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
    _pp_gap_count = _count_internal_pp_gaps(
        pp_tier, words_tier, _PP_GAP_THRESHOLD_S)
    if _pp_gap_count > 0:
        _record_filterable_qc(
            report, filter_reasons, args.filter_suspicious,
            "pp_tier_gaps", {"count": _pp_gap_count}
        )

    # ── Case 35: words_tier_gaps — direct gaps between content words.
    #     Punctuation and explicit silence intervals own their boundary gaps;
    #     those are not alignment holes and must remain preserved.
    _wt_gap_count = 0
    _wt_gap_examples: list[str] = []
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
            _gap = round(_nxt.xmin - _cur.xmax, 4)
            if _gap > _WT_GAP_THRESHOLD_S:
                _wt_gap_count += 1
                if len(_wt_gap_examples) < 5:
                    _wt_gap_examples.append(
                        f"{_cl!r}→{_nl!r}[{_gap*1000:.0f}ms]")
    if _wt_gap_count > 0:
        _record_filterable_qc(
            report, filter_reasons, args.filter_suspicious,
            "words_tier_gaps", {"count": _wt_gap_count,
                                 "examples": _wt_gap_examples}
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
    new_tg.tiers = [t for t in new_tg.tiers if t.name != "phones"]
    _finalize_textgrid(new_tg)

    # strict-ok is intentionally stricter than the legacy best-effort mode:
    # every already-computed diagnostic is a veto.  It does not invent a new
    # acoustic judgement; the independent disk auditor records that judgement
    # as not evaluated.
    if getattr(args, "strict_ok", False) and report.get("warnings"):
        filter_reasons.append("warnings")
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
                 _filt_d, _raw_text_index):
    import os as _os
    for ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        _os.environ[ev] = "1"
    global _W
    _W = (_ipa, _py_dict, _py_case, _a, _txt_d, _wav_d, _out_d, _filt_d,
          _raw_text_index)


def _worker_fn(tgp):
    (_ipa, _py_dict, _py_case, _a, _txt_d, _wav_d, _out_d, _filt_d,
     _raw_text_index) = _W
    return process_one(tgp, _txt_d, _wav_d, _out_d, _filt_d, _a,
                       _ipa, _py_dict, _py_case, _raw_text_index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process MFA TextGrids for Chinese alignment.")
    parser.add_argument("--txt-dir", type=Path, default=PROJECT_ROOT / "corpus_clean" / "txt")
    parser.add_argument("--textgrid-dir", type=Path, default=PROJECT_ROOT / "aligned")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--filtered-dir", type=Path, default=PROJECT_ROOT / "filtered")
    parser.add_argument("--wav-dir", type=Path, default=PROJECT_ROOT / "corpus_clean" / "wav")
    parser.add_argument("--raw-text-dir", type=Path, default=None,
                        help="Directory with original Chinese text files")
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
                        help="Merge short sil intervals into previous phone based on energy.")
    parser.add_argument("--merge-max-sil-sec", type=float, default=0.2,
                        help="Max silence duration to consider for merging (default: 0.2s).")
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
                        help="Flag word if energy < noise_floor * N.")
    parser.add_argument("--filter-min-phone-coverage", type=float, default=0.35)
    parser.add_argument("--filter-edge-gap-sec", type=float, default=0.25)
    parser.add_argument("--copy-errors", action="store_true")
    parser.add_argument("--allow-filtered-integrity-failures", action="store_true",
                        help="Continue when mandatory-integrity failures were isolated in filtered/.")
    parser.add_argument("--strict-ok", action="store_true",
                        help="Treat every executed QC positive and warning as filterable.")
    parser.add_argument("--enable-text-correction", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-check punctuation against silence gaps and emit corrected_text tier.")
    parser.add_argument("--handle-unexpected-sil", action=argparse.BooleanOptionalAction, default=True,
                        help="Merge <sp0> gaps without punct; flag <sp1-3> gaps for filtering.")
    args = parser.parse_args()

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
                                           raw_text_index))
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
                         args.output_dir, args.filtered_dir, raw_text_index)
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
                                               raw_text_index)) as pool:
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
