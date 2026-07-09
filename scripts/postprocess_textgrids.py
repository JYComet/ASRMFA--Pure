#!/usr/bin/env python3
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
import json
import math
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

try:
    from pypinyin import lazy_pinyin, Style
except ModuleNotFoundError:
    raise SystemExit("pypinyin is not installed. Run: pip install pypinyin")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SILENCE_LABELS = {"<eps>", "<sil>", "sil", "<sp0>", "<sp1>", "<sp2>", "<sp3>"}
SHORT_PAUSE_PUNCT = set("，、：；,")
LONG_PAUSE_PUNCT = set("。？！…!?.")
SHORT_PAUSE_TOKEN = "[PAUSE]"
LONG_PAUSE_TOKEN = "<PAUSE>"

CHINESE_SHORT_WORDS = {
    "的", "了", "着", "呢", "吗", "吧", "啊", "嘛", "呀", "哦",
    "是", "在", "个", "和", "就", "也", "都", "不", "没",
    "de5", "le5", "zhe5", "ne5", "ma5", "ba5", "a5", "ya5",
}


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

    Returns (dict, case_map) where dict maps token→[phones] and case_map
    maps lowercase→canonical form (so MFA's lowercase output can be fixed).
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


# ---------------------------------------------------------------------------
# Static IPA → Pinyin phone mapping table
# Covers all phones in the MFA mandarin_mfa acoustic model and our dicts.
# ---------------------------------------------------------------------------

# Consonant mapping: IPA → pinyin
IPA_CONSONANT_MAP = {
    # Stops & affricates
    'p': 'b', 'pʰ': 'p',
    't': 'd', 'tʰ': 't',
    'k': 'g', 'kʰ': 'k',
    'tɕ': 'j', 'tɕʰ': 'q',
    'ʈʂ': 'zh', 'ʈʂʰ': 'ch',
    'ts': 'z', 'tsʰ': 'c',
    # Fricatives
    'f': 'f', 's': 's', 'ɕ': 'x', 'ʂ': 'sh', 'x': 'h',
    # Sonorants
    'm': 'm', 'n': 'n', 'l': 'l', 'ɻ': 'r',
    # Glides
    'j': 'i', 'w': 'u', 'ɥ': 'v',
    # Nasal finals
    'ŋ': 'ng',
    # Special
    'ʔ': '',  # glottal stop (unwritten in pinyin)
    'z̩': 'i0', 'ʐ̩': 'ir',
}

# Vowel tone mapping: base IPA vowel → (tone_marks_pattern → pinyin_tone_digit)
# Tones are applied to the vowel by replacing tone marks with the digit.
IPA_TONE_TO_DIGIT = {
    '˥˥': '1', '˥': '1',   # high level (also single ˥)
    '˧˥': '2',              # rising
    '˨˩˦': '3',             # dipping
    '˥˩': '4',              # falling
    '˩': '5',               # neutral
}

# Base vowel mapping: IPA (without tone) → pinyin vowel base
IPA_VOWEL_BASE_MAP = {
    'a': 'a', 'o': 'o', 'ə': 'e', 'e': 'e',
    'i': 'i', 'u': 'u', 'y': 'v',
    'z̩': 'i0', 'ʐ̩': 'ir',
}

TONE_MARK_CHARS = set('˥˧˨˩˦')

# Decompose pinyin finals into individual phone components for 1:1 IPA alignment.
# Derived from FINAL_SEGMENTS in convert_dict_to_ipa.py.
FINAL_DECOMPOSE = {
    'a': ['a'], 'o': ['o'], 'e': ['e'], 'e2': ['e'],
    'i': ['i'], 'u': ['u'], 'v': ['v'],
    'i0': ['i0'], 'u0': ['u0'], 'v0': ['v0'], 'ir': ['ir'],
    'ai': ['a', 'i'], 'ei': ['e', 'i'], 'ao': ['a', 'u'], 'ou': ['o', 'u'],
    'an': ['a', 'n'], 'en': ['e', 'n'], 'in': ['i', 'n'],
    'ang': ['a', 'ng'], 'eng': ['e', 'ng'], 'ing': ['i', 'ng'], 'ong': ['u', 'ng'],
    'ia': ['i', 'a'], 'ie': ['i', 'e'],
    'iao': ['i', 'a', 'u'], 'iu': ['i', 'o', 'u'], 'iou': ['i', 'o', 'u'],
    'ian': ['i', 'e', 'n'], 'iang': ['i', 'a', 'ng'], 'iong': ['i', 'u', 'ng'],
    'ua': ['u', 'a'], 'uo': ['u', 'o'],
    'uai': ['u', 'a', 'i'], 'ui': ['u', 'e', 'i'], 'uei': ['u', 'e', 'i'],
    'uan': ['u', 'a', 'n'], 'un': ['u', 'e', 'n'], 'uen': ['u', 'e', 'n'],
    'uang': ['u', 'a', 'ng'], 'ueng': ['u', 'e', 'ng'],
    've': ['v', 'e'], 'vn': ['v', 'n'], 'van': ['v', 'e', 'n'],
    'er': ['e', 'r'], 'io': ['i', 'o'],
    'n': ['n'], 'm': ['m'],
}

# Which component carries the tone (0-based index into FINAL_DECOMPOSE list).
FINAL_TONE_INDEX = {
    'a': 0, 'o': 0, 'e': 0, 'e2': 0, 'i': 0, 'u': 0, 'v': 0,
    'i0': 0, 'u0': 0, 'v0': 0, 'ir': 0,
    'ai': 0, 'ei': 0, 'ao': 0, 'ou': 0,
    'an': 0, 'en': 0, 'in': 0,
    'ang': 0, 'eng': 0, 'ing': 0, 'ong': 0,
    'ia': 1, 'ie': 1, 'iao': 1, 'iu': 1, 'iou': 1,
    'ian': 1, 'iang': 1, 'iong': 1,
    'ua': 1, 'uo': 1, 'uai': 1, 'ui': 1, 'uei': 1,
    'uan': 1, 'un': 1, 'uen': 1,
    'uang': 1, 'ueng': 1,
    've': 1, 'vn': 0, 'van': 1,
    'er': 0, 'io': 1,
    'n': 0, 'm': 0,
}

# Chinese initials (consonant phones without tone numbers)
CHINESE_INITIALS_SET = {
    "p", "pʰ", "t", "tʰ", "k", "kʰ",
    "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ", "ts", "tsʰ",
    "f", "s", "ɕ", "ʂ", "x",
    "m", "n", "l", "ɻ",
    "j", "w", "ɥ",
    "ŋ", "ʔ",
}


def decompose_pinyin_phone(phone: str) -> list[str]:
    """Decompose a pinyin phone into individual components for 1:1 IPA alignment.

    E.g., 'ai1' → ['a1', 'i'], 'ian3' → ['i', 'e3', 'n'], 'b' → ['b'].
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
    Build IPA→pinyin phone mapping: static table + dict-based cross-referencing.
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

            key = f"{base} → {py_p}"
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

def is_silence(text: str) -> bool:
    t = text.strip()
    return t in SILENCE_LABELS or t.startswith("<sp")


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
# IPA → Pinyin reverse-mapped phone tier
# ---------------------------------------------------------------------------

def build_pinyin_phones_tier(phones_tier: Tier,
                              ipa_to_pinyin: dict[str, str],
                              words_tier: Tier | None = None,
                              pinyin_dict: dict[str, list[str]] | None = None) -> Tier:
    """Build pinyin_phones tier using fullpinyin dict's initial+final format.

    For each word, look up the fullpinyin dict entry (e.g. pao4 → [p, ao4]),
    then use MFA phone boundaries to split the word interval into the dict's
    phone segments.  Punctuation and silence pass through unchanged.
    """
    if words_tier is None or pinyin_dict is None:
        # Fallback: 1:1 IPA→pinyin mapping
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

        if not word_phones:
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, word))
            continue

        # Look up dict entry for this word
        dict_phones = None
        for key in pinyin_dict:
            if key.lower() == word:
                dict_phones = pinyin_dict[key]
                break

        # Punctuation: pass through as-is
        if _is_punct(w_iv.text):
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
            continue

        # NVV token: one self-referential phone
        if is_nvv_token(w_iv.text):
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
            continue

        if dict_phones and len(dict_phones) >= 1:
            # Initial + final from fullpinyin dict
            if len(dict_phones) == 1 or len(word_phones) <= 1:
                # Zero-initial or single phone: entire interval = dict phone
                new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dict_phones[0]))
            else:
                # Initial: first MFA phone → dict initial
                new_intervals.append(Interval(word_phones[0][0], word_phones[0][1], dict_phones[0]))
                # Final: remaining MFA phones combined → dict final
                final_start = word_phones[1][0] if len(word_phones) > 1 else word_phones[0][1]
                final_end = word_phones[-1][1]
                final_label = " ".join(dict_phones[1:]) if len(dict_phones) > 2 else dict_phones[1]
                new_intervals.append(Interval(final_start, final_end, final_label))
        else:
            # Fallback: 1:1 IPA→pinyin
            for s, e, txt in word_phones:
                new_intervals.append(Interval(s, e, ipa_to_pinyin.get(txt, txt)))

    return Tier("pinyin_phones", phones_tier.xmin, phones_tier.xmax, new_intervals)


def _build_pinyin_phones_1to1(phones_tier: Tier, ipa_to_pinyin: dict[str, str]) -> Tier:
    """Fallback: 1:1 IPA→pinyin mapping when words_tier/pinyin_dict unavailable."""
    new_intervals = []
    for iv in phones_tier.intervals:
        txt = iv.text.strip()
        if is_silence(txt):
            new_intervals.append(Interval(iv.xmin, iv.xmax, silence_label(iv.duration)))
        else:
            new_intervals.append(Interval(iv.xmin, iv.xmax, ipa_to_pinyin.get(txt, txt)))
    return Tier("pinyin_phones", phones_tier.xmin, phones_tier.xmax, new_intervals)


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
    """Merge sp0 gaps that lack punctuation; flag sp1-3 gaps for filtering.

    After the punctuation–silence cross-check, any silence between words that
    has *no* corresponding punctuation is an unexpected pause:
      - ``<sp0>`` (< 0.2 s)  → merge into the previous word (extend phone,
        word, and pinyin_phones tiers in sync)
      - ``<sp1-3>`` (≥ 0.2 s) → return as filter reasons
    """
    words_tier = tier_by_name(textgrid, "words")
    phones_tier = tier_by_name(textgrid, "phones")
    pp_tier = tier_by_name(textgrid, "pinyin_phones")
    if words_tier is None or phones_tier is None or pp_tier is None:
        return []

    pinyin_tokens = pinyin_text.split()
    word_items = [(iv.text.strip(), is_silence(iv.text)) for iv in words_tier.intervals]
    tg_word_idx = [i for i, (text, is_sil) in enumerate(word_items)
                   if not is_sil and not _is_punct(text)]
    py_word_idx = [i for i, t in enumerate(pinyin_tokens) if _is_word_like(t)]

    if len(tg_word_idx) != len(py_word_idx) or len(tg_word_idx) == 0:
        return []

    n = len(tg_word_idx)

    # Build gap_sil (only inter-word gaps, index 1..n-1 → words k-1 → k)
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
        gap_punct[k] = any(_is_punct(pinyin_tokens[i]) for i in range(lo, hi))

    filter_reasons = []

    # Process gaps in REVERSE order (to preserve indices during deletion)
    for k in range(n - 1, 0, -1):
        sil_label = gap_sil[k]
        has_punct = gap_punct[k]
        if sil_label is None or has_punct:
            continue  # no silence, or silence expected (punctuation present)

        # Silence without punctuation — unexpected pause
        if sil_label in ("<sp1>", "<sp2>", "<sp3>"):
            filter_reasons.append("unexpected_silence")
            continue

        # <sp0>: merge into previous word (word index = tg_word_idx[k-1])
        prev_word_idx = tg_word_idx[k - 1]
        sil_idx = None
        for j in range(prev_word_idx + 1, tg_word_idx[k]):
            if word_items[j][1]:
                sil_idx = j
                break
        if sil_idx is None:
            continue

        sil_iv = words_tier.intervals[sil_idx]
        prev_w_iv = words_tier.intervals[prev_word_idx]

        # Extend word
        prev_w_iv.xmax = sil_iv.xmax

        # Extend last phone of previous word
        for pi in range(len(phones_tier.intervals) - 1, -1, -1):
            p = phones_tier.intervals[pi]
            if not is_silence(p.text) and p.text != 'spn' and abs(p.xmax - sil_iv.xmin) < 0.01:
                p.xmax = sil_iv.xmax
                # Fix next interval's xmin
                if pi + 1 < len(phones_tier.intervals):
                    phones_tier.intervals[pi + 1].xmin = sil_iv.xmax
                break

        # Remove silence from all three tiers
        for tier, idxs in [
            (words_tier, (sil_idx,)),
            (phones_tier, ()),
            (pp_tier, ()),
        ]:
            pass  # handled below

        # Remove from words tier
        del words_tier.intervals[sil_idx]
        # Remove matching from phones tier (find by time match)
        for pi, p in enumerate(phones_tier.intervals):
            if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 and abs(p.xmax - sil_iv.xmax) < 0.01:
                del phones_tier.intervals[pi]
                break
        # Remove matching from pinyin_phones tier
        for pi, p in enumerate(pp_tier.intervals):
            if is_silence(p.text) and abs(p.xmin - sil_iv.xmin) < 0.01 and abs(p.xmax - sil_iv.xmax) < 0.01:
                del pp_tier.intervals[pi]
                break

        # Update tg_word_idx for subsequent iterations (indices shifted after deletion)
        for i in range(len(tg_word_idx)):
            if tg_word_idx[i] > sil_idx:
                tg_word_idx[i] -= 1

    # Clean up zero-duration remnants in all tiers
    for tier in (words_tier, phones_tier, pp_tier):
        tier.intervals = [iv for iv in tier.intervals
                          if iv.duration > 0.001 or not iv.text.strip()]

    return filter_reasons


def _finalise_textgrid(textgrid: TextGrid, raw_text: str, pinyin_text: str,
                       args) -> TextGrid:
    """Clean up corrected text and restructure tiers for final output.

    1. Remove ``[sp]`` markers from corrected_text (merged as sp0).
    2. Prefix ``<sp1>`` to mark leading silence.
    3. Replace raw_text tier with the final text.
    4. Sync pinyin tier punctuation + ``<sp1>`` prefix.
    5. Insert a hanzi tier (one CJK char per word interval).
    6. Reorder: raw_text, pinyin, hanzi, words, phones, pinyin_phones.
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
    hanzi_tier = _build_hanzi_tier(words_tier, raw_text) if words_tier else None

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
    py_words = [t for t in pinyin_text.split() if _is_word_like(t)]
    # Build final_text character sequence: word chars vs punct
    final_chars = list(final_text.replace('<sp1>', ''))
    result = []
    word_idx = 0
    for ch in final_chars:
        if _is_word_like(ch):
            if word_idx < len(py_words):
                result.append(py_words[word_idx])
                word_idx += 1
        elif _is_punct(ch):
            result.append(ch)
        else:
            result.append(ch)

    return '<sp1> ' + ' '.join(result)


def _extract_word_chars(text: str) -> list[str]:
    """Extract word-like chars from raw text, grouping consecutive non-CJK alpha chars
    and trailing digits (pinyin tone numbers)."""
    result = []
    buf = ""
    for c in text:
        if _is_cjk(c):
            if buf:
                result.append(buf)
                buf = ""
            result.append(c)
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


def _build_hanzi_tier(words_tier: Tier, raw_text: str) -> Tier:
    """Create a hanzi tier with one CJC character per non-silence word interval."""
    # Strip <sp1> prefix for character alignment
    chars = _extract_word_chars(raw_text.replace('<sp1>', ''))
    intervals = []
    char_idx = 0
    for iv in words_tier.intervals:
        if is_silence(iv.text) or not iv.text.strip():
            intervals.append(Interval(iv.xmin, iv.xmax, silence_label(iv.duration)))
        else:
            if char_idx < len(chars):
                intervals.append(Interval(iv.xmin, iv.xmax, chars[char_idx]))
                char_idx += 1
            else:
                intervals.append(Interval(iv.xmin, iv.xmax, iv.text))
    return Tier("hanzi", words_tier.xmin, words_tier.xmax, intervals)


def _word_rms(audio: list[float], sr: int, xmin: float, xmax: float) -> float:
    """Mean absolute amplitude of a time slice."""
    s = max(0, int(xmin * sr))
    e = min(len(audio), int(xmax * sr))
    if e <= s:
        return 0.0
    seg = [abs(v) for v in audio[s:e]]
    return sum(seg) / len(seg) if seg else 0.0


def _is_cjk(ch: str) -> bool:
    return '一' <= ch <= '鿿'


def is_nvv_token(s: str) -> bool:
    """Check if token is an NVV uppercase label (BREATHING, QUESTION-YI, etc.)."""
    import re as _re
    return bool(_re.match(r'^[A-Z][A-Z0-9-]*[A-Z0-9]$', s))


def _is_word_like(s: str) -> bool:
    """True for CJK characters, pinyin syllables (ni3), English words, digits, NVV labels."""
    if not s:
        return False
    return _is_cjk(s) or s[0].isalpha() or s.isdigit() or is_nvv_token(s)


def _is_punct(s: str) -> bool:
    return bool(s.strip()) and not _is_word_like(s)


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
                   if _is_word_like(t) and not is_nvv_token(t)]
    tg_word_idx = [i for i, (text, is_sil) in enumerate(word_items)
                   if not is_sil and not is_nvv_token(text) and not _is_punct(text)]

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
        gap_punct[0] = any(_is_punct(pinyin_tokens[i]) for i in range(0, py_word_idx[0]))

    # between-word punct
    for k in range(n - 1):
        lo = py_word_idx[k] + 1
        hi = py_word_idx[k + 1]
        gap_punct[k + 1] = any(_is_punct(pinyin_tokens[i]) for i in range(lo, hi))

    # trailing punct
    if py_word_idx[-1] < len(pinyin_tokens) - 1:
        gap_punct[n] = any(_is_punct(pinyin_tokens[i])
                           for i in range(py_word_idx[-1] + 1, len(pinyin_tokens)))

    # ---- walk raw Chinese text and produce corrected version ----
    result = []
    word_idx = 0  # how many word-characters have been emitted

    for ch in raw_text:
        if _is_word_like(ch):
            if word_idx > 0:
                gap_pos = word_idx  # gap before this word
                if gap_pos < len(gap_sil) and gap_sil[gap_pos] and not gap_punct[gap_pos]:
                    result.append('[sp]')
            result.append(ch)
            word_idx += 1
        elif _is_punct(ch):
            gap_pos = word_idx  # gap after the last word
            if gap_pos < len(gap_sil):
                if gap_sil[gap_pos]:
                    result.append(ch)
                # else: silence missing → drop the punctuation
            else:
                result.append(ch)
        else:
            result.append(ch)  # whitespace, etc.

    return ''.join(result)


# ---------------------------------------------------------------------------
# Energy-based fix (unchanged)
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav:
        ch = wav.getnchannels()
        sw = wav.getsampwidth()
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sw == 1:
        return [(s - 128) / 128.0 for s in frames[::ch]], sr
    if sw in {2, 4}:
        tc = "h" if sw == 2 else "i"
        scale = float(2 ** (8 * sw - 1))
        samples = array.array(tc)
        samples.frombytes(frames)
        return [samples[i] / scale for i in range(0, len(samples), ch)], sr
    raise ValueError(f"Unsupported sample width: {sw}")


def frame_rms(audio: list[float], frame_size: int, hop_size: int) -> list[float]:
    if len(audio) < frame_size:
        return []
    return [math.sqrt(sum(s * s for s in audio[s:s + frame_size]) / frame_size + 1e-12)
            for s in range(0, len(audio) - frame_size + 1, hop_size)]


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def find_speech_in_silence(
    audio: list[float], sr: int, sil_start: float, sil_end: float,
    search_sec: float, frame_ms: float, hop_ms: float,
    thresh_ratio: float, min_region_sec: float,
) -> tuple[float, float] | None:
    search_end = min(sil_end, sil_start + search_sec)
    ss = max(0, int(sil_start * sr))
    es = min(len(audio), int(search_end * sr))
    if es <= ss:
        return None
    seg = audio[ss:es]
    fs = max(1, int(frame_ms / 1000.0 * sr))
    hs = max(1, int(hop_ms / 1000.0 * sr))
    rms = frame_rms(seg, fs, hs)
    if not rms:
        return None
    tail = rms[max(0, int(len(rms) * 0.6)):]
    noise = median(tail) if tail else median(rms)
    peak = max(rms)
    threshold = max(noise * thresh_ratio, peak * 0.15)
    active = [v > threshold for v in rms]
    min_f = max(1, int(min_region_sec / (hop_ms / 1000.0)))
    first = None
    for i in range(len(active)):
        if sum(active[i:i + min_f]) >= min_f:
            first = i
            break
    if first is None:
        return None
    last = None
    for i in range(first, len(active)):
        if not active[i] and sum(1 for j in range(i, min(i + min_f, len(active))) if not active[j]) >= min_f:
            last = i
            break
    if last is None:
        last = max(i for i, v in enumerate(active) if v) + 1
    sp_start = sil_start + first * hop_ms / 1000.0
    sp_end = sil_start + last * hop_ms / 1000.0 + frame_ms / 1000.0
    sp_end = min(sp_end, sil_end)
    if sp_end - sp_start < min_region_sec or sp_start - sil_start > 0.35:
        return None
    return sp_start, sp_end


def nonzero_mean(segment: list[float]) -> float:
    """Mean absolute amplitude of non-zero values in a segment."""
    if not segment:
        return 0.0
    nonzero = [abs(v) for v in segment if abs(v) > 1e-12]
    if not nonzero:
        return 0.0
    return sum(nonzero) / len(nonzero)


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
        if (iv.text.strip().lower().rstrip('12345') in {w.rstrip('12345') for w in CHINESE_SHORT_WORDS}
                and iv.duration < args.fix_short_word_sec
                and is_silence(next_iv.text)
                and next_iv.duration >= args.fix_min_silence_sec):
            candidates.append(idx)
    if not candidates:
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
        for s in range(0, len(seg) - frame_size + 1, hop_size):
            frame = seg[s:s + frame_size]
            sil_rms_vals.append(math.sqrt(sum(v * v for v in frame) / frame_size + 1e-12))

    if sil_rms_vals:
        sorted_sil = sorted(sil_rms_vals)
        # Use bottom 10% median as noise floor — avoids circular pollution
        # where loud mislabeled silences inflate the median
        noise_floor = median(sorted_sil[:max(1, int(len(sorted_sil) * 0.1))])
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
        if seg:
            speech_rms.append(median([abs(s) for s in seg]))
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
        if not seg:
            continue
        nonzero = [abs(s) for s in seg if abs(s) > 0]
        sil_energy = sum(nonzero) / len(nonzero) if nonzero else 0.0

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
    phones = tier_by_name(textgrid, "phones")
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
        # Word energy at silence level → likely misaligned into a silence gap
        if args.filter_word_energy_ratio > 0 and noise_floor > 1e-8:
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
        if (w.text.strip() and w.duration < 0.12
                and prev_w and is_silence(prev_w.text) and next_w and is_silence(next_w.text)
                and prev_w.duration >= args.filter_flank_silence_sec
                and next_w.duration >= args.filter_flank_silence_sec):
            issues.append({"rule": "short_word_between_silences", "text": w.text})
    for pi, p in enumerate(phones.intervals):
        if not p.text.strip() or is_silence(p.text):
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

def find_original_text(stem: str, raw_text_dir: Path | None) -> str:
    """Find the original Chinese text for a given output stem (searches recursively)."""
    if not raw_text_dir or not raw_text_dir.exists():
        return ""
    # Try stem.txt (flat or recursive)
    candidates = list(raw_text_dir.rglob(f"{stem}.txt"))
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
        candidates = list(raw_text_dir.rglob(f"{base}.txt"))
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
    for pi in range(len(resolved)):
        ps, pe, ptext, pkind = resolved[pi]
        if pkind != "punct":
            continue
        for wi in range(len(resolved)):
            ws, we, wtext, wkind = resolved[wi]
            if wkind != "word" or is_silence(wtext):
                continue
            if ws < pe and we > ps:  # overlap exists
                if ws <= ps and we >= pe:
                    resolved[pi] = (0, 0, "", pkind)
                elif ws <= ps:
                    resolved[pi] = (we, pe, ptext, pkind)
                    ps = we  # punct start 推到 word end
                elif we >= pe:
                    resolved[pi] = (ps, ws, ptext, pkind)
                    pe = ws  # punct end 拉到 word start
                else:
                    resolved[pi] = (0, 0, "", pkind)

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

    # 微小静音间隙合并到后续标点或 NVV (<sp> → 吸收进标点/NVV)
    for gi in range(len(merged)):
        gs, ge, gtext, gkind = merged[gi]
        if not (gkind == "word" and is_silence(gtext)):
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
        if not (gkind == "word" and is_silence(gtext)):
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
        if not (gkind == "word" and is_silence(gtext)):
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
        # 重建: 保留非静音 + 最后标点(延伸)
        new_merged = []
        for m in merged:
            if m is last_punct:
                new_merged.append((punct_start, words_tier.xmax, punct_text, "punct"))
            elif m[0] < punct_start:
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
        new_pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax, pp_intervals)
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

    fs = max(1, int(0.02 * sr))
    hs = max(1, int(0.01 * sr))
    all_rms = frame_rms(audio, fs, hs)
    nf = median(sorted(all_rms)[:max(1, int(len(all_rms) * 0.15))]) if all_rms else 1e-6
    threshold = max(nf * 2.5, 0.005)

    intervals = list(words_tier.intervals)
    n = len(intervals)

    for i in range(n - 1):
        iv_curr = intervals[i]
        iv_next = intervals[i + 1]

        if is_nvv_token(iv_curr.text) or _is_punct(iv_curr.text):
            continue
        if iv_curr.text.strip() in SILENCE_LABELS:
            continue
        if not _is_word_like(iv_curr.text):
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

        ss = int(ellipsis_start * sr)
        ee = int(ellipsis_end * sr)
        seg = audio[ss:ee]

        el_fs = max(1, int(0.01 * sr))
        el_hs = max(1, int(0.005 * sr))
        rms_vals = frame_rms(seg, el_fs, el_hs)
        if not rms_vals:
            continue

        # Find energy decay: ≥2 consecutive fine frames below threshold
        consecutive_needed = 2
        decay_idx = len(rms_vals)
        below = 0
        for j, v in enumerate(rms_vals):
            if v < threshold:
                below += 1
                if below >= consecutive_needed:
                    decay_idx = j - consecutive_needed + 1
                    break
            else:
                below = 0

        if decay_idx <= 0:
            extend_target = ellipsis_start + dur * 0.35
        elif decay_idx >= len(rms_vals):
            continue
        else:
            decay_time = max(0.0, ellipsis_start + decay_idx * (el_hs / sr))
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
            if is_nvv_token(pp_cur.text) or _is_punct(pp_cur.text):
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

    # 估算噪声地板 (frame_rms 用 sample 数)
    fs = max(1, int(0.02 * sr))
    hs = max(1, int(0.01 * sr))
    all_rms = frame_rms(audio, fs, hs)
    nf = median(sorted(all_rms)[:max(1, int(len(all_rms) * 0.15))]) if all_rms else 1e-6
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

        # 检查 … 段是否有足够能量
        ellipsis_start = iv_next.xmin
        ellipsis_end = iv_next.xmax
        ss = int(ellipsis_start * sr)
        ee = int(ellipsis_end * sr)
        if ee <= ss:
            continue
        seg = audio[ss:ee]
        el_fs = max(1, int(0.01 * sr))
        el_hs = max(1, int(0.005 * sr))
        rms_vals = frame_rms(seg, el_fs, el_hs)
        if not rms_vals:
            continue
        energy_ratio = sum(1 for v in rms_vals if v > threshold) / len(rms_vals)

        # ≥30% 帧有能量 → 合并; NVV 后极短省略号 (<100ms) 无条件合并
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
                                  min_word_dur: float = 0.03) -> Tier:
    """词落在静音段时向后搜索语音起点, 整体后移 (不越过后词)."""
    fs = max(1, int(0.02 * sr))
    hs = max(1, int(0.01 * sr))
    all_rms = frame_rms(audio, fs, hs)
    if not all_rms:
        return words_tier
    nf = median(sorted(all_rms)[:max(1, int(len(all_rms) * 0.15))])
    threshold = max(nf * 3.0, 0.005)

    intervals = list(words_tier.intervals)
    n = len(intervals)

    # 从右往左处理: 后面的词先移, 给前面的词腾空间
    for i in range(n - 1, -1, -1):
        iv = intervals[i]
        if is_silence(iv.text) or not iv.text.strip():
            continue
        word_start = iv.xmin
        word_end = iv.xmax
        dur = word_end - word_start

        # 检查整词能量: 是否完全在静音中
        w_ss = max(0, int(word_start * sr))
        w_ee = min(len(audio), int(word_end * sr))
        word_seg = audio[w_ss:w_ee] if w_ee > w_ss else []
        word_rms = float(sum(abs(v) for v in word_seg)) / len(word_seg) if len(word_seg) else 0

        if word_rms >= threshold:
            continue  # 词有能量, 不需要整体移动

        # 词在静音中 → 搜索后方的语音起点
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
        onset = None
        for pos in range(s_sample, e_sample, frame_s):
            pe = min(pos + frame_s, len(audio))
            chunk = audio[pos:pe]
            rms_val = float(sum(abs(v) for v in chunk)) / len(chunk) if len(chunk) else 0
            if rms_val > threshold:
                onset = pos / sr
                break

        if onset is None or onset <= word_start:
            continue

        # 整体后移: 不越过后词, 空间不够则放弃
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

    intervals = [iv for iv in intervals if iv.xmax > iv.xmin + 0.001]
    return Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)


def _snap_to_ctc(words_tier: Tier, pp_tier: Tier | None,
                  ctc_tokens: list[dict],
                  snap_threshold: float = 0.3,
                  punct_entries: list[dict] | None = None) -> tuple[Tier, Tier | None]:
    """Snap MFA word boundaries to CTC anchors only when they differ too much.

    If |MFA - CTC| <= snap_threshold: trust MFA, keep MFA boundaries.
    If |MFA - CTC| >  snap_threshold: MFA likely misaligned, snap to CTC.

    When keeping MFA boundaries, silence gaps use CTC gap positions to
    correctly place punctuation between words.
    """
    mfa_words = [(i, iv) for i, iv in enumerate(words_tier.intervals)
                 if not is_silence(iv.text) and iv.text.strip() not in ("", "<eps>")
                 and not _is_punct(iv.text)]

    if len(mfa_words) != len(ctc_tokens):
        import sys
        print(f"  _snap_to_ctc: token count mismatch (MFA={len(mfa_words)}, CTC={len(ctc_tokens)}) — "
              f"skipping boundary snap", file=sys.stderr)
        return words_tier, pp_tier

    new_word_ivs = []        # (xmin, xmax, text, source)
    new_phone_ivs = []       # (xmin, xmax, text)

    prev_end = 0.0
    prev_ctc_end = 0.0

    for idx, (wi, mfa_iv) in enumerate(mfa_words):
        ctc = ctc_tokens[idx]
        ctc_start = ctc["start_s"]
        ctc_end = ctc["end_s"]
        mfa_start = mfa_iv.xmin
        mfa_end = mfa_iv.xmax
        mfa_dur = mfa_end - mfa_start if mfa_end > mfa_start else 0.001

        start_diff = abs(mfa_start - ctc_start)
        end_diff = abs(mfa_end - ctc_end)
        use_mfa = (start_diff <= snap_threshold and end_diff <= snap_threshold)
        # NVV 不参考 MFA: MFA 没有 NVV 声学模型, 直接用 CTC 锚点
        if is_nvv_token(mfa_iv.text):
            use_mfa = False
        # 保护: MFA 词过短时信任 CTC (避免 yi4 30ms 这类情况)
        ctc_dur = ctc_end - ctc_start
        if use_mfa and mfa_dur < 0.06 and ctc_dur > 0.15:
            use_mfa = False

        if use_mfa:
            word_start = mfa_start
            word_end = mfa_end
            # 差异较大时用中间点: 前半间隙归前词, 后半间隙归当前词
            if start_diff > 0.15:
                word_start = round((ctc_start + mfa_start) / 2, 3)
            if end_diff > 0.15:
                word_end = round((ctc_end + mfa_end) / 2, 3)
            # MFA 把词放在长静音之后, CTC 说更早 → 取标点之后的纯静音间隙
            # 如果纯静音间隙 > 100ms, 优先用 CTC 起点
            SILENCE_GAP_SNAP_THRESH = 0.10
            if mfa_start > ctc_start and start_diff <= snap_threshold:
                gap_start = prev_end
                if punct_entries:
                    for p in punct_entries:
                        if p["start_s"] < mfa_start and p["end_s"] > prev_end:
                            gap_start = max(gap_start, p["end_s"])
                pure_silence_gap = mfa_start - gap_start
                if pure_silence_gap > SILENCE_GAP_SNAP_THRESH:
                    word_start = max(ctc_start, gap_start)
        else:
            word_start = ctc_start
            word_end = ctc_end

        # 防止词间重叠: start 不能在前一词 end 之前
        if word_start < prev_end - 0.002:
            word_start = prev_end

        # NVV 吸收前方小间隙 (<200ms): 天然包含周围静音
        # 超过 200ms 或间隙中有标点 → 不吸收
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
        prev_ctc_end = ctc_end

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

    # Build new tiers
    new_words_tier = Tier(words_tier.name, words_tier.xmin, words_tier.xmax,
                          [Interval(s, e, t) for s, e, t, _ in new_word_ivs])

    new_pp_tier = None
    if pp_tier is not None and new_phone_ivs:
        new_pp_tier = Tier(pp_tier.name, pp_tier.xmin, pp_tier.xmax,
                           [Interval(s, e, t) for s, e, t in new_phone_ivs])

    return new_words_tier, new_pp_tier


def process_one(tg_path: Path, txt_dir: Path, wav_dir: Path,
                output_dir: Path, filtered_dir: Path, args,
                ipa_to_pinyin: dict[str, str],
                pinyin_dict: dict[str, list[str]],
                pinyin_case: dict[str, str] | None = None) -> dict:
    stem = tg_path.stem
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

    # Fix MFA's forced lowercase: use dictionary's canonical form
    if pinyin_case:
        for iv in words_tier.intervals:
            word = iv.text.strip()
            if word and not is_silence(word):
                canonical = pinyin_case.get(word.lower())
                if canonical is not None and canonical != word:
                    iv.text = canonical

    # Tier 1: original Chinese text (from data_dir)
    raw_text = find_original_text(stem, args.raw_text_dir)
    if not raw_text:
        # Try NVASR Chinese ASR output
        cn_path = txt_dir / f"{stem}_text_cn.txt"
        if cn_path.exists():
            raw_text = cn_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        # Fallback: use the pinyin txt content
        raw_text = txt_path.read_text(encoding="utf-8").strip()

    # Tier 2: pinyin with punctuation (from corpus txt)
    pinyin_text = txt_path.read_text(encoding="utf-8").strip()

    # Fix <unk> from MFA: self-referential NVV tokens (BREATHING etc.) not in
    # MFA acoustic model get replaced with <unk>.  Restore from .lab tokens.
    lab_tokens = pinyin_text.split()
    lab_idx = 0
    for iv in words_tier.intervals:
        if is_silence(iv.text) or iv.text.strip() in ("", "<eps>"):
            continue
        if lab_idx < len(lab_tokens):
            if iv.text.strip() == "<unk>" and is_nvv_token(lab_tokens[lab_idx]):
                iv.text = lab_tokens[lab_idx]
            lab_idx += 1

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

    merge_report = []
    if args.merge_silence:
        new_tg, merge_report = merge_short_silences(
            new_tg, wav_path if wav_path.exists() else None, args, wav_audio, wav_sr)
        report["silence_merges"] = merge_report

    # Energy fix
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

    # Filter
    align_issues = []
    if args.filter_suspicious:
        align_issues = detect_issues(new_tg, args, wav_path if wav_path.exists() else None,
                                     wav_audio, wav_sr)

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

    # Merge sp0 gaps that lack punctuation; flag sp1-3 gaps for filtering
    sil_filter_reasons = []
    if args.handle_unexpected_sil:
        sil_filter_reasons = handle_unexpected_silences(new_tg, pinyin_text)
        if sil_filter_reasons:
            report["unexpected_silence"] = sil_filter_reasons

    # Finalise: strip [sp] markers (merged), add <sp1> prefix,
    # sync pinyin, insert hanzi tier, reorder everything.
    if args.enable_text_correction:
        new_tg = _finalise_textgrid(new_tg, raw_text, pinyin_text, args)

    # 输出路径先默认 output, 最终检查时再决定是否重定向到 filtered
    out_path = output_dir / tg_path.name
    stale = filtered_dir / tg_path.name

    # Snap MFA word boundaries to CTC anchors, remap phones proportionally
    tokens_path = txt_dir / f"{stem}_tokens.jsonl"
    punct_path = txt_dir / f"{stem}_punct.json"
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
                                                   punct_entries=punct_entries)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                elif t.name == "pinyin_phones" and pp_tier is not None:
                    new_tg.tiers[i] = pp_tier

    # 能量微调: 词边界落在静音段时向后搜索语音起点
    if wav_audio is not None:
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            words_tier = _refine_boundaries_by_energy(words_tier, wav_audio, wav_sr)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "words":
                    new_tg.tiers[i] = words_tier
                    break

    # Inject punctuation from CTC anchors AFTER snapping
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

    # NVV 后紧跟的短省略号无条件合并到 NVV (不依赖能量检测)
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

    # 原有的能量检测 NVV+省略号合并 (处理有能量的长省略号)
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

    # 能量检测: content word + … → 延长词边界 (如 后→… 中的拖长音)
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

    # 检测被吞掉的标点: CTC punct 条目在 words tier 中时间匹配不到 → 从文本删除
    # 删除对应位置的那个标点, 而非总是第一个
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
                # 标点没对应 → 检查是否有 … 在同一位置 (CTC 长停顿替换了原标点)
                replaced = False
                if p_char in '，。！？':
                    for wi, iv in enumerate(words_tier.intervals):
                        if iv.text.strip() == '…' and abs(iv.xmin - p_start) < 0.3:
                            iv.text = p_char  # 用原标点替换省略号
                            replaced = True
                            break
                if not replaced:
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

    # Rebuild hanzi from final words tier (now with punct), using Chinese raw_text
    final_words_tier = tier_by_name(new_tg, "words")
    if final_words_tier:
        hanzi_tier = _build_hanzi_tier(final_words_tier, raw_text)
        if hanzi_tier:
            for i, t in enumerate(new_tg.tiers):
                if t.name == "hanzi":
                    new_tg.tiers[i] = hanzi_tier
                    break
        # Rebuild pinyin_phones from phones_tier with final word boundaries
        final_phones_tier = tier_by_name(new_tg, "phones")
        if final_phones_tier and final_words_tier:
            synced_pp = build_pinyin_phones_tier(final_phones_tier, ipa_to_pinyin,
                                                  final_words_tier, pinyin_dict)
            # Extend first/last phone to word boundaries (fix unvoiced stop gaps)
            w_idx = 0
            new_pp_ivs = list(synced_pp.intervals)
            for w_iv in final_words_tier.intervals:
                if is_silence(w_iv.text) or not w_iv.text.strip():
                    continue
                # Find phones in this word
                while w_idx < len(new_pp_ivs) and new_pp_ivs[w_idx].xmax <= w_iv.xmin + 0.005:
                    w_idx += 1
                word_pps = []
                while w_idx < len(new_pp_ivs) and new_pp_ivs[w_idx].xmin < w_iv.xmax - 0.005:
                    word_pps.append(w_idx)
                    w_idx += 1
                if word_pps:
                    first = word_pps[0]
                    last = word_pps[-1]
                    if new_pp_ivs[first].xmin > w_iv.xmin + 0.005:
                        new_pp_ivs[first] = Interval(w_iv.xmin, new_pp_ivs[first].xmax, new_pp_ivs[first].text)
                    if w_iv.xmax > new_pp_ivs[last].xmax + 0.005:
                        new_pp_ivs[last] = Interval(new_pp_ivs[last].xmin, w_iv.xmax, new_pp_ivs[last].text)
            synced_pp = Tier(synced_pp.name, synced_pp.xmin, synced_pp.xmax, new_pp_ivs)
            for i, t in enumerate(new_tg.tiers):
                if t.name == "pinyin_phones":
                    new_tg.tiers[i] = synced_pp
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
            raw_tokens = [iv.text for iv in hanzi_after.intervals
                          if not is_silence(iv.text) and iv.text.strip()]
            raw_tier.intervals[0].text = "".join(raw_tokens) if raw_tokens else raw_tier.intervals[0].text

    # 最终恢复: CTC 长停顿注入 … 覆盖了原标点, 用 CTC punct 替换回去
    if punct_entries:
        words_tier = tier_by_name(new_tg, "words")
        if words_tier:
            for p in punct_entries:
                if p["word"] not in '，。！？':
                    continue
                # 检查 words tier 中是否有 …, 且位置接近 CTC punct
                for iv in words_tier.intervals:
                    if iv.text.strip() == '…' and abs(iv.xmin - p["start_s"]) < 0.3:
                        iv.text = p["word"]
                        break

    # ================================================================
    # 最终筛选: 所有处理完成后再统一判断 (用最终的边界和静音结构)
    # ================================================================
    filter_reasons = []

    # sp3 / mid_sp: 检查最终 words 层的静音结构
    words_tier = tier_by_name(new_tg, "words")
    if words_tier:
        for iv in words_tier.intervals:
            if iv.text.strip() == "<sp3>":
                filter_reasons.append("sp3")
        sp_in_mid = False
        for i, iv in enumerate(words_tier.intervals):
            if i > 0 and is_silence(iv.text) and iv.text.strip():
                sp_in_mid = True
                break
        if sp_in_mid:
            filter_reasons.append("mid_sp")

    # suspicious_alignment (来自 detect_issues, 已包含 short_phone 等)
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
            all_rms = frame_rms(wav_audio, fs, hs)
            nf_bgm = median(sorted(all_rms)[:max(1, int(len(all_rms) * 0.6))]) if all_rms else 1e-6
            nf_bgm = max(nf_bgm, 1e-6)
            bgm_threshold = max(nf_bgm * args.bgm_noise_floor_ratio, 0.005)
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
                sil_energy = _word_rms(wav_audio, wav_sr, iv.xmin, iv.xmax)
                total_sil_dur += iv.xmax - iv.xmin
                if sil_energy > bgm_threshold:
                    suspect_intervals.append({"xmin": round(iv.xmin, 3), "xmax": round(iv.xmax, 3),
                                              "duration": round(iv.xmax - iv.xmin, 3),
                                              "energy": round(sil_energy, 6),
                                              "noise_floor": round(nf_bgm, 6)})
                    suspect_dur += iv.xmax - iv.xmin
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
            fs = max(1, int(0.02 * wav_sr))
            hs = max(1, int(0.01 * wav_sr))
            all_rms = frame_rms(wav_audio, fs, hs)
            nf = median(sorted(all_rms)[:max(1, int(len(all_rms) * 0.15))]) if all_rms else 1e-6
            threshold = max(nf * args.filter_word_energy_ratio, 0.003)
            for iv in words_tier.intervals:
                if is_silence(iv.text) or not iv.text.strip():
                    continue
                if iv.text.strip() in '，。…！？、；：':
                    continue
                w_energy = _word_rms(wav_audio, wav_sr, iv.xmin, iv.xmax)
                if 0 < w_energy < threshold:
                    align_issues.append({"rule": "word_in_silence", "text": iv.text,
                                         "energy": round(w_energy, 6),
                                         "noise_floor": round(nf, 6)})
        # 更新 BGM + word_in_silence 到过滤原因
        if bgm_issues and "bgm_suspect" not in filter_reasons:
            filter_reasons.append("bgm_suspect")
            report["bgm_issues"] = bgm_issues
        if any(i["rule"] == "word_in_silence" for i in align_issues):
            if "word_in_silence" not in filter_reasons:
                filter_reasons.append("word_in_silence")

    # 统一设置过滤状态和输出路径
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

    # Drop phones tier (IPA) — used internally, not needed in final output
    new_tg.tiers = [t for t in new_tg.tiers if t.name != "phones"]

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path}")
    if stale.exists() and args.overwrite:
        stale.unlink()
    write_textgrid(new_tg, out_path)
    report["output"] = str(out_path)
    report["textgrid_duration"] = round(tg.xmax - tg.xmin, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description="Post-process MFA TextGrids for Chinese alignment.")
    parser.add_argument("--txt-dir", type=Path, default=PROJECT_ROOT / "corpus_clean" / "txt")
    parser.add_argument("--textgrid-dir", type=Path, default=PROJECT_ROOT / "aligned")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--filtered-dir", type=Path, default=PROJECT_ROOT / "filtered")
    parser.add_argument("--wav-dir", type=Path, default=PROJECT_ROOT / "corpus_clean" / "wav")
    parser.add_argument("--raw-text-dir", type=Path, default=None,
                        help="Directory with original Chinese text files")
    parser.add_argument("--pinyin-dict", type=Path, default=PROJECT_ROOT / "dict" / "fullpinyin_enword.dict")
    parser.add_argument("--ipa-dict", type=Path, default=PROJECT_ROOT / "dict" / "mfa_ipa.dict")
    parser.add_argument("--tone-ref", type=Path, default=PROJECT_ROOT / "output" / "tone_mapping.json",
                        help="Output path for tone reference table")
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
    parser.add_argument("--filter-min-word-sec", type=float, default=0.15)
    parser.add_argument("--filter-min-word-dur-sec", type=float, default=0.02,
                        help="Absolute minimum word duration (below = misaligned).")
    parser.add_argument("--filter-word-energy-ratio", type=float, default=2.0,
                        help="Flag word if energy < noise_floor * N (0 = disabled).")
    parser.add_argument("--filter-min-phone-coverage", type=float, default=0.35)
    parser.add_argument("--filter-edge-gap-sec", type=float, default=0.25)
    parser.add_argument("--copy-errors", action="store_true")
    parser.add_argument("--enable-text-correction", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-check punctuation against silence gaps and emit corrected_text tier.")
    parser.add_argument("--handle-unexpected-sil", action=argparse.BooleanOptionalAction, default=True,
                        help="Merge <sp0> gaps without punct; flag <sp1-3> gaps for filtering.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_dir.mkdir(parents=True, exist_ok=True)

    # Load dictionaries and build IPA→pinyin mapping
    print("Loading dictionaries...")
    pinyin_dict, pinyin_case = load_dict(args.pinyin_dict)
    ipa_dict, _ = load_dict(args.ipa_dict)
    print(f"  Pinyin dict: {len(pinyin_dict)} entries")
    print(f"  IPA dict: {len(ipa_dict)} entries")

    ipa_to_pinyin = build_ipa_to_pinyin_map(pinyin_dict, ipa_dict)
    print(f"  IPA→Pinyin phone mappings: {len(ipa_to_pinyin)}")

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
        return

    reports = []
    for tgp in tg_paths:
        try:
            reports.append(process_one(tgp, args.txt_dir, args.wav_dir,
                                       args.output_dir, args.filtered_dir, args,
                                       ipa_to_pinyin, pinyin_dict, pinyin_case))
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


if __name__ == "__main__":
    main()
