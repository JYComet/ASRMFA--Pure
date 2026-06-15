#!/usr/bin/env python3
"""
Post-process MFA TextGrids for Chinese forced alignment (pinyin + tone numbers).

Builds 5-tier TextGrid:
  raw_text      — original Chinese sentence
  pinyin        — pinyin with tone numbers + [PAUSE]/<PAUSE>
  words         — MFA-aligned pinyin words (with tone numbers)
  phones        — MFA-aligned phones (IPA notation)
  pinyin_phones — IPA phones reverse-mapped to pinyin tone-number notation

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

def load_dict(path: Path) -> dict[str, list[str]]:
    """Load a pronunciation dictionary: {token: [phone1, phone2, ...]}."""
    d = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1:]
    return d


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
    'j': 'y', 'w': 'w', 'ɥ': 'y',
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

# Chinese initials (consonant phones without tone numbers)
CHINESE_INITIALS_SET = {
    "p", "pʰ", "t", "tʰ", "k", "kʰ",
    "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ", "ts", "tsʰ",
    "f", "s", "ɕ", "ʂ", "x",
    "m", "n", "l", "ɻ",
    "j", "w", "ɥ",
    "ŋ", "ʔ",
}


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

    # 2. Fill from dict-based cross-referencing (for compound finals, etc.)
    for token, pinyin_phones in pinyin_dict.items():
        ipa_phones = ipa_dict.get(token)
        if ipa_phones and len(pinyin_phones) == len(ipa_phones):
            for ipa_p, py_p in zip(ipa_phones, pinyin_phones):
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


def build_tone_reference_table(ipa_to_pinyin: dict[str, str]) -> dict:
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

def build_pinyin_phones_tier(words_tier: Tier, phones_tier: Tier,
                              ipa_to_pinyin: dict[str, str],
                              pinyin_dict: dict[str, list[str]]) -> Tier:
    """
    Create a pinyin_phones tier with syllable-structured intervals.
    Uses word-level time intervals from the words tier + dict syllable structure.
    Splits each word's time range into initial + final_with_tone.
    """
    new_intervals = []

    for w_iv in words_tier.intervals:
        w_text = w_iv.text.strip()

        # Silence/pause: keep as-is
        if is_silence(w_text) or w_text == '<eps>' or w_text == '<pause>' or w_text == '[pause]':
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, silence_label(w_iv.duration)))
            continue

        # Look up syllable structure from pinyin dict
        py_phones = pinyin_dict.get(w_text)

        if py_phones and 1 <= len(py_phones) <= 2:
            if len(py_phones) == 1:
                # Standalone final: single interval covering the whole word
                new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, py_phones[0]))
            else:
                # Initial + Final: split the word interval
                # Find the approximate boundary from the phones tier
                # Default: 40% initial, 60% final
                split = w_iv.xmin + w_iv.duration * 0.4

                # Try to find the actual phone boundary from the phones tier
                phone_ivs = [p for p in phones_tier.intervals
                            if not is_silence(p.text) and p.text != 'spn'
                            and p.xmax > w_iv.xmin + 0.005 and p.xmin < w_iv.xmax - 0.005]
                if len(phone_ivs) >= 2:
                    # Use the boundary between first and second phone
                    split = phone_ivs[0].xmax

                new_intervals.append(Interval(w_iv.xmin, split, py_phones[0]))
                new_intervals.append(Interval(split, w_iv.xmax, py_phones[1]))
        else:
            # Fallback: use the whole word interval with the token itself
            new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_text))

    return Tier("pinyin_phones", phones_tier.xmin, phones_tier.xmax, new_intervals)


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
    """Mean of non-zero values in a segment."""
    if not segment:
        return 0.0
    nonzero = [v for v in segment if v > 0]
    if not nonzero:
        return 0.0
    return sum(nonzero) / len(nonzero)


def merge_short_silences(textgrid: TextGrid, wav_path: Path | None, args) -> tuple[TextGrid, list[dict]]:
    """
    Merge short sil intervals into the previous phone when energy conditions are met.

    For each 'sil' interval in the phones tier:
    1. Duration must be < merge_max_sil_sec
    2. Non-zero energy mean > previous phone non-zero mean * merge_energy_threshold

    If both pass, the sil is merged into the previous phone (extend its xmax),
    and the matching <eps> in the words tier is merged into the previous word.
    """
    if wav_path is None or not wav_path.exists():
        return textgrid, []
    words = tier_by_name(textgrid, "words")
    phones = tier_by_name(textgrid, "phones")
    if words is None or phones is None:
        return textgrid, []

    audio, sr = load_audio(wav_path)
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


def fix_short_words(textgrid: TextGrid, wav_path: Path | None, args) -> tuple[TextGrid, list[dict]]:
    if wav_path is None or not wav_path.exists():
        return textgrid, []
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
    audio, sr = load_audio(wav_path)
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
        word_iv.xmax = sp_end
        sil_iv.xmin = sp_end
        for pi in [i for i, p in enumerate(phones.intervals)
                   if p.xmax > word_iv.xmax - 0.5 and p.xmin < sp_end]:
            if not is_silence(phones.intervals[pi].text):
                phones.intervals[pi].xmax = sp_end
        fixes.append({"rule": "short_word_fix", "word": word_iv.text})
    return textgrid, fixes


# ---------------------------------------------------------------------------
# BGM / noise detection (global noise floor + per-silence energy check)
# ---------------------------------------------------------------------------

def detect_bgm_suspect(textgrid: TextGrid, wav_path: Path | None, args) -> list[dict]:
    """
    Detect if silence intervals have abnormally high energy (BGM/noise residual).

    Uses global noise floor estimation (bottom 60% RMS median of entire audio),
    then checks each silence interval against it. Flags the file if too many
    silence intervals are above the noise floor.
    """
    if wav_path is None or not wav_path.exists():
        return []

    phones = tier_by_name(textgrid, "phones")
    if phones is None:
        return []

    audio, sr = load_audio(wav_path)

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


def detect_issues(textgrid: TextGrid, args) -> list[dict]:
    issues = []
    words = tier_by_name(textgrid, "words")
    phones = tier_by_name(textgrid, "phones")
    if words is None or phones is None:
        return [{"rule": "missing_tier"}]
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
        if p.duration < args.filter_short_phone_sec:
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


def process_one(tg_path: Path, txt_dir: Path, wav_dir: Path,
                output_dir: Path, filtered_dir: Path, args,
                ipa_to_pinyin: dict[str, str],
                pinyin_dict: dict[str, list[str]]) -> dict:
    stem = tg_path.stem
    report: dict = {"stem": stem, "status": "ok", "warnings": []}
    txt_path = txt_dir / f"{stem}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing txt: {txt_path}")
    tg = parse_textgrid(tg_path)
    if len(tg.tiers) < 2:
        raise ValueError(f"Need at least 2 tiers in {tg_path}")
    words_tier = tg.tiers[0]
    phones_tier = tg.tiers[1]

    # Tier 1: original Chinese text (from data_dir)
    raw_text = find_original_text(stem, args.raw_text_dir)
    if not raw_text:
        # Fallback: use the pinyin txt content
        raw_text = txt_path.read_text(encoding="utf-8").strip()

    # Tier 2: pinyin with punctuation (from corpus txt)
    pinyin_text = txt_path.read_text(encoding="utf-8").strip()

    # Build 5 tiers
    raw_tier = Tier("raw_text", tg.xmin, tg.xmax, [Interval(tg.xmin, tg.xmax, raw_text)])
    pinyin_tier = Tier("pinyin", tg.xmin, tg.xmax, [Interval(tg.xmin, tg.xmax, pinyin_text)])
    pinyin_phones_tier = build_pinyin_phones_tier(words_tier, phones_tier, ipa_to_pinyin, pinyin_dict)
    new_tg = TextGrid(tg.xmin, tg.xmax,
                      [raw_tier, pinyin_tier, words_tier, phones_tier, pinyin_phones_tier])

    # Find WAV recursively (may be in subdirectory)
    wav_path = wav_dir / f"{stem}.wav"
    if not wav_path.exists():
        candidates = list(wav_dir.rglob(f"{stem}.wav"))
        if candidates:
            wav_path = candidates[0]
    merge_report = []
    if args.merge_silence:
        new_tg, merge_report = merge_short_silences(
            new_tg, wav_path if wav_path.exists() else None, args)
        report["silence_merges"] = merge_report

    # Energy fix
    if args.fix_short_word:
        new_tg, fixes = fix_short_words(new_tg, wav_path if wav_path.exists() else None, args)
        report["fixes"] = fixes

    # BGM/noise detection
    bgm_issues = []
    if args.detect_bgm:
        bgm_issues = detect_bgm_suspect(new_tg, wav_path if wav_path.exists() else None, args)
        if bgm_issues:
            report["bgm_issues"] = bgm_issues

    # Filter
    align_issues = []
    if args.filter_suspicious:
        align_issues = detect_issues(new_tg, args)

    # Relabel all silences
    new_tiers = []
    for tier in new_tg.tiers:
        relabeled = [Interval(iv.xmin, iv.xmax,
                              silence_label(iv.duration) if is_silence(iv.text) else iv.text)
                     for iv in tier.intervals]
        new_tiers.append(Tier(tier.name, tier.xmin, tier.xmax, relabeled))
    new_tg = TextGrid(new_tg.xmin, new_tg.xmax, new_tiers)

    # Check sp3
    filter_reasons = []
    for tier in new_tg.tiers:
        for iv in tier.intervals:
            if iv.text.strip() == "<sp3>":
                filter_reasons.append("sp3")
    if align_issues:
        filter_reasons.append("suspicious_alignment")
    if bgm_issues:
        filter_reasons.append("bgm_suspect")

    if filter_reasons:
        out_path = filtered_dir / tg_path.name
        stale = output_dir / tg_path.name
        report["status"] = "filtered_" + "_".join(filter_reasons)
        report["filter_reasons"] = filter_reasons
        if align_issues:
            report["alignment_issues"] = align_issues
    else:
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
    parser.add_argument("--filter-short-phone-sec", type=float, default=0.015)
    parser.add_argument("--filter-long-consonant-sec", type=float, default=999.0,
                        help="Max consonant phone duration (default: disabled).")
    parser.add_argument("--filter-long-vowel-sec", type=float, default=999.0,
                        help="Max vowel phone duration (default: disabled).")
    parser.add_argument("--filter-min-word-sec", type=float, default=0.15)
    parser.add_argument("--filter-min-phone-coverage", type=float, default=0.35)
    parser.add_argument("--filter-edge-gap-sec", type=float, default=0.25)
    parser.add_argument("--copy-errors", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.filtered_dir.mkdir(parents=True, exist_ok=True)

    # Load dictionaries and build IPA→pinyin mapping
    print("Loading dictionaries...")
    pinyin_dict = load_dict(args.pinyin_dict)
    ipa_dict = load_dict(args.ipa_dict)
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
                                       ipa_to_pinyin, pinyin_dict))
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
