"""Mandatory Qwen lexical pause normalization before the existing MFA stages.

Coordinates use the TextGrid microsecond precision.  No waveform, model or MFA
dependency is needed; subsequent acoustic boundary refinement remains enabled.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

SCHEMA = "qwen3-timestamp-normalization-v2"
ELLIPSIS = "…"
_QWEN_MARKUP_RE = re.compile(
    r"<(?:sp[0-3]|[A-Za-z][A-Za-z0-9_-]*)>|\[[A-Za-z][A-Za-z0-9 _-]*\]",
    re.IGNORECASE,
)
_QWEN_PUNCTUATION_MAP = {
    ",": "，", ".": "。", "!": "！", "?": "？",
    "~": ELLIPSIS, "～": ELLIPSIS,
}
_QWEN_DECORATIVE_CHARS = frozenset(
    "\"“”‘’`´「」『』《》〈〉（）()【】[]〔〕［］{}:;：；"
)
_REFERENCE_TAG_RE = re.compile(r"<[^>]*>")
_REFERENCE_EVENT_RE = re.compile(r"\[[A-Za-z][^\]]*\]")
_REFERENCE_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_REFERENCE_CONDITIONAL_RUN_RE = re.compile(
    r"(?:\{[FM]#[^{}]*\}\s*)+", re.IGNORECASE)
_REFERENCE_CONDITIONAL_ITEM_RE = re.compile(
    r"\{([FM])#([^{}]*)\}", re.IGNORECASE)
_REFERENCE_PUNCTUATION = frozenset("，。！？…、")
_CN_DIGITS = "零一二三四五六七八九"
_MAX_QUANTIZED_OVERSHOOT_TICKS = 80_000


def normalize_qwen_input_text(text: str) -> str:
    """Normalize Qwen surface punctuation without hiding lexical evidence.

    Known angle/square markup is copied as an opaque token so existing NVV or
    unsupported-label validation still sees it. Decorative delimiters are
    removed while their enclosed content, whitespace, and semantic operators
    remain visible to the existing lexical contract.
    """
    if not isinstance(text, str):
        raise TypeError("Qwen input text must be a string")
    output: list[str] = []
    index = 0
    while index < len(text):
        match = _QWEN_MARKUP_RE.match(text, index)
        if match is not None:
            output.append(match.group(0))
            index = match.end()
            continue
        char = text[index]
        if char == "\\" and index + 1 < len(text) and text[index + 1] == "~":
            output.append(ELLIPSIS)
            index += 2
            continue
        if char in _QWEN_PUNCTUATION_MAP:
            output.append(_QWEN_PUNCTUATION_MAP[char])
        elif char in "'’":
            # Apostrophes are meaningful inside English words, but quote
            # delimiters and CJK interpuncts are decorative input noise.
            left, right = (text[index - 1], text[index + 1]) if (
                index > 0 and index + 1 < len(text)
            ) else ("", "")
            if (left.isascii() and left.isalpha()
                    and right.isascii() and right.isalpha()):
                output.append(char)
        elif char in _QWEN_DECORATIVE_CHARS:
            pass
        else:
            output.append(char)
        index += 1
    return re.sub(r"…{2,}", ELLIPSIS, "".join(output))


def _reference_integer(digits: str) -> str:
    if len(digits) > 1 and digits.startswith("0"):
        return "".join(_CN_DIGITS[int(char)] for char in digits)
    number = int(digits)
    if number == 0:
        return "零"
    if number >= 10 ** 16:
        return "".join(_CN_DIGITS[int(char)] for char in digits)

    def four(value: int) -> str:
        units = ("", "十", "百", "千")
        result = []
        pending_zero = False
        for power in range(3, -1, -1):
            divisor = 10 ** power
            digit, value = divmod(value, divisor)
            if digit:
                if pending_zero and result:
                    result.append("零")
                if not (power == 1 and digit == 1 and not result):
                    result.append(_CN_DIGITS[digit])
                result.append(units[power])
                pending_zero = False
            elif result and value:
                pending_zero = True
        return "".join(result)

    groups = []
    value = number
    while value:
        groups.append(value % 10_000)
        value //= 10_000
    group_units = ("", "万", "亿", "兆")
    result = []
    pending_zero = False
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if not group:
            if result:
                pending_zero = True
            continue
        if result and (pending_zero or group < 1000):
            result.append("零")
        result.append(four(group))
        result.append(group_units[index])
        pending_zero = False
    return "".join(result)


def normalize_qwen_reference_text(text: str, *, speaker: str | None = None) -> str:
    """Normalize authority text to Qwen/MFA-supported spoken units."""
    if not isinstance(text, str):
        raise TypeError("Qwen reference text must be a string")
    # NFKC expands U+2026 to three ASCII periods; protect the canonical
    # ellipsis while still folding full-width digits/letters elsewhere.
    ellipsis_sentinel = "\ue000"
    text = text.replace(ELLIPSIS, ellipsis_sentinel)
    text = unicodedata.normalize("NFKC", text).replace(ellipsis_sentinel, ELLIPSIS)
    text = text.replace("\\n", " ")

    def conditional_run(match: re.Match) -> str:
        variants = _REFERENCE_CONDITIONAL_ITEM_RE.findall(match.group(0))
        if not variants:
            return match.group(0)
        folded_speaker = (speaker or "").casefold()
        preferred = "M" if "wise" in folded_speaker else (
            "F" if "belle" in folded_speaker else variants[0][0].upper())
        return next((body for label, body in variants
                     if label.upper() == preferred), variants[0][1])

    text = _REFERENCE_CONDITIONAL_RUN_RE.sub(conditional_run, text)
    text = _REFERENCE_TAG_RE.sub("", text)
    text = _REFERENCE_EVENT_RE.sub("", text)

    def number(match: re.Match) -> str:
        value = match.group(0)
        left = text[match.start() - 1] if match.start() else ""
        right = text[match.end()] if match.end() < len(text) else ""
        if "." in value:
            integer, fraction = value.split(".", 1)
            return _reference_integer(integer) + "点" + "".join(
                _CN_DIGITS[int(char)] for char in fraction)
        if ((left.isascii() and left.isalpha())
                or (right.isascii() and right.isalpha())
                or (len(value) == 4 and right == "年")):
            return "".join(_CN_DIGITS[int(char)] for char in value)
        return _reference_integer(value)

    text = _REFERENCE_NUMBER_RE.sub(number, text)
    text = normalize_qwen_input_text(text)
    output = []
    for char in text:
        codepoint = ord(char)
        is_cjk = (0x3400 <= codepoint <= 0x4DBF
                  or 0x4E00 <= codepoint <= 0x9FFF
                  or 0xF900 <= codepoint <= 0xFAFF)
        if is_cjk or (char.isascii() and char.isalpha()) or char in _REFERENCE_PUNCTUATION:
            output.append(char)
        elif char.isspace():
            output.append(" ")
        elif unicodedata.category(char)[0] in {"P", "S", "C"}:
            continue
        else:
            raise ValueError(f"unsupported Qwen/MFA reference character: {char!r}")
    normalized = re.sub(r"\s+", " ", "".join(output)).strip()
    normalized = re.sub(r"…{2,}", ELLIPSIS, normalized)
    has_lexical = any(
        (0x3400 <= ord(char) <= 0x4DBF)
        or (0x4E00 <= ord(char) <= 0x9FFF)
        or (0xF900 <= ord(char) <= 0xFAFF)
        or (char.isascii() and char.isalpha()) for char in normalized)
    if not normalized or not has_lexical:
        raise ValueError("normalized Qwen reference has no spoken lexical units")
    return normalized


def normalized_bundle_text(rows, text):
    """Read the mandatory normalized authority, rejecting mixed/stale evidence."""
    marked = [row for row in rows if row.get("timestamp_normalization")]
    if not marked:
        return None  # historical/imported bundles retain their existing policy
    digest = hashlib.sha256(text.strip().encode()).hexdigest()
    if len(marked) != len(rows) or any(
            row.get("provider") != "qwen3_hf"
            or row["timestamp_normalization"].get("schema") != SCHEMA
            or row.get("normalized_text_sha256") != digest for row in rows):
        raise ValueError("Qwen normalized transcript evidence mismatch")
    return text.strip()


def read_normalized_transcript(root, stem):
    """Independent disk readers use the same receipt-bound text contract."""
    tokens = Path(root) / f"{stem}_tokens.jsonl"
    if not tokens.exists():
        return None
    if tokens.is_symlink():
        raise ValueError("Qwen token evidence must be a regular file")
    rows = [json.loads(line) for line in tokens.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(row.get("timestamp_normalization") for row in rows):
        return None
    path = Path(root) / f"{stem}_text_cn.txt"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Qwen normalized transcript is missing")
    return normalized_bundle_text(rows, path.read_text(encoding="utf-8"))


def _ticks(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite Qwen timestamp")
    return round(number * 1_000_000)


def normalize_timestamps(rows: list[dict], text: str, duration: float):
    """Return copied lexical rows, repaired text and timed pause punctuation.

    Leading gaps stay boundary silence at every duration. Other sp0 gaps
    extend the preceding word. Longer gaps use existing punctuation, adding
    an ellipsis only when none exists, never spoken MFA dictionary entries.
    Raw model coordinates and pre-normalization coordinates remain available.
    """
    end_tick = _ticks(duration)
    if end_tick <= 0 or not rows:
        raise ValueError("empty Qwen timeline")
    fixed = copy.deepcopy(rows)
    cursor = 0
    gaps = []
    for index, row in enumerate(fixed):
        prior = row.get("timestamp_normalization")
        if prior is not None:
            if prior.get("schema") != SCHEMA:
                raise ValueError("unknown Qwen timestamp normalization version")
            row["start_s"] = prior["original_start_s"]
            row["end_s"] = prior["original_end_s"]
        start, original_end = _ticks(row["start_s"]), _ticks(row["end_s"])
        end = original_end
        if (start < end_tick < original_end
                and original_end - end_tick <= _MAX_QUANTIZED_OVERSHOOT_TICKS):
            end = end_tick
        if not 0 <= cursor <= start < end <= end_tick:
            raise ValueError("invalid or overlapping Qwen timestamp interval")
        row["timestamp_normalization"] = {
            "schema": SCHEMA, "original_start_s": start / 1e6,
            "original_end_s": original_end / 1e6,
        }
        row["start_s"], row["end_s"] = start / 1e6, end / 1e6
        if start > cursor:
            gaps.append((index, cursor, start))
        cursor = end
    if cursor < end_tick:
        gaps.append((len(fixed), cursor, end_tick))

    # Locate lexical boundaries before assigning pauses, so original punctuation
    # wins over synthetic ellipses. Explicit sp tags are timing annotations;
    # they must neither overwrite punctuation nor manufacture a leading pause.
    source = normalize_qwen_input_text(re.sub(r"<sp[0-3]>", "", text, flags=re.I))
    spans = []
    cursor = 0
    for row in fixed:
        surface = row.get("surface_text", row["unit"])
        position = source.find(surface, cursor)
        if position < 0:
            raise ValueError("Qwen timestamp/text lexical mismatch")
        spans.append((position, position + len(surface)))
        cursor = position + len(surface)
    boundaries = [source[:spans[0][0]]]
    boundaries.extend(source[left[1]:right[0]] for left, right in zip(spans, spans[1:]))
    boundaries.append(source[spans[-1][1]:])

    pauses = []
    for boundary, start, end in gaps:
        if boundary == 0:
            continue
        length = end - start
        if length < 200_000:
            fixed[boundary - 1]["end_s"] = end / 1e6
            continue
        punctuation = "".join(char for char in boundaries[boundary]
                              if unicodedata.category(char).startswith("P"))
        if not punctuation:
            punctuation = ELLIPSIS
            boundaries[boundary] = ELLIPSIS
        label = "sp1" if length < 500_000 else "sp2" if length < 1_500_000 else "sp3"
        pauses.append({
            "schema": "ctc-punctuation-evidence-v2", "word": punctuation,
            "start_s": start / 1e6, "end_s": end / 1e6,
            "start_ms": start / 1000, "end_ms": end / 1000,
            "raw_start_s": start / 1e6, "raw_end_s": end / 1e6,
            "candidate_id": f"qwen-pause-{boundary:04d}",
            # `source` is the legacy handoff namespace, not a CTC model claim.
            "source": "ctc", "provider": "qwen3_hf",
            "lexical_timing_source": "qwen3_forced_aligner_hf",
            "normalization_schema": SCHEMA, "pause_label": label,
            "left_lexical_ordinal": boundary - 1,
            "right_lexical_ordinal": boundary if boundary < len(fixed) else None,
        })

    pieces = []
    for boundary, (start, end) in enumerate(spans):
        pieces.extend((boundaries[boundary], source[start:end]))
    pieces.append(boundaries[-1])
    repaired = "".join(pieces).strip()

    for row in fixed:
        # Keep the canonical English geometry in the same coordinate system as
        # its lexical token; raw_start/end retain the model's evidence.
        if "canonical_unit" in row:
            unit = row["canonical_unit"]
            unit["canonical_start"], unit["canonical_end"] = row["start_s"], row["end_s"]
            unit["canonical_span"] = [row["start_s"], row["end_s"]]
            row["canonical_span"] = [row["start_s"], row["end_s"]]
            row["canonical_unit_sha256"] = hashlib.sha256(json.dumps(
                unit, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
        row["normalized_text_sha256"] = hashlib.sha256(repaired.encode()).hexdigest()
        if "reference_identity" in row:
            row.setdefault("original_reference_identity", row["reference_identity"])
            row["reference_identity"] = row["normalized_text_sha256"]
    return fixed, repaired, pauses
