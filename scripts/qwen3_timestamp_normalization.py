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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SCHEMA = "qwen3-timestamp-normalization-v2"
MACRO_SCHEMA = "qwen3-reference-macro-resolutions-v1"
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

# GAMEDATA authority text embeds localization macros.  Braces and `#` are
# stripped by the character filter below while their ASCII letters survive, so
# an unresolved macro reaches the aligner as a bare token such as NICKNAME and
# is then G2P'd into nonsense phonemes.  `{M#…}{F#…}` was already handled; the
# families below were not.  Only single-level braces are recognised, so an
# adjacent `…}{NICKNAME}` cannot be swallowed by its neighbour.
_REFERENCE_MACRO_RE = re.compile(
    r"\{(?![FfMm]#)([A-Za-z][A-Za-z0-9_]*)([^{}]*)\}")
_REFERENCE_ALIAS_RE = re.compile(r"(?<![A-Za-z0-9])(player|TA)(?![A-Za-z0-9])")
_REFERENCE_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
_REFERENCE_PARAM_RE = re.compile(r"#\s*([A-Za-z0-9_]+)")
# `{RUBY#[S]希望}` is a ruby gloss: `杜麦{RUBY#[S]希望}尼` reads as the base
# text 杜麦尼 with 希望 typeset above it.  The whole macro is dropped so the
# base text survives intact; the gloss is never spoken.
_REFERENCE_RUBY_MACRO = "RUBY"
# Macros whose whole value is the player's chosen name.
_REFERENCE_NAME_MACROS = frozenset({"NICKNAME", "REALNAME"})
# Label suffix token -> spoken replacement.  Keyed on the pronoun token, never
# on the INFO_MALE_/INFO_FEMALE_ prefix: the corpus contains
# `[INFO_MALE_PRONOUN_SHE|INFO_FEMALE_PRONOUN_HE]`, so the prefix contradicts
# the pronoun it labels.
_SEXPRO_WORDS = {
    "HE": "他", "SHE": "她",
    "BROTHER": "哥哥", "SISTER": "姐姐", "SISTERA": "姐姐",
    "BOY": "男孩", "BOYD": "男孩",
    "GIRL": "女孩", "GIRLD": "女孩", "GIRLC": "女孩",
    "YING": "荧", "KONG": "空",
}
# 他/她/它 are indistinguishable to an acoustic model, so the branch is chosen
# by fixed policy rather than by ASR.  Any other pair is audibly distinct and
# must be resolved from evidence.
_HOMOPHONE_GROUPS = (frozenset({"他", "她", "它"}),)
_HOMOPHONE_POLICY = "他"
# A resolution that reintroduces one of these would leak exactly as the bug did.
_REFERENCE_WORD_RE = re.compile(r"[A-Za-z]+")
_REFERENCE_RESERVED_WORDS = frozenset({
    "nickname", "playeravatar", "mateavatar", "textjoin", "realname",
    "ruby", "gender", "sexpro",
})


@dataclass(frozen=True)
class ReferenceMacroSlot:
    """One localization macro occurrence and the choices it presents."""

    literal: str
    macro: str
    param: str | None
    labels: tuple[str, ...]
    alternatives: tuple[str, ...]
    start: int
    end: int

    @property
    def audibly_distinct(self) -> bool:
        return len(self.alternatives) > 1 and not _same_homophone_group(
            self.alternatives)

    @property
    def default(self) -> str | None:
        """The value implied by policy alone, without ASR evidence.

        A sole alternative is unambiguous.  A homophone set is resolved by
        policy rather than by its first member, so an ordered
        `[SHE|HE]` branch list still yields 他.
        """
        if len(self.alternatives) == 1:
            return self.alternatives[0]
        if self.alternatives and not self.audibly_distinct:
            return _HOMOPHONE_POLICY
        return None

    @property
    def key(self) -> str:
        return self.literal


def _same_homophone_group(words) -> bool:
    return any(set(words) <= group for group in _HOMOPHONE_GROUPS)


def _label_word(label: str) -> str | None:
    """Map one SEXPRO alternative label to its spoken word, if known."""
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", label.upper()) if token]
    for token in reversed(tokens):
        if token in _SEXPRO_WORDS:
            return _SEXPRO_WORDS[token]
    return None


def reference_macro_slots(text: str) -> tuple[ReferenceMacroSlot, ...]:
    """Locate localization macros; returns () for macro-free text."""
    if "{" not in text:
        return ()
    slots = []
    for match in _REFERENCE_MACRO_RE.finditer(text):
        macro, body = match.group(1), match.group(2)
        param_match = _REFERENCE_PARAM_RE.search(body)
        bracket = _REFERENCE_BRACKET_RE.search(body)
        labels: tuple[str, ...] = ()
        alternatives: tuple[str, ...] = ()
        if bracket is not None:
            labels = tuple(part.strip() for part in bracket.group(1).split("|")
                           if part.strip())
            alternatives = tuple(
                word for word in (_label_word(label) for label in labels)
                if word is not None)
        slots.append(ReferenceMacroSlot(
            literal=match.group(0), macro=macro,
            param=param_match.group(1) if param_match else None,
            labels=labels, alternatives=alternatives,
            start=match.start(), end=match.end()))
    return tuple(slots)


def _resolve_reference_macros(text: str, slot: ReferenceMacroSlot,
                              resolutions: Mapping[str, str] | None) -> str:
    """Return the spoken replacement for one macro, or raise if unresolvable."""
    if slot.macro == _REFERENCE_RUBY_MACRO:
        return ""
    if slot.macro in _REFERENCE_NAME_MACROS:
        resolved = resolutions.get(slot.key) if resolutions else None
        if resolved is None:
            raise ValueError(
                f"unresolved Qwen reference macro (no resolution): {slot.literal!r}")
        return resolved
    if slot.macro.endswith("AVATAR") or _REFERENCE_BRACKET_RE.search(slot.literal):
        if not slot.alternatives:
            raise ValueError(
                f"unrecognized Qwen reference macro labels: {slot.literal!r}")
        implied = slot.default
        if implied is not None:
            return implied
        resolved = resolutions.get(slot.key) if resolutions else None
        if resolved is None:
            raise ValueError(
                f"unresolved Qwen reference macro (ambiguous branch): {slot.literal!r}")
        if resolved not in slot.alternatives:
            raise ValueError(
                f"Qwen reference macro resolution is not one of its branches: {slot.literal!r}")
        return resolved
    raise ValueError(f"unrecognized Qwen reference macro: {slot.literal!r}")


def _reference_replacement_is_spoken(value: str) -> bool:
    """Reject a substitution that would be altered by, or re-leak through, the filter.

    A replacement must consist only of units that survive the reference filter
    verbatim, so punctuation that would be silently stripped fails closed.  It
    must also not reintroduce a macro-shaped token: an all-caps run or a known
    macro name would leak exactly as the original bug did.
    """
    if not value:
        return False
    for char in value:
        codepoint = ord(char)
        is_cjk = (0x3400 <= codepoint <= 0x4DBF
                  or 0x4E00 <= codepoint <= 0x9FFF
                  or 0xF900 <= codepoint <= 0xFAFF)
        if not (is_cjk or (char.isascii() and char.isalpha())
                or char in _REFERENCE_PUNCTUATION or char.isspace()):
            return False
    for token in _REFERENCE_WORD_RE.findall(value):
        if len(token) >= 3 and token.isupper():
            return False
        if token.casefold() in _REFERENCE_RESERVED_WORDS:
            return False
    return True


def _resolve_reference_alias(word: str, resolutions: Mapping[str, str]) -> str:
    """Resolve a bare `player`/`TA` alias; only reachable with a table."""
    resolved = resolutions.get(f"alias:{word}")
    if resolved is None:
        raise ValueError(f"unresolved Qwen reference alias: {word!r}")
    if not _reference_replacement_is_spoken(resolved):
        raise ValueError(f"unsafe Qwen reference alias resolution: {word!r}")
    return resolved


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


def normalize_qwen_reference_text(
        text: str, *, speaker: str | None = None,
        resolutions: Mapping[str, str] | None = None) -> str:
    """Normalize authority text to Qwen/MFA-supported spoken units.

    ``resolutions`` maps a macro literal (see :func:`reference_macro_slots`) to
    the spoken word chosen for it.  It is only consulted for branches that
    policy cannot decide; homophone branches are always taken to the fixed
    policy value so an acoustic model is never asked to distinguish them.
    Macro-free text is returned byte-identical to the previous behaviour.
    """
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

    # Macro resolution must precede _REFERENCE_EVENT_RE, which would otherwise
    # delete a SEXPRO branch list before it can be read, and precede number
    # verbalization, which would turn `{TEXTJOIN#54}` into `TEXTJOIN五十四`.
    slots = reference_macro_slots(text)
    if slots:
        pieces: list[str] = []
        cursor = 0
        for slot in slots:
            pieces.append(text[cursor:slot.start])
            replacement = _resolve_reference_macros(text, slot, resolutions)
            if replacement and not _reference_replacement_is_spoken(replacement):
                raise ValueError(
                    f"unsafe Qwen reference macro resolution: {slot.literal!r}")
            pieces.append(replacement)
            cursor = slot.end
        pieces.append(text[cursor:])
        text = "".join(pieces)
    if resolutions is not None:
        text = _REFERENCE_ALIAS_RE.sub(
            lambda match: _resolve_reference_alias(match.group(1), resolutions), text)

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
