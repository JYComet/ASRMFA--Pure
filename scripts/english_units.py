"""Canonical English lexical units shared by the alignment pipeline.

This module deliberately contains no pipeline-side mutation or fallback
matching.  A reference spelling is the authority for the surface form and a
CTC fragment group is accepted only when it is an exact, ordered, contiguous
match for that spelling.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

try:  # Consumers commonly put ``scripts`` directly on sys.path.
    from pipeline_utils import NVV_NAMES, is_pinyin_syllable
except ImportError:  # pragma: no cover - package-style imports
    from scripts.pipeline_utils import NVV_NAMES, is_pinyin_syllable


# The compound grammar is intentionally narrow.  In particular, apostrophes,
# Unicode dashes, and underscores are not English-unit syntax here.  An
# alpha+digit token is one authority unit; finite-inventory pinyin remains a
# hard exclusion (for example, ``jin1`` is not English).
ENGLISH_COMPOUND_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")
ENGLISH_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)+|[A-Za-z]+[0-9]+|[A-Za-z]+")
LEXICAL_COMPOUND_RE = ENGLISH_COMPOUND_RE

MERGE_KIND_DIRECT = "direct"
MERGE_KIND_COMPOUND = "compound"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CANONICAL_UNITS_SCHEMA = "canonical-english-units-v1"
_ORDINAL_KEYS = ("ordinal", "ctc_ordinal", "source_ctc_ordinal")
_TEXT_KEYS = ("text", "surface_text", "word", "token")
_START_KEYS = ("start", "xmin", "start_s")
_END_KEYS = ("end", "xmax", "end_s")


class EnglishUnitError(ValueError):
    """Raised when an English unit or authority merge is not valid.

    ``code`` is stable enough for consumers to route a rejected item without
    parsing human-readable error text.
    """

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def _is_nvv(text: str) -> bool:
    """Check NVV before lexical classification (NVV has precedence)."""
    candidate = text.strip().strip("<>").upper()
    return candidate in NVV_NAMES


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _canonical_token(text: str) -> str:
    if not isinstance(text, str) or not text:
        raise EnglishUnitError("invalid_surface_text", "surface text must be non-empty str")
    if _is_nvv(text):
        raise EnglishUnitError("nvv_is_not_english", text)
    if not ENGLISH_TOKEN_RE.fullmatch(text):
        raise EnglishUnitError("invalid_english_syntax", text)
    if is_pinyin_syllable(text):
        raise EnglishUnitError("pinyin_is_not_english", text)
    # NVV precedence must also apply to a bare hyphenated NVV label such as
    # QUESTION-YI, which otherwise satisfies the compound grammar.
    if _is_nvv(text):
        raise EnglishUnitError("nvv_is_not_english", text)
    token = text.replace("-", "").lower()
    # Dictionary lookup uses the alphabetic base while surface/unit identity
    # retains the numeric suffix (target1 and target2 remain distinct units).
    return re.sub(r"[0-9]+$", "", token)


def canonicalize_english_token(text: str) -> str:
    """Return the lowercase, hyphenless alignment token for *text*.

    This is strict: it never strips arbitrary punctuation or guesses through
    a malformed token.
    """
    return _canonical_token(text)


def is_english_fragment_token(text: str) -> bool:
    """Return whether *text* can participate in an authority CTC group.

    Digits-only text is admitted only as a candidate final suffix; the merge
    validator enforces that position and exact authority spelling.  This
    keeps ``target`` + ``1`` visible to the producer without classifying a
    standalone numeric token as a complete English unit.  Finite-inventory
    pinyin (``jin1``/``rui4``) is excluded here as well as in the authority
    parser.
    """
    if not isinstance(text, str) or not text or _is_nvv(text) or _is_cjk(text):
        return False
    if re.fullmatch(r"[0-9]+", text):
        return True
    try:
        canonicalize_english_token(text)
    except EnglishUnitError:
        return False
    return True


def _unit_id(reference_ordinal: int) -> str:
    return f"en-u{reference_ordinal:04d}"


@dataclass(frozen=True, slots=True)
class EnglishUnit:
    """Immutable canonical English unit.

    ``canonical_start``/``canonical_end`` are character offsets for parsed
    reference units and source timing values for merged CTC units.  They are
    kept as one pair so downstream code can carry a unit without changing its
    identity; ``canonical_span`` exposes the pair directly.
    """

    surface_text: str
    alignment_token: str
    match_key: str
    unit_id: str
    reference_ordinal: int
    source_ctc_ordinals: tuple[int, ...] = ()
    merge_kind: str = MERGE_KIND_DIRECT
    canonical_start: int | float | None = None
    canonical_end: int | float | None = None

    @property
    def lexical_ordinal(self) -> int:
        """Stable reference lexical ordinal (an explicit API alias)."""
        return self.reference_ordinal

    @property
    def start(self) -> int | float | None:
        return self.canonical_start

    @property
    def end(self) -> int | float | None:
        return self.canonical_end

    @property
    def canonical_span(self) -> tuple[int | float | None, int | float | None]:
        return self.canonical_start, self.canonical_end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        # The producer-side sidecar is an authority artifact.  Persist the
        # binding in the unit itself so downstream stages can distinguish it
        # from a display-only word reconstructed from a TextGrid.
        data["canonical_binding"] = CANONICAL_UNITS_SCHEMA
        data["source_ctc_ordinals"] = list(self.source_ctc_ordinals)
        data["canonical_span"] = [self.canonical_start, self.canonical_end]
        return data

    as_dict = to_dict

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _make_unit(
    surface_text: str,
    reference_ordinal: int,
    *,
    source_ctc_ordinals: Iterable[int] = (),
    canonical_start: int | float | None = None,
    canonical_end: int | float | None = None,
) -> EnglishUnit:
    if not isinstance(reference_ordinal, int) or isinstance(reference_ordinal, bool):
        raise EnglishUnitError("invalid_reference_ordinal")
    if reference_ordinal < 0:
        raise EnglishUnitError("invalid_reference_ordinal")
    alignment_token = _canonical_token(surface_text)
    ordinals = tuple(source_ctc_ordinals)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in ordinals):
        raise EnglishUnitError("invalid_source_ctc_ordinal")
    if tuple(sorted(set(ordinals))) != ordinals:
        raise EnglishUnitError("source_ctc_ordinals_not_monotonic")
    if (canonical_start is None) != (canonical_end is None):
        raise EnglishUnitError("incomplete_canonical_span")
    if canonical_start is not None and canonical_end is not None:
        if not isinstance(canonical_start, (int, float)) or isinstance(canonical_start, bool):
            raise EnglishUnitError("invalid_canonical_span")
        if not isinstance(canonical_end, (int, float)) or isinstance(canonical_end, bool):
            raise EnglishUnitError("invalid_canonical_span")
        if canonical_end < canonical_start:
            raise EnglishUnitError("invalid_canonical_span")
    kind = MERGE_KIND_COMPOUND if "-" in surface_text else MERGE_KIND_DIRECT
    return EnglishUnit(
        surface_text=surface_text,
        alignment_token=alignment_token,
        match_key=alignment_token,
        unit_id=_unit_id(reference_ordinal),
        reference_ordinal=reference_ordinal,
        source_ctc_ordinals=ordinals,
        merge_kind=kind,
        canonical_start=canonical_start,
        canonical_end=canonical_end,
    )


def _inside_label(text: str, start: int, end: int) -> bool:
    """Return whether a regex match is inside an NVV-style label."""
    before = text[:start].rfind("<")
    close = text[:start].rfind(">")
    if before > close and text.find(">", end) >= 0:
        return True
    before = text[:start].rfind("[")
    close = text[:start].rfind("]")
    return before > close and text.find("]", end) >= 0


def parse_english_units(text: str) -> tuple[EnglishUnit, ...]:
    """Parse lexical English units from a reference string.

    Only ASCII alpha words and ASCII-hyphen compounds are emitted.  NVV
    labels are skipped before lexical classification, including labels such as
    ``<QUESTION-YI>`` whose inner spelling resembles a compound.
    Character spans are half-open offsets into the original reference.
    """
    if not isinstance(text, str):
        raise EnglishUnitError("invalid_reference_text")
    units: list[EnglishUnit] = []
    for match in ENGLISH_TOKEN_RE.finditer(text):
        if _inside_label(text, match.start(), match.end()):
            continue
        surface = match.group(0)
        if _is_nvv(surface):
            continue
        if is_pinyin_syllable(surface):
            continue
        # A token containing CJK cannot be produced by this regex, but the
        # explicit check documents and protects the crossing boundary.
        if _is_cjk(surface):
            continue
        units.append(_make_unit(surface, len(units),
                                canonical_start=match.start(),
                                canonical_end=match.end()))
    return tuple(units)


def project_authority_semantics(text: str) -> tuple[dict[str, Any], ...]:
    """Project an authority string into one exact, ordered semantic stream.

    This is the shared projection used by postprocess and the disk auditors.
    English entries retain their surface and distinct ``unit_id`` while their
    dictionary-facing ``alignment_token`` may be shared (``target1`` and
    ``target2`` both use ``target``).  Silence labels are non-semantic;
    pinyin, including ``jin1`` and ``rui4``, is ``other`` rather than English.
    """
    if not isinstance(text, str):
        raise EnglishUnitError("invalid_reference_text")
    result: list[dict[str, Any]] = []
    english_ordinal = 0
    index = 0
    punctuation = {",": "，", ".": "。", "?": "？", "!": "！",
                   ";": "；", ":": "："}
    while index < len(text):
        silence = re.match(r"<sp[0-3]>", text[index:], re.I)
        if silence:
            index += silence.end()
            continue
        if text[index].isspace():
            index += 1
            continue
        if text[index] == "<":
            close = text.find(">", index + 1)
            if close >= 0:
                surface = text[index:close + 1]
                label = surface[1:-1].upper()
                kind = "nvv" if _is_nvv(label) else "other"
                result.append({"kind": kind, "surface": label if kind == "nvv" else surface,
                               "alignment_token": None, "unit_id": None,
                               "reference_ordinal": None})
                index = close + 1
                continue
        if _CJK_RE.fullmatch(text[index]):
            result.append({"kind": "cjk", "surface": text[index],
                           "alignment_token": None, "unit_id": None,
                           "reference_ordinal": None})
            index += 1
            continue
        match = re.match(r"[A-Za-z][A-Za-z0-9-]*", text[index:])
        if match:
            surface = match.group(0)
            if is_pinyin_syllable(surface):
                kind, alignment, unit_id, ordinal = "other", None, None, None
            elif _is_nvv(surface):
                kind, alignment, unit_id, ordinal = "nvv", None, None, None
            else:
                try:
                    alignment = _canonical_token(surface)
                except EnglishUnitError:
                    kind, alignment, unit_id, ordinal = "other", None, None, None
                else:
                    kind, ordinal = "english", english_ordinal
                    unit_id = _unit_id(english_ordinal)
                    english_ordinal += 1
            result.append({"kind": kind, "surface": surface if kind != "nvv" else surface.upper(),
                           "alignment_token": alignment, "unit_id": unit_id,
                           "reference_ordinal": ordinal})
            index += match.end()
            continue
        char = text[index]
        if char in punctuation or char in "，。！？；：、…～~":
            result.append({"kind": "punct", "surface": punctuation.get(char, char),
                           "alignment_token": None, "unit_id": None,
                           "reference_ordinal": None})
        # Curly braces are placeholder syntax used by authority references
        # (for example ``{target一}``), not semantic lexical units.  Ignore
        # them just like the legacy square-bracket wrappers so reference and
        # derived hanzi streams compare on the same target/一/二 sequence.
        elif char not in "[]{}":
            result.append({"kind": "other", "surface": char,
                           "alignment_token": None, "unit_id": None,
                           "reference_ordinal": None})
        index += 1
    return tuple(result)


def _field(item: Any, keys: Sequence[str], label: str) -> Any:
    if isinstance(item, Mapping):
        for key in keys:
            if key in item:
                return item[key]
    else:
        for key in keys:
            if hasattr(item, key):
                return getattr(item, key)
    raise EnglishUnitError(f"missing_{label}")


def _fragment(item: Any) -> tuple[str, int, int | float | None, int | float | None]:
    text = _field(item, _TEXT_KEYS, "fragment_text")
    ordinal = _field(item, _ORDINAL_KEYS, "fragment_ordinal")
    if not isinstance(text, str) or not text:
        raise EnglishUnitError("invalid_fragment_text")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise EnglishUnitError("invalid_fragment_ordinal")
    start = end = None
    for key in _START_KEYS:
        if isinstance(item, Mapping) and key in item:
            start = item[key]
            break
        if not isinstance(item, Mapping) and hasattr(item, key):
            start = getattr(item, key)
            break
    for key in _END_KEYS:
        if isinstance(item, Mapping) and key in item:
            end = item[key]
            break
        if not isinstance(item, Mapping) and hasattr(item, key):
            end = getattr(item, key)
            break
    if (start is None) != (end is None):
        raise EnglishUnitError("incomplete_fragment_span")
    if start is not None:
        if (not isinstance(start, (int, float)) or isinstance(start, bool)
                or not isinstance(end, (int, float)) or isinstance(end, bool)
                or end < start):
            raise EnglishUnitError("invalid_fragment_span")
    return text, ordinal, start, end


def _authority_unit(authority: EnglishUnit | str) -> EnglishUnit:
    if isinstance(authority, EnglishUnit):
        return authority
    if isinstance(authority, str):
        return _make_unit(authority, 0)
    raise EnglishUnitError("invalid_authority_unit")


def validate_authority_fragment_group(
    authority: EnglishUnit | str,
    fragments: Sequence[Any],
) -> EnglishUnit:
    """Validate one exact authority-mode CTC fragment group.

    The group must be non-empty, its source ordinals must be strictly
    increasing (and normally contiguous), and its concatenated ASCII spelling
    must equal the authority surface identity.  A missing CTC hyphen is
    tolerated only for a hyphenated authority unit: the lexical fragments
    still have to be ordered and exact, and the output span covers them all.
    Every other punctuation, CJK, NVV, gap, reordering, partial match, and
    extra fragment is rejected.
    """
    unit = _authority_unit(authority)
    if not fragments:
        raise EnglishUnitError("empty_fragment_group")
    parsed = [_fragment(item) for item in fragments]
    texts = [item[0] for item in parsed]
    ordinals = [item[1] for item in parsed]
    if any(left >= right for left, right in zip(ordinals, ordinals[1:])):
        raise EnglishUnitError("noncontiguous_source_ctc_ordinals")
    contiguous = ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))
    if (not contiguous and ("-" not in unit.surface_text
                            or len(texts) < 2
                            or any(not re.fullmatch(r"[A-Za-z]+(?:[0-9]+)?", text)
                                   for text in texts))):
        raise EnglishUnitError("noncontiguous_source_ctc_ordinals")
    for index, text in enumerate(texts):
        if _is_nvv(text):
            raise EnglishUnitError("nvv_crossing")
        if _is_cjk(text):
            raise EnglishUnitError("cjk_crossing")
        if len(texts) == 1:
            if not ENGLISH_TOKEN_RE.fullmatch(text) or is_pinyin_syllable(text):
                raise EnglishUnitError("punctuation_fragment")
        elif re.fullmatch(r"[0-9]+", text):
            if index != len(texts) - 1:
                raise EnglishUnitError("numeric_suffix_not_final")
        elif text == "-" and "-" in unit.surface_text:
            # NVASR may emit the visible hyphen between the fragments of a
            # reference compound (for example ``V`` + ``-`` + ``tuber``).
            # It is part of the authority surface, not a language boundary,
            # so retain its source ordinal while validating the lexical
            # fragments around it.
            continue
        elif not re.fullmatch(r"[A-Za-z]+(?:[0-9]+)?", text):
            if any(char in ",.!?;:\u3001\u3002，。！？；：" for char in text):
                raise EnglishUnitError("punctuation_fragment")
            raise EnglishUnitError("invalid_fragment_syntax")
        elif is_pinyin_syllable(text):
            raise EnglishUnitError("pinyin_fragment")

    # Compare surface identity, not the dictionary token: target1 and target2
    # both align to ``target`` but must never merge into one another.
    compact = "".join(text.replace("-", "").lower() for text in texts)
    authority_surface = unit.surface_text.replace("-", "").lower()
    if compact != authority_surface:
        if len(compact) < len(authority_surface):
            raise EnglishUnitError("partial_fragment_match")
        if len(compact) > len(authority_surface):
            raise EnglishUnitError("extra_fragment")
        raise EnglishUnitError("fragment_match_mismatch")

    starts = [item[2] for item in parsed]
    ends = [item[3] for item in parsed]
    if any(value is None for value in starts + ends):
        # No timing was supplied: retain the canonical reference span rather
        # than erasing it while attaching source ordinals.
        start, end = unit.canonical_start, unit.canonical_end
    else:
        start, end = starts[0], ends[-1]
        if any(left > right for left, right in zip(ends, starts[1:])):
            raise EnglishUnitError("fragment_span_reordered")
    return replace(unit, source_ctc_ordinals=tuple(ordinals),
                   canonical_start=start, canonical_end=end)


def merge_authority_fragment_group(
    authority: EnglishUnit | str,
    fragments: Sequence[Any],
) -> EnglishUnit:
    """Validate and return the canonical unit for one fragment group."""
    return validate_authority_fragment_group(authority, fragments)


# Short alias used by workers that describe the operation as a merge.
merge_authority_fragments = merge_authority_fragment_group


def _strict_source_ordinals(value: Any, *, label: str) -> tuple[int, ...]:
    """Validate the immutable ordinal shape used by strict English evidence."""
    if not isinstance(value, (list, tuple)) or not value:
        raise EnglishUnitError(f"{label}_invalid")
    if any(type(item) is not int or item < 0 for item in value):
        raise EnglishUnitError(f"{label}_invalid")
    ordinals = tuple(value)
    if any(left >= right for left, right in zip(ordinals, ordinals[1:])):
        raise EnglishUnitError(f"{label}_not_increasing")
    return ordinals


def resolve_processed_english_token(
        tokens: Sequence[Any] | None,
        source_ctc_ordinals: Sequence[int],
) -> Mapping[str, Any] | None:
    """Resolve exactly one processed CTC token by its source ordinal tuple.

    This deliberately does not fall back to text, timing, or ordinal-nearby
    matching.  A non-contiguous strict-English span is safe only when its
    immutable source tuple identifies one and only one processed token.
    """
    requested = _strict_source_ordinals(source_ctc_ordinals,
                                         label="source_ctc_ordinals")
    matches: list[Mapping[str, Any]] = []
    for token in tokens or ():
        if not isinstance(token, Mapping) or token.get("type", "word") != "word":
            continue
        values = token.get("source_ctc_ordinals")
        if isinstance(values, (list, tuple)) and tuple(values) == requested:
            matches.append(token)
    if len(matches) != 1:
        return None
    return matches[0]


def validate_processed_english_token_binding(
        record: Mapping[str, Any],
        token: Mapping[str, Any] | None,
) -> None:
    """Validate a strict-English ledger record against processed CTC evidence.

    Contiguous source spans remain valid without a processed-token marker for
    compatibility with existing v10 ledgers.  A non-contiguous span is
    accepted only when the exact token is independently resolved and carries
    the producer's explicit ``hyphen_separator_omitted is True`` marker.
    """
    if not isinstance(record, Mapping):
        raise EnglishUnitError("strict_english_record_invalid")
    ordinals = _strict_source_ordinals(
        record.get("source_ctc_ordinals"), label="source_ctc_ordinals")
    ctc_ordinal = record.get("ctc_ordinal")
    if type(ctc_ordinal) is not int or ctc_ordinal < 0 or ctc_ordinal != ordinals[0]:
        raise EnglishUnitError("strict_english_ctc_ordinal_binding_invalid")
    contiguous = all(right - left == 1
                     for left, right in zip(ordinals, ordinals[1:]))
    if contiguous and token is None:
        return
    if token is None:
        raise EnglishUnitError("strict_english_processed_token_missing")
    if token.get("type", "word") != "word":
        raise EnglishUnitError("strict_english_processed_token_invalid")

    token_ordinals = _strict_source_ordinals(
        token.get("source_ctc_ordinals"), label="processed_source_ctc_ordinals")
    if token_ordinals != ordinals:
        raise EnglishUnitError("strict_english_source_ordinals_mismatch")
    for key in ("source_ctc_ordinal", "ordinal"):
        if key in token:
            value = token[key]
            if type(value) is not int or value < 0 or value != ordinals[0]:
                raise EnglishUnitError("strict_english_ctc_ordinal_binding_invalid")

    canonical = token.get("canonical_unit")
    if not isinstance(canonical, Mapping):
        raise EnglishUnitError("strict_english_canonical_unit_missing")
    if canonical.get("canonical_binding") != CANONICAL_UNITS_SCHEMA:
        raise EnglishUnitError("strict_english_canonical_binding_invalid")
    surface = canonical.get("surface_text")
    try:
        parsed = parse_english_units(surface)
    except (EnglishUnitError, TypeError, ValueError) as exc:
        raise EnglishUnitError("strict_english_canonical_unit_invalid") from exc
    if len(parsed) != 1:
        raise EnglishUnitError("strict_english_canonical_unit_invalid")
    unit = parsed[0]
    reference_ordinal = canonical.get("reference_ordinal")
    if (type(reference_ordinal) is not int or reference_ordinal < 0
            or canonical.get("unit_id") != _unit_id(reference_ordinal)
            or canonical.get("alignment_token") != unit.alignment_token
            or canonical.get("match_key") != unit.match_key
            or canonical.get("merge_kind") != unit.merge_kind
            or canonical.get("source_ctc_ordinals") != list(ordinals)
            or canonical.get("canonical_span") != [
                canonical.get("canonical_start"), canonical.get("canonical_end")]
            or token.get("surface_text") != surface
            or token.get("canonical_span") != canonical.get("canonical_span")
            or token.get("word") != unit.alignment_token
            or ("alignment_token" in token
                and token.get("alignment_token") != canonical.get("alignment_token"))
            or ("unit_id" in token and token.get("unit_id") != canonical.get("unit_id"))
            or ("reference_ordinal" in token
                and token.get("reference_ordinal") != reference_ordinal)):
        raise EnglishUnitError("strict_english_canonical_unit_mismatch")
    if "canonical_unit_sha256" in token:
        encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if token.get("canonical_unit_sha256") != hashlib.sha256(encoded).hexdigest():
            raise EnglishUnitError("strict_english_canonical_unit_hash_mismatch")
    try:
        record_token = canonicalize_english_token(str(record.get("ctc_text", "")))
        processed_token = canonicalize_english_token(str(token.get("word", "")))
    except (EnglishUnitError, TypeError, ValueError) as exc:
        raise EnglishUnitError("strict_english_ledger_ctc_mismatch") from exc
    if (record.get("unit_id") != canonical.get("unit_id")
            or record.get("alignment_token") != canonical.get("alignment_token")
            or record_token != processed_token
            or processed_token != canonical.get("alignment_token")
            or record.get("canonical_span") != token.get("canonical_span")):
        raise EnglishUnitError("strict_english_ledger_ctc_mismatch")

    if not contiguous:
        if token.get("hyphen_separator_omitted") is not True:
            raise EnglishUnitError("strict_english_hyphen_marker_missing")
        if "-" not in unit.surface_text:
            raise EnglishUnitError("strict_english_nonhyphen_gap")
        deltas = [right - left for left, right in zip(ordinals, ordinals[1:])]
        if any(delta not in (1, 2) for delta in deltas):
            raise EnglishUnitError("strict_english_source_ordinal_gap_invalid")
        omitted = sum(delta - 1 for delta in deltas)
        if omitted != unit.surface_text.count("-"):
            raise EnglishUnitError("strict_english_hyphen_omission_count_invalid")


def merge_authority_units(
    authority_units: Sequence[EnglishUnit],
    fragment_groups: Sequence[Sequence[Any]],
) -> tuple[EnglishUnit, ...]:
    """Validate an ordered set of authority units and fragment groups."""
    if len(authority_units) != len(fragment_groups):
        raise EnglishUnitError("authority_group_count_mismatch")
    result: list[EnglishUnit] = []
    previous_ordinal = -1
    for unit, group in zip(authority_units, fragment_groups):
        if not isinstance(unit, EnglishUnit):
            raise EnglishUnitError("invalid_authority_unit")
        if unit.reference_ordinal <= previous_ordinal:
            raise EnglishUnitError("authority_ordinals_not_monotonic")
        merged = validate_authority_fragment_group(unit, group)
        if merged.source_ctc_ordinals and merged.source_ctc_ordinals[0] <= (
                result[-1].source_ctc_ordinals[-1] if result and result[-1].source_ctc_ordinals else -1):
            raise EnglishUnitError("source_ctc_ordinals_not_monotonic")
        result.append(merged)
        previous_ordinal = unit.reference_ordinal
    return tuple(result)


def serialize_english_units(units: Iterable[EnglishUnit]) -> list[dict[str, Any]]:
    """Serialize an iterable of units without exposing mutable internals."""
    return [unit.to_dict() for unit in units]


__all__ = [
    "ENGLISH_COMPOUND_RE", "ENGLISH_TOKEN_RE", "LEXICAL_COMPOUND_RE",
    "MERGE_KIND_DIRECT", "MERGE_KIND_COMPOUND", "EnglishUnit",
    "EnglishUnitError", "canonicalize_english_token", "is_english_fragment_token",
    "parse_english_units",
    "project_authority_semantics",
    "validate_authority_fragment_group", "merge_authority_fragment_group",
    "merge_authority_fragments", "merge_authority_units",
    "resolve_processed_english_token", "validate_processed_english_token_binding",
    "serialize_english_units",
]
