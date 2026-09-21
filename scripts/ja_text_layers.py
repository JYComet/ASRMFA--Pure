"""Immutable text layers used by the Japanese frontend.

The frontend receives a canonical string, while every serialized unit keeps a
span in both the canonical and caller domains.  Unicode normalization is
explicitly lossy: these maps provide provenance, not a claim that the
normalization can be reversed.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping, Sequence

try:
    from .ja_en_schema import stable_digest, sha256_bytes
except ImportError:  # direct script execution
    from ja_en_schema import stable_digest, sha256_bytes


TEXT_LAYER_VERSION = "ja-text-layer-v1"


def _prefix_boundaries(text: str, mode: str) -> list[int]:
    return [len(unicodedata.normalize(mode, text[:index])) for index in range(len(text) + 1)]


def _offset_maps(original: str, canonical: str, mode: str) -> tuple[list[list[int]], list[list[int]]]:
    """Build monotonic half-open maps, including composing-character spans."""
    boundaries = _prefix_boundaries(original, mode)
    orig_to_canonical = [[boundaries[i], boundaries[i + 1]] for i in range(len(original))]
    canonical_to_orig: list[list[int]] = []
    for canonical_index in range(len(canonical)):
        # Find the original interval whose normalized prefix contains this
        # codepoint.  Keeping equal prefix boundaries on the right captures
        # composing sequences such as halfwidth ``ﾊﾞ`` -> ``バ``.
        left = max((i for i, boundary in enumerate(boundaries) if boundary <= canonical_index), default=0)
        right = next((i for i, boundary in enumerate(boundaries) if boundary > canonical_index), len(original))
        while right < len(original) and right + 1 < len(boundaries) and boundaries[right] == boundaries[right + 1]:
            right += 1
        if right <= left:
            right = min(len(original), left + 1)
        canonical_to_orig.append([left, right])
    return orig_to_canonical, canonical_to_orig


def canonicalize_text(text: str, normalize_mode: str = "NFKC") -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if normalize_mode not in {"None", "NFC", "NFKC"}:
        raise ValueError(f"unsupported normalize_mode: {normalize_mode!r}")
    mode = "NFC" if normalize_mode == "None" else normalize_mode
    canonical = text if normalize_mode == "None" else unicodedata.normalize(mode, text)
    if normalize_mode == "None":
        orig_to_canonical = [[index, index + 1] for index in range(len(text))]
        canonical_to_orig = [[index, index + 1] for index in range(len(text))]
    else:
        orig_to_canonical, canonical_to_orig = _offset_maps(text, canonical, mode)
    return {
        "version": TEXT_LAYER_VERSION,
        "orig_text": text,
        "canonical_text": canonical,
        "normalize_mode": normalize_mode,
        "normalization": {"name": mode, "lossy": canonical != text},
        "orig_to_canonical": orig_to_canonical,
        "canonical_to_orig": canonical_to_orig,
        "canonical_sha256": sha256_bytes(canonical.encode("utf-8")),
        "text_layer_digest": stable_digest({
            "version": TEXT_LAYER_VERSION,
            "orig_text": text,
            "canonical_text": canonical,
            "normalize_mode": normalize_mode,
            "orig_to_canonical": orig_to_canonical,
            "canonical_to_orig": canonical_to_orig,
        }),
    }


def canonical_span_to_orig(layer: Mapping[str, Any], span: Sequence[int]) -> list[int]:
    start, end = (int(span[0]), int(span[1]))
    canonical = layer["canonical_text"]
    if start < 0 or end < start or end > len(canonical):
        raise ValueError("canonical span is outside canonical text")
    if start == end:
        return [0, 0]
    spans = layer["canonical_to_orig"][start:end]
    return [min(item[0] for item in spans), max(item[1] for item in spans)]


def orig_span_to_canonical(layer: Mapping[str, Any], span: Sequence[int]) -> list[int]:
    start, end = (int(span[0]), int(span[1]))
    original = layer["orig_text"]
    if start < 0 or end < start or end > len(original):
        raise ValueError("original span is outside original text")
    if start == end:
        return [0, 0]
    spans = layer["orig_to_canonical"][start:end]
    return [min(item[0] for item in spans), max(item[1] for item in spans)]


def validate_monotonic_spans(spans: Sequence[Sequence[int]], length: int, *, allow_gaps: bool = False) -> None:
    cursor = 0
    for index, span in enumerate(spans):
        if len(span) != 2:
            raise ValueError(f"span {index} is not a pair")
        start, end = int(span[0]), int(span[1])
        if start < 0 or end < start or end > length or (not allow_gaps and start != cursor):
            raise ValueError(f"invalid span at index {index}: {span}")
        cursor = end
    if not allow_gaps and cursor != length:
        raise ValueError("spans do not cover text exactly")


def compare_text_layers(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return true only when text and normalization provenance are identical."""
    keys = ("canonical_sha256", "text_layer_digest", "canonical_text", "normalize_mode", "orig_text")
    return all(left.get(key) == right.get(key) for key in keys)


__all__ = [
    "TEXT_LAYER_VERSION", "canonicalize_text", "canonical_span_to_orig", "orig_span_to_canonical",
    "validate_monotonic_spans", "compare_text_layers",
]
