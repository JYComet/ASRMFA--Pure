#!/usr/bin/env python3
"""Shared occurrence-aware contract for non-verbal vocalisations.

NVV labels are semantic owners, not punctuation or generic angle-bracketed
text.  This module deliberately only recognizes names from ``NVV_NAMES`` and
compares their complete ordered occurrence sequence across the five published
tiers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

try:
    from pipeline_utils import NVV_NAMES
except ImportError:  # package-style imports in tests/tools
    from scripts.pipeline_utils import NVV_NAMES


NVV_CONTRACT_SCHEMA = "nvv-cross-tier-contract-v1"
NVV_CONTRACT_REASON = "cross_tier_nvv_sequence_mismatch"
NVV_EXPECTED_SEQUENCE_REASON = "frozen_expected_nvv_sequence_mismatch"
NVV_TIER_NAMES = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")

_NAMES = tuple(sorted(
    (str(name).strip("<>").upper() for name in NVV_NAMES),
    key=len, reverse=True))
_NAME_PATTERN = "|".join(re.escape(name) for name in _NAMES)
_MARKED = re.compile(r"(?:<|\[)(" + _NAME_PATTERN + r")(?:>|\])", re.I)
_BARE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + _NAME_PATTERN + r")(?![A-Za-z0-9_-])")


def _tier_text(value: object) -> str:
    """Flatten a tier-like value without depending on TextGrid classes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    intervals = getattr(value, "intervals", None)
    if intervals is not None:
        return " ".join(str(getattr(item, "text", "") or "")
                         for item in intervals)
    if isinstance(value, Mapping):
        return str(value.get("text", value.get("label", "")) or "")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_tier_text(item) for item in value)
    return str(value)


def nvv_sequence(value: object) -> list[str]:
    """Return canonical known-NVV labels in exact surface order.

    Both ``[Breathing]``/``<BREATHING>`` markup and bare emitted labels are
    accepted.  Unknown bracketed or angle-bracketed text is intentionally
    ignored and cannot satisfy this contract.
    """
    text = _tier_text(value)
    matches: list[tuple[int, str]] = []
    marked_spans: list[tuple[int, int]] = []
    for match in _MARKED.finditer(text):
        token = f"<{match.group(1).upper()}>"
        matches.append((match.start(), token))
        marked_spans.append(match.span())
    for match in _BARE.finditer(text):
        if any(start <= match.start() < end for start, end in marked_spans):
            continue
        matches.append((match.start(), f"<{match.group(1).upper()}>"))
    matches.sort(key=lambda item: item[0])
    return [token for _, token in matches]


def _tier_mapping(tiers: object) -> dict[str, object]:
    if isinstance(tiers, Mapping):
        return {name: tiers.get(name) for name in NVV_TIER_NAMES}
    values = getattr(tiers, "tiers", None)
    if values is not None:
        return {str(getattr(tier, "name", "")): tier for tier in values}
    if isinstance(tiers, Sequence) and not isinstance(tiers, (str, bytes, bytearray)):
        return {name: value for name, value in zip(NVV_TIER_NAMES, tiers)}
    return {}


def audit_nvv_contract(tiers: object) -> list[str]:
    """Return stable veto reasons for a missing or mismatched five-tier NVV axis."""
    mapping = _tier_mapping(tiers)
    sequences = [nvv_sequence(mapping.get(name)) for name in NVV_TIER_NAMES]
    if any(mapping.get(name) is None for name in NVV_TIER_NAMES):
        return [NVV_CONTRACT_REASON]
    if any(sequence != sequences[0] for sequence in sequences[1:]):
        return [NVV_CONTRACT_REASON]
    return []


def audit_expected_nvv_sequence(
        observed_tiers: object,
        expected_occurrence_sequence: object) -> list[str]:
    """Require each published tier to match a frozen ordered NVV sequence.

    Repeated labels are intentionally retained: their list positions are the
    occurrence ordinals.  An empty expected sequence remains valid only when
    all five present tiers also contain no known NVV labels.
    """
    mapping = _tier_mapping(observed_tiers)
    expected = nvv_sequence(expected_occurrence_sequence)
    if any(mapping.get(name) is None for name in NVV_TIER_NAMES):
        return [NVV_EXPECTED_SEQUENCE_REASON]
    if any(nvv_sequence(mapping.get(name)) != expected
           for name in NVV_TIER_NAMES):
        return [NVV_EXPECTED_SEQUENCE_REASON]
    return []


def build_nvv_contract(tiers: object) -> dict[str, object]:
    """Build a sealed contract payload for a final five-tier TextGrid."""
    mapping = _tier_mapping(tiers)
    sequences = {name: nvv_sequence(mapping.get(name))
                 for name in NVV_TIER_NAMES}
    reasons = audit_nvv_contract(tiers)
    payload: dict[str, object] = {
        "schema": NVV_CONTRACT_SCHEMA,
        "status": "rejected" if reasons else "verified",
        "reasons": reasons,
        "tiers": list(NVV_TIER_NAMES),
        "sequences": sequences,
        "occurrence_count": len(sequences[NVV_TIER_NAMES[0]]),
    }
    identity = dict(payload)
    payload["digest"] = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return payload


def validate_nvv_contract(contract: object, tiers: object | None = None) -> bool:
    """Validate a sealed contract and optionally bind it to final tier data."""
    if not isinstance(contract, Mapping):
        return False
    if contract.get("schema") != NVV_CONTRACT_SCHEMA:
        return False
    if contract.get("status") != "verified" or contract.get("reasons") != []:
        return False
    if contract.get("tiers") != list(NVV_TIER_NAMES):
        return False
    digest = contract.get("digest")
    identity = {key: value for key, value in contract.items() if key != "digest"}
    expected = hashlib.sha256(json.dumps(
        identity, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != expected:
        return False
    if tiers is not None:
        actual = build_nvv_contract(tiers)
        if actual.get("sequences") != contract.get("sequences"):
            return False
    return True
