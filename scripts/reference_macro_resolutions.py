"""Classify GAMEDATA reference macros and build the repair resolution table.

``normalize_qwen_reference_text`` deliberately refuses to guess: a homophone
branch is taken by fixed policy, and every other branch must arrive in the
``resolutions`` table or normalization fails closed.  This module produces that
table, separating what policy already decides from what needs acoustic
evidence and from what cannot be decided at all.

The decisive classification:

* ``policy``     - the branch is a homophone set (他/她), so no acoustic model
                   can distinguish it and the fixed policy value is used.
* ``default``    - a sole alternative, or a name macro resolved to the game's
                   established default for the player's name.
* ``evidence``   - an audibly distinct branch.  The spoken word must appear in
                   the item's transcript, and exactly one branch may match.
* ``unresolved`` - the macro's meaning is not recoverable from the data
                   (e.g. ``{TEXTJOIN#54}``, whose payload is a runtime join id,
                   or a macro name this project has never seen).  These are
                   withheld rather than guessed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping

try:
    from scripts.qwen3_timestamp_normalization import (
        ReferenceMacroSlot, reference_macro_slots)
except ModuleNotFoundError:  # direct script execution from repository root
    from qwen3_timestamp_normalization import (
        ReferenceMacroSlot, reference_macro_slots)

SCHEMA = "qwen3-reference-macro-resolutions-v1"

# The player's own name, as each game's voice track actually speaks it.  These
# are hypotheses about this corpus, not facts about the games: the evidence
# stage confirms or refutes them, and an unconfirmed default stays visible in
# the manifest as review_required.
GAME_NICKNAME_DEFAULTS = {
    "原神": "旅行者",
    "崩铁": "开拓者",
    "物华弥新": "收藏家",
}
_NON_SPOKEN = re.compile(r"[^0-9A-Za-z㐀-䶿一-鿿豈-﫿]+")


@dataclass(frozen=True)
class SlotPlan:
    """What is known about one macro slot, before any table is built."""

    run_stem: str
    literal: str
    macro: str
    kind: str
    value: str | None
    alternatives: tuple[str, ...]
    reason: str
    review_required: bool = False

    @property
    def key(self) -> str:
        return self.literal


def game_nickname_default(game: str | None) -> str | None:
    return GAME_NICKNAME_DEFAULTS.get(game or "")


def transcript_has(transcript: str | None, word: str) -> bool:
    """True when ``word`` is spoken in ``transcript``, ignoring punctuation."""
    if not transcript or not word:
        return False
    return _NON_SPOKEN.sub("", word) in _NON_SPOKEN.sub("", transcript)


def classify_slot(slot: ReferenceMacroSlot, *, run_stem: str, game: str | None,
                  transcript: str | None = None) -> SlotPlan:
    """Decide one slot from policy, then evidence, and withhold otherwise."""
    def plan(kind, value, reason, review=False):
        return SlotPlan(run_stem, slot.literal, slot.macro, kind, value,
                        slot.alternatives, reason, review)

    if slot.macro == "RUBY":
        # A ruby gloss is typeset over the base text and is never spoken.
        return plan("policy", "", "ruby gloss dropped")

    if slot.macro in ("NICKNAME", "REALNAME"):
        default = game_nickname_default(game)
        if default is None:
            return plan("unresolved", None, f"no nickname default for game {game!r}")
        if transcript_has(transcript, default):
            return plan("evidence", default, "default confirmed by transcript")
        return plan("default", default,
                    "game default (not confirmed by transcript)", review=True)

    if slot.alternatives:
        implied = slot.default
        if implied is not None:
            kind = "policy" if len(slot.alternatives) > 1 else "default"
            return plan(kind, implied, "homophone branch fixed by policy"
                        if len(slot.alternatives) > 1 else "sole alternative")
        matched = tuple(word for word in dict.fromkeys(slot.alternatives)
                        if transcript_has(transcript, word))
        if len(matched) == 1:
            return plan("evidence", matched[0], "unique transcript match")
        if not matched:
            return plan("unresolved", None, "no branch heard in transcript")
        return plan("unresolved", None,
                    f"ambiguous transcript match: {list(matched)}")

    if slot.labels:
        return plan("unresolved", None,
                    f"unrecognized macro labels: {list(slot.labels)}")
    return plan("unresolved", None, "unrecognized macro")


def plan_text(text: str, *, run_stem: str, game: str | None,
              transcript: str | None = None) -> tuple[SlotPlan, ...]:
    return tuple(classify_slot(slot, run_stem=run_stem, game=game,
                               transcript=transcript)
                 for slot in reference_macro_slots(text))


def is_actionable(plans: Iterable[SlotPlan]) -> bool:
    """True when every slot has a decided value, so the item may be repaired."""
    plans = tuple(plans)
    return bool(plans) and all(p.value is not None for p in plans)


def build_table(plans: Iterable[SlotPlan]) -> dict[str, str]:
    """Collect decided slots into the mapping the normalizer consumes.

    A literal repeated with two different values inside one item is a
    contradiction and is rejected rather than silently collapsed.
    """
    table: dict[str, str] = {}
    for plan in plans:
        if plan.value is None:
            continue
        existing = table.get(plan.key)
        if existing is not None and existing != plan.value:
            raise ValueError(
                f"conflicting macro resolutions for {plan.literal!r}: "
                f"{existing!r} vs {plan.value!r}")
        table[plan.key] = plan.value
    return table


def empty_plan(run_stem: str, literal: str, macro: str, reason: str) -> SlotPlan:
    return SlotPlan(run_stem, literal, macro, "unresolved", None, (), reason)


def table_digest(table: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(
        dict(sorted(table.items())), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_manifest_row(plan: SlotPlan) -> dict:
    row = asdict(plan)
    row["schema"] = SCHEMA
    return row


def write_resolutions(root: Path, tables: Mapping[str, Mapping[str, str]]) -> dict:
    """Persist the per-stem tables plus a digest binding them to the run."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = {stem: dict(sorted(table.items())) for stem, table in sorted(tables.items())}
    digest = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    target = root / "reference_macro_resolutions.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        {"schema": SCHEMA, "digest": digest, "tables": payload},
        ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return {"path": str(target), "digest": digest, "stems": len(payload)}


def read_resolutions(root: Path) -> tuple[dict[str, dict[str, str]], str]:
    payload = json.loads((Path(root) / "reference_macro_resolutions.json")
                         .read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unknown reference macro resolution schema")
    tables = payload["tables"]
    if hashlib.sha256(json.dumps(
            tables, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest() != payload.get("digest"):
        raise ValueError("reference macro resolution digest mismatch")
    return tables, payload["digest"]
