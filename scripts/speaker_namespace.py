"""Canonical public speaker namespace for flat multi-game outputs."""
from __future__ import annotations

import re
from pathlib import Path

from pypinyin import Style, lazy_pinyin


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def game_initial_prefix(game: str) -> str:
    """Return uppercase pinyin initials for the first two Hanzi in a game."""
    hanzi = _CJK.findall(str(game or ""))[:2]
    if len(hanzi) != 2:
        raise ValueError(f"game name needs at least two Hanzi for speaker prefix: {game!r}")
    prefix = "".join(lazy_pinyin(hanzi, style=Style.FIRST_LETTER)).upper()
    if len(prefix) != 2 or not prefix.isascii() or not prefix.isalpha():
        raise ValueError(f"invalid game speaker prefix: {game!r} -> {prefix!r}")
    return prefix


def publication_speaker(game: str | None, speaker: str) -> str:
    """Namespace game speakers while preserving already-prefixed names."""
    speaker = str(speaker or "")
    if not speaker or speaker in {".", ".."} or Path(speaker).name != speaker:
        raise ValueError(f"unsafe speaker: {speaker!r}")
    if not game:
        return speaker
    prefix = game_initial_prefix(game)
    return speaker if speaker[:2].upper() == prefix else prefix + speaker

