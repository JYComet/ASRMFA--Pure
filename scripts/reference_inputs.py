"""Shared lookup for exact and legacy-suffixed transcript references."""
from __future__ import annotations

from pathlib import Path


class ReferenceInputError(ValueError):
    """Reference files cannot be mapped unambiguously to input stems."""


LEGACY_SUFFIXES = ("_qwen3-api", "_qwen3", "_firered")


def _candidate(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    for suffix in LEGACY_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[:-len(suffix)], 1
    return stem, 0


def resolve_references(root: Path, stems=None) -> dict[str, Path]:
    """Resolve logical stems; exact names win and ties at each priority fail."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ReferenceInputError(f"invalid reference root: {root}")
    wanted = set(stems) if stems is not None else None
    candidates: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(root.rglob("*.txt"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ReferenceInputError(f"symlink reference: {path}")
        if wanted is not None and path.stem in wanted:
            # A selected audio stem is always a literal exact match, even if
            # its own name happens to end in a legacy provider suffix.
            candidates.setdefault(path.stem, []).append((0, path))
            continue
        mapped = _candidate(path)
        if mapped is None:
            continue
        stem, priority = mapped
        if wanted is None or stem in wanted:
            candidates.setdefault(stem, []).append((priority, path))
    result = {}
    for stem, entries in candidates.items():
        priority = min(item[0] for item in entries)
        best = [path for rank, path in entries if rank == priority]
        if len(best) != 1:
            raise ReferenceInputError(f"ambiguous references for {stem}: {best}")
        result[stem] = best[0]
    return result


def load_reference_texts(root: Path, stems=None) -> dict[str, str]:
    return {stem: path.read_text(encoding="utf-8-sig").strip()
            for stem, path in resolve_references(root, stems).items()}
