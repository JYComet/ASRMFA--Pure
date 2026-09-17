#!/usr/bin/env python3
"""Fail-closed CPU acceptance gate for a frozen NVV rerun manifest.

The gate binds every frozen ``game/speaker/stem`` row to exactly one final
TextGrid underneath a fresh rerun root.  It never changes TextGrids or WAVs:
its only outputs are a deterministic decision report and the accepted/rejected
subsets of the input manifest.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from nvv_contract import (
    NVV_TIER_NAMES,
    audit_expected_nvv_sequence,
    audit_nvv_contract,
    nvv_sequence,
)
from pipeline_utils import is_nvv_token, is_punct
from postprocess_textgrids import AXIS_EPS, parse_textgrid


_FORBIDDEN_COMPONENTS = {"", ".", ".."}
_PARSE_REASON = "textgrid_parse_error"
_MISSING_GRID_REASON = "missing_rerun_textgrid"
_TEMPORAL_OWNER_REASON = "nvv_temporal_owner_mismatch"
_TEMPORAL_TIER_NAMES = ("hanzi", "words", "pinyin_phones")


def _safe_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value in _FORBIDDEN_COMPONENTS:
        raise ValueError(f"invalid {label} component: {value!r}")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(f"path escape in {label} component: {value!r}")
    return value


def _key(record: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        _safe_component(record.get("game"), label="game"),
        _safe_component(record.get("speaker"), label="speaker"),
        _safe_component(record.get("stem"), label="stem"),
    )


def _expected_sequence(record: Mapping[str, object]) -> list[str]:
    if "expected_nvv_sequence" not in record:
        raise ValueError("missing expected_nvv_sequence")
    value = record["expected_nvv_sequence"]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("expected_nvv_sequence must be a JSON array")
    return nvv_sequence(value)


def _read_manifest(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest is missing, invalid, or symlinked: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank manifest JSONL line: {line_number}")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSONL at line {line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"manifest JSONL row must be object: line {line_number}")
        _key(item)
        _expected_sequence(item)
        rows.append(item)
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def _safe_root(root: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"TextGrid root is missing, invalid, or symlinked: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"TextGrid root contains symlink: {path}")
    return root.resolve()


def _grid_paths(root: Path) -> dict[tuple[str, str, str], Path]:
    paths: dict[tuple[str, str, str], Path] = {}
    for path in sorted(root.rglob("*.TextGrid")):
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            raise ValueError(f"TextGrid does not use game/speaker/stem layout: {path}")
        game, speaker, filename = relative.parts
        if path.suffix != ".TextGrid" or not path.stem:
            raise ValueError(f"invalid TextGrid filename: {path}")
        key = (
            _safe_component(game, label="game"),
            _safe_component(speaker, label="speaker"),
            _safe_component(path.stem, label="stem"),
        )
        if key in paths:
            raise ValueError(f"duplicate rerun TextGrid key: {key!r}")
        paths[key] = path
    return paths


def _validate_universe(
    rows: list[dict[str, object]], *, textgrid_root: Path,
) -> tuple[list[tuple[tuple[str, str, str], dict[str, object]]], dict[tuple[str, str, str], Path]]:
    ordered: list[tuple[tuple[str, str, str], dict[str, object]]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate manifest key: {key!r}")
        seen.add(key)
        ordered.append((key, row))
    grids = _grid_paths(textgrid_root)
    unlisted = sorted(set(grids) - seen)
    if unlisted:
        raise ValueError(f"rerun TextGrid not present in frozen manifest: {unlisted[0]!r}")
    return sorted(ordered), grids


def _observed_sequences(grid: object) -> dict[str, list[str]]:
    tiers = {str(getattr(tier, "name", "")): tier
             for tier in getattr(grid, "tiers", [])}
    return {name: nvv_sequence(tiers.get(name)) for name in NVV_TIER_NAMES}


def _tier_mapping(grid: object) -> dict[str, object]:
    return {str(getattr(tier, "name", "")): tier
            for tier in getattr(grid, "tiers", [])}


def _nvv_interval_label(interval: object) -> str | None:
    """Return a canonical label only for a complete NVV owner interval."""
    text = str(getattr(interval, "text", "") or "").strip()
    if not is_nvv_token(text):
        return None
    sequence = nvv_sequence(text)
    return sequence[0] if len(sequence) == 1 else None


def _nvv_owners(tier: object) -> list[tuple[str, float, float]]:
    owners: list[tuple[str, float, float]] = []
    for interval in getattr(tier, "intervals", []):
        label = _nvv_interval_label(interval)
        if label is not None:
            owners.append((label, float(interval.xmin), float(interval.xmax)))
    return owners


def _same_owner_spans(
    left: Sequence[tuple[str, float, float]],
    right: Sequence[tuple[str, float, float]],
) -> bool:
    return (len(left) == len(right)
            and all(left_label == right_label
                    and abs(left_start - right_start) <= AXIS_EPS
                    and abs(left_end - right_end) <= AXIS_EPS
                    for (left_label, left_start, left_end),
                    (right_label, right_start, right_end) in zip(left, right)))


def _punctuation_overlaps_owner(
    tier: object, owners: Sequence[tuple[str, float, float]],
) -> bool:
    for interval in getattr(tier, "intervals", []):
        if not is_punct(str(getattr(interval, "text", "") or "")):
            continue
        start, end = float(interval.xmin), float(interval.xmax)
        if any(start < owner_end - AXIS_EPS and end > owner_start + AXIS_EPS
               for _, owner_start, owner_end in owners):
            return True
    return False


def _audit_temporal_nvv_ownership(grid: object) -> list[str]:
    """Require one exact, punctuation-free NVV time owner in all timed tiers."""
    tiers = _tier_mapping(grid)
    if any(tiers.get(name) is None for name in _TEMPORAL_TIER_NAMES):
        return [_TEMPORAL_OWNER_REASON]
    owners = {name: _nvv_owners(tiers[name]) for name in _TEMPORAL_TIER_NAMES}
    baseline = owners["hanzi"]
    if any(not _same_owner_spans(baseline, owners[name])
           for name in _TEMPORAL_TIER_NAMES[1:]):
        return [_TEMPORAL_OWNER_REASON]
    if any(_punctuation_overlaps_owner(tiers[name], owners[name])
           for name in _TEMPORAL_TIER_NAMES):
        return [_TEMPORAL_OWNER_REASON]
    return []


def _decision(
    *, key: tuple[str, str, str], row: Mapping[str, object], grid_path: Path | None,
    expected_path: Path,
) -> dict[str, object]:
    game, speaker, stem = key
    expected = _expected_sequence(row)
    reasons: list[str] = []
    observed = {name: [] for name in NVV_TIER_NAMES}
    if grid_path is None:
        reasons.append(_MISSING_GRID_REASON)
    else:
        try:
            grid = parse_textgrid(grid_path)
            observed = _observed_sequences(grid)
            contract_reasons = audit_nvv_contract(grid)
            reasons.extend(contract_reasons)
            reasons.extend(audit_expected_nvv_sequence(grid, expected))
            if not contract_reasons:
                reasons.extend(_audit_temporal_nvv_ownership(grid))
        except Exception:
            reasons.append(_PARSE_REASON)
    reasons = list(dict.fromkeys(reasons))
    return {
        "expected_nvv_sequence": expected,
        "game": game,
        "observed_nvv_sequences": observed,
        "path": str(grid_path if grid_path is not None else expected_path),
        "reasons": reasons,
        "speaker": speaker,
        "status": "rejected" if reasons else "accepted",
        "stem": stem,
    }


def _validate_output_paths(paths: Sequence[Path], *, protected: Sequence[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate output path: {path}")
        seen.add(resolved)
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite output: {path}")
        for root in protected:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            raise ValueError(f"output must not be inside protected input: {path}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _write_stems(path: Path, keys: Sequence[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for game, speaker, stem in keys:
            handle.write(f"{game}/{speaker}/{stem}\n")


def verify_nvv_rerun_manifest(
    *, manifest_path: Path, textgrid_root: Path, report_path: Path,
    accepted_manifest_path: Path, rejected_manifest_path: Path,
    accepted_stems_path: Path, rejected_stems_path: Path,
) -> dict[str, int]:
    """Audit every frozen rerun row and write deterministic decision subsets.

    Structural problems (duplicate keys, path escapes, or unexpected rerun
    TextGrids) are hard failures and produce no decision files.  A missing
    rerun target, parse failure, or semantic contract problem is a normal
    rejected row, retaining the complete input universe for review.
    """
    manifest = Path(manifest_path)
    root = _safe_root(Path(textgrid_root))
    output_paths = [
        Path(report_path), Path(accepted_manifest_path),
        Path(rejected_manifest_path), Path(accepted_stems_path),
        Path(rejected_stems_path),
    ]
    _validate_output_paths(output_paths, protected=(root, manifest.resolve()))
    rows = _read_manifest(manifest)
    ordered, grids = _validate_universe(rows, textgrid_root=root)

    report = [_decision(
        key=key, row=row, grid_path=grids.get(key),
        expected_path=root / key[0] / key[1] / f"{key[2]}.TextGrid")
              for key, row in ordered]
    accepted_pairs = [(item, result) for item, result in zip(ordered, report)
                      if result["status"] == "accepted"]
    rejected_pairs = [(item, result) for item, result in zip(ordered, report)
                      if result["status"] == "rejected"]
    accepted_rows = [row for (_, row), _ in accepted_pairs]
    rejected_rows = [row for (_, row), _ in rejected_pairs]
    accepted_keys = [key for (key, _), _ in accepted_pairs]
    rejected_keys = [key for (key, _), _ in rejected_pairs]

    _write_jsonl(output_paths[0], report)
    _write_jsonl(output_paths[1], accepted_rows)
    _write_jsonl(output_paths[2], rejected_rows)
    _write_stems(output_paths[3], accepted_keys)
    _write_stems(output_paths[4], rejected_keys)
    return {"accepted": len(accepted_rows), "rejected": len(rejected_rows),
            "total": len(report)}


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--textgrid-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--accepted-manifest", required=True, type=Path)
    parser.add_argument("--rejected-manifest", required=True, type=Path)
    parser.add_argument("--accepted-stems", required=True, type=Path)
    parser.add_argument("--rejected-stems", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    try:
        result = verify_nvv_rerun_manifest(
            manifest_path=args.manifest, textgrid_root=args.textgrid_root,
            report_path=args.report,
            accepted_manifest_path=args.accepted_manifest,
            rejected_manifest_path=args.rejected_manifest,
            accepted_stems_path=args.accepted_stems,
            rejected_stems_path=args.rejected_stems,
        )
    except ValueError as exc:
        _arguments().error(str(exc))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
