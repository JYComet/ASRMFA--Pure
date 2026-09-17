#!/usr/bin/env python3
"""Collect flat NVV rerun outputs into fresh, speaker-classified private trees.

This collector is intentionally narrower than a quality gate: an absent
TextGrid or WAV is recorded as a per-row missing pair, not promoted to a batch
failure.  Namespace, provenance, and filesystem-safety errors are batch
failures, because accepting them could blend an unrelated rerun into the
frozen repair universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline_utils import (PIPELINE_ACCOUNTING_RECEIPT_NAME,
                            read_pipeline_accounting_receipt)


_MANIFEST_RECEIPT_SCHEMA = "gamedata-nvv-rerun-materialization-receipt-v1"
_COLLECTION_RECEIPT_SCHEMA = "gamedata-nvv-rerun-flat-collection-receipt-v1"
_FORBIDDEN_COMPONENTS = {"", ".", ".."}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


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


def _key_record(key: tuple[str, str, str]) -> dict[str, str]:
    game, speaker, stem = key
    return {"game": game, "speaker": speaker, "stem": stem}


def _has_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _absolute_path(value: Path | str, *, label: str) -> Path:
    if not isinstance(value, (Path, str)):
        raise ValueError(f"missing or invalid {label}")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path escape in {label}: {path}")
    if _has_symlink_component(path):
        raise ValueError(f"{label} is symlinked: {path}")
    return path


def _safe_file(value: Path | str, *, label: str) -> Path:
    path = _absolute_path(value, label=label)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing, invalid, or symlinked: {path}")
    return path


def _safe_flat_root(value: Path | str, *, label: str) -> Path:
    root = _absolute_path(value, label=label)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is missing, invalid, or symlinked: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} root contains symlink: {path}")
    return root.resolve()


def _load_manifest(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    manifest = _safe_file(path, label="frozen manifest")
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank frozen manifest JSONL line: {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid frozen manifest JSONL at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"frozen manifest row must be an object: line {line_number}")
        key = _key(row)
        if key in rows:
            raise ValueError(f"duplicate frozen manifest key: {key!r}")
        rows[key] = row
    if not rows:
        raise ValueError("frozen manifest is empty")
    return rows


def _load_receipt(
    path: Path, *, manifest_path: Path,
) -> tuple[dict[str, object], list[tuple[str, str, str]], dict[tuple[str, str, str], Mapping[str, object]]]:
    receipt_path = _safe_file(path, label="materialization receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid materialization receipt JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != _MANIFEST_RECEIPT_SCHEMA:
        raise ValueError("unexpected materialization receipt schema")
    receipt_manifest = receipt.get("manifest_path")
    if not isinstance(receipt_manifest, str):
        raise ValueError("materialization receipt missing manifest path")
    literal_manifest = _safe_file(Path(receipt_manifest), label="receipt manifest")
    if literal_manifest.resolve() != manifest_path.resolve():
        raise ValueError("materialization receipt manifest path mismatch")
    if receipt.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("materialization receipt manifest hash mismatch")
    raw_inputs = receipt.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("materialization receipt inputs missing or empty")
    selected: list[tuple[str, str, str]] = []
    inputs: dict[tuple[str, str, str], Mapping[str, object]] = {}
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(raw_inputs, 1):
        if not isinstance(item, Mapping):
            raise ValueError(f"materialization receipt input is not an object: {index}")
        key = _key(item)
        if key in seen:
            raise ValueError(f"duplicate materialization receipt input key: {key!r}")
        seen.add(key)
        selected.append(key)
        inputs[key] = item
    return receipt, sorted(selected), inputs


def _validate_selected_universe(
    manifest_rows: Mapping[tuple[str, str, str], Mapping[str, object]],
    selected: Sequence[tuple[str, str, str]],
) -> None:
    unknown = sorted(set(selected) - set(manifest_rows))
    if unknown:
        raise ValueError(f"materialization receipt key outside frozen manifest: {unknown[0]!r}")
    flat_names: set[tuple[str, str]] = set()
    for game, _, stem in selected:
        name = (game, stem)
        if name in flat_names:
            raise ValueError(f"selected flat stem collision: {name!r}")
        flat_names.add(name)


def _validate_game_roots(
    roots: Mapping[str, Path], *, selected_games: set[str], label: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for game, root in roots.items():
        safe_game = _safe_component(game, label=f"{label} game")
        if safe_game in result:
            raise ValueError(f"duplicate {label} root game: {safe_game}")
        result[safe_game] = _safe_flat_root(root, label=f"{label} {safe_game}")
    if set(result) != selected_games:
        raise ValueError(
            f"{label} roots must exactly cover selected games: "
            f"missing={sorted(selected_games - set(result))!r} "
            f"extra={sorted(set(result) - selected_games)!r}")
    return result


def _validate_game_receipts(
    receipts: Mapping[str, Path], *, selected_games: set[str],
    textgrid_roots: Mapping[str, Path], selected_stems: Mapping[str, set[str]],
) -> dict[str, dict[str, object]]:
    """Bind every flat TextGrid root to its own complete v2 pipeline receipt."""
    normalized: dict[str, Path] = {}
    for game, receipt_path in receipts.items():
        safe_game = _safe_component(game, label="flat TextGrid receipt game")
        if safe_game in normalized:
            raise ValueError(f"duplicate flat TextGrid receipt game: {safe_game}")
        normalized[safe_game] = _safe_file(receipt_path, label=f"pipeline receipt {safe_game}")
    if set(normalized) != selected_games:
        raise ValueError(
            "flat TextGrid receipts must exactly cover selected games: "
            f"missing={sorted(selected_games - set(normalized))!r} "
            f"extra={sorted(set(normalized) - selected_games)!r}")

    records: dict[str, dict[str, object]] = {}
    for game in sorted(selected_games):
        root = textgrid_roots[game]
        receipt_path = normalized[game]
        if (receipt_path.name != PIPELINE_ACCOUNTING_RECEIPT_NAME
                or receipt_path.parent.resolve() != root.resolve()):
            raise ValueError(f"pipeline receipt path mismatch for {game}: {receipt_path}")
        try:
            receipt = read_pipeline_accounting_receipt(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid pipeline receipt for {game}: {receipt_path}") from exc
        paths = receipt.get("paths")
        output_path = paths.get("output") if isinstance(paths, Mapping) else None
        if not isinstance(output_path, str):
            raise ValueError(f"pipeline receipt output root missing for {game}")
        bound_output = _absolute_path(output_path, label=f"pipeline receipt output {game}")
        if bound_output.resolve() != root.resolve():
            raise ValueError(f"pipeline receipt output root mismatch for {game}")
        source = receipt.get("source")
        eligible = receipt.get("eligible")
        output = receipt.get("output")
        filtered = receipt.get("filtered")
        if not all(isinstance(bucket, Mapping)
                   for bucket in (source, eligible, output, filtered)):
            raise ValueError(f"pipeline receipt accounting buckets missing for {game}")
        expected = selected_stems[game]
        source_stems = set(source["stems"])
        eligible_stems = set(eligible["stems"])
        output_stems = set(output["stems"])
        filtered_stems = set(filtered["stems"])
        if source_stems != expected or eligible_stems != expected:
            raise ValueError(f"pipeline receipt input/eligible stem mismatch for {game}")
        if receipt.get("exclusions") != []:
            raise ValueError(f"pipeline receipt exclusions not permitted for {game}")
        if output_stems | filtered_stems != expected:
            raise ValueError(f"pipeline receipt output accounting mismatch for {game}")
        records[game] = {
            "path": str(receipt_path.resolve()),
            "sha256": _sha256(receipt_path),
            "output_root": str(bound_output.resolve()),
            "output_stems": output_stems,
        }
    return records


def _receipt_input_wav_binding(
    materialization: Mapping[str, object],
    inputs: Mapping[tuple[str, str, str], Mapping[str, object]],
    selected: Sequence[tuple[str, str, str]], *, wav_roots: Mapping[str, Path],
) -> dict[tuple[str, str, str], dict[str, object]]:
    """Require each supplied game WAV root to be the receipt's inputs/game root."""
    run_root_value = materialization.get("run_root")
    run_root = _absolute_path(run_root_value, label="materialization receipt run root")
    bindings: dict[tuple[str, str, str], dict[str, object]] = {}
    expected_roots: dict[str, Path] = {}
    for key in selected:
        game, _, stem = key
        item = inputs[key]
        target_value = item.get("target_wav_path")
        target = _absolute_path(target_value, label=f"receipt target WAV {game}/{stem}")
        expected_root = run_root / "inputs" / game
        if target.parent.resolve() != expected_root.resolve() or target.name != f"{stem}.wav":
            raise ValueError(f"receipt target WAV path mismatch for {game}/{stem}")
        existing_root = expected_roots.setdefault(game, expected_root)
        if existing_root.resolve() != expected_root.resolve():  # pragma: no cover - defensive
            raise ValueError(f"receipt-bound inputs root mismatch for {game}")
        source_hash = item.get("source_wav_sha256")
        target_hash = item.get("target_sha256")
        if (not isinstance(source_hash, str) or not isinstance(target_hash, str)
                or _SHA256.fullmatch(source_hash) is None
                or _SHA256.fullmatch(target_hash) is None
                or source_hash != target_hash):
            raise ValueError(f"receipt WAV hash binding mismatch for {game}/{stem}")
        bindings[key] = {
            "target_path": target,
            "sha256": target_hash,
        }
    for game, expected_root in expected_roots.items():
        if wav_roots[game].resolve() != expected_root.resolve():
            raise ValueError(f"flat WAV root is not receipt-bound inputs root for {game}")
    return bindings


def _validate_supplied_wav_bindings(
    bindings: Mapping[tuple[str, str, str], Mapping[str, object]],
    wavs: Mapping[str, Mapping[str, Path]],
) -> None:
    """A present WAV must be the exact materialized receipt input and bytes."""
    for (game, _speaker, stem), binding in bindings.items():
        source = wavs[game].get(stem)
        if source is None:
            continue
        target = binding["target_path"]
        if not isinstance(target, Path) or source.resolve() != target.resolve():
            raise ValueError(f"flat WAV path mismatch for {game}/{stem}")
        expected_hash = binding["sha256"]
        if not isinstance(expected_hash, str) or _sha256(source) != expected_hash:
            raise ValueError(f"flat WAV hash mismatch for {game}/{stem}")


def _validate_publish_manifests(
    *, textgrid_roots: Mapping[str, Path], grids: Mapping[str, Mapping[str, Path]],
    pipeline_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    """Byte-bind every collected TextGrid to its same-root schema-2 manifest."""
    records: dict[str, dict[str, str]] = {}
    for game, root in sorted(textgrid_roots.items()):
        manifest_path = _safe_file(root / ".publish_manifest.json",
                                   label=f"publish manifest {game}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid publish manifest JSON for {game}") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema") != 2:
            raise ValueError(f"publish manifest schema mismatch for {game}")
        source_value = manifest.get("source")
        source = _absolute_path(source_value, label=f"publish manifest source {game}")
        pipeline_output = pipeline_receipts[game].get("output_root")
        if (source.resolve() != root.resolve()
                or not isinstance(pipeline_output, str)
                or source.resolve() != _absolute_path(
                    pipeline_output, label=f"pipeline receipt output {game}").resolve()):
            raise ValueError(f"publish manifest source mismatch for {game}")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ValueError(f"publish manifest files missing for {game}")
        indexed: dict[str, tuple[int, str]] = {}
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, Mapping):
                raise ValueError(f"invalid publish manifest entry for {game}: {index}")
            relative = entry.get("path")
            size = entry.get("size")
            digest = entry.get("sha256")
            if (not isinstance(relative, str) or not relative or "\\" in relative
                    or not isinstance(size, int) or size < 0
                    or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None):
                raise ValueError(f"invalid publish manifest entry for {game}: {index}")
            relative_path = Path(relative)
            if (relative_path.is_absolute() or ".." in relative_path.parts
                    or "." in relative_path.parts
                    or relative_path.as_posix() != relative):
                raise ValueError(f"publish manifest path escape for {game}: {relative!r}")
            if relative in indexed:
                raise ValueError(f"duplicate publish manifest path for {game}: {relative!r}")
            indexed[relative] = (size, digest)
        for stem, grid_path in grids[game].items():
            relative = grid_path.relative_to(root).as_posix()
            expected = indexed.get(relative)
            if expected is None:
                raise ValueError(f"publish manifest missing TextGrid for {game}/{stem}")
            size, digest = expected
            if grid_path.stat().st_size != size:
                raise ValueError(f"publish manifest TextGrid size mismatch for {game}/{stem}")
            if _sha256(grid_path) != digest:
                raise ValueError(f"publish manifest TextGrid hash mismatch for {game}/{stem}")
        records[game] = {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
        }
    return records


def _scan_flat_root(
    root: Path, *, game: str, suffix: str, selected_stems: set[str], label: str,
) -> dict[str, Path]:
    """Return direct flat files and reject stale or nested target artifacts."""
    result: dict[str, Path] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid {label} source: {path}")
        relative = path.relative_to(root)
        if len(relative.parts) != 1 or path.suffix != suffix or not path.stem:
            raise ValueError(f"{label} source is not a flat {suffix} file: {path}")
        stem = path.stem
        if stem not in selected_stems:
            raise ValueError(
                f"{label} stem outside selected universe: {(game, stem)!r}")
        if stem in result:
            raise ValueError(f"duplicate {label} flat stem: {(game, stem)!r}")
        result[stem] = path
    return result


def _validate_flat_media_namespace(
    root: Path, *, game: str, selected_stems: set[str], label: str,
) -> None:
    """Reject any stale TextGrid/WAV artifact in either supplied flat root."""
    for suffix in (".TextGrid", ".wav"):
        _scan_flat_root(root, game=game, suffix=suffix,
                        selected_stems=selected_stems, label=label)


def _validate_fresh_output_root(
    value: Path | str, *, label: str, protected: Sequence[Path],
) -> Path:
    root = _absolute_path(value, label=label)
    if root.exists() or root.is_symlink():
        raise ValueError(f"{label} must be fresh and must not overwrite: {root}")
    resolved = root.resolve()
    for protected_path in protected:
        try:
            resolved.relative_to(protected_path.resolve())
        except ValueError:
            continue
        raise ValueError(f"{label} must not be inside protected input: {root}")
    return root


def _validate_fresh_receipt(
    value: Path | str, *, protected: Sequence[Path], output_roots: Sequence[Path],
) -> Path:
    path = _absolute_path(value, label="collection receipt")
    if path.exists() or path.is_symlink():
        raise ValueError(f"collection receipt must not overwrite: {path}")
    resolved = path.resolve()
    for root in [*protected, *output_roots]:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise ValueError(f"collection receipt must not be inside protected output/input: {path}")
    return path


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno != getattr(os, "EXDEV", 18):
            raise
        shutil.copy2(source, target)
        return "copy"


def _copy_record(
    source: Path, target: Path,
) -> dict[str, object]:
    method = _link_or_copy(source, target)
    source_hash = _sha256(source)
    target_hash = _sha256(target)
    if source_hash != target_hash:
        raise ValueError(f"collected target hash mismatch: {target}")
    return {
        "method": method,
        "source_path": str(source),
        "source_sha256": source_hash,
        "target_path": str(target),
        "target_sha256": target_hash,
    }


def collect_nvv_rerun_outputs(
    *, manifest_path: Path, materialization_receipt_path: Path,
    flat_textgrid_roots: Mapping[str, Path], flat_wav_roots: Mapping[str, Path],
    flat_textgrid_receipts: Mapping[str, Path],
    textgrid_output_root: Path, wav_output_root: Path,
    collection_receipt_path: Path,
) -> dict[str, object]:
    """Classify selected flat pairs into fresh private trees and write a receipt.

    A missing half pair is not copied.  It remains in ``missing`` so the
    semantic NVV gate can issue its normal per-row rejection afterwards.
    """
    manifest = _safe_file(manifest_path, label="frozen manifest")
    manifest_rows = _load_manifest(manifest)
    materialization, selected, receipt_inputs = _load_receipt(
        materialization_receipt_path, manifest_path=manifest)
    _validate_selected_universe(manifest_rows, selected)
    selected_games = {game for game, _, _ in selected}
    grid_roots = _validate_game_roots(
        flat_textgrid_roots, selected_games=selected_games, label="flat TextGrid")
    wav_roots = _validate_game_roots(
        flat_wav_roots, selected_games=selected_games, label="flat WAV")
    selected_stems = {
        game: {stem for current_game, _, stem in selected if current_game == game}
        for game in selected_games
    }
    pipeline_receipts = _validate_game_receipts(
        flat_textgrid_receipts, selected_games=selected_games,
        textgrid_roots=grid_roots, selected_stems=selected_stems)
    wav_bindings = _receipt_input_wav_binding(
        materialization, receipt_inputs, selected, wav_roots=wav_roots)
    for game in sorted(selected_games):
        _validate_flat_media_namespace(
            grid_roots[game], game=game, selected_stems=selected_stems[game],
            label="flat TextGrid")
        _validate_flat_media_namespace(
            wav_roots[game], game=game, selected_stems=selected_stems[game],
            label="flat WAV")
    grids = {
        game: _scan_flat_root(grid_roots[game], game=game, suffix=".TextGrid",
                              selected_stems=selected_stems[game], label="TextGrid")
        for game in sorted(selected_games)
    }
    wavs = {
        game: _scan_flat_root(wav_roots[game], game=game, suffix=".wav",
                              selected_stems=selected_stems[game], label="WAV")
        for game in sorted(selected_games)
    }
    _validate_supplied_wav_bindings(wav_bindings, wavs)
    for game, pipeline in pipeline_receipts.items():
        if set(grids[game]) != pipeline["output_stems"]:
            raise ValueError(f"pipeline receipt/TextGrid output mismatch for {game}")
    publish_manifests = _validate_publish_manifests(
        textgrid_roots=grid_roots, grids=grids,
        pipeline_receipts=pipeline_receipts)
    protected = [manifest, _safe_file(materialization_receipt_path,
                                      label="materialization receipt"),
                 *grid_roots.values(), *wav_roots.values()]
    grid_output = _validate_fresh_output_root(
        textgrid_output_root, label="TextGrid output root", protected=protected)
    wav_output = _validate_fresh_output_root(
        wav_output_root, label="WAV output root", protected=protected)
    if grid_output.resolve() == wav_output.resolve():
        raise ValueError("TextGrid and WAV output roots must differ")
    try:
        grid_output.resolve().relative_to(wav_output.resolve())
    except ValueError:
        try:
            wav_output.resolve().relative_to(grid_output.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("TextGrid and WAV output roots must not be nested")
    else:
        raise ValueError("TextGrid and WAV output roots must not be nested")
    receipt_output = _validate_fresh_receipt(
        collection_receipt_path, protected=protected,
        output_roots=(grid_output, wav_output))

    missing: list[dict[str, object]] = []
    complete: list[tuple[str, str, str]] = []
    for key in selected:
        game, speaker, stem = key
        grid_present = stem in grids[game]
        wav_present = stem in wavs[game]
        if grid_present and wav_present:
            complete.append(key)
        else:
            missing.append({
                "game": game, "grid_present": grid_present,
                "reason": "missing_flat_pair", "speaker": speaker,
                "stem": stem, "wav_present": wav_present,
            })

    # The semantic gate consumes roots, not an optional pair list.  Materialize
    # both fresh roots even when every selected item is a normal missing-pair
    # rejection, while still leaving them absent for all preflight hard-fails.
    grid_output.mkdir(parents=True, exist_ok=False)
    wav_output.mkdir(parents=True, exist_ok=False)
    collected: list[dict[str, object]] = []
    for game, speaker, stem in complete:
        grid_target = grid_output / game / speaker / f"{stem}.TextGrid"
        wav_target = wav_output / game / speaker / f"{stem}.wav"
        collected.append({
            "game": game,
            "speaker": speaker,
            "stem": stem,
            "textgrid": _copy_record(grids[game][stem], grid_target),
            "wav": _copy_record(wavs[game][stem], wav_target),
        })

    selected_counts = Counter(game for game, _, _ in selected)
    collected_counts = Counter(item["game"] for item in collected)
    missing_counts = Counter(str(item["game"]) for item in missing)
    game_counts = {
        game: {
            "selected": selected_counts[game], "collected": collected_counts[game],
            "missing": missing_counts[game],
        }
        for game in sorted(selected_games)
    }
    receipt: dict[str, object] = {
        "schema": _COLLECTION_RECEIPT_SCHEMA,
        "frozen_manifest": {"path": str(manifest.resolve()), "sha256": _sha256(manifest)},
        "materialization_receipt": {
            "path": str(_safe_file(materialization_receipt_path,
                                     label="materialization receipt").resolve()),
            "sha256": _sha256(_safe_file(materialization_receipt_path,
                                           label="materialization receipt")),
        },
        "materialization_selected_count": materialization.get("selected_count"),
        "pipeline_receipts": {
            game: {
                "path": record["path"], "sha256": record["sha256"],
                "publish_manifest": publish_manifests[game],
            }
            for game, record in sorted(pipeline_receipts.items())
        },
        "textgrid_output_root": str(grid_output),
        "wav_output_root": str(wav_output),
        "selected": [_key_record(key) for key in selected],
        "collected": collected,
        "missing": missing,
        "selected_count": len(selected),
        "collected_count": len(collected),
        "missing_count": len(missing),
        "game_counts": game_counts,
    }
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    with receipt_output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_canonical_json(receipt).decode("utf-8"))
    receipt["receipt_path"] = str(receipt_output)
    receipt["receipt_sha256"] = _sha256(receipt_output)
    return receipt


def _parse_game_root(values: Sequence[str] | None, *, label: str) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{label} must use GAME=DIR: {value!r}")
        game, raw_path = value.split("=", 1)
        game = _safe_component(game, label=f"{label} game")
        if game in roots or not raw_path:
            raise ValueError(f"duplicate or invalid {label}: {value!r}")
        roots[game] = Path(raw_path)
    if not roots:
        raise ValueError(f"at least one {label} is required")
    return roots


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--materialization-receipt", required=True, type=Path)
    parser.add_argument("--flat-textgrid-root", action="append", metavar="GAME=DIR")
    parser.add_argument("--flat-textgrid-receipt", action="append", metavar="GAME=PATH",
                        help="per-game .pipeline_run_receipt_v2.json beside the flat TextGrid root")
    parser.add_argument("--flat-wav-root", action="append", metavar="GAME=DIR")
    parser.add_argument("--textgrid-output-root", required=True, type=Path)
    parser.add_argument("--wav-output-root", required=True, type=Path)
    parser.add_argument("--collection-receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _arguments()
    args = parser.parse_args(argv)
    try:
        receipt = collect_nvv_rerun_outputs(
            manifest_path=args.manifest,
            materialization_receipt_path=args.materialization_receipt,
            flat_textgrid_roots=_parse_game_root(args.flat_textgrid_root,
                                                  label="flat TextGrid root"),
            flat_textgrid_receipts=_parse_game_root(args.flat_textgrid_receipt,
                                                     label="flat TextGrid receipt"),
            flat_wav_roots=_parse_game_root(args.flat_wav_root,
                                             label="flat WAV root"),
            textgrid_output_root=args.textgrid_output_root,
            wav_output_root=args.wav_output_root,
            collection_receipt_path=args.collection_receipt,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
