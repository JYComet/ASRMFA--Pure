#!/usr/bin/env python3
"""Create a private, hash-bound input tree for an NVV-only rerun.

This is deliberately a preparation step.  It never invokes the pipeline and
never writes a public aligned or audio directory.  The output run root must be
new so an operator cannot accidentally blend a rerun with a previous attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


MANIFEST_SCHEMA = "gamedata-nvv-repair-manifest-v1"
RECEIPT_SCHEMA = "gamedata-nvv-rerun-materialization-receipt-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_COMPONENTS = {"", ".", ".."}


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


def _key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        _safe_component(row.get("game"), label="game"),
        _safe_component(row.get("speaker"), label="speaker"),
        _safe_component(row.get("stem"), label="stem"),
    )


def _has_symlink_component(path: Path) -> bool:
    """Detect symlinks in the literal path, including a symlinked parent."""
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == current.parent:
            return False
        current = current.parent


def _absolute_literal_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or invalid {label}")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path escape in {label}: {path}")
    return path


def _source(row: Mapping[str, object]) -> tuple[Path, str]:
    source = _absolute_literal_path(row.get("source_wav_path"), label="source_wav_path")
    if _has_symlink_component(source) or source.is_symlink():
        raise ValueError(f"source WAV is symlinked: {source}")
    if not source.is_file() or source.suffix.lower() != ".wav":
        raise ValueError(f"source WAV missing or invalid: {source}")
    source_root = _absolute_literal_path(
        row.get("source_wav_root"), label="source_wav_root")
    if _has_symlink_component(source_root) or source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"source_wav_root missing, invalid, or symlinked: {source_root}")
    try:
        source.resolve().relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"source WAV outside protected source root: {source}") from exc
    expected = row.get("source_wav_sha256")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError("invalid source_wav_sha256")
    actual = _sha256(source)
    if actual != expected:
        raise ValueError(f"source WAV hash mismatch: {source}")
    return source, actual


def _expected_nvv(row: Mapping[str, object]) -> tuple[str, ...]:
    sequence = row.get("expected_nvv_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("missing or empty expected_nvv_sequence")
    if any(not isinstance(label, str) or not label for label in sequence):
        raise ValueError("invalid expected_nvv_sequence")
    return tuple(sequence)


def _validate_row(row: object, *, index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"manifest row {index} is not an object")
    schema = row.get("schema")
    if schema is not None and schema != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema in row {index}: {schema!r}")
    key = _key(row)
    if row.get("asr_mode") != "fallback" or row.get("reference_mode") != "fallback":
        raise ValueError(f"manifest row {index} is not a fallback ASR row")
    source, source_sha256 = _source(row)
    expected_nvv = _expected_nvv(row)
    return {
        "key": key,
        "source": source,
        "source_sha256": source_sha256,
        "expected_nvv_sequence": list(expected_nvv),
        "row": dict(row),
    }


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"manifest missing, invalid, or symlinked: {manifest_path}")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, str]] = set()
    target_sources: dict[tuple[str, str], str] = {}
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank manifest line: {line_no}")
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON at line {line_no}") from exc
        item = _validate_row(parsed, index=line_no)
        key = item["key"]
        if key in keys:
            raise ValueError(f"duplicate manifest key: {'/'.join(key)}")
        keys.add(key)
        target_key = (key[0], key[2])
        old_source = target_sources.setdefault(target_key, item["source_sha256"])
        if old_source != item["source_sha256"]:
            raise ValueError(
                "target stem collision with different source: "
                f"{target_key[0]}/{target_key[1]}")
        rows.append(item)
    if not rows:
        raise ValueError("frozen manifest is empty")
    return sorted(rows, key=lambda item: item["key"])


def _parse_keys_file(keys_file: Path) -> set[tuple[str, str, str]]:
    if not keys_file.is_file() or keys_file.is_symlink():
        raise ValueError(f"keys file missing, invalid, or symlinked: {keys_file}")
    selected: set[tuple[str, str, str]] = set()
    for line_no, raw in enumerate(keys_file.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise ValueError(f"blank keys-file line: {line_no}")
        parts = raw.split("/")
        if len(parts) != 3:
            raise ValueError(f"invalid keys-file key at line {line_no}: {raw!r}")
        key = tuple(_safe_component(part, label="keys-file") for part in parts)
        if key in selected:
            raise ValueError(f"duplicate key in keys file: {'/'.join(key)}")
        selected.add(key)
    if not selected:
        raise ValueError("keys file is empty")
    return selected


def _select_rows(rows: list[dict[str, Any]], keys_file: Path | None) -> list[dict[str, Any]]:
    if keys_file is None:
        return rows
    selected_keys = _parse_keys_file(keys_file)
    known = {item["key"] for item in rows}
    unknown = sorted(selected_keys - known)
    if unknown:
        raise ValueError(f"keys file contains key outside manifest: {'/'.join(unknown[0])}")
    return [item for item in rows if item["key"] in selected_keys]


def _config_for_game(run_root: Path, game: str) -> dict[str, Any]:
    return {
        "mode": "nvrasr_fallback",
        "reference_mode": "fallback",
        "data_dir": str(run_root / "inputs" / game),
        "output_dir": str(run_root / "staging" / game),
        "output_staging": True,
        "keep_16k_audio": True,
        "workspace": str(run_root / "workspaces" / game),
        "pad_silence": {"enabled": False},
        "ctc_prealign": {
            "enabled": True,
            "python": "/home/user/miniconda3/envs/asr/bin/python",
            "model_path": "/mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR",
            "device": "cuda:0",
            "all_gpus": True,
            "limit": 0,
            "timeout": 7200,
            "nvv_enabled": True,
            "allow_missing_reference": True,
        },
        "ctc_adjust": {"enabled": True, "limit": 0},
        "ctc_ready": {"allow_missing_reference": True},
        "mfa": {
            "num_jobs": 64,
            "single_speaker": True,
            "output_format": "long_textgrid",
            "clean": False,
            "fine_tune": False,
            "skip_validate": True,
            "beam": 20,
            "retry_beam": 80,
            "allow_partial": True,
            "min_output_ratio": 0.90,
        },
        "mfa_en": {
            "enabled": True,
            "strict_provenance": False,
            "num_jobs": 16,
            "beam": 10,
            "retry_beam": 40,
            "fine_tune": False,
        },
        "postprocess": {
            "strict_ok": True,
            "allow_filtered_integrity_failures": False,
            "merge_silence": True,
            "min_sil_merge_sec": 0.2,
            "fix_short_word": True,
            "short_word_max_sec": 0.25,
            "flank_silence_sec": 0.4,
            "detect_bgm": True,
            "filter_suspicious": True,
            "filter_min_phone_coverage": 0.35,
            "enable_text_correction": True,
            "handle_unexpected_sil": True,
            "workers": 0,
        },
    }


def _write_link_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno != getattr(os, "EXDEV", 18):
            raise
        shutil.copyfile(source, target)
        return "copy"


def materialize_nvv_rerun_inputs(
    *, manifest_path: Path, run_root: Path, keys_file: Path | None = None,
) -> dict[str, Any]:
    """Materialize a new private rerun root from a frozen JSONL manifest."""
    manifest_path = Path(manifest_path)
    run_root = Path(run_root)
    if run_root.exists() or run_root.is_symlink():
        raise ValueError(f"run root already exists or is symlinked: {run_root}")
    if not run_root.is_absolute():
        raise ValueError(f"run root must be absolute: {run_root}")
    if ".." in run_root.parts or _has_symlink_component(run_root.parent):
        raise ValueError(f"path escape or symlinked parent for run root: {run_root}")

    all_rows = _load_manifest(manifest_path)
    selected = _select_rows(all_rows, Path(keys_file) if keys_file is not None else None)
    if not selected:
        raise ValueError("selected manifest subset is empty")

    # All validation and all source hashing happened before this first write.
    run_root.mkdir(parents=True, exist_ok=False)
    configs: dict[str, dict[str, str]] = {}
    inputs: list[dict[str, Any]] = []
    games = sorted({item["key"][0] for item in selected})
    try:
        for game in games:
            config_path = run_root / "configs" / f"{game}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(_config_for_game(run_root, game), allow_unicode=True,
                               sort_keys=True), encoding="utf-8")
            configs[game] = {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            }
        for item in selected:
            game, speaker, stem = item["key"]
            target = run_root / "inputs" / game / f"{stem}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            method = _write_link_or_copy(item["source"], target)
            target_hash = _sha256(target)
            if target_hash != item["source_sha256"]:
                raise ValueError(f"materialized target hash mismatch: {target}")
            inputs.append({
                "game": game,
                "speaker": speaker,
                "stem": stem,
                "source_wav_path": str(item["source"]),
                "source_wav_sha256": item["source_sha256"],
                "target_wav_path": str(target),
                "target_sha256": target_hash,
                "method": method,
                "expected_nvv_sequence": item["expected_nvv_sequence"],
            })
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "manifest_count": len(all_rows),
            "selected_count": len(selected),
            "selected_subset_sha256": hashlib.sha256(
                _canonical_json([item["row"] for item in selected])).hexdigest(),
            "run_root": str(run_root),
            "stem_count_by_game": {
                game: sum(1 for item in selected if item["key"][0] == game)
                for game in games
            },
            "inputs": inputs,
            "configs": configs,
        }
        receipt_path = run_root / "materialization_receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))
        receipt["receipt_path"] = str(receipt_path)
        return receipt
    except Exception:
        # The caller receives an explicit error; a partial fresh root remains
        # visible for forensic inspection and cannot be reused due to new-root
        # creation rules.  Do not delete data behind an operator's back.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--keys-file", type=Path,
                        help="explicit game/speaker/stem subset, one key per line")
    args = parser.parse_args()
    receipt = materialize_nvv_rerun_inputs(
        manifest_path=args.manifest, run_root=args.run_root,
        keys_file=args.keys_file)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
