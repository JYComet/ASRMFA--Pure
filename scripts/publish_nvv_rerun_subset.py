#!/usr/bin/env python3
"""Publish a verified NVV-rerun subset into a fresh speaker-classified root.

The publisher is deliberately append-free: the destination must be absent or
empty, every listed input pair must exist exactly once, and only rows already
accepted by the NVV gate may enter the output.  A game is fully built and
re-verified in private staging before its TextGrid and GAMESL trees move into
the new public root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import soundfile as sf

from nvv_contract import NVV_TIER_NAMES, audit_nvv_contract
from finalize_gamedata_speakers import (
    detect_edge_silence_rms,
    normalize_edge_silence,
    retime_textgrid_to_audio,
)
from postprocess_textgrids import AXIS_EPS, parse_textgrid
from verify_nvv_rerun_manifest import verify_nvv_rerun_manifest


_SCHEMA = "nvv-rerun-fresh-publish-v1"
_SILENCE_SECONDS = 0.5
_FORBIDDEN_COMPONENTS = {"", ".", ".."}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_accepted_manifest(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"accepted manifest is missing, invalid, or symlinked: {path}")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank accepted manifest JSONL line: {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid accepted manifest JSONL at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"accepted manifest row must be object: line {line_number}")
        if row.get("status") not in (None, "accepted"):
            raise ValueError(f"accepted manifest row is not accepted: line {line_number}")
        if not isinstance(row.get("expected_nvv_sequence"), list):
            raise ValueError(f"missing expected_nvv_sequence: line {line_number}")
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate accepted manifest key: {key!r}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("accepted manifest is empty")
    return sorted(rows, key=_key)


def _safe_input_root(root: Path, *, label: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is missing, invalid, or symlinked: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} root contains symlink: {path}")
    return root.resolve()


def _paths_by_key(root: Path, *, suffix: str, label: str) -> dict[tuple[str, str, str], Path]:
    paths: dict[tuple[str, str, str], Path] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"invalid {label} source: {path}")
        relative = path.relative_to(root)
        if len(relative.parts) != 3 or path.suffix != suffix or not path.stem:
            raise ValueError(f"{label} source has invalid game/speaker/stem layout: {path}")
        game, speaker, _ = relative.parts
        key = (_safe_component(game, label="game"),
               _safe_component(speaker, label="speaker"),
               _safe_component(path.stem, label="stem"))
        if key in paths:
            raise ValueError(f"duplicate {label} source key: {key!r}")
        paths[key] = path
    return paths


def _validate_source_universe(
        rows: Sequence[Mapping[str, object]], *, textgrid_root: Path,
        wav_root: Path) -> tuple[dict[tuple[str, str, str], Path], dict[tuple[str, str, str], Path]]:
    expected = {_key(row) for row in rows}
    grids = _paths_by_key(textgrid_root, suffix=".TextGrid", label="TextGrid")
    wavs = _paths_by_key(wav_root, suffix=".wav", label="WAV")
    if set(grids) != expected or set(wavs) != expected:
        missing_grids = sorted(expected - set(grids))
        missing_wavs = sorted(expected - set(wavs))
        extra_grids = sorted(set(grids) - expected)
        extra_wavs = sorted(set(wavs) - expected)
        raise ValueError(
            "source universe mismatch: "
            f"missing_grids={missing_grids[:1]!r} missing_wavs={missing_wavs[:1]!r} "
            f"extra_grids={extra_grids[:1]!r} extra_wavs={extra_wavs[:1]!r}")
    return grids, wavs


def _validate_output_root(root: Path, *, protected: Sequence[Path]) -> Path:
    if root.is_symlink():
        raise ValueError(f"output root must not be symlinked: {root}")
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"output root must be a directory: {root}")
        if any(root.iterdir()):
            raise ValueError(f"output root must be empty: {root}")
    resolved = root.resolve()
    for input_root in protected:
        try:
            resolved.relative_to(input_root)
        except ValueError:
            continue
        raise ValueError(f"output root must not be inside an input: {root}")
    return resolved


def _require_close(left: float, right: float, *, label: str, tolerance: float = AXIS_EPS) -> None:
    if abs(left - right) > tolerance:
        raise ValueError(f"{label} mismatch: {left} != {right}")


def _validate_source_grid(grid_path: Path, *, duration: float) -> None:
    grid = parse_textgrid(grid_path)
    _require_close(float(grid.xmin), 0.0, label=f"TextGrid xmin {grid_path}")
    _require_close(float(grid.xmax), duration, label=f"TextGrid/WAV duration {grid_path}")
    names = {str(tier.name) for tier in grid.tiers}
    missing = set(NVV_TIER_NAMES) - names
    if missing:
        raise ValueError(f"TextGrid missing NVV tiers {sorted(missing)!r}: {grid_path}")
    for tier in grid.tiers:
        _require_close(float(tier.xmin), 0.0, label=f"tier xmin {tier.name}")
        _require_close(float(tier.xmax), duration, label=f"tier xmax {tier.name}")
        cursor = 0.0
        if not tier.intervals:
            raise ValueError(f"TextGrid tier has no intervals: {grid_path}:{tier.name}")
        for interval in tier.intervals:
            _require_close(float(interval.xmin), cursor, label=f"tier discontinuity {tier.name}")
            if float(interval.xmax) <= float(interval.xmin):
                raise ValueError(f"non-positive TextGrid interval: {grid_path}:{tier.name}")
            cursor = float(interval.xmax)
        _require_close(cursor, duration, label=f"tier final boundary {tier.name}")


def _normalize_wav_and_retime_grid(
        source_wav: Path, source_grid: Path, *, wav_target: Path,
        grid_target: Path) -> float:
    """Normalize edges and retime every grid tier to the final WAV axis."""
    source_duration = sf.info(str(source_wav)).duration
    if source_duration <= 0:
        raise ValueError(f"invalid empty WAV: {source_wav}")
    _validate_source_grid(source_grid, duration=source_duration)
    wav_target.parent.mkdir(parents=True, exist_ok=True)
    padding = normalize_edge_silence(
        source_wav, wav_target, target_silence_sec=_SILENCE_SECONDS)
    duration = float(padding["duration"])
    grid_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_grid, grid_target)
    retime_textgrid_to_audio(
        grid_target, time_offset=float(padding["time_offset"]), duration=duration)
    return duration


def _validate_final_pair(grid_path: Path, wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    if (info.format, info.subtype, info.channels) != ("WAV", "PCM_16", 1):
        raise ValueError(f"published WAV format mismatch: {wav_path}")
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"published WAV is empty: {wav_path}")
    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    # The shared detector reports the end of its first 10 ms speech-bearing
    # window.  Its project-wide normalization therefore has a one-window
    # measurement tolerance, especially for audio that begins at sample zero.
    tolerance = 0.011 + 2 / sample_rate
    if (abs(detect_edge_silence_rms(audio, sample_rate) - _SILENCE_SECONDS) > tolerance
            or abs(detect_edge_silence_rms(audio[::-1], sample_rate) - _SILENCE_SECONDS) > tolerance):
        raise ValueError(f"published WAV edge silence mismatch: {wav_path}")
    duration = len(audio) / sample_rate
    _validate_source_grid(grid_path, duration=duration)
    grid = parse_textgrid(grid_path)
    if audit_nvv_contract(grid):
        raise ValueError(f"published TextGrid NVV contract mismatch: {grid_path}")
    return duration


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _verify_staged_game(
        rows: Sequence[Mapping[str, object]], *, game: str, stage_root: Path) -> dict[str, int]:
    verification = stage_root / "verification" / game
    manifest = verification / "accepted.jsonl"
    _write_jsonl(manifest, rows)
    return verify_nvv_rerun_manifest(
        manifest_path=manifest, textgrid_root=stage_root / "textgrids",
        report_path=verification / "report.jsonl",
        accepted_manifest_path=verification / "accepted-final.jsonl",
        rejected_manifest_path=verification / "rejected-final.jsonl",
        accepted_stems_path=verification / "accepted-final.stems",
        rejected_stems_path=verification / "rejected-final.stems",
    )


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def publish_nvv_rerun_subset(
        *, accepted_manifest_path: Path, textgrid_root: Path, wav_root: Path,
        output_root: Path) -> dict[str, object]:
    """Fresh-publish only accepted rerun pairs and return an audit receipt.

    This function never modifies input roots and rejects a populated output
    root before it creates any destination content.
    """
    manifest = Path(accepted_manifest_path)
    rows = _read_accepted_manifest(manifest)
    grid_root = _safe_input_root(Path(textgrid_root), label="TextGrid")
    audio_root = _safe_input_root(Path(wav_root), label="WAV")
    output = _validate_output_root(
        Path(output_root), protected=(manifest.resolve(), grid_root, audio_root))
    grid_sources, audio_sources = _validate_source_universe(
        rows, textgrid_root=grid_root, wav_root=audio_root)

    by_game: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_game[_key(row)[0]].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.nvv-rerun-stage.", dir=output.parent))
    staged_games: dict[str, dict[str, object]] = {}
    try:
        for game, game_rows in sorted(by_game.items()):
            tg_stage = stage / "textgrids" / game
            wav_stage = stage / "GAMESL" / game
            pair_rows: list[dict[str, object]] = []
            for row in game_rows:
                key = _key(row)
                _, speaker, stem = key
                tg_target = tg_stage / speaker / f"{stem}.TextGrid"
                wav_target = wav_stage / speaker / f"{stem}.wav"
                padded_duration = _normalize_wav_and_retime_grid(
                    audio_sources[key], grid_sources[key], wav_target=wav_target,
                    grid_target=tg_target)
                duration = _validate_final_pair(tg_target, wav_target)
                _require_close(padded_duration, duration, label=f"padded duration {key!r}")
                pair_rows.append({
                    "speaker": speaker,
                    "stem": stem,
                    "textgrid_sha256": _sha256(tg_target),
                    "wav_sha256": _sha256(wav_target),
                    "duration_seconds": round(duration, 6),
                })
            verification = _verify_staged_game(game_rows, game=game, stage_root=stage)
            if verification != {"accepted": len(game_rows), "rejected": 0, "total": len(game_rows)}:
                raise ValueError(f"final NVV verification rejected {game}: {verification!r}")
            pair_rows.sort(key=lambda item: (str(item["speaker"]), str(item["stem"])))
            pair_digest = hashlib.sha256(json.dumps(
                pair_rows, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            staged_games[game] = {
                "pair_count": len(pair_rows),
                "duration_seconds": round(sum(float(item["duration_seconds"]) for item in pair_rows), 6),
                "pair_digest": pair_digest,
                "pairs": pair_rows,
                "final_nvv_verification": verification,
            }

        output.mkdir(exist_ok=True)
        gamesl_output = output / "GAMESL"
        gamesl_output.mkdir()
        for game in sorted(staged_games):
            tg_target = output / game
            wav_target = gamesl_output / game
            if tg_target.exists() or wav_target.exists():
                raise ValueError(f"refusing to overwrite published game: {game}")
            os.replace(stage / "textgrids" / game, tg_target)
            try:
                os.replace(stage / "GAMESL" / game, wav_target)
            except Exception:
                os.replace(tg_target, stage / "textgrids" / game)
                raise

        receipt: dict[str, object] = {
            "schema": _SCHEMA,
            "accepted_manifest": {"path": str(manifest.resolve()), "sha256": _sha256(manifest)},
            "textgrid_root": str(grid_root),
            "wav_root": str(audio_root),
            "output_root": str(output),
            "games": staged_games,
            "total_pair_count": sum(int(item["pair_count"]) for item in staged_games.values()),
            "total_duration_seconds": round(sum(float(item["duration_seconds"]) for item in staged_games.values()), 6),
        }
        _atomic_write_json(output / ".nvv_rerun_publish_receipt.json", receipt)
        return receipt
    except Exception:
        # Staging is intentionally retained for investigation; no failed game
        # has a partially written pair in the public destination.
        raise


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-manifest", required=True, type=Path)
    parser.add_argument("--textgrid-root", required=True, type=Path)
    parser.add_argument("--wav-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    try:
        receipt = publish_nvv_rerun_subset(
            accepted_manifest_path=args.accepted_manifest,
            textgrid_root=args.textgrid_root,
            wav_root=args.wav_root,
            output_root=args.output_root,
        )
    except ValueError as exc:
        _arguments().error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
