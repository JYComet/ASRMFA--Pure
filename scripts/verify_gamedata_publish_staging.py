#!/usr/bin/env python3
"""Read-only verifier for speaker-classified GAMEDATA publish staging.

An approval returned by this module binds an exact accepted stem universe to
an exact TextGrid/WAV pair tree.  The verifier never writes inside either
staged tree; callers may serialize the returned receipt elsewhere only after
all checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from finalize_gamedata_speakers import (
    _load_manifest,
    detect_edge_silence_rms,
    speaker_for_stem,
)
from postprocess_textgrids import parse_textgrid
from nvv_contract import (audit_nvv_contract, build_nvv_contract,
                          NVV_CONTRACT_SCHEMA, NVV_CONTRACT_REASON)
EXPECTED_TIERS = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")


def fingerprint_tree(root: Path) -> dict[str, object]:
    """Return a deterministic metadata fingerprint without reading payloads."""
    if root.is_symlink():
        raise ValueError(f"fingerprint root must not be a symlink: {root}")
    if not root.exists():
        return {
            "root": str(root), "exists": False, "files": 0, "dirs": 0,
            "symlinks": 0, "bytes": 0,
            "metadata_sha256": hashlib.sha256(b"").hexdigest(),
        }
    digest = hashlib.sha256()
    counts = Counter()
    total_bytes = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        directory_path = Path(directory)
        for name in list(dirnames) + filenames:
            path = directory_path / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                kind = "symlink"
                counts["symlinks"] += 1
                if name in dirnames:
                    dirnames.remove(name)
                extra = os.readlink(path)
            elif stat.S_ISDIR(info.st_mode):
                kind = "dir"
                counts["dirs"] += 1
                extra = ""
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
                counts["files"] += 1
                total_bytes += info.st_size
                extra = ""
            else:
                kind = "other"
                counts["other"] += 1
                extra = ""
            record = {
                "kind": kind, "mtime_ns": info.st_mtime_ns,
                "path": relative, "size": info.st_size, "target": extra,
            }
            digest.update(json.dumps(
                record, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    if counts["other"]:
        raise ValueError(f"tree contains unsupported filesystem entries: {root}")
    return {
        "root": str(root), "exists": True, "files": counts["files"],
        "dirs": counts["dirs"], "symlinks": counts["symlinks"],
        "bytes": total_bytes, "metadata_sha256": digest.hexdigest(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_detect_verification_edge_silence = detect_edge_silence_rms


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"staging root is missing, invalid, or symlinked: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staging contains symlink: {path}")


def _pair_paths(root: Path, suffix: str) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        relative = path.relative_to(root).with_suffix("")
        if relative in result:
            raise ValueError(f"duplicate staged relative stem: {relative}")
        result[relative] = path
    return result


def _accepted_stems(root: Path) -> set[str]:
    stems: set[str] = set()
    for path in sorted(root.rglob("*.TextGrid")):
        if path.stem in stems:
            raise ValueError(f"duplicate accepted stem: {path.stem}")
        stems.add(path.stem)
    if not stems:
        raise ValueError(f"accepted root has no TextGrids: {root}")
    return stems


def _verify_grid(path: Path, duration: float, *, axis_tolerance: float) -> str:
    grid = parse_textgrid(path)
    names = tuple(tier.name for tier in grid.tiers)
    if names != EXPECTED_TIERS:
        raise ValueError(
            f"unexpected TextGrid tiers: {path}: {names!r}")
    if not math.isclose(grid.xmin, 0.0, abs_tol=axis_tolerance):
        raise ValueError(f"TextGrid xmin is not zero: {path}: {grid.xmin}")
    if not math.isclose(grid.xmax, duration, abs_tol=axis_tolerance):
        raise ValueError(
            f"TextGrid duration mismatch: {path}: grid={grid.xmax}, wav={duration}")
    for tier in grid.tiers:
        if (not math.isclose(tier.xmin, 0.0, abs_tol=axis_tolerance)
                or not math.isclose(tier.xmax, duration,
                                    abs_tol=axis_tolerance)):
            raise ValueError(f"TextGrid tier domain mismatch: {path}: {tier.name}")
        if not tier.intervals:
            raise ValueError(f"empty TextGrid tier: {path}: {tier.name}")
        cursor = 0.0
        for interval in tier.intervals:
            if interval.xmin < -axis_tolerance or interval.xmax > duration + axis_tolerance:
                raise ValueError(f"TextGrid interval out of bounds: {path}: {tier.name}")
            if interval.xmax <= interval.xmin:
                raise ValueError(f"non-positive TextGrid interval: {path}: {tier.name}")
            if not math.isclose(interval.xmin, cursor, abs_tol=axis_tolerance):
                raise ValueError(f"TextGrid interval gap/overlap: {path}: {tier.name}")
            cursor = interval.xmax
        if not math.isclose(cursor, duration, abs_tol=axis_tolerance):
            raise ValueError(f"TextGrid tier does not cover WAV: {path}: {tier.name}")
    return _sha256(path)


def _verify_pair(
    item: tuple[Path, Path, Path], *, target_silence_sec: float,
    silence_threshold: float, frame_length: int,
) -> dict[str, object]:
    relative, grid_path, wav_path = item
    info = sf.info(wav_path)
    if (info.frames <= 0 or info.samplerate <= 0 or info.channels != 1
            or info.subtype != "PCM_16" or info.format != "WAV"):
        raise ValueError(
            f"invalid WAV format: {wav_path}: frames={info.frames}, "
            f"sr={info.samplerate}, channels={info.channels}, "
            f"format={info.format}, subtype={info.subtype}")
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"invalid WAV samples: {wav_path}")
    if not np.any(np.abs(audio) >= silence_threshold):
        raise ValueError(f"WAV contains no detected speech: {wav_path}")
    head = _detect_verification_edge_silence(
        audio, sample_rate, silence_threshold=silence_threshold)
    tail = _detect_verification_edge_silence(
        audio[::-1], sample_rate, silence_threshold=silence_threshold)
    silence_tolerance = max(frame_length / sample_rate, 0.03)
    if not math.isclose(head, target_silence_sec, abs_tol=silence_tolerance):
        raise ValueError(
            f"head silence mismatch: {wav_path}: {head:.6f}s")
    if not math.isclose(tail, target_silence_sec, abs_tol=silence_tolerance):
        raise ValueError(
            f"tail silence mismatch: {wav_path}: {tail:.6f}s")
    duration = info.frames / info.samplerate
    axis_tolerance = max(1.0 / info.samplerate, 1e-5)
    grid_digest = _verify_grid(
        grid_path, duration, axis_tolerance=axis_tolerance)
    grid = parse_textgrid(grid_path)
    nvv_contract = build_nvv_contract(grid)
    if audit_nvv_contract(grid):
        raise ValueError(f"{NVV_CONTRACT_REASON}: {grid_path}")
    return {
        "relative": relative.as_posix(),
        "duration": duration,
        "grid_bytes": grid_path.stat().st_size,
        "wav_bytes": wav_path.stat().st_size,
        "grid_sha256": grid_digest,
        "wav_sha256": _sha256(wav_path),
        "head_silence": head,
        "tail_silence": tail,
        "nvv_contract": nvv_contract,
    }


def verify_game_staging(
    *, game: str, run_id: str, accepted_root: Path,
    manifest_path: Path, aligned_root: Path, gamesl_root: Path,
    workers: int = 16, target_silence_sec: float = 0.5,
    silence_threshold: float = 0.001, frame_length: int = 1024,
) -> dict[str, object]:
    """Verify one complete staged game and return its approval payload."""
    for root in (accepted_root, aligned_root, gamesl_root):
        _reject_symlinks(root)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing or symlinked manifest: {manifest_path}")

    unexpected_aligned = [
        path for path in aligned_root.rglob("*") if path.is_file()
        and path.suffix != ".TextGrid"
        and path.name != ".speaker_classification_receipt.json"]
    unexpected_gamesl = [
        path for path in gamesl_root.rglob("*") if path.is_file()
        and path.suffix.lower() != ".wav"]
    if unexpected_aligned or unexpected_gamesl:
        raise ValueError(
            f"unexpected staged files: aligned={unexpected_aligned[:3]}, "
            f"gamesl={unexpected_gamesl[:3]}")

    grids = _pair_paths(aligned_root, ".TextGrid")
    wavs = _pair_paths(gamesl_root, ".wav")
    if set(grids) != set(wavs):
        only_grids = sorted(set(grids) - set(wavs))[:5]
        only_wavs = sorted(set(wavs) - set(grids))[:5]
        raise ValueError(
            f"TextGrid/WAV relative path mismatch: "
            f"only_grids={only_grids}, only_wavs={only_wavs}")

    accepted_stems = _accepted_stems(accepted_root)
    staged_stems = {relative.name for relative in grids}
    if staged_stems != accepted_stems or len(staged_stems) != len(grids):
        raise ValueError(
            f"accepted/staged stem mismatch: accepted={len(accepted_stems)}, "
            f"staged={len(staged_stems)}, pairs={len(grids)}")

    manifest = _load_manifest(manifest_path)
    speaker_counts: Counter[str] = Counter()
    for relative in grids:
        expected = speaker_for_stem(manifest, relative.name)
        actual = relative.parent.as_posix()
        if actual != expected:
            raise ValueError(
                f"speaker path mismatch: {relative}: expected={expected}, "
                f"actual={actual}")
        speaker_counts[actual] += 1

    items = [(relative, grids[relative], wavs[relative])
             for relative in sorted(grids)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        records = list(executor.map(
            lambda item: _verify_pair(
                item, target_silence_sec=target_silence_sec,
                silence_threshold=silence_threshold,
                frame_length=frame_length),
            items))

    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    duration_seconds = sum(float(record["duration"]) for record in records)
    contract_rows = [{"relative": record["relative"],
                      "digest": record["nvv_contract"]["digest"]}
                     for record in records]
    contract_digest = hashlib.sha256(json.dumps(
        contract_rows, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema": "gamedata-staging-approval-v2",
        "status": "STAGING_APPROVED",
        "game": game,
        "run_id": run_id,
        "accepted_root": str(accepted_root),
        "manifest_path": str(manifest_path),
        "staged_aligned_root": str(aligned_root),
        "staged_gamesl_root": str(gamesl_root),
        "pair_count": len(records),
        "speaker_count": len(speaker_counts),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "tier_names": list(EXPECTED_TIERS),
        "wav_contract": "WAV/PCM_16/mono",
        "target_edge_silence_seconds": target_silence_sec,
        "duration_seconds": duration_seconds,
        "duration_hours": duration_seconds / 3600.0,
        "pair_digest": digest.hexdigest(),
        "nvv_contract": {
            "schema": NVV_CONTRACT_SCHEMA,
            "status": "verified",
            "pair_contracts": contract_rows,
            "digest": contract_digest,
        },
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--gamesl-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    receipt = verify_game_staging(
        game=args.game, run_id=args.run_id,
        accepted_root=args.accepted_root, manifest_path=args.manifest,
        aligned_root=args.aligned_root, gamesl_root=args.gamesl_root,
        workers=args.workers)
    if args.receipt:
        _atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
