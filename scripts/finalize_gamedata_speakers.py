#!/usr/bin/env python3
"""Finalize GAMEDATA TextGrids and padded WAVs by original speaker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from postprocess_textgrids import Interval, parse_textgrid, write_textgrid
PUBLISHED_ROOT = Path("/mnt/Raw/GAMEDATA_对齐_20260903")
GAMESL_ROOT = Path("/mnt/Raw/GAMESL")
STAGE_ROOT = Path("/mnt/nvme3/gamedata_20260903")


@dataclass(frozen=True)
class GameSpec:
    game: str
    accepted_root: Path
    published_root: Path
    manifest_path: Path
    padded_audio_root: Path | None
    gamesl_root: Path
    expected_count: int | None = None


def dynamic_game_spec(
    task_config: Path | str,
    game: str,
    accepted_root: Path,
    *,
    manifest_path: Path | None = None,
    padded_audio_root: Path | None = None,
) -> GameSpec:
    """Build a finalizer spec from the incremental rebuild task YAML.

    Importing here keeps the legacy production spec table independent from
    the new preparation script.  ``accepted_root`` is supplied by the caller
    after the alignment pipeline has completed; this function never runs it.
    """
    from rebuild_gamedata import load_config

    config = load_config(Path(task_config))
    selected = next((item for item in config.games if item.codename == game), None)
    if selected is None:
        raise ValueError(f"unknown game in rebuild task: {game}")
    manifest = manifest_path or (config.staging_root / game / ".rebuild_manifest.json")
    return GameSpec(
        game=game,
        accepted_root=Path(accepted_root),
        published_root=config.output_root / game,
        manifest_path=manifest,
        padded_audio_root=padded_audio_root,
        gamesl_root=config.gamesl_root,
    )


@dataclass(frozen=True)
class PlanEntry:
    stem: str
    speaker: str
    grid_source: Path
    grid_target: Path
    audio_source: Path
    audio_target: Path
    generate_padding: bool


def _safe_speaker(value: str | None) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        return "default"
    return value


def speaker_for_stem(manifest: dict, stem: str) -> str:
    row = manifest.get(stem)
    if not isinstance(row, dict):
        return "default"
    audio = row.get("audio")
    if not isinstance(audio, str) or not audio:
        return "default"
    parent = Path(audio).parent
    if parent == Path("."):
        return "default"
    return _safe_speaker(parent.name)


def detect_edge_silence_rms(
    audio: np.ndarray, sample_rate: int, *, silence_threshold: float = 0.001,
    window_seconds: float = 0.01,
) -> float:
    """Return edge silence using a shift-invariant sliding RMS window.

    Reverse ``audio`` before calling to measure the trailing edge.  Reporting
    the end of the first speech-bearing window makes the estimated boundary
    stable when the audio is shifted during normalization.
    """
    window = max(1, round(window_seconds * sample_rate))
    if len(audio) < window:
        return 0.0
    squared = np.asarray(audio, dtype=np.float64) ** 2
    cumulative = np.empty(len(squared) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(squared, out=cumulative[1:])
    means = (cumulative[window:] - cumulative[:-window]) / window
    speech = np.flatnonzero(means >= silence_threshold ** 2)
    if not len(speech):
        return len(audio) / sample_rate
    return min(len(audio), int(speech[0]) + window) / sample_rate


def normalize_edge_silence(
    source: Path,
    target: Path,
    *,
    target_silence_sec: float = 0.5,
    silence_threshold: float = 0.001,
    frame_length: int = 1024,
) -> dict[str, float | str]:
    """Write a WAV with normalized edge silence and return the head offset."""
    audio, sr = sf.read(str(source))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)
    head_sil = detect_edge_silence_rms(
        audio, sr, silence_threshold=silence_threshold)
    target_samples = int(target_silence_sec * sr)
    head_samples = round(head_sil * sr)
    if head_sil > target_silence_sec + 0.001:
        audio = audio[head_samples - target_samples:]
        time_offset = -head_sil + target_silence_sec
    elif head_sil < target_silence_sec - 0.001:
        audio = np.concatenate([
            np.zeros(target_samples - head_samples, dtype=np.float32), audio])
        time_offset = target_silence_sec - head_sil
    else:
        time_offset = 0.0

    tail_sil = detect_edge_silence_rms(
        audio[::-1], sr, silence_threshold=silence_threshold)
    tail_samples = round(tail_sil * sr)
    if tail_sil > target_silence_sec + 0.001:
        audio = audio[:max(0, len(audio) - (tail_samples - target_samples))]
    elif tail_sil < target_silence_sec - 0.001:
        audio = np.concatenate([
            audio, np.zeros(target_samples - tail_samples, dtype=np.float32)])

    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, sr, subtype="PCM_16")
    return {
        "source": str(source),
        "target": str(target),
        "time_offset": round(time_offset, 6),
        "sample_rate": int(sr),
        "duration": round(len(audio) / sr, 6),
    }


def retime_textgrid_to_audio(
        path: Path, *, time_offset: float, duration: float) -> None:
    """Map every interval onto the padded WAV's complete [0, duration] axis."""
    if duration <= 0:
        raise ValueError("padded audio duration must be positive")
    grid = parse_textgrid(path)
    epsilon = 1e-7
    for tier in grid.tiers:
        remapped: list[Interval] = []
        cursor = 0.0
        for interval in tier.intervals:
            start = max(0.0, min(duration, interval.xmin + time_offset))
            end = max(0.0, min(duration, interval.xmax + time_offset))
            start = max(cursor, start)
            if start > cursor + epsilon:
                remapped.append(Interval(cursor, start, ""))
            if end > start + epsilon:
                remapped.append(Interval(start, end, interval.text))
                cursor = end
        if cursor < duration - epsilon:
            remapped.append(Interval(cursor, duration, ""))
        if not remapped:
            remapped = [Interval(0.0, duration, "")]
        remapped[0].xmin = 0.0
        remapped[-1].xmax = duration
        tier.xmin = 0.0
        tier.xmax = duration
        tier.intervals = remapped
    grid.xmin = 0.0
    grid.xmax = duration
    write_textgrid(grid, path)


def _load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    # The incremental rebuild writer uses an ordered ``items`` list so its
    # collision decisions are auditable.  Normalize it to the legacy lookup
    # shape while retaining source (not staged) audio for speaker ownership.
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        normalized = {}
        for item in value["items"]:
            if not isinstance(item, dict) or not isinstance(item.get("stem"), str):
                raise ValueError(f"invalid rebuild manifest item: {path}")
            normalized[item["stem"]] = {
                "audio": item.get("source") or item.get("audio"),
                "txt": item.get("reference_text"),
            }
        return normalized
    if not isinstance(value, dict):
        raise ValueError(f"stage manifest must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_content(left: Path, right: Path) -> bool:
    return (left.stat().st_size == right.stat().st_size
            and _sha256(left) == _sha256(right))


def build_game_plan(spec: GameSpec) -> list[PlanEntry]:
    manifest = _load_manifest(spec.manifest_path)
    scan_root = spec.accepted_root
    receipt_path = spec.published_root / ".speaker_classification_receipt.json"
    if receipt_path.is_file() and spec.accepted_root != spec.published_root:
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        published_grids = sorted(spec.published_root.rglob("*.TextGrid"))
        if (prior.get("schema") == "gamedata-speaker-classification-v1"
                and prior.get("accepted_source") == str(spec.accepted_root)
                and prior.get("accepted_count") == len(published_grids)):
            scan_root = spec.published_root
    grids = sorted(scan_root.rglob("*.TextGrid"))
    if not grids:
        raise ValueError(f"no accepted TextGrids: {spec.accepted_root}")
    seen_stems: set[str] = set()
    seen_targets: set[Path] = set()
    plan: list[PlanEntry] = []
    for grid in grids:
        stem = grid.stem
        if stem in seen_stems:
            raise ValueError(f"duplicate accepted stem: {stem}")
        seen_stems.add(stem)
        speaker = speaker_for_stem(manifest, stem)
        grid_target = spec.published_root / speaker / grid.name
        if grid_target in seen_targets:
            raise ValueError(f"duplicate destination: {grid_target}")
        seen_targets.add(grid_target)
        row = manifest.get(stem)
        original_audio = (Path(row["audio"])
                          if isinstance(row, dict)
                          and isinstance(row.get("audio"), str) else None)
        padded = (spec.padded_audio_root / f"{stem}.wav"
                  if spec.padded_audio_root is not None else None)
        staged_audio = spec.manifest_path.parent / f"{stem}.wav"
        if padded is not None and padded.is_file():
            audio_source = padded
            generate_padding = False
        elif staged_audio.is_file():
            audio_source = staged_audio
            generate_padding = True
        elif original_audio is not None and original_audio.is_file():
            audio_source = original_audio
            generate_padding = True
        else:
            raise FileNotFoundError(f"missing audio for accepted stem: {stem}")
        audio_target = spec.gamesl_root / spec.game / speaker / f"{stem}.wav"
        if grid_target.exists() and grid_target != grid and not _same_content(
                grid, grid_target):
            raise FileExistsError(f"conflicting destination: {grid_target}")
        plan.append(PlanEntry(
            stem, speaker, grid, grid_target, audio_source, audio_target,
            generate_padding))
    if spec.expected_count is not None and len(plan) != spec.expected_count:
        raise ValueError(
            f"accepted count mismatch for {spec.game}: "
            f"expected={spec.expected_count}, actual={len(plan)}")
    return plan


def _temporary_path(target: Path) -> Path:
    return target.with_name(
        f".{target.stem}.tmp.{os.getpid()}.{threading.get_ident()}"
        f"{target.suffix}")


def _finalize_entry(entry: PlanEntry) -> dict[str, object]:
    entry.grid_target.parent.mkdir(parents=True, exist_ok=True)
    entry.audio_target.parent.mkdir(parents=True, exist_ok=True)
    if entry.grid_source == entry.grid_target and entry.audio_target.is_file():
        return {"generated": entry.generate_padding, "reused": True}

    audio_tmp = _temporary_path(entry.audio_target)
    grid_tmp = _temporary_path(entry.grid_target)
    try:
        if not entry.audio_target.exists():
            if entry.generate_padding:
                padding = normalize_edge_silence(entry.audio_source, audio_tmp)
            else:
                shutil.copy2(entry.audio_source, audio_tmp)
                padding = {"time_offset": 0.0}
            os.replace(audio_tmp, entry.audio_target)
        else:
            padding = {"time_offset": 0.0}

        if entry.grid_source != entry.grid_target:
            if not entry.grid_target.exists():
                shutil.copy2(entry.grid_source, grid_tmp)
                if entry.generate_padding:
                    retime_textgrid_to_audio(
                        grid_tmp,
                        time_offset=float(padding["time_offset"]),
                        duration=float(padding["duration"]),
                    )
                os.replace(grid_tmp, entry.grid_target)
            elif not _same_content(entry.grid_source, entry.grid_target):
                raise FileExistsError(
                    f"conflicting destination: {entry.grid_target}")
            if (entry.grid_source.parent == entry.grid_target.parent.parent
                    and entry.grid_source.is_file()):
                entry.grid_source.unlink()
        return {"generated": entry.generate_padding, "reused": False}
    finally:
        audio_tmp.unlink(missing_ok=True)
        grid_tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def finalize_game(spec: GameSpec, *, workers: int = 16) -> dict:
    plan = build_game_plan(spec)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(executor.map(_finalize_entry, plan))
    speaker_counts = dict(sorted(Counter(
        entry.speaker for entry in plan).items()))
    receipt = {
        "schema": "gamedata-speaker-classification-v1",
        "game": spec.game,
        "accepted_count": len(plan),
        "speaker_count": len(speaker_counts),
        "speaker_counts": speaker_counts,
        "generated_padded_count": sum(
            bool(result["generated"]) for result in results),
        "reused_count": sum(bool(result["reused"]) for result in results),
        "accepted_source": str(spec.accepted_root),
        "published_root": str(spec.published_root),
        "gamesl_root": str(spec.gamesl_root / spec.game),
        "manifest": str(spec.manifest_path),
        "unresolved_speaker_count": speaker_counts.get("default", 0),
    }
    _atomic_json(
        spec.published_root / ".speaker_classification_receipt.json", receipt)
    return receipt


def production_specs() -> dict[str, GameSpec]:
    accepted = {
        "snowbreak": PUBLISHED_ROOT / "snowbreak",
        "huanxing": PUBLISHED_ROOT / "huanxing",
        "baijing": PUBLISHED_ROOT / "baijing",
        "zhongmodi": PUBLISHED_ROOT / "zhongmodi",
        "yihuan": PUBLISHED_ROOT / "yihuan",
        "zzz": PUBLISHED_ROOT / "zzz",
        "genshin": Path("/mnt/nvme3/mfa_work_gamedata_genshin_20260903/"
                        "output_staging/recover_20260908T1420Z"),
        "reverse1999": Path("/mnt/nvme3/mfa_work_gamedata_reverse1999_20260903/"
                            "output_staging/20260904T065249Z_3677774_3677774"),
    }
    padded = {
        game: Path(f"/mnt/nvme3/mfa_work_gamedata_{game}_20260903/padded_audio")
        for game in accepted
    }
    expected = {
        "snowbreak": 1221, "huanxing": 1910, "baijing": 1421,
        "zhongmodi": 7815, "yihuan": 4894, "zzz": 35911,
        "genshin": 86782, "reverse1999": 11923,
    }
    return {
        game: GameSpec(
            game=game,
            accepted_root=root,
            published_root=PUBLISHED_ROOT / game,
            manifest_path=STAGE_ROOT / game / ".stage_manifest.json",
            padded_audio_root=padded[game],
            gamesl_root=GAMESL_ROOT,
            expected_count=expected[game],
        )
        for game, root in accepted.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default=",".join(production_specs()))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = production_specs()
    games = [game.strip() for game in args.games.split(",") if game.strip()]
    unknown = sorted(set(games) - set(specs))
    if unknown:
        parser.error(f"unknown games: {', '.join(unknown)}")
    total = 0
    for game in games:
        spec = specs[game]
        if args.dry_run:
            plan = build_game_plan(spec)
            print(f"{game}: planned={len(plan)}")
            total += len(plan)
        else:
            receipt = finalize_game(spec, workers=args.workers)
            print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
            total += int(receipt["accepted_count"])
    print(f"TOTAL accepted={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
