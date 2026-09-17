#!/usr/bin/env python3
"""Prepare incremental GAMEDATA rebuilds without executing alignment jobs.

The module deliberately stops at a journaled, reproducible data hand-off.  A
caller may use the generated per-game pipeline configurations to invoke the
alignment runner separately; this script never starts that runner itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import soundfile as sf

try:
    import yaml
except ImportError:  # pragma: no cover - the repository already depends on PyYAML
    yaml = None


SOURCE_ROOT = Path("/mnt/Raw/GAMEDATA")
OUTPUT_ROOT = Path("/mnt/Raw/GAMEDATA_对齐_20260903")
GAMESL_ROOT = Path("/mnt/Raw/GAMESL")
STAGING_ROOT = Path("/mnt/nvme3/gamedata_rebuild_20260909")
ARCHIVE_ROOT = STAGING_ROOT / "archive"
AUDIO_EXTENSIONS = frozenset({".wav", ".ogg", ".mp3", ".flac", ".m4a", ".aac", ".opus", ".wma"})


@dataclass(frozen=True)
class GameInput:
    codename: str
    source_dir: Path
    source_name: str
    reference_mode: str = "auto"


@dataclass(frozen=True)
class InventoryRow:
    stem: str
    source: Path
    reference_text_path: Path | None
    reference_text: str | None
    text_mode: str
    collision_index: int = 0


@dataclass(frozen=True)
class RebuildConfig:
    source_root: Path
    staging_root: Path
    archive_root: Path
    output_root: Path
    gamesl_root: Path
    games: tuple[GameInput, ...]
    ffmpeg: str = "ffmpeg"
    allow_test_roots: bool = False
    publish_staging_root: Path = Path("/mnt/Raw/.gamedata_publish_staging")
    publish_archive_root: Path = Path("/mnt/Raw/.gamedata_publish_archive")
    split_by_text_mode: bool = True


def _path(value: str | os.PathLike[str]) -> Path:
    raw = str(value).replace("\\", "/")
    # The NAS spelling is useful in a checked-in config while execution takes
    # place on the Linux mount used by this host.
    prefixes = (
        ("//RS3621/Research_TTS/Data/Raw", "/mnt/Raw"),
        ("/mnt/nas/Research_TTS/Data/Raw", "/mnt/Raw"),
    )
    for prefix, local in prefixes:
        if raw == prefix or raw.startswith(prefix + "/"):
            raw = local + raw[len(prefix):]
            break
    return Path(raw)


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def validate_config(config: RebuildConfig) -> None:
    """Fail closed on unsafe roots, traversal, duplicate codenames, or escapes."""
    if not config.source_root.is_absolute():
        raise ValueError("source_root must be absolute")
    if not config.split_by_text_mode:
        raise ValueError("split_by_text_mode must be enabled for mixed inventories")
    if not config.allow_test_roots:
        exact = (("output_root", config.output_root, OUTPUT_ROOT),
                 ("gamesl_root", config.gamesl_root, GAMESL_ROOT))
        for label, actual, expected in exact:
            if actual.resolve(strict=False) != expected.resolve(strict=False):
                raise ValueError(f"{label} must be the exact target root {expected}")
        if config.source_root.resolve(strict=False) != SOURCE_ROOT.resolve(strict=False):
            raise ValueError(f"source_root must be the exact target root {SOURCE_ROOT}")
        if not _within(config.staging_root, Path("/mnt/nvme3")):
            raise ValueError("staging_root must be under /mnt/nvme3")
        if not (_within(config.archive_root, Path("/mnt/nvme3"))
                or _within(config.archive_root, Path("/mnt/Raw/.gamedata_publish_archive"))):
            raise ValueError("archive_root must be under /mnt/nvme3 or NAS publish archive")
        if not _within(config.publish_staging_root, Path("/mnt/Raw/.gamedata_publish_staging")):
            raise ValueError("publish_staging_root must be under NAS publish staging")
        if not _within(config.publish_archive_root, Path("/mnt/Raw/.gamedata_publish_archive")):
            raise ValueError("publish_archive_root must be under NAS publish archive")
    for root_name, root in (("staging_root", config.staging_root),
                            ("archive_root", config.archive_root),
                            ("output_root", config.output_root),
                            ("gamesl_root", config.gamesl_root),
                            ("publish_staging_root", config.publish_staging_root),
                            ("publish_archive_root", config.publish_archive_root)):
        if not root.is_absolute():
            raise ValueError(f"{root_name} must be absolute")
    names: set[str] = set()
    for game in config.games:
        if not game.codename or game.codename in {".", ".."} or Path(game.codename).name != game.codename:
            raise ValueError(f"unsafe game codename: {game.codename!r}")
        if game.codename in names:
            raise ValueError(f"duplicate game codename: {game.codename}")
        names.add(game.codename)
        if not _within(game.source_dir, config.source_root):
            raise ValueError(f"game source escapes source_root: {game.source_dir}")


def build_inventory(source_dir: Path) -> list[InventoryRow]:
    """Recursively index supported audio and same-directory reference text.

    A non-empty sibling ``stem.txt`` wins over ASR.  Duplicate stems are
    assigned deterministic ``__dupNN`` suffixes after sorting by source path.
    """
    # Walk the NAS tree once.  Looking up sibling TXT with ``iterdir`` for
    # every audio file turns an 80k-file inventory into tens of thousands of
    # network directory RPCs and can stall for hours.
    walked: list[Path] = []
    for directory, _dirnames, filenames in os.walk(source_dir):
        walked.extend(Path(directory) / name for name in filenames)
    files = sorted(walked, key=lambda p: p.as_posix().casefold())
    audio_files = [p for p in files if p.suffix.lower() in AUDIO_EXTENSIONS]
    refs_by_parent_stem: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in files:
        if path.suffix.lower() == ".txt":
            refs_by_parent_stem[(path.parent.as_posix().casefold(),
                                 path.stem.casefold())].append(path)
    rows: list[InventoryRow] = []
    collisions: dict[str, int] = defaultdict(int)
    reserved_stems = {p.stem for p in audio_files}
    assigned_stems: set[str] = set()
    for source in audio_files:
        base = source.stem
        collision_key = base.casefold()
        index = collisions[collision_key]
        collisions[collision_key] += 1
        if index == 0 and base not in assigned_stems:
            stem = base
        else:
            suffix = max(1, index)
            stem = f"{base}__dup{suffix:02d}"
            while stem in reserved_stems or stem in assigned_stems:
                suffix += 1
                stem = f"{base}__dup{suffix:02d}"
        assigned_stems.add(stem)
        candidates = refs_by_parent_stem.get(
            (source.parent.as_posix().casefold(), source.stem.casefold()), [])
        text_path = next((p for p in candidates if p.read_text(encoding="utf-8", errors="replace").strip()), None)
        text = text_path.read_text(encoding="utf-8") if text_path else None
        rows.append(InventoryRow(stem, source, text_path, text,
                                 "reference" if text is not None else "asr", index))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_audio_to_wav(source: Path, target: Path, *, ffmpeg: str = "ffmpeg") -> None:
    """Convert any supported input into mono PCM16 WAV atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        try:
            audio, sample_rate = sf.read(str(source), always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio[:, 0]
            sf.write(str(temporary), audio, sample_rate, subtype="PCM_16", format="WAV")
        except Exception:
            command = [ffmpeg, "-y", "-v", "error", "-i", str(source),
                       "-ac", "1", "-c:a", "pcm_s16le", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode != 0 or not temporary.is_file():
                raise RuntimeError(f"audio conversion failed for {source}: {result.stderr[-500:]}")
        info = sf.info(str(temporary))
        if info.channels != 1 or info.subtype != "PCM_16":
            raise RuntimeError(f"conversion did not produce mono PCM16 WAV: {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def build_cohorts(rows: Sequence[InventoryRow]) -> dict[str, list[InventoryRow]]:
    """Partition the frozen inventory into mutually exclusive text cohorts."""
    cohorts: dict[str, list[InventoryRow]] = {"reference": [], "asr": []}
    seen: set[str] = set()
    for row in rows:
        if row.stem in seen:
            raise ValueError(f"duplicate cohort stem: {row.stem}")
        seen.add(row.stem)
        if row.text_mode not in cohorts:
            raise ValueError(f"unknown inventory text mode: {row.text_mode}")
        cohorts[row.text_mode].append(row)
    for values in cohorts.values():
        values.sort(key=lambda item: item.stem)
    if {row.stem for values in cohorts.values() for row in values} != seen:
        raise ValueError("cohort partition does not conserve inventory")
    return cohorts


def _append_journal(path: Path, event: str, game: GameInput, **details: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "gamedata-rebuild-journal-v1", "event": event,
               "game": game.codename, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stage_game(config: RebuildConfig, game: GameInput, *, journal_path: Path) -> Path:
    validate_config(config)
    rows = build_inventory(game.source_dir)
    destination = config.staging_root / game.codename
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"staging destination is non-empty: {destination}")
    config.staging_root.mkdir(parents=True, exist_ok=True)
    temporary = config.staging_root / f".{game.codename}.stage.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"staging transaction collision: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    manifest = []
    try:
        for row in rows:
            output = temporary / f"{row.stem}.wav"
            convert_audio_to_wav(row.source, output, ffmpeg=config.ffmpeg)
            if row.reference_text is not None:
                (temporary / f"{row.stem}.txt").write_text(row.reference_text, encoding="utf-8")
            manifest.append({
                "stem": row.stem, "audio": str(destination / f"{row.stem}.wav"), "source": str(row.source),
                "reference_text": str(destination / f"{row.stem}.txt") if row.reference_text is not None else None,
                "text_mode": row.text_mode, "sha256": _sha256(output),
            })
        # Each cohort is a self-contained data root.  In particular, the
        # fallback root deliberately contains no TXT, so stale authority text
        # cannot turn a no-reference run into a mixed batch.
        cohorts = build_cohorts(rows)
        for mode, cohort_rows in cohorts.items():
            if not cohort_rows:
                continue
            cohort_root = temporary / "cohorts" / mode
            cohort_root.mkdir(parents=True, exist_ok=True)
            cohort_items = []
            for row in cohort_rows:
                shutil.copy2(temporary / f"{row.stem}.wav",
                             cohort_root / f"{row.stem}.wav")
                reference_path = None
                if mode == "reference":
                    reference_path = cohort_root / f"{row.stem}.txt"
                    reference_path.write_text(row.reference_text or "", encoding="utf-8")
                cohort_items.append({
                    "stem": row.stem,
                    "audio": str(cohort_root / f"{row.stem}.wav"),
                    "source": str(row.source),
                    "reference_text": str(reference_path) if reference_path else None,
                    "text_mode": mode,
                    "sha256": _sha256(cohort_root / f"{row.stem}.wav"),
                })
            _write_json(cohort_root / ".rebuild_manifest.json", {
                "schema": "gamedata-rebuild-cohort-v1", "game": game.codename,
                "text_mode": mode, "items": cohort_items,
            })
        _write_json(temporary / ".rebuild_manifest.json", {"schema": "gamedata-rebuild-manifest-v1", "game": game.codename, "items": manifest})
        _write_json(temporary / ".pipeline_configs.json", {"configs": resolve_pipeline_configs(config, game, sorted({row.text_mode for row in rows}))})
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _append_journal(journal_path, "stage", game, destination=str(destination), count=len(rows))
    return destination


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.publish.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"copy transaction collision: {temporary}")
    try:
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _append_publish_journal(path: Path, event: str, **details: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "gamedata-publish-journal-v1",
                                 "event": event, **details},
                                ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_publish_pair(
    staged_output: Path, staged_gamesl: Path,
    output: Path, gamesl: Path,
    archive_root: Path, *, journal_path: Path, run_id: str,
) -> None:
    """Replace aligned and normalized trees with rollback on partial failure."""
    paths = (staged_output, staged_gamesl, output, gamesl)
    if any(path.is_symlink() for path in paths):
        raise ValueError("publish paths must not be symlinks")
    if staged_output.resolve(strict=False) == output.resolve(strict=False):
        raise ValueError("staged output must be distinct from public output")
    backup_root = archive_root / run_id
    backups = ((output, backup_root / "aligned"),
               (gamesl, backup_root / "gamesl"))
    moved_old: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        gamesl.parent.mkdir(parents=True, exist_ok=True)
        for target, backup in backups:
            if target.exists():
                if backup.exists() or backup.is_symlink():
                    raise FileExistsError(f"publish backup already exists: {backup}")
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                moved_old.append((target, backup))
                _append_publish_journal(journal_path, "archive_previous",
                                        source=str(target), destination=str(backup),
                                        run_id=run_id)
        os.replace(staged_output, output)
        installed.append((output, staged_output))
        _append_publish_journal(journal_path, "publish_replace",
                                source=str(staged_output), destination=str(output),
                                run_id=run_id)
        os.replace(staged_gamesl, gamesl)
        installed.append((gamesl, staged_gamesl))
        _append_publish_journal(journal_path, "publish_replace",
                                source=str(staged_gamesl), destination=str(gamesl),
                                run_id=run_id)
    except Exception:
        for target, source in reversed(installed):
            if target.exists() and not source.exists():
                os.replace(target, source)
        for target, backup in reversed(moved_old):
            if backup.exists() and not target.exists():
                os.replace(backup, target)
        _append_publish_journal(journal_path, "publish_rollback", run_id=run_id)
        raise


def archive_game(config: RebuildConfig, game: GameInput, staged: Path, *, journal_path: Path) -> Path:
    validate_config(config)
    if staged.resolve(strict=False) != (config.staging_root / game.codename).resolve(strict=False):
        raise ValueError("staged path is outside configured staging root")
    destination = config.archive_root / game.codename
    _copy_tree(staged, destination)
    _append_journal(journal_path, "archive", game, source=str(staged), destination=str(destination))
    return destination


def _require_staging_approval(
    config: RebuildConfig, game: GameInput, archived: Path,
    approval_path: Path | None = None,
    run_id: str | None = None,
) -> tuple[Path | None, str | None]:
    """Fail closed unless an independent verifier approved this exact root."""
    if config.allow_test_roots:
        return None, run_id
    if not run_id or Path(run_id).name != run_id:
        raise PermissionError("publish requires a fresh run_id")
    approval = approval_path or (config.publish_staging_root / run_id
                                 / ".STAGING_APPROVED.json")
    if approval.is_symlink() or not approval.is_file():
        raise PermissionError(f"publish requires STAGING_APPROVED: {approval}")
    try:
        payload = json.loads(approval.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PermissionError(f"invalid STAGING_APPROVED receipt: {approval}") from exc
    aligned_root = config.publish_staging_root / run_id / "aligned" / game.codename
    gamesl_root = config.publish_staging_root / run_id / "gamesl" / game.codename
    pair_digest = str(payload.get("pair_digest", ""))
    nvv_contract = payload.get("nvv_contract")
    contract_digest = ""
    if isinstance(nvv_contract, dict):
        contract_rows = nvv_contract.get("pair_contracts")
        if isinstance(contract_rows, list):
            contract_digest = hashlib.sha256(json.dumps(
                contract_rows, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
    if (payload.get("status") != "STAGING_APPROVED"
            or payload.get("schema") != "gamedata-staging-approval-v2"
            or payload.get("game") != game.codename
            or payload.get("run_id") != run_id
            or Path(str(payload.get("accepted_root", ""))).resolve(strict=False)
            != archived.resolve(strict=False)
            or Path(str(payload.get("staged_aligned_root", ""))).resolve(strict=False)
            != aligned_root.resolve(strict=False)
            or Path(str(payload.get("staged_gamesl_root", ""))).resolve(strict=False)
            != gamesl_root.resolve(strict=False)
            or not re.fullmatch(r"[0-9a-f]{64}", pair_digest)
            or not isinstance(nvv_contract, dict)
            or nvv_contract.get("schema") != "nvv-cross-tier-contract-v1"
            or nvv_contract.get("status") != "verified"
            or nvv_contract.get("digest") != contract_digest
            or not re.fullmatch(r"[0-9a-f]{64}", str(nvv_contract.get("digest", "")))
            or not aligned_root.is_dir() or not gamesl_root.is_dir()):
        raise PermissionError(f"STAGING_APPROVED receipt does not bind archive: {approval}")
    return approval, run_id


def publish_game(config: RebuildConfig, game: GameInput, archived: Path, *,
                 journal_path: Path, approval_path: Path | None = None,
                 run_id: str | None = None) -> Path:
    """Publish completed alignment artifacts through the speaker finalizer.

    A pre-alignment archive contains only flat WAV/TXT and is intentionally
    rejected.  This gate prevents accidentally presenting preparation files
    as usable TextGrid output.
    """
    validate_config(config)
    approval, approved_run_id = _require_staging_approval(
        config, game, archived, approval_path, run_id)
    if archived.resolve(strict=False) != (config.archive_root / game.codename).resolve(strict=False):
        raise ValueError("archived path is outside configured archive root")
    if not any(archived.rglob("*.TextGrid")):
        raise ValueError("publish requires completed TextGrid artifacts")
    from finalize_gamedata_speakers import GameSpec, finalize_game

    output = config.output_root / game.codename
    if config.allow_test_roots:
        spec = GameSpec(
            game=game.codename,
            accepted_root=archived,
            published_root=output,
            manifest_path=archived / ".rebuild_manifest.json",
            padded_audio_root=None,
            gamesl_root=config.gamesl_root,
        )
        receipt = finalize_game(spec, workers=16)
        _append_journal(journal_path, "publish", game, output=str(output),
                        gamesl=str(config.gamesl_root / game.codename),
                        approval=str(approval) if approval else None,
                        accepted_count=receipt["accepted_count"])
        return output

    # Finalize into private trees first.  The public pair is changed only by
    # the journaled, rollback-capable replacement below.
    run_id = approved_run_id or f"publish-{os.getpid()}"
    private_output = output.parent / f".{game.codename}.publish.{os.getpid()}"
    private_gamesl = config.gamesl_root.parent / f".{game.codename}.gamesl.{os.getpid()}"
    if private_output.exists() or private_gamesl.exists():
        raise FileExistsError("publish private staging collision")
    spec = GameSpec(
        game=game.codename,
        accepted_root=archived,
        published_root=private_output,
        manifest_path=archived / ".rebuild_manifest.json",
        padded_audio_root=None,
        gamesl_root=private_gamesl,
    )
    try:
        receipt = finalize_game(spec, workers=16)
        atomic_publish_pair(
            private_output, private_gamesl / game.codename, output,
            config.gamesl_root / game.codename, config.publish_archive_root,
            journal_path=journal_path, run_id=run_id)
    except Exception:
        shutil.rmtree(private_output, ignore_errors=True)
        shutil.rmtree(private_gamesl, ignore_errors=True)
        raise
    _append_journal(journal_path, "publish", game, output=str(output),
                    gamesl=str(config.gamesl_root / game.codename),
                    approval=str(approval) if approval else None,
                    accepted_count=receipt["accepted_count"])
    return output


def resolve_pipeline_configs(config: RebuildConfig, game: GameInput, modes: Iterable[str]) -> list[dict[str, object]]:
    """Return resolved authority/fallback configs for a mixed game inventory."""
    stage = config.staging_root / game.codename
    result = []
    order = {"reference": 0, "asr": 1}
    for mode in sorted(set(modes), key=lambda item: order.get(item, 99)):
        if mode not in {"reference", "asr"}:
            raise ValueError(f"unknown inventory text mode: {mode}")
        reference = mode == "reference"
        cohort = stage / "cohorts" / mode
        run_root = stage / "pipeline" / mode
        result.append({
            "game": game.codename, "pipeline_kind": "reference" if reference else "noref",
            "reference_mode": "authority" if reference else "fallback",
            "data_dir": str(cohort), "audio_dir": str(cohort),
            "text_dir": str(cohort) if reference else None,
            "raw_text_dir": str(cohort) if reference else None,
            "workspace": str(run_root / "workspace"),
            "output_dir": str(run_root / "output"),
            "aligned_dir": str(run_root / "aligned"),
            "gamesl_root": str(run_root / "gamesl"),
            "allow_missing_reference": not reference,
            "source_name": game.source_name,
        })
    return result


def load_config(path: Path) -> RebuildConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load rebuild config")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source_root = _path(raw["source_root"])
    games = []
    for item in raw.get("games", []):
        source_dir = _path(item["source_dir"])
        if not source_dir.is_absolute():
            source_dir = source_root / source_dir
        games.append(GameInput(str(item["codename"]), source_dir,
                               str(item.get("source_name", item["source_dir"])),
                               str(item.get("reference_mode", "auto"))))
    config = RebuildConfig(source_root, _path(raw["staging_root"]),
                           _path(raw["archive_root"]), _path(raw["output_root"]),
                           _path(raw["gamesl_root"]), tuple(games),
                           str(raw.get("ffmpeg", "ffmpeg")),
                           bool(raw.get("allow_test_roots", False)),
                           _path(raw.get("publish_staging_root",
                                         "/mnt/Raw/.gamedata_publish_staging")),
                           _path(raw.get("publish_archive_root",
                                         "/mnt/Raw/.gamedata_publish_archive")),
                           bool(raw.get("pipeline", {}).get("split_by_text_mode", True)))
    validate_config(config)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", help="comma-separated game codenames")
    parser.add_argument("--action", choices=("inventory", "stage"), default="inventory",
                        help="Preparation only; archive/publish are library APIs to require explicit caller control")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    selected = {value.strip() for value in args.only.split(",")} if args.only else {g.codename for g in config.games}
    unknown = selected - {g.codename for g in config.games}
    if unknown:
        parser.error(f"unknown games: {', '.join(sorted(unknown))}")
    journal = config.staging_root / "rebuild.jsonl"
    for game in config.games:
        if game.codename not in selected:
            continue
        rows = build_inventory(game.source_dir)
        print(json.dumps({"game": game.codename, "audio": len(rows), "reference": sum(r.text_mode == "reference" for r in rows), "asr": sum(r.text_mode == "asr" for r in rows)}, ensure_ascii=False, sort_keys=True))
        if args.action == "stage":
            stage_game(config, game, journal_path=journal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
