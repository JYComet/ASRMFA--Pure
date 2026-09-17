#!/usr/bin/env python3
"""Freeze an immutable, fail-closed NVV repair input manifest.

This tool is intentionally read-only with respect to accepted TextGrids and
their paired WAVs.  It only writes a new JSONL file after the complete selected
universe has been validated, so it is safe to use as the hand-off between an
NVV audit and an affected-items-only rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

from nvv_contract import NVV_TIER_NAMES, nvv_sequence
from postprocess_textgrids import parse_textgrid


MANIFEST_SCHEMA = "gamedata-nvv-repair-manifest-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT_FORBIDDEN = {"", ".", ".."}
EXPECTED_20260910_GAME_COUNTS = {
    "baijing": 514,
    "persona": 834,
    "punishing_gray_raven": 25,
    "reverse1999": 50,
}
PRODUCTION_20260910_GAMES = tuple(EXPECTED_20260910_GAME_COUNTS)
EXPECTED_20260910_FILE_COUNT = 1423
EXPECTED_20260910_OCCURRENCE_COUNT = 1452
SNAPSHOT_20260910 = {
    "baijing": {
        "textgrid_root": Path("/mnt/nvme3/mfa_work_gamedata_baijing_20260903/output_staging/20260903T162031Z_2492831_2492831"),
        "source_wav_root": Path("/mnt/nvme3/mfa_work_gamedata_baijing_20260903/audio_16k"),
        "stage_manifest": Path("/mnt/nvme3/gamedata_20260903/baijing/.stage_manifest.json"),
    },
    "persona": {"textgrid_root": Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/publish_accepted_v1/persona"),
                "receipt": Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/staging/persona/.rebuild_manifest.json")},
    "punishing_gray_raven": {"textgrid_root": Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/accepted/punishing_gray_raven"),
                             "receipt": Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/staging/punishing_gray_raven/.rebuild_manifest.json")},
    "reverse1999": {"textgrid_root": Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/accepted/reverse1999"),
                    "receipt": Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/staging/reverse1999/.rebuild_manifest.json")},
}
SNAPSHOT_FRESH4_STAGING_ROOT = Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/staging")
SNAPSHOT_FRESH4_GAMESL_ROOT = Path("/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4/final_staging_nvme_v3/gamesl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_directory(root: Path, *, label: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is missing, invalid, or symlinked: {root}")
    return root.resolve()


def _safe_root(root: Path, *, label: str) -> Path:
    resolved = _safe_directory(root, label=label)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} root contains symlink: {path}")
    return resolved


def _safe_game_tree(root: Path, *, game: str, label: str) -> Path:
    game_root = root / game
    if game_root.is_symlink() or not game_root.is_dir():
        raise ValueError(f"{label} game root is missing, invalid, or symlinked: {game_root}")
    for path in game_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} game tree contains symlink: {path}")
    return game_root.resolve()


def _vetted_source_roots(paths: Iterable[Path]) -> tuple[Path, ...]:
    roots = tuple(sorted({_safe_root(Path(path), label="source WAV") for path in paths},
                         key=str))
    if not roots:
        raise ValueError("at least one allowed source root is required")
    for index, root in enumerate(roots):
        for other in roots[index + 1:]:
            try:
                other.relative_to(root)
            except ValueError:
                continue
            raise ValueError(f"allowed source roots overlap: {root} and {other}")
    return roots


def _safe_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or value in _COMPONENT_FORBIDDEN:
        raise ValueError(f"invalid {label} component: {value!r}")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValueError(f"path escape in {label} component: {value!r}")
    return value


def _key_from_record(record: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        _safe_component(record.get("game"), label="game"),
        _safe_component(record.get("speaker"), label="speaker"),
        _safe_component(record.get("stem"), label="stem"),
    )


def _mode(record: Mapping[str, object], name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid {name}")
    return value


def _expected_sha256(value: object) -> str | None:
    """Validate an optional caller claim that will be checked against bytes."""
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("invalid source_wav_sha256: expected lowercase SHA-256 hex")
    return value


def _paths_for_key(
    key: tuple[str, str, str], *, textgrid_root: Path, wav_root: Path,
) -> tuple[Path, Path]:
    game, speaker, stem = key
    grid = textgrid_root / game / speaker / f"{stem}.TextGrid"
    wav = wav_root / game / speaker / f"{stem}.wav"
    for path, root, label in ((grid, textgrid_root, "TextGrid"),
                              (wav, wav_root, "WAV")):
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escape for {label}: {path}") from exc
        if path.is_symlink():
            raise ValueError(f"symlinked {label} is not allowed: {path}")
    return grid, wav


def _audited_old_nvv(
    grid_path: Path,
) -> tuple[list[str], list[dict[str, object]], dict[str, list[str]], str]:
    """Freeze old NVV evidence without requiring the old broken grid to pass.

    The repair scope is selected precisely because four aligned tiers may have
    lost a leading NVV.  Raw text is the frozen authority for the expected
    sequence; all five old observations are retained as audit evidence.
    """
    if not grid_path.is_file():
        raise ValueError(f"accepted TextGrid missing: {grid_path}")
    grid = parse_textgrid(grid_path)
    names = tuple(tier.name for tier in grid.tiers)
    if names != tuple(NVV_TIER_NAMES):
        raise ValueError(f"missing or unexpected five TextGrid tiers: {grid_path}: {names!r}")
    by_tier = {tier.name: nvv_sequence(tier) for tier in grid.tiers}
    sequence = by_tier[NVV_TIER_NAMES[0]]
    if not sequence:
        raise ValueError(f"no NVV in selected TextGrid: {grid_path}")
    expected_positions = [
        {"tier": "raw_text", "occurrence_ordinal": ordinal, "label": label}
        for ordinal, label in enumerate(sequence)
    ]
    if all(not by_tier[tier] for tier in NVV_TIER_NAMES[1:]):
        failure_class = "raw_only"
    elif any(by_tier[tier] != sequence for tier in NVV_TIER_NAMES[1:]):
        failure_class = "cross_tier_mismatch"
    else:
        # Sequence agreement alone cannot attest that historical interval
        # timing was sound.  Preserve this distinct audit state for rerun.
        failure_class = "temporal_or_other"
    return sequence, expected_positions, by_tier, failure_class


def _real_source_wav(
    path_value: object, *, expected_sha256: object = None,
    allowed_roots: tuple[Path, ...],
) -> tuple[Path, Path]:
    """Return a real, non-symlinked source WAV; never trust a caller hash."""
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("missing or invalid source_wav_path")
    source = Path(path_value)
    if not source.is_absolute():
        raise ValueError(f"source WAV path must be absolute: {source}")
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source WAV missing, invalid, or symlinked: {source}")
    if source.suffix.lower() != ".wav":
        raise ValueError(f"source WAV must use .wav extension: {source}")
    containing = []
    for root in allowed_roots:
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        candidate = root
        for component in relative.parts:
            candidate = candidate / component
            if candidate.is_symlink():
                raise ValueError(f"source WAV path contains symlink: {candidate}")
        containing.append(root)
    if len(containing) != 1:
        raise ValueError(f"source WAV outside allowed source roots: {source}")
    source = source.resolve()
    expected = _expected_sha256(expected_sha256)
    if expected is not None and _sha256(source) != expected:
        raise ValueError(f"source WAV hash mismatch: {source}")
    return source, containing[0]


def _source_for_scan(
    source_root: Path, key: tuple[str, str, str], *, layout: str,
) -> tuple[Path, Path]:
    game, speaker, stem = key
    if layout == "game_speaker_stem":
        candidate = source_root / game / speaker / f"{stem}.wav"
    elif layout == "flat_stem":
        candidate = source_root / f"{stem}.wav"
    else:
        raise ValueError(f"unsupported source_wav_layout: {layout}")
    try:
        candidate.resolve().relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"source WAV path escape: {candidate}") from exc
    return _real_source_wav(str(candidate), allowed_roots=(source_root,))


def _record_for_key(
    key: tuple[str, str, str], *, source_wav: Path, source_wav_root: Path,
    asr_mode: str, reference_mode: str, textgrid_root: Path, wav_root: Path,
    run_id: str,
) -> dict[str, object]:
    grid, wav = _paths_for_key(
        key, textgrid_root=textgrid_root, wav_root=wav_root)
    if not wav.is_file():
        raise ValueError(f"paired WAV missing: {wav}")
    sequence, positions, observed, failure_class = _audited_old_nvv(grid)
    game, speaker, stem = key
    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "game": game,
        "speaker": speaker,
        "stem": stem,
        "old_textgrid_relative_path": grid.relative_to(textgrid_root).as_posix(),
        "old_wav_relative_path": wav.relative_to(wav_root).as_posix(),
        "textgrid_sha256": _sha256(grid),
        "wav_sha256": _sha256(wav),
        "source_wav_path": str(source_wav),
        "source_wav_root": str(source_wav_root),
        "source_wav_sha256": _sha256(source_wav),
        "asr_mode": asr_mode,
        "reference_mode": reference_mode,
        "expected_nvv_sequence": sequence,
        "expected_nvv_positions": positions,
        "observed_nvv_sequences": observed,
        "old_failure_class": failure_class,
    }


def _scan_keys(textgrid_root: Path) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    for grid in sorted(textgrid_root.rglob("*.TextGrid")):
        if grid.is_symlink():
            raise ValueError(f"symlinked TextGrid is not allowed: {grid}")
        relative = grid.relative_to(textgrid_root)
        if len(relative.parts) != 3:
            raise ValueError(f"TextGrid does not use game/speaker/stem layout: {grid}")
        game, speaker, filename = relative.parts
        if grid.suffix != ".TextGrid" or not grid.stem:
            raise ValueError(f"invalid TextGrid filename: {grid}")
        # Only NVV-bearing grids enter generated scan scope.  Their full
        # five-tier contract is then checked by _record_for_key.
        try:
            parsed = parse_textgrid(grid)
        except Exception as exc:  # parser errors must not be silently skipped
            raise ValueError(f"cannot parse TextGrid during scan: {grid}: {exc}") from exc
        if any(nvv_sequence(tier) for tier in parsed.tiers):
            keys.append((_safe_component(game, label="game"),
                         _safe_component(speaker, label="speaker"),
                         _safe_component(grid.stem, label="stem")))
    if not keys:
        raise ValueError("no NVV-bearing TextGrids found during scan")
    return keys


def _production_problem_keys(textgrid_root: Path) -> list[tuple[str, str, str]]:
    """Discover only old raw-NVV/cross-tier-mismatch repair targets."""
    keys: list[tuple[str, str, str]] = []
    for game in PRODUCTION_20260910_GAMES:
        game_root = _safe_game_tree(textgrid_root, game=game, label="accepted TextGrid")
        for grid in sorted(game_root.rglob("*.TextGrid")):
            relative = grid.relative_to(textgrid_root)
            if len(relative.parts) != 3:
                raise ValueError(f"TextGrid does not use game/speaker/stem layout: {grid}")
            _, speaker, _ = relative.parts
            try:
                expected, _, observed, _ = _audited_old_nvv(grid)
            except ValueError:
                # Non-NVV grids are expected in the public root.  A malformed grid
                # that does contain a raw NVV must not be silently skipped.
                try:
                    parsed = parse_textgrid(grid)
                    raw = next((tier for tier in parsed.tiers
                                if tier.name == "raw_text"), None)
                except Exception:
                    raw = None
                if raw is not None and nvv_sequence(raw):
                    raise
                continue
            if any(observed[tier] != expected for tier in NVV_TIER_NAMES[1:]):
                keys.append((game, _safe_component(speaker, label="speaker"),
                             _safe_component(grid.stem, label="stem")))
    return keys


def _load_fresh4_receipt(path: Path, *, game: str) -> dict[str, dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"fresh4 receipt missing, invalid, or symlinked: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid fresh4 receipt JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("game") != game:
        raise ValueError(f"fresh4 receipt game mismatch: {path}")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"fresh4 receipt items missing: {path}")
    by_stem: dict[str, dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"fresh4 receipt item is invalid: {path}")
        stem = _safe_component(item.get("stem"), label="receipt stem")
        if stem in by_stem:
            raise ValueError(f"duplicate fresh4 receipt stem: {game}/{stem}")
        by_stem[stem] = item
    return by_stem


def freeze_20260910_nvv_production_manifest(
    *, accepted_textgrid_root: Path, paired_wav_root: Path,
    baijing_source_wav_root: Path, fresh4_staging_root: Path,
    fresh4_receipts: Mapping[str, Path], run_id: str,
    output_path: Path | None = None,
    expected_game_counts: Mapping[str, int] = EXPECTED_20260910_GAME_COUNTS,
    expected_total_occurrences: int = EXPECTED_20260910_OCCURRENCE_COUNT,
) -> list[dict[str, object]]:
    """Build the 2026-09-10 repair manifest without hand-authored records.

    Baijing sources use the trusted flat ``{stem}.wav`` store.  The three
    fresh4 games are bound to an ASR-only item in their rebuild receipt, using
    its staged ``audio`` path.  Fixed count checks turn any audit drift into a
    hard error before an output manifest can exist.
    """
    expected_counts = {str(game): int(count)
                       for game, count in expected_game_counts.items()}
    if set(expected_counts) != set(PRODUCTION_20260910_GAMES):
        raise ValueError("production expected counts must cover exactly the fixed four games")
    if any(count < 1 for count in expected_counts.values()):
        raise ValueError("invalid expected game counts")
    expected_total_files = sum(expected_counts.values())
    baijing_root = _safe_root(Path(baijing_source_wav_root), label="baijing source WAV")
    fresh4_root = _safe_root(Path(fresh4_staging_root), label="fresh4 staging source WAV")
    expected_receipt_games = set(expected_counts) - {"baijing"}
    if set(fresh4_receipts) != expected_receipt_games:
        raise ValueError("fresh4 receipt games do not match production scope")
    receipt_items = {
        game: _load_fresh4_receipt(Path(path), game=game)
        for game, path in fresh4_receipts.items()
    }

    root = _safe_directory(Path(accepted_textgrid_root), label="accepted TextGrid")
    paired_root = _safe_directory(Path(paired_wav_root), label="paired WAV")
    for game in PRODUCTION_20260910_GAMES:
        _safe_game_tree(root, game=game, label="accepted TextGrid")
        _safe_game_tree(paired_root, game=game, label="paired WAV")
    selected_keys = _production_problem_keys(root)
    game_counts = Counter(key[0] for key in selected_keys)
    if dict(sorted(game_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(
            "game count conservation failed: "
            f"observed={dict(sorted(game_counts.items()))}, expected={dict(sorted(expected_counts.items()))}")
    if len(selected_keys) != expected_total_files:
        raise ValueError(
            f"file count conservation failed: observed={len(selected_keys)}, "
            f"expected={expected_total_files}")

    records: list[dict[str, object]] = []
    for game, speaker, stem in selected_keys:
        if game == "baijing":
            source = baijing_root / f"{stem}.wav"
            expected_hash = None
        else:
            item = receipt_items[game].get(stem)
            if item is None:
                raise ValueError(f"fresh4 receipt lacks selected stem: {game}/{stem}")
            if item.get("text_mode") != "asr":
                raise ValueError(f"selected fresh4 stem is not ASR: {game}/{stem}")
            source = item.get("audio")
            expected_hash = item.get("sha256")
        record: dict[str, object] = {
            "game": game, "speaker": speaker, "stem": stem,
            "source_wav_path": str(source), "asr_mode": "fallback",
            "reference_mode": "fallback",
        }
        if expected_hash is not None:
            record["source_wav_sha256"] = expected_hash
        records.append(record)

    rows = freeze_nvv_repair_manifest(
        accepted_textgrid_root=root, paired_wav_root=paired_root,
        records=records, run_id=run_id, output_path=None,
        allowed_source_roots=(baijing_root, fresh4_root),
        prevalidated_input_roots=True,
        prevalidated_allowed_source_roots=True,
    )
    occurrence_count = sum(len(row["expected_nvv_sequence"]) for row in rows)
    if occurrence_count != expected_total_occurrences:
        raise ValueError(
            f"NVV occurrence conservation failed: observed={occurrence_count}, "
            f"expected={expected_total_occurrences}")
    if output_path is not None:
        _write_manifest_output(Path(output_path), rows, protected_roots=(root, paired_root))
    return rows


def _snapshot_problem_stems(root: Path) -> list[tuple[str, list[str], dict[str, list[str]], str]]:
    root = _safe_root(root, label="snapshot TextGrid")
    result = []
    for grid in sorted(root.glob("*.TextGrid")):
        # A snapshot contains many ordinary non-NVV grids.  They are outside
        # this repair scope; only a raw NVV opts a grid into strict auditing.
        try:
            parsed = parse_textgrid(grid)
        except Exception:
            continue
        raw = next((tier for tier in parsed.tiers if tier.name == "raw_text"), None)
        if raw is None or not nvv_sequence(raw):
            continue
        # Once raw NVV is present, a missing/faulty five-tier structure is an
        # integrity failure rather than a reason to silently lose the target.
        expected, _, observed, failure = _audited_old_nvv(grid)
        if any(observed[tier] != expected for tier in NVV_TIER_NAMES[1:]):
            result.append((grid.stem, expected, observed, failure))
    return result


def _baijing_stage_speakers(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"baijing stage manifest missing or symlinked: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baijing stage manifest must be a dict")
    result = {}
    for stem, item in payload.items():
        if not isinstance(stem, str) or not isinstance(item, dict):
            raise ValueError("invalid baijing stage entry")
        audio = Path(str(item.get("audio", "")))
        if (audio.suffix.lower() != ".wav" or audio.stem != stem
                or not audio.parent.name or stem in result):
            raise ValueError("invalid or duplicate baijing stage audio")
        result[stem] = audio.parent.name
    return result


def freeze_20260910_nvv_snapshot_manifest(
    *, run_id: str, output_path: Path | None = None,
) -> list[dict[str, object]]:
    """Freeze repair inputs from immutable 2026-09-10 snapshots only."""
    if set(SNAPSHOT_20260910) != set(PRODUCTION_20260910_GAMES):
        raise ValueError("snapshot authority must cover exactly the fixed four games")
    expected_counts = EXPECTED_20260910_GAME_COUNTS
    baijing = SNAPSHOT_20260910["baijing"]
    baijing_source_root = _safe_root(Path(baijing["source_wav_root"]), label="baijing source WAV")
    baijing_speakers = _baijing_stage_speakers(Path(baijing["stage_manifest"]))
    fresh_root = _safe_root(SNAPSHOT_FRESH4_STAGING_ROOT, label="fresh4 snapshot source WAV")
    gamesl_root = _safe_directory(SNAPSHOT_FRESH4_GAMESL_ROOT, label="fresh4 old WAV snapshot")
    rows: list[dict[str, object]] = []
    protected = [baijing_source_root, fresh_root]
    for game in PRODUCTION_20260910_GAMES:
        spec = SNAPSHOT_20260910[game]
        problems = _snapshot_problem_stems(Path(spec["textgrid_root"]))
        if len(problems) != expected_counts[game]:
            raise ValueError(f"snapshot game count conservation failed: {game}={len(problems)}")
        receipt = None if game == "baijing" else _load_fresh4_receipt(Path(spec["receipt"]), game=game)
        for stem, expected, observed, failure in problems:
            grid = Path(spec["textgrid_root"]) / f"{stem}.TextGrid"
            if game == "baijing":
                speaker = baijing_speakers.get(stem)
                source = baijing_source_root / f"{stem}.wav"
                if not speaker:
                    raise ValueError(f"baijing stage speaker missing: {stem}")
                old_wav_status = "unavailable_due_external_cleanup"
                old_wav_path = None
            else:
                item = receipt.get(stem) if receipt else None
                if item is None or item.get("text_mode") != "asr":
                    raise ValueError(f"fresh4 ASR receipt missing: {game}/{stem}")
                source = Path(str(item.get("audio", "")))
                speaker = Path(str(item.get("source", ""))).parent.name
                if not speaker:
                    raise ValueError(f"fresh4 speaker missing: {game}/{stem}")
                old_wav_path = gamesl_root / game / speaker / f"{stem}.wav"
                old_wav_status = "verified" if old_wav_path.is_file() and not old_wav_path.is_symlink() else "unavailable"
            source, source_root = _real_source_wav(str(source), allowed_roots=(baijing_source_root, fresh_root))
            row = {
                "schema": MANIFEST_SCHEMA, "run_id": run_id, "game": game,
                "speaker": speaker, "stem": stem,
                "old_textgrid_relative_path": f"{game}/{stem}.TextGrid",
                "textgrid_sha256": _sha256(grid),
                "source_wav_path": str(source), "source_wav_root": str(source_root),
                "source_wav_sha256": _sha256(source), "asr_mode": "fallback",
                "reference_mode": "fallback", "expected_nvv_sequence": expected,
                "expected_nvv_positions": [{"tier": "raw_text", "occurrence_ordinal": i, "label": label}
                                           for i, label in enumerate(expected)],
                "observed_nvv_sequences": observed, "old_failure_class": failure,
                "old_wav_snapshot_status": old_wav_status,
            }
            if old_wav_path is not None and old_wav_status == "verified":
                row["old_wav_snapshot_relative_path"] = old_wav_path.relative_to(gamesl_root).as_posix()
                row["old_wav_sha256"] = _sha256(old_wav_path)
            rows.append(row)
    if len(rows) != sum(expected_counts.values()):
        raise ValueError("snapshot file count conservation failed")
    if sum(len(row["expected_nvv_sequence"]) for row in rows) != EXPECTED_20260910_OCCURRENCE_COUNT:
        raise ValueError("snapshot NVV occurrence conservation failed")
    if output_path is not None:
        _write_manifest_output(Path(output_path), rows, protected_roots=protected)
    return rows


def _assert_new_output_path(output_path: Path, roots: Iterable[Path]) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"refusing to overwrite manifest output: {output_path}")
    output_resolved = output_path.resolve()
    for root in roots:
        try:
            output_resolved.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"manifest output must not be inside protected input root: {output_path}")


def _write_manifest_output(
    output_path: Path, rows: Iterable[Mapping[str, object]], *,
    protected_roots: Iterable[Path],
) -> None:
    """Write exactly one new JSONL manifest after every caller check passes."""
    _assert_new_output_path(output_path, protected_roots)
    payload = "".join(json.dumps(
        row, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")) + "\n" for row in rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ``x`` ensures a concurrent producer cannot replace a checked target.
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def freeze_nvv_repair_manifest(
    *, accepted_textgrid_root: Path, paired_wav_root: Path,
    records: Iterable[Mapping[str, object]] | None, run_id: str,
    output_path: Path | None = None, source_wav_root: Path | None = None,
    source_wav_layout: str = "game_speaker_stem",
    allowed_source_roots: Iterable[Path] | None = None,
    prevalidated_input_roots: bool = False,
    prevalidated_allowed_source_roots: bool = False,
    asr_mode: str | None = None, reference_mode: str | None = None,
) -> list[dict[str, object]]:
    """Return validated repair rows and optionally write one fresh JSONL file.

    ``records`` mode freezes an explicit audited subset.  Each record must
    carry game/speaker/stem, an actual absolute source WAV path, ASR mode, and
    reference mode.  The source hash is always calculated from that path.
    Scan mode discovers only NVV-bearing grids and derives source WAV paths
    from a vetted root and layout.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("missing or invalid run_id")
    input_validator = _safe_directory if prevalidated_input_roots else _safe_root
    textgrid_root = input_validator(Path(accepted_textgrid_root), label="accepted TextGrid")
    wav_root = input_validator(Path(paired_wav_root), label="paired WAV")
    if output_path is not None:
        _assert_new_output_path(Path(output_path), (textgrid_root, wav_root))

    selected: list[tuple[tuple[str, str, str], Path, Path, str, str]] = []
    if records is None:
        if source_wav_root is None:
            raise ValueError("missing source_wav_root for scan")
        source_root = _safe_root(Path(source_wav_root), label="source WAV")
        if not isinstance(asr_mode, str) or not asr_mode.strip():
            raise ValueError("missing or invalid asr_mode for scan")
        if not isinstance(reference_mode, str) or not reference_mode.strip():
            raise ValueError("missing or invalid reference_mode for scan")
        selected = [
            (key, *_source_for_scan(source_root, key, layout=source_wav_layout),
             asr_mode, reference_mode)
            for key in _scan_keys(textgrid_root)
        ]
    else:
        if allowed_source_roots is None:
            raise ValueError("allowed_source_roots is required for records")
        if prevalidated_allowed_source_roots:
            allowed_roots = tuple(sorted(
                {_safe_directory(Path(path), label="source WAV")
                 for path in allowed_source_roots}, key=str))
        else:
            allowed_roots = _vetted_source_roots(allowed_source_roots)
        for item in records:
            if not isinstance(item, Mapping):
                raise ValueError("records JSONL row must be an object")
            selected.append((_key_from_record(item), *_real_source_wav(
                                 item.get("source_wav_path"),
                                 expected_sha256=item.get("source_wav_sha256"),
                                 allowed_roots=allowed_roots),
                             _mode(item, "asr_mode"),
                             _mode(item, "reference_mode")))
        if not selected:
            raise ValueError("records selection is empty")

    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, object]] = []
    for key, source_wav, source_root, row_asr_mode, row_reference_mode in sorted(selected):
        if key in seen:
            raise ValueError(f"duplicate manifest key: {key!r}")
        seen.add(key)
        rows.append(_record_for_key(
            key, source_wav=source_wav, source_wav_root=source_root,
            asr_mode=row_asr_mode,
            reference_mode=row_reference_mode, textgrid_root=textgrid_root,
            wav_root=wav_root, run_id=run_id))

    if output_path is not None:
        _write_manifest_output(
            Path(output_path), rows, protected_roots=(textgrid_root, wav_root))
    return rows


def _read_records(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"records JSONL missing, invalid, or symlinked: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank records JSONL line: {line_number}")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid records JSONL at line {line_number}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"records JSONL row must be object: line {line_number}")
        rows.append(item)
    return rows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-textgrid-root", type=Path)
    parser.add_argument("--paired-wav-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--records-jsonl", type=Path)
    parser.add_argument("--source-wav-root", type=Path)
    parser.add_argument("--source-wav-layout", choices=("game_speaker_stem", "flat_stem"),
                        default="game_speaker_stem")
    parser.add_argument("--asr-mode")
    parser.add_argument("--reference-mode")
    parser.add_argument("--allowed-source-root", type=Path, action="append")
    parser.add_argument("--production-20260910", action="store_true",
                        help="derive all 1,423 selected rows from public audit roots and fresh4 receipts")
    parser.add_argument("--baijing-source-wav-root", type=Path)
    parser.add_argument("--fresh4-staging-root", type=Path)
    parser.add_argument("--fresh4-receipt", action="append", default=[],
                        metavar="GAME=PATH")
    parser.add_argument("--snapshot-production-20260910", action="store_true",
                        help="use only the fixed immutable snapshot authorities")
    return parser.parse_args(argv)


def _receipt_argument_map(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        game, separator, path = value.partition("=")
        if not separator or not game or not path:
            raise ValueError(f"invalid --fresh4-receipt (expected GAME=PATH): {value!r}")
        game = _safe_component(game, label="receipt game")
        if game in result:
            raise ValueError(f"duplicate --fresh4-receipt game: {game}")
        result[game] = Path(path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.snapshot_production_20260910:
        if (args.production_20260910 or args.accepted_textgrid_root
                or args.paired_wav_root or args.records_jsonl
                or args.source_wav_root or args.allowed_source_root):
            raise ValueError("snapshot production rejects dynamic public/input roots")
        rows = freeze_20260910_nvv_snapshot_manifest(
            run_id=args.run_id, output_path=args.output)
    elif args.production_20260910:
        if args.accepted_textgrid_root is None or args.paired_wav_root is None:
            raise ValueError("production mode requires accepted TextGrid and paired WAV roots")
        if args.baijing_source_wav_root is None or args.fresh4_staging_root is None:
            raise ValueError(
                "production mode requires --baijing-source-wav-root and --fresh4-staging-root")
        if args.records_jsonl or args.source_wav_root or args.allowed_source_root:
            raise ValueError("production mode derives its own records and source roots")
        rows = freeze_20260910_nvv_production_manifest(
            accepted_textgrid_root=args.accepted_textgrid_root,
            paired_wav_root=args.paired_wav_root,
            baijing_source_wav_root=args.baijing_source_wav_root,
            fresh4_staging_root=args.fresh4_staging_root,
            fresh4_receipts=_receipt_argument_map(args.fresh4_receipt),
            run_id=args.run_id,
            output_path=args.output,
        )
    else:
        if args.accepted_textgrid_root is None or args.paired_wav_root is None:
            raise ValueError("generic mode requires accepted TextGrid and paired WAV roots")
        records = _read_records(args.records_jsonl) if args.records_jsonl else None
        rows = freeze_nvv_repair_manifest(
            accepted_textgrid_root=args.accepted_textgrid_root,
            paired_wav_root=args.paired_wav_root,
            records=records,
            run_id=args.run_id,
            output_path=args.output,
            source_wav_root=args.source_wav_root,
            source_wav_layout=args.source_wav_layout,
            allowed_source_roots=args.allowed_source_root,
            asr_mode=args.asr_mode,
            reference_mode=args.reference_mode,
        )
    print(json.dumps({"status": "ok", "rows": len(rows),
                      "output": str(args.output)}, ensure_ascii=False,
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"freeze-nvv-repair-manifest: {exc}", file=sys.stderr)
        raise SystemExit(2)
