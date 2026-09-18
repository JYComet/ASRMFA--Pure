import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

import sys

from rebuild_gamedata import (  # noqa: E402
    GameInput,
    RebuildConfig,
    archive_game,
    atomic_publish_pair,
    _require_staging_approval,
    build_inventory,
    build_cohorts,
    convert_audio_to_wav,
    publish_game,
    resolve_pipeline_configs,
    stage_game,
    validate_config,
)


def _wav(path: Path, *, channels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((800, channels), dtype=np.float32)
    sf.write(path, data, 8000)


def _grid(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 0.1\ntiers? <exists>\nsize = 1\nitem []:\n'
        '    item [1]:\n        class = "IntervalTier"\n'
        '        name = "words"\n        xmin = 0\n        xmax = 0.1\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 0.1\n'
        '            text = ""\n', encoding="utf-8")


def test_inventory_is_recursive_and_nonempty_same_dir_text_wins(tmp_path):
    root = tmp_path / "input"
    _wav(root / "voice" / "line.wav")
    (root / "voice" / "line.txt").write_text("你好\n", encoding="utf-8")
    _wav(root / "voice" / "empty.ogg")
    (root / "voice" / "empty.txt").write_text("", encoding="utf-8")
    _wav(root / "other" / "line.wav")
    rows = build_inventory(root)

    assert [row.stem for row in rows] == ["line", "empty", "line__dup01"]
    by_path = {row.source.parent.name + "/" + row.source.name: row for row in rows}
    assert by_path["voice/line.wav"].reference_text == "你好\n"
    assert by_path["voice/empty.ogg"].reference_text is None
    assert by_path["voice/empty.ogg"].text_mode == "asr"
    assert by_path["voice/line.wav"].text_mode == "reference"


def test_convert_audio_to_mono_pcm16_wav(tmp_path):
    source = tmp_path / "stereo.flac"
    target = tmp_path / "out.wav"
    _wav(source)
    convert_audio_to_wav(source, target)
    info = sf.info(target)
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_collision_suffix_does_not_overwrite_natural_stem(tmp_path):
    root = tmp_path / "input"
    _wav(root / "a.wav")
    _wav(root / "nested" / "a.wav")
    _wav(root / "nested" / "a__dup01.wav")
    assert {row.stem for row in build_inventory(root)} == {"a", "a__dup01", "a__dup02"}


def test_resolve_mixed_pipeline_configs(tmp_path):
    game = GameInput("new", tmp_path / "source", "new-source")
    config = RebuildConfig(
        source_root=tmp_path / "source",
        staging_root=tmp_path / "stage",
        archive_root=tmp_path / "archive",
        output_root=tmp_path / "out",
        gamesl_root=tmp_path / "gamesl",
        games=(game,),
    )
    resolved = resolve_pipeline_configs(config, game, ["reference", "asr"])
    assert [item["reference_mode"] for item in resolved] == ["authority", "fallback"]
    assert {item["pipeline_kind"] for item in resolved} == {"reference", "noref"}


def test_build_cohorts_is_disjoint_and_fallback_has_no_reference_text():
    rows = [
        type("Row", (), {"stem": "ref", "text_mode": "reference"})(),
        type("Row", (), {"stem": "asr", "text_mode": "asr"})(),
    ]
    cohorts = build_cohorts(rows)
    assert {row.stem for row in cohorts["reference"]} == {"ref"}
    assert {row.stem for row in cohorts["asr"]} == {"asr"}
    assert not ({row.stem for row in cohorts["reference"]}
                & {row.stem for row in cohorts["asr"]})
    assert {row.stem for values in cohorts.values() for row in values} == {"ref", "asr"}


def test_resolved_cohort_paths_are_under_fresh_staging_root(tmp_path):
    game = GameInput("persona", tmp_path / "source", "persona")
    config = RebuildConfig(
        source_root=tmp_path / "source",
        staging_root=tmp_path / "run" / "stage",
        archive_root=tmp_path / "run" / "archive",
        output_root=tmp_path / "out",
        gamesl_root=tmp_path / "gamesl",
        games=(game,), allow_test_roots=True,
    )
    resolved = resolve_pipeline_configs(config, game, ["asr"])
    item = resolved[0]
    assert Path(item["data_dir"]).is_relative_to(config.staging_root)
    assert Path(item["workspace"]).is_relative_to(config.staging_root)
    assert Path(item["output_dir"]).is_relative_to(config.staging_root)
    assert Path(item["gamesl_root"]).is_relative_to(config.staging_root)
    assert item["text_dir"] is None


def test_config_rejects_output_roots_outside_exact_targets(tmp_path):
    config = RebuildConfig(
        source_root=tmp_path / "source",
        staging_root=tmp_path / "stage",
        archive_root=tmp_path / "archive",
        output_root=tmp_path / "not-raw",
        gamesl_root=tmp_path / "gamesl",
        games=(),
    )
    with pytest.raises(ValueError, match="output_root"):
        validate_config(config)


def test_journaled_stage_archive_publish(tmp_path):
    source_root = tmp_path / "source"
    _wav(source_root / "Game" / "Alice" / "a.wav")
    game = GameInput("game", source_root, "Game")
    config = RebuildConfig(
        source_root=source_root,
        staging_root=tmp_path / "stage",
        archive_root=tmp_path / "archive",
        output_root=tmp_path / "out",
        gamesl_root=tmp_path / "gamesl",
        games=(game,),
        allow_test_roots=True,
    )
    validate_config(config)
    journal = tmp_path / "journal.jsonl"
    staged = stage_game(config, game, journal_path=journal)
    archived = archive_game(config, game, staged, journal_path=journal)
    _grid(archived / "a.TextGrid")
    published = publish_game(config, game, archived, journal_path=journal)
    assert (staged / "a.wav").is_file()
    assert (staged / "cohorts" / "asr" / "a.wav").is_file()
    assert not (staged / "cohorts" / "asr" / "a.txt").exists()
    assert (archived / "a.wav").is_file()
    assert (published / "Alice" / "a.TextGrid").is_file()
    assert (tmp_path / "gamesl" / "game" / "Alice" / "a.wav").is_file()
    events = [json.loads(line)["event"] for line in journal.read_text().splitlines()]
    assert events == ["stage", "archive", "publish"]


def test_real_publish_requires_staging_approved_before_any_public_write() -> None:
    source_root = Path("/mnt/Raw/GAMEDATA")
    game = GameInput("persona", source_root / "女神异闻录", "女神异闻录")
    config = RebuildConfig(
        source_root=source_root,
        staging_root=Path("/mnt/nvme3/gamedata_rebuild_test/staging"),
        archive_root=Path("/mnt/Raw/.gamedata_publish_archive/test"),
        output_root=Path("/mnt/Raw/GAMEDATA_对齐_20260903"),
        gamesl_root=Path("/mnt/Raw/GAMESL"),
        games=(game,),
    )
    with pytest.raises(PermissionError, match="STAGING_APPROVED"):
        publish_game(config, game, config.archive_root / game.codename,
                     journal_path=Path("/mnt/nvme3/gamedata_rebuild_test/journal"),
                     run_id="run")


def test_atomic_publish_pair_rolls_back_when_second_replace_fails(tmp_path, monkeypatch):
    from rebuild_gamedata import os as rebuild_os

    staged_output = tmp_path / "staged-output"
    staged_gamesl = tmp_path / "staged-gamesl"
    output = tmp_path / "output"
    gamesl = tmp_path / "gamesl"
    staged_output.mkdir()
    staged_gamesl.mkdir()
    (staged_output / "grid").write_text("grid", encoding="utf-8")
    (staged_gamesl / "audio").write_text("audio", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    original_replace = rebuild_os.replace
    calls = {"count": 0}

    def fail_second(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(rebuild_os, "replace", fail_second)
    with pytest.raises(OSError, match="injected publish failure"):
        atomic_publish_pair(staged_output, staged_gamesl, output, gamesl,
                            tmp_path / "archive", journal_path=journal,
                            run_id="run")
    assert staged_output.is_dir()
    assert staged_gamesl.is_dir()
    assert not output.exists()
    assert not gamesl.exists()


def test_staging_approval_binds_run_pair_roots_and_digest(tmp_path):
    config = RebuildConfig(
        source_root=tmp_path / "source",
        staging_root=tmp_path / "stage",
        archive_root=tmp_path / "archive",
        output_root=tmp_path / "out",
        gamesl_root=tmp_path / "gamesl",
        games=(GameInput("game", tmp_path / "source", "Game"),),
        publish_staging_root=tmp_path / "publish-staging",
        publish_archive_root=tmp_path / "publish-archive",
    )
    run_id = "run-1"
    aligned = config.publish_staging_root / run_id / "aligned" / "game"
    gamesl = config.publish_staging_root / run_id / "gamesl" / "game"
    aligned.mkdir(parents=True)
    gamesl.mkdir(parents=True)
    approval = config.publish_staging_root / run_id / ".STAGING_APPROVED.json"
    contract_rows = []
    contract_digest = __import__("hashlib").sha256(
        json.dumps(contract_rows, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    approval.write_text(json.dumps({
        "schema": "gamedata-staging-approval-v2",
        "status": "STAGING_APPROVED", "game": "game", "run_id": run_id,
        "accepted_root": str(config.archive_root / "game"),
        "staged_aligned_root": str(aligned),
        "staged_gamesl_root": str(gamesl),
        "pair_digest": "0" * 64,
        "nvv_contract": {
            "schema": "nvv-cross-tier-contract-v1", "status": "verified",
            "pair_contracts": contract_rows, "digest": contract_digest,
        },
    }), encoding="utf-8")
    result = _require_staging_approval(
        config, config.games[0], config.archive_root / "game", run_id=run_id)
    assert result == (approval, run_id)


def test_publish_approval_rejects_legacy_missing_nvv_contract(tmp_path):
    config = RebuildConfig(
        source_root=tmp_path / "source", staging_root=tmp_path / "stage",
        archive_root=tmp_path / "archive", output_root=tmp_path / "out",
        gamesl_root=tmp_path / "gamesl",
        games=(GameInput("game", tmp_path / "source", "Game"),),
        publish_staging_root=tmp_path / "publish-staging",
        publish_archive_root=tmp_path / "publish-archive")
    run_id = "run-legacy"
    aligned = config.publish_staging_root / run_id / "aligned" / "game"
    gamesl = config.publish_staging_root / run_id / "gamesl" / "game"
    aligned.mkdir(parents=True)
    gamesl.mkdir(parents=True)
    approval = config.publish_staging_root / run_id / ".STAGING_APPROVED.json"
    approval.write_text(json.dumps({
        "schema": "gamedata-staging-approval-v1", "status": "STAGING_APPROVED",
        "game": "game", "run_id": run_id,
        "accepted_root": str(config.archive_root / "game"),
        "staged_aligned_root": str(aligned), "staged_gamesl_root": str(gamesl),
        "pair_digest": "0" * 64}), encoding="utf-8")
    with pytest.raises(PermissionError, match="STAGING_APPROVED"):
        _require_staging_approval(
            config, config.games[0], config.archive_root / "game", run_id=run_id)
