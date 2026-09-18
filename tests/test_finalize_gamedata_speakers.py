import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


from finalize_gamedata_speakers import (  # noqa: E402
    GameSpec,
    build_game_plan,
    detect_edge_silence_rms,
    finalize_game,
    normalize_edge_silence,
    speaker_for_stem,
    dynamic_game_spec,
)
from postprocess_textgrids import parse_textgrid


def _write_wav(path: Path, *, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate([
        np.zeros(int(0.1 * sr), dtype=np.float32),
        np.full(int(0.4 * sr), 0.25, dtype=np.float32),
        np.zeros(int(0.2 * sr), dtype=np.float32),
    ])
    sf.write(path, audio, sr)


def _write_grid(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n\n'
        'xmin = 0\n'
        'xmax = 0.7\n'
        'tiers? <exists>\n'
        'size = 1\n'
        'item []:\n'
        '    item [1]:\n'
        '        class = "IntervalTier"\n'
        '        name = "words"\n'
        '        xmin = 0\n'
        '        xmax = 0.7\n'
        '        intervals: size = 1\n'
        '        intervals [1]:\n'
        '            xmin = 0.1\n'
        '            xmax = 0.5\n'
        '            text = "line"\n',
        encoding="utf-8",
    )


def test_speaker_uses_original_audio_parent_and_defaults() -> None:
    manifest = {
        "line_a": {"audio": "/source/原神/Amber/line_a.wav"},
        "line_b": {"audio": "line_b.wav"},
    }
    assert speaker_for_stem(manifest, "line_a") == "Amber"
    assert speaker_for_stem(manifest, "line_b") == "default"
    assert speaker_for_stem(manifest, "missing") == "default"


def test_normalize_edge_silence_writes_half_second_edges(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    _write_wav(source)

    result = normalize_edge_silence(
        source, target, target_silence_sec=0.5, frame_length=80)

    audio, sr = sf.read(target)
    assert result["time_offset"] == pytest.approx(0.4, abs=0.02)
    assert np.max(np.abs(audio[: int(0.49 * sr)])) == 0
    assert np.max(np.abs(audio[-int(0.49 * sr) :])) == 0


def test_normalize_edge_silence_is_not_quantized_to_1024_sample_frames(
        tmp_path: Path) -> None:
    sr = 44100
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    audio = np.concatenate([
        np.zeros(round(0.553 * sr), dtype=np.float32),
        np.full(round(0.4 * sr), 0.25, dtype=np.float32),
        np.zeros(round(0.537 * sr), dtype=np.float32),
    ])
    sf.write(source, audio, sr, subtype="PCM_16")

    normalize_edge_silence(source, target)

    normalized, actual_sr = sf.read(target, dtype="float32")
    assert detect_edge_silence_rms(normalized, actual_sr) == pytest.approx(
        0.5, abs=1 / actual_sr)
    assert detect_edge_silence_rms(normalized[::-1], actual_sr) == pytest.approx(
        0.5, abs=1 / actual_sr)


def test_build_plan_rejects_two_stems_with_same_destination(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    published = tmp_path / "published"
    gamesl = tmp_path / "gamesl"
    manifest_path = tmp_path / "manifest.json"
    original = tmp_path / "source" / "game" / "Speaker" / "a.wav"
    _write_grid(accepted / "a.TextGrid")
    _write_grid(accepted / "nested" / "a.TextGrid")
    _write_wav(original)
    manifest_path.write_text(
        json.dumps({"a": {"audio": str(original)}}),
        encoding="utf-8",
    )
    spec = GameSpec("game", accepted, published, manifest_path, None, gamesl)

    with pytest.raises(ValueError, match="duplicate accepted stem"):
        build_game_plan(spec)


def test_build_plan_uses_staged_audio_when_original_is_missing(
        tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    published = tmp_path / "published"
    gamesl = tmp_path / "gamesl"
    stage = tmp_path / "stage"
    manifest_path = stage / ".stage_manifest.json"
    _write_grid(accepted / "a.TextGrid")
    _write_wav(stage / "a.wav")
    manifest_path.write_text(
        json.dumps({"a": {"audio": "/missing/game/Speaker/a.wav"}}),
        encoding="utf-8",
    )
    spec = GameSpec("game", accepted, published, manifest_path, None, gamesl)

    plan = build_game_plan(spec)

    assert plan[0].speaker == "Speaker"
    assert plan[0].audio_source == stage / "a.wav"
    assert plan[0].generate_padding is True


def test_build_plan_prefers_validated_staged_wav_over_existing_original(
        tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    published = tmp_path / "published"
    gamesl = tmp_path / "gamesl"
    stage = tmp_path / "stage"
    manifest_path = stage / ".rebuild_manifest.json"
    original = tmp_path / "source" / "game" / "Speaker" / "line.m4a"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"source container")
    _write_grid(accepted / "line.TextGrid")
    _write_wav(stage / "line.wav")
    manifest_path.write_text(json.dumps({
        "line": {"audio": str(original)},
    }), encoding="utf-8")
    spec = GameSpec("game", accepted, published, manifest_path, None, gamesl)

    plan = build_game_plan(spec)

    assert plan[0].speaker == "Speaker"
    assert plan[0].audio_source == stage / "line.wav"


def test_finalize_moves_flat_grid_and_copies_existing_padded_audio(tmp_path: Path) -> None:
    published = tmp_path / "published" / "game"
    gamesl = tmp_path / "GAMESL"
    padded = tmp_path / "padded"
    manifest_path = tmp_path / "manifest.json"
    _write_grid(published / "line.TextGrid")
    _write_wav(padded / "line.wav")
    manifest_path.write_text(
        json.dumps({"line": {"audio": "/source/game/Alice/line.wav"}}),
        encoding="utf-8",
    )
    spec = GameSpec("game", published, published, manifest_path, padded, gamesl)

    receipt = finalize_game(spec, workers=1)

    assert not (published / "line.TextGrid").exists()
    assert (published / "Alice" / "line.TextGrid").is_file()
    assert (gamesl / "game" / "Alice" / "line.wav").is_file()
    assert receipt["accepted_count"] == 1
    assert receipt["speaker_counts"] == {"Alice": 1}
    assert receipt["generated_padded_count"] == 0

    second = finalize_game(spec, workers=1)
    assert second["accepted_count"] == 1


def test_finalize_generates_padding_and_shifts_grid_copy(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    published = tmp_path / "published" / "game"
    gamesl = tmp_path / "GAMESL"
    original = tmp_path / "source" / "game" / "Bob" / "line.wav"
    manifest_path = tmp_path / "manifest.json"
    _write_grid(accepted / "line.TextGrid")
    _write_wav(original)
    manifest_path.write_text(
        json.dumps({"line": {"audio": str(original)}}), encoding="utf-8")
    spec = GameSpec("game", accepted, published, manifest_path, None, gamesl)

    receipt = finalize_game(spec, workers=1)

    wav = gamesl / "game" / "Bob" / "line.wav"
    grid = parse_textgrid(published / "Bob" / "line.TextGrid")
    duration = sf.info(wav).duration
    assert grid.xmin == 0
    assert grid.xmax == pytest.approx(duration, abs=1e-6)
    assert grid.tiers[0].xmin == 0
    assert grid.tiers[0].xmax == pytest.approx(duration, abs=1e-6)
    assert grid.tiers[0].intervals[0].text == ""
    assert grid.tiers[0].intervals[0].xmax >= 0.49
    assert grid.tiers[0].intervals[-1].text == ""
    assert grid.tiers[0].intervals[-1].xmax == pytest.approx(duration, abs=1e-6)
    assert receipt["generated_padded_count"] == 1

    second = finalize_game(spec, workers=1)
    assert second["accepted_count"] == 1


def test_finalize_rejects_conflicting_existing_destination(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    published = tmp_path / "published" / "game"
    gamesl = tmp_path / "GAMESL"
    padded = tmp_path / "padded"
    manifest_path = tmp_path / "manifest.json"
    _write_grid(accepted / "line.TextGrid")
    _write_grid(published / "Alice" / "line.TextGrid")
    (published / "Alice" / "line.TextGrid").write_text("conflict", encoding="utf-8")
    _write_wav(padded / "line.wav")
    manifest_path.write_text(
        json.dumps({"line": {"audio": "/source/game/Alice/line.wav"}}),
        encoding="utf-8",
    )
    spec = GameSpec("game", accepted, published, manifest_path, padded, gamesl)

    with pytest.raises(FileExistsError, match="conflicting destination"):
        finalize_game(spec, workers=1)


def test_manifest_items_format_maps_source_parent_to_speaker(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted"
    stage = tmp_path / "stage"
    source = tmp_path / "source" / "Alice" / "line.wav"
    _write_grid(accepted / "line.TextGrid")
    _write_wav(source)
    manifest = stage / ".rebuild_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"items": [{
        "stem": "line", "audio": str(stage / "line.wav"), "source": str(source),
        "text_mode": "reference",
    }]}), encoding="utf-8")
    spec = GameSpec("game", accepted, tmp_path / "published", manifest, None, tmp_path / "gamesl")
    assert build_game_plan(spec)[0].speaker == "Alice"


def test_dynamic_game_spec_reads_rebuild_task_config(tmp_path: Path) -> None:
    config = tmp_path / "task.yaml"
    config.write_text(
        "allow_test_roots: true\n"
        "source_root: /tmp/source\n"
        "staging_root: /tmp/stage\n"
        "archive_root: /tmp/archive\n"
        "output_root: /tmp/out\n"
        "gamesl_root: /tmp/gamesl\n"
        "games:\n"
        "  - codename: persona\n"
        "    source_dir: /tmp/source/persona\n"
        "    source_name: 女神异闻录\n", encoding="utf-8")
    spec = dynamic_game_spec(config, "persona", tmp_path / "accepted")
    assert spec.game == "persona"
    assert spec.published_root == Path("/tmp/out/persona")
    assert spec.manifest_path == Path("/tmp/stage/persona/.rebuild_manifest.json")


def test_publish_rejects_flat_pre_alignment_archive(tmp_path: Path) -> None:
    from rebuild_gamedata import GameInput, RebuildConfig, publish_game
    source = tmp_path / "source"
    _write_wav(source / "line.wav")
    game = GameInput("game", source, "Game")
    config = RebuildConfig(source, tmp_path / "stage", tmp_path / "archive",
                           tmp_path / "out", tmp_path / "gamesl", (game,),
                           allow_test_roots=True)
    archived = tmp_path / "archive" / "game"
    archived.mkdir(parents=True)
    _write_wav(archived / "line.wav")
    with pytest.raises(ValueError, match="TextGrid"):
        publish_game(config, game, archived, journal_path=tmp_path / "journal")
