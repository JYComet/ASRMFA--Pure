import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


from postprocess_textgrids import (Interval, TextGrid, Tier, parse_textgrid,
                                   write_textgrid)  # noqa: E402
from verify_gamedata_publish_staging import (  # noqa: E402
    _detect_verification_edge_silence,
    fingerprint_tree,
    verify_game_staging,
)


TIERS = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")


def test_verification_edge_ignores_isolated_subframe_noise() -> None:
    sr = 8000
    audio = np.concatenate([
        np.zeros(4000, dtype=np.float32),
        np.full(800, 0.25, dtype=np.float32),
    ])
    audio[3600] = 0.002

    assert _detect_verification_edge_silence(
        audio, sr, silence_threshold=0.001) == pytest.approx(0.5, abs=0.01)


def test_tree_fingerprint_changes_with_relative_path_or_size(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "speaker").mkdir(parents=True)
    item = root / "speaker" / "line.wav"
    item.write_bytes(b"audio")
    first = fingerprint_tree(root)

    item.rename(root / "speaker" / "renamed.wav")
    renamed = fingerprint_tree(root)
    (root / "speaker" / "renamed.wav").write_bytes(b"longer-audio")
    resized = fingerprint_tree(root)

    assert first["files"] == renamed["files"] == resized["files"] == 1
    assert first["metadata_sha256"] != renamed["metadata_sha256"]
    assert renamed["metadata_sha256"] != resized["metadata_sha256"]


def _fixture(tmp_path: Path, *, head_silence: float = 0.5):
    accepted = tmp_path / "accepted" / "game"
    aligned = tmp_path / "publish" / "aligned" / "game"
    gamesl = tmp_path / "publish" / "gamesl" / "game"
    speaker = "Alice"
    stem = "line"
    accepted.mkdir(parents=True)
    aligned.joinpath(speaker).mkdir(parents=True)
    gamesl.joinpath(speaker).mkdir(parents=True)

    duration = head_silence + 0.5 + 0.5
    tiers = [Tier(name, 0.0, duration, [
        Interval(0.0, head_silence, ""),
        Interval(head_silence, head_silence + 0.5, "line"),
        Interval(head_silence + 0.5, duration, ""),
    ]) for name in TIERS]
    grid = TextGrid(0.0, duration, tiers)
    write_textgrid(grid, accepted / f"{stem}.TextGrid")
    write_textgrid(grid, aligned / speaker / f"{stem}.TextGrid")

    sr = 8192
    audio = np.concatenate([
        np.zeros(round(head_silence * sr), dtype=np.float32),
        np.full(round(0.5 * sr), 0.25, dtype=np.float32),
        np.zeros(round(0.5 * sr), dtype=np.float32),
    ])
    sf.write(gamesl / speaker / f"{stem}.wav", audio, sr, subtype="PCM_16")
    manifest = accepted / ".rebuild_manifest.json"
    manifest.write_text(json.dumps({"items": [{
        "stem": stem,
        "source": f"/source/game/{speaker}/{stem}.ogg",
    }]}), encoding="utf-8")
    return accepted, aligned, gamesl, manifest


def test_verify_game_staging_accepts_exact_five_tier_pcm16_pair(tmp_path: Path):
    accepted, aligned, gamesl, manifest = _fixture(tmp_path)

    receipt = verify_game_staging(
        game="game", run_id="run-1", accepted_root=accepted,
        manifest_path=manifest, aligned_root=aligned, gamesl_root=gamesl,
        workers=1,
    )

    assert receipt["status"] == "STAGING_APPROVED"
    assert receipt["pair_count"] == 1
    assert receipt["speaker_count"] == 1
    assert receipt["duration_seconds"] == pytest.approx(1.5)
    assert len(receipt["pair_digest"]) == 64
    assert receipt["tier_names"] == list(TIERS)


def test_verify_game_staging_rejects_raw_only_nvv(tmp_path: Path):
    accepted, aligned, gamesl, manifest = _fixture(tmp_path)
    grid = parse_textgrid(aligned / "Alice" / "line.TextGrid")
    next(tier for tier in grid.tiers if tier.name == "raw_text").intervals[1].text = (
        "line <BREATHING>")
    write_textgrid(grid, aligned / "Alice" / "line.TextGrid")
    with pytest.raises(ValueError, match="cross_tier_nvv_sequence_mismatch"):
        verify_game_staging(
            game="game", run_id="run-1", accepted_root=accepted,
            manifest_path=manifest, aligned_root=aligned, gamesl_root=gamesl,
            workers=1,
        )


def test_verify_game_staging_accepts_matching_ordered_repeated_nvv_contract(
        tmp_path: Path):
    accepted, aligned, gamesl, manifest = _fixture(tmp_path)
    labels = {
        "raw_text": "你 [Breathing] [Breathing] [Cough] 好",
        "pinyin": "ni3 <BREATHING> <BREATHING> <COUGH> hao3",
        "hanzi": "你 <BREATHING> <BREATHING> <COUGH> 好",
        "words": "ni3 <BREATHING> <BREATHING> <COUGH> hao3",
        "pinyin_phones": "n <BREATHING> <BREATHING> <COUGH> h",
    }
    for grid_path in (accepted / "line.TextGrid",
                      aligned / "Alice" / "line.TextGrid"):
        grid = parse_textgrid(grid_path)
        for tier in grid.tiers:
            tier.intervals = [Interval(0.0, grid.xmax, labels[tier.name])]
        write_textgrid(grid, grid_path)

    receipt = verify_game_staging(
        game="game", run_id="run-nvv", accepted_root=accepted,
        manifest_path=manifest, aligned_root=aligned, gamesl_root=gamesl,
        workers=1,
    )
    assert receipt["schema"] == "gamedata-staging-approval-v2"
    assert receipt["status"] == "STAGING_APPROVED"
    assert receipt["nvv_contract"]["status"] == "verified"
    assert len(receipt["nvv_contract"]["pair_contracts"]) == 1


def test_verify_game_staging_rejects_wrong_speaker_path(tmp_path: Path):
    accepted, aligned, gamesl, manifest = _fixture(tmp_path)
    aligned.joinpath("Alice", "line.TextGrid").rename(
        aligned.joinpath("line.TextGrid"))
    gamesl.joinpath("Alice", "line.wav").rename(gamesl.joinpath("line.wav"))

    with pytest.raises(ValueError, match="speaker path mismatch"):
        verify_game_staging(
            game="game", run_id="run-1", accepted_root=accepted,
            manifest_path=manifest, aligned_root=aligned, gamesl_root=gamesl,
            workers=1,
        )


def test_verify_game_staging_rejects_short_edge_silence(tmp_path: Path):
    accepted, aligned, gamesl, manifest = _fixture(
        tmp_path, head_silence=0.1)

    with pytest.raises(ValueError, match="head silence"):
        verify_game_staging(
            game="game", run_id="run-1", accepted_root=accepted,
            manifest_path=manifest, aligned_root=aligned, gamesl_root=gamesl,
            workers=1, frame_length=80,
        )


def test_verify_game_staging_rejects_textgrid_axis_mismatch(tmp_path: Path):
    accepted, aligned, gamesl, manifest = _fixture(tmp_path)
    grid = aligned / "Alice" / "line.TextGrid"
    grid.write_text(grid.read_text(encoding="utf-8").replace(
        "xmax = 1.5", "xmax = 1.3", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="TextGrid duration mismatch"):
        verify_game_staging(
            game="game", run_id="run-1", accepted_root=accepted,
            manifest_path=manifest, aligned_root=aligned, gamesl_root=gamesl,
            workers=1,
        )
