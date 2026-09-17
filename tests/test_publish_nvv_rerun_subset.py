import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from postprocess_textgrids import Interval, TextGrid, Tier, parse_textgrid, write_textgrid  # noqa: E402
from publish_nvv_rerun_subset import publish_nvv_rerun_subset  # noqa: E402
from finalize_gamedata_speakers import detect_edge_silence_rms  # noqa: E402


TIERS = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")


def _row(*, game="persona", speaker="Alice", stem="line", status=None):
    row = {
        "game": game,
        "speaker": speaker,
        "stem": stem,
        "expected_nvv_sequence": ["<BREATHING>"],
    }
    if status is not None:
        row["status"] = status
    return row


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _write_grid(root: Path, *, game="persona", speaker="Alice", stem="line",
                duration=1.0, nvv_start=0.0) -> Path:
    tiers = []
    labels = {
        "raw_text": "<BREATHING> 你好",
        "pinyin": "<BREATHING> ni3 hao3",
        "hanzi": "<BREATHING> 你 好",
        "words": "<BREATHING> ni3 hao3",
        "pinyin_phones": "<BREATHING> n i h ao",
    }
    for name in TIERS:
        if name in {"hanzi", "words", "pinyin_phones"}:
            intervals = []
            if nvv_start:
                intervals.append(Interval(0.0, nvv_start, ""))
            intervals.extend((
                Interval(nvv_start, nvv_start + 0.2, "<BREATHING>"),
                Interval(nvv_start + 0.2, duration,
                         labels[name].removeprefix("<BREATHING> ")),
            ))
        else:
            intervals = [Interval(0.0, duration, labels[name])]
        tiers.append(Tier(name, 0.0, duration, intervals))
    path = root / game / speaker / f"{stem}.TextGrid"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_textgrid(TextGrid(0.0, duration, tiers), path)
    return path


def _write_wav(root: Path, *, game="persona", speaker="Alice", stem="line",
               leading_silence=0.0, trailing_silence=0.0) -> Path:
    path = root / game / speaker / f"{stem}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stereo intentionally verifies mono normalization.
    sample_rate = 8000
    mono = np.concatenate((
        np.zeros(round(leading_silence * sample_rate), dtype=np.float32),
        np.full(sample_rate, 0.25, dtype=np.float32),
        np.zeros(round(trailing_silence * sample_rate), dtype=np.float32),
    ))
    audio = np.column_stack((mono, mono))
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path


def _publish(tmp_path: Path, rows: list[dict], *, output: Path | None = None):
    manifest = _write_manifest(tmp_path / "accepted.jsonl", rows)
    grids = tmp_path / "grids"
    wavs = tmp_path / "wavs"
    for row in rows:
        _write_grid(grids, game=row["game"], speaker=row["speaker"], stem=row["stem"])
        _write_wav(wavs, game=row["game"], speaker=row["speaker"], stem=row["stem"])
    return publish_nvv_rerun_subset(
        accepted_manifest_path=manifest,
        textgrid_root=grids,
        wav_root=wavs,
        output_root=output or tmp_path / "对齐0910",
    )


def test_publishes_only_accepted_subset_with_padded_wav_and_shifted_grid(tmp_path: Path):
    output = tmp_path / "对齐0910"

    receipt = _publish(tmp_path, [_row()], output=output)

    grid_path = output / "persona" / "Alice" / "line.TextGrid"
    wav_path = output / "GAMESL" / "persona" / "Alice" / "line.wav"
    assert grid_path.is_file()
    assert wav_path.is_file()
    info = sf.info(wav_path)
    assert (info.format, info.subtype, info.channels) == ("WAV", "PCM_16", 1)
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    assert detect_edge_silence_rms(audio, sample_rate) == pytest.approx(
        0.5, abs=0.011 + 2 / sample_rate)
    assert detect_edge_silence_rms(audio[::-1], sample_rate) == pytest.approx(
        0.5, abs=0.011 + 2 / sample_rate)
    grid = parse_textgrid(grid_path)
    assert grid.xmin == 0.0
    assert grid.xmax == pytest.approx(len(audio) / sample_rate, abs=1e-6)
    for tier in grid.tiers:
        assert tier.xmin == 0.0
        assert tier.xmax == pytest.approx(len(audio) / sample_rate, abs=1e-6)
        assert tier.intervals[0].text == ""
        assert tier.intervals[0].xmax == pytest.approx(
            0.5, abs=0.011 + 2 / sample_rate)
    assert next(tier for tier in grid.tiers if tier.name == "words").intervals[1].text == "<BREATHING>"
    assert receipt["games"]["persona"]["pair_count"] == 1
    assert (output / ".nvv_rerun_publish_receipt.json").is_file()


@pytest.mark.parametrize("leading_silence,trailing_silence", (
    (0.0, 0.0),
    (0.8, 0.8),
    (0.1, 0.7),
))
def test_normalizes_all_edge_shapes_to_half_second_and_preserves_nvv_owner_timing(
        tmp_path: Path, leading_silence: float, trailing_silence: float):
    manifest = _write_manifest(tmp_path / "accepted.jsonl", [_row()])
    grids = tmp_path / "grids"
    wavs = tmp_path / "wavs"
    _write_grid(grids, duration=1.0 + leading_silence + trailing_silence,
                nvv_start=leading_silence)
    _write_wav(wavs, leading_silence=leading_silence,
               trailing_silence=trailing_silence)

    publish_nvv_rerun_subset(
        accepted_manifest_path=manifest, textgrid_root=grids, wav_root=wavs,
        output_root=tmp_path / "对齐0910")

    wav = tmp_path / "对齐0910" / "GAMESL" / "persona" / "Alice" / "line.wav"
    audio, sample_rate = sf.read(wav, dtype="float32")
    assert detect_edge_silence_rms(audio, sample_rate) == pytest.approx(
        0.5, abs=0.011 + 2 / sample_rate)
    assert detect_edge_silence_rms(audio[::-1], sample_rate) == pytest.approx(
        0.5, abs=0.011 + 2 / sample_rate)
    grid = parse_textgrid(
        tmp_path / "对齐0910" / "persona" / "Alice" / "line.TextGrid")
    owners = []
    for name in ("hanzi", "words", "pinyin_phones"):
        tier = next(item for item in grid.tiers if item.name == name)
        owners.append([(interval.xmin, interval.xmax) for interval in tier.intervals
                       if interval.text == "<BREATHING>"])
    assert owners == [owners[0], owners[0], owners[0]]
    assert len(owners[0]) == 1


def test_rejects_nonaccepted_row_before_any_publication(tmp_path: Path):
    output = tmp_path / "对齐0910"

    with pytest.raises(ValueError, match="not accepted"):
        _publish(tmp_path, [_row(status="rejected")], output=output)

    assert not output.exists()


def test_refuses_nonempty_output_root(tmp_path: Path):
    output = tmp_path / "对齐0910"
    output.mkdir()
    (output / "existing").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        _publish(tmp_path, [_row()], output=output)

    assert (output / "existing").read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize("kind", ("missing-wav", "extra-grid"))
def test_rejects_missing_or_extra_source_pair_before_publication(tmp_path: Path, kind: str):
    output = tmp_path / "对齐0910"
    manifest = _write_manifest(tmp_path / "accepted.jsonl", [_row()])
    grids = tmp_path / "grids"
    wavs = tmp_path / "wavs"
    _write_grid(grids)
    _write_wav(wavs)
    if kind == "missing-wav":
        (wavs / "persona" / "Alice" / "line.wav").unlink()
    else:
        _write_grid(grids, stem="unexpected")

    with pytest.raises(ValueError, match="universe"):
        publish_nvv_rerun_subset(
            accepted_manifest_path=manifest, textgrid_root=grids,
            wav_root=wavs, output_root=output)

    assert not output.exists()
