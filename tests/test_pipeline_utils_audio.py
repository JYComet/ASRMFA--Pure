from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipeline_utils import (
    _axis_audio_metadata,
    normalize_authority_reference_numerals,
)


def _write_float_wav(path: Path, *, frames: int, sample_rate: int) -> None:
    samples = b"\0\0\0\0" * frames
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    fact = struct.pack("<I", frames)
    chunks = [
        b"fmt " + struct.pack("<I", len(fmt)) + fmt,
        b"fact" + struct.pack("<I", len(fact)) + fact,
        b"data" + struct.pack("<I", len(samples)) + samples,
    ]
    path.write_bytes(
        b"RIFF" + struct.pack("<I", 4 + sum(map(len, chunks)))
        + b"WAVE" + b"".join(chunks)
    )


def test_axis_audio_metadata_supports_ieee_float_and_pcm_wav(tmp_path: Path):
    float_wav = tmp_path / "float.wav"
    _write_float_wav(float_wav, frames=12_000, sample_rate=16_000)
    float_meta = _axis_audio_metadata(float_wav)
    assert float_meta["duration_s"] == 0.75
    assert float_meta["sample_rate"] == 16_000
    assert float_meta["frames"] == 12_000
    assert float_meta["channels"] == 1
    assert float_meta["sample_width"] == 4

    pcm_wav = tmp_path / "pcm.wav"
    with wave.open(str(pcm_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 8_000)
    pcm_meta = _axis_audio_metadata(pcm_wav)
    assert pcm_meta["duration_s"] == 0.5
    assert pcm_meta["sample_rate"] == 16_000
    assert pcm_meta["frames"] == 8_000
    assert pcm_meta["channels"] == 1
    assert pcm_meta["sample_width"] == 2


def test_authority_reference_numerals_are_scoped_and_reported():
    normalized, report = normalize_authority_reference_numerals(
        "target1 target2 rui4 <BREATHING> OK2 ABC1 12", return_report=True)

    assert normalized == "target一 target二 rui4 <BREATHING> OK2 ABC1 十二"
    assert report["schema"] == "reference-numeral-normalization-v1"
    assert report["changed"] is True
    assert [item["replacement"] for item in report["mappings"]] == [
        "target一", "target二", "十二"]


def test_authority_reference_numerals_use_an2cn_transform_without_touching_tokens():
    calls = []

    def transform(value, mode):
        calls.append((value, mode))
        return {"1": "一", "2": "二", "12": "十二"}.get(value.strip(), value)

    normalized = normalize_authority_reference_numerals(
        "target1 target2 rui4 OK2", transform)
    assert normalized == "target一 target二 rui4 OK2"
    assert calls[:2] == [("1", "an2cn"), ("2", "an2cn")]
    assert all(mode == "an2cn" for _, mode in calls)
