from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ctc_prealign import _build_txt_index, _wav_duration_s


def _write_float_wav(path: Path, *, frames: int, sample_rate: int) -> None:
    samples = b"\0\0\0\0" * frames
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    fact = struct.pack("<I", frames)
    chunks = [b"fmt " + struct.pack("<I", len(fmt)) + fmt,
              b"fact" + struct.pack("<I", len(fact)) + fact,
              b"data" + struct.pack("<I", len(samples)) + samples]
    riff_size = 4 + sum(len(chunk) for chunk in chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + b"".join(chunks))


def test_wav_duration_supports_ieee_float_and_pcm(tmp_path: Path):
    float_wav = tmp_path / "float.wav"
    _write_float_wav(float_wav, frames=12_000, sample_rate=16_000)
    assert _wav_duration_s(float_wav) == 0.75

    pcm_wav = tmp_path / "pcm.wav"
    with wave.open(str(pcm_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 8_000)
    assert _wav_duration_s(pcm_wav) == 0.5


def test_reference_index_includes_nested_authority_texts(tmp_path: Path):
    shallow = tmp_path / "speaker" / "shallow.txt"
    nested = tmp_path / "ria新增" / "ria" / "nested.txt"
    shallow.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    shallow.write_text("浅\n", encoding="utf-8")
    nested.write_text("深\n", encoding="utf-8")

    index = _build_txt_index(tmp_path)

    assert index["shallow"] == shallow
    assert index["nested"] == nested
