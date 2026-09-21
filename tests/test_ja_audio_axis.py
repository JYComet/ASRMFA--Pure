import wave
from pathlib import Path

import numpy as np

from scripts.ja_audio import (inspect_wav, make_audio_receipt,
                               prepare_alignment_wav, prepare_training_wav,
                               transform_audio)


def _wav(path: Path, rate=8000, channels=2):
    samples = np.zeros((rate // 10, channels), dtype=np.int16)
    samples[:, 0] = 1000
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())


def _wav24(path: Path, rate=44100, channels=2):
    """Write deterministic little-endian signed PCM24 for axis tests."""
    frames = max(1, rate // 100)
    values = np.zeros((frames, channels), dtype=np.int32)
    values[:, 0] = 1 << 20
    if channels > 1:
        values[:, 1] = -(1 << 19)
    packed = bytearray()
    for value in values.reshape(-1):
        packed.extend(int(value).to_bytes(3, "little", signed=True))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(3)
        out.setframerate(rate)
        out.writeframes(bytes(packed))


def test_alignment_is_pcm16_mono_16k_and_axis_is_integer(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "alignment.wav"
    _wav(source)
    receipt = prepare_alignment_wav(source, target)
    header = inspect_wav(target)
    assert header["sample_rate"] == 16000
    assert header["channels"] == 1
    assert header["sample_width"] == 2
    assert receipt["sample_transform"]["source_start"] == 0
    assert receipt["sample_transform"]["output_start"] == 0
    assert receipt["sample_transform"]["output_frames"] == 1600
    assert receipt["sample_transform"]["method"] == "scipy_resample_poly_v1"
    assert receipt["sample_transform"]["frame_policy"] == "round_half_up_v1"
    assert receipt["sample_transform"]["downmix"] == "mean"
    assert receipt["sample_transform"]["dtype"] == "float64"
    assert receipt["sample_transform"]["pad_before_frames"] == 0
    assert receipt["sample_transform"]["pad_after_frames"] == 0


def test_transform_does_not_delete_low_energy_frames(tmp_path):
    source = tmp_path / "source.wav"
    _wav(source, rate=16000, channels=1)
    data = transform_audio(source, sample_rate=16000)
    assert data["source_frames"] == data["output_frames"] == 1600
    assert data["trimmed_frames"] == 0


def test_same_rate_mono_pcm16_is_bit_exact(tmp_path):
    source = tmp_path / "source.wav"
    values = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(values.tobytes())
    target = tmp_path / "train.wav"
    from scripts.ja_audio import prepare_training_wav
    info = prepare_training_wav(source, target)
    with wave.open(str(target), "rb") as out:
        assert out.readframes(out.getnframes()) == values.tobytes()
    assert info["sample_transform"]["method"] == "pcm16_copy_v1"


def test_44100_stereo_pcm24_records_polyphase_replay_contract(tmp_path):
    source = tmp_path / "source-44k24.wav"
    target = tmp_path / "alignment.wav"
    _wav24(source, rate=44100)
    info = prepare_alignment_wav(source, target, sample_rate=16000)
    transform = info["sample_transform"]
    assert transform["method"] == "scipy_resample_poly_v1"
    assert transform["scipy_version"]
    assert (transform["up"], transform["down"]) == (160, 441)
    assert transform["window"] == {"name": "kaiser", "beta": 5.0}
    assert transform["padtype"] == "constant"
    assert transform["downmix"] == "mean"
    assert transform["dtype"] == "float64"
    assert transform["source_header"]["sample_width"] == 3
    assert transform["source_header"]["channels"] == 2
    assert transform["output_header"]["sample_rate"] == 16000
    assert transform["output_header"]["sample_width"] == 2
    assert transform["crop_start_frames"] == 0
    assert transform["crop_end_frames"] == 0
    assert transform["pad_before_frames"] == 0
    assert transform["pad_after_frames"] == 0
    assert transform["frame_policy"] == "round_half_up_v1"


def test_48000_stereo_pcm24_same_rate_is_explicit_reencode(tmp_path):
    source = tmp_path / "source-48k24.wav"
    target = tmp_path / "alignment.wav"
    _wav24(source, rate=48000)
    info = prepare_alignment_wav(source, target, sample_rate=48000)
    transform = info["sample_transform"]
    assert transform["method"] == "pcm16_reencode_v1"
    assert transform["method"] != "pcm16_copy_v1"
    assert transform["method"] != "scipy_resample_poly_v1"
    assert transform["source_rate"] == transform["target_rate"] == 48000
    assert transform["frame_policy"] == "identity_v1"
    assert transform["downmix"] == "mean"
    assert transform["source_header"]["sample_width"] == 3
    assert transform["output_header"]["sample_width"] == 2


def test_audio_receipt_binds_train_and_alignment_transforms(tmp_path):
    source = tmp_path / "source.wav"
    train = tmp_path / "train.wav"
    alignment = tmp_path / "alignment.wav"
    _wav24(source, rate=44100)
    train_info = prepare_training_wav(source, train)
    alignment_info = prepare_alignment_wav(source, alignment)
    receipt = make_audio_receipt(
        "u1", source, train, alignment,
        train_transform=train_info["sample_transform"],
        alignment_transform=alignment_info["sample_transform"],
    )
    assert receipt["train_transform"]["method"] == "pcm16_reencode_v1"
    assert receipt["alignment_transform"]["method"] == "scipy_resample_poly_v1"
    # Legacy consumers use this alias for the alignment axis.
    assert receipt["sample_transform"] == receipt["alignment_transform"]


def test_audio_stage_rejects_sanitized_uid_collision(tmp_path):
    from scripts.ja_audio import AudioContractError, _safe_stem
    assert _safe_stem("a/b", 0) != _safe_stem("a?b", 1)
