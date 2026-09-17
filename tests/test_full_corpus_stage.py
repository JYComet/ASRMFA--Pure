from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf

from scripts.full_corpus_inventory import InventoryItem
from scripts.full_corpus_stage import stage_item, verify_padded_wav, publish_gamesl


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _item(tmp_path, source_id="wuwa", padded=True):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.r_[np.zeros(8000), np.full(8000, .1), np.zeros(8000)], 16000)
    text = audio.with_suffix(".lab" if source_id == "wuwa" else ".txt")
    text.write_text("「你好」~")
    return InventoryItem(source_id, "鸣潮" if source_id == "wuwa" else "g",
                         "今汐", audio, "source.wav", "source", "u1", _sha256(audio),
                         audio.stat().st_size, 1.5, text, _sha256(text), text.suffix,
                         "reference", padded)


def test_reference_lab_is_normalized_and_padded_before_flattening(tmp_path):
    item = _item(tmp_path)
    receipt = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    assert receipt.text_path.read_text(encoding="utf-8").strip() == "你好…"
    edge = verify_padded_wav(receipt.pipeline_wav, .5)
    assert edge["head_ok"] and edge["tail_ok"]
    assert receipt.pipeline_wav_sha256 == receipt.gamesl_wav_sha256


def test_existing_gamesl_target_is_replaced_with_rollback_record(tmp_path):
    item = _item(tmp_path)
    receipt = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    target = tmp_path / "GAMESL" / "鸣潮" / "今汐" / "u1.wav"
    target.parent.mkdir(parents=True); target.write_bytes(b"old")
    result = publish_gamesl([receipt], tmp_path / "GAMESL", tmp_path / "rollback")
    assert result["replaced_count"] == 1
    assert len(result["replacements"]) == 1
    assert (tmp_path / "rollback" / "鸣潮" / "今汐" / "u1.wav").read_bytes() == b"old"


def test_padding_verification_is_fail_closed(tmp_path, monkeypatch):
    item = _item(tmp_path)
    def bad_normalize(source, target):
        sf.write(target, np.ones(16000) * .1, 16000, subtype="PCM_16", format="WAV")
    monkeypatch.setattr("scripts.full_corpus_stage._normalize_audio", bad_normalize)
    try:
        stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    except ValueError as exc:
        assert "edge verification failed" in str(exc)
    else:
        raise AssertionError("non-silent edges were accepted")


def test_v5_stage_keeps_pipeline_axis_without_gamesl(tmp_path):
    item = _item(tmp_path, source_id="v5_0707", padded=False)
    receipt = stage_item(item, tmp_path / "flat")
    assert receipt.gamesl_wav is None
    assert receipt.gamesl_wav_sha256 is None
    assert verify_padded_wav(receipt.pipeline_wav, .5)["sample_rate"] == 16000


def test_v5_nonwav_or_multichannel_is_materialized_as_mono_wav(tmp_path):
    source = tmp_path / "source.flac"
    samples = np.stack([np.linspace(-.1, .1, 1600), np.linspace(-.05, .15, 1600)], axis=1)
    sf.write(source, samples, 16000)
    item = InventoryItem("v5_0707", None, "LAria", source, "LAria/source.flac",
                         "source", "u2", _sha256(source), source.stat().st_size, .1,
                         None, None, None, "fallback", False)
    receipt = stage_item(item, tmp_path / "flat")
    info = sf.info(str(receipt.pipeline_wav))
    assert info.format == "WAV"
    assert info.channels == 1
    assert info.frames == 1600


def test_stage_receipt_allows_hash_bound_resume(tmp_path):
    item = _item(tmp_path)
    first = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    second = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    assert first.pipeline_wav_sha256 == second.pipeline_wav_sha256


def test_stage_resume_repairs_tampered_normalized_reference(tmp_path):
    item = _item(tmp_path)
    receipt = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    receipt.text_path.write_text("被篡改", encoding="utf-8")
    repaired = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    assert repaired.text_path.read_text(encoding="utf-8") == "你好…"
    assert repaired.normalized_text == "你好…"


def test_stage_rejects_source_changed_after_preflight(tmp_path):
    item = _item(tmp_path)
    sf.write(item.source_path, np.ones(24000) * .2, 16000)
    try:
        stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    except ValueError as exc:
        assert "frozen source hash drift" in str(exc)
    else:
        raise AssertionError("changed frozen audio was staged")


def test_stage_rejects_reference_changed_after_preflight(tmp_path):
    item = _item(tmp_path)
    item.reference_path.write_text("变化后的文本", encoding="utf-8")
    try:
        stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    except ValueError as exc:
        assert "frozen reference hash drift" in str(exc)
    else:
        raise AssertionError("changed frozen reference was staged")
