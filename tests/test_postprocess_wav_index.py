from __future__ import annotations

from pathlib import Path

from scripts.postprocess_textgrids import _build_wav_index, _find_wav


def test_wav_index_handles_flat_and_nested_candidates(tmp_path: Path):
    flat = tmp_path / "flat.wav"
    nested = tmp_path / "nested" / "deep.wav"
    nested.parent.mkdir()
    flat.write_bytes(b"flat")
    nested.write_bytes(b"nested")

    index = _build_wav_index(tmp_path)

    assert index == {"flat": flat, "deep": nested}
    assert _find_wav("flat", tmp_path, index) == flat
    assert _find_wav("deep", tmp_path, index) == nested


def test_wav_index_prefers_top_level_and_is_deterministic_for_nested_duplicates(
        tmp_path: Path):
    top = tmp_path / "dup.wav"
    first_nested = tmp_path / "a" / "dup.wav"
    second_nested = tmp_path / "b" / "dup.wav"
    first_nested.parent.mkdir()
    second_nested.parent.mkdir()
    top.write_bytes(b"top")
    first_nested.write_bytes(b"a")
    second_nested.write_bytes(b"b")

    assert _build_wav_index(tmp_path)["dup"] == top

    top.unlink()
    assert _build_wav_index(tmp_path)["dup"] == first_nested


def test_indexed_wav_disappearance_falls_back_to_recursive_lookup(tmp_path: Path):
    indexed = tmp_path / "vanishing.wav"
    fallback = tmp_path / "nested" / "vanishing.wav"
    fallback.parent.mkdir()
    indexed.write_bytes(b"indexed")
    fallback.write_bytes(b"fallback")
    index = _build_wav_index(tmp_path)
    assert index["vanishing"] == indexed

    indexed.unlink()

    assert _find_wav("vanishing", tmp_path, index) == fallback
