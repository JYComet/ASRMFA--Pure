from pathlib import Path

import pytest

from reference_inputs import ReferenceInputError, load_reference_texts, resolve_references


def test_nested_exact_reference_wins_over_compatibility_suffix(tmp_path: Path):
    (tmp_path / "episode" / "a.txt").parent.mkdir()
    (tmp_path / "episode" / "a.txt").write_text("exact", encoding="utf-8")
    (tmp_path / "episode" / "a_qwen3-api.txt").write_text("legacy", encoding="utf-8")

    assert resolve_references(tmp_path, ["a"]) == {"a": tmp_path / "episode" / "a.txt"}
    assert load_reference_texts(tmp_path, ["a"]) == {"a": "exact"}


def test_equal_priority_nested_reference_collision_is_rejected(tmp_path: Path):
    for folder in ("one", "two"):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "a_qwen3.txt").write_text(folder, encoding="utf-8")

    with pytest.raises(ReferenceInputError, match="ambiguous"):
        resolve_references(tmp_path, ["a"])


def test_audio_stem_ending_in_legacy_suffix_keeps_exact_reference(tmp_path: Path):
    reference = tmp_path / "episode" / "foo_qwen3.txt"
    reference.parent.mkdir()
    reference.write_text("exact transcript", encoding="utf-8")

    assert resolve_references(tmp_path, ["foo_qwen3"]) == {
        "foo_qwen3": reference,
    }
    assert load_reference_texts(tmp_path, ["foo_qwen3"]) == {
        "foo_qwen3": "exact transcript",
    }
