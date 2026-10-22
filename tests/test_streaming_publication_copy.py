"""Dataset publication copy must be linear and hash-exact across shards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


from scripts import streaming_pipeline as stream


def test_copy_publication_tree_accepts_existing_sibling_shard(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "new.TextGrid").write_text("new", encoding="utf-8")
    (target / "old.TextGrid").write_text("old", encoding="utf-8")

    manifest = stream._copy_publication_tree(source, target)

    assert (target / "old.TextGrid").read_text(encoding="utf-8") == "old"
    assert (target / "new.TextGrid").read_text(encoding="utf-8") == "new"
    assert manifest == {
        "new.TextGrid": {
            "size": 3,
            "sha256": stream._sha256_file(source / "new.TextGrid"),
        },
    }


def test_copy_publication_tree_hashes_only_current_batch_targets(
        tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    target_hashes = []
    original_hash = stream._sha256_file

    def counted(path):
        digest = original_hash(path)
        if path.parent == target:
            target_hashes.append(path.name)
        return digest

    monkeypatch.setattr(stream, "_sha256_file", counted)
    for index in range(10):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        (source / f"batch-{index}.TextGrid").write_text(
            f"batch-{index}", encoding="utf-8")
        stream._copy_publication_tree(source, target)

    assert target_hashes == [f"batch-{index}.TextGrid" for index in range(10)]


def test_copy_publication_tree_uses_expected_manifest_without_source_hash(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "one.TextGrid").write_text("one", encoding="utf-8")
    expected = stream._publication_tree(source)
    original_hash = stream._sha256_file
    source_hashes = []

    def counted(path):
        if path.parent == source:
            source_hashes.append(path)
        return original_hash(path)

    monkeypatch.setattr(stream, "_sha256_file", counted)
    assert stream._copy_publication_tree(
        source, target, expected_manifest=expected) == expected
    assert source_hashes == []


def test_copy_publication_tree_rejects_symlink_directory(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (real / "nested.TextGrid").write_text("nested", encoding="utf-8")
    (source / "nested").symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        stream._copy_publication_tree(source, target)


def test_copy_publication_tree_rejects_conflicting_hash(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "same.TextGrid").write_text("expected", encoding="utf-8")
    (target / "same.TextGrid").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="publication target hash conflict"):
        stream._copy_publication_tree(source, target)


def test_batch_path_rename_retries_transient_permission_denial(
        tmp_path, monkeypatch):
    source = tmp_path / "batch.tmp"
    target = tmp_path / "batch"
    source.mkdir()
    (source / "receipt.json").write_text("{}", encoding="utf-8")
    original_replace = Path.replace
    calls = 0

    def flaky_replace(path, destination):
        nonlocal calls
        if path == source:
            calls += 1
            if calls == 1:
                raise PermissionError("transient NAS directory handle")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(stream.time, "sleep", lambda _seconds: None)

    stream._replace_batch_path_with_retry(source, target)

    assert calls == 2
    assert not source.exists()
    assert (target / "receipt.json").is_file()
