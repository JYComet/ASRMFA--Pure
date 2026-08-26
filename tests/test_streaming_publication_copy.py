"""Dataset publication copy must be linear and hash-exact across shards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import streaming_pipeline as stream


def test_copy_publication_tree_accepts_existing_sibling_shard(tmp_path, monkeypatch):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "new.TextGrid").write_text("new", encoding="utf-8")
    (target / "old.TextGrid").write_text("old", encoding="utf-8")

    original = stream._publication_tree
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(stream, "_publication_tree", counted)
    stream._copy_publication_tree(source, target)

    assert (target / "old.TextGrid").read_text(encoding="utf-8") == "old"
    assert (target / "new.TextGrid").read_text(encoding="utf-8") == "new"
    assert calls.count(target) == 1


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
