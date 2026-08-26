import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ctc_prealign import (  # noqa: E402
    _commit_all_gpu_candidate,
    _prepare_dictionary_candidate,
)


def _transaction_fixture(tmp_path):
    live_output = tmp_path / "output"
    live_output.mkdir()
    (live_output / "old.TextGrid").write_text("old", encoding="utf-8")

    candidate_output = tmp_path / ".output.merge"
    candidate_output.mkdir()
    (candidate_output / "new.TextGrid").write_text("new", encoding="utf-8")

    dict_path = tmp_path / "dict.txt"
    dict_path.write_text("old old\n", encoding="utf-8")
    dict_candidate, new_tokens = _prepare_dictionary_candidate(
        dict_path,
        [{"_words": [{"word": "hello"}]}],
        no_update=False,
    )
    assert dict_candidate is not None
    assert new_tokens == ["hello"]
    assert dict_path.read_text(encoding="utf-8") == "old old\n"
    return live_output, candidate_output, dict_path, dict_candidate


def _commit_args(live_output, candidate_output, dict_path, dict_candidate):
    return {
        "live_output": live_output,
        "candidate_output": candidate_output,
        "old_output_backup": live_output.with_name("output.partial"),
        "dict_path": dict_path,
        "dict_candidate": dict_candidate,
    }


def test_all_gpu_commit_success_publishes_validated_pair(tmp_path):
    live, candidate, dictionary, dict_candidate = _transaction_fixture(tmp_path)

    old_backup, dict_backup = _commit_all_gpu_candidate(
        **_commit_args(live, candidate, dictionary, dict_candidate)
    )

    assert (live / "new.TextGrid").read_text(encoding="utf-8") == "new"
    assert not (live / "old.TextGrid").exists()
    assert (old_backup / "old.TextGrid").read_text(encoding="utf-8") == "old"
    assert "hello hello\n" in dictionary.read_text(encoding="utf-8")
    assert dict_backup is not None and dict_backup.read_text(encoding="utf-8") == "old old\n"


def test_all_gpu_commit_output_publish_failure_restores_old_pair(tmp_path, monkeypatch):
    live, candidate, dictionary, dict_candidate = _transaction_fixture(tmp_path)
    import ctc_prealign

    real_replace = ctc_prealign.os.replace

    def fail_output_publish(source, destination):
        if Path(source) == candidate and Path(destination) == live:
            raise OSError("injected output publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(ctc_prealign.os, "replace", fail_output_publish)
    with pytest.raises(OSError, match="output publish failure"):
        _commit_all_gpu_candidate(
            **_commit_args(live, candidate, dictionary, dict_candidate)
        )

    assert (live / "old.TextGrid").read_text(encoding="utf-8") == "old"
    assert not (live / "new.TextGrid").exists()
    assert dictionary.read_text(encoding="utf-8") == "old old\n"
    assert candidate.exists()


def test_all_gpu_commit_dictionary_publish_failure_restores_old_pair(tmp_path, monkeypatch):
    live, candidate, dictionary, dict_candidate = _transaction_fixture(tmp_path)
    import ctc_prealign

    real_replace = ctc_prealign.os.replace

    def fail_dictionary_publish(source, destination):
        if Path(source) == dict_candidate and Path(destination) == dictionary:
            raise OSError("injected dictionary publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(ctc_prealign.os, "replace", fail_dictionary_publish)
    with pytest.raises(OSError, match="dictionary publish failure"):
        _commit_all_gpu_candidate(
            **_commit_args(live, candidate, dictionary, dict_candidate)
        )

    assert (live / "old.TextGrid").read_text(encoding="utf-8") == "old"
    assert not (live / "new.TextGrid").exists()
    assert dictionary.read_text(encoding="utf-8") == "old old\n"
    assert dict_candidate.exists()
    assert list(tmp_path.glob(".output.merge.FAILED-*"))

