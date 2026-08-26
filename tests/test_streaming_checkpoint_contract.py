"""Focused checkpoint identity and no-work result contracts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import streaming_pipeline as streaming


def _args(tmp_path: Path, config: dict | None = None, **overrides):
    values = {
        "batch_size": 2,
        "parallel_datasets": 1,
        "mfa_jobs": None,
        "mfa_en_jobs": None,
        "gpus": 1,
        "pipelined": False,
        "_config": config or {"streaming": {"batch_size": 2}},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cache(*datasets):
    return {"datasets": list(datasets), "output_root": "output"}


def _write_checkpoint(path: Path, identity: dict, completed=None):
    streaming._save_checkpoint(path, set(completed or []), set(), identity)


def test_checkpoint_identity_is_order_stable_and_resume_succeeds(tmp_path):
    first = {"name": "b", "audio_dir": "b", "stems": ["2", "1"]}
    second = {"name": "a", "audio_dir": "a", "stems": ["3"]}
    args = _args(tmp_path)
    identity_a = streaming._checkpoint_identity(_cache(first, second), args)
    identity_b = streaming._checkpoint_identity(_cache(second, first), args)
    assert identity_a == identity_b

    checkpoint = tmp_path / "run.checkpoint.json"
    _write_checkpoint(checkpoint, identity_a, completed=["a"])
    assert streaming._load_checkpoint(checkpoint, identity_b) == {"a"}


@pytest.mark.parametrize("mutation", ["input", "config", "model"])
def test_checkpoint_identity_change_requires_recovery(tmp_path, mutation):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model-v1")
    config = {"model_path": str(model), "streaming": {"batch_size": 2}}
    args = _args(tmp_path, config=config)
    cache = _cache({"name": "a", "audio_dir": "a", "stems": ["1", "2"]})
    identity = streaming._checkpoint_identity(cache, args)
    checkpoint = tmp_path / "run.checkpoint.json"
    _write_checkpoint(checkpoint, identity, completed=["a"])

    if mutation == "input":
        cache["datasets"][0]["stems"].append("3")
    elif mutation == "config":
        args._config["streaming"]["batch_size"] = 3
    else:
        model.write_bytes(b"model-v2")

    changed = streaming._checkpoint_identity(cache, args)
    with pytest.raises(RuntimeError, match="recovery required"):
        streaming._load_checkpoint(checkpoint, changed)


def test_legacy_checkpoint_fails_closed_with_recovery_message(tmp_path):
    checkpoint = tmp_path / "legacy.checkpoint.json"
    checkpoint.write_text(json.dumps({"completed": ["a"]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy checkpoint requires recovery"):
        streaming._load_checkpoint(checkpoint)


def test_legacy_batch_progress_fails_closed(tmp_path):
    checkpoint = tmp_path / "run.cache.checkpoint.json"
    progress = checkpoint.with_name("run.cache.checkpoint.batch_progress.json")
    progress.write_text(json.dumps({"a": {"done": 1, "fail": 0, "total": 1}}),
                        encoding="utf-8")
    with pytest.raises(RuntimeError, match="recovery required"):
        streaming._load_batch_progress(checkpoint)


def test_run_batch_reports_already_completed_as_success(tmp_path):
    cache_path = tmp_path / "run.cache.json"
    cache = _cache({"name": "a", "audio_dir": "a", "stems": ["1"]})
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    args = _args(
        tmp_path,
        batch_size=2,
        batch_cache=cache_path,
        no_resume=False,
        limit_datasets=0,
    )
    identity = streaming._checkpoint_identity(cache, args)
    _write_checkpoint(cache_path.with_name("run.cache.checkpoint.json"), identity,
                      completed=["a"])
    assert streaming.run_batch(args) is True


@pytest.mark.parametrize(("runner_result", "expected_rc"), [(True, 0), (False, 1)])
def test_streaming_main_maps_batch_result_to_cli_exit(monkeypatch, tmp_path,
                                                       runner_result, expected_rc):
    monkeypatch.setattr(streaming, "run_batch", lambda _args: runner_result)
    monkeypatch.setattr(sys, "argv", [
        "streaming_pipeline.py", "--batch-cache", str(tmp_path / "cache.json"),
        "--local-work", str(tmp_path / "work"), "--gpus", "1",
    ])
    assert streaming.main() == expected_rc
