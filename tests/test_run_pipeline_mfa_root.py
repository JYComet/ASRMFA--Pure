"""MFA command-history isolation contracts for direct run_pipeline runs."""

import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import pipeline_utils, run_pipeline


def test_default_mfa_root_is_created_below_workspace_and_reaches_child_env(
        tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("MFA_ROOT_DIR", raising=False)

    assert run_pipeline._configure_mfa_root_dir(workspace) == 0
    expected = workspace.resolve() / "mfa_root"
    assert expected.is_dir()
    assert os.environ["MFA_ROOT_DIR"] == str(expected)
    child_env = pipeline_utils.get_mfa_env(Path(sys.executable), tmp_path / "models")
    assert child_env["MFA_ROOT_DIR"] == str(expected)


def test_explicit_parent_mfa_root_is_preserved_without_default_creation(
        tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    explicit = tmp_path / "parent-mfa-root"
    monkeypatch.setenv("MFA_ROOT_DIR", str(explicit))

    assert run_pipeline._configure_mfa_root_dir(workspace) == 0
    assert os.environ["MFA_ROOT_DIR"] == str(explicit)
    assert not (workspace / "mfa_root").exists()


def test_mfa_root_creation_failure_fails_closed_without_shared_fallback(
        tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("MFA_ROOT_DIR", raising=False)

    def fail_mkdir(self, *args, **kwargs):
        raise OSError("mkdir denied")

    monkeypatch.setattr(pipeline_utils.Path, "mkdir", fail_mkdir)
    assert run_pipeline._configure_mfa_root_dir(workspace) == 1
    assert "MFA_ROOT_DIR" not in os.environ
    assert not (workspace / "mfa_root").exists()


def test_helper_rejects_existing_non_directory_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "mfa_root").write_text("not a directory", encoding="utf-8")
    monkeypatch.delenv("MFA_ROOT_DIR", raising=False)

    with pytest.raises(OSError, match="ordinary directory"):
        pipeline_utils.ensure_mfa_root_dir(workspace)


def test_mfa_dither_defaults_to_deterministic_zero_and_rejects_bad_values():
    assert pipeline_utils.resolve_mfa_dither() == 0.0
    assert run_pipeline.DEFAULT_CFG["mfa"]["dither"] == 0.0
    assert run_pipeline.DEFAULT_CFG["mfa_en"]["dither"] == 0.0
    for value in (True, -0.1, float("nan"), float("inf"), "not-a-number"):
        with pytest.raises(ValueError, match="finite non-negative"):
            pipeline_utils.resolve_mfa_dither(value)


def test_primary_mfa_command_and_shard_contract_explicitly_disable_dither(
        tmp_path, monkeypatch):
    ctc = tmp_path / "ctc"
    audio = tmp_path / "audio"
    aligned = tmp_path / "aligned"
    temp = tmp_path / "temp"
    workspace = tmp_path / "workspace"
    models = tmp_path / "models"
    pinyin = tmp_path / "pinyin"
    for path in (ctc, audio, workspace, models, pinyin):
        path.mkdir()
    (ctc / "demo.lab").write_text("ni3\n", encoding="utf-8")
    (ctc / "demo.TextGrid").write_text("anchor\n", encoding="utf-8")
    (audio / "demo.wav").write_bytes(b"wav")
    dictionary = tmp_path / "dict"
    dictionary.write_text("ni3 n i\n", encoding="utf-8")
    captured = {}

    def fake_sharded(**kwargs):
        captured["sharded"] = kwargs
        return None

    def fake_run(mfa_args, *_args, **_kwargs):
        captured["command"] = list(mfa_args)
        aligned.mkdir(exist_ok=True)
        (aligned / "demo.TextGrid").write_text("aligned\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_pipeline, "_guard_mfa_axis", lambda *_args: 0)
    monkeypatch.setattr(run_pipeline, "_run_mfa_sharded", fake_sharded)
    monkeypatch.setattr(run_pipeline, "run_mfa", fake_run)
    monkeypatch.setattr(
        run_pipeline, "_write_mfa_alignment_axis_receipt", lambda *_args: 0)
    monkeypatch.setattr(
        run_pipeline, "validate_strict_mfa_textgrid", lambda *_args: [])

    cfg = deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["mfa"]["num_jobs"] = 8
    ctx = {
        "ctc_pretg_adj": ctc, "ctc_pretg": ctc, "pinyin_dir": pinyin,
        "aligned_dir": aligned, "temp_dir": temp, "models_dir": models,
        "mfa_dict": dictionary, "mfa_audio_dir": audio,
        "workspace": workspace, "expected_stems": ("demo",),
        "strict_ready": False,
    }
    rc = run_pipeline.step_mfa_align(
        SimpleNamespace(overwrite=False), cfg, Path(sys.executable), ctx)

    assert rc == 0
    assert captured["sharded"]["dither"] == 0.0
    command = captured["command"]
    assert command[command.index("--dither") + 1] == "0.0"
