"""Focused resource and failure contracts for streaming_pipeline."""

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "streaming_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("streaming_pipeline", _SCRIPT)
assert _SPEC and _SPEC.loader
streaming = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(streaming)


def test_resource_plan_caps_ordinary_workers_and_both_mfa_pools():
    plan = streaming.plan_streaming_resources(
        cpu_budget=12,
        requested_gpu_workers=8,
        requested_cpu_workers=8,
        requested_mfa_jobs=99,
        config_mfa_en_jobs=99,
        batch_size=500,
        batch_count=3,
    )

    assert plan["cpu_workers"] == 3
    assert plan["gpu_workers"] == 3
    assert plan["mfa_jobs_per_worker"] == 4
    assert plan["mfa_en_jobs_per_worker"] == 4
    assert plan["cpu_workers"] * plan["mfa_jobs_per_worker"] <= plan["cpu_budget"]
    assert plan["cpu_workers"] * plan["mfa_en_jobs_per_worker"] <= plan["cpu_budget"]


def test_resource_plan_pipelined_defaults_and_queues_are_bounded():
    plan = streaming.plan_streaming_resources(
        cpu_budget=32,
        requested_gpu_workers=4,
        requested_cpu_workers=0,
        config_mfa_jobs=64,
        config_mfa_en_jobs=64,
        batch_count=10,
        pipelined=True,
    )

    assert plan["cpu_workers"] == 4  # cpu_count // 8
    assert plan["mfa_jobs_per_worker"] == 8
    assert plan["mfa_en_jobs_per_worker"] == 8
    assert plan["gpu_queue_size"] == 8
    assert plan["cpu_queue_size"] == 8


def test_gpu_phase_fails_before_subprocess_when_audio_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "find_wav", lambda *_: None)
    called = False

    def unexpected_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("GPU subprocess must not run with missing audio")

    monkeypatch.setattr(streaming.subprocess, "run", unexpected_run)
    ok = streaming._run_gpu_phase(
        ds={"audio_dir": str(tmp_path / "audio"), "ctc_dir": str(tmp_path / "ctc")},
        batch_idx=0,
        batch_stems=["missing"],
        layout_map={}, wav_index={}, text_index=None,
        local_base=tmp_path / "work", config=tmp_path / "config.yaml",
        mfa_python=tmp_path / "python", models_dir=tmp_path / "models",
        batch_size=1, python_path=None, device="cuda:0",
    )

    assert ok is False
    assert called is False


@pytest.mark.parametrize("use_link_or_copy", [False, True])
def test_gpu_staging_copy_retries_once_then_succeeds(
    tmp_path, monkeypatch, use_link_or_copy
):
    calls = 0
    delays = []

    def transient_copy(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary NAS I/O error")
        return True

    if use_link_or_copy:
        monkeypatch.setattr(streaming, "link_or_copy_file", transient_copy)
    else:
        monkeypatch.setattr(streaming.shutil, "copy2", transient_copy)
    monkeypatch.setattr(streaming.time, "sleep", delays.append)

    assert streaming._copy_gpu_staging_file(
        tmp_path / "source", tmp_path / "target", "demo",
        use_link_or_copy=use_link_or_copy,
    ) is True
    assert calls == 2
    assert delays == [streaming._GPU_STAGING_RETRY_DELAY_S]


@pytest.mark.parametrize("use_link_or_copy", [False, True])
def test_gpu_staging_copy_persistent_failure_returns_false_with_diagnostics(
    tmp_path, monkeypatch, capsys, use_link_or_copy
):
    calls = 0
    delays = []

    def persistent_copy(*_args):
        nonlocal calls
        calls += 1
        if use_link_or_copy:
            return False
        raise OSError("NAS unavailable")

    if use_link_or_copy:
        monkeypatch.setattr(streaming, "link_or_copy_file", persistent_copy)
    else:
        monkeypatch.setattr(streaming.shutil, "copy2", persistent_copy)
    monkeypatch.setattr(streaming.time, "sleep", delays.append)
    source = tmp_path / "source"
    target = tmp_path / "target"

    assert streaming._copy_gpu_staging_file(
        source, target, "合成ria_85474", use_link_or_copy=use_link_or_copy,
    ) is False
    assert calls == streaming._GPU_STAGING_COPY_ATTEMPTS
    assert delays == [
        streaming._GPU_STAGING_RETRY_DELAY_S,
        streaming._GPU_STAGING_RETRY_DELAY_S * 2,
    ]
    output = capsys.readouterr().out
    assert "stem=合成ria_85474" in output
    assert f"source={source}" in output
    assert f"target={target}" in output
    assert f"attempts={streaming._GPU_STAGING_COPY_ATTEMPTS}" in output


def test_gpu_phase_staging_failure_uses_audio_and_text_denominator(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "source.wav"
    text = tmp_path / "source.txt"
    audio.write_bytes(b"wav")
    text.write_text("text", encoding="utf-8")
    monkeypatch.setattr(streaming, "_copy_gpu_staging_file", lambda *args, **kwargs: False)

    ok = streaming._run_gpu_phase(
        ds={"audio_dir": str(tmp_path), "ctc_dir": str(tmp_path / "ctc")},
        batch_idx=0,
        batch_stems=["source"],
        layout_map={},
        wav_index={"source": audio},
        text_index={"source": text},
        local_base=tmp_path / "work",
        config=tmp_path / "config.yaml",
        mfa_python=tmp_path / "python",
        models_dir=tmp_path / "models",
        batch_size=1,
        python_path=None,
        device="cuda:0",
    )

    assert ok is False
    assert "2/2 staging copies failed" in capsys.readouterr().out


def test_stale_batch_workspace_is_quarantined_without_deletion(tmp_path):
    batch = tmp_path / "batch_0065"
    batch.mkdir()
    marker = batch / "old_ctc_marker.txt"
    marker.write_text("old", encoding="utf-8")

    quarantined = streaming._quarantine_existing_path(batch, label="STALE")

    assert quarantined is not None
    assert not batch.exists()
    assert quarantined.name.startswith("batch_0065.STALE.")
    assert (quarantined / marker.name).read_text(encoding="utf-8") == "old"


def test_failed_batch_preserves_previous_failure_evidence(tmp_path):
    batch = tmp_path / "batch_0001"
    batch.mkdir()
    (batch / "current.txt").write_text("current", encoding="utf-8")
    previous = tmp_path / "batch_0001.FAILED"
    previous.mkdir()
    (previous / "previous.txt").write_text("previous", encoding="utf-8")

    failed = streaming._preserve_failed_batch(batch)

    assert failed == previous
    assert (failed / "current.txt").read_text(encoding="utf-8") == "current"
    preserved = list(tmp_path.glob("batch_0001.FAILED.PREVIOUS.*"))
    assert len(preserved) == 1
    assert (preserved[0] / "previous.txt").read_text(encoding="utf-8") == "previous"


def test_dataset_batch_workspaces_are_isolated(tmp_path):
    first = streaming._batch_local_dir(tmp_path, 68, "GSpaimeng")
    second = streaming._batch_local_dir(tmp_path, 68, "HKyiyi")

    assert first != second
    assert first.name.startswith("batch_0068_GSpaimeng-")
    assert second.name.startswith("batch_0068_HKyiyi-")


def test_dataset_batch_token_keeps_sanitized_names_distinct():
    assert streaming._dataset_batch_token("a/b") != streaming._dataset_batch_token("a_b")


def test_flatten_ctc_shards_recovers_complete_bundle(tmp_path):
    shard = tmp_path / "ctc_pretg" / "_shard_gpu0"
    shard.mkdir(parents=True)
    stem = "合成ria_01551"
    suffixes = (".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
                "_text_cn.txt", "_text_raw.txt")
    for suffix in suffixes:
        (shard / f"{stem}{suffix}").write_text("artifact", encoding="utf-8")

    complete = streaming._flatten_ctc_shards(
        tmp_path / "ctc_pretg", {stem}, suffixes)

    assert complete == {stem}
    assert all((tmp_path / "ctc_pretg" / f"{stem}{s}").is_file()
               for s in suffixes)
