"""Focused resource and failure contracts for streaming_pipeline."""

import importlib.util
from pathlib import Path


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
