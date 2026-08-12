"""Dry-run contracts for the legacy multi-GPU launcher entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_8GPU = ROOT / "scripts" / "launch_8gpu.py"
LAUNCH_MULTI = ROOT / "scripts" / "launch_multi_gpu.sh"
BATCH_CONFIG = ROOT / "configs" / "batch_all.yaml"
STRICT_CONFIG = ROOT / "configs" / "hecheng_english_mfa.yaml"


def _run(command: list[str], **env_values: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_values)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_launch_8gpu_is_one_pipelined_streaming_command_without_shards():
    result = _run([
        sys.executable, str(LAUNCH_8GPU), "--dry-run", "--config", str(BATCH_CONFIG),
        "--gpus", "3,5", "--total-stems", "17",
    ])

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("streaming_pipeline.py") == 1
    assert "--pipelined" in result.stdout
    assert "--gpus 2" in result.stdout
    assert "--limit 17" in result.stdout
    assert "run_pipeline.py" not in result.stdout
    assert "shard_configs" not in result.stdout
    assert "CUDA_VISIBLE_DEVICES: 3,5" in result.stdout


def test_launch_8gpu_rejects_strict_ctc_ready_config_before_launching():
    result = _run([
        sys.executable, str(LAUNCH_8GPU), "--dry-run", "--config", str(STRICT_CONFIG),
    ])

    assert result.returncode != 0
    assert "strict ctc_ready configs are old shard targets" in result.stderr
    assert "streaming_pipeline.py" not in result.stdout


def test_multi_gpu_streaming_dry_run_is_pipelined_and_preserves_cuda_visibility(tmp_path):
    result = _run([
        "bash", str(LAUNCH_MULTI), "--config", "configs/batch_all.yaml",
        "--streaming", "--gpus", "1", "--log-dir", str(tmp_path / "logs"), "--dry-run",
    ], MFA_PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="2,4")

    assert result.returncode == 0, result.stderr
    assert "--pipelined" in result.stdout
    assert "CUDA visible: 2,4" in result.stdout
    assert 'export CUDA_VISIBLE_DEVICES=""' not in LAUNCH_MULTI.read_text(encoding="utf-8")


def test_multi_gpu_no_pipelined_omits_pipeline_flag(tmp_path):
    result = _run([
        "bash", str(LAUNCH_MULTI), "--config", "configs/batch_all.yaml",
        "--streaming", "--no-pipelined", "--gpus", "1",
        "--log-dir", str(tmp_path / "logs"), "--dry-run",
    ], MFA_PYTHON=sys.executable)

    assert result.returncode == 0, result.stderr
    assert "Pipelined:    false" in result.stdout
    assert "--pipelined" not in result.stdout
