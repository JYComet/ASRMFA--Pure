"""Input failures and selectors must survive the preprocessing boundary."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import wave
from argparse import Namespace
from pathlib import Path

from scripts import run_pipeline


def wav(path: Path, frames: int = 3200, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x10\x01" * frames)


def run_trim(source: Path, output: Path, *extra: str):
    return subprocess.run([
        sys.executable, str(run_pipeline.SCRIPTS_DIR / "trim_silence_batch.py"),
        "--input-dir", str(source), "--output-dir", str(output),
        "--workers", "1", *extra,
    ], capture_output=True, text=True, timeout=30)


def test_trim_failure_is_nonzero_and_keeps_full_input_ledger(tmp_path):
    source, output = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "good.wav")
    (source / "bad.wav").write_bytes(b"not a WAV")
    result = run_trim(source, output)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((output / ".trim_receipt.json").read_text())
    assert receipt["status"] == "PARTIAL"
    assert [row["stem"] for row in receipt["inputs"]] == ["bad", "good"]
    assert [row["stem"] for row in receipt["failures"]] == ["bad"]
    assert [row["stem"] for row in receipt["outputs"]] == ["good"]


def test_failed_trim_keeps_original_source_denominator_in_parent(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "good.wav")
    (source / "bad.wav").write_bytes(b"not a WAV")
    ctx = {"data_dir": source, "audio_dir": output,
           "workspace": tmp_path / "workspace", "models_dir": tmp_path / "models"}

    def child(script, arguments, *_args, **_kwargs):
        return subprocess.run([sys.executable, str(script), *arguments],
                              capture_output=True, text=True, timeout=30).returncode

    monkeypatch.setattr(run_pipeline, "run_python", child)
    cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["trim"]["workers"] = 1
    rc = run_pipeline.step_trim_silence(
        Namespace(overwrite=False, force=False), cfg, Path(sys.executable), ctx)
    assert rc != 0
    assert ctx["accounting_source_stems"] == ("bad", "good")
    assert ctx["accounting_eligible_stems"] == ("bad", "good")


def test_trim_resume_rejects_changed_input_and_overwrite_rebuilds(tmp_path):
    source, output = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "line.wav", 3200)
    assert run_trim(source, output).returncode == 0
    old = (output / "line.wav").read_bytes()
    wav(source / "line.wav", 6400)
    assert run_trim(source, output).returncode != 0
    assert (output / "line.wav").read_bytes() == old
    result = run_trim(source, output, "--overwrite")
    assert result.returncode == 0, result.stdout + result.stderr
    with wave.open(str(output / "line.wav")) as handle:
        assert handle.getnframes() == 6400


def test_trim_resume_checks_output_bytes_and_completeness(tmp_path):
    source, output = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "line.wav")
    assert run_trim(source, output).returncode == 0
    (output / "line.wav").write_bytes(b"changed")
    assert run_trim(source, output).returncode != 0


def test_trim_overwrite_flag_reaches_child(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "line.wav", 3200)
    assert run_trim(source, output).returncode == 0
    wav(source / "line.wav", 6400)

    def child(script, arguments, *_args, **_kwargs):
        return subprocess.run([sys.executable, str(script), *arguments],
                              capture_output=True, text=True, timeout=30).returncode

    monkeypatch.setattr(run_pipeline, "run_python", child)
    cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["trim"].update(workers=1, normalize_edges=False)
    ctx = {"data_dir": source, "audio_dir": output,
           "workspace": tmp_path / "workspace", "models_dir": tmp_path / "models"}
    assert run_pipeline.step_trim_silence(
        Namespace(overwrite=True, force=False), cfg, Path(sys.executable), ctx) == 0
    with wave.open(str(output / "line.wav")) as handle:
        assert handle.getnframes() == 6400


def selection_ctx(tmp_path):
    source = tmp_path / "source"
    for stem in "abc":
        wav(source / "speaker" / f"{stem}.wav")
        (source / "speaker" / f"{stem}.txt").write_text("你好", encoding="utf-8")
    return {"mode": "full", "audio_dir": source, "data_dir": source,
            "workspace": tmp_path / "workspace"}


def test_frozen_selection_applies_offset_before_limit(tmp_path):
    ctx = selection_ctx(tmp_path)
    assert run_pipeline._freeze_pre_ctc_stems(
        {"ctc_prealign": {"offset": 1, "limit": 2}}, ctx) == ("b", "c")


def test_authority_selection_uses_nested_and_supported_suffix_references(tmp_path):
    ctx = selection_ctx(tmp_path)
    root = ctx["data_dir"] / "speaker"
    (root / "b.txt").rename(root / "b_qwen3-api.txt")
    assert run_pipeline._freeze_pre_ctc_stems(
        {"reference_mode": "authority", "ctc_prealign": {}}, ctx) == ("a", "b", "c")


def test_resumed_selection_rejects_changed_offset(tmp_path):
    ctx = selection_ctx(tmp_path)
    cfg = {"ctc_prealign": {"offset": 0, "limit": 2}}
    assert run_pipeline._freeze_pre_ctc_stems(cfg, ctx) == ("a", "b")
    ctx.pop("expected_stems")
    cfg["ctc_prealign"]["offset"] = 1
    assert run_pipeline._freeze_pre_ctc_stems(cfg, ctx) is None


def test_execution_receipt_allows_only_verified_preprocessing_resume(tmp_path, monkeypatch):
    import pytest
    source, output = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "line.wav")
    ctx = {"data_dir": source, "audio_dir": output,
           "workspace": tmp_path / "workspace", "models_dir": tmp_path / "models"}
    ctx["workspace"].mkdir()
    cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["trim"].update(workers=1, normalize_edges=False)
    request = Namespace(step=None, skip_to=None, stop_after=None, scan_only=False,
                        overwrite=False, force=False)
    plan = run_pipeline.plan_pipeline_steps(request, cfg, "full")
    run_pipeline.initialize_stage_execution(ctx, cfg, plan)

    def child(script, arguments, *_args, **_kwargs):
        return subprocess.run([sys.executable, str(script), *arguments],
                              capture_output=True, text=True, timeout=30).returncode

    monkeypatch.setattr(run_pipeline, "run_python", child)
    assert run_pipeline.step_trim_silence(request, cfg, Path(sys.executable), ctx) == 0
    run_pipeline.record_stage_execution(ctx, "trim")
    request.skip_to = "resample"
    resume = run_pipeline.plan_pipeline_steps(request, cfg, "full")
    run_pipeline.initialize_stage_execution(ctx, cfg, resume)
    assert ctx["pipeline_execution_state"]["completed_steps"] == ["trim"]
    wav(source / "line.wav", frames=6400)
    with pytest.raises(ValueError, match="changed"):
        run_pipeline.initialize_stage_execution(ctx, cfg, resume)


def test_resample_outputs_cannot_be_recertified_after_content_tamper(tmp_path, monkeypatch):
    import pytest
    source, audio = tmp_path / "source", tmp_path / "trimmed"
    wav(source / "line.wav", sample_rate=48000)
    ctx = {"mode": "full", "data_dir": source, "audio_dir": audio,
           "workspace": tmp_path / "workspace", "models_dir": tmp_path / "models",
           "ctc_pretg": tmp_path / "raw", "ctc_pretg_adj": tmp_path / "work",
           "mfa_audio_dir": tmp_path / "audio_16k"}
    ctx["workspace"].mkdir()
    cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["trim"].update(workers=1, normalize_edges=False)
    cfg["mfa"]["num_jobs"] = 1
    request = Namespace(step=None, skip_to=None, stop_after=None, scan_only=False,
                        overwrite=False, force=False)
    run_pipeline.initialize_stage_execution(ctx, cfg, run_pipeline.plan_pipeline_steps(request, cfg, "full"))

    def child(script, arguments, *_args, **_kwargs):
        return subprocess.run([sys.executable, str(script), *arguments],
                              capture_output=True, text=True, timeout=30).returncode

    monkeypatch.setattr(run_pipeline, "run_python", child)
    assert run_pipeline.step_trim_silence(request, cfg, Path(sys.executable), ctx) == 0
    run_pipeline.record_stage_execution(ctx, "trim")
    assert run_pipeline.step_resample_for_mfa(request, cfg, Path(sys.executable), ctx) == 0
    run_pipeline.record_stage_execution(ctx, "resample")
    request.skip_to = "prealign"
    plan = run_pipeline.plan_pipeline_steps(request, cfg, "full")
    run_pipeline.initialize_stage_execution(ctx, cfg, plan)
    (ctx["mfa_audio_dir"] / "line.wav").write_bytes(b"changed resampled waveform")
    with pytest.raises(ValueError, match="resample"):
        run_pipeline.initialize_stage_execution(ctx, cfg, plan)
