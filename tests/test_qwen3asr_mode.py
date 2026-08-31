"""No-network contract tests for the isolated qwen3asr mode."""

from __future__ import annotations

import importlib
import json
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qwen3asr_transcribe as qwen
from scripts import run_pipeline, streaming_pipeline


def _wav(path: Path, marker: int = 0) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((marker.to_bytes(2, "little", signed=False)) * 160)


def _float_wav(path: Path, frames: int = 160) -> None:
    samples = b"\0\0\0\0" * frames
    fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 16_000 * 4, 4, 32)
    fact = struct.pack("<I", frames)
    chunks = [
        b"fmt " + struct.pack("<I", len(fmt)) + fmt,
        b"fact" + struct.pack("<I", len(fact)) + fact,
        b"data" + struct.pack("<I", len(samples)) + samples,
    ]
    path.write_bytes(
        b"RIFF" + struct.pack("<I", 4 + sum(map(len, chunks)))
        + b"WAVE" + b"".join(chunks))


def _fixture(tmp_path: Path, stems=("a", "b")):
    data = tmp_path / "data"
    model = tmp_path / "model"
    output = tmp_path / "qwen-output"
    data.mkdir()
    model.mkdir()
    (model / "config.json").write_text("fake-model", encoding="utf-8")
    for index, stem in enumerate(stems):
        _wav(data / f"{stem}.wav", index)
    cfg = {
        "data_dir": str(data),
        "qwen3asr": {
            "model_path": str(model),
            "output_dir": str(output),
            "batch_size": 1,
            "language": "Chinese",
            "context": "fixture",
        },
    }
    return cfg, data, output


class FakeBackend:
    def __init__(self, *, fail=(), mismatch=False, empty=(), detected_language="Chinese"):
        self.fail = set(fail)
        self.mismatch = mismatch
        self.empty = set(empty)
        self.detected_language = detected_language
        self.calls = []
        self.languages = []

    def transcribe(self, *, audio, language, context, return_time_stamps):
        assert return_time_stamps is False
        self.calls.append(list(audio))
        self.languages.append(language)
        if self.mismatch:
            return []
        results = []
        for path in audio:
            stem = Path(path).stem
            if stem in self.fail:
                raise RuntimeError(f"fake failure {stem}")
            results.append({
                "text": "" if stem in self.empty else f"text-{stem}",
                "language": self.detected_language,
            })
        return results


def _run(cfg, tmp_path, backend):
    return qwen.run_qwen3asr(cfg, tmp_path, backend=backend,
                             qwen_asr_version="0.0.6",
                             argv=["run_pipeline.py", "--mode", "qwen3asr"])


def test_import_is_dependency_isolated():
    assert "qwen_asr" not in sys.modules
    importlib.import_module("scripts.qwen3asr_transcribe")
    assert "qwen_asr" not in sys.modules


def test_two_stem_success_has_exact_dedicated_artifacts_and_complete_receipt(tmp_path):
    cfg, _, output = _fixture(tmp_path)
    backend = FakeBackend()
    assert _run(cfg, tmp_path, backend) == 0
    assert sorted(path.relative_to(output).as_posix()
                  for path in output.rglob("*") if path.is_file()) == [
                      ".qwen3asr_run_receipt.json", "qwen3asr_checkpoint.json",
                      "qwen3asr_manifest.json", "transcripts/a_qwen3.txt",
                      "transcripts/b_qwen3.txt"]
    manifest = json.loads((output / "qwen3asr_manifest.json").read_text())
    receipt = json.loads((output / ".qwen3asr_run_receipt.json").read_text())
    assert manifest["success"] == ["a", "b"]
    assert manifest["failed"] == []
    assert receipt["status"] == "COMPLETE"
    assert receipt["return_code"] == 0
    assert manifest["success_records"][0]["language"] == "Chinese"
    assert receipt["identity_digest"] == qwen.stable_json_digest(receipt["identity"])
    assert receipt["source_stems_digest"] == qwen.stable_json_digest(["a", "b"])
    assert receipt["success_stems_digest"] == qwen.stable_json_digest(["a", "b"])
    assert receipt["failed_stems_digest"] == qwen.stable_json_digest([])
    assert receipt["transcript_records_digest"] == qwen.stable_json_digest(
        manifest["success_records"])
    assert receipt["manifest"] == {
        "path": "qwen3asr_manifest.json",
        "sha256": qwen._sha256_file(output / "qwen3asr_manifest.json"),
    }
    assert receipt["manifest_path"] == "qwen3asr_manifest.json"
    assert receipt["argv"] == ["run_pipeline.py", "--mode", "qwen3asr"]
    assert receipt["timestamp_utc"].endswith("Z")


def test_auto_language_is_normalized_to_none_for_official_backend(tmp_path):
    cfg, _, _ = _fixture(tmp_path)
    cfg["qwen3asr"]["language"] = "Auto"
    backend = FakeBackend()
    assert _run(cfg, tmp_path, backend) == 0
    assert backend.languages == [None, None]


def test_ieee_float_wav_is_accepted_by_shared_audio_contract(tmp_path):
    cfg, data, output = _fixture(tmp_path, stems=())
    _float_wav(data / "float.wav")
    backend = FakeBackend()
    assert _run(cfg, tmp_path, backend) == 0
    assert (output / "transcripts/float_qwen3.txt").read_text() == "text-float\n"


def test_resume_skips_intact_success_and_retries_failed_stem(tmp_path):
    cfg, _, _ = _fixture(tmp_path)
    first = FakeBackend(fail=("b",))
    assert _run(cfg, tmp_path, first) == 1
    second = FakeBackend()
    assert _run(cfg, tmp_path, second) == 0
    assert [Path(path).stem for call in second.calls for path in call] == ["b"]


def test_complete_resume_does_not_load_backend(tmp_path, monkeypatch):
    cfg, _, _ = _fixture(tmp_path)
    assert _run(cfg, tmp_path, FakeBackend()) == 0
    monkeypatch.setattr(qwen, "_installed_qwen_version", lambda: "0.0.6")
    monkeypatch.setattr(qwen, "_load_backend",
                        lambda *_: (_ for _ in ()).throw(AssertionError("backend loaded")))
    assert qwen.run_qwen3asr(cfg, tmp_path, argv=["resume"]) == 0


@pytest.mark.parametrize("mutation", ["wav", "model"])
def test_resume_rejects_identity_change_before_inference(tmp_path, mutation):
    cfg, data, _ = _fixture(tmp_path)
    assert _run(cfg, tmp_path, FakeBackend(fail=("b",))) == 1
    if mutation == "wav":
        _wav(data / "a.wav", 99)
    else:
        (Path(cfg["qwen3asr"]["model_path"]) / "new.bin").write_bytes(b"changed")
    backend = FakeBackend()
    with pytest.raises(qwen.Qwen3ASRError, match="identity mismatch"):
        _run(cfg, tmp_path, backend)
    assert backend.calls == []


def test_resume_rejects_missing_or_tampered_success_before_inference(tmp_path):
    cfg, _, output = _fixture(tmp_path)
    assert _run(cfg, tmp_path, FakeBackend()) == 0
    (output / "transcripts/a_qwen3.txt").write_text("tampered\n")
    backend = FakeBackend()
    with pytest.raises(qwen.Qwen3ASRError, match="tampered"):
        _run(cfg, tmp_path, backend)
    assert backend.calls == []


def test_output_root_symlink_is_rejected_before_resolve(tmp_path):
    cfg, _, output = _fixture(tmp_path)
    target = tmp_path / "actual-output"
    target.mkdir()
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(qwen.Qwen3ASRError, match="output root must not be a symlink"):
        _run(cfg, tmp_path, FakeBackend())


def test_nested_model_tree_symlink_is_rejected(tmp_path):
    cfg, _, output = _fixture(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    (Path(cfg["qwen3asr"]["model_path"]) / "linked.bin").symlink_to(external)
    with pytest.raises(qwen.Qwen3ASRError, match="symlink not allowed"):
        _run(cfg, tmp_path, FakeBackend())
    assert not output.exists()


@pytest.mark.parametrize("backend", [FakeBackend(mismatch=True), FakeBackend(fail=("a",)),
                                      FakeBackend(empty=("a",))])
def test_partial_exception_empty_or_length_mismatch_accounts_every_stem(tmp_path, backend):
    cfg, _, output = _fixture(tmp_path)
    assert _run(cfg, tmp_path, backend) == 1
    manifest = json.loads((output / "qwen3asr_manifest.json").read_text())
    assert set(manifest["success"]) | set(manifest["failed"]) == {"a", "b"}
    assert set(manifest["success"]) & set(manifest["failed"]) == set()
    for row in manifest["failed_records"]:
        assert set(row) == {"stem", "code", "exception_type", "message"}
        assert row["code"] in {"backend_exception", "batch_length_mismatch",
                               "invalid_transcript"}
        assert "\n" not in row["message"]
    assert json.loads((output / ".qwen3asr_run_receipt.json").read_text())["status"] == "PARTIAL"


def test_empty_source_fails_before_backend_or_output_creation(tmp_path, monkeypatch):
    cfg, _, output = _fixture(tmp_path, stems=())
    monkeypatch.setattr(qwen, "_load_backend",
                        lambda *_: (_ for _ in ()).throw(AssertionError("backend loaded")))
    with pytest.raises(qwen.Qwen3ASRError, match="source inventory is empty"):
        qwen.run_qwen3asr(cfg, tmp_path, qwen_asr_version="0.0.6")
    assert not output.exists()


def test_run_pipeline_qwen_branch_returns_before_mfa_or_workspace(tmp_path, monkeypatch):
    cfg, _, output = _fixture(tmp_path)
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(
        "mode: qwen3asr\n"
        f"data_dir: {cfg['data_dir']}\n"
        "qwen3asr:\n"
        f"  model_path: {cfg['qwen3asr']['model_path']}\n"
        f"  output_dir: {cfg['qwen3asr']['output_dir']}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--config", str(config_path), "--mode", "qwen3asr",
    ])
    monkeypatch.setattr(run_pipeline, "find_mfa_python",
                        lambda *_: (_ for _ in ()).throw(AssertionError("MFA probed")))
    # qwen-asr is intentionally not required by fake-mode tests; the runtime
    # capability error must still happen before any dedicated output exists.
    assert run_pipeline.main() == 1
    assert not output.exists()


def test_streaming_rejects_qwen_before_local_work_or_gpu_resolution(tmp_path, monkeypatch):
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text("mode: qwen3asr\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "streaming_pipeline.py", "--config", str(config_path), "--mode", "qwen3asr",
    ])
    monkeypatch.setattr(streaming_pipeline.subprocess, "run",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("resource probe")))
    assert streaming_pipeline.main() == 1


class _FakeModelClass:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        raise AssertionError("check must not load weights")


def _patch_check_modules(monkeypatch, torch_module):
    monkeypatch.setattr(qwen, "_installed_qwen_version", lambda: "0.0.6")

    def loader(name):
        if name == "qwen_asr":
            return SimpleNamespace(Qwen3ASRModel=_FakeModelClass)
        if name == "torch":
            return torch_module
        raise AssertionError(name)

    monkeypatch.setattr(qwen.importlib, "import_module", loader)


def test_check_rejects_missing_torch_dtype_without_loading_weights(tmp_path, monkeypatch):
    cfg, _, _ = _fixture(tmp_path)
    cfg["qwen3asr"].update({"dtype": "missing_dtype", "device": "cpu"})
    _patch_check_modules(monkeypatch, SimpleNamespace())
    ok, message = qwen.check_qwen3asr(cfg, tmp_path)
    assert not ok
    assert "does not expose configured dtype" in message


@pytest.mark.parametrize(("device", "available", "count", "message"), [
    ("cuda:0", False, 0, "CUDA is unavailable"),
    ("cuda:2", True, 1, "torch reports 1 CUDA device"),
])
def test_check_rejects_unavailable_cuda_device(tmp_path, monkeypatch, device,
                                                available, count, message):
    cfg, _, _ = _fixture(tmp_path)
    cfg["qwen3asr"]["device"] = device
    torch_module = SimpleNamespace(
        bfloat16=object(),
        cuda=SimpleNamespace(is_available=lambda: available,
                             device_count=lambda: count),
    )
    _patch_check_modules(monkeypatch, torch_module)
    ok, detail = qwen.check_qwen3asr(cfg, tmp_path)
    assert not ok
    assert message in detail


def _cli_args(**overrides):
    values = {
        "python": None, "device": None, "data_dir": None, "output_dir": None,
        "workspace": None, "step": None, "skip_to": None, "stop_after": None,
        "scan_only": False, "ctc_ready": None, "force": False, "overwrite": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_cli_reexec_prefers_args_python_and_preserves_exact_argv(tmp_path, monkeypatch):
    config_python = tmp_path / "config-python"
    cli_python = tmp_path / "cli-python"
    for executable in (config_python, cli_python):
        executable.write_bytes(b"python")
        executable.chmod(0o755)
    cfg = {"qwen3asr": {"python": str(config_python)}}
    argv = ["scripts/run_pipeline.py", "--mode", "qwen3asr",
            "--python", str(cli_python)]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def fake_run(command):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(qwen.subprocess, "run", fake_run)
    assert qwen.run_cli(_cli_args(python=str(cli_python)), cfg, tmp_path) == 7
    assert calls == [[str(cli_python.resolve()), *argv]]
    calls.clear()
    config_argv = ["scripts/run_pipeline.py", "--mode", "qwen3asr"]
    monkeypatch.setattr(sys, "argv", config_argv)
    assert qwen.run_cli(_cli_args(), cfg, tmp_path) == 7
    assert calls == [[str(config_python.resolve()), *config_argv]]


def test_old_mode_does_not_use_qwen_interpreter_reexec(tmp_path, monkeypatch):
    executable = tmp_path / "qwen-python"
    executable.write_bytes(b"python")
    executable.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [
        "run_pipeline.py", "--mode", "full", "--list-steps",
        "--python", str(executable),
    ])
    monkeypatch.setattr(qwen.subprocess, "run",
                        lambda *_: (_ for _ in ()).throw(AssertionError("qwen re-exec")))
    assert run_pipeline.main() == 0


def test_qwen_cli_does_not_reexec_same_resolved_interpreter(tmp_path, monkeypatch):
    cfg = {"qwen3asr": {"python": sys.executable}}
    monkeypatch.setattr(qwen.subprocess, "run",
                        lambda *_: (_ for _ in ()).throw(AssertionError("recursive re-exec")))
    monkeypatch.setattr(qwen, "check_qwen3asr",
                        lambda *_args, **_kwargs: (True, "ok"))
    assert qwen.run_cli(_cli_args(), cfg, tmp_path, check=True) == 0


def test_qwen_cli_applies_device_override_and_rejects_workspace(tmp_path, monkeypatch):
    captured = {}

    def fake_run(*_args, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(qwen, "run_qwen3asr", fake_run)
    cfg = {"qwen3asr": {"device": "cuda:0"}}
    assert qwen.run_cli(_cli_args(device="cuda:3"), cfg, tmp_path) == 0
    assert captured["device_override"] == "cuda:3"
    captured.clear()
    assert qwen.run_cli(_cli_args(workspace=str(tmp_path / "work")), cfg, tmp_path) == 1
    assert captured == {}


class AnchoredQwenBackend:
    def __init__(self, events):
        self.events = events
        self.calls = 0

    def transcribe(self, *, audio, language, context, return_time_stamps):
        assert language == "Chinese"
        assert return_time_stamps is False
        self.events.append("qwen")
        self.calls += 1
        return [{"text": "你好", "language": "Chinese"} for _ in audio]

    def close(self):
        self.events.append("close-qwen")


class AnchoredAligner:
    def __init__(self, events):
        self.events = events
        self.calls = 0

    def align(self, *, audio_path, text, units, model_path, stem):
        assert text == "你好"
        assert units == ["你", "好"]
        assert model_path.is_dir()
        self.events.append("aligner")
        self.calls += 1
        return [
            {"unit": "你", "start_s": 0.0, "end_s": 0.6},
            {"unit": "好", "start_s": 1.2, "end_s": 1.8},
        ]

    def close(self):
        self.events.append("close-aligner")


class AnchoredNVASR:
    def __init__(self, events, *, rejected=False):
        self.events = events
        self.rejected = rejected
        self.calls = 0

    def transcribe(self, *, audio_path, stem):
        self.events.append("nvasr")
        self.calls += 1
        lexical_occurrences = [
            {
                "lexical_ordinal": 0, "surface": "你", "token_id": 21,
                "token_ids": [21], "raw_start_frame": 4, "raw_end_frame": 6,
                "raw_start_s": 0.24, "raw_end_s": 0.36,
                "speech_start_frame": 0, "speech_end_frame": 2,
                "speech_start_s": 0.0, "speech_end_s": 0.12,
            },
            {
                "lexical_ordinal": 1, "surface": "好", "token_id": 22,
                "token_ids": [22], "raw_start_frame": 24, "raw_end_frame": 26,
                "raw_start_s": 1.44, "raw_end_s": 1.56,
                "speech_start_frame": 20, "speech_end_frame": 22,
                "speech_start_s": 1.2, "speech_end_s": 1.32,
            },
        ]
        candidate = {
            "label": "[Breathing]" if not self.rejected else "[Cough]",
            "surface": "[Breathing]" if not self.rejected else "[Cough]",
            "source": "ctc",
            "kind": "candidate",
            "token_id": 99,
            "token_ids": [99],
            "left_lexical_ordinal": 0 if not self.rejected else 1,
            "right_lexical_ordinal": 1,
            "candidate_id": "nvasr-candidate-0000",
            "occurrence": 0,
            "diagnostic": "anchored fixture",
            "raw_start_frame": 14,
            "raw_end_frame": 16,
            "raw_start_s": 0.84,
            "raw_end_s": 0.96,
            "speech_start_frame": 10,
            "speech_end_frame": 12,
            "speech_start_s": 0.6,
            "speech_end_s": 0.72,
        }
        return {
            "schema": "nvasr-candidate-timeline-v1", "stem": stem,
            "query_frames": 4, "frame_ms": 60,
            "duration_s": 10.0,
            "diagnostic": "anchored fixture",
            "lexical_occurrences": lexical_occurrences,
            "candidates": [candidate],
        }

    def close(self):
        self.events.append("close-nvasr")


def _anchored_fixture(tmp_path):
    data = tmp_path / "data"
    model = tmp_path / "model"
    aligner_model = tmp_path / "forced-aligner"
    nvasr_model = tmp_path / "nvasr-model"
    output = tmp_path / "anchored-output"
    data.mkdir()
    model.mkdir()
    aligner_model.mkdir()
    nvasr_model.mkdir()
    (model / "config.json").write_text("qwen", encoding="utf-8")
    (aligner_model / "config.json").write_text("aligner", encoding="utf-8")
    (nvasr_model / "config.json").write_text("nvasr", encoding="utf-8")
    _wav(data / "a.wav", 0)
    cfg = {
        "data_dir": str(data),
        "qwen3asr": {
            "profile": "anchored_nvv",
            "model_path": str(model),
            "forced_aligner_model_path": str(aligner_model),
            "nvasr_model_path": str(nvasr_model),
            "output_dir": str(output),
            "language": "Chinese",
            "device": "cpu",
            "batch_size": 1,
        },
    }
    return cfg, output


def _run_anchored(cfg, tmp_path, events, *, nvasr=None):
    return qwen.run_qwen3asr(
        cfg, tmp_path, backend=AnchoredQwenBackend(events),
        forced_aligner_backend=AnchoredAligner(events),
        nvasr_backend=nvasr or AnchoredNVASR(events),
        qwen_asr_version="0.0.6", argv=["run_pipeline.py", "--mode", "qwen3asr"],
    )


def test_anchored_profile_is_chinese_only_and_validates_aligner_before_output(tmp_path, monkeypatch):
    cfg, output = _anchored_fixture(tmp_path)
    monkeypatch.setattr(qwen, "_installed_qwen_version", lambda: "0.0.6")
    monkeypatch.setattr(qwen.importlib, "import_module", lambda name: (
        SimpleNamespace(Qwen3ASRModel=_FakeModelClass)
        if name == "qwen_asr" else SimpleNamespace(bfloat16=object())
    ))
    ok, message = qwen.check_qwen3asr(cfg, tmp_path)
    assert ok
    assert "profile=anchored_nvv" in message
    assert "forced_aligner_model=" in message
    cfg["qwen3asr"]["language"] = "Auto"
    ok, message = qwen.check_qwen3asr(cfg, tmp_path)
    assert not ok
    assert "requires language=Chinese" in message
    assert not output.exists()


def test_anchored_profile_is_serial_and_writes_only_dedicated_artifacts(tmp_path):
    cfg, output = _anchored_fixture(tmp_path)
    events = []
    assert _run_anchored(cfg, tmp_path, events) == 0
    assert events == ["qwen", "aligner", "nvasr"]
    assert sorted(path.relative_to(output).as_posix()
                  for path in output.rglob("*") if path.is_file()) == [
        ".anchored_nvv_run_receipt.json", "anchored_nvv_checkpoint.json",
        "anchored_nvv_manifest.json", "anchors/a.qwen3_forced_aligner.json",
        "fused/a.anchored_nvv.json",
        "nvasr_candidates/a.candidate_timeline.json",
    ]
    fused = json.loads((output / "fused/a.anchored_nvv.json").read_text())
    manifest = json.loads((output / "anchored_nvv_manifest.json").read_text())
    identity = manifest["identity"]
    assert identity["qwen_model_tree_digest"]
    assert identity["forced_aligner_model_tree_digest"]
    assert identity["nvasr_model_tree_digest"]
    assert identity["anchor_schema"] == qwen.ANCHOR_SCHEMA
    assert identity["timeline_schema"] == qwen.TIMELINE_SCHEMA
    assert identity["fusion_schema"] == qwen.FUSION_SCHEMA
    assert identity["provider_contract_version"] == qwen.PROVIDER_CONTRACT_VERSION
    assert identity["zero_width_policy_version"] == qwen.ZERO_WIDTH_POLICY_VERSION
    assert identity["nvv_bias"] == 4.0
    assert identity["pause_threshold"] == 8
    assert fused["lexical_authority"] == "qwen"
    assert fused["lexical_timing_source"] == "qwen3_forced_aligner"
    assert fused["fused_lexical_units"] == ["你", "[Breathing]", "好"]
    assert not list(output.rglob("*.TextGrid"))
    assert not list(output.rglob("*.lab"))


def test_anchored_heavy_phases_do_not_interleave_across_stems(tmp_path):
    cfg, output = _anchored_fixture(tmp_path)
    _wav(Path(cfg["data_dir"]) / "b.wav", 1)
    events = []
    assert _run_anchored(cfg, tmp_path, events) == 0
    assert events == ["qwen", "qwen", "aligner", "aligner", "nvasr", "nvasr"]
    assert json.loads((output / "anchored_nvv_manifest.json").read_text())["success"] == [
        "a", "b"
    ]


def test_anchored_complete_resume_is_tamper_checked_before_providers(tmp_path):
    cfg, output = _anchored_fixture(tmp_path)
    events = []
    assert _run_anchored(cfg, tmp_path, events) == 0
    (output / "fused/a.anchored_nvv.json").write_text("tampered", encoding="utf-8")
    second_events = []
    with pytest.raises(qwen.Qwen3ASRError, match="tampered"):
        _run_anchored(cfg, tmp_path, second_events)
    assert second_events == []


def test_anchored_candidate_rejection_is_fail_closed_and_accounted(tmp_path):
    cfg, output = _anchored_fixture(tmp_path)
    events = []
    assert _run_anchored(
        cfg, tmp_path, events, nvasr=AnchoredNVASR(events, rejected=True)
    ) == 1
    manifest = json.loads((output / "anchored_nvv_manifest.json").read_text())
    fused = json.loads((output / "fused/a.anchored_nvv.json").read_text())
    assert manifest["success"] == []
    assert manifest["failed"] == ["a"]
    assert fused["status"] == "FAILED"
    assert fused["candidate_conservation"]["exactly_once"]


def test_anchored_qwen_units_exclude_punctuation_but_preserve_raw_text():
    assert qwen._chinese_lexical_units("你，好！") == ["你", "好"]
    with pytest.raises(qwen.TranscriptResultError, match="non-Chinese"):
        qwen._chinese_lexical_units("你好abc")


def test_forced_aligner_adapter_unwraps_official_single_batch_result():
    calls = []

    class Model:
        timestamp_segment_time = 80

        def align(self, audio, text, *, language):
            calls.append((audio, text, language))
            return [SimpleNamespace(items=[
                SimpleNamespace(text="你", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="好", start_time=0.5, end_time=0.9),
            ])]

    adapter = qwen._ForcedAlignerAdapter(Model())
    assert adapter.align(Path("fixture.wav"), "你好") == [
        {"unit": "你", "start_s": 0.0, "end_s": 0.4,
         "raw_start_s": 0.0, "raw_end_s": 0.4},
        {"unit": "好", "start_s": 0.5, "end_s": 0.9,
         "raw_start_s": 0.5, "raw_end_s": 0.9},
    ]
    assert calls == [("fixture.wav", "你好", "Chinese")]


def test_forced_aligner_adapter_rejects_multiple_official_batch_results():
    item = SimpleNamespace(text="你", start_time=0.0, end_time=0.4)

    class Model:
        def align(self, *_args, **_kwargs):
            return [SimpleNamespace(items=[item]), SimpleNamespace(items=[item])]

    with pytest.raises(qwen.Qwen3ASRError, match="exactly one batch result"):
        qwen._ForcedAlignerAdapter(Model()).align(Path("fixture.wav"), "你")


def test_forced_aligner_adapter_expands_consecutive_zero_duration_items_right():
    class Model:
        timestamp_segment_time = 80

        def align(self, *_args, **_kwargs):
            return [SimpleNamespace(items=[
                SimpleNamespace(text="大", start_time=3.28, end_time=3.28),
                SimpleNamespace(text="家", start_time=3.36, end_time=3.36),
                SimpleNamespace(text="族", start_time=3.44, end_time=3.60),
            ])]

    result = qwen._ForcedAlignerAdapter(Model()).align(Path("fixture.wav"), "大家族")
    assert [(item["start_s"], item["end_s"]) for item in result] == [
        (3.28, 3.36), (3.36, 3.44), (3.44, 3.60)
    ]
    assert result[0]["raw_start_s"] == result[0]["raw_end_s"] == 3.28
    assert result[1]["raw_start_s"] == result[1]["raw_end_s"] == 3.36
    assert result[0]["timing_adjustment"] == {
        "reason": "zero_duration_expand_right", "quantum_s": 0.08,
        "raw_start_s": 3.28, "raw_end_s": 3.28,
    }
    assert result[1]["timing_adjustment"] == {
        "reason": "zero_duration_expand_right", "quantum_s": 0.08,
        "raw_start_s": 3.36, "raw_end_s": 3.36,
    }
    assert "timing_adjustment" not in result[2]


def test_forced_aligner_adapter_expands_zero_duration_item_left_when_right_unavailable():
    class Model:
        timestamp_segment_time = 80

        def align(self, *_args, **_kwargs):
            return [SimpleNamespace(items=[
                SimpleNamespace(text="前", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="大", start_time=1.0, end_time=1.0),
            ])]

    result = qwen._ForcedAlignerAdapter(Model()).align(Path("fixture.wav"), "前大")
    assert result[1]["start_s"] == 0.92
    assert result[1]["end_s"] == 1.0
    assert result[1]["timing_adjustment"]["reason"] == "zero_duration_expand_left"
    assert result[1]["timing_adjustment"]["quantum_s"] == 0.08


def test_forced_aligner_adapter_rejects_zero_duration_item_without_full_quantum_gap():
    class Model:
        timestamp_segment_time = 80

        def align(self, *_args, **_kwargs):
            return [SimpleNamespace(items=[
                SimpleNamespace(text="前", start_time=0.0, end_time=1.0),
                SimpleNamespace(text="大", start_time=1.0, end_time=1.0),
            ])]

    with pytest.raises(qwen.Qwen3ASRError, match="no full quantum gap"):
        qwen._ForcedAlignerAdapter(Model()).align(Path("fixture.wav"), "前大")


def test_anchored_auto_factories_load_in_phase_order_and_release(tmp_path, monkeypatch):
    cfg, _ = _anchored_fixture(tmp_path)
    events = []

    def load_qwen(_settings):
        events.append("load-qwen")
        return AnchoredQwenBackend(events), "0.0.6"

    def load_aligner(_settings):
        events.append("load-aligner")
        return AnchoredAligner(events)

    def load_nvasr(_settings):
        events.append("load-nvasr")
        return AnchoredNVASR(events)

    monkeypatch.setattr(qwen, "_load_backend", load_qwen)
    monkeypatch.setattr(qwen, "_load_forced_aligner", load_aligner)
    monkeypatch.setattr(qwen, "_load_nvasr", load_nvasr)
    assert qwen.run_qwen3asr(
        cfg, tmp_path, qwen_asr_version="0.0.6",
        argv=["run_pipeline.py", "--mode", "qwen3asr"],
    ) == 0
    assert events == [
        "load-qwen", "qwen", "close-qwen",
        "load-aligner", "aligner", "close-aligner",
        "load-nvasr", "nvasr", "close-nvasr",
    ]


def test_anchored_auto_factory_cleanup_runs_after_inference_failure(tmp_path, monkeypatch):
    cfg, output = _anchored_fixture(tmp_path)
    events = []

    class FailingQwen(AnchoredQwenBackend):
        def transcribe(self, **_kwargs):
            self.events.append("qwen")
            raise RuntimeError("inference failed")

    monkeypatch.setattr(qwen, "_load_backend",
                        lambda _settings: (events.append("load-qwen") or
                                           (FailingQwen(events), "0.0.6")))
    monkeypatch.setattr(qwen, "_load_forced_aligner",
                        lambda _settings: (_ for _ in ()).throw(AssertionError("aligner loaded")))
    monkeypatch.setattr(qwen, "_load_nvasr",
                        lambda _settings: (_ for _ in ()).throw(AssertionError("nvasr loaded")))
    assert qwen.run_qwen3asr(
        cfg, tmp_path, qwen_asr_version="0.0.6",
        argv=["run_pipeline.py", "--mode", "qwen3asr"],
    ) == 1
    assert events == ["load-qwen", "qwen", "close-qwen"]
    assert json.loads((output / "anchored_nvv_manifest.json").read_text())["failed"] == ["a"]


def test_anchored_complete_resume_does_not_instantiate_any_provider(tmp_path, monkeypatch):
    cfg, _ = _anchored_fixture(tmp_path)
    first_events = []
    monkeypatch.setattr(qwen, "_load_backend",
                        lambda _settings: (AnchoredQwenBackend(first_events), "0.0.6"))
    monkeypatch.setattr(qwen, "_load_forced_aligner",
                        lambda _settings: AnchoredAligner(first_events))
    monkeypatch.setattr(qwen, "_load_nvasr",
                        lambda _settings: AnchoredNVASR(first_events))
    assert qwen.run_qwen3asr(cfg, tmp_path, qwen_asr_version="0.0.6") == 0
    for name in ("_load_backend", "_load_forced_aligner", "_load_nvasr"):
        monkeypatch.setattr(qwen, name,
                            lambda _settings, name=name: (_ for _ in ()).throw(
                                AssertionError(f"{name} loaded on resume")))
    assert qwen.run_qwen3asr(cfg, tmp_path, qwen_asr_version="0.0.6") == 0


def test_anchored_v3_fused_artifact_resumes_without_provider_factories(tmp_path, monkeypatch):
    cfg, output = _anchored_fixture(tmp_path)
    first_events = []
    assert _run_anchored(cfg, tmp_path, first_events) == 0
    fused_path = output / "fused/a.anchored_nvv.json"
    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    fused.pop("timing_label", None)
    assert fused["lexical_timing_source"] == "qwen3_forced_aligner"
    fused_path.write_text(json.dumps(fused, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    checkpoint_path = output / "anchored_nvv_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["records"][0]
    fused_record = record["phases"]["fused"]
    fused_record["size"] = fused_path.stat().st_size
    fused_record["sha256"] = qwen._sha256_file(fused_path)
    checkpoint_path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def fail_factory(_settings):
        raise AssertionError("provider factory called on v3 complete resume")

    monkeypatch.setattr(qwen, "_load_backend", fail_factory)
    monkeypatch.setattr(qwen, "_load_forced_aligner", fail_factory)
    monkeypatch.setattr(qwen, "_load_nvasr", fail_factory)
    assert qwen.run_qwen3asr(cfg, tmp_path, qwen_asr_version="0.0.6") == 0
