from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen3_hf_backend as backend


class Inputs(dict):
    def to(self, *args):
        return self


def native_runtime(monkeypatch, alignment=None):
    calls = []
    if alignment is None:
        alignment = [{"text": "Hello", "start_time": 0.08, "end_time": 0.4}]

    class Processor:
        def apply_transcription_request(self, **kwargs):
            calls.append(("transcribe", kwargs))
            return Inputs(input_ids=np.array([[10, 11]]))

        def decode(self, ids, **kwargs):
            assert ids.tolist() == [[12, 13]]
            return [{"transcription": "Hello!", "language": None}]

        def prepare_forced_aligner_inputs(self, **kwargs):
            calls.append(("align", kwargs))
            return Inputs(input_ids=np.array([[1, 2]])), [["Hello"]]

        def decode_forced_alignment(self, **kwargs):
            assert kwargs["timestamp_token_id"] == 42
            return [alignment]

    class Model:
        device = "cpu"
        dtype = "float32"
        config = SimpleNamespace(timestamp_token_id=42, timestamp_segment_time=80)

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 123
            return np.array([[10, 11, 12, 13]])

        def __call__(self, **kwargs):
            return SimpleNamespace(logits="logits")

    def factory(name, instance):
        def load(path, **kwargs):
            assert kwargs["local_files_only"] is True
            calls.append((name, path))
            return instance()
        return SimpleNamespace(from_pretrained=load)

    transformers = SimpleNamespace(
        __version__="5.13.0",
        AutoProcessor=factory("processor", Processor),
        AutoModelForMultimodalLM=factory("asr_model", Model),
        AutoModelForTokenClassification=factory("aligner_model", Model),
    )
    torch = SimpleNamespace(float32="float32", inference_mode=nullcontext,
                            cuda=SimpleNamespace(is_available=lambda: False))

    def import_module(name):
        assert name in {"transformers", "torch"}, "must not import legacy qwen_asr"
        return transformers if name == "transformers" else torch

    monkeypatch.setattr(backend.importlib, "import_module", import_module)
    return calls, transformers


def settings(tmp_path):
    for name in ("asr", "aligner"):
        (tmp_path / name).mkdir()
    return backend.Qwen3HFSettings(tmp_path / "asr", tmp_path / "aligner",
                                  device="cpu", dtype="float32", language="English",
                                  max_new_tokens=123)


def test_native_asr_and_forced_alignment_contract(monkeypatch, tmp_path):
    calls, _ = native_runtime(monkeypatch)
    loaded = backend.load_backend(settings(tmp_path))
    assert loaded.transcribe(Path("sample.wav"), language="English", context="Hello") == "Hello!"
    assert loaded.last_language == "English"
    assert loaded.align(Path("sample.wav"), "Hello!", language="English") == [
        {"unit": "Hello", "start_s": .08, "end_s": .4,
         "raw_start_s": .08, "raw_end_s": .4}]
    assert ("transcribe", {"audio": "sample.wav", "language": "English", "prompt": "Hello"}) in calls
    assert ("align", {"audio": "sample.wav", "transcript": "Hello!", "language": "English"}) in calls
    loaded.close()


def test_reference_alignment_does_not_load_asr(monkeypatch, tmp_path):
    calls, _ = native_runtime(monkeypatch)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)
    loaded.align(Path("sample.wav"), "Hello", language="English")
    assert not any(name == "asr_model" for name, _ in calls)
    with pytest.raises(backend.Qwen3HFError, match="ASR"):
        loaded.transcribe(Path("sample.wav"))


def test_models_can_use_distinct_whole_gpu_devices(monkeypatch, tmp_path):
    from dataclasses import replace
    _, package = native_runtime(monkeypatch)
    placements = {}
    for kind, factory in [("asr", package.AutoModelForMultimodalLM),
                           ("aligner", package.AutoModelForTokenClassification)]:
        original = factory.from_pretrained
        def record(path, _kind=kind, _original=original, **kwargs):
            placements[_kind] = kwargs["device_map"]
            return _original(path, **kwargs)
        monkeypatch.setattr(factory, "from_pretrained", record)
    cfg = replace(settings(tmp_path), device="cuda:1", forced_aligner_device="cuda:0")
    loaded = backend.load_backend(cfg)
    assert placements == {"asr": "cuda:1", "aligner": "cuda:0"}
    loaded.close()


def test_native_alignment_expands_model_declared_zero_width_quantum(monkeypatch, tmp_path):
    raw = [
        {"text": "你", "start_time": 0.08, "end_time": 0.56},
        {"text": "好", "start_time": 0.64, "end_time": 0.64},
        {"text": "呀", "start_time": 0.72, "end_time": 1.04},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    assert loaded.align(Path("sample.wav"), "你好呀", language="Chinese") == [
        {"unit": "你", "start_s": .08, "end_s": .56,
         "raw_start_s": .08, "raw_end_s": .56},
        {"unit": "好", "start_s": .64, "end_s": .72,
         "raw_start_s": .64, "raw_end_s": .64,
         "timing_adjustment": {
             "reason": "zero_duration_expand_right", "quantum_s": .08,
             "raw_start_s": .64, "raw_end_s": .64,
         }},
        {"unit": "呀", "start_s": .72, "end_s": 1.04,
         "raw_start_s": .72, "raw_end_s": 1.04},
    ]


def test_native_alignment_borrows_shared_boundary_quantum_from_previous_unit(monkeypatch, tmp_path):
    raw = [
        {"text": "大", "start_time": 0.48, "end_time": 0.64},
        {"text": "家", "start_time": 0.64, "end_time": 0.64},
        {"text": "现", "start_time": 0.64, "end_time": 0.80},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    assert loaded.align(Path("sample.wav"), "大家现", language="Chinese") == [
        {"unit": "大", "start_s": .48, "end_s": .56,
         "raw_start_s": .48, "raw_end_s": .64,
         "timing_adjustment": {
             "reason": "yield_right_quantum_to_zero_duration", "quantum_s": .08,
             "raw_start_s": .48, "raw_end_s": .64,
         }},
        {"unit": "家", "start_s": .56, "end_s": .64,
         "raw_start_s": .64, "raw_end_s": .64,
         "timing_adjustment": {
             "reason": "zero_duration_borrow_left", "quantum_s": .08,
             "raw_start_s": .64, "raw_end_s": .64,
         }},
        {"unit": "现", "start_s": .64, "end_s": .80,
         "raw_start_s": .64, "raw_end_s": .80},
    ]


def test_native_alignment_borrows_shared_boundary_quantum_from_next_unit(monkeypatch, tmp_path):
    raw = [
        {"text": "家", "start_time": 0.64, "end_time": 0.72},
        {"text": "现", "start_time": 0.72, "end_time": 0.72},
        {"text": "在", "start_time": 0.72, "end_time": 0.88},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    assert loaded.align(Path("sample.wav"), "家现在", language="Chinese") == [
        {"unit": "家", "start_s": .64, "end_s": .72,
         "raw_start_s": .64, "raw_end_s": .72},
        {"unit": "现", "start_s": .72, "end_s": .80,
         "raw_start_s": .72, "raw_end_s": .72,
         "timing_adjustment": {
             "reason": "zero_duration_borrow_right", "quantum_s": .08,
             "raw_start_s": .72, "raw_end_s": .72,
         }},
        {"unit": "在", "start_s": .80, "end_s": .88,
         "raw_start_s": .72, "raw_end_s": .88,
         "timing_adjustment": {
             "reason": "yield_left_quantum_to_zero_duration", "quantum_s": .08,
             "raw_start_s": .72, "raw_end_s": .88,
         }},
    ]


def test_native_alignment_redistributes_subquantum_shared_boundary_neighbors(monkeypatch, tmp_path):
    raw = [
        {"text": "很", "start_time": 0.64, "end_time": 0.72},
        {"text": "好", "start_time": 0.72, "end_time": 0.72},
        {"text": "玩", "start_time": 0.72, "end_time": 0.80},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    rows = loaded.align(Path("sample.wav"), "很好玩", language="Chinese")

    assert [(row["start_s"], row["end_s"]) for row in rows] == [
        (.64, .693333333), (.693333333, .746666667), (.746666667, .80)]
    assert [row.get("timing_adjustment", {}).get("reason") for row in rows] == [
        "redistribute_shared_boundary_for_zero_duration",
        "zero_duration_redistribute_shared_boundary",
        "redistribute_shared_boundary_for_zero_duration",
    ]
    assert [(row["raw_start_s"], row["raw_end_s"]) for row in rows] == [
        (.64, .72), (.72, .72), (.72, .80)]


def test_native_alignment_redistributes_consecutive_shared_boundary_points(monkeypatch, tmp_path):
    raw = [
        {"text": "前", "start_time": 0.64, "end_time": 0.72},
        {"text": "甲", "start_time": 0.72, "end_time": 0.72},
        {"text": "乙", "start_time": 0.72, "end_time": 0.72},
        {"text": "后", "start_time": 0.72, "end_time": 0.80},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    rows = loaded.align(Path("sample.wav"), "前甲乙后", language="Chinese")

    assert [(row["start_s"], row["end_s"]) for row in rows] == [
        (.64, .68), (.68, .72), (.72, .76), (.76, .80)]
    assert all(row["end_s"] > row["start_s"] for row in rows)
    assert [row["raw_start_s"] for row in rows[1:3]] == [.72, .72]


def test_native_alignment_redistributes_terminal_zero_point_with_left_neighbor(monkeypatch, tmp_path):
    raw = [
        {"text": "我", "start_time": 7.28, "end_time": 7.36},
        {"text": "我", "start_time": 7.36, "end_time": 7.36},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    rows = loaded.align(Path("sample.wav"), "我我", language="Chinese")

    assert [(row["start_s"], row["end_s"]) for row in rows] == [
        (7.28, 7.32), (7.32, 7.36)]
    assert [row.get("timing_adjustment", {}).get("reason") for row in rows] == [
        "redistribute_shared_boundary_for_zero_duration",
        "zero_duration_redistribute_shared_boundary",
    ]


def test_native_alignment_expands_zero_point_into_subquantum_adjacent_gap(monkeypatch, tmp_path):
    raw = [
        {"text": "前", "start_time": .40, "end_time": .45},
        {"text": "中", "start_time": .48, "end_time": .48},
        {"text": "后", "start_time": .52, "end_time": .60},
    ]
    native_runtime(monkeypatch, alignment=raw)
    loaded = backend.load_backend(settings(tmp_path), require_asr=False)

    rows = loaded.align(Path("sample.wav"), "前中后", language="Chinese")

    assert rows[1]["start_s"] == .45
    assert rows[1]["end_s"] == .52
    assert rows[1]["timing_adjustment"]["reason"] == "zero_duration_expand_subquantum_gap"


def test_old_transformers_fails_before_model_load(monkeypatch):
    calls, transformers = native_runtime(monkeypatch)
    transformers.__version__ = "4.57.6"
    with pytest.raises(backend.Qwen3HFError, match="5.13"):
        backend.runtime_capabilities()
    assert calls == []


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (0, float("inf")), (0, 0), (-1, 1)])
def test_invalid_timestamps_rejected(start, end):
    with pytest.raises(backend.Qwen3HFError):
        backend.normalize_alignment_items([{"text": "Hi", "start_time": start, "end_time": end}])
