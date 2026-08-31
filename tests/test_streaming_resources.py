"""Focused resource and failure contracts for streaming_pipeline."""

import importlib.util
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "streaming_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("streaming_pipeline", _SCRIPT)
assert _SPEC and _SPEC.loader
streaming = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(streaming)


def test_streaming_cli_propagates_main_exit_status():
    probe = r'''
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = int(sys.argv[2])
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "main":
        node.body = [ast.Return(value=ast.Constant(status))]
        break
else:
    raise SystemExit(97)
ast.fix_missing_locations(tree)
try:
    exec(compile(tree, str(path), "exec"), {
        "__name__": "__main__", "__file__": str(path),
        "__package__": None,
    })
except SystemExit as exc:
    raise SystemExit(exc.code)
raise SystemExit(99)
'''
    for status in (0, 1):
        result = subprocess.run(
            [sys.executable, "-c", probe, str(_SCRIPT), str(status)],
            capture_output=True, text=True,
        )
        assert result.returncode == status, result.stderr


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


def test_exact_gpu_bundle_rejects_subset_and_empty_bindings(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    ctc = workspace / "ctc_pretg"
    ctc.mkdir(parents=True)
    stems = ["s00", "s01"]
    for stem in stems:
        for suffix in streaming.CTC_SUFFIXES:
            (ctc / f"{stem}{suffix}").write_text("artifact", encoding="utf-8")
    receipt = {
        "schema": "ctc-run-receipt-v2",
        "input_stems": stems,
        "output_stems": stems,
        "audio_bindings": [{"stem": stem} for stem in stems],
    }
    receipt_path = ctc / ".ctc_run_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(streaming, "validate_ctc_run_receipt_v2", lambda *_a, **_k: [])

    assert streaming._validate_exact_ctc_bundle(workspace, tmp_path / "audio", stems)

    receipt["output_stems"] = [stems[0]]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not streaming._validate_exact_ctc_bundle(workspace, tmp_path / "audio", stems)

    receipt["output_stems"] = stems
    receipt["audio_bindings"] = []
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not streaming._validate_exact_ctc_bundle(workspace, tmp_path / "audio", stems)


def test_complete_receipt_repair_reconstructs_exact_bindings(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    ctc = workspace / "ctc_pretg"
    ctc.mkdir(parents=True)
    stems = ["s00", "s01"]
    for stem in stems:
        for suffix in streaming.CTC_SUFFIXES:
            (ctc / f"{stem}{suffix}").write_text("artifact", encoding="utf-8")
    receipt_path = ctc / ".ctc_run_receipt.json"
    receipt_path.write_text(json.dumps({
        "schema": "ctc-run-receipt-v2", "input_stems": stems,
        "output_stems": stems, "audio_bindings": [],
    }), encoding="utf-8")

    def fake_ensure(ctx):
        ctx["ctc_axis_receipt"] = {
            "schema": "ctc-run-receipt-v2", "input_stems": stems,
            "output_stems": stems,
            "audio_bindings": [{"stem": stem} for stem in stems],
        }
        (Path(ctx["ctc_pretg"]) / ".ctc_run_receipt.json").write_text(
            json.dumps(ctx["ctc_axis_receipt"]), encoding="utf-8")
        return 0

    monkeypatch.setattr(streaming, "validate_ctc_run_receipt_v2", lambda *_a, **_k: [])
    import pipeline_utils
    import run_pipeline
    monkeypatch.setattr(pipeline_utils, "validate_ctc_run_receipt_v2",
                        lambda *_a, **_k: [])
    monkeypatch.setattr(run_pipeline, "_ensure_ctc_axis_receipt", fake_ensure)
    assert streaming._repair_complete_ctc_receipt(
        workspace, tmp_path / "audio", stems) is True
    repaired = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert repaired["input_stems"] == stems
    assert repaired["output_stems"] == stems
    assert [row["stem"] for row in repaired["audio_bindings"]] == stems


@pytest.mark.parametrize("invalid_kind", ["subset", "empty_bindings"])
def test_gpu_phase_retries_three_times_without_queuing_invalid_bundle(
        tmp_path, monkeypatch, invalid_kind):
    source = tmp_path / "s00.wav"
    source.write_bytes(b"source")
    stems = [f"s{index:02d}" for index in range(13)]
    calls = []
    quarantine_counts = []

    monkeypatch.setattr(streaming, "_copy_gpu_staging_file",
                        lambda *_args, **_kwargs: True)
    monkeypatch.setattr(streaming, "get_mfa_env", lambda *_args: {})
    monkeypatch.setattr(streaming, "validate_ctc_run_receipt_v2",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(streaming, "_repair_complete_ctc_receipt",
                        lambda *_args, **_kwargs: False)

    def fake_run(command, **_kwargs):
        workspace = Path(command[command.index("--workspace") + 1])
        ctc = workspace / "ctc_pretg"
        quarantine_counts.append(len(list(workspace.glob("ctc_pretg.ATTEMPT*"))))
        ctc.mkdir(parents=True)
        produced = stems[:-1] if invalid_kind == "subset" else stems
        for stem in produced:
            for suffix in streaming.CTC_SUFFIXES:
                (ctc / f"{stem}{suffix}").write_text("artifact", encoding="utf-8")
        receipt_stems = produced
        receipt = {
            "schema": "ctc-run-receipt-v2",
            "input_stems": receipt_stems,
            "output_stems": receipt_stems,
            "audio_bindings": ([{"stem": stem} for stem in receipt_stems]
                                if invalid_kind == "subset" else []),
        }
        (ctc / ".ctc_run_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8")
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(streaming.subprocess, "run", fake_run)
    ok = streaming._run_gpu_phase(
        ds={"name": "demo", "audio_dir": str(tmp_path),
            "ctc_dir": str(tmp_path / "ctc")},
        batch_idx=0, batch_stems=stems, layout_map={},
        wav_index={stem: source for stem in stems}, text_index={},
        local_base=tmp_path / "work", config=tmp_path / "config.yaml",
        mfa_python=Path("/bin/python"), models_dir=tmp_path,
        batch_size=13, python_path=None, device="cuda:0",
        persist_cache=False, allow_overwrite=False, allow_force=False)

    assert ok is False
    assert len(calls) == 3
    assert quarantine_counts == [0, 1, 2]
    assert list((tmp_path / "work").glob("batch_*.FAILED"))


def test_gpu_phase_returns_when_third_full_bundle_is_exact(tmp_path, monkeypatch):
    source = tmp_path / "s00.wav"
    source.write_bytes(b"source")
    stems = [f"s{index:02d}" for index in range(13)]
    calls = []
    quarantine_counts = []

    monkeypatch.setattr(streaming, "_copy_gpu_staging_file",
                        lambda *_args, **_kwargs: True)
    monkeypatch.setattr(streaming, "get_mfa_env", lambda *_args: {})
    monkeypatch.setattr(streaming, "validate_ctc_run_receipt_v2",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(streaming, "_repair_complete_ctc_receipt",
                        lambda *_args, **_kwargs: False)

    def fake_run(command, **_kwargs):
        workspace = Path(command[command.index("--workspace") + 1])
        ctc = workspace / "ctc_pretg"
        quarantine_counts.append(len(list(workspace.glob("ctc_pretg.ATTEMPT*"))))
        ctc.mkdir(parents=True)
        produced = stems if len(calls) == 2 else stems[:-1]
        for stem in produced:
            for suffix in streaming.CTC_SUFFIXES:
                (ctc / f"{stem}{suffix}").write_text("artifact", encoding="utf-8")
        (ctc / ".ctc_run_receipt.json").write_text(json.dumps({
            "schema": "ctc-run-receipt-v2", "input_stems": produced,
            "output_stems": produced,
            "audio_bindings": [{"stem": stem} for stem in produced],
        }), encoding="utf-8")
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(streaming.subprocess, "run", fake_run)
    ok = streaming._run_gpu_phase(
        ds={"name": "demo", "audio_dir": str(tmp_path),
            "ctc_dir": str(tmp_path / "ctc")},
        batch_idx=0, batch_stems=stems, layout_map={},
        wav_index={stem: source for stem in stems}, text_index={},
        local_base=tmp_path / "work", config=tmp_path / "config.yaml",
        mfa_python=Path("/bin/python"), models_dir=tmp_path,
        batch_size=13, python_path=None, device="cuda:0",
        persist_cache=False, allow_overwrite=False, allow_force=False)

    assert ok is True
    assert len(calls) == 3
    assert quarantine_counts == [0, 1, 2]
    assert not list((tmp_path / "work").glob("batch_*.FAILED"))


def test_delayed_cpu_failure_drains_and_preserves_bounded_queue_work(tmp_path):
    sentinel = object()
    started = threading.Event()

    class SignalQueue(streaming.queue.Queue):
        def put(self, item, block=True, timeout=None):
            if item is sentinel:
                started.set()
            return super().put(item, block=block, timeout=timeout)

    cpu_queue = SignalQueue(maxsize=1)
    local_base = tmp_path / "work"
    local_dir = streaming._batch_local_dir(local_base, 0, "demo")
    local_dir.mkdir(parents=True)
    (local_dir / "producer-evidence.txt").write_text("evidence", encoding="utf-8")
    queued_item = ({"name": "demo"}, 0, ["s00"], local_base,
                   "nvrasr_fallback")
    # Fill the bounded queue without calling the instrumented put method.
    with cpu_queue.mutex:
        cpu_queue.queue.append(queued_item)
        cpu_queue.unfinished_tasks += 1

    stop_event = threading.Event()
    failure_event = threading.Event()
    accounted = []

    def drain():
        while True:
            try:
                item = cpu_queue.get_nowait()
            except streaming.queue.Empty:
                return
            if item is not sentinel:
                accounted.append(item)
                streaming._preserve_failed_batch(local_dir)

    def delayed_failure():
        assert started.wait(1.0)
        failure_event.set()
        stop_event.set()

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(delayed_failure)
        result = streaming._shutdown_pipelined_cpu_queue(
            cpu_queue, 1, stop_event, failure_event, sentinel, drain)
        future.result(timeout=1.0)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed < 1.0
    assert len(accounted) == 1
    failed_dirs = list(local_base.glob("batch_*.FAILED"))
    assert len(failed_dirs) == 1
    assert (failed_dirs[0] / "producer-evidence.txt").read_text(
        encoding="utf-8") == "evidence"


def test_failure_after_successful_sentinels_drains_after_cpu_futures_quiesce(tmp_path):
    sentinel = object()
    cpu_queue = streaming.queue.Queue(maxsize=2)
    local_base = tmp_path / "work"
    local_dir = streaming._batch_local_dir(local_base, 0, "demo")
    local_dir.mkdir(parents=True)
    (local_dir / "producer-evidence.txt").write_text("evidence", encoding="utf-8")
    queued_item = ({"name": "demo"}, 0, ["s00"], local_base,
                   "nvrasr_fallback")
    with cpu_queue.mutex:
        cpu_queue.queue.append(queued_item)
        cpu_queue.unfinished_tasks += 1

    stop_event = threading.Event()
    failure_event = threading.Event()
    allow_failure = threading.Event()
    accounted = []

    def delayed_failure():
        assert allow_failure.wait(1.0)
        failure_event.set()
        stop_event.set()
        return 0, []

    def drain():
        while True:
            try:
                item = cpu_queue.get_nowait()
            except streaming.queue.Empty:
                return
            if item is not sentinel:
                accounted.append(item)
                streaming._preserve_failed_batch(local_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(delayed_failure)
        result = streaming._shutdown_pipelined_cpu_queue(
            cpu_queue, 1, stop_event, failure_event, sentinel, lambda: None)
        assert result is True
        assert not failure_event.is_set()
        allow_failure.set()

    start = time.monotonic()
    ok_count, fail_list = streaming._collect_pipelined_cpu_futures(
        [future], failure_event, stop_event, drain)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert (ok_count, fail_list) == (0, [])
    assert failure_event.is_set() and stop_event.is_set()
    assert len(accounted) == 1
    failed_dirs = list(local_base.glob("batch_*.FAILED"))
    assert len(failed_dirs) == 1
    assert (failed_dirs[0] / "producer-evidence.txt").read_text(
        encoding="utf-8") == "evidence"


def _pipelined_exception_args(tmp_path, *, stems, gpus=1,
                              upload_buffer=2):
    audio = tmp_path / "audio"
    ctc = tmp_path / "ctc"
    output = tmp_path / "output"
    audio.mkdir()
    for stem in stems:
        (audio / f"{stem}.wav").write_bytes(b"wav")
    cache = {"output_root": str(output), "datasets": [{
        "name": "demo", "audio_dir": str(audio), "ctc_dir": str(ctc),
        "stems": list(stems),
    }]}
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    local_work = tmp_path / "local"
    return SimpleNamespace(
        batch_cache=cache_path, config=config_path,
        _batch_cache_data=cache, _config={"pipelined": {
            "restore_ctc_cache": False}},
        gpus=gpus, cpu_workers=1, batch_size=1, python="/bin/python",
        limit=0, no_resume=True, local_work=local_work,
        _local_work_drives=(local_work,), prefetch_buffer=2,
        upload_buffer=upload_buffer, mfa_jobs=None, mfa_en_jobs=None,
        _effective_mfa_jobs=0, _effective_mfa_en_jobs=0,
        parallel_datasets=None, pipelined=True,
        _allow_overwrite=False, _allow_force=False,
    )


def test_production_entrypoints_compute_and_propagate_implementation_fingerprint(
        tmp_path, monkeypatch):
    implementation = "f" * 64
    fingerprint_calls = []
    inventory_calls = []

    def fingerprint():
        fingerprint_calls.append(True)
        return implementation

    monkeypatch.setattr(streaming, "_implementation_sha256", fingerprint)
    monkeypatch.setattr(
        streaming, "_validate_frozen_cache_inventory",
        lambda cache: inventory_calls.append(cache),
    )
    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted",
                        lambda: True)
    args = _pipelined_exception_args(tmp_path, stems=["s00"])
    seen_batch = {}
    real_pipelined = streaming.run_pipelined_batch

    def fake_pipelined(received, *, validated_frozen_cache=None):
        seen_batch["implementation_sha256"] = getattr(
            received, "_implementation_sha256", "")
        seen_batch["validated_frozen_cache"] = validated_frozen_cache
        return True

    monkeypatch.setattr(streaming, "run_pipelined_batch", fake_pipelined)
    args.python = sys.executable
    args.limit_datasets = 0
    args.pipelined = True
    args.stage_all = False
    args.force_stage = False
    args.no_overwrite = False
    args.no_force = False
    assert streaming.run_batch(args) is True
    assert seen_batch["implementation_sha256"] == implementation
    assert seen_batch["validated_frozen_cache"] is args._batch_cache_data
    assert inventory_calls == [args._batch_cache_data]

    pipe_root = tmp_path / "pipe"
    pipe_root.mkdir()
    args = _pipelined_exception_args(pipe_root, stems=["s00"])
    args.python = sys.executable
    inventory_calls.clear()
    monkeypatch.setattr(streaming, "run_pipelined_batch", real_pipelined)
    seen_calls = {"trusted": [], "final": {}}
    monkeypatch.setattr(
        streaming, "_load_trusted_batch_evidence",
        lambda *_args, **kwargs: seen_calls["trusted"].append(kwargs)
        or {"trusted": True},
    )

    def fake_finalize(_plans, _trackers, _root, completed, _failed, **kwargs):
        seen_calls["final"].update(kwargs)
        completed.add("demo")
        return []

    monkeypatch.setattr(streaming, "_finalize_dataset_publications", fake_finalize)
    monkeypatch.setattr(streaming, "_save_checkpoint", lambda *_args, **_kwargs: None)
    assert streaming.run_pipelined_batch(args) is True
    assert inventory_calls == [args._batch_cache_data]
    assert fingerprint_calls
    assert seen_calls["trusted"]
    assert all(call.get("implementation_sha256") == implementation
               for call in seen_calls["trusted"])
    assert seen_calls["final"]["implementation_sha256"] == implementation


def test_unexpected_cpu_phase_exception_stops_workers_and_drains_queued_work(
        tmp_path, monkeypatch):
    args = _pipelined_exception_args(tmp_path, stems=["s00", "s01"])
    cpu_started = threading.Event()
    sentinel_success = threading.Event()
    allow_failure = threading.Event()
    gpu_calls = []
    cpu_calls = []
    checkpoint = {}

    monkeypatch.setattr(streaming, "_validate_frozen_cache_inventory",
                        lambda _cache: None)
    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted",
                        lambda: True)
    monkeypatch.setattr(streaming, "_load_trusted_batch_evidence",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(streaming, "_finalize_dataset_publications",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        streaming, "_save_checkpoint",
        lambda _path, completed, failed, _identity: checkpoint.update(
            completed=set(completed), failed=set(failed)),
    )

    def fake_gpu(**kwargs):
        batch_dir = streaming._batch_local_dir(
            kwargs["local_base"], kwargs["batch_idx"], kwargs["ds"]["name"])
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "gpu-evidence.txt").write_text(
            str(kwargs["batch_idx"]), encoding="utf-8")
        gpu_calls.append(kwargs["batch_idx"])
        return True

    def fake_cpu(**kwargs):
        batch_idx = kwargs["batch_idx"]
        cpu_calls.append(batch_idx)
        if batch_idx != 0:
            raise AssertionError("CPU must not consume work after sentinel/failure")
        cpu_started.set()
        assert allow_failure.wait(2.0)
        raise OSError("simulated MFA environment failure")

    monkeypatch.setattr(streaming, "_run_gpu_phase", fake_gpu)
    monkeypatch.setattr(streaming, "_run_cpu_phase", fake_cpu)
    original_shutdown = streaming._shutdown_pipelined_cpu_queue

    def wrapped_shutdown(*args, **kwargs):
        result = original_shutdown(*args, **kwargs)
        if result:
            sentinel_success.set()
            allow_failure.set()
        return result

    monkeypatch.setattr(streaming, "_shutdown_pipelined_cpu_queue",
                        wrapped_shutdown)

    result_holder = {}

    def run():
        result_holder["result"] = streaming.run_pipelined_batch(args)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert cpu_started.wait(2.0)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result_holder["result"] is False
    assert sentinel_success.is_set()
    assert gpu_calls == [0, 1]
    assert cpu_calls == [0]
    assert checkpoint["failed"] == {"demo"}
    failed_dirs = sorted(args.local_work.rglob("batch_*.FAILED"))
    assert len(failed_dirs) == 2
    assert all((path / "gpu-evidence.txt").is_file() for path in failed_dirs)


def test_unexpected_gpu_phase_exception_is_accounted_and_preserved(
        tmp_path, monkeypatch):
    args = _pipelined_exception_args(tmp_path, stems=["s00"])
    gpu_started = threading.Event()
    gpu_calls = []
    cpu_calls = []
    checkpoint = {}

    monkeypatch.setattr(streaming, "_validate_frozen_cache_inventory",
                        lambda _cache: None)
    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted",
                        lambda: True)
    monkeypatch.setattr(streaming, "_load_trusted_batch_evidence",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(streaming, "_finalize_dataset_publications",
                        lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        streaming, "_save_checkpoint",
        lambda _path, completed, failed, _identity: checkpoint.update(
            completed=set(completed), failed=set(failed)),
    )

    def unexpected_gpu(**kwargs):
        batch_dir = streaming._batch_local_dir(
            kwargs["local_base"], kwargs["batch_idx"], kwargs["ds"]["name"])
        batch_dir.mkdir(parents=True, exist_ok=True)
        (batch_dir / "gpu-evidence.txt").write_text("partial", encoding="utf-8")
        gpu_calls.append(kwargs["batch_idx"])
        gpu_started.set()
        raise OSError("simulated NVASR failure")

    def unexpected_cpu(**kwargs):
        cpu_calls.append(kwargs["batch_idx"])
        raise AssertionError("CPU must not run after GPU exception")

    monkeypatch.setattr(streaming, "_run_gpu_phase", unexpected_gpu)
    monkeypatch.setattr(streaming, "_run_cpu_phase", unexpected_cpu)

    result_holder = {}

    def run():
        result_holder["result"] = streaming.run_pipelined_batch(args)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert gpu_started.wait(2.0)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result_holder["result"] is False
    assert gpu_calls == [0]
    assert cpu_calls == []
    assert checkpoint["failed"] == {"demo"}
    failed_dirs = list(args.local_work.rglob("batch_*.FAILED"))
    assert len(failed_dirs) == 1
    assert (failed_dirs[0] / "gpu-evidence.txt").read_text(
        encoding="utf-8") == "partial"


def _make_successful_pipelined_finalize(monkeypatch, checkpoint):
    monkeypatch.setattr(streaming, "_validate_frozen_cache_inventory",
                        lambda _cache: None)
    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted",
                        lambda: True)
    monkeypatch.setattr(streaming, "_save_checkpoint",
                        lambda _path, completed, failed, _identity:
                        checkpoint.update(completed=set(completed),
                                          failed=set(failed)))

    def finalize(_plans, _trackers, _root, completed, _failed, **_kwargs):
        completed.add("demo")
        return []

    monkeypatch.setattr(streaming, "_finalize_dataset_publications", finalize)


def test_pipelined_summary_counts_all_trusted_batches_as_success(
        tmp_path, monkeypatch, capsys):
    args = _pipelined_exception_args(
        tmp_path, stems=[f"s{index:02d}" for index in range(24)])
    checkpoint = {}
    _make_successful_pipelined_finalize(monkeypatch, checkpoint)
    monkeypatch.setattr(streaming, "_load_trusted_batch_evidence",
                        lambda *_args, **_kwargs: {"trusted": True})

    assert streaming.run_pipelined_batch(args) is True
    assert checkpoint["completed"] == {"demo"}
    assert checkpoint["failed"] == set()
    assert "24/24 batches OK" in capsys.readouterr().out


def test_pipelined_summary_counts_trusted_and_new_successes_only(
        tmp_path, monkeypatch, capsys):
    args = _pipelined_exception_args(tmp_path, stems=["s00", "s01"])
    checkpoint = {}
    _make_successful_pipelined_finalize(monkeypatch, checkpoint)
    trusted_calls = []
    gpu_calls = []
    cpu_calls = []

    def trusted(_root, batch_idx, *_args, **_kwargs):
        trusted_calls.append(batch_idx)
        return {"trusted": True} if batch_idx == 0 else None

    monkeypatch.setattr(streaming, "_load_trusted_batch_evidence", trusted)
    monkeypatch.setattr(streaming, "_run_gpu_phase",
                        lambda **kwargs: gpu_calls.append(kwargs["batch_idx"]) or True)
    monkeypatch.setattr(streaming, "_run_cpu_phase",
                        lambda **kwargs: cpu_calls.append(kwargs["batch_idx"]) or True)

    assert streaming.run_pipelined_batch(args) is True
    assert trusted_calls == [0, 1]
    assert gpu_calls == [1]
    assert cpu_calls == [1]
    assert "2/2 batches OK" in capsys.readouterr().out
