"""Focused checkpoint identity and no-work result contracts."""

import json
import inspect
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


def test_staged_limit_caps_partitioned_inventory_and_checkpoint_identity(tmp_path):
    complete, incomplete = streaming._limit_stem_partitions(
        ["ready-1", "ready-2"], ["fallback-1", "fallback-2"], 3)
    assert complete == ["ready-1", "ready-2"]
    assert incomplete == ["fallback-1"]

    cache = _cache({"name": "a", "audio_dir": "a",
                    "stems": ["1", "2", "3", "4"]})
    limited_identity = streaming._checkpoint_identity(
        cache, _args(tmp_path, limit=3))
    unlimited_identity = streaming._checkpoint_identity(
        cache, _args(tmp_path, limit=0))
    assert limited_identity != unlimited_identity
    assert len(limited_identity["batch_identities"]) == 2


def test_batch_config_output_dir_overrides_frozen_cache_destination(tmp_path):
    cache = _cache({"name": "a", "audio_dir": "a", "ctc_dir": "old/a",
                    "stems": ["1"]})
    args = _args(tmp_path)
    args.output_dir = None
    args.nas_output = None
    args._config = {"output_dir": str(tmp_path / "fresh")}

    overridden = streaming._apply_batch_output_override(cache, args)

    assert overridden is not cache
    assert overridden["output_root"] == str(tmp_path / "fresh")
    assert overridden["datasets"][0]["ctc_dir"] == str(tmp_path / "fresh" / "a")
    assert cache["datasets"][0]["ctc_dir"] == "old/a"


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


def test_no_resume_replaces_stale_batch_progress_identity(tmp_path):
    checkpoint = tmp_path / "run.cache.checkpoint.json"
    stale = {"input_manifest_digest": "stale"}
    current = {"input_manifest_digest": "current"}
    streaming._save_batch_progress(checkpoint, "a", 1, 0, 2, stale)

    with pytest.raises(RuntimeError, match="identity changed"):
        streaming._save_batch_progress(checkpoint, "a", 1, 0, 2, current)

    streaming._save_batch_progress(
        checkpoint, "a", 1, 0, 2, current,
        reset_on_identity_mismatch=True)
    progress = checkpoint.with_name(
        "run.cache.checkpoint.batch_progress.json")
    payload = json.loads(progress.read_text(encoding="utf-8"))
    assert payload["identity"] == current
    assert payload["datasets"] == {"a": {"done": 1, "fail": 0, "total": 2}}


def test_run_batch_reports_already_completed_as_success(tmp_path):
    args, cache = _completed_run_fixture(tmp_path)
    dataset_root = tmp_path / "output" / "a"
    local = _checkpoint_batch_fixture(tmp_path, ["1"])
    assert streaming._publish_batch_to_staging(
        local, dataset_root, 0, ["1"],
        config_sha256=streaming._sha256_file(args.config),
        cache_sha256=streaming._sha256_file(args.batch_cache),
        implementation_sha256=args._implementation_sha256)
    streaming._aggregate_dataset_publication(
        dataset_root, "a", ["1"], [0],
        config_sha256=streaming._sha256_file(args.config),
        cache_sha256=streaming._sha256_file(args.batch_cache),
        implementation_sha256=args._implementation_sha256,
        receipt_mode="staged")
    identity = streaming._checkpoint_identity(cache, args)
    _write_checkpoint(args.batch_cache.with_name("run.cache.checkpoint.json"), identity,
                      completed=["a"])
    assert streaming.run_batch(args) is True


def test_completed_scan_cache_uses_verified_final_source_stems(tmp_path):
    args, cache = _completed_run_fixture(tmp_path)
    dataset = cache["datasets"][0]
    dataset.pop("stems")
    row = dataset["source_inventory"]["files"][0]
    row["path"] = "1.wav"
    args.batch_cache.write_text(json.dumps(cache), encoding="utf-8")
    args._batch_cache_data = cache
    args.no_resume = False
    root = tmp_path / "output" / "a"
    local = _checkpoint_batch_fixture(tmp_path, ["1"])
    assert streaming._publish_batch_to_staging(
        local, root, 0, ["1"],
        config_sha256=streaming._sha256_file(args.config),
        cache_sha256=streaming._sha256_file(args.batch_cache),
        implementation_sha256=args._implementation_sha256)
    streaming._aggregate_dataset_publication(
        root, "a", ["1"], [0],
        config_sha256=streaming._sha256_file(args.config),
        cache_sha256=streaming._sha256_file(args.batch_cache),
        implementation_sha256=args._implementation_sha256,
        receipt_mode="staged")
    _write_checkpoint(
        args.batch_cache.with_name("run.cache.checkpoint.json"),
        streaming._checkpoint_identity(cache, args), completed=["a"])

    assert streaming.run_batch(args) is True


def test_run_batch_rejects_completed_checkpoint_when_source_hash_changed(tmp_path):
    args, cache = _completed_run_fixture(tmp_path)
    cache["datasets"][0]["source_inventory"]["files"][0]["sha256"] = "0" * 64
    args.batch_cache.write_text(json.dumps(cache), encoding="utf-8")
    identity = streaming._checkpoint_identity(cache, args)
    _write_checkpoint(args.batch_cache.with_name("run.cache.checkpoint.json"),
                      identity, completed=["a"])

    assert streaming.run_batch(args) is False


def test_run_batch_rejects_completed_checkpoint_without_valid_final_receipt(tmp_path):
    args, cache = _completed_run_fixture(tmp_path)
    identity = streaming._checkpoint_identity(cache, args)
    _write_checkpoint(args.batch_cache.with_name("run.cache.checkpoint.json"),
                      identity, completed=["a"])
    output = tmp_path / "output" / "a" / "output"
    output.mkdir(parents=True)
    (output / "1.TextGrid").write_text("unreceipted\n", encoding="utf-8")

    assert streaming.run_batch(args) is False


def test_run_batch_no_resume_freezes_and_persists_legacy_source_inventory(
        tmp_path, monkeypatch):
    args, cache = _completed_run_fixture(tmp_path)
    cache["datasets"][0].pop("stems")
    cache["datasets"][0].pop("source_inventory")
    args.batch_cache.write_text(json.dumps(cache), encoding="utf-8")
    args._batch_cache_data = cache
    args.no_resume = True
    args.pipelined = True
    args.cpu_workers = 1
    args._local_work_drives = (args.local_work,)
    args.output_dir = None
    args.nas_output = None
    args.force_stage = False

    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted", lambda: None)
    monkeypatch.setattr(streaming, "plan_streaming_resources", lambda **_kwargs: {
        "cpu_workers": 1, "mfa_jobs_per_worker": 1,
        "mfa_en_jobs_per_worker": 1, "worker_cpu_budget": 1,
    })
    received = {}

    def dispatch(received_args, *, validated_frozen_cache=None):
        received["cache"] = validated_frozen_cache
        return True

    monkeypatch.setattr(streaming, "run_pipelined_batch", dispatch)

    assert streaming.run_batch(args) is True
    persisted = json.loads(args.batch_cache.read_text(encoding="utf-8"))
    frozen = persisted["datasets"][0]["source_inventory"]
    assert frozen["count"] == 1
    assert frozen["files"] == [{
        "stem": "1", "path": "1.wav", "size": 3,
        "sha256": streaming._sha256_file(tmp_path / "audio" / "1.wav"),
    }]
    assert received["cache"]["datasets"][0]["source_inventory"] == frozen


def test_legacy_completed_checkpoint_cannot_survive_live_inventory_freeze(
        tmp_path):
    args, cache = _completed_run_fixture(tmp_path)
    cache["datasets"][0].pop("source_inventory")
    args.batch_cache.write_text(json.dumps(cache), encoding="utf-8")
    args._batch_cache_data = cache
    old_cache_sha = streaming._sha256_file(args.batch_cache)
    local = _checkpoint_batch_fixture(tmp_path, ["1"])
    dataset_root = tmp_path / "output" / "a"
    assert streaming._publish_batch_to_staging(
        local, dataset_root, 0, ["1"],
        config_sha256=streaming._sha256_file(args.config),
        cache_sha256=old_cache_sha,
        implementation_sha256=streaming._implementation_sha256())
    streaming._aggregate_dataset_publication(
        dataset_root, "a", ["1"], [0],
        config_sha256=streaming._sha256_file(args.config),
        cache_sha256=old_cache_sha,
        implementation_sha256=streaming._implementation_sha256(),
        receipt_mode="staged")
    old_identity = streaming._checkpoint_identity(cache, args)
    _write_checkpoint(args.batch_cache.with_name("run.cache.checkpoint.json"),
                      old_identity, completed=["a"])

    assert streaming.run_batch(args) is False
    updated = json.loads(args.batch_cache.read_text(encoding="utf-8"))
    assert "source_inventory" in updated["datasets"][0]


def test_direct_pipelined_entrypoint_freezes_before_checkpoint_identity(tmp_path):
    args, cache = _completed_run_fixture(tmp_path)
    cache["datasets"][0].pop("stems")
    cache["datasets"][0].pop("source_inventory")
    args.batch_cache.write_text(json.dumps(cache), encoding="utf-8")
    args._batch_cache_data = cache
    args.no_resume = False
    args.pipelined = True
    old_identity = streaming._checkpoint_identity(cache, args)
    _write_checkpoint(args.batch_cache.with_name("run.cache.checkpoint.json"),
                      old_identity, completed=["a"])

    assert streaming.run_pipelined_batch(args) is False
    persisted = json.loads(args.batch_cache.read_text(encoding="utf-8"))
    assert "source_inventory" in persisted["datasets"][0]


def _completed_run_fixture(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = audio / "1.wav"
    wav.write_bytes(b"wav")
    cache = _cache({
        "name": "a", "audio_dir": str(audio), "stems": ["1"],
        "source_inventory": {"count": 1, "files": [{
            "stem": "1", "size": wav.stat().st_size,
            "sha256": streaming._sha256_file(wav),
        }]},
    })
    cache["output_root"] = str(tmp_path / "output")
    cache_path = tmp_path / "run.cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    args = _args(
        tmp_path, batch_size=2, batch_cache=cache_path, config=config,
        no_resume=False, limit_datasets=0, local_work=tmp_path / "work",
        python=sys.executable, gpus=1, cpu_workers=1, pipelined=False,
        stage_all=True, no_overwrite=False, no_force=False,
        parallel_datasets=1, mfa_jobs=None, mfa_en_jobs=None,
    )
    args.config = config
    args._config = {"streaming": {"batch_size": 2}}
    args._implementation_sha256 = streaming._implementation_sha256()
    return args, cache


def _checkpoint_batch_fixture(tmp_path, stems):
    local = tmp_path / "local-complete"
    output = local / "output"
    filtered = local / "workspace" / "filtered"
    output.mkdir(parents=True)
    filtered.mkdir(parents=True)
    for stem in stems:
        (output / f"{stem}.TextGrid").write_text(
            f"grid-{stem}\n", encoding="utf-8")
    (output / "postprocess_report.jsonl").write_text(
        "".join(json.dumps({"stem": stem, "status": "ok"}) + "\n"
                for stem in stems), encoding="utf-8")
    receipt = streaming.make_pipeline_accounting_receipt(
        stems, stems, [], stems, [], run_id="fixture", mode="pipelined",
        route=["fixture"], shards=[{"shard_id": "fixture", "stems": stems}],
    )
    (local / "workspace" / ".pipeline_run_receipt_v2.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    return local


@pytest.mark.parametrize(("runner_result", "expected_rc"), [(True, 0), (False, 1)])
def test_streaming_main_maps_batch_result_to_cli_exit(monkeypatch, tmp_path,
                                                       runner_result, expected_rc):
    monkeypatch.setattr(streaming, "run_batch", lambda _args: runner_result)
    monkeypatch.setattr(sys, "argv", [
        "streaming_pipeline.py", "--batch-cache", str(tmp_path / "cache.json"),
        "--local-work", str(tmp_path / "work"), "--gpus", "1",
    ])
    assert streaming.main() == expected_rc


def _single_dataset_paths(tmp_path: Path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "LAria.wav").write_bytes(b"wav")
    ctc = tmp_path / "ctc"
    output = tmp_path / "output" / "LAria"
    config = tmp_path / "config.yaml"
    config.write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    return audio, ctc, output, config


def test_single_dataset_zero_of_one_staged_fails_closed(
        monkeypatch, tmp_path):
    audio, ctc, output, config = _single_dataset_paths(tmp_path)
    calls = {"process": 0, "upload": 0}

    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted", lambda: None)

    def stage_failure(**kwargs):
        local = streaming._batch_local_dir(
            kwargs["local_base"], kwargs["batch_idx"], kwargs["ds"]["name"])
        local.mkdir(parents=True)
        assert kwargs["mode"] == "nvrasr_fallback"
        return local, 0.0, 18

    monkeypatch.setattr(streaming, "_stage_one_batch", stage_failure)
    monkeypatch.setattr(
        streaming, "_process_one_batch",
        lambda **_kwargs: calls.__setitem__("process", calls["process"] + 1))
    monkeypatch.setattr(
        streaming, "_upload_one_batch",
        lambda **_kwargs: calls.__setitem__("upload", calls["upload"] + 1))

    assert streaming.run_single_dataset(
        nas_ctc=str(ctc), nas_audio=str(audio), nas_output=str(output),
        config=config, local_work=tmp_path / "work", batch_size=1,
        stems_override=["LAria"], python_path=sys.executable,
        parallel_batches=1, mode=None,
    ) is False
    assert calls == {"process": 0, "upload": 0}


def test_single_dataset_propagates_fallback_mode_from_config(
        monkeypatch, tmp_path):
    audio, ctc, output, config = _single_dataset_paths(tmp_path)
    calls = {"stage_mode": None, "process_mode": None, "upload": 0}

    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted", lambda: None)

    def stage_ok(**kwargs):
        calls["stage_mode"] = kwargs["mode"]
        local = streaming._batch_local_dir(
            kwargs["local_base"], kwargs["batch_idx"], kwargs["ds"]["name"])
        local.mkdir(parents=True)
        return local, 0.0, 0

    def process_ok(**kwargs):
        calls["process_mode"] = kwargs["mode"]
        return True

    def upload_ok(**_kwargs):
        calls["upload"] += 1
        return True

    monkeypatch.setattr(streaming, "_stage_one_batch", stage_ok)
    monkeypatch.setattr(streaming, "_process_one_batch", process_ok)
    monkeypatch.setattr(streaming, "_upload_one_batch", upload_ok)

    assert streaming.run_single_dataset(
        nas_ctc=str(ctc), nas_audio=str(audio), nas_output=str(output),
        config=config, local_work=tmp_path / "work", batch_size=1,
        stems_override=["LAria"], python_path=sys.executable,
        parallel_batches=1, mode=None,
    ) is True
    assert calls == {"stage_mode": "nvrasr_fallback",
                     "process_mode": "nvrasr_fallback", "upload": 1}


def test_single_dataset_process_failure_is_not_uploaded(monkeypatch, tmp_path):
    audio, ctc, output, config = _single_dataset_paths(tmp_path)
    uploads = []

    monkeypatch.setattr(streaming, "_ensure_mfa_model_extracted", lambda: None)

    def stage_ok(**kwargs):
        local = streaming._batch_local_dir(
            kwargs["local_base"], kwargs["batch_idx"], kwargs["ds"]["name"])
        local.mkdir(parents=True)
        return local, 0.0, 0

    monkeypatch.setattr(streaming, "_stage_one_batch", stage_ok)
    monkeypatch.setattr(streaming, "_process_one_batch", lambda **_kwargs: False)
    monkeypatch.setattr(streaming, "_upload_one_batch",
                        lambda **kwargs: uploads.append(kwargs) or True)

    assert streaming.run_single_dataset(
        nas_ctc=str(ctc), nas_audio=str(audio), nas_output=str(output),
        config=config, local_work=tmp_path / "work", batch_size=1,
        stems_override=["LAria"], python_path=sys.executable,
        parallel_batches=1, mode="nvrasr_fallback",
    ) is False
    assert uploads == []


def test_sequential_failure_checkpoint_never_marks_completed(
        monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    checkpoint = tmp_path / "run.checkpoint.json"
    completed: set[str] = set()
    failed: set[str] = set()
    modes = []
    args = SimpleNamespace(
        local_work=tmp_path / "work",
        _local_work_drives=(tmp_path / "work",),
        _config={"mode": "nvrasr_fallback"}, config=config,
        batch_size=1, limit=0, python=sys.executable,
        prefetch_buffer=1, upload_buffer=1, stage_all=True,
    )
    cache = {"mode": "nvrasr_fallback", "output_root": str(tmp_path / "out"),
             "datasets": [{"name": "LAria", "ctc_dir": str(tmp_path / "ctc"),
                           "audio_dir": str(tmp_path / "audio"),
                           "stems": ["LAria"]}]}
    monkeypatch.setattr(streaming, "_checkpoint_identity",
                        lambda *_args: {"identity": "test"})

    def fail_dataset(**kwargs):
        modes.append(kwargs["mode"])
        return False

    monkeypatch.setattr(streaming, "run_single_dataset", fail_dataset)

    assert streaming._run_batch_sequential(
        args, cache["datasets"], cache, checkpoint, completed, failed) is False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["completed"] == []
    assert payload["failed"] == ["LAria"]
    assert modes == ["nvrasr_fallback"]


def test_single_worker_batch_mode_keeps_per_batch_classifier():
    source = inspect.getsource(streaming.run_batch)
    assert "return _run_batch_sequential(" not in source
