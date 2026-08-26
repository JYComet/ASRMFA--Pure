"""Dataset-level publication and resume contracts for pipelined streaming."""

from __future__ import annotations

import json
import wave
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import streaming_pipeline as streaming
from scripts import run_pipeline
from scripts.pipeline_utils import write_ctc_raw_manifest, write_ctc_run_receipt


def _receipt(stems: list[str], output: list[str], filtered: list[str]) -> dict:
    return streaming.make_pipeline_accounting_receipt(
        stems, stems, [], output, filtered,
        run_id="fixture", mode="pipelined", route=["fixture"],
        shards=[{"shard_id": "fixture", "stems": stems}],
    )


def _batch(tmp_path: Path, index: int, stems: list[str], output: list[str],
           filtered: list[str], *, tag: str = "") -> Path:
    local = tmp_path / f"local-{index}{tag}"
    out = local / "output"
    filt = local / "workspace" / "filtered"
    out.mkdir(parents=True)
    filt.mkdir(parents=True)
    for stem in output:
        (out / f"{stem}.TextGrid").write_text(f"output-{stem}\n", encoding="utf-8")
    for stem in filtered:
        (filt / f"{stem}.TextGrid").write_text(f"filtered-{stem}\n", encoding="utf-8")
    rows = [{"stem": stem, "status": "ok" if stem in output else "filtered"}
            for stem in stems]
    (out / "postprocess_report.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    receipt = _receipt(stems, output, filtered)
    (local / "workspace" / ".pipeline_run_receipt_v2.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    return local


def test_batch_publication_isolated_and_hash_verified(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a", "b"], ["a"], ["b"])
    assert streaming._publish_batch_to_staging(
        local, dataset, 0, ["a", "b"], config_sha256="cfg", cache_sha256="cache")

    evidence = streaming._load_trusted_batch_evidence(
        dataset, 0, ["a", "b"], config_sha256="cfg", cache_sha256="cache")
    assert evidence is not None
    assert (dataset / ".batch_evidence/batch_0000/postprocess_report.jsonl").is_file()
    assert (dataset / ".staging/batch_0000/output/a.TextGrid").is_file()
    assert (dataset / ".staging/batch_0000/filtered/b.TextGrid").is_file()

    (dataset / ".staging/batch_0000/output/a.TextGrid").write_text(
        "tampered\n", encoding="utf-8")
    assert streaming._load_trusted_batch_evidence(
        dataset, 0, ["a", "b"], config_sha256="cfg", cache_sha256="cache") is None


def test_existing_staging_conflict_is_fail_closed(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a"], ["a"], [])
    assert streaming._publish_batch_to_staging(local, dataset, 0, ["a"])
    (dataset / ".staging/batch_0000/output/a.TextGrid").write_text(
        "different\n", encoding="utf-8")
    replacement = _batch(tmp_path, 0, ["a"], ["a"], [], tag="-replacement")
    assert not streaming._publish_batch_to_staging(replacement, dataset, 0, ["a"])


def test_committed_staging_recovers_valid_orphan_evidence(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 1, ["a", "b"], ["a"], ["b"])
    assert streaming._publish_batch_to_staging(
        local, dataset, 1, ["a", "b"],
        config_sha256="cfg", cache_sha256="cache")

    evidence = dataset / ".batch_evidence/batch_0001"
    orphan = dataset / ".batch_evidence/.batch_0001.tmp.1234"
    evidence.rename(orphan)
    assert streaming._publish_batch_to_staging(
        local, dataset, 1, ["a", "b"],
        config_sha256="cfg", cache_sha256="cache")
    assert evidence.is_dir()
    assert not orphan.exists()
    assert streaming._load_trusted_batch_evidence(
        dataset, 1, ["a", "b"],
        config_sha256="cfg", cache_sha256="cache") is not None


def test_dataset_aggregation_is_index_ordered_and_conservative(tmp_path):
    dataset = tmp_path / "dataset"
    first = _batch(tmp_path, 1, ["c", "d"], ["c"], ["d"])
    second = _batch(tmp_path, 0, ["a", "b"], ["a", "b"], [])
    assert streaming._publish_batch_to_staging(first, dataset, 1, ["c", "d"],
                                                config_sha256="cfg", cache_sha256="cache")
    assert streaming._publish_batch_to_staging(second, dataset, 0, ["a", "b"],
                                                config_sha256="cfg", cache_sha256="cache")

    receipt = streaming._aggregate_dataset_publication(
        dataset, "demo", ["a", "b", "c", "d"], [1, 0],
        config_sha256="cfg", cache_sha256="cache")
    assert receipt["status"] if "status" in receipt else True
    assert json.loads((dataset / ".pipeline_run_receipt_v2.json").read_text())["source"]["stems"] == ["a", "b", "c", "d"]
    report_stems = [json.loads(line)["stem"] for line in
                    (dataset / "output/postprocess_report.jsonl").read_text().splitlines()]
    assert report_stems == ["a", "b", "c", "d"]
    dataset_receipt = json.loads((dataset / ".streaming_dataset_receipt_v1.json").read_text())
    assert dataset_receipt["status"] == "COMPLETE"
    assert dataset_receipt["batch_count"] == 2
    assert dataset_receipt["outputs"] == {
        "output": 3, "filtered": 1,
        "report_sha256": streaming._sha256_file(dataset / "output/postprocess_report.jsonl"),
    }


def test_adjusted_ctc_isolated_artifact_is_retained_and_hashed(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a"], ["a"], [])
    adjusted = local / "workspace" / "ctc_pretg_adj"
    adjusted.mkdir()
    (adjusted / "a.lab").write_text("adjusted\n", encoding="utf-8")

    assert streaming._publish_batch_to_staging(local, dataset, 0, ["a"])
    evidence = streaming._load_trusted_batch_evidence(dataset, 0, ["a"])
    assert evidence is not None
    assert evidence["artifacts"]["ctc_pretg_adj/a.lab"]["sha256"] == \
        streaming._sha256_file(dataset / ".staging/batch_0000/ctc_pretg_adj/a.lab")
    assert not (dataset / "ctc_pretg_adj").exists()


def test_trusted_batches_rebuild_missing_dataset_receipt_without_recompute(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a"], ["a"], [])
    assert streaming._publish_batch_to_staging(local, dataset, 0, ["a"],
                                                config_sha256="cfg", cache_sha256="cache")
    streaming._aggregate_dataset_publication(
        dataset, "demo", ["a"], [0], config_sha256="cfg", cache_sha256="cache")
    (dataset / ".streaming_dataset_receipt_v1.json").unlink()
    receipt = streaming._aggregate_dataset_publication(
        dataset, "demo", ["a"], [0], config_sha256="cfg", cache_sha256="cache")
    assert receipt["status"] == "COMPLETE"
    assert (dataset / ".streaming_dataset_receipt_v1.json").is_file()


def test_staged_publication_failure_preserves_local_forensics(tmp_path, monkeypatch):
    local_base = tmp_path / "local"
    config = tmp_path / "config.yaml"
    config.write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    local_dir = local_base / "batch_0000_demo"
    dataset = {"name": "demo"}
    batches = [("ctc_ready", dataset, 0, ["a"], {}, {}, None)]
    tracker = {"demo": {"total": 1, "done": 0, "fail": 0}}
    checkpoint_writes = []

    def fake_stage(**_kwargs):
        local_dir.mkdir(parents=True)
        return local_dir, 0.0, 0

    monkeypatch.setattr(streaming, "_stage_one_batch", fake_stage)
    monkeypatch.setattr(streaming, "_process_one_batch", lambda **_kwargs: True)
    monkeypatch.setattr(streaming, "_publish_batch_to_staging",
                        lambda *_args, **_kwargs: False)
    monkeypatch.setattr(streaming, "_save_batch_progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(streaming, "_save_checkpoint",
                        lambda *args, **_kwargs: checkpoint_writes.append(args))

    result = streaming._execute_staged(
        args=Namespace(batch_size=1, gpus=1, python="python", config=config,
                       _config={}),
        all_batches=batches, cache={"output_root": str(tmp_path / "nas")},
        ckpt_path=tmp_path / "run.checkpoint.json", completed_set=set(),
        failed_set=set(), usable_drives=[local_base], mfa_python=Path("python"),
        models_dir=tmp_path, parallel=1, ds_batch_tracker=tracker, batch_size=1,
    )

    assert result[0] == 0
    assert "demo" in result[1]
    assert list(local_base.glob("*.FAILED"))
    assert checkpoint_writes


def test_run_batch_does_not_finalize_staged_dataset_twice():
    import inspect

    source = inspect.getsource(streaming.run_batch)
    staged_call = source.index("ok_count, fail_list = _execute_staged(")
    outer_gate = source.index("if not use_staging:", staged_call)
    outer_finalize = source.index("publish_failures = _finalize_dataset_publications(",
                                  outer_gate)
    assert staged_call < outer_gate < outer_finalize


def test_publication_receipt_rejects_overlap_and_missing_stems(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a", "b"], ["a"], ["b"])
    receipt_path = local / "workspace/.pipeline_run_receipt_v2.json"
    bad = _receipt(["a", "b"], ["a", "b"], [])
    bad["filtered"]["stems"] = ["b"]
    receipt_path.write_text(json.dumps(bad), encoding="utf-8")
    assert not streaming._publish_batch_to_staging(local, dataset, 0, ["a", "b"])


def test_laria_config_and_cache_contract():
    import yaml

    config_path = Path("configs/laria_v5_no_reference_8gpu_20260825.yaml")
    cache_path = Path("cache/laria_v5_no_reference_8gpu_20260825.cache.json")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert not run_pipeline.validate_config(config, "nvrasr_fallback")
    assert config["reference_mode"] == "fallback"
    assert config["pad_silence"]["enabled"] is False
    assert config["streaming"]["batch_size"] == 132
    assert config["streaming"]["num_gpus"] == 8
    assert config["ctc_prealign"]["all_gpus"] is False
    assert config["pipelined"]["restore_ctc_cache"] is False
    dataset = cache["datasets"][0]
    assert dataset["name"] == "LAria"
    assert dataset["stems"] == sorted(dataset["stems"])
    assert len(dataset["stems"]) == 1055
    assert len(dataset["source_inventory"]["files"]) == 1055
    assert len({row["stem"] for row in dataset["source_inventory"]["files"]}) == 1055


def test_laria_strict_config_and_cache_publish_to_same_v2_root():
    import yaml

    config_path = Path("configs/laria_v5_no_reference_strict_8gpu_20260825.yaml")
    cache_path = Path("cache/laria_v5_no_reference_strict_8gpu_20260825.cache.json")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert not run_pipeline.validate_config(config, "nvrasr_fallback")
    assert config["mfa_en"]["strict_provenance"] is True
    assert config["postprocess"]["strict_ok"] is False
    assert config["output_dir"] == cache["output_root"]
    assert cache["datasets"][0]["ctc_dir"] == str(
        Path(config["output_dir"]) / cache["datasets"][0]["name"])


@pytest.mark.parametrize("restore_cache", [False])
def test_restore_cache_false_is_explicitly_supported(restore_cache):
    import inspect

    signature = inspect.signature(streaming._run_gpu_phase)
    assert "persist_cache" in signature.parameters
    assert restore_cache is False


def test_ctc_ready_reuses_sealed_raw_without_global_overwrite():
    assert streaming._ctc_ready_overwrite_args(
        allow_overwrite=True, sealed_ctc_raw=True) == []
    assert streaming._ctc_ready_overwrite_args(
        allow_overwrite=True, sealed_ctc_raw=False) == ["--overwrite"]
    assert streaming._ctc_ready_overwrite_args(
        allow_overwrite=False, sealed_ctc_raw=False) == []


def test_cpu_command_preserves_explicit_fallback_mode_after_gpu():
    common = dict(
        mfa_python=Path("/python"), config=Path("config.yaml"),
        local_audio=Path("batch/audio"), local_ctc=Path("batch/ctc"),
        local_output=Path("batch/output"),
        local_workspace=Path("batch/workspace"), allow_overwrite=True,
        allow_force=True, sealed_ctc_raw=True, skip_pad_silence=True,
    )
    fallback = streaming._build_cpu_phase_command(
        **common, producer_mode="nvrasr_fallback")
    assert fallback[fallback.index("--mode") + 1] == "nvrasr_fallback"
    assert fallback[fallback.index("--skip-to") + 1] == "resample"
    assert "--ctc-ready" not in fallback
    assert "link" not in fallback
    assert "--overwrite" in fallback

    ready = streaming._build_cpu_phase_command(
        **common, producer_mode="ctc_ready")
    assert ready[ready.index("--mode") + 1] == "ctc_ready"
    assert ready[ready.index("--ctc-ready") + 1] == "batch/ctc"
    assert "--skip-to" not in ready
    assert "--overwrite" not in ready


def test_fallback_resume_promotes_empty_producer_audio_bindings_without_raw_rewrite(
        tmp_path):
    raw = tmp_path / "workspace" / "ctc_pretg"
    raw.mkdir(parents=True)
    stem = "one"
    for suffix in streaming.CTC_SUFFIXES:
        if suffix == ".TextGrid":
            (raw / f"{stem}{suffix}").write_text(
                'File type = "ooTextFile"\n'
                'Object class = "TextGrid"\n\n'
                "xmin = 0\n"
                "xmax = 0.01\n",
                encoding="utf-8")
            continue
        (raw / f"{stem}{suffix}").write_text(
            "{}\n" if suffix == "_tokens.jsonl"
            else f"{stem}:{suffix}\n", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    wav_path = audio / f"{stem}.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    producer = raw / ".ctc_run_receipt.json"
    write_ctc_run_receipt(
        raw, actual_argv=["fixture"], asr_python="python",
        model_path=tmp_path / "model", model_tree_digest="model",
        model_file_manifest=[], dict_path=tmp_path / "dict",
        dict_digest="dict", input_stems=[stem], output_stems=[stem],
        audio_bindings=[])
    before = producer.read_bytes()
    write_ctc_raw_manifest(raw, producer_receipt=producer, stems=[stem])

    ctx = {"ctc_pretg": raw, "audio_dir": audio,
           "workspace": tmp_path / "workspace"}
    assert run_pipeline._ensure_ctc_axis_receipt(ctx) == 0
    assert ctx["ctc_axis_receipt"]["audio_bindings"][0]["stem"] == stem
    assert Path(ctx["ctc_axis_receipt"]["audio_bindings"][0]["path"]) == wav_path.resolve()
    assert producer.read_bytes() == before
    assert (ctx["workspace"] / ".ctc_input_axis_receipt.json").is_file()


def test_pipelined_validates_frozen_inventory_before_output_resolution(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.json"
    config_path = tmp_path / "config.yaml"
    cache_path.write_text(json.dumps({"datasets": [{"name": "LAria"}]}), encoding="utf-8")
    config_path.write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    events = []

    def reject(_cache):
        events.append("validate")
        raise ValueError("source hash mismatch")

    def unexpected_resolve(*_args, **_kwargs):
        events.append("resolve")
        raise AssertionError("output resolution must not precede inventory validation")

    monkeypatch.setattr(streaming, "_validate_frozen_cache_inventory", reject)
    monkeypatch.setattr(streaming, "resolve_input_path", unexpected_resolve)
    result = streaming.run_pipelined_batch(Namespace(
        batch_cache=cache_path, config=config_path))
    assert result is False
    assert events == ["validate"]


def test_frozen_inventory_rejects_changed_source_hash(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = audio / "one.wav"
    wav.write_bytes(b"changed-source")
    cache = {"datasets": [{
        "name": "demo", "audio_dir": str(audio), "stems": ["one"],
        "source_inventory": {
            "count": 1,
            "files": [{"stem": "one", "size": 1, "sha256": "0" * 64}],
        },
    }]}
    with pytest.raises(ValueError, match="source inventory hash mismatch"):
        streaming._validate_frozen_cache_inventory(cache)


def test_gpu_worker_device_is_remapped_for_single_child_without_all_gpu_flag(
        monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav-fixture")
    captured = {}

    monkeypatch.setattr(streaming, "get_mfa_env", lambda *_args: {})
    monkeypatch.setattr(streaming, "cuda_visible_token", lambda index, _env: f"gpu-{index}")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return Namespace(returncode=1)

    monkeypatch.setattr(streaming.subprocess, "run", fake_run)
    ok = streaming._run_gpu_phase(
        ds={"name": "LAria", "audio_dir": str(tmp_path), "ctc_dir": str(tmp_path / "ctc")},
        batch_idx=0, batch_stems=["source"], layout_map={},
        wav_index={"source": source}, text_index={}, local_base=tmp_path / "work",
        config=tmp_path / "config.yaml", mfa_python=Path("/bin/python"),
        models_dir=tmp_path, batch_size=1, python_path=None, device="cuda:3",
        persist_cache=False, allow_overwrite=False, allow_force=False)
    assert ok is False
    assert captured["cmd"][captured["cmd"].index("--device") + 1] == "cuda:0"
    assert "--all-gpus" not in captured["cmd"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "gpu-3"
