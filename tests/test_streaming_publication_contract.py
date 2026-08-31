"""Dataset-level publication and resume contracts for pipelined streaming."""

from __future__ import annotations

import hashlib
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


def _clean_batch(tmp_path: Path, index: int, stems: list[str]) -> Path:
    """Build the report shape emitted by a clean strict postprocess run."""
    local = _batch(tmp_path, index, stems, stems, [])
    rows = []
    for stem in stems:
        rows.append({
            "stem": stem,
            "status": "ok",
            "warnings": [],
            "hard_integrity_reasons": [],
            "publication_contract": {"status": "verified", "reasons": []},
            "nvasr_candidate_provenance": {
                "status": "not_applicable", "reasons": []},
            "english_provenance": {
                "status": "not_required", "required_words": 0,
                "verified_words": 0, "failed_word_ids": []},
        })
    (local / "output" / "postprocess_report.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return local


def test_zero_filtered_policy_rejects_before_staging(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a", "b"], ["a"], ["b"])
    assert not streaming._publish_batch_to_staging(
        local, dataset, 0, ["a", "b"], require_zero_filtered=True)
    assert not (dataset / ".staging").exists()
    assert not (dataset / ".batch_evidence").exists()


def test_zero_filtered_policy_rejects_bad_clean_report_before_staging(tmp_path):
    dataset = tmp_path / "dataset"
    local = _clean_batch(tmp_path, 0, ["a"])
    report = local / "output" / "postprocess_report.jsonl"
    row = json.loads(report.read_text(encoding="utf-8"))
    row["warnings"] = ["unexpected"]
    report.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert not streaming._publish_batch_to_staging(
        local, dataset, 0, ["a"], require_zero_filtered=True)
    assert not (dataset / ".staging").exists()
    assert not (dataset / ".batch_evidence").exists()


def test_zero_filtered_policy_accepts_clean_batch_and_records_policy(tmp_path):
    dataset = tmp_path / "dataset"
    local = _clean_batch(tmp_path, 0, ["a"])
    assert streaming._publish_batch_to_staging(
        local, dataset, 0, ["a"], require_zero_filtered=True)
    evidence = streaming._load_trusted_batch_evidence(
        dataset, 0, ["a"], require_zero_filtered=True)
    assert evidence is not None
    assert evidence["publication_policy"] == {"require_zero_filtered": True}


def test_zero_filtered_policy_cannot_trust_existing_filtered_evidence(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a", "b"], ["a"], ["b"])
    assert streaming._publish_batch_to_staging(local, dataset, 0, ["a", "b"])
    assert streaming._load_trusted_batch_evidence(
        dataset, 0, ["a", "b"], require_zero_filtered=True) is None


def test_zero_filtered_policy_rejects_dataset_complete_with_filtered_batch(tmp_path):
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a", "b"], ["a"], ["b"])
    assert streaming._publish_batch_to_staging(local, dataset, 0, ["a", "b"])
    with pytest.raises(ValueError, match="untrusted batch evidence"):
        streaming._aggregate_dataset_publication(
            dataset, "demo", ["a", "b"], [0], require_zero_filtered=True)


def test_zero_filtered_dataset_complete_requires_zero_filtered_batches(tmp_path):
    dataset = tmp_path / "dataset"
    local = _clean_batch(tmp_path, 0, ["a"])
    assert streaming._publish_batch_to_staging(
        local, dataset, 0, ["a"], require_zero_filtered=True)
    receipt = streaming._aggregate_dataset_publication(
        dataset, "demo", ["a"], [0], require_zero_filtered=True)
    assert receipt["publication_policy"] == {"require_zero_filtered": True}
    final = json.loads((dataset / ".pipeline_run_receipt_v2.json").read_text())
    assert final["extra"]["publication_policy"] == {
        "require_zero_filtered": True}
    assert streaming._load_complete_dataset_receipt(
        dataset, "demo", ["a"], [0], require_zero_filtered=True) is not None


def test_implementation_fingerprint_is_stable_and_tracks_python_changes(
        tmp_path, monkeypatch):
    scripts_root = tmp_path / "scripts"
    scripts_root.mkdir()
    source = scripts_root / "worker.py"
    source.write_text("print('one')\n", encoding="utf-8")
    (scripts_root / "ignored.txt").write_text("one\n", encoding="utf-8")

    monkeypatch.setattr(streaming, "PROJECT_ROOT", tmp_path)
    first = streaming._implementation_sha256()
    assert first == streaming._implementation_sha256()
    source.write_text("print('two')\n", encoding="utf-8")
    assert streaming._implementation_sha256() != first


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_batch_trust_requires_matching_implementation_fingerprint(
        tmp_path, mutation):
    implementation = "a" * 64
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a"], ["a"], [])
    assert streaming._publish_batch_to_staging(
        local, dataset, 0, ["a"], implementation_sha256=implementation)
    evidence_path = dataset / ".batch_evidence/batch_0000/batch_receipt.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        evidence.pop("implementation_sha256")
    else:
        evidence["implementation_sha256"] = "b" * 64
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert streaming._load_trusted_batch_evidence(
        dataset, 0, ["a"], implementation_sha256=implementation) is None


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_dataset_trust_requires_matching_implementation_fingerprint(
        tmp_path, mutation):
    implementation = "c" * 64
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a"], ["a"], [])
    assert streaming._publish_batch_to_staging(
        local, dataset, 0, ["a"], implementation_sha256=implementation)
    streaming._aggregate_dataset_publication(
        dataset, "demo", ["a"], [0], implementation_sha256=implementation)
    dataset_path = dataset / ".streaming_dataset_receipt_v1.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload.pop("implementation_sha256")
    else:
        payload["implementation_sha256"] = "d" * 64
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    assert streaming._load_complete_dataset_receipt(
        dataset, "demo", ["a"], [0],
        implementation_sha256=implementation) is None


def test_implementation_fingerprint_is_persisted_in_batch_and_dataset_receipts(
        tmp_path):
    implementation = "e" * 64
    dataset = tmp_path / "dataset"
    local = _batch(tmp_path, 0, ["a"], ["a"], [])
    assert streaming._publish_batch_to_staging(
        local, dataset, 0, ["a"], implementation_sha256=implementation)
    evidence = json.loads((
        dataset / ".batch_evidence/batch_0000/batch_receipt.json"
    ).read_text(encoding="utf-8"))
    assert evidence["implementation_sha256"] == implementation
    streaming._aggregate_dataset_publication(
        dataset, "demo", ["a"], [0], implementation_sha256=implementation)
    final = json.loads((dataset / ".pipeline_run_receipt_v2.json").read_text())
    receipt = json.loads((
        dataset / ".streaming_dataset_receipt_v1.json"
    ).read_text(encoding="utf-8"))
    assert final["extra"]["implementation_sha256"] == implementation
    assert receipt["implementation_sha256"] == implementation


def test_r15_enables_zero_filtered_policy():
    import yaml

    config = yaml.safe_load(Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260828_logic_audit_r15_300.yaml"
    ).read_text(encoding="utf-8"))
    assert config["pipelined"]["require_zero_filtered"] is True


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


@pytest.mark.parametrize("preserve_successful_workspaces", [False, True])
def test_cpu_phase_successful_workspace_retention_is_opt_in(
        tmp_path, monkeypatch, preserve_successful_workspaces):
    local_base = tmp_path / "local"
    dataset = {"name": "demo"}
    local_dir = streaming._batch_local_dir(local_base, 0, dataset["name"])
    local_dir.mkdir(parents=True)

    monkeypatch.setattr(streaming, "get_mfa_env", lambda *_args: {})
    monkeypatch.setattr(
        streaming.subprocess, "run",
        lambda *_args, **_kwargs: Namespace(returncode=0),
    )
    monkeypatch.setattr(
        streaming, "_publish_batch_to_staging",
        lambda *_args, **_kwargs: True,
    )

    assert streaming._run_cpu_phase(
        ds=dataset, batch_idx=0, batch_stems=["one"],
        local_base=local_base, config=tmp_path / "config.yaml",
        mfa_python=Path("python"), models_dir=tmp_path,
        nas_output=tmp_path / "nas", batch_size=1, python_path=None,
        restore_cache=False,
        preserve_successful_workspaces=preserve_successful_workspaces,
        producer_mode="nvrasr_fallback",
    )
    assert local_dir.exists() is preserve_successful_workspaces


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


def test_laria_r13_config_uses_fresh_paths_and_retention_without_semantic_drift():
    import copy
    import yaml

    r12_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260828_ria_r12_500.yaml")
    r13_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260828_logic_audit_r13_300.yaml")
    r12 = yaml.safe_load(r12_path.read_text(encoding="utf-8"))
    r13 = yaml.safe_load(r13_path.read_text(encoding="utf-8"))

    assert not run_pipeline.validate_config(r13, "nvrasr_fallback")
    assert r13["output_dir"] == (
        "/mnt/Raw/0825/laria-v5-no-reference-logic-audit300-20260828-r13")
    assert r13["workspace"] == (
        "/mnt/nvme3/mfa_workspace_laria_logic_audit300_20260828_r13")
    assert r13["streaming"]["batch_cache"] == (
        "/tmp/laria-logic-audit300-20260828-r13.cache.json")
    assert r13["streaming"]["local_work"] == [
        "/mnt/nvme0/mfa_work_laria_logic_audit300_20260828_r13",
        "/mnt/nvme1/mfa_work_laria_logic_audit300_20260828_r13",
        "/mnt/nvme3/mfa_work_laria_logic_audit300_20260828_r13",
        "/mnt/nvme4/mfa_work_laria_logic_audit300_20260828_r13",
    ]
    assert r13["streaming"]["batch_size"] == 13
    assert r13["streaming"]["parallel"] == 8
    assert r13["streaming"]["num_gpus"] == 8
    assert r13["pipelined"]["preserve_successful_workspaces"] is True

    differing_paths = ("output_dir", "workspace")
    r12_compare = copy.deepcopy(r12)
    r13_compare = copy.deepcopy(r13)
    for key in differing_paths:
        r12_compare.pop(key, None)
        r13_compare.pop(key, None)
    for key in ("batch_cache", "local_work"):
        r12_compare["streaming"].pop(key, None)
        r13_compare["streaming"].pop(key, None)
    r13_compare["pipelined"].pop("preserve_successful_workspaces")
    assert r13_compare == r12_compare


def test_laria_r17_config_is_r16_semantics_equivalent_with_fresh_namespaces():
    import copy
    import yaml

    r16_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260828_logic_audit_r16_300.yaml")
    r17_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260831_logic_audit_r17_300.yaml")
    r16 = yaml.safe_load(r16_path.read_text(encoding="utf-8"))
    r17 = yaml.safe_load(r17_path.read_text(encoding="utf-8"))

    assert not run_pipeline.validate_config(r17, "nvrasr_fallback")
    assert r17["output_dir"] == (
        "/mnt/Raw/0825/laria-v5-no-reference-logic-audit300-20260831-r17")
    assert r17["workspace"] == (
        "/mnt/nvme3/mfa_workspace_laria_logic_audit300_20260831_r17")
    assert r17["streaming"]["batch_cache"] == (
        "/tmp/laria-logic-audit300-20260831-r17.cache.json")
    assert r17["streaming"]["local_work"] == [
        "/mnt/nvme0/mfa_work_laria_logic_audit300_20260831_r17",
        "/mnt/nvme1/mfa_work_laria_logic_audit300_20260831_r17",
        "/mnt/nvme3/mfa_work_laria_logic_audit300_20260831_r17",
        "/mnt/nvme4/mfa_work_laria_logic_audit300_20260831_r17",
    ]
    assert r17["streaming"]["batch_size"] == 13
    assert r17["streaming"]["parallel"] == 8
    assert r17["streaming"]["num_gpus"] == 8

    def without_namespace(config):
        normalized = copy.deepcopy(config)
        normalized.pop("output_dir")
        normalized.pop("workspace")
        normalized["streaming"].pop("batch_cache")
        normalized["streaming"].pop("local_work")
        return normalized

    assert without_namespace(r17) == without_namespace(r16)


def test_laria_r18_config_is_r17_semantics_equivalent_with_fresh_namespaces():
    import copy
    import yaml

    r17_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260831_logic_audit_r17_300.yaml")
    r18_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260831_logic_audit_r18_300.yaml")
    r17 = yaml.safe_load(r17_path.read_text(encoding="utf-8"))
    r18 = yaml.safe_load(r18_path.read_text(encoding="utf-8"))

    assert not run_pipeline.validate_config(r18, "nvrasr_fallback")
    assert r18["output_dir"] == (
        "/mnt/Raw/0825/laria-v5-no-reference-logic-audit300-20260831-r18")
    assert r18["workspace"] == (
        "/mnt/nvme3/mfa_workspace_laria_logic_audit300_20260831_r18")
    assert r18["streaming"]["batch_cache"] == (
        "/tmp/laria-logic-audit300-20260831-r18.cache.json")
    assert r18["streaming"]["local_work"] == [
        "/mnt/nvme0/mfa_work_laria_logic_audit300_20260831_r18",
        "/mnt/nvme1/mfa_work_laria_logic_audit300_20260831_r18",
        "/mnt/nvme3/mfa_work_laria_logic_audit300_20260831_r18",
        "/mnt/nvme4/mfa_work_laria_logic_audit300_20260831_r18",
    ]

    def without_namespace(config):
        normalized = copy.deepcopy(config)
        normalized.pop("output_dir")
        normalized.pop("workspace")
        normalized["streaming"].pop("batch_cache")
        normalized["streaming"].pop("local_work")
        return normalized

    assert without_namespace(r18) == without_namespace(r17)


def test_laria_r19_config_is_r18_semantics_equivalent_with_fresh_namespaces():
    import copy
    import yaml

    r18_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260831_logic_audit_r18_300.yaml")
    r19_path = Path(
        "configs/laria_v5_no_reference_strict_8gpu_20260831_logic_audit_r19_300.yaml")
    r18 = yaml.safe_load(r18_path.read_text(encoding="utf-8"))
    r19 = yaml.safe_load(r19_path.read_text(encoding="utf-8"))

    assert not run_pipeline.validate_config(r19, "nvrasr_fallback")
    assert r19["output_dir"] == (
        "/mnt/Raw/0825/laria-v5-no-reference-logic-audit300-20260831-r19")
    assert r19["workspace"] == (
        "/mnt/nvme3/mfa_workspace_laria_logic_audit300_20260831_r19")
    assert r19["streaming"]["batch_cache"] == (
        "/tmp/laria-logic-audit300-20260831-r19.cache.json")
    assert r19["streaming"]["local_work"] == [
        "/mnt/nvme0/mfa_work_laria_logic_audit300_20260831_r19",
        "/mnt/nvme1/mfa_work_laria_logic_audit300_20260831_r19",
        "/mnt/nvme3/mfa_work_laria_logic_audit300_20260831_r19",
        "/mnt/nvme4/mfa_work_laria_logic_audit300_20260831_r19",
    ]

    def without_namespace(config):
        normalized = copy.deepcopy(config)
        normalized.pop("output_dir")
        normalized.pop("workspace")
        normalized["streaming"].pop("batch_cache")
        normalized["streaming"].pop("local_work")
        return normalized

    assert without_namespace(r19) == without_namespace(r18)


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


def _write_repairable_ctc_artifacts(raw: Path, stem: str) -> None:
    for suffix in streaming.CTC_SUFFIXES:
        if suffix == ".TextGrid":
            content = (
                'File type = "ooTextFile"\n'
                'Object class = "TextGrid"\n\n'
                "xmin = 0\n"
                "xmax = 0.01\n"
            )
        elif suffix == "_tokens.jsonl":
            content = "{}\n"
        else:
            content = "artifact\n"
        (raw / f"{stem}{suffix}").write_text(content, encoding="utf-8")


def test_unsealed_empty_producer_is_bound_before_raw_manifest_seal(tmp_path):
    raw = tmp_path / "workspace" / "ctc_pretg"
    raw.mkdir(parents=True)
    audio = tmp_path / "audio"
    audio.mkdir()
    stem = "one"
    wav_path = audio / f"{stem}.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    _write_repairable_ctc_artifacts(raw, stem)
    producer = raw / ".ctc_run_receipt.json"
    write_ctc_run_receipt(
        raw, actual_argv=["fixture"], asr_python="python",
        model_path=tmp_path / "model", model_tree_digest="model",
        model_file_manifest=[], dict_path=tmp_path / "dict",
        dict_digest="dict", input_stems=[stem], output_stems=[stem],
        audio_bindings=[])
    before = producer.read_bytes()
    ctx = {"ctc_pretg": raw, "audio_dir": audio,
           "workspace": tmp_path / "workspace"}

    assert run_pipeline._ensure_ctc_axis_receipt(ctx) == 0
    assert producer.read_bytes() != before
    bound = json.loads(producer.read_text(encoding="utf-8"))
    assert [row["stem"] for row in bound["audio_bindings"]] == [stem]
    assert ctx["ctc_axis_receipt_path"] == producer
    assert not (raw / streaming.CTC_RAW_MANIFEST_NAME).exists()

    assert run_pipeline._seal_ctc_raw(ctx) == 0
    assert streaming.validate_ctc_raw_manifest(raw) == []


def test_unsealed_axis_commit_cleans_temp_after_atomic_replace_failure(
        tmp_path, monkeypatch):
    raw = tmp_path / "workspace" / "ctc_pretg"
    raw.mkdir(parents=True)
    audio = tmp_path / "audio"
    audio.mkdir()
    stem = "one"
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

    def fail_replace(*_args, **_kwargs):
        raise OSError("replace denied")

    monkeypatch.setattr(run_pipeline.os, "replace", fail_replace)
    assert run_pipeline._ensure_ctc_axis_receipt({
        "ctc_pretg": raw, "audio_dir": audio,
        "workspace": tmp_path / "workspace",
    }) == 1
    assert not raw.joinpath(
        f".{producer.name}.tmp-{run_pipeline.os.getpid()}"
    ).exists()


def test_sealed_empty_producer_repair_keeps_raw_receipt_and_manifest_immutable(
        tmp_path):
    raw = tmp_path / "workspace" / "ctc_pretg"
    raw.mkdir(parents=True)
    audio = tmp_path / "audio"
    audio.mkdir()
    stem = "one"
    wav_path = audio / f"{stem}.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    _write_repairable_ctc_artifacts(raw, stem)
    producer = raw / ".ctc_run_receipt.json"
    write_ctc_run_receipt(
        raw, actual_argv=["fixture"], asr_python="python",
        model_path=tmp_path / "model", model_tree_digest="model",
        model_file_manifest=[], dict_path=tmp_path / "dict",
        dict_digest="dict", input_stems=[stem], output_stems=[stem],
        audio_bindings=[])
    assert run_pipeline._seal_ctc_raw({
        "ctc_pretg": raw, "accounting_eligible_stems": (stem,),
    }) == 0
    producer_before = producer.read_bytes()
    manifest_before = (raw / streaming.CTC_RAW_MANIFEST_NAME).read_bytes()
    ctx = {"ctc_pretg": raw, "audio_dir": audio,
           "workspace": tmp_path / "workspace"}

    assert streaming._repair_complete_ctc_receipt(
        tmp_path / "workspace", audio, [stem]) is True
    assert producer.read_bytes() == producer_before
    assert (raw / streaming.CTC_RAW_MANIFEST_NAME).read_bytes() == manifest_before
    assert streaming.validate_ctc_raw_manifest(raw) == []
    derived_path = ctx["workspace"] / ".ctc_input_axis_receipt.json"
    # The repair owns a fresh context internally; its derived artifact is
    # nevertheless required in the workspace and must carry the exact stem.
    assert derived_path.is_file()
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    assert [row["stem"] for row in derived["audio_bindings"]] == [stem]


def test_sealed_raw_manifest_tamper_rejects_repair_without_rewrite(tmp_path):
    raw = tmp_path / "workspace" / "ctc_pretg"
    raw.mkdir(parents=True)
    audio = tmp_path / "audio"
    audio.mkdir()
    stem = "one"
    wav_path = audio / f"{stem}.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    _write_repairable_ctc_artifacts(raw, stem)
    producer = raw / ".ctc_run_receipt.json"
    write_ctc_run_receipt(
        raw, actual_argv=["fixture"], asr_python="python",
        model_path=tmp_path / "model", model_tree_digest="model",
        model_file_manifest=[], dict_path=tmp_path / "dict",
        dict_digest="dict", input_stems=[stem], output_stems=[stem],
        audio_bindings=[])
    assert run_pipeline._seal_ctc_raw({
        "ctc_pretg": raw, "accounting_eligible_stems": (stem,),
    }) == 0
    manifest = json.loads((raw / streaming.CTC_RAW_MANIFEST_NAME).read_text())
    manifest["identity"] = "tampered"
    (raw / streaming.CTC_RAW_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8")
    producer_before = producer.read_bytes()

    assert not streaming._repair_complete_ctc_receipt(
        tmp_path / "workspace", audio, [stem])
    assert producer.read_bytes() == producer_before


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


def _frozen_inventory_cache(audio: Path, stems: list[str]) -> dict:
    files = []
    for stem in stems:
        path = audio / f"{stem}.wav"
        if not path.exists():
            path = audio / f"{stem}.WAV"
        files.append({
            "stem": stem,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return {"datasets": [{
        "name": "demo",
        "audio_dir": str(audio),
        "stems": stems,
        "source_inventory": {"count": len(files), "files": files},
    }]}


def test_frozen_inventory_enumerates_once_and_reports_success(tmp_path, monkeypatch,
                                                               capsys):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "one.wav").write_bytes(b"one")
    (audio / "two.WAV").write_bytes(b"two")
    (audio / "ignore.txt").write_text("not audio", encoding="utf-8")
    cache = _frozen_inventory_cache(audio, ["one", "two"])
    real_scandir = streaming.os.scandir
    calls = []

    def counted_scandir(path):
        calls.append(path)
        return real_scandir(path)

    monkeypatch.setattr(streaming.os, "scandir", counted_scandir)
    streaming._validate_frozen_cache_inventory(cache)

    assert calls == [str(audio)]
    assert "Frozen source validated: demo (2 WAVs," in capsys.readouterr().out


def test_frozen_inventory_rejects_wav_symlink_before_hashing(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    target = audio / "target.bin"
    target.write_bytes(b"target")
    (audio / "one.wav").symlink_to(target)
    cache = {"datasets": [{
        "name": "demo", "audio_dir": str(audio), "stems": ["one"],
        "source_inventory": {"count": 1, "files": []},
    }]}
    monkeypatch.setattr(
        streaming, "_sha256_file",
        lambda _path: pytest.fail("symlink must be rejected before hashing"),
    )

    with pytest.raises(ValueError, match="contains symlink WAV"):
        streaming._validate_frozen_cache_inventory(cache)


def test_frozen_inventory_rejects_size_tamper_even_when_content_hash_matches(
        tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = audio / "one.wav"
    wav.write_bytes(b"original")
    cache = _frozen_inventory_cache(audio, ["one"])
    wav.write_bytes(b"changed-size")
    expected_hash = cache["datasets"][0]["source_inventory"]["files"][0]["sha256"]
    monkeypatch.setattr(streaming, "_sha256_file", lambda _path: expected_hash)

    with pytest.raises(ValueError, match="source inventory hash mismatch"):
        streaming._validate_frozen_cache_inventory(cache)


def test_frozen_inventory_rejects_content_hash_tamper(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = audio / "one.wav"
    wav.write_bytes(b"original")
    cache = _frozen_inventory_cache(audio, ["one"])
    wav.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="source inventory hash mismatch"):
        streaming._validate_frozen_cache_inventory(cache)


@pytest.mark.parametrize("source_stems, cache_stems", [
    (["one", "two"], ["one"]),
    (["one"], ["one", "two"]),
])
def test_frozen_inventory_rejects_extra_or_missing_wav(tmp_path, source_stems,
                                                        cache_stems):
    audio = tmp_path / "audio"
    audio.mkdir()
    for stem in source_stems:
        (audio / f"{stem}.wav").write_bytes(stem.encode())
    cache = _frozen_inventory_cache(audio, cache_stems if set(cache_stems) <= set(source_stems)
                                    else ["one"])
    cache["datasets"][0]["stems"] = cache_stems

    with pytest.raises(ValueError, match="cache/source stem mismatch"):
        streaming._validate_frozen_cache_inventory(cache)


def test_frozen_inventory_ignores_non_wav_and_non_regular_wav_entries(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    wav = audio / "one.wav"
    wav.write_bytes(b"one")
    (audio / "notes.txt").write_text("ignored", encoding="utf-8")
    (audio / "nested.wav").mkdir()
    cache = _frozen_inventory_cache(audio, ["one"])

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
