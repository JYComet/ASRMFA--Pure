"""Regression tests for the commit-local CTC work inventory."""

from __future__ import annotations

import hashlib
import json
import pipeline_utils as runtime_pipeline_utils
import os
from pathlib import Path

import pytest

from scripts import pipeline_utils
from scripts import run_pipeline
from scripts.pipeline_utils import (
    CTC_SUFFIXES,
    CTC_RAW_MANIFEST_NAME,
    CTC_WORK_RECEIPT_NAME,
    CtcWorkCommitSnapshot,
    materialize_ctc_work,
    write_ctc_raw_manifest,
    write_ctc_work_receipt,
)


def _bundle(tmp_path: Path, stem_count: int) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    work = tmp_path / "work"
    raw.mkdir(parents=True)
    stems = [f"clip-{index:03d}" for index in range(stem_count)]
    for stem in stems:
        for suffix in CTC_SUFFIXES:
            (raw / f"{stem}{suffix}").write_text(
                f"{stem}:{suffix}\n", encoding="utf-8")
        (raw / f"{stem}_ref.txt").write_text(
            f"reference {stem}\n", encoding="utf-8")
    producer = raw / ".ctc_run_receipt.json"
    producer.write_text('{"schema":"ctc-run-receipt-v2"}\n', encoding="utf-8")
    write_ctc_raw_manifest(raw, producer_receipt=producer, stems=stems)
    manifest = raw / CTC_RAW_MANIFEST_NAME
    materialize_ctc_work(raw, work, raw_manifest_path=manifest)
    return raw, work, manifest


def _ctx(raw: Path, work: Path, stems: list[str]) -> dict:
    return {
        "ctc_pretg": raw,
        "ctc_pretg_adj": work,
        "ctc_raw_manifest": raw / CTC_RAW_MANIFEST_NAME,
        "expected_stems": stems,
        "config": {"ctc_prealign": {"provider": "nvasr"}},
        "data_dir": raw,
        "reference_mode": "auto",
    }


@pytest.mark.parametrize("stem_count", [1, 10, 100])
def test_record_ctc_work_stage_hashes_each_work_artifact_once(
        tmp_path, monkeypatch, stem_count):
    raw, work, manifest = _bundle(tmp_path, stem_count)
    stems = [f"clip-{index:03d}" for index in range(stem_count)]
    ctx = _ctx(raw, work, stems)
    calls: list[Path] = []
    original = pipeline_utils._sha256_file

    def counted(path):
        path = Path(path)
        if path.parent == work:
            calls.append(path)
        return original(path)

    monkeypatch.setattr(pipeline_utils, "_sha256_file", counted)
    monkeypatch.setattr(run_pipeline, "_sha256_file", counted)
    monkeypatch.setattr(runtime_pipeline_utils, "_sha256_file", counted)

    assert run_pipeline._record_ctc_work_stage(ctx, "normalize", status="skipped") == 0
    expected = [path for path in sorted(work.iterdir())
                if path.name != CTC_WORK_RECEIPT_NAME]
    assert sorted(calls) == expected
    assert len(calls) == len(expected)


def test_snapshot_preserves_legacy_receipt_ledger_and_freshness(
        tmp_path, monkeypatch):
    raw, work, manifest = _bundle(tmp_path, 3)
    stems = [f"clip-{index:03d}" for index in range(3)]
    ctx = _ctx(raw, work, stems)
    prior = json.loads((work / CTC_WORK_RECEIPT_NAME).read_text())
    ledger = list(prior["transform_ledger"]) + [{
        "stage": "normalize", "status": "skipped",
        "work_files_digest": "legacy-oracle-placeholder",
    }]
    legacy_rows = [{"name": p.name, "sha256": hashlib.sha256(
        p.read_bytes()).hexdigest()} for p in sorted(work.iterdir())
                   if p.is_file() and p.name != CTC_WORK_RECEIPT_NAME]
    ledger[-1]["work_files_digest"] = pipeline_utils.stable_json_digest(legacy_rows)

    expected = write_ctc_work_receipt(
        work, manifest, transform_ledger=ledger, work_root=work)
    expected = run_pipeline._write_ctc_work_freshness(ctx, work, expected)
    (work / CTC_WORK_RECEIPT_NAME).unlink()
    materialize_ctc_work(raw, work, raw_manifest_path=manifest,
                         overwrite=True, transform_ledger=prior["transform_ledger"])
    ctx["ctc_work_receipt"] = json.loads(
        (work / CTC_WORK_RECEIPT_NAME).read_text())

    assert run_pipeline._record_ctc_work_stage(ctx, "normalize", status="skipped") == 0
    actual = json.loads((work / CTC_WORK_RECEIPT_NAME).read_text())
    assert actual == expected


def test_snapshot_rejects_mutation_between_capture_and_commit(tmp_path):
    raw, work, manifest = _bundle(tmp_path, 1)
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    artifact = work / "clip-000.lab"
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|mutat|snapshot"):
        write_ctc_work_receipt(work, manifest, snapshot=snapshot)


def test_snapshot_rejects_same_size_rewrite_with_restored_mtime(tmp_path):
    raw, work, manifest = _bundle(tmp_path, 1)
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    artifact = work / "clip-000.lab"
    original_stat = artifact.stat()
    artifact.write_bytes(b"x" * original_stat.st_size)
    os.utime(artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed|snapshot"):
        write_ctc_work_receipt(work, manifest, snapshot=snapshot)


def test_snapshot_rejects_raw_mutation_between_capture_and_commit(tmp_path):
    raw, work, manifest = _bundle(tmp_path, 1)
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    (raw / "clip-000.lab").write_text("changed raw\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|digest|raw"):
        write_ctc_work_receipt(work, manifest, snapshot=snapshot)


def test_freshness_commit_rechecks_snapshot_after_fingerprint_generation(
        tmp_path, monkeypatch):
    raw, work, manifest = _bundle(tmp_path, 1)
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    ctx = _ctx(raw, work, ["clip-000"])
    receipt = json.loads((work / CTC_WORK_RECEIPT_NAME).read_text())
    original = run_pipeline._ctc_work_freshness_fingerprints

    def mutate_after_fingerprints(*args, **kwargs):
        result = original(*args, **kwargs)
        (work / "clip-000.lab").write_text("changed after fingerprint\n",
                                             encoding="utf-8")
        return result

    monkeypatch.setattr(run_pipeline, "_ctc_work_freshness_fingerprints",
                        mutate_after_fingerprints)
    with pytest.raises(ValueError, match="changed|snapshot"):
        run_pipeline._write_ctc_work_freshness(
            ctx, work, receipt, snapshot=snapshot)


def test_snapshot_keeps_ordinary_work_subdirectories_out_of_inventory(tmp_path):
    raw, work, manifest = _bundle(tmp_path, 1)
    (work / "ignored-cache").mkdir()
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    receipt = write_ctc_work_receipt(work, manifest, snapshot=snapshot)
    assert all(row["name"] != "ignored-cache"
               for row in snapshot.freshness_rows)
    assert receipt["files"]


def test_snapshot_rejects_raw_and_work_identity_mismatch(tmp_path):
    raw, work, manifest = _bundle(tmp_path, 1)
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    other_raw, _, other_manifest = _bundle(tmp_path / "other", 1)

    with pytest.raises(ValueError, match="raw|identity|root"):
        write_ctc_work_receipt(work, other_manifest, snapshot=snapshot)
    with pytest.raises(ValueError, match="raw|identity|root"):
        write_ctc_work_receipt(other_raw / "work", manifest, snapshot=snapshot)


@pytest.mark.parametrize("kind", ["missing", "symlink", "extra"])
def test_snapshot_rejects_unsafe_work_inventory(tmp_path, kind):
    raw, work, manifest = _bundle(tmp_path, 1)
    snapshot = CtcWorkCommitSnapshot.capture(work, manifest)
    artifact = work / "clip-000.lab"
    if kind == "missing":
        artifact.unlink()
    elif kind == "symlink":
        artifact.unlink()
        artifact.symlink_to(raw / "clip-000.lab")
    else:
        (work / "new-metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing|symlink|inventory|changed"):
        write_ctc_work_receipt(work, manifest, snapshot=snapshot)
