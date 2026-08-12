"""Stdlib regression harness for exact-206 filtered replay package preparation."""
from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.filtered_recovery_package import PackageError, prepare_package, preflight_package
from scripts.postprocess_textgrids import (
    STRICT_EN_MFA_SCHEMA,
    Interval,
    Tier,
    load_strict_en_provenance,
)
from scripts.run_pipeline import (
    _read_filtered_recovery_evidence,
    _validate_filtered_recovery_english_ledger_scope,
    validate_filtered_recovery_manifest,
    validate_config,
)


def _fixture(base: Path) -> Path:
    root = base / "source"
    output = root / "workspace" / "strict_ok_runs" / "run" / "output"
    filtered = output.parent / "filtered"
    output.mkdir(parents=True); filtered.mkdir()
    stems = [f"s{i:04d}" for i in range(1000)]
    frozen = set(stems[:206]); strict_only = stems[205]
    report = []
    for stem in stems:
        status = "ok" if stem not in frozen or stem == strict_only else "filtered_short_word"
        report.append({"stem": stem, "status": status, "filter_reasons": ["short_word"] if status.startswith("filtered") else []})
    (output / "postprocess_report.jsonl").write_text("".join(json.dumps(row) + "\n" for row in report), encoding="utf-8")
    (output / "strict_ok_manifest.json").write_text(json.dumps({
        "ok": [{"stem": stem} for stem in stems[206:]],
        "rejected": {stem: ["strict_rejected"] for stem in frozen},
        "output_dir": str(output),
    }), encoding="utf-8")
    (output / ".pipeline_run_receipt_v2.json").write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}), encoding="utf-8")
    for stem in frozen:
        (filtered / f"{stem}.TextGrid").write_text("grid", encoding="utf-8")
    (root / "selected_manifest.json").write_text(json.dumps({"samples": [{"stem": stem} for stem in stems]}), encoding="utf-8")
    workspace = root / "workspace"
    (workspace / ".mfa_alignment_axis_receipt_recovered.json").write_text("{}", encoding="utf-8")
    (workspace / "en_phones").mkdir()
    (workspace / "en_phones" / "en_alignment_manifest.json").write_text(json.dumps({"schema": "strict-en-mfa-v1"}), encoding="utf-8")
    (root / "resolved_gpu1000_nvrasr_fallback.yaml").write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    return root


def test_prepare_and_read_only_preflight_exact206(tmp_path: Path):
    source = _fixture(tmp_path); package = tmp_path / "package"
    prepare_package(source, package)
    result = preflight_package(package)
    assert result["ok"] and result["selected"] == 1000 and result["accepted"] == 794 and result["frozen"] == 206
    assert all(path.stat().st_mode & 0o222 == 0 for path in package.iterdir())
    frozen_payload = json.loads((package / "frozen_filtered.json").read_text())
    accepted_payload = json.loads((package / "accepted_manifest.json").read_text())
    plan = json.loads((package / "filtered_recovery_manifest.json").read_text())
    evidence = _read_filtered_recovery_evidence(package / "evidence_receipt.json",
                                                frozen_payload["stems"], plan)
    accepted = [row["stem"] for row in accepted_payload["ok"]]
    validated = validate_filtered_recovery_manifest(
        plan, frozen_payload["stems"], accepted,
        expected_mismatch=evidence["declared_vs_actual_inner_receipt"])
    assert validated["count"] == 206
    ledger = [json.loads(line) for line in (package / "filtered_reason_ledger.jsonl").read_text().splitlines()]
    strict_only = next(row for row in ledger if row["stem"] == "s0205")
    assert strict_only["strict_reasons"] == ["strict_rejected"]
    assert "strict_rejected" in strict_only["reasons"]


def test_negative_source_drift_and_partition_mismatch_fail_closed(tmp_path: Path):
    source = _fixture(tmp_path); package = tmp_path / "package"; prepare_package(source, package)
    report = next((source / "workspace" / "strict_ok_runs").glob("*/output/postprocess_report.jsonl"))
    report.write_text(report.read_text() + "\n", encoding="utf-8")
    try:
        preflight_package(package)
    except PackageError:
        pass
    else:
        raise AssertionError("source drift was not rejected")


def test_negative_filtered_file_and_strict_only_mismatch(tmp_path: Path):
    source = _fixture(tmp_path)
    filtered = next((source / "workspace" / "strict_ok_runs").glob("*/filtered"))
    next(filtered.glob("*.TextGrid")).unlink()
    try:
        prepare_package(source, tmp_path / "package")
    except PackageError:
        pass
    else:
        raise AssertionError("filtered-file mismatch was not rejected")
    source = _fixture(tmp_path / "strict_only")
    report = next((source / "workspace" / "strict_ok_runs").glob("*/output/postprocess_report.jsonl"))
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    rows[205]["status"] = "filtered_short_word"
    report.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    try:
        prepare_package(source, tmp_path / "strict_only_package")
    except PackageError:
        pass
    else:
        raise AssertionError("strict-only mismatch was not rejected")


def test_negative_existing_and_symlink_package_targets(tmp_path: Path):
    source = _fixture(tmp_path)
    existing = tmp_path / "existing"; existing.mkdir()
    try:
        prepare_package(source, existing)
    except PackageError:
        pass
    else:
        raise AssertionError("existing package target was accepted")
    link = tmp_path / "link"; link.symlink_to(tmp_path / "missing")
    try:
        prepare_package(source, link)
    except PackageError:
        pass
    else:
        raise AssertionError("symlink package target was accepted")


def test_negative_unexpected_package_entry(tmp_path: Path):
    source = _fixture(tmp_path); package = tmp_path / "package"; prepare_package(source, package)
    (package / "unexpected.json").write_text("{}", encoding="utf-8")
    try:
        preflight_package(package)
    except PackageError:
        pass
    else:
        raise AssertionError("unexpected package entry was accepted")


def test_filtered_recovery_mode_all_gpus_validation_is_narrow():
    base = {"ctc_prealign": {"enabled": True, "all_gpus": True}}
    assert validate_config(base, "filtered_recovery") == []
    assert any("enabled=false" in error for error in validate_config(
        {"ctc_prealign": {"enabled": False, "all_gpus": True}}, "filtered_recovery"))
    for mode in ("full", "ctc_ready", "nvrasr_fallback", "strict_replay"):
        assert not any("ctc_prealign.all_gpus=true" in error for error in validate_config(base, mode))
    assert any("requires full or nvrasr_fallback" in error for error in validate_config(base, "batch_ctc_ready"))


def test_filtered_recovery_english_ledger_universe_is_exact_and_unique():
    frozen = ["frozen-a", "frozen-b"]
    accepted = ["accepted-a", "accepted-b"]
    frozen_rows = [{"stem": stem} for stem in frozen]
    full_rows = [{"stem": stem} for stem in frozen + accepted]
    assert _validate_filtered_recovery_english_ledger_scope(frozen_rows, frozen, accepted) == set(frozen)
    assert _validate_filtered_recovery_english_ledger_scope(full_rows, frozen, accepted) == set(frozen + accepted)
    for invalid_rows in (
        frozen_rows[:1],
        full_rows + [{"stem": "extra"}],
        frozen_rows + [{"stem": "frozen-a"}],
    ):
        try:
            _validate_filtered_recovery_english_ledger_scope(invalid_rows, frozen, accepted)
        except ValueError:
            pass
        else:
            raise AssertionError("partial, expanded, or duplicate English ledger universe was accepted")


def test_grouped_english_keeps_source_segment_phone_ordinals(tmp_path: Path):
    """Grouping split spellings must not rewrite immutable MFA ordinals."""
    stem = "fixture"
    evidence = tmp_path / "evidence.TextGrid"
    evidence.write_text("source", encoding="utf-8")
    evidence_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()

    def word(segment: int, ordinal: int, text: str, phone_ordinal: int) -> dict:
        start = phone_ordinal / 10.0
        return {
            "word_id": f"{stem}:s{segment}:w{ordinal}",
            "ctc_text": text,
            "ctc_ordinal": ordinal,
            "status": "verified",
            "provenance": "english_mfa_textgrid",
            "mfa_word": {"ordinal": ordinal, "text": text,
                         "start": start, "end": start + 0.05},
            "phones": [{"ordinal": 0, "mfa_phone_ordinal": phone_ordinal,
                        "label": "AH1", "start": start, "end": start + 0.05}],
        }

    segments = [
        {"segment_id": f"{stem}:s0", "segment_ordinal": 0, "status": "verified",
         "mfa_textgrid": {"path": str(evidence), "sha256": evidence_sha},
         "words": [word(0, 0, "X", 0), word(0, 1, "A", 2), word(0, 2, "B", 3)]},
        {"segment_id": f"{stem}:s1", "segment_ordinal": 1, "status": "verified",
         "mfa_textgrid": {"path": str(evidence), "sha256": evidence_sha},
         "words": [word(1, 3, "C", 1), word(1, 4, "D", 2)]},
    ]
    ledger = tmp_path / f"{stem}_en_phones.json"
    ledger.write_text(json.dumps({"schema": STRICT_EN_MFA_SCHEMA, "stem": stem,
                                  "segments": segments}), encoding="utf-8")
    ledger_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    manifest = {
        "schema": STRICT_EN_MFA_SCHEMA,
        "strict_provenance": True,
        "status": "success",
        "expected_segments": [f"{stem}:s0", f"{stem}:s1"],
        "produced_segments": [f"{stem}:s0", f"{stem}:s1"],
        "rejected_segments": [],
        "stem_ledgers": [{"stem": stem, "path": str(ledger), "sha256": ledger_sha}],
    }
    (tmp_path / "en_alignment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.1, "X"), Interval(0.1, 0.4, "AB"),
        Interval(0.4, 0.7, "CD")])
    report, pairs = load_strict_en_provenance(stem, words, tmp_path)
    assert report["status"] == "verified", report
    assert [[phone["mfa_phone_ordinal"] for phone in record["phones"]]
            for _, record in pairs] == [[0], [2, 3], [1, 2]]


def main() -> int:
    tests = [test_prepare_and_read_only_preflight_exact206,
             test_negative_source_drift_and_partition_mismatch_fail_closed,
             test_negative_filtered_file_and_strict_only_mismatch,
             test_negative_existing_and_symlink_package_targets,
             test_negative_unexpected_package_entry,
             test_filtered_recovery_mode_all_gpus_validation_is_narrow,
             test_filtered_recovery_english_ledger_universe_is_exact_and_unique,
             test_grouped_english_keeps_source_segment_phone_ordinals]
    for test in tests:
        with TemporaryDirectory() as directory:
            test(Path(directory)) if test.__code__.co_argcount else test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
