"""Paired disk-contract fixtures for authority and no-reference modes."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scripts import audit_strict_ok as audit
from scripts import ctc_prealign as ctc
from scripts import postprocess_textgrids as post
from scripts import run_pipeline
from scripts import verify_strict_ok as verifier
from scripts.postprocess_textgrids import Interval, Tier, parse_textgrid, write_textgrid


def _audit_fixture(paths: dict[str, Path], reference_mode: str = "auto"):
    english_manifest = json.loads(paths["en_manifest"].read_text(encoding="utf-8"))
    english_manifest["canonical_units"] = "canonical-english-units-v1"
    paths["en_manifest"].write_text(json.dumps(english_manifest), encoding="utf-8")
    return audit.audit(Namespace(
        output_dir=paths["output"], filtered_dir=paths["filtered"],
        ctc_dir=paths["ctc"], reference_dir=paths["refs"],
        wav_dir=paths["wavs"], aligned_dir=paths["aligned"],
        en_phones_dir=paths["en_phones"], en_aligned_dir=paths["en_aligned"],
        en_manifest=paths["en_manifest"], pipeline_receipt=paths["pipeline_receipt"],
        mfa_input_axis_receipt=paths["axis_input"],
        mfa_alignment_axis_receipt=paths["axis_alignment"],
        mfa_axis_audio_root=paths["wavs"],
        tts_authoritative_audio_root=paths["tts"],
        report=paths["output"] / "postprocess_report.jsonl", isolate=True,
        reference_mode=reference_mode,
    ))


def _fallback(paths: dict[str, Path], source: str) -> None:
    (paths["refs"] / "demo.txt").unlink()
    if source == "asr_fallback":
        source_path = paths["ctc"] / "demo_text_cn.txt"
        source_path.write_text("你好\n", encoding="utf-8")
    else:
        source_path = paths["ctc"] / "demo.lab"
        source_path.write_text("ni3 hao3\n", encoding="utf-8")
    report_path = paths["output"] / "postprocess_report.jsonl"
    row = json.loads(report_path.read_text(encoding="utf-8").strip())
    row.update({
        "reference_mode": "fallback",
        "reference_source": source,
        "reference_text_authoritative": False,
        "fallback_transcript": {
            "source": source, "path": str(source_path.resolve()),
            "sha256": audit._sha256(source_path),
        },
    })
    report_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _publish_and_verify(paths: dict[str, Path], manifest: dict) -> list[str]:
    manifest_path = paths["output"] / "strict_ok_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return verifier.verify(manifest_path, paths["output"])


@pytest.mark.parametrize("schema", [
    audit.CTC_LIFECYCLE_SCHEMA,
    audit.NVASR_SPIKE_ANCHOR_SCHEMA,
    audit.CTC_RAW_TOKEN_ROW_SCHEMA,
    audit.NVASR_CANDIDATE_PROVENANCE_SCHEMA,
    audit.NVASR_PRODUCER_AUTHORITY_SCHEMA,
    audit.NVASR_IMMUTABLE_PROJECTION_SCHEMA,
    "nvasr-candidate-timeline-v1",
    "nvasr-raw-timeline-neighbors-v1",
    "ctc-frame-support-v1",
    "nvasr-owner-selection-v2",
])
def test_audit_rejects_schema_only_v3_sidecar_without_lifecycle_markers(
        tmp_path, schema):
    paths = verifier._write_fixture(tmp_path)
    sidecar = paths["ctc"] / "demo_tokens.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    rows[0] = {"schema": schema}
    sidecar.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    manifest, clean = _audit_fixture(paths)

    assert not clean
    assert not manifest["ok"]
    assert "ctc_lifecycle_missing_for_v3:demo" in manifest["global_reasons"]


@pytest.mark.parametrize("key", sorted(audit._MODERN_REPORT_CONTRACT_KEYS))
def test_audit_rejects_empty_modern_report_contract_without_lifecycle(
        tmp_path, key):
    paths = verifier._write_fixture(tmp_path)
    report = paths["output"] / "postprocess_report.jsonl"
    row = json.loads(report.read_text(encoding="utf-8").strip())
    row[key] = {}
    report.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest, clean = _audit_fixture(paths)

    assert not clean
    assert not manifest["ok"]
    assert "postprocess_v3_claim_without_ctc_lifecycle" in \
        manifest["global_reasons"]


def test_audit_rejects_broken_default_raw_manifest_symlink(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    os.symlink("missing_raw_manifest.json", paths["ctc"] /
               audit.CTC_RAW_MANIFEST_NAME)

    manifest, clean = _audit_fixture(paths)

    assert not clean
    assert not manifest["ok"]
    assert "ctc_raw_manifest_missing_or_symlink" in \
        manifest["global_reasons"]


def test_audit_rejects_report_only_v3_lifecycle_claim(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    report = paths["output"] / "postprocess_report.jsonl"
    row = json.loads(report.read_text(encoding="utf-8").strip())
    row["ctc_lifecycle"] = {
        "schema": audit.CTC_LIFECYCLE_SCHEMA,
        "status": "legacy_single_directory_fixture",
    }
    report.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest, clean = _audit_fixture(paths)

    assert not clean
    assert not manifest["ok"]
    assert "postprocess_v3_claim_without_ctc_lifecycle" in \
        manifest["global_reasons"]


@pytest.mark.parametrize("kind", ["symlink", "malformed"])
def test_audit_does_not_downgrade_unsafe_sidecar_to_legacy(tmp_path, kind):
    paths = verifier._write_fixture(tmp_path)
    sidecar = paths["ctc"] / "demo_tokens.jsonl"
    if kind == "symlink":
        sidecar.unlink()
        os.symlink("missing_tokens.jsonl", sidecar)
    else:
        sidecar.write_text("{not-json}\n", encoding="utf-8")

    manifest, clean = _audit_fixture(paths)

    assert not clean
    assert not manifest["ok"]
    assert "ctc_v3_token_sidecar_unreadable:demo" in \
        manifest["global_reasons"]


def test_asr_fallback_positive_has_exclusive_disk_evidence(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    _fallback(paths, "asr_fallback")
    manifest, clean = _audit_fixture(paths)

    assert clean
    assert not manifest["global_reasons"]
    entry = manifest["ok"][0]
    assert entry["mode"] == "fallback"
    assert entry["fallback_transcript"]["source"] == "asr_fallback"
    assert "reference" not in entry
    assert _publish_and_verify(paths, manifest) == []
    report = json.loads((paths["output"] / "postprocess_report.jsonl").read_text().strip())
    assert report["reference_mode"] == "fallback"
    assert report["fallback_transcript"]["source"] == "asr_fallback"
    assert not any("reference_" in reason or reason == "cjk_token_count_mismatch"
                   for reason in manifest["rejected"].get("demo", []))


def test_lab_fallback_and_ctc_ready_share_the_same_contract(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    _fallback(paths, "lab_fallback")
    manifest, clean = _audit_fixture(paths)

    assert clean
    assert manifest["ok"][0]["fallback_transcript"]["path"].endswith("/demo.lab")
    assert _publish_and_verify(paths, manifest) == []


def test_authority_reference_semantic_drift_remains_rejected(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    (paths["refs"] / "demo.txt").write_text("世界!\n", encoding="utf-8")
    manifest, clean = _audit_fixture(paths)

    assert not manifest["ok"]
    assert "reference_semantic_sequence_mismatch" in manifest["rejected"]["demo"]


def test_common_cjk_ownership_corruption_rejects_authority_and_fallback(tmp_path):
    for fallback in (False, True):
        root = tmp_path / ("fallback" if fallback else "authority")
        root.mkdir()
        paths = verifier._write_fixture(root)
        if fallback:
            _fallback(paths, "asr_fallback")
        grid = parse_textgrid(paths["output"] / "demo.TextGrid")
        words = next(tier for tier in grid.tiers if tier.name == "words")
        words.intervals[1].text = "not_pinyin"
        write_textgrid(grid, paths["output"] / "demo.TextGrid")
        manifest, clean = _audit_fixture(paths)

        assert not manifest["ok"]
        reasons = manifest["rejected"]["demo"]
        if fallback:
            assert "cjk_pinyin_ownership_mismatch" in reasons
            assert "cjk_token_count_mismatch" not in reasons
        else:
            assert {"cjk_pinyin_count_mismatch", "cjk_without_toned_pinyin_word"} <= set(reasons)


def test_fallback_source_report_hash_and_mode_tampering_fail_closed(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    _fallback(paths, "asr_fallback")
    manifest, clean = _audit_fixture(paths)
    assert clean
    assert _publish_and_verify(paths, manifest) == []

    source = paths["ctc"] / "demo_text_cn.txt"
    source.write_text("篡改\n", encoding="utf-8")
    assert "fallback_transcript_hash_mismatch:demo" in verifier.verify(
        paths["output"] / "strict_ok_manifest.json", paths["output"])

    source.write_text("你好\n", encoding="utf-8")
    report = paths["output"] / "postprocess_report.jsonl"
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert "postprocess_report_hash_mismatch" in verifier.verify(
        paths["output"] / "strict_ok_manifest.json", paths["output"])

    manifest["ok"][0]["mode"] = "authority"
    assert "authority_fallback_evidence_conflict:demo" in _publish_and_verify(paths, manifest)


def test_explicit_fallback_ignores_incidental_reference_file(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    source_path = paths["ctc"] / "demo_text_cn.txt"
    source_path.write_text("你好\n", encoding="utf-8")
    report_path = paths["output"] / "postprocess_report.jsonl"
    row = json.loads(report_path.read_text(encoding="utf-8").strip())
    row.update({
        "reference_mode": "fallback",
        "reference_source": "asr_fallback",
        "reference_text_authoritative": False,
        "fallback_transcript": {
            "source": "asr_fallback", "path": str(source_path.resolve()),
            "sha256": audit._sha256(source_path),
        },
    })
    report_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest, clean = _audit_fixture(paths, reference_mode="fallback")
    assert clean
    assert manifest["reference_mode_policy"] == "fallback"
    assert manifest["ok"][0]["mode"] == "fallback"


def test_explicit_authority_rejects_missing_reference_instead_of_fallback(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    (paths["refs"] / "demo.txt").unlink()
    source_path = paths["ctc"] / "demo_text_cn.txt"
    source_path.write_text("你好\n", encoding="utf-8")

    manifest, clean = _audit_fixture(paths, reference_mode="authority")
    assert not manifest["ok"]
    assert "authority_reference_missing" in manifest["rejected"]["demo"]


def test_fallback_report_corrections_do_not_use_authority_count_contract():
    fallback_row = {
        "status": "ok",
        "reference_mode": "fallback",
        "reference_coverage": {"reference_validation_applied": False},
        "fallback_lexical_alignment": {"safe": True},
        "text_corrected": True,
        "pinyin_displacement": {"mismatch_rate": 0.0, "displacement_runs": 0},
        "text_order": {"in_order": True, "ref_cjk_count": 3, "hanzi_cjk_count": 2},
    }
    assert audit._report_reasons(fallback_row) == []

    fallback_row["reference_mode"] = "authority"
    assert "report_positive:text_corrected" in audit._report_reasons(fallback_row)


def test_ctc_fallback_inventory_excludes_incidental_reference_text(tmp_path):
    wav = tmp_path / "demo.wav"
    wav.write_bytes(b"RIFF")
    ref = tmp_path / "demo.txt"
    ref.write_text("你好\n", encoding="utf-8")

    selected, refs, exclusions = ctc._source_inventory(
        [wav], tmp_path, {"demo": ref}, True, "fallback")
    assert [path.stem for path in selected] == ["demo"]
    assert refs == {}
    assert exclusions == {}


def test_reference_mode_resolution_and_schema_contract():
    assert run_pipeline.resolve_reference_mode({"reference_mode": "fallback"}) == "fallback"
    assert run_pipeline.resolve_reference_mode({"ctc_prealign": {"allow_missing_reference": False}}) == "authority"
    assert run_pipeline.resolve_reference_mode({"ctc_prealign": {"allow_missing_reference": True}}) == "fallback"
    assert run_pipeline.validate_config({"reference_mode": "invalid"}, "full")
    assert run_pipeline.validate_config({
        "reference_mode": "authority",
        "ctc_prealign": {"allow_missing_reference": True},
    }, "full")


def test_no_reference_empty_english_manifest_is_v2_compatible():
    manifest = run_pipeline._validate_strict_english_manifest_payload({
        "schema": "strict-en-mfa-v2",
        "canonical_units": "canonical-english-units-v1",
        "strict_provenance": True,
        "status": "no_english",
    }, "no-reference fixture")
    assert manifest["status"] == "no_english"
    with pytest.raises(ValueError):
        run_pipeline._validate_strict_english_manifest_payload({
            "schema": "strict-en-mfa-v1",
            "strict_provenance": True,
            "status": "success",
        }, "historical fixture")


def test_old_authority_manifest_without_report_binding_remains_compatible(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    manifest, clean = _audit_fixture(paths)
    assert clean
    manifest.pop("postprocess_report")

    assert _publish_and_verify(paths, manifest) == []


def test_fallback_without_report_binding_is_rejected(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    _fallback(paths, "asr_fallback")
    manifest, clean = _audit_fixture(paths)
    assert clean
    manifest.pop("postprocess_report")

    assert "postprocess_report_binding_missing" in _publish_and_verify(paths, manifest)

    manifest["policy_version"] = verifier.LEGACY_AUTHORITY_POLICY_VERSION
    assert "postprocess_report_binding_missing" in _publish_and_verify(paths, manifest)


def test_declared_report_binding_path_or_hash_tamper_is_rejected(tmp_path):
    paths = verifier._write_fixture(tmp_path)
    manifest, clean = _audit_fixture(paths)
    assert clean

    manifest["postprocess_report"]["path"] = str((paths["output"] / "other.jsonl").resolve())
    assert "postprocess_report_hash_mismatch" in _publish_and_verify(paths, manifest)

    manifest["postprocess_report"]["path"] = str(
        (paths["output"] / "postprocess_report.jsonl").resolve())
    manifest["postprocess_report"]["sha256"] = "0" * 64
    assert "postprocess_report_hash_mismatch" in _publish_and_verify(paths, manifest)


@pytest.mark.parametrize("mode", ["authority", "asr_fallback", "lab_fallback"])
def test_boundary_gap_is_legal_for_all_transcript_modes(mode):
    words = Tier("words", 10.0, 11.5, [
        Interval(10.0, 10.83, "chuan1"),
        Interval(10.83, 11.5, "yi1"),
    ])
    phones = Tier("pinyin_phones", 10.0, 11.5, [
        Interval(10.0, 10.83, "ch"),
        Interval(10.89, 11.5, "i1"),
    ])

    assert post._find_internal_pp_gaps(phones, words) == []


@pytest.mark.parametrize("mode", ["authority", "asr_fallback", "lab_fallback"])
def test_true_same_word_gap_remains_rejected_for_all_transcript_modes(mode):
    words = Tier("words", 10.0, 11.0, [Interval(10.0, 11.0, "chuan1")])
    phones = Tier("pinyin_phones", 10.0, 11.0, [
        Interval(10.0, 10.83, "ch"),
        Interval(10.89, 11.0, "uan1"),
    ])

    gaps = post._find_internal_pp_gaps(phones, words)
    assert len(gaps) == 1
    assert gaps[0]["word"] == "chuan1"


def test_fallback_projection_skips_source_insertion_without_global_shift():
    words = Tier("words", 0.0, 1.5, [
        Interval(0.0, 0.5, "ni3"),
        Interval(0.5, 1.0, "ma3"),
        Interval(1.0, 1.5, "ni3"),
    ])

    alignment = post._fallback_cjk_alignment("你好吗你", words)
    assert alignment["safe"] is True
    assert alignment["source_only"] == 1
    assert alignment["actual_only"] == 0
    assert alignment["actual_to_source"] == {0: 0, 1: 2, 2: 3}

    tier = post._build_hanzi_tier(
        words, "你好吗你", reference_authoritative=False)
    assert [iv.text for iv in tier.intervals] == ["你", "吗", "你"]


def test_authority_projection_remains_positional_and_is_not_fallback_repaired():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.5, "ni3"),
        Interval(0.5, 1.0, "ma3"),
    ])
    tier = post._build_hanzi_tier(
        words, "你好吗", reference_authoritative=True)
    assert [iv.text for iv in tier.intervals] == ["你", "好"]
