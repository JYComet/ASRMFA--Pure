from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import audit_strict_ok as audit
from scripts import postprocess_textgrids as post


def _row(indices):
    intervals = [SimpleNamespace(text=("<sp1>" if i == 0 else "词")) for i in range(3)]
    tg = SimpleNamespace(tiers=[SimpleNamespace(name="words", intervals=intervals)])
    return {"status": "ok", "output": "/tmp/final.TextGrid", "sp3": {"count": len(indices), "details": [{"index": i} for i in indices]}} , tg


def test_edge_sp3_report_is_benign():
    row, tg = _row({0})
    with patch.object(audit, "parse_textgrid", return_value=tg):
        assert "report_positive:sp3" not in audit._report_reasons(row)


def test_internal_sp3_report_remains_veto():
    row, tg = _row({1})
    with patch.object(audit, "parse_textgrid", return_value=tg):
        assert "report_positive:sp3" in audit._report_reasons(row)


@pytest.mark.parametrize("key", [
    "nvasr_owner_selection", "nvasr_frame_support",
    "nvasr_candidate_provenance",
])
def test_rejected_nvasr_subreport_is_an_independent_audit_veto(key):
    row = {
        "status": "ok",
        key: {"status": "rejected", "reasons": [
            "multi_frame_anchor_final_60ms:0"], "candidates": [{}]},
    }

    reasons = audit._report_reasons(row)

    assert f"report_{key}_not_verified" in reasons
    assert f"report_{key}_has_reasons" in reasons


def test_evidence_digest_is_json_roundtrip_stable_and_ignores_prior_seal():
    payload = {
        "schema": "fixture-v1",
        "alignment": {"actual_to_source": {0: 0, 2: 10, 10: 20}},
    }
    producer_digest = post._evidence_digest(payload)
    persisted = json.loads(json.dumps({**payload, "digest": producer_digest}))

    assert audit._evidence_digest(persisted) == producer_digest
    persisted["digest"] = "f" * 64
    assert post._evidence_digest(persisted) == producer_digest


def _valid_fallback_bookkeeping_row():
    source_text = "你，好"
    surface = {
        "schema": "fallback-punctuation-surface-v1",
        "source_text": source_text,
        "source_digest": hashlib.sha256(source_text.encode()).hexdigest(),
        "lexical_count": 2,
        "punctuation": [{
            "source_index": 1, "lexical_boundary": 1, "label": "，"}],
    }
    surface["digest"] = audit._evidence_digest(surface)
    projection = {
        "schema": "fallback-punctuation-projection-v1",
        "source_text": source_text,
        "source_digest": surface["source_digest"],
        "surface_ledger_digest": surface["digest"],
        "alignment": {"actual_to_source": {0: 0, 1: 1}},
        "source_lexical_count": 2,
        "final_lexical_count": 2,
        "mapped": [{}, {}],
        "entries": [{}],
        "safe": True,
        "status": "verified",
        "reasons": [],
    }
    projection["digest"] = audit._evidence_digest(projection)
    correspondence = {
        "schema": "fallback-lexical-correspondence-v2",
        "status": "mapped", "safe": True, "reasons": [],
        "source_count": 2, "ctc_count": 2, "final_count": 2,
        "entries": [{}, {}],
    }
    correspondence["digest"] = audit._evidence_digest(correspondence)
    unknown = {
        "schema": "fallback-source-ctc-projection-v1",
        "status": "mapped", "safe": True, "solution_count": 1,
        "source_count": 2, "ctc_count": 2,
        "entries": [{}, {}], "first_mismatch": None, "recovered": [],
    }
    unknown["digest"] = audit._evidence_digest(unknown)
    return {
        "status": "ok",
        "fixes": [{"rule": "short_word_fix", "word": "hao3"}],
        "fallback_punctuation_surface": surface,
        "fallback_punctuation_projection": projection,
        "fallback_surface_final_commit": {
            "schema": "fallback-punctuation-surface-v1",
            "status": "verified", "reasons": [],
            "source_digest": surface["source_digest"],
            "ledger_digest": surface["digest"],
        },
        "fallback_correspondence": correspondence,
        "fallback_unknown_projection": unknown,
        "bgm_ctc_gap_selection": {
            "schema": "fallback-bgm-ctc-gap-selection-v1",
            "selection_mode": "ctc_gap_supported",
            "evaluated_intervals": [],
            "validation": {
                "status": "verified", "reasons": [],
                "digest": correspondence["digest"],
            },
        },
        "publication_contract": {
            "status": "verified", "reasons": [], "details": {
                "fallback_surface_authority": {
                    "status": "verified",
                    "source_digest": surface["source_digest"],
                    "ledger_digest": surface["digest"],
                },
                "fallback_punctuation_projection_authority": {
                    "status": "verified",
                    "ledger_digest": projection["digest"],
                },
                "fallback_correspondence_projection": {
                    "digest": correspondence["digest"],
                },
            },
        },
    }


def test_verified_fallback_bookkeeping_is_not_an_unknown_positive_veto():
    row = _valid_fallback_bookkeeping_row()
    reasons = audit._report_reasons(row)

    for key in (
            "fixes", "fallback_punctuation_surface",
            "fallback_punctuation_projection", "fallback_surface_final_commit",
            "fallback_correspondence", "fallback_unknown_projection",
            "bgm_ctc_gap_selection"):
        assert f"report_positive:{key}" not in reasons


@pytest.mark.parametrize("key", [
    "fixes", "fallback_punctuation_surface",
    "fallback_punctuation_projection", "fallback_surface_final_commit",
    "fallback_correspondence", "fallback_unknown_projection",
    "bgm_ctc_gap_selection",
])
def test_malformed_fallback_bookkeeping_remains_an_audit_veto(key):
    row = deepcopy(_valid_fallback_bookkeeping_row())
    if key == "fixes":
        row[key][0]["rule"] = "unreviewed_repair"
    elif key == "fallback_surface_final_commit":
        row[key]["status"] = "rejected"
    elif key == "bgm_ctc_gap_selection":
        row[key]["validation"]["status"] = "rejected"
    else:
        row[key]["digest"] = "0" * 64

    assert f"report_positive:{key}" in audit._report_reasons(row)


def test_audit_semantics_ignore_canonical_silence_but_not_foreign_text():
    reference = audit._semantic_tokens("你 ～ 好")
    assert audit._semantic_tokens("你 ～<sp2> 好") == reference
    assert audit._semantic_tokens("你 ～<sp2> BADTOKEN 好") != reference


def test_sp1_contract_allows_derived_tiers_to_start_with_lexical_owner():
    tg = SimpleNamespace(tiers=[
        SimpleNamespace(name="raw_text", xmin=0.0, intervals=[
            SimpleNamespace(xmin=0.0, text="<sp1>你好")]),
        SimpleNamespace(name="pinyin", xmin=0.0, intervals=[
            SimpleNamespace(xmin=0.0, text="<sp1> ni3 hao3")]),
        SimpleNamespace(name="hanzi", xmin=0.0, intervals=[
            SimpleNamespace(xmin=0.0, text="你")]),
        SimpleNamespace(name="words", xmin=0.0, intervals=[
            SimpleNamespace(xmin=0.0, text="ni3")]),
        SimpleNamespace(name="pinyin_phones", xmin=0.0, intervals=[
            SimpleNamespace(xmin=0.0, text="n")]),
    ])
    assert audit._sp1_reasons(tg) == []


def test_evidence_repair_resolves_final_pair_after_index_shift():
    words = [
        SimpleNamespace(xmin=0.0, xmax=0.4, text="ni3"),
        SimpleNamespace(xmin=0.4, xmax=0.7, text="er4"),
        SimpleNamespace(xmin=0.7, xmax=0.9, text="de5"),
    ]
    tg = SimpleNamespace(tiers=[None, None, None,
                                SimpleNamespace(intervals=words)])
    row = {"evidence_repairs": [{
        "schema": "evidence-constrained-repair-v1",
        "stem": "fixture",
        # This is the old/pre-rebuild index and intentionally points one slot
        # past the final pair.
        "word_indices": [2, 3],
        "source_words": [
            {"ordinal": 1, "start": 0.4, "end": 0.7, "text": "er4"},
            {"ordinal": 2, "start": 0.7, "end": 0.9, "text": "de5"},
        ],
        "ctc_tokens": [
            {"ordinal": 1, "word": "er4", "start_s": 0.4, "end_s": 0.7},
            {"ordinal": 2, "word": "de5", "start_s": 0.7, "end_s": 0.9},
        ],
        "boundary_s": 0.7,
        "proof": "source_mfa_ctc_unique_monotone_boundary",
    }]}
    assert audit._evidence_repair_reasons("fixture", tg, row) == []


@pytest.mark.parametrize("case", [
    "legacy",
    "missing_work",
    "missing_raw",
    "missing_both",
    "broken_work",
    "broken_raw",
    "work_directory",
    "raw_directory",
])
def test_ctc_lifecycle_markers_fail_closed_for_explicit_unsafe_paths(
        tmp_path, case):
    ctc_dir = tmp_path / "ctc"
    ctc_dir.mkdir()
    raw_path = ctc_dir / "raw_manifest.json"
    work_path = ctc_dir / "work_receipt.json"

    raw_arg = None
    work_arg = None
    if case == "missing_work":
        work_arg = work_path
    elif case == "missing_raw":
        raw_arg = raw_path
    elif case == "missing_both":
        raw_arg, work_arg = raw_path, work_path
    elif case == "broken_work":
        os.symlink("missing_work_receipt.json", work_path)
        work_arg = work_path
    elif case == "broken_raw":
        os.symlink("missing_raw_manifest.json", raw_path)
        raw_arg = raw_path
    elif case == "work_directory":
        work_path.mkdir()
        work_arg = work_path
    elif case == "raw_directory":
        raw_path.mkdir()
        raw_arg = raw_path

    reasons, lifecycle = audit._ctc_lifecycle_reasons(SimpleNamespace(
        ctc_dir=ctc_dir,
        ctc_raw_manifest=raw_arg,
        ctc_work_receipt=work_arg,
    ))

    if case == "legacy":
        assert reasons == []
        assert lifecycle is None
        return

    assert reasons
    assert lifecycle is None
    if case in {"missing_work", "missing_both", "broken_work",
                "work_directory"}:
        assert "ctc_work_receipt_missing_or_symlink" in reasons
    if case in {"missing_raw", "missing_both", "broken_raw",
                "raw_directory"}:
        assert "ctc_raw_manifest_missing_or_symlink" in reasons
