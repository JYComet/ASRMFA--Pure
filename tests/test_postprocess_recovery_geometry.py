"""Evidence-constrained recovery fixtures and mandatory negative cases."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_strict_ok as audit
from scripts import postprocess_textgrids as post
from scripts.english_units import parse_english_units, validate_processed_english_token_binding


def _source(*items):
    return [{"ordinal": index, "start": start, "end": end, "text": text}
            for index, (start, end, text) in enumerate(items)]


def _ctc(*items):
    return [{"word": text, "start_s": start, "end_s": end, "type": "word"}
            for start, end, text in items]


def _initial_mira_fixture():
    source = _source((0.0, 0.8, "<eps>"), (0.8, 1.28, "<unk>"),
                     (1.28, 1.55, "<eps>"), (1.55, 1.73, "ni3"))
    ctc = _ctc((0.81, 1.29, "Mira"), (1.55, 1.73, "ni3"))
    words = post.Tier("words", 0.0, 1.73, [
        post.Interval(0.0, 0.81, "<sp1>"),
        post.Interval(0.81, 1.29, "Mira"),
        post.Interval(1.29, 1.55, "："),
        post.Interval(1.55, 1.73, "ni3"),
    ])
    hanzi = post.Tier("hanzi", 0.0, 1.73, [
        post.Interval(0.0, 0.81, "<sp1>"),
        post.Interval(0.81, 1.29, "Mira"),
        post.Interval(1.29, 1.55, "："),
        post.Interval(1.55, 1.73, "你"),
    ])
    record = {"word_id": "fixture:s0:w0", "ctc_ordinal": 0,
              "ctc_text": "Mira", "status": "verified",
              "provenance": "english_mfa_textgrid"}
    return source, ctc, words, hanzi, record


def test_unknown_proof_is_structured_and_bound_to_all_sources():
    source, ctc, words, hanzi, record = _initial_mira_fixture()
    proof = post._build_mfa_unknown_recovery_proof(
        "fixture", source, ctc, "Mira：你", words, hanzi,
        {"status": "verified", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record)])
    assert proof["schema"] == "mfa-unknown-recovery-proof-v1"
    assert proof["source"]["lexical_ordinal"] == 0
    assert proof["ctc"]["token"]["ordinal"] == 0
    assert proof["reference"]["token"] == "mira"
    assert proof["english_ledger"]["word_sha256"]
    assert proof["final"]["semantic_sequence_sha256"]


def test_initial_mira_proof_survives_valid_hyphen_provenance():
    source, ctc, words, hanzi, record = _initial_mira_fixture()
    unit = parse_english_units("v-tuber")[0]
    canonical = unit.to_dict()
    canonical["source_ctc_ordinals"] = [18, 20, 21]
    hyphen_record = {
        "ctc_ordinal": 18, "source_ctc_ordinals": [18, 20, 21],
        "ctc_text": "v-tuber", "unit_id": unit.unit_id,
        "alignment_token": unit.alignment_token,
        "canonical_span": list(unit.canonical_span),
    }
    hyphen_token = {
        "type": "word", "word": "vtuber", "surface_text": "v-tuber",
        "source_ctc_ordinals": [18, 20, 21],
        "canonical_span": list(unit.canonical_span),
        "canonical_unit": canonical, "hyphen_separator_omitted": True,
    }
    validate_processed_english_token_binding(hyphen_record, hyphen_token)
    proof = post._build_mfa_unknown_recovery_proof(
        "fixture", source, ctc, "Mira：你", words, hanzi,
        {"status": "verified", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record),
         (post.Interval(0.0, 0.1, "v-tuber"), hyphen_record)])
    assert proof["scenario"] == "initial_mira"


def test_unknown_proof_rejects_multiple_or_nonfirst_unknown():
    source, ctc, words, hanzi, record = _initial_mira_fixture()
    multiple = source[:2] + [{"ordinal": 2, "start": 1.28, "end": 1.4,
                              "text": "<unk>"}] + source[2:]
    assert post._build_mfa_unknown_recovery_proof(
        "fixture", multiple, ctc, "Mira：你", words, hanzi,
        {"status": "verified", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record)]) is None
    nonfirst = _source((0.0, 0.2, "ni3"), (0.2, 0.8, "<eps>"),
                       (0.8, 1.28, "<unk>"), (1.28, 1.5, "<eps>"))
    assert post._build_mfa_unknown_recovery_proof(
        "fixture", nonfirst, ctc, "Mira：你", words, hanzi,
        {"status": "verified", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record)]) is None


def test_unknown_proof_rejects_absent_or_mismatched_ledger():
    source, ctc, words, hanzi, record = _initial_mira_fixture()
    assert post._build_mfa_unknown_recovery_proof(
        "fixture", source, ctc, "Mira：你", words, hanzi,
        {"status": "rejected", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record)]) is None
    assert post._build_mfa_unknown_recovery_proof(
        "fixture", source, ctc, "Mira：你", words, hanzi,
        {"status": "verified"}, [(words.intervals[1], record)]) is None


def test_unknown_proof_rejects_reordered_ctc_tokens():
    source, ctc, words, hanzi, record = _initial_mira_fixture()
    reordered = [ctc[1], ctc[0]]
    assert post._build_mfa_unknown_recovery_proof(
        "fixture", source, reordered, "Mira：你", words, hanzi,
        {"status": "verified", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record)]) is None


def test_unknown_proof_uses_ordinal_when_mfa_and_ctc_do_not_overlap():
    source = _source((0.0, 0.20, "<eps>"), (0.20, 0.25, "<unk>"),
                     (0.25, 0.40, "<eps>"), (0.40, 0.58, "ni3"))
    ctc = _ctc((0.80, 1.10, "Mira"), (1.10, 1.28, "ni3"))
    words = post.Tier("words", 0.0, 1.28, [
        post.Interval(0.0, 0.80, "<sp1>"),
        post.Interval(0.80, 1.10, "Mira"),
        post.Interval(1.10, 1.12, "："),
        post.Interval(1.12, 1.28, "ni3"),
    ])
    hanzi = post.Tier("hanzi", 0.0, 1.28, [
        post.Interval(0.0, 0.80, "<sp1>"),
        post.Interval(0.80, 1.10, "Mira"),
        post.Interval(1.10, 1.12, "："),
        post.Interval(1.12, 1.28, "你"),
    ])
    record = {"word_id": "fixture:s0:w0", "ctc_ordinal": 0,
              "ctc_text": "Mira", "status": "verified",
              "provenance": "english_mfa_textgrid"}
    proof = post._build_mfa_unknown_recovery_proof(
        "fixture-no-overlap", source, ctc, "Mira：你", words, hanzi,
        {"status": "verified", "ledger_sha256": "a" * 64},
        [(words.intervals[1], record)])
    assert proof is not None
    assert proof["binding"]["temporal_overlap_required"] is False
    assert proof["binding"]["temporal_overlap_s"] == 0.0


def test_dual_boundary_repair_rejects_sub30ms_source_or_ctc():
    source = _source((0.0, 0.020, "yi1"), (0.020, 0.2, "er4"))
    ctc = _ctc((0.0, 0.020, "yi1"), (0.020, 0.2, "er4"))
    left, right = post.Interval(0.0, 0.1, "yi1"), post.Interval(0.1, 0.2, "er4")
    assert post._dual_evidence_boundary_solution(
        source, ctc, left, right, allow_overlap=True) is None


def test_overlap_repair_rejects_source_ctc_order_conflict():
    source = _source((0.0, 0.11, "ke3"), (0.11, 0.16, "yi3"))
    ctc = _ctc((0.0, 0.12, "ke3"), (0.12, 0.30, "yi3"))
    # Final output has the reverse order, matching the known unsafe 000407.
    left, right = post.Interval(0.0, 0.2, "yi3"), post.Interval(0.15, 0.3, "ke3")
    assert post._dual_evidence_boundary_solution(
        source, ctc, left, right, allow_overlap=True) is None


def test_true_internal_mid_sp_is_not_repaired():
    grid = post.TextGrid(0.0, 1.0, [
        post.Tier("words", 0.0, 1.0, [
            post.Interval(0.0, 0.1, "ni3"),
            post.Interval(0.1, 0.7, "<sp3>"),
            post.Interval(0.7, 1.0, "hao3"),
        ])
    ])
    before = [(iv.xmin, iv.xmax, iv.text) for iv in grid.tiers[0].intervals]
    assert post._apply_evidence_constrained_repairs(
        "001234_fixture", [], [], grid) == []
    assert [(iv.xmin, iv.xmax, iv.text) for iv in grid.tiers[0].intervals] == before


def test_publication_geometry_fills_edges_without_extending_lexical_owner():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.1, 0.4, "ni3"),
        post.Interval(0.7, 0.9, "hao3"),
    ])
    canonical = post._reconcile_publication_geometry(words)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in canonical.intervals] == [
        (0.0, 0.1, "<sp0>"),
        (0.1, 0.4, "ni3"),
        (0.4, 0.7, "<sp1>"),
        (0.7, 0.9, "hao3"),
        (0.9, 1.0, "<sp0>"),
    ]


def test_independent_audit_rejects_hole_overlap_and_cross_owner_phone():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.6, "ni3"), post.Interval(0.5, 1.0, "hao3")])
    hanzi = post.Tier("hanzi", 0.0, 1.0, [
        post.Interval(0.0, 0.6, "你"), post.Interval(0.5, 1.0, "好")])
    grid = post.TextGrid(0.0, 1.0, [
        post.Tier("raw_text", 0.0, 1.0, [post.Interval(0.0, 1.0, "<sp1>你好")]),
        post.Tier("pinyin", 0.0, 1.0, [post.Interval(0.0, 1.0, "<sp1> ni3 hao3")]),
        hanzi, words,
        post.Tier("pinyin_phones", 0.0, 1.0, [post.Interval(0.4, 0.7, "n")]),
    ])
    reasons = audit._publication_geometry_reasons(grid)
    assert "words_overlap" in reasons
    assert "hanzi_overlap" in reasons
    assert "phone_owner_mismatch" in reasons


def test_sos_consumer_rejects_reordered_policy_and_accepts_exact_record(tmp_path):
    dictionary = tmp_path / "run.dict"
    dictionary.write_text("SOS EH2 S OW2 EH1 S\n", encoding="utf-8")
    provenance = {"path": str(dictionary),
                  "sha256": hashlib.sha256(dictionary.read_bytes()).hexdigest()}
    phones = [{"label": label} for label in post.SOS_EXPECTED_PRONUNCIATION]
    record = {
        "alignment_token": "sos",
        "pronunciation_policy_id": post.SOS_PRONUNCIATION_POLICY_ID,
        "dictionary_provenance": provenance,
        "phones": phones,
        "pronunciation_policy": {
            "policy_id": post.SOS_PRONUNCIATION_POLICY_ID,
            "expected_pronunciation": list(post.SOS_EXPECTED_PRONUNCIATION),
            "actual_source_sequence": list(post.SOS_EXPECTED_PRONUNCIATION),
            "dictionary_provenance": provenance,
        },
    }
    ledger = {"dictionary_provenance": provenance}
    assert post._strict_en_pronunciation_reason(record, ledger) is None
    assert audit._pronunciation_consumer_reasons(
        record, {"phones": phones}, ledger) == []
    record["pronunciation_policy"]["actual_source_sequence"] = ["EH2", "OW2", "S", "EH1", "S"]
    assert post._strict_en_pronunciation_reason(record, ledger) == "sos_pronunciation_policy_mismatch"


def test_disk_audit_rejects_old_words_geometry_after_processed_freeze():
    words = post.Tier("words", 0.0, 0.9, [
        post.Interval(0.0, 0.5, "ni3"),
        post.Interval(0.5, 0.9, "hao3"),
    ])
    final_grid = post.TextGrid(0.0, 0.9, [words])
    digest = audit._processed_geometry_digest(final_grid)
    ledger = [{"operation": "boundary_freeze"}]
    row = {
        "processed_geometry_digest": digest,
        "processed_operation_ledger": ledger,
        "processed_geometry": {
            "schema": audit.PROCESSED_GEOMETRY_SCHEMA,
            "frozen": True,
            "digest": digest,
            "ledger": ledger,
        },
        "publication_contract": {
            "status": "verified",
            "reasons": [],
            "details": {
                "ctc_lexical_evidence_proof": [
                    {"published_span": [0.0, 0.5]},
                    {"published_span": [0.5, 0.9]},
                ],
            },
        },
    }
    assert audit._postprocess_contract_reasons(row, final_grid, None) == []

    stale_grid = post.TextGrid(0.0, 0.9, [post.Tier("words", 0.0, 0.9, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.9, "hao3"),
    ])])
    reasons = audit._postprocess_contract_reasons(row, stale_grid, None)
    assert "processed_geometry_digest_mismatch" in reasons
