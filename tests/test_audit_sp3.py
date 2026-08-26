from types import SimpleNamespace
from unittest.mock import patch

from scripts import audit_strict_ok as audit


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
