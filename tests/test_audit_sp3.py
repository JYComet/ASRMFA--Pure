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
