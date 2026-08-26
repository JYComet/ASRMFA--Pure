import json
from pathlib import Path

import pytest

from scripts import compare_alignment_runs as compare


def _bundle(root: Path, expected: list[str], ok: set[str], statuses=None):
    root.mkdir()
    (root / "en_phones").mkdir()
    statuses = statuses or {stem: ("ok" if stem in ok else "filtered")
                            for stem in expected}
    (root / "strict_ok_manifest.json").write_text(json.dumps({
        "expected_stems": expected,
        "ok": [{"stem": stem} for stem in sorted(ok)],
        "rejected": {stem: ["reason"] for stem in expected if stem not in ok},
    }), encoding="utf-8")
    (root / ".pipeline_run_receipt_v2.json").write_text(json.dumps({
        "eligible": {"stems": sorted(expected)},
    }), encoding="utf-8")
    (root / "postprocess_report.jsonl").write_text(
        "".join(json.dumps({"stem": stem, "status": statuses[stem],
                           "filter_reasons": [] if statuses[stem] == "ok"
                           else ["overlap"]}) + "\n" for stem in expected),
        encoding="utf-8")


def test_compare_fails_closed_without_all_input_artifacts(tmp_path):
    with pytest.raises(compare.ComparisonError, match="strict output"):
        compare.compare_alignment_runs(tmp_path / "v7", tmp_path / "v9",
                                       tmp_path / "v10")


def test_compare_reports_partition_and_matched_stem_deltas(tmp_path, monkeypatch):
    expected = ["a", "b"]
    monkeypatch.setattr(compare, "EXPECTED_COUNT", 2)
    monkeypatch.setattr(compare, "EXPECTED_STEMS_SHA256", compare._digest(expected))
    v7, v9, v10 = (tmp_path / name for name in ("v7", "v9", "v10"))
    _bundle(v7, expected, {"a"})
    _bundle(v9, expected, set())
    _bundle(v10, expected, {"b"})

    result = compare.compare_alignment_runs(v7, v9, v10)
    assert result["expected_stems"] == {"count": 2, "sha256": compare._digest(expected)}
    assert result["runs"]["v7"]["strict"] == {"ok": 1, "filtered": 1}
    assert result["v7_pass_to_v10_fail"] == ["a"]
    assert result["v9_fail_to_v10_recovered"] == ["b"]
    assert result["v10_new_regressions"] == []
    assert result["per_stem"]["a"]["v10"]["strict"] is False
