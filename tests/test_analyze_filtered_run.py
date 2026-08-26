from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analyze_filtered_run as analyzer


def test_duplicate_stems_are_rejected() -> None:
    with pytest.raises(analyzer.TaxonomyError, match="duplicate report stem"):
        analyzer._unique_report_stems([{"stem": "000001_demo"}, {"stem": "000001_demo"}])


def test_count_mismatch_is_rejected(tmp_path: Path) -> None:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    (filtered / "000001_demo.TextGrid").write_text("fixture", encoding="utf-8")
    with pytest.raises(analyzer.TaxonomyError, match="filtered candidate count mismatch"):
        analyzer._candidate_stems(filtered)


def test_status_count_mismatch_is_rejected() -> None:
    rows = [{"stem": f"000{i:03d}_demo", "status": "filtered_short_word",
             "filter_reasons": ["short_word"], "output": f"/tmp/{i}.TextGrid"}
            for i in range(2)]
    paths = {row["stem"]: Path(row["output"]) for row in rows}
    with pytest.raises(analyzer.TaxonomyError, match="parent report row count mismatch"):
        analyzer._validate_report(rows, set(paths), paths)


def test_artifact_path_is_anchored(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    with pytest.raises(analyzer.TaxonomyError, match="unsafe artifact filename"):
        analyzer.resolve_anchored_artifact(root, "../outside.txt")


def test_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (root / "evidence.txt").symlink_to(outside)
    with pytest.raises(analyzer.TaxonomyError, match="regular file"):
        analyzer.resolve_anchored_artifact(root, "evidence.txt")


def test_primary_partition_prioritizes_unknown_and_missing() -> None:
    assert analyzer._primary_partition(["mfa_unknown_source", "short_word"]) == "unknown_gate"
    assert analyzer._primary_partition(["missing_mfa_alignment"]) == "missing_mfa_axis"
    assert analyzer._primary_partition(["short_word", "words_tier_gaps"]) == "qc_audio"


def test_repair_eligibility_keeps_missing_mfa_out_of_repair_lane() -> None:
    assert analyzer._repair_eligibility("missing_mfa_axis")["eligible"] is False
    assert analyzer._repair_eligibility("qc_audio")["eligible"] is True
