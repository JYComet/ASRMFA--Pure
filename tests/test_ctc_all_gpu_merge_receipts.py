import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from ctc_prealign import (
    _operator_bounded_accounting_universe,
    _plan_all_gpu_shard,
    CTC_NORMALIZATION_MARKER,
    balanced_gpu_shard_ranges,
    expected_shard_artifact_names,
    manifest_stem_set_matches,
    refresh_ctc_summary_counts,
    validate_shard_accounting_receipt,
)
from pipeline_utils import make_pipeline_accounting_receipt


def _receipt(universe, output):
    return make_pipeline_accounting_receipt(
        source_stems=universe,
        eligible_stems=universe,
        exclusions={},
        output_stems=output,
        filtered_stems=sorted(set(universe) - set(output)),
        run_id="test",
        mode="ctc_prealign",
        route=["ctc_prealign"],
        shards=[{"shard_id": "single", "stems": universe}],
        extra={"processed_stems": universe},
    )


def test_child_accounting_receipt_accepts_exact_shard_output(tmp_path):
    universe = ["a"]
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(_receipt(universe, ["a"])), encoding="utf-8")
    validate_shard_accounting_receipt(path, {"a"}, {"a", "b"})


def test_child_accounting_receipt_rejects_wrong_shard_output(tmp_path):
    universe = ["b"]
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(_receipt(universe, ["b"])), encoding="utf-8")
    try:
        validate_shard_accounting_receipt(path, {"a"}, {"a", "b"})
    except ValueError as exc:
        assert "source mismatch" in str(exc)
    else:
        raise AssertionError("wrong shard source was accepted")


def test_child_accounting_receipt_rejects_parent_universe_buckets(tmp_path):
    # This was the old child model: source/eligible described every parent
    # stem even though output/processed described only one shard.
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(_receipt(["a", "b"], ["a"])), encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        validate_shard_accounting_receipt(path, {"a"}, {"a", "b"})


def test_child_accounting_receipt_rejects_expected_stem_outside_parent(tmp_path):
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(_receipt(["a"], ["a"])), encoding="utf-8")
    with pytest.raises(ValueError, match="outside parent universe"):
        validate_shard_accounting_receipt(path, {"a"}, {"b"})


def test_child_accounting_receipt_accepts_filtered_stems(tmp_path):
    # A shard may legitimately skip stems (e.g. an empty reference producing
    # no text).  Those skipped stems belong in filtered, not output, while
    # processed_stems still covers the full expected set.
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(make_pipeline_accounting_receipt(
        source_stems=["a", "b", "c"],
        eligible_stems=["a", "b", "c"],
        exclusions={},
        output_stems=["a", "b"],
        filtered_stems=["c"],
        run_id="test",
        mode="ctc_prealign",
        route=["ctc_prealign"],
        shards=[{"shard_id": "single", "stems": ["a", "b", "c"]}],
        extra={"processed_stems": ["a", "b", "c"]},
    )), encoding="utf-8")
    validate_shard_accounting_receipt(path, {"a", "b", "c"}, {"a", "b", "c"})


def test_unbounded_authority_keeps_frozen_exclusions():
    result = _operator_bounded_accounting_universe(
        ["a", "b", "c"], ["a", "b"], {"c": "missing_reference"}
    )
    assert result == (
        ["a", "b", "c"], ["a", "b"], {"c": "missing_reference"}
    )


def test_stems_file_subset_has_exact_exclusion_free_universe():
    result = _operator_bounded_accounting_universe(
        ["a", "b", "c"], ["a", "b"], {"c": "missing_reference"}, ["b"]
    )
    assert result == (["b"], ["b"], {})


def test_offset_limit_selection_has_exact_exclusion_free_universe():
    result = _operator_bounded_accounting_universe(
        ["a", "b", "c", "d"], ["a", "b", "c", "d"], {}, ["b", "c"]
    )
    assert result == (["b", "c"], ["b", "c"], {})


def test_all_gpu_bounded_parent_uses_selected_denominator():
    # Parent shard construction supplies the selected eligible stems to the
    # same helper used by the child; unselected source stems are not exclusions.
    result = _operator_bounded_accounting_universe(
        ["a", "b", "c", "d"], ["a", "b", "c", "d"], {}, ["b", "d"]
    )
    assert result == (["b", "d"], ["b", "d"], {})


def test_all_gpu_expected_artifacts_allows_ref_sidecar_only_for_reference_stems():
    expected = expected_shard_artifact_names(
        produced_stems={"ref_line", "asr_line"},
        reference_stems={"ref_line"},
    )
    assert "ref_line_ref.txt" in expected
    assert "asr_line_ref.txt" not in expected
    assert "ref_line.TextGrid" in expected
    assert "asr_line.TextGrid" in expected


def test_all_gpu_expected_artifacts_rejects_reference_sidecar_for_fallback_stem():
    expected = expected_shard_artifact_names(
        produced_stems={"fallback"},
        reference_stems=set(),
    )
    assert "fallback_ref.txt" not in expected


def test_all_gpu_manifest_accepts_permuted_prefix_stems_without_duplicates():
    # Full filename sorting puts the extension before a suffix beginning with
    # ``&``; manifest order is therefore not a validity condition.  The
    # preflight contract is conservation plus uniqueness.
    expected = {
        "剧情__ac59_haogan_juqing__ac59_haogan_juqing_05_26_younghelentine",
        "剧情__ac59_haogan_juqing__ac59_haogan_juqing_05_26_younghelentine&helentine",
    }
    manifest = [
        "剧情__ac59_haogan_juqing__ac59_haogan_juqing_05_26_younghelentine&helentine",
        "剧情__ac59_haogan_juqing__ac59_haogan_juqing_05_26_younghelentine",
    ]
    assert manifest_stem_set_matches(manifest, expected)


def test_all_gpu_reuses_filtered_complete_shard_without_lab_count_threshold(tmp_path):
    shard = tmp_path / "_shard_gpu3"
    shard.mkdir()
    expected = {"a", "b", "c"}
    produced = ["a", "b"]
    for stem in produced:
        for suffix in (".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
                       "_text_cn.txt", "_text_raw.txt"):
            (shard / f"{stem}{suffix}").write_text("\n", encoding="utf-8")
    (shard / "selected_stems.txt").write_text("a\nb\nc\n", encoding="utf-8")
    (shard / ".ctc_normalized").write_text(CTC_NORMALIZATION_MARKER, encoding="utf-8")
    (shard / "summary.txt").write_text(
        "Files: 3 total, 2 OK, 0 failed\n", encoding="utf-8")
    (shard / ".ctc_run_receipt.json").write_text("{}\n", encoding="utf-8")
    (shard / ".pipeline_run_receipt_v2.json").write_text(
        json.dumps(_receipt(sorted(expected), produced)), encoding="utf-8")
    (shard / "manifest.json").write_text(json.dumps([
        {"audio": f"/input/{stem}.wav", "textgrid": str(shard / f"{stem}.TextGrid"),
         "lab": str(shard / f"{stem}.lab")}
        for stem in produced
    ]), encoding="utf-8")

    selected, reused, recovered = _plan_all_gpu_shard(
        tmp_path, 3, 3, expected_stems=expected)
    assert selected == shard and reused and not recovered

    # A shard produced by an older partition layout remains reusable when its
    # own selected manifest is complete; the parent merge later enforces the
    # selected-set union against the current invocation.
    selected, reused, recovered = _plan_all_gpu_shard(
        tmp_path, 3, 4, expected_stems={"a", "b", "c", "legacy"})
    assert selected == shard and reused and not recovered


def test_all_gpu_summary_counts_are_resealed_after_invalid_bundle_drop():
    summary = "Files: 14165 total, 14015 OK, 0 failed\n"
    refreshed = refresh_ctc_summary_counts(summary, total=14165, ok=14013, failed=0)
    assert "Files: 14165 total, 14013 OK, 0 failed" in refreshed


def test_all_gpu_balanced_ranges_use_all_eight_gpus_for_seventeen_items():
    ranges = balanced_gpu_shard_ranges(17, 8)
    assert [limit for _, limit in ranges] == [3, 2, 2, 2, 2, 2, 2, 2]
    assert [offset for offset, _ in ranges] == [0, 3, 5, 7, 9, 11, 13, 15]
    assert sum(limit for _, limit in ranges) == 17


@pytest.mark.parametrize(
    "source, eligible, exclusions, selected, message",
    [
        (["a", "a"], ["a"], {}, None, "duplicate"),
        (["a", "b"], ["a"], {"b": "missing_reference"}, ["c"], "outside eligible"),
        (["a", "b"], ["b", "a"], {}, None, "must be sorted"),
        (["a", "b"], ["a", "b"], {"a": "missing_reference"}, None, "overlap"),
    ],
)
def test_accounting_universe_rejects_invalid_or_out_of_scope_selection(
    source, eligible, exclusions, selected, message
):
    with pytest.raises(ValueError, match=message):
        _operator_bounded_accounting_universe(source, eligible, exclusions, selected)
