import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ctc_prealign import (
    _operator_bounded_accounting_universe,
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
        extra={"processed_stems": output},
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
