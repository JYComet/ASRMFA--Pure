import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ctc_prealign import validate_shard_accounting_receipt
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
    universe = ["a", "b"]
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(_receipt(universe, ["a"])), encoding="utf-8")
    validate_shard_accounting_receipt(path, {"a"}, set(universe))


def test_child_accounting_receipt_rejects_wrong_shard_output(tmp_path):
    universe = ["a", "b"]
    path = tmp_path / ".pipeline_run_receipt_v2.json"
    path.write_text(json.dumps(_receipt(universe, ["b"])), encoding="utf-8")
    try:
        validate_shard_accounting_receipt(path, {"a"}, set(universe))
    except ValueError as exc:
        assert "output mismatch" in str(exc)
    else:
        raise AssertionError("wrong shard output was accepted")
