#!/usr/bin/env python3
"""Synthetic verifier for the pipeline source-denominator accounting contract.

This test is intentionally self-contained: it creates no production artifacts
and proves the invariants needed by runners, CTC, strict audit, and publish.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import (  # noqa: E402
    PIPELINE_ACCOUNTING_SCHEMA,
    make_pipeline_accounting_receipt,
    read_pipeline_accounting_receipt,
    validate_pipeline_accounting_receipt,
    write_pipeline_accounting_receipt,
)


def _expect_invalid(receipt: dict, needle: str) -> None:
    errors = validate_pipeline_accounting_receipt(receipt)
    assert errors, "tampered receipt unexpectedly validated"
    assert any(needle in error for error in errors), (needle, errors)


def main() -> int:
    # Four WAVs with only three authoritative TXT references.  The fourth WAV
    # is an explicit missing_reference exclusion and is never ASR fallback.
    source = ["a", "b", "c", "d"]
    receipt = make_pipeline_accounting_receipt(
        source_stems=source,
        eligible_stems=["a", "b", "c"],
        exclusions={"d": "missing_reference"},
        output_stems=["a", "b", "c"],
        filtered_stems=[],
        run_id="synthetic",
        shards=[
            {"shard_id": "gpu0", "stems": ["a", "b"]},
            {"shard_id": "gpu1", "stems": ["c"]},
        ],
    )
    assert receipt["schema"] == PIPELINE_ACCOUNTING_SCHEMA
    assert receipt["source"]["count"] == 4
    assert receipt["eligible"]["count"] == 3
    assert receipt["exclusions"] == [{"stem": "d", "reason": "missing_reference"}]
    assert receipt["output"]["count"] == 3 and receipt["filtered"]["count"] == 0
    assert receipt["silent_loss"] == 0
    assert receipt["run_health"] == "healthy"
    assert validate_pipeline_accounting_receipt(receipt) == []

    # Exact shard union is required; duplicate and cross-shard assignments are
    # rejected even when aggregate counts happen to look right.
    duplicate_shards = copy.deepcopy(receipt)
    duplicate_shards["shards"][1]["stems"] = ["b", "c"]
    _expect_invalid(duplicate_shards, "cross-shard duplicate")
    missing_shard = copy.deepcopy(receipt)
    missing_shard["shards"] = [
        {"shard_id": "gpu0", "stems": ["a"]},
        {"shard_id": "gpu1", "stems": ["b"]},
    ]
    _expect_invalid(missing_shard, "shard union")

    # Any stem evidence tamper (digest or membership) must fail validation.
    tampered = copy.deepcopy(receipt)
    tampered["output"]["stems"][0] = "z"
    _expect_invalid(tampered, "output stems digest")
    tampered_digest = copy.deepcopy(receipt)
    tampered_digest["source"]["stems_digest"] = "0" * 64
    _expect_invalid(tampered_digest, "source stems digest")

    # The observed 54,000 source arithmetic is represented explicitly rather
    # than inferred from a v1 count-only receipt.
    large = make_pipeline_accounting_receipt(
        source_stems=[f"s{i:05d}" for i in range(54000)],
        eligible_stems=[f"s{i:05d}" for i in range(53998)],
        exclusions={"s53998": "missing_reference", "s53999": "missing_reference"},
        output_stems=[f"s{i:05d}" for i in range(53998)],
        filtered_stems=[],
    )
    assert (large["source_count"], large["eligible_count"], large["excluded_count"]) == (54000, 53998, 2)
    assert large["source_count"] == large["eligible_count"] + large["excluded_count"]

    # Processed quality rejection belongs in filtered; attempting to encode it
    # as a source exclusion is rejected at receipt construction time.
    try:
        make_pipeline_accounting_receipt(
            source_stems=["ok", "bad"], eligible_stems=["ok"],
            exclusions={"bad": "processed_quality_rejection"},
            output_stems=["ok"], filtered_stems=[],
        )
    except ValueError as exc:
        assert "filtered" in str(exc)
    else:
        raise AssertionError("processed rejection accepted as exclusion")
    filtered = make_pipeline_accounting_receipt(
        source_stems=["ok", "bad"], eligible_stems=["ok", "bad"],
        exclusions={}, output_stems=["ok"], filtered_stems=["bad"],
    )
    assert validate_pipeline_accounting_receipt(filtered) == []

    # Atomic write/read and explicit legacy policy.
    with tempfile.TemporaryDirectory(prefix="receipt-accounting-") as td:
        root = Path(td)
        written = write_pipeline_accounting_receipt(root, receipt)
        path = root / ".pipeline_run_receipt_v2.json"
        assert path.is_file() and read_pipeline_accounting_receipt(path) == written
        legacy = root / "legacy.json"
        legacy.write_text(json.dumps({"schema": "pipeline-run-receipt-v1"}), encoding="utf-8")
        try:
            read_pipeline_accounting_receipt(legacy)
        except ValueError as exc:
            assert "cannot be promoted" in str(exc)
        else:
            raise AssertionError("v1 receipt was promoted by inference")

    print("receipt accounting verifier: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

