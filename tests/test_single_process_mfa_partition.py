from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pipeline import _record_single_process_mfa_partition


def test_single_process_partial_mfa_records_missing_partition():
    ctx = {}
    missing, extra = _record_single_process_mfa_partition(
        ctx, ["a", "b", "c"], {"a", "b"}, allow_partial=True)

    assert missing == ["c"]
    assert extra == []
    assert ctx["mfa_missing_stems"] == ("c",)
    assert ctx["mfa_aligned_stems"] == ("a", "b")
    assert len(ctx["mfa_aligned_stems"]) + len(ctx["mfa_missing_stems"]) == 3


def test_single_process_complete_mfa_has_empty_partition():
    ctx = {}
    missing, extra = _record_single_process_mfa_partition(
        ctx, ["a", "b"], {"a", "b"}, allow_partial=True)

    assert missing == []
    assert extra == []
    assert ctx == {}


def test_single_process_partial_mfa_does_not_admit_extra_stems():
    ctx = {}
    missing, extra = _record_single_process_mfa_partition(
        ctx, ["a", "b"], {"a", "x"}, allow_partial=True)

    assert missing == ["b"]
    assert extra == ["x"]
    assert ctx["mfa_missing_stems"] == ("b",)


def test_single_process_mfa_without_partial_keeps_missing_as_failure():
    ctx = {}
    missing, extra = _record_single_process_mfa_partition(
        ctx, ["a", "b"], {"a"}, allow_partial=False)

    assert missing == ["b"]
    assert extra == []
    assert ctx == {}
