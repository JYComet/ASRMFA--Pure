#!/usr/bin/env python3
"""Regression checks for semantic tier-discontinuity detection."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from postprocess_textgrids import (
    Interval,
    TextGrid,
    Tier,
    _collect_tier_discontinuities,
    _count_internal_pp_gaps,
    _record_filterable_qc,
)


def _base_words() -> Tier:
    return Tier(
        "words",
        0.0,
        1.2,
        [
            Interval(0.0, 0.4, "ni3"),
            Interval(0.4, 0.8, "<sp2>"),
            Interval(0.8, 1.2, "hao3"),
        ],
    )


def _case_phone_gap_across_pause_is_not_discontinuity() -> None:
    words = _base_words()
    hanzi = Tier(
        "hanzi", 0.0, 1.2,
        [Interval(0.0, 0.4, "你"), Interval(0.4, 0.8, "<sp2>"),
         Interval(0.8, 1.2, "好")],
    )
    pp = Tier(
        "pinyin_phones", 0.0, 1.2,
        [Interval(0.0, 0.4, "n"), Interval(0.8, 1.2, "h")],
    )
    tg = TextGrid(0.0, 1.2, [hanzi, words, pp])

    assert _count_internal_pp_gaps(pp, words) == 0
    assert _collect_tier_discontinuities(tg, words) == []


def _case_phone_gap_inside_word_is_discontinuity() -> None:
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.2, "<sp1>"),
        Interval(0.2, 0.8, "chang4"),
        Interval(0.8, 1.0, "<sp1>"),
        Interval(1.0, 1.1, "<sp1>"),
        Interval(1.1, 1.2, "<sp1>"),
    ])
    hanzi = Tier("hanzi", 0.0, 1.2, [
        Interval(0.0, 0.2, "<sp1>"),
        Interval(0.2, 0.8, "唱"),
        Interval(0.8, 1.0, "<sp1>"),
        Interval(1.0, 1.1, "<sp1>"),
        Interval(1.1, 1.2, "<sp1>"),
    ])
    pp = Tier("pinyin_phones", 0.0, 1.2, [
        Interval(0.2, 0.4, "ch"),
        Interval(0.5, 0.8, "ang4"),
        Interval(0.8, 1.0, "<sp1>"),
        Interval(1.0, 1.1, "<sp1>"),
        Interval(1.1, 1.2, "<sp1>"),
    ])
    tg = TextGrid(0.0, 1.2, [hanzi, words, pp])

    assert _count_internal_pp_gaps(pp, words) == 1
    assert _collect_tier_discontinuities(tg, words) == ["pinyin_phones(1/5)"]


def _case_disabled_filter_keeps_diagnostic_only() -> None:
    report: dict = {}
    reasons: list[str] = []
    _record_filterable_qc(report, reasons, False, "tier_discontinuity", {"tiers": ["words(2/8)"]})
    assert report["tier_discontinuity"]["tiers"] == ["words(2/8)"]
    assert reasons == []


def main() -> int:
    cases = [
        _case_phone_gap_across_pause_is_not_discontinuity,
        _case_phone_gap_inside_word_is_discontinuity,
        _case_disabled_filter_keeps_diagnostic_only,
    ]
    failures = 0
    for case in cases:
        try:
            case()
            print(f"OK {case.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {case.__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
