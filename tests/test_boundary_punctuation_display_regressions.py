"""Synthetic publication/display geometry fixtures for historical stems.

These fixtures intentionally model the final display tiers without copying
runtime artifacts.  ``words`` owns the time axis; ``hanzi`` is always derived
from it.  The two marked historical assertions are expected to expose the
current punctuation-tail behavior for W3.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import postprocess_textgrids as post


AXIS_XMIN = 0.0
AXIS_XMAX = 1.0
EPS = 1e-9
PUNCTUATION = set("，。…！？、；：,.!?;:～")


@dataclass
class GeometryFixture:
    stem: str
    reference: str
    ctc_tokens: list[dict]
    words: post.Tier

    @property
    def punct_entries(self) -> list[dict]:
        return [row for row in self.ctc_tokens
                if row["word"] in PUNCTUATION]


def _tier(name: str, *items: tuple[float, float, str]) -> post.Tier:
    return post.Tier(name, AXIS_XMIN, AXIS_XMAX,
                     [post.Interval(start, end, text)
                      for start, end, text in items])


def _clone_tier(tier: post.Tier) -> post.Tier:
    return post.Tier(
        tier.name,
        tier.xmin,
        tier.xmax,
        [post.Interval(iv.xmin, iv.xmax, iv.text)
         for iv in tier.intervals],
    )


def _fixture_52697() -> GeometryFixture:
    # The reference ends at 欢.  CTC's extra period reaches the audio end,
    # while the explicit tail silence remains the axis-closing display owner.
    return GeometryFixture(
        stem="52697",
        reference="欢",
        ctc_tokens=[
            {"word": "huan1", "start_s": 0.05, "end_s": 0.72,
             "type": "word"},
            {"word": ".", "start_s": 0.72, "end_s": 1.0,
             "type": "punct"},
        ],
        words=_tier(
            "words",
            (0.00, 0.05, "<sp0>"),
            (0.05, 0.72, "huan1"),
            (0.72, 1.00, "<sp1>"),
        ),
    )


def _fixture_16735() -> GeometryFixture:
    # Authority punctuation owns only its local post-lexical gap.  The final
    # edge after 绪！ is still explicit silence, not lexical time.
    return GeometryFixture(
        stem="16735",
        reference="备，绪！",
        ctc_tokens=[
            {"word": "bei4", "start_s": 0.05, "end_s": 0.38,
             "type": "word"},
            {"word": "，", "start_s": 0.38, "end_s": 0.52,
             "type": "punct"},
            {"word": "xu4", "start_s": 0.52, "end_s": 0.78,
             "type": "word"},
            {"word": "！", "start_s": 0.78, "end_s": 0.91,
             "type": "punct"},
        ],
        words=_tier(
            "words",
            (0.00, 0.05, "<sp0>"),
            (0.05, 0.38, "bei4"),
            (0.38, 0.40, "，"),
            (0.40, 0.52, "<sp1>"),
            (0.52, 0.78, "xu4"),
            (0.78, 0.82, "！"),
            (0.82, 1.00, "<sp0>"),
        ),
    )


def _fixture_16806() -> GeometryFixture:
    # The 520 ms interior pause is acoustic silence, not punctuation.  It is
    # canonical <sp2> and should make this candidate filterable.
    pause = post.silence_label(0.52)
    assert pause == "<sp2>"
    return GeometryFixture(
        stem="16806",
        reference="你好吗",
        ctc_tokens=[
            {"word": "ni3", "start_s": 0.05, "end_s": 0.30,
             "type": "word"},
            {"word": "hao3", "start_s": 0.82, "end_s": 0.92,
             "type": "word"},
            {"word": "ma1", "start_s": 0.92, "end_s": 1.00,
             "type": "word"},
        ],
        words=_tier(
            "words",
            (0.00, 0.05, "<sp0>"),
            (0.05, 0.30, "ni3"),
            (0.30, 0.82, pause),
            (0.82, 0.92, "hao3"),
            (0.92, 1.00, "ma1"),
        ),
    )


def _materialize(fixture: GeometryFixture) -> tuple[post.Tier, post.Tier]:
    """Apply only the display punctuation projection and derive hanzi."""
    words = _clone_tier(fixture.words)
    if fixture.punct_entries:
        words, _ = post._inject_punctuation(
            words, None, deepcopy(fixture.punct_entries))
        post._restore_reference_punctuation(
            words, fixture.reference, deepcopy(fixture.punct_entries))
    hanzi = post._build_hanzi_tier(
        words,
        fixture.reference,
        reference_authoritative=True,
    )
    return words, hanzi


def _geometry_reasons(words: post.Tier, hanzi: post.Tier,
                      *, reject_interior_sp: bool = True) -> list[str]:
    """Small publication/display oracle kept local until W3 centralizes it."""
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for tier in (words, hanzi):
        if not tier.intervals:
            add(f"{tier.name}_empty")
            continue
        if tier.intervals[0].xmin > tier.xmin + EPS:
            add(f"{tier.name}_coverage_hole")
        previous = tier.intervals[0]
        for current in tier.intervals[1:]:
            delta = current.xmin - previous.xmax
            if delta > EPS:
                add(f"{tier.name}_coverage_hole")
            elif delta < -EPS:
                add(f"{tier.name}_overlap")
            previous = current
        if tier.intervals[-1].xmax < tier.xmax - EPS:
            add(f"{tier.name}_coverage_hole")

    if len(words.intervals) != len(hanzi.intervals):
        add("hanzi_words_count_mismatch")
    for word, label in zip(words.intervals, hanzi.intervals):
        if (abs(word.xmin - label.xmin) > EPS
                or abs(word.xmax - label.xmax) > EPS):
            add("hanzi_words_boundary_mismatch")
        if ((post.is_silence(word.text) or word.text.strip() in PUNCTUATION)
                and word.text != label.text):
            add("silence_or_punctuation_label_mismatch")

    word_silence = [(iv.xmin, iv.xmax, iv.text)
                    for iv in words.intervals if post.is_silence(iv.text)]
    hanzi_silence = [(iv.xmin, iv.xmax, iv.text)
                     for iv in hanzi.intervals if post.is_silence(iv.text)]
    if word_silence != hanzi_silence:
        add("silence_label_split")

    if reject_interior_sp:
        for iv in words.intervals:
            if (post.is_silence(iv.text)
                    and iv.xmin > words.xmin + EPS
                    and iv.xmax < words.xmax - EPS):
                add("strict_interior_sp")
    return reasons


def _fixture_with_aux_tiers(words: post.Tier) -> post.TextGrid:
    # The filtering helper only needs named tiers; identical synthetic labels
    # keep the fixture deterministic and make silence ownership inspectable.
    phones = _clone_tier(words)
    phones.name = "phones"
    pinyin_phones = _clone_tier(words)
    pinyin_phones.name = "pinyin_phones"
    return post.TextGrid(AXIS_XMIN, AXIS_XMAX,
                         [words, phones, pinyin_phones])


def test_52697_extra_terminal_ctc_period_does_not_extend_huan_or_drop_edge_silence():
    fixture = _fixture_52697()
    words, hanzi = _materialize(fixture)

    huan = next(iv for iv in words.intervals if iv.text == "huan1")
    assert fixture.ctc_tokens[-1]["word"] == "."
    assert fixture.ctc_tokens[-1]["end_s"] == AXIS_XMAX
    assert huan.xmax == 0.72
    assert words.intervals[-1].text == "<sp1>"
    assert words.intervals[-1].xmax == AXIS_XMAX
    assert not _geometry_reasons(words, hanzi), _geometry_reasons(words, hanzi)


def test_16735_authority_punctuation_keeps_local_gaps_off_lexical_words():
    fixture = _fixture_16735()
    words, hanzi = _materialize(fixture)

    lexical = {iv.text: iv for iv in words.intervals if iv.text in {"bei4", "xu4"}}
    punctuation = {iv.text: iv for iv in words.intervals
                   if iv.text in {"，", "！"}}
    assert lexical["bei4"].xmax == 0.38
    assert lexical["xu4"].xmax == 0.78
    assert (punctuation["，"].xmin, punctuation["，"].xmax) == (0.38, 0.52)
    assert (punctuation["！"].xmin, punctuation["！"].xmax) == (0.78, 0.91)
    assert words.intervals[-1].text == "<sp0>"
    assert words.intervals[-1].xmax == AXIS_XMAX
    assert not _geometry_reasons(words, hanzi), _geometry_reasons(words, hanzi)


def test_006035_authority_punctuation_keeps_local_geometry_and_allows_missing_tail():
    reference = "哈喽哈喽，欢迎回来！在OK上发现一个超赞的dancer创作者。不用着急哦。"
    units = post._extract_word_chars(reference)
    lexical_units = [unit for unit in units if post.is_word_like(unit)]
    puncts = [(unit, index) for index, unit in enumerate(units)
              if post.is_punct(unit) and unit in PUNCTUATION]
    assert [char for char, _ in puncts] == ["，", "！", "。", "。"]

    # Boundaries from the v9 006035 evidence.  The final 400 ms pause is
    # intentionally not owned by the final period.
    local_gaps = {
        puncts[0][1]: (1.17, 1.29),
        puncts[1][1]: (2.00, 2.61),
        puncts[2][1]: (5.54, 6.395),
        puncts[3][1]: (7.13, 7.53),
    }
    intervals = []
    cursor = 0.05
    lexical_cursor = 0
    for punct_index, (char, unit_index) in enumerate(puncts):
        boundary = sum(1 for unit in units[:unit_index] if post.is_word_like(unit))
        target_end, next_start = local_gaps[unit_index]
        count = boundary - lexical_cursor
        step = (target_end - cursor) / count
        for offset in range(count):
            end = target_end if offset == count - 1 else cursor + step
            intervals.append(post.Interval(cursor, end, lexical_units[lexical_cursor]))
            cursor = end
            lexical_cursor += 1
        if next_start > target_end:
            intervals.append(post.Interval(target_end, next_start, "<sp1>"))
        cursor = next_start
    # Last lexical group ends before the final unexplained pause.
    if lexical_cursor < len(lexical_units):
        step = (7.13 - cursor) / (len(lexical_units) - lexical_cursor)
        while lexical_cursor < len(lexical_units):
            end = (7.13 if lexical_cursor == len(lexical_units) - 1
                   else cursor + step)
            intervals.append(post.Interval(cursor, end, lexical_units[lexical_cursor]))
            cursor = end
            lexical_cursor += 1
    words = post.Tier("words", 0.0, 8.0, sorted(intervals, key=lambda iv: iv.xmin))
    punct_entries = [
        {"word": "，", "start_s": 1.17, "end_s": 1.29},
        {"word": "！", "start_s": 2.135, "end_s": 2.61},
        {"word": "。", "start_s": 5.54, "end_s": 6.395},
        {"word": "。", "start_s": 7.53, "end_s": 7.6347},
    ]

    post._restore_reference_punctuation(words, reference, punct_entries)
    comma = next(iv for iv in words.intervals if iv.text == "，")
    first_period = next(iv for iv in words.intervals
                        if iv.text == "。" and abs(iv.xmin - 5.54) < 1e-6)
    assert (comma.xmin, comma.xmax) == (1.17, 1.29)
    assert (first_period.xmin, first_period.xmax) == (5.54, 6.395)
    # The final CTC anchor starts after the only explicit local silence and
    # has no silence support of its own.  The authority mark is therefore
    # missing_allowed; no voiced or implicit edge duration is invented.
    assert not any(iv.text == "。" and abs(iv.xmin - 7.53) < 1e-6
                   for iv in words.intervals)
    assert any(iv.text == "<sp1>" and iv.xmin == 7.13 and iv.xmax == 7.53
               for iv in words.intervals)


def test_16806_long_interior_pause_is_canonical_sp_and_filterable():
    fixture = _fixture_16806()
    words, hanzi = _materialize(fixture)
    pause = next(iv for iv in words.intervals if post.is_silence(iv.text)
                 and iv.xmin > 0.0)

    assert pause.text == "<sp2>"
    assert _geometry_reasons(words, hanzi) == ["strict_interior_sp"]
    filter_reasons = post.handle_unexpected_silences(
        _fixture_with_aux_tiers(_clone_tier(words)),
        "<sp1> ni3 hao3 ma1",
    )
    assert "unexpected_silence" in filter_reasons
    # A substantive unowned pause is retained as an explicit owner.  It is
    # filtered, not hidden by extending the preceding lexical word.
    preserved = _clone_tier(words)
    post.handle_unexpected_silences(
        post.TextGrid(AXIS_XMIN, AXIS_XMAX,
                      [preserved, _clone_tier(preserved), _clone_tier(preserved)]),
        "<sp1> ni3 hao3 ma1",
    )
    pause_after = next(iv for iv in preserved.intervals if post.is_silence(iv.text)
                       and iv.xmin > 0.0 and iv.xmax < 1.0)
    assert (pause_after.xmin, pause_after.xmax) == (0.30, 0.82)


def test_short_internal_sp0_and_sp1_merge_left_without_audio():
    args = SimpleNamespace(merge_max_sil_sec=0.5, min_sil_merge_sec=0.5,
                           merge_silence=True, merge_energy_threshold=0.5)
    for label, end in (("<sp0>", 0.20), ("<sp1>", 0.50)):
        grid = post.TextGrid(0.0, 1.0, [post.Tier("words", 0.0, 1.0, [
            post.Interval(0.0, 0.10, "ni3"),
            post.Interval(0.10, end, label),
            post.Interval(end, 1.0, "hao3"),
        ])])
        decisions = post.resolve_visual_short_silence_merges(
            grid, None, 16000, args)
        assert [iv.text for iv in grid.tiers[0].intervals] == ["ni3", "hao3"]
        assert decisions[0]["decision"] == "merged_left"
        assert decisions[0]["direction_source"] in {
            "forced_left", "forced_left_fallback"}


def test_long_internal_pause_is_not_redeemed_by_correspondence():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.10, "ni3"),
        post.Interval(0.10, 0.60, "<sp2>"),
        post.Interval(0.60, 1.0, "hao3"),
    ])
    grid = post.TextGrid(0.0, 1.0, [words])
    args = SimpleNamespace(merge_max_sil_sec=0.5, min_sil_merge_sec=0.5,
                           merge_silence=True, merge_energy_threshold=0.5)
    post.resolve_visual_short_silence_merges(grid, None, 16000, args)
    assert [iv.text for iv in words.intervals] == ["ni3", "<sp2>", "hao3"]
    assert post._terminal_punctuation_evidence_missing(
        words, reference_authoritative=False) is None


def test_exact_fallback_punctuation_candidate_owns_internal_gap():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.20, "ni3"),
        post.Interval(0.20, 0.70, "<sp2>"),
        post.Interval(0.70, 1.0, "hao3"),
    ])
    updated, _ = post._inject_punctuation(words, None, [{
        "schema": post.PUNCTUATION_EVIDENCE_SCHEMA,
        "word": "，", "raw_start_s": 0.20, "raw_end_s": 0.26,
        "start_s": 0.20, "end_s": 0.26,
        "candidate_id": "ctc-punct-0000", "source": "ctc",
        "left_lexical_ordinal": 0, "right_lexical_ordinal": 1,
    }], reference_authoritative=False)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.20, "ni3"), (0.20, 0.70, "，"), (0.70, 1.0, "hao3")]
    assert updated._punctuation_evidence_ledger["owners"][0]["raw_span"] == [
        0.20, 0.26]


@pytest.mark.parametrize(
    ("stem", "source", "left", "pause_start", "pause_end", "right", "label"),
    [
        ("00015", "了，明", "le5", 4.89, 5.55, "ming2", "，"),
        ("00039", "吗？让", "ma5", 4.43, 5.55, "rang4", "？"),
        ("00046", "吗？没", "ma5", 6.29, 7.17, "mei2", "？"),
        ("00054", "吧！嘉", "ba5", 4.29, 5.31, "jia1", "！"),
        ("00057", "<LAUGHTER>，谢", "LAUGHTER", 0.99, 1.39, "xie4", "，"),
        ("00084", "池，拿", "chi2", 7.11, 7.94, "na4", "，"),
        ("00096", "的！一", "de5", 4.26, 4.83, "yi1", "！"),
    ],
)
def test_seven_filtered_stems_source_punctuation_owns_explicit_silence(
        stem, source, left, pause_start, pause_end, right, label):
    words = post.Tier("words", 0.0, pause_end + 0.2, [
        post.Interval(0.0, pause_start, left),
        post.Interval(pause_start, pause_end, "<sp2>"),
        post.Interval(pause_end, pause_end + 0.2, right),
    ])
    ledger = post._fallback_punctuation_surface_ledger(source)
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, pause_start, left), (pause_start, pause_end, label),
        (pause_end, pause_end + 0.2, right),
    ], stem


def test_source_only_punctuation_uses_exact_positive_ctc_gap():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.20, "ni3"),
        post.Interval(0.70, 1.0, "hao3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger("你，好")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.20},
            {"type": "word", "word": "hao3", "start_s": 0.70, "end_s": 1.0},
        ])
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.20, "ni3"), (0.20, 0.70, "，"), (0.70, 1.0, "hao3")]
    owner = updated._punctuation_evidence_ledger["owners"][0]
    assert owner["source"] == "fallback_surface_ctc_gap"


def test_source_only_zero_width_ctc_gap_fails_closed():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.50, "ni3"),
        post.Interval(0.50, 1.0, "hao3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger("你，好")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.50},
            {"type": "word", "word": "hao3", "start_s": 0.50, "end_s": 1.0},
        ])
    assert updated is words
    assert updated._punctuation_evidence_ledger["rejected"][0]["reason"] == \
        "punctuation_gap_zero_width"


def test_source_only_zero_width_gap_allocates_one_frame_from_adjacent_nvv():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.30, "<BREATHING>"),
        post.Interval(0.30, 1.0, "hao3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger("Breathing，好")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "BREATHING",
             "start_s": 0.0, "end_s": 0.30},
            {"type": "word", "word": "hao3",
             "start_s": 0.30, "end_s": 1.0},
        ])
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, pytest.approx(0.24), "<BREATHING>"),
        (pytest.approx(0.24), 0.30, "，"),
        (0.30, 1.0, "hao3"),
    ]
    owner = updated._punctuation_evidence_ledger["owners"][0]
    assert owner["source"] == "fallback_surface_adjacent_nvv_frame"
    assert owner["nvv_side"] == "left"
    assert owner["allocation_width_s"] == pytest.approx(0.060)


def test_punctuation_owners_trim_one_nvv_without_duplicating_lexical_ordinal():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.40, "zi5"),
        post.Interval(0.40, 0.60, "<BREATHING>"),
        post.Interval(0.60, 1.0, "ni3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger(
        "子， Breathing …你")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "zi5", "start_s": 0.0,
             "end_s": 0.46},
            {"type": "word", "word": "BREATHING", "start_s": 0.54,
             "end_s": 0.60},
            {"type": "word", "word": "ni3", "start_s": 0.70,
             "end_s": 1.0},
        ])
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.46, "zi5"),
        (0.46, 0.54, "，"),
        (0.54, 0.60, "<BREATHING>"),
        (0.60, 0.70, "…"),
        (0.70, 1.0, "ni3"),
    ]
    lexical = [iv for iv in updated.intervals
               if not post.is_punct(iv.text) and not post.is_silence(iv.text)]
    assert [iv.text for iv in lexical] == ["zi5", "<BREATHING>", "ni3"]


def test_short_nvv_between_two_punctuation_owners_uses_ctc_geometry():
    """LAria_00285: punctuation must not erase a displaced 60 ms NVV."""
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.20, "kan4"),
        post.Interval(0.20, 0.23, "BREATHING"),
        post.Interval(0.70, 1.0, "ni3"),
    ])
    ctc = [
        {"type": "word", "word": "kan4", "start_s": 0.0, "end_s": 0.20},
        {"type": "word", "word": "BREATHING", "start_s": 0.32,
         "end_s": 0.38},
        {"type": "word", "word": "ni3", "start_s": 0.70, "end_s": 1.0},
    ]
    punct = [
        {"schema": post.PUNCTUATION_EVIDENCE_SCHEMA, "word": "，",
         "start_s": 0.20, "end_s": 0.26, "raw_start_s": 0.20,
         "raw_end_s": 0.26, "candidate_id": "ctc-punct-0000",
         "source": "ctc", "left_lexical_ordinal": 0,
         "right_lexical_ordinal": 1},
        {"schema": post.PUNCTUATION_EVIDENCE_SCHEMA, "word": "…",
         "start_s": 0.38, "end_s": 0.44, "raw_start_s": 0.38,
         "raw_end_s": 0.44, "candidate_id": "ctc-punct-0001",
         "source": "ctc", "left_lexical_ordinal": 1,
         "right_lexical_ordinal": 2},
    ]

    snapped, _ = post._snap_to_ctc(words, None, ctc, punct_entries=punct)
    breathing = next(iv for iv in snapped.intervals
                     if post.is_nvv_token(iv.text))
    assert (breathing.xmin, breathing.xmax) == pytest.approx((0.32, 0.38))
    authority = snapped._ctc_word_authority[1]
    assert authority["boundary_source"] == "ctc"

    updated, _ = post._inject_punctuation(
        snapped, None, punct, reference_authoritative=False,
        source_surface_ledger=post._fallback_punctuation_surface_ledger(
            "看， BREATHING …你"),
        ctc_tokens=ctc)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.20, "kan4"),
        (0.20, 0.32, "，"),
        (0.32, 0.38, "BREATHING"),
        (0.38, 0.70, "…"),
        (0.70, 1.0, "ni3"),
    ]


def test_boundary_owner_does_not_trim_disjoint_adjacent_nvv():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.35, "zi5"),
        post.Interval(0.35, 0.47, "<BREATHING>"),
        post.Interval(0.70, 1.0, "ni3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger(
        "子， Breathing …你")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "zi5", "start_s": 0.0,
             "end_s": 0.49},
            {"type": "word", "word": "BREATHING", "start_s": 0.55,
             "end_s": 0.61},
            {"type": "word", "word": "ni3", "start_s": 0.70,
             "end_s": 1.0},
        ])
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.35, "zi5"),
        (0.35, 0.47, "<BREATHING>"),
        (0.49, 0.55, "，"),
        (0.61, 0.70, "…"),
        (0.70, 1.0, "ni3"),
    ]


def test_ctc_edge_completion_closes_ownerless_gap_before_source_punctuation():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.24, "zi5"),
        post.Interval(0.30, 0.36, "<BREATHING>"),
        post.Interval(0.70, 1.0, "ni3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger(
        "子， Breathing …你")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "zi5", "start_s": 0.0,
             "end_s": 0.30},
            {"type": "word", "word": "BREATHING", "start_s": 0.36,
             "end_s": 0.42},
            {"type": "word", "word": "ni3", "start_s": 0.70,
             "end_s": 1.0},
        ])
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.30, "zi5"),
        (0.30, 0.36, "，"),
        (0.36, 0.42, "<BREATHING>"),
        (0.42, 0.70, "…"),
        (0.70, 1.0, "ni3"),
    ]
    repairs = updated._punctuation_evidence_ledger["edge_repairs"]
    assert repairs == [
        {
            "source": "fallback_surface_ctc_lexical_edge_completion",
            "side": "right", "lexical_ordinal": 0,
            "old_span": [0.0, 0.24], "new_span": [0.0, 0.30],
            "supporting_ctc_span": [0.0, 0.30],
        },
        {
            "source": "fallback_surface_ctc_lexical_edge_completion",
            "side": "right", "lexical_ordinal": 1,
            "old_span": [0.36, 0.36], "new_span": [0.36, 0.42],
            "supporting_ctc_span": [0.36, 0.42],
        },
    ]


def test_ctc_edge_completion_rejects_lexical_identity_mismatch():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.24, "zi5"),
        post.Interval(0.30, 1.0, "hao3"),
    ])
    ledger = post._fallback_punctuation_surface_ledger("子，好")
    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=ledger,
        ctc_tokens=[
            {"type": "word", "word": "ta1", "start_s": 0.0,
             "end_s": 0.30},
            {"type": "word", "word": "hao3", "start_s": 0.36,
             "end_s": 1.0},
        ])
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.24, "zi5"),
        (0.30, 0.36, "，"),
        (0.36, 1.0, "hao3"),
    ]
    assert updated._punctuation_evidence_ledger["edge_repairs"] == []


def test_fallback_publication_audit_rejects_shifted_punctuation_boundary():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.3, "ni3"),
        post.Interval(0.3, 0.6, "hao3"),
        post.Interval(0.6, 0.7, "，"),
        post.Interval(0.7, 1.0, "ma1"),
    ])
    ledger = post._fallback_punctuation_surface_ledger("你，好吗")
    reasons, details = post._publication_contract_audit(
        words, None, None, None, "你，好吗", None, None, False,
        reference_mode="fallback", fallback_surface_ledger=ledger)
    assert "fallback_punctuation_ownership_mismatch" in reasons
    projection = details["fallback_punctuation_projection"]
    assert projection["expected"][0]["boundary"] == 1
    assert projection["observed"][0]["boundary"] == 2


def test_terminal_punctuation_candidate_absorbs_tail_to_axis_end():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.40, "hao3"),
        post.Interval(0.40, 1.0, "<sp1>"),
    ])
    updated, _ = post._inject_punctuation(words, None, [{
        "schema": post.PUNCTUATION_EVIDENCE_SCHEMA,
        "word": "。", "raw_start_s": 0.40, "raw_end_s": 0.46,
        "start_s": 0.40, "end_s": 0.46,
        "candidate_id": "ctc-punct-0000", "source": "ctc",
        "left_lexical_ordinal": 0, "right_lexical_ordinal": None,
    }], reference_authoritative=False)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.40, "hao3"), (0.40, 1.0, "。")]
    assert post._terminal_punctuation_evidence_missing(
        updated, reference_authoritative=False) is None


def test_terminal_silence_without_candidate_is_explicitly_missing():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.40, "hao3"),
        post.Interval(0.40, 1.0, "<sp1>"),
    ])
    evidence = post._terminal_punctuation_evidence_missing(
        words, reference_authoritative=False)
    assert evidence["reason"] == "terminal_punctuation_evidence_missing"


def test_final_publication_transaction_preserves_fallback_source_surface():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.6, "，"),
        post.Interval(0.6, 1.0, "hao3"),
    ])
    grid = post.TextGrid(0.0, 1.0, [
        post.Tier("raw_text", 0.0, 1.0, [post.Interval(0.0, 1.0, " stale")]),
        post.Tier("pinyin", 0.0, 1.0, [post.Interval(0.0, 1.0, " stale")]),
        post.Tier("hanzi", 0.0, 1.0, []), words,
    ])
    grid._processed_geometry_frozen = True
    post._rebuild_derived_from_frozen_words(
        grid, {}, {}, "你好，", pinyin_text="ni3 hao3")
    assert grid.tiers[0].intervals[0].text == "<sp1>你好，"
    assert [iv.text for iv in grid.tiers[2].intervals] == ["你", "，", "好"]
    assert grid.tiers[1].intervals[0].text == "<sp1> ni3 ， hao3"
    assert grid._derived_publication_transaction["schema"] == \
        post.PUBLICATION_TRANSACTION_SCHEMA


def test_final_publication_normalizes_bare_nvv_and_audits_its_punctuation():
    # This models the r1 surface defect: the recognized bare label contains a
    # lexical hyphen, but the final raw_text owner must publish canonical NVV
    # markup so that the hyphen cannot enter punctuation accounting.
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.20, "da4"),
        post.Interval(0.20, 0.30, "？"),
        post.Interval(0.30, 0.65, "SURPRISE-WA"),
        post.Interval(0.65, 0.90, "hao3"),
        post.Interval(0.90, 1.00, "。"),
    ])
    raw = post.Tier("raw_text", 0.0, 1.0,
                    [post.Interval(0.0, 1.0, "stale")])
    pinyin = post.Tier("pinyin", 0.0, 1.0,
                       [post.Interval(0.0, 1.0, "stale")])
    hanzi = post.Tier("hanzi", 0.0, 1.0, [])
    grid = post.TextGrid(0.0, 1.0, [raw, pinyin, hanzi, words])
    frozen, freeze_reasons = post._freeze_processed_geometry(grid)
    assert frozen is not None and freeze_reasons == []
    post._rebuild_derived_from_frozen_words(
        grid, {}, {}, "大家？Surprise-wa好。",
        reference_authoritative=True,
        pinyin_text="da4 ? Surprise-wa hao3 .")

    assert raw.intervals[0].text == "<sp1>大家？<SURPRISE-WA>好。"
    reasons, details = post._publication_contract_audit(
        words, post.tier_by_name(grid, "hanzi"), None, None,
        "大家？Surprise-wa好。", None, None, True,
        raw_text_tier=raw, pinyin_tier=pinyin)
    assert "raw_text_punctuation_sequence_mismatch" not in reasons
    assert "pinyin_punctuation_sequence_mismatch" not in reasons
    assert post._surface_punctuation(raw) == ["？", "。"]
    lexical_hyphen = post.Tier("raw_text", 0.0, 1.0, [
        post.Interval(0.0, 1.0, "open-ai<SURPRISE-WA>")])
    assert post._surface_punctuation(lexical_hyphen) == ["-"]

    # The r1 path is fallback-mode: raw_text is rebuilt from the derived
    # surface, so canonicalization must happen there as well.
    fallback_words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.20, "da4"),
        post.Interval(0.20, 0.30, "？"),
        post.Interval(0.30, 0.65, "Surprise-wa"),
        post.Interval(0.65, 0.90, "hao3"),
        post.Interval(0.90, 1.00, "。"),
    ])
    fallback_raw = post.Tier("raw_text", 0.0, 1.0,
                             [post.Interval(0.0, 1.0, "stale")])
    fallback_pinyin = post.Tier("pinyin", 0.0, 1.0,
                                [post.Interval(0.0, 1.0, "stale")])
    fallback_hanzi = post.Tier("hanzi", 0.0, 1.0, [])
    fallback_grid = post.TextGrid(
        0.0, 1.0, [fallback_raw, fallback_pinyin, fallback_hanzi,
                   fallback_words])
    frozen, freeze_reasons = post._freeze_processed_geometry(fallback_grid)
    assert frozen is not None and freeze_reasons == []
    post._rebuild_derived_from_frozen_words(
        fallback_grid, {}, {}, "大？Surprise-wa好。",
        reference_authoritative=False)
    assert fallback_raw.intervals[0].text == "<sp1>大？<SURPRISE-WA>好。"


def test_nvv_markup_is_not_counted_as_surface_punctuation():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.4, "da4"),
        post.Interval(0.4, 0.5, "？"),
        post.Interval(0.5, 0.8, "BREATHING"),
        post.Interval(0.8, 0.9, "hao3"),
        post.Interval(0.9, 1.0, "。"),
    ])
    raw_text = post.Tier("raw_text", 0.0, 1.0, [
        post.Interval(0.0, 1.0, "大家？<BREATHING>好。"),
    ])
    pinyin = post.Tier("pinyin", 0.0, 1.0, [
        post.Interval(0.0, 1.0, "da4 ？ <BREATHING> hao3 。"),
    ])
    reasons, details = post._publication_contract_audit(
        words, None, None, None, "大家？好。", None, None, False,
        raw_text_tier=raw_text, pinyin_tier=pinyin)
    assert "raw_text_punctuation_sequence_mismatch" not in reasons
    assert "pinyin_punctuation_sequence_mismatch" not in reasons


def test_fallback_punctuation_commit_precedes_resolver_and_freeze():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.40, "ni3"),
        post.Interval(0.40, 0.50, "<sp0>"),
        post.Interval(0.50, 0.90, "hao3"),
    ])
    surface = post._fallback_punctuation_surface_ledger("你，好")
    words, _ = post._inject_fallback_punctuation_gaps(
        words, None, [], source_surface_ledger=surface)
    grid = post.TextGrid(0.0, 1.0, [
        post.Tier("raw_text", 0.0, 1.0,
                  [post.Interval(0.0, 1.0, "stale")]),
        post.Tier("pinyin", 0.0, 1.0,
                  [post.Interval(0.0, 1.0, "stale")]),
        post.Tier("hanzi", 0.0, 1.0, []), words,
    ])
    decisions = post.resolve_visual_short_silence_merges(
        grid, np.zeros(16000, dtype="float32"), 16000,
        SimpleNamespace(merge_silence=False, merge_max_sil_sec=0.001,
                        merge_energy_threshold=0.5), {})
    assert decisions == []
    assert [iv.text for iv in words.intervals] == ["ni3", "，", "hao3"]
    frozen, reasons = post._freeze_processed_geometry(grid)
    assert frozen is words and reasons == []
    post._rebuild_derived_from_frozen_words(
        grid, {}, {}, "你，好", fallback_surface_ledger=surface)
    assert grid.tiers[0].intervals[0].text == "<sp1>你，好"

def _published_52697() -> tuple[post.Tier, post.Tier]:
    fixture = _fixture_52697()
    words = _clone_tier(fixture.words)
    return words, post._build_hanzi_tier(
        words, fixture.reference, reference_authoritative=True)


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("coverage_hole", "words_coverage_hole"),
        ("overlap", "words_overlap"),
        ("split_silence_labels", "silence_label_split"),
        ("strict_interior_sp", "strict_interior_sp"),
    ],
)
def test_geometry_tamper_variants_are_rejected(tamper: str, expected: str):
    words, hanzi = _published_52697()
    if tamper == "coverage_hole":
        words.intervals[1].xmax = 0.70
    elif tamper == "overlap":
        words.intervals[2].xmin = 0.70
    elif tamper == "split_silence_labels":
        words.intervals[-1].text = "<sp2>"
    elif tamper == "strict_interior_sp":
        words.intervals[1].text = "<sp2>"
        hanzi.intervals[1].text = "<sp2>"

    reasons = _geometry_reasons(words, hanzi)
    assert expected in reasons, (tamper, reasons)


def test_fallback_projection_maps_source_mark_after_omitted_cjk():
    words = post.Tier("words", 0.0, 1.5, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.8, "hao3"),
        post.Interval(0.8, 1.0, "<sp2>"),
        post.Interval(1.0, 1.5, "ni3"),
    ])
    ctc = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.4},
        {"type": "word", "word": "hao3", "start_s": 0.4, "end_s": 0.8},
        {"type": "word", "word": "ni3", "start_s": 1.0, "end_s": 1.5},
    ]
    projection = post._fallback_punctuation_projection(
        "你好吗，你", words, ctc)
    assert projection["safe"] is True
    assert projection["alignment"]["actual_to_source"] == {
        0: 0, 1: 1, 2: 3}
    assert [(item["label"], item["source_boundary"],
             item["final_boundary"]) for item in projection["entries"]] == [
                 ("，", 3, 2)]

    updated, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=post._fallback_punctuation_surface_ledger(
            "你好吗，你"), ctc_tokens=ctc,
        punctuation_projection=projection)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.4, "ni3"), (0.4, 0.8, "hao3"),
        (0.8, 1.0, "，"), (1.0, 1.5, "ni3")]


def _laria_r9_projection_fixture(stem: str):
    """Return the real 00394/00395 lexical topology with local gap owners."""
    fixtures = {
        "LAria_00394": (
            "Surprise-wa，今天大赛也太给力了吧！狗在耍可爱！哎我都不好意思了！"
            "看到你们这么热情，我超级开心的！",
            11.202,
            [
                (0.51, 0.99, "<SURPRISE-WA>"),
                (0.99, 1.11, "jin1"), (1.11, 1.29, "tian1"),
                (1.29, 1.44, "da4"), (1.44, 1.60, "sai4"),
                (1.60, 1.68, "ye3"), (1.68, 1.93, "tai4"),
                (1.93, 2.13, "gei3"), (2.13, 2.31, "li4"),
                (2.31, 2.46, "le5"), (2.46, 2.79, "ba5"),
                (3.30, 3.45, "gou3"), (3.45, 3.63, "zai4"),
                (3.63, 3.99, "shua3"), (3.99, 4.23, "ke3"),
                (4.23, 4.59, "RIA"),
                (5.31, 5.43, "wo3"), (5.43, 5.55, "dou1"),
                (5.55, 5.61, "bu4"), (5.61, 5.75, "hao3"),
                (5.75, 5.91, "yi4"), (5.91, 6.11, "si1"),
                (6.11, 6.37, "le5"),
                (6.96, 7.05, "kan4"), (7.05, 7.17, "dao4"),
                (7.17, 7.29, "ni3"), (7.29, 7.41, "men5"),
                (7.41, 7.53, "zhe4"), (7.53, 7.60, "me5"),
                (7.60, 7.86, "re4"), (7.86, 8.31, "qing2"),
                (9.025, 9.09, "wo3"), (9.09, 9.45, "chao1"),
                (9.45, 9.63, "ji2"), (9.63, 9.93, "kai1"),
                (9.93, 10.23, "xin1"), (10.23, 10.74, "de5"),
            ],
        ),
        "LAria_00395": (
            "SURPRISE-WA ！怎么突然来了这么多人啊！欢迎大家， BREATHING "
            "欢迎所有新来的朋友们，谢谢大家的礼物，还有弹幕！",
            10.094,
            [
                (1.35, 1.59, "<SURPRISE-WA>"),
                (2.07, 2.13, "zen3"), (2.13, 2.30, "me5"),
                (2.30, 2.49, "tu1"), (2.49, 2.67, "ran2"),
                (2.67, 2.79, "lai2"), (2.79, 2.97, "le5"),
                (2.97, 3.09, "zhe4"), (3.09, 3.16, "me5"),
                (3.16, 3.33, "duo1"), (3.33, 3.51, "ren2"),
                (3.51, 3.69, "RIA"),
                (4.12, 4.23, "huan1"), (4.23, 4.375, "ying2"),
                (4.375, 4.52, "da4"), (4.52, 4.83, "jia1"),
                (4.95, 5.25, "<BREATHING>"),
                (5.25, 5.43, "huan1"), (5.43, 5.61, "ying2"),
                (5.61, 5.73, "suo3"), (5.73, 5.90, "you3"),
                (5.90, 6.08, "xin1"), (6.08, 6.21, "lai2"),
                (6.21, 6.39, "de5"), (6.39, 6.51, "peng2"),
                (6.51, 6.63, "you3"), (6.63, 6.79, "men5"),
                (7.35, 7.53, "xie4"), (7.53, 7.65, "xie4"),
                (7.65, 7.83, "da4"), (7.83, 8.01, "jia1"),
                (8.01, 8.19, "de5"), (8.19, 8.37, "li3"),
                (8.37, 8.67, "wu4"),
                (8.73, 8.85, "hai2"), (8.85, 9.09, "you3"),
                (9.09, 9.21, "dan4"), (9.21, 9.60, "mu4"),
            ],
        ),
    }
    source, xmax, lexical = fixtures[stem]
    intervals = []
    cursor = 0.0
    for start, end, label in lexical:
        if start > cursor + EPS:
            intervals.append(post.Interval(cursor, start, "<sp2>"))
        intervals.append(post.Interval(start, end, label))
        cursor = end
    if xmax > cursor + EPS:
        intervals.append(post.Interval(cursor, xmax, "<sp2>"))
    words = post.Tier("words", 0.0, xmax, intervals)
    ctc = [
        {"type": "word", "word": label.strip("<>"),
         "start_s": start, "end_s": end}
        for start, end, label in lexical
    ]
    return source, words, ctc


def _pre_display_nvv_tier(words: post.Tier) -> post.Tier:
    """Undo only the known NVV wrapper applied after projection creation."""
    tier = _clone_tier(words)
    for interval in tier.intervals:
        if post._canonical_nvv_identity(interval.text) is not None:
            interval.text = interval.text.strip().strip("<>")
    return tier


def test_laria_00394_r9_projection_with_ria_is_exact_and_publishable():
    source_text, words, ctc = _laria_r9_projection_fixture("LAria_00394")
    pre_display_words = _pre_display_nvv_tier(words)
    projection = post._fallback_punctuation_projection(
        source_text, pre_display_words, ctc)

    assert projection["safe"] is True
    assert projection["reasons"] == []
    assert [(item["label"], item["final_boundary"])
            for item in projection["entries"]] == [
                ("，", 1), ("！", 11), ("！", 16),
                ("！", 23), ("，", 31), ("！", 37),
            ]
    assert projection["unanchored_final_lexical_ordinals"] == [15]
    assert not any(item["final_text"] == "RIA"
                   for item in projection["mapped"])
    assert projection["mapped"][0]["final_text"] == "SURPRISE-WA"
    omitted_interval = projection["entries"][2]
    assert omitted_interval["candidate_final_boundaries"] == [15, 16]
    assert omitted_interval["positive_owner_candidates"] == [16]

    valid, authority = post._validate_fallback_punctuation_projection(
        projection, source_text, words, ctc)
    assert valid is True
    assert authority["status"] == "verified"
    assert authority["reasons"] == []

    published, _ = post._inject_punctuation(
        words, None, [], reference_authoritative=False,
        source_surface_ledger=post._fallback_punctuation_surface_ledger(
            source_text), ctc_tokens=ctc,
        punctuation_projection=projection)
    observed = []
    lexical_before = 0
    for interval in published.intervals:
        if (interval.text.strip() and not post.is_silence(interval.text)
                and not post.is_punct(interval.text)):
            lexical_before += 1
        elif post.is_punct(interval.text) and not post.is_silence(interval.text):
            observed.append((interval.text, lexical_before))
    assert observed == [
        ("，", 1), ("！", 11), ("！", 16),
        ("！", 23), ("，", 31), ("！", 37),
    ]
    reasons, details = post._publication_contract_audit(
        published, None, None, None, source_text, None, ctc, False,
        reference_mode="fallback",
        fallback_surface_ledger=post._fallback_punctuation_surface_ledger(
            source_text),
        fallback_punctuation_projection=projection)
    assert "fallback_punctuation_ownership_mismatch" not in reasons
    assert details["fallback_punctuation_projection_authority"]["status"] == \
        "verified"
    assert "fallback_cjk_cross_kind_owner_unproven" in reasons
    cross_kind = details["fallback_cjk_cross_kind_owner_unproven"]
    assert cross_kind["buckets"]
    assert any(owner["final_text"] == "RIA"
               for bucket in cross_kind["buckets"]
               for owner in bucket["final_unanchored_owners"])


def test_laria_00395_r9_ria_boundary_and_boundary_zero_are_separate():
    source_text, words, ctc = _laria_r9_projection_fixture("LAria_00395")
    pre_display_words = _pre_display_nvv_tier(words)
    projection = post._fallback_punctuation_projection(
        source_text, pre_display_words, ctc)

    assert projection["safe"] is True
    assert [(item["label"], item["final_boundary"])
            for item in projection["entries"][:2]] == [("！", 1), ("！", 12)]
    assert projection["entries"][1]["candidate_final_boundaries"] == [11, 12]
    assert projection["entries"][1]["positive_owner_candidates"] == [12]
    assert projection["unanchored_final_lexical_ordinals"] == [11]
    assert not any(item["final_text"] == "RIA"
                   for item in projection["mapped"])
    assert projection["mapped"][0]["final_text"] == "SURPRISE-WA"
    valid, authority = post._validate_fallback_punctuation_projection(
        projection, source_text, words, ctc)
    assert valid is True
    assert authority["status"] == "verified"
    assert authority["reasons"] == []

    raw_boundary_zero = [{
        "type": "punct", "word": "！", "start_s": 0.87, "end_s": 0.93,
        "left_lexical_ordinal": None, "right_lexical_ordinal": 0,
    }]
    valid_projected_reference = {
        "type": "punct", "word": "！", "start_s": 1.80, "end_s": 1.90,
        "left_lexical_ordinal": 0, "right_lexical_ordinal": 1,
    }
    resolver_words = _clone_tier(words)
    grid = post.TextGrid(0.0, words.xmax, [resolver_words])
    decisions = post._resolve_visual_short_silence_merges(
        grid, np.zeros(int(words.xmax * 16000), dtype=np.float32), 16000,
        SimpleNamespace(merge_silence=True, merge_max_sil_sec=0.5,
                        merge_energy_threshold=0.5),
        ctc_tokens=ctc,
        reference_punct_entries=[raw_boundary_zero[0],
                                 valid_projected_reference],
        fallback_punctuation_projection=projection)
    leading = next(item for item in decisions
                   if item["left_lexical_ordinal"] is None
                   and item["right_lexical_ordinal"] == 0)
    assert leading["punctuation_owner"] is None
    assert leading["punctuation_gap_restore"] is False
    valid_reference = next(item for item in decisions
                           if item["left_lexical_ordinal"] == 0
                           and item["right_lexical_ordinal"] == 1)
    assert valid_reference["punctuation_owner"] == \
        "reference_punctuation_owner"
    assert valid_reference["punctuation_gap_restore"] is True
    resolver_observed = []
    lexical_before = 0
    for interval in resolver_words.intervals:
        if (interval.text.strip() and not post.is_silence(interval.text)
                and not post.is_punct(interval.text)):
            lexical_before += 1
        elif post.is_punct(interval.text) and not post.is_silence(interval.text):
            resolver_observed.append((interval.text, lexical_before))
    assert ("！", 1) in resolver_observed
    assert not any(boundary == 0 for _, boundary in resolver_observed)

    raw_boundary_zero[0].update({
        "schema": post.PUNCTUATION_EVIDENCE_SCHEMA,
        "raw_start_s": 0.87, "raw_end_s": 0.93,
        "candidate_id": "00395-leading-boundary-zero", "source": "ctc",
    })
    published, _ = post._inject_punctuation(
        _clone_tier(words), None, raw_boundary_zero,
        reference_authoritative=False,
        source_surface_ledger=post._fallback_punctuation_surface_ledger(
            source_text), ctc_tokens=ctc,
        punctuation_projection=projection)
    observed = []
    lexical_before = 0
    for interval in published.intervals:
        if (interval.text.strip() and not post.is_silence(interval.text)
                and not post.is_punct(interval.text)):
            lexical_before += 1
        elif post.is_punct(interval.text) and not post.is_silence(interval.text):
            observed.append((interval.text, lexical_before))
    assert observed == [
        ("！", 1), ("！", 12), ("，", 16),
        ("，", 27), ("，", 34), ("！", 38),
    ]
    assert not any(boundary == 0 for _, boundary in observed)

    reasons, details = post._publication_contract_audit(
        published, None, None, None, source_text, None, ctc, False,
        reference_mode="fallback",
        fallback_surface_ledger=post._fallback_punctuation_surface_ledger(
            source_text),
        fallback_punctuation_projection=projection)
    assert "fallback_punctuation_ownership_mismatch" not in reasons
    assert details["fallback_punctuation_projection_authority"]["status"] == \
        "verified"
    assert "fallback_cjk_cross_kind_owner_unproven" in reasons
    cross_kind = details["fallback_cjk_cross_kind_owner_unproven"]
    assert any(owner["final_text"] == "RIA"
               for bucket in cross_kind["buckets"]
               for owner in bucket["final_unanchored_owners"])


def _fallback_ria_topology(raw_text: str, final_labels: list[str]):
    intervals = []
    ctc = []
    source_words = []
    for index, label in enumerate(final_labels):
        start = index * 0.2
        end = start + 0.2
        intervals.append(post.Interval(start, end, label))
        ctc.append({"type": "word", "word": label,
                    "start_s": start, "end_s": end})
        source_words.append({"ordinal": index, "start": start, "end": end,
                             "text": label})
    return (post.Tier("words", 0.0, len(final_labels) * 0.2, intervals),
            ctc, source_words)


def test_laria_00014_omitted_cjk_and_unanchored_ria_fail_closed():
    words, ctc, source_words = _fallback_ria_topology(
        "破费了啊", ["po4", "fei4", "le5", "RIA"])
    projection = post._fallback_punctuation_projection("破费了啊", words, ctc)
    safe, details = post._fallback_cjk_cross_kind_owner_audit(
        "破费了啊", words, projection, ctc)
    assert safe is False
    assert details["buckets"]
    assert details["buckets"][0]["source_cjk_omitted"][-1]["source_text"] == "啊"
    assert details["buckets"][0]["final_unanchored_owners"][-1]["final_text"] == "RIA"

    reasons, audit_details = post._publication_contract_audit(
        words, None, None, None, "破费了啊", source_words, ctc, False,
        reference_mode="fallback",
        fallback_surface_ledger=post._fallback_punctuation_surface_ledger(
            "破费了啊"),
        fallback_punctuation_projection=projection)
    assert "fallback_cjk_cross_kind_owner_unproven" in reasons
    assert audit_details["fallback_cjk_cross_kind_owner"]["status"] == "rejected"


def test_corrected_a5_topology_has_no_cross_kind_owner_veto():
    words, ctc, _source_words = _fallback_ria_topology(
        "破费了啊", ["po4", "fei4", "le5", "a5"])
    projection = post._fallback_punctuation_projection("破费了啊", words, ctc)
    safe, details = post._fallback_cjk_cross_kind_owner_audit(
        "破费了啊", words, projection, ctc)
    assert safe is True
    assert details["buckets"] == []
    assert projection["omitted_source_lexical_ordinals"] == []
    assert projection["unanchored_final_lexical_ordinals"] == []


def test_native_raw_english_ria_remains_allowed_by_cross_kind_veto():
    words, ctc, _source_words = _fallback_ria_topology(
        "你RIA好", ["ni3", "RIA", "hao3"])
    projection = post._fallback_punctuation_projection("你RIA好", words, ctc)
    safe, details = post._fallback_cjk_cross_kind_owner_audit(
        "你RIA好", words, projection, ctc)
    assert safe is True
    assert details["buckets"] == []
    assert any(anchor["source_text"] == "RIA"
               for anchor in projection["mapped"])


def test_cross_kind_veto_rejects_non_monotonic_mapped_anchors(monkeypatch):
    words, ctc, _source_words = _fallback_ria_topology(
        "你好吗", ["ni3", "hao3", "ma5"])
    malformed = {
        "mapped": [
            {"source_lexical_ordinal": 0, "final_lexical_ordinal": 1,
             "anchor_kind": "cjk", "source_text": "你",
             "final_text": "hao3"},
            {"source_lexical_ordinal": 1, "final_lexical_ordinal": 0,
             "anchor_kind": "cjk", "source_text": "好",
             "final_text": "ni3"},
        ],
    }
    monkeypatch.setattr(
        post, "_fallback_punctuation_projection",
        lambda *_args, **_kwargs: malformed)

    safe, details = post._fallback_cjk_cross_kind_owner_audit(
        "你好吗", words, None, ctc)
    assert safe is False
    assert details["status"] == "rejected"
    assert details["reason"] == "mapped_anchors_not_strictly_monotonic"
    assert details["independent_identity_proof"] is False


def _laria_00234_repeated_breathing_fixture():
    source = (
        "Breathing，谢谢大家， BREATHING 看了这么多弹幕我好开心啊，"
        "看来…你们跟我一样都特别喜欢刚的我。"
    )
    surface = post._fallback_punctuation_surface_ledger(source)
    punctuation_boundaries = {
        item["lexical_boundary"] for item in surface["punctuation"]}
    source_units = [
        unit for unit in post._extract_word_chars(source)
        if post.is_word_like(unit)]
    intervals = [post.Interval(0.0, 0.33, "<sp2>")]
    ctc = []
    cursor = 0.33
    for ordinal, unit in enumerate(source_units):
        label = (
            post.lazy_pinyin(
                unit, style=post.Style.TONE3,
                neutral_tone_with_five=True)[0]
            if post.is_cjk(unit) else "BREATHING"
        )
        end = cursor + 0.08
        intervals.append(post.Interval(cursor, end, label))
        ctc.append({"type": "word", "word": label,
                    "start_s": cursor, "end_s": end})
        cursor = end
        boundary = ordinal + 1
        # Match the r10 failure topology: the leading silence is the only
        # positive owner among pre-fix candidates [0, 1].  Other source marks
        # retain explicit local gaps so injection itself remains realistic.
        if boundary in punctuation_boundaries and boundary != 1:
            intervals.append(post.Interval(cursor, cursor + 0.04, "<sp0>"))
            cursor += 0.04
    return source, post.Tier("words", 0.0, cursor, intervals), ctc


def test_laria_00234_repeated_breathing_is_bounded_by_cjk_anchors():
    source, pre_display_words, ctc = \
        _laria_00234_repeated_breathing_fixture()
    projection = post._fallback_punctuation_projection(
        source, pre_display_words, ctc)

    assert projection["safe"] is True
    breathing_anchors = [
        item for item in projection["mapped"]
        if post._lexical_identity(item["source_text"]) == "<BREATHING>"]
    assert [(item["source_lexical_ordinal"],
             item["final_lexical_ordinal"], item["anchor_kind"])
            for item in breathing_anchors] == [
                (0, 0, "bounded_repeated_non_cjk_identity"),
                (5, 5, "bounded_repeated_non_cjk_identity"),
            ]
    first = projection["entries"][0]
    assert (first["label"], first["source_boundary"],
            first["final_boundary"]) == ("，", 1, 1)
    assert first["candidate_final_boundaries"] == [1]

    display_words = _clone_tier(pre_display_words)
    for interval in display_words.intervals:
        if interval.text == "BREATHING":
            interval.text = "<BREATHING>"
    published, _ = post._inject_punctuation(
        display_words, None, [], reference_authoritative=False,
        source_surface_ledger=post._fallback_punctuation_surface_ledger(
            source), ctc_tokens=ctc,
        punctuation_projection=projection)
    valid, authority = post._validate_fallback_punctuation_projection(
        projection, source, published, ctc)
    assert valid is True
    assert authority["status"] == "verified"
    assert authority["reasons"] == []

    observed = []
    lexical_before = 0
    for interval in published.intervals:
        if (interval.text.strip() and not post.is_silence(interval.text)
                and not post.is_punct(interval.text)):
            lexical_before += 1
        elif post.is_punct(interval.text) and not post.is_silence(interval.text):
            observed.append((interval.text, lexical_before))
    assert observed == [
        ("，", 1), ("，", 5), ("，", 18), ("…", 20), ("。", 34)]
    assert observed[0] == ("，", 1)
    assert not any(boundary == 0 for _, boundary in observed)
    reasons, details = post._publication_contract_audit(
        published, None, None, None, source, None, ctc, False,
        reference_mode="fallback",
        fallback_surface_ledger=post._fallback_punctuation_surface_ledger(
            source),
        fallback_punctuation_projection=projection)
    assert "fallback_punctuation_ownership_mismatch" not in reasons
    assert details["fallback_punctuation_projection_authority"]["status"] == \
        "verified"


def test_repeated_identity_inside_one_anchor_interval_fails_closed():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "ni3"),
        post.Interval(0.2, 0.4, "BREATHING"),
        post.Interval(0.4, 0.6, "BREATHING"),
        post.Interval(0.6, 0.8, "<sp1>"),
        post.Interval(0.8, 1.0, "hao3"),
    ])
    ctc = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.2},
        {"type": "word", "word": "BREATHING",
         "start_s": 0.2, "end_s": 0.4},
        {"type": "word", "word": "BREATHING",
         "start_s": 0.4, "end_s": 0.6},
        {"type": "word", "word": "hao3", "start_s": 0.8, "end_s": 1.0},
    ]
    projection = post._fallback_punctuation_projection(
        "你 BREATHING BREATHING，好", words, ctc)

    assert projection["safe"] is False
    assert "bounded_repeated_non_cjk_identity_ambiguous" in \
        projection["reasons"]
    assert not any(item["anchor_kind"] ==
                   "bounded_repeated_non_cjk_identity"
                   for item in projection["mapped"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_lexical_ordinal", 1),
        ("final_lexical_ordinal", 1),
        ("anchor_kind", "forged_anchor"),
        ("source_text", "SURPRISE-OH"),
        ("final_text", "SURPRISE-OH"),
    ],
)
def test_display_invariant_projection_keeps_mapped_authority_strict(
        field: str, replacement: object):
    source_text, words, ctc = _laria_r9_projection_fixture("LAria_00394")
    projection = post._fallback_punctuation_projection(
        source_text, _pre_display_nvv_tier(words), ctc)
    tampered = deepcopy(projection)
    tampered["mapped"][0][field] = replacement
    tampered["digest"] = post._evidence_digest(
        {key: value for key, value in tampered.items() if key != "digest"})

    valid, details = post._validate_fallback_punctuation_projection(
        tampered, source_text, words, ctc)
    assert valid is False
    assert "mapped_mismatch" in details["reasons"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("final_boundary", 2), ("label", "？")],
)
def test_display_invariant_projection_keeps_entry_authority_strict(
        field: str, replacement: object):
    source_text, words, ctc = _laria_r9_projection_fixture("LAria_00394")
    projection = post._fallback_punctuation_projection(
        source_text, _pre_display_nvv_tier(words), ctc)
    tampered = deepcopy(projection)
    tampered["entries"][0][field] = replacement
    tampered["digest"] = post._evidence_digest(
        {key: value for key, value in tampered.items() if key != "digest"})

    valid, details = post._validate_fallback_punctuation_projection(
        tampered, source_text, words, ctc)
    assert valid is False
    assert "entries_mismatch" in details["reasons"]


def test_fallback_projection_rejects_two_positive_owners_in_anchor_interval():
    words = post.Tier("words", 0.0, 1.4, [
        post.Interval(0.0, 0.3, "ni3"),
        post.Interval(0.3, 0.5, "<sp2>"),
        post.Interval(0.5, 0.8, "RIA"),
        post.Interval(0.8, 1.0, "<sp2>"),
        post.Interval(1.0, 1.4, "ni3"),
    ])
    projection = post._fallback_punctuation_projection(
        "你好吗，你", words, [
            {"type": "word", "word": "ni3",
             "start_s": 0.0, "end_s": 0.3},
            {"type": "word", "word": "RIA",
             "start_s": 0.5, "end_s": 0.8},
            {"type": "word", "word": "ni3",
             "start_s": 1.0, "end_s": 1.4},
        ])
    assert projection["safe"] is False
    assert "punctuation_boundary_owner_candidate_not_unique" in \
        projection["reasons"]


def test_fallback_projection_rejects_ambiguous_or_tampered_authority():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.6, "<sp2>"),
        post.Interval(0.6, 1.0, "hao3"),
    ])
    ctc = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.4},
        {"type": "word", "word": "hao3", "start_s": 0.6, "end_s": 1.0},
    ]
    projection = post._fallback_punctuation_projection("你好吗", words, ctc)
    projection["safe"] = True
    valid, details = post._validate_fallback_punctuation_projection(
        projection, "你好吗", words, ctc)
    assert valid is False
    assert "digest_mismatch" in details["reasons"]


def test_fallback_projection_rejects_raw_candidate_neighbor_mismatch():
    words = post.Tier("words", 0.0, 1.5, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.8, "hao3"),
        post.Interval(0.8, 1.0, "<sp2>"),
        post.Interval(1.0, 1.5, "ni3"),
    ])
    ctc = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.4},
        {"type": "word", "word": "hao3", "start_s": 0.4, "end_s": 0.8},
        {"type": "word", "word": "ni3", "start_s": 1.0, "end_s": 1.5},
    ]
    projection = post._fallback_punctuation_projection(
        "你好吗，你", words, ctc)
    wrong = [{"schema": post.PUNCTUATION_EVIDENCE_SCHEMA, "word": "，",
              "raw_start_s": 0.8, "raw_end_s": 0.9,
              "start_s": 0.8, "end_s": 0.9, "candidate_id": "wrong",
              "source": "ctc", "left_lexical_ordinal": None,
              "right_lexical_ordinal": 0}]
    updated, _ = post._inject_punctuation(
        words, None, wrong, reference_authoritative=False,
        source_surface_ledger=post._fallback_punctuation_surface_ledger(
            "你好吗，你"), ctc_tokens=ctc,
        punctuation_projection=projection)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in updated.intervals] == [
        (0.0, 0.4, "ni3"), (0.4, 0.8, "hao3"),
        (0.8, 1.0, "，"), (1.0, 1.5, "ni3")]
    assert updated._punctuation_evidence_ledger["rejected"][0]["reason"] == \
        "source_punctuation_boundary_missing"


def test_fallback_projection_authority_rejects_boundary_zero_in_visual_resolver():
    words = post.Tier("words", 0.0, 1.5, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.8, "hao3"),
        post.Interval(0.8, 1.0, "<sp2>"),
        post.Interval(1.0, 1.5, "ni3"),
    ])
    ctc_words = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.4},
        {"type": "word", "word": "hao3", "start_s": 0.4, "end_s": 0.8},
        {"type": "word", "word": "ni3", "start_s": 1.0, "end_s": 1.5},
    ]
    projection = post._fallback_punctuation_projection(
        "你好吗，你", words, ctc_words)
    wrong_candidate = [{
        "type": "punct", "word": "！", "start_s": 0.8, "end_s": 0.9,
        "left_lexical_ordinal": None, "right_lexical_ordinal": 0,
    }]
    grid = post.TextGrid(0.0, 1.5, [words])
    decision = post._resolve_visual_short_silence_merges(
        grid, np.zeros(24000, dtype=np.float32), 16000,
        SimpleNamespace(merge_silence=True, merge_max_sil_sec=0.5,
                        merge_energy_threshold=0.5),
        ctc_tokens=wrong_candidate,
        fallback_punctuation_projection=projection)[0]
    assert decision["punctuation_owner"] is None
    assert decision["punctuation_gap_restore"] is False
    assert not any(iv.text in {"，", "！"} for iv in words.intervals)


def test_publication_audit_uses_projected_boundary_but_source_surface_labels():
    words = post.Tier("words", 0.0, 1.5, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.8, "hao3"),
        post.Interval(0.8, 1.0, "，"),
        post.Interval(1.0, 1.5, "ni3"),
    ])
    ctc = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.4},
        {"type": "word", "word": "hao3", "start_s": 0.4, "end_s": 0.8},
        {"type": "word", "word": "ni3", "start_s": 1.0, "end_s": 1.5},
    ]
    source = post._fallback_punctuation_surface_ledger("你好吗，你")
    projection = post._fallback_punctuation_projection(
        source["source_text"], words, ctc)
    reasons, details = post._publication_contract_audit(
        words, None, None, None, source["source_text"], None, ctc, False,
        reference_mode="fallback", fallback_surface_ledger=source,
        fallback_punctuation_projection=projection)
    assert "fallback_punctuation_ownership_mismatch" not in reasons
    projected = details["fallback_punctuation_projection"]["expected"]
    assert projected[0]["source_boundary"] == 3
    assert projected[0]["boundary"] == 2
