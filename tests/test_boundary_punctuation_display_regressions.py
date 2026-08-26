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

import pytest

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
