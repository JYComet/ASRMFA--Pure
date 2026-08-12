from types import SimpleNamespace

from scripts.postprocess_textgrids import (
    Interval,
    Tier,
    TextGrid,
    _duration_ticks,
    _find_internal_pp_gaps,
    _fix_overlapping_boundaries,
    absorb_silence_into_punct,
    handle_unexpected_silences,
    _repair_authority_punctuation_geometry,
    _repair_reference_authority_colocated_silence,
    _restore_reference_punctuation,
    _strict_semantic_tokens,
    detect_issues,
)


def _qc_args(**overrides):
    values = dict(
        filter_min_word_dur_sec=0.0,
        filter_min_word_sec=0.0,
        filter_min_phone_coverage=0.0,
        filter_edge_gap_sec=99.0,
        filter_long_word_sec=99.0,
        filter_flank_silence_sec=0.0,
        filter_word_energy_ratio=0.0,
        enable_word_in_silence_filter=False,
        filter_short_phone=False,
        filter_short_phone_sec=0.015,
        filter_long_consonant_sec=999.0,
        filter_long_vowel_sec=999.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_duration_ticks_are_exact_at_30ms_boundary():
    assert _duration_ticks(0.0, 0.030000) == 30_000
    assert _duration_ticks(0.0, 0.029999) == 29_999


def test_late_mild_overlap_is_repaired_but_large_overlap_is_retained():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.30, "er2"), Interval(0.29, 0.50, "hui4"),
        Interval(0.46, 0.80, "lai2"),
    ])
    assert _fix_overlapping_boundaries(words) == 1
    assert words.intervals[0].xmax == words.intervals[1].xmin
    assert words.intervals[1].xmax - words.intervals[2].xmin > 0.03


def test_short_word_uses_strict_integer_tick_boundary_and_reports_details():
    words = Tier("words", 0.0, 0.06, [
        Interval(0.0, 0.030000, "ni3"),
        Interval(0.030001, 0.059, "hao3"),
    ])
    phones = Tier("pinyin_phones", 0.0, 0.06, [
        Interval(0.0, 0.030000, "i3"),
        Interval(0.030001, 0.059, "ao3"),
    ])
    assert _duration_ticks(words.intervals[0].xmin, words.intervals[0].xmax) == 30_000
    words.intervals[0] = Interval(0.0, 0.029999, "ni3")
    assert _duration_ticks(words.intervals[0].xmin, words.intervals[0].xmax) == 29_999


def test_gap_diagnostics_include_word_and_phone_root_cause():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.45, 1.0, "hao3"),
    ])
    phones = Tier("pinyin_phones", 0.0, 1.0, [
        Interval(0.0, 0.1, "n"), Interval(0.13, 0.4, "i3"),
        Interval(0.45, 0.6, "h"), Interval(0.6, 1.0, "ao3"),
    ])
    args = _qc_args(filter_suspicious=True)
    issues = detect_issues(TextGrid(0.0, 1.0, [words, phones]), args)
    # The inter-word pause is not an internal pinyin gap; the within-word
    # 30-ms hole is retained with its owner and neighbouring phone labels.
    gaps = _find_internal_pp_gaps(phones, words)
    assert len(gaps) == 1
    assert gaps[0]["word"] == "ni3"
    assert gaps[0]["duration_us"] == 30_000
    assert all(row["rule"] != "pp_tier_gaps" for row in issues)


def test_authority_punctuation_absorbs_only_colocated_silence():
    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.3, "ni3"),
        Interval(0.3, 0.31, "，"),
        Interval(0.31, 0.5, "<sp1>"),
        Interval(0.5, 0.8, "hao3"),
    ])
    entries = [{"word": "，", "start_s": 0.31, "end_s": 0.5}]
    assert _restore_reference_punctuation(words, "你，好", entries) == 1
    punct = next(iv for iv in words.intervals if iv.text == "，")
    assert (punct.xmin, punct.xmax) == (0.3, 0.5)
    assert not any(iv.text == "<sp1>" for iv in words.intervals)

    negative = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.3, "ni3"),
        Interval(0.3, 0.31, "，"),
        Interval(0.31, 0.5, "<sp1>"),
        Interval(0.5, 0.8, "hao3"),
    ])
    assert _repair_authority_punctuation_geometry(negative, "你，好",
                                                   [{"word": "，", "start_s": 0.2,
                                                     "end_s": 0.25}]) == 0
    assert any(iv.text == "<sp1>" for iv in negative.intervals)


def test_literal_sp2_adjacent_tilde_is_punctuation_owned():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "你"), Interval(0.3, 0.35, "～"),
        Interval(0.35, 0.70, "<sp2>"), Interval(0.70, 1.0, "好")])
    phones = Tier("phones", 0.0, 1.0, [Interval(0.35, 0.70, "<sp2>")])
    pp = Tier("pinyin_phones", 0.0, 1.0, [Interval(0.35, 0.70, "<sp2>")])
    tg = TextGrid(0.0, 1.0, [words, phones, pp])
    assert handle_unexpected_silences(tg, "ni3 ～<sp2> hao3") == []
    absorb_silence_into_punct(tg)
    assert not any(iv.text == "<sp2>" for iv in words.intervals)
    tilde = next(iv for iv in words.intervals if iv.text == "～")
    assert tilde.xmax == 0.70


def test_literal_sp2_semantics_are_nonlexical_but_foreign_tokens_still_fail():
    # Canonical silence labels must not alter the lexical sequence, even when
    # attached directly to punctuation.  Unknown lexical text remains visible
    # to the strict semantic contract and therefore does not compare equal.
    reference = _strict_semantic_tokens("你 ～ 好")
    assert _strict_semantic_tokens("你 ～<sp2> 好") == reference
    assert _strict_semantic_tokens("你 ～<sp2> BADTOKEN 好") != reference


def test_authority_geometry_removes_exactly_colocated_silence():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.30, "ni3"),
        Interval(0.30, 0.52, "～"),
        # Deliberately identical geometry to the punctuation: this used to
        # trigger an early return because punctuation bounds needed no update.
        Interval(0.30, 0.52, "<sp2>"),
        Interval(0.52, 1.0, "hao3"),
    ])
    changed = _repair_authority_punctuation_geometry(
        words, "你～好", [{"word": "～", "start_s": 0.30, "end_s": 0.52}])
    assert changed == 1
    assert not any(iv.text == "<sp2>" for iv in words.intervals)
    tilde = next(iv for iv in words.intervals if iv.text == "～")
    assert (tilde.xmin, tilde.xmax) == (0.30, 0.52)


def test_reference_only_tilde_absorbs_actual_shaped_exact_sp0():
    # 007089 shape: the reference owns ``～``, but CTC punctuation does not
    # provide an anchor.  Reference restoration creates the punctuation over
    # the exact <sp0> interval, which can then be safely deduplicated.
    words = Tier("words", 0.0, 9.0, [
        Interval(0.0, 7.91, "ni3"),
        Interval(7.91, 8.52, "<sp0>"),
        Interval(8.52, 9.0, "hao3"),
    ])
    assert _restore_reference_punctuation(words, "你～好", []) == 2
    tilde = next(iv for iv in words.intervals if iv.text == "～")
    assert (tilde.xmin, tilde.xmax) == (7.91, 8.52)
    assert not any(iv.text == "<sp0>" for iv in words.intervals)


def test_reference_only_fallback_preserves_partial_and_foreign_silence():
    partial = Tier("words", 0.0, 9.0, [
        Interval(0.0, 7.91, "ni3"), Interval(7.91, 8.52, "～"),
        Interval(7.91, 8.53, "<sp0>"), Interval(8.52, 9.0, "hao3"),
    ])
    assert _repair_reference_authority_colocated_silence(partial, "你～好") == 0
    assert any(iv.text == "<sp0>" for iv in partial.intervals)

    foreign = Tier("words", 0.0, 9.0, [
        Interval(0.0, 7.91, "ni3"), Interval(7.91, 8.52, "～"),
        Interval(7.91, 8.52, "<sp0>"), Interval(8.52, 9.0, "hao3"),
    ])
    assert _repair_reference_authority_colocated_silence(foreign, "你，好") == 0
    assert any(iv.text == "<sp0>" for iv in foreign.intervals)
