from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.postprocess_textgrids import (
    Interval,
    Tier,
    TextGrid,
    _duration_ticks,
    _absorb_tiny_gaps,
    _build_gap_ownership_evidence,
    _bind_source_phone_lineage,
    _rebuild_phones_from_lineage,
    _reassert_ctc_word_authority_tier,
    _snap_to_ctc,
    _reconcile_publication_geometry,
    _find_internal_pp_gaps,
    _fix_overlapping_boundaries,
    absorb_silence_into_punct,
    handle_unexpected_silences,
    _repair_authority_punctuation_geometry,
    _repair_reference_authority_colocated_silence,
    _restore_reference_punctuation,
    _normalize_authority_short_interword_silence,
    _resolve_visual_short_silence_merges,
    _normalize_final_internal_silence_labels,
    _publication_contract_audit,
    _pure_silence_label,
    _freeze_processed_geometry,
    _rebuild_derived_from_frozen_words,
    _inject_punctuation,
    _strict_semantic_tokens,
    detect_issues,
    is_silence,
    tier_by_name,
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


def _visual_args(**overrides):
    values = dict(merge_max_sil_sec=0.2, merge_energy_threshold=0.5)
    values.update(overrides)
    return SimpleNamespace(**values)


def _visual_grid(gap=(0.4, 0.5), left="ni3", right="hao3"):
    start, end = gap
    return TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
        Interval(0.0, start, left), Interval(start, end, "<sp0>"),
        Interval(end, 0.9, right),
    ])])


def _gap_audio(left_level=0.2, right_level=0.0, gap=(0.4, 0.5),
               sr=16000):
    audio = np.zeros(sr, dtype=np.float32)
    start, end = (int(gap[0] * sr), int(gap[1] * sr))
    midpoint = (start + end) // 2
    audio[start:midpoint] = left_level
    audio[midpoint:end] = right_level
    # Give both context windows a deterministic voiced reference.
    audio[max(0, start - int(0.05 * sr)):start] = 0.2
    audio[end:min(len(audio), end + int(0.05 * sr))] = 0.2
    return audio


def test_duration_ticks_are_exact_at_30ms_boundary():
    assert _duration_ticks(0.0, 0.030000) == 30_000
    assert _duration_ticks(0.0, 0.029999) == 29_999
    assert _duration_ticks(11.54, 11.569999999999999) == 30_000


def test_ctc_authority_is_restored_after_late_tier_mutation():
    words = Tier("words", 0.0, 1.0, [Interval(0.10, 0.20, "ni3")])
    words._ctc_word_authority = [{
        "lexical_ordinal": 0,
        "text": "ni3",
        "boundary_source": "ctc",
        "ctc_span": [0.60, 0.80],
        "mfa_span": [0.10, 0.20],
        "resolved_span": [0.60, 0.80],
    }]
    words.intervals[0] = Interval(0.25, 0.35, "ni3")
    restored, changed = _reassert_ctc_word_authority_tier(words)
    assert changed == 1
    assert (restored.intervals[0].xmin, restored.intervals[0].xmax) == (0.60, 0.80)


def test_ctc_boundary_compensates_mfa_overlap_without_squeezing_next_word():
    """A late MFA end must not push the next CTC word past its anchor."""
    words = Tier("words", 0.0, 2.0, [
        Interval(0.88, 1.04, "wan3"),
        Interval(1.04, 1.16, "fan4"),
    ])
    ctc = [
        {"type": "word", "word": "wan3", "start_s": 0.87, "end_s": 0.99},
        {"type": "word", "word": "fan4", "start_s": 0.99, "end_s": 1.05},
    ]
    snapped, _ = _snap_to_ctc(words, None, ctc)
    lexical = [iv for iv in snapped.intervals if not is_silence(iv.text)]
    assert [(iv.xmin, iv.xmax) for iv in lexical] == [(0.87, 0.99), (0.99, 1.05)]
    assert snapped._ctc_word_authority[1]["resolved_span"] == [0.99, 1.05]
    assert snapped._ctc_word_authority[1]["boundary_source"] == "ctc"
    assert snapped._ctc_word_authority[1]["arbitration"] == "ctc_neighbor_compensation"


def test_same_sized_real_gap_is_not_consumed_by_derived_sync():
    for gap in (0.010, 0.029999):
        words = Tier("words", 0.0, 1.0, [
            Interval(0.0, 0.4, "ni3"),
            Interval(0.4 + gap, 0.8, "hao3"),
        ])
        evidence = {(0, 1): {
            "mechanical_frame_residual": True,
            "source_ctc_contiguous": True,
            "source_owner": False,
            "ctc_owner": False,
            "reference_owner": False,
            "gap_duration_us": round(gap * 1_000_000),
        }}
        canonical = _absorb_tiny_gaps(words, ownership_evidence=evidence)
        canonical = _reconcile_publication_geometry(canonical)
        assert any(is_silence(iv.text)
                   and abs(iv.xmin - 0.4) < 1e-9
                   and abs(iv.xmax - (0.4 + gap)) < 1e-9
                   for iv in canonical.intervals)


def test_same_sized_gap_without_ownership_evidence_is_preserved():
    for gap in (0.010, 0.029999):
        words = Tier("words", 0.0, 1.0, [
            Interval(0.0, 0.4, "ni3"),
            Interval(0.4 + gap, 0.8, "hao3"),
        ])
        canonical = _reconcile_publication_geometry(_absorb_tiny_gaps(words))
        assert any(is_silence(iv.text) and iv.xmin == 0.4
                   and iv.xmax == 0.4 + gap
                   for iv in canonical.intervals)


def test_30ms_direct_gap_remains_substantive_silence():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"),
        Interval(0.430000, 0.8, "hao3"),
    ])
    canonical = _reconcile_publication_geometry(_absorb_tiny_gaps(words))
    assert any(
        is_silence(iv.text)
        and _duration_ticks(iv.xmin, iv.xmax) == 30_000
        for iv in canonical.intervals
    )


def test_source_silence_or_punctuation_owner_blocks_same_sized_gap():
    for owner in ("<sp1>", "，"):
        words = Tier("words", 0.0, 1.0, [
            Interval(0.0, 0.4, "ni3"),
            Interval(0.4, 0.41, owner),
            Interval(0.41, 0.8, "hao3"),
        ])
        evidence = {(0, 2): {
            "mechanical_frame_residual": False,
            "source_ctc_contiguous": True,
            "source_owner": True,
            "ctc_owner": owner == "，",
            "reference_owner": False,
        }}
        canonical = _reconcile_publication_geometry(
            _absorb_tiny_gaps(words, ownership_evidence=evidence))
        if is_silence(owner):
            assert any(is_silence(iv.text) and iv.xmin == 0.4
                       and iv.xmax == 0.41 for iv in canonical.intervals)
        else:
            assert any(iv.text == owner for iv in canonical.intervals)


def test_gap_evidence_builder_requires_exact_sequences_and_authority():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.41, 0.8, "hao3")])
    source = [
        {"text": "ni3", "start": 0.0, "end": 0.4},
        {"text": "hao3", "start": 0.41, "end": 0.8},
    ]
    ctc = [
        {"word": "ni3", "start_s": 0.0, "end_s": 0.4, "type": "word"},
        {"word": "hao3", "start_s": 0.4, "end_s": 0.8, "type": "word"},
    ]
    evidence = _build_gap_ownership_evidence(
        words, source, ctc, reference_text="你，好", reference_authoritative=True)
    assert evidence[(0, 1)]["mechanical_frame_residual"] is False
    assert evidence[(0, 1)]["reference_owner"] is True


def test_ctc_only_terminal_punctuation_is_removed_without_consuming_edge_silence():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.05, "<sp0>"),
        Interval(0.05, 0.72, "huan1"),
        Interval(0.72, 1.0, "<sp1>"),
    ])
    words, _ = _inject_punctuation(
        words, None, [{"word": ".", "start_s": 0.72, "end_s": 1.0}])
    assert not any(iv.text == "." for iv in words.intervals)
    assert any(iv.text == "<sp1>" and iv.xmin == 0.72 and iv.xmax == 1.0
               for iv in words.intervals)


def test_reference_punctuation_is_bound_to_local_gap_not_unrelated_audio():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.6, "<sp1>"),
        Interval(0.6, 0.9, "hao3"), Interval(0.9, 1.0, "<sp0>")])
    words, _ = _inject_punctuation(
        words, None, [{"word": "，", "start_s": 0.31, "end_s": 0.32}],
        reference_text="你，好", reference_authoritative=True)
    punct = next(iv for iv in words.intervals if iv.text == "，")
    assert punct.xmin >= 0.3 and punct.xmax <= 0.6
    assert punct.xmax < 0.6


def test_reference_anchor_without_silence_does_not_cut_lexical_audio():
    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.8, "hao3")])
    words, _ = _inject_punctuation(
        words, None, [{"word": "，", "start_s": 0.4, "end_s": 0.5}])
    assert not any(iv.text == "，" for iv in words.intervals)
    assert _restore_reference_punctuation(
        words, "你，好", [{"word": "，", "start_s": 0.4, "end_s": 0.5}]) == 0
    assert not any(iv.text == "，" for iv in words.intervals)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.4, "ni3"), (0.4, 0.8, "hao3")]


def test_reference_punctuation_inside_extended_left_word_without_pause_is_ignored():
    """A CTC anchor alone must not steal voiced audio from an extended word."""
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.60, "ni3"),
        Interval(0.60, 0.80, "hao3"),
    ])
    entries = [{"word": "，", "start_s": 0.40, "end_s": 0.50}]

    words, _ = _inject_punctuation(
        words, None, entries,
        reference_text="你，好", reference_authoritative=True)

    ni = next(iv for iv in words.intervals if iv.text == "ni3")
    assert ni.xmax == 0.60
    assert not any(iv.text == "，" for iv in words.intervals)


def test_reference_restore_replaces_pause_not_voiced_prefix_of_owner():
    """Only the silent part of an overlapped anchor becomes punctuation."""
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.60, "ni3"),
        Interval(0.60, 0.70, "<sp1>"),
        Interval(0.70, 0.80, "hao3"),
    ])
    entries = [{"word": "，", "start_s": 0.40, "end_s": 0.65}]

    assert _restore_reference_punctuation(words, "你，好", entries) == 1
    punct = next(iv for iv in words.intervals if iv.text == "，")
    assert (punct.xmin, punct.xmax) == (0.60, 0.70)
    ni = next(iv for iv in words.intervals if iv.text == "ni3")
    assert ni.xmax == 0.60


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


def test_source_phone_lineage_affinely_remaps_each_semantic_owner():
    source_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.5, "shang1"), Interval(0.5, 1.0, "chuang1")])
    source_phones = Tier("phones", 0.0, 1.0, [
        Interval(0.0, 0.15, "ʂ"), Interval(0.15, 0.5, "ang1"),
        Interval(0.5, 0.62, "ʈʂʰ"), Interval(0.62, 1.0, "uang1")])
    lineage = _bind_source_phone_lineage(source_words, source_phones)
    assert lineage["status"] == "verified"

    final_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "shang1"), Interval(0.4, 1.0, "chuang1")])
    rebuilt = _rebuild_phones_from_lineage(final_words, source_phones, lineage)
    assert rebuilt is not None
    assert [iv.text for iv in rebuilt.intervals] == ["ʂ", "ang1", "ʈʂʰ", "uang1"]
    assert rebuilt.intervals[1].xmax == 0.4
    assert rebuilt.intervals[2].xmin == 0.4


def test_crossing_source_phone_has_ambiguous_lineage_and_is_not_assigned():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.5, "shang1"), Interval(0.5, 1.0, "chuang1")])
    phones = Tier("phones", 0.0, 1.0, [Interval(0.45, 0.55, "ʂ")])
    lineage = _bind_source_phone_lineage(words, phones)
    assert lineage["status"] == "rejected"
    assert lineage["reasons"][0]["reason"] == "phone_lineage_ambiguous"
    assert _rebuild_phones_from_lineage(words, phones, lineage) is None


def test_nonlexical_source_phone_is_split_by_final_silence_and_punctuation_owners():
    source_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.8, "<sp2>"),
        Interval(0.8, 1.0, "hao3")])
    source_phones = Tier("phones", 0.0, 1.0, [
        Interval(0.4, 0.8, "sil")])
    lineage = _bind_source_phone_lineage(source_words, source_phones)
    assert lineage["status"] == "verified"
    final_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.6, "<sp1>"),
        Interval(0.6, 0.7, "，"), Interval(0.7, 0.8, "<sp1>"),
        Interval(0.8, 1.0, "hao3")])
    rebuilt = _rebuild_phones_from_lineage(final_words, source_phones, lineage)
    assert rebuilt is not None
    assert [(iv.xmin, iv.xmax, iv.text) for iv in rebuilt.intervals] == [
        (0.4, 0.6, "sil"), (0.6, 0.7, "sil"), (0.7, 0.8, "sil")]


def test_nonlexical_source_phone_crossing_final_lexical_owner_is_clipped():
    source_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.8, "<sp2>"),
        Interval(0.8, 1.0, "hao3")])
    source_phones = Tier("phones", 0.0, 1.0, [Interval(0.4, 0.8, "sil")])
    lineage = _bind_source_phone_lineage(source_words, source_phones)
    final_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.6, "<sp1>"),
        Interval(0.6, 0.7, "hao3"), Interval(0.7, 0.8, "<sp1>"),
        Interval(0.8, 1.0, "hao3")])
    rebuilt = _rebuild_phones_from_lineage(final_words, source_phones, lineage)
    assert rebuilt is not None
    assert [(iv.xmin, iv.xmax, iv.text) for iv in rebuilt.intervals] == [
        (0.4, 0.6, "sil"), (0.7, 0.8, "sil")]


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
    assert (punct.xmin, punct.xmax) == (0.31, 0.5)
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


def test_reference_punctuation_anchor_can_clip_only_immediate_overlapping_owners():
    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.5, "<sp1>"),
        Interval(0.5, 0.8, "hao3")])
    _restore_reference_punctuation(
        words, "你，好", [{"word": "，", "start_s": 0.30 - 0.004,
                           "end_s": 0.50 + 0.007}])
    punct = next(iv for iv in words.intervals if iv.text == "，")
    assert (punct.xmin, punct.xmax) == (0.3, 0.5)
    assert next(iv for iv in words.intervals if iv.text == "ni3").xmax == 0.3
    assert next(iv for iv in words.intervals if iv.text == "hao3").xmin == 0.5


def test_broad_punctuation_anchor_owns_local_pause_without_consuming_next_words():
    # CTC punctuation may remain active through a long realised pause and
    # even report an end after subsequent MFA lexical intervals.  The final
    # punctuation owner must absorb the local pause, but may not consume the
    # next spoken word or the words after it.
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.30, "ni3"),
        Interval(0.30, 0.70, "<sp2>"),
        Interval(0.70, 0.80, "hao3"),
        Interval(0.80, 1.00, "ma1"),
    ])
    entries = [{"word": "。", "start_s": 0.35, "end_s": 1.00}]

    assert _restore_reference_punctuation(words, "你。好吗", entries) == 1
    labels = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    assert labels == [
        (0.0, 0.30, "ni3"),
        (0.30, 0.70, "。"),
        (0.70, 0.80, "hao3"),
        (0.80, 1.00, "ma1"),
    ]


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
    assert _restore_reference_punctuation(words, "你～好", []) == 1
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


def test_punctuation_absorbs_pause_on_either_side_but_not_lexical_gap():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.4, "<sp1>"),
        Interval(0.4, 0.45, "，"), Interval(0.45, 0.5, "<sp1>"),
        Interval(0.5, 1.0, "hao3")])
    phones = Tier("phones", 0.0, 1.0, [
        Interval(0.3, 0.4, "sil"), Interval(0.45, 0.5, "sil")])
    pp = Tier("pinyin_phones", 0.0, 1.0, [
        Interval(0.3, 0.4, "sil"), Interval(0.45, 0.5, "sil")])
    tg = TextGrid(0.0, 1.0, [words, phones, pp])
    absorb_silence_into_punct(tg)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.3, "ni3"), (0.3, 0.5, "，"), (0.5, 1.0, "hao3")]
    assert not any(is_silence(iv.text) for iv in phones.intervals)


def test_restore_partial_projection_preserves_existing_mark_and_adds_only_missing_one():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.2, "ni3"), Interval(0.2, 0.25, "，"),
        Interval(0.25, 0.35, "<sp1>"), Interval(0.35, 0.5, "hao3"),
        Interval(0.5, 0.6, "<sp1>"), Interval(0.6, 1.0, "ma1"),
    ])
    changed = _restore_reference_punctuation(
        words, "你，好。吗", [{"word": "。", "start_s": 0.5, "end_s": 0.6}])
    assert changed == 1
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.2, "ni3"), (0.2, 0.25, "，"),
        (0.25, 0.35, "<sp0>"), (0.35, 0.5, "hao3"),
        (0.5, 0.6, "。"), (0.6, 1.0, "ma1"),
    ]


def test_restore_repeated_same_char_does_not_reuse_later_occurrence_for_missing_one():
    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.2, "ni3"), Interval(0.2, 0.3, "<sp1>"),
        Interval(0.3, 0.4, "hao3"), Interval(0.4, 0.5, "。"),
        Interval(0.5, 0.8, "ma1"),
    ])
    _restore_reference_punctuation(
        words, "你。好。吗", [{"word": "。", "start_s": 0.4, "end_s": 0.5}])
    periods = [iv for iv in words.intervals if iv.text == "。"]
    assert len(periods) == 1
    assert (periods[0].xmin, periods[0].xmax) == (0.4, 0.5)
    assert any(is_silence(iv.text) and iv.xmin == 0.2
               and iv.xmax == 0.3 for iv in words.intervals)


def test_restore_is_idempotent_for_partial_projection():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.4, "<sp1>"),
        Interval(0.4, 0.7, "hao3"), Interval(0.7, 1.0, "ma1"),
    ])
    entries = [{"word": "，", "start_s": 0.3, "end_s": 0.4}]
    assert _restore_reference_punctuation(words, "你，好。吗", entries) == 1
    first = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    first_digest = tuple(first)
    assert _restore_reference_punctuation(words, "你，好。吗", entries) == 0
    assert tuple((iv.xmin, iv.xmax, iv.text) for iv in words.intervals) == first_digest


def test_restore_keeps_true_lexical_pause_without_reference_or_ctc_punctuation():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.7, "<sp2>"),
        Interval(0.7, 1.0, "hao3"),
    ])
    before = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    assert _restore_reference_punctuation(words, "你好", []) == 0
    after = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    assert [(start, end) for start, end, _ in after] == [
        (start, end) for start, end, _ in before]
    assert is_silence(words.intervals[1].text)


def test_restore_does_not_consume_valid_edge_tail_silence_for_ctc_only_mark():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.1, "<sp0>"), Interval(0.1, 0.7, "huan1"),
        Interval(0.7, 1.0, "<sp1>"),
    ])
    assert _restore_reference_punctuation(
        words, "欢", [{"word": ".", "start_s": 0.7, "end_s": 1.0}]) == 0
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals][-1] == (
        0.7, 1.0, "<sp1>")


def test_restore_does_not_turn_internal_phone_hole_into_punctuation():
    # The phone tier can expose a hole, but without an explicit words-tier
    # silence there is no local punctuation owner to claim.
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.5, "ni3"), Interval(0.5, 1.0, "hao3")])
    phones = Tier("phones", 0.0, 1.0, [Interval(0.0, 0.2, "n"),
                                        Interval(0.7, 1.0, "h")])
    before = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    assert _restore_reference_punctuation(
        words, "你，好", [{"word": "，", "start_s": 0.45, "end_s": 0.55}]) == 0
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == before
    assert phones.intervals[0].xmax == 0.2


def test_authority_final_commit_merges_unowned_sp0_and_syncs_phone_tiers():
    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.42, "<sp0>"),
        Interval(0.42, 0.8, "hao3")])
    phones = Tier("phones", 0.0, 0.8, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.3, "i"),
        Interval(0.3, 0.42, "sil"), Interval(0.42, 0.65, "h"),
        Interval(0.65, 0.8, "ao")])
    pp = Tier("pinyin_phones", 0.0, 0.8, [
        Interval(0.0, 0.3, "i"), Interval(0.3, 0.42, "sil"),
        Interval(0.42, 0.8, "ao")])
    tg = TextGrid(0.0, 0.8, [words, phones, pp])
    before = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    assert _normalize_authority_short_interword_silence(tg, "你好", []) == 0
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == before
    assert any(iv.text == "sil" for iv in phones.intervals)
    assert any(iv.text == "sil" for iv in pp.intervals)
    assert _normalize_authority_short_interword_silence(tg, "你好", []) == 0


def test_authority_short_sp_normalization_preserves_substantive_and_punct_gaps():
    for label in ("<sp1>", "<sp2>"):
        words = Tier("words", 0.0, 1.0, [
            Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.3 + (0.3 if label == "<sp1>" else 0.5), label),
            Interval(0.3 + (0.3 if label == "<sp1>" else 0.5), 1.0, "hao3")])
        tg = TextGrid(0.0, 1.0, [words])
        assert _normalize_authority_short_interword_silence(tg, "你好", []) == 0
        assert words.intervals[1].text == label

    words = Tier("words", 0.0, 0.8, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.35, "，"),
        Interval(0.35, 0.42, "<sp0>"), Interval(0.42, 0.8, "hao3")])
    tg = TextGrid(0.0, 0.8, [words])
    assert _normalize_authority_short_interword_silence(
        tg, "你，好", [{"word": "，", "start_s": 0.3, "end_s": 0.35}]) == 0
    assert any(iv.text == "<sp0>" for iv in words.intervals)


def test_visual_energy_left_and_right_owner_are_committed_from_snapshot():
    left_grid = _visual_grid()
    left_report = {}
    left_decisions = _resolve_visual_short_silence_merges(
        left_grid, _gap_audio(), 16000, _visual_args(), left_report)
    assert left_decisions[0]["decision"] == "merged_left"
    assert left_decisions[0]["winner_share"] >= 0.55
    assert [(iv.xmin, iv.xmax, iv.text)
            for iv in left_grid.tiers[0].intervals] == [
                (0.0, 0.5, "ni3"), (0.5, 0.9, "hao3")]
    assert left_report["silence_merges"][0]["visual_reference_digest"]

    right_grid = _visual_grid()
    right_report = {}
    right_decisions = _resolve_visual_short_silence_merges(
        right_grid, _gap_audio(0.0, 0.2), 16000, _visual_args(), right_report)
    assert right_decisions[0]["decision"] == "merged_right"
    assert [(iv.xmin, iv.xmax, iv.text)
            for iv in right_grid.tiers[0].intervals] == [
                (0.0, 0.4, "ni3"), (0.4, 0.9, "hao3")]


@pytest.mark.parametrize(
    ("duration_us", "stale", "expected"),
    [
        (200_000, "<sp0>", "<sp1>"),
        (199_999, "<sp0>", "<sp0>"),
        (100_000, "<sp1>", "<sp0>"),
        (500_000, "<sp1>", "<sp2>"),
        (570_000, "<sp2>", "<sp2>"),
        (870_000, "<sp2>", "<sp2>"),
    ],
)
def test_final_internal_silence_labels_use_integer_ticks_after_owner_commit(
        duration_us, stale, expected):
    end = 0.4 + duration_us / 1_000_000
    grid = TextGrid(0.0, 1.4, [Tier("words", 0.0, 1.4, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, end, stale),
        Interval(end, 1.4, "hao3")])])
    report = {"silence_merges": [{
        "gap_span": [0.4, end], "label": stale,
        "expected_label": expected, "reason": "silence_label_duration_mismatch",
    }]}

    before = [(iv.xmin, iv.xmax) for iv in grid.tiers[0].intervals]
    records = _normalize_final_internal_silence_labels(grid, report)
    silence = grid.tiers[0].intervals[1]
    assert (silence.xmin, silence.xmax) == before[1]
    assert silence.text == expected
    assert report["silence_merges"][0]["label"] == stale
    assert report["silence_merges"][0]["expected_label"] == expected
    assert report["silence_merges"][0]["reason"] == (
        "silence_label_duration_mismatch")
    assert report["silence_merges"][0]["serialized_label"] == expected
    assert report["silence_merges"][0]["normalization_status"] == (
        "relabelled" if stale != expected else "unchanged")
    assert len(records) == (stale != expected)


def test_exact_200ms_stale_sp0_is_not_reclassified_before_owner_decision():
    grid = _visual_grid((0.4, 0.6))
    report = {}
    decisions = _resolve_visual_short_silence_merges(
        grid, None, 16000, _visual_args(merge_max_sil_sec=0.5), report)
    # The stale label is still the resolver input and therefore does not
    # activate Case 161's forced-left policy.
    assert decisions[0]["label"] == "<sp0>"
    assert decisions[0]["expected_label"] == "<sp1>"
    assert decisions[0]["decision"] == "preserve"
    assert decisions[0]["reason"] == "silence_label_duration_mismatch"
    assert [(iv.xmin, iv.xmax, iv.text)
            for iv in grid.tiers[0].intervals] == [
                (0.0, 0.4, "ni3"), (0.4, 0.6, "<sp0>"),
                (0.6, 0.9, "hao3")]

    records = _normalize_final_internal_silence_labels(grid, report)
    assert records and records[0]["from_label"] == "<sp0>"
    assert grid.tiers[0].intervals[1].text == "<sp1>"
    assert len(grid.tiers[0].intervals) == 3
    assert [(iv.xmin, iv.xmax) for iv in grid.tiers[0].intervals] == [
        (0.0, 0.4), (0.4, 0.6), (0.6, 0.9)]
    assert report["silence_merges"][0]["reason"] == (
        "silence_label_duration_mismatch")
    assert report["silence_merges"][0]["normalization_status"] == "relabelled"


def test_final_internal_silence_label_normalization_is_idempotent_and_ledgered():
    grid = _visual_grid((0.4, 0.6))
    report = {}
    _resolve_visual_short_silence_merges(
        grid, None, 16000, _visual_args(merge_max_sil_sec=0.5), report)
    first = _normalize_final_internal_silence_labels(grid, report)
    ledger = list(grid.tiers[0]._processed_geometry_ledger)
    second = _normalize_final_internal_silence_labels(grid, report)
    assert len(first) == 1
    assert second == []
    assert grid.tiers[0]._processed_geometry_ledger == ledger
    assert sum(item["operation"] == "final_internal_silence_label_normalization"
               for item in ledger) == 1


def test_leading_sp1_is_excluded_and_freeze_rebuild_preserves_final_label():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.4, "ni3"),
        Interval(0.4, 0.6, "<sp0>"), Interval(0.6, 1.0, "hao3")])
    tg = TextGrid(0.0, 1.0, [words])
    records = _normalize_final_internal_silence_labels(tg, {})
    assert len(records) == 1
    assert [iv.text for iv in words.intervals] == [
        "<sp1>", "ni3", "<sp1>", "hao3"]
    frozen, reasons = _freeze_processed_geometry(tg)
    assert frozen is words and reasons == []
    _rebuild_derived_from_frozen_words(
        tg, {}, None, "你好")
    assert [iv.text for iv in words.intervals] == [
        "<sp1>", "ni3", "<sp1>", "hao3"]


def test_frozen_hanzi_rebuild_preserves_exact_threshold_silence_labels():
    words = Tier("words", 0.0, 1.7, [
        Interval(0.0, 0.2, "ni3"),
        Interval(0.2, 0.4, "<sp1>"),
        Interval(0.4, 0.7, "hao3"),
        Interval(0.7, 1.2, "<sp2>"),
        Interval(1.2, 1.7, "ma5"),
    ])
    grid = TextGrid(0.0, 1.7, [
        words,
        Tier("hanzi", 0.0, 1.7, []),
    ])
    frozen, reasons = _freeze_processed_geometry(grid)
    assert frozen is words and reasons == []

    _rebuild_derived_from_frozen_words(grid, {}, None, "你好吗")

    hanzi = tier_by_name(grid, "hanzi")
    assert hanzi is not None
    assert [iv.text for iv in hanzi.intervals] == [
        "你", "<sp1>", "好", "<sp2>", "吗"]
    assert [(iv.xmin, iv.xmax) for iv in hanzi.intervals] == [
        (iv.xmin, iv.xmax) for iv in words.intervals]


def test_normalized_internal_pause_remains_mid_and_strict_interior_rejected():
    grid = _visual_grid((0.4, 0.6))
    report = {}
    _resolve_visual_short_silence_merges(
        grid, None, 16000, _visual_args(merge_max_sil_sec=0.5), report)
    _normalize_final_internal_silence_labels(grid, report)
    words = grid.tiers[0]
    interior = [iv for iv in words.intervals if is_silence(iv.text)]
    assert len(interior) == 1 and interior[0].text == "<sp1>"
    assert any(iv.xmin > words.xmin and iv.xmax < words.xmax
               for iv in interior)
    # Both final QC classifications are label-independent: the retained
    # interval remains an internal pause after its display label is fixed.
    details = [{"label": iv.text, "start_s": iv.xmin, "end_s": iv.xmax}
               for iv in interior]
    assert details[0]["label"] == "<sp1>"
    mid_sp = [iv for index, iv in enumerate(words.intervals)
              if 0 < index < len(words.intervals) - 1
              and is_silence(iv.text)]
    assert mid_sp == interior

    hanzi = Tier("hanzi", 0.0, 1.0, [
        Interval(0.0, 0.4, "你"), Interval(0.4, 0.6, "<sp1>"),
        Interval(0.6, 0.9, "好")])
    phones = Tier("phones", 0.0, 1.0, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "i"),
        Interval(0.6, 0.7, "h"), Interval(0.7, 0.9, "ao")])
    pp = Tier("pinyin_phones", 0.0, 1.0, list(phones.intervals))
    reasons, details = _publication_contract_audit(
        words, hanzi, pp, phones, "你好",
        [{"text": "ni3", "start": 0.0, "end": 0.4},
         {"text": "hao3", "start": 0.6, "end": 0.9}],
        [{"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.4},
         {"type": "word", "word": "hao3", "start_s": 0.6, "end_s": 0.9}],
        False, None)
    assert "strict_interior_sp" in reasons
    assert details["strict_interior_sp"][0]["label"] == "<sp1>"


def test_visual_energy_sp1_always_forwards_to_left_owner():
    for right_level in (0.0, 0.2):
        grid = _visual_grid((0.4, 0.6))
        grid.tiers[0].intervals[1].text = "<sp1>"
        report = {}
        decisions = _resolve_visual_short_silence_merges(
            grid, _gap_audio(0.2 if right_level == 0.0 else 0.0,
                             right_level, gap=(0.4, 0.6)), 16000,
                             _visual_args(merge_max_sil_sec=0.5), report)
        assert decisions[0]["decision"] == "merged_left"
        assert decisions[0]["reason"] == "forced_internal_sp1_forward"
        assert decisions[0]["policy"] == "forced_internal_sp1_forward"
        assert decisions[0]["direction_source"] == "forced_left"
        assert decisions[0]["label"] == "<sp1>"
        assert decisions[0]["expected_label"] == "<sp1>"
        assert decisions[0]["effective_max_sil_sec"] == 0.5
        assert report["silence_merges"][0]["gap_duration"] == 0.2


def test_visual_energy_sp1_default_bound_and_duration_boundaries():
    default_grid = _visual_grid((0.4, 0.6))
    default_grid.tiers[0].intervals[1].text = "<sp1>"
    default_decision = _resolve_visual_short_silence_merges(
        default_grid, _gap_audio(gap=(0.4, 0.6)), 16000,
        _visual_args(), {})[0]
    assert default_decision["decision"] == "merged_left"
    assert default_decision["reason"] == "forced_internal_sp1_forward"
    assert default_decision["policy"] == "forced_internal_sp1_forward"
    assert default_decision["direction_source"] == "forced_left"
    assert default_decision["effective_max_sil_sec"] == 0.2

    for gap, label, expected_decision, expected_label in [
        ((0.4, 0.6), "<sp1>", "merged_left", "<sp1>"),       # 0.2
        ((0.4, 0.899999), "<sp1>", "merged_left", "<sp1>"),   # 0.499999
        ((0.4, 0.9), "<sp2>", "preserve", "<sp2>"),           # 0.5
    ]:
        grid = _visual_grid(gap)
        grid.tiers[0].intervals[1].text = label
        decision = _resolve_visual_short_silence_merges(
            grid, _gap_audio(gap=gap), 16000,
            _visual_args(merge_max_sil_sec=0.5), {})[0]
        assert decision["decision"] == expected_decision
        assert decision["label"] == expected_label
        if label == "<sp2>":
            assert decision["reason"] == "unsupported_silence_label"

    configured_grid = _visual_grid((0.4, 0.65))
    configured_grid.tiers[0].intervals[1].text = "<sp1>"
    decision = _resolve_visual_short_silence_merges(
        configured_grid, _gap_audio(gap=(0.4, 0.65)), 16000,
        _visual_args(merge_max_sil_sec=0.3), {})[0]
    assert decision["decision"] == "merged_left"
    assert decision["effective_max_sil_sec"] == 0.3


def test_visual_energy_sp1_low_zero_ambiguous_and_single_frame_are_preserved():
    cases = [
        (_gap_audio(0.2, 0.2, gap=(0.4, 0.6)), "preserve_ambiguous_energy"),
        (_gap_audio(0.0005, 0.0005, gap=(0.4, 0.6)),
         "preserve_no_continuous_active"),
        (_gap_audio(0.2, 0.0, gap=(0.4, 0.6)) * 0.0,
         "preserve_all_zero_audio"),
    ]
    one_frame = np.zeros(16000, dtype=np.float32)
    one_frame[6400:6450] = 0.2
    cases.append((one_frame, "preserve_no_continuous_active"))
    for audio, reason in cases:
        grid = _visual_grid((0.4, 0.6))
        grid.tiers[0].intervals[1].text = "<sp1>"
        decision = _resolve_visual_short_silence_merges(
            grid, audio, 16000, _visual_args(merge_max_sil_sec=0.5), {})[0]
        assert decision["decision"] == "merged_left"
        assert decision["reason"] == "forced_internal_sp1_forward"
        assert decision["policy"] == "forced_internal_sp1_forward"
        assert decision["direction_source"] == "forced_left"
        assert decision["forced_original_reason"] == reason
        assert len(grid.tiers[0].intervals) == 2


def test_visual_energy_sp1_protects_labels_owners_and_lineage_evidence():
    # The pure-label helper must reject attached/non-canonical text.
    assert _pure_silence_label(" <SP1> ") == "<sp1>"
    assert _pure_silence_label("<sp1>tail") is None

    cases = []
    punct = _visual_grid((0.4, 0.6))
    punct.tiers[0].intervals[1].text = "<sp1>"
    cases.append((punct, {"reference_punct_entries": [
        {"word": "，", "start_s": 0.44, "end_s": 0.46}]},
        "reference_punctuation_owner"))

    nvv = _visual_grid((0.4, 0.6), left="<LAUGHTER>")
    nvv.tiers[0].intervals[1].text = "<sp1>"
    cases.append((nvv, {}, "nvv_owner"))

    edge = _visual_grid((0.0, 0.2))
    edge.tiers[0].intervals[1].text = "<sp1>"
    cases.append((edge, {}, "edge"))

    hole = _visual_grid((0.4, 0.6))
    hole.tiers[0].intervals[1].text = "<sp1>"
    hole.tiers.append(Tier("phones", 0.0, 1.0, [
        Interval(0.0, 0.4, "n"), Interval(0.6, 0.9, "h")]))
    cases.append((hole, {}, "phone_hole"))

    ambiguous = _visual_grid((0.4, 0.6))
    ambiguous.tiers[0].intervals[1].text = "<sp1>"
    ambiguous._phone_lineage = {"status": "rejected", "reasons": ["ambiguous"]}
    cases.append((ambiguous, {}, "phone_lineage_ambiguous"))

    for grid, kwargs, reason in cases:
        decision = _resolve_visual_short_silence_merges(
            grid, _gap_audio(gap=(0.4, 0.6)), 16000,
            _visual_args(merge_max_sil_sec=0.5), {}, **kwargs)[0]
        if reason in {"phone_hole", "phone_lineage_ambiguous"}:
            assert decision["decision"] == "merged_left"
            assert decision["reason"] == "forced_internal_sp1_forward"
            assert decision["direction_source"] == "forced_left"
            assert decision["phone_reason"] == reason
            assert len(grid.tiers[0].intervals) == 2
        else:
            assert decision["decision"] == "preserve"
            assert decision["reason"] == reason
            assert len(grid.tiers[0].intervals) == 3

    ctc_punctuation = _visual_grid((0.4, 0.6))
    ctc_punctuation.tiers[0].intervals[1].text = "<sp1>"
    ctc_punctuation_decision = _resolve_visual_short_silence_merges(
        ctc_punctuation, _gap_audio(gap=(0.4, 0.6)), 16000,
        _visual_args(merge_max_sil_sec=0.5), {}, ctc_tokens=[
            {"type": "punctuation", "word": "，", "start_s": 0.45,
             "end_s": 0.55}]
    )[0]
    assert ctc_punctuation_decision["decision"] == "preserve"
    assert ctc_punctuation_decision["reason"] == "ctc_punctuation_owner"

    lexical_ctc = _visual_grid((0.4, 0.6))
    lexical_ctc.tiers[0].intervals[1].text = "<sp1>"
    lexical_ctc_decision = _resolve_visual_short_silence_merges(
        lexical_ctc, _gap_audio(gap=(0.4, 0.6)), 16000,
        _visual_args(merge_max_sil_sec=0.5), {}, ctc_tokens=[
            {"type": "word", "word": "ctc", "start_s": 0.45, "end_s": 0.55}]
    )[0]
    assert lexical_ctc_decision["decision"] == "merged_left"
    assert lexical_ctc_decision["reason"] == "forced_internal_sp1_forward"
    assert lexical_ctc_decision["policy"] == "forced_internal_sp1_forward"

    mixed = TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.5, "<sp0>"),
        Interval(0.5, 0.6, "<sp1>"), Interval(0.6, 0.9, "hao3")])])
    mixed_decision = _resolve_visual_short_silence_merges(
        mixed, _gap_audio(gap=(0.4, 0.6)), 16000,
        _visual_args(merge_max_sil_sec=0.5), {})[0]
    assert mixed_decision["reason"] == "mixed_or_noncanonical_silence_labels"

    mismatch = _visual_grid((0.4, 0.5))
    mismatch.tiers[0].intervals[1].text = "<sp1>"
    mismatch_decision = _resolve_visual_short_silence_merges(
        mismatch, _gap_audio(gap=(0.4, 0.5)), 16000,
        _visual_args(merge_max_sil_sec=0.5), {})[0]
    assert mismatch_decision["reason"] == "silence_label_duration_mismatch"

    for label in ("<sp2>", "<sp3>"):
        grid = _visual_grid((0.4, 0.6))
        grid.tiers[0].intervals[1].text = label
        decision = _resolve_visual_short_silence_merges(
            grid, _gap_audio(gap=(0.4, 0.6)), 16000,
            _visual_args(merge_max_sil_sec=0.5), {})[0]
        assert decision["decision"] == "preserve"
        assert decision["reason"] == "unsupported_silence_label"


def test_visual_forced_internal_sp1_ignores_switch_and_max_but_keeps_diagnostics():
    for overrides in (
        {"merge_silence": False, "merge_max_sil_sec": 0.2},
        {"merge_silence": True, "merge_max_sil_sec": 0.001},
        {"merge_silence": False, "merge_max_sil_sec": 0.5},
    ):
        grid = _visual_grid((0.4, 0.6))
        grid.tiers[0].intervals[1].text = "<sp1>"
        decision = _resolve_visual_short_silence_merges(
            grid, _gap_audio(gap=(0.4, 0.6)), 16000,
            _visual_args(**overrides), {})[0]
        assert decision["decision"] == "merged_left"
        assert decision["policy"] == "forced_internal_sp1_forward"
        assert decision["direction_source"] == "forced_left"
        assert "merge_disabled" in decision["forced_gate_reasons"] or (
            "long_pause" in decision["forced_gate_reasons"])
        assert len(grid.tiers[0].intervals) == 2


def test_visual_terminal_punctuation_tail_absorbs_once_and_preserves_negative_cases():
    head = TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.2, "<sp1>"), Interval(0.2, 0.6, "ni3"),
        Interval(0.6, 1.0, "hao3")])])
    head_decision = _resolve_visual_short_silence_merges(
        head, None, 16000, _visual_args(merge_max_sil_sec=0.5), {})[0]
    assert head_decision["reason"] == "edge"
    assert head.tiers[0].intervals[0].text == "<sp1>"

    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "xu4"), Interval(0.4, 0.55, "！"),
        Interval(0.55, 1.0, "<sp1>")])
    grid = TextGrid(0.0, 1.0, [words])
    report = {}
    decisions = _resolve_visual_short_silence_merges(
        grid, None, 16000, _visual_args(), report)
    assert decisions[0]["operation"] == "terminal_punctuation_tail_absorption"
    assert decisions[0]["decision"] == "terminal_punctuation_tail_absorption"
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.4, "xu4"), (0.4, 1.0, "！")]
    assert report["terminal_punctuation_tail_absorption"][0]["reason"] == (
        "terminal_punctuation_tail_absorption")
    assert _resolve_visual_short_silence_merges(
        grid, None, 16000, _visual_args(), {}) == []

    # Historical 16735 shape: the final existing punctuation owns the tail.
    tail_16735 = TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.05, "<sp0>"), Interval(0.05, 0.38, "bei4"),
        Interval(0.38, 0.52, "，"), Interval(0.52, 0.78, "xu4"),
        Interval(0.78, 0.82, "！"), Interval(0.82, 1.0, "<sp0>")])])
    _resolve_visual_short_silence_merges(
        tail_16735, None, 16000, _visual_args(), {})
    assert [(iv.xmin, iv.xmax, iv.text)
            for iv in tail_16735.tiers[0].intervals][-1] == (0.78, 1.0, "！")
    assert tail_16735.tiers[0].intervals[0].text == "<sp0>"

    # 52697/006035: missing final punctuation must not synthesize an owner.
    for tail in ("<sp1>", "<sp0>"):
        missing = TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
            Interval(0.0, 0.72, "huan1"), Interval(0.72, 1.0, tail)])])
        _resolve_visual_short_silence_merges(
            missing, None, 16000, _visual_args(merge_max_sil_sec=0.5), {})
        assert missing.tiers[0].intervals[-1].text == tail

    ctc_only = TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.72, "huan1"), Interval(0.72, 1.0, "<sp1>")])])
    _resolve_visual_short_silence_merges(
        ctc_only, None, 16000, _visual_args(merge_max_sil_sec=0.5), {},
        ctc_tokens=[{"type": "punct", "word": ".", "start_s": 0.72,
                     "end_s": 1.0}])
    assert ctc_only.tiers[0].intervals[-1].text == "."

    nvv_tail = TextGrid(0.0, 1.0, [Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.7, "<LAUGHTER>"), Interval(0.7, 1.0, "<sp1>")])])
    _resolve_visual_short_silence_merges(
        nvv_tail, None, 16000, _visual_args(merge_max_sil_sec=0.5), {})
    assert nvv_tail.tiers[0].intervals[-1].text == "<sp1>"


def test_visual_sp1_merge_is_idempotent_through_freeze_and_derived_sync():
    words = Tier("words", 0.0, 0.9, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.6, "<sp1>"),
        Interval(0.6, 0.9, "hao3")])
    phones = Tier("phones", 0.0, 0.9, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "i"),
        Interval(0.4, 0.6, "sil"), Interval(0.6, 0.7, "h"),
        Interval(0.7, 0.9, "ao")])
    hanzi = Tier("hanzi", 0.0, 0.9, [
        Interval(0.0, 0.4, "你"), Interval(0.4, 0.6, "<sp1>"),
        Interval(0.6, 0.9, "好")])
    pp = Tier("pinyin_phones", 0.0, 0.9, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "i"),
        Interval(0.4, 0.6, "sil"), Interval(0.6, 0.7, "h"),
        Interval(0.7, 0.9, "ao")])
    tg = TextGrid(0.0, 0.9, [words, phones, hanzi, pp])
    tg._phone_lineage = _bind_source_phone_lineage(words, phones)
    first = _resolve_visual_short_silence_merges(
        tg, _gap_audio(gap=(0.4, 0.6)), 16000,
        _visual_args(merge_max_sil_sec=0.5), {})
    assert first[0]["decision"] == "merged_left"
    geometry = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
    assert _resolve_visual_short_silence_merges(
        tg, _gap_audio(gap=(0.4, 0.6)), 16000,
        _visual_args(merge_max_sil_sec=0.5), {}) == []
    frozen, reasons = _freeze_processed_geometry(tg)
    assert frozen is not None and reasons == []
    _rebuild_derived_from_frozen_words(
        tg, {"n": "n", "i": "i", "h": "h", "ao": "ao"},
        {"ni3": ["n", "i"], "hao3": ["h", "ao"]}, "你好")
    assert geometry == [(iv.xmin, iv.xmax, iv.text)
                        for iv in words.intervals]
    assert not any(is_silence(iv.text) for iv in words.intervals)
    assert not any(is_silence(iv.text)
                   for iv in next(t for t in tg.tiers if t.name == "phones").intervals)


def test_visual_energy_ambiguous_or_missing_audio_forwards_unknown_sp0():
    cases = [
        (_gap_audio(0.2, 0.2), "preserve_ambiguous_energy"),
        (_gap_audio(0.2 * 0.54, 0.2 * 0.46), "preserve_ambiguous_energy"),
        (_gap_audio(0.2, 0.0) * 0.0, "preserve_all_zero_audio"),
        (_gap_audio(0.0005, 0.0005), "preserve_no_continuous_active"),
    ]
    # A single active frame cannot establish the required three-frame run.
    one_frame = np.zeros(16000, dtype=np.float32)
    one_frame[6400:6450] = 0.2
    cases.append((one_frame, "preserve_no_continuous_active"))
    for audio, reason in cases:
        grid = _visual_grid()
        decisions = _resolve_visual_short_silence_merges(
            grid, audio, 16000, _visual_args(), {})
        assert decisions[0]["decision"] == "merged_left"
        assert decisions[0]["reason"] == "unknown_sp0_forward"
        assert decisions[0]["policy"] == "unknown_sp0_forward"
        assert decisions[0]["forced_original_reason"] == reason
        assert len(grid.tiers[0].intervals) == 2

    grid = _visual_grid()
    decisions = _resolve_visual_short_silence_merges(
        grid, None, 16000, _visual_args(), {})
    assert decisions[0]["reason"] == "unknown_sp0_forward"
    assert decisions[0]["policy"] == "unknown_sp0_forward"
    assert decisions[0]["forced_original_reason"] == "missing_audio"
    assert len(grid.tiers[0].intervals) == 2


def test_visual_energy_structural_owners_and_long_pause_are_preserved():
    for label, gap, words in [
        ("<sp1>", (0.4, 0.5), None),
        ("<sp2>", (0.4, 0.5), None),
        ("<sp3>", (0.4, 0.5), None),
        ("<sp0>", (0.4, 0.6), None),
        ("<sp0>", (0.0, 0.1), None),
        ("<sp0>", (0.4, 0.5), ("ni3", "<LAUGHTER>")),
    ]:
        left, right = words or ("ni3", "hao3")
        grid = _visual_grid(gap, left, right)
        grid.tiers[0].intervals[1].text = label
        decisions = _resolve_visual_short_silence_merges(
            grid, _gap_audio(gap=gap), 16000, _visual_args(), {})
        assert decisions[0]["decision"] == "preserve"
        assert len(grid.tiers[0].intervals) == 3

    grid = _visual_grid()
    decisions = _resolve_visual_short_silence_merges(
        grid, _gap_audio(), 16000, _visual_args(), {},
        reference_punct_entries=[{"word": "，", "start_s": 0.44,
                                  "end_s": 0.46}])
    assert decisions[0]["reason"] == "reference_punctuation_owner"


def test_visual_snapshot_ignores_raw_ctc_spans_and_is_idempotent():
    grid = _visual_grid()
    report = {}
    ctc = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.1},
        {"type": "word", "word": "hao3", "start_s": 0.1, "end_s": 0.2},
    ]
    first = _resolve_visual_short_silence_merges(
        grid, _gap_audio(), 16000, _visual_args(), report, ctc_tokens=ctc)
    digest = first[0]["visual_reference_digest"]
    geometry = [(iv.xmin, iv.xmax, iv.text) for iv in grid.tiers[0].intervals]
    second = _resolve_visual_short_silence_merges(
        grid, _gap_audio(), 16000, _visual_args(), {}, ctc_tokens=ctc)
    assert first[0]["decision"] == "merged_left"
    assert second == []
    assert geometry == [(0.0, 0.5, "ni3"), (0.5, 0.9, "hao3")]
    assert digest


def test_visual_multiple_gap_commit_is_snapshot_order_independent():
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    # Gap 1 is owned right; gap 2 is owned left.  Both contexts are voiced.
    audio[int(0.15 * sr):int(0.2 * sr)] = 0.2
    audio[int(0.25 * sr):int(0.3 * sr)] = 0.2
    audio[int(0.5 * sr):int(0.55 * sr)] = 0.2
    audio[int(0.6 * sr):int(0.65 * sr)] = 0.2
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.2, "ni3"), Interval(0.2, 0.3, "<sp0>"),
        Interval(0.3, 0.5, "hao3"), Interval(0.5, 0.6, "<sp0>"),
        Interval(0.6, 0.9, "ma1")])
    grid = TextGrid(0.0, 1.0, [words])
    decisions = _resolve_visual_short_silence_merges(
        grid, audio, sr, _visual_args(), {})
    assert [item["decision"] for item in decisions] == [
        "merged_right", "merged_left"]
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.2, "ni3"), (0.2, 0.6, "hao3"), (0.6, 0.9, "ma1")]
    second = _resolve_visual_short_silence_merges(
        grid, audio, sr, _visual_args(), {})
    assert second == []


def test_visual_gap_rejects_phone_hole_and_ambiguous_lineage():
    grid = _visual_grid()
    grid.tiers.append(Tier("phones", 0.0, 1.0, [
        Interval(0.0, 0.4, "n"), Interval(0.5, 0.9, "h")]))
    decisions = _resolve_visual_short_silence_merges(
        grid, _gap_audio(), 16000, _visual_args(), {})
    assert decisions[0]["reason"] == "unknown_sp0_forward"
    assert decisions[0]["policy"] == "unknown_sp0_forward"
    assert decisions[0]["direction_source"] == "forced_left_fallback"
    assert decisions[0]["operation"] == "unknown_sp0_forward_merge"
    assert len(grid.tiers[0].intervals) == 2

    grid = _visual_grid()
    grid._phone_lineage = {
        "status": "rejected",
        "reasons": [{"reason": "phone_lineage_ambiguous"}],
    }
    decisions = _resolve_visual_short_silence_merges(
        grid, _gap_audio(), 16000, _visual_args(), {})
    assert decisions[0]["reason"] == "unknown_sp0_forward"
    assert decisions[0]["policy"] == "unknown_sp0_forward"
    assert len(grid.tiers[0].intervals) == 2


def test_unknown_internal_sp0_forwards_to_left_owner():
    grid = _visual_grid((0.4, 0.5))
    decisions = _resolve_visual_short_silence_merges(
        grid, np.zeros(16000, dtype=np.float32), 16000,
        _visual_args(), {})
    assert decisions[0]["decision"] == "merged_left"
    assert decisions[0]["reason"] == "unknown_sp0_forward"
    assert decisions[0]["policy"] == "unknown_sp0_forward"
    assert decisions[0]["direction_source"] == "forced_left_fallback"
    assert [(iv.xmin, iv.xmax, iv.text) for iv in grid.tiers[0].intervals] == [
        (0.0, 0.5, "ni3"), (0.5, 0.9, "hao3")]


def test_broad_punctuation_span_does_not_protect_later_internal_sp0():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.4, "<sp0>"),
        Interval(0.4, 0.5, "hao3"), Interval(0.5, 0.6, "<sp0>"),
        Interval(0.6, 0.9, "ma1"),
    ])
    grid = TextGrid(0.0, 1.0, [words])
    decisions = _resolve_visual_short_silence_merges(
        grid, np.zeros(16000, dtype=np.float32), 16000, _visual_args(), {},
        ctc_tokens=[{"type": "punctuation", "word": "，",
                     "start_s": 0.25, "end_s": 0.55}])
    assert [item["punctuation_owner"] for item in decisions] == [None, None]
    assert [item["policy"] for item in decisions] == [
        "unknown_sp0_forward", "unknown_sp0_forward"]
    assert [(iv.xmin, iv.xmax, iv.text) for iv in grid.tiers[0].intervals] == [
        (0.0, 0.4, "ni3"), (0.4, 0.6, "hao3"), (0.6, 0.9, "ma1")]


def test_local_explicit_punctuation_gap_is_restored_as_punctuation():
    grid = _visual_grid((0.4, 0.5))
    decisions = _resolve_visual_short_silence_merges(
        grid, _gap_audio(gap=(0.4, 0.5)), 16000, _visual_args(), {},
        ctc_tokens=[{"type": "punctuation", "word": "，",
                     "start_s": 0.44, "end_s": 0.46}])
    decision = decisions[0]
    assert decision["decision"] == "preserve"
    assert decision["reason"] == "ctc_punctuation_owner"
    assert decision["operation"] == "punctuation_gap_restore"
    assert decision["punctuation_gap_restore"] is True
    assert [(iv.xmin, iv.xmax, iv.text) for iv in grid.tiers[0].intervals] == [
        (0.0, 0.4, "ni3"), (0.4, 0.5, "，"), (0.5, 0.9, "hao3")]


def test_visual_merge_freeze_rebuild_keeps_words_hanzi_and_phone_owners_in_sync():
    words = Tier("words", 0.0, 0.9, [
        Interval(0.0, 0.4, "ni3"), Interval(0.4, 0.5, "<sp0>"),
        Interval(0.5, 0.9, "hao3")])
    phones = Tier("phones", 0.0, 0.9, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "i"),
        Interval(0.4, 0.5, "sil"), Interval(0.5, 0.7, "h"),
        Interval(0.7, 0.9, "ao")])
    hanzi = Tier("hanzi", 0.0, 0.9, [
        Interval(0.0, 0.4, "你"), Interval(0.4, 0.5, "<sp0>"),
        Interval(0.5, 0.9, "好")])
    pp = Tier("pinyin_phones", 0.0, 0.9, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "i"),
        Interval(0.4, 0.5, "sil"), Interval(0.5, 0.7, "h"),
        Interval(0.7, 0.9, "ao")])
    tg = TextGrid(0.0, 0.9, [words, phones, hanzi, pp])
    tg._phone_lineage = _bind_source_phone_lineage(words, phones)
    _resolve_visual_short_silence_merges(
        tg, _gap_audio(), 16000, _visual_args(), {})
    frozen, reasons = _freeze_processed_geometry(tg)
    assert frozen is not None and reasons == []

    # Simulate a late writer carrying the pre-freeze derived tiers.  The
    # frozen words list is the only publication authority; these stale tiers
    # must be replaced rather than allowed to reopen geometry arbitration.
    tg.tiers[2] = Tier("hanzi", 0.0, 0.9, [
        Interval(0.0, 0.4, "你"), Interval(0.4, 0.5, "<sp0>"),
        Interval(0.5, 0.9, "好")])
    tg.tiers[1] = Tier("phones", 0.0, 0.9, [
        Interval(0.0, 0.2, "stale"), Interval(0.2, 0.4, "tier")])
    tg.tiers[3] = Tier("pinyin_phones", 0.0, 0.9, [
        Interval(0.0, 0.4, "stale"), Interval(0.4, 0.9, "tier")])
    _rebuild_derived_from_frozen_words(
        tg, {"n": "n", "i": "i", "h": "h", "ao": "ao"},
        {"ni3": ["n", "i"], "hao3": ["h", "ao"]}, "你好")
    hanzi = next(tier for tier in tg.tiers if tier.name == "hanzi")
    assert [(iv.xmin, iv.xmax) for iv in hanzi.intervals] == [
        (iv.xmin, iv.xmax) for iv in words.intervals]
    assert not any(is_silence(iv.text) for iv in words.intervals)
    rebuilt_phones = next(tier for tier in tg.tiers if tier.name == "phones")
    assert not any(is_silence(iv.text) for iv in rebuilt_phones.intervals)

    # Every final owner is bounded by the frozen words partition, including
    # both outer edges.  This covers the source phones and the two derived
    # display/phone projections together, not just hanzi.
    frozen_words = next(tier for tier in tg.tiers if tier.name == "words")
    for tier_name in ("hanzi", "phones", "pinyin_phones"):
        tier = next(tier for tier in tg.tiers if tier.name == tier_name)
        assert tier.intervals[0].xmin == frozen_words.intervals[0].xmin
        assert tier.intervals[-1].xmax == frozen_words.intervals[-1].xmax
        for interval in tier.intervals:
            owners = [word for word in frozen_words.intervals
                      if interval.xmin >= word.xmin - 1e-9
                      and interval.xmax <= word.xmax + 1e-9]
            assert len(owners) == 1, (tier_name, interval, owners)


def test_frozen_rebuild_preserves_canonical_english_units_and_duplicate_phone_edges():
    words = Tier("words", 0.0, 0.9, [
        Interval(0.0, 0.3, "KPop"),
        Interval(0.3, 0.6, "app"),
        Interval(0.6, 0.9, "app"),
    ])
    phones = Tier("phones", 0.0, 0.9, [
        Interval(0.0, 0.08, "en:K"), Interval(0.08, 0.3, "en:P"),
        Interval(0.3, 0.4, "en:AE1"), Interval(0.4, 0.6, "en:P"),
        Interval(0.6, 0.7, "en:AE1"), Interval(0.7, 0.9, "en:P"),
    ])
    tg = TextGrid(0.0, 0.9, [
        Tier("hanzi", 0.0, 0.9, [Interval(0.0, 0.9, "stale")]),
        words, phones,
        Tier("pinyin_phones", 0.0, 0.9, [Interval(0.0, 0.9, "stale")]),
    ])
    frozen, reasons = _freeze_processed_geometry(tg)
    assert frozen is words and reasons == []
    _rebuild_derived_from_frozen_words(
        tg, {}, {}, "K-Pop app app", reference_authoritative=True)

    rebuilt_hanzi = next(tier for tier in tg.tiers if tier.name == "hanzi")
    assert [iv.text for iv in rebuilt_hanzi.intervals] == ["K-Pop", "app", "app"]
    rebuilt_pp = next(tier for tier in tg.tiers if tier.name == "pinyin_phones")
    assert [iv.text for iv in rebuilt_pp.intervals] == [
        "en:K", "en:P", "en:AE1", "en:P", "en:AE1", "en:P"]
    assert rebuilt_pp.intervals[0].xmin == words.intervals[0].xmin
    assert rebuilt_pp.intervals[-1].xmax == words.intervals[-1].xmax
    assert [(iv.xmin, iv.xmax) for iv in rebuilt_hanzi.intervals] == [
        (iv.xmin, iv.xmax) for iv in words.intervals]
