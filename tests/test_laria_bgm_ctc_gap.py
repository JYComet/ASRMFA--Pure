"""Exact CTC-gap support for no-reference BGM candidate selection."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import postprocess_textgrids as post


def _fixture(*, silence_start=0.6, silence_end=2.4):
    words = post.Tier("words", 0.0, 3.0, [
        post.Interval(0.0, silence_start, "ni3"),
        post.Interval(silence_start, silence_end, "<sp1>"),
        post.Interval(silence_end, 3.0, "hao3"),
    ])
    source = [
        {"ordinal": 0, "start": 0.0, "end": 1.0, "text": "ni3"},
        {"ordinal": 1, "start": 2.0, "end": 3.0, "text": "hao3"},
    ]
    ctc = [
        {"ordinal": 0, "word": "ni3", "start_s": 0.0,
         "end_s": 1.0, "type": "word"},
        {"ordinal": 1, "word": "hao3", "start_s": 2.0,
         "end_s": 3.0, "type": "word"},
    ]
    ledger = post._fallback_lexical_correspondence_ledger(source, ctc, words)
    assert ledger["safe"] is True
    return words, source, ctc, ledger


def test_paired_downstream_geometries_share_the_exact_ctc_supported_candidate():
    first = _fixture(silence_start=0.6, silence_end=2.4)
    second = _fixture(silence_start=0.8, silence_end=2.2)

    first_selection = post._fallback_bgm_ctc_gap_selection(
        first[0], first[1], first[2], first[3], (0.0, 3.0))
    second_selection = post._fallback_bgm_ctc_gap_selection(
        second[0], second[1], second[2], second[3], (0.0, 3.0))

    assert first_selection["validation"]["status"] == "verified"
    assert first_selection["validation"]["digest"] == first[3]["digest"]
    assert first_selection["selection_mode"] == "ctc_gap_supported"
    assert second_selection["selection_mode"] == "ctc_gap_supported"
    first_interval = first_selection["evaluated_intervals"][0]
    second_interval = second_selection["evaluated_intervals"][0]
    assert first_interval["original_silence_span"] != second_interval[
        "original_silence_span"]
    assert first_interval["ctc_gap"] == second_interval["ctc_gap"] == [1.0, 2.0]
    assert first_interval["evaluated_intersection"] == second_interval[
        "evaluated_intersection"] == [1.0, 2.0]
    assert first_interval["left_ctc_ordinal"] == 0
    assert first_interval["right_ctc_ordinal"] == 1
    assert first_interval["excluded_duration"] == 0.8
    assert second_interval["excluded_duration"] == 0.4


def test_leading_and_trailing_silence_use_audio_axis_and_exact_owner():
    words = post.Tier("words", 0.0, 4.0, [
        post.Interval(0.0, 0.8, "<sp1>"),
        post.Interval(0.8, 1.4, "ni3"),
        post.Interval(1.4, 2.8, "hao3"),
        post.Interval(2.8, 4.0, "<sp1>"),
    ])
    source = [
        {"ordinal": 0, "start": 0.8, "end": 1.4, "text": "ni3"},
        {"ordinal": 1, "start": 1.4, "end": 2.8, "text": "hao3"},
    ]
    ctc = [
        {"ordinal": 0, "word": "ni3", "start_s": 1.0,
         "end_s": 1.4, "type": "word"},
        {"ordinal": 1, "word": "hao3", "start_s": 1.8,
         "end_s": 2.8, "type": "word"},
    ]
    ledger = post._fallback_lexical_correspondence_ledger(source, ctc, words)
    selection = post._fallback_bgm_ctc_gap_selection(
        words, source, ctc, ledger, (0.2, 3.2))

    leading, trailing = selection["evaluated_intervals"]
    assert leading["ctc_gap"] == [0.2, 1.0]
    assert leading["evaluated_intersection"] == [0.2, 0.8]
    assert leading["right_ctc_ordinal"] == 0
    assert trailing["ctc_gap"] == [2.8, 3.2]
    assert trailing["evaluated_intersection"] == [2.8, 3.2]
    assert trailing["left_ctc_ordinal"] == 1
    assert leading["excluded_duration"] == 0.2
    assert trailing["excluded_duration"] == 0.8


def test_missing_or_unsafe_proof_keeps_legacy_full_final_silence_scan():
    words, source, ctc, ledger = _fixture()
    unsafe = copy.deepcopy(ledger)
    unsafe["safe"] = False

    for proof in (None, unsafe, {"schema": post.FALLBACK_CORRESPONDENCE_SCHEMA}):
        selection = post._fallback_bgm_ctc_gap_selection(
            words, source, ctc, proof, (0.0, 3.0))
        item = selection["evaluated_intervals"][0]
        assert selection["selection_mode"] == "legacy_full_final_silence"
        assert item["selection_mode"] == "legacy_full_final_silence"
        assert item["evaluated_intersection"] == item["original_silence_span"]
        assert item["ctc_gap"] is None
        assert item["excluded_duration"] == 0.0
        assert selection["validation"]["status"] == "rejected"


def test_lexical_evidence_narrows_ctc_gap_before_bgm_scan():
    words = post.Tier("words", 0.0, 1.2, [
        post.Interval(0.0, 0.75, "<sp1>"),
        post.Interval(0.75, 1.2, "ni3"),
    ])
    source = [
        {"ordinal": 0, "start": 0.49, "end": 0.70, "text": "ni3"},
    ]
    ctc = [{"ordinal": 0, "word": "ni3", "start_s": 0.75,
            "end_s": 0.95, "type": "word"}]
    # The final silence is leading, so CTC alone would scan [0, 0.75].
    # Source MFA evidence proves lexical onset at 0.49 and excludes the
    # source-MFA/CTC disagreement from the BGM evidence window.
    ledger = post._fallback_lexical_correspondence_ledger(source, ctc, words)
    selection = post._fallback_bgm_ctc_gap_selection(
        words, source, ctc, ledger, (0.0, 1.2))

    item = selection["evaluated_intervals"][0]
    assert item["ctc_gap"] == [0.0, 0.75]
    assert item["lexical_evidence_gap"] == [0.0, 0.49]
    assert item["evaluated_intersection"] == [0.0, 0.49]
    assert item["lexical_exclusions"] == [[0.49, 0.75]]
    assert item["narrowing_basis"] == "lexical_evidence_gap"
