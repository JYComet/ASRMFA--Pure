"""Fail-closed LAria fallback pause-veto qualification contracts."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import postprocess_textgrids as post


def _fixture(*labels):
    labels = labels or ("ni3", "<sp2>", "hao3")
    words = post.Tier(
        "words", 0.0, float(len(labels)),
        [post.Interval(float(i), float(i + 1), label)
         for i, label in enumerate(labels)],
    )
    hanzi = post.Tier(
        "hanzi", 0.0, float(len(labels)),
        [post.Interval(iv.xmin, iv.xmax, iv.text) for iv in words.intervals],
    )
    phones = post.Tier("phones", 0.0, float(len(labels)), [
        post.Interval(0.0, 1.0, "n"),
        post.Interval(2.0, 3.0, "h"),
    ])
    pinyin_phones = post.Tier(
        "pinyin_phones", 0.0, float(len(labels)), [
            post.Interval(0.0, 1.0, "n"),
            post.Interval(2.0, 3.0, "h"),
        ],
    )
    source = [{"ordinal": i, "start": float(i), "end": float(i + 1),
               "text": label}
              for i, label in enumerate(labels)]
    ctc = []
    for i, label in enumerate(labels):
        if not post.is_silence(label):
            ctc.append({"ordinal": len(ctc), "word": label,
                        "start_s": float(i), "end_s": float(i + 1),
                        "type": "word"})
    ledger = post._fallback_lexical_correspondence_ledger(source, ctc, words)
    return words, hanzi, pinyin_phones, phones, source, ctc, ledger


_DEFAULT_LEDGER = object()


def _publication(fixture, *, reference_mode="fallback", authoritative=False,
                 ledger=_DEFAULT_LEDGER):
    words, hanzi, pp, phones, source, ctc, built = fixture
    reasons, details = post._publication_contract_audit(
        words, hanzi, pp, phones, "ni3 hao3", source, ctc,
        authoritative, None,
        fallback_correspondence=built if ledger is _DEFAULT_LEDGER else ledger,
        reference_mode=reference_mode,
    )
    return reasons, details


def test_fallback_pause_correspondence_never_redeems_pause_vetoes():
    for pause in ("<sp0>", "<sp1>", "<sp2>"):
        fixture = _fixture("ni3", pause, "hao3")
        gate = post._fallback_pause_qualification(
            fixture[0], "fallback", fixture[6], fixture[4], fixture[5])
        assert gate["all_qualified"] is True
        assert gate["details"][0]["label"] == pause
        assert gate["details"][0]["qualified"] is True

        reasons, details = _publication(fixture)
        assert "strict_interior_sp" in reasons
        assert details["strict_interior_sp"]
        assert details["fallback_pause_qualification"]["pause_count"] == 1

        filtered = post._apply_fallback_pause_veto_qualification(
            ["mid_sp", "strict_interior_sp", "unexpected_silence", "bgm_suspect"],
            gate,
        )
        assert filtered == ["mid_sp", "strict_interior_sp",
                            "unexpected_silence", "bgm_suspect"]


def test_leading_sp1_reindex_does_not_invalidate_exact_lexical_ledger():
    """A display-only leading owner may reindex, never change lexical proof."""
    words = post.Tier("words", 0.0, 3.0, [
        post.Interval(0.40, 1.0, "ni3"),
        post.Interval(1.0, 2.0, "<sp2>"),
        post.Interval(2.0, 3.0, "BREATHING"),
    ])
    raw_text = post.Tier("raw_text", 0.0, 3.0, [
        post.Interval(0.0, 3.0, "你好"),
    ])
    textgrid = post.TextGrid(0.0, 3.0, [raw_text, words])
    source = [
        {"ordinal": 0, "start": 0.40, "end": 1.0, "text": "ni3"},
        {"ordinal": 1, "start": 1.0, "end": 2.0, "text": "<sp2>"},
        {"ordinal": 2, "start": 2.0, "end": 3.0, "text": "BREATHING"},
    ]
    ctc = [
        {"ordinal": 0, "word": "ni3", "start_s": 0.40,
         "end_s": 1.0, "type": "word"},
        {"ordinal": 1, "word": "BREATHING", "start_s": 2.0,
         "end_s": 3.0, "type": "word"},
    ]

    ledger = post._fallback_lexical_correspondence_ledger(
        source, ctc, post.tier_by_name(textgrid, "words"))
    post._finalize_textgrid(textgrid)
    final_words = post.tier_by_name(textgrid, "words")
    assert [iv.text for iv in final_words.intervals] == [
        "<sp1>", "ni3", "<sp2>", "<BREATHING>"]
    gate = post._fallback_pause_qualification(
        final_words, "fallback", ledger, source, ctc)
    assert gate["ledger"]["status"] == "verified"
    assert gate["ledger"]["final_interval_reindexed"] is True
    assert gate["ledger"]["final_surface_normalized"] is True
    assert gate["all_qualified"] is True


def test_authority_equivalent_pause_is_not_redeemed():
    fixture = _fixture()
    reasons, details = _publication(fixture, reference_mode="authority",
                                    authoritative=True)
    assert "strict_interior_sp" in reasons
    assert details["strict_interior_sp"]
    gate = details["fallback_pause_qualification"]
    assert gate["all_qualified"] is False
    assert "reference_mode_not_fallback" in gate["details"][0]["qualification_reasons"]


def test_sp3_remains_a_veto_even_with_exact_ctc_gap_support():
    fixture = _fixture("ni3", "<sp3>", "hao3")
    gate = post._fallback_pause_qualification(
        fixture[0], "fallback", fixture[6], fixture[4], fixture[5])
    assert gate["all_qualified"] is True
    assert gate["details"][0]["ctc_gap_evidence"]["pause_coverage"] == 1.0
    assert post._apply_fallback_pause_veto_qualification(
        ["mid_sp", "strict_interior_sp", "unexpected_silence", "sp3"], gate
    ) == ["mid_sp", "strict_interior_sp", "unexpected_silence", "sp3"]

    unsupported = _fixture("ni3", "<sp3>", "hao3")
    unsupported[5][0]["end_s"] = 1.8
    unsupported[5][1]["start_s"] = 2.0
    unsupported_ledger = post._fallback_lexical_correspondence_ledger(
        unsupported[4], unsupported[5], unsupported[0])
    gate = post._fallback_pause_qualification(
        unsupported[0], "fallback", unsupported_ledger,
        unsupported[4], unsupported[5])
    assert gate["all_qualified"] is False
    assert "sp3_ctc_gap_not_supported" in gate["details"][0]["qualification_reasons"]


def test_missing_unsafe_and_malformed_ledgers_fail_closed():
    fixture = _fixture()
    for ledger in (None, {"schema": post.FALLBACK_CORRESPONDENCE_SCHEMA},
                   {**fixture[6], "safe": False}):
        reasons, _details = _publication(fixture, ledger=ledger)
        assert "strict_interior_sp" in reasons

    malformed = copy.deepcopy(fixture[6])
    malformed["entries"][0]["final_text"] = "tampered"
    reasons, details = _publication(fixture, ledger=malformed)
    assert "strict_interior_sp" in reasons
    assert "digest_mismatch" in details["fallback_pause_qualification"]["ledger"]["reasons"]


def test_nonleading_pure_silence_is_a_final_publication_veto():
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "ni3"),
        post.Interval(0.2, 0.8, "<sp2>"),
        post.Interval(0.8, 1.0, "hao3"),
    ])
    details = post._published_nonleading_silence_details(words)
    assert len(details) == 1
    assert details[0]["label"] == "<sp2>"

    leading = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.1, "<sp1>"),
        post.Interval(0.1, 0.5, "ni3"),
        post.Interval(0.5, 1.0, "hao3"),
    ])
    assert post._published_nonleading_silence_details(leading) == []


@pytest.mark.parametrize("label", ("<SP0>", "<sP1>", "<Sp2>", "<SP3>",
                                    "sp0", "sp1", "sp2", "sp3"))
def test_uppercase_canonical_and_bare_sp_are_final_nonleading_vetoes(label):
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "ni3"),
        post.Interval(0.2, 0.8, label),
        post.Interval(0.8, 1.0, "hao3"),
    ])

    details = post._published_nonleading_silence_details(words)
    assert len(details) == 1
    assert details[0]["label"] == label
    assert details[0]["reason"] == "nonleading_pure_silence_owner"


def test_unqualified_silence_and_mixed_pause_do_not_get_global_redemption():
    fixture = _fixture("ni3", "<sp2>", "hao3", "<sil>", "ma1")
    # Qualification is retained as evidence, but cannot redeem any retained
    # substantive pause or silence veto.
    words = fixture[0]
    gate = post._fallback_pause_qualification(
        words, "fallback", fixture[6], fixture[4], fixture[5])
    assert gate["qualified_indices"] == [1]
    assert gate["unqualified_indices"] == [3]
    assert gate["reason_qualification"]["unexpected_silence"] is True
    assert gate["reason_qualification"]["strict_interior_sp"] is False
    assert post._apply_fallback_pause_veto_qualification(
        ["mid_sp", "strict_interior_sp", "unexpected_silence", "short_word"], gate
    ) == ["mid_sp", "strict_interior_sp", "unexpected_silence", "short_word"]


def test_laria_expectation_is_static_only():
    # Packet expectation from the supplied audit: 671 candidates, 11 expected
    # to remain after this narrow pause-veto correction.  This is not a claim
    # about a production run or a measured result.
    expected_candidates = 671
    expected_qualified_pause_veto_outputs = 11
    assert (expected_candidates, expected_qualified_pause_veto_outputs) == (671, 11)
