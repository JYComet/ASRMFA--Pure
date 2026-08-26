"""No-reference lexical correspondence and ordinal ownership contracts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import postprocess_textgrids as post


def _source(*labels):
    return [{"ordinal": i, "start": float(i), "end": float(i + 1), "text": label}
            for i, label in enumerate(labels)]


def _ctc(*labels):
    return [{"word": label, "start_s": float(i), "end_s": float(i + 1),
             "type": "word"} for i, label in enumerate(labels)]


def _final(*labels):
    return post.Tier(
        "words", 0.0, float(len(labels)),
        [post.Interval(float(i), float(i + 1), label)
         for i, label in enumerate(labels)],
    )


def _ledger(source, ctc, final):
    return post._fallback_lexical_correspondence_ledger(source, ctc, final)


def test_ledger_is_complete_stable_and_digest_bound():
    source = _source("<unk>", "<unk>")
    ctc = _ctc("SURPRISE-WA", "BREATHING")
    final = _final("SURPRISE-WA", "BREATHING")

    first = _ledger(source, ctc, final)
    second = _ledger(source, ctc, final)

    assert first == second
    assert first["safe"] is True
    assert first["status"] == "mapped"
    assert (first["source_count"], first["ctc_count"], first["final_count"]) == (2, 2, 2)
    assert len(first["entries"]) == 2
    assert first["omissions"] == []
    assert first["first_mismatch"] is None
    assert first["digest"] == post._evidence_digest(
        {key: value for key, value in first.items() if key != "digest"})
    assert [entry["status"] for entry in first["entries"]] == ["mapped", "mapped"]


def test_multiple_unknowns_require_exact_same_nvv_at_each_ordinal():
    positive = _ledger(
        _source("<unk>", "<unk>"),
        _ctc("SURPRISE-WA", "BREATHING"),
        _final("SURPRISE-WA", "BREATHING"),
    )
    assert positive["safe"] is True
    assert [entry["resolved_text"] for entry in positive["entries"]] == [
        "SURPRISE-WA", "BREATHING"]

    # LAria_00160: the second unknown cannot consume the first repeated final
    # owner when CTC proves BREATHING at that ordinal.
    negative = _ledger(
        _source("<unk>", "<unk>"),
        _ctc("SURPRISE-WA", "BREATHING"),
        _final("SURPRISE-WA", "SURPRISE-WA"),
    )
    assert negative["safe"] is False
    assert negative["status"] == "rejected"
    assert negative["first_mismatch"]["reason"] == "substitution"
    assert negative["entries"][1]["status"] == "mismatch"


def test_restore_unknowns_consumes_distinct_ordered_ctc_ordinals_and_spans():
    words = _final("<unk>", "<unk>")
    restored = post._restore_fallback_unknown_surfaces(words, [
        {"ordinal": 0, "word": "SURPRISE-WA", "start_s": 0.20,
         "end_s": 1.50, "type": "word"},
        {"ordinal": 1, "word": "BREATHING", "start_s": 1.00,
         "end_s": 2.40, "type": "word"},
    ])

    assert [iv.text for iv in restored.intervals] == [
        "SURPRISE-WA", "BREATHING"]
    projection = restored._fallback_unknown_projection
    recovered = [entry["recovery_evidence"] for entry in projection["entries"]]
    assert [item["ctc_ordinal"] for item in recovered] == [0, 1]
    assert [item["ctc_span"] for item in recovered] == [
        [0.2, 1.5], [1.0, 2.4]]


def test_restore_known_omission_is_explicit_and_does_not_shift_unknown_owner():
    words = _final("a", "<unk>", "c")
    restored = post._restore_fallback_unknown_surfaces(words, [
        {"ordinal": 0, "word": "BREATHING", "start_s": 1.0,
         "end_s": 2.0, "type": "word"},
        {"ordinal": 1, "word": "c", "start_s": 2.0,
         "end_s": 3.0, "type": "word"},
    ])

    assert [iv.text for iv in restored.intervals] == ["a", "BREATHING", "c"]
    projection = restored._fallback_unknown_projection
    assert projection["safe"] is True
    assert projection["entries"][0]["status"] == "omitted"
    assert projection["entries"][0]["omission_evidence"]["source_identity"] == "a"
    assert projection["entries"][1]["ctc_ordinal"] == 0
    assert projection["entries"][2]["ctc_ordinal"] == 1


def test_restore_rejected_projection_is_transactional_for_invalid_or_ambiguous_evidence():
    cases = [
        # Unknown may not consume a Chinese/pinyin target.
        [{"word": "ni3", "start_s": 0.0, "end_s": 1.0, "type": "word"}],
        # Known source reorder cannot be repaired by time overlap.
        [{"word": "b", "start_s": 0.0, "end_s": 1.0, "type": "word"},
         {"word": "a", "start_s": 1.0, "end_s": 2.0, "type": "word"}],
        # Two identical known-source owners admit two omission/mapping paths.
        [{"word": "a", "start_s": 0.0, "end_s": 1.0, "type": "word"}],
        # Missing end_s is malformed evidence, even when overlap is available.
        [{"word": "BREATHING", "start_s": 0.0, "type": "word"}],
    ]
    source_labels = ["<unk>", "a", "a", "<unk>"]
    for labels, ctc in zip(
            (("<unk>",), ("a", "b"), ("a", "a"), ("<unk>",)), cases):
        words = _final(*labels)
        before = [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals]
        restored = post._restore_fallback_unknown_surfaces(words, ctc)
        assert [(iv.xmin, iv.xmax, iv.text) for iv in restored.intervals] == before
        assert restored._fallback_unknown_projection["safe"] is False


def test_reorder_substitution_final_only_and_malformed_are_rejected():
    cases = [
        (_source("A", "B"), _ctc("B", "A"), _final("B", "A")),
        (_source("A"), _ctc("A"), _final("B")),
        (_source("A"), _ctc("A"), _final("A", "B")),
        (_source("A"), [{"word": "A", "start_s": 0.0, "type": "word"}], _final("A")),
    ]
    for source, ctc, final in cases:
        ledger = _ledger(source, ctc, final)
        assert ledger["safe"] is False
        assert ledger["status"] == "rejected"
        assert ledger["first_mismatch"] is not None


def test_safe_source_omission_does_not_shift_later_phone_owner():
    source_words = post.Tier("words", 0.0, 3.0, [
        post.Interval(0.0, 1.0, "a"),
        post.Interval(1.0, 2.0, "b"),
        post.Interval(2.0, 3.0, "c"),
    ])
    source_phones = post.Tier("phones", 0.0, 3.0, [
        post.Interval(0.1, 0.9, "A_PHONE"),
        post.Interval(1.1, 1.9, "OMITTED_PHONE"),
        post.Interval(2.1, 2.9, "C_PHONE"),
    ])
    source_snapshot = [
        {"ordinal": i, "start": iv.xmin, "end": iv.xmax, "text": iv.text}
        for i, iv in enumerate(source_words.intervals)
    ]
    ledger = _ledger(source_snapshot, _ctc("a", "c"), _final("a", "c"))
    assert ledger["safe"] is True
    lineage = post._bind_source_phone_lineage(source_words, source_phones, ledger)
    final_words = _final("a", "c")
    rebuilt = post._rebuild_phones_from_lineage(final_words, source_phones, lineage)

    assert rebuilt is not None
    assert [phone.text for phone in rebuilt.intervals] == ["A_PHONE", "C_PHONE"]


def test_wide_english_anchor_cannot_relabel_or_swallow_chinese_pinyin():
    words = _final("yang4", "ria")
    restored = post._restore_fallback_unknown_surfaces(words, [{
        "word": "ria", "start_s": 0.0, "end_s": 2.0, "type": "word",
    }])

    assert [(iv.xmin, iv.xmax, iv.text) for iv in restored.intervals] == [
        (0.0, 1.0, "yang4"), (1.0, 2.0, "ria"),
    ]


def test_explicit_unknown_can_still_restore_nvv_from_ctc():
    words = _final("<unk>", "hao3")
    restored = post._restore_fallback_unknown_surfaces(words, [{
        "word": "BREATHING", "start_s": 0.0, "end_s": 1.0,
        "type": "word",
    }])

    assert [iv.text for iv in restored.intervals] == ["BREATHING", "hao3"]


def test_exact_unknown_english_owner_requires_verified_strict_provenance():
    correspondence = _ledger(
        _source("<unk>", "wo3"), _ctc("ria", "wo3"),
        _final("ria", "wo3"))
    assert correspondence["safe"] is True

    assert post._fallback_redeemed_unknown_entries(
        correspondence, {"status": "rejected"}) == []
    recovered = post._fallback_redeemed_unknown_entries(
        correspondence, {"status": "verified"})
    assert len(recovered) == 1
    assert recovered[0]["resolved_text"] == "ria"
    assert recovered[0]["recovery_target_kind"] == "strict_english"
    assert recovered[0]["recovery_provenance"] == \
        "exact_correspondence+strict_english_ledger"
