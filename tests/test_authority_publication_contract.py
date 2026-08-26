from pathlib import Path

import pytest

from scripts.audit_authority_ok100 import audit_run, load_selection
from scripts.postprocess_textgrids import (
    Interval,
    Tier,
    _inject_punctuation,
    _final_unexpected_silence_reasons,
    _is_substantive_interior_silence,
    _lexical_identity,
    _publication_contract_audit,
    _resolve_phone_owner_overlaps,
    _restore_reference_punctuation,
    assess_reference_coverage,
)


def _contract_fixture(*, include_comma=True, interior_silence=False,
                      ctc_end=0.8, english=False):
    if english:
        words = [Interval(0.0, 0.8, "app"), Interval(0.8, 1.0, "<sp1>")]
        hanzi = [Interval(0.0, 0.8, "APP"), Interval(0.8, 1.0, "<sp1>")]
        phones = Tier("phones", 0.0, 1.0, [
            Interval(0.0, 0.4, "en:AE1"),
            Interval(0.4, 0.8, "en:P"),
        ])
        pinyin_phones = Tier("pinyin_phones", 0.0, 1.0, [
            Interval(0.0, 0.4, "en:AE1"),
            Interval(0.4, 0.8, "en:P"),
        ])
        reference = "APP"
        source = [{"text": "app", "start": 0.0, "end": 0.8}]
        ctc = [{"type": "word", "word": "app", "start_s": 0.0,
                "end_s": ctc_end}]
        provenance = {"status": "verified", "verified_words": 1}
    else:
        if include_comma:
            words = [
                Interval(0.0, 0.3, "ni3"),
                Interval(0.3, 0.4, "，"),
                Interval(0.4, 0.8, "hao3"),
                Interval(0.8, 1.0, "<sp1>"),
            ]
            hanzi = [
                Interval(0.0, 0.3, "你"),
                Interval(0.3, 0.4, "，"),
                Interval(0.4, 0.8, "好"),
                Interval(0.8, 1.0, "<sp1>"),
            ]
            reference = "你，好"
            source = [
                {"text": "ni3", "start": 0.0, "end": 0.3},
                {"text": "hao3", "start": 0.4, "end": 0.8},
            ]
            ctc = [
                {"type": "word", "word": "ni3", "start_s": 0.0,
                 "end_s": 0.3},
                {"type": "punct", "word": "，", "start_s": 0.3,
                 "end_s": 0.4},
                {"type": "word", "word": "hao3", "start_s": 0.4,
                 "end_s": 0.8},
            ]
        else:
            words = [Interval(0.0, 0.4, "ni3"),
                     Interval(0.4, 0.8, "hao3"),
                     Interval(0.8, 1.0, "<sp1>")]
            hanzi = [Interval(0.0, 0.4, "你"),
                     Interval(0.4, 0.8, "好"),
                     Interval(0.8, 1.0, "<sp1>")]
            reference = "你好"
            source = [{"text": "ni3", "start": 0.0, "end": 0.4},
                      {"text": "hao3", "start": 0.4, "end": 0.8}]
            ctc = [{"type": "word", "word": "ni3", "start_s": 0.0,
                    "end_s": 0.4},
                   {"type": "word", "word": "hao3", "start_s": 0.4,
                    "end_s": 0.8}]
        phones = Tier("phones", 0.0, 1.0, [
            Interval(0.0, 0.3 if include_comma else 0.4, "n"),
            Interval(0.4, 0.8, "h"),
        ])
        pinyin_phones = Tier("pinyin_phones", 0.0, 1.0, [
            Interval(0.0, 0.3 if include_comma else 0.4, "n"),
            Interval(0.4, 0.8, "h"),
        ])
        provenance = None

    if interior_silence:
        words.insert(-1, Interval(0.8, 0.85, "<sp2>"))
        hanzi.insert(-1, Interval(0.8, 0.85, "<sp2>"))
        words[-1] = Interval(0.85, 1.0, "<sp1>")
        hanzi[-1] = Interval(0.85, 1.0, "<sp1>")
    return (Tier("words", 0.0, 1.0, words),
            Tier("hanzi", 0.0, 1.0, hanzi),
            pinyin_phones, phones, reference, source, ctc, provenance)


def _audit(fixture, *, authoritative=True):
    words, hanzi, pp, phones, reference, source, ctc, provenance = fixture
    return _publication_contract_audit(
        words, hanzi, pp, phones, reference, source, ctc,
        authoritative, provenance)


def test_exact_owner_partition_and_reference_sequence_is_publishable():
    reasons, details = _audit(_contract_fixture())
    assert reasons == []
    assert details["ctc_lexical_evidence_proof"]
    assert all(row["accepted"] for row in details["ctc_lexical_evidence_proof"])


def test_words_hanzi_mismatch_and_interior_silence_are_structured_vetoes():
    fixture = list(_contract_fixture(interior_silence=True))
    # Make the tested silence a genuine lexical-silence-lexical gap rather
    # than the tail-silence chain used by the fixture's other contract cases.
    fixture[0].intervals = [
        Interval(0.0, 0.3, "ni3"),
        Interval(0.3, 0.7, "<sp2>"),
        Interval(0.7, 1.0, "hao3"),
    ]
    fixture[1].intervals = [
        Interval(0.0, 0.29, "你"),
        Interval(0.3, 0.7, "<sp2>"),
        Interval(0.7, 1.0, "好"),
    ]
    reasons, details = _audit(tuple(fixture))
    assert "words_hanzi_bounds_mismatch" in reasons
    assert "strict_interior_sp" in reasons
    assert details["strict_interior_sp"][0]["label"] == "<sp2>"


@pytest.mark.parametrize(
    ("labels", "strict_expected"),
    [
        (["ni3", "<sp2>", "。"], False),
        (["。", "<sp2>", "ni3"], False),
        (["ni3", "<sp2>", "hao3"], True),
    ],
)
def test_strict_silence_requires_lexical_owner_on_both_sides(
        labels, strict_expected):
    intervals = [Interval(start, end, label)
                 for start, end, label in zip(
                     (0.0, 0.3, 0.6), (0.3, 0.6, 1.0), labels)]
    words = Tier("words", 0.0, 1.0, intervals)
    hanzi = Tier("hanzi", 0.0, 1.0,
                 [Interval(iv.xmin, iv.xmax, iv.text) for iv in intervals])
    phones = Tier("phones", 0.0, 1.0, [])
    pp = Tier("pinyin_phones", 0.0, 1.0, [])
    reasons, _details = _publication_contract_audit(
        words, hanzi, pp, phones, "你", [], [], False, None)
    assert ("strict_interior_sp" in reasons) is strict_expected


def test_final_unexpected_silence_rebuild_drops_punctuation_owned_stale_reason():
    tail_owned = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.6, "ni3"),
        Interval(0.6, 0.8, "<sp2>"),
        Interval(0.8, 1.0, "。"),
    ])
    lexical_gap = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"),
        Interval(0.3, 0.7, "<sp2>"),
        Interval(0.7, 1.0, "hao3"),
    ])
    assert _final_unexpected_silence_reasons(tail_owned) == []
    assert _final_unexpected_silence_reasons(lexical_gap) == [
        "unexpected_silence"]


@pytest.mark.parametrize(
    ("labels", "sp3_expected"),
    [
        (["ni3", "<sp3>", "。"], False),
        (["ni3", "<sp3>", "hao3"], True),
    ],
)
def test_sp3_uses_the_same_semantic_owner_gate(labels, sp3_expected):
    intervals = [Interval(start, end, label)
                 for start, end, label in zip(
                     (0.0, 0.3, 0.6), (0.3, 0.6, 1.0), labels)]
    assert _is_substantive_interior_silence(intervals, 1) is sp3_expected


def test_known_nvv_labels_share_identity_but_arbitrary_angle_text_does_not():
    assert _lexical_identity("breathing") == _lexical_identity("<BREATHING>")
    assert _lexical_identity("laughter") == _lexical_identity("<LAUGHTER>")
    assert _lexical_identity("foobar") != _lexical_identity("<FOOBAR>")


def test_processed_word_boundary_outside_source_evidence_is_advisory():
    fixture = list(_contract_fixture(english=True))
    fixture[0].intervals[0] = Interval(0.8, 1.0, "app")
    fixture[1].intervals[0] = Interval(0.8, 1.0, "APP")
    reasons, details = _audit(tuple(fixture))
    assert "ctc_lexical_boundary_outside_evidence" not in reasons
    assert details["ctc_lexical_evidence_proof"][0]["accepted"] is True


def test_ctc_authoritative_word_may_replace_far_mfa_span():
    """A CTC-owned word must not be rejected merely because MFA was late."""
    words = Tier("words", 0.0, 1.0, [Interval(0.50, 0.70, "ni3")])
    words._ctc_word_authority = [{
        "lexical_ordinal": 0,
        "text": "ni3",
        "boundary_source": "ctc",
        "ctc_span": [0.50, 0.70],
        "mfa_span": [0.05, 0.20],
        "resolved_span": [0.50, 0.70],
    }]
    hanzi = Tier("hanzi", 0.0, 1.0, [Interval(0.50, 0.70, "你")])
    phones = Tier("phones", 0.0, 1.0, [Interval(0.50, 0.70, "n")])
    pp = Tier("pinyin_phones", 0.0, 1.0, [Interval(0.50, 0.70, "n")])
    reasons, details = _publication_contract_audit(
        words, hanzi, pp, phones, "你",
        [{"text": "ni3", "start": 0.05, "end": 0.20}],
        [{"type": "word", "word": "ni3", "start_s": 0.50, "end_s": 0.70}],
        False, None)
    assert "ctc_lexical_boundary_outside_evidence" not in reasons
    proof = details["ctc_lexical_evidence_proof"][0]
    assert proof["ctc_authoritative"] is True
    assert proof["source_overlap_s"] == 0.0


def test_contract_consumes_unknown_resolution_and_normalizes_english_surface():
    fixture = list(_contract_fixture(english=True))
    fixture[0].intervals[0] = Interval(0.0, 0.8, "v-tuber")
    fixture[1].intervals[0] = Interval(0.0, 0.8, "V-TUBER")
    fixture[4] = "V-Tuber"
    fixture[5] = [{"ordinal": 0, "text": "<unk>", "start": 0.0, "end": 0.8}]
    fixture[6] = [{"type": "word", "word": "vtuber", "start_s": 0.0,
                   "end_s": 0.8}]
    proof = {"schema": "mfa-unknown-recovery-proof-v1",
             "source": {"interval": {"ordinal": 0}},
             "ctc": {"token": {"word": "vtuber"}}}
    reasons, details = _publication_contract_audit(
        fixture[0], fixture[1], fixture[2], fixture[3], fixture[4],
        fixture[5], fixture[6], True, fixture[7],
        unknown_recovery_proof=proof)
    assert "ctc_lexical_sequence_mismatch" not in reasons
    assert details["ctc_lexical_source_resolution"]["resolved_text"] == "vtuber"


def test_evidence_envelope_allows_source_extended_word_with_positive_ctc_overlap():
    fixture = list(_contract_fixture(english=True))
    fixture[0].intervals[0] = Interval(0.0, 0.85, "app")
    fixture[1].intervals[0] = Interval(0.0, 0.85, "APP")
    fixture[5] = [{"text": "app", "start": 0.0, "end": 0.85}]
    reasons, details = _audit(tuple(fixture))
    assert "ctc_lexical_boundary_outside_evidence" not in reasons
    proof = details["ctc_lexical_evidence_proof"][0]
    assert proof["ctc_overlap_s"] > 0
    assert proof["evidence_envelope"] == [0.0, 0.85]


def test_adjacent_evidence_envelopes_may_overlap_when_each_sequence_is_ordered():
    fixture = list(_contract_fixture(include_comma=False))
    fixture[0].intervals[0] = Interval(0.0, 0.58, "ni3")
    fixture[0].intervals[1] = Interval(0.58, 0.8, "hao3")
    fixture[1].intervals[0] = Interval(0.0, 0.58, "你")
    fixture[1].intervals[1] = Interval(0.58, 0.8, "好")
    fixture[3].intervals = [
        Interval(0.0, 0.3, "n"), Interval(0.3, 0.58, "i3"),
        Interval(0.58, 0.7, "h"), Interval(0.7, 0.8, "ao3"),
    ]
    fixture[2].intervals = [
        Interval(0.0, 0.3, "n"), Interval(0.3, 0.58, "i3"),
        Interval(0.58, 0.7, "h"), Interval(0.7, 0.8, "ao3"),
    ]
    fixture[5] = [
        {"text": "ni3", "start": 0.0, "end": 0.61},
        {"text": "hao3", "start": 0.57, "end": 0.8},
    ]
    fixture[6] = [
        {"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.61},
        {"type": "word", "word": "hao3", "start_s": 0.57, "end_s": 0.8},
    ]
    reasons, details = _audit(tuple(fixture))
    assert "ctc_lexical_boundary_outside_evidence" not in reasons
    assert all(row["final_word_order"] and row["source_order"]
               and row["ctc_order"]
               for row in details["ctc_lexical_evidence_proof"])


def test_missing_authority_or_source_evidence_is_not_guessed():
    fixture = list(_contract_fixture(include_comma=False))
    fixture[4] = ""
    fixture[5] = []
    fixture[6] = []
    reasons, _ = _audit(tuple(fixture), authoritative=False)
    assert "ctc_lexical_evidence_missing" in reasons


def test_reference_punctuation_with_malformed_source_span_is_removed():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.6, "hao3"),
        Interval(0.6, 1.0, "<sp1>"),
    ])
    result, _ = _inject_punctuation(
        words, None, [{"word": "，", "start_s": "bad", "end_s": 0.4}],
        reference_text="你，好", reference_authoritative=True)
    assert [iv.text for iv in result.intervals] == ["ni3", "hao3", "<sp1>"]


def test_missing_reference_punctuation_is_allowed_but_extra_is_not():
    fixture = list(_contract_fixture())
    # Remove the displayed comma while keeping the lexical geometry valid.
    fixture[0].intervals = [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.8, "hao3"),
        Interval(0.8, 1.0, "<sp1>"),
    ]
    fixture[1].intervals = [
        Interval(0.0, 0.3, "你"), Interval(0.3, 0.8, "好"),
        Interval(0.8, 1.0, "<sp1>"),
    ]
    reasons, details = _audit(tuple(fixture))
    assert "reference_punctuation_ownership_mismatch" not in reasons
    assert details["reference_punctuation_projection"]["missing_allowed"]

    extra = list(_contract_fixture(include_comma=False))
    extra[0].intervals.insert(1, Interval(0.4, 0.45, "，"))
    extra[1].intervals.insert(1, Interval(0.4, 0.45, "，"))
    reasons, _ = _audit(tuple(extra))
    assert "reference_punctuation_ownership_mismatch" in reasons


def test_missing_earlier_duplicate_punctuation_is_not_misclassified_as_wrong_boundary():
    """A retained later ``。`` must not be greedily matched to an earlier one."""
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.2, "ni3"),
        Interval(0.2, 0.4, "hao3"),
        Interval(0.4, 0.5, "。"),
        Interval(0.5, 0.8, "ma1"),
        Interval(0.8, 1.0, "<sp1>"),
    ])
    hanzi = Tier("hanzi", 0.0, 1.0, [
        Interval(0.0, 0.2, "你"),
        Interval(0.2, 0.4, "好"),
        Interval(0.4, 0.5, "。"),
        Interval(0.5, 0.8, "吗"),
        Interval(0.8, 1.0, "<sp1>"),
    ])
    phones = Tier("phones", 0.0, 1.0, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "h"),
        Interval(0.5, 0.8, "m"),
    ])
    pp = Tier("pinyin_phones", 0.0, 1.0, [
        Interval(0.0, 0.2, "n"), Interval(0.2, 0.4, "h"),
        Interval(0.5, 0.8, "m"),
    ])
    reasons, details = _publication_contract_audit(
        words, hanzi, pp, phones, "你。好。吗",
        [{"text": "ni3", "start": 0.0, "end": 0.2},
         {"text": "hao3", "start": 0.2, "end": 0.4},
         {"text": "ma1", "start": 0.5, "end": 0.8}],
        [{"type": "word", "word": "ni3", "start_s": 0.0, "end_s": 0.2},
         {"type": "word", "word": "hao3", "start_s": 0.2, "end_s": 0.4},
         {"type": "word", "word": "ma1", "start_s": 0.5, "end_s": 0.8}],
        True, None)
    projection = details["reference_punctuation_projection"]
    assert "reference_punctuation_ownership_mismatch" not in reasons
    assert [item["index"] for item in projection["missing_allowed"]] == [0]
    assert projection["boundary_errors"] == []


def test_restore_keeps_extra_wrong_boundary_punctuation_observable():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.2, "ni3"), Interval(0.2, 0.25, "！"),
        Interval(0.25, 0.4, "<sp1>"), Interval(0.4, 1.0, "hao3"),
    ])
    _restore_reference_punctuation(
        words, "你，好", [{"word": "，", "start_s": 0.25, "end_s": 0.4}])
    assert [iv.text for iv in words.intervals if iv.text in {"！", "，"}] == [
        "！", "，"]


def test_misplaced_reference_punctuation_is_not_treated_as_missing():
    fixture = list(_contract_fixture())
    fixture[0].intervals = [
        Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.8, "hao3"),
        Interval(0.8, 0.9, "，"), Interval(0.9, 1.0, "<sp1>"),
    ]
    fixture[1].intervals = [
        Interval(0.0, 0.3, "你"), Interval(0.3, 0.8, "好"),
        Interval(0.8, 0.9, "，"), Interval(0.9, 1.0, "<sp1>"),
    ]
    reasons, details = _audit(tuple(fixture))
    assert "reference_punctuation_ownership_mismatch" in reasons
    assert details["reference_punctuation_projection"]["boundary_errors"]


def test_phone_owner_is_clipped_only_for_a_clear_majority():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.5, "ni3"), Interval(0.5, 1.0, "hao3")])
    clear = Tier("phones", 0.0, 1.0, [Interval(0.45, 0.65, "i3")])
    fixed, ambiguous = _resolve_phone_owner_overlaps(clear, words)
    assert fixed == 1
    assert ambiguous == []
    assert (clear.intervals[0].xmin, clear.intervals[0].xmax) == (0.5, 0.65)

    tie = Tier("phones", 0.0, 1.0, [Interval(0.45, 0.55, "i3")])
    fixed, ambiguous = _resolve_phone_owner_overlaps(tie, words)
    assert fixed == 0
    assert ambiguous[0]["reason"] == "owner_tie_or_weak_majority"


def test_reference_coverage_allows_missing_punctuation_but_rejects_replacement():
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.5, "ni3"), Interval(0.5, 1.0, "hao3")])
    hanzi = Tier("hanzi", 0.0, 1.0, [
        Interval(0.0, 0.5, "你"), Interval(0.5, 1.0, "好")])
    coverage, reasons = assess_reference_coverage(
        "你，好", words, hanzi, reference_source="original_or_ref",
        reference_authoritative=True)
    assert "reference_semantic_sequence_mismatch" not in reasons
    assert coverage["exact_semantic_sequence"] is True

    replaced = Tier("hanzi", 0.0, 1.0, [
        Interval(0.0, 0.4, "你"), Interval(0.4, 0.5, "。"),
        Interval(0.5, 1.0, "好")])
    _, reasons = assess_reference_coverage(
        "你，好", words, replaced, reference_source="original_or_ref",
        reference_authoritative=True)
    assert "reference_semantic_sequence_mismatch" in reasons


def test_ok100_audit_reports_per_stem_and_conservation_failure(tmp_path: Path):
    selection_path = Path(__file__).parents[1] / "configs" / \
        "hecheng_ria_ok100_authority.selection.json"
    selection = load_selection(selection_path)
    assert len(selection["stems"]) == 100
    result = audit_run(selection_path, tmp_path / "fresh", tmp_path / "evidence")
    assert result["count"] == 100
    assert result["missing"] == 100
    assert result["conservation"] is False
    assert result["reference_ok_all"] is False
    assert result["ok"] is False
    assert len(result["records"]) == 100
    assert all(row["verdict"] == "missing" for row in result["records"])


def test_ok100_audit_rechecks_reference_token_boundary(tmp_path: Path):
    selection_path = Path(__file__).parents[1] / "configs" / \
        "hecheng_ria_ok100_authority.selection.json"
    stem = load_selection(selection_path)["stems"][0]
    ctc_dir = tmp_path / "evidence" / "ctc_pretg"
    ctc_dir.mkdir(parents=True)
    (ctc_dir / f"{stem}_ref.txt").write_text(
        "not okayish", encoding="utf-8")
    result = audit_run(selection_path, tmp_path / "fresh", tmp_path / "evidence")
    row = next(item for item in result["records"] if item["stem"] == stem)
    assert "reference_ok_missing" in row["reasons"]
    assert row["reference_ok"] is False
    assert row["ok"] is False
