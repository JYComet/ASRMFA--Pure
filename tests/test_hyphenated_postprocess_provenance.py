"""Synthetic strict-en-mfa-v2 canaries for the 012871 K-Pop owner."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_strict_ok as audit
from scripts import postprocess_textgrids as post


def test_strict_english_provenance_does_not_require_global_strict_ok():
    assert post._strict_en_provenance_enabled(SimpleNamespace(
        strict_ok=False, strict_en_provenance=True))
    assert post._strict_en_provenance_enabled(SimpleNamespace(
        strict_ok=True, strict_en_provenance=False))
    assert not post._strict_en_provenance_enabled(SimpleNamespace(
        strict_ok=False, strict_en_provenance=False))


def _ledger_fixture(tmp_path: Path, *, schema: str = post.STRICT_EN_MFA_SCHEMA,
                    phones: list[dict] | None = None,
                    phone_start: float = 0.0) -> tuple[Path, dict]:
    stem = "012871"
    source = tmp_path / "012871_seg0.TextGrid"
    source.write_text("synthetic MFA evidence; no MFA process\n", encoding="utf-8")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    phones = phones if phones is not None else [{
        "ordinal": 0, "mfa_phone_ordinal": 0, "label": "K",
        "start": phone_start, "end": phone_start + 0.4,
    }]
    record = {
        "word_id": "012871:s0:w0", "unit_id": "en-u0000",
        "ctc_ordinal": 0, "source_ctc_ordinals": [0],
        "ctc_text": "kpop", "alignment_token": "kpop",
        "canonical_span": [4, 9],
        "canonical_binding": post.CANONICAL_UNITS_SCHEMA,
        "status": "verified", "provenance": "english_mfa_textgrid",
        "mfa_word": {"ordinal": 0, "text": "kpop", "start": 0.0, "end": 0.4},
        "phones": phones,
    }
    ledger = {
        "schema": schema, "stem": stem,
        "canonical_units": post.CANONICAL_UNITS_SCHEMA,
        "segments": [{
            "segment_id": "012871:s0", "segment_ordinal": 0,
            "status": "verified",
            "mfa_textgrid": {"path": str(source), "sha256": source_sha},
            "words": [record],
        }],
    }
    ledger_path = tmp_path / "012871_en_phones.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    manifest = {
        "schema": schema, "strict_provenance": True,
        "canonical_units": post.CANONICAL_UNITS_SCHEMA,
        "status": "success", "expected_segments": ["012871:s0"],
        "produced_segments": ["012871:s0"], "rejected_segments": [],
        "stem_ledgers": [{"stem": stem, "path": str(ledger_path),
                           "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest()}],
    }
    (tmp_path / "en_alignment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return ledger_path, record


def test_012871_canary_restores_one_surface_owner_without_mfa(tmp_path):
    _ledger_fixture(tmp_path)
    words = post.Tier("words", 0.0, 0.4, [post.Interval(0.0, 0.4, "KPop")])
    hanzi = post.Tier("hanzi", 0.0, 0.4, [post.Interval(0.0, 0.4, "KPop")])

    assert post._restore_reference_surfaces(words, hanzi, "你好K-Pop") == ["en-u0000"]
    assert [iv.text for iv in words.intervals] == ["K-Pop"]
    assert [iv.text for iv in hanzi.intervals] == ["K-Pop"]
    report, pairs = post.load_strict_en_provenance(
        "012871", words, tmp_path, hanzi_tier=hanzi, reference_text="你好K-Pop")
    assert report["status"] == "verified"
    assert len(pairs) == 1


def test_fallback_english_projects_processed_raw_ordinal_to_compact_ctc_axis(
        tmp_path):
    """Processed CTC and MFA-source raw interval ordinals are different axes."""
    ledger_path, record = _ledger_fixture(tmp_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    record = ledger["segments"][0]["words"][0]
    record["ctc_ordinal"] = 7
    record["source_ctc_ordinals"] = [7]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    manifest_path = tmp_path / "en_alignment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stem_ledgers"][0]["sha256"] = hashlib.sha256(
        ledger_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    words = post.Tier("words", 0.0, 0.4, [
        post.Interval(0.0, 0.4, "K-Pop"),
    ])
    correspondence = {
        "safe": True,
        "mapping": {
            # The English producer's raw processed-CTC interval 7 projects to
            # compact CTC lexical ordinal 0.  ``source_to_final`` belongs to
            # the MFA source tier and must not be consulted here.
            "ctc_to_final": {"0": 0},
            "source_to_final": {"7": 9},
        },
    }
    processed_ctc_words = post.Tier("words", 0.0, 0.8, [
        *[post.Interval(index / 10, (index + 1) / 10, "")
          for index in range(7)],
        post.Interval(0.7, 0.8, "kpop"),
    ])
    report, pairs = post.load_strict_en_provenance(
        "012871", words, tmp_path,
        ctc_tokens=[{"type": "word", "word": "kpop"}],
        correspondence=correspondence,
        processed_ctc_words_tier=processed_ctc_words)
    assert report["status"] == "verified"
    assert len(pairs) == 1


def test_012871_split_authoritative_compound_is_rejected(tmp_path):
    _ledger_fixture(tmp_path)
    words = post.Tier("words", 0.0, 0.5, [
        post.Interval(0.0, 0.2, "kp"), post.Interval(0.2, 0.4, "op")])
    report, _ = post.load_strict_en_provenance(
        "012871", words, tmp_path, reference_text="你好K-Pop")
    assert report["status"] == "rejected"
    assert report["reason"] == "strict_en_authoritative_compound_split"


def test_authority_alpha_digit_fragments_share_words_and_hanzi_owner():
    words = post.Tier("words", 0.0, 0.6, [
        post.Interval(0.0, 0.2, "target"),
        post.Interval(0.2, 0.3, "1"),
        post.Interval(0.3, 0.6, "OK"),
    ])
    ctc = [
        {"type": "word", "word": "target", "start_s": 0.0,
         "end_s": 0.2, "source_ctc_ordinal": 0},
        {"type": "word", "word": "1", "start_s": 0.2,
         "end_s": 0.3, "source_ctc_ordinal": 1},
        {"type": "word", "word": "OK", "start_s": 0.3,
         "end_s": 0.6, "source_ctc_ordinal": 2},
    ]
    assert post._merge_authority_alpha_digit_fragments(
        words, "target1 OK", ctc) == ["en-u0000"]
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.3, "target1"), (0.3, 0.6, "OK")
    ]
    hanzi = post._build_hanzi_tier(words, "target1 OK", reference_authoritative=True)
    assert [iv.text for iv in hanzi.intervals] == ["target1", "OK"]


def test_authority_target_numerals_project_to_cjk_pinyin_not_target_digit_english():
    words = post.Tier("words", 0.0, 0.7, [
        post.Interval(0.0, 0.2, "target"),
        post.Interval(0.2, 0.3, "1"),
        post.Interval(0.3, 0.5, "target"),
        post.Interval(0.5, 0.6, "2"),
        post.Interval(0.6, 0.7, "OK"),
    ])
    ctc = [
        {"type": "word", "word": text, "start_s": start, "end_s": end}
        for text, start, end in (
            ("target", 0.0, 0.2), ("1", 0.2, 0.3),
            ("target", 0.3, 0.5), ("2", 0.5, 0.6),
            ("OK", 0.6, 0.7))
    ]
    report = {}
    post._merge_authority_alpha_digit_fragments(
        words, "target一 target二 OK", ctc, report=report)

    assert [iv.text for iv in words.intervals] == [
        "target", "yi1", "target", "er4", "OK"]
    assert report["authority_compound_reconciliation"]["numeral_fragments"][0][
        "surface"] == "一"
    hanzi = post._build_hanzi_tier(
        words, "target一 target二 OK", reference_authoritative=True)
    assert [iv.text for iv in hanzi.intervals] == [
        "target", "一", "target", "二", "OK"]
    assert post._reference_pinyin_text(
        "target一 target二", "target1 target2") == (
            "<sp1> target yi1 target er4")


def test_authority_target_numerals_inside_braces_keep_semantic_projection():
    """Placeholder braces are syntax and must not block 一/二 ownership."""
    words = post.Tier("words", 0.0, 0.7, [
        post.Interval(0.0, 0.2, "target"),
        post.Interval(0.2, 0.3, "yi1"),
        post.Interval(0.3, 0.5, "he2"),
        post.Interval(0.5, 0.6, "target"),
        post.Interval(0.6, 0.7, "er4"),
    ])
    hanzi = post._build_hanzi_tier(
        words, "{target一}和{target二}", reference_authoritative=True)
    assert [iv.text for iv in hanzi.intervals] == [
        "target", "一", "和", "target", "二"
    ]


def test_authority_fragments_without_ordinals_preserve_interleaved_cjk_and_units():
    words = post.Tier("words", 0.0, 0.5, [
        post.Interval(0.0, 0.1, "target"),
        post.Interval(0.1, 0.2, "1"),
        post.Interval(0.2, 0.3, "he2"),
        post.Interval(0.3, 0.4, "target"),
        post.Interval(0.4, 0.5, "2"),
    ])
    ctc = [
        {"type": "word", "word": "target", "start_s": 0.0, "end_s": 0.1},
        {"type": "word", "word": "1", "start_s": 0.1, "end_s": 0.2},
        {"type": "word", "word": "he2", "start_s": 0.2, "end_s": 0.3},
        {"type": "word", "word": "target", "start_s": 0.3, "end_s": 0.4},
        {"type": "word", "word": "2", "start_s": 0.4, "end_s": 0.5},
    ]

    restored = post._merge_authority_alpha_digit_fragments(
        words, "target1 和 target2", ctc)

    assert restored == ["en-u0000", "en-u0001"]
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.2, "target1"), (0.2, 0.3, "he2"), (0.3, 0.5, "target2")]
    assert words._canonical_authority_units[0]["source_ctc_indices"] == [0, 1]
    assert words._canonical_authority_units[1]["unit_id"] != words._canonical_authority_units[0]["unit_id"]


def test_authority_alpha_digit_five_ctc_fragments_merge_without_count_limit():
    words = post.Tier("words", 0.0, 0.6, [
        post.Interval(0.00, 0.10, "t"),
        post.Interval(0.10, 0.20, "ar"),
        post.Interval(0.20, 0.30, "ge"),
        post.Interval(0.30, 0.40, "t"),
        post.Interval(0.40, 0.45, "1"),
        post.Interval(0.45, 0.60, "OK"),
    ])

    ctc = [
        {"type": "word", "word": text, "start_s": start,
         "end_s": end, "source_ctc_ordinal": ordinal}
        for ordinal, (text, start, end) in enumerate((
            ("t", 0.00, 0.10), ("ar", 0.10, 0.20),
            ("ge", 0.20, 0.30), ("t", 0.30, 0.40),
            ("1", 0.40, 0.45), ("OK", 0.45, 0.60)))
    ]
    assert post._merge_authority_alpha_digit_fragments(
        words, "target1 OK", ctc) == ["en-u0000"]
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.45, "target1"), (0.45, 0.60, "OK")
    ]


def test_authority_merge_rebuilds_over_evidence_covered_sp_only():
    words = post.Tier("words", 0.0, 0.5, [
        post.Interval(0.00, 0.10, "target1"),
        post.Interval(0.10, 0.20, "ar"),
        post.Interval(0.20, 0.30, "ge"),
        post.Interval(0.30, 0.40, "t"),
        post.Interval(0.40, 0.41, "<sp0>"),
        post.Interval(0.41, 0.50, "1"),
    ])
    ctc = [
        {"type": "word", "word": "t", "start_s": 0.00,
         "end_s": 0.10, "source_ctc_ordinal": 0},
        {"type": "word", "word": "ar", "start_s": 0.10,
         "end_s": 0.20, "source_ctc_ordinal": 1},
        {"type": "word", "word": "ge", "start_s": 0.20,
         "end_s": 0.30, "source_ctc_ordinal": 2},
        # This source token covers the displayed 10 ms <sp0> residual.
        {"type": "word", "word": "t", "start_s": 0.30,
         "end_s": 0.41, "source_ctc_ordinal": 3},
        {"type": "word", "word": "1", "start_s": 0.41,
         "end_s": 0.50, "source_ctc_ordinal": 4},
    ]

    assert post._merge_authority_alpha_digit_fragments(
        words, "target1", ctc) == ["en-u0000"]
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.5, "target1")
    ]


def test_authority_merge_keeps_long_uncovered_sp_as_hard_boundary():
    words = post.Tier("words", 0.0, 0.7, [
        post.Interval(0.00, 0.10, "target"),
        post.Interval(0.10, 0.50, "<sp0>"),
        post.Interval(0.50, 0.60, "1"),
    ])
    ctc = [
        {"type": "word", "word": "target", "start_s": 0.00,
         "end_s": 0.10, "source_ctc_ordinal": 0},
        {"type": "word", "word": "1", "start_s": 0.50,
         "end_s": 0.60, "source_ctc_ordinal": 1},
    ]

    assert post._merge_authority_alpha_digit_fragments(
        words, "target1", ctc) == []


def test_authority_alpha_digit_fragments_do_not_cross_punctuation_or_pinyin():
    words = post.Tier("words", 0.0, 0.7, [
        post.Interval(0.0, 0.2, "target"),
        post.Interval(0.2, 0.3, ","),
        post.Interval(0.3, 0.4, "1"),
        post.Interval(0.4, 0.7, "jin1"),
    ])
    assert post._merge_authority_alpha_digit_fragments(words, "target1 jin1") == []
    assert [iv.text for iv in words.intervals] == ["target", ",", "1", "jin1"]


def test_012871_missing_phone_and_span_drift_fail_closed(tmp_path):
    for phones, expected in (
            ([], "strict_en_word_identity_or_evidence_invalid"),
            ([{"ordinal": 0, "mfa_phone_ordinal": 0, "label": "K",
              "start": -0.010, "end": 0.4}], "strict_en_phone_invalid")):
        _ledger_fixture(tmp_path, phones=phones)
        words = post.Tier("words", 0.0, 0.4, [post.Interval(0.0, 0.4, "K-Pop")])
        report, _ = post.load_strict_en_provenance("012871", words, tmp_path)
        assert report["status"] == "rejected"
        assert report["reason"] == expected
        for path in tmp_path.iterdir():
            if path.is_file():
                path.unlink()


def test_012871_legacy_v1_is_not_v2_success(tmp_path):
    _ledger_fixture(tmp_path, schema=post.HISTORICAL_STRICT_EN_MFA_SCHEMA)
    words = post.Tier("words", 0.0, 0.4, [post.Interval(0.0, 0.4, "K-Pop")])
    report, _ = post.load_strict_en_provenance("012871", words, tmp_path)
    assert report["reason"] == "strict_en_manifest_legacy_schema"

    manifest, reasons = audit._load_english_manifest(type("Args", (), {
        "en_phones_dir": tmp_path,
        "en_manifest": "en_alignment_manifest.json",
    })())
    assert manifest is None
    assert reasons == ["english_provenance_legacy_schema"]


def test_partial_manifest_is_partitioned_per_stem(tmp_path):
    good_ledger, _ = _ledger_fixture(tmp_path)
    bad_ledger_path = tmp_path / "bad_en_phones.json"
    bad_ledger_path.write_text(json.dumps({
        "schema": post.STRICT_EN_MFA_SCHEMA,
        "stem": "bad",
        "canonical_units": post.CANONICAL_UNITS_SCHEMA,
        "segments": [{
            "segment_id": "bad:s0", "segment_ordinal": 0,
            "status": "rejected", "reason": "segment_too_short",
            "words": [{"word_id": "bad:s0:w0"}],
        }],
    }), encoding="utf-8")
    good_hash = hashlib.sha256(good_ledger.read_bytes()).hexdigest()
    bad_hash = hashlib.sha256(bad_ledger_path.read_bytes()).hexdigest()
    manifest = {
        "schema": post.STRICT_EN_MFA_SCHEMA,
        "strict_provenance": True,
        "canonical_units": post.CANONICAL_UNITS_SCHEMA,
        "status": "partial",
        "expected_segments": ["012871:s0", "bad:s0"],
        "produced_segments": ["012871:s0"],
        "rejected_segments": [{"id": "bad:s0", "reason": "segment_too_short"}],
        "stem_ledgers": [
            {"stem": "012871", "path": str(good_ledger), "sha256": good_hash},
            {"stem": "bad", "path": str(bad_ledger_path), "sha256": bad_hash},
        ],
        "mfa": {"return_code": 0, "timed_out": False, "exception": "",
                "command": [], "timeout_seconds": 1,
                "acoustic_model_sha256": "0" * 64,
                "dictionary_sha256": "1" * 64},
        "counts": {"english_stems": 2, "english_segments": 2,
                   "english_words": 2, "verified_words": 1,
                   "rejected_words": 1},
    }
    (tmp_path / "en_alignment_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")

    good_words = post.Tier("words", 0.0, 0.4,
                           [post.Interval(0.0, 0.4, "K-Pop")])
    bad_words = post.Tier("words", 0.0, 0.4,
                          [post.Interval(0.0, 0.4, "K-Pop")])
    good_report, good_pairs = post.load_strict_en_provenance(
        "012871", good_words, tmp_path)
    bad_report, bad_pairs = post.load_strict_en_provenance(
        "bad", bad_words, tmp_path)

    assert good_report["status"] == "verified"
    assert len(good_pairs) == 1
    assert bad_report["status"] == "rejected"
    assert bad_report["reason"] == "strict_en_segment_rejected"
    assert bad_report["failed_word_ids"] == ["bad:s0:w0"]
    assert bad_pairs == []

    audit_args = type("Args", (), {
        "en_phones_dir": tmp_path,
        "en_manifest": "en_alignment_manifest.json",
    })()
    loaded, reasons = audit._load_english_manifest(audit_args)
    assert reasons == []
    assert loaded["status"] == "partial"
    fake_tg = type("TextGrid", (), {"tiers": [None, None, None, bad_words]})()
    audit_reasons, _ = audit._english_provenance_reasons(
        "bad", fake_tg, tmp_path, audit_args, loaded)
    assert audit_reasons == ["english_segment_rejected"]
