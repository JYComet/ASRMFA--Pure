"""Contract tests for immutable CTC raw inputs and processed publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_strict_ok as audit  # noqa: E402
from scripts import adjust_ctc_boundaries as adjust  # noqa: E402
from scripts import ctc_prealign as ctc  # noqa: E402
from scripts import postprocess_textgrids as post  # noqa: E402
from scripts import run_pipeline  # noqa: E402
from scripts import streaming_pipeline as streaming  # noqa: E402
from scripts.pipeline_utils import (  # noqa: E402
    CTC_SUFFIXES,
    CTC_RAW_MANIFEST_NAME,
    CTC_WORK_RECEIPT_NAME,
    materialize_ctc_work,
    validate_ctc_raw_manifest,
    validate_ctc_work_receipt,
    validate_ctc_authority_bundle,
    make_pipeline_accounting_receipt,
    make_pipeline_resume_fingerprints,
    validate_pipeline_resume_receipt,
    write_ctc_raw_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_fixture(root: Path, stem: str = "demo") -> tuple[Path, Path, dict]:
    raw = root / "raw"
    raw.mkdir(parents=True)
    for index, suffix in enumerate(CTC_SUFFIXES):
        (raw / f"{stem}{suffix}").write_text(
            f"{stem}:{suffix}:{index}\n", encoding="utf-8")
    producer = raw / ".ctc_run_receipt.json"
    producer.write_text(json.dumps({"schema": "ctc-run-receipt-v2", "stem": stem}),
                        encoding="utf-8")
    (raw / f"{stem}_ref.txt").write_text("你，好\n", encoding="utf-8")
    manifest = write_ctc_raw_manifest(raw, producer_receipt=producer, stems=[stem])
    return raw, raw / CTC_RAW_MANIFEST_NAME, manifest


def test_raw_manifest_is_immutable_and_work_is_physical_six_artifact_copy(tmp_path):
    raw, manifest_path, manifest = _raw_fixture(tmp_path)
    before = {path.name: _sha256(path) for path in raw.iterdir()
              if path.is_file()}

    work = tmp_path / "workspace" / "ctc_pretg_adj"
    receipt = materialize_ctc_work(raw, work, raw_manifest_path=manifest_path)

    after = {path.name: _sha256(path) for path in raw.iterdir()
             if path.is_file()}
    assert before == after
    assert _sha256(manifest_path) == before[manifest_path.name]
    assert validate_ctc_raw_manifest(raw) == []
    assert validate_ctc_work_receipt(work, manifest_path, receipt) == []

    assert {row["suffix"] for row in receipt["files"]} == set(CTC_SUFFIXES)
    assert len(receipt["files"]) == len(CTC_SUFFIXES)
    assert [row["suffix"] for row in receipt["ref"]] == ["_ref.txt"]
    assert receipt["transform_ledger"][0]["operation"] == "physical_copy"
    for row in receipt["files"] + receipt["ref"]:
        source = raw / row["name"]
        target = work / row["name"]
        assert target.is_file() and not target.is_symlink()
        assert source.read_bytes() == target.read_bytes()
        assert not os.path.samestat(source.stat(), target.stat())


def test_adjust_disabled_still_binds_independent_work_directory(monkeypatch, tmp_path):
    raw, manifest_path, _ = _raw_fixture(tmp_path)
    work = tmp_path / "workspace" / "ctc_pretg_adj"
    called = []
    monkeypatch.setattr(run_pipeline, "run_python",
                        lambda *args, **kwargs: called.append(args) or 0)
    ctx = {
        "ctc_pretg": raw,
        "ctc_pretg_adj": work,
        "ctc_raw_manifest": manifest_path,
        "audio_dir": tmp_path / "audio",
        "tts_authoritative_audio_dir": tmp_path / "padded_audio",
        "models_dir": tmp_path / "models",
    }
    result = run_pipeline.step_adjust_ctc(
        SimpleNamespace(overwrite=True), {"ctc_adjust": {"enabled": False}},
        tmp_path / "python", ctx)

    assert result == 0
    assert "--geometry-only" in called[0][1]
    assert called[0][1][called[0][1].index("--audio-dir") + 1] == str(
        tmp_path / "padded_audio")
    assert work.resolve() != raw.resolve()
    assert work.is_dir() and work != raw
    receipt = json.loads((work / CTC_WORK_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["work_root"] == str(work.resolve())
    assert receipt["transform_ledger"][-1]["stage"] == "adjust"
    assert receipt["transform_ledger"][-1]["status"] == "geometry_only"
    assert receipt["transform_ledger"][-1]["work_files_digest"]
    assert validate_ctc_work_receipt(work, manifest_path) == []
    assert validate_ctc_raw_manifest(raw) == []


def test_raw_digest_mismatch_and_mixed_cache_restore_fail_closed(tmp_path):
    source_root = tmp_path / "source"
    raw_one, manifest_one, _ = _raw_fixture(source_root, "one")
    work_one = tmp_path / "work-one"
    materialize_ctc_work(raw_one, work_one, raw_manifest_path=manifest_one)

    nas = tmp_path / "nas"
    shutil.copytree(work_one, nas / "ctc_pretg_adj")
    local = tmp_path / "local"
    raw_two, manifest_two, _ = _raw_fixture(local, "two")
    local_workspace = tmp_path / "local-workspace"
    shutil.copytree(raw_two, local_workspace / "ctc_pretg")

    assert streaming._restore_ctc_adj_cache(
        local_workspace, nas, raw_manifest_path=manifest_two) is False
    assert not (local_workspace / "ctc_pretg_adj").exists()

    # Even a cache with the right path cannot be used after producer bytes
    # change under the sealed raw manifest.
    (raw_two / "two.lab").write_text("tampered\n", encoding="utf-8")
    assert validate_ctc_raw_manifest(raw_two)
    assert streaming._restore_ctc_adj_cache(
        local_workspace, nas, raw_manifest_path=manifest_two) is False
    assert not (local_workspace / "ctc_pretg_adj").exists()


def test_processed_authority_geometry_is_required_without_mutating_raw_unit(tmp_path):
    row = ctc._merge_reference_english_fragments(
        [{"word": "K", "start": 0.06, "end": 0.12,
          "source_ctc_ordinal": 0}], "K")[0]
    raw_span = list(row["canonical_span"])
    raw_hash = row["canonical_unit_sha256"]
    row["start"], row["end"] = 0.06, 0.42
    row["processed_ctc_span"] = [0.06, 0.42]
    row["processed_ctc_boundary_source"] = "next_lexical_token_start"
    ctc.write_textgrid([row], 1.0, tmp_path / "demo.TextGrid")
    (tmp_path / "demo.lab").write_text("k\n", encoding="utf-8")
    (tmp_path / "demo_ref.txt").write_text("K\n", encoding="utf-8")
    (tmp_path / "demo_tokens.jsonl").write_text(json.dumps({
        "word": "k", "start_s": 0.06, "end_s": 0.42,
        "canonical_span": raw_span, "canonical_unit": row["canonical_unit"],
        "canonical_unit_sha256": raw_hash,
        "surface_text": "K", "source_ctc_ordinals": [0],
        "reference_identity": hashlib.sha256(b"K").hexdigest(),
        "reference_ordinal": 0,
        "processed_ctc_span": [0.06, 0.42],
        "processed_ctc_boundary_source": "next_lexical_token_start",
    }) + "\n", encoding="utf-8")

    assert validate_ctc_authority_bundle(tmp_path, "demo", "K") == []
    rows = [json.loads(line) for line in
            (tmp_path / "demo_tokens.jsonl").read_text().splitlines()]
    del rows[0]["processed_ctc_span"]
    (tmp_path / "demo_tokens.jsonl").write_text(
        json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert validate_ctc_authority_bundle(tmp_path, "demo", "K")
    assert row["canonical_span"] == raw_span
    assert row["canonical_unit_sha256"] == raw_hash


def test_processed_span_shorter_than_canonical_end_is_rejected_with_context(tmp_path):
    row = ctc._merge_reference_english_fragments(
        [{"word": "K", "start": 0.06, "end": 0.42,
          "source_ctc_ordinal": 0}], "K")[0]
    raw_span = list(row["canonical_span"])
    raw_hash = row["canonical_unit_sha256"]
    row["start"], row["end"] = 0.06, 0.10
    row["processed_ctc_span"] = [0.06, 0.10]
    row["processed_ctc_boundary_source"] = "raw_end_fallback"
    ctc.write_textgrid([row], 1.0, tmp_path / "demo.TextGrid")
    (tmp_path / "demo.lab").write_text("k\n", encoding="utf-8")
    (tmp_path / "demo_ref.txt").write_text("K\n", encoding="utf-8")
    (tmp_path / "demo_tokens.jsonl").write_text(json.dumps({
        "word": "k", "start_s": 0.06, "end_s": 0.10,
        "canonical_span": raw_span, "canonical_unit": row["canonical_unit"],
        "canonical_unit_sha256": raw_hash,
        "surface_text": "K", "source_ctc_ordinals": [0],
        "reference_identity": hashlib.sha256(b"K").hexdigest(),
        "reference_ordinal": 0,
        "processed_ctc_span": [0.06, 0.10],
        "processed_ctc_boundary_source": "raw_end_fallback",
    }) + "\n", encoding="utf-8")

    errors = validate_ctc_authority_bundle(tmp_path, "demo", "K")
    assert any("reason_code=processed_geometry_end_before_canonical_end" in error
               and "stem=demo" in error and "unit=en-u0000" in error
               and "scale=1.0" in error for error in errors)
    assert row["canonical_span"] == raw_span
    assert row["canonical_unit_sha256"] == raw_hash


@pytest.mark.parametrize("value, reason", [
    ([0.06, "bad"], "processed_geometry_non_numeric"),
    (None, "processed_geometry_missing"),
])
def test_processed_geometry_failure_reason_codes_are_stable(tmp_path, value, reason):
    row = ctc._merge_reference_english_fragments(
        [{"word": "K", "start": 0.06, "end": 0.42,
          "source_ctc_ordinal": 0}], "K")[0]
    row["start"], row["end"] = 0.06, 0.42
    row["processed_ctc_span"] = value
    row["processed_ctc_boundary_source"] = "raw_end_fallback"
    ctc.write_textgrid([row], 1.0, tmp_path / "demo.TextGrid")
    (tmp_path / "demo.lab").write_text("k\n", encoding="utf-8")
    (tmp_path / "demo_ref.txt").write_text("K\n", encoding="utf-8")
    token = {
        "word": "k", "start_s": 0.06, "end_s": 0.42,
        "canonical_span": row["canonical_span"],
        "canonical_unit": row["canonical_unit"],
        "canonical_unit_sha256": row["canonical_unit_sha256"],
        "surface_text": "K", "source_ctc_ordinals": [0],
        "reference_identity": hashlib.sha256(b"K").hexdigest(),
        "reference_ordinal": 0,
        "processed_ctc_span": value,
        "processed_ctc_boundary_source": "raw_end_fallback",
    }
    (tmp_path / "demo_tokens.jsonl").write_text(
        json.dumps(token, ensure_ascii=False) + "\n", encoding="utf-8")
    errors = validate_ctc_authority_bundle(tmp_path, "demo", "K")
    assert any(f"reason_code={reason}" in error and "stem=demo" in error
               and "unit=en-u0000" in error and "scale=1.0" in error
               for error in errors)


def test_legacy_accounting_receipt_is_audit_only_and_not_resumable():
    legacy = {"schema": "pipeline-run-receipt-v1"}
    assert validate_pipeline_resume_receipt(legacy) == [
        "legacy_receipt_not_resumable"]

    fingerprints = make_pipeline_resume_fingerprints(
        producer={"sha256": "producer"}, effective_config={"sha256": "config"},
        dependencies={"sha256": "deps"}, inputs={"sha256": "inputs"},
        expected_stems={"sha256": "stems"}, outputs={"sha256": "outputs"})
    receipt = make_pipeline_accounting_receipt(
        ["demo"], ["demo"], [], ["demo"], [], fingerprints=fingerprints)
    assert validate_pipeline_resume_receipt(receipt) == []
    without_fingerprints = dict(receipt)
    without_fingerprints.pop("fingerprints")
    assert "fresh_resume_fingerprints_missing" in validate_pipeline_resume_receipt(
        without_fingerprints)


def test_raw_authority_bundle_allows_missing_processed_geometry_until_adjust(tmp_path):
    row = ctc._merge_reference_english_fragments(
        [{"word": "K", "start": 0.06, "end": 0.12,
          "source_ctc_ordinal": 0}], "K")[0]
    bundle = tmp_path
    ctc.write_textgrid([row], 1.0, bundle / "demo.TextGrid")
    (bundle / "demo.lab").write_text("k\n", encoding="utf-8")
    (bundle / "demo_ref.txt").write_text("K\n", encoding="utf-8")
    path = bundle / "demo_tokens.jsonl"
    path.write_text(json.dumps({
        "word": "k", "start_s": 0.06, "end_s": 0.12,
        "canonical_span": row["canonical_span"],
        "canonical_unit": row["canonical_unit"],
        "canonical_unit_sha256": row["canonical_unit_sha256"],
        "surface_text": "K", "source_ctc_ordinals": [0],
        "reference_identity": hashlib.sha256(b"K").hexdigest(),
        "reference_ordinal": 0,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    assert validate_ctc_authority_bundle(
        bundle, "demo", "K", require_processed=False) == []
    assert validate_ctc_authority_bundle(bundle, "demo", "K")


def test_adjusted_words_tier_keeps_punctuation_out_of_lexical_ctc_words(tmp_path):
    source = tmp_path / "raw.TextGrid"
    source.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n'
        'xmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n'
        'item [1]:\nclass = "IntervalTier"\nname = "words"\n'
        'xmin = 0\nxmax = 1\nintervals: size = 1\nintervals [1]:\n'
        'xmin = 0\nxmax = 1\ntext = "k"\n'
        'item [2]:\nclass = "IntervalTier"\nname = "pauses"\n'
        'xmin = 0\nxmax = 1\nintervals: size = 1\nintervals [1]:\n'
        'xmin = 0.4\nxmax = 0.7\ntext = "300ms"\n',
        encoding="utf-8")
    out = tmp_path / "processed.TextGrid"
    adjust.rebuild_textgrid(
        source, out,
        [{"word": "k", "start_s": 0.06, "end_s": 0.4}],
        [{"word": "，", "start_s": 0.4, "end_s": 0.7}],
    )
    text = out.read_text(encoding="utf-8")
    assert 'text = "，"' not in text.split('name = "words"', 1)[1].split(
        'name = "pauses"', 1)[0]
    assert 'text = "k"' in text


def test_adjust_stage_creates_processed_span_in_work_without_mutating_raw(tmp_path):
    raw = tmp_path / "raw"
    work = tmp_path / "work"
    audio_dir = tmp_path / "audio"
    raw.mkdir()
    audio_dir.mkdir()
    wav_path = audio_dir / "demo.wav"
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)

    english = ctc._merge_reference_english_fragments(
        [{"word": "K", "start": 0.06, "end": 0.12,
          "source_ctc_ordinal": 0}], "K")[0]
    following = {"word": "hao3", "start": 0.42, "end": 0.55}
    ctc.write_textgrid([english, following], 1.0, raw / "demo.TextGrid")
    (raw / "demo.lab").write_text("k hao3\n", encoding="utf-8")
    raw_tokens = [
        {"word": "k", "start_s": 0.06, "end_s": 0.12,
         "canonical_span": english["canonical_span"],
         "canonical_unit": english["canonical_unit"],
         "canonical_unit_sha256": english["canonical_unit_sha256"],
         "surface_text": "K", "source_ctc_ordinals": [0],
         "reference_identity": hashlib.sha256(b"K").hexdigest(),
         "reference_ordinal": 0},
        {"word": "hao3", "start_s": 0.42, "end_s": 0.55},
    ]
    (raw / "demo_tokens.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in raw_tokens) + "\n",
        encoding="utf-8")

    result = adjust.process_one("demo", raw, audio_dir, work)

    assert "error" not in result
    assert "processed_ctc_span" not in raw_tokens[0]
    processed = [json.loads(line) for line in
                 (work / "demo_tokens.jsonl").read_text().splitlines()]
    assert processed[0]["processed_ctc_span"] == [0.06, 0.42]
    assert processed[0]["end_s"] == 0.42


def test_adjust_cache_does_not_accept_raw_copy_as_processed(tmp_path):
    stem = "demo"
    token_dir = tmp_path / "work"
    token_dir.mkdir()
    token_path = token_dir / f"{stem}_tokens.jsonl"
    token = {
        "word": "k",
        "canonical_span": [0.06, 0.12],
        "canonical_unit": {"canonical_span": [0.06, 0.12]},
    }
    token_path.write_text(json.dumps(token) + "\n", encoding="utf-8")
    assert not adjust._processed_geometry_cache_complete(token_dir, {stem})

    token["processed_ctc_span"] = [0.06, 0.42]
    token["processed_ctc_boundary_source"] = "next_lexical_token_start"
    token_path.write_text(json.dumps(token) + "\n", encoding="utf-8")
    assert adjust._processed_geometry_cache_complete(token_dir, {stem})

@pytest.mark.parametrize("pause", ["<sp1>", "<sp2>", "<sp3>"])
def test_substantive_unowned_pause_remains_filterable(pause):
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "ni3"),
        post.Interval(0.2, 0.5, pause),
        post.Interval(0.5, 1.0, "hao3"),
    ])
    phones = post.Tier("phones", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "n"), post.Interval(0.2, 0.5, "sil"),
        post.Interval(0.5, 1.0, "h"),
    ])
    pp = post.Tier("pinyin_phones", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "n"), post.Interval(0.2, 0.5, "sil"),
        post.Interval(0.5, 1.0, "h"),
    ])
    grid = post.TextGrid(0.0, 1.0, [words, phones, pp])

    reasons = post.handle_unexpected_silences(grid, "ni3 hao3")

    assert reasons == ["unexpected_silence"]
    assert [(iv.text, iv.xmin, iv.xmax) for iv in words.intervals] == [
        ("ni3", 0.0, 0.2), (pause, 0.2, 0.5), ("hao3", 0.5, 1.0)]


def test_sp0_merge_survives_sync_barrier_and_textgrid_roundtrip(tmp_path):
    words = post.Tier("words", 0.0, 0.8, [
        post.Interval(0.0, 0.2, "ni3"),
        post.Interval(0.2, 0.21, "<sp0>"),
        post.Interval(0.21, 0.8, "hao3"),
    ])
    words._ctc_word_authority = [
        {"lexical_ordinal": 0, "text": "ni3", "boundary_source": "ctc",
         "resolved_span": [0.0, 0.2]},
        {"lexical_ordinal": 1, "text": "hao3", "boundary_source": "ctc",
         "resolved_span": [0.21, 0.8]},
    ]
    phones = post.Tier("phones", 0.0, 0.8, [
        post.Interval(0.0, 0.2, "n"), post.Interval(0.2, 0.21, "sil"),
        post.Interval(0.21, 0.8, "h"),
    ])
    pp = post.Tier("pinyin_phones", 0.0, 0.8, [
        post.Interval(0.0, 0.2, "n"), post.Interval(0.2, 0.21, "sil"),
        post.Interval(0.21, 0.8, "h"),
    ])
    grid = post.TextGrid(0.0, 0.8, [words, phones, pp])

    assert post.handle_unexpected_silences(grid, "ni3 hao3") == []
    post._sync_derived_tiers(grid, {}, {}, "你好", report_warnings=[])
    frozen, freeze_reasons = post._freeze_processed_geometry(grid)
    assert freeze_reasons == []
    assert frozen is not None
    # A real 10 ms visual gap is preserved by derived sync; only the final
    # energy owner resolver may consume it.
    assert any(iv.text == "<sp0>" for iv in frozen.intervals)
    digest = post._processed_geometry_digest(frozen)
    assert grid._processed_geometry_digest == digest

    path = tmp_path / "processed.TextGrid"
    post.write_textgrid(grid, path)
    loaded = post.parse_textgrid(path)
    assert post._processed_geometry_digest(post.tier_by_name(loaded, "words")) == digest
    assert any(iv.text == "<sp0>"
               for iv in post.tier_by_name(loaded, "words").intervals)


@pytest.mark.parametrize("layout", ["punct-sp", "sp-punct"])
def test_punctuation_owns_adjacent_pause_in_both_orders(layout):
    if layout == "punct-sp":
        intervals = [
            post.Interval(0.0, 0.3, "ni3"),
            post.Interval(0.3, 0.4, "，"),
            post.Interval(0.4, 0.6, "<sp2>"),
            post.Interval(0.6, 1.0, "hao3"),
        ]
    else:
        intervals = [
            post.Interval(0.0, 0.3, "ni3"),
            post.Interval(0.3, 0.5, "<sp2>"),
            post.Interval(0.5, 0.6, "，"),
            post.Interval(0.6, 1.0, "hao3"),
        ]
    words = post.Tier("words", 0.0, 1.0, intervals)
    phones = post.Tier("phones", 0.0, 1.0, [
        post.Interval(0.3, 0.5, "sil")])
    pp = post.Tier("pinyin_phones", 0.0, 1.0, [
        post.Interval(0.3, 0.5, "sil")])
    grid = post.TextGrid(0.0, 1.0, [words, phones, pp])

    post.absorb_silence_into_punct(grid)

    punct = next(iv for iv in words.intervals if iv.text == "，")
    assert (punct.xmin, punct.xmax) == ((0.3, 0.6) if layout == "punct-sp"
                                        else (0.3, 0.6))
    assert not any(post.is_silence(iv.text) for iv in words.intervals)


def test_published_geometry_digest_and_spans_are_authoritative_over_resolved_span(tmp_path):
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.4, "ni3"),
        post.Interval(0.4, 0.6, "，"),
        post.Interval(0.6, 1.0, "hao3"),
    ])
    words._ctc_word_authority = [
        {"lexical_ordinal": 0, "text": "ni3", "boundary_source": "ctc",
         "resolved_span": [0.8, 0.9], "ctc_span": [0.8, 0.9]},
        {"lexical_ordinal": 1, "text": "hao3", "boundary_source": "ctc",
         "resolved_span": [0.1, 0.2], "ctc_span": [0.1, 0.2]},
    ]
    grid = post.TextGrid(0.0, 1.0, [words])
    frozen, reasons = post._freeze_processed_geometry(grid)

    assert reasons == []
    assert frozen is words
    assert post._processed_geometry_digest(words) == words._processed_geometry_digest
    assert [entry["published_span"] for entry in words._ctc_word_authority] == [
        [0.0, 0.4], [0.6, 1.0]]
    assert [entry["resolved_span"] for entry in words._ctc_word_authority] == [
        [0.8, 0.9], [0.1, 0.2]]

    # The independent audit accepts the final publication spans and rejects
    # a legacy proof that offers only resolved_span as publication authority.
    ledger = list(words._processed_geometry_ledger)
    row = {
        "stem": "demo", "status": "ok",
        "processed_geometry_digest": words._processed_geometry_digest,
        "processed_operation_ledger": ledger,
        "processed_geometry": {
            "schema": "processed-words-geometry-v1", "frozen": True,
            "digest": words._processed_geometry_digest, "ledger": ledger,
        },
        "publication_contract": {
            "status": "verified", "details": {
                "ctc_lexical_evidence_proof": [
                    {"published_span": [0.0, 0.4]},
                    {"published_span": [0.6, 1.0]},
                ]
            }
        },
    }
    assert audit._postprocess_contract_reasons(row, grid, None) == []
    legacy = json.loads(json.dumps(row))
    legacy["publication_contract"]["details"]["ctc_lexical_evidence_proof"] = [
        {"resolved_span": [0.8, 0.9]}, {"resolved_span": [0.1, 0.2]}]
    legacy_reasons = audit._postprocess_contract_reasons(legacy, grid, None)
    assert all(reason.startswith("processed_published_span_missing:")
               for reason in legacy_reasons)

    path = tmp_path / "processed.TextGrid"
    post.write_textgrid(grid, path)
    loaded = post.parse_textgrid(path)
    assert post._processed_geometry_digest(post.tier_by_name(loaded, "words")) == \
        words._processed_geometry_digest


def test_report_lifecycle_fields_and_legacy_authority_fields_are_compatible(
        monkeypatch, tmp_path):
    raw, manifest_path, _ = _raw_fixture(tmp_path)
    work = tmp_path / "work"
    materialize_ctc_work(raw, work, raw_manifest_path=manifest_path)
    monkeypatch.setenv("CTC_RAW_MANIFEST", str(manifest_path))
    monkeypatch.setenv("CTC_WORK_RECEIPT", str(work / CTC_WORK_RECEIPT_NAME))

    lifecycle = post._load_ctc_lifecycle(work, "demo")
    assert lifecycle["schema"] == "ctc-processed-input-lifecycle-v1"
    assert set(lifecycle) == {"schema", "raw_manifest", "work_receipt", "stem"}
    assert set(lifecycle["raw_manifest"]) == {"path", "sha256", "identity"}
    assert set(lifecycle["work_receipt"]) == {
        "path", "sha256", "identity", "lineage_entries"}

    # These are historical report/authority fields.  They remain accepted as
    # bookkeeping while the new lifecycle contract is independently checked.
    legacy_report = {
        "stem": "demo", "status": "ok", "output": str(tmp_path / "out.TextGrid"),
        "reference_mode": "authority",
        "reference_source": "original_or_ref",
        "reference_text_authoritative": True,
        "ctc_word_authority": [{"resolved_span": [0.0, 0.1]}],
    }
    # ctc_word_authority is not a report-positive field; it is old evidence,
    # and must not make a valid authority report fail merely because new
    # lifecycle keys are absent.
    assert "report_positive:ctc_word_authority" not in audit._report_reasons(
        {key: value for key, value in legacy_report.items()
         if key != "ctc_word_authority"})
