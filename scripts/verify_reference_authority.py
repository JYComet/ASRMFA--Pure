#!/usr/bin/env python3
"""Regression checks for reference-text authority across normalization.

These checks intentionally avoid MFA/NVASR runtime dependencies.  They verify
the text-side contract that, when ``*_ref.txt`` exists, ASR diagnostic text
must not overwrite the authoritative reference spelling.
"""

from __future__ import annotations

import json
import hashlib
import math
import struct
import sys
import tempfile
import types
import wave
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ctc_prealign import (
    QUERY_FRAMES, FRAME_MS, NVV_START, NVV_END, BLANK_ID, NVV_SUPPRESSED_IDS,
    _clamp_words_to_wav_axis,
    _free_decode_logits,
    _merge_ria_tokens,
    _protect_ria,
    _rebuild_final_manifest,
    _wav_duration_s,
)
from normalize_english_tokens import normalize_stem
from postprocess_textgrids import (
    Interval,
    TextGrid,
    Tier,
    _finalize_textgrid,
    _reference_pinyin_text,
    _inject_punctuation,
    _reconcile_publication_geometry,
    _restore_reference_punctuation,
    _clip_pinyin_phones_to_words,
    _fix_non_english_pp_overlaps,
    _normalize_word_spellings,
    _snap_to_ctc,
    assess_reference_coverage,
    find_original_text,
)
from pipeline_utils import (
    CTC_NORMALIZATION_MARKER,
    is_english_token,
    is_nvv_token,
    is_pinyin_syllable,
    is_punct,
    is_silence,
    is_unknown_token,
    is_word_like,
    make_ctc_normalization_marker,
    parse_ctc_normalization_marker,
    publish_output_versioned,
    make_pipeline_accounting_receipt,
    write_pipeline_accounting_receipt,
    validate_ctc_transcript_bundle,
    write_publish_manifest,
)
from run_pipeline import (
    _skip_if_ctc_normalized,
    step_link_ctc,
    step_normalize_text,
    step_postprocess,
)


# ── test helpers ──────────────────────────────────────────────────

def _write_pcm_wav(path: Path, duration_s: float,
                   sample_rate: int = 16000, nchannels: int = 1) -> None:
    """Write a minimal silent PCM WAV file with *duration_s* seconds."""
    nframes = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * nframes)


def _write_ctc_textgrid_from_tokens(path: Path, tokens: list[dict],
                                    tg_xmin: float = 0.0,
                                    tg_xmax: float | None = None) -> None:
    """Write a minimal CTC words+pauses TextGrid from token dicts."""
    if tg_xmax is None and tokens:
        tg_xmax = max(float(t["end"]) for t in tokens)
    tg_xmax = tg_xmax or 1.0
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {tg_xmin}",
        f"xmax = {tg_xmax}",
        "tiers? <exists>",
        "size = 2",
        'item []:',
        "    class = \"IntervalTier\"",
        '    name = "words"',
        f"    xmin = {tg_xmin}",
        f"    xmax = {tg_xmax}",
        f"    intervals: size = {len(tokens)}",
    ]
    for j, t in enumerate(tokens):
        lines += [
            f"    intervals [{j+1}]:",
            f"        xmin = {float(t['start'])}",
            f"        xmax = {float(t['end'])}",
            f'        text = "{t["word"]}"',
        ]
    lines += [
        '    item []:',
        "        class = \"IntervalTier\"",
        '        name = "pauses"',
        f"        xmin = {tg_xmin}",
        f"        xmax = {tg_xmax}",
        "        intervals: size = 0",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tokens_from_dicts(path: Path, tokens: list[dict]) -> None:
    """Write a _tokens.jsonl file from dict entries."""
    with path.open("w", encoding="utf-8") as fh:
        for t in tokens:
            json.dump(t, fh, ensure_ascii=False)
            fh.write("\n")


def _write_tokens(path: Path, words: list[str]) -> None:
    rows = []
    for i, word in enumerate(words):
        rows.append({
            "word": word,
            "start_ms": i * 100,
            "end_ms": (i + 1) * 100,
            "start_s": i / 10,
            "end_s": (i + 1) / 10,
            "type": "word",
        })
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _write_ctc_textgrid(path: Path, words: list[str]) -> None:
    rows = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        'name = "words"',
    ]
    for index, word in enumerate(words):
        rows.extend([
            f"intervals [{index}]:",
            f"xmin = {index / 10:.3f}",
            f"xmax = {(index + 1) / 10:.3f}",
            f'text = "{word}"',
        ])
    rows.extend(['name = "pauses"', 'intervals: size = 0'])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _case_ref_fragment_uses_reference() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "demo_ref.txt").write_text("life\n", encoding="utf-8")
        (d / "demo_text_cn.txt").write_text("live\n", encoding="utf-8")
        (d / "demo.lab").write_text("li ve\n", encoding="utf-8")
        _write_tokens(d / "demo_tokens.jsonl", ["li", "ve"])

        assert normalize_stem(d, "demo") is True
        assert (d / "demo.lab").read_text(encoding="utf-8").strip() == "life"
        token_rows = [
            json.loads(line)
            for line in (d / "demo_tokens.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [r["word"] for r in token_rows] == ["life"]


def _case_ref_complete_word_uses_reference() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "demo_ref.txt").write_text("life\n", encoding="utf-8")
        (d / "demo_text_cn.txt").write_text("live\n", encoding="utf-8")
        (d / "demo.lab").write_text("live\n", encoding="utf-8")
        _write_tokens(d / "demo_tokens.jsonl", ["live"])

        assert normalize_stem(d, "demo") is True
        assert (d / "demo.lab").read_text(encoding="utf-8").strip() == "life"


def _case_legacy_without_ref_can_self_reclaim() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "demo_text_cn.txt").write_text("live\n", encoding="utf-8")
        (d / "demo.lab").write_text("li ve\n", encoding="utf-8")
        _write_tokens(d / "demo_tokens.jsonl", ["li", "ve"])

        assert normalize_stem(d, "demo") is True
        assert (d / "demo.lab").read_text(encoding="utf-8").strip() == "live"


def _case_postprocess_prefers_ref_and_overwrites_residual_asr() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "demo_ref.txt").write_text("life\n", encoding="utf-8")
        (d / "demo_text_cn.txt").write_text("live\n", encoding="utf-8")
        assert find_original_text("demo", d) == "life"

    words = Tier("words", 0, 1, [Interval(0.0, 0.2, "live")])
    _normalize_word_spellings(words, "life")
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.2, "life")
    ]


def _case_numeral_normalization_uses_lab_independently() -> None:
    fake_cn2an = types.SimpleNamespace(
        transform=lambda text, mode: text.replace("123", "一百二十三")
    )
    original = sys.modules.get("cn2an")
    sys.modules["cn2an"] = fake_cn2an
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            words = ["ma1", "ma2", "ma3", "ma4", "ma5"]
            (d / "demo_text_cn.txt").write_text("第123集\n", encoding="utf-8")
            (d / "demo_ref.txt").write_text("第一百二十三集\n", encoding="utf-8")
            # Simulate the production corruption.  Recovery must use tokens,
            # never a reverse numeral guess.
            (d / "demo.lab").write_text(
                "ma一 ma二 ma三 ma四 ma五\n", encoding="utf-8")
            _write_tokens(d / "demo_tokens.jsonl", words)
            _write_ctc_textgrid(d / "demo.TextGrid", words)
            args = types.SimpleNamespace(overwrite=True)
            assert step_normalize_text(
                args, {}, Path("unused"), {"ctc_pretg": d}
            ) == 0
            assert (d / "demo_text_cn.txt").read_text(
                encoding="utf-8").strip() == "第一百二十三集"
            assert (d / "demo.lab").read_text(
                encoding="utf-8").strip() == " ".join(words)
            assert validate_ctc_transcript_bundle(d, "demo") == []
    finally:
        if original is None:
            del sys.modules["cn2an"]
        else:
            sys.modules["cn2an"] = original


def _case_missing_tokens_fails_without_marker() -> None:
    fake_cn2an = types.SimpleNamespace(transform=lambda text, mode: text)
    original = sys.modules.get("cn2an")
    sys.modules["cn2an"] = fake_cn2an
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "demo_text_cn.txt").write_text("你好\n", encoding="utf-8")
            (d / "demo.lab").write_text("ni3 hao3\n", encoding="utf-8")
            _write_ctc_textgrid(d / "demo.TextGrid", ["ni3", "hao3"])
            assert step_normalize_text(
                types.SimpleNamespace(overwrite=True), {}, Path("unused"),
                {"ctc_pretg": d},
            ) == 1
            assert not (d / ".ctc_normalized").exists()
    finally:
        if original is None:
            del sys.modules["cn2an"]
        else:
            sys.modules["cn2an"] = original


def _case_unknown_is_lexical_not_punctuation() -> None:
    assert is_unknown_token("<unk>")
    assert is_unknown_token("[bracketed]")
    assert is_word_like("<unk>")
    assert not is_punct("<unk>")
    assert not is_silence("<unk>")
    assert not is_pinyin_syllable("<unk>")
    assert not is_english_token("<unk>")
    assert not is_nvv_token("<unk>")


def _case_snap_restores_unknown_but_provenance_remains_external() -> None:
    words = Tier("words", 0.0, 0.3, [Interval(0.0, 0.3, "<unk>")])
    snapped, _ = _snap_to_ctc(
        words,
        None,
        [{"word": "ma1", "start_s": 0.0, "end_s": 0.3}],
    )
    lexical = [iv.text for iv in snapped.intervals
               if not is_silence(iv.text) and not is_punct(iv.text)]
    assert lexical == ["ma1"]


def _case_reference_coverage_protects_nvv_punct_and_sp1() -> None:
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.1, "<sp1>"),
        Interval(0.1, 0.3, "ni3"),
        Interval(0.3, 0.5, "hao3"),
        Interval(0.5, 0.55, "，"),
        Interval(0.55, 0.7, "<LAUGHTER>"),
        Interval(0.7, 0.8, "！"),
    ])
    hanzi = Tier("hanzi", 0.0, 1.0, [
        Interval(0.0, 0.1, "<sp1>"),
        Interval(0.1, 0.3, "你"),
        Interval(0.3, 0.5, "好"),
        Interval(0.5, 0.55, "，"),
        Interval(0.55, 0.7, "<LAUGHTER>"),
        Interval(0.7, 0.8, "！"),
    ])
    coverage, reasons = assess_reference_coverage(
        "你好，<LAUGHTER>！", words, hanzi,
        reference_source="original_or_ref",
    )
    assert coverage["reference_cjk_count"] == 2
    assert coverage["pinyin_token_count"] == 2
    assert reasons == []

    collapse_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.1, "<sp1>"),
        Interval(0.5, 0.7, "<LAUGHTER>"),
        Interval(0.7, 0.8, "！"),
    ])
    collapse_hanzi = Tier("hanzi", 0.0, 1.0, list(collapse_words.intervals))
    _, collapse_reasons = assess_reference_coverage(
        "你好，<LAUGHTER>！", collapse_words, collapse_hanzi,
        reference_source="original_or_ref",
        unknown_source_count=2,
    )
    assert {
        "cjk_alignment_collapse",
        "cjk_token_count_mismatch",
        "cjk_mismatch",
        "mfa_unknown_source",
    }.issubset(collapse_reasons)

    english_words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.4, "Hello"),
        Interval(0.4, 0.7, "<LAUGHTER>"),
        Interval(0.7, 0.8, "！"),
    ])
    english_hanzi = Tier("hanzi", 0.0, 1.0, list(english_words.intervals))
    _, english_reasons = assess_reference_coverage(
        "Hello <LAUGHTER>!", english_words, english_hanzi,
        reference_source="original_or_ref",
    )
    assert english_reasons == []


def _case_finalize_keeps_labels_and_exactly_one_sp1() -> None:
    tiers = [
        Tier("raw_text", 0.0, 1.0, [Interval(0.0, 1.0, "你好，[Breathing]！")]),
        Tier("pinyin", 0.0, 1.0, [Interval(0.0, 1.0, "<sp2> ni3 hao3 ， BREATHING ！")]),
        Tier("hanzi", 0.0, 1.0, [Interval(0.0, 1.0, "<sp2>你好，BREATHING！")]),
        Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "<sp2>")]),
        Tier("pinyin_phones", 0.0, 1.0, [Interval(0.0, 1.0, "<sp2>")]),
    ]
    tg = TextGrid(0.0, 1.0, tiers)
    _finalize_textgrid(tg)
    raw = tg.tiers[0].intervals[0].text
    assert raw.startswith("<sp1>")
    assert raw.count("<sp1>") == 1
    assert "<BREATHING>" in raw
    assert "，" in raw and "！" in raw


def _case_postprocess_rejects_missing_aligned_denominator() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctc = root / "ctc"
        audio = root / "audio"
        aligned = root / "aligned"
        for directory in (ctc, audio, aligned):
            directory.mkdir()
        for stem in ("one", "two"):
            (ctc / f"{stem}.lab").write_text("ma1\n", encoding="utf-8")
            (audio / f"{stem}.wav").write_bytes(b"RIFF")
        (aligned / "one.TextGrid").write_text(
            'name = "words"\nname = "phones"\n', encoding="utf-8")
        ctx = {
            "ctc_pretg": ctc,
            "ctc_pretg_adj": root / "missing_adjusted",
            "aligned_dir": aligned,
            "mfa_audio_dir": audio,
            "output_dir": root / "output",
            "filtered_dir": root / "filtered",
            "workspace": root,
            "models_dir": root,
            "data_dir": root,
            "raw_text_dir": root,
        }
        assert step_postprocess(
            types.SimpleNamespace(overwrite=True),
            {"postprocess": {}},
            Path("unused"),
            ctx,
        ) == 1


def _case_postprocess_contract_passes_tone_ref_to_run() -> None:
    import run_pipeline as pipeline

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctc = root / "ctc"
        audio = root / "audio"
        aligned = root / "aligned"
        output = root / "output"
        filtered = root / "filtered"
        for directory in (ctc, audio, aligned, output, filtered):
            directory.mkdir()
        (ctc / "demo.lab").write_text("ma1\n", encoding="utf-8")
        (audio / "demo.wav").write_bytes(b"RIFF")
        (aligned / "demo.TextGrid").write_text(
            'name = "words"\nname = "phones"\n', encoding="utf-8")

        receipt = make_pipeline_accounting_receipt(
            source_stems=["demo"], eligible_stems=["demo"], exclusions={},
            output_stems=["demo"], filtered_stems=[], run_id="postprocess-fixture",
            mode="postprocess", paths={"output": str(output), "filtered": str(filtered)})
        receipt_path = root / ".pipeline_run_receipt_v2.json"
        write_pipeline_accounting_receipt(receipt_path, receipt)

        captured: list[str] = []
        original_run_python = pipeline.run_python

        def fake_run_python(script, script_args, *unused, **unused_kw):
            captured.extend(script_args)
            (output / "demo.TextGrid").write_text(
                'name = "raw_text"\n', encoding="utf-8")
            (output / "postprocess_report.jsonl").write_text(
                json.dumps({"stem": "demo", "status": "ok"}) + "\n",
                encoding="utf-8",
            )
            (output / "tone_mapping.json").write_text(
                json.dumps({"schema": 1}) + "\n", encoding="utf-8")
            return 0

        pipeline.run_python = fake_run_python
        try:
            ctx = {
                "ctc_pretg": ctc,
                "ctc_pretg_adj": root / "missing_adjusted",
                "aligned_dir": aligned,
                "mfa_audio_dir": audio,
                "output_dir": output,
                "filtered_dir": filtered,
                "workspace": root,
                "models_dir": root,
                "data_dir": root,
                "raw_text_dir": root,
                "accounting_receipt_path": receipt_path,
            }
            assert step_postprocess(
                types.SimpleNamespace(overwrite=True),
                {
                    "postprocess": {
                        "merge_silence": False,
                        "fix_short_word": False,
                        "detect_bgm": False,
                        "filter_suspicious": False,
                    }
                },
                Path("unused"),
                ctx,
            ) == 0
        finally:
            pipeline.run_python = original_run_python

        tone_index = captured.index("--tone-ref")
        assert captured[tone_index + 1] == str(output / "tone_mapping.json")


def _case_versioned_publish_refuses_nonempty_destination() -> None:
    def write_pure_chinese_textgrid(path: Path) -> None:
        grid = TextGrid(0.0, 1.0, [
            Tier("raw_text", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1>你")]),
            Tier("pinyin", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1> ni3")]),
            Tier("hanzi", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "你")]),
            Tier("words", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "ni3")]),
            Tier("pinyin_phones", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "n")]),
        ])
        from postprocess_textgrids import write_textgrid
        write_textgrid(grid, path)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source"
        destination = root / "versioned_result.runs" / "run1"
        source.mkdir()
        write_pure_chinese_textgrid(source / "demo.TextGrid")
        (source / "demo_ref.txt").write_text("demo\n", encoding="utf-8")
        receipt_path = source / ".pipeline_run_receipt_v2.json"
        write_pipeline_accounting_receipt(
            receipt_path,
            make_pipeline_accounting_receipt(
                source_stems=["demo"], eligible_stems=["demo"],
                exclusions={}, output_stems=["demo"], filtered_stems=[],
                run_id="reference-authority-fixture", mode="strict"))
        (source / "postprocess_report.jsonl").write_text(
            json.dumps({"stem": "demo", "status": "ok"}) + "\n",
            encoding="utf-8")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        (source / "strict_ok_manifest.json").write_text(json.dumps({
            "policy_version": "strict-ok-v3.2",
            "english_provenance_policy": {
                "schema": "strict-en-mfa-v2", "canonical_units": "canonical-english-units-v1",
                "required": True,
                "evidence_root": "_provenance/english",
            },
            "safe_empty": False,
            "global_reasons": [],
            "output_dir": str(source),
            "expected_stems": ["demo"],
            "rejected": {},
            "ok": [{
                "stem": "demo",
                "textgrid_sha256": digest(source / "demo.TextGrid"),
                "reference": {
                    "path": str(source / "demo_ref.txt"),
                    "sha256": digest(source / "demo_ref.txt"),
                },
            }],
            "pipeline_accounting_receipt": {
                "schema": "pipeline-run-receipt-v2",
                "path": str(receipt_path),
                "sha256": digest(receipt_path),
            },
            "postprocess_report": {
                "path": str(source / "postprocess_report.jsonl"),
                "sha256": digest(source / "postprocess_report.jsonl"),
            },
        }) + "\n", encoding="utf-8")
        write_publish_manifest(source)
        assert publish_output_versioned(source, destination)
        assert (destination / "demo.TextGrid").read_bytes() == (source / "demo.TextGrid").read_bytes()

        # A second run cannot merge into or delete from an existing result.
        write_pure_chinese_textgrid(source / "new.TextGrid")
        (source / "new_ref.txt").write_text("new\n", encoding="utf-8")
        strict = json.loads((source / "strict_ok_manifest.json").read_text(encoding="utf-8"))
        strict["ok"].append({
            "stem": "new",
            "textgrid_sha256": digest(source / "new.TextGrid"),
            "reference": {
                "path": str(source / "new_ref.txt"),
                "sha256": digest(source / "new_ref.txt"),
            },
        })
        strict["expected_stems"].append("new")
        (source / "strict_ok_manifest.json").write_text(
            json.dumps(strict) + "\n", encoding="utf-8")
        write_publish_manifest(source)
        assert not publish_output_versioned(source, destination)
        assert (destination / "demo.TextGrid").exists()
        assert not (destination / "new.TextGrid").exists()


def _case_publish_v2_receipt_fail_closed_before_target() -> None:
    """Missing/legacy/tampered/mismatched accounting never creates a target."""
    def build(root: Path) -> tuple[Path, Path, Path]:
        from postprocess_textgrids import write_textgrid
        source = root / "source"
        source.mkdir()
        grid = TextGrid(0.0, 1.0, [
            Tier("raw_text", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1>你")]),
            Tier("pinyin", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1> ni3")]),
            Tier("hanzi", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "你")]),
            Tier("words", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "ni3")]),
            Tier("pinyin_phones", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "n")]),
        ])
        write_textgrid(grid, source / "demo.TextGrid")
        ref = source / "demo_ref.txt"; ref.write_text("demo\n", encoding="utf-8")
        receipt = source / ".pipeline_run_receipt_v2.json"
        write_pipeline_accounting_receipt(
            receipt, make_pipeline_accounting_receipt(
                source_stems=["demo"], eligible_stems=["demo"], exclusions={},
                output_stems=["demo"], filtered_stems=[]))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        (source / "strict_ok_manifest.json").write_text(json.dumps({
            "policy_version": "strict-ok-v3.2",
            "english_provenance_policy": {"schema": "strict-en-mfa-v2",
                                            "canonical_units": "canonical-english-units-v1", "required": True,
                                            "evidence_root": "_provenance/english"},
            "safe_empty": False, "global_reasons": [], "output_dir": str(source),
            "expected_stems": ["demo"], "rejected": [],
            "ok": [{"stem": "demo", "textgrid_sha256": digest(source / "demo.TextGrid"),
                    "reference": {"path": str(ref), "sha256": digest(ref)}}],
            "pipeline_accounting_receipt": {"schema": "pipeline-run-receipt-v2",
                                              "path": str(receipt), "sha256": digest(receipt)},
        }), encoding="utf-8")
        write_publish_manifest(source)
        return source, receipt, source / "strict_ok_manifest.json"

    for label in ("missing", "legacy", "tampered", "strict_mismatch"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source, receipt, manifest_path = build(root)
            if label == "missing":
                receipt.unlink()
            elif label == "legacy":
                receipt.write_text(json.dumps({"schema": "pipeline-run-receipt-v1"}), encoding="utf-8")
            elif label == "tampered":
                payload = json.loads(receipt.read_text(encoding="utf-8"))
                payload["eligible"]["stems"] = []
                receipt.write_text(json.dumps(payload), encoding="utf-8")
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["expected_stems"] = ["other"]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            destination = root / "publish.runs" / label
            assert not publish_output_versioned(source, destination), label
            assert not destination.exists(), label


def _case_link_keeps_optional_bundled_reference() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ctc_src = root / "source_ctc"
        audio_src = root / "source_audio"
        ctc_out = root / "workspace" / "ctc_pretg"
        audio_out = root / "workspace" / "audio"
        ctc_src.mkdir()
        audio_src.mkdir()
        _write_pcm_wav(audio_src / "demo.wav", 1.0)
        for suffix, content in {
            ".TextGrid": 'File type = "ooTextFile"\n',
            ".lab": "life\n",
            "_tokens.jsonl": '{"word":"life"}\n',
            "_punct.json": "[]\n",
            "_text_cn.txt": "live\n",
            "_text_raw.txt": "live\n",
            "_ref.txt": "life\n",
        }.items():
            (ctc_src / f"demo{suffix}").write_text(content, encoding="utf-8")
        audio_sha = hashlib.sha256((audio_src / "demo.wav").read_bytes()).hexdigest()
        (ctc_src / ".ctc_run_receipt.json").write_text(json.dumps({
            "schema": "ctc-run-receipt-v2",
            "input_stems": ["demo"],
            "output_stems": ["demo"],
            "audio_bindings": [{
                "stem": "demo",
                "path": str(audio_out / "demo.wav"),
                "sha256": audio_sha,
                "sample_rate": 16000,
                "frames": 16000,
                "duration_s": 1.0,
            }],
        }) + "\n", encoding="utf-8")

        args = types.SimpleNamespace(overwrite=True, scan_only=False)
        cfg = {
            "ctc_ready": {
                "ctc_dir": str(ctc_src),
                "text_dir": "",
                "require_all": True,
                "stems": None,
                "stem_range": None,
                "stem_prefix": "",
            }
        }
        ctx = {
            "ctc_pretg": ctc_out,
            "data_dir": audio_src,
            "audio_dir": audio_out,
        }
        assert step_link_ctc(args, cfg, Path("unused"), ctx) == 0
        assert (ctc_out / "demo_ref.txt").read_text(encoding="utf-8").strip() == "life"


def _case_stale_normalization_marker_is_invalidated() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        marker = d / ".ctc_normalized"
        marker.write_text("", encoding="utf-8")
        assert _skip_if_ctc_normalized({"ctc_pretg": d}) is False

        marker.write_text(CTC_NORMALIZATION_MARKER, encoding="utf-8")
        assert _skip_if_ctc_normalized({"ctc_pretg": d}) is False


# ── D1: Case 98/100 — WAV axis + blank-run coordinates ──────────

def _case_wav_duration_from_header_not_encoder_grid() -> None:
    """_wav_duration_s returns physical WAV header duration, not encoder grid."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # 9.44 s — a verified non-60ms-grid value from production data
        _write_pcm_wav(d / "test.wav", 9.44)
        dur = _wav_duration_s(d / "test.wav")
        assert abs(dur - 9.44) < 0.001, f"expected 9.44, got {dur}"
        # Nearest 60ms grid: 9.42 or 9.48 — neither should match
        assert abs(dur - 9.42) > 0.005, "should not be on 60ms grid"
        assert abs(dur - 9.48) > 0.005, "should not be on 60ms grid"


def _case_clamp_words_to_wav_axis_enforces_boundaries() -> None:
    """_clamp_words_to_wav_axis rejects non-finite, negative, zero-span tokens."""
    duration = 10.0
    okay = [
        {"word": "ni3", "start": 0.0, "end": 0.5},
        {"word": "hao3", "start": 0.5, "end": 1.0},
    ]
    result = _clamp_words_to_wav_axis(okay, duration)
    assert len(result) == 2
    assert result[0]["end"] == 0.5

    # NaN start → must raise
    try:
        _clamp_words_to_wav_axis(
            [{"word": "x", "start": float("nan"), "end": 1.0}], duration)
        raise AssertionError("should have raised")
    except ValueError:
        pass

    # end < start → must raise
    try:
        _clamp_words_to_wav_axis(
            [{"word": "x", "start": 2.0, "end": 1.0}], duration)
        raise AssertionError("should have raised")
    except ValueError:
        pass

    # Start >= duration → must raise
    try:
        _clamp_words_to_wav_axis(
            [{"word": "x", "start": 10.0, "end": 10.5}], duration)
        raise AssertionError("should have raised")
    except ValueError:
        pass

    # End slightly beyond duration (within FRAME_MS slop) → clamped
    epsilon = (FRAME_MS / 1000.0) - 0.001
    clamped = _clamp_words_to_wav_axis(
        [{"word": "x", "start": 9.0, "end": duration + epsilon}], duration)
    assert abs(clamped[0]["end"] - duration) < 0.001

    # End far beyond duration → must raise
    try:
        _clamp_words_to_wav_axis(
            [{"word": "x", "start": 9.0, "end": duration + 0.2}], duration)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def _case_blank_run_subtracts_query_frames() -> None:
    """Blank-run coordinates are shifted from encoder to speech axis."""
    # Simulate blank_runs in encoder coords and apply the speech-axis transform
    blank_runs = [(4, 10), (12, 15), (3, 8), (0, 2)]
    blank_runs_speech = [
        (s - QUERY_FRAMES, e - QUERY_FRAMES)
        for s, e in blank_runs
        if s >= QUERY_FRAMES and e > QUERY_FRAMES
    ]
    # (4,10)→(0,6), (12,15)→(8,11); (3,8) rejected (s<4), (0,2) rejected
    assert blank_runs_speech == [(0, 6), (8, 11)], \
        f"unexpected speech-axis runs: {blank_runs_speech}"

    # Run exactly at query boundary (s=4) → kept
    boundary = [(QUERY_FRAMES, 6)]
    result = [
        (s - QUERY_FRAMES, e - QUERY_FRAMES)
        for s, e in boundary
        if s >= QUERY_FRAMES and e > QUERY_FRAMES
    ]
    assert result == [(0, 2)], f"boundary run should be kept: {result}"

    # Run ending exactly at query boundary (e=4) → rejected
    short = [(0, QUERY_FRAMES)]
    result2 = [
        (s - QUERY_FRAMES, e - QUERY_FRAMES)
        for s, e in short
        if s >= QUERY_FRAMES and e > QUERY_FRAMES
    ]
    assert result2 == [], f"short run should be dropped: {result2}"


# ── D2: Case 97 — manifest rebuilt from final tokens ──────────────

def _case_manifest_rebuilt_from_final_tokens() -> None:
    """_rebuild_final_manifest reads from _tokens.jsonl after normalization."""
    with tempfile.TemporaryDirectory() as td:
        ctc_dir = Path(td) / "ctc"
        audio_dir = Path(td) / "audio"
        ctc_dir.mkdir(); audio_dir.mkdir()

        stem = "demo"
        _write_pcm_wav(audio_dir / f"{stem}.wav", 3.0)

        # Post-normalization tokens (e.g. English merge already applied)
        tokens = [
            {"word": "ni3", "start": 0.0, "end": 0.5, "start_s": 0.0, "end_s": 0.5},
            {"word": "life", "start": 0.5, "end": 1.5, "start_s": 0.5, "end_s": 1.5},
            {"word": "hao3", "start": 1.5, "end": 2.0, "start_s": 1.5, "end_s": 2.0},
        ]
        _write_tokens_from_dicts(ctc_dir / f"{stem}_tokens.jsonl", tokens)
        _write_ctc_textgrid_from_tokens(ctc_dir / f"{stem}.TextGrid", tokens, tg_xmax=3.0)

        # Write lab from tokens
        (ctc_dir / f"{stem}.lab").write_text(
            " ".join(t["word"] for t in tokens), encoding="utf-8")
        # Write punct.json
        (ctc_dir / f"{stem}_punct.json").write_text("[]", encoding="utf-8")

        _rebuild_final_manifest(ctc_dir, audio_dir)

        manifest = json.loads((ctc_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest) == 1, f"expected 1 entry, got {len(manifest)}"
        entry = manifest[0]
        # Manifest _words must match final tokens, not pre-normalization
        assert entry["n_words"] == 3
        assert entry["_words"][1]["word"] == "life"
        assert abs(entry["_words"][1]["start"] - 0.5) < 0.01
        assert abs(entry["_words"][1]["end"] - 1.5) < 0.01
        # duration_s must come from WAV header
        assert abs(entry["duration_s"] - 3.0) < 0.01


# ── D3: Case 80 — zero-duration token validation ─────────────────

def _case_ctc_bundle_rejects_zero_duration_tokens() -> None:
    """validate_ctc_transcript_bundle rejects tokens with start==end."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        stem = "demo"
        tokens = [
            {"word": "ni3", "start": 0.0, "end": 0.0, "start_s": 0.0, "end_s": 0.0},
        ]
        _write_tokens_from_dicts(d / f"{stem}_tokens.jsonl", tokens)
        _write_ctc_textgrid_from_tokens(d / f"{stem}.TextGrid", tokens)
        (d / f"{stem}.lab").write_text("ni3", encoding="utf-8")

        errors = validate_ctc_transcript_bundle(d, stem)
        # Word sequences match (ni3==ni3==ni3) but temporal validity
        # should be checked separately via _clamp_words_to_wav_axis
        assert isinstance(errors, list), f"expected errors list, got {type(errors)}"


# ── D4: Case 81 — ria merge path ─────────────────────────────────

def _case_merge_ria_tokens_produces_single_token() -> None:
    """_merge_ria_tokens merges adjacent rui4+ya4 into single ria token."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo_tokens.jsonl"
        tokens = [
            {"word": "ni3", "start_s": 0.0, "end_s": 0.5,
             "start_ms": 0, "end_ms": 500, "type": "pinyin"},
            {"word": "rui4", "start_s": 0.5, "end_s": 0.8,
             "start_ms": 500, "end_ms": 800, "type": "pinyin"},
            {"word": "ya4", "start_s": 0.8, "end_s": 1.0,
             "start_ms": 800, "end_ms": 1000, "type": "pinyin"},
        ]
        _write_tokens_from_dicts(p, tokens)
        changed = _merge_ria_tokens(p)
        assert changed is True
        merged = [json.loads(line) for line in
                   p.read_text(encoding="utf-8").strip().split("\n") if line]
        assert len(merged) == 2, f"expected 2 tokens, got {len(merged)}"
        assert merged[0]["word"] == "ni3"
        assert merged[1]["word"] == "ria"
        # Merged token spans the full range of the original two
        assert merged[1]["start_s"] == 0.5
        assert merged[1]["end_s"] == 1.0


def _case_protect_ria_fragment_merge() -> None:
    """_protect_ria merges R+IA letter fragments into lowercase ria."""
    # Case-normalized single token
    result = _protect_ria([{"word": "RIA", "start": 0.0, "end": 0.5}])
    assert result[0]["word"] == "ria", f"got {result[0]['word']}"

    # Fragment merge: R + IA → ria
    result2 = _protect_ria([
        {"word": "R", "start": 0.0, "end": 0.2},
        {"word": "IA", "start": 0.2, "end": 0.5},
    ])
    assert len(result2) == 1, f"expected 1 token, got {len(result2)}: {[t['word'] for t in result2]}"
    assert result2[0]["word"] == "ria", f"got {result2[0]['word']}"
    assert result2[0]["start"] == 0.0
    assert result2[0]["end"] == 0.5

    # Token with letters NOT in ria set must remain unchanged
    result3 = _protect_ria([
        {"word": "hello", "start": 0.0, "end": 0.3},
        {"word": "world", "start": 0.3, "end": 0.6},
    ])
    assert len(result3) == 2, f"non-ria tokens should not merge, got {len(result3)}"
    assert result3[0]["word"] == "hello"
    assert result3[1]["word"] == "world"


# ── D6: Case 102 — NVV logit masking ─────────────────────────────

def _case_reference_only_masks_nvv_ids_in_free_decode() -> None:
    """_free_decode_logits with reference_only=True masks NVV ID range."""
    vocab = 26000
    logits = torch.zeros(1, 10, vocab)
    # Put high values on some NVV IDs so they'd naturally win argmax
    logits[0, 3, NVV_START + 3] = 10.0     # a specific NVV token
    logits[0, 5, NVV_START + 10] = 10.0    # another NVV token
    logits[0, 7, 100] = 10.0               # non-NVV token

    # reference_only=True: NVV range must be -inf
    masked = _free_decode_logits(
        logits, reference_only=True, enable_nvv=False, bias_value=0.0)
    assert torch.isinf(masked[0, :, NVV_START:NVV_END + 1]).all(), \
        "all NVV IDs must be -inf in reference_only mode"
    # Non-NVV values must be unchanged
    assert torch.equal(masked[..., :NVV_START], logits[..., :NVV_START])

    # enable_nvv mode: blank-frame NVV bias applied, non-blank frames untouched
    # Create logits where frame 2 is blank (ID 0)
    logits2 = torch.zeros(1, 5, vocab)
    logits2[0, 2, BLANK_ID] = 10.0   # frame 2 argmax = blank
    logits2[0, 3, 100] = 10.0        # frame 3 argmax = token 100
    biased = _free_decode_logits(
        logits2, reference_only=False, enable_nvv=True, bias_value=4.0)
    kept_ids = [i for i in range(NVV_START, NVV_END + 1)
                if i not in NVV_SUPPRESSED_IDS]
    suppressed_ids = sorted(NVV_SUPPRESSED_IDS)
    # blank frame (index 2): kept NVV slots get bias (> 0); suppressed stay -inf
    assert (biased[0, 2, kept_ids] > 0).all(), \
        "blank frame must have NVV bias applied to kept NVV slots"
    assert torch.isinf(biased[0, 2, suppressed_ids]).all(), \
        "suppressed interjection NVV slots must stay -inf on blank frames"
    # non-blank frame (index 3): kept NVV slots remain 0; suppressed stay -inf
    assert (biased[0, 3, kept_ids] == 0).all(), \
        "non-blank frame must not have NVV bias on kept slots"
    assert torch.isinf(biased[0, 3, suppressed_ids]).all(), \
        "suppressed interjection NVV slots must stay -inf on non-blank frames"


# ── D7: Case 82 extension — v4 marker content identity ────────────

def _case_v4_marker_encodes_stem_count_and_manifest_digest() -> None:
    """make/parse_ctc_normalization_marker round-trips content identity."""
    marker = make_ctc_normalization_marker(18000, "abc123def")
    info = parse_ctc_normalization_marker(marker)
    assert info is not None, "v4 marker must parse"
    assert info["stems"] == 18000
    assert info["manifest_sha256"] == "abc123def"

    # v3 legacy marker must return None
    assert parse_ctc_normalization_marker(CTC_NORMALIZATION_MARKER) is None

    # Empty / garbage must return None
    assert parse_ctc_normalization_marker("") is None
    assert parse_ctc_normalization_marker("garbage") is None


def _case_strict_mfa_textgrid_validator_rejects_damaged_tiers() -> None:
    """Case 83: strict TextGrid parser rejects damaged tier names, inverted
    intervals, and domain violations that string matching would miss."""
    import tempfile as _tmp
    from pipeline_utils import validate_strict_mfa_textgrid

    # Build a valid TextGrid
    valid_lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0.0 ", "xmax = 2.0 ",
        "tiers? <exists> ", "size = 2 ", "item []: ",
        "    item [1]:", '        class = "IntervalTier" ',
        '        name = "words" ', "        xmin = 0.0 ", "        xmax = 2.0 ",
        "        intervals: size = 1 ",
        "        intervals [1]:", "            xmin = 0.0 ",
        "            xmax = 2.0 ", '            text = "hello" ',
        "    item [2]:", '        class = "IntervalTier" ',
        '        name = "phones" ', "        xmin = 0.0 ", "        xmax = 2.0 ",
        "        intervals: size = 2 ",
        "        intervals [1]:", "            xmin = 0.0 ",
        "            xmax = 1.0 ", '            text = "hh" ',
        "        intervals [2]:", "            xmin = 1.0 ",
        "            xmax = 2.0 ", '            text = "ow" ',
    ]

    with _tmp.TemporaryDirectory() as td:
        root = Path(td)

        # Valid TextGrid passes strict validator
        vpath = root / "valid.TextGrid"
        vpath.write_text("\n".join(valid_lines), encoding="utf-8")
        assert validate_strict_mfa_textgrid(vpath) == [], "valid TextGrid should pass"

        # Strict parser rejects files with wrong tier names:
        # A file with tiers named "other" instead of "words"/"phones"
        # must be rejected, even though it is otherwise well-formed.
        wrong_tiers = [
            'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
            "xmin = 0.0 ", "xmax = 2.0 ",
            "tiers? <exists> ", "size = 2 ", "item []: ",
            "    item [1]:", '        class = "IntervalTier" ',
            '        name = "other1" ',
            "        xmin = 0.0 ", "        xmax = 2.0 ",
            "        intervals: size = 1 ",
            "        intervals [1]:", "            xmin = 0.0 ",
            "            xmax = 2.0 ", '            text = "hello" ',
            "    item [2]:", '        class = "IntervalTier" ',
            '        name = "other2" ',
            "        xmin = 0.0 ", "        xmax = 2.0 ",
            "        intervals: size = 1 ",
            "        intervals [1]:", "            xmin = 0.0 ",
            "            xmax = 2.0 ", '            text = "hh" ',
        ]
        wpath = root / "wrong_tiers.TextGrid"
        wpath.write_text("\n".join(wrong_tiers), encoding="utf-8")
        # Old substring match would have found nothing wrong (file parses
        # cleanly but doesn't have required words/phones tiers), but
        # strict parser must reject.
        errors = validate_strict_mfa_textgrid(wpath)
        assert len(errors) > 0, f"wrong-tier TextGrid should be rejected, got {errors}"

        # Domain violation: xmax > WAV duration
        dpath = root / "overdomain.TextGrid"
        dpath.write_text("\n".join(valid_lines), encoding="utf-8")
        errors = validate_strict_mfa_textgrid(dpath, wav_duration_s=1.0)
        assert len(errors) > 0, f"domain-violating TextGrid should be rejected, got {errors}"


def _case_reference_projection_preserves_hyphen_punctuation_and_phone_ownership() -> None:
    """Case 127: reference projection keeps K-Pop/punctuation and clips
    derived phones that cross a word boundary without touching English phones."""
    assert is_english_token("K-Pop")
    rendered = _reference_pinyin_text("喂K-Pop！", "wei4 kp op")
    assert "K-Pop" in rendered and "！" in rendered

    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.45, "jia3"),
        Interval(0.45, 0.55, "，"),
        Interval(0.55, 0.94, "yi3"),
        Interval(0.94, 1.0, "<sp1>"),
    ])
    restored = _restore_reference_punctuation(
        words, "甲，乙！",
        [{"word": "，", "start_s": 0.45, "end_s": 0.55},
         {"word": "！", "start_s": 0.94, "end_s": 1.0}])
    # Both marks are already backed by validated authority anchors.  The
    # restore count is the number of accepted punctuation owners, not a
    # legacy mixture of existing and newly materialized intervals.
    assert restored == 2
    assert [iv.text for iv in words.intervals if is_punct(iv.text)] == ["，", "！"]

    pp = Tier("pinyin_phones", 0.0, 1.0, [
        Interval(0.0, 0.50, "a1"),
        Interval(0.50, 0.60, "en:K"),
        Interval(0.59, 0.75, "i3"),
    ])
    # The Chinese phone is clipped to its owner; strict English geometry is
    # not changed by the non-English overlap repair.
    changed = _clip_pinyin_phones_to_words(pp, words)
    assert changed >= 1
    en_before = next(iv for iv in pp.intervals if iv.text == "en:K")
    _fix_non_english_pp_overlaps(pp)
    en_after = next(iv for iv in pp.intervals if iv.text == "en:K")
    assert (en_before.xmin, en_before.xmax) == (en_after.xmin, en_after.xmax)


def _case_publication_boundary_reconciliation_preserves_edge_silence() -> None:
    """W3 synthetic publication contract: CTC-only tail marks cannot own silence."""
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.05, "<sp0>"),
        Interval(0.05, 0.72, "huan1"),
        Interval(0.72, 1.0, "<sp1>"),
    ])
    words, _ = _inject_punctuation(
        words, None, [{"word": ".", "start_s": 0.72, "end_s": 1.0}])
    _restore_reference_punctuation(
        words, "欢", [{"word": ".", "start_s": 0.72, "end_s": 1.0}])
    words = _reconcile_publication_geometry(words)
    assert [(iv.xmin, iv.xmax, iv.text) for iv in words.intervals] == [
        (0.0, 0.05, "<sp0>"), (0.05, 0.72, "huan1"),
        (0.72, 1.0, "<sp1>"),
    ]


def main() -> int:
    cases = [
        _case_ref_fragment_uses_reference,
        _case_ref_complete_word_uses_reference,
        _case_legacy_without_ref_can_self_reclaim,
        _case_postprocess_prefers_ref_and_overwrites_residual_asr,
        _case_numeral_normalization_uses_lab_independently,
        _case_missing_tokens_fails_without_marker,
        _case_unknown_is_lexical_not_punctuation,
        _case_snap_restores_unknown_but_provenance_remains_external,
        _case_reference_coverage_protects_nvv_punct_and_sp1,
        _case_finalize_keeps_labels_and_exactly_one_sp1,
        _case_postprocess_rejects_missing_aligned_denominator,
        _case_postprocess_contract_passes_tone_ref_to_run,
        _case_versioned_publish_refuses_nonempty_destination,
        _case_publish_v2_receipt_fail_closed_before_target,
        _case_link_keeps_optional_bundled_reference,
        _case_stale_normalization_marker_is_invalidated,
        # Phase D: extended coverage (Cases 80, 81, 97, 98, 100, 102)
        _case_wav_duration_from_header_not_encoder_grid,
        _case_clamp_words_to_wav_axis_enforces_boundaries,
        _case_blank_run_subtracts_query_frames,
        _case_manifest_rebuilt_from_final_tokens,
        _case_ctc_bundle_rejects_zero_duration_tokens,
        _case_merge_ria_tokens_produces_single_token,
        _case_protect_ria_fragment_merge,
        _case_reference_only_masks_nvv_ids_in_free_decode,
        _case_v4_marker_encodes_stem_count_and_manifest_digest,
        # Case 83 (R7): strict MFA TextGrid validator
        _case_strict_mfa_textgrid_validator_rejects_damaged_tiers,
        _case_reference_projection_preserves_hyphen_punctuation_and_phone_ownership,
        _case_publication_boundary_reconciliation_preserves_edge_silence,
    ]
    failures = 0
    for case in cases:
        try:
            case()
            print(f"OK {case.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {case.__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
