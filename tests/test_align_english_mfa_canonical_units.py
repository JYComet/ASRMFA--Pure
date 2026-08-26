import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.align_english_mfa as producer


def _textgrid(path: Path, tiers: dict[str, list[tuple[float, float, str]]]) -> None:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "xmin = 0",
        "xmax = 2",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for index, (name, intervals) in enumerate(tiers.items(), 1):
        lines += [
            f"    item [{index}]:",
            '        class = "IntervalTier"',
            f'        name = "{name}"',
            "        xmin = 0",
            "        xmax = 2",
            f"        intervals: size = {len(intervals)}",
        ]
        for ordinal, (start, end, text) in enumerate(intervals, 1):
            lines += [
                f"        intervals [{ordinal}]:",
                f"            xmin = {start}",
                f"            xmax = {end}",
                f'            text = "{text}"',
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kpop_source(tmp_path: Path, *, repeated: bool = False) -> tuple[Path, dict]:
    ctc = tmp_path / "ctc"
    ctc.mkdir()
    words = [(0.0, 0.4, "K"), (0.4, 1.0, "Pop")]
    reference = "K-Pop"
    if repeated:
        words += [(1.0, 1.35, "K"), (1.35, 2.0, "Pop")]
        reference = "K-Pop K-Pop"
    _textgrid(ctc / "demo.TextGrid", {"words": words})
    (ctc / "demo.lab").write_text(" ".join(item[2] for item in words) + "\n", encoding="utf-8")
    (ctc / "demo_ref.txt").write_text(reference + "\n", encoding="utf-8")
    return ctc, producer.find_english_segments(ctc, ["demo"])


def test_canonical_compound_is_one_mfa_word_with_alignment_token(tmp_path: Path):
    ctc, segments = _kpop_source(tmp_path)

    word = segments["demo"][0]["words"][0]
    assert len(segments["demo"][0]["words"]) == 1
    assert word["text"] == "K-Pop"
    assert word["alignment_token"] == "kpop"
    assert word["unit_id"] == "en-u0000"
    assert word["source_ctc_ordinals"] == [0, 1]
    assert word["canonical_span"] == [0.0, 1.0]
    assert word["canonical_unit"]["canonical_binding"] == "canonical-english-units-v1"


def test_repeated_adjacent_compounds_keep_distinct_unit_identity(tmp_path: Path):
    _, segments = _kpop_source(tmp_path, repeated=True)

    words = segments["demo"][0]["words"]
    assert [word["unit_id"] for word in words] == ["en-u0000", "en-u0001"]
    assert [word["source_ctc_ordinals"] for word in words] == [[0, 1], [2, 3]]
    assert [word["alignment_token"] for word in words] == ["kpop", "kpop"]


def test_token_authority_processed_geometry_drives_mfa_segment(tmp_path: Path):
    ctc, segments = _kpop_source(tmp_path)
    source_word = segments["demo"][0]["words"][0]
    processed = [0.0, 1.4]
    _textgrid(ctc / "demo.TextGrid", {"words": [(0.0, 1.4, "kpop")]})
    (ctc / "demo.lab").write_text("kpop\n", encoding="utf-8")
    token = {
        "word": "kpop", "start_s": processed[0], "end_s": processed[1],
        "processed_ctc_span": processed,
        "processed_ctc_boundary_source": "next_lexical_token_start",
        "surface_text": source_word["text"],
        "source_ctc_ordinals": source_word["source_ctc_ordinals"],
        "canonical_span": source_word["canonical_span"],
        "canonical_unit": source_word["canonical_unit"],
        "canonical_unit_sha256": hashlib.sha256(
            json.dumps(source_word["canonical_unit"], ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reference_identity": hashlib.sha256(b"K-Pop").hexdigest(),
        "reference_ordinal": 0,
    }
    (ctc / "demo_tokens.jsonl").write_text(
        json.dumps(token, ensure_ascii=False) + "\n", encoding="utf-8")

    result = producer.find_english_segments(ctc, ["demo"])
    word = result["demo"][0]["words"][0]
    assert word["canonical_span"] == [0.0, 1.0]
    assert word["processed_ctc_span"] == processed
    assert word["start"] == 0.0 and word["end"] == 1.4
    assert result["demo"][0]["seg_start"] == 0.0
    assert result["demo"][0]["seg_end"] == 1.4


def test_token_authority_processed_geometry_mismatch_rejects_without_fallback(tmp_path: Path):
    ctc, segments = _kpop_source(tmp_path)
    source_word = segments["demo"][0]["words"][0]
    _textgrid(ctc / "demo.TextGrid", {"words": [(0.0, 1.0, "kpop")]})
    (ctc / "demo.lab").write_text("kpop\n", encoding="utf-8")
    token = {
        "word": "kpop", "start_s": 0.0, "end_s": 1.4,
        "processed_ctc_span": [0.0, 1.4],
        "processed_ctc_boundary_source": "next_lexical_token_start",
        "surface_text": source_word["text"],
        "source_ctc_ordinals": source_word["source_ctc_ordinals"],
        "canonical_span": source_word["canonical_span"],
        "canonical_unit": source_word["canonical_unit"],
        "canonical_unit_sha256": hashlib.sha256(
            json.dumps(source_word["canonical_unit"], ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reference_identity": hashlib.sha256(b"K-Pop").hexdigest(),
        "reference_ordinal": 0,
    }
    (ctc / "demo_tokens.jsonl").write_text(
        json.dumps(token, ensure_ascii=False) + "\n", encoding="utf-8")

    result = producer.find_english_segments(ctc, ["demo"])
    assert result["demo"][0]["canonical_reject_reason"] == (
        "canonical_token_textgrid_geometry_mismatch")


def test_token_authority_short_processed_end_rejects_without_fallback(tmp_path: Path):
    ctc, segments = _kpop_source(tmp_path)
    source_word = segments["demo"][0]["words"][0]
    short_span = [0.0, 0.9]
    _textgrid(ctc / "demo.TextGrid", {"words": [(0.0, 0.9, "kpop")]})
    (ctc / "demo.lab").write_text("kpop\n", encoding="utf-8")
    token = {
        "word": "kpop", "start_s": short_span[0], "end_s": short_span[1],
        "processed_ctc_span": short_span,
        "processed_ctc_boundary_source": "raw_end_fallback",
        "surface_text": source_word["text"],
        "source_ctc_ordinals": source_word["source_ctc_ordinals"],
        "canonical_span": source_word["canonical_span"],
        "canonical_unit": source_word["canonical_unit"],
        "canonical_unit_sha256": hashlib.sha256(
            json.dumps(source_word["canonical_unit"], ensure_ascii=False,
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reference_identity": hashlib.sha256(b"K-Pop").hexdigest(),
        "reference_ordinal": 0,
    }
    (ctc / "demo_tokens.jsonl").write_text(
        json.dumps(token, ensure_ascii=False) + "\n", encoding="utf-8")

    result = producer.find_english_segments(ctc, ["demo"])
    assert result["demo"][0]["canonical_reject_reason"] == (
        "processed_geometry_end_before_canonical_end")


def test_strict_ledger_rejects_short_processed_end_before_phone_validation():
    word = _direct_word("OK")
    word["processed_ctc_span"] = [0.0, 0.8]
    word["processed_ctc_boundary_source"] = "raw_end_fallback"
    with pytest.raises(ValueError, match="processed_geometry_end_before_canonical_end"):
        producer._strict_verified_words(
            "demo:s0", {"words": [word]},
            [{"ordinal": 0, "text": "ok", "start": 0.0, "end": 0.8}],
            [{"ordinal": 0, "text": "K", "start": 0.0, "end": 0.8}],
        )


def test_alpha_digit_authority_units_merge_ordered_ctc_fragments(tmp_path: Path):
    ctc = tmp_path / "ctc"
    ctc.mkdir()
    _textgrid(ctc / "demo.TextGrid", {"words": [
        (0.0, 0.2, "target"), (0.2, 0.3, "1"),
        (0.3, 0.6, "target2"),
    ]})
    (ctc / "demo.lab").write_text("target 1 target2\n", encoding="utf-8")
    (ctc / "demo_ref.txt").write_text("target1 target2\n", encoding="utf-8")

    segments = producer.find_english_segments(ctc, ["demo"])
    words = segments["demo"][0]["words"]

    assert [word["text"] for word in words] == ["target1", "target2"]
    assert [word["alignment_token"] for word in words] == ["target", "target"]
    assert [word["unit_id"] for word in words] == ["en-u0000", "en-u0001"]
    assert [word["source_ctc_ordinals"] for word in words] == [[0, 1], [2]]
    assert [word["canonical_span"] for word in words] == [[0.0, 0.3], [0.3, 0.6]]


def test_corpus_min_duration_is_checked_after_effective_padding(tmp_path: Path):
    from scipy.io import wavfile
    import numpy as np

    audio_dir = tmp_path / "audio"
    corpus_dir = tmp_path / "corpus"
    audio_dir.mkdir()
    corpus_dir.mkdir()
    wavfile.write(audio_dir / "demo.wav", 16000, np.zeros(16000, dtype=np.int16))
    segment = {
        "seg_idx": 0, "segment_ordinal": 0,
        "words": [_direct_word("OK")],
        "seg_start": 0.40, "seg_end": 0.52,
    }

    stem, result = producer._build_corpus_stem(
        "demo", [segment], audio_dir, corpus_dir,
        padding_s=0.05, min_dur_s=0.15, strict=True)

    assert stem == "demo"
    assert result[0]["skipped"] is False
    assert result[0]["raw_duration_s"] == pytest.approx(0.12)
    assert result[0]["padded_duration_s"] == pytest.approx(0.22)
    assert result[0]["padding_s"] == {
        "requested": 0.05, "left": pytest.approx(0.05),
        "right": pytest.approx(0.05)}


def test_exact_150ms_effective_clip_is_not_rejected_by_float_roundoff(tmp_path: Path):
    from scipy.io import wavfile
    import numpy as np

    audio_dir = tmp_path / "audio"
    corpus_dir = tmp_path / "corpus"
    audio_dir.mkdir()
    corpus_dir.mkdir()
    wavfile.write(audio_dir / "demo.wav", 32000, np.zeros(4 * 32000, dtype=np.int16))
    segment = {
        "seg_idx": 0, "segment_ordinal": 0,
        "words": [_direct_word("ria")],
        "seg_start": 2.62, "seg_end": 2.67,
    }

    _, result = producer._build_corpus_stem(
        "demo", [segment], audio_dir, corpus_dir,
        padding_s=0.05, min_dur_s=0.15, strict=True)

    assert result[0]["padded_duration_s"] == pytest.approx(0.15)
    assert result[0]["skipped"] is False


def test_corrected_100ms_owner_passes_200ms_corpus_floor_without_weakening(tmp_path: Path):
    from scipy.io import wavfile
    import numpy as np

    audio_dir = tmp_path / "audio"
    corpus_dir = tmp_path / "corpus"
    audio_dir.mkdir()
    corpus_dir.mkdir()
    wavfile.write(audio_dir / "demo.wav", 16000, np.zeros(16000, dtype=np.int16))
    segments = [
        {"seg_idx": 0, "segment_ordinal": 0,
         "words": [_direct_word("ria")],
         "seg_start": 0.40, "seg_end": 0.50},
        {"seg_idx": 1, "segment_ordinal": 1,
         "words": [_direct_word("Go", 1)],
         "seg_start": 0.70, "seg_end": 0.79},
    ]

    _, result = producer._build_corpus_stem(
        "demo", segments, audio_dir, corpus_dir,
        padding_s=0.05, min_dur_s=0.20, strict=True)

    assert result[0]["raw_duration_s"] == pytest.approx(0.10)
    assert result[0]["padded_duration_s"] == pytest.approx(0.20)
    assert result[0]["skipped"] is False
    assert result[1]["raw_duration_s"] == pytest.approx(0.09)
    assert result[1]["padded_duration_s"] == pytest.approx(0.19)
    assert result[1]["skipped"] is True
    assert result[1]["reject_reason"] == "segment_too_short"


def test_pinyin_tone_token_is_not_an_english_authority_unit(tmp_path: Path):
    ctc = tmp_path / "ctc"
    ctc.mkdir()
    _textgrid(ctc / "demo.TextGrid", {"words": [(0.0, 0.3, "jin1")]})
    (ctc / "demo.lab").write_text("jin1\n", encoding="utf-8")
    (ctc / "demo_ref.txt").write_text("jin1\n", encoding="utf-8")

    assert producer.find_english_segments(ctc, ["demo"]) == {}


def _nonzero_ordinal_source(tmp_path: Path) -> tuple[Path, dict]:
    ctc = tmp_path / "ctc"
    ctc.mkdir()
    words = [(0.0, 0.4, "target"), (0.4, 0.8, "OK"),
             (0.8, 1.2, "target")]
    _textgrid(ctc / "demo.TextGrid", {"words": words})
    (ctc / "demo.lab").write_text("target OK target\n", encoding="utf-8")
    (ctc / "demo_ref.txt").write_text("target OK target\n", encoding="utf-8")
    return ctc, producer.find_english_segments(ctc, ["demo"])


def test_validated_unit_uses_raw_nonzero_reference_ordinal(tmp_path: Path):
    _, segments = _nonzero_ordinal_source(tmp_path)
    words = segments["demo"][0]["words"]

    ok_unit = producer._validated_unit(words[1])

    assert words[1]["text"] == "OK"
    assert words[1]["unit_id"] == "en-u0001"
    assert ok_unit.unit_id == "en-u0001"
    assert ok_unit.reference_ordinal == 1
    assert ok_unit.source_ctc_ordinals == (1,)
    assert [producer._validated_unit(word).unit_id for word in (words[0], words[2])] == [
        "en-u0000", "en-u0002"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [("unit_id", "en-u0000"), ("reference_ordinal", 0)],
)
def test_nonzero_ordinal_unit_id_or_ordinal_tamper_fails_closed(
        tmp_path: Path, field: str, value):
    _, segments = _nonzero_ordinal_source(tmp_path)
    tampered = deepcopy(segments["demo"][0]["words"][1])
    tampered["canonical_unit"][field] = value

    with pytest.raises(producer.EnglishUnitError):
        producer._validated_unit(tampered)


def test_missing_reference_supports_direct_hyphenless_unit(tmp_path: Path):
    ctc = tmp_path / "ctc"
    ctc.mkdir()
    _textgrid(ctc / "demo.TextGrid", {"words": [(0.2, 1.0, "OpenAI")]})
    (ctc / "demo.lab").write_text("OpenAI\n", encoding="utf-8")

    segments = producer.find_english_segments(ctc, ["demo"])
    word = segments["demo"][0]["words"][0]
    assert word["text"] == "OpenAI"
    assert word["alignment_token"] == "openai"
    assert word["canonical_unit"]["merge_kind"] == "direct"


def test_strict_v2_ledger_has_one_verified_word_and_full_binding(tmp_path: Path):
    ctc, segments = _kpop_source(tmp_path)
    seg = segments["demo"][0]
    seg["seg_name"] = "demo_seg0"
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    _textgrid(aligned / "demo_seg0.TextGrid", {
        "phones": [(0.0, 0.4, "K"), (0.4, 0.7, "AA"), (0.7, 1.0, "P")],
        "words": [(0.0, 1.0, "kpop")],
    })
    output = tmp_path / "output"
    dictionary = tmp_path / "dict.dict"
    dictionary.write_text("hello HH EH L OW\n", encoding="utf-8")
    mfa = {
        "dictionary": str(dictionary),
        "dictionary_sha256": hashlib.sha256(dictionary.read_bytes()).hexdigest(),
    }
    expected, counts = producer._strict_expected_snapshot(segments)

    manifest_path = producer.produce_strict_ledgers(
        segments, ctc, aligned, output, mfa,
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads((output / "demo_en_phones.json").read_text(encoding="utf-8"))
    word = ledger["segments"][0]["words"][0]

    assert manifest["schema"] == "strict-en-mfa-v2"
    assert manifest["status"] == "success"
    assert ledger["schema"] == "strict-en-mfa-v2"
    assert ledger["canonical_units"] == "canonical-english-units-v1"
    assert len(ledger["segments"]) == 1
    assert len(ledger["segments"][0]["words"]) == 1
    assert word["status"] == "verified"
    assert word["unit_id"] == "en-u0000"
    assert word["alignment_token"] == "kpop"
    assert word["canonical_span"] == [0.0, 1.0]
    assert word["source_ctc_ordinals"] == [0, 1]
    assert word["canonical_binding"] == "canonical-english-units-v1"
    assert word["dictionary_provenance"]["sha256"] == mfa["dictionary_sha256"]
    assert [phone["label"] for phone in word["phones"]] == ["K", "AA", "P"]


def test_tampered_unit_is_rejected_without_fabricated_phones(tmp_path: Path):
    ctc, segments = _kpop_source(tmp_path)
    seg = segments["demo"][0]
    seg["seg_name"] = "demo_seg0"
    seg["words"][0]["canonical_unit"]["alignment_token"] = "not-kpop"
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    _textgrid(aligned / "demo_seg0.TextGrid", {
        "phones": [(0.0, 0.4, "K"), (0.4, 0.7, "AA"), (0.7, 1.0, "P")],
        "words": [(0.0, 1.0, "kpop")],
    })
    output = tmp_path / "output"
    expected, counts = producer._strict_expected_snapshot(segments)
    manifest_path = producer.produce_strict_ledgers(
        segments, ctc, aligned, output, {},
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads((output / "demo_en_phones.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "strict-en-mfa-v2"
    assert manifest["status"] == "partial"
    record = ledger["segments"][0]
    assert record["status"] == "rejected"
    assert record["words"][0]["phones"] == []


def test_partial_manifest_accepts_rejected_segment_with_complete_partition(tmp_path: Path):
    ctc, good_segments = _kpop_source(tmp_path)
    good_segments["demo"][0]["seg_name"] = "demo_seg0"
    _textgrid(ctc / "bad.TextGrid", {"words": [(0.0, 0.2, "OK")]})
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    _textgrid(aligned / "demo_seg0.TextGrid", {
        "phones": [(0.0, 0.4, "K"), (0.4, 0.7, "AA"), (0.7, 1.0, "P")],
        "words": [(0.0, 1.0, "kpop")],
    })
    segments = {
        "demo": good_segments["demo"],
        "bad": [{"segment_ordinal": 0, "seg_name": "bad_seg0",
                 "skipped": True, "reject_reason": "segment_too_short",
                 "words": [_direct_word("OK")]}],
    }
    expected, counts = producer._strict_expected_snapshot(segments)
    manifest_path = producer.produce_strict_ledgers(
        segments, ctc, aligned, tmp_path / "out", {},
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "partial"
    assert manifest["produced_segments"] == ["demo:s0"]
    assert manifest["rejected_segments"] == [
        {"id": "bad:s0", "reason": "segment_too_short"}
    ]
    assert producer._strict_manifest_succeeded(manifest_path) is True


def test_historical_v1_manifest_is_not_fresh_v2_success(tmp_path: Path):
    path = tmp_path / "en_alignment_manifest.json"
    path.write_text(json.dumps({"schema": "strict-en-mfa-v1", "status": "success"}), encoding="utf-8")
    assert producer._strict_manifest_succeeded(path) is False


def test_dictionary_is_run_local_and_does_not_modify_repository_bytes(tmp_path: Path, monkeypatch):
    base = tmp_path / "base.dict"
    original = b"hello HH EH L OW\n"
    base.write_bytes(original)
    model = tmp_path / "g2p.zip"
    model.write_bytes(b"model")
    unit = producer.parse_english_units("K-Pop")[0]
    merged = producer.merge_authority_fragment_group(
        unit, [{"text": "KPop", "ordinal": 0, "start": 0.0, "end": 1.0}],
    )
    word = {
        "text": merged.surface_text,
        "unit_id": merged.unit_id,
        "alignment_token": merged.alignment_token,
        "source_ctc_ordinals": list(merged.source_ctc_ordinals),
        "canonical_span": list(merged.canonical_span),
        "canonical_unit": {**merged.to_dict(), "canonical_binding": "canonical-english-units-v1"},
    }

    def fake_run(command, **kwargs):
        Path(command[6]).write_text("kpop K P\n", encoding="utf-8")
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(producer.subprocess, "run", fake_run)
    result = producer.build_en_dict(
        {"demo": [{"words": [word]}]}, base, model,
        Path("/usr/bin/python"), tmp_path / "models", tmp_path / "run",
    )

    assert result.parent == tmp_path / "run"
    assert base.read_bytes() == original
    assert "kpop K P" in result.read_text(encoding="utf-8")


def _direct_word(surface: str, ordinal: int = 0) -> dict:
    unit = producer.parse_english_units(surface)[0]
    merged = producer.merge_authority_fragment_group(
        unit, [{"text": surface, "ordinal": ordinal, "start": 0.0, "end": 1.0}],
    )
    return {
        "text": merged.surface_text,
        "unit_id": merged.unit_id,
        "alignment_token": merged.alignment_token,
        "source_ctc_ordinals": list(merged.source_ctc_ordinals),
        "canonical_span": list(merged.canonical_span),
        "canonical_unit": {
            **merged.to_dict(), "canonical_binding": "canonical-english-units-v1",
        },
        "start": 0.0,
        "end": 1.0,
        "ordinal": ordinal,
    }


def test_sos_run_local_override_preserves_app_and_base_bytes(tmp_path: Path):
    base = tmp_path / "base.dict"
    original = b"APP AE1 P\nSOS EH2 OW2 EH1 S\nSOSA S OW1 S AH0\n"
    base.write_bytes(original)
    segments = {"demo": [{"words": [_direct_word("SOS"), _direct_word("APP", 1)]}]}

    result = producer.build_en_dict(
        segments, base, tmp_path / "unused-g2p.zip", Path("python"), tmp_path / "models",
        tmp_path / "run", strict=True,
    )
    rows = [line.split() for line in result.read_text(encoding="utf-8").splitlines()]
    assert [row for row in rows if row[0].split("(", 1)[0].casefold() == "sos"] == [
        ["SOS", "EH2", "S", "OW2", "EH1", "S"]
    ]
    assert [row for row in rows if row[0].split("(", 1)[0].casefold() == "app"] == [
        ["APP", "AE1", "P"]
    ]
    assert base.read_bytes() == original


def test_sos_five_phone_record_has_exact_policy_and_provenance(tmp_path: Path):
    base = tmp_path / "base.dict"
    base.write_text("APP AE1 P\nSOS EH2 OW2 EH1 S\n", encoding="utf-8")
    dictionary = producer.build_en_dict(
        {"demo": [{"words": [_direct_word("SOS")]}]}, base,
        tmp_path / "unused-g2p.zip", Path("python"), tmp_path / "models", tmp_path / "run",
        strict=True,
    )
    _textgrid(tmp_path / "ctc.TextGrid", {"words": [(0.0, 1.0, "SOS")]})
    (tmp_path / "ctc_ref.txt").write_text("SOS\n", encoding="utf-8")
    _textgrid(tmp_path / "aligned.TextGrid", {
        "words": [(0.0, 1.0, "sos")],
        "phones": [
            (0.0, 0.2, "EH2"), (0.2, 0.4, "S"), (0.4, 0.6, "OW2"),
            (0.6, 0.8, "EH1"), (0.8, 1.0, "S"),
        ],
    })
    segments = producer.find_english_segments(tmp_path, ["ctc"])
    segments["ctc"][0]["seg_name"] = "aligned"
    expected, counts = producer._strict_expected_snapshot(segments)
    manifest_path = producer.produce_strict_ledgers(
        segments, tmp_path, tmp_path, tmp_path / "out",
        {"dictionary": str(dictionary), "dictionary_sha256": producer._sha256(dictionary)},
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    word = json.loads((tmp_path / "out" / "ctc_en_phones.json").read_text(encoding="utf-8"))["segments"][0]["words"][0]
    assert manifest["status"] == "success"
    assert [phone["label"] for phone in word["phones"]] == list(producer.SOS_EXPECTED_PRONUNCIATION)
    assert word["pronunciation_policy"]["policy_id"] == producer.SOS_PRONUNCIATION_POLICY_ID
    assert word["pronunciation_policy"]["expected_pronunciation"] == list(producer.SOS_EXPECTED_PRONUNCIATION)
    assert word["pronunciation_policy"]["actual_source_sequence"] == list(producer.SOS_EXPECTED_PRONUNCIATION)
    assert word["pronunciation_policy"]["dictionary_provenance"]["sha256"] == producer._sha256(dictionary)


@pytest.mark.parametrize("labels", [
    ("EH2", "OW2", "EH1", "S"),
    ("S", "OW2", "EH1", "S"),
    ("EH2", "S", "EH1", "OW2", "S"),
])
def test_sos_old_missing_or_reordered_source_sequence_fails_closed(tmp_path: Path, labels):
    dictionary = tmp_path / "dict.dict"
    dictionary.write_text("SOS EH2 S OW2 EH1 S\n", encoding="utf-8")
    segment = {"words": [_direct_word("SOS")]}
    phones = []
    step = 1.0 / len(labels)
    for index, label in enumerate(labels):
        phones.append((index * step, (index + 1) * step, label))
    _textgrid(tmp_path / "ctc.TextGrid", {"words": [(0.0, 1.0, "SOS")]})
    _textgrid(tmp_path / "aligned.TextGrid", {
        "words": [(0.0, 1.0, "sos")], "phones": phones,
    })
    segment["seg_name"] = "aligned"
    segments = {"ctc": [{"segment_ordinal": 0, **segment}]}
    expected, counts = producer._strict_expected_snapshot(segments)
    manifest_path = producer.produce_strict_ledgers(
        segments, tmp_path, tmp_path, tmp_path / "out",
        {"dictionary": str(dictionary), "dictionary_sha256": producer._sha256(dictionary)},
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["rejected_segments"][0]["reason"] == "sos_expected_pronunciation_mismatch"


def test_sos_tampered_dictionary_hash_fails_closed(tmp_path: Path):
    dictionary = tmp_path / "dict.dict"
    dictionary.write_text("SOS EH2 S OW2 EH1 S\n", encoding="utf-8")
    _textgrid(tmp_path / "ctc.TextGrid", {"words": [(0.0, 1.0, "SOS")]})
    _textgrid(tmp_path / "aligned.TextGrid", {
        "words": [(0.0, 1.0, "sos")],
        "phones": [
            (0.0, 0.2, "EH2"), (0.2, 0.4, "S"), (0.4, 0.6, "OW2"),
            (0.6, 0.8, "EH1"), (0.8, 1.0, "S"),
        ],
    })
    segment = _direct_word("SOS")
    segments = {"ctc": [{"segment_ordinal": 0, "seg_name": "aligned", "words": [segment]}]}
    expected, counts = producer._strict_expected_snapshot(segments)
    manifest_path = producer.produce_strict_ledgers(
        segments, tmp_path, tmp_path, tmp_path / "out",
        {"dictionary": str(dictionary), "dictionary_sha256": "tampered"},
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["rejected_segments"][0]["reason"] == "sos_dictionary_hash_mismatch"


def test_app_fake_second_p_is_rejected(tmp_path: Path):
    base = tmp_path / "base.dict"
    base.write_text("APP AE1 P P\n", encoding="utf-8")
    with pytest.raises(producer.StrictG2PError, match="app_pronunciation_missing_or_tampered"):
        producer.build_en_dict(
            {"demo": [{"words": [_direct_word("APP")]}]}, base,
            tmp_path / "unused-g2p.zip", Path("python"), tmp_path / "models", tmp_path / "run",
            strict=True,
        )


def test_app_fake_second_p_in_mfa_evidence_is_rejected(tmp_path: Path):
    dictionary = tmp_path / "dict.dict"
    dictionary.write_text("APP AE1 P\n", encoding="utf-8")
    _textgrid(tmp_path / "ctc.TextGrid", {"words": [(0.0, 1.0, "APP")]})
    _textgrid(tmp_path / "aligned.TextGrid", {
        "words": [(0.0, 1.0, "app")],
        "phones": [(0.0, 0.25, "AE1"), (0.25, 0.6, "P"), (0.6, 1.0, "P")],
    })
    segments = {"ctc": [{"segment_ordinal": 0, "seg_name": "aligned",
                         "words": [_direct_word("APP")]}]}
    expected, counts = producer._strict_expected_snapshot(segments)
    manifest_path = producer.produce_strict_ledgers(
        segments, tmp_path, tmp_path, tmp_path / "out",
        {"dictionary": str(dictionary), "dictionary_sha256": producer._sha256(dictionary)},
        expected_segments=expected, expected_counts=counts,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["rejected_segments"][0]["reason"] == "app_expected_pronunciation_mismatch"


def test_singleton_retry_recovers_only_validated_missing_textgrid(
        tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    aligned = tmp_path / "aligned"
    corpus.mkdir(); aligned.mkdir()
    (corpus / "demo_seg0.wav").write_bytes(b"exact wav evidence")
    (corpus / "demo_seg0.lab").write_text("mira vtuber\n", encoding="utf-8")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        retry_aligned = args[3]
        retry_aligned.mkdir(parents=True)
        _textgrid(retry_aligned / "demo_seg0.TextGrid", {
            "words": [(0.0, 0.4, "mira"), (0.4, 1.0, "vtuber")],
            "phones": [(0.0, 0.2, "M"), (0.2, 0.4, "AH0"),
                       (0.4, 0.7, "V"), (0.7, 1.0, "ER0")],
        })
        return {"return_code": 0, "command": ["mfa"],
                "timed_out": False, "timeout_seconds": kwargs["timeout"],
                "acoustic_model": "model", "exception": ""}

    monkeypatch.setattr(producer, "run_en_mfa", fake_run)
    records = producer.retry_missing_en_segments(
        {"demo": [{"seg_name": "demo_seg0", "seg_idx": 0,
                   "skipped": False}]},
        corpus, aligned, tmp_path / "dict", "model", tmp_path / "temp",
        tmp_path / "python", tmp_path / "models",
        beam=100, retry_beam=1000, timeout=321, limit=1)

    assert records[0]["status"] == "recovered"
    assert (aligned / "demo_seg0.TextGrid").is_file()
    assert records[0]["textgrid_sha256"] == producer._sha256(
        aligned / "demo_seg0.TextGrid")
    assert calls[0][1]["beam"] == 100
    assert calls[0][1]["retry_beam"] == 1000
    assert calls[0][1]["timeout"] == 321


def test_singleton_retry_limit_keeps_missing_segment_fail_closed(
        tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    aligned = tmp_path / "aligned"
    corpus.mkdir(); aligned.mkdir()
    (corpus / "demo_seg0.wav").write_bytes(b"wav")
    (corpus / "demo_seg0.lab").write_text("mira\n", encoding="utf-8")
    monkeypatch.setattr(
        producer, "run_en_mfa",
        lambda *args, **kwargs: pytest.fail("retry must respect limit"))

    records = producer.retry_missing_en_segments(
        {"demo": [{"seg_name": "demo_seg0", "seg_idx": 0,
                   "skipped": False}]},
        corpus, aligned, tmp_path / "dict", "model", tmp_path / "temp",
        tmp_path / "python", tmp_path / "models", limit=0)

    assert records == [{"segment_name": "demo_seg0",
                        "status": "retry_limit_exceeded",
                        "reason": "singleton_retry_limit_exceeded"}]
    assert not (aligned / "demo_seg0.TextGrid").exists()


def test_run_en_mfa_overrides_shared_mutable_environment(
        tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "demo.wav").write_bytes(b"wav")
    captured = {}

    def fake_subprocess(command, **kwargs):
        captured.update({"command": command, **kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("MFA_ROOT_DIR", str(tmp_path / "shared-mfa-root"))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "shared-numba"))
    monkeypatch.setattr(producer.subprocess, "run", fake_subprocess)
    outcome = producer.run_en_mfa(
        corpus, tmp_path / "dict", str(tmp_path / "model"),
        tmp_path / "aligned", tmp_path / "run-local", tmp_path / "python",
        tmp_path / "models", strict=True)

    assert outcome["return_code"] == 0
    assert captured["env"]["MFA_ROOT_DIR"] == str(
        tmp_path / "run-local" / "mfa_root")
    assert captured["env"]["NUMBA_CACHE_DIR"] == str(
        tmp_path / "run-local" / "numba_cache")
    assert captured["env"]["MFA_ROOT_DIR"] != os.environ["MFA_ROOT_DIR"]
    assert outcome["environment"] == {
        "MFA_ROOT_DIR": captured["env"]["MFA_ROOT_DIR"],
        "NUMBA_CACHE_DIR": captured["env"]["NUMBA_CACHE_DIR"],
    }
    strict_record = producer._strict_mfa_record(
        outcome, tmp_path / "model", tmp_path / "dict")
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest_path = producer.write_strict_manifest(
        manifest_dir, "failed", mfa=strict_record,
        expected_segments=[], produced_segments=[], rejected_segments=[],
        stem_ledgers=[], counts={}, reason="test")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mfa"]["environment"] == outcome["environment"]
