"""Focused authority-mode CTC English-unit producer/validator tests."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import ctc_prealign as ctc
from scripts import ctc_processed_geometry as geometry
from scripts.ctc_processed_geometry import (
    CanonicalNextOwnerConflict,
    resolve_processed_english_spans,
)
from scripts.adjust_ctc_boundaries import _read_pause_intervals
from scripts.pipeline_utils import validate_ctc_authority_bundle


def _row(text: str, ordinal: int, start: float, end: float) -> dict:
    return {
        "word": text,
        "start": start,
        "end": end,
        "source_ctc_ordinal": ordinal,
    }


def _positive_bundle(tmp_path: Path) -> tuple[Path, dict]:
    reference = "你K-Pop好"
    merged = ctc._merge_reference_english_fragments(
        [
            _row("ni3", 3, 0.10, 0.20),
            _row("K", 4, 0.20, 0.30),
            _row("Pop", 5, 0.30, 0.50),
            _row("hao3", 6, 0.50, 0.60),
        ],
        reference,
    )
    canonical = next(row for row in merged if row["word"] == "kpop")
    assert canonical["surface_text"] == "K-Pop"
    assert canonical["source_ctc_ordinals"] == [4, 5]
    assert canonical["canonical_span"] == [0.20, 0.50]

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "demo_ref.txt").write_text(reference + "\n", encoding="utf-8")
    (tmp_path / "demo.lab").write_text("ni3 kpop hao3\n", encoding="utf-8")
    ctc.write_textgrid(merged, 1.0, tmp_path / "demo.TextGrid")
    lines = []
    for row in merged:
        line = {
            "word": row["word"],
            "start_s": row["start"],
            "end_s": row["end"],
        }
        for key in (
                "surface_text", "source_ctc_ordinals", "canonical_span",
                "canonical_unit", "canonical_unit_sha256",
                "reference_identity", "reference_ordinal"):
            if key in row:
                line[key] = row[key]
        if "canonical_unit" in row:
            line["processed_ctc_span"] = [row["start"], row["end"]]
            line["processed_ctc_boundary_source"] = "raw_end_fallback"
        lines.append(json.dumps(line, ensure_ascii=False))
    (tmp_path / "demo_tokens.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path, canonical


def test_authority_kpop_is_one_canonical_ctc_word_and_lexical_interval(tmp_path):
    bundle, canonical = _positive_bundle(tmp_path)

    assert [row["word"] for row in json.loads(
        "[" + ",".join(
            line for line in (bundle / "demo_tokens.jsonl").read_text().splitlines()
        ) + "]"
    )] == ["ni3", "kpop", "hao3"]
    assert ctc._validate_all_ctc_bundles(bundle, {"demo": "你K-Pop好"})
    assert validate_ctc_authority_bundle(bundle, "demo", "你K-Pop好") == []
    textgrid = (bundle / "demo.TextGrid").read_text(encoding="utf-8")
    assert textgrid.count('text = "kpop"') == 1
    assert canonical["canonical_unit_sha256"]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (("KP",), "partial"),
        (("K", "Pop", "extra"), "extra"),
        (("op", "KP"), "mismatch"),
        (("K", "中", "Pop"), "cjk"),
        (("K", "QUESTION-YI", "Pop"), "nvv"),
        (("K", ",", "Pop"), "punctuation"),
    ],
)
def test_authority_candidates_fail_closed(candidate, expected):
    rows = [_row(text, index, index * 0.1, (index + 1) * 0.1)
            for index, text in enumerate(candidate)]
    with pytest.raises(ValueError, match=expected):
        ctc._merge_reference_english_fragments(rows, "K-Pop")


def test_authority_dropped_hyphen_gap_is_bound_to_the_compound_owner():
    merged = ctc._merge_reference_english_fragments(
        [_row("K", 1, 0.1, 0.2), _row("Pop", 3, 0.3, 0.4)],
        "K-Pop",
    )
    assert len(merged) == 1
    assert merged[0]["surface_text"] == "K-Pop"
    assert merged[0]["source_ctc_ordinals"] == [1, 3]
    assert merged[0]["hyphen_separator_omitted"] is True


@pytest.mark.parametrize("field", ["reference_identity", "canonical_unit_sha256"])
def test_authority_metadata_tamper_is_rejected(tmp_path, field):
    bundle, _ = _positive_bundle(tmp_path)
    path = bundle / "demo_tokens.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1][field] = "0" * 64
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    errors = validate_ctc_authority_bundle(bundle, "demo", "你K-Pop好")
    assert any("mismatch" in error for error in errors)


def test_authority_span_tolerance_is_three_ms(tmp_path):
    bundle, _ = _positive_bundle(tmp_path)
    path = bundle / "demo_tokens.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["canonical_span"][1] += 0.003
    rows[1]["canonical_unit"]["canonical_end"] += 0.003
    rows[1]["canonical_unit_sha256"] = hashlib.sha256(
        json.dumps(rows[1]["canonical_unit"], ensure_ascii=False,
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert validate_ctc_authority_bundle(bundle, "demo", "你K-Pop好") == []


def test_authority_source_ordinal_tamper_is_rejected(tmp_path):
    bundle, _ = _positive_bundle(tmp_path)
    path = bundle / "demo_tokens.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["source_ctc_ordinals"] = [4, 6]
    rows[1]["canonical_unit"]["source_ctc_ordinals"] = [4, 6]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    errors = validate_ctc_authority_bundle(bundle, "demo", "你K-Pop好")
    assert any("source ordinal" in error for error in errors)


def _canonical_timed_row(start: float = 0.06, end: float = 0.12) -> dict:
    merged = ctc._merge_reference_english_fragments(
        [_row("K", 0, start, end)], "K")[0]
    return merged


def _energy_wav(path: Path, *, duration: float = 0.8,
                floor: float = 0.0001,
                active: tuple[tuple[float, float, float], ...] = ()) -> Path:
    import numpy as np
    from scipy.io import wavfile

    sample_rate = 16000
    samples = int(round(duration * sample_rate))
    signs = np.where(np.arange(samples) % 2, 1.0, -1.0).astype(np.float32)
    audio = signs * np.float32(floor)
    for start, end, amplitude in active:
        left = int(round(start * sample_rate))
        right = int(round(end * sample_rate))
        audio[left:right] = signs[left:right] * np.float32(amplitude)
    wavfile.write(path, sample_rate, audio)
    return path


def test_ctc_prealign_only_aliases_the_shared_production_resolver():
    assert ctc.resolve_processed_english_spans is resolve_processed_english_spans
    assert ctc._resolve_processed_english_spans is resolve_processed_english_spans


def test_read_pause_intervals_discards_empty_pause_labels(tmp_path):
    textgrid = tmp_path / "demo.TextGrid"
    textgrid.write_text(
        '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "pauses"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0.1
            xmax = 0.3
            text = "   "
        intervals [2]:
            xmin = 0.4
            xmax = 0.7
            text = "long pause"
''',
        encoding="utf-8",
    )
    assert _read_pause_intervals(textgrid) == [
        {"start_s": 0.4, "end_s": 0.7}
    ]


def test_processed_boundary_inside_canonical_word_is_ignored():
    english = _canonical_timed_row(0.06, 0.12)
    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.70, "end": 0.80}],
        [{"start_s": 0.08, "end_s": 0.10}],
        [{"start_ms": 80, "end_ms": 280}],
        1.0,
    )
    assert english["processed_ctc_span"] == [0.06, 0.70]
    assert english["processed_ctc_boundary_source"] == "next_lexical_token_start"


@pytest.mark.parametrize(
    ("pause_end_ms", "expected_end", "expected_source"),
    [(399, 0.70, "next_lexical_token_start"),
     (400, 0.20, "raw_end_long_pause")],
)
def test_long_pause_threshold_is_199_or_200_ms(
        pause_end_ms, expected_end, expected_source):
    english = _canonical_timed_row()
    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.70, "end": 0.80}],
        [], [{"start_ms": 200, "end_ms": pause_end_ms}], 1.0)
    assert english["processed_ctc_span"] == [0.06, expected_end]
    assert english["processed_ctc_boundary_source"] == expected_source


def test_active_declared_pause_uses_first_sustained_silence_onset(tmp_path):
    wav = _energy_wav(
        tmp_path / "active-tail.wav",
        active=((0.08, 0.235, 0.01), (0.60, 0.72, 0.01)))
    english = _canonical_timed_row(0.10, 0.16)
    original = deepcopy(english)

    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.60, "end": 0.72}],
        [], [{"start_s": 0.19, "end_s": 0.50}], 0.8, wav)

    assert english["processed_ctc_span"] == [0.10, 0.235]
    assert english["processed_ctc_boundary_source"] == "energy_end_hard_boundary"
    assert english["canonical_span"] == original["canonical_span"]
    assert english["canonical_unit"] == original["canonical_unit"]
    assert english["canonical_unit_sha256"] == original["canonical_unit_sha256"]


def test_true_silence_at_declared_pause_onset_is_unchanged(tmp_path):
    wav = _energy_wav(
        tmp_path / "true-pause.wav",
        active=((0.08, 0.20, 0.01), (0.60, 0.72, 0.01)))
    english = _canonical_timed_row(0.10, 0.16)

    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.60, "end": 0.72}],
        [], [{"start_s": 0.20, "end_s": 0.50}], 0.8, wav)

    assert english["processed_ctc_span"] == [0.10, 0.20]
    assert english["processed_ctc_boundary_source"] == "raw_end_long_pause"


def test_false_pause_without_sustained_silence_falls_to_next_lexical(tmp_path):
    wav = _energy_wav(
        tmp_path / "no-silence.wav",
        active=((0.08, 0.50, 0.01), (0.60, 0.72, 0.01)))
    english = _canonical_timed_row(0.10, 0.16)

    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.60, "end": 0.72}],
        [], [{"start_s": 0.20, "end_s": 0.50}], 0.8, wav)

    assert english["processed_ctc_span"] == [0.10, 0.60]
    assert english["processed_ctc_boundary_source"] == "next_lexical_token_start"


def test_punctuation_before_acoustic_fall_remains_hard(tmp_path):
    wav = _energy_wav(
        tmp_path / "punctuation.wav",
        active=((0.08, 0.35, 0.01), (0.60, 0.72, 0.01)))
    english = _canonical_timed_row(0.10, 0.16)

    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.60, "end": 0.72}],
        [{"start_s": 0.30, "end_s": 0.32}],
        [{"start_s": 0.20, "end_s": 0.50}], 0.8, wav)

    assert english["processed_ctc_span"] == [0.10, 0.30]
    assert english["processed_ctc_boundary_source"] == "raw_end_punctuation"


@pytest.mark.parametrize("wav_name", [None, "unreadable.wav"])
def test_missing_or_unreadable_audio_keeps_declared_pause_boundary(
        tmp_path, wav_name):
    wav = None
    if wav_name is not None:
        wav = tmp_path / wav_name
        wav.write_text("not a wav\n", encoding="utf-8")
    english = _canonical_timed_row(0.10, 0.16)

    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.60, "end": 0.72}],
        [], [{"start_s": 0.20, "end_s": 0.50}], 0.8, wav)

    assert english["processed_ctc_span"] == [0.10, 0.20]
    assert english["processed_ctc_boundary_source"] == "raw_end_long_pause"


def test_noisy_floor_and_exact_six_frame_silence_contract(tmp_path):
    wav = _energy_wav(
        tmp_path / "noisy-floor.wav", floor=0.0005,
        active=((0.08, 0.235, 0.01), (0.260, 0.265, 0.01),
                (0.295, 0.50, 0.01), (0.60, 0.72, 0.01)))
    profile = geometry._load_local_rms_profile(wav)
    assert profile is not None
    assert profile["frame_s"] == pytest.approx(0.005)
    assert profile["noise_floor"] == pytest.approx(0.0005, abs=1e-7)
    assert profile["threshold"] == pytest.approx(0.0015, abs=1e-7)
    english = _canonical_timed_row(0.10, 0.16)

    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 0.60, "end": 0.72}],
        [], [{"start_s": 0.20, "end_s": 0.50}], 0.8, wav)

    # The 0.235-0.260 run is only five frames.  The first valid six-frame run
    # is exactly 0.265-0.295, so its onset owns the boundary.
    assert english["processed_ctc_span"] == [0.10, 0.265]
    assert english["processed_ctc_boundary_source"] == "energy_end_hard_boundary"


def test_canonical_next_owner_conflict_is_structured_and_fail_closed():
    english = _canonical_timed_row(0.06, 0.12)
    english.update({"stem": "000009", "surface_text": "K"})
    with pytest.raises(CanonicalNextOwnerConflict) as caught:
        resolve_processed_english_spans(
            [english, {"word": "hao3", "start": 0.10, "end": 0.20}],
            [], [], 1.0)
    error = caught.value
    assert error.reason_code == "canonical_next_owner_conflict"
    assert error.context["canonical_span"] == [0.06, 0.12]
    assert error.context["candidate_start"] == 0.10
    assert error.context["next_lexical_start"] == 0.10
    assert error.context["source"] == "ctc_processed_geometry"
    assert error.context["stem_context"] == "000009"


def test_short_processed_span_is_extended_past_representative_raw_anchor():
    english = _canonical_timed_row(4.65, 4.68)
    resolve_processed_english_spans(
        [english, {"word": "hao3", "start": 4.90, "end": 5.0}],  [], [], 5.2)
    assert english["processed_ctc_span"] == [4.65, 4.90]
    assert english["processed_ctc_span"][1] >= english["canonical_span"][1]


def test_processed_span_extends_from_raw_60ms_anchor_to_next_lexical_start():
    english = _canonical_timed_row()
    original = deepcopy(english)
    words = [english, {"word": "hao3", "start": 0.42, "end": 0.55}]

    resolve_processed_english_spans(words, [], [], 1.0)

    assert english["canonical_span"] == original["canonical_span"]
    assert english["canonical_unit"] == original["canonical_unit"]
    assert english["canonical_unit_sha256"] == original["canonical_unit_sha256"]
    assert english["processed_ctc_span"] == [0.06, 0.42]
    assert english["processed_ctc_boundary_source"] == "next_lexical_token_start"
    assert [english["start"], english["end"]] == english["processed_ctc_span"]


@pytest.mark.parametrize(
    ("punct", "pauses", "source"),
    [
        ([{"start_s": 0.20, "end_s": 0.26}], [], "raw_end_punctuation"),
        ([], [{"start_ms": 200, "end_ms": 500}], "raw_end_long_pause"),
    ],
)
def test_processed_span_keeps_punctuation_and_long_pause_as_hard_boundaries(
        punct, pauses, source):
    english = _canonical_timed_row()
    words = [english, {"word": "hao3", "start": 0.70, "end": 0.80}]

    resolve_processed_english_spans(words, punct, pauses, 1.0)

    assert english["processed_ctc_span"] == [0.06, 0.20]
    assert english["processed_ctc_boundary_source"] == source


def test_sentence_final_processed_span_uses_valid_vad_end_and_raw_fallback(
        monkeypatch):
    english = _canonical_timed_row()
    monkeypatch.setattr(geometry, "_vad_speech_end", lambda *_: 0.48)
    resolve_processed_english_spans([english], [], [], 0.9, "demo.wav")
    assert english["processed_ctc_span"] == [0.06, 0.48]
    assert english["processed_ctc_boundary_source"] == "vad_speech_end"

    fallback = _canonical_timed_row()
    monkeypatch.setattr(geometry, "_vad_speech_end", lambda *_: 1.2)
    resolve_processed_english_spans([fallback], [], [], 0.9, "demo.wav")
    assert fallback["processed_ctc_span"] == [0.06, 0.12]
    assert fallback["processed_ctc_boundary_source"] == "raw_end_fallback"


def test_authority_processed_span_missing_or_tampered_is_rejected(tmp_path):
    bundle, _ = _positive_bundle(tmp_path)
    path = bundle / "demo_tokens.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[1]["processed_ctc_span"]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")
    assert any("processed span missing" in error
               for error in validate_ctc_authority_bundle(bundle, "demo", "你K-Pop好"))

    rows[1]["processed_ctc_span"] = [0.3, 0.9]
    rows[1]["start_s"], rows[1]["end_s"] = 0.3, 0.9
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")
    assert validate_ctc_authority_bundle(bundle, "demo", "你K-Pop好")
