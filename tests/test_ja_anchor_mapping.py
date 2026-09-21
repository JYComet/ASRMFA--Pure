from pathlib import Path
import json

import pytest

from scripts.ja_en_anchors import (
    AnchorConflictError,
    align_units_to_lexical_items,
    build_anchor_plan,
    build_language_runs,
    compute_integer_seams,
    run_dual_forced_alignment,
    handle_anchors,
)


def test_exact_monotonic_mapping_uses_character_spans_without_substring_search():
    units = [
        {"unit_id": "u0", "text": "東京", "char_span": [0, 2], "language": "ja"},
        {"unit_id": "u1", "text": "game", "char_span": [2, 6], "language": "en"},
    ]
    items = [
        {"unit": "東京", "start_s": 0.10, "end_s": 0.80},
        {"unit": "game", "start_s": 0.82, "end_s": 1.40},
    ]
    mapped = align_units_to_lexical_items("東京game", units, items, sample_rate=16000)
    assert [row["unit_id"] for row in mapped] == ["u0", "u1"]
    assert mapped[0]["start_sample"] == 1600
    assert mapped[1]["end_sample"] == 22400


def test_character_stream_mismatch_fails_closed_even_if_substring_would_match():
    units = [{"unit_id": "u0", "text": "game", "char_span": [0, 4], "language": "en"}]
    items = [{"unit": "games", "start_s": 0.0, "end_s": 1.0}]
    with pytest.raises(ValueError, match="lexical character stream"):
        align_units_to_lexical_items("game", units, items, sample_rate=16000)


def test_mixed_runs_keep_adjacent_language_ownership_and_dual_seam_conflict():
    units = [
        {"unit_id": "a", "char_span": [0, 1], "language": "ja", "start_sample": 1000, "end_sample": 8000},
        {"unit_id": "b", "char_span": [1, 5], "language": "en", "start_sample": 8000, "end_sample": 20000},
        {"unit_id": "c", "char_span": [5, 6], "language": "ja", "start_sample": 20000, "end_sample": 30000},
    ]
    runs = build_language_runs(units, padding_samples=1600, total_samples=32000)
    assert [(r["language"], r["unit_ids"]) for r in runs] == [
        ("ja", ["a"]), ("en", ["b"]), ("ja", ["c"])
    ]
    assert all(r["ownership_start_sample"] <= r["ownership_end_sample"] for r in runs)
    with pytest.raises(AnchorConflictError):
        compute_integer_seams(
            [{"left_unit_id": "a", "right_unit_id": "b", "left_end_sample": 1000, "right_start_sample": 2000}],
            [{"left_unit_id": "a", "right_unit_id": "b", "left_end_sample": 1000, "right_start_sample": 5000}],
            max_disagreement_ms=80,
            sample_rate=16000,
        )


def test_dual_runner_calls_both_languages_for_same_audio_and_text(tmp_path: Path):
    calls = []

    class Fake:
        def align(self, audio, text, language):
            calls.append((audio, text, language))
            return [{"unit": "東京game", "start_s": 0.0, "end_s": 1.0}]

    result = run_dual_forced_alignment(Fake(), tmp_path / "x.wav", "東京game", sample_rate=16000)
    assert [call[2] for call in calls] == ["Japanese", "English"]
    assert calls[0][:2] == calls[1][:2] == (str(tmp_path / "x.wav"), "東京game")
    assert set(result) == {"Japanese", "English"}


def test_pure_route_calls_only_its_native_language(tmp_path: Path):
    calls = []
    class Fake:
        def align(self, audio, text, language):
            calls.append(language)
            return [{"unit": text, "start_s": 0.0, "end_s": 1.0}]
    from scripts.ja_en_anchors import run_forced_alignment
    run_forced_alignment(Fake(), tmp_path / "x.wav", "game", route="en", sample_rate=16000)
    assert calls == ["English"]


def test_anchor_plan_routes_seams_and_runs_from_dual_passes(tmp_path: Path):
    class Fake:
        def align(self, audio, text, language):
            if language == "Japanese":
                return [{"unit": "東京", "start_s": 0.0, "end_s": 0.5}, {"unit": "game", "start_s": 0.5, "end_s": 1.0}]
            return [{"unit": "東京", "start_s": 0.0, "end_s": 0.55}, {"unit": "game", "start_s": 0.55, "end_s": 1.0}]
    plan = build_anchor_plan(
        uid="u1", aligner=Fake(), audio=tmp_path / "x.wav", spoken_text="東京game",
        units=[
            {"unit_id": "a", "text": "東京", "char_span": [0, 2], "language": "ja"},
            {"unit_id": "b", "text": "game", "char_span": [2, 6], "language": "en"},
        ], route="mixed", sample_rate=16000, total_samples=16000,
    )
    assert plan["schema"] == "ja-en-alignment-plan-v2"
    assert len(plan["runs"]) == 2 and len(plan["seams"]) == 1
    assert plan["runs"][0]["ownership_end_sample"] == plan["seams"][0]["ownership_seam_sample"]
    assert plan["runs"][1]["ownership_start_sample"] == plan["seams"][0]["ownership_seam_sample"]
    assert plan["units"][0]["end_sample"] == 8000
    assert plan["units"][1]["start_sample"] == 8800


def test_anchor_stage_consumes_prepared_input_and_writes_plan(tmp_path: Path):
    class Fake:
        def align(self, audio, text, language):
            return [{"unit": "東京", "start_s": 0.0, "end_s": 0.5}, {"unit": "game", "start_s": 0.5, "end_s": 1.0}]
    stage = tmp_path / "stages" / "anchors"
    result = handle_anchors({"anchors": {
        "uid": "u1", "audio": tmp_path / "x.wav", "spoken_text": "東京game",
        "units": [
            {"unit_id": "a", "text": "東京", "char_span": [0, 2], "language": "ja"},
            {"unit_id": "b", "text": "game", "char_span": [2, 6], "language": "en"},
        ], "route": "mixed", "sample_rate": 16000, "total_samples": 16000, "qwen_aligner": Fake(),
    }}, stage)
    assert result.status == "COMPLETE"
    assert (stage / "anchor_plan.json").is_file()


def test_frozen_mixed_worker_output_selects_target_pass_and_allows_other_pass_crossing():
    fixture = Path(__file__).resolve().parent / "fixtures" / "frozen_mixed_qwen_worker_output.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    class Static:
        def align(self, audio, text, language):
            return payload["passes"][language]

    spoken = "桜が咲きましたpeople"
    units = [
        {"unit_id": "tok_桜", "text": "桜", "char_span": [0, 1], "language": "ja"},
        {"unit_id": "tok_が", "text": "が", "char_span": [1, 2], "language": "ja"},
        {"unit_id": "tok_咲き", "text": "咲き", "char_span": [2, 4], "language": "ja"},
        {"unit_id": "tok_まし", "text": "まし", "char_span": [4, 6], "language": "ja"},
        {"unit_id": "tok_た", "text": "た", "char_span": [6, 7], "language": "ja"},
        {"unit_id": "tok_people", "text": "people", "char_span": [7, 13], "language": "en"},
    ]
    plan = build_anchor_plan(uid="mixed", aligner=Static(), audio=Path("fixture.wav"),
                             spoken_text=spoken, units=units, route="mixed",
                             sample_rate=16000, total_samples=48000)
    assert [unit["language"] for unit in plan["units"]] == ["ja"] * 5 + ["en"]
    assert plan["units"][-1]["start_sample"] == 33280
    assert plan["passes"]["English"][2]["crosses_non_target_unit"] is True
    assert plan["passes"]["English"][5]["crosses_non_target_unit"] is False
    assert plan["boundary_evidence"][0]["English"][0]["right_item"] == "people"
    assert plan["seams"][0]["ownership_seam_sample"] > 0


def test_selected_target_unit_crossing_is_rejected():
    units = [
        {"unit_id": "a", "text": "きま", "char_span": [0, 2], "language": "en"},
        {"unit_id": "b", "text": "した", "char_span": [2, 4], "language": "ja"},
    ]
    with pytest.raises(ValueError, match="crosses unit char span"):
        align_units_to_lexical_items("きました", units,
                                     [{"unit": "きました", "start_s": 0.0, "end_s": 1.0}],
                                     sample_rate=16000, target_language="en")


def test_mixed_stream_incomplete_and_missing_seam_edge_fail_closed():
    spoken = "桜people"
    units = [
        {"unit_id": "ja", "text": "桜", "char_span": [0, 1], "language": "ja"},
        {"unit_id": "en", "text": "people", "char_span": [1, 7], "language": "en"},
    ]
    with pytest.raises(ValueError, match="stream is incomplete"):
        align_units_to_lexical_items(spoken, units, [{"unit": "桜", "start_s": 0, "end_s": 1}],
                                     sample_rate=16000, target_language="ja")

    class MissingEdge:
        def align(self, audio, text, language):
            if language == "Japanese":
                return [{"unit": "桜", "start_s": 0, "end_s": .5}, {"unit": "people", "start_s": .5, "end_s": 1}]
            return [{"unit": "桜people", "start_s": 0, "end_s": 1}]

    with pytest.raises((AnchorConflictError, ValueError)):
        build_anchor_plan(uid="missing-edge", aligner=MissingEdge(), audio=Path("fixture.wav"),
                          spoken_text=spoken, units=units, route="mixed",
                          sample_rate=16000, total_samples=16000)
