"""No-provider contract tests for Qwen/NVASR candidate fusion."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import ctc_prealign
from scripts import qwen3asr_fusion as fusion


def _coordinates(start_frame: int, end_frame: int) -> dict:
    return {
        "raw_start_frame": start_frame,
        "raw_end_frame": end_frame,
        "raw_start_s": start_frame * fusion.FRAME_SECONDS,
        "raw_end_s": end_frame * fusion.FRAME_SECONDS,
        "speech_start_frame": start_frame - fusion.QUERY_FRAMES,
        "speech_end_frame": end_frame - fusion.QUERY_FRAMES,
        "speech_start_s": (start_frame - fusion.QUERY_FRAMES) * fusion.FRAME_SECONDS,
        "speech_end_s": (end_frame - fusion.QUERY_FRAMES) * fusion.FRAME_SECONDS,
    }


def _lexical(ordinal: int, surface: str, start: int, token_id: int = 1) -> dict:
    return {
        "lexical_ordinal": ordinal,
        "surface": surface,
        "kind": "lexical",
        "token_id": token_id,
        "token_ids": [token_id],
        **_coordinates(start, start + 1),
    }


def _candidate(
    *,
    occurrence: int = 0,
    left: int | None = 0,
    right: int | None = 1,
    label: str = "[Breathing]",
    start: int = 5,
    **extra,
) -> dict:
    candidate = {
        "candidate_id": fusion.candidate_id_for_occurrence(occurrence),
        "occurrence": occurrence,
        "label": label,
        "surface": label,
        "source": "ctc",
        "kind": "nvv",
        "token_id": 25025,
        "token_ids": [25025],
        "left_lexical_ordinal": left,
        "right_lexical_ordinal": right,
        "diagnostic": "fixture",
        **_coordinates(start, start + 1),
    }
    candidate.update(extra)
    return candidate


def _timeline(
    lexical_occurrences=None,
    candidates=None,
):
    if lexical_occurrences is None:
        lexical_occurrences = (_lexical(0, "你", 4), _lexical(1, "好", 6, 2))
    if candidates is None:
        candidates = (_candidate(),)
    return {
        "schema": "nvasr-candidate-timeline-v1",
        "stem": "stem",
        "query_frames": 4,
        "frame_ms": 60,
        "duration_s": 0.18,
        "diagnostic": "fixture",
        "lexical_occurrences": list(lexical_occurrences),
        "candidates": list(candidates),
    }


def _forced(units=("你", "好")):
    return [
        {"unit": units[0], "start_s": 0.00, "end_s": 0.06},
        {"unit": units[1], "start_s": 0.12, "end_s": 0.18},
    ]


def test_module_import_is_provider_free():
    tree = ast.parse(
        Path(fusion.__file__).read_text(encoding="utf-8"),
        filename=str(fusion.__file__),
    )
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imported.intersection({"qwen_asr", "FunASR", "torch"})


def test_timeline_validator_requires_ordered_lexical_occurrences_and_adjacency():
    timeline = _timeline()
    assert fusion.validate_candidate_timeline(timeline)["lexical_occurrences"][1][
        "lexical_ordinal"
    ] == 1
    timeline["lexical_occurrences"][1]["lexical_ordinal"] = 3
    with pytest.raises(fusion.FusionError, match="contiguous and ordered"):
        fusion.validate_candidate_timeline(timeline)


def test_acceptance_derives_boundary_and_preserves_qwen_order():
    result = fusion.fuse_qwen_nvasr_candidates(["你", "好"], _forced(), _timeline())
    assert result["status"] == "COMPLETE"
    assert result["lexical_timing_source"] == "qwen3_forced_aligner"
    assert result["lexical_authority"] == "qwen"
    assert result["fused_lexical_units"] == ["你", "[Breathing]", "好"]
    assert result["accepted"][0]["qwen_insertion_boundary"] == 1
    assert result["rejected"] == []
    assert result["candidate_conservation"]["exactly_once"]


def test_case_insensitive_latin_numeric_matching_is_monotonic():
    timeline = _timeline(
        lexical_occurrences=(
            _lexical(0, "ABC12", 4), _lexical(1, "next", 6, 2),
        ),
    )
    result = fusion.fuse_qwen_nvasr_candidates(
        ["abc12", "NEXT"], _forced(("abc12", "NEXT")), timeline
    )
    assert result["status"] == "COMPLETE"
    assert result["lexical_alignment"] == [[0, 0], [1, 1]]


def test_mismatched_qwen_and_forced_aligner_units_fail_closed():
    with pytest.raises(fusion.FusionError, match="exactly match"):
        fusion.fuse_qwen_nvasr_candidates(["你", "错"], _forced(), _timeline())


def test_lexical_substitution_does_not_replace_qwen_authority():
    timeline = _timeline(
        lexical_occurrences=(
            _lexical(0, "你", 4), _lexical(1, "坏", 6, 2),
        ),
    )
    result = fusion.fuse_qwen_nvasr_candidates(
        ["你", "好"], _forced(), timeline
    )
    assert result["status"] == "FAILED"
    assert result["fused_lexical_units"] == ["你", "好"]
    assert result["rejected"][0]["reason"] == "unmapped_lexical_neighbor"


def test_one_sided_non_edge_is_rejected():
    result = fusion.fuse_qwen_nvasr_candidates(
        ["你", "好"], _forced(),
        _timeline(candidates=(_candidate(left=None, right=1),)),
    )
    assert result["status"] == "FAILED"
    assert result["rejected"][0]["reason"] == "edge_owner_not_qwen_first"
    assert result["candidate_conservation"]["exactly_once"]


def test_laria_before_overlay_after_shape_and_provenance():
    lexical = (
        _lexical(0, "甲", 4), _lexical(1, "丁", 11, 2),
    )
    qwen = ["甲", "乙", "丙", "丁"]
    forced = [
        {"unit": "甲", "start_s": 0.00, "end_s": 0.06},
        {"unit": "乙", "start_s": 0.12, "end_s": 0.18},
        {"unit": "丙", "start_s": 0.24, "end_s": 0.30},
        {"unit": "丁", "start_s": 0.36, "end_s": 0.42},
    ]
    candidates = (
        _candidate(start=5),       # .06..12, before 乙
        _candidate(occurrence=1, start=6),  # .12..18, overlay 乙
        _candidate(occurrence=2, start=9),  # .30..36, after 丙
    )
    timeline = _timeline(lexical_occurrences=lexical, candidates=candidates)
    timeline["duration_s"] = 0.48
    result = fusion.fuse_qwen_nvasr_candidates(qwen, forced, timeline)
    assert result["status"] == "COMPLETE"
    assert result["fused_lexical_units"] == [
        "甲", "[Breathing]", "[Breathing]", "乙", "丙", "[Breathing]", "丁",
    ]
    assert [row["temporal_relation"] for row in result["accepted"]] == [
        "before", "overlap", "after",
    ]
    assert [row["qwen_insertion_boundary"] for row in result["accepted"]] == [1, 1, 3]
    overlay = result["accepted"][1]
    assert overlay["qwen_overlap_ordinal"] == 1
    assert overlay["qwen_projected_start_s"] == pytest.approx(0.06)
    assert overlay["qwen_projected_end_s"] == pytest.approx(0.12)
    assert result["accepted"][0]["source"] == "ctc"
    assert result["accepted"][0]["kind"] == "nvv"
    assert result["accepted"][0]["token_id"] == 25025
    assert result["accepted"][0]["token_ids"] == [25025]
    assert all(row["timing_source"] == "nvasr_ctc_free_decode"
               for row in result["accepted"])
    assert result["lexical_timing_source"] == "qwen3_forced_aligner"


def test_terminal_edges_are_accepted_only_on_correct_exterior_side():
    lexical = (_lexical(0, "甲", 6),)
    forced = [{"unit": "甲", "start_s": 0.12, "end_s": 0.18}]
    leading = _timeline(
        lexical_occurrences=lexical,
        candidates=(_candidate(left=None, right=0, start=4),),
    )
    leading["duration_s"] = 0.24
    result = fusion.fuse_qwen_nvasr_candidates(["甲"], forced, leading)
    assert result["status"] == "COMPLETE"
    assert result["accepted"][0]["temporal_relation"] == "before"
    assert result["accepted"][0]["qwen_insertion_boundary"] == 0

    trailing = _timeline(
        lexical_occurrences=lexical,
        candidates=(_candidate(left=0, right=None, start=7),),
    )
    trailing["duration_s"] = 0.24
    result = fusion.fuse_qwen_nvasr_candidates(["甲"], forced, trailing)
    assert result["status"] == "COMPLETE"
    assert result["accepted"][0]["temporal_relation"] == "after"
    assert result["accepted"][0]["qwen_insertion_boundary"] == 1


def test_multi_overlap_and_straddle_are_rejected_with_conservation():
    lexical = (_lexical(0, "甲", 4), _lexical(1, "丁", 11, 2))
    forced = [
        {"unit": "甲", "start_s": 0.00, "end_s": 0.06},
        {"unit": "乙", "start_s": 0.12, "end_s": 0.18},
        {"unit": "丙", "start_s": 0.24, "end_s": 0.30},
        {"unit": "丁", "start_s": 0.36, "end_s": 0.42},
    ]
    multi = _timeline(
        lexical_occurrences=lexical,
        candidates=(_candidate(start=6, **{"raw_end_frame": 9,
                                           "raw_end_s": 0.54,
                                           "speech_end_frame": 5,
                                           "speech_end_s": 0.30}),),
    )
    multi["duration_s"] = 0.48
    result = fusion.fuse_qwen_nvasr_candidates(["甲", "乙", "丙", "丁"], forced, multi)
    assert result["status"] == "FAILED"
    assert result["rejected"][0]["reason"] == "straddling_qwen_ambiguity"
    assert result["candidate_conservation"]["exactly_once"]


def test_three_unmatched_qwen_units_are_rejected_as_multi_overlap():
    lexical = (_lexical(0, "甲", 4), _lexical(1, "丁", 13, 2))
    forced = [
        {"unit": "甲", "start_s": 0.00, "end_s": 0.06},
        {"unit": "乙", "start_s": 0.12, "end_s": 0.18},
        {"unit": "丙", "start_s": 0.24, "end_s": 0.30},
        {"unit": "戊", "start_s": 0.36, "end_s": 0.42},
        {"unit": "丁", "start_s": 0.48, "end_s": 0.54},
    ]
    candidate = _candidate(start=6, **{
        "raw_end_frame": 11, "raw_end_s": 0.66,
        "speech_end_frame": 7, "speech_end_s": 0.42,
    })
    timeline = _timeline(lexical_occurrences=lexical, candidates=(candidate,))
    timeline["duration_s"] = 0.60
    result = fusion.fuse_qwen_nvasr_candidates(
        ["甲", "乙", "丙", "戊", "丁"], forced, timeline
    )
    assert result["status"] == "FAILED"
    assert result["rejected"][0]["reason"] == "multi_qwen_overlap"
    assert result["candidate_conservation"]["exactly_once"]


def test_blank_run_provenance_uses_heuristic_timing_source():
    timeline = _timeline()
    timeline["candidates"][0]["source"] = "blank_run"
    timeline["candidates"][0]["kind"] = "punctuation"
    timeline["candidates"][0]["label"] = "…"
    timeline["candidates"][0]["surface"] = "…"
    result = fusion.fuse_qwen_nvasr_candidates(["你", "好"], _forced(), timeline)
    assert result["status"] == "COMPLETE"
    assert result["accepted"][0]["timing_source"] == "nvasr_blank_pause_heuristic"
    assert "qwen3_forced_aligner" not in result["accepted"][0]["timing_source"]


def test_repeated_lexical_mapping_is_ambiguous_even_when_time_favors_one_gap():
    timeline = _timeline(
        lexical_occurrences=(
            _lexical(0, "你", 4), _lexical(1, "好", 6, 2),
        ),
    )
    result = fusion.fuse_qwen_nvasr_candidates(
        ["你", "好", "好"],
        [
            {"unit": "你", "start_s": 0.00, "end_s": 0.03},
            {"unit": "好", "start_s": 0.06, "end_s": 0.09},
            {"unit": "好", "start_s": 0.12, "end_s": 0.15},
        ],
        timeline,
    )
    assert result["status"] == "FAILED"
    assert result["optimal_mapping_count"] == 2
    assert result["rejected"][0]["reason"] == "ambiguous_lexical_mapping"
    assert result["candidate_conservation"]["exactly_once"]


def test_unmatched_qwen_region_gets_exact_inter_anchor_boundary():
    timeline = _timeline(candidates=(_candidate(start=7),))
    timeline["duration_s"] = 0.48
    result = fusion.fuse_qwen_nvasr_candidates(
        ["你", "插入", "更多", "好"],
        [
            {"unit": "你", "start_s": 0.00, "end_s": 0.06},
            {"unit": "插入", "start_s": 0.12, "end_s": 0.18},
            {"unit": "更多", "start_s": 0.24, "end_s": 0.30},
            {"unit": "好", "start_s": 0.36, "end_s": 0.42},
        ],
        timeline,
    )
    assert result["status"] == "COMPLETE"
    assert result["accepted"][0]["temporal_relation"] == "inter_anchor"
    assert result["accepted"][0]["qwen_insertion_boundary"] == 2


def test_candidate_outside_derived_qwen_gap_is_rejected():
    candidate = _candidate(start=4)
    result = fusion.fuse_qwen_nvasr_candidates(
        ["你", "好"], _forced(), _timeline(candidates=(candidate,))
    )
    assert result["status"] == "FAILED"
    assert result["rejected"][0]["reason"] == "candidate_outside_qwen_anchor_envelope"


def test_end_to_end_ctc_raw_frames_feed_repeated_ambiguity_to_fusion():
    timeline = ctc_prealign.extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 1, 25025, 2],
        "raw-frame fixture",
        token_surfaces={1: "你", 2: "好", 25025: "[Breathing]"},
        stem="stem",
    )
    assert [row["surface"] for row in timeline["lexical_occurrences"]] == ["你", "好"]
    assert timeline["candidates"][0]["left_lexical_ordinal"] == 0
    assert timeline["candidates"][0]["right_lexical_ordinal"] == 1
    result = fusion.fuse_qwen_nvasr_candidates(
        ["你", "好", "好"],
        [
            {"unit": "你", "start_s": 0.00, "end_s": 0.03},
            {"unit": "好", "start_s": 0.06, "end_s": 0.09},
            {"unit": "好", "start_s": 0.12, "end_s": 0.15},
        ],
        timeline,
    )
    assert result["status"] == "FAILED"
    assert result["rejected"][0]["reason"] == "ambiguous_lexical_mapping"


def test_tampered_axis_coordinates_are_rejected_before_fusion():
    timeline = _timeline()
    timeline["lexical_occurrences"][0]["speech_start_s"] = 0.01
    with pytest.raises(fusion.FusionError, match="do not match frame"):
        fusion.fuse_qwen_nvasr_candidates(["你", "好"], _forced(), timeline)
