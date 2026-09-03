from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.ctc_prealign import (
    _ctc_token_sidecar_row,
    _deduplicate_adjacent_nvv_rows,
    _nvasr_anchor_from_frames,
    _validate_emitted_nvasr_provenance,
    attach_nvasr_candidate_provenance,
    extract_nvasr_candidate_timeline,
)


def test_schema_v3_extract_attach_and_raw_locator_serialization_contract():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 32], "你[Breathing]好",
        token_surfaces={31: "你", 11: "[Breathing]", 32: "好"},
        stem="demo")
    candidate = timeline["candidates"][0]

    assert timeline["nvasr_candidate_schema_version"] == 3
    assert candidate["nvasr_candidate_schema_version"] == 3
    assert candidate["query_frames"] == 4
    assert candidate["frame_ms"] == 60
    assert candidate["ctc_spike_anchor"] == _nvasr_anchor_from_frames(5, 6)
    assert candidate["ctc_spike_anchor"]["schema"] == "ctc_spike_anchor_v2"
    assert candidate["ctc_spike_anchor"]["end"] - \
        candidate["ctc_spike_anchor"]["start"] == pytest.approx(0.06)

    words = [
        {"word": "ni3", "start": 0.0, "end": 0.06},
        {"word": "BREATHING", "start": 0.06, "end": 0.12},
        {"word": "hao3", "start": 0.12, "end": 0.18},
    ]
    assert attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True) == []
    assert _validate_emitted_nvasr_provenance(words) == []

    serialized = [
        _ctc_token_sidecar_row(
            row, row["start"], row["end"], stem="demo",
            row_ordinal=ordinal)
        for ordinal, row in enumerate(words)
    ]
    assert [row["ctc_raw_token_row"] for row in serialized] == [
        {"schema": "ctc_raw_token_row_v1", "stem": "demo",
         "sidecar": "demo_tokens.jsonl", "row_ordinal": ordinal}
        for ordinal in range(3)
    ]
    assert serialized[1]["ctc_spike_anchor"] == \
        candidate["ctc_spike_anchor"]


def test_timeline_is_provider_free_and_preserves_duplicate_occurrences():
    surfaces = {
        101: "你",
        102: "好",
        11: "[Breathing]",
        12: "，",
        13: "[Breathing]",
        14: "#",  # not in the CTC punctuation whitelist after decoding
    }
    frame_ids = [0, 0, 0, 0, 101, 101, 11, 12, 13, 102, 102, 101, 14]

    first = extract_nvasr_candidate_timeline(
        frame_ids,
        "你[Breathing]，[Breathing]好你#",
        token_surfaces=surfaces,
        stem="fixture",
    )
    second = extract_nvasr_candidate_timeline(
        frame_ids,
        "你[Breathing]，[Breathing]好你#",
        token_surfaces=surfaces,
        stem="fixture",
    )

    assert first == second
    assert first["schema"] == "nvasr-candidate-timeline-v1"
    assert first["duration_s"] == 0.54
    assert first["query_frames"] == 4
    assert first["frame_ms"] == 60
    assert first["stem"] == "fixture"
    assert first["diagnostic"] == first["diagnostic_text"]
    assert first["diagnostic_text"] == "你[Breathing]，[Breathing]好你#"
    assert [row["surface"] for row in first["lexical_occurrences"]] == [
        "你", "好", "你",
    ]
    assert [row["lexical_ordinal"] for row in first["lexical_occurrences"]] == [
        0, 1, 2,
    ]
    assert [row["surface_occurrence"] for row in first["lexical_occurrences"]] == [
        0, 0, 1,
    ]
    assert first["lexical_occurrences"][0]["raw_start_frame"] == 4
    assert first["lexical_occurrences"][0]["speech_start_frame"] == 0
    assert first["lexical_occurrences"][2]["raw_start_frame"] == 11
    assert first["lexical_occurrences"][2]["speech_start_s"] == 0.42
    assert [candidate["surface"] for candidate in first["candidates"]] == [
        "[Breathing]", "，", "[Breathing]",
    ]
    assert [candidate["candidate_id"] for candidate in first["candidates"]] == [
        "nvasr-candidate-0000",
        "nvasr-candidate-0001",
        "nvasr-candidate-0002",
    ]
    assert first["candidates"][0]["label"] == "[Breathing]"
    assert first["candidates"][0]["source"] == "ctc"
    assert first["candidates"][0]["kind"] == "nvv"
    assert first["candidates"][0]["token_id"] == 11
    assert first["candidates"][0]["token_ids"] == [11]
    assert first["candidates"][0]["raw_start_frame"] == 6
    assert first["candidates"][0]["raw_end_frame"] == 7
    assert first["candidates"][0]["speech_start_frame"] == 2
    assert first["candidates"][0]["speech_end_frame"] == 3
    assert first["candidates"][0]["left_lexical_ordinal"] == 0
    assert first["candidates"][0]["right_lexical_ordinal"] == 1
    assert first["candidates"][1]["left_lexical_ordinal"] == 0
    assert first["candidates"][1]["right_lexical_ordinal"] == 1
    assert first["candidates"][1]["source"] == "ctc"
    assert first["candidates"][1]["kind"] == "punctuation"
    assert first["candidates"][1]["token_id"] == 12
    assert first["candidates"][1]["token_ids"] == [12]
    assert first["candidates"][2]["raw_start_frame"] == 8
    assert first["candidates"][2]["speech_start_s"] == 0.24
    assert not any("gap_index" in row for row in first["candidates"])
    assert not any("left_unit" in row for row in first["candidates"])


def test_long_blank_run_emits_one_provenance_preserving_ellipsis():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, 0, 21, 32],
        "你…[Cough]好",
        token_surfaces={31: "你", 21: "[Cough]", 32: "好"},
        stem="fixture",
        pause_threshold=8,
    )

    assert [candidate["surface"] for candidate in timeline["candidates"]] == [
        "…", "[Cough]",
    ]
    assert [row["surface"] for row in timeline["lexical_occurrences"]] == [
        "你", "好",
    ]
    ellipsis = timeline["candidates"][0]
    assert ellipsis["source"] == "blank_run"
    assert ellipsis["kind"] == "punctuation"
    assert ellipsis["token_id"] == 9724
    assert ellipsis["token_ids"] == [9724]
    assert timeline["duration_s"] == 0.66
    assert ellipsis["raw_start_frame"] == 9
    assert ellipsis["raw_end_frame"] == 10
    assert ellipsis["speech_start_frame"] == 5
    assert ellipsis["speech_end_frame"] == 6
    assert ellipsis["raw_start_s"] == 0.54
    assert ellipsis["speech_start_s"] == 0.3
    assert ellipsis["left_lexical_ordinal"] == 0
    assert ellipsis["right_lexical_ordinal"] == 1
    assert timeline["candidates"][1]["left_lexical_ordinal"] == 0
    assert timeline["candidates"][1]["right_lexical_ordinal"] == 1


def test_locked_frame_axis_is_applied_without_mutating_raw_coordinates():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 31, 0],
        "[Laughter]",
        token_surfaces={31: "[Laughter]"},
        stem="fixture",
    )

    candidate = timeline["candidates"][0]
    assert candidate["raw_start_frame"] == 4
    assert candidate["raw_end_frame"] == 6
    assert candidate["raw_start_s"] == 0.24
    assert candidate["raw_end_s"] == 0.36
    assert candidate["speech_start_frame"] == 0
    assert candidate["speech_end_frame"] == 2
    assert candidate["speech_start_s"] == 0.0
    assert candidate["speech_end_s"] == 0.12


def test_candidate_neighbors_are_null_only_at_utterance_edges():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 11, 0, 101, 0, 12],
        "[Breathing]你，",
        token_surfaces={11: "[Breathing]", 101: "你", 12: "，"},
        stem="fixture",
    )

    assert [row["surface"] for row in timeline["lexical_occurrences"]] == ["你"]
    assert timeline["candidates"][0]["left_lexical_ordinal"] is None
    assert timeline["candidates"][0]["right_lexical_ordinal"] == 0
    assert timeline["candidates"][1]["left_lexical_ordinal"] == 0
    assert timeline["candidates"][1]["right_lexical_ordinal"] is None


def test_nvv_row_gets_unique_raw_speech_forced_and_mapping_provenance():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 32],
        "你好[Breathing]",
        token_surfaces={31: "你", 32: "好", 11: "[Breathing]"},
    )
    words = [
        {"word": "ni3", "start": 0.00, "end": 0.06},
        {"word": "hao3", "start": 0.12, "end": 0.18},
        {"word": "BREATHING", "start": 0.24, "end": 0.30},
    ]
    # Put the NVV between the lexical rows, matching its raw neighbor key.
    words = [words[0], words[2], words[1]]
    assert attach_nvasr_candidate_provenance(words, [], timeline) == []
    row = words[1]
    assert row["candidate_id"] == "nvasr-candidate-0000"
    assert row["raw_span"] == [0.3, 0.36]
    assert row["speech_span"] == [0.06, 0.12]
    assert row["forced_span"] == [0.24, 0.30]
    assert row["mapping_basis"] == "raw_ctc_label_neighbors_forced_overlap-v2"
    assert row["mapping_outcome"] == "unique"


def test_adjacent_laughter_is_deduplicated_before_unique_candidate_binding():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 0, 11, 0, 0, 11],
        "你[Laughter][Laughter]",
        token_surfaces={31: "你", 11: "[Laughter]"},
    )
    # Mirrors LAria_00137: the merged forced envelope overlaps the first raw
    # candidate for a full frame and the second for only half a frame.
    words = [
        {"word": "ni3", "start": 0.00, "end": 0.06,
         "source_ctc_ordinal": 0},
        {"word": "LAUGHTER", "start": 0.12, "end": 0.18,
         "source_ctc_ordinal": 1},
        {"word": "LAUGHTER", "start": 0.27, "end": 0.33,
         "source_ctc_ordinal": 2},
    ]

    deduplicated = _deduplicate_adjacent_nvv_rows(words)

    assert len(deduplicated) == 2
    laughter = deduplicated[1]
    assert laughter["start"] == 0.12
    assert laughter["end"] == 0.33
    assert laughter["source_ctc_ordinals"] == [1, 2]
    assert laughter["nvv_deduplication"] == {
        "schema": "nvv-adjacent-deduplication-v1",
        "label": "LAUGHTER",
        "occurrence_count": 2,
        "forced_occurrence_spans": [[0.12, 0.18], [0.27, 0.33]],
    }
    assert attach_nvasr_candidate_provenance(
        deduplicated, [], timeline) == []
    assert laughter["candidate_id"] == "nvasr-candidate-0000"
    assert laughter["forced_span"] == [0.12, 0.33]
    assert laughter["mapping_selection"] == "unique_max_forced_speech_overlap"
    assert _validate_emitted_nvasr_provenance(deduplicated) == []


def test_final_nvv_without_candidate_provenance_is_producer_error():
    rows = [{"word": "LAUGHTER", "start": 0.12, "end": 0.33}]

    errors = _validate_emitted_nvasr_provenance(rows)

    assert errors
    assert "missing provenance fields" in errors[0]


def test_repeated_identical_nvv_neighbor_key_is_rejected():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 0, 11, 32],
        "你[Breathing][Breathing]好",
        token_surfaces={31: "你", 32: "好", 11: "[Breathing]"},
    )
    words = [
        {"word": "ni3", "start": 0.00, "end": 0.06},
        {"word": "BREATHING", "start": 0.12, "end": 0.18},
        {"word": "BREATHING", "start": 0.18, "end": 0.24},
        {"word": "hao3", "start": 0.30, "end": 0.36},
    ]
    errors = attach_nvasr_candidate_provenance(words, [], timeline)
    assert errors
    assert "ambiguous" in errors[0]
    assert not any("candidate_id" in row for row in words if row["word"] == "BREATHING")


def test_unemitted_raw_nvv_candidate_remains_diagnostic_without_rejecting_row():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 32, 11, 33],
        "你[Breathing]好吗",
        token_surfaces={31: "你", 32: "好", 33: "吗", 11: "[Breathing]"},
    )
    words = [
        {"word": "ni3", "start": 0.00, "end": 0.06},
        {"word": "BREATHING", "start": 0.06, "end": 0.12},
        {"word": "hao3", "start": 0.12, "end": 0.18},
        {"word": "ma5", "start": 0.24, "end": 0.30},
    ]

    assert attach_nvasr_candidate_provenance(words, [], timeline) == []
    assert words[1]["candidate_id"] == "nvasr-candidate-0000"
    assert len(timeline["candidates"]) == 2


def test_same_neighbor_candidates_use_unique_forced_speech_overlap():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 0, 11, 11, 32],
        "你[Breathing]好",
        token_surfaces={31: "你", 32: "好", 11: "[Breathing]"},
    )
    words = [
        {"word": "ni3", "start": 0.00, "end": 0.06},
        {"word": "BREATHING", "start": 0.24, "end": 0.30},
        {"word": "hao3", "start": 0.30, "end": 0.36},
    ]

    assert attach_nvasr_candidate_provenance(words, [], timeline) == []
    assert words[1]["candidate_id"] == "nvasr-candidate-0001"
    assert words[1]["mapping_selection"] == "unique_max_forced_speech_overlap"
    assert words[1]["mapping_forced_speech_overlap_s"] == 0.06


def test_same_neighbor_candidates_fall_back_to_unique_forced_raw_overlap():
    timeline = {"candidates": [
        {"candidate_id": "candidate-0", "kind": "nvv",
         "surface": "[Breathing]", "token_id": 11, "token_ids": [11],
         "left_lexical_ordinal": 0, "right_lexical_ordinal": 1,
         "raw_start_s": 0.96, "raw_end_s": 1.02,
         "speech_start_s": 0.72, "speech_end_s": 0.78,
         "raw_start_frame": 16, "raw_end_frame": 17,
         "speech_start_frame": 12, "speech_end_frame": 13},
        {"candidate_id": "candidate-2", "kind": "nvv",
         "surface": "[Breathing]", "token_id": 11, "token_ids": [11],
         "left_lexical_ordinal": 0, "right_lexical_ordinal": 1,
         "raw_start_s": 1.26, "raw_end_s": 1.38,
         "speech_start_s": 1.02, "speech_end_s": 1.14,
         "raw_start_frame": 21, "raw_end_frame": 23,
         "speech_start_frame": 17, "speech_end_frame": 19},
    ]}
    words = [
        {"word": "ni3", "start": 0.0, "end": 0.5},
        {"word": "BREATHING", "start": 0.93, "end": 0.99},
        {"word": "hao3", "start": 1.4, "end": 1.5},
    ]

    assert attach_nvasr_candidate_provenance(words, [], timeline) == []
    assert words[1]["candidate_id"] == "candidate-0"
    assert words[1]["mapping_selection"] == "unique_max_forced_raw_overlap"
    assert "mapping_forced_raw_overlap_s" not in words[1]
    assert "mapping_forced_speech_overlap_s" not in words[1]


def _ambiguous_raw_timeline(first_raw, second_raw):
    def candidate(candidate_id, raw_span):
        return {
            "candidate_id": candidate_id, "kind": "nvv",
            "surface": "[Breathing]", "token_id": 11, "token_ids": [11],
            "left_lexical_ordinal": 0, "right_lexical_ordinal": 1,
            "raw_start_s": raw_span[0], "raw_end_s": raw_span[1],
            "speech_start_s": 2.0, "speech_end_s": 2.1,
            "raw_start_frame": 16, "raw_end_frame": 17,
            "speech_start_frame": 32, "speech_end_frame": 33,
        }
    return {"candidates": [
        candidate("candidate-0", first_raw),
        candidate("candidate-1", second_raw),
    ]}


def _ambiguous_raw_words():
    return [
        {"word": "ni3", "start": 0.0, "end": 0.5},
        {"word": "BREATHING", "start": 1.0, "end": 1.1},
        {"word": "hao3", "start": 1.4, "end": 1.5},
    ]


def test_ambiguous_candidates_with_all_zero_raw_overlap_are_rejected():
    words = _ambiguous_raw_words()
    errors = attach_nvasr_candidate_provenance(
        words, [], _ambiguous_raw_timeline([0.0, 0.1], [0.2, 0.3]))
    assert errors and "ambiguous" in errors[0]
    assert not any("candidate_id" in row for row in words)


def test_ambiguous_candidates_with_equal_positive_raw_overlap_are_rejected():
    words = _ambiguous_raw_words()
    errors = attach_nvasr_candidate_provenance(
        words, [], _ambiguous_raw_timeline([0.95, 1.05], [0.95, 1.05]))
    assert errors and "ambiguous" in errors[0]
    assert not any("candidate_id" in row for row in words)


def test_ambiguous_candidates_with_malformed_raw_evidence_are_rejected():
    words = _ambiguous_raw_words()
    errors = attach_nvasr_candidate_provenance(
        words, [], _ambiguous_raw_timeline(["bad", 1.05], [1.02, 1.08]))
    assert errors and "malformed candidate mapping" in errors[0]
    assert not any("candidate_id" in row for row in words)


def test_ambiguous_candidates_with_malformed_speech_competitor_are_rejected():
    words = _ambiguous_raw_words()
    timeline = _ambiguous_raw_timeline([0.96, 1.02], [1.26, 1.38])
    timeline["candidates"][0]["speech_start_s"] = "bad"
    timeline["candidates"][1]["speech_start_s"] = 0.95
    timeline["candidates"][1]["speech_end_s"] = 1.05

    errors = attach_nvasr_candidate_provenance(words, [], timeline)

    assert errors and "malformed candidate mapping" in errors[0]
    assert not any("candidate_id" in row for row in words)


def test_single_candidate_missing_frame_field_is_structurally_rejected():
    timeline = _ambiguous_raw_timeline([0.96, 1.02], [1.26, 1.38])
    timeline["candidates"] = timeline["candidates"][:1]
    timeline["candidates"][0].pop("raw_start_frame")
    words = _ambiguous_raw_words()
    before = [dict(row) for row in words]

    errors = attach_nvasr_candidate_provenance(words, [], timeline)

    assert errors and "malformed candidate mapping" in errors[0]
    assert words == before


def test_ambiguous_candidate_missing_frame_field_is_atomic_rejection():
    words = _ambiguous_raw_words()
    before = [dict(row) for row in words]
    timeline = _ambiguous_raw_timeline([0.96, 1.02], [1.26, 1.38])
    timeline["candidates"][1].pop("speech_end_frame")

    errors = attach_nvasr_candidate_provenance(words, [], timeline)

    assert errors and "malformed candidate mapping" in errors[0]
    assert words == before


def _topology_candidate(candidate_id, surface, key, speech_span, source="ctc",
                        raw_span=None, token_id=9724):
    raw_span = raw_span or speech_span
    return {
        "candidate_id": candidate_id,
        "occurrence": int(candidate_id.split("-")[-1]),
        "kind": "punctuation" if surface == "…" else "nvv",
        "surface": surface,
        "token_id": token_id,
        "token_ids": [token_id],
        "source": source,
        "left_lexical_ordinal": key[0],
        "right_lexical_ordinal": key[1],
        "raw_start_s": raw_span[0], "raw_end_s": raw_span[1],
        "speech_start_s": speech_span[0], "speech_end_s": speech_span[1],
        "raw_start_frame": 100 + int(candidate_id.split("-")[-1]),
        "raw_end_frame": 101 + int(candidate_id.split("-")[-1]),
        "speech_start_frame": 200 + int(candidate_id.split("-")[-1]),
        "speech_end_frame": 201 + int(candidate_id.split("-")[-1]),
    }


def _punctuation_topology_fixture():
    key = (21, 22)
    timeline = {"candidates": [
        _topology_candidate("candidate-0006", "[Breathing]", (None, 0),
                            [6.36, 6.48], raw_span=[6.36, 6.48], token_id=11),
        _topology_candidate("candidate-0007", "…", key, [6.48, 6.60]),
        _topology_candidate("candidate-0008", "TSK", key, [6.60, 6.66],
                            raw_span=[8.00, 8.06], token_id=25032),
        _topology_candidate("candidate-0009", "…", key, [6.66, 6.72]),
        _topology_candidate("candidate-0010", "TSK", key, [6.72, 6.78],
                            raw_span=[8.20, 8.26], token_id=25032),
        _topology_candidate("candidate-0011", "…", key, [7.02, 7.08],
                            source="blank_run"),
    ]}
    words = [
        {"word": "BREATHING", "start": 6.39, "end": 6.45},
        {"word": "TSK", "start": 6.51, "end": 6.57,
         "mapping_key": {"left_lexical_ordinal": 21,
                          "right_lexical_ordinal": 22}},
        {"word": "wo3", "start": 7.23, "end": 7.29},
    ]
    punct_entries = [
        {"word": "…", "start": 6.45, "end": 6.51,
         "raw_start_s": 6.45, "raw_end_s": 6.51,
         "candidate_id": "ctc-punct-pre", "source": "ctc",
         "mapping_key": {"left_lexical_ordinal": 21,
                          "right_lexical_ordinal": 22}},
        {"word": "…", "start": 6.63, "end": 6.69,
         "raw_start_s": 6.63, "raw_end_s": 6.69,
         "candidate_id": "ctc-punct-post", "source": "ctc",
         "mapping_key": {"left_lexical_ordinal": 21,
                          "right_lexical_ordinal": 22}},
    ]
    return words, punct_entries, timeline


def test_ambiguous_nvv_uses_unique_ctc_punctuation_topology_binding():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors == []
    assert words[1]["candidate_id"] == "candidate-0008"
    assert words[1]["mapping_selection"] == "unique_punctuation_topology_bound"
    assert punct_entries == before


@pytest.mark.parametrize("missing_index", [0, 1])
def test_punctuation_topology_requires_both_immediate_boundaries(missing_index):
    words, punct_entries, timeline = _punctuation_topology_fixture()
    punct_entries.pop(missing_index)
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_rejects_tied_boundary_binding():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    duplicate = deepcopy(timeline["candidates"][1])
    duplicate["candidate_id"] = "candidate-0012"
    timeline["candidates"].insert(2, duplicate)
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


@pytest.mark.parametrize("bad_source", ["blank_run", "decoder"])
def test_non_ctc_punctuation_cannot_substitute_for_topology_boundary(bad_source):
    words, punct_entries, timeline = _punctuation_topology_fixture()
    if bad_source == "blank_run":
        punct_entries[0]["source"] = bad_source
    else:
        timeline["candidates"][1]["source"] = bad_source
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_rejects_two_matching_nvv_candidates_inside_window():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    duplicate = deepcopy(timeline["candidates"][2])
    duplicate["candidate_id"] = "candidate-0012"
    duplicate["raw_start_s"], duplicate["raw_end_s"] = 8.40, 8.46
    timeline["candidates"].insert(3, duplicate)
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_rejects_matching_nvv_candidates_only_outside_window():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    timeline["candidates"].pop(2)  # remove candidate-0008 from inside
    before_candidate = deepcopy(timeline["candidates"][0])
    before_candidate["candidate_id"] = "candidate-0012"
    before_candidate["surface"] = "TSK"
    before_candidate["kind"] = "nvv"
    before_candidate["left_lexical_ordinal"] = 21
    before_candidate["right_lexical_ordinal"] = 22
    before_candidate["raw_start_s"], before_candidate["raw_end_s"] = 5.0, 5.06
    before_candidate["speech_start_s"], before_candidate["speech_end_s"] = 5.0, 5.06
    after_candidate = deepcopy(before_candidate)
    after_candidate["candidate_id"] = "candidate-0013"
    after_candidate["raw_start_s"], after_candidate["raw_end_s"] = 8.4, 8.46
    after_candidate["speech_start_s"], after_candidate["speech_end_s"] = 8.4, 8.46
    timeline["candidates"].insert(1, before_candidate)
    timeline["candidates"].insert(5, after_candidate)
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_rejects_zero_boundary_overlaps():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    for index in (1, 3):
        timeline["candidates"][index]["speech_start_s"] = 8.0
        timeline["candidates"][index]["speech_end_s"] = 8.1
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_rejects_reversed_timeline_boundary_order():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    timeline["candidates"][1], timeline["candidates"][3] = (
        timeline["candidates"][3], timeline["candidates"][1])
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


@pytest.mark.parametrize(
    "entry_index, start, end",
    [(0, 6.45, 6.52), (1, 6.56, 6.69)],
)
def test_punctuation_topology_rejects_emitted_boundary_overlapping_nvv(
        entry_index, start, end):
    words, punct_entries, timeline = _punctuation_topology_fixture()
    punct_entries[entry_index]["start"] = start
    punct_entries[entry_index]["end"] = end
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_cannot_resolve_positive_nvv_speech_tie():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    for index in (2, 4):
        timeline["candidates"][index]["speech_start_s"] = 6.52
        timeline["candidates"][index]["speech_end_s"] = 6.56
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_punctuation_topology_cannot_resolve_positive_nvv_raw_tie():
    words, punct_entries, timeline = _punctuation_topology_fixture()
    for index in (2, 4):
        timeline["candidates"][index]["raw_start_s"] = 6.52
        timeline["candidates"][index]["raw_end_s"] = 6.56
    before = deepcopy(punct_entries)

    errors = attach_nvasr_candidate_provenance(words, punct_entries, timeline)

    assert errors and "ambiguous" in errors[-1]
    assert not any("candidate_id" in row for row in words)
    assert punct_entries == before


def test_token_sidecar_round_trip_keeps_candidate_provenance():
    word = {
        "word": "BREATHING",
        "candidate_id": "nvasr-candidate-0004",
        "candidate_kind": "nvv",
        "provenance_schema": "nvasr-candidate-provenance-v1",
        "raw_span": [3.66, 3.72],
        "raw_start_s": 3.66,
        "raw_end_s": 3.72,
        "raw_frame_count": 1,
        "frame_ms": 60,
        "speech_span": [3.42, 3.48],
        "forced_span": [3.39, 3.45],
        "mapping_basis": "raw_ctc_label_neighbors_forced_overlap-v2",
        "mapping_outcome": "unique",
        "mapping_selection": "label_neighbors",
    }

    serialized = _ctc_token_sidecar_row(word, 3.39, 3.45)
    for key, value in word.items():
        assert serialized[key] == value


def test_nvv_binding_does_not_overwrite_punctuation_coordinate_contract():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 12, 32],
        "你，好",
        token_surfaces={31: "你", 32: "好", 12: "，"},
    )
    words = [
        {"word": "ni3", "start": 0.00, "end": 0.06},
        {"word": "hao3", "start": 0.12, "end": 0.18},
    ]
    punctuation = [{
        "word": "，", "start": 0.06, "end": 0.12,
        "raw_start_s": 0.06, "raw_end_s": 0.12,
        "candidate_id": "ctc-punct-0000", "source": "ctc",
    }]
    before = dict(punctuation[0])

    assert attach_nvasr_candidate_provenance(
        words, punctuation, timeline) == []
    assert punctuation[0] == before
