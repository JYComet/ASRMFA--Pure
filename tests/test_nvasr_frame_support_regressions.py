from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts.audit_strict_ok import (
    _nvasr_authority_candidate_reasons as
    _audit_authority_candidate_reasons,
    _nvasr_candidate_immutable_projection as
    _audit_candidate_immutable_projection,
)
from scripts.ctc_prealign import (
    NVASR_CANDIDATE_SCHEMA_VERSION,
    NVASR_MAPPING_AXIS,
    NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA,
    NVASR_SEMANTIC_AXIS_SCHEMA,
    _finalize_nvasr_canonical_neighbors,
    _nvasr_anchor_from_frames,
    _nvasr_raw_timeline_evidence_sha256,
    _rebase_final_token_sidecars,
    attach_nvasr_candidate_provenance,
    extract_nvasr_candidate_timeline,
)
from scripts.postprocess_textgrids import (
    Interval,
    NVASR_IMMUTABLE_PROJECTION_SCHEMA,
    NVASR_PRODUCER_AUTHORITY_SCHEMA,
    Tier,
    _nvasr_authority_candidate_reasons,
    _nvasr_candidate_immutable_projection,
    _contain_nvasr_frame_support,
    _nvasr_candidate_provenance_audit,
    _nvasr_frame_support,
    _nvasr_select_owner,
    _nvasr_stable_json_digest,
    _publish_nvasr_owner_report,
)


def _row(*, raw_start_frame: int, frame_count: int, candidate_id: str,
         key: tuple[int | None, int | None], forced_span: list[float],
         label: str = "BREATHING") -> dict:
    raw_end_frame = raw_start_frame + frame_count
    speech_start_frame = raw_start_frame - 4
    speech_end_frame = raw_end_frame - 4
    raw_span = [raw_start_frame * 0.06, raw_end_frame * 0.06]
    speech_span = [speech_start_frame * 0.06, speech_end_frame * 0.06]
    return {
        "candidate_kind": "nvv",
        "candidate_id": candidate_id,
        "word": label,
        "start_s": forced_span[0],
        "end_s": forced_span[1],
        "raw_start_s": raw_span[0],
        "raw_end_s": raw_span[1],
        "raw_start_frame": raw_start_frame,
        "raw_end_frame": raw_end_frame,
        "raw_frame_count": frame_count,
        "speech_start_frame": speech_start_frame,
        "speech_end_frame": speech_end_frame,
        "speech_frame_count": frame_count,
        "frame_ms": 60,
        "raw_span": raw_span,
        "speech_span": speech_span,
        "forced_span": list(forced_span),
        "adjusted_span": list(forced_span),
        "provenance_schema": "nvasr-candidate-provenance-v1",
        "mapping_basis": "raw_ctc_label_neighbors_forced_overlap-v2",
        "mapping_axis": NVASR_SEMANTIC_AXIS_SCHEMA,
        "mapping_outcome": "unique",
        "mapping_selection": "label_neighbors",
        "mapping_key": {
            "left_lexical_ordinal": key[0],
            "right_lexical_ordinal": key[1],
        },
    }


def _containment_fixture(row: dict, *, nvv_span: tuple[float, float],
                         axis: tuple[float, float] = (0.0, 1.0)):
    words = Tier("words", axis[0], axis[1], [
        Interval(axis[0], nvv_span[0], "ni3"),
        Interval(nvv_span[0], nvv_span[1], f"<{row['word']}>"),
        Interval(nvv_span[1], axis[1], "hao3"),
    ])
    ctc = [
        {"type": "word", "word": "ni3", "start_s": axis[0],
         "end_s": nvv_span[0]},
        row,
        {"type": "word", "word": "hao3", "start_s": nvv_span[1],
         "end_s": axis[1]},
    ]
    return words, ctc


def _v3_row(*, raw_start_frame: int, frame_count: int,
            candidate_id: str, ctc_ordinal: int,
            key: tuple[int | None, int | None],
            forced_span: list[float], adjusted_span: list[float],
            neighbors: list[dict], mapping_selection: str = "label_neighbors",
            anchor_overlap: float | None = None,
            label: str = "BREATHING",
            raw_left_surface: str | None = None,
            raw_right_surface: str | None = None) -> dict:
    """Build one serialized schema-v3 candidate for strict owner tests."""
    raw_end_frame = raw_start_frame + frame_count
    speech_start_frame = raw_start_frame - 4
    speech_end_frame = raw_end_frame - 4
    anchor = _nvasr_anchor_from_frames(raw_start_frame, raw_end_frame)
    row = {
        "candidate_kind": "nvv",
        "candidate_id": candidate_id,
        "word": label,
        "candidate_surface": f"[{label.title()}]",
        "candidate_source": "ctc",
        "candidate_token_id": 11,
        "candidate_token_ids": [11] * frame_count,
        "ctc_lexical_ordinal": ctc_ordinal,
        "start_s": forced_span[0],
        "end_s": forced_span[1],
        "raw_start_s": raw_start_frame * 0.06,
        "raw_end_s": raw_end_frame * 0.06,
        "raw_start_frame": raw_start_frame,
        "raw_end_frame": raw_end_frame,
        "raw_frame_count": frame_count,
        "speech_start_frame": speech_start_frame,
        "speech_end_frame": speech_end_frame,
        "speech_frame_count": frame_count,
        "query_frames": 4,
        "frame_ms": 60,
        "raw_span": [raw_start_frame * 0.06, raw_end_frame * 0.06],
        "speech_span": [speech_start_frame * 0.06, speech_end_frame * 0.06],
        "speech_start_s": speech_start_frame * 0.06,
        "speech_end_s": speech_end_frame * 0.06,
        "forced_span": list(forced_span),
        "adjusted_span": list(adjusted_span),
        "provenance_schema": "nvasr-candidate-provenance-v1",
        "mapping_basis": "raw_ctc_label_neighbors_forced_overlap-v2",
        "mapping_axis": NVASR_MAPPING_AXIS,
        "mapping_outcome": "unique",
        "mapping_selection": mapping_selection,
        "mapping_key": {
            "left_lexical_ordinal": key[0],
            "right_lexical_ordinal": key[1],
        },
        "nvasr_candidate_schema_version": NVASR_CANDIDATE_SCHEMA_VERSION,
        "ordered_semantic_neighbors": deepcopy(neighbors),
        "raw_timeline_neighbors_schema":
            NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA,
        "ctc_spike_anchor": anchor,
        "raw_timeline_mapping_key": {
            "left_lexical_ordinal": key[0],
            "right_lexical_ordinal": key[1],
        },
        "adjusted_span_basis":
            "ctc_spike_anchor_forced_correspondence_envelope_v1",
        "adjusted_span_is_acoustic_evidence": False,
        "ctc_raw_token_row": {
            "schema": "ctc_raw_token_row_v1",
            "stem": "fixture",
            "sidecar": "fixture_tokens.jsonl",
            "row_ordinal": ctc_ordinal,
        },
    }
    semantic_by_side = {item["side"]: item for item in neighbors}
    left_present = key[0] is not None
    right_present = key[1] is not None
    row["raw_timeline_index"] = 1 if left_present else 0
    row["raw_timeline_event_count"] = (
        1 + int(left_present) + int(right_present))
    row["raw_timeline_neighbors"] = {
        "left": ({
            "surface": (raw_left_surface if raw_left_surface is not None else
                        semantic_by_side.get("left", {}).get("surface", "raw-left")),
            "source": "ctc",
            "token_id": 31,
            "ordered_source_frame_ids": [raw_start_frame - 1],
        } if left_present else None),
        "right": ({
            "surface": (raw_right_surface if raw_right_surface is not None else
                        semantic_by_side.get("right", {}).get("surface", "raw-right")),
            "source": "ctc",
            "token_id": 32,
            "ordered_source_frame_ids": [raw_end_frame],
        } if right_present else None),
    }
    row["raw_timeline_evidence_sha256"] = (
        _nvasr_raw_timeline_evidence_sha256(row))
    row["adjusted_span"] = [
        min(adjusted_span[0], forced_span[0], anchor["start"]),
        max(adjusted_span[1], forced_span[1], anchor["end"]),
    ]
    if anchor_overlap is not None:
        row["mapping_forced_ctc_anchor_overlap_s"] = anchor_overlap
    return row


def _ordinary(word: str, ordinal: int, occurrence: int = 0) -> dict:
    return {
        "type": "word", "word": word,
        "semantic_occurrence_id": f"nvasr-lexical-{ordinal:04d}",
        "semantic_surface_occurrence": occurrence,
        "start_s": 0.0, "end_s": 1.0,
    }


def _strict_authority(ctc: list[dict]) -> dict:
    """Build the private manifest-authority shape for isolated consumer tests."""
    candidates = [row for row in ctc
                  if isinstance(row, dict)
                  and row.get("candidate_kind") == "nvv"]
    projections = [
        _nvasr_candidate_immutable_projection(row) for row in candidates]
    stem = (candidates[0].get("ctc_raw_token_row", {}).get("stem")
            if candidates else None)
    return {
        "summary": {
            "schema": NVASR_PRODUCER_AUTHORITY_SCHEMA,
            "status": "verified",
            "raw_manifest_identity": "fixture-manifest",
            "raw_tokens_sha256": "0" * 64,
            "work_receipt_identity": "fixture-work-receipt",
            "candidate_count": len(candidates),
            "ordered_projection_sha256": _nvasr_stable_json_digest({
                "schema": NVASR_IMMUTABLE_PROJECTION_SCHEMA,
                "stem": stem,
                "candidates": projections,
            }),
            "reasons": [],
        },
        "ordered_raw_projections": projections,
    }


def _strict_words_ctc(row: dict, *, left: str = "chi2", right: str = "hao3",
                      nvv_span: tuple[float, float] = (10.35, 10.41),
                      left_span: tuple[float, float] = (9.82, 9.94),
                      right_span: tuple[float, float] = (10.41, 11.0)):
    left_row = _ordinary(left, 0)
    right_row = _ordinary(right, 1)
    ctc = [left_row, row, right_row]
    words = Tier("words", 0.0, 11.0, [
        Interval(left_span[0], left_span[1], left),
        Interval(nvv_span[0], nvv_span[1], f"<{row['word']}>") ,
        Interval(right_span[0], right_span[1], right),
    ])
    return words, ctc


def test_00452_forced_span_is_correspondence_only_for_one_frame_support():
    row = _row(raw_start_frame=177, frame_count=1, candidate_id="00452",
               key=(0, 1), forced_span=[10.41, 10.47])
    support, source, frame_limited, reasons = _nvasr_frame_support(row)

    assert support == pytest.approx([10.35, 10.41])
    assert source == "raw_ctc_frames_shifted_to_speech_axis"
    assert frame_limited is True
    assert reasons == []

    words, ctc = _containment_fixture(
        row, nvv_span=(10.35, 10.41), axis=(0.0, 11.0))
    result = _contain_nvasr_frame_support(words, ctc, wav_duration_s=11.0)
    assert result["status"] == "verified"
    assert result["candidates"][0]["final_contains_frame_support"] is True


@pytest.mark.parametrize("frame_count", [2, 3, 5])
def test_physical_support_duration_is_raw_frame_count_times_60ms(frame_count):
    row = _row(raw_start_frame=20, frame_count=frame_count,
               candidate_id=f"duration-{frame_count}", key=(0, 1),
               forced_span=[1.17, 1.23])
    support, _source, frame_limited, reasons = _nvasr_frame_support(row)

    assert reasons == []
    assert support[1] - support[0] == pytest.approx(frame_count * 0.06)
    assert frame_limited is False
    assert support[1] - support[0] != pytest.approx(0.06)


def test_00460_producer_consumer_round_trip_uses_compact_semantic_axis():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 90, 11, 32],
        "你 sil [Breathing] 好",
        token_surfaces={31: "你", 90: "sil", 11: "[Breathing]", 32: "好"},
        stem="00460",
    )
    candidate = timeline["candidates"][0]
    assert timeline["semantic_axis"]["schema"] == NVASR_SEMANTIC_AXIS_SCHEMA
    assert (candidate["left_lexical_ordinal"],
            candidate["right_lexical_ordinal"]) == (0, 1)

    words = [
        {"word": "ni3", "start": 0.0, "end": 0.3},
        {"word": "BREATHING", "start": 0.3, "end": 0.36},
        {"word": "hao3", "start": 0.36, "end": 0.6},
    ]
    assert attach_nvasr_candidate_provenance(words, [], timeline) == []
    assert words[1]["mapping_axis"] == NVASR_SEMANTIC_AXIS_SCHEMA
    assert words[1]["mapping_key"] == {
        "left_lexical_ordinal": 0, "right_lexical_ordinal": 1}

    words_tier, ctc = _containment_fixture(
        words[1], nvv_span=(0.30, 0.36), axis=(0.0, 1.0))
    ctc.insert(1, {"type": "word", "word": "sil",
                   "start_s": 0.12, "end_s": 0.18})
    result = _contain_nvasr_frame_support(
        words_tier, ctc, wav_duration_s=1.0)
    assert result["status"] == "verified"


def test_real_mapping_axis_mismatch_is_rejected():
    row = _row(raw_start_frame=11, frame_count=1, candidate_id="mismatch",
               key=(0, 1), forced_span=[0.39, 0.45])
    row["mapping_axis"] = "wrong-axis"
    words, ctc = _containment_fixture(row, nvv_span=(0.30, 0.36))
    result = _contain_nvasr_frame_support(words, ctc, wav_duration_s=1.0)

    assert result["status"] == "rejected"
    assert "frame_support_mapping_axis_mismatch:0" in result["reasons"]


def test_00462_joint_repartition_keeps_two_frame_supports_and_third_owner():
    first = _row(raw_start_frame=36, frame_count=2, candidate_id="00462-a",
                 key=(0, 1), forced_span=[1.89, 2.01])
    second = _row(raw_start_frame=70, frame_count=2, candidate_id="00462-b",
                  key=(2, 3), forced_span=[3.93, 4.05])
    third = _row(raw_start_frame=107, frame_count=1, candidate_id="00462-c",
                 key=(3, 4), forced_span=[6.15, 6.21])
    words = Tier("words", 0.0, 7.0, [
        Interval(0.0, 1.95, "a3"),
        Interval(1.95, 2.01, "<BREATHING>"),
        Interval(2.01, 3.81, "b3"),
        Interval(3.81, 3.93, "c3"),
        Interval(3.93, 3.99, ""),
        Interval(3.99, 4.05, "<BREATHING>"),
        Interval(4.05, 6.15, "d3"),
        Interval(6.15, 6.21, "<BREATHING>"),
        Interval(6.21, 7.0, "e3"),
    ])
    ctc = [
        {"type": "word", "word": "a3", "start_s": 0.0, "end_s": 1.95},
        first,
        {"type": "word", "word": "b3", "start_s": 2.01, "end_s": 3.81},
        {"type": "word", "word": "c3", "start_s": 3.81, "end_s": 3.93},
        second,
        {"type": "word", "word": "d3", "start_s": 4.05, "end_s": 6.15},
        third,
        {"type": "word", "word": "e3", "start_s": 6.21, "end_s": 7.0},
    ]

    result = _contain_nvasr_frame_support(words, ctc, wav_duration_s=7.0)

    assert result["status"] == "verified"
    contained = result["_contained_tier"]
    owners = [contained.intervals[index] for index in (1, 5, 7)]
    assert [owner.text for owner in owners] == [
        "<BREATHING>", "<BREATHING>", "<BREATHING>"]
    assert owners[0].xmin <= 1.89 and owners[0].xmax >= 2.01
    assert owners[1].xmin <= 3.93 and owners[1].xmax >= 4.05
    assert owners[2].xmin <= 6.15 and owners[2].xmax >= 6.21
    assert all(interval.xmax > interval.xmin
               for interval in contained.intervals)
    assert all(left.xmax <= right.xmin + 1e-9
               for left, right in zip(contained.intervals,
                                      contained.intervals[1:]))
    assert [item["frame_limited"] for item in result["candidates"]] == [
        False, False, True]
    assert result["repartitioned_intervals"] >= 5


def test_joint_repartition_rejects_when_positive_geometry_is_infeasible():
    first = _row(raw_start_frame=11, frame_count=2, candidate_id="bad-a",
                 key=(0, 1), forced_span=[0.39, 0.51])
    second = _row(raw_start_frame=13, frame_count=2, candidate_id="bad-b",
                  key=(1, 2), forced_span=[0.51, 0.63])
    words = Tier("words", 0.0, 0.12, [
        Interval(0.0, 0.06, "a3"),
        Interval(0.06, 0.12, "<BREATHING>"),
        Interval(0.12, 0.12, "<BREATHING>"),
    ])
    ctc = [
        {"type": "word", "word": "a3", "start_s": 0.0, "end_s": 0.06},
        first, second,
    ]
    result = _contain_nvasr_frame_support(words, ctc, wav_duration_s=0.12)

    assert result["status"] == "rejected"
    assert "frame_support_out_of_axis:0" in result["reasons"]
    assert result["changed"] == 0


def test_v3_00452_compatible_source_owner_protects_adjacent_chi2():
    row = _v3_row(
        raw_start_frame=177, frame_count=1, candidate_id="00452-v3",
        ctc_ordinal=1, key=(0, 1), forced_span=[10.41, 10.47],
        adjusted_span=[10.41, 10.47], neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "chi2",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "hao3",
             "surface_occurrence": 0},
        ])
    words, ctc = _strict_words_ctc(
        row, left_span=(9.82, 9.90), nvv_span=(10.35, 10.41),
        right_span=(10.41, 11.0))
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 9.82, "end": 9.94, "text": "chi2"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 9.94, "end": 10.35, "text": "breathing"},
        {"ordinal": 2, "ctc_lexical_ordinal": 2,
         "start": 10.35, "end": 11.0, "text": "hao3"},
    ]
    lineage = {"schema": "source-phone-lineage-v1", "status": "verified",
               "owners": {
                   "0": [{"label": "j", "start": 9.82, "end": 9.94}],
                   "1": [{"label": "spn", "start": 9.94,
                           "end": 10.35}],
               }}
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage=lineage,
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    candidate = result["candidates"][0]
    assert candidate["owner_branch"] == "compatible_source_mfa"
    assert candidate["owner_selected_span"] == pytest.approx([9.94, 10.35])
    assert candidate["anchor_span"] == pytest.approx([10.35, 10.41])
    assert row["ctc_lexical_ordinal"] == \
        source_words[1]["ctc_lexical_ordinal"] == 1
    assert ctc[1]["forced_span"] == pytest.approx([10.41, 10.47])
    assert candidate["owner_selected_span"][1] < ctc[1]["forced_span"][1]
    contained = result["_contained_tier"]
    assert [(iv.xmin, iv.xmax) for iv in contained.intervals] == [
        (pytest.approx(9.82), pytest.approx(9.94)),
        (pytest.approx(9.94), pytest.approx(10.35)),
        (pytest.approx(10.35), pytest.approx(11.0)),
    ]


def test_v3_00460_canonical_round_trip_uses_gap_owner_and_anchor_overlap_proof():
    row = _v3_row(
        raw_start_frame=50, frame_count=5, candidate_id="00460-v3",
        ctc_ordinal=1, key=(0, 1), forced_span=[2.73, 3.03],
        adjusted_span=[2.73, 3.03], mapping_selection=
        "unique_label_forced_ctc_overlap", anchor_overlap=0.30,
        neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "ni3",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "allin",
             "surface_occurrence": 0},
        ])
    words = Tier("words", 0.0, 4.0, [
        Interval(2.0, 2.10, "ni3"),
        Interval(2.30, 2.40, "<BREATHING>"),
        Interval(3.03, 3.50, "allin"),
    ])
    ctc = [_ordinary("ni3", 0), row, _ordinary("allin", 1)]
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 2.0, "end": 2.10, "text": "ni3"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 2.10, "end": 2.15, "text": "breathing"},
        {"ordinal": 2, "ctc_lexical_ordinal": 2,
         "start": 3.03, "end": 3.50, "text": "allin"},
    ]
    lineage = {"schema": "source-phone-lineage-v1", "status": "verified",
               "owners": {"1": [{"label": "spn", "start": 2.10,
                                     "end": 2.15}]}}
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=4.0, source_words=source_words,
        source_phone_lineage=lineage,
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    candidate = result["candidates"][0]
    assert candidate["owner_branch"] == "adjusted_ctc_energy_gap"
    assert candidate["owner_selected_span"] == pytest.approx([2.73, 3.03])
    assert candidate["anchor_span"] == pytest.approx([2.73, 3.03])
    assert row["ctc_lexical_ordinal"] == \
        source_words[1]["ctc_lexical_ordinal"] == 1
    assert source_words[1]["end"] + 0.03 < candidate["anchor_span"][0]
    assert candidate["owner_reason"] == \
        "anchor_and_forced_in_source_nonlexical_gap"
    assert result["reasons"] == []


def test_final_rebase_changes_only_final_coordinates_and_preserves_raw_evidence(
        tmp_path):
    candidate = _v3_row(
        raw_start_frame=177, frame_count=1, candidate_id="rebase-nvv",
        ctc_ordinal=1, key=(1, 2), forced_span=[10.41, 10.47],
        adjusted_span=[10.41, 10.47], neighbors=[
            {"side": "left", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "lo",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 2,
             "occurrence_id": "nvasr-lexical-0002", "surface": "beta",
             "surface_occurrence": 0},
        ])
    raw_snapshot = {
        key: deepcopy(candidate[key]) for key in (
            "raw_timeline_mapping_key", "raw_timeline_neighbors",
            "ctc_spike_anchor", "raw_timeline_evidence_sha256")}
    rows = [
        {"word": "hello", "start_s": 0.0, "end_s": 0.2,
         "source_ctc_ordinals": [0, 1]},
        candidate,
        {"word": "beta", "start_s": 0.2, "end_s": 0.3,
         "source_ctc_ordinals": [3]},
    ]
    token_path = tmp_path / "demo_tokens.jsonl"
    token_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")

    assert _rebase_final_token_sidecars(tmp_path) == 1
    rebased = [json.loads(line) for line in token_path.read_text(
        encoding="utf-8").splitlines()]
    assert [row["ctc_raw_token_row"]["row_ordinal"] for row in rebased] == [0, 1, 2]
    assert [row["ctc_raw_token_row"]["sidecar"] for row in rebased] == [
        "demo_tokens.jsonl"] * 3
    assert [row.get("ctc_lexical_ordinal") for row in rebased] == [0, 1, 2]
    assert rebased[1]["mapping_key"] == {
        "left_lexical_ordinal": 0, "right_lexical_ordinal": 1}
    assert rebased[1]["ordered_semantic_neighbors"] == [
        {"side": "left", "lexical_ordinal": 0,
         "occurrence_id": "nvasr-lexical-0000", "surface": "hello",
         "surface_occurrence": 0},
        {"side": "right", "lexical_ordinal": 1,
         "occurrence_id": "nvasr-lexical-0001", "surface": "beta",
         "surface_occurrence": 0},
    ]
    assert {key: rebased[1][key] for key in raw_snapshot} == raw_snapshot


def test_rebased_canonical_and_raw_mapping_axes_are_independently_authoritative():
    words, ctc, source_words = _valid_v3_case()
    row = ctc[1]
    # Model the 00460 English contraction: final neighbours were rebased to
    # 0/1, while immutable raw decoder neighbours remain at 1/2.
    row["raw_timeline_mapping_key"] = {
        "left_lexical_ordinal": 1, "right_lexical_ordinal": 2}
    _rehash_raw_timeline(row)

    assert _nvasr_authority_candidate_reasons(row) == []
    assert _audit_authority_candidate_reasons(row) == []
    assert _nvasr_candidate_immutable_projection(row) == \
        _audit_candidate_immutable_projection(row)

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "verified"


@pytest.mark.parametrize("tamper", [
    "canonical_mapping_key", "canonical_neighbors", "raw_mapping_key",
])
def test_manifest_authority_seals_both_rebased_and_raw_mapping_axes(tamper):
    words, ctc, source_words = _valid_v3_case()
    row = ctc[1]
    row["raw_timeline_mapping_key"] = {
        "left_lexical_ordinal": 1, "right_lexical_ordinal": 2}
    _rehash_raw_timeline(row)
    authority = _strict_authority(ctc)

    if tamper == "canonical_mapping_key":
        row["mapping_key"]["left_lexical_ordinal"] = 9
    elif tamper == "canonical_neighbors":
        row["ordered_semantic_neighbors"][0]["surface"] = "tampered"
    else:
        row["raw_timeline_mapping_key"]["left_lexical_ordinal"] = 9
        _rehash_raw_timeline(row)

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=authority, strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert "nvasr_manifest_anchored_projection_mismatch" in result["reasons"]


def test_v3_00462_two_frame_source_owner_preserves_ya5_and_phone_cores():
    row = _v3_row(
        raw_start_frame=70, frame_count=2, candidate_id="00462-breathing-2",
        ctc_ordinal=1, key=(0, 1), forced_span=[3.93, 4.05],
        adjusted_span=[3.93, 4.05], neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "ya5",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "ni3",
             "surface_occurrence": 0},
        ])
    words = Tier("words", 3.0, 5.0, [
        Interval(3.83, 3.98, "ya5"),
        Interval(3.93, 3.99, "<BREATHING>"),
        Interval(4.10, 5.0, "ni3"),
    ])
    ctc = [_ordinary("ya5", 0), row, _ordinary("ni3", 1)]
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 3.83, "end": 3.98, "text": "ya5"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 3.98, "end": 4.10, "text": "breathing"},
        {"ordinal": 2, "ctc_lexical_ordinal": 2,
         "start": 4.10, "end": 5.0, "text": "ni3"},
    ]
    lineage = {"schema": "source-phone-lineage-v1", "status": "verified",
               "owners": {"0": [
                   {"label": "j", "start": 3.83, "end": 3.90},
                   {"label": "a", "start": 3.90, "end": 3.98},
               ], "1": [
                   {"label": "spn", "start": 3.98, "end": 4.10},
               ]}}
    lineage_before = deepcopy(lineage)
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=5.0, source_words=source_words,
        source_phone_lineage=lineage,
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    candidate = result["candidates"][0]
    assert candidate["owner_branch"] == "compatible_source_mfa"
    assert candidate["owner_selected_span"] == pytest.approx([3.98, 4.10])
    assert candidate["anchor_span"] == pytest.approx([3.93, 4.05])
    assert row["ctc_lexical_ordinal"] == \
        source_words[1]["ctc_lexical_ordinal"] == 1
    assert lineage == lineage_before
    contained = result["_contained_tier"]
    assert [(iv.xmin, iv.xmax) for iv in contained.intervals] == [
        (pytest.approx(3.83), pytest.approx(3.98)),
        (pytest.approx(3.98), pytest.approx(4.10)),
        (pytest.approx(4.10), pytest.approx(5.0)),
    ]
    assert candidate["owner_selected_span"][1] - candidate["owner_selected_span"][0] == pytest.approx(0.12)


def test_v3_canonical_neighbor_fallback_persists_distinct_anchor_overlap():
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 90, 91, 32],
        "你[Breathing]all in 好",
        token_surfaces={31: "你", 11: "[Breathing]", 90: "all",
                        91: "in", 32: "好"})
    words = [
        {"word": "ni3", "start": 0.0, "end": 0.1},
        {"word": "BREATHING", "start": 0.03, "end": 0.09,
         "mapping_key": {"left_lexical_ordinal": 9,
                          "right_lexical_ordinal": 9}},
        {"word": "all", "start": 0.09, "end": 0.15},
        {"word": "in", "start": 0.15, "end": 0.21},
        {"word": "hao3", "start": 0.21, "end": 0.3},
    ]
    assert attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True) == []
    assert words[1]["mapping_selection"] == "unique_label_forced_ctc_overlap"
    assert words[1]["mapping_forced_ctc_anchor_overlap_s"] > 0
    assert "mapping_forced_speech_overlap_s" not in words[1]
    raw_neighbors = deepcopy(words[1]["raw_timeline_neighbors"])
    words[2] = {"word": "allin", "start": words[2]["start"],
                "end": words[3]["end"]}
    del words[3]
    _finalize_nvasr_canonical_neighbors(words)
    assert words[1]["ordered_semantic_neighbors"][1]["surface"] == "allin"
    assert words[1]["ordered_semantic_neighbors"][1]["occurrence_id"] == \
        "nvasr-lexical-0001"
    assert words[1]["raw_timeline_neighbors"] == raw_neighbors
    assert raw_neighbors["right"]["surface"] == "all"


def _tamper_anchor_binding(row: dict, mutation: str) -> None:
    raw_start = row["raw_start_frame"]
    raw_end = row["raw_end_frame"]
    if mutation == "coordinate_only":
        row["ctc_spike_anchor"]["start"] += 0.001
        return
    if mutation == "anchor_frames_recomputed":
        row["ctc_spike_anchor"] = _nvasr_anchor_from_frames(
            raw_start + 1, raw_end + 1)
        return
    assert mutation == "runtime_recomputed"
    query_frames = 5
    frame_ms = 50
    row["query_frames"] = query_frames
    row["frame_ms"] = frame_ms
    row["raw_start_s"] = raw_start * frame_ms / 1000.0
    row["raw_end_s"] = raw_end * frame_ms / 1000.0
    row["speech_start_frame"] = raw_start - query_frames
    row["speech_end_frame"] = raw_end - query_frames
    row["speech_start_s"] = (raw_start - query_frames) * frame_ms / 1000.0
    row["speech_end_s"] = (raw_end - query_frames) * frame_ms / 1000.0
    if "raw_span" in row:
        row["raw_span"] = [row["raw_start_s"], row["raw_end_s"]]
    if "speech_span" in row:
        row["speech_span"] = [row["speech_start_s"], row["speech_end_s"]]
    row["ctc_spike_anchor"] = _nvasr_anchor_from_frames(
        raw_start, raw_end, query_frames=query_frames, frame_ms=frame_ms)
    row["raw_timeline_evidence_sha256"] = \
        _nvasr_raw_timeline_evidence_sha256(row)


def _strict_attach_case(*, edge: bool = False):
    frames = ([0, 0, 0, 0, 11, 31] if edge else
              [0, 0, 0, 0, 31, 11, 32])
    surfaces = ({11: "[Breathing]", 31: "你"} if edge else
                {31: "你", 11: "[Breathing]", 32: "好"})
    timeline = extract_nvasr_candidate_timeline(
        frames, "[Breathing]你" if edge else "你[Breathing]好",
        token_surfaces=surfaces)
    words = ([
        {"word": "BREATHING", "start": 0.0, "end": 0.06},
        {"word": "ni3", "start": 0.06, "end": 0.12},
    ] if edge else [
        {"word": "ni3", "start": 0.0, "end": 0.06},
        {"word": "BREATHING", "start": 0.06, "end": 0.12},
        {"word": "hao3", "start": 0.12, "end": 0.18},
    ])
    return timeline, words


@pytest.mark.parametrize("mutation,reason", [
    ("coordinate_only", "ctc_spike_anchor_coordinate_binding_invalid"),
    ("anchor_frames_recomputed", "ctc_spike_anchor_frame_binding_invalid"),
    ("runtime_recomputed", "ctc_spike_anchor_query_frames_mismatch"),
])
def test_v3_strict_attach_rejects_displaced_or_runtime_recomputed_anchor(
        mutation, reason):
    timeline, words = _strict_attach_case()
    _tamper_anchor_binding(timeline["candidates"][0], mutation)

    errors = attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True)

    assert errors
    assert reason in errors[0]
    assert not any("candidate_id" in row for row in words)


@pytest.mark.parametrize("edge", [False, True])
def test_v3_attach_accepts_exact_raw_timeline_window_and_edge_contract(edge):
    timeline, words = _strict_attach_case(edge=edge)

    assert attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True) == []

    row = words[0] if edge else words[1]
    raw = row["raw_timeline_neighbors"]
    assert row["raw_timeline_neighbors_schema"] == \
        NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA
    assert row["raw_timeline_evidence_sha256"] == \
        _nvasr_raw_timeline_evidence_sha256(row)
    assert (raw["left"] is None) is edge
    assert raw["right"] is not None
    assert raw != row["ordered_semantic_neighbors"]


@pytest.mark.parametrize("mutation", [
    "empty_top_timeline", "required_side_null", "fabricated_identity",
    "malformed_token", "malformed_frames", "wrong_edge_presence",
])
def test_v3_attach_rejects_invalid_raw_timeline_content(mutation):
    edge = mutation == "wrong_edge_presence"
    timeline, words = _strict_attach_case(edge=edge)
    candidate = timeline["candidates"][0]
    if mutation == "empty_top_timeline":
        timeline["raw_timeline_neighbors"] = []
    elif mutation == "required_side_null":
        candidate["raw_timeline_neighbors"]["left"] = None
        candidate["raw_timeline_evidence_sha256"] = (
            _nvasr_raw_timeline_evidence_sha256(candidate))
    elif mutation == "fabricated_identity":
        candidate["raw_timeline_neighbors"]["left"]["surface"] = "fabricated"
        candidate["raw_timeline_evidence_sha256"] = (
            _nvasr_raw_timeline_evidence_sha256(candidate))
    elif mutation == "malformed_token":
        candidate["raw_timeline_neighbors"]["left"]["token_id"] = "31"
        candidate["raw_timeline_evidence_sha256"] = (
            _nvasr_raw_timeline_evidence_sha256(candidate))
    elif mutation == "malformed_frames":
        candidate["raw_timeline_neighbors"]["left"][
            "ordered_source_frame_ids"] = [4, 3]
        candidate["raw_timeline_evidence_sha256"] = (
            _nvasr_raw_timeline_evidence_sha256(candidate))
    else:
        candidate["raw_timeline_neighbors"]["left"] = {
            "surface": "fabricated-edge", "source": "ctc", "token_id": 99,
            "ordered_source_frame_ids": [3],
        }
        candidate["raw_timeline_evidence_sha256"] = (
            _nvasr_raw_timeline_evidence_sha256(candidate))

    errors = attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True)

    assert errors
    assert "raw" in errors[0]
    assert not any("candidate_id" in row for row in words)


def _valid_v3_case():
    row = _v3_row(
        raw_start_frame=177, frame_count=1, candidate_id="v3-negative-base",
        ctc_ordinal=1, key=(0, 1), forced_span=[10.41, 10.47],
        adjusted_span=[10.41, 10.47], neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "chi2",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "hao3",
             "surface_occurrence": 0},
        ])
    words, ctc = _strict_words_ctc(row)
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 9.82, "end": 9.94, "text": "chi2"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 9.94, "end": 10.35, "text": "breathing"},
        {"ordinal": 2, "ctc_lexical_ordinal": 2,
         "start": 10.35, "end": 11.0, "text": "hao3"},
    ]
    return words, ctc, source_words


@pytest.mark.parametrize("strict_flag", [
    "strict_schema_v3", "strict_schema_v2",
])
def test_strict_containment_requires_manifest_anchored_authority(strict_flag):
    words, ctc, source_words = _valid_v3_case()

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {"1": [
            {"label": "spn", "start": 9.94, "end": 10.35},
        ]}}, producer_authority=None, **{strict_flag: True})

    assert result["status"] == "rejected"
    assert result["reasons"] == [
        "nvasr_manifest_anchored_producer_authority_required"]
    assert result["changed"] == 0
    assert "_contained_tier" not in result


@pytest.mark.parametrize("strict_flag", [
    "strict_schema_v3", "strict_schema_v2",
])
def test_strict_provenance_audit_requires_manifest_anchored_authority(
        strict_flag):
    words, ctc, source_words = _valid_v3_case()

    report = _nvasr_candidate_provenance_audit(
        ctc, words, required=True, wav_duration_s=11.0,
        source_words=source_words,
        source_phone_lineage={"owners": {"1": [
            {"label": "spn", "start": 9.94, "end": 10.35},
        ]}}, producer_authority=None, **{strict_flag: True})

    assert report["status"] == "rejected"
    assert report["reasons"] == [
        "nvasr_manifest_anchored_producer_authority_required"]
    assert report["candidates"] == []


def test_non_strict_legacy_consumers_do_not_require_producer_authority():
    row = _row(raw_start_frame=11, frame_count=1,
               candidate_id="legacy-no-authority", key=(0, 1),
               forced_span=[0.39, 0.45])
    words, ctc = _containment_fixture(
        row, nvv_span=(0.39, 0.45), axis=(0.0, 1.0))

    containment = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=1.0, producer_authority=None)
    assert containment["status"] == "verified"
    assert "nvasr_manifest_anchored_producer_authority_required" not in \
        containment["reasons"]

    audit = _nvasr_candidate_provenance_audit(
        ctc, containment.get("_contained_tier", words), required=True,
        wav_duration_s=1.0, producer_authority=None)
    assert audit["status"] == "verified"
    assert "nvasr_manifest_anchored_producer_authority_required" not in \
        audit["reasons"]


@pytest.mark.parametrize("mutation,reason", [
    ("coordinate_only", "ctc_spike_anchor_coordinate_binding_invalid"),
    ("anchor_frames_recomputed", "ctc_spike_anchor_frame_binding_invalid"),
    ("runtime_recomputed", "ctc_spike_anchor_query_frames_mismatch"),
])
def test_v3_consumer_rejects_displaced_or_runtime_recomputed_anchor(
        mutation, reason):
    words, ctc, source_words = _valid_v3_case()
    _tamper_anchor_binding(ctc[1], mutation)

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "rejected"
    assert any(reason in item for item in result["reasons"])


@pytest.mark.parametrize("missing", [
    "nvasr_candidate_schema_version", "mapping_axis",
    "ordered_semantic_neighbors", "raw_timeline_neighbors",
    "candidate_source", "raw_timeline_neighbors_schema",
    "raw_timeline_index", "raw_timeline_event_count",
    "raw_timeline_evidence_sha256", "query_frames", "frame_ms",
    "ctc_raw_token_row", "raw_timeline_mapping_key",
    "ctc_spike_anchor",
])
def test_v3_missing_required_candidate_field_fails_closed(missing):
    words, ctc, source_words = _valid_v3_case()
    ctc[1].pop(missing)
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"


@pytest.mark.parametrize("missing", ["mapping_axis", "semantic_axis",
                                      "ordered_semantic_neighbors",
                                      "raw_timeline_neighbors"])
def test_v3_missing_top_level_timeline_contract_fails_closed(missing):
    timeline = extract_nvasr_candidate_timeline(
        [0, 0, 0, 0, 31, 11, 32], "你[Breathing]好",
        token_surfaces={31: "你", 11: "[Breathing]", 32: "好"})
    timeline.pop(missing)
    words = [
        {"word": "ni3", "start": 0.0, "end": 0.1},
        {"word": "BREATHING", "start": 0.1, "end": 0.2},
        {"word": "hao3", "start": 0.2, "end": 0.3},
    ]
    assert attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True)


def test_legacy_unlabeled_sidecar_is_rejected_by_production_v3_barrier():
    row = _row(raw_start_frame=177, frame_count=1, candidate_id="legacy",
               key=(0, 1), forced_span=[10.41, 10.47])
    words, ctc = _containment_fixture(row, nvv_span=(10.35, 10.41),
                                      axis=(0.0, 11.0))
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, strict_schema_v3=True,
        source_words=[], source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc))
    assert result["status"] == "rejected"
    assert "nvasr_candidate_schema_v3_required" in result["reasons"]


def test_schema_v2_candidate_is_rejected_by_production_v3_barrier():
    words, ctc, source_words = _valid_v3_case()
    ctc[1]["nvasr_candidate_schema_version"] = 2
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "rejected"
    assert "nvasr_candidate_schema_version_invalid:0" in result["reasons"]


def test_schema_v2_timeline_is_rejected_by_strict_v3_attach():
    timeline, words = _strict_attach_case()
    timeline["nvasr_candidate_schema_version"] = 2
    timeline["candidates"][0]["nvasr_candidate_schema_version"] = 2

    errors = attach_nvasr_candidate_provenance(
        words, [], timeline, strict_schema_v3=True)

    assert errors == ["candidate timeline schema-v3 metadata is invalid"]
    assert not any("candidate_id" in row for row in words)


@pytest.mark.parametrize("tamper", ["swap", "surface", "occurrence"])
def test_v3_semantic_neighbor_identity_tampering_fails_with_same_count(tamper):
    words, ctc, source_words = _valid_v3_case()
    neighbors = ctc[1]["ordered_semantic_neighbors"]
    if tamper == "swap":
        neighbors.reverse()
    elif tamper == "surface":
        neighbors[0]["surface"] = "tampered"
    else:
        neighbors[0].pop("occurrence_id")
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert any("semantic_identity" in reason for reason in result["reasons"])


def test_v3_audit_preserves_canonical_neighbor_mismatch_reason():
    words, ctc, source_words = _valid_v3_case()
    ctc[1]["ordered_semantic_neighbors"][0]["surface"] = "tampered"
    report = _nvasr_candidate_provenance_audit(
        ctc, words, required=True, wav_duration_s=11.0,
        source_words=source_words, source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert report["status"] == "rejected"
    assert "canonical_semantic_identity_order_mismatch" in report["reasons"]


@pytest.mark.parametrize("mutation", ["reversed_forced", "topology"])
def test_v3_forced_span_and_topology_never_supply_anchor(mutation):
    words, ctc, source_words = _valid_v3_case()
    if mutation == "reversed_forced":
        ctc[1]["forced_span"] = [10.47, 10.41]
    else:
        ctc[1]["mapping_selection"] = "unique_punctuation_topology_bound"
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert any("forced_span" in reason or "topology" in reason
               for reason in result["reasons"])


def test_v3_source_owner_ignores_stale_word_authority_but_legacy_keeps_conflict():
    words, ctc, source_words = _valid_v3_case()
    words._ctc_word_authority = [{
        "text": "BREATHING", "mfa_span": [10.35, 10.50],
        "ctc_lexical_ordinal": 1,
    }]
    strict = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc), strict_schema_v3=True)
    assert strict["status"] == "verified"
    assert strict["candidates"][0]["owner_branch"] == "compatible_source_mfa"

    words, ctc, _source_words = _valid_v3_case()
    words._ctc_word_authority = [{
        "text": "BREATHING", "mfa_span": [9.94, 10.35],
        "ctc_lexical_ordinal": 1,
    }]
    words._nvasr_source_mfa = [{
        "text": "BREATHING", "span": [9.94, 10.35],
        "ctc_lexical_ordinal": 1,
    }]
    words._source_mfa = deepcopy(words._nvasr_source_mfa)
    words._source_intervals = deepcopy(words._nvasr_source_mfa)
    words._nvasr_source_intervals = deepcopy(words._nvasr_source_mfa)
    words._nvasr_source_phones = [
        {"text": "a", "span": [10.35, 10.40]}]
    words._source_phones = deepcopy(words._nvasr_source_phones)
    words._source_phone_lineage = {"owners": {"0": [
        {"label": "a", "start": 10.35, "end": 10.40},
    ]}}
    ctc[1]["source_mfa_span"] = [9.94, 10.35]
    ctc[1]["mfa_span"] = [9.94, 10.35]
    source_free = _nvasr_select_owner(
        words, ctc[1], [10.35, 10.41], ctc, source_words=None,
        source_phone_lineage=None, strict_schema_v3=True)
    assert source_free[0] == "adjusted_ctc_energy_gap"

    words, ctc, source_words = _valid_v3_case()
    ctc[1]["adjusted_span"] = [10.41, 10.47]
    words._ctc_word_authority = [{
        "text": "BREATHING", "mfa_span": [10.35, 10.50],
        "ctc_lexical_ordinal": 1,
    }]
    legacy = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}}, producer_authority=None,
        strict_schema_v3=False)
    assert legacy["status"] == "rejected"
    assert "frame_support_authority_mapping_conflict:0" in legacy["reasons"]


def test_v3_provenance_api_never_falls_back_to_stale_tier_source_evidence():
    words, ctc, _source_words = _valid_v3_case()
    words.intervals[1] = Interval(10.35, 10.47, "<BREATHING>")
    words.intervals[2] = Interval(10.47, 11.0, "hao3")
    words._nvasr_source_mfa = [{
        "text": "BREATHING", "span": [9.94, 10.35],
        "ctc_lexical_ordinal": 1,
    }]
    words._source_phone_lineage = {"owners": {"0": [
        {"label": "a", "start": 10.35, "end": 10.40},
    ]}}

    report = _nvasr_candidate_provenance_audit(
        ctc, words, required=True, wav_duration_s=11.0,
        source_words=None, source_phone_lineage=None,
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert report["status"] == "verified"
    assert report["candidates"][0]["owner_branch"] == \
        "adjusted_ctc_energy_gap"
    assert report["candidates"][0]["owner_selected_span"] == \
        pytest.approx([10.35, 10.47])


def test_00452_terminal_punctuation_geometry_keeps_exact_source_nvv_owner():
    row = _v3_row(
        raw_start_frame=177, frame_count=1, candidate_id="00452-terminal",
        ctc_ordinal=1, key=(0, None), forced_span=[10.41, 10.47],
        adjusted_span=[10.35, 10.64], neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "chi2",
             "surface_occurrence": 0},
        ])
    row["start_s"], row["end_s"] = 10.41, 10.64
    words = Tier("words", 9.0, 11.0, [
        Interval(9.75, 10.35, "chi2"),
        Interval(10.35, 10.41, "…"),
        Interval(10.41, 10.64, "<BREATHING>"),
        Interval(10.64, 11.0, "！"),
    ])
    ctc = [_ordinary("chi2", 0), row]
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 9.82, "end": 9.94, "text": "chi2"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 9.94, "end": 10.35, "text": "breathing"},
    ]

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    assert result["source_core_protection"]["status"] == \
        "fallback_to_final_display_axis"
    contained = result["_contained_tier"]
    breathing = next(item for item in contained.intervals
                     if item.text == "<BREATHING>")
    assert [breathing.xmin, breathing.xmax] == pytest.approx([9.94, 10.35])
    assert all(item.xmax > item.xmin for item in contained.intervals)
    assert all(left.xmax == pytest.approx(right.xmin, abs=1e-9)
               for left, right in zip(contained.intervals,
                                      contained.intervals[1:]))
    assert [item.text for item in contained.intervals] == [
        "chi2", "…", "<BREATHING>", "！"]
    terminal_punctuation = contained.intervals[-1]
    assert [terminal_punctuation.xmin, terminal_punctuation.xmax] == \
        pytest.approx([10.35, 11.0])
    assert any(item["index"] == 3
               and item["after"] == pytest.approx([10.35, 11.0])
               for item in result["reconciliations"])


def test_00462_internal_punctuation_geometry_keeps_two_frame_source_owner():
    row = _v3_row(
        raw_start_frame=70, frame_count=2, candidate_id="00462-middle",
        ctc_ordinal=1, key=(0, 1), forced_span=[3.99, 4.05],
        adjusted_span=[3.93, 4.05], neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "ya5",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "rang4",
             "surface_occurrence": 0},
        ])
    words = Tier("words", 3.0, 5.5, [
        Interval(3.81, 3.93, "ya5"),
        Interval(3.93, 3.99, "…"),
        Interval(3.99, 4.05, "<BREATHING>"),
        Interval(4.05, 5.25, "…"),
        Interval(5.25, 5.43, "rang4"),
    ])
    ctc = [_ordinary("ya5", 0), row, _ordinary("rang4", 1)]
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 3.83, "end": 3.95, "text": "ya5"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 3.95, "end": 4.07, "text": "breathing"},
        {"ordinal": 2, "ctc_lexical_ordinal": 2,
         "start": 4.07, "end": 4.20, "text": "rang4"},
    ]

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=5.5, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    assert result["source_core_protection"]["status"] == \
        "fallback_to_final_display_axis"
    contained = result["_contained_tier"]
    assert [(item.xmin, item.xmax) for item in contained.intervals] == [
        (pytest.approx(3.81), pytest.approx(3.93)),
        (pytest.approx(3.93), pytest.approx(3.95)),
        (pytest.approx(3.95), pytest.approx(4.07)),
        (pytest.approx(4.07), pytest.approx(5.25)),
        (pytest.approx(5.25), pytest.approx(5.43)),
    ]
    assert all(item.xmax > item.xmin for item in contained.intervals)
    assert all(left.xmax <= right.xmin + 1e-9
               for left, right in zip(contained.intervals,
                                      contained.intervals[1:]))


@pytest.mark.parametrize("end,expected", [(10.32, "compatible_source_mfa"),
                                           (10.319999, None)])
def test_v3_source_mfa_endpoint_tolerance_is_inclusive_but_not_beyond(end,
                                                                       expected):
    words, ctc, source_words = _valid_v3_case()
    source_words[1]["start"] = 9.90
    source_words[1]["end"] = end
    source_words[0]["end"] = 9.90
    source_words[2]["start"] = end
    # Strict-v3 adjusted_span is evidence envelope, not just the current
    # geometry: include the persisted spike anchor at [10.35, 10.41].
    ctc[1]["adjusted_span"] = [10.35, 10.47]
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    if expected is None:
        assert result["status"] == "rejected"
    else:
        assert result["status"] == "verified"
        assert result["candidates"][0]["owner_branch"] == expected


def test_v3_lexical_and_phone_core_crossing_rejects_owner():
    words, ctc, source_words = _valid_v3_case()
    source_words[1]["start"], source_words[1]["end"] = 9.94, 10.35
    source_words[2]["start"], source_words[2]["end"] = 10.20, 10.80
    ctc[1]["adjusted_span"] = [10.35, 10.47]
    ctc[1]["start_s"], ctc[1]["end_s"] = 10.35, 10.47
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert any("lexical_or_phone_core" in reason
               for reason in result["reasons"])

    words, ctc, source_words = _valid_v3_case()
    ctc[1]["adjusted_span"] = [10.35, 10.47]
    ctc[1]["start_s"], ctc[1]["end_s"] = 10.35, 10.47
    lineage = {"owners": {"0": [{"label": "j", "start": 10.30,
                                        "end": 10.40}]}}
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage=lineage,
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert any("lexical_or_phone_core" in reason
               for reason in result["reasons"])

    words, ctc, source_words = _valid_v3_case()
    pause_lineage = {"owners": {"0": [{"label": "sp", "start": 10.0,
                                          "end": 10.10}]}}
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage=pause_lineage,
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "verified"
    assert result["candidates"][0]["owner_branch"] == "compatible_source_mfa"


def test_v3_no_compatible_owner_and_missing_source_order_reject():
    words, ctc, source_words = _valid_v3_case()
    source_words[1]["start"], source_words[1]["end"] = 2.10, 2.15
    ctc[1]["adjusted_span"] = [10.0, 10.47]
    ctc[1]["start_s"], ctc[1]["end_s"] = 10.0, 10.1
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert any("owner_selection_rejected" in reason
               for reason in result["reasons"])

    words, ctc, source_words = _valid_v3_case()
    source_words[1].pop("ctc_lexical_ordinal")
    source_words[1].pop("ordinal")
    ctc[1]["adjusted_span"] = [10.0, 10.47]
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"


def test_v3_distinct_compatible_source_owners_are_not_first_match_wins():
    words, ctc, source_words = _valid_v3_case()
    source_words.append({"ordinal": 3, "ctc_lexical_ordinal": 1,
                         "start": 9.95, "end": 10.35, "text": "spn"})
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert any("multiple_compatible_source_mfa_owners" in reason
               for reason in result["reasons"])


def test_v3_identical_source_owner_evidence_is_deduplicated():
    words, ctc, source_words = _valid_v3_case()
    source_words.append({
        "ordinal": 1, "start": 9.94, "end": 10.35,
        "text": "breathing",
    })

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {"1": [
            {"label": "spn", "start": 9.94, "end": 10.35},
        ]}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    assert result["candidates"][0]["owner_branch"] == \
        "compatible_source_mfa"


def test_v3_multi_frame_final_60ms_is_a_real_rejection():
    row = _v3_row(
        raw_start_frame=70, frame_count=2, candidate_id="multi-60",
        ctc_ordinal=1, key=(0, 1), forced_span=[3.93, 4.05],
        adjusted_span=[3.93, 4.05], neighbors=[
            {"side": "left", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "ya5",
             "surface_occurrence": 0},
            {"side": "right", "lexical_ordinal": 1,
             "occurrence_id": "nvasr-lexical-0001", "surface": "ni3",
             "surface_occurrence": 0},
        ])
    words = Tier("words", 3.0, 5.0, [
        Interval(3.83, 3.98, "ya5"),
        Interval(3.93, 3.99, "<BREATHING>"),
        Interval(4.10, 5.0, "ni3"),
    ])
    ctc = [_ordinary("ya5", 0), row, _ordinary("ni3", 1)]
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 3.83, "end": 3.98, "text": "ya5"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 3.98, "end": 4.04, "text": "breathing"},
        {"ordinal": 2, "ctc_lexical_ordinal": 2,
         "start": 4.04, "end": 5.0, "text": "ni3"},
    ]
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=5.0, source_words=source_words,
        source_phone_lineage={"owners": {"1": [
            {"label": "spn", "start": 3.98, "end": 4.04},
        ]}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "rejected"
    assert "multi_frame_anchor_final_60ms:0" in result["reasons"]


def test_v3_edge_straddling_anchor_is_evidence_but_owner_stays_in_wav():
    row = _v3_row(
        raw_start_frame=4, frame_count=1, candidate_id="edge-anchor",
        ctc_ordinal=0, key=(None, 0), forced_span=[0.0, 0.03],
        adjusted_span=[0.0, 0.03], neighbors=[
            {"side": "right", "lexical_ordinal": 0,
             "occurrence_id": "nvasr-lexical-0000", "surface": "ni3",
             "surface_occurrence": 0},
        ])
    words = Tier("words", 0.0, 1.0, [
        Interval(0.0, 0.03, "<BREATHING>"),
        Interval(0.03, 1.0, "ni3"),
    ])
    ctc = [row, _ordinary("ni3", 0)]
    source_words = [
        {"ordinal": 0, "ctc_lexical_ordinal": 0,
         "start": 0.0, "end": 0.06, "text": "breathing"},
        {"ordinal": 1, "ctc_lexical_ordinal": 1,
         "start": 0.06, "end": 1.0, "text": "ni3"},
    ]
    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=1.0, source_words=source_words,
        source_phone_lineage={"owners": {"0": [
            {"label": "spn", "start": 0.0, "end": 0.06},
        ]}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert result["status"] == "verified"
    assert result["candidates"][0]["anchor_span"] == pytest.approx([-0.03, 0.03])
    assert result["candidates"][0]["owner_selected_span"] == pytest.approx([0.0, 0.06])
    assert result["candidates"][0]["owner_selected_span"][0] >= 0.0


def test_v3_anchor_quantization_metadata_matches_round_half_up_coordinates():
    anchor = _nvasr_anchor_from_frames(177, 178)
    assert anchor["start"] == 10.35
    assert anchor["end"] == 10.41
    assert anchor["quantization"].endswith("round_half_up")


def _rehash_raw_timeline(row: dict) -> None:
    row["raw_timeline_evidence_sha256"] = (
        _nvasr_raw_timeline_evidence_sha256(row))


def test_v3_containment_accepts_exact_persisted_raw_neighbor_binding():
    words, ctc, source_words = _valid_v3_case()

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "verified"
    assert result["candidates"][0]["raw_timeline_neighbors_schema"] == \
        NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA
    assert result["candidates"][0]["raw_timeline_evidence_sha256"] == \
        ctc[1]["raw_timeline_evidence_sha256"]


@pytest.mark.parametrize("mutation", [
    "empty", "required_side_null", "tampered_identity", "malformed_fields",
    "wrong_raw_index",
])
def test_v3_containment_rejects_invalid_persisted_raw_neighbor_binding(mutation):
    words, ctc, source_words = _valid_v3_case()
    row = ctc[1]
    if mutation == "empty":
        row["raw_timeline_neighbors"] = {}
        _rehash_raw_timeline(row)
    elif mutation == "required_side_null":
        row["raw_timeline_neighbors"]["left"] = None
        _rehash_raw_timeline(row)
    elif mutation == "tampered_identity":
        row["raw_timeline_neighbors"]["left"]["surface"] = "fabricated"
    elif mutation == "malformed_fields":
        row["raw_timeline_neighbors"]["right"]["token_id"] = None
        row["raw_timeline_neighbors"]["right"][
            "ordered_source_frame_ids"] = []
        _rehash_raw_timeline(row)
    else:
        row["raw_timeline_index"] = 0
        _rehash_raw_timeline(row)

    result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert result["status"] == "rejected"
    assert any("raw_timeline_" in reason for reason in result["reasons"])


def test_v3_provenance_audit_accepts_exact_raw_neighbor_binding():
    words, ctc, source_words = _valid_v3_case()
    containment = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert containment["status"] == "verified"

    audit = _nvasr_candidate_provenance_audit(
        ctc, containment["_contained_tier"], required=True,
        wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert audit["status"] == "verified"
    assert audit["candidates"][0]["raw_timeline_neighbors_schema"] == \
        NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA


@pytest.mark.parametrize("mutation", ["tampered_identity", "null_required_side"])
def test_v3_provenance_audit_rejects_raw_neighbor_tampering(mutation):
    words, ctc, source_words = _valid_v3_case()
    containment = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert containment["status"] == "verified"
    if mutation == "tampered_identity":
        ctc[1]["raw_timeline_neighbors"]["right"]["surface"] = "fabricated"
    else:
        ctc[1]["raw_timeline_neighbors"]["right"] = None
        _rehash_raw_timeline(ctc[1])

    audit = _nvasr_candidate_provenance_audit(
        ctc, containment["_contained_tier"], required=True,
        wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)

    assert audit["status"] == "rejected"
    assert any("raw_timeline_" in reason for reason in audit["reasons"])


def test_v3_owner_report_and_compatibility_alias_are_json_serializable():
    words, ctc, source_words = _valid_v3_case()
    owner_result = _contain_nvasr_frame_support(
        words, ctc, wav_duration_s=11.0, source_words=source_words,
        source_phone_lineage={"owners": {}},
        producer_authority=_strict_authority(ctc),
        strict_schema_v3=True)
    assert owner_result["status"] == "verified"
    assert owner_result["changed"] > 0
    assert isinstance(owner_result.get("_contained_tier"), Tier)

    report = {}
    contained = _publish_nvasr_owner_report(report, owner_result)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)

    assert isinstance(contained, Tier)
    assert "_contained_tier" not in serialized
    assert report["nvasr_owner_selection"]["schema"] == \
        "nvasr-owner-selection-v2"
    alias = report["nvasr_frame_support"]
    assert alias["schema"] == "nvasr-owner-selection-v2"
    assert alias["deprecated"] is True
    assert alias["deprecated_alias"] == "nvasr_owner_selection"
    assert alias["compatibility_alias_semantics"] == \
        "owner_selection_provenance"
