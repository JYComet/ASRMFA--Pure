import pytest
import scripts.merge_ja_en_mfa as merge_module
from scripts.ja_en_schema import JAContractError

from scripts.merge_ja_en_mfa import (
    MergeRejected,
    merge_global_sample_axis,
    merge_with_seam_retries,
    retry_seam_both_sides,
    handle_merge,
)


def _native_phone(phone_id, label, alias, raw_interval_id, *, token_id=None, start=0, end=100, language="ja"):
    token = token_id or alias
    return {
        "phone_id": phone_id, "uid": "u1", "unit_id": token, "token_id": token,
        "alias": alias, "language": language, "native_phone": label, "phone": f"{language}:{label}",
        "raw_interval_id": raw_interval_id, "raw_interval_index": raw_interval_id,
        "run_id": "ja-run", "start_sample": start, "end_sample": end,
        "source_axis": {"name": "source", "sample_rate": 16000},
        "alignment_axis": {"name": "alignment", "sample_rate": 16000},
        "training_axis": {"name": "training", "sample_rate": 16000},
    }


def _semantic_graph(alias, token_id, labels):
    return {
        "schema": "ja-semantic-phone-graph-v2", "uid": "u1", "token_id": token_id,
        # The template aliases are intentionally absent: the locked alias
        # receipt, rather than graph position, supplies that evidence.
        "native_phone_templates": [
            {"native_phone_id": f"np-{index}", "native_phone": label,
             "token_id": token_id, "basic_phone_ids": [f"bp-{index}"],
             "mora_ids": [f"m-{index}"], "transform": "identity"}
            for index, label in enumerate(labels)
        ],
    }


def _alignment(phones, locked_aliases):
    return {
        "schema": "ja-en-alignment-v2", "uid": "u1", "words": [],
        "native_phones": phones, "locked_aliases": locked_aliases,
    }


def test_repeated_native_labels_join_by_composite_identity_and_occurrence_order():
    phones = [
        _native_phone("p0", "a", "ju_000001", 4, token_id="tok-a", start=0, end=100),
        _native_phone("p1", "a", "ju_000001", 5, token_id="tok-a", start=100, end=200),
        _native_phone("p2", "a", "ju_000002", 1, token_id="tok-b", start=200, end=300),
    ]
    bound = merge_module.bind_native_phone_graph(
        _alignment(phones, [
            {"alias": "ju_000001", "token_id": "tok-a", "pronunciation": ["a", "a"]},
            {"alias": "ju_000002", "token_id": "tok-b", "pronunciation": ["a"]},
        ]),
        [_semantic_graph("ju_000001", "tok-a", ["a", "a"]), _semantic_graph("ju_000002", "tok-b", ["a"])],
    )
    assert [row["phone_id"] for row in bound["native_phones"]] == ["p0", "p1", "p2"]
    assert [row["raw_interval_id"] for row in bound["native_phones"]] == [4, 5, 1]
    assert bound["native_phones"][2]["alias"] == "ju_000002"
    assert bound["native_phones"][1]["basic_phone_ids"] == ["bp-1"]


def test_alias_mismatch_never_falls_back_to_all_actual_phones():
    with pytest.raises(JAContractError, match="native_basic_mapping_ambiguous"):
        merge_module.bind_native_phone_graph(
            _alignment([_native_phone("p0", "a", "ju_wrong", 1, token_id="tok")], [
                {"alias": "ju_expected", "token_id": "tok", "pronunciation": ["a"]},
            ]),
            [_semantic_graph("ju_expected", "tok", ["a"])],
        )


def test_alignment_v3_retains_raw_mfa_boundaries_and_axes(tmp_path):
    stage = tmp_path / "stages" / "merge"
    raw = _native_phone("p0", "a", "ju_000001", 4, token_id="tok-a", start=100, end=300)
    result = handle_merge({"merge": {
        "uid": "u1", "ownership": [0, 400], "sample_rate": 16000,
        "expected_languages": {"ju_000001": "ja"},
        "japanese_intervals": [raw], "english_intervals": [],
        "locked_aliases": [{"alias": "ju_000001", "token_id": "tok-a", "pronunciation": ["a"]}],
        "semantic_graphs": [_semantic_graph("ju_000001", "tok-a", ["a"])],
    }}, stage)
    assert result.status == "COMPLETE"
    row = __import__("json").loads((stage / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert row["schema"] == "ja-en-alignment-v3"
    for phone, source in zip(row["native_phones"], row["raw_mfa"]["phones"], strict=True):
        assert (phone["start_sample"], phone["end_sample"]) == (source["start_sample"], source["end_sample"])
        assert phone["boundary_source"] == "mfa_native_interval"
        assert phone["source_axis"] == source["source_axis"]
        assert phone["alignment_axis"] == source["alignment_axis"]
        assert phone["training_axis"] == source["training_axis"]
        assert phone["run_id"] == source["run_id"]


def test_merge_rejects_clipping_overlap_and_scale_edits():
    ja = [{"raw_interval_id": 1, "alias": "ju_000000", "start_sample": 0, "end_sample": 1000, "phone": "ja:t"}]
    en = [{"raw_interval_id": 1, "alias": "eu_000001", "start_sample": 1100, "end_sample": 2000, "phone": "en:G"}]
    merged = merge_global_sample_axis(
        ja, en, ownership=(0, 2000), sample_rate=16000,
        expected_languages={"ju_000000": "ja", "eu_000001": "en"},
    )
    assert [row["start_sample"] for row in merged] == [0, 1100]
    assert all("scale" not in row and "clipped" not in row for row in merged)

    with pytest.raises(MergeRejected, match="overlap"):
        merge_global_sample_axis(
            ja, [{**en[0], "start_sample": 900}], ownership=(0, 2000), sample_rate=16000,
            expected_languages={"ju_000000": "ja", "eu_000001": "en"},
        )


def test_failed_seam_reruns_both_sides_and_rejects_after_bound():
    calls = []

    def rerun(side, padding):
        calls.append((side, padding))
        return {"side": side, "padding": padding, "ok": False}

    result = retry_seam_both_sides(
        seam_id="s0", initial_padding_samples=100,
        max_retries=2, rerun=rerun, validate=lambda _: False,
    )
    assert result["status"] == "REJECTED"
    assert calls == [("left", 200), ("right", 200), ("left", 400), ("right", 400)]


def test_successful_merge_returns_accepted_intervals_and_checks_run_ownership():
    ja = [{"raw_interval_id": 1, "alias": "ju_000000", "start_sample": 100, "end_sample": 500, "phone": "ja:t", "run_id": "ja0"}]
    en = [{"raw_interval_id": 2, "alias": "eu_000001", "start_sample": 600, "end_sample": 900, "phone": "en:G", "run_id": "en0"}]
    result = merge_with_seam_retries(
        ja, en, ownership=(0, 1000), sample_rate=16000,
        expected_languages={"ju_000000": "ja", "eu_000001": "en"},
        seam_id="s0", initial_padding_samples=50, max_retries=1,
        rerun=lambda side, padding: (ja, en),
        run_ownership={"ja0": (0, 550), "en0": (550, 1000)},
        reject_edge_touch=False,
    )
    assert result["status"] == "VERIFIED"
    assert len(result["intervals"]) == 2

    with pytest.raises(MergeRejected, match="run ownership"):
        merge_global_sample_axis(
            [{**ja[0], "end_sample": 700}], en, ownership=(0, 1000), sample_rate=16000,
            expected_languages={"ju_000000": "ja", "eu_000001": "en"},
            run_ownership={"ja0": (0, 550), "en0": (550, 1000)},
        )


def test_retry_path_returns_deduplicated_accepted_intervals_after_both_reruns():
    bad_ja = [{"raw_interval_id": 1, "alias": "ju_000000", "start_sample": 0, "end_sample": 500, "phone": "ja:t"}]
    bad_en = [{"raw_interval_id": 2, "alias": "eu_000001", "start_sample": 600, "end_sample": 1000, "phone": "en:G"}]
    good_ja = [{**bad_ja[0], "start_sample": 100, "end_sample": 500}]
    good_en = [{**bad_en[0], "start_sample": 600, "end_sample": 900}]
    calls = []
    def rerun(side, padding):
        calls.append(side)
        return good_ja, good_en
    result = merge_with_seam_retries(
        bad_ja, bad_en, ownership=(0, 1000), sample_rate=16000,
        expected_languages={"ju_000000": "ja", "eu_000001": "en"},
        seam_id="s0", initial_padding_samples=100, max_retries=1,
        rerun=rerun, reject_edge_touch=True,
    )
    assert result["status"] == "VERIFIED"
    assert len(result["intervals"]) == 2
    assert calls == ["left", "right"]


def test_merge_stage_consumes_prepared_ledgers_and_writes_alignment(tmp_path):
    stage = tmp_path / "stages" / "merge"
    ja = _native_phone("p0", "t", "ju_000000", 1, token_id="tok-ja", start=0, end=1000)
    en = _native_phone("p1", "G", "eu_000001", 2, token_id="tok-en", start=1100, end=2000, language="en")
    result = handle_merge({"merge": {
        "uid": "u1", "ownership": [0, 2000], "sample_rate": 16000,
        "expected_languages": {"ju_000000": "ja", "eu_000001": "en"},
        "japanese_intervals": [ja], "english_intervals": [en],
        "locked_aliases": [{"alias": "ju_000000", "token_id": "tok-ja", "pronunciation": ["t"]},
                           {"alias": "eu_000001", "token_id": "tok-en", "pronunciation": ["G"]}],
        "semantic_graphs": [_semantic_graph("ju_000000", "tok-ja", ["t"]), _semantic_graph("eu_000001", "tok-en", ["G"])],
    }}, stage)
    assert result.status == "COMPLETE"
    payload = __import__("json").loads((stage / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "ja-en-alignment-v3"
    assert len(payload["native_phones"]) == 2


def test_merge_stage_serialized_retry_plan_reruns_both_sides(tmp_path):
    stage = tmp_path / "stages" / "merge"
    bad_ja = [_native_phone("p0", "t", "ju_000000", 1, token_id="tok-ja", start=0, end=500)]
    bad_en = [_native_phone("p1", "G", "eu_000001", 2, token_id="tok-en", start=600, end=1000, language="en")]
    good_ja = [{**bad_ja[0], "start_sample": 100, "end_sample": 500}]
    good_en = [{**bad_en[0], "start_sample": 600, "end_sample": 900}]
    config = {"merge": {
        "uid": "u1", "ownership": [0, 1000], "sample_rate": 16000,
        "expected_languages": {"ju_000000": "ja", "eu_000001": "en"}, "reject_edge_touch": True,
        "initial_padding_samples": 100, "max_retries": 1,
        "japanese_intervals": bad_ja, "english_intervals": bad_en,
        "locked_aliases": [{"alias": "ju_000000", "token_id": "tok-ja", "pronunciation": ["t"]},
                           {"alias": "eu_000001", "token_id": "tok-en", "pronunciation": ["G"]}],
        "semantic_graphs": [_semantic_graph("ju_000000", "tok-ja", ["t"]), _semantic_graph("eu_000001", "tok-en", ["G"])],
        "rerun_plan": {"left": {"japanese_intervals": good_ja, "english_intervals": good_en}, "right": {"japanese_intervals": good_ja, "english_intervals": good_en}},
    }}
    result = handle_merge(config, stage)
    assert result.status == "COMPLETE"
    payload = __import__("json").loads((stage / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert len(payload["native_phones"]) == 2
