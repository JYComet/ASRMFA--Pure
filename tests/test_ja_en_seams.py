import pytest

from scripts.merge_ja_en_mfa import (
    MergeRejected,
    merge_global_sample_axis,
    merge_with_seam_retries,
    retry_seam_both_sides,
    handle_merge,
)


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
    result = handle_merge({"merge": {
        "uid": "u1", "ownership": [0, 2000], "sample_rate": 16000,
        "expected_languages": {"ju_000000": "ja", "eu_000001": "en"},
        "japanese_intervals": [{"raw_interval_id": 1, "alias": "ju_000000", "start_sample": 0, "end_sample": 1000, "phone": "ja:t"}],
        "english_intervals": [{"raw_interval_id": 2, "alias": "eu_000001", "start_sample": 1100, "end_sample": 2000, "phone": "en:G"}],
    }}, stage)
    assert result.status == "COMPLETE"
    payload = __import__("json").loads((stage / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "ja-en-alignment-v2"
    assert len(payload["phones"]) == 2


def test_merge_stage_serialized_retry_plan_reruns_both_sides(tmp_path):
    stage = tmp_path / "stages" / "merge"
    bad_ja = [{"raw_interval_id": 1, "alias": "ju_000000", "start_sample": 0, "end_sample": 500, "phone": "ja:t"}]
    bad_en = [{"raw_interval_id": 2, "alias": "eu_000001", "start_sample": 600, "end_sample": 1000, "phone": "en:G"}]
    good_ja = [{**bad_ja[0], "start_sample": 100, "end_sample": 500}]
    good_en = [{**bad_en[0], "start_sample": 600, "end_sample": 900}]
    config = {"merge": {
        "uid": "u1", "ownership": [0, 1000], "sample_rate": 16000,
        "expected_languages": {"ju_000000": "ja", "eu_000001": "en"}, "reject_edge_touch": True,
        "initial_padding_samples": 100, "max_retries": 1,
        "japanese_intervals": bad_ja, "english_intervals": bad_en,
        "rerun_plan": {"left": {"japanese_intervals": good_ja, "english_intervals": good_en}, "right": {"japanese_intervals": good_ja, "english_intervals": good_en}},
    }}
    result = handle_merge(config, stage)
    assert result.status == "COMPLETE"
    payload = __import__("json").loads((stage / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert len(payload["phones"]) == 2
