import pytest
import scripts.merge_ja_en_mfa as merge_module
from scripts.ja_en_schema import JAContractError
from scripts.align_japanese_mfa import handle_align

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
    languages = {row["alias"]: row["language"] for row in phones}
    return {
        "schema": "ja-en-alignment-v2", "uid": "u1", "words": [],
        "native_phones": phones,
        "locked_aliases": [{**row, "language": row.get("language", languages.get(row["alias"]))} for row in locked_aliases],
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


def test_english_and_mixed_aliases_bind_without_japanese_templates():
    en_phone = _native_phone("en0", "AH", "eu_000001", 1, token_id="tok-en", language="en")
    english = merge_module.bind_native_phone_graph(
        _alignment([en_phone], [{"alias": "eu_000001", "token_id": "tok-en", "language": "en", "pronunciation": ["AH"]}]),
        [],
    )
    assert english["native_phones"][0]["basic_phone_ids"] == []
    assert english["native_phones"][0]["mora_ids"] == []
    assert english["native_phones"][0]["transform"] == "identity"

    ja_phone = _native_phone("ja0", "t", "ju_000002", 2, token_id="tok-ja", start=100, end=200)
    mixed = merge_module.bind_native_phone_graph(
        _alignment([en_phone, ja_phone], [
            {"alias": "eu_000001", "token_id": "tok-en", "language": "en", "pronunciation": ["AH"]},
            {"alias": "ju_000002", "token_id": "tok-ja", "language": "ja", "pronunciation": ["t"]},
        ]),
        [_semantic_graph("ju_000002", "tok-ja", ["t"])],
    )
    assert [row["phone_id"] for row in mixed["native_phones"]] == ["en0", "ja0"]


def test_japanese_template_without_ordered_basic_and_mora_ids_is_rejected():
    graph = _semantic_graph("ju_000001", "tok", ["a"])
    graph["native_phone_templates"][0]["basic_phone_ids"] = []
    with pytest.raises(JAContractError, match="native_basic_mapping_ambiguous"):
        merge_module.bind_native_phone_graph(
            _alignment([_native_phone("p0", "a", "ju_000001", 1, token_id="tok")], [
                {"alias": "ju_000001", "token_id": "tok", "language": "ja", "pronunciation": ["a"]},
            ]), [graph],
        )


def test_merge_receipt_preserves_contract_error_code_and_path(tmp_path):
    stage = tmp_path / "stages" / "merge"
    result = handle_merge({"merge": {
        "uid": "u1", "ownership": [0, 100], "sample_rate": 16000,
        "expected_languages": {"ju_000001": "ja"},
        "japanese_intervals": [_native_phone("p0", "a", "ju_000001", 1, token_id="tok", end=100)],
        "english_intervals": [],
        "locked_aliases": [{"alias": "ju_000001", "token_id": "tok", "language": "ja", "pronunciation": ["a"]}],
        "semantic_graphs": [],
    }}, stage)
    assert result.status == "REJECTED"
    error = __import__("json").loads((stage / "receipt.json").read_text(encoding="utf-8"))["errors"][0]
    assert error["code"] == "native_basic_mapping_ambiguous"
    assert error["path"] == "$.semantic_graphs"


def test_strict_ledger_projects_axes_and_merges_with_bound_evidence(tmp_path):
    raw_grid = tmp_path / "raw.TextGrid"
    raw_grid.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n'
        'item [1]:\nclass = "IntervalTier"\nname = "words"\nxmin = 0\nxmax = 1\nintervals: size = 1\n'
        'intervals [1]:\nxmin = 0\nxmax = 1\ntext = "ju_000001"\n'
        'item [2]:\nclass = "IntervalTier"\nname = "phones"\nxmin = 0\nxmax = 1\nintervals: size = 1\n'
        'intervals [1]:\nxmin = 0\nxmax = 1\ntext = "a"\n', encoding="utf-8")
    receipt = {
        "schema": "audio-transform-receipt-v2", "uid": "u1",
        "source": {"path": "/source.wav", "sha256": "source", "sample_rate": 8000},
        "alignment": {"path": "/align.wav", "sha256": "align", "sample_rate": 16000},
        "train": {"path": "/train.wav", "sha256": "train", "sample_rate": 24000},
        "sample_transform": {"source_start": 10}, "alignment_transform": {"source_start": 10},
        "train_transform": {"source_start": 0},
    }
    run = {"run_id": "ja-run", "language": "ja", "unit_ids": ["unit-ja"],
           "aliases": [{"alias": "ju_000001", "token_id": "tok-ja", "unit_id": "unit-ja", "pronunciation": ["a"]}],
           "ownership_start_sample": 0, "ownership_end_sample": 16000,
           "context_start_sample": 0, "context_end_sample": 16000, "sample_rate": 16000,
           "audio_receipt": receipt}
    align_stage = tmp_path / "stages" / "align"
    aligned = handle_align({"align": {"uid": "u1", "runs": [run],
                                        "mfa_runner": lambda *_: {"status": "COMPLETE", "textgrid": str(raw_grid)}}}, align_stage)
    assert aligned.status == "COMPLETE"
    merged = handle_merge({"merge": {
        "uid": "u1", "ownership": [0, 16000], "sample_rate": 16000,
        "expected_languages": {"ju_000001": "ja"},
        "japanese_ledger": str(align_stage / "strict_ja_mfa.json"), "english_ledger": [],
        "locked_aliases": [{"alias": "ju_000001", "token_id": "tok-ja", "language": "ja", "pronunciation": ["a"]}],
        "semantic_graphs": [_semantic_graph("ju_000001", "tok-ja", ["a"])],
        "runs": [run], "audio_receipt": receipt,
    }}, tmp_path / "stages" / "merge")
    assert merged.status == "COMPLETE"
    payload = __import__("json").loads((tmp_path / "stages" / "merge" / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert payload["native_phones"][0]["token_id"] == "tok-ja"
    assert payload["raw_mfa"]["runs"][0]["run_id"] == "ja-run"
    assert payload["audio_receipt"] == receipt


def test_alignment_v3_retains_raw_mfa_boundaries_and_axes(tmp_path):
    stage = tmp_path / "stages" / "merge"
    raw = _native_phone("p0", "a", "ju_000001", 4, token_id="tok-a", start=100, end=300)
    result = handle_merge({"merge": {
        "uid": "u1", "ownership": [0, 400], "sample_rate": 16000,
        "expected_languages": {"ju_000001": "ja"},
        "japanese_intervals": [raw], "english_intervals": [],
        "locked_aliases": [{"alias": "ju_000001", "token_id": "tok-a", "language": "ja", "pronunciation": ["a"]}],
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
        "locked_aliases": [{"alias": "ju_000000", "token_id": "tok-ja", "language": "ja", "pronunciation": ["t"]},
                           {"alias": "eu_000001", "token_id": "tok-en", "language": "en", "pronunciation": ["G"]}],
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
        "locked_aliases": [{"alias": "ju_000000", "token_id": "tok-ja", "language": "ja", "pronunciation": ["t"]},
                           {"alias": "eu_000001", "token_id": "tok-en", "language": "en", "pronunciation": ["G"]}],
        "semantic_graphs": [_semantic_graph("ju_000000", "tok-ja", ["t"]), _semantic_graph("eu_000001", "tok-en", ["G"])],
        "rerun_plan": {"left": {"japanese_intervals": good_ja, "english_intervals": good_en}, "right": {"japanese_intervals": good_ja, "english_intervals": good_en}},
    }}
    result = handle_merge(config, stage)
    assert result.status == "COMPLETE"
    payload = __import__("json").loads((stage / "ja_en_alignment.json").read_text(encoding="utf-8"))
    assert len(payload["native_phones"]) == 2
