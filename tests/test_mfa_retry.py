from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pipeline import (MFA_RETRY_SCHEMA, prepare_mfa_retry_packet,
                                  reconcile_mfa_outputs, mfa_retry_state_machine,
                                  run_mfa_retry_coordinator)


def test_cli_retry_and_rescue_modes_are_reachable():
    """The mode branches must dispatch before ordinary pipeline setup."""
    import scripts.run_pipeline as pipeline

    original = {
        "load_config": pipeline.load_config,
        "validate_config": pipeline.validate_config,
        "prepare": pipeline.prepare_mfa_retry_packet,
        "rescue": pipeline.run_mfa_singleton_rescue,
        "argv": sys.argv,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="mfa-cli-dispatch-") as raw:
            root = Path(raw)
            parent = root / "parent"
            parent.mkdir()
            (parent / "frozen_filtered.json").write_text(
                json.dumps({"stems": ["s0"]}), encoding="utf-8")
            (parent / "strict_ok_manifest.json").write_text(
                json.dumps({"ok": []}), encoding="utf-8")
            stems_file = root / "stems.txt"
            stems_file.write_text("s0\n", encoding="utf-8")
            calls = []
            pipeline.load_config = lambda _path: {}
            pipeline.validate_config = lambda _cfg, _mode: []
            pipeline.prepare_mfa_retry_packet = (
                lambda *args, **kwargs: calls.append(("retry", args, kwargs)) or {})
            pipeline.run_mfa_singleton_rescue = (
                lambda *args, **kwargs: calls.append(("rescue", args, kwargs)) or {})

            sys.argv = ["run_pipeline.py", "--mode", "mfa_retry",
                        "--mfa-retry-parent-root", str(parent),
                        "--mfa-retry-stems-file", str(stems_file),
                        "--mfa-retry-workspace", str(root / "retry")]
            assert pipeline.main() == 0
            assert calls and calls[-1][0] == "retry"

            prior = root / "prior.json"
            prior.write_text("{}", encoding="utf-8")
            sys.argv = ["run_pipeline.py", "--mode", "mfa_rescue",
                        "--mfa-retry-parent-root", str(parent),
                        "--mfa-rescue-stem", "s0",
                        "--mfa-rescue-workspace", str(root / "rescue"),
                        "--mfa-rescue-prior-receipt", str(prior)]
            assert pipeline.main() == 0
            assert calls[-1][0] == "rescue"
    finally:
        pipeline.load_config = original["load_config"]
        pipeline.validate_config = original["validate_config"]
        pipeline.prepare_mfa_retry_packet = original["prepare"]
        pipeline.run_mfa_singleton_rescue = original["rescue"]
        sys.argv = original["argv"]


def _fixture_parent(tmp_path: Path, stems: list[str]) -> Path:
    parent = tmp_path / "parent"
    ws = parent / "workspace"
    (ws / "ctc_pretg").mkdir(parents=True)
    (ws / "audio_16k").mkdir(parents=True)
    (ws / "strict_ok_runs" / "r" / "output").mkdir(parents=True)
    (ws / "mfa_logs" / "r").mkdir(parents=True)
    for stem in stems:
        (ws / "ctc_pretg" / f"{stem}.lab").write_text("sil\n", encoding="utf-8")
        (ws / "ctc_pretg" / f"{stem}.txt").write_text("词", encoding="utf-8")
        (ws / "audio_16k" / f"{stem}.wav").write_bytes(b"wav")
    rows = [{"stem": s, "status": "missing_mfa_alignment"} for s in stems]
    (ws / ".mfa_alignment_axis_receipt.json").write_text(json.dumps({"alignments": rows}), encoding="utf-8")
    report = "\n".join(json.dumps({"stem": s, "filter_reasons": ["missing_mfa_alignment"]}) for s in stems)
    (ws / "strict_ok_runs" / "r" / "output" / "postprocess_report.jsonl").write_text(report, encoding="utf-8")
    (ws / "strict_ok_runs" / "r" / "output" / "strict_ok_manifest.json").write_text(json.dumps({"ok": []}), encoding="utf-8")
    (ws / "mfa_logs" / "r" / "mfa_output_manifest.json").write_text(json.dumps({"shards": [{"missing": stems}]}), encoding="utf-8")
    return parent


def test_mfa_retry_packet_is_exact_and_bound(tmp_path: Path):
    stems = ["s0", "s1"]
    parent = _fixture_parent(tmp_path, stems)
    packet = prepare_mfa_retry_packet(parent, Path("/tmp") / f"mfa-retry-test-{tmp_path.name}", stems,
                                      frozen_stems=stems, accepted_stems=[])
    assert packet["schema"] == MFA_RETRY_SCHEMA
    assert packet["stems"] == stems
    assert packet["accepted_intersection"] == 0
    assert packet["execution"]["attempted"] is False
    assert packet["options"]["num_jobs"] == 12


def test_mfa_retry_rejects_accepted_intersection(tmp_path: Path):
    parent = _fixture_parent(tmp_path, ["s0"])
    try:
        prepare_mfa_retry_packet(parent, Path("/tmp") / f"mfa-retry-test-bad-{tmp_path.name}", ["s0"],
                                 frozen_stems=["s0"], accepted_stems=["s0"])
    except ValueError as exc:
        assert "accepted" in str(exc)
    else:
        raise AssertionError("accepted intersection was not rejected")


def test_rc0_incomplete_retries_only_missing():
    result = reconcile_mfa_outputs(["a", "b", "c"], ["a", "b"], return_code=0)
    assert result["retry_missing"] and result["missing"] == ["c"]
    state = mfa_retry_state_machine(["a", "b", "c"], [
        {"return_code": 0, "produced": ["a", "b"]},
        {"return_code": 0, "produced": ["a", "b", "c"]},
    ])
    assert state["merge_allowed"] and state["state"] == "complete"


def test_persistent_generic_failure_has_no_rescue():
    state = mfa_retry_state_machine(["a"], [
        {"return_code": 1, "produced": [], "exception": "GenericError"},
    ], rescue_stem="a")
    assert state["state"] == "permanent_failure" and not state["rescue_used"]


def test_batch_10_to_9_then_singleton_rc0_to_1_completes_without_rescue():
    expected = [f"s{i}" for i in range(10)]
    calls = []
    def retry(missing):
        calls.append(list(missing))
        if len(calls) == 1:
            return {"return_code": 0, "produced": expected[:-1]}
        return {"return_code": 0, "produced": [expected[-1]]}
    state = run_mfa_retry_coordinator(
        expected, {"return_code": 0, "produced": []}, retry,
        rescue_stem=expected[-1],
        rescue_executor=lambda stem: (_ for _ in ()).throw(AssertionError("rescue forbidden")))
    assert state["merge_allowed"] and not state["rescue_used"]
    assert calls == [expected, [expected[-1]]]
    assert state["attempts"][1]["produced_individual"] == expected[:-1]
    assert state["history"][1]["produced"] == expected[:-1]
    assert state["history"][2]["produced"] == expected


def test_noalignments_isolation_allows_one_rescue():
    calls = []
    def retry(missing):
        calls.append(("retry", list(missing)))
        if len([x for x in calls if x[0] == "retry"]) == 1:
            return {"return_code": 0, "produced": ["a", "b"]}
        return {"return_code": 1, "produced": [], "exception": "NoAlignmentsError"}
    state = run_mfa_retry_coordinator(
        ["a", "b", "c"], {"return_code": 0, "produced": []}, retry,
        rescue_stem="c",
        rescue_executor=lambda stem: (calls.append(("rescue", stem)) or
                                       {"return_code": 0, "produced": [stem]}))
    assert state["merge_allowed"] and state["rescue_used"]
    assert calls == [("retry", ["a", "b", "c"]), ("retry", ["c"]), ("rescue", "c")]


def test_rescue_stem_is_derived_after_large_batch():
    expected = [f"s{i}" for i in range(10)]
    retry_calls = []
    def retry(missing):
        retry_calls.append(list(missing))
        if len(retry_calls) == 1:
            return {"return_code": 0, "produced": expected[:-1]}
        return {"return_code": 1, "produced": [], "exception": "NoAlignmentsError"}
    rescue_calls = []
    state = run_mfa_retry_coordinator(
        expected, {"return_code": 0, "produced": []}, retry,
        rescue_executor=lambda stem: (rescue_calls.append(stem) or
                                       {"return_code": 0, "produced": [stem]}))
    assert state["merge_allowed"] and rescue_calls == [expected[-1]]
    assert retry_calls == [expected, [expected[-1]]]


def test_rc0_singleton_missing_generic_extra_invalid_permanently_fail():
    expected = ["a", "b"]
    cases = [
        {"return_code": 0, "produced": []},
        {"return_code": 1, "produced": [], "exception": "GenericError"},
        {"return_code": 0, "produced": ["c"]},
        {"return_code": 0, "produced": [], "invalid": ["b"]},
    ]
    for isolation in cases:
        calls = []
        def retry(missing, isolation=isolation):
            calls.append(list(missing))
            return {"return_code": 0, "produced": ["a"]} if len(calls) == 1 else isolation
        state = run_mfa_retry_coordinator(
            expected, {"return_code": 0, "produced": []}, retry,
            rescue_stem="b",
            rescue_executor=lambda stem: (_ for _ in ()).throw(AssertionError("rescue forbidden")))
        assert state["state"] == "permanent_failure" and not state["merge_allowed"]
        assert calls == [expected, ["b"]]


def test_rescue_cap_is_one_attempt():
    calls = []
    def retry(missing):
        calls.append(("retry", list(missing)))
        return {"return_code": 0, "produced": ["a"]} if len(calls) == 1 else {
            "return_code": 1, "produced": [], "exception": "NoAlignmentsError"}
    state = run_mfa_retry_coordinator(
        ["a", "b"], {"return_code": 0, "produced": []}, retry,
        rescue_stem="b",
        rescue_executor=lambda stem: (calls.append(("rescue", stem)) or {
            "return_code": 1, "produced": [], "exception": "NoAlignmentsError"}))
    assert state["state"] == "permanent_failure" and state["rescue_used"]
    assert calls.count(("rescue", "b")) == 1


def test_multi_missing_rescue_is_not_singleton():
    state = mfa_retry_state_machine(["a", "b"], [
        {"return_code": 0, "produced": ["a"]},
        {"return_code": 0, "produced": []},
        {"return_code": 0, "produced": []},
    ], rescue_stem="a")
    assert state["state"] == "permanent_failure"


def test_mocked_shard_integration_retains_incomplete_and_merges_exact_set():
    """Exercise the production reconciliation contract before partial admission."""
    first = reconcile_mfa_outputs(["a", "b", "c"], ["a", "b"], return_code=0)
    assert first["retry_missing"] and first["missing"] == ["c"]
    retry = reconcile_mfa_outputs(["c"], ["c"], return_code=0)
    assert retry["complete"]
    final = set(first["produced"]) | set(retry["produced"])
    assert final == {"a", "b", "c"}
    calls = []
    state = run_mfa_retry_coordinator(["a", "b", "c"],
        {"return_code": 0, "produced": ["a", "b"]},
        lambda missing: (calls.append(missing) or {"return_code": 0, "produced": ["c"]}))
    assert calls == [["c"]] and state["merge_allowed"]


def self_test() -> None:
    test_cli_retry_and_rescue_modes_are_reachable()
    test_rc0_incomplete_retries_only_missing()
    test_persistent_generic_failure_has_no_rescue()
    test_batch_10_to_9_then_singleton_rc0_to_1_completes_without_rescue()
    test_noalignments_isolation_allows_one_rescue()
    test_rc0_singleton_missing_generic_extra_invalid_permanently_fail()
    test_rescue_cap_is_one_attempt()
    test_multi_missing_rescue_is_not_singleton()
    test_mocked_shard_integration_retains_incomplete_and_merges_exact_set()


if __name__ == "__main__":
    self_test()
    print("mfa retry self-test passed")
