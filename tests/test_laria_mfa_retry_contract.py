from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pipeline import (  # noqa: E402
    MFA_ATTEMPT_RECEIPT_SCHEMA,
    _execute_single_process_mfa_retry,
    _prepare_retained_mfa_retry_inputs,
    mfa_retry_state_machine,
    run_mfa_retry_coordinator,
)


def test_single_process_retry_command_keeps_anchor(tmp_path: Path, monkeypatch):
    import scripts.run_pipeline as pipeline

    corpus = tmp_path / "source_corpus"
    audio = tmp_path / "source_audio"
    anchors = tmp_path / "source_anchors"
    for path in (corpus, audio, anchors):
        path.mkdir()
    (corpus / "demo.lab").write_text("ni3\n", encoding="utf-8")
    (audio / "demo.wav").write_bytes(b"wav")
    (anchors / "demo.TextGrid").write_text("anchor\n", encoding="utf-8")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        output = Path(command[command.index("align") + 4])
        (output / "demo.TextGrid").write_text("aligned\n", encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "validate_strict_mfa_textgrid", lambda _path: [])
    result = _execute_single_process_mfa_retry(
        stem="demo", retry_root=tmp_path / "retry",
        source_corpus=corpus, source_audio=audio, source_anchors=anchors,
        dict_path=tmp_path / "dict", acoustic_model="model",
        mfa_python=Path(sys.executable), models_dir=tmp_path / "models",
        output_format="long_textgrid", beam=200, retry_beam=800,
        boost_silence=1.0, single_speaker=True, no_tokenization=True,
        timeout=30, attempt_ordinal=2)

    assert "--textgrid_directory" in seen["command"]
    assert result["produced"] == ["demo"]
    assert result["attempt_receipt"]["inputs"]["anchor_sha256"]


def test_retained_retry_preserves_ctc_anchor_and_exact_inputs(tmp_path: Path):
    source = tmp_path / "shard"
    for name in ("corpus", "audio", "anchors"):
        (source / name).mkdir(parents=True, exist_ok=True)
    (source / "corpus" / "demo.lab").write_text("ni3 hao3\n", encoding="utf-8")
    (source / "audio" / "demo.wav").write_bytes(b"wav")
    (source / "anchors" / "demo.TextGrid").write_text(
        'File type = "ooTextFile"\n', encoding="utf-8")

    copied = _prepare_retained_mfa_retry_inputs(
        source / "corpus", source / "audio", source / "anchors",
        tmp_path / "retry", "demo")

    assert copied["lab"].read_bytes() == (source / "corpus" / "demo.lab").read_bytes()
    assert copied["wav"].read_bytes() == (source / "audio" / "demo.wav").read_bytes()
    assert copied["anchor"].read_bytes() == (
        source / "anchors" / "demo.TextGrid").read_bytes()


def test_retained_retry_fails_closed_when_anchor_is_missing(tmp_path: Path):
    source = tmp_path / "shard"
    (source / "corpus").mkdir(parents=True)
    (source / "audio").mkdir()
    (source / "anchors").mkdir()
    (source / "corpus" / "demo.lab").write_text("ni3\n", encoding="utf-8")
    (source / "audio" / "demo.wav").write_bytes(b"wav")

    try:
        _prepare_retained_mfa_retry_inputs(
            source / "corpus", source / "audio", source / "anchors",
            tmp_path / "retry", "demo")
    except ValueError as exc:
        assert "anchor" in str(exc)
    else:
        raise AssertionError("missing CTC retry anchor was accepted")


def test_laria_recovery_is_per_stem_and_rescues_only_explicit_noalignments():
    stems = ["ok", "noalign", "timeout", "generic"]
    retry_calls: list[list[str]] = []
    rescue_calls: list[str] = []

    def retry(missing: list[str]) -> dict:
        retry_calls.append(list(missing))
        stem = missing[0]
        if stem == "noalign":
            return {"return_code": 1, "produced": [],
                    "exception_type": "NoAlignmentsError",
                    "exception": "NoAlignmentsError"}
        if stem == "timeout":
            return {"return_code": "timeout", "produced": [],
                    "invocation_outcome": "timeout",
                    "exception_type": "TimeoutExpired"}
        if stem == "generic":
            return {"return_code": 1, "produced": [],
                    "exception_type": "RuntimeError",
                    "exception": "ordinary failure"}
        return {"return_code": 0, "produced": [stem],
                "produced_output_paths": [f"aligned/{stem}.TextGrid"]}

    def rescue(stem: str) -> dict:
        rescue_calls.append(stem)
        return {"return_code": 0, "produced": [stem],
                "produced_output_paths": [f"rescue/{stem}.TextGrid"]}

    state = run_mfa_retry_coordinator(
        stems, {"return_code": 0, "produced": []}, retry,
        rescue_executor=rescue)

    assert retry_calls == [[stem] for stem in sorted(stems)]
    assert rescue_calls == ["noalign"]
    assert state["rescue_used"] is True
    assert state["merge_allowed"] is False
    assert {row["stem"] for row in state["receipts"]} == set(stems)
    assert [row["ordinal"] for row in state["receipts"] if row["stem"] == "noalign"] == [1, 2]
    assert all(row["schema"] == MFA_ATTEMPT_RECEIPT_SCHEMA
               for row in state["receipts"])
    assert all(row["isolation"] == "singleton" for row in state["receipts"])
    assert all(row["settings"] == {"beam": 20, "retry_beam": 80, "num_jobs": 1}
               for row in state["receipts"] if row["ordinal"] == 1)
    assert [row["settings"] for row in state["receipts"]
            if row["stem"] == "noalign" and row["ordinal"] == 2] == [
                {"beam": 200, "retry_beam": 800, "num_jobs": 1}]


def test_laria_no_fabricated_output_on_failed_rescue():
    calls: list[tuple[str, object]] = []

    def retry(missing: list[str]) -> dict:
        calls.append(("retry", missing))
        return {"return_code": 1, "produced": [],
                "exception": "NoAlignmentsError"}

    def rescue(stem: str) -> dict:
        calls.append(("rescue", stem))
        return {"return_code": 0, "produced": [],
                "produced_output_paths": []}

    state = run_mfa_retry_coordinator(
        ["missing"], {"return_code": 0, "produced": []}, retry,
        rescue_executor=rescue)

    assert calls == [("retry", ["missing"]), ("rescue", "missing")]
    assert state["merge_allowed"] is False
    assert state["missing_stems"] == ["missing"]
    assert state["receipts"][-1]["produced_outputs"] == []
    assert state["receipts"][-1]["final_disposition"] == "unrecovered"


def test_laria_state_machine_compatibility_accepts_complete_legacy_attempt():
    state = mfa_retry_state_machine(
        ["a"], [{"return_code": 0, "produced": ["a"]}])
    assert state["state"] == "complete"
    assert state["merge_allowed"] is True
