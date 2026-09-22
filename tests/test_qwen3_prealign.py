from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_prealign import (  # noqa: E402
    Qwen3PrealignError,
    _lexical_plan,
    _identity_payload,
    _write_textgrid,
    run_qwen3_hf,
)
from run_pipeline import validate_config  # noqa: E402


class FakeBackend:
    def __init__(self, transcripts: dict[str, str]):
        self.transcripts = transcripts
        self.transcribe_calls: list[str] = []
        self.align_calls: list[tuple[str, str]] = []
        self.last_language = "Chinese"

    def transcribe(self, audio: Path, *, language=None, context="") -> str:
        self.transcribe_calls.append(audio.stem)
        return self.transcripts[audio.stem]

    def align(self, audio: Path, text: str, *, language="Chinese") -> list[dict]:
        self.align_calls.append((audio.stem, text))
        units = []
        word = []
        for char in text:
            if char.isalnum() and ord(char) < 128:
                word.append(char)
            else:
                if word:
                    units.append("".join(word))
                    word = []
                if not char.isspace():
                    units.append(char)
        if word:
            units.append("".join(word))
        return [{"unit": unit, "start_s": i * 0.2, "end_s": i * 0.2 + 0.1}
                for i, unit in enumerate(units)]

    def close(self):
        pass


def test_qwen_lexical_plan_splits_uppercase_ascii_into_letter_units():
    plan = _lexical_plan("CPU a")

    assert [(item["unit"], item["authority"].surface_text)
            for item in plan] == [("C", "C"), ("P", "P"), ("U", "U"), ("a", "a")]


def test_qwen_identity_binds_english_units_policy_and_hash():
    identity = _identity_payload(
        settings=SimpleNamespace(model_path=Path("asr"),
                                 forced_aligner_model_path=Path("aligner"),
                                 device="cpu", dtype="float32", language="auto",
                                 max_new_tokens=1, batch_size=1, context="",
                                 forced_aligner_device=None),
        runtime={}, asr_digest="a", aligner_digest="b", input_digest="c",
        reference_digest="d", output_digest="e", stems=["demo"])

    assert identity["english_units_policy_id"] == "uppercase-ascii-letter-names-v1"
    assert len(identity["english_units_policy_sha256"]) == 64


def test_producer_keeps_punctuation_as_pinyin_phrase_boundary(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "sample.wav", seconds=2.0)
    output = tmp_path / "out"
    backend = FakeBackend({"sample": "完成。都认真！"})
    assert run_qwen3_hf(_cfg(tmp_path, data, output, reference_mode="fallback"),
                       tmp_path, backend=backend) == 0
    rows = [json.loads(line) for line in (output / "sample_tokens.jsonl").read_text().splitlines()]
    assert [row["word"] for row in rows] == ["wan2", "cheng2", "dou1", "ren4", "zhen1"]


def _wav(path: Path, seconds: float = 1.0) -> None:
    frames = int(16000 * seconds)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(struct.pack("<h", 0) * frames)


def test_qwen_textgrid_does_not_serialize_submicrosecond_trailing_interval(
        tmp_path: Path):
    """A float gap that rounds away must not become a zero-length interval."""
    path = tmp_path / "sample.TextGrid"
    duration = 3.006281179138322
    _write_textgrid(path, duration, [{
        "word": "zhe3", "start_s": 2.84, "end_s": 3.006281,
    }])

    text = path.read_text(encoding="utf-8")
    assert 'xmin = 3.006281\n            xmax = 3.006281' not in text
    assert "intervals: size = 2" in text


def _cfg(root: Path, data: Path, output: Path, *, reference_mode="auto") -> dict:
    asr = root / "asr"
    aligner = root / "aligner"
    asr.mkdir(exist_ok=True)
    aligner.mkdir(exist_ok=True)
    (asr / "config.json").write_text("asr", encoding="utf-8")
    (aligner / "config.json").write_text("aligner", encoding="utf-8")
    return {
        "data_dir": str(data),
        "audio_dir": str(data),
        "output_dir": str(output),
        "mfa_dict": "",
        "reference_mode": reference_mode,
        "ctc_prealign": {
            "model_path": str(asr),
            "forced_aligner_model_path": str(aligner),
            "language": "Chinese",
            "runtime_capabilities": {"backend": "fake", "transformers_version": "5.13.0"},
            "nvv_enabled": False,
        },
    }


def test_reference_authority_skips_asr_and_writes_six_artifacts(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "ref.wav")
    (data / "ref.txt").write_text("你好", encoding="utf-8")
    output = tmp_path / "out"
    backend = FakeBackend({"ref": "ignored ASR"})

    assert run_qwen3_hf(_cfg(tmp_path, data, output), tmp_path, backend=backend) == 0
    assert backend.transcribe_calls == []
    assert (output / "ref_ref.txt").is_file()
    for suffix in (".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
                   "_text_cn.txt", "_text_raw.txt"):
        assert (output / f"ref{suffix}").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["content_authority"] == "reference"
    assert manifest[0]["text_asr"] == ""
    assert json.loads((output / ".qwen3_hf_identity.json").read_text(encoding="utf-8"))["models"]


def test_no_reference_uses_asr_and_honors_audio_selector(tmp_path: Path):
    data = tmp_path / "data"
    audio = tmp_path / "audio"
    data.mkdir(); audio.mkdir()
    _wav(audio / "a.wav"); _wav(audio / "b.wav")
    stems = tmp_path / "stems.txt"
    stems.write_text("b\n", encoding="utf-8")
    output = tmp_path / "out"
    cfg = _cfg(tmp_path, data, output, reference_mode="auto")
    cfg["audio_dir"] = str(audio)
    cfg["ctc_prealign"].update({"audio_dir": str(audio), "stems_file": str(stems)})
    backend = FakeBackend({"a": "unused", "b": "hello world"})

    assert run_qwen3_hf(cfg, tmp_path, backend=backend) == 0
    assert backend.transcribe_calls == ["b"]
    assert (output / "b_tokens.jsonl").is_file()
    assert not (output / "b_ref.txt").exists()
    assert not (output / "a_tokens.jsonl").exists()

    # The content and model identity lets a resume reuse the complete bundle
    # without loading either model or invoking the fake backend again.
    second = FakeBackend({"b": "must not run"})
    assert run_qwen3_hf(cfg, tmp_path, backend=second) == 0
    assert second.transcribe_calls == []


def test_prealign_preserves_forced_aligner_raw_zero_point_and_adjustment(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    (data / "a.txt").write_text("家", encoding="utf-8")

    class AdjustingBackend(FakeBackend):
        def align(self, audio: Path, text: str, *, language="Chinese") -> list[dict]:
            return [{
                "unit": "家", "start_s": .56, "end_s": .64,
                "raw_start_s": .64, "raw_end_s": .64,
                "timing_adjustment": {
                    "reason": "zero_duration_borrow_left", "quantum_s": .08,
                    "raw_start_s": .64, "raw_end_s": .64,
                },
            }]

    output = tmp_path / "out"
    assert run_qwen3_hf(
        _cfg(tmp_path, data, output), tmp_path,
        backend=AdjustingBackend({}),
    ) == 0
    row = json.loads((output / "a_tokens.jsonl").read_text(encoding="utf-8"))
    assert row["start_s"] == .56
    assert row["raw_start_s"] == row["raw_end_s"] == .64
    assert row["timing_adjustment"]["reason"] == "zero_duration_borrow_left"


def test_rejects_nvv_and_long_audio_before_backend(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "long.wav", seconds=300.1)
    cfg = _cfg(tmp_path, data, tmp_path / "out")
    backend = FakeBackend({"long": "hello"})
    with pytest.raises(Qwen3PrealignError, match="at most 300"):
        run_qwen3_hf(cfg, tmp_path, backend=backend)
    assert backend.transcribe_calls == []

    _wav(data / "short.wav")
    cfg["ctc_prealign"]["nvv_enabled"] = True
    with pytest.raises(Qwen3PrealignError, match="NVV"):
        run_qwen3_hf(cfg, tmp_path, backend=backend)


def test_qwen_provider_validation_is_conditional_on_legacy_nvv_rules():
    base = {"reference_mode": "fallback", "ctc_prealign": {
        "enabled": True, "provider": "qwen3_hf", "nvv_enabled": False,
        "reference_nvv_enabled": False,
    }}
    assert not any("requires nvv_enabled" in error
                   for error in validate_config(base, "full"))
    bad = {**base, "ctc_prealign": {**base["ctc_prealign"], "nvv_enabled": True}}
    assert any("requires nvv_enabled=false" in error
               for error in validate_config(bad, "full"))
    bad_reference = {**base, "ctc_prealign": {
        **base["ctc_prealign"], "reference_nvv_enabled": True}}
    assert any("forbids reference NVV" in error
               for error in validate_config(bad_reference, "full"))


@pytest.mark.parametrize("mutation", ["model", "text", "mode", "reference_sidecar", "manifest", "foreign", "receipt"])
def test_changed_identity_or_artifact_fails_before_inference(tmp_path, mutation):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    (data / "a.txt").write_text("你好")
    output = tmp_path / "out"
    cfg = _cfg(tmp_path, data, output)
    assert run_qwen3_hf(cfg, tmp_path, backend=FakeBackend({})) == 0
    if mutation == "model":
        (tmp_path / "aligner" / "config.json").write_text("changed")
    elif mutation == "text":
        (data / "a.txt").write_text("再见")
    elif mutation == "mode":
        cfg["reference_mode"] = "fallback"
    elif mutation == "reference_sidecar":
        (output / "a_ref.txt").write_text("wrong")
    elif mutation == "manifest":
        (output / "manifest.json").write_text("[]")
    elif mutation == "foreign":
        (output / "foreign.lab").write_text("wrong")
    else:
        path = output / ".ctc_run_receipt.json"
        receipt = json.loads(path.read_text())
        receipt["model"]["tree_digest"] = "0" * 64
        path.write_text(json.dumps(receipt))
    fake = FakeBackend({"a": "你好"})
    with pytest.raises(Qwen3PrealignError, match="identity|tamper|changed"):
        run_qwen3_hf(cfg, tmp_path, backend=fake)
    assert fake.transcribe_calls == fake.align_calls == []


def test_reference_english_authority_metadata_and_raw_text(tmp_path):
    from pipeline_utils import validate_ctc_authority_bundle
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav", seconds=2)
    reference = "你好 well-known target1!"
    (data / "a.txt").write_text(reference)
    output = tmp_path / "out"
    assert run_qwen3_hf(_cfg(tmp_path, data, output), tmp_path, backend=FakeBackend({})) == 0
    assert (output / "a_text_raw.txt").read_text().strip() == reference
    normalized = (output / "a_text_cn.txt").read_text().strip()
    assert normalized == "你好 well-known target1！"
    assert validate_ctc_authority_bundle(output, "a", normalized, require_processed=False) == []
    rows = [json.loads(line) for line in (output / "a_tokens.jsonl").read_text().splitlines()]
    assert all(row["provider"] == "qwen3_hf" for row in rows)
    assert all(row["lexical_timing_source"] == "qwen3_forced_aligner_hf" for row in rows)
    assert not any("raw_start_frame" in row for row in rows)


@pytest.mark.parametrize("reference_mode", ["authority", "fallback"])
def test_repeated_ellipsis_normalizes_before_plan_and_preserves_raw_provenance(
        tmp_path, reference_mode, monkeypatch):
    import qwen3_prealign

    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    raw = "你好……"
    if reference_mode == "authority":
        (data / "a.txt").write_text(raw, encoding="utf-8")
        transcripts = {"a": "ignored ASR"}
    else:
        transcripts = {"a": raw}
    output = tmp_path / "out"
    seen_plan_inputs = []
    original_plan = qwen3_prealign._lexical_plan

    def capture_plan(text):
        seen_plan_inputs.append(text)
        return original_plan(text)

    monkeypatch.setattr(qwen3_prealign, "_lexical_plan", capture_plan)
    backend = FakeBackend(transcripts)
    assert run_qwen3_hf(
        _cfg(tmp_path, data, output, reference_mode=reference_mode),
        tmp_path,
        backend=backend,
    ) == 0

    assert seen_plan_inputs and all("……" not in text for text in seen_plan_inputs)
    assert (output / "a_text_raw.txt").read_text(encoding="utf-8").strip() == raw
    assert (output / "a_text_cn.txt").read_text(encoding="utf-8").strip() == "你好…"
    assert (output / "a_ref.txt").exists() is (reference_mode == "authority")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))[0]
    assert manifest["text_original"] == raw
    assert manifest["text_normalized"] == "你好…"
    assert manifest["text_asr"] == ("" if reference_mode == "authority" else raw)
    token_rows = [json.loads(line) for line in
                  (output / "a_tokens.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_digest = __import__("hashlib").sha256("你好…".encode()).hexdigest()
    assert all(row["normalized_text_sha256"] == expected_digest for row in token_rows)


@pytest.mark.parametrize("reference_mode", ["authority", "fallback"])
def test_input_punctuation_normalizes_before_qwen_plan_and_keeps_raw_source(
        tmp_path, reference_mode):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    raw = "「你」~（好）！"
    if reference_mode == "authority":
        (data / "a.txt").write_text(raw, encoding="utf-8")
        transcripts = {"a": "ignored"}
    else:
        transcripts = {"a": raw}
    output = tmp_path / "out"

    class InputBackend(FakeBackend):
        def align(self, audio: Path, text: str, *, language="Chinese") -> list[dict]:
            assert text == "你 好"
            return [dict(unit=unit, start_s=index * .2, end_s=index * .2 + .1)
                    for index, unit in enumerate(("你", "好"))]

    backend = InputBackend(transcripts)
    assert run_qwen3_hf(
        _cfg(tmp_path, data, output, reference_mode=reference_mode),
        tmp_path, backend=backend) == 0
    assert (output / "a_text_raw.txt").read_text(encoding="utf-8").strip() == raw
    assert (output / "a_text_cn.txt").read_text(encoding="utf-8").strip() == "你…好！"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))[0]
    assert manifest["text_original"] == raw
    assert manifest["text_normalized"] == "你…好！"
    assert manifest["text_asr"] == ("" if reference_mode == "authority" else raw)


def test_pure_ellipsis_reference_still_has_no_lexical_units(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    (data / "a.txt").write_text("……", encoding="utf-8")
    backend = FakeBackend({})
    with pytest.raises(Qwen3PrealignError, match="no spoken lexical units"):
        run_qwen3_hf(_cfg(tmp_path, data, tmp_path / "out"), tmp_path, backend=backend)
    assert backend.align_calls == []


@pytest.mark.parametrize("reference_mode", ["authority", "fallback"])
def test_normalized_transcript_reaches_real_postprocess(tmp_path, reference_mode, monkeypatch):
    from scripts import postprocess_textgrids as post
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav", seconds=1.4)
    (data / "a.txt").write_text("你好，啊！")
    output = tmp_path / "prealign"

    class GapBackend(FakeBackend):
        def align(self, audio, text, *, language="Chinese"):
            return [dict(unit=u, start_s=s, end_s=e) for u, s, e in [
                ("你", .1, .3), ("好", .4, .6), ("啊", 1.1, 1.3)]]

    fake = GapBackend({"a": "你好，啊！"})
    assert run_qwen3_hf(_cfg(tmp_path, data, output, reference_mode=reference_mode), tmp_path, backend=fake) == 0
    assert fake.transcribe_calls == ([] if reference_mode == "authority" else ["a"])
    assert (output / "a_text_cn.txt").read_text().strip() == "你好，啊！"
    from scripts.pipeline_utils import write_ctc_raw_manifest, materialize_ctc_work
    write_ctc_raw_manifest(output)
    work = tmp_path / "work"
    materialize_ctc_work(output, work)
    monkeypatch.setenv("CTC_RAW_MANIFEST", str(output / ".ctc_raw_manifest.json"))
    monkeypatch.setenv("CTC_WORK_RECEIPT", str(work / ".ctc_work_receipt.json"))
    output = work
    tokens = [json.loads(line) for line in (output / "a_tokens.jsonl").read_text().splitlines()]
    assert tokens[0]["end_s"] == .4
    # Synthetic MFA output; run the real postprocessor, including its normal
    # boundary correction and publication logic, without loading GPU models.
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    word_spans = [(0, .1, ""), (.1, .4, "ni3"), (.4, .6, "hao3"), (.6, 1.1, ""), (1.1, 1.4, "a5")]
    grid = post.TextGrid(0, 1.4, [post.Tier(name, 0, 1.4, [post.Interval(s,e,w) for s,e,w in word_spans]) for name in ("words", "phones")])
    post.write_textgrid(grid, aligned / "a.TextGrid")
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / "scripts/postprocess_textgrids.py"),
               "--txt-dir", str(output), "--raw-text-dir", str(data),
               "--original-txt-dir", str(data),
               "--reference-mode", reference_mode, "--textgrid-dir", str(aligned),
               "--wav-dir", str(data), "--output-dir", str(tmp_path / "final"),
               "--filtered-dir", str(tmp_path / "filtered"), "--tone-ref", str(tmp_path / "tone.json"),
               "--workers", "1", "--allow-filtered-integrity-failures"]
    parsed = []
    def capture_args(args):
        parsed.append(args)
        raise RuntimeError("captured parser defaults")
    monkeypatch.setattr(sys, "argv", command[1:])
    monkeypatch.setattr(post, "_load_axis_contract", capture_args)
    with pytest.raises(RuntimeError, match="captured parser defaults"):
        post.main()
    args = parsed[0]
    args._axis_stem_reasons = {}
    args.output_dir.mkdir()
    args.filtered_dir.mkdir()
    report = post.process_one(aligned / "a.TextGrid", output, data,
                              args.output_dir, args.filtered_dir, args,
                              {}, {"ni3": ["ni3"], "hao3": ["hao3"], "a5": ["a5"]})
    candidates = [Path(report["output"])]
    final_grid = post.parse_textgrid(candidates[0])
    final_text = post.tier_by_name(final_grid, "raw_text").intervals[0].text
    assert final_text == "<sp1>你好，啊"
    assert [(iv.xmin, iv.xmax) for iv in post.tier_by_name(final_grid, "words").intervals
            if iv.text == "，"] == [(.6, 1.1)]
    words_punctuation = [mark for iv in post.tier_by_name(final_grid, "words").intervals
                         if post.is_punct(iv.text) and not post.is_silence(iv.text)
                         for mark in iv.text.strip()]
    assert post._surface_punctuation(post.tier_by_name(final_grid, "raw_text")) == words_punctuation
    assert post._surface_punctuation(post.tier_by_name(final_grid, "pinyin")) == words_punctuation
    from scripts.qwen3_timestamp_normalization import read_normalized_transcript
    from scripts import audit_strict_ok as audit
    assert read_normalized_transcript(output, "a") == "你好，啊！"
    transaction = report["derived_publication_transaction"]
    assert transaction["removed_source_punctuation"] == [{
        "label": "！", "source_boundary": 3,
        "reason": "absent_from_frozen_words",
    }]
    assert "report_positive:timestamp_normalized_transcript" not in audit._report_reasons(report)
    assert not any("punctuation_evidence_schema" in reason or "nvasr_candidate" in reason
                   for reason in report.get("filter_reasons", []))


def test_reference_only_does_not_require_asr_model_tree(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    (data / "a.txt").write_text("你好")
    cfg = _cfg(tmp_path, data, tmp_path / "out", reference_mode="authority")
    cfg["ctc_prealign"]["model_path"] = str(tmp_path / "absent-asr-model")
    fake = FakeBackend({})
    assert run_qwen3_hf(cfg, tmp_path, backend=fake) == 0
    assert fake.transcribe_calls == []
    assert run_qwen3_hf(cfg, tmp_path, backend=FakeBackend({})) == 0


def test_qwen_failure_is_accounted_and_retried_without_reference_fallback(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    (data / "a.txt").write_text("不能回退到这份参考文本")
    output = tmp_path / "out"
    cfg = _cfg(tmp_path, data, output, reference_mode="fallback")
    class FailingBackend(FakeBackend):
        def transcribe(self, audio, **kwargs):
            self.transcribe_calls.append(audio.stem)
            raise RuntimeError("Qwen inference failed")
    failing = FailingBackend({})
    assert run_qwen3_hf(cfg, tmp_path, backend=failing) == 1
    assert failing.transcribe_calls == ["a"] and failing.align_calls == []
    assert not (output / "a_text_cn.txt").exists()
    retry = FakeBackend({"a": "你好"})
    assert run_qwen3_hf(cfg, tmp_path, backend=retry) == 0
    assert retry.transcribe_calls == ["a"]
    assert (output / "a_text_cn.txt").read_text().strip() == "你好…"


def test_configured_item_failures_are_filtered_without_failing_the_batch(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    _wav(data / "b.wav")
    (data / "a.txt").write_text("你好")
    (data / "b.txt").write_text("再见")
    output = tmp_path / "out"
    cfg = _cfg(tmp_path, data, output)
    cfg["ctc_prealign"]["allow_item_failures"] = True

    class OneFailure(FakeBackend):
        def align(self, audio, text, **kwargs):
            if audio.stem == "b":
                raise RuntimeError("bad alignment")
            return super().align(audio, text, **kwargs)

    assert run_qwen3_hf(cfg, tmp_path, backend=OneFailure({})) == 0
    receipt = json.loads((output / ".pipeline_run_receipt_v2.json").read_text())
    assert receipt["output"]["stems"] == ["a"]
    assert receipt["filtered"]["stems"] == ["b"]


def test_reference_nvv_is_rejected_without_publishing(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav")
    (data / "a.txt").write_text("你好 [Breathing]")
    output = tmp_path / "out"
    fake = FakeBackend({})
    with pytest.raises(Qwen3PrealignError, match="NVV"):
        run_qwen3_hf(_cfg(tmp_path, data, output), tmp_path, backend=fake)
    assert not output.exists()
    assert fake.align_calls == []


def test_pipeline_dispatch_seals_qwen_bundle_and_resumes(tmp_path, monkeypatch):
    import run_pipeline as pipeline
    from types import SimpleNamespace
    data = tmp_path / "data"
    data.mkdir()
    _wav(data / "a.wav", seconds=2)
    (data / "a.txt").write_text("你好 hello")
    output = tmp_path / "out"
    cfg = _cfg(tmp_path, data, output)
    cfg["ctc_prealign"].update(provider="qwen3_hf", enabled=True)
    ctx = {"data_dir": data, "audio_dir": data, "ctc_pretg": output,
           "models_dir": tmp_path, "mfa_dict": tmp_path / "dict",
           "workspace": tmp_path, "mfa_audio_dir": data,
           "reference_mode": "authority", "expected_stems": ("a",),
           "accounting_receipt_path": output / ".pipeline_run_receipt_v2.json"}
    args = SimpleNamespace(overwrite=False, device=None)
    transform_stems = []
    def invoke(script, cli, *unused, **kwargs):
        assert script.name == "qwen3_prealign.py"
        assert cli[cli.index("--audio-dir") + 1] == str(data)
        assert "--forced-aligner-model-path" in cli
        return run_qwen3_hf(cfg, tmp_path, backend=FakeBackend({}))
    monkeypatch.setattr(pipeline, "run_python", invoke)
    monkeypatch.setattr(
        pipeline, "_ensure_mfa_transform_receipts",
        lambda _ctx, stems: transform_stems.append(tuple(stems)) or 0,
    )
    assert pipeline.step_prealign(args, cfg, Path(sys.executable), ctx) == 0
    assert transform_stems == [("a",)]
    cfg["ctc_prealign"]["provider"] = "nvasr"
    assert pipeline.step_prealign(args, cfg, Path(sys.executable), ctx) == 1
    assert (output / ".ctc_raw_manifest.json").is_file()
    cfg["ctc_prealign"]["provider"] = "qwen3_hf"
    assert pipeline.step_prealign(args, cfg, Path(sys.executable), ctx) == 0
    assert transform_stems == [("a",), ("a",)]
