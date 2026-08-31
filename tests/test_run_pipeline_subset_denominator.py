from argparse import Namespace
import json
import sys
import wave
from pathlib import Path

import pytest

from scripts import run_pipeline
from scripts.pipeline_utils import _axis_audio_metadata
from scripts.pipeline_utils import make_pipeline_accounting_receipt


def _ctx(tmp_path: Path, *, subset: bool):
    source = tmp_path / "full_data_dir"
    output = tmp_path / "audio_16k"
    source.mkdir()
    return {
        "audio_dir": source,
        "mfa_audio_dir": output,
        "ctc_pretg": tmp_path / "ctc_pretg",
        "expected_stems": ("ok_a", "ok_b"),
        "ctc_ready_subset": subset,
    }, source, output


def _resume_scope_fixture(tmp_path: Path):
    data_dir = tmp_path / "data"
    selected_a = data_dir / "speaker_a" / "selected_a.wav"
    selected_b = data_dir / "speaker_b" / "selected_b.wav"
    nonselected = data_dir / "unselected" / "nested" / "other.wav"
    for path in (selected_a, selected_b, nonselected):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))
    (selected_a.parent / "selected_a.txt").write_text(
        "authority-a\n", encoding="utf-8")
    return {
        "data_dir": data_dir,
        "expected_stems": ("selected_b", "selected_a"),
        "pre_ctc_audio_index": {
            "selected_b": selected_b,
            "selected_a": selected_a,
        },
        "ctc_pretg": tmp_path / "raw_ctc",
        "models_dir": tmp_path / "models",
        "ctc_pretg_adj": tmp_path / "ctc_work",
        "mfa_audio_dir": tmp_path / "mfa_audio",
    }, selected_a, selected_b, nonselected


def test_resume_fingerprint_scopes_to_frozen_index_without_rglob(
        tmp_path, monkeypatch):
    ctx, selected_a, selected_b, _nonselected = _resume_scope_fixture(tmp_path)

    def forbidden_rglob(_self, _pattern):
        raise AssertionError("scoped fingerprint scanned an unselected tree")

    monkeypatch.setattr(Path, "rglob", forbidden_rglob)
    fingerprints = run_pipeline._pipeline_resume_fingerprints(
        ctx, {}, outputs={})
    scoped = fingerprints["inputs"]["data_dir"]
    assert scoped["schema"] == "data-dir-scoped-fingerprint-v1"
    assert scoped["scope"] == "frozen_expected_stems"
    assert scoped["selected_count"] == 2
    assert scoped["selected_stems"] == ["selected_a", "selected_b"]
    assert [row["path"] for row in scoped["files"]] == [
        "speaker_a/selected_a.wav", "speaker_a/selected_a.txt",
        "speaker_b/selected_b.wav", "speaker_b/selected_b.txt",
    ]
    assert scoped["files"][-1]["status"] == "missing"
    reordered = dict(ctx)
    reordered["expected_stems"] = ("selected_a", "selected_b")
    reordered["pre_ctc_audio_index"] = {
        "selected_a": selected_a,
        "selected_b": selected_b,
    }
    assert run_pipeline._scoped_data_dir_fingerprint(reordered) == scoped


def test_resume_scoped_fingerprint_ignores_nonselected_content(
        tmp_path):
    ctx, _selected_a, _selected_b, nonselected = _resume_scope_fixture(tmp_path)
    before = run_pipeline._scoped_data_dir_fingerprint(ctx)
    nonselected.write_bytes(b"changed nonselected content")
    after = run_pipeline._scoped_data_dir_fingerprint(ctx)
    assert after == before


def test_resume_scoped_fingerprint_tracks_selected_wav_and_authority_txt(
        tmp_path):
    ctx, selected_a, _selected_b, _nonselected = _resume_scope_fixture(tmp_path)
    before = run_pipeline._scoped_data_dir_fingerprint(ctx)

    selected_a.write_bytes(b"changed selected wav")
    wav_changed = run_pipeline._scoped_data_dir_fingerprint(ctx)
    assert wav_changed["digest"] != before["digest"]

    selected_a.write_bytes(b"selected_a.wav")
    (selected_a.parent / "selected_a.txt").write_text(
        "changed authority-a\n", encoding="utf-8")
    txt_changed = run_pipeline._scoped_data_dir_fingerprint(ctx)
    assert txt_changed["digest"] != before["digest"]


def test_resume_scoped_fingerprint_tracks_creation_of_missing_authority_txt(
        tmp_path):
    ctx, _selected_a, selected_b, _nonselected = _resume_scope_fixture(tmp_path)
    before = run_pipeline._scoped_data_dir_fingerprint(ctx)
    missing = next(row for row in before["files"]
                   if row["stem"] == "selected_b" and row["role"] == "authority_txt")
    assert missing["status"] == "missing"

    (selected_b.parent / "selected_b.txt").write_text(
        "authority-b\n", encoding="utf-8")
    after = run_pipeline._scoped_data_dir_fingerprint(ctx)
    present = next(row for row in after["files"]
                   if row["stem"] == "selected_b" and row["role"] == "authority_txt")
    assert present["status"] == "present"
    assert after["digest"] != before["digest"]


def test_resume_invalid_or_incomplete_index_falls_back_to_full_fingerprint(
        tmp_path, monkeypatch):
    ctx, selected_a, selected_b, _nonselected = _resume_scope_fixture(tmp_path)
    original = run_pipeline._path_fingerprint
    seen = []

    def record_path_fingerprint(path):
        seen.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(run_pipeline, "_path_fingerprint", record_path_fingerprint)
    for invalid_index in (
            {"selected_a": selected_a},
            {"selected_a": tmp_path / "outside" / "selected_a.wav",
             "selected_b": selected_b},
    ):
        if invalid_index["selected_a"].name == "selected_a.wav" \
                and invalid_index["selected_a"].parent.name == "outside":
            invalid_index["selected_a"].parent.mkdir(parents=True, exist_ok=True)
            invalid_index["selected_a"].write_bytes(b"outside")
        ctx["pre_ctc_audio_index"] = invalid_index
        fingerprints = run_pipeline._pipeline_resume_fingerprints(
            ctx, {}, outputs={})
        assert ctx["data_dir"].resolve() in seen
        assert fingerprints["inputs"]["data_dir"] == original(ctx["data_dir"])


def test_resample_explicit_subset_reads_only_selected_stems_from_full_source(
        tmp_path, monkeypatch):
    ctx, source, output = _ctx(tmp_path, subset=True)
    for stem in ("ok_a", "ok_b", "extra_000"):
        (source / f"{stem}.wav").write_bytes(b"source")

    seen = []
    receipt_stems = []

    def fake_resample(wav, audio_dir, mfa_audio_dir, target_sr, overwrite):
        seen.append(wav.name)
        (mfa_audio_dir / wav.name).write_bytes(b"resampled")
        return wav.name, True, "resampled"

    monkeypatch.setattr(run_pipeline, "_resample_one", fake_resample)
    monkeypatch.setattr(
        run_pipeline, "_ensure_mfa_transform_receipts",
        lambda _ctx, stems: receipt_stems.append(tuple(stems)) or 0,
    )

    rc = run_pipeline.step_resample_for_mfa(
        Namespace(overwrite=True), {"mfa": {"num_jobs": 1}}, None, ctx)

    assert rc == 0
    assert seen == ["ok_a.wav", "ok_b.wav"]
    assert {path.name for path in output.iterdir()} == {"ok_a.wav", "ok_b.wav"}
    assert receipt_stems == [("ok_a", "ok_b")]


def test_resample_flattens_nested_source_for_mfa(tmp_path):
    source = tmp_path / "source"
    nested = source / "speaker" / "nested.wav"
    output = tmp_path / "audio_16k"
    nested.parent.mkdir(parents=True)
    output.mkdir()
    with wave.open(str(nested), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)

    rel, ok, action = run_pipeline._resample_one(
        nested, source, output, 16000, True)

    assert (rel, ok, action) == ("nested.wav", True, "linked")
    assert (output / "nested.wav").is_file()
    assert not (output / "speaker" / "nested.wav").exists()


def test_resample_non_filtered_expected_set_keeps_full_namespace_contract(
        tmp_path, monkeypatch):
    ctx, source, _output = _ctx(tmp_path, subset=False)
    for stem in ("ok_a", "ok_b", "extra_000"):
        (source / f"{stem}.wav").write_bytes(b"source")
    called = []
    monkeypatch.setattr(
        run_pipeline, "_resample_one",
        lambda *args: called.append(args) or ("", True, "resampled"),
    )

    rc = run_pipeline.step_resample_for_mfa(
        Namespace(overwrite=True), {"mfa": {"num_jobs": 1}}, None, ctx)

    assert rc == 1
    assert called == []


def test_resample_subset_rejects_existing_extra_mfa_audio(tmp_path, monkeypatch):
    ctx, source, output = _ctx(tmp_path, subset=True)
    for stem in ("ok_a", "ok_b"):
        (source / f"{stem}.wav").write_bytes(b"source")
    output.mkdir()
    (output / "ok_a.wav").write_bytes(b"old")
    (output / "stale_54k.wav").write_bytes(b"old")
    called = []
    monkeypatch.setattr(
        run_pipeline, "_resample_one",
        lambda *args: called.append(args) or ("", True, "resampled"),
    )

    rc = run_pipeline.step_resample_for_mfa(
        Namespace(overwrite=False), {"mfa": {"num_jobs": 1}}, None, ctx)

    assert rc == 1
    assert called == []


def test_resample_subset_without_frozen_expected_stems_fails_closed(
        tmp_path, monkeypatch):
    ctx, source, _output = _ctx(tmp_path, subset=True)
    ctx.pop("expected_stems")
    (source / "000000.wav").write_bytes(b"must-not-scan")
    called = []
    monkeypatch.setattr(
        run_pipeline, "_resample_one",
        lambda *args: called.append(args) or ("", True, "resampled"),
    )

    rc = run_pipeline.step_resample_for_mfa(
        Namespace(overwrite=True), {"mfa": {"num_jobs": 1}}, None, ctx)

    assert rc == 1
    assert called == []


def test_pre_ctc_limit_freezes_sorted_manifest_and_accounting(tmp_path):
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    (source / "speaker_a").mkdir(parents=True)
    (source / "speaker_b").mkdir(parents=True)
    for stem in ("000003", "000001", "000002", "000004"):
        parent = source / ("speaker_a" if stem < "000003" else "speaker_b")
        (parent / f"{stem}.wav").write_bytes(b"wav")

    ctx = {
        "mode": "nvrasr_fallback", "audio_dir": source,
        "workspace": workspace,
    }
    selected = run_pipeline._freeze_pre_ctc_stems(
        {"ctc_prealign": {"limit": 2}}, ctx)

    assert selected == ("000001", "000002")
    assert (workspace / "pre_ctc_stems.txt").read_text() == (
        "000001\n000002\n")
    assert ctx["expected_stems"] == selected
    assert ctx["accounting_source_stems"] == (
        "000001", "000002", "000003", "000004")
    assert ctx["accounting_eligible_stems"] == selected
    assert len(ctx["accounting_exclusions"]) == 2
    assert ctx["pre_ctc_selection"]["selected_count"] == 2


def _pre_ctc_accounting_ctx(tmp_path: Path):
    receipt_path = tmp_path / "formal" / ".pipeline_run_receipt_v2.json"
    return {
        "accounting_receipt_path": receipt_path,
        "expected_stems": ("a", "b"),
        "pre_ctc_selection": {
            "schema": "pre-ctc-stem-selection-v1",
            "stems": ["a", "b"],
            "selected_count": 2,
            "source_count": 4,
            "limit": 2,
        },
        "accounting_source_stems": ("a", "b", "c", "d"),
        "accounting_eligible_stems": ("a", "b"),
        "accounting_exclusions": (
            {"stem": "c", "reason": "pre_ctc_limit"},
            {"stem": "d", "reason": "pre_ctc_limit"},
        ),
    }


def _write_ctc_accounting(path: Path, *, source, eligible, output, filtered,
                          extra=None):
    receipt = make_pipeline_accounting_receipt(
        source, eligible, [], output, filtered,
        run_id="producer", mode="ctc_prealign", route=["ctc_prealign"],
        extra=extra,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def test_load_ctc_accounting_projects_bounded_producer_onto_pipeline_scope(
        tmp_path):
    ctx = _pre_ctc_accounting_ctx(tmp_path)
    producer = _write_ctc_accounting(
        ctx["accounting_receipt_path"],
        source=["a", "b"], eligible=["a", "b"], output=["a"], filtered=["b"],
        extra={"producer_note": "preserve"},
    )

    assert run_pipeline._load_ctc_accounting(ctx, required=True) == 0
    formal = json.loads(ctx["accounting_receipt_path"].read_text())
    assert formal["source"]["stems"] == ["a", "b", "c", "d"]
    assert formal["eligible"]["stems"] == ["a", "b"]
    assert formal["exclusions"] == [
        {"stem": "c", "reason": "pre_ctc_limit"},
        {"stem": "d", "reason": "pre_ctc_limit"},
    ]
    assert formal["output"]["stems"] == ["a"]
    assert formal["filtered"]["stems"] == ["b"]
    assert formal["route"].count("project_pre_ctc_selection") == 1
    assert formal["extra"]["producer_note"] == "preserve"
    assert formal["extra"]["denominator_projection"] == (
        "frozen_pre_ctc_selection")
    assert formal["extra"]["producer_scope_identity"] == {
        "source_stems_digest": producer["source"]["stems_digest"],
        "eligible_stems_digest": producer["eligible"]["stems_digest"],
        "output_stems_digest": producer["output"]["stems_digest"],
        "filtered_stems_digest": producer["filtered"]["stems_digest"],
    }
    assert ctx["accounting_source_stems"] == ("a", "b", "c", "d")
    assert ctx["accounting_eligible_stems"] == ("a", "b")


def test_load_ctc_accounting_projection_is_byte_stable_on_resume(tmp_path):
    ctx = _pre_ctc_accounting_ctx(tmp_path)
    producer = _write_ctc_accounting(
        ctx["accounting_receipt_path"],
        source=["a", "b"], eligible=["a", "b"], output=["a"], filtered=["b"],
    )

    assert run_pipeline._load_ctc_accounting(ctx, required=True) == 0
    first_bytes = ctx["accounting_receipt_path"].read_bytes()
    first_formal = json.loads(first_bytes)
    first_identity = first_formal["extra"]["producer_scope_identity"]
    assert first_identity["source_stems_digest"] == producer["source"]["stems_digest"]
    assert first_identity["eligible_stems_digest"] == producer["eligible"]["stems_digest"]

    assert run_pipeline._load_ctc_accounting(ctx, required=True) == 0
    assert ctx["accounting_receipt_path"].read_bytes() == first_bytes
    second_formal = json.loads(ctx["accounting_receipt_path"].read_bytes())
    assert second_formal["extra"]["producer_scope_identity"] == first_identity


@pytest.mark.parametrize("tampered", [
    {"accounting_source_stems": ("a", "b", "c")},
    {"accounting_exclusions": (
        {"stem": "c", "reason": "pre_ctc_limit"},)},
])
def test_load_ctc_accounting_rejects_tampered_upper_scope(tmp_path, tampered):
    ctx = _pre_ctc_accounting_ctx(tmp_path)
    _write_ctc_accounting(
        ctx["accounting_receipt_path"],
        source=["a", "b"], eligible=["a", "b"], output=["a"], filtered=["b"],
    )
    original = ctx["accounting_receipt_path"].read_bytes()
    ctx.update(tampered)

    assert run_pipeline._load_ctc_accounting(ctx, required=True) == 1
    assert ctx["accounting_receipt_path"].read_bytes() == original


def test_load_ctc_accounting_without_selection_keeps_producer_scope(tmp_path):
    receipt_path = tmp_path / ".pipeline_run_receipt_v2.json"
    producer = _write_ctc_accounting(
        receipt_path,
        source=["a", "b"], eligible=["a", "b"], output=["a"], filtered=["b"],
        extra={"producer_note": "direct"},
    )
    ctx = {"accounting_receipt_path": receipt_path,
           "expected_stems": ("a", "b")}

    assert run_pipeline._load_ctc_accounting(ctx, required=True) == 0
    formal = json.loads(receipt_path.read_text())
    assert formal == producer
    assert "denominator_projection" not in formal.get("extra", {})


def test_load_ctc_accounting_projects_legacy_full_producer_compatibly(tmp_path):
    ctx = _pre_ctc_accounting_ctx(tmp_path)
    producer = _write_ctc_accounting(
        ctx["accounting_receipt_path"],
        source=["a", "b", "c", "d"],
        eligible=["a", "b", "c", "d"],
        output=["a", "c"], filtered=["b", "d"],
    )

    assert run_pipeline._load_ctc_accounting(ctx, required=True) == 0
    formal = json.loads(ctx["accounting_receipt_path"].read_text())
    assert formal["source"]["stems"] == ["a", "b", "c", "d"]
    assert formal["eligible"]["stems"] == ["a", "b"]
    assert formal["output"]["stems"] == ["a"]
    assert formal["filtered"]["stems"] == ["b"]
    assert formal["extra"]["producer_scope_identity"]["source_stems_digest"] == (
        producer["source"]["stems_digest"])


def test_pre_ctc_manifest_is_reused_by_prealign_without_second_limit(
        tmp_path, monkeypatch):
    source = tmp_path / "audio"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    for stem in ("000001", "000002", "000003"):
        (source / f"{stem}.wav").write_bytes(b"wav")
    ctx = {"mode": "full", "data_dir": source, "audio_dir": source,
           "workspace": workspace, "pinyin_dir": tmp_path / "pinyin",
           "ctc_pretg": tmp_path / "ctc", "mfa_dict": tmp_path / "dict",
           "models_dir": tmp_path / "models"}
    seen = {}
    def fake_run(_script, argv, *_args, **_kwargs):
        seen["argv"] = argv
        return 0
    monkeypatch.setattr(run_pipeline, "run_python", fake_run)
    monkeypatch.setattr(run_pipeline, "_load_ctc_accounting", lambda *args, **kwargs: 0)
    monkeypatch.setattr(run_pipeline, "_ensure_ctc_axis_receipt", lambda *args: 0)
    monkeypatch.setattr(run_pipeline, "_seal_ctc_raw", lambda *args: 0)

    rc = run_pipeline.step_prealign(
        type("Args", (), {"device": "cpu", "overwrite": False})(),
        {"ctc_prealign": {"enabled": True, "python": sys.executable,
                           "model_path": "model", "limit": 2}},
        None, ctx)

    assert rc == 0
    assert ctx["expected_stems"] == ("000001", "000002")
    manifest_arg = seen["argv"][seen["argv"].index("--stems-file") + 1]
    assert Path(manifest_arg).read_text() == "000001\n000002\n"
    assert "--limit" not in seen["argv"]


def test_ctc_axis_receipt_prefers_current_padded_audio_root(
        tmp_path, monkeypatch):
    """Pad-silence must not bind padded CTC geometry to the raw WAV axis."""
    raw = tmp_path / "raw" / "stem.wav"
    padded = tmp_path / "padded" / "stem.wav"
    ctc = tmp_path / "ctc"
    workspace = tmp_path / "workspace"
    raw.parent.mkdir()
    padded.parent.mkdir()
    ctc.mkdir()
    workspace.mkdir()

    for path, frames in ((raw, 16000), (padded, 32000)):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * frames)

    (ctc / ".ctc_run_receipt.json").write_text(json.dumps({
        "schema": "ctc-run-receipt-v2",
        "input_stems": ["stem"],
        "output_stems": ["stem"],
        "audio_bindings": [],
    }), encoding="utf-8")
    monkeypatch.setattr(run_pipeline, "validate_ctc_run_receipt_v2",
                        lambda *_args, **_kwargs: [])

    ctx = {
        "ctc_pretg": ctc,
        "audio_dir": padded.parent,
        "pre_ctc_audio_index": {"stem": raw},
        "workspace": workspace,
    }
    assert run_pipeline._ensure_ctc_axis_receipt(ctx) == 0
    binding = ctx["ctc_axis_receipt"]["audio_bindings"][0]
    assert Path(binding["path"]) == padded.resolve()
    assert binding["duration_s"] == 2.0


def _mfa_axis_guard_fixture(tmp_path: Path):
    raw = tmp_path / "ctc_pretg"
    work = tmp_path / "ctc_pretg_adj"
    workspace = tmp_path / "workspace"
    mfa_audio = workspace / "audio_16k"
    raw.mkdir()
    work.mkdir()
    workspace.mkdir(exist_ok=True)
    mfa_audio.mkdir()
    raw_audio = tmp_path / "padded_audio" / "stem.wav"
    raw_audio.parent.mkdir()
    with wave.open(str(raw_audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000)
    return raw, work, workspace, mfa_audio, raw_audio


def _stub_mfa_axis_guard_dependencies(monkeypatch, mfa_audio: Path):
    real_axis_audio_metadata = run_pipeline._axis_audio_metadata

    def fake_transform_receipts(ctx, stems):
        input_row = ctx["ctc_axis_receipt"]["audio_bindings"][0]
        input_path = input_row["path"]
        ctx["mfa_transform_receipts"] = {
            stem: {
                "path": str(ctx["workspace"] / "transform.json"),
                "receipt": {
                    "schema": "audio-transform-receipt-v1",
                    "scale": 1.0,
                    "head_transform_s": 0.0,
                    "tail_transform_s": 0.0,
                    "shift_s": 0.0,
                    "input": {
                        "path": input_path,
                        "sha256": input_row["sha256"],
                    },
                    "output": {
                        "path": str(mfa_audio / f"{stem}.wav"),
                        "sha256": "mfa-audio-hash",
                    },
                },
            }
            for stem in stems
        }
        return 0

    monkeypatch.setattr(run_pipeline, "_ensure_mfa_transform_receipts",
                        fake_transform_receipts)
    monkeypatch.setattr(
        run_pipeline, "_axis_audio_metadata",
        lambda path: (
            real_axis_audio_metadata(path)
            if Path(path).is_file()
            else {"sha256": "mfa-audio-hash", "duration_s": 1.0}
        ),
    )
    monkeypatch.setattr(run_pipeline, "validate_mfa_axis_receipts",
                        lambda *_args, **_kwargs: [])


def _ctc_axis_receipt(raw_audio: Path):
    metadata = _axis_audio_metadata(raw_audio)
    return {
        "schema": run_pipeline.CTC_RUN_RECEIPT_SCHEMA,
        "input_stems": ["stem"],
        "audio_bindings": [{
            "stem": "stem",
            "path": str(raw_audio.resolve()),
            **metadata,
            "ctc_bounds": {"xmin": 0.0, "xmax": 1.0},
        }],
    }


def test_cold_mfa_axis_resume_restores_from_raw_ctc_owner(
        tmp_path, monkeypatch):
    raw, work, workspace, mfa_audio, raw_audio = _mfa_axis_guard_fixture(tmp_path)
    receipt = _ctc_axis_receipt(raw_audio)
    receipt_path = raw / ".ctc_run_receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    before = receipt_path.read_bytes()
    _stub_mfa_axis_guard_dependencies(monkeypatch, mfa_audio)

    ctx = {
        "workspace": workspace,
        "ctc_pretg": raw,
        "ctc_pretg_adj": work,
        "audio_dir": raw_audio.parent,
        "mfa_audio_dir": mfa_audio,
    }
    assert run_pipeline._guard_mfa_axis(ctx, ["stem"], work) == 0
    assert ctx["ctc_axis_receipt"]["schema"] == receipt["schema"]
    assert Path(ctx["ctc_axis_receipt"]["audio_bindings"][0]["path"]) == raw_audio.resolve()
    assert receipt_path.read_bytes() == before
    assert not (work / ".ctc_run_receipt.json").exists()
    assert (workspace / ".mfa_input_axis_receipt.json").is_file()


@pytest.mark.parametrize("raw_payload", [None, {"schema": "legacy-ctc-receipt"}])
def test_cold_mfa_axis_resume_ignores_work_root_receipt_when_raw_invalid(
        tmp_path, monkeypatch, raw_payload):
    raw, work, workspace, mfa_audio, raw_audio = _mfa_axis_guard_fixture(tmp_path)
    if raw_payload is not None:
        (raw / ".ctc_run_receipt.json").write_text(
            json.dumps(raw_payload) + "\n", encoding="utf-8")
    # This is deliberately a valid-looking producer receipt in the wrong
    # owner.  A cold resume must fail before consuming it.
    (work / ".ctc_run_receipt.json").write_text(
        json.dumps(_ctc_axis_receipt(raw_audio)) + "\n", encoding="utf-8")
    _stub_mfa_axis_guard_dependencies(monkeypatch, mfa_audio)
    called = []
    original = run_pipeline._ensure_ctc_axis_receipt

    def record_raw_restore(ctx):
        called.append(Path(ctx["ctc_pretg"]))
        return original(ctx)

    monkeypatch.setattr(run_pipeline, "_ensure_ctc_axis_receipt",
                        record_raw_restore)
    ctx = {
        "workspace": workspace,
        "ctc_pretg": raw,
        "ctc_pretg_adj": work,
        "audio_dir": raw_audio.parent,
        "mfa_audio_dir": mfa_audio,
    }

    assert run_pipeline._guard_mfa_axis(ctx, ["stem"], work) == 1
    assert called == [raw]
    assert "ctc_axis_receipt" not in ctx
    assert not (workspace / ".mfa_input_axis_receipt.json").exists()


def test_warm_mfa_axis_path_does_not_restore_ctc_receipt(
        tmp_path, monkeypatch):
    raw, work, workspace, mfa_audio, raw_audio = _mfa_axis_guard_fixture(tmp_path)
    receipt = _ctc_axis_receipt(raw_audio)
    _stub_mfa_axis_guard_dependencies(monkeypatch, mfa_audio)
    monkeypatch.setattr(
        run_pipeline, "_ensure_ctc_axis_receipt",
        lambda _ctx: (_ for _ in ()).throw(AssertionError("warm path restored")),
    )
    ctx = {
        "workspace": workspace,
        "ctc_pretg": raw,
        "ctc_pretg_adj": work,
        "audio_dir": raw_audio.parent,
        "mfa_audio_dir": mfa_audio,
        "ctc_axis_receipt": receipt,
    }

    assert run_pipeline._guard_mfa_axis(ctx, ["stem"], work) == 0


def test_adjust_cache_requires_processed_geometry_not_only_textgrids(tmp_path):
    ctc_dir = tmp_path / "ctc_pretg_adj"
    ctc_dir.mkdir()
    stem = "stem"
    (ctc_dir / f"{stem}.lab").write_text("hello\n", encoding="utf-8")
    (ctc_dir / f"{stem}.TextGrid").write_text("placeholder\n", encoding="utf-8")
    token = {
        "word": "hello",
        "canonical_span": [0.1, 0.2],
        "canonical_unit": {"canonical_span": [0.1, 0.2]},
    }
    token_path = ctc_dir / f"{stem}_tokens.jsonl"
    token_path.write_text(json.dumps(token) + "\n", encoding="utf-8")

    assert not run_pipeline._processed_geometry_cache_complete(
        ctc_dir, {stem})

    token["processed_ctc_span"] = [0.1, 0.35]
    token["processed_ctc_boundary_source"] = "next_lexical_token_start"
    token_path.write_text(json.dumps(token) + "\n", encoding="utf-8")
    assert run_pipeline._processed_geometry_cache_complete(ctc_dir, {stem})


def test_pad_pre_ctc_limit_passes_frozen_manifest_and_pre_ctc_flag(
        tmp_path, monkeypatch):
    source = tmp_path / "audio"
    workspace = tmp_path / "workspace"
    ctc = tmp_path / "ctc"
    output = tmp_path / "output"
    source.mkdir(); workspace.mkdir(); ctc.mkdir(); output.mkdir()
    for stem in ("000001", "000002", "000003"):
        (source / f"{stem}.wav").write_bytes(b"wav")
    seen = {}

    def fake_run(_script, argv, *_args, **_kwargs):
        seen["argv"] = argv
        padded = Path(argv[argv.index("--padded-audio-dir") + 1])
        padded.mkdir(parents=True, exist_ok=True)
        selected = Path(argv[argv.index("--stems-file") + 1]).read_text().splitlines()
        for stem in selected:
            (padded / f"{stem}.wav").write_bytes(b"padded")
        return 0

    monkeypatch.setattr(run_pipeline, "run_python", fake_run)
    monkeypatch.setattr(
        run_pipeline, "make_audio_transform_receipt",
        lambda source_path, output_path: {"source": str(source_path),
                                           "output": str(output_path)})
    monkeypatch.setattr(run_pipeline, "write_audio_transform_receipt",
                        lambda *_args, **_kwargs: None)

    ctx = {"mode": "nvrasr_fallback", "audio_dir": source,
           "workspace": workspace, "ctc_pretg": ctc,
           "output_dir": output, "models_dir": tmp_path / "models"}
    rc = run_pipeline.step_pad_silence(
        Namespace(), {"ctc_prealign": {"limit": 2},
                      "pad_silence": {"enabled": True}}, None, ctx)

    assert rc == 0
    assert "--pre-ctc" in seen["argv"]
    assert ctx["expected_stems"] == ("000001", "000002")
    assert {path.name for path in (workspace / "padded_audio").iterdir()} == {
        "000001.wav", "000002.wav"}


def test_ok100_canary_enables_explicit_partial_policy():
    config_path = Path(__file__).parents[1] / "configs" / \
        "hecheng_ria_ok100_authority.yaml"
    cfg = run_pipeline.load_config(config_path)
    assert cfg["mfa"]["allow_partial"] is True
    assert cfg["mfa"]["min_output_ratio"] == 0.99
    assert run_pipeline.validate_config(cfg, "ctc_ready") == []


def test_partial_missing_mfa_reason_is_bound_to_accounting_receipt(tmp_path):
    missing = "047714_弹幕互动_回应恶意攻击"
    aligned = "ok_aligned"
    output = tmp_path / "output"
    filtered = tmp_path / "filtered"
    output.mkdir(); filtered.mkdir()
    receipt_path = tmp_path / ".pipeline_run_receipt_v2.json"
    source = make_pipeline_accounting_receipt(
        [aligned, missing], [aligned, missing], [], [aligned, missing], [],
        run_id="test", mode="ctc_ready", route=["link"],
        paths={"output": str(output), "filtered": str(filtered)})
    ctx = {
        "accounting_receipt": source,
        "accounting_receipt_path": receipt_path,
        "output_dir": output,
        "filtered_dir": filtered,
        "filtered_reason_map": {"missing_mfa_alignment": [missing]},
    }

    assert run_pipeline._refresh_postprocess_accounting(
        ctx, {aligned}, {missing}) == 0
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stored["eligible"]["stems"] == sorted([aligned, missing])
    assert stored["output"]["stems"] == [aligned]
    assert stored["filtered"]["stems"] == [missing]
    assert stored["extra"]["filtered_reasons"] == {
        "missing_mfa_alignment": [missing]
    }


def test_link_fast_path_rejects_manifest_from_full_workspace(tmp_path):
    source = tmp_path / "source"
    ctc_source = tmp_path / "source_ctc"
    workspace = tmp_path / "workspace"
    source.mkdir()
    ctc_source.mkdir()
    stems = ("ok_a", "ok_b")
    for stem in stems:
        wav_path = source / f"{stem}.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16)
        for suffix in run_pipeline._CTC_SUFFIXES:
            (ctc_source / f"{stem}{suffix}").write_bytes(b"evidence")
        (ctc_source / f"{stem}_ref.txt").write_text("ok\n", encoding="utf-8")
    # Simulate an old full-run workspace.  The explicit subset must not
    # accept its manifest or let the downstream resample fallback scan it.
    ctc_out = workspace / "ctc_pretg"
    ctc_out.mkdir(parents=True)
    (ctc_out / "ctc_ready_manifest.json").write_text(
        json.dumps({"stems": ["000000", *stems]}), encoding="utf-8")
    cfg = {"ctc_ready": {"ctc_dir": str(ctc_source), "stems": list(stems),
                          "require_all": True}}
    ctx = {"data_dir": source, "audio_dir": source, "ctc_pretg": ctc_out,
           "workspace": workspace, "mode": "ctc_ready"}

    rc = run_pipeline.step_link_ctc(
        Namespace(overwrite=False, scan_only=False), cfg, None, ctx)

    assert rc == 1
    assert "expected_stems" not in ctx


def test_link_freezes_explicit_ctc_ready_subset_before_receipt_reuse(tmp_path):
    source = tmp_path / "full_data_dir"
    ctc_source = tmp_path / "source_ctc"
    workspace = tmp_path / "workspace"
    source.mkdir()
    ctc_source.mkdir()

    stems = ("ok_a", "ok_b")
    for stem in (*stems, "extra_000"):
        wav_path = source / f"{stem}.wav"
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16)
        if stem == "extra_000":
            continue
        for suffix in run_pipeline._CTC_SUFFIXES:
            (ctc_source / f"{stem}{suffix}").write_bytes(b"evidence")
        (ctc_source / f"{stem}_ref.txt").write_text("ok\n", encoding="utf-8")

    bindings = []
    for stem in stems:
        wav_path = source / f"{stem}.wav"
        meta = _axis_audio_metadata(wav_path)
        bindings.append({
            "stem": stem,
            "path": str(wav_path),
            **meta,
            "ctc_bounds": {"xmin": 0.0, "xmax": meta["duration_s"]},
        })
    ordered = sorted(stems)
    receipt = {
        "schema": "ctc-run-receipt-v2",
        "argv": [],
        "asr_python": "test",
        "model": {},
        "dictionary": {},
        "input_stems": ordered,
        "input_stems_digest": run_pipeline.stable_json_digest(ordered),
        "output_stems": ordered,
        "output_stems_digest": run_pipeline.stable_json_digest(ordered),
        "audio_bindings": bindings,
    }
    (ctc_source / ".ctc_run_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8")

    ctx = {
        "data_dir": source,
        "audio_dir": source,
        "ctc_pretg": workspace / "ctc_pretg",
        "workspace": workspace,
        "mode": "ctc_ready",
        "output_dir": workspace / "output",
        "accounting_required": False,
    }
    args = Namespace(overwrite=True, scan_only=False)
    cfg = {"ctc_ready": {"ctc_dir": str(ctc_source), "stems": list(stems),
                          "require_all": True}}

    assert run_pipeline.step_link_ctc(args, cfg, None, ctx) == 0
    assert ctx["ctc_ready_subset"] is True
    assert ctx["expected_stems"] == stems
    assert set(ctx["expected_source_audio_map"]) == set(stems)


def test_require_fresh_workspace_rejects_existing_targets_across_modes(tmp_path):
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    workspace.mkdir()
    for mode in ("full", "ctc_ready", "nvrasr_fallback", "strict_replay",
                 "filtered_recovery", "mfa_retry", "mfa_rescue", "batch_ctc_ready"):
        try:
            run_pipeline.guard_fresh_workspace_targets(
                workspace, output, required=True, mode=mode)
        except ValueError as exc:
            assert "require_fresh_workspace=true" in str(exc)
            assert mode in str(exc)
        else:
            raise AssertionError(f"fresh guard accepted existing target in {mode}")


def test_require_fresh_workspace_rejects_existing_output_before_workspace(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    try:
        run_pipeline.guard_fresh_workspace_targets(
            tmp_path / "new_workspace", output, required=True, mode="full")
    except ValueError as exc:
        assert "output=" in str(exc)
    else:
        raise AssertionError("fresh guard accepted existing output")


def test_english_namespace_is_run_local_and_rejects_aliases(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert run_pipeline.resolve_workspace_namespace(
        workspace, "replays/en_phones_v2", "mfa_en.output_dir") == (
            workspace / "replays" / "en_phones_v2")

    for unsafe in ("../old_en_phones", str(tmp_path / "absolute"), "."):
        with pytest.raises(ValueError):
            run_pipeline.resolve_workspace_namespace(
                workspace, unsafe, "mfa_en.output_dir")

    target = workspace / "real_en_phones"
    target.mkdir()
    (workspace / "aliased_en_phones").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_pipeline.resolve_workspace_namespace(
            workspace, "aliased_en_phones", "mfa_en.output_dir")


def test_stale_or_legacy_ctc_work_receipt_cannot_resume(tmp_path):
    raw = tmp_path / "raw"
    work = tmp_path / "work"
    raw.mkdir()
    for suffix in run_pipeline._CTC_SUFFIXES:
        (raw / f"sample{suffix}").write_text("evidence\n", encoding="utf-8")
    (raw / ".ctc_run_receipt.json").write_text("{}\n", encoding="utf-8")
    ctx = {
        "ctc_pretg": raw, "ctc_pretg_adj": work, "workspace": tmp_path,
        "data_dir": tmp_path / "data", "models_dir": tmp_path / "models",
        "expected_stems": ("sample",), "config": {"revision": 1},
        "reference_mode": "authority",
    }
    assert run_pipeline._ensure_ctc_work(ctx) == work
    ctx["config"] = {"revision": 2}
    assert run_pipeline._ensure_ctc_work(ctx) is None

    receipt_path = work / run_pipeline.CTC_WORK_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("fingerprints", None)
    identity = dict(receipt)
    identity.pop("identity", None)
    receipt["identity"] = run_pipeline.stable_json_digest(identity)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert run_pipeline._ensure_ctc_work(ctx) is None


def test_v10_fixed_manifest_hash_and_complete_route():
    root = Path(__file__).parents[1]
    cfg = run_pipeline.load_config(
        root / "configs/hecheng_en_1000_test_v10_fresh.yaml")
    manifest = json.loads((root / cfg["ctc_prealign"]["selection_manifest"]).read_text())
    assert manifest["count"] == 1000
    assert len(manifest["stems"]) == 1000
    assert run_pipeline.stable_json_digest(sorted(manifest["stems"])) == (
        "e22f9d2ea067ef7e9bfa3abbe6fe36d594e8197a10eaddc02536c025be91357b")
    assert cfg["require_fresh_workspace"] is True
    assert cfg["mfa_en"]["min_segment_dur_ms"] == 200
    assert run_pipeline.validate_config(cfg, "nvrasr_fallback") == []
    assert run_pipeline.NVASR_FALLBACK_STEP_ORDER == [
        "pad_silence", "prealign", "normalize_punct", "normalize",
        "normalize_ria", "normalize_en", "resample", "adjust", "align",
        "align_en", "postprocess", "strict_ok"]
