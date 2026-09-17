from pathlib import Path
from types import SimpleNamespace
import json
import sys
import wave

import pytest

from scripts import run_pipeline


def _runtime(tmp_path, monkeypatch, *, supports_anchors):
    package = tmp_path / "runtime" / "montreal_forced_aligner"
    (package / "alignment").mkdir(parents=True)
    (package / "command_line").mkdir()
    for directory in (package, package / "command_line"):
        (directory / "__init__.py").write_text("")
    result = "{'textgrid_directory': unknown_args[1]}" if supports_anchors else "{}"
    (package / "alignment" / "__init__.py").write_text(
        "class PretrainedAligner:\n"
        "    @classmethod\n"
        "    def parse_parameters(cls, config_path, args, unknown_args):\n"
        f"        return {result}\n"
    )
    marker = tmp_path / "alignment-launched"
    (package / "command_line" / "mfa.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['MFA_TEST_LAUNCH_MARKER']).write_text('launched')\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(package.parent))
    monkeypatch.setenv("MFA_TEST_LAUNCH_MARKER", str(marker))
    return Path(sys.executable), marker


@pytest.mark.parametrize("supports_anchors", [False, True])
def test_anchor_argument_must_be_consumed_by_selected_runtime(tmp_path, monkeypatch,
                                                            supports_anchors):
    python, marker = _runtime(tmp_path, monkeypatch, supports_anchors=supports_anchors)
    rc = run_pipeline.run_mfa(
        ["align", "corpus", "dict", "model", "output",
         "--textgrid_directory", str(tmp_path / "anchors")], python, tmp_path)
    assert (rc == 0) is supports_anchors
    assert marker.exists() is supports_anchors


def test_anchor_free_invocation_does_not_require_custom_runtime(tmp_path, monkeypatch):
    python, marker = _runtime(tmp_path, monkeypatch, supports_anchors=False)
    assert run_pipeline.run_mfa(
        ["align", "corpus", "dict", "model", "output"], python, tmp_path) == 0
    assert marker.exists()


@pytest.mark.parametrize("supports_anchors", [False, True])
def test_failed_preflight_does_not_delete_previous_alignment(tmp_path, monkeypatch,
                                                            supports_anchors):
    python, _ = _runtime(tmp_path, monkeypatch, supports_anchors=supports_anchors)
    corpus = tmp_path / "ctc"
    corpus.mkdir()
    (corpus / "a.lab").write_text("ni3")
    (corpus / "a.TextGrid").write_text("anchor")
    aligned = tmp_path / "aligned"
    aligned.mkdir()
    previous = aligned / "a.TextGrid"
    previous.write_text("previous alignment")
    monkeypatch.setattr(run_pipeline, "_guard_mfa_axis", lambda *args: 1)
    ctx = dict(ctc_pretg=corpus, aligned_dir=aligned, temp_dir=tmp_path / "temp",
               models_dir=tmp_path / "models", mfa_dict=tmp_path / "dict",
               mfa_audio_dir=tmp_path, expected_stems=())
    cfg = dict(mfa={}, acoustic_model="mandarin_mfa")
    assert run_pipeline.step_mfa_align(SimpleNamespace(overwrite=True), cfg, python, ctx) == 1
    assert previous.read_text() == "previous alignment"


def test_shards_reject_unsupported_runtime_before_staging(tmp_path, monkeypatch):
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 8)
    python, marker = _runtime(tmp_path, monkeypatch, supports_anchors=False)
    rc = run_pipeline._run_mfa_sharded(
        stems=[f"a{i}" for i in range(400)], corpus_dir=tmp_path / "corpus",
        audio_dir=tmp_path / "audio", anchors_dir=tmp_path / "anchors",
        dict_path=tmp_path / "dict", acoustic_model="model",
        aligned_dir=tmp_path / "aligned", workspace=tmp_path,
        mfa_python=python, models_dir=tmp_path / "models", num_jobs=2,
        single_speaker=True, no_tokenization=True, beam=20, retry_beam=80,
        boost_silence=1.0, dither=0.0, clean=False, overwrite=True)
    assert rc == 1
    assert not (tmp_path / "mfa_shards").exists()
    assert not marker.exists()


def test_retry_rejects_unsupported_runtime_before_staging(tmp_path, monkeypatch):
    python, marker = _runtime(tmp_path, monkeypatch, supports_anchors=False)
    retry_root = tmp_path / "retry"
    result = run_pipeline._execute_single_process_mfa_retry(
        stem="a", retry_root=retry_root, source_corpus=tmp_path / "corpus",
        source_audio=tmp_path / "audio", source_anchors=tmp_path / "anchors",
        dict_path=tmp_path / "dict", acoustic_model="model", mfa_python=python,
        models_dir=tmp_path / "models", output_format="long_textgrid",
        beam=20, retry_beam=80, boost_silence=1.0, single_speaker=True,
        no_tokenization=True, timeout=10, attempt_ordinal=1)
    assert result["invocation_outcome"] == "not_started"
    assert "--textgrid_directory" in result["exception"]
    assert not retry_root.exists()
    assert not marker.exists()


def _write_anchor(path):
    path.write_text(
        '''File type = "ooTextFile"\nObject class = "TextGrid"\n\n'''
        'xmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n'
        '    item [1]:\n        class = "IntervalTier"\n'
        '        name = "words"\n        xmin = 0\n        xmax = 1\n'
        '        intervals: size = 2\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 0.4\n'
        '            text = "ni3"\n        intervals [2]:\n'
        '            xmin = 0.4\n            xmax = 1\n'
        '            text = "hao3"\n'
        '    item [2]:\n        class = "IntervalTier"\n'
        '        name = "pauses"\n        xmin = 0\n        xmax = 1\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 1\n            text = ""\n',
        encoding="utf-8",
    )


def test_native_anchor_corpus_renames_only_words_and_checks_bounds(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    anchor = source / "a.TextGrid"
    _write_anchor(anchor)
    native = tmp_path / "native"
    created = run_pipeline._build_native_anchor_corpus(source, ["a"], native)
    assert created == ["a"]
    text = (native / "a.TextGrid").read_text(encoding="utf-8")
    assert 'name = "speaker"' in text
    assert 'name = "words"' not in text
    assert 'name = "pauses"' not in text
    metadata = json.loads((native / "native_anchor_groups.json").read_text())
    assert metadata["schema"] == "native-anchor-groups-v1"
    assert metadata["stems"]["a"]["groups"]

    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0, 0.2, "ni"), (0.2, 0.4, "3"),
                                   (0.4, 0.7, "hao"), (0.7, 1, "3")],
                         phones=[(0, 0.2, "n")])
    assert run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=native / "native_anchor_groups.json") == []
    assert run_pipeline._validate_native_anchor_bounds(anchor, aligned, tolerance=0) == []


def test_native_group_mapping_is_parsed_once_per_immutable_file(
        tmp_path, monkeypatch):
    """Batch validation must not reparse a corpus-sized mapping per stem."""
    source = tmp_path / "source"
    source.mkdir()
    anchor = source / "a.TextGrid"
    _write_anchor(anchor)
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(source, ["a"], native)
    mapping_path = native / "native_anchor_groups.json"
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0, 0.2, "ni"), (0.2, 0.4, "3"),
                                   (0.4, 0.7, "hao"), (0.7, 1, "3")])
    original_read_text = Path.read_text
    mapping_reads = 0

    def counted_read_text(path, *args, **kwargs):
        nonlocal mapping_reads
        if path == mapping_path:
            mapping_reads += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    for _ in range(2):
        assert run_pipeline._validate_native_anchor_bounds(
            anchor, aligned, group_map_path=mapping_path) == []

    assert mapping_reads == 1


def test_native_anchor_grouping_merges_short_utterances_deterministically(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    anchor = source / "a.TextGrid"
    _write_anchor(anchor)
    text = anchor.read_text(encoding="utf-8")
    text = text.replace("xmax = 0.4", "xmax = 0.06", 1)
    text = text.replace("xmin = 0.4\n            xmax = 1",
                        "xmin = 0.20\n            xmax = 0.26", 1)
    anchor.write_text(text, encoding="utf-8")
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(source, ["a"], native)
    metadata = json.loads((native / "native_anchor_groups.json").read_text())
    groups = metadata["stems"]["a"]["groups"]
    assert len(groups) == 2
    assert [group["anchor_indices"] for group in groups] == [[0], [1]]
    assert all(group["window_end"] - group["window_start"] >= 0.1
               for group in groups)
    native_text = (native / "a.TextGrid").read_text()
    assert 'text = "ni3"' in native_text
    assert 'text = "hao3"' in native_text


def test_native_anchor_grouping_handles_multiple_short_anchors_in_order():
    groups = run_pipeline._group_native_anchor_intervals(
        [("a", 0.00, 0.04), ("b", 0.04, 0.08),
         ("c", 0.08, 0.12), ("d", 0.50, 0.54)], 0.0, 1.0)
    assert [index for group in groups for index in group["anchor_indices"]] == [0, 1, 2, 3]
    assert all(group["window_end"] - group["window_start"] >= 0.1
               for group in groups)


def test_native_anchor_bounds_reject_one_ms_escape(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(anchor.parent, ["a"], native)
    aligned = tmp_path / "aligned.TextGrid"
    aligned.write_text(
        (native / "a.TextGrid").read_text(encoding="utf-8")
        .replace('name = "speaker"', 'name = "words"')
        .replace("xmax = 0.4", "xmax = 0.401", 1),
        encoding="utf-8")
    errors = run_pipeline._validate_native_anchor_bounds(anchor, aligned)
    assert any("escaped CTC interval" in error for error in errors)


def _write_aligned_words(path, intervals, phones=()):
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0", "xmax = 1", "tiers? <exists>",
        f"size = {2 if phones else 1}", "item []:",
        "    item [1]:", '        class = "IntervalTier"',
        '        name = "words"', "        xmin = 0", "        xmax = 1",
        f"        intervals: size = {len(intervals)}",
    ]
    for index, (start, end, text) in enumerate(intervals, 1):
        lines += [f"        intervals [{index}]:", f"            xmin = {start}",
                  f"            xmax = {end}", f'            text = "{text}"']
    if phones:
        lines += ["    item [2]:", '        class = "IntervalTier"',
                  '        name = "phones"', "        xmin = 0", "        xmax = 1",
                  f"        intervals: size = {len(phones)}"]
        for index, (start, end, text) in enumerate(phones, 1):
            lines += [f"        intervals [{index}]:", f"            xmin = {start}",
                      f"            xmax = {end}", f'            text = "{text}"']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_native_anchor_bounds_accept_split_words(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0, 0.2, "ni"), (0.2, 0.4, "3"),
                                   (0.4, 0.7, "hao"), (0.7, 1, "3")])
    assert run_pipeline._validate_native_anchor_bounds(anchor, aligned) == []


def test_native_anchor_bounds_reject_missing_anchor(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0, 0.4, "ni")])
    errors = run_pipeline._validate_native_anchor_bounds(anchor, aligned)
    assert any("has no MFA word" in error for error in errors)


def test_native_anchor_bounds_reject_phone_escape(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0, 0.4, "ni"), (0.4, 1, "hao")],
                         phones=[(0.39, 0.41, "bad")])
    errors = run_pipeline._validate_native_anchor_bounds(anchor, aligned)
    assert any("phone 0 escaped" in error for error in errors)


def test_native_group_validator_accepts_cross_anchor_words_inside_window(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    source_text = anchor.read_text(encoding="utf-8")
    source_text = source_text.replace("xmax = 0.4", "xmax = 0.05", 1)
    source_text = source_text.replace("xmin = 0.4\n            xmax = 1",
                                    "xmin = 0.05\n            xmax = 0.10", 1)
    anchor.write_text(source_text, encoding="utf-8")
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(anchor.parent, ["a"], native)
    mapping_path = native / "native_anchor_groups.json"
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0.02, 0.08, "ni3 hao3")],
                         phones=[(0.03, 0.07, "n")])
    assert run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=mapping_path) == []


def test_native_group_validator_accounts_for_mfa_four_decimal_export(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    anchor.write_text(anchor.read_text().replace("0.4", "0.333333"))
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(anchor.parent, ["a"], native)
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0, 0.3333, "ni3"), (0.3333, 1, "hao3")],
                         phones=[(0.3333, 0.4, "h")])
    mapping = native / "native_anchor_groups.json"
    assert run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=mapping) == []
    # A different export tick is an actual escape, not decimal rounding.
    _write_aligned_words(aligned, [(0, 0.3333, "ni3"), (0.3332, 1, "hao3")],
                         phones=[(0.3332, 0.4, "h")])
    errors = run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=mapping)
    assert any("outside a unique native utterance window" in e for e in errors)
    assert any("phone 0 escaped" in e for e in errors)


def test_native_group_validator_rejects_window_escape_and_missing_group(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(anchor.parent, ["a"], native)
    mapping_path = native / "native_anchor_groups.json"
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0.0, 1.001, "ni3 hao3")],
                         phones=[(0.39, 0.41, "cross")])
    errors = run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=mapping_path)
    assert any("outside a unique native utterance window" in error
               for error in errors)
    assert any("phone 0 escaped native utterance window" in error
               for error in errors)
    _write_aligned_words(aligned, [(0.0, 0.4, "ni3")])
    errors = run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=mapping_path)
    assert any("native utterance group 1 has no MFA word" in error
               for error in errors)


def test_native_group_validator_accepts_oov_and_eps_coverage(tmp_path):
    anchor = tmp_path / "a.TextGrid"
    _write_anchor(anchor)
    native = tmp_path / "native"
    run_pipeline._build_native_anchor_corpus(anchor.parent, ["a"], native)
    aligned = tmp_path / "aligned.TextGrid"
    _write_aligned_words(aligned, [(0.0, 0.2, "<unk><eps>"),
                                   (0.4, 1.0, "<unk>")])
    assert run_pipeline._validate_native_anchor_bounds(
        anchor, aligned, group_map_path=native / "native_anchor_groups.json") == []


def test_overwrite_clears_nested_mfa_temp_state_without_touching_source(tmp_path):
    workspace = tmp_path / "run"
    temp = workspace / "mfa" / "temp"
    nested_db = temp / "corpus" / "corpus.db"
    nested_db.parent.mkdir(parents=True)
    nested_db.write_bytes(b"stale database")
    source = tmp_path / "ctc_raw" / "source.TextGrid"
    source.parent.mkdir()
    source.write_text("immutable", encoding="utf-8")

    assert run_pipeline._clear_mfa_temp_derived_state(temp, workspace) == 0
    assert not nested_db.exists()
    assert source.read_text(encoding="utf-8") == "immutable"


def test_native_anchor_mode_never_builds_custom_anchor_argument(tmp_path, monkeypatch):
    source = tmp_path / "ctc"
    source.mkdir()
    _write_anchor(source / "a.TextGrid")
    (source / "a.lab").write_text("ni3 hao3\n", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"wav")
    captured = {}

    def fake_sharded(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(run_pipeline, "require_mfa_anchor_support",
                        lambda *_args: (_ for _ in ()).throw(
                            AssertionError("native mode must not probe custom anchors")))
    monkeypatch.setattr(run_pipeline, "_run_mfa_sharded", fake_sharded)
    cfg = {"mfa": {"native_anchor_fallback": True, "num_jobs": 64,
                    "single_speaker": True, "no_tokenization": True,
                    "output_format": "long_textgrid"},
           "acoustic_model": "model"}
    ctx = {"ctc_pretg": source, "ctc_pretg_adj": source,
           "pinyin_dir": tmp_path / "pinyin", "mfa_dict": tmp_path / "dict",
           "mfa_audio_dir": audio, "aligned_dir": tmp_path / "aligned",
           "temp_dir": tmp_path / "temp", "workspace": tmp_path,
           "models_dir": tmp_path / "models", "expected_stems": ("a",),
           "ctc_axis_receipt": {"schema": "ctc-run-receipt-v2"}}
    monkeypatch.setattr(run_pipeline, "_guard_mfa_axis", lambda *_args: 0)
    rc = run_pipeline.step_mfa_align(SimpleNamespace(overwrite=False), cfg,
                                     Path("/bin/python"), ctx)
    assert rc == 1
    assert captured["anchors_dir"] is None
    assert captured["native_anchor_dir"] == source
    assert all(path.suffix == ".TextGrid"
               for path in (captured["corpus_dir"] / "a.TextGrid",
                            captured["native_anchor_dir"] / "a.TextGrid"))


def test_resume_resample_uses_only_ctc_output_but_preserves_formal_partition(
        tmp_path, monkeypatch):
    """A resumed producer subset must not ask resample for filtered CTC stems."""
    source = tmp_path / "source"
    source.mkdir()
    for stem in ("a", "b"):
        with wave.open(str(source / f"{stem}.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 160)

    ctc = tmp_path / "canonical_ctc"
    ctc.mkdir()
    formal = run_pipeline.make_pipeline_accounting_receipt(
        ["a", "b"], ["a", "b"], [], ["a"], ["b"],
        run_id="canonical", mode="nvrasr_fallback", route=["prealign"])
    run_pipeline.write_pipeline_accounting_receipt(ctc, formal)
    (ctc / ".ctc_run_receipt.json").write_text(
        json.dumps({"schema": run_pipeline.CTC_RUN_RECEIPT_SCHEMA,
                    "input_stems": [], "output_stems": ["a"],
                    "audio_bindings": []}), encoding="utf-8")
    workspace = tmp_path / "workspace"
    output = workspace / "formal"
    mfa_audio = workspace / "audio_16k"
    transformed = []

    monkeypatch.setattr(run_pipeline, "_ensure_mfa_transform_receipts",
                        lambda ctx, stems: transformed.append(tuple(stems)) or 0)
    calls = []
    monkeypatch.setattr(run_pipeline, "run_python",
                        lambda *args, **kwargs: calls.append(args) or 99)
    ctx = {
        "mode": "nvrasr_fallback", "audio_dir": source,
        "mfa_audio_dir": mfa_audio, "ctc_pretg": ctc,
        "workspace": workspace, "expected_stems": (),
        "accounting_receipt_path": output / ".pipeline_run_receipt_v2.json",
        "accounting_source_receipt_path": ctc / ".pipeline_run_receipt_v2.json",
        "ctc_axis_receipt": {"schema": run_pipeline.CTC_RUN_RECEIPT_SCHEMA},
    }
    cfg = {"mfa": {"num_jobs": 1}}
    rc = run_pipeline.step_resample_for_mfa(
        SimpleNamespace(overwrite=True), cfg, Path(sys.executable), ctx)

    assert rc == 0
    assert sorted(p.name for p in mfa_audio.glob("*.wav")) == ["a.wav"]
    assert transformed == [("a",)]
    assert calls == []
    assert set(ctx["accounting_eligible_stems"]) == {"a", "b"}
    assert set(ctx["accounting_receipt"]["output"]["stems"]) == {"a"}
    assert set(ctx["accounting_receipt"]["filtered"]["stems"]) == {"b"}
