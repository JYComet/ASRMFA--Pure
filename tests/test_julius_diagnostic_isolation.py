from pathlib import Path
import hashlib
from types import SimpleNamespace
import wave
import numpy as np
import pytest

from scripts.run_julius_diagnostic import JATTS_COMMIT, JULIUS4SEG_COMMIT, convert_reading_to_julius, parse_lab, run_julius_diagnostic, validate_pcm16
from scripts.run_julius_diagnostic import _inside


def test_lab_parser_uses_100ns_ticks_and_keeps_diagnostic_namespace(tmp_path: Path):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
        out.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    assert validate_pcm16(audio)["sample_rate"] == 16000
    rows = parse_lab("0 5000000 s\n5000000 10000000 a\n", sample_rate=16000, timebase="100ns")
    assert rows[0]["start_sample"] == 0 and rows[0]["end_sample"] == 8000
    result = run_julius_diagnostic(audio, "さくら", tmp_path / "diag", enabled=False)
    assert result["status"] == "diagnostic_unavailable"
    assert result["production_write_back"] is False


def test_bad_pcm_is_rejected(tmp_path: Path):
    p = tmp_path / "bad.wav"
    with wave.open(str(p), "wb") as out:
        out.setnchannels(2); out.setsampwidth(2); out.setframerate(8000); out.writeframes(b"\0" * 32)
    try:
        validate_pcm16(p)
    except ValueError as exc:
        assert "PCM16" in str(exc) or "16000" in str(exc)
    else:
        raise AssertionError("invalid Julius input accepted")


def _assets(tmp_path: Path):
    paths = {name: tmp_path / name for name in ("binary", "model", "dictionary", "converter", "jatts")}
    for path in paths.values():
        path.write_text("def conv2julius(s): return 's a'\n" if path.name == "converter" else "fixture\n", encoding="utf-8")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    return paths, hashes


def test_disabled_namespace_does_not_expose_reading_or_commands(tmp_path: Path):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b"\0" * 3200)
    production = tmp_path / "production.json"
    production.write_text('{"timestamp": 1}', encoding="utf-8")
    before = production.read_bytes()
    result = run_julius_diagnostic(audio, "さ", tmp_path / "diag", enabled=False)
    assert set(result) == {"schema", "uid", "production_write_back", "status", "reason"}
    assert "selected_reading" not in result and "commands" not in result
    assert production.read_bytes() == before


def test_enabled_commit_only_trust_is_unavailable(tmp_path: Path):
    result = run_julius_diagnostic(tmp_path / "missing.wav", "さ", tmp_path / "diag", enabled=True, converter_commit=JULIUS4SEG_COMMIT, jatts_reference_commit=JATTS_COMMIT)
    assert result["status"] == "diagnostic_unavailable"
    assert "asset" in result["reason"] or "WAV" in result["reason"] or "required" in result["reason"]


def test_lab_rejects_malformed_and_nonmonotonic_rows():
    with pytest.raises(ValueError):
        parse_lab("0.0 0.1 a extra\n")
    with pytest.raises(ValueError):
        parse_lab("0.1 0.2 a\n0.15 0.3 b\n")
    with pytest.raises(ValueError, match=r"invalid \.lab time"):
        parse_lab("0 1 a\n", timebase="auto")
    with pytest.raises(ValueError, match=r"invalid \.lab time"):
        parse_lab("0.0 inf a\n")


def test_controlled_workflow_invocation_produces_receipt(tmp_path: Path):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b"\0" * 3200)
    paths, hashes = _assets(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        Path(kwargs["cwd"]) .joinpath("a.lab").write_text("0.0000000 0.0500000 s\n0.0500000 0.1000000 a\n", encoding="utf-8")
        return SimpleNamespace(stdout="fixture stdout", stderr="fixture stderr")

    semantic = [{"id": "p0", "unit_id": "u0", "julius_phone": "s"}, {"id": "p1", "unit_id": "u0", "julius_phone": "a"}]
    result = run_julius_diagnostic(audio, "さ", tmp_path / "diag", enabled=True, binary=paths["binary"], model=paths["model"], dictionary=paths["dictionary"], converter_path=paths["converter"], jatts_path=paths["jatts"], semantic_phones=semantic, selected_unit_ids=["u0"], mapping_provenance={"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT}, converter_commit=JULIUS4SEG_COMMIT, jatts_reference_commit=JATTS_COMMIT, asset_hashes=hashes, asset_licenses={key: "approved" for key in hashes}, workflow=["{jatts}", "--converter", "{converter}", "--binary", "{binary}", "--model", "{model}", "--dictionary", "{dictionary}", "--out", "{stage_dir}"], converter=lambda _: "s a", runner=runner)
    assert result["status"] == "COMPLETE"
    assert len(calls) == 1
    assert result["production_write_back"] is False
    assert result["commands"][0]["stdout"] == "fixture stdout"
    assert result["raw_intervals"][0]["start_sample"] == 0


def test_native_phone_passthrough_is_rejected(tmp_path: Path):
    paths, _ = _assets(tmp_path)
    with pytest.raises(ValueError, match="Julius mapping|source event"):
        from scripts.run_julius_diagnostic import convert_reading_to_julius
        convert_reading_to_julius("さ", semantic_phones=[{"id": "p0", "unit_id": "u0", "phone": "s"}], converter=lambda _: "s", converter_commit=JULIUS4SEG_COMMIT, mapping_provenance={"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT}, selected_unit_ids=["u0"])


def test_graph_source_events_drive_pinned_converter_and_katakana_provenance(tmp_path: Path):
    graph = {
        "schema": "ja-semantic-phone-graph-v1",
        "frontend_phone_nodes": [
            {"id": "oj0", "phone": "t"}, {"id": "oj1", "phone": "o"}, {"id": "oj1b", "phone": "o"},
            {"id": "oj2", "phone": "k"}, {"id": "oj3", "phone": "y"},
            {"id": "oj4", "phone": "o"}, {"id": "oj4b", "phone": "o"},
        ],
        "semantic_phone_nodes": [
            {"id": "p0", "unit_id": "u0", "source_openjtalk_ids": ["oj0"]},
            {"id": "p1", "unit_id": "u0", "source_openjtalk_ids": ["oj1", "oj1b"]},
            {"id": "p2", "unit_id": "u0", "source_openjtalk_ids": ["oj2", "oj3"]},
            {"id": "p3", "unit_id": "u0", "source_openjtalk_ids": ["oj4", "oj4b"]},
        ],
    }
    converter_source = tmp_path / "converter.py"
    converter_source.write_text("def conv2julius(reading):\n    if reading in {'とうきょう', 'とーきょー'}: return 't o: ky o:'\n    raise ValueError(reading)\n", encoding="utf-8")
    result = convert_reading_to_julius("トウキョウ", semantic_graph=graph, converter_path=converter_source, converter_commit=JULIUS4SEG_COMMIT, mapping_provenance={"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT}, selected_unit_ids=["u0"])
    assert result["converter_reading"] == "とうきょう"
    assert result["phones"] == ["t", "o:", "ky", "o:"]
    assert result["reading_normalization"]["method"] == "katakana_to_hiragana_codepoint"
    long_result = convert_reading_to_julius("とーきょー", semantic_graph=graph, converter_path=converter_source, converter_commit=JULIUS4SEG_COMMIT, mapping_provenance={"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT}, selected_unit_ids=["u0"])
    assert long_result["converter_reading"] == "とーきょー"
    assert long_result["phones"] == ["t", "o:", "ky", "o:"]
    with pytest.raises(ValueError, match="hidden G2P"):
        convert_reading_to_julius("東京", semantic_graph=graph, converter_path=converter_source, converter_commit=JULIUS4SEG_COMMIT, mapping_provenance={"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT}, selected_unit_ids=["u0"])


@pytest.mark.skipif(not Path("/home/user/ja-en-task-artifacts/julius4seg-converter.py").is_file(), reason="pinned converter artifact is unavailable")
def test_actual_pinned_converter_sourcegraph_smoke():
    converter = Path("/home/user/ja-en-task-artifacts/julius4seg-converter.py")

    def graph(groups):
        source_nodes = []
        semantic_nodes = []
        for index, events in enumerate(groups):
            ids = []
            for event_index, event in enumerate(events):
                node_id = f"oj{index}_{event_index}"
                ids.append(node_id)
                source_nodes.append({"id": node_id, "phone": event})
            semantic_nodes.append({"id": f"p{index}", "unit_id": "u0", "source_openjtalk_ids": ids})
        return {"schema": "ja-semantic-phone-graph-v1", "frontend_phone_nodes": source_nodes, "semantic_phone_nodes": semantic_nodes}

    provenance = {"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT, "converter_sha256": "066bd90c23c4fb8683fb9e7ddcf66fc6156b12d452570034c06395b301ca74a3"}
    cases = [
        ("とうきょう", [["t"], ["o", "o"], ["k", "y"], ["o", "o"]], ["t", "o", "u", "ky", "o", "u"]),
        ("がっこう", [["g"], ["a"], ["cl", "k"], ["o", "o"]], ["g", "a", "q", "k", "o", "u"]),
        ("さくら", [["s"], ["a"], ["k"], ["u"], ["r"], ["a"]], ["s", "a", "k", "u", "r", "a"]),
    ]
    for reading, groups, expected in cases:
        result = convert_reading_to_julius(reading, semantic_graph=graph(groups), converter_path=converter, converter_commit=JULIUS4SEG_COMMIT, mapping_provenance=provenance, selected_unit_ids=["u0"])
        assert result["phones"] == expected
        assert result["mapping_provenance"]["converter_sha256"] == provenance["converter_sha256"]


def test_stage_parent_symlink_and_stale_workflow_are_rejected(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        run_julius_diagnostic(tmp_path / "missing.wav", "さ", linked / "diag", enabled=False)
    with pytest.raises(ValueError, match="escapes"):
        _inside("../outside.lab", tmp_path / "stage")

    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b"\0" * 3200)
    paths, hashes = _assets(tmp_path)
    stale = tmp_path / "stale"
    (stale / "workflow").mkdir(parents=True)
    (stale / "workflow" / "old.lab").write_text("0 1 a\n", encoding="utf-8")
    result = run_julius_diagnostic(audio, "さ", stale, enabled=True, binary=paths["binary"], model=paths["model"], dictionary=paths["dictionary"], converter_path=paths["converter"], jatts_path=paths["jatts"], semantic_phones=[{"id": "p", "unit_id": "u", "julius_phone": "s"}], selected_unit_ids=["u"], mapping_provenance={"method": "conv2julius", "converter_commit": JULIUS4SEG_COMMIT}, converter_commit=JULIUS4SEG_COMMIT, jatts_reference_commit=JATTS_COMMIT, asset_hashes=hashes, asset_licenses={key: "approved" for key in hashes}, workflow=["{jatts}", "{binary}", "{model}", "{dictionary}"] , converter=lambda _: "s", runner=lambda *a, **k: SimpleNamespace(stdout="", stderr=""))
    assert result["status"] == "diagnostic_unavailable"
    assert "fresh" in result["reason"]
