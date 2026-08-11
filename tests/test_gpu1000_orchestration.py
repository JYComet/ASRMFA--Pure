from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import analyze_gpu1000_run as analyze
from scripts import gpu1000_orchestrate as tool


def _source(root: Path, *, extras: int = 0) -> Path:
    source = root / "source"
    for speaker, count in tool.QUOTAS.items():
        folder = source / speaker; folder.mkdir(parents=True)
        for index in range(count + extras):
            stem = f"{speaker}_{index:04d}"
            (folder / f"{stem}.wav").write_bytes((stem + " wav").encode())
            (folder / f"{stem}.txt").write_text("reference", encoding="utf-8")
    return source


def _config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("mode: nvrasr_fallback\noutput_dir: /mnt/Raw/unsafe\nctc_prealign: {}\n", encoding="utf-8")
    return path


def _fake_gpus(path: Path, *, count: int = 8, busy: bool = False) -> Path:
    path.write_text(json.dumps({"gpus": [{"index": i, "memory_free_mib": 50000,
                                           "compute_pids": ["123"] if busy and i == 0 else []}
                                          for i in range(count)]}), encoding="utf-8")
    return path


def _prepare(tmp_path: Path) -> Path:
    root = tmp_path / "gpu1000_test"
    args = type("Args", (), {"root": root, "source": _source(tmp_path), "config": _config(tmp_path / "base.yaml")})()
    assert tool.prepare(args) == 0
    return root


def _bucket(stems):
    ordered = sorted(stems)
    return {"count": len(ordered), "stems": ordered, "stems_digest": analyze._digest(ordered)}


def _receipt(stems, output, filtered, *, shards=None, paths=None):
    value = {"schema": "pipeline-run-receipt-v2", "source": _bucket(stems), "eligible": _bucket(stems),
             "exclusions": [], "output": _bucket(output), "filtered": _bucket(filtered), "paths": paths or {}}
    if shards is not None:
        value["shards"] = [{"shard_id": f"gpu{row['gpu']}", "count": len(row["stems"]),
                            "stems": sorted(row["stems"]), "stems_digest": analyze._digest(sorted(row["stems"]))}
                           for row in shards]
    return value


def _nested_rc0_evidence(root: Path, *, telemetry=True):
    stems = json.loads((root / "selected_stems.json").read_text(encoding="utf-8"))["stems"]
    shards = json.loads((root / "shard_plan.json").read_text(encoding="utf-8"))["shards"]
    workspace = root / "workspace"; ctc = workspace / "ctc_pretg"; ctc.mkdir(parents=True)
    (ctc / ".pipeline_run_receipt_v2.json").write_text(json.dumps(_receipt(stems, stems, [], shards=shards)), encoding="utf-8")
    output = workspace / "strict_ok_runs" / "run" / "output"; filtered = output.parent / "filtered"; output.mkdir(parents=True); filtered.mkdir()
    paths = {"output": str(output.resolve()), "filtered": str(filtered.resolve())}
    (output / ".pipeline_run_receipt_v2.json").write_text(json.dumps(_receipt(stems, stems, [], paths=paths)), encoding="utf-8")
    (output / "strict_ok_manifest.json").write_text(json.dumps({"ok": [{"stem": stem} for stem in stems], "rejected": {}, "global_reasons": []}), encoding="utf-8")
    (root / "run_receipt.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
    if telemetry:
        rows = [{"at_utc": f"2026-01-01T00:00:0{i}Z", "gpus": [{"index": gpu, "memory_free_mib": 49000,
                                                                        "memory_used_mib": 1000, "utilization_gpu_pct": 50,
                                                                        "compute_pids": []} for gpu in range(8)]}
                for i in range(2)]
        (root / "nvidia-smi.telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return stems, shards, output


def _child_ctc_receipt(child: Path, stems):
    ordered = sorted(stems)
    selected = str((child / "selected_stems.txt").resolve())
    return {"schema": "ctc-run-receipt-v2", "argv": ["python", "ctc_prealign.py", "--output-dir", str(child.resolve()),
                                                           "--stems-file", selected],
            "input_stems": ordered, "input_stems_digest": analyze._digest(ordered),
            "output_stems": ordered, "output_stems_digest": analyze._digest(ordered)}


def test_prepare_is_deterministic_copies_and_rejects_existing_root(tmp_path):
    root = _prepare(tmp_path)
    manifest = json.loads((root / "selected_manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 1000
    assert {row["speaker"] for row in manifest["samples"]} == set(tool.SPEAKERS)
    input_entries = list((root / "input").iterdir())
    assert len(input_entries) == 2000 and all(path.is_file() and not path.is_symlink() for path in input_entries)
    assert not any(path.is_dir() for path in input_entries)
    assert "audio_dir.iterdir()" in (tool.PROJECT / "scripts/run_pipeline.py").read_text(encoding="utf-8")
    again = type("Args", (), {"root": root, "source": tmp_path / "source", "config": tmp_path / "base.yaml"})()
    try:
        tool.prepare(again)
    except tool.SafetyError as exc:
        assert "must not exist" in str(exc)
    else: raise AssertionError("existing root was accepted")
    root2 = tmp_path / "gpu1000_repeat"
    args = type("Args", (), {"root": root2, "source": tmp_path / "source", "config": tmp_path / "base.yaml"})()
    assert tool.prepare(args) == 0
    assert json.loads((root / "selected_manifest.json").read_text(encoding="utf-8"))["selection_digest"] == json.loads((root2 / "selected_manifest.json").read_text(encoding="utf-8"))["selection_digest"]


def test_prepare_records_unpaired_when_quota_remains_and_fails_when_it_does_not(tmp_path):
    source = _source(tmp_path, extras=1)
    (source / "ria" / "ria_0000.txt").unlink()
    root = tmp_path / "unpaired_ok"
    args = type("Args", (), {"root": root, "source": source, "config": _config(tmp_path / "base.yaml")})()
    assert tool.prepare(args) == 0
    inventory = json.loads((root / "source_inventory.json").read_text(encoding="utf-8"))
    assert {row["reason"] for row in inventory["exclusions"]} >= {"missing_sibling_txt"}
    insufficient = _source(tmp_path / "insufficient")
    (insufficient / "ria" / "ria_0000.txt").unlink()
    bad_root = tmp_path / "insufficient_root"
    args = type("Args", (), {"root": bad_root, "source": insufficient, "config": _config(tmp_path / "insufficient/base.yaml")})()
    try: tool.prepare(args)
    except tool.SafetyError as exc: assert "eligible samples" in str(exc)
    else: raise AssertionError("insufficient paired inventory accepted")
    assert not bad_root.exists()


def test_prepare_rejects_duplicate_global_stem(tmp_path):
    source = _source(tmp_path / "second")
    (source / "花礼" / "ria_0000.wav").write_bytes(b"duplicate")
    (source / "花礼" / "ria_0000.txt").write_text("duplicate", encoding="utf-8")
    args = type("Args", (), {"root": tmp_path / "duplicate", "source": source, "config": _config(tmp_path / "second/base.yaml")})()
    try: tool.prepare(args)
    except tool.SafetyError as exc: assert "duplicate global eligible stem" in str(exc)
    else: raise AssertionError("duplicate stem accepted")
    assert not (tmp_path / "duplicate").exists()


def test_flat_input_validation_rejects_subdirectories_and_tamper(tmp_path):
    root = _prepare(tmp_path)
    (root / "input" / "nested").mkdir()
    assert "flat_input_namespace_not_exact" in tool.validate_prepared(root)
    (root / "input" / "nested").rmdir()
    wav = next((root / "input").glob("*.wav")); wav.write_bytes(b"tampered")
    assert any(error.startswith("input_hash_mismatch:") for error in tool.validate_prepared(root))


def test_shard_plan_uses_sorted_flat_ctc_filename_order_not_speaker_grouping(tmp_path):
    source = _source(tmp_path)
    # Force an interleaving that differs from the fixed ria/花礼/雪狐桑
    # selection grouping while preserving all pair and quota invariants.
    folder = source / "花礼"
    (folder / "花礼_0000.wav").rename(folder / "0000_cross_speaker.wav")
    (folder / "花礼_0000.txt").rename(folder / "0000_cross_speaker.txt")
    root = tmp_path / "flat_order"
    args = type("Args", (), {"root": root, "source": source, "config": _config(tmp_path / "base.yaml"),
                              "count": None, "selected_stems": None})()
    assert tool.prepare(args) == 0
    plan = json.loads((root / "shard_plan.json").read_text(encoding="utf-8"))["shards"]
    flat = sorted((root / "input").glob("*.wav"), key=lambda path: path.name)
    assert [stem for shard in plan for stem in shard["stems"]] == [path.stem for path in flat]
    assert plan[0]["stems"][0] == "0000_cross_speaker"
    _nested_rc0_evidence(root)
    assert analyze.analyze(root)["ok"]


def test_confirmation_count_is_labeled_nonpublishing_and_sharded(tmp_path):
    root = tmp_path / "confirmation"
    args = type("Args", (), {"root": root, "source": _source(tmp_path), "config": _config(tmp_path / "base.yaml"),
                              "count": 8, "selected_stems": None})()
    assert tool.prepare(args) == 0
    manifest = json.loads((root / "selected_manifest.json").read_text(encoding="utf-8"))
    shards = json.loads((root / "shard_plan.json").read_text(encoding="utf-8"))["shards"]
    assert manifest["run_label"] == "confirmation_nonpublish" and manifest["count"] == 8
    assert [len(shard["stems"]) for shard in shards] == [1] * 8
    assert tool.validate_prepared(root) == []
    _nested_rc0_evidence(root)
    assert analyze.analyze(root)["ok"]


def test_hash_tamper_preflight_gpu_safety_and_dry_run(tmp_path):
    root = _prepare(tmp_path); fake = _fake_gpus(tmp_path / "gpus.json")
    assert tool.preflight(type("Args", (), {"root": root, "fake_gpus": str(fake)})()) == 0
    assert tool.run(type("Args", (), {"root": root, "python": "python", "fake_gpus": str(fake), "dry_run": True})()) == 0
    argv = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))["argv"]
    assert argv[argv.index("--python") + 1] == "python"
    try: tool.run(type("Args", (), {"root": root, "python": "python", "fake_gpus": str(fake), "dry_run": True})())
    except tool.SafetyError as exc: assert "cannot be run twice" in str(exc)
    else: raise AssertionError("run retry accepted")
    second = _prepare(tmp_path / "again")
    busy = _fake_gpus(tmp_path / "busy.json", busy=True)
    try: tool.preflight(type("Args", (), {"root": second, "fake_gpus": str(busy)})())
    except tool.SafetyError as exc: assert "compute_pids" in str(exc)
    else: raise AssertionError("occupied gpu accepted")


def test_gpu_queries_are_separate_map_pids_and_preserve_diagnostics():
    calls = []
    original = tool.subprocess.run
    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu":
            return SimpleNamespace(returncode=0, stdout="0, GPU-a, 50000, 100, 20\n1, GPU-b, 50001, 101, 0\n", stderr="")
        if argv[1] == "--query-compute-apps=pid,gpu_uuid":
            return SimpleNamespace(returncode=0, stdout="42, GPU-b\n", stderr="")
        raise AssertionError(f"unexpected query: {argv}")
    tool.subprocess.run = fake_run
    try:
        rows = tool.gpu_snapshot()
    finally:
        tool.subprocess.run = original
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[1]["compute_pids"] == ["42"]
    assert all("compute_applications.pid" not in part for argv in calls for part in argv)
    assert calls[0][1] == "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu"
    assert calls[1][1] == "--query-compute-apps=pid,gpu_uuid"


def test_mfa_dependency_uses_sibling_interpreter_environment(tmp_path):
    python = tmp_path / "mfa-dev" / "bin" / "python"; python.parent.mkdir(parents=True)
    mfa = python.parent / "mfa"; mfa.write_text("fake", encoding="utf-8"); mfa.chmod(0o755)
    resolved, method = tool.resolve_mfa_dependency(str(python))
    assert resolved == str(mfa) and method == "sibling_executable"
    mfa.unlink()
    original = tool.subprocess.run
    tool.subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=7, stdout="", stderr="module missing")
    try:
        resolved, diagnostic = tool.resolve_mfa_dependency(str(python))
    finally:
        tool.subprocess.run = original
    assert resolved is None and "module missing" in diagnostic


def test_analyzer_detects_nested_ambiguity_tamper_and_telemetry_then_accepts(tmp_path):
    root = _prepare(tmp_path)
    stems, shards, output = _nested_rc0_evidence(root)
    report = analyze.analyze(root)
    assert report["ok"] and report["output_count"] == 1000 and report["publication"] == "forbidden"
    duplicate = root / "workspace" / "strict_ok_runs" / "other" / "output"; duplicate.mkdir(parents=True)
    (duplicate / ".pipeline_run_receipt_v2.json").write_text((output / ".pipeline_run_receipt_v2.json").read_text(encoding="utf-8"), encoding="utf-8")
    assert "strict_output_receipt_not_exactly_one" in analyze.analyze(root)["errors"]
    (duplicate / ".pipeline_run_receipt_v2.json").unlink(); duplicate.rmdir(); duplicate.parent.rmdir()
    ctc = root / "workspace" / "ctc_pretg" / ".pipeline_run_receipt_v2.json"
    receipt = json.loads(ctc.read_text(encoding="utf-8")); receipt["shards"][1]["stems"][0] = receipt["shards"][0]["stems"][0]
    ctc.write_text(json.dumps(receipt), encoding="utf-8")
    assert any("ctc_gpu1" in error or "ctc_shard_overlap" == error for error in analyze.analyze(root)["errors"])
    ctc.write_text(json.dumps(_receipt(stems, stems, [], shards=shards)), encoding="utf-8")
    (root / "nvidia-smi.telemetry.jsonl").write_text(json.dumps({"at_utc": "now", "gpus": []}) + "\n", encoding="utf-8")
    assert "gpu0_telemetry_activity_below_2" in analyze.analyze(root)["errors"]


def test_analyzer_accepts_validated_quarantined_child_receipts_when_old_telemetry_misses_short_work(tmp_path):
    root = _prepare(tmp_path); _, plan, _ = _nested_rc0_evidence(root)
    # A 15-second sample can see only insignificant driver bookkeeping drift.
    rows = [{"at_utc": f"2026-01-01T00:00:0{i}Z", "gpus": [{"index": gpu, "memory_free_mib": 50000 - i * 3,
                                                                        "memory_used_mib": 1000 + i * 3, "utilization_gpu_pct": 0,
                                                                        "compute_pids": []} for gpu in range(8)]} for i in range(2)]
    (root / "nvidia-smi.telemetry.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    quarantine = root / "workspace" / "ctc_pretg.partial-999"
    for shard in plan:
        child = quarantine / f"_shard_gpu{shard['gpu']}"; child.mkdir(parents=True)
        (child / "selected_stems.txt").write_text("\n".join(shard["stems"]) + "\n", encoding="utf-8")
        (child / ".ctc_run_receipt.json").write_text("{}", encoding="utf-8")
        (child / "summary.txt").write_text(f"Files: {len(shard['stems'])} total, {len(shard['stems'])} OK, 0 failed\n", encoding="utf-8")
    assert "gpu0_telemetry_activity_below_2" in analyze.analyze(root)["errors"]
    for shard in plan:
        child = quarantine / f"_shard_gpu{shard['gpu']}"
        (child / ".ctc_run_receipt.json").write_text(json.dumps(_child_ctc_receipt(child, shard["stems"])), encoding="utf-8")
    report = analyze.analyze(root)
    assert report["ok"] and set(report["gpu_activity_evidence"].values()) == {"quarantined_child_receipt"}
    child = quarantine / "_shard_gpu0"; receipt = json.loads((child / ".ctc_run_receipt.json").read_text(encoding="utf-8"))
    receipt["output_stems_digest"] = "0" * 64
    (child / ".ctc_run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert "gpu0_telemetry_activity_below_2" in analyze.analyze(root)["errors"]
    receipt = _child_ctc_receipt(child, plan[0]["stems"]); receipt["output_stems"] = receipt["output_stems"][1:]
    (child / ".ctc_run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert "gpu0_telemetry_activity_below_2" in analyze.analyze(root)["errors"]
