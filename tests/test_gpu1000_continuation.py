from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import gpu1000_orchestrate as tool
from scripts import analyze_gpu1000_run as analyzer


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    root = tmp_path / "failed"
    (root / "input").mkdir(parents=True)
    (root / "workspace" / "ctc_pretg").mkdir(parents=True)
    stems = [f"s{i:04d}" for i in range(1000)]
    samples = []
    for stem in stems:
        wav, txt = root / "input" / f"{stem}.wav", root / "input" / f"{stem}.txt"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(1); handle.setframerate(16000); handle.writeframes(b"\0" * 16000)
        txt.write_text(stem, encoding="utf-8")
        samples.append({"speaker": "s", "stem": stem, "source_relative_wav": f"{stem}.wav",
                        "source_relative_txt": f"{stem}.txt", "wav_sha256": tool.sha_file(wav),
                        "txt_sha256": tool.sha_file(txt), "wav_destination": str(wav), "txt_destination": str(txt)})
    identity = [{k: row[k] for k in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")} for row in samples]
    (root / "selected_manifest.json").write_text(json.dumps({"count": 1000, "run_label": "full1000", "samples": samples,
        "selection_digest": tool.digest(identity)}), encoding="utf-8")
    valid_grid = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "yi1"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "yi"
'''
    for stem in stems[:-1]: (root / "workspace" / "ctc_pretg" / f"{stem}.TextGrid").write_text(valid_grid, encoding="utf-8")
    (root / "source_inventory.json").write_text("{}")
    (root / "prepare_evidence.json").write_text(json.dumps({"run_pipeline_sha256": tool.sha_file(tool.PROJECT / "scripts/run_pipeline.py"),
        "ctc_prealign_sha256": tool.sha_file(tool.PROJECT / "scripts/ctc_prealign.py")}))
    (root / "shard_plan.json").write_text("{}")
    (root / "resolved_gpu1000_nvrasr_fallback.yaml").write_text("mode: nvrasr_fallback\n")
    (root / "run_receipt.json").write_text(json.dumps({"schema": "gpu1000-run-v1", "returncode": 1}))
    scope_grid = tmp_path / "s0999.TextGrid"; scope_grid.write_text(valid_grid, encoding="utf-8")
    # Retained retry input is current-root evidence; the selected source copy
    # remains under input/ to exercise path-different proof binding.
    retry = root / "workspace" / "mfa_shards" / "run" / "retry_missing"
    retry.mkdir(parents=True)
    (retry / "s0999.lab").write_text("s0999", encoding="utf-8")
    (retry / "s0999.wav").write_bytes(b"s0999")
    scope = tmp_path / "scope.json"; scope.write_text(json.dumps({"schema": tool.CONTINUATION_SCOPE_SCHEMA,
        "stems": [stems[-1]], "textgrid": str(scope_grid)}), encoding="utf-8")
    return root, scope, stems


def _store_preflight(root: Path, scope: Path) -> None:
    exe = root / "mfa"; dep = root / "fstcompile"; dictionary = root / "dict"; model = root / "model.zip"
    for path in (exe, dep, dictionary, model): path.write_bytes(path.name.encode())
    env_root = root / "mfa_root"; cache = root / "numba_cache"; env_root.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    stem = json.loads(scope.read_text(encoding="utf-8"))["stems"][0]
    lab = root / "proof-input" / f"{stem}.lab"
    wav = root / "proof-input" / f"{stem}.wav"
    lab.parent.mkdir(exist_ok=True)
    current_lab = root / "workspace" / "mfa_shards" / "run" / "retry_missing" / f"{stem}.lab"
    current_wav = root / "workspace" / "mfa_shards" / "run" / "retry_missing" / f"{stem}.wav"
    lab.write_text(current_lab.read_text(encoding="utf-8"), encoding="utf-8")
    wav.write_bytes(current_wav.read_bytes())
    proof = root / "proof.json"; proof.write_text(json.dumps({"schema": "mfa-retry-evidence-v1", "command": [str(exe), "align", "corpus", str(dictionary), str(model), "output", "--beam", "20", "--retry_beam", "80"],
        "mfa_executable": {"path": str(exe), "sha256": tool.sha_file(exe)}, "mfa_dependency": {"path": str(dep), "sha256": tool.sha_file(dep)},
        "dictionary": {"path": str(dictionary), "sha256": tool.sha_file(dictionary)}, "model": {"path": str(model), "sha256": tool.sha_file(model)},
        "environment": {"MFA_ROOT_DIR": str(env_root), "NUMBA_CACHE_DIR": str(cache)},
        "inputs": [{"stem": stem, "role": "anchor.lab", "source": str(lab), "sha256": tool.sha_file(lab)},
                   {"stem": stem, "role": "mfa_axis_audio", "source": str(wav), "sha256": tool.sha_file(wav)}]}), encoding="utf-8")
    receipt = tool.continuation_preflight(root, scope, proven_mfa_retry_receipt=proof)
    (root / "continuation_preflight_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def _store_preflight_v2(root: Path, scope: Path) -> None:
    tool.continuation_preflight_v2(root, scope)


def test_continuation_preflight_requires_exact_one_and_binds_999(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight(root, scope)
    receipt = json.loads((root / "continuation_preflight_receipt.json").read_text())
    assert receipt["ok"] and receipt["original_grid_count"] == 999
    bad = tmp_path / "bad_scope.json"; bad.write_text(json.dumps({"stems": ["s0999", "s0998"]}), encoding="utf-8")
    proof = root / "proof.json"
    blocked = tool.continuation_preflight(root, bad, proven_mfa_retry_receipt=proof)
    assert not blocked["ok"] and any("exactly one" in error for error in blocked["errors"])


def test_discover_retry_plan_is_read_only_and_returns_original_path(tmp_path: Path):
    root, _, stems = _fixture(tmp_path)
    logs = root / "workspace" / "mfa_logs" / "run"
    logs.mkdir(parents=True)
    plan = logs / "mfa_missing_retry_plan.json"
    plan.write_text(json.dumps({"coordinator": {"history": [{"missing": [stems[-1]]}]}}), encoding="utf-8")
    scratch = tmp_path / "tmp-scratch"
    scratch.mkdir()
    with patch.object(tool.tempfile, "gettempdir", return_value=str(scratch)):
        returned, scope = tool._discover_continuation_scope(root, None)
    assert returned == plan.resolve()
    assert scope["stems"] == [stems[-1]]
    assert list(scratch.iterdir()) == []


def test_continue_refuses_stale_v1_without_v2(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    try:
        tool.continue_after_mfa(root, scope)
    except tool.SafetyError as exc:
        assert "v2" in str(exc)
        return
    raise AssertionError("v1-only continuation was accepted")


def test_v2_selection_and_tamper_are_bound(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight_v2(root, scope)
    v2_path = root / "continuation_preflight_receipt_v2.json"
    value = json.loads(v2_path.read_text())
    value["v1_binding"]["sha256"] = "0" * 64
    os.chmod(v2_path, 0o644)
    v2_path.write_text(json.dumps(value), encoding="utf-8")
    try:
        tool.continue_after_mfa(root, scope)
    except tool.SafetyError as exc:
        assert "v1 preflight binding" in str(exc)
        return
    raise AssertionError("tampered v2 binding was accepted")


def test_v2_accepts_empty_canonical_dirs_and_historical_strict(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    (root / "workspace" / "aligned").mkdir()
    (root / "workspace" / "en_phones").mkdir()
    historical = root / "workspace" / "strict_ok_runs" / "historical" / "output"
    historical.mkdir(parents=True)
    (historical / ".pipeline_run_receipt_v2.json").write_text(
        json.dumps({"schema": "pipeline-run-receipt-v2"}), encoding="utf-8")
    receipt = tool.continuation_preflight_v2(root, scope, proven_mfa_retry_receipt=root / "proof.json")
    assert receipt["ok"] and receipt["schema"] == tool.CONTINUATION_PREFLIGHT_V2_SCHEMA


def _prepare_downstream_resume(root: Path, scope: Path, *, valid_grids: bool = False, legacy_resume: bool = False) -> None:
    if valid_grids:
        valid = '''File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n item [1]:\n  class = "IntervalTier"\n  name = "words"\n  xmin = 0\n  xmax = 1\n  intervals: size = 1\n  intervals [1]:\n   xmin = 0\n   xmax = 1\n   text = "yi1"\n item [2]:\n  class = "IntervalTier"\n  name = "phones"\n  xmin = 0\n  xmax = 1\n  intervals: size = 1\n  intervals [1]:\n   xmin = 0\n   xmax = 1\n   text = "yi"\n'''
        for grid in (root / "workspace" / "ctc_pretg").glob("*.TextGrid"):
            grid.write_text(valid, encoding="utf-8")
    _store_preflight(root, scope)
    _store_preflight_v2(root, scope)
    tool.continue_after_mfa(root, scope)
    attempts = {"schema": "gpu1000-singleton-attempts-v1", "attempts": [
        {"attempt": "20_80", "returncode": 1, "stderr": "NoAlignmentsError: no path",
         "argv": ["mfa", "align", "corpus", "dict", "model", "out", "--beam", "20", "--retry_beam", "80"]},
        {"attempt": "200_800", "returncode": 0, "stderr": "",
         "argv": ["mfa", "align", "corpus", "dict", "model", "out", "--beam", "200", "--retry_beam", "800"]},
    ]}
    (root / "workspace" / "continuation_singleton_attempts.json").write_text(json.dumps(attempts), encoding="utf-8")
    historical = root / "workspace" / "strict_ok_runs" / "historical" / "output"
    historical.mkdir(parents=True)
    (historical / ".pipeline_run_receipt_v2.json").write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}), encoding="utf-8")
    failed = root / "workspace" / "strict_ok_runs" / "continuation_20260811T132821Z" / "output"
    failed.mkdir(parents=True)
    (failed / ".pipeline_run_receipt_v2.json").write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}), encoding="utf-8")
    (failed / "postprocess_report.jsonl").write_text("{}\n", encoding="utf-8")
    (failed / "strict_ok_manifest.json").write_text("{}", encoding="utf-8")
    tool.downstream_resume_preflight(root)
    if legacy_resume:
        resume_path = root / "downstream_resume_preflight_receipt.json"
        legacy = json.loads(resume_path.read_text())
        for key in ("alignment_axis_path", "alignment_axis_sha256", "failed_strict_path", "failed_strict_sha256", "failed_report_sha256", "failed_manifest_sha256", "english_manifest_path", "english_manifest_sha256", "english_ledger_digest", "orchestrator_delta_reason"):
            legacy.pop(key, None)
        legacy["current_code_hashes"]["orchestrator_sha256"] = "legacy-orchestrator"
        os.chmod(resume_path, 0o644); resume_path.write_text(json.dumps(legacy), encoding="utf-8")


def test_downstream_resume_success_and_tamper(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _prepare_downstream_resume(root, scope, valid_grids=True)
    stored = json.loads((root / "downstream_resume_preflight_receipt.json").read_text())
    assert stored["ok"] and stored["schema"] == "gpu1000-downstream-resume-preflight-v1"
    def fake_downstream(run_root, python, **kwargs):
        output = run_root / "workspace" / "strict_ok_runs" / "continuation_new" / "output"
        output.mkdir(parents=True)
        receipt = output / ".pipeline_run_receipt_v2.json"
        receipt.write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}), encoding="utf-8")
        return {"strict_receipt": str(receipt), "strict_receipt_sha256": tool.sha_file(receipt)}
    with patch.object(tool, "_run_permitted_downstream", side_effect=fake_downstream):
        result = tool.continue_downstream(root)
    assert result["status"] == "PASS_WITH_CONTINUATION"

    root2, scope2, _ = _fixture(tmp_path / "tamper")
    _prepare_downstream_resume(root2, scope2, valid_grids=True)
    resume_path = root2 / "downstream_resume_preflight_receipt.json"
    value = json.loads(resume_path.read_text()); value["v2_sha256"] = "0" * 64
    os.chmod(resume_path, 0o644); resume_path.write_text(json.dumps(value), encoding="utf-8")
    try:
        tool.continue_downstream(root2)
    except tool.SafetyError:
        pass
    else:
        raise AssertionError("tampered downstream resume receipt was accepted")


def test_downstream_resume_allows_orchestrator_delta_after_v2(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _prepare_downstream_resume(root, scope, valid_grids=True)
    v2_path = root / "continuation_preflight_receipt_v2.json"
    v2 = json.loads(v2_path.read_text())
    v2["fresh_preflight"]["current_code_hashes"]["orchestrator_sha256"] = "old-orchestrator-proof"
    v2["fresh_preflight_digest"] = tool.digest(v2["fresh_preflight"])
    os.chmod(v2_path, 0o644); v2_path.write_text(json.dumps(v2), encoding="utf-8")
    v2_digest = tool.digest(v2); v2_sha = tool.sha_file(v2_path)
    started_path = root / "continuation_started.json"
    started = json.loads(started_path.read_text()); started["preflight_v2_sha256"] = v2_sha; started["fresh_preflight_digest"] = v2["fresh_preflight_digest"]
    os.chmod(started_path, 0o644); started_path.write_text(json.dumps(started), encoding="utf-8")
    merge_path = root / "alignment_merge_receipt.json"
    merge = json.loads(merge_path.read_text()); merge["preflight_v2_sha256"] = v2_sha; merge["preflight_v2_digest"] = v2_digest; merge["fresh_preflight_digest"] = v2["fresh_preflight_digest"]
    os.chmod(merge_path, 0o644); merge_path.write_text(json.dumps(merge), encoding="utf-8")
    assert tool._downstream_resume_preflight(root)["ok"]


def test_axis_recovery_corrects_missing_fields_and_binds_new_resume(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _prepare_downstream_resume(root, scope, valid_grids=True, legacy_resume=True)
    old_axis = root / "workspace" / ".mfa_alignment_axis_receipt.json"
    old_hash = tool.sha_file(old_axis)
    recovery_preflight = tool.axis_recovery_preflight(root)
    assert recovery_preflight["needs_repair"] and recovery_preflight["missing_required_fields"] == 1000
    chain = tool.recover_alignment_axis(root)
    corrected = Path(chain["corrected_axis_path"])
    payload = json.loads(corrected.read_text())
    assert tool.sha_file(old_axis) == old_hash and corrected != old_axis
    assert all("audio_sha256" in row and "xmax" in row for row in payload["alignments"])
    v2 = tool.downstream_resume_preflight_v2(root)
    assert v2["ok"] and v2["alignment_axis_path"] == str(corrected.resolve())
    seen: dict[str, str] = {}
    def fake_downstream(run_root, python, **kwargs):
        seen["alignment_axis_path"] = str(kwargs.get("alignment_axis_path"))
        receipt = run_root / "workspace" / "strict_ok_runs" / "continuation_new" / "output" / ".pipeline_run_receipt_v2.json"
        receipt.parent.mkdir(parents=True); receipt.write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}), encoding="utf-8")
        return {"strict_receipt": str(receipt), "strict_receipt_sha256": tool.sha_file(receipt)}
    with patch.object(tool, "_run_permitted_downstream", side_effect=fake_downstream):
        result = tool.continue_downstream(root)
    assert result["status"] == "PASS_WITH_CONTINUATION" and seen["alignment_axis_path"] == str(corrected.resolve())


def test_axis_recovery_rejects_grid_or_audio_tamper(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _prepare_downstream_resume(root, scope)
    grid = next((root / "workspace" / "aligned").glob("*.TextGrid"))
    grid.write_text(grid.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try: tool.recover_alignment_axis(root)
    except tool.SafetyError: pass
    else: raise AssertionError("grid tamper accepted")
    root2, scope2, _ = _fixture(tmp_path / "audio")
    _prepare_downstream_resume(root2, scope2, valid_grids=True)
    audio = next((root2 / "input").glob("*.wav"))
    audio.write_bytes(audio.read_bytes() + b"tamper")
    try: tool.recover_alignment_axis(root2)
    except tool.SafetyError: pass
    else: raise AssertionError("audio tamper accepted")


def test_axis_recovery_rejects_stale_old_resume_without_writes(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _prepare_downstream_resume(root, scope, valid_grids=True)
    resume = root / "downstream_resume_preflight_receipt.json"
    value = json.loads(resume.read_text()); value["v2_sha256"] = "tampered"
    os.chmod(resume, 0o644); resume.write_text(json.dumps(value), encoding="utf-8")
    try: tool.recover_alignment_axis(root)
    except tool.SafetyError: pass
    else: raise AssertionError("stale old resume accepted")
    assert not (root / "workspace" / ".mfa_alignment_axis_receipt_recovered.json").exists()


def test_direct_entrypoint_exposes_downstream_resume_commands():
    proc = subprocess.run([sys.executable, str(tool.PROJECT / "scripts" / "gpu1000_orchestrate.py"), "--help"],
                          cwd=tool.PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert proc.returncode == 0
    assert "downstream-resume-preflight" in proc.stdout and "continue-downstream" in proc.stdout


def test_preflight_binds_990_shard_plus_9_retry_origins(tmp_path: Path):
    root, scope, stems = _fixture(tmp_path)
    ctc = root / "workspace" / "ctc_pretg"
    grids = sorted(ctc.glob("*.TextGrid"))
    for grid in grids: grid.unlink()
    shard = root / "workspace" / "mfa_shards" / "run" / "shard_0" / "output"; shard.mkdir(parents=True)
    retry = root / "workspace" / "mfa_shards" / "run" / "retry_missing" / "output"; retry.mkdir(parents=True)
    for grid, destination in zip(grids[:990], [shard] * 990): (destination / grid.name).write_text("grid")
    for grid, destination in zip(grids[990:], [retry] * 9): (destination / grid.name).write_text("grid")
    _store_preflight(root, scope)
    receipt = json.loads((root / "continuation_preflight_receipt.json").read_text())
    assert receipt["ok"] and receipt["original_grid_count"] == 999


def test_continue_after_mfa_is_atomic_new_root_and_original_immutable(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight_v2(root, scope)
    original_hash = tool.sha_file(root / "run_receipt.json")
    receipt = tool.continue_after_mfa(root, scope)
    assert receipt["status"] == "READY_FOR_DOWNSTREAM" and receipt["merged_count"] == 1000
    assert root.is_dir() and tool.sha_file(root / "run_receipt.json") == original_hash
    started = json.loads((root / "continuation_started.json").read_text())
    v2_path = root / "continuation_preflight_receipt_v2.json"
    assert started["preflight_v2_path"] == str(v2_path.resolve())
    assert started["preflight_v2_sha256"] == tool.sha_file(v2_path)
    assert receipt["preflight_v2_path"] == str(v2_path.resolve())
    assert receipt["preflight_v2_sha256"] == tool.sha_file(v2_path)
    axis = json.loads((root / "workspace" / ".mfa_alignment_axis_receipt.json").read_text())
    assert len(axis["stems"]) == 1000


def test_continuation_rejects_downstream_namespace(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight(root, scope)
    (root / "workspace" / "aligned").mkdir()
    (root / "workspace" / "aligned" / "stale.TextGrid").write_text("stale")
    receipt = tool.continuation_preflight(root, scope, proven_mfa_retry_receipt=root / "proof.json")
    assert not receipt["ok"] and "downstream_namespace_present:aligned" in receipt["errors"]


def test_composite_lineage_rejects_tampered_original_receipt(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight_v2(root, scope)
    tool.continue_after_mfa(root, scope)
    errors: list[str] = []
    lineage = analyzer._continuation_lineage(root, errors)
    assert lineage["status"] == "NONE" and not errors
    output = root / "workspace" / "strict_ok_runs" / "continuation_mock" / "output"; output.mkdir(parents=True)
    (output / ".pipeline_run_receipt_v2.json").write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}))
    designated = str(next((root / "workspace" / "strict_ok_runs").glob("continuation_mock/output/.pipeline_run_receipt_v2.json")))
    tool.finalize_continuation(root, downstream={"strict_receipt": designated})
    errors = []
    lineage = analyzer._continuation_lineage(root, errors)
    assert lineage["status"] == "PASS_WITH_CONTINUATION" and not errors
    (root / "run_receipt.json").write_text("tampered", encoding="utf-8")
    errors = []
    lineage = analyzer._continuation_lineage(root, errors)
    assert lineage["status"] == "BLOCKED" and "continuation_original_receipt_binding_invalid" in errors


def test_finalize_requires_single_strict_receipt_and_promotes_gate(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight_v2(root, scope)
    tool.continue_after_mfa(root, scope)
    output = root / "workspace" / "strict_ok_runs" / "continuation_mock" / "output"; output.mkdir(parents=True)
    receipt = output / ".pipeline_run_receipt_v2.json"; receipt.write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}))
    designated = str(receipt.resolve())
    result = tool.finalize_continuation(root, downstream={"strict_receipt": designated})
    assert result["status"] == "PASS_WITH_CONTINUATION" and result["strict_receipt_sha256"] == tool.sha_file(receipt)


def test_stored_preflight_drift_blocks_continuation(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path); _store_preflight(root, scope); _store_preflight_v2(root, scope)
    path = root / "continuation_preflight_receipt_v2.json"; value = json.loads(path.read_text()); value["fresh_preflight_digest"] = "0" * 64; os.chmod(path, 0o644); path.write_text(json.dumps(value))
    try: tool.continue_after_mfa(root, scope)
    except tool.SafetyError: return
    raise AssertionError("drifted stored preflight accepted")


def test_dual_strict_designation_prefers_bound_continuation(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path); _store_preflight(root, scope); _store_preflight_v2(root, scope); tool.continue_after_mfa(root, scope)
    old = root / "workspace" / "strict_ok_runs" / "historical" / "output"; old.mkdir(parents=True); (old / "old.TextGrid").write_text("x"); (old / ".pipeline_run_receipt_v2.json").write_text("{}")
    new = root / "workspace" / "strict_ok_runs" / "continuation_new" / "output"; new.mkdir(parents=True); (new / "new.TextGrid").write_text("x"); receipt = new / ".pipeline_run_receipt_v2.json"; receipt.write_text("{}")
    import hashlib
    (root / "continuation_result.json").write_text(json.dumps({"status":"PASS_WITH_CONTINUATION","strict_receipt_path":str(receipt),"strict_receipt_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest()}))
    from scripts import analyze_gpu1000_run as a
    assert a._partition_sets(root)[0] == {"new"}


def test_singleton_batch_policy_is_narrow():
    assert tool.continuation_batch_policy(initial_pass=False)["expanded_allowed"] is False
    assert tool.continuation_batch_policy(initial_pass=True)["expanded"]["split"] == [200, 800]


def test_canonical_nonempty_bootstrap_builder():
    from scripts.pipeline_utils import make_pipeline_accounting_receipt, validate_pipeline_accounting_receipt
    receipt = make_pipeline_accounting_receipt(["a"], ["a"], [], ["a"], [], run_id="continuation", route=["align_en", "postprocess", "strict_ok"], paths={"output":"/tmp/o", "filtered":"/tmp/f"})
    assert validate_pipeline_accounting_receipt(receipt) == []


def test_mocked_downstream_route_uses_only_permitted_steps(tmp_path: Path):
    root, scope, _ = _fixture(tmp_path)
    _store_preflight(root, scope)
    _store_preflight_v2(root, scope)
    tool.continue_after_mfa(root, scope)
    ctc = root / "workspace" / "ctc_pretg"
    (ctc / ".pipeline_run_receipt_v2.json").write_text(json.dumps({"schema": "pipeline-run-receipt-v2"}))
    en_root = root / "workspace" / "en_phones"; en_root.mkdir()
    ledgers = []
    for stem in sorted(f"s{i:04d}" for i in range(1000)):
        ledger = en_root / f"{stem}_en_phones.json"; ledger.write_text("{}", encoding="utf-8")
        ledgers.append({"stem": stem, "path": str(ledger.resolve()), "sha256": tool.sha_file(ledger)})
    (en_root / "en_alignment_manifest.json").write_text(json.dumps({"schema": "strict-en-mfa-v1", "status": "success", "strict_provenance": True, "stem_ledgers": ledgers}), encoding="utf-8")
    input_axis_path = ctc / ".mfa_input_axis_receipt.json"; os.chmod(input_axis_path, 0o644)
    input_axis_path.write_text(json.dumps({"schema": "mfa-input-axis-receipt-v1", "stems": sorted(f"s{i:04d}" for i in range(1000)), "axis_root": str((root / "workspace" / "aligned").resolve()), "tts_authoritative_audio_root": str((root / "input").resolve())}), encoding="utf-8")
    import scripts.run_pipeline as pipeline
    calls = []
    def fake(name):
        def run(args, cfg, python, ctx):
            calls.append((name, args.python, args.overwrite, str(ctx["mfa_dict"])))
            return 0
        return run
    with patch.object(pipeline, "step_mfa_align_en", fake("align_en")), patch.object(pipeline, "step_postprocess", fake("postprocess")), patch.object(pipeline, "step_strict_ok", fake("strict_ok")), patch.object(tool, "validate_pipeline_accounting_receipt", return_value=[]), patch.object(tool, "make_pipeline_accounting_receipt", return_value={"schema": "pipeline-run-receipt-v2"}):
        result = tool._run_permitted_downstream(root, "python")
    assert [row[0] for row in calls] == ["postprocess", "strict_ok"]
    assert all(row[1] == "python" and row[2] is False for row in calls)
    assert all("mfa_ipa.dict" in row[3] for row in calls)
    assert result["strict_receipt_sha256"]
