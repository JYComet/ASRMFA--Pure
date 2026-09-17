import json
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts.full_corpus_orchestrator import (ChunkLimits, build_chunks, materialize_pipeline_config,
    _result_roots, prepare, preflight, audit, _load_valid_gamesl_receipt,
    _load_bound_gamesl_receipt_fast, _load_bound_output_receipt_fast,
    _gamesl_bound, _fresh_canary_root, _stage_chunk_items, _qwen_evidence,
    _downstream_command, _prepare_command, _prepared_chunk, run_full)


def test_frozen_stage_receipt_reuses_nvme_without_reading_nas(tmp_path, monkeypatch):
    import scripts.full_corpus_orchestrator as orchestrator
    input_dir = tmp_path / "input"; input_dir.mkdir()
    pipeline = input_dir / "u1.wav"; pipeline.write_bytes(b"pipeline")
    digest = hashlib.sha256(pipeline.read_bytes()).hexdigest()
    (input_dir / "u1.stage_receipt.json").write_text(json.dumps({
        "run_stem": "u1", "source_sha256": "frozen-source",
        "pipeline_wav": str(pipeline), "pipeline_wav_sha256": digest,
        "gamesl_wav": None, "gamesl_wav_sha256": None,
        "text_path": None, "text_sha256": None, "normalized_text": None,
    }))
    source = tmp_path / "nas.wav"; source.write_bytes(b"nas")
    item = SimpleNamespace(run_stem="u1", source_id="v5_0707", game=None,
                           speaker="LAria", source_path=source,
                           audio_sha256="frozen-source", reference_path=None)
    real_sha = orchestrator._sha256
    def reject_nas(path):
        assert Path(path) != source
        return real_sha(path)
    monkeypatch.setattr(orchestrator, "_sha256", reject_nas)

    receipt = orchestrator._load_frozen_stage_receipt(item, input_dir)
    assert receipt.pipeline_wav == pipeline
    assert receipt.source_sha256 == "frozen-source"


def test_prepared_stage_receipt_reuses_sealed_hash_without_rereading_audio(
        tmp_path, monkeypatch):
    import scripts.full_corpus_orchestrator as orchestrator
    input_dir = tmp_path / "input"; input_dir.mkdir()
    pipeline = input_dir / "u1.wav"; pipeline.write_bytes(b"pipeline")
    (input_dir / "u1.stage_receipt.json").write_text(json.dumps({
        "run_stem": "u1", "source_sha256": "frozen-source",
        "pipeline_wav": str(pipeline), "pipeline_wav_sha256": "sealed",
        "gamesl_wav": None, "gamesl_wav_sha256": None,
        "text_path": None, "text_sha256": None, "normalized_text": None,
    }))
    item = SimpleNamespace(run_stem="u1", source_id="v5_0707", game=None,
                           speaker="LAria", audio_sha256="frozen-source",
                           reference_path=None, needs_gamesl_padding=False)
    monkeypatch.setattr(orchestrator, "_sha256", lambda _: (_ for _ in ()).throw(
        AssertionError("prepared audio must not be rehashed before MFA")))

    receipt = orchestrator._load_frozen_stage_receipt(
        item, input_dir, verify_content=False)
    assert receipt.pipeline_wav_sha256 == "sealed"


def test_daemon_prefetch_task_runs_without_blocking_caller():
    from scripts.full_corpus_orchestrator import _PrefetchTask
    started = threading.Event(); release = threading.Event()
    def work():
        started.set(); release.wait(2); return "ready"
    task = _PrefetchTask("chunk-2", work)
    task.start()
    assert started.wait(1)
    assert not task.done()
    release.set()
    assert task.result(1) == "ready"


def test_prefetched_chunk_promotes_only_after_current_pipeline_finishes():
    from scripts.full_corpus_orchestrator import _prefetch_can_promote, _PrefetchTask
    task = _PrefetchTask("chunk-2", lambda: "ready").start()
    assert task.result(1) == "ready"
    assert not _prefetch_can_promote(task, producer=object(), downstream=None)
    assert not _prefetch_can_promote(task, producer=None, downstream=object())
    assert _prefetch_can_promote(task, producer=None, downstream=None)


def test_resumed_downstream_refreshes_mutable_ctc_work_once(tmp_path):
    command = _downstream_command(
        "/venv/bin/python", tmp_path / "chunk.yaml", refresh_ctc_work=True)
    assert command[-1] == "--refresh-ctc-work"
    assert "--overwrite" not in command

    fresh = _downstream_command(
        "/venv/bin/python", tmp_path / "chunk.yaml", refresh_ctc_work=False)
    assert "--refresh-ctc-work" not in fresh


def test_prepared_mfa_command_can_defer_step_validation_to_final_gate(tmp_path):
    command = _downstream_command(
        "/venv/bin/python", tmp_path / "chunk.yaml",
        skip_to="align", validate=False)
    assert "--validate" not in command
    assert command[-2:] == ["--skip-to", "align"]


def test_prepare_command_stops_after_adjust_and_refreshes_only_resumed_qwen(tmp_path):
    fresh = _prepare_command("/venv/bin/python", tmp_path / "chunk.yaml", qwen_sealed=False)
    assert fresh[-3:] == ["--stop-after", "adjust", "--validate"]
    assert "--skip-to" not in fresh

    resumed = _prepare_command("/venv/bin/python", tmp_path / "chunk.yaml", qwen_sealed=True)
    assert ["--skip-to", "normalize_punct"] == resumed[
        resumed.index("--skip-to"):resumed.index("--skip-to") + 2]
    assert ["--stop-after", "adjust"] == resumed[
        resumed.index("--stop-after"):resumed.index("--stop-after") + 2]
    assert resumed[-1] == "--refresh-ctc-work"


def test_prepared_chunk_preserves_exact_ready_failure_partition():
    chunk = build_chunks(
        [Item("u1", "reference"), Item("u2", "reference"), Item("u3", "reference")],
        ChunkLimits(10, 1, 10**9))[0]
    prepared, failures = _prepared_chunk(chunk, {
        "ready_stems": ["u1", "u3"],
        "stage_failures": [{"stem": "u2", "state": "pipeline_failure", "reason": "bad audio"}],
    })
    assert [item.run_stem for item in prepared.items] == ["u1", "u3"]
    assert failures == [{"stem": "u2", "state": "pipeline_failure", "reason": "bad audio"}]


def test_publication_bound_chunk_excludes_explicit_producer_failures():
    items = [Item("u1", "reference"), Item("u2", "reference"),
             Item("u3", "reference")]
    chunk = build_chunks(items, ChunkLimits(10, 1, 10**9))[0]
    publication = SimpleNamespace(accepted_stems=("u1",), filtered_stems=("u2",),
                                  failed_stems=("u3",))
    from scripts.full_corpus_orchestrator import _publication_bound_chunk
    public = _publication_bound_chunk(chunk, publication)
    assert [item.run_stem for item in public.items] == ["u1", "u2"]


@dataclass
class Item:
    run_stem: str
    text_mode: str
    duration_seconds: float = 10
    audio_bytes: int = 100


def test_chunks_never_mix_reference_modes_and_conserve_inventory():
    items = [Item("u3", "fallback"), Item("u1", "reference"), Item("u2", "reference")]
    chunks = build_chunks(items, ChunkLimits(2, 1, 10**9))
    assert all(len({row.text_mode for row in chunk.items}) == 1 for chunk in chunks)
    assert sorted(row.run_stem for chunk in chunks for row in chunk.items) == ["u1", "u2", "u3"]


def test_resolved_pipeline_is_qwen_only(tmp_path):
    chunk = build_chunks([Item("u1", "reference")], ChunkLimits(4, 1, 10**9))[0]
    task = {"ctc_prealign": {"provider": "nvasr", "model_path": "old"},
            "trim": {"normalize_edges": True}}
    cfg = materialize_pipeline_config(chunk, task, tmp_path / "run.yaml")
    assert cfg["ctc_prealign"]["provider"] == "qwen3_hf"
    assert cfg["reference_mode"] == "authority"
    assert "nvasr" not in json.dumps(cfg, ensure_ascii=False).lower()
    assert cfg["trim"]["normalize_edges"] is False
    assert cfg["trim"]["max_silence_sec"] == 1_000_000_000.0
    assert cfg["ctc_prealign"]["allow_item_failures"] is True


def test_qwen_evidence_accepts_accounted_item_filter_subset(tmp_path):
    raw = tmp_path / "workspace" / "ctc_pretg"
    raw.mkdir(parents=True)
    manifest = [{"audio": "/input/u1.wav", "duration_s": 1.0,
                 "text_original": "你好", "_words": [{"word": "ni3"}]}]
    (raw / "manifest.json").write_text(json.dumps(manifest))
    identity = {
        "schema": "qwen3-hf-prealign-identity-v1", "provider": "qwen3_hf",
        "models": {"forced_aligner_tree_digest": "a", "asr": "not_required"},
        "inputs": "i", "references_digest": "r", "output_digest": "o",
        "identity_digest": "identity",
    }
    (raw / ".qwen3_hf_identity.json").write_text(json.dumps(identity))
    def part(stems):
        stems = sorted(stems)
        return {"count": len(stems), "stems": stems,
                "stems_digest": hashlib.sha256(json.dumps(
                    stems, ensure_ascii=False, separators=(",", ":")
                ).encode()).hexdigest()}
    receipt = {
        "schema": "pipeline-run-receipt-v2", "run_health": "healthy",
        "silent_loss": 0, "source": part(["u1", "u2"]),
        "eligible": part(["u1", "u2"]), "output": part(["u1"]),
        "filtered": part(["u2"]),
        "route": ["qwen3_hf", "qwen3_forced_aligner"],
        "extra": {"reference_mode": "authority", "identity_digest": "identity",
                  "forced_aligner_model_tree_digest": "a",
                  "processed_stems": ["u1", "u2"]},
    }
    (raw / ".pipeline_run_receipt_v2.json").write_text(json.dumps(receipt))

    evidence = _qwen_evidence(tmp_path / "workspace", ["u1", "u2"], "reference")
    assert evidence["output_stems"] == ["u1"]
    assert evidence["filtered_stems"] == ["u2"]


def test_result_roots_follow_sealed_accounting_paths(tmp_path):
    workspace = tmp_path / "workspace"
    output = tmp_path / "authoritative-output"
    filtered = tmp_path / "authoritative-filtered"
    receipt_dir = workspace / "runs" / "run" / "output"
    receipt_dir.mkdir(parents=True)
    output.mkdir(); filtered.mkdir()
    digest = hashlib.sha256(b'["u1"]').hexdigest()
    (receipt_dir / ".pipeline_run_receipt_v2.json").write_text(json.dumps({
        "schema":"pipeline-run-receipt-v2", "run_health":"healthy", "silent_loss":0,
        "source":{"count":1,"stems":["u1"],"stems_digest":digest}, "eligible":{"count":1,"stems":["u1"],"stems_digest":digest},
        "output":{"count":1,"stems":["u1"],"stems_digest":digest}, "filtered":{"count":0,"stems":[],"stems_digest":hashlib.sha256(b'[]').hexdigest()},
        "extra":{"reference_mode":"reference","identity_digest":"i","forced_aligner_model_tree_digest":"m"},
        "paths": {"output": str(output), "filtered": str(filtered)}}))
    assert _result_roots(workspace, ["u1"], "reference") == (output, filtered)


def test_chunk_limits_reject_nonpositive_values():
    try:
        build_chunks([Item("u1", "fallback")], ChunkLimits(0, 1, 10))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("invalid limits were accepted")


def test_failed_canary_retry_uses_fresh_private_workspace(tmp_path):
    first = _fresh_canary_root(tmp_path)
    assert first == tmp_path / "canaries"
    first.mkdir()
    rerun = _fresh_canary_root(tmp_path)
    assert rerun.parent == tmp_path
    assert rerun.name.startswith("canaries-rerun-")
    assert rerun != first


def test_chunk_staging_isolates_one_bad_item(tmp_path):
    items = [Item("u1", "reference"), Item("u2", "reference"),
             Item("u3", "reference")]
    chunk = build_chunks(items, ChunkLimits(10, 1, 10**9))[0]
    status = {"items": {"u1":{"state":"producer_failure"}},
              "terminal": {chunk.chunk_id:"producer_failure"}}
    stale = tmp_path / "chunk" / "input" / "u2.wav"
    stale.parent.mkdir(parents=True); stale.write_bytes(b"stale")
    def stage_one(item, input_dir, gamesl_dir):
        if item.run_stem == "u2":
            raise ValueError("no speech")
        return "receipt-" + item.run_stem
    ready, receipts, failures = _stage_chunk_items(
        tmp_path, status, chunk, {item.run_stem: item for item in items},
        tmp_path / "chunk", stage_one)
    assert [item.run_stem for item in ready.items] == ["u1", "u3"]
    assert receipts == ["receipt-u1", "receipt-u3"]
    assert failures == 1
    assert "u1" not in status["items"]
    assert chunk.chunk_id not in status["terminal"]
    assert not stale.exists()
    assert status["items"]["u2"]["state"] == "pipeline_failure"
    assert "stage_failure:ValueError:no speech" in status["items"]["u2"]["reason"]


def test_preflight_validates_frozen_canaries_and_disjoint_roots(tmp_path, monkeypatch):
    import soundfile as sf
    import numpy as np
    roots = {key: tmp_path / key for key in ("gamedata", "wuwa", "v5_0707")}
    wuwa = roots["wuwa"] / "今汐" / "zh_vo_Chengxiaoshan_main_1_1_145_11.wav"
    wuwa.parent.mkdir(parents=True); sf.write(wuwa, np.r_[np.zeros(800), np.ones(800)], 16000)
    wuwa.with_suffix(".lab").write_text("你好")
    laria = roots["v5_0707"] / "LAria" / "wavs" / "LAria_00001.wav"
    laria.parent.mkdir(parents=True); sf.write(laria, np.r_[np.zeros(800), np.ones(800)], 16000)
    for root in roots.values(): root.mkdir(parents=True, exist_ok=True)
    qwen = tmp_path / "qwen"; qwen.mkdir(); mfa = tmp_path / "mfa"; mfa.mkdir()
    for name in ("asr", "align", "dict", "pin", "model"):
        (tmp_path / name).write_text("x")
    cfg = {"run_root":str(tmp_path/"run"), "output_root":str(tmp_path/"output"), "gamesl_root":str(tmp_path/"gamesl"),
           "sources":{k:str(v) for k,v in roots.items()}, "chunk_limits":{"max_files":4,"max_audio_hours":1,"max_bytes":10**9},
           "qwen":{"python":"/home/user/miniconda3/bin/python","asr_model":str(tmp_path/"asr"),"forced_aligner_model":str(tmp_path/"align")},
           "mfa":{"python":"/home/user/miniconda3/envs/mfa-dev/bin/python"}, "mfa_dictionary":str(tmp_path/"dict"),
           "pinyin_dictionary":str(tmp_path/"pin"), "mfa_models":str(tmp_path/"model"), "required_gpu_count":0,
           "canaries":[{"source_id":"wuwa","speaker":"今汐","relative_stem":"zh_vo_Chengxiaoshan_main_1_1_145_11"},{"source_id":"v5_0707","speaker":"LAria","relative_path":"wavs/LAria_00001.wav"}]}
    prepare(cfg)
    assert preflight(cfg) == 0
    receipt = json.loads((tmp_path / "run" / "preflight_receipt.json").read_text())
    assert receipt["source_deferred_to_stage"] == 2
    assert receipt["canary_hash_verified"] == 2
    assert receipt["source_hash_policy"] == "frozen_at_prepare_reverified_at_stage"


def test_preflight_does_not_reread_all_frozen_audio_bytes(tmp_path, monkeypatch):
    import soundfile as sf
    import numpy as np
    import scripts.full_corpus_orchestrator as orchestrator
    roots = {key: tmp_path / key for key in ("gamedata", "wuwa", "v5_0707")}
    wuwa = roots["wuwa"] / "今汐" / "zh_vo_Chengxiaoshan_main_1_1_145_11.wav"
    wuwa.parent.mkdir(parents=True); sf.write(wuwa, np.ones(1600), 16000)
    wuwa.with_suffix(".lab").write_text("你好")
    laria = roots["v5_0707"] / "LAria" / "wavs" / "LAria_00001.wav"
    laria.parent.mkdir(parents=True); sf.write(laria, np.ones(1600), 16000)
    roots["gamedata"].mkdir(parents=True)
    for name in ("asr", "align", "dict", "pin", "model"):
        (tmp_path / name).write_text("x")
    cfg = {"run_root":str(tmp_path/"run"), "output_root":str(tmp_path/"output"), "gamesl_root":str(tmp_path/"gamesl"),
           "sources":{k:str(v) for k,v in roots.items()}, "chunk_limits":{"max_files":4,"max_audio_hours":1,"max_bytes":10**9},
           "qwen":{"python":"/home/user/miniconda3/bin/python","asr_model":str(tmp_path/"asr"),"forced_aligner_model":str(tmp_path/"align")},
           "mfa":{"python":"/home/user/miniconda3/envs/mfa-dev/bin/python"}, "mfa_dictionary":str(tmp_path/"dict"),
           "pinyin_dictionary":str(tmp_path/"pin"), "mfa_models":str(tmp_path/"model"), "required_gpu_count":0,
           "canaries":[{"source_id":"wuwa","speaker":"今汐","relative_stem":"zh_vo_Chengxiaoshan_main_1_1_145_11"},{"source_id":"v5_0707","speaker":"LAria","relative_path":"wavs/LAria_00001.wav"}]}
    prepare(cfg)
    real_sha256 = orchestrator._sha256
    audio_reads = []
    def record_audio_reads(path):
        if str(path).endswith(".wav"):
            audio_reads.append(str(path))
        return real_sha256(path)
    monkeypatch.setattr(orchestrator, "_sha256", record_audio_reads)
    assert preflight(cfg) == 0
    assert sorted(audio_reads) == sorted([str(wuwa), str(laria)])


def test_audit_accepts_conserved_failure_and_writes_report(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    item = {"run_stem":"u1","source_id":"v5_0707","speaker":"LAria","text_mode":"fallback","duration_seconds":1}
    (root / "frozen_inventory.json").write_text(json.dumps({"items":[item],"excluded":[],"invalid":[]}))
    (root / "chunks.json").write_text(json.dumps({"chunks":[{"chunk_id":"fallback-c","items":["u1"]}]}))
    (root / "status.json").write_text(json.dumps({"schema":"qwen3-0915all-status-v1","state":"complete_with_failures",
        "terminal":{"fallback-c":"producer_failure"},"items":{"u1":{"state":"producer_failure","reason":"qwen_prealign_failed"}}}))
    status = {"schema":"qwen3-0915all-status-v1","state":"complete_with_failures",
        "terminal":{"fallback-c":"producer_failure"},"items":{"u1":{"state":"producer_failure","reason":"qwen_prealign_failed"}}}
    (root / "status.json").write_text(json.dumps(status))
    (root / "final_report.json").write_text(json.dumps({"terminal_stems":["u1"],"schema":"qwen3-0915all-report-v1",
        "status":status,"publication_receipts":[],
        "counts":{"accepted":0,"filtered":0,"producer_failure":1,"pipeline_or_mfa_failure":0}}))
    assert audit({"run_root":str(root),"reports_root":str(tmp_path / "reports")}) == 0
    assert (tmp_path / "reports" / "final_report.json").is_file()


def test_gamesl_resume_receipt_requires_matching_target_hash(tmp_path):
    target = tmp_path / "gamesl.wav"; target.write_bytes(b"new")
    import hashlib
    bound = {"inventory_sha256":"i", "chunk_stems_digest":"d", "chunk_stems":["u1"], "config_sha256":"c"}
    receipt = {**bound, "replacements":[{"stem":"u1","target":str(target),"new_sha256":hashlib.sha256(b"new").hexdigest()}]}
    path = tmp_path / "gamesl_publication.json"; path.write_text(json.dumps(receipt))
    assert _load_valid_gamesl_receipt(path, bound) == receipt
    target.write_bytes(b"tampered")
    assert _load_valid_gamesl_receipt(path, bound) is None


def test_prepared_gamesl_receipt_fast_path_checks_binding_without_rehashing_audio(
        tmp_path, monkeypatch):
    target = tmp_path / "gamesl.wav"; target.write_bytes(b"audio")
    bound = {"chunk_id":"c", "inventory_sha256":"i", "config_sha256":"x",
             "chunk_stems":["u1"], "chunk_stems_digest":"d"}
    receipt = {**bound, "published_count":1, "replaced_count":0,
               "replacements":[{"stem":"u1", "target":str(target),
                                  "new_sha256":"sealed-in-prepare"}]}
    path = tmp_path / "gamesl_publication.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    import scripts.full_corpus_orchestrator as orchestrator
    monkeypatch.setattr(orchestrator, "_sha256", lambda _: (_ for _ in ()).throw(
        AssertionError("prepared GAMESL audio must not be rehashed after MFA")))

    assert _load_bound_gamesl_receipt_fast(path, bound) == receipt


def test_completed_output_receipt_fast_path_does_not_rehash_public_targets(
        tmp_path, monkeypatch):
    target = tmp_path / "u1.TextGrid"; target.write_text("sealed")
    bound = {"chunk_id":"c", "inventory_sha256":"i", "config_sha256":"x",
             "chunk_stems":["u1"], "chunk_stems_digest":"d"}
    receipt = {**bound, "replacements":[{
        "stem":"u1", "target":str(target), "new_sha256":"sealed-at-publication"}]}
    path = tmp_path / "output_publication.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    import scripts.full_corpus_orchestrator as orchestrator
    monkeypatch.setattr(orchestrator, "_sha256", lambda _: (_ for _ in ()).throw(
        AssertionError("completed public output must not be rehashed on resume")))

    assert _load_bound_output_receipt_fast(path, bound) == receipt


def test_gamesl_resume_projects_a_valid_prior_superset(tmp_path):
    targets = {}
    rows = []
    for stem in ("u1", "u2"):
        target = tmp_path / f"{stem}.wav"; target.write_bytes(stem.encode())
        targets[stem] = target
        rows.append({"stem":stem,"target":str(target),
                     "new_sha256":hashlib.sha256(stem.encode()).hexdigest(),
                     "old_sha256":None})
    receipt = {"chunk_id":"c","inventory_sha256":"i","config_sha256":"x",
               "chunk_stems":["u1","u2"],"chunk_stems_digest":"old",
               "replacements":rows}
    path = tmp_path / "receipt.json"; path.write_text(json.dumps(receipt))
    bound = {"chunk_id":"c","inventory_sha256":"i","config_sha256":"x",
             "chunk_stems":["u1"],"chunk_stems_digest":"new"}
    projected = _load_valid_gamesl_receipt(path, bound)
    assert projected["chunk_stems"] == ["u1"]
    assert [row["stem"] for row in projected["replacements"]] == ["u1"]


def test_gamesl_receipt_binding_excludes_unpadded_v5_items(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    (root / "frozen_inventory.json").write_text("{}")
    padded = Item("u1", "reference"); padded.needs_gamesl_padding = True
    v5 = Item("u2", "reference"); v5.needs_gamesl_padding = False
    chunk = build_chunks([padded, v5], ChunkLimits(4, 1, 10**9))[0]
    bound = _gamesl_bound(root, chunk, {})
    assert bound["chunk_stems"] == ["u1"]


def test_audit_rejects_missing_publication_receipt_for_accepted_item(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    item = {"run_stem":"u1","source_id":"v5_0707","speaker":"LAria","text_mode":"fallback","duration_seconds":1}
    status = {"schema":"qwen3-0915all-status-v1","state":"complete",
              "terminal":{"fallback-c":"complete"},"items":{"u1":{"state":"accepted"}}}
    (root / "frozen_inventory.json").write_text(json.dumps({"items":[item],"excluded":[],"invalid":[]}))
    (root / "chunks.json").write_text(json.dumps({"chunks":[{"chunk_id":"fallback-c","items":["u1"]}]}))
    (root / "status.json").write_text(json.dumps(status))
    (root / "final_report.json").write_text(json.dumps({"schema":"qwen3-0915all-report-v1",
        "terminal_stems":["u1"],"status":status,"publication_receipts":[],
        "counts":{"accepted":1,"filtered":0,"producer_failure":0,"pipeline_or_mfa_failure":0}}))
    try:
        audit({"run_root":str(root),"reports_root":str(tmp_path / "reports"),"output_root":str(tmp_path / "out")})
    except ValueError as exc:
        assert "published output denominator" in str(exc)
    else:
        raise AssertionError("accepted item without publication receipt passed audit")


def test_run_full_rejects_forged_canary_before_any_stage(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    source = tmp_path / "a.wav"; source.write_bytes(b"a")
    item = {"run_stem":"u1","source_id":"wuwa","speaker":"今汐","source_path":str(source),
            "source_relative_path":"今汐/zh_vo_Chengxiaoshan_main_1_1_145_11.wav","text_mode":"reference",
            "game":"鸣潮","needs_gamesl_padding":True}
    (root / "frozen_inventory.json").write_text(json.dumps({"items":[item],"excluded":[],"invalid":[]}))
    (root / "canary_receipt.json").write_text(json.dumps({"schema":"qwen3-0915all-canary-v1","success":True,"exact_count":2}))
    cfg = {"run_root":str(root),"canaries":[{"source_id":"wuwa","speaker":"今汐","relative_stem":"zh_vo_Chengxiaoshan_main_1_1_145_11"},{"source_id":"v5_0707","speaker":"LAria","relative_path":"wavs/LAria_00001.wav"}]}
    try:
        run_full(cfg)
    except RuntimeError as exc:
        assert "canary" in str(exc)
    else:
        raise AssertionError("forged canary receipt was accepted")
