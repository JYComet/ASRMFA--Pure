# 0915ALL Qwen3 Full Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a resumable, manifest-bound full-corpus runner that flattens globally unique inputs, uses only Qwen3 and MFA, pre-normalizes GAMEDATA/Wuthering Waves edge silence, and republishes TextGrids by speaker.

**Architecture:** A read-only inventory module freezes every source item and assigns a stable opaque stem. A staging module materializes bounded NVMe chunks and publishes verified padded game audio to GAMESL. An orchestrator drives existing `run_pipeline.py` in separate reference and fallback chunks, overlaps Qwen GPU work with downstream CPU work, and delegates validated result publication to a focused publisher.

**Tech Stack:** Python 3.11, PyYAML, soundfile, NumPy, subprocess, existing Qwen3 HF/ForcedAligner and MFA pipeline, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-0915all-qwen3-full-corpus-design.md`

## Global Constraints

- Treat `/mnt/Raw/GAMEDATA`, `/mnt/Raw/v5_0707`, and `/mnt/Raw/Onlinedataset/鸣潮/中文/WutheringWaves2.2_CN/中文 - Chinese` as read-only.
- Publish accepted TextGrids under `/mnt/Raw/0915ALL/<speaker>/` and filtered TextGrids under `/mnt/Raw/0915ALL/_filtered/<speaker>/`.
- Publish normalized GAMEDATA and Wuthering Waves WAVs under `/mnt/Raw/GAMESL/<game>/<speaker>/`; do not publish v5_0707 audio there.
- Exclude every `其它语音 - Others` and `带变量语音 - Placeholder` subtree.
- Use `qwen3_hf` exclusively; provider fallback and NVASR model loading are forbidden.
- Run exactly two end-to-end canaries first; launch full processing automatically only if both pass.
- Preserve unrelated dirty-worktree changes and all existing sealed pipeline evidence.

---

### Task 1: Freeze the cross-source inventory and stable flat namespace

**Files:**
- Create: `scripts/full_corpus_inventory.py`
- Create: `tests/test_full_corpus_inventory.py`

**Interfaces:**
- Consumes: three configured source roots and declarative selection rules.
- Produces: `InventoryItem`, `InventorySnapshot`, `scan_sources(config: dict) -> InventorySnapshot`, `stable_run_stem(...) -> str`, and `write_frozen_inventory(snapshot, path: Path) -> dict`.

- [ ] **Step 1: Write failing tests for source selection, references, speakers, and IDs**

```python
def test_scan_sources_applies_all_layout_rules(tmp_path):
    roots = make_three_source_fixture(tmp_path)
    snapshot = scan_sources(config_for(roots))
    by_original = {row.original_stem: row for row in snapshot.items}
    assert by_original["game_ref"].text_mode == "reference"
    assert by_original["game_ref"].reference_suffix == ".txt"
    assert by_original["wuwa_ref"].speaker == "今汐"
    assert by_original["wuwa_ref"].reference_suffix == ".lab"
    assert by_original["laria_noref"].text_mode == "fallback"
    assert {row.original_stem for row in snapshot.excluded} == {"other", "placeholder", "chinese0707"}

def test_stable_id_is_path_stable_and_content_changes_are_detected(tmp_path):
    first = stable_run_stem("gamedata", "崩铁", "白露", "白露/a.wav")
    second = stable_run_stem("gamedata", "崩铁", "白露", "白露/a.wav")
    assert first == second and re.fullmatch(r"u[0-9a-f]{32}", first)
```

- [ ] **Step 2: Run the inventory tests and confirm they fail**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_inventory.py`

Expected: collection fails because `scripts.full_corpus_inventory` does not exist.

- [ ] **Step 3: Implement a one-pass, fail-closed inventory**

```python
@dataclass(frozen=True)
class InventoryItem:
    source_id: str
    game: str | None
    speaker: str
    source_path: Path
    source_relative_path: str
    original_stem: str
    run_stem: str
    audio_sha256: str
    audio_bytes: int
    duration_seconds: float
    reference_path: Path | None
    reference_sha256: str | None
    reference_suffix: str | None
    text_mode: Literal["reference", "fallback"]
    needs_gamesl_padding: bool

def stable_run_stem(source_id: str, game: str | None, speaker: str,
                    relative_path: str) -> str:
    identity = [source_id, game or "", speaker,
                unicodedata.normalize("NFC", relative_path)]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return "u" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
```

Walk every tree once with sorted directories and files. For GAMEDATA use first relative directory as speaker and a non-empty sibling `.txt`; for Wuthering Waves use first directory as speaker, skip either excluded directory name at any depth, and use a non-empty sibling `.lab`; for v5_0707 apply the case-insensitive allow rules and sibling `.txt` discovery. Hash files while inventorying, reject symlinks and unsafe speakers, and fail if one run stem maps to two identities.

- [ ] **Step 4: Run tests and inspect a read-only real-source inventory sample**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_inventory.py`

Run: `/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_inventory.py --config configs/qwen3_0915all_full_20260914.yaml --sample-only 20`

Expected: tests pass; the sample output includes all three source IDs, excludes both Wuthering Waves categories, and performs no source writes.

- [ ] **Step 5: Commit the inventory unit**

```bash
git add scripts/full_corpus_inventory.py tests/test_full_corpus_inventory.py
git commit -m "Add 0915ALL frozen corpus inventory"
```

### Task 2: Materialize normalized flat chunks and verified GAMESL audio

**Files:**
- Create: `scripts/full_corpus_stage.py`
- Create: `tests/test_full_corpus_stage.py`
- Modify: `scripts/full_corpus_inventory.py`

**Interfaces:**
- Consumes: `InventoryItem`, a chunk root, and `normalize_qwen_input_text(text: str) -> str`.
- Produces: `stage_item(item, chunk_input: Path, gamesl_stage: Path | None) -> StageReceipt`, `verify_padded_wav(path: Path, target_seconds: float) -> dict`, and `publish_gamesl(receipts, gamesl_root, rollback_root) -> dict`.

- [ ] **Step 1: Write failing tests for padding, text normalization, and atomic replacement**

```python
def test_reference_lab_is_normalized_and_padded_before_flattening(tmp_path):
    item = wuwa_item(tmp_path, text="「你好」~")
    receipt = stage_item(item, tmp_path / "flat", tmp_path / "gamesl_stage")
    assert (tmp_path / "flat" / f"{item.run_stem}.txt").read_text() == "你好…"
    edge = verify_padded_wav(tmp_path / "flat" / f"{item.run_stem}.wav", .5)
    assert edge["head_ok"] and edge["tail_ok"]
    assert receipt.pipeline_wav_sha256 == receipt.gamesl_wav_sha256

def test_existing_gamesl_target_is_replaced_with_rollback_record(tmp_path):
    result = publish_gamesl([prepared_receipt(tmp_path)], tmp_path / "GAMESL",
                            tmp_path / "rollback")
    assert result["replaced_count"] == 1
    assert len(result["replacements"]) == 1
```

- [ ] **Step 2: Run stage tests and confirm they fail**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_stage.py`

Expected: collection fails because `scripts.full_corpus_stage` does not exist.

- [ ] **Step 3: Implement atomic staging and exact edge normalization**

Use `finalize_gamedata_speakers.normalize_edge_silence` for GAMEDATA and Wuthering Waves, then validate mono PCM16, finite non-silent samples, and 0.5-second head/tail silence within `max(1024 / sample_rate, 0.03)` seconds. Materialize Wuthering Waves `.lab` and GAMEDATA `.txt` through `normalize_qwen_input_text`; fallback items receive no `.txt`. Write each file to a run-local temporary path, fsync, hash, then `os.replace` into staging. Copy the verified GAMESL WAV into the flat chunk so both hashes match exactly.

`publish_gamesl` must resolve every target under the configured root, move an existing regular target into `<rollback_root>/<game>/<speaker>/`, then atomically install the staged WAV. Reject symlinks and write a replacement row containing target, old hash, and new hash.

- [ ] **Step 4: Run focused tests and real-audio private staging checks**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_stage.py tests/test_qwen3_timestamp_normalization.py`

Run: `/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_stage.py --config configs/qwen3_0915all_full_20260914.yaml --private-check gamedata,wuwa`

Expected: all tests pass; private outputs verify at 0.5 seconds and no public GAMESL target changes.

- [ ] **Step 5: Commit the staging unit**

```bash
git add scripts/full_corpus_inventory.py scripts/full_corpus_stage.py tests/test_full_corpus_stage.py
git commit -m "Add flat staging and GAMESL normalization"
```

### Task 3: Generate bounded Qwen-only chunks and schedule pipeline stages

**Files:**
- Create: `scripts/full_corpus_orchestrator.py`
- Create: `tests/test_full_corpus_orchestrator.py`
- Create: `configs/qwen3_0915all_full_20260914.yaml`

**Interfaces:**
- Consumes: frozen inventory JSON, `stage_item`, and existing `scripts/run_pipeline.py` CLI.
- Produces: `build_chunks(items, limits) -> list[Chunk]`, `materialize_pipeline_config(chunk, task_config, path) -> dict`, `run_canary(config) -> int`, `run_full(config) -> int`, `status.json`, and append-only `events.jsonl`.

- [ ] **Step 1: Write failing scheduler/config tests**

```python
def test_chunks_never_mix_reference_modes_and_conserve_inventory(items):
    chunks = build_chunks(items, ChunkLimits(max_files=4, max_hours=1, max_bytes=10**9))
    assert all(len({row.text_mode for row in chunk.items}) == 1 for chunk in chunks)
    assert sorted(row.run_stem for chunk in chunks for row in chunk.items) == \
           sorted(row.run_stem for row in items)

def test_resolved_pipeline_is_qwen_only(tmp_path, reference_chunk):
    cfg = materialize_pipeline_config(reference_chunk, task_config(), tmp_path / "run.yaml")
    assert cfg["ctc_prealign"]["provider"] == "qwen3_hf"
    assert cfg["reference_mode"] == "authority"
    assert "nvasr" not in json.dumps(cfg, ensure_ascii=False).lower()
    assert cfg["trim"]["normalize_edges"] is False
```

- [ ] **Step 2: Run orchestrator tests and confirm they fail**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_orchestrator.py`

Expected: collection fails because `scripts.full_corpus_orchestrator` does not exist.

- [ ] **Step 3: Add the production task configuration**

```yaml
schema: qwen3-0915all-full-v1
run_root: /mnt/nvme3/qwen3_0915all_full_20260914
output_root: /mnt/Raw/0915ALL
gamesl_root: /mnt/Raw/GAMESL
sources:
  gamedata: /mnt/Raw/GAMEDATA
  v5_0707: /mnt/Raw/v5_0707
  wuwa: /mnt/Raw/Onlinedataset/鸣潮/中文/WutheringWaves2.2_CN/中文 - Chinese
chunk_limits: {max_files: 10000, max_audio_hours: 20, max_bytes: 107374182400}
qwen:
  python: /home/user/miniconda3/bin/python
  asr_model: /mnt/nvme3/models/Qwen3-ASR-1.7B-hf
  forced_aligner_model: /mnt/nvme3/models/Qwen3-ForcedAligner-0.6B-hf
  all_gpus: true
  batch_size: 4
mfa:
  python: /home/user/miniconda3/envs/mfa-dev/bin/python
  num_jobs: 40
postprocess: {workers: 32}
```

Include the exact v5_0707 allow policy, Wuthering Waves exclusions, canary source paths, target edge silence, report roots, model/dictionary paths, timeouts, and current production postprocess thresholds from `configs/bailu_qwen3_0915_reference_test100_punctsync.yaml`.

- [ ] **Step 4: Implement resumable two-lane scheduling**

For each chunk, generate a fresh config with flat `data_dir`, unique `workspace`, `output_staging: false`, `ctc_prealign.provider: qwen3_hf`, `all_gpus: true`, no NVASR key, fixed reference mode, `trim.normalize_edges: false`, and the existing MFA/postprocess settings. Invoke GPU production with:

```python
[mfa_python, "scripts/run_pipeline.py", "--config", cfg,
 "--python", mfa_python, "--stop-after", "prealign", "--validate"]
```

After the raw Qwen manifest is sealed, launch downstream with `--skip-to normalize_punct --validate`. Allow one GPU chunk and one downstream chunk concurrently; write child PID, command, started time, return code, log paths, and receipt hashes. Never use `--overwrite` on a sealed Qwen workspace. Resume only from validated stage receipts.

- [ ] **Step 5: Run tests and dry-run generated commands**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_orchestrator.py tests/test_qwen3_gpu_process_isolation.py tests/test_run_pipeline_subset_denominator.py`

Run: `/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_orchestrator.py prepare --config configs/qwen3_0915all_full_20260914.yaml --sample-only 20 --dry-run`

Expected: tests pass; commands contain Qwen3 model paths, all-GPU mode, distinct workspaces, and no NVASR model path or provider.

- [ ] **Step 6: Commit the orchestrator and task configuration**

```bash
git add scripts/full_corpus_orchestrator.py tests/test_full_corpus_orchestrator.py configs/qwen3_0915all_full_20260914.yaml
git commit -m "Add resumable 0915ALL Qwen scheduler"
```

### Task 4: Classify, publish, and audit every final result

**Files:**
- Create: `scripts/full_corpus_publish.py`
- Create: `tests/test_full_corpus_publish.py`
- Modify: `scripts/full_corpus_orchestrator.py`

**Interfaces:**
- Consumes: a sealed chunk manifest, pipeline output/filtered roots, and frozen inventory.
- Produces: `validate_chunk_result(...) -> ChunkPublication`, `publish_chunk(...) -> dict`, and `build_final_report(...) -> dict`.

- [ ] **Step 1: Write failing conservation and publication tests**

```python
def test_publication_reclassifies_flat_results_and_conserves_denominator(tmp_path):
    publication = validate_chunk_result(chunk_fixture(tmp_path), inventory_fixture())
    assert publication.accepted[0].target.parts[-2:] == ("今汐", "u1.TextGrid")
    assert set(publication.accepted_stems) | set(publication.filtered_stems) | \
           set(publication.failed_stems) == set(publication.input_stems)

def test_publish_replaces_only_exact_target_and_keeps_rollback(tmp_path):
    receipt = publish_chunk(publication_fixture(tmp_path), output_root=tmp_path / "0915ALL")
    assert receipt["replaced_count"] == 1
    assert Path(receipt["rollback_root"]).is_dir()
```

- [ ] **Step 2: Run publication tests and confirm they fail**

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_publish.py`

Expected: collection fails because `scripts.full_corpus_publish` does not exist.

- [ ] **Step 3: Implement fail-closed result validation and atomic merge**

Require each stem to appear in exactly one terminal bucket. Parse every TextGrid, require the five final tiers, require each tier domain to match the pipeline-axis WAV, verify Qwen producer/model receipts and final punctuation projection, then derive the destination speaker only from the frozen manifest. Accepted and filtered files go through private staging and per-file `os.replace`; existing exact targets move to the run rollback tree first. Unknown files, unknown stems, duplicate buckets, symlinks, or denominator gaps block publication.

- [ ] **Step 4: Add final grouped statistics and tests**

`build_final_report` must group input, excluded, accepted, filtered, invalid, producer failure, MFA failure, replacement, duration, and filter reasons by source, game, speaker, and text mode. Store the frozen manifest digest, configuration digest, code revision, per-chunk receipt digests, Qwen model identities, GAMESL receipt digest, and publication digest.

Run: `PYTHONPATH=. python -m pytest -q tests/test_full_corpus_publish.py tests/test_authority_publication_contract.py tests/test_qwen3_filter_compat.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the publication unit**

```bash
git add scripts/full_corpus_publish.py scripts/full_corpus_orchestrator.py tests/test_full_corpus_publish.py
git commit -m "Add audited 0915ALL speaker publication"
```

### Task 5: Verify, run two canaries, and launch the monitored full job

**Files:**
- Modify: `docs/superpowers/plans/2026-09-14-0915all-qwen3-full-corpus.md`
- Runtime artifacts: `/mnt/nvme3/qwen3_0915all_full_20260914/`

**Interfaces:**
- Consumes: completed Tasks 1-4 and production models/hardware.
- Produces: successful canary receipt followed by a running/completed full-run receipt and Luna monitoring evidence.

- [ ] **Step 1: Run static and complete regression verification**

Run:

```bash
/home/user/miniconda3/envs/mfa-dev/bin/python -m py_compile \
  scripts/full_corpus_inventory.py scripts/full_corpus_stage.py \
  scripts/full_corpus_orchestrator.py scripts/full_corpus_publish.py
PYTHONPATH=. /home/user/miniconda3/envs/mfa-dev/bin/python -m pytest -q
```

Expected: compilation succeeds and the full test suite passes.

- [ ] **Step 2: Freeze the complete inventory and preflight dependencies**

Run:

```bash
/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_orchestrator.py \
  prepare --config configs/qwen3_0915all_full_20260914.yaml
/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_orchestrator.py \
  preflight --config configs/qwen3_0915all_full_20260914.yaml
```

Expected: immutable inventory and chunk manifests are sealed; source/exclusion conservation passes; model, dictionary, disk, process, and GPU checks pass.

- [ ] **Step 3: Execute and audit exactly two end-to-end canaries**

Run:

```bash
/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_orchestrator.py \
  canary --config configs/qwen3_0915all_full_20260914.yaml
```

Expected: the fixed Wuthering Waves reference item and LAria fallback item each produce a valid five-tier TextGrid, correct speaker mapping, Qwen-only model identity, and zero fallback/publication contract errors. Canary public output remains under the run root.

- [ ] **Step 4: Launch full processing without another approval gate**

Run:

```bash
/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_orchestrator.py \
  run --config configs/qwen3_0915all_full_20260914.yaml
```

The command must refuse to start without the successful exact canary receipt. Capture its PTY/session ID and start a Luna medium read-only monitor against `status.json`, `events.jsonl`, stage logs, process table, `nvidia-smi`, NVMe use, and published counts.

- [ ] **Step 5: Maintain monitored recovery until the frozen denominator is terminal**

On a recoverable process failure, validate the last completed receipt and resume only the affected unsealed stage. On a systemic contract error, stop new publication and report the exact blocker. Continue until every frozen input has one terminal state and all in-flight chunks are closed.

- [ ] **Step 6: Run final audit and record measured results**

Run:

```bash
/home/user/miniconda3/envs/mfa-dev/bin/python scripts/full_corpus_orchestrator.py \
  audit --config configs/qwen3_0915all_full_20260914.yaml
```

Record actual inventory, exclusion, reference/fallback, accepted, filtered, failed, replacement, duration, throughput, eight-GPU utilization, and final report paths in this plan. Commit only task-owned source, tests, configuration, and measured plan updates.
