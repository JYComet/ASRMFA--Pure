#!/usr/bin/env python3
"""
Streaming batch pipeline — 预取→处理→回传 三阶段流水线并行。

设计原则:
  1. 只移动当前批次需要的数据（避免全量拷贝的长等待）
  2. 预取和回传在后台线程中运行，与处理并行
  3. 本地 SSD 工作区，处理完毕后自动清理
  4. 处理 batch N 时，batch N+1 已在预取，batch N-1 正在回传

架构:
  线程1 (Prefetch): NAS → 本地SSD
  线程2 (Main):     本地管线处理 (调用 run_pipeline.py)
  线程3 (Upload):   本地SSD → NAS

用法:
  # 单数据集
  python scripts/streaming_pipeline.py \
      --nas-ctc //RS3621/.../ASRNEW/my_dataset/wavs \
      --nas-audio //RS3621/.../v5_0707/my_dataset/wavs \
      --nas-output //RS3621/.../ASR_MFA/my_dataset \
      --local-work /ssd/mfa_work --batch-size 500

  # 测试模式
  python scripts/streaming_pipeline.py \
      --nas-ctc /nas/ctc --nas-audio /nas/audio \
      --nas-output /nas/output --local-work /ssd/mfa_work \
      --limit 1000 --batch-size 300
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import shared utilities — path translation, file discovery, MFA env
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_utils import (
    translate_path, resolve_input_path, find_mfa_python, get_mfa_env,
    find_wav, link_or_copy_file, sync_tree_back,
    discover_stems, discover_stems_separated, build_ctc_presence, build_file_index,
    CTC_SUFFIXES, stable_json_digest, cuda_visible_token,
    CTC_RAW_MANIFEST_NAME, CTC_WORK_RECEIPT_NAME,
    validate_ctc_raw_manifest, validate_ctc_work_receipt,
    validate_ctc_run_receipt_v2,
    make_pipeline_accounting_receipt, validate_pipeline_accounting_receipt,
)


def plan_streaming_resources(
    *,
    cpu_budget: int | None = None,
    requested_gpu_workers: int | None = None,
    requested_cpu_workers: int | None = None,
    requested_mfa_jobs: int | None = None,
    requested_mfa_en_jobs: int | None = None,
    config_mfa_jobs: int = 0,
    config_mfa_en_jobs: int = 0,
    batch_size: int = 500,
    batch_count: int | None = None,
    gpu_queue_size: int = 0,
    cpu_queue_size: int = 0,
    pipelined: bool = False,
) -> dict[str, int]:
    """Return one bounded CPU/GPU allocation for every streaming mode.

    ``run_pipeline.py`` can start a process pool for both MFA stages.  Keep
    each pool within the host CPU budget even when several dataset/batch
    workers run concurrently.  Explicit job requests are honoured only up to
    that safe per-worker ceiling; configuration defaults follow the same cap.
    The result intentionally contains only plain integers so launchers and
    tests can use it without constructing argparse objects.
    """
    budget = max(1, int(cpu_budget or (os.cpu_count() or 1)))
    requested_gpu = max(1, int(requested_gpu_workers or 1))
    if batch_count is not None:
        requested_gpu = min(requested_gpu, max(1, batch_count))

    if requested_cpu_workers is None or requested_cpu_workers <= 0:
        # Pipelined CPU work is intentionally less granular by default: an
        # MFA worker is itself a process pool.  Ordinary mode has one CPU
        # worker per dataset/batch worker.
        requested_cpu = max(1, budget // 8) if pipelined else requested_gpu
    else:
        requested_cpu = int(requested_cpu_workers)
    if batch_count is not None:
        requested_cpu = min(requested_cpu, max(1, batch_count))
    cpu_workers = max(1, min(requested_cpu, budget))
    jobs_ceiling = max(1, budget // cpu_workers)

    def _jobs(explicit: int | None, configured: int, default: int) -> int:
        # ``explicit`` is not a license to oversubscribe the machine; it can
        # only reduce the planner's safe ceiling.
        requested = explicit if explicit is not None else configured
        if requested is None or requested <= 0:
            requested = default
        return max(1, min(int(requested), jobs_ceiling))

    mfa_jobs = _jobs(requested_mfa_jobs, config_mfa_jobs, budget)
    mfa_en_jobs = _jobs(requested_mfa_en_jobs, config_mfa_en_jobs, max(1, budget // 16))
    default_gpu_queue = max(2, 2 * requested_gpu)
    default_cpu_queue = max(2, 2 * requested_gpu)
    return {
        "cpu_budget": budget,
        "gpu_workers": requested_gpu,
        "cpu_workers": cpu_workers,
        "mfa_jobs_per_worker": mfa_jobs,
        "mfa_en_jobs_per_worker": mfa_en_jobs,
        "batch_size": max(1, int(batch_size)),
        "gpu_queue_size": max(1, int(gpu_queue_size or default_gpu_queue)),
        "cpu_queue_size": max(1, int(cpu_queue_size or default_cpu_queue)),
    }


def _shutdown_pipelined_cpu_queue(
        cpu_queue: queue.Queue, n_cpu_workers: int,
        stop_event: threading.Event, failure_event: threading.Event,
        sentinel: object, drain_callback) -> bool:
    """Finish CPU sentinels or fail closed on a delayed worker error.

    The sentinel path is intentionally bounded.  A CPU failure may race with
    the first ``put`` while the queue is full; every retry checks both events,
    then delegates queued-work accounting/preservation to the caller.
    """
    if failure_event.is_set() or stop_event.is_set():
        stop_event.set()
        drain_callback()
        return False

    for _ in range(n_cpu_workers):
        while True:
            if failure_event.is_set() or stop_event.is_set():
                stop_event.set()
                drain_callback()
                return False
            try:
                cpu_queue.put(sentinel, timeout=0.25)
                break
            except queue.Full:
                continue

    if failure_event.is_set() or stop_event.is_set():
        stop_event.set()
        drain_callback()
        return False
    return True


def _collect_pipelined_cpu_futures(
        cpu_futures, failure_event: threading.Event,
        stop_event: threading.Event, drain_callback) -> tuple[int, list[str]]:
    """Collect quiescent CPU workers, then drain work left by delayed failure.

    Sentinel insertion can complete before an active worker reports failure.
    The executor context has to quiesce every CPU future first; only then is
    it safe to classify the remaining queue entries as unconsumed.  The
    caller's accounting callback is idempotent for any entries already
    handled by the bounded shutdown path.
    """
    import concurrent.futures

    ok_count = 0
    fail_list: list[str] = []
    try:
        for fut in concurrent.futures.as_completed(cpu_futures):
            try:
                w_ok, w_fails = fut.result()
            except Exception as exc:
                print(f"  [CPU] worker exception: {type(exc).__name__}: {exc}")
                failure_event.set()
                stop_event.set()
                continue
            ok_count += w_ok
            fail_list.extend(w_fails)
    finally:
        if failure_event.is_set() or stop_event.is_set():
            drain_callback()
    return ok_count, fail_list


def _index_wavs_for_stems(audio_dir: Path, stems: list[str]) -> dict[str, Path]:
    """Resolve explicit stems with one directory scan plus narrow fallbacks.

    ``find_wav`` performs several filesystem probes (and may recurse) for every
    stem.  Explicit-stem datasets commonly contain thousands of stems, so scan
    the flat directory and its immediate subdirectories once, retaining the
    same precedence as ``find_wav`` (flat, then ``stem/stem.wav``).  Ambiguous
    or deeper layouts still use ``find_wav`` so path-selection semantics are
    unchanged.
    """
    wanted = set(stems)
    if not wanted:
        return {}

    flat: dict[str, Path] = {}
    nested_direct: dict[str, Path] = {}
    nested_other: dict[str, list[Path]] = {}
    try:
        with os.scandir(str(audio_dir)) as entries:
            top_entries = list(entries)
    except OSError:
        top_entries = []

    for entry in top_entries:
        try:
            is_file = entry.is_file()
            is_dir = entry.is_dir()
        except OSError:
            continue
        if is_file and entry.name.endswith(".wav"):
            stem = entry.name[:-4]
            if stem in wanted and stem not in flat:
                flat[stem] = Path(entry.path)
        elif is_dir:
            try:
                with os.scandir(entry.path) as children:
                    for child in children:
                        try:
                            child_is_file = child.is_file()
                        except OSError:
                            continue
                        if not child_is_file or not child.name.endswith(".wav"):
                            continue
                        stem = child.name[:-4]
                        if stem not in wanted:
                            continue
                        child_path = Path(child.path)
                        if entry.name == stem:
                            nested_direct.setdefault(stem, child_path)
                        else:
                            nested_other.setdefault(stem, []).append(child_path)
            except OSError:
                continue

    resolved: dict[str, Path] = {}
    unresolved: list[str] = []
    for stem in stems:
        if stem in resolved:
            continue
        path = flat.get(stem) or nested_direct.get(stem)
        # A unique one-level nested candidate is equivalent to find_wav's
        # recursive fallback; defer ambiguous candidates to preserve its choice.
        if path is None:
            candidates = nested_other.get(stem, [])
            if len(candidates) == 1:
                path = candidates[0]
        if path is not None:
            resolved[stem] = path
        else:
            unresolved.append(stem)

    # Preserve zero-padded/numeric and deeply nested fallback behaviour.
    for stem in unresolved:
        path = find_wav(audio_dir, stem)
        if path is not None:
            resolved[stem] = path
    return resolved


# ═══════════════════════════════════════════════════════════════
# Batch-level processing — single batch (2000 stems), no threading
# ═══════════════════════════════════════════════════════════════

def _persist_ctc_adj_cache(local_workspace: Path, nas_speaker: Path,
                           raw_manifest_path: Path | None = None) -> bool:
    """Persist one manifest-bound batch without replacing sibling batches."""
    local_adj = local_workspace / "ctc_pretg_adj"
    if not local_adj.exists() or not any(local_adj.iterdir()):
        return False
    raw_manifest_path = raw_manifest_path or (
        local_workspace / "ctc_pretg" / CTC_RAW_MANIFEST_NAME)
    if not raw_manifest_path.is_file() or raw_manifest_path.is_symlink():
        return False
    if validate_ctc_raw_manifest(raw_manifest_path.parent):
        return False
    receipt_path = local_adj / CTC_WORK_RECEIPT_NAME
    if (not receipt_path.is_file() or receipt_path.is_symlink()
            or validate_ctc_work_receipt(local_adj, raw_manifest_path)):
        return False
    # A pipelined dataset has multiple GPU workers in one process.  A single
    # ``ctc_pretg_adj`` target and PID-only staging name makes those workers
    # race; each successful batch replaces the previous batch, leaving only
    # a nondeterministic tail of the run.  Bind both the target and staging
    # path to the immutable raw-manifest bytes instead.
    raw_digest = hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest()
    cache_root = nas_speaker / "ctc_pretg_adj_batches"
    nas_adj = cache_root / raw_digest
    cache_stage = cache_root / (
        f".{raw_digest}.{os.getpid()}.{threading.get_ident()}.staging")
    if nas_adj.is_dir() and not nas_adj.is_symlink():
        return not validate_ctc_work_receipt(nas_adj, raw_manifest_path)
    if cache_stage.exists() or cache_stage.is_symlink():
        return False
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_stage.mkdir(parents=True, exist_ok=False)
        for source in local_adj.rglob("*"):
            if source.is_file() and not source.is_symlink():
                target = cache_stage / source.relative_to(local_adj)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        if validate_ctc_work_receipt(cache_stage, raw_manifest_path):
            return False
        os.replace(cache_stage, nas_adj)
        return not validate_ctc_work_receipt(nas_adj, raw_manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    finally:
        if cache_stage.exists() and not cache_stage.is_symlink():
            shutil.rmtree(cache_stage, ignore_errors=True)


def _restore_ctc_adj_cache(local_workspace: Path, nas_speaker: Path,
                           raw_manifest_path: Path | None = None) -> bool:
    """Restore only a cache bound to the current raw manifest digest."""
    raw_manifest_path = raw_manifest_path or (
        local_workspace / "ctc_pretg" / CTC_RAW_MANIFEST_NAME)
    if not raw_manifest_path.is_file() or raw_manifest_path.is_symlink():
        return False
    if validate_ctc_raw_manifest(raw_manifest_path.parent):
        return False
    raw_digest = hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest()
    nas_adj = nas_speaker / "ctc_pretg_adj_batches" / raw_digest
    if not nas_adj.exists():
        # Read-only compatibility with caches produced before batch isolation.
        nas_adj = nas_speaker / "ctc_pretg_adj"
    if (not nas_adj.exists() or nas_adj.is_symlink()
            or not any(nas_adj.iterdir())):
        return False
    try:
        cached_receipt = json.loads(
            (nas_adj / CTC_WORK_RECEIPT_NAME).read_text(encoding="utf-8"))
        binding = cached_receipt.get("raw_manifest", {})
        if (binding.get("sha256") != hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest()
                or binding.get("identity") != json.loads(
                    raw_manifest_path.read_text(encoding="utf-8")).get("identity")):
            return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    local_adj = local_workspace / "ctc_pretg_adj"
    staging = local_workspace / f".ctc_adj_cache.{os.getpid()}.staging"
    if staging.exists() or staging.is_symlink():
        return False
    staging.mkdir(parents=True)
    try:
        for f in nas_adj.rglob("*"):
            if f.is_file():
                rel = f.relative_to(nas_adj)
                tgt = staging / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(f), str(tgt))
        if local_adj.exists() or local_adj.is_symlink():
            _quarantine_existing_path(local_adj, label="CACHE")
        receipt = json.loads((staging / CTC_WORK_RECEIPT_NAME).read_text(encoding="utf-8"))
        receipt["work_root"] = str(local_adj.resolve())
        receipt["raw_manifest"]["path"] = str(raw_manifest_path.resolve())
        identity = dict(receipt)
        identity.pop("identity", None)
        receipt["identity"] = stable_json_digest(identity)
        (staging / CTC_WORK_RECEIPT_NAME).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        os.replace(staging, local_adj)
        return not validate_ctc_work_receipt(local_adj, raw_manifest_path)
    except Exception:
        return False


def _quarantine_existing_path(path: Path, *, label: str) -> Path | None:
    """Move an old batch artifact aside without deleting it.

    A resumed streaming run must never merge a prior batch workspace into a
    newly staged source set.  In particular, stale CTC shards can make the
    downstream accounting receipt describe fewer stems than the current
    audio directory.  Quarantining is recoverable and leaves the old evidence
    available for forensic inspection.
    """
    if not (path.exists() or path.is_symlink()):
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    candidate = path.with_name(f"{path.name}.{label}.{stamp}.{os.getpid()}")
    suffix = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = path.with_name(
            f"{path.name}.{label}.{stamp}.{os.getpid()}.{suffix}"
        )
        suffix += 1
    shutil.move(str(path), str(candidate))
    print(f"  Quarantined stale batch artifact: {path} -> {candidate}")
    return candidate


def _dataset_batch_token(dataset_name: str) -> str:
    """Return a readable, filesystem-safe identity for a dataset.

    Batch numbers are regenerated when a run resumes from a checkpoint.  The
    local workspace therefore needs the dataset identity as well as the
    number; otherwise a new dataset can inherit an older dataset's CTC
    receipt.  Keep the digest so names that only differ by a path separator
    or another sanitized character remain distinct.
    """
    raw = str(dataset_name)
    safe = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_"
        for ch in raw
    ).strip("._") or "dataset"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe}-{digest}"


def _batch_local_dir(
    local_base: Path, batch_idx: int, dataset_name: str | None = None,
) -> Path:
    """Return an isolated local workspace path for one dataset batch."""
    suffix = ""
    if dataset_name:
        suffix = f"_{_dataset_batch_token(dataset_name)}"
    return local_base / f"batch_{batch_idx:04d}{suffix}"


def _preserve_failed_batch(local_dir: Path) -> Path:
    """Preserve a failed batch while retaining any earlier failure evidence."""
    failed_dir = local_dir.with_name(local_dir.name + ".FAILED")
    if failed_dir.exists() or failed_dir.is_symlink():
        _quarantine_existing_path(failed_dir, label="PREVIOUS")
    shutil.move(str(local_dir), str(failed_dir))
    return failed_dir


def _complete_ctc_producer_stems(ctc_dir: Path) -> set[str]:
    """Return stems with every required producer artifact, flat and exact."""
    ctc_dir = Path(ctc_dir)
    by_suffix: dict[str, set[str]] = {suffix: set() for suffix in CTC_SUFFIXES}
    if not ctc_dir.is_dir():
        return set()
    for path in ctc_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        for suffix in CTC_SUFFIXES:
            if path.name.endswith(suffix):
                by_suffix[suffix].add(path.name[:-len(suffix)])
                break
    producer_stems = set().union(*by_suffix.values()) if by_suffix else set()
    return {stem for stem in producer_stems
            if all(stem in by_suffix[suffix] for suffix in CTC_SUFFIXES)}


def _ctc_producer_stems(ctc_dir: Path) -> set[str]:
    """Return every flat stem mentioned by a producer artifact."""
    ctc_dir = Path(ctc_dir)
    stems: set[str] = set()
    if not ctc_dir.is_dir():
        return stems
    for path in ctc_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        for suffix in CTC_SUFFIXES:
            if path.name.endswith(suffix):
                stems.add(path.name[:-len(suffix)])
                break
    return stems


def _validate_exact_ctc_bundle(
        local_workspace: Path, local_audio: Path, batch_stems: list[str],
        receipt_override: dict | None = None,
    ) -> bool:
    """Require one exact, nonempty producer/receipt/audio axis for a batch."""
    requested = list(batch_stems)
    expected = sorted(requested)
    if not expected or len(set(requested)) != len(requested):
        return False
    ctc_dir = Path(local_workspace) / "ctc_pretg"
    if (_ctc_producer_stems(ctc_dir) != set(expected)
            or _complete_ctc_producer_stems(ctc_dir) != set(expected)):
        return False
    if receipt_override is None:
        receipt_path = ctc_dir / ".ctc_run_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    else:
        receipt = receipt_override
    if not isinstance(receipt, dict):
        return False
    if (receipt.get("input_stems") != expected
            or receipt.get("output_stems") != expected):
        return False
    rows = receipt.get("audio_bindings")
    if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
            or [row.get("stem") for row in rows] != expected):
        return False
    return not validate_ctc_run_receipt_v2(
        receipt, expected_stems=expected, audio_root=Path(local_audio))


def _repair_complete_ctc_receipt(
    local_workspace: Path, local_audio: Path, batch_stems: list[str],
) -> bool:
    """Rebuild the producer receipt when NVASR left a complete bundle behind.

    A late NVASR wrapper error can occur after all TextGrids/labs are written
    but before the receipt is atomically committed.  The bundle is still
    usable; create the minimal producer receipt and let the canonical pipeline
    helper fill and validate audio bindings and artifact hashes.
    """
    ctc_dir = local_workspace / "ctc_pretg"
    if not ctc_dir.exists():
        return False
    requested = list(batch_stems)
    expected = sorted(requested)
    if (not expected or len(set(requested)) != len(requested)
            or _ctc_producer_stems(ctc_dir) != set(expected)
            or _complete_ctc_producer_stems(ctc_dir) != set(expected)):
        return False
    try:
        from pipeline_utils import (compute_model_tree_digest,
                                    write_ctc_run_receipt,
                                    validate_ctc_run_receipt_v2)
        from run_pipeline import _ensure_ctc_axis_receipt
        _existing = ctc_dir / ".ctc_run_receipt.json"
        if _existing.is_file():
            receipt = json.loads(_existing.read_text(encoding="utf-8"))
            if (receipt.get("schema") != "ctc-run-receipt-v2"
                    or receipt.get("input_stems") != expected
                    or receipt.get("output_stems") != expected):
                # A producer receipt for a subset is evidence of an invalid
                # attempt, not permission to redefine this requested batch.
                return False
        else:
            receipt = None

        if receipt is None:
            model_path = Path("/mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR")
            dict_path = PROJECT_ROOT / "dict" / "mfa_ipa.dict"
            model_digest, model_manifest = compute_model_tree_digest(model_path)
            dict_digest = hashlib.sha256(dict_path.read_bytes()).hexdigest()
            write_ctc_run_receipt(
                ctc_dir, actual_argv=["repaired-complete-bundle"],
                asr_python="/home/user/miniconda3/envs/asr/bin/python",
                model_path=model_path, model_tree_digest=model_digest,
                model_file_manifest=model_manifest, dict_path=dict_path,
                dict_digest=dict_digest, input_stems=expected,
                output_stems=expected, audio_bindings=[])

        raw_manifest_path = ctc_dir / CTC_RAW_MANIFEST_NAME
        raw_manifest_before = (
            raw_manifest_path.read_bytes() if raw_manifest_path.is_file() else None
        )
        ctx = {"ctc_pretg": ctc_dir, "audio_dir": local_audio,
               "workspace": local_workspace}
        if _ensure_ctc_axis_receipt(ctx) != 0:
            return False
        repaired = ctx.get("ctc_axis_receipt")
        if not isinstance(repaired, dict):
            return False
        if (repaired.get("input_stems") != expected
                or repaired.get("output_stems") != expected):
            return False
        if validate_ctc_run_receipt_v2(
                repaired, expected_stems=expected, audio_root=local_audio):
            return False
        if raw_manifest_before is not None:
            try:
                raw_manifest_after = raw_manifest_path.read_bytes()
                manifest = json.loads(raw_manifest_after.decode("utf-8"))
                if (raw_manifest_after != raw_manifest_before
                        or validate_ctc_raw_manifest(ctc_dir, manifest)):
                    return False
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return False
            derived_path = Path(ctx.get("ctc_axis_receipt_path", ""))
            if (not derived_path.is_file()
                    or derived_path.is_symlink()):
                return False
            try:
                derived = json.loads(derived_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return False
            if derived != repaired:
                return False
            exact = _validate_exact_ctc_bundle(
                local_workspace, local_audio, requested,
                receipt_override=repaired)
        else:
            exact = _validate_exact_ctc_bundle(
                local_workspace, local_audio, requested)
        if exact:
            print(f"  [GPU] Repaired and validated CTC receipt for {len(expected)} stems")
            return True
        return False
    except Exception as exc:
        print(f"  [GPU] CTC receipt repair failed: {exc}")
    return False


def _flatten_ctc_shards(
    ctc_dir: Path, stems: set[str], suffixes: tuple[str, ...],
) -> set[str]:
    """Project one unambiguous sharded CTC bundle into the flat layout.

    NVASR can finish writing every artifact but exit before its shard merge.
    The canonical receipt validator consumes the flat layout, so copy each
    uniquely identified shard artifact to the batch root before attempting
    receipt repair.  Ambiguous or incomplete stems are excluded instead of
    mixing artifacts from different attempts.
    """
    if not ctc_dir.exists():
        return set()
    indexed: dict[tuple[str, str], list[Path]] = {}
    for artifact in ctc_dir.rglob("*"):
        if not artifact.is_file() or artifact.parent == ctc_dir:
            continue
        for suffix in suffixes:
            if artifact.name.endswith(suffix):
                stem = artifact.name[:-len(suffix)]
                indexed.setdefault((stem, suffix), []).append(artifact)
                break

    complete: set[str] = set()
    for stem in sorted(stems):
        sources: list[tuple[Path, Path]] = []
        valid = True
        for suffix in suffixes:
            target = ctc_dir / f"{stem}{suffix}"
            if target.is_file():
                continue
            candidates = indexed.get((stem, suffix), [])
            if len(candidates) != 1:
                valid = False
                break
            sources.append((candidates[0], target))
        if not valid:
            continue
        try:
            for source, target in sources:
                shutil.copy2(str(source), str(target))
            complete.add(stem)
        except OSError:
            continue
    if complete:
        print(f"  [GPU] Flattened sharded CTC artifacts for "
              f"{len(complete)} stems")
    return complete


# ═══════════════════════════════════════════════════════════════
# Composable batch operations — Stage / Process / Upload / Cleanup
# ═══════════════════════════════════════════════════════════════

def _stage_one_batch(
    ds: dict, batch_idx: int, batch_stems: list[str],
    layout_map: dict, wav_index: dict,
    local_base: Path, mode: str = "ctc_ready",
    text_index: dict[str, Path] | None = None,
) -> tuple[Path, float, int]:
    """Phase 1: NAS → NVMe. Copy WAV + CTC files for one batch.

    Returns:
        (local_dir, elapsed_seconds, unavailable_input_count).
        Callers must treat a non-zero count as a failed stage.
    """
    local_dir = _batch_local_dir(local_base, batch_idx, ds.get("name"))
    # Never reuse a partially processed workspace.  It may contain an older
    # CTC shard/receipt whose stem set differs from this run's staged audio.
    # Empty directories are harmless; non-empty ones are preserved intact.
    if local_dir.exists() or local_dir.is_symlink():
        if local_dir.is_dir() and not local_dir.is_symlink() \
                and not any(local_dir.iterdir()):
            local_dir.rmdir()
        else:
            _quarantine_existing_path(local_dir, label="STALE")
    local_audio = local_dir / "audio"
    local_ctc = local_dir / "ctc"

    nas_ctc_dir = resolve_input_path(ds.get("ctc_dir", ""))
    nas_audio_dir = resolve_input_path(ds.get("audio_dir", ""))
    is_fallback = (mode == "nvrasr_fallback")

    t0 = time.time()
    local_audio.mkdir(parents=True, exist_ok=True)
    if not is_fallback:
        local_ctc.mkdir(parents=True, exist_ok=True)

    import concurrent.futures as _cf2
    copy_tasks: list[tuple[Path, Path]] = []
    regular_audio_tasks: list[tuple[Path, Path]] = []
    missing_audio = 0

    for stem in batch_stems:
        src_wav = wav_index.get(stem)
        if src_wav is None:
            src_wav = find_wav(nas_audio_dir, stem)
        if src_wav:
            # Strict CTC/MFA axis receipts reject symlink audio.  In the raw
            # NVASR fallback route the batch WAV is therefore a real local
            # file; ctc_ready keeps the historical hard-link/symlink path.
            if is_fallback:
                regular_audio_tasks.append((src_wav, local_audio / f"{stem}.wav"))
            else:
                copy_tasks.append((src_wav, local_audio / f"{stem}.wav"))
        else:
            missing_audio += 1

        if is_fallback:
            # Copy .txt reference text alongside audio for NVASR
            txt_src = None
            if text_index and stem in text_index:
                txt_src = text_index[stem]
            else:
                for txt_path in (nas_audio_dir / f"{stem}.txt",
                                 nas_audio_dir / stem / f"{stem}.txt"):
                    if txt_path.exists():
                        txt_src = txt_path
                        break
            if txt_src:
                copy_tasks.append((txt_src, local_audio / f"{stem}.txt"))
        else:
            # ctc_ready: copy CTC files
            layout = layout_map.get(stem, "flat")
            ctc_base = nas_ctc_dir / stem if layout == "nested" else nas_ctc_dir
            for suffix in CTC_SUFFIXES:
                copy_tasks.append(
                    (ctc_base / f"{stem}{suffix}",
                     local_ctc / f"{stem}{suffix}")
                )
            ref_source = ctc_base / f"{stem}_ref.txt"
            if ref_source.is_file() and not ref_source.is_symlink():
                copy_tasks.append((ref_source, local_ctc / ref_source.name))

    # Carry raw lineage metadata only when it describes exactly this batch.
    # A dataset-level manifest must never be copied into a subset workspace.
    if not is_fallback:
        source_raw_manifest = nas_ctc_dir / CTC_RAW_MANIFEST_NAME
        try:
            if source_raw_manifest.is_file() and not source_raw_manifest.is_symlink():
                raw_payload = json.loads(source_raw_manifest.read_text(encoding="utf-8"))
                if sorted(raw_payload.get("stems", [])) == sorted(batch_stems):
                    copy_tasks.append((source_raw_manifest,
                                       local_ctc / CTC_RAW_MANIFEST_NAME))
                    source_receipt = nas_ctc_dir / ".ctc_run_receipt.json"
                    if source_receipt.is_file() and not source_receipt.is_symlink():
                        copy_tasks.append((source_receipt,
                                           local_ctc / source_receipt.name))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # The pipeline will create a fresh batch raw manifest; cache
            # restore remains fail-closed until that manifest exists.
            pass

    if not copy_tasks and not regular_audio_tasks:
        elapsed = time.time() - t0
        print(f"  [STAGE {batch_idx:04d}] WARNING: no files to copy "
              f"(missing_audio={missing_audio})")
        return local_dir, elapsed, missing_audio

    n_workers = min(8, max(1, (len(copy_tasks) + len(regular_audio_tasks)) // 100))
    failed_copies = 0
    with _cf2.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(shutil.copy2, str(s), str(d))
                   for s, d in regular_audio_tasks]
        futures += [pool.submit(shutil.copy2, str(s), str(d))
                    for s, d in copy_tasks]
        for f in _cf2.as_completed(futures):
            try:
                f.result()
            except Exception:
                failed_copies += 1

    if failed_copies:
        print(f"  [STAGE {batch_idx:04d}] WARNING: {failed_copies}/{len(copy_tasks)} "
              f"copies failed (source files missing on NAS?)")

    # Write manifest for run_pipeline.py (ctc_ready only)
    if not is_fallback:
        manifest = {"schema": "ctc-ready-manifest-v2",
                    "stems": batch_stems, "n_stems": len(batch_stems)}
        (local_ctc / "ctc_ready_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False))

    elapsed = time.time() - t0
    return local_dir, elapsed, missing_audio + failed_copies


def _process_one_batch(
    ds: dict, batch_idx: int, batch_stems: list[str],
    local_base: Path, config: Path,
    mfa_python: Path, models_dir: Path,
    nas_output_root: Path,
    batch_size: int, python_path: str | None = None,
    mode: str = "ctc_ready",
    device: str = "",
    restore_cache: bool = True,
    persist_cache_on_failure: bool = True,
    mfa_num_jobs: int = 0,
    mfa_en_num_jobs: int = 0,
    allow_overwrite: bool = True,
    allow_force: bool = True,
) -> bool:
    """Phase 2: NVMe → NVMe. Run run_pipeline.py on locally-staged data.

    All reads and writes are on local NVMe — zero CIFS/NAS I/O during
    processing (unless *restore_cache* is True, which does one NAS read
    to fetch cached CTC adjust output).

    Args:
        restore_cache: If True, attempt to restore ctc_pretg_adj from NAS
                       before processing (saves re-running adjust_ctc).
                       Set False in staged mode (adjust runs fast on NVMe).
        persist_cache_on_failure: If True, upload CTC cache to NAS on failure.
                                  Set False in staged mode (upload happens in Phase 3).
    """
    local_dir = _batch_local_dir(local_base, batch_idx, ds.get("name"))
    local_audio = local_dir / "audio"
    local_ctc = local_dir / "ctc"
    local_output = local_dir / "output"
    local_workspace = local_dir / "workspace"

    nas_output = nas_output_root / ds["name"]
    is_fallback = (mode == "nvrasr_fallback")

    # ── Restore cached adjust output (optional NAS read) ──
    if restore_cache:
        cache_raw_manifest = local_ctc / CTC_RAW_MANIFEST_NAME
        if not cache_raw_manifest.is_file():
            cache_raw_manifest = local_workspace / "ctc_pretg" / CTC_RAW_MANIFEST_NAME
        adj_cached = _restore_ctc_adj_cache(
            local_workspace, nas_output,
            raw_manifest_path=cache_raw_manifest if cache_raw_manifest.is_file() else None)
        if adj_cached:
            print(f"  [PROC  {batch_idx:04d}] Restored ctc_pretg_adj from NAS cache")

    # ── Run pipeline ──
    local_workspace.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(mfa_python),
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(config),
        "--mode", mode,
        "--data-dir", str(local_audio),
        "--output-dir", str(local_output),
        "--workspace", str(local_workspace),
        "--python", str(mfa_python),
    ]
    if allow_overwrite:
        cmd.append("--overwrite")
    if allow_force:
        cmd.append("--force")
    if not is_fallback:
        cmd += ["--ctc-ready", str(local_ctc)]
    if device:
        cmd += ["--device", "cuda:0"]
    if mfa_num_jobs > 0:
        cmd += ["--mfa-jobs", str(mfa_num_jobs)]
    if mfa_en_num_jobs > 0:
        cmd += ["--mfa-en-jobs", str(mfa_en_num_jobs)]

    t0 = time.time()
    env = get_mfa_env(mfa_python, models_dir)
    _mfa_root = local_workspace / "mfa_root"
    _mfa_root.mkdir(parents=True, exist_ok=True)
    env["MFA_ROOT_DIR"] = str(_mfa_root)
    if device:
        gpu_idx = device.replace("cuda:", "")
        if gpu_idx.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_token(int(gpu_idx), env)

    try:
        rc = subprocess.run(
            cmd, env=env,
            timeout=7200, capture_output=False,
        ).returncode
    except subprocess.TimeoutExpired:
        rc = 1

    elapsed = time.time() - t0

    if rc != 0:
        print(f"  [PROC  {batch_idx:04d}] {ds['name']} FAIL (rc={rc}) "
              f"({elapsed:.0f}s)")
        if persist_cache_on_failure:
            _persist_ctc_adj_cache(local_workspace, nas_output)
        # Preserve failed batch directory for forensic analysis
        _failed_dir = local_dir.with_name(local_dir.name + ".FAILED")
        if _failed_dir.exists():
            shutil.rmtree(_failed_dir, ignore_errors=True)
        shutil.move(str(local_dir), str(_failed_dir))
        print(f"  [PROC  {batch_idx:04d}] Preserved: {_failed_dir}")
        return False

    print(f"  [PROC  {batch_idx:04d}] {ds['name']} OK ({elapsed:.0f}s)")
    return True


def _detect_strict_output(workspace: Path) -> Path | None:
    """Detect if strict_ok redirected output to a run-specific directory.

    When postprocess.strict_ok is True, run_pipeline.py writes results to
    workspace/strict_ok_runs/{run_id}/output instead of the configured
    output-dir. Returns the run-specific output path, or None.
    """
    strict_root = workspace / "strict_ok_runs"
    if not strict_root.exists():
        return None
    # Find the most recent run directory
    run_dirs = sorted(strict_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        if run_dir.is_dir() and (run_dir / "output").exists():
            return run_dir / "output"
    return None


def _upload_one_batch(
    local_dir: Path, nas_output_root: Path, ds_name: str,
    batch_idx: int = 0,
) -> bool:
    """Phase 3: NVMe → NAS. rsync batch results to per-batch staging.

    Each batch uploads to .staging/batch_XXXX/ to prevent cross-batch
    file collisions. A separate merge step combines all batch staging
    dirs into the final output.

    Returns True on success, False if any upload failed.
    """
    local_output = local_dir / "output"
    local_workspace = local_dir / "workspace"
    nas_dataset = nas_output_root / ds_name
    # Per-batch isolation: upload to staging, merge happens after all batches
    staging_dir = nas_dataset / ".staging" / f"batch_{batch_idx:04d}"

    t0 = time.time()
    upload_ok = True

    def _verify_tree(source: Path, target: Path) -> bool:
        """Verify the published staging tree before it can be checkpointed."""
        expected = {
            path.relative_to(source).as_posix(): (
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in source.rglob("*") if path.is_file()
        }
        actual = {
            path.relative_to(target).as_posix(): (
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in target.rglob("*") if path.is_file()
        }
        if actual != expected:
            print(f"    Upload verification failed: {source} → {target}")
            return False
        return True

    # Detect strict_ok output redirect: if configured output-dir is empty,
    # check workspace/strict_ok_runs/{run_id}/output
    _strict_output = None
    if local_output.exists() and not any(local_output.iterdir()):
        _strict_output = _detect_strict_output(local_workspace)
    if _strict_output:
        print(f"    strict_ok output detected: {_strict_output}")

    for local_src, nas_rel in [
        (_strict_output if _strict_output else local_output,
         staging_dir / "output"),
        (local_workspace / "filtered", staging_dir / "filtered"),
        (local_workspace / "ctc_pretg_adj", staging_dir / "ctc_pretg_adj"),
    ]:
        if not local_src.exists() or not any(local_src.iterdir()):
            continue
        nas_rel.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            try:
                rc_up = subprocess.run(
                    [rsync, "-a",
                     str(local_src) + "/", str(nas_rel) + "/"],
                    capture_output=True, text=True, timeout=600).returncode
                if rc_up != 0:
                    print(f"    rsync FAILED: rc={rc_up} for {local_src} → {nas_rel}")
                    upload_ok = False
                    break
            except subprocess.TimeoutExpired:
                print(f"    rsync TIMEOUT for {local_src}, falling back to copy")
                rsync = None
        if not rsync:
            try:
                for f in local_src.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(local_src)
                        target = nas_rel / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(f), str(target))
            except Exception as e:
                print(f"    Upload copy error: {e}")
                upload_ok = False
        if upload_ok and not _verify_tree(local_src, nas_rel):
            upload_ok = False
        if not upload_ok:
            break

    elapsed = time.time() - t0
    if upload_ok:
        print(f"  [UPLOAD] {ds_name} batch done ({elapsed:.1f}s)")
    else:
        print(f"  [UPLOAD] {ds_name} batch FAILED")
    return upload_ok


def _cleanup_one_batch_dir(local_dir: Path) -> None:
    """Remove one batch directory from NVMe."""
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)


# ── Convenience: all-in-one batch (original streaming behavior) ──

def run_single_batch(
    ds: dict, batch_idx: int, batch_stems: list[str],
    layout_map: dict, wav_index: dict,
    local_base: Path, config: Path,
    mfa_python: Path, models_dir: Path,
    nas_output_root: Path,
    batch_size: int, python_path: str | None = None,
    mode: str = "ctc_ready",
    text_index: dict[str, Path] | None = None,
    device: str = "",
    mfa_num_jobs: int = 0,
    mfa_en_num_jobs: int = 0,
    config_sha256: str = "",
    cache_sha256: str = "",
    implementation_sha256: str = "",
    receipt_mode: str = "streaming",
    receipt_route: list[str] | None = None,
    require_zero_filtered: bool = False,
) -> bool:
    """Process a single batch end-to-end (original streaming behavior).

    Stage → Process → Upload → Cleanup.  Kept for backward compatibility
    with pipelined mode and direct callers.  For the new staged model,
    use _stage_one_batch / _process_one_batch / _upload_one_batch directly.
    """
    t_start = time.time()
    local_dir, prefetch_elapsed, missing = _stage_one_batch(
        ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
        layout_map=layout_map, wav_index=wav_index,
        local_base=local_base, mode=mode, text_index=text_index,
    )
    if missing:
        print(f"  [BATCH {batch_idx:04d}] {ds['name']} STAGE FAIL "
              f"({missing} missing audio files)")
        return False

    ok = _process_one_batch(
        ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
        local_base=local_base, config=config,
        mfa_python=mfa_python, models_dir=models_dir,
        nas_output_root=nas_output_root,
        batch_size=batch_size, python_path=python_path,
        mode=mode, device=device,
        restore_cache=True, persist_cache_on_failure=True,
        mfa_num_jobs=mfa_num_jobs, mfa_en_num_jobs=mfa_en_num_jobs,
    )
    if not ok:
        return False

    dataset_root = nas_output_root / ds["name"]
    try:
        upload_ok = _publish_batch_to_staging(
        local_dir, dataset_root, batch_idx, batch_stems,
            config_sha256=config_sha256, cache_sha256=cache_sha256,
            implementation_sha256=implementation_sha256,
            require_zero_filtered=require_zero_filtered)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  [BATCH {batch_idx:04d}] publication exception: {exc}")
        upload_ok = False
    if upload_ok:
        _cleanup_one_batch_dir(local_dir)
    else:
        # Keep the local tree available for forensic recovery.  In particular,
        # a failed publication must not become an apparently successful batch
        # merely because its workspace was cleaned up.
        try:
            _preserve_failed_batch(local_dir)
        except (OSError, ValueError):
            pass

    total = time.time() - t_start
    print(f"  [BATCH {batch_idx:04d}] {ds['name']} "
          f"{'OK' if upload_ok else 'UPLOAD FAIL'} ({total:.0f}s)")
    return upload_ok


def _merge_to_nas(src: Path, dst: Path) -> bool:
    """Merge *src* files into *dst* directory on NAS without removing source.

    Uses rsync -a if available, otherwise copy file-by-file.
    Unlike sync_tree_back, this does NOT delete source files (cleanup is
    handled separately).
    """
    dst.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        try:
            rc = subprocess.run(
                [rsync, "-a",
                 str(src) + "/", str(dst) + "/"],
                capture_output=True, text=True, timeout=300).returncode
            if rc == 0:
                return True
            print(f"  rsync failed (rc={rc}) for {src} → {dst}")
            return False
        except subprocess.TimeoutExpired:
            print(f"  rsync timed out after 300s — falling back to file-by-file copy")
    # Fallback: copy file-by-file
    try:
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(target))
        return True
    except Exception as e:
        print(f"  Merge failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset publication contract
# ═══════════════════════════════════════════════════════════════════════════════

STREAMING_BATCH_EVIDENCE_SCHEMA = "streaming-batch-evidence-v1"
STREAMING_DATASET_RECEIPT_SCHEMA = "streaming-dataset-receipt-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Return the deterministic fingerprint of the runnable pipeline code."""
    scripts_root = PROJECT_ROOT / "scripts"
    records: list[tuple[str, str]] = []
    if scripts_root.is_dir() and not scripts_root.is_symlink():
        for path in sorted(scripts_root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                continue
            records.append((path.relative_to(PROJECT_ROOT).as_posix(),
                            _sha256_file(path)))
    if not records:
        raise ValueError(f"no ordinary pipeline Python files under {scripts_root}")
    digest = hashlib.sha256()
    for relative, file_digest in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _publication_tree(root: Path) -> dict[str, dict[str, object]]:
    """Return a deterministic file/size/hash manifest for a publication tree."""
    if not root.is_dir() or root.is_symlink():
        return {}
    result: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"publication tree contains symlink: {path}")
        rel = path.relative_to(root).as_posix()
        result[rel] = {"size": path.stat().st_size,
                       "sha256": _sha256_file(path)}
    return result


def _publication_policy(require_zero_filtered: bool) -> dict[str, bool]:
    """Return the private publication policy carried by new evidence."""
    return {"require_zero_filtered": bool(require_zero_filtered)}


def _evidence_policy_matches(evidence: dict, require_zero_filtered: bool) -> bool:
    """Validate the optional policy without breaking legacy permissive runs."""
    policy = evidence.get("publication_policy")
    if policy is None:
        return not require_zero_filtered
    if (not isinstance(policy, dict)
            or not isinstance(policy.get("require_zero_filtered"), bool)):
        return False
    # A permissive caller may consume an already stricter zero-filter artifact;
    # the strict caller must see an explicit strict policy.
    return (policy["require_zero_filtered"]
            if require_zero_filtered else True)


def _validate_zero_filtered_report_rows(
        rows: list[object], output_stems: set[str]) -> bool:
    """Validate the clean-report semantics required by zero-filter policy."""
    if len(rows) != len(output_stems):
        return False
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        stem = row.get("stem")
        if not isinstance(stem, str) or stem in seen or stem not in output_stems:
            return False
        seen.add(stem)
        if row.get("status") != "ok":
            return False
        if row.get("warnings") != [] or row.get("hard_integrity_reasons") != []:
            return False

        publication = row.get("publication_contract")
        if (not isinstance(publication, dict)
                or publication.get("status") != "verified"
                or publication.get("reasons") != []):
            return False

        nvasr = row.get("nvasr_candidate_provenance")
        if (not isinstance(nvasr, dict)
                or nvasr.get("status") not in {"verified", "not_applicable"}
                or nvasr.get("reasons") != []):
            return False

        english = row.get("english_provenance")
        if (not isinstance(english, dict)
                or english.get("status") not in {"verified", "not_required"}
                or english.get("failed_word_ids") != []):
            return False
        required = english.get("required_words")
        verified = english.get("verified_words")
        if (isinstance(required, bool) or not isinstance(required, int)
                or isinstance(verified, bool) or not isinstance(verified, int)
                or required != verified):
            return False
    return seen == output_stems


def _validate_zero_filtered_report(report_path: Path,
                                   output_stems: set[str]) -> bool:
    try:
        rows = [json.loads(line) for line in
                report_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return _validate_zero_filtered_report_rows(rows, output_stems)


def _validate_frozen_cache_inventory(cache: dict) -> None:
    """Fail closed when an explicit cache no longer names the source WAV set."""
    for dataset in cache.get("datasets", []):
        validation_started = time.monotonic()
        stems = dataset.get("stems")
        if not isinstance(stems, list):
            continue  # legacy multi-dataset caches are validated by their CTC scan
        expected = [str(stem) for stem in stems]
        if expected != sorted(expected) or len(expected) != len(set(expected)):
            raise ValueError(f"cache stems are not sorted and unique: {dataset.get('name')}")
        audio_dir = resolve_input_path(str(dataset.get("audio_dir", "")))
        entries: list[tuple[Path, int]] = []
        with os.scandir(str(audio_dir)) as directory_entries:
            for entry in directory_entries:
                if Path(entry.name).suffix.lower() != ".wav":
                    continue
                entry_stat = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ValueError(f"cache source contains symlink WAV: {audio_dir}")
                if not stat.S_ISREG(entry_stat.st_mode):
                    continue
                entries.append((Path(entry.path),
                                entry_stat.st_size))
        actual = sorted(path.stem for path, _size in entries)
        if actual != expected:
            raise ValueError(
                f"cache/source stem mismatch for {dataset.get('name')}: "
                f"cache={len(expected)} source={len(actual)}")
        inventory = dataset.get("source_inventory")
        if not isinstance(inventory, dict):
            continue
        rows = inventory.get("files")
        if not isinstance(rows, list) or inventory.get("count") != len(rows):
            raise ValueError(f"invalid source inventory: {dataset.get('name')}")
        by_stem = {row.get("stem"): row for row in rows if isinstance(row, dict)}
        if set(by_stem) != set(expected) or len(by_stem) != len(rows):
            raise ValueError(f"source inventory stem mismatch: {dataset.get('name')}")
        for path, size in entries:
            row = by_stem[path.stem]
            actual_sha256 = _sha256_file(path)
            if (row.get("size") != size
                    or row.get("sha256") != actual_sha256):
                raise ValueError(f"source inventory hash mismatch: {path}")
        print(f"  Frozen source validated: {dataset.get('name')} "
              f"({len(entries)} WAVs, "
              f"{time.monotonic() - validation_started:.2f}s)")


def _copy_publication_tree(source: Path, target: Path) -> None:
    """Copy a tree without overwriting a conflicting existing artifact."""
    expected = _publication_tree(source)
    target.mkdir(parents=True, exist_ok=True)
    for rel, metadata in expected.items():
        src = source / rel
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or not dst.is_file():
                raise ValueError(f"publication target conflict: {dst}")
            actual = {"size": dst.stat().st_size, "sha256": _sha256_file(dst)}
            if actual != metadata:
                raise ValueError(f"publication target hash conflict: {dst}")
            continue
        shutil.copyfile(str(src), str(dst))
    # ``target`` accumulates sibling batches, so equality with this shard is
    # neither required nor expected.  Scan/hash it once and verify every file
    # from this shard against its exact metadata.  The former expression
    # rebuilt and re-hashed the whole NAS target once per source file (O(n²))
    # and, when extra sibling files existed, checked only names rather than
    # the expected size/hash pair.
    actual = _publication_tree(target)
    mismatches = {
        rel: {"expected": metadata, "actual": actual.get(rel)}
        for rel, metadata in expected.items()
        if actual.get(rel) != metadata
    }
    if mismatches:
        sample = next(iter(mismatches.items()))
        raise ValueError(
            f"publication verification failed: {source} -> {target}: {sample}")


def _copy_textgrid_tree(source: Path, target: Path) -> None:
    """Copy only terminal TextGrid artifacts into a publication tree."""
    target.mkdir(parents=True, exist_ok=True)
    expected: dict[str, dict[str, object]] = {}
    if source.is_dir() and not source.is_symlink():
        for path in sorted(source.glob("*.TextGrid")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"invalid TextGrid artifact: {path}")
            expected[path.name] = {"size": path.stat().st_size,
                                   "sha256": _sha256_file(path)}
    for rel, metadata in expected.items():
        dst = target / rel
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or not dst.is_file():
                raise ValueError(f"publication target conflict: {dst}")
            if {"size": dst.stat().st_size, "sha256": _sha256_file(dst)} != metadata:
                raise ValueError(f"publication target hash conflict: {dst}")
        else:
            shutil.copyfile(str(source / rel), str(dst))
    if _publication_tree(target) != expected:
        raise ValueError(f"TextGrid publication verification failed: {source} -> {target}")


def _batch_evidence_path(dataset_root: Path, batch_idx: int) -> Path:
    return (dataset_root / ".batch_evidence" /
            f"batch_{batch_idx:04d}" / "batch_receipt.json")


def _load_trusted_batch_evidence(
    dataset_root: Path, batch_idx: int, batch_stems: list[str],
    *, config_sha256: str = "", cache_sha256: str = "",
    implementation_sha256: str = "", evidence_dir: Path | None = None,
    require_zero_filtered: bool = False,
) -> dict | None:
    """Trust a resumed batch only when evidence and published files agree."""
    evidence_path = ((evidence_dir / "batch_receipt.json") if evidence_dir is not None
                     else _batch_evidence_path(dataset_root, batch_idx))
    try:
        if evidence_path.is_symlink() or not evidence_path.is_file():
            return None
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected_stems = sorted(str(stem) for stem in batch_stems)
        if evidence.get("schema") != STREAMING_BATCH_EVIDENCE_SCHEMA:
            return None
        if evidence.get("batch_index") != batch_idx:
            return None
        if evidence.get("stems") != expected_stems:
            return None
        if evidence.get("stems_digest") != stable_json_digest(expected_stems):
            return None
        if not _evidence_policy_matches(evidence, require_zero_filtered):
            return None
        if config_sha256 and evidence.get("config_sha256") != config_sha256:
            return None
        if cache_sha256 and evidence.get("cache_sha256") != cache_sha256:
            return None
        if (implementation_sha256
                and evidence.get("implementation_sha256") != implementation_sha256):
            return None
        receipt = evidence.get("receipt")
        if not isinstance(receipt, dict) or validate_pipeline_accounting_receipt(receipt):
            return None
        if evidence.get("receipt_sha256") != stable_json_digest(receipt):
            return None
        report_path = evidence_path.parent / "postprocess_report.jsonl"
        if evidence.get("report_sha256") != _sha256_file(report_path):
            return None
        receipt = evidence["receipt"]
        if require_zero_filtered and set(receipt["filtered"]["stems"]):
            return None
        if (require_zero_filtered
                and not _validate_zero_filtered_report(
                    report_path, set(receipt["output"]["stems"]))):
            return None
        staging = dataset_root / ".staging" / f"batch_{batch_idx:04d}"
        actual = _publication_tree(staging)
        if actual != evidence.get("artifacts", {}):
            return None
        return evidence
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _find_batch_accounting_receipt(local_dir: Path) -> tuple[dict, Path] | None:
    candidates = sorted(local_dir.rglob(".pipeline_run_receipt_v2.json"))
    for path in candidates:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(receipt, dict) and not validate_pipeline_accounting_receipt(receipt):
                return receipt, path
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


_BATCH_RENAME_ATTEMPTS = 5
_BATCH_RENAME_RETRY_DELAY_S = 0.25


def _replace_batch_path_with_retry(source: Path, target: Path) -> None:
    """Commit one exact batch path despite transient NAS rename denials."""
    last_error: OSError | None = None
    for attempt in range(1, _BATCH_RENAME_ATTEMPTS + 1):
        try:
            source.replace(target)
            return
        except OSError as exc:
            last_error = exc
            if attempt < _BATCH_RENAME_ATTEMPTS:
                time.sleep(_BATCH_RENAME_RETRY_DELAY_S * attempt)
    assert last_error is not None
    raise last_error


def _publish_batch_to_staging(
    local_dir: Path, dataset_root: Path, batch_idx: int, batch_stems: list[str],
    *, config_sha256: str = "", cache_sha256: str = "",
    implementation_sha256: str = "", require_zero_filtered: bool = False,
) -> bool:
    """Publish one batch into isolated staging and immutable batch evidence."""
    expected_stems = sorted(str(stem) for stem in batch_stems)
    # Do the strict local gate before trusted/orphan recovery can rename any
    # evidence.  This preserves the policy's no-write boundary for a batch
    # that is visibly filtered or has an invalid clean report.
    if require_zero_filtered and local_dir.is_dir() and not local_dir.is_symlink():
        early_output = local_dir / "output"
        early_filtered = local_dir / "workspace" / "filtered"
        if any(early_filtered.glob("*.TextGrid")):
            print(f"    Batch publication rejected: filtered stems are forbidden "
                  f"by publication policy for batch_{batch_idx:04d}")
            return False
        early_report = early_output / "postprocess_report.jsonl"
        if not early_report.is_file():
            return False
        early_output_stems = {path.stem for path in early_output.glob("*.TextGrid")}
        if not _validate_zero_filtered_report(early_report, early_output_stems):
            print(f"    Batch publication rejected: clean report policy failed "
                  f"for batch_{batch_idx:04d}")
            return False
    trusted = _load_trusted_batch_evidence(
        dataset_root, batch_idx, expected_stems,
        config_sha256=config_sha256, cache_sha256=cache_sha256,
        implementation_sha256=implementation_sha256,
        require_zero_filtered=require_zero_filtered)
    if trusted is not None:
        return True

    staging = dataset_root / ".staging" / f"batch_{batch_idx:04d}"
    evidence_dir = dataset_root / ".batch_evidence" / f"batch_{batch_idx:04d}"
    # A NAS rename can transiently fail after the immutable staging tree has
    # already committed.  Recover only a temp evidence directory that passes
    # the same stem/config/cache/report/artifact gate as final evidence.  This
    # avoids recomputing a successful batch while remaining fail-closed.
    if ((staging.is_dir() and not staging.is_symlink())
            and not (evidence_dir.exists() or evidence_dir.is_symlink())):
        evidence_parent = evidence_dir.parent
        for orphan in sorted(evidence_parent.glob(
                f".batch_{batch_idx:04d}.tmp.*")):
            if not orphan.is_dir() or orphan.is_symlink():
                continue
            orphan_evidence = _load_trusted_batch_evidence(
                dataset_root, batch_idx, expected_stems,
                config_sha256=config_sha256, cache_sha256=cache_sha256,
                implementation_sha256=implementation_sha256,
                require_zero_filtered=require_zero_filtered,
                evidence_dir=orphan)
            if orphan_evidence is None:
                continue
            try:
                _replace_batch_path_with_retry(orphan, evidence_dir)
            except OSError as exc:
                print(f"    Batch evidence recovery rename failed: {exc}")
                continue
            if _load_trusted_batch_evidence(
                    dataset_root, batch_idx, expected_stems,
                    config_sha256=config_sha256,
                    cache_sha256=cache_sha256,
                    implementation_sha256=implementation_sha256,
                    require_zero_filtered=require_zero_filtered) is not None:
                print(f"    Recovered committed batch evidence: batch_{batch_idx:04d}")
                return True

    local_output = local_dir / "output"
    local_filtered = local_dir / "workspace" / "filtered"
    report = local_output / "postprocess_report.jsonl"
    found = _find_batch_accounting_receipt(local_dir)
    if found is None:
        print(f"    Batch publication rejected: no valid accounting receipt "
              f"for batch_{batch_idx:04d}")
        return False
    if not report.is_file():
        print(f"    Batch publication rejected: postprocess report missing "
              f"for batch_{batch_idx:04d}: {report}")
        return False
    receipt, receipt_path = found
    try:
        output_stems = {p.stem for p in local_output.glob("*.TextGrid")}
        filtered_stems = {p.stem for p in local_filtered.glob("*.TextGrid")}
        if (output_stems & filtered_stems
                or output_stems | filtered_stems != set(receipt["eligible"]["stems"])
                or set(receipt["source"]["stems"]) != set(expected_stems)):
            print(f"    Batch publication rejected: source/output/filtered partition "
                  f"mismatch for batch_{batch_idx:04d}")
            return False
        if set(receipt["output"]["stems"]) != output_stems \
                or set(receipt["filtered"]["stems"]) != filtered_stems:
            print(f"    Batch publication rejected: receipt artifact membership "
                  f"mismatch for batch_{batch_idx:04d}")
            return False
        if require_zero_filtered:
            if filtered_stems:
                print(f"    Batch publication rejected: filtered stems are forbidden "
                      f"by publication policy for batch_{batch_idx:04d}")
                return False
            if not _validate_zero_filtered_report(report, output_stems):
                print(f"    Batch publication rejected: clean report policy failed "
                      f"for batch_{batch_idx:04d}")
                return False
        report_rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        report_stems = [row["stem"] for row in report_rows]
        if len(report_stems) != len(set(report_stems)) \
                or set(report_stems) != set(receipt["eligible"]["stems"]):
            print(f"    Batch publication rejected: report membership mismatch "
                  f"for batch_{batch_idx:04d}")
            return False
        if receipt_path.is_symlink():
            print(f"    Batch publication rejected: accounting receipt is a symlink "
                  f"for batch_{batch_idx:04d}")
            return False
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"    Batch publication rejected: invalid local evidence for "
              f"batch_{batch_idx:04d}: {exc}")
        return False

    if (staging.exists() or staging.is_symlink() or evidence_dir.exists()
            or evidence_dir.is_symlink()):
        # Existing artifacts are never overwritten.  They can only be accepted
        # through the evidence/hash gate above.
        print(f"    Batch publication rejected: existing staging/evidence failed "
              f"identity or hash validation for batch_{batch_idx:04d}")
        return False
    staging_tmp = dataset_root / ".staging" / f".batch_{batch_idx:04d}.tmp.{os.getpid()}"
    evidence_tmp = dataset_root / ".batch_evidence" / f".batch_{batch_idx:04d}.tmp.{os.getpid()}"
    try:
        staging_tmp.mkdir(parents=True, exist_ok=False)
        _copy_textgrid_tree(local_output, staging_tmp / "output")
        _copy_textgrid_tree(local_filtered, staging_tmp / "filtered")
        # Adjusted CTC is an isolated batch artifact.  It is part of the
        # trusted evidence manifest, but intentionally is not copied by the
        # dataset aggregator into terminal output/filtered trees.
        local_adj = local_dir / "workspace" / "ctc_pretg_adj"
        if local_adj.exists() or local_adj.is_symlink():
            _copy_publication_tree(local_adj, staging_tmp / "ctc_pretg_adj")
        artifacts = _publication_tree(staging_tmp)
        evidence_tmp.mkdir(parents=True, exist_ok=False)
        report_target = evidence_tmp / "postprocess_report.jsonl"
        shutil.copyfile(str(report), str(report_target))
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        evidence = {
            "schema": STREAMING_BATCH_EVIDENCE_SCHEMA,
            "batch_index": batch_idx,
            "stems": expected_stems,
            "stems_digest": stable_json_digest(expected_stems),
            "config_sha256": config_sha256,
            "cache_sha256": cache_sha256,
            "publication_policy": _publication_policy(require_zero_filtered),
            "receipt": receipt_payload,
            "receipt_sha256": stable_json_digest(receipt_payload),
            "report_sha256": _sha256_file(report_target),
            "artifacts": artifacts,
        }
        if implementation_sha256:
            evidence["implementation_sha256"] = implementation_sha256
        (evidence_tmp / "batch_receipt.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _replace_batch_path_with_retry(staging_tmp, staging)
        _replace_batch_path_with_retry(evidence_tmp, evidence_dir)
        if _load_trusted_batch_evidence(
                dataset_root, batch_idx, expected_stems,
                config_sha256=config_sha256, cache_sha256=cache_sha256,
                implementation_sha256=implementation_sha256,
                require_zero_filtered=require_zero_filtered) is None:
            return False
        return True
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"    Batch publication exception: {exc}")
        return False


def _load_complete_dataset_receipt(
    dataset_root: Path, dataset_name: str, source_stems: list[str],
    batch_indices: list[int], *, config_sha256: str = "", cache_sha256: str = "",
    implementation_sha256: str = "", receipt_mode: str = "pipelined",
    require_zero_filtered: bool = False,
) -> dict | None:
    """Accept a dataset resume only when both final receipts are complete."""
    dataset_path = dataset_root / ".streaming_dataset_receipt_v1.json"
    final_path = dataset_root / ".pipeline_run_receipt_v2.json"
    try:
        if (dataset_path.is_symlink() or final_path.is_symlink()
                or not dataset_path.is_file() or not final_path.is_file()):
            return None
        dataset_receipt = json.loads(dataset_path.read_text(encoding="utf-8"))
        final = json.loads(final_path.read_text(encoding="utf-8"))
        expected_source = sorted(str(stem) for stem in source_stems)
        expected_indices = sorted(set(batch_indices))
        if (dataset_receipt.get("schema") != STREAMING_DATASET_RECEIPT_SCHEMA
                or dataset_receipt.get("status") != "COMPLETE"
                or dataset_receipt.get("dataset") != dataset_name
                or dataset_receipt.get("source", {}).get("count") != len(expected_source)
                or dataset_receipt.get("source", {}).get("stems_digest")
                != stable_json_digest(expected_source)
                or dataset_receipt.get("config_sha256") != config_sha256
                or dataset_receipt.get("cache_sha256") != cache_sha256
                or (implementation_sha256
                    and dataset_receipt.get("implementation_sha256")
                    != implementation_sha256)
                or dataset_receipt.get("batch_count") != len(expected_indices)
                or dataset_receipt.get("final_receipt_sha256") != _sha256_file(final_path)
                or dataset_receipt.get("receipt_mode", receipt_mode) != receipt_mode
                or validate_pipeline_accounting_receipt(final)
                or final.get("mode") != receipt_mode):
            return None
        batch_hashes = dataset_receipt.get("batch_receipt_sha256")
        final_extra = final.get("extra", {})
        final_output_bucket = final.get("output", {})
        final_filtered_bucket = final.get("filtered", {})
        if (not isinstance(batch_hashes, list)
                or len(batch_hashes) != len(expected_indices)
                or not isinstance(final_extra, dict)
                or not isinstance(final_output_bucket, dict)
                or not isinstance(final_filtered_bucket, dict)
                or final_extra.get("batch_receipts") != batch_hashes):
            return None
        if (implementation_sha256
                and final_extra.get("implementation_sha256")
                != implementation_sha256):
            return None
        if (not _evidence_policy_matches(dataset_receipt,
                                          require_zero_filtered)
                or not _evidence_policy_matches(
                    {"publication_policy": final_extra.get(
                        "publication_policy")}, require_zero_filtered)):
            return None
        if require_zero_filtered and final_filtered_bucket.get("stems", []):
            return None
        for batch_index, expected_hash in zip(expected_indices, batch_hashes):
            evidence_path = _batch_evidence_path(dataset_root, batch_index)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (evidence.get("receipt_sha256") != expected_hash
                    or _load_trusted_batch_evidence(
                        dataset_root, batch_index, evidence.get("stems", []),
                        config_sha256=config_sha256,
                        cache_sha256=cache_sha256,
                        implementation_sha256=implementation_sha256,
                        require_zero_filtered=require_zero_filtered) is None):
                return None
        final_output = {p.stem for p in (dataset_root / "output").glob("*.TextGrid")}
        final_filtered = {p.stem for p in (dataset_root / "filtered").glob("*.TextGrid")}
        if (final_output & final_filtered
                or final_output != set(final_output_bucket.get("stems", []))
                or final_filtered != set(final_filtered_bucket.get("stems", []))
                or dataset_receipt.get("outputs", {}).get("output") != len(final_output)
                or dataset_receipt.get("outputs", {}).get("filtered") != len(final_filtered)):
            return None
        report_path = dataset_root / "output" / "postprocess_report.jsonl"
        if (dataset_receipt.get("outputs", {}).get("report_sha256")
                != _sha256_file(report_path)):
            return None
        if (require_zero_filtered
                and not _validate_zero_filtered_report(
                    report_path, final_output)):
            return None
        return dataset_receipt
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _verify_complete_dataset_receipt(
    dataset_root: Path, dataset_name: str, source_stems: list[str],
    batch_indices: list[int], *, config_sha256: str = "", cache_sha256: str = "",
    implementation_sha256: str = "", receipt_mode: str = "pipelined",
    require_zero_filtered: bool = False,
) -> dict:
    receipt = _load_complete_dataset_receipt(
        dataset_root, dataset_name, source_stems, batch_indices,
        config_sha256=config_sha256, cache_sha256=cache_sha256,
        implementation_sha256=implementation_sha256,
        receipt_mode=receipt_mode,
        require_zero_filtered=require_zero_filtered)
    if receipt is None:
        raise ValueError(f"dataset publication is not COMPLETE: {dataset_name}")
    return receipt


def _aggregate_dataset_publication(
    dataset_root: Path, dataset_name: str, source_stems: list[str],
    batch_indices: list[int], *, config_sha256: str = "", cache_sha256: str = "",
    receipt_mode: str = "pipelined", receipt_route: list[str] | None = None,
    batch_plan: dict[int, list[str]] | None = None,
    implementation_sha256: str = "", require_zero_filtered: bool = False,
) -> dict:
    """Merge verified batches and atomically publish the dataset receipt."""
    expected_source = sorted(str(stem) for stem in source_stems)
    if len(batch_indices) != len(set(batch_indices)):
        raise ValueError("batch indices are not unique")
    existing = _load_complete_dataset_receipt(
        dataset_root, dataset_name, expected_source, batch_indices,
        config_sha256=config_sha256, cache_sha256=cache_sha256,
        implementation_sha256=implementation_sha256,
        receipt_mode=receipt_mode,
        require_zero_filtered=require_zero_filtered)
    if existing is not None:
        return existing
    evidences = []
    for batch_idx in sorted(batch_indices):
        evidence_path = _batch_evidence_path(dataset_root, batch_idx)
        try:
            evidence_stems = json.loads(
                evidence_path.read_text(encoding="utf-8"))["stems"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            raise ValueError(f"missing batch evidence: {batch_idx}")
        if batch_plan is not None:
            planned = sorted(str(stem) for stem in batch_plan.get(batch_idx, []))
            if planned != sorted(str(stem) for stem in evidence_stems):
                raise ValueError(f"batch plan mismatch: {batch_idx}")
            evidence_stems = planned
        evidence = _load_trusted_batch_evidence(
            dataset_root, batch_idx, evidence_stems,
            config_sha256=config_sha256, cache_sha256=cache_sha256,
            implementation_sha256=implementation_sha256,
            require_zero_filtered=require_zero_filtered)
        if evidence is None:
            raise ValueError(f"untrusted batch evidence: {batch_idx}")
        evidences.append(evidence)
    if sorted({e["batch_index"] for e in evidences}) != sorted(batch_indices):
        raise ValueError("batch indices are not unique")

    source: set[str] = set()
    eligible: set[str] = set()
    exclusions: list[dict[str, str]] = []
    output: set[str] = set()
    filtered: set[str] = set()
    shards = []
    report_rows: list[str] = []
    for evidence in sorted(evidences, key=lambda row: row["batch_index"]):
        receipt = evidence["receipt"]
        batch_source = set(receipt["source"]["stems"])
        if source & batch_source:
            raise ValueError("source stems overlap across batches")
        source.update(batch_source)
        eligible.update(receipt["eligible"]["stems"])
        exclusions.extend(receipt.get("exclusions", []))
        output.update(receipt["output"]["stems"])
        filtered.update(receipt["filtered"]["stems"])
        shards.append({"shard_id": f"batch_{evidence['batch_index']:04d}",
                       "stems": receipt["eligible"]["stems"]})
        report_rows.extend((dataset_root / ".batch_evidence" /
                            f"batch_{evidence['batch_index']:04d}" /
                            "postprocess_report.jsonl").read_text(
                                encoding="utf-8").splitlines())
    if source != set(expected_source):
        raise ValueError("source does not equal frozen cache stems")
    if require_zero_filtered:
        if filtered:
            raise ValueError("filtered stems are forbidden by publication policy")
        try:
            aggregate_rows = [json.loads(line) for line in report_rows if line.strip()]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("dataset clean report is invalid") from exc
        if not _validate_zero_filtered_report_rows(aggregate_rows, output):
            raise ValueError("dataset clean report policy failed")
    final_extra = {
        "config_sha256": config_sha256,
        "cache_sha256": cache_sha256,
        "publication_policy": _publication_policy(require_zero_filtered),
        "batch_count": len(evidences),
        "batch_receipts": [e["receipt_sha256"] for e in sorted(
        evidences, key=lambda row: row["batch_index"])],
    }
    if implementation_sha256:
        final_extra["implementation_sha256"] = implementation_sha256
    final_path = dataset_root / ".pipeline_run_receipt_v2.json"
    # Preserve immutable legacy final receipts when a permissive resume is
    # aggregating old evidence that predates this optional policy marker.
    if not require_zero_filtered and final_path.is_file() and not final_path.is_symlink():
        try:
            old_final_probe = json.loads(final_path.read_text(encoding="utf-8"))
            if (isinstance(old_final_probe, dict)
                    and "publication_policy" not in old_final_probe.get("extra", {})):
                final_extra.pop("publication_policy", None)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            pass
    final = make_pipeline_accounting_receipt(
        expected_source, sorted(eligible), exclusions, sorted(output),
        sorted(filtered), run_id=f"streaming-{dataset_name}", mode=receipt_mode,
        route=list(receipt_route or ["gpu", "cpu", "dataset_publish"]),
        paths={"output": str((dataset_root / "output").resolve()),
               "filtered": str((dataset_root / "filtered").resolve()),
               "report": str((dataset_root / "output" / "postprocess_report.jsonl").resolve())},
        shards=shards, extra=final_extra)

    for evidence in evidences:
        batch_root = dataset_root / ".staging" / f"batch_{evidence['batch_index']:04d}"
        _copy_publication_tree(batch_root / "output", dataset_root / "output")
        _copy_publication_tree(batch_root / "filtered", dataset_root / "filtered")
    report_text = "\n".join(row for row in report_rows if row.strip()) + "\n"
    report_path = dataset_root / "output" / "postprocess_report.jsonl"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report_text:
        raise ValueError("dataset report conflict")
    report_tmp = report_path.with_name(report_path.name + f".tmp.{os.getpid()}")
    report_tmp.write_text(report_text, encoding="utf-8")
    report_tmp.replace(report_path)
    final_already_valid = False
    if final_path.exists() or final_path.is_symlink():
        try:
            old_final = json.loads(final_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"conflicting final receipt: {final_path}") from exc
        if old_final != final:
            raise ValueError(f"conflicting final receipt: {final_path}")
        final_already_valid = True
    final_tmp = final_path.with_name(final_path.name + f".tmp.{os.getpid()}")
    if not final_already_valid:
        final_tmp.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        final_tmp.replace(final_path)
    dataset_receipt = {
        "schema": STREAMING_DATASET_RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "dataset": dataset_name,
        "source": {"count": len(expected_source),
                   "stems_digest": stable_json_digest(expected_source)},
        "config_sha256": config_sha256,
        "cache_sha256": cache_sha256,
        "publication_policy": _publication_policy(require_zero_filtered),
        "batch_count": len(evidences),
        "receipt_mode": receipt_mode,
        "receipt_route": list(receipt_route or ["gpu", "cpu", "dataset_publish"]),
        "batch_receipt_sha256": [e["receipt_sha256"] for e in sorted(
            evidences, key=lambda row: row["batch_index"])],
        "final_receipt_sha256": _sha256_file(final_path),
        "outputs": {"output": len(output), "filtered": len(filtered),
                    "report_sha256": _sha256_file(report_path)},
    }
    if implementation_sha256:
        dataset_receipt["implementation_sha256"] = implementation_sha256
    dataset_path = dataset_root / ".streaming_dataset_receipt_v1.json"
    if dataset_path.exists() or dataset_path.is_symlink():
        try:
            old_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"conflicting dataset receipt: {dataset_path}") from exc
        if old_dataset != dataset_receipt:
            raise ValueError(f"conflicting dataset receipt: {dataset_path}")
        return _verify_complete_dataset_receipt(
            dataset_root, dataset_name, expected_source, batch_indices,
            config_sha256=config_sha256, cache_sha256=cache_sha256,
            implementation_sha256=implementation_sha256,
            receipt_mode=receipt_mode,
            require_zero_filtered=require_zero_filtered)
    dataset_tmp = dataset_path.with_name(dataset_path.name + f".tmp.{os.getpid()}")
    dataset_tmp.write_text(json.dumps(dataset_receipt, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    dataset_tmp.replace(dataset_path)
    return dataset_receipt


def _finalize_dataset_publications(
    plans: dict[str, list[tuple[int, list[str]]]],
    trackers: dict[str, dict], nas_output_root: Path,
    completed_set: set[str], failed_set: set[str],
    *, config_sha256: str = "", cache_sha256: str = "",
    receipt_mode: str = "pipelined", receipt_route: list[str] | None = None,
    implementation_sha256: str = "", require_zero_filtered: bool = False,
) -> list[str]:
    """Finalize only datasets whose complete batch plan is trusted."""
    failures: list[str] = []
    for dataset_name, planned in plans.items():
        tracker = trackers.get(dataset_name, {})
        indices = [index for index, _stems in planned]
        batch_plan = {index: stems for index, stems in planned}
        if (tracker.get("fail", 0) or tracker.get("done", 0) != len(planned)
                or len(indices) != len(set(indices))):
            failed_set.add(dataset_name)
            failures.append(dataset_name)
            continue
        source_stems = sorted({stem for _index, stems in planned for stem in stems})
        dataset_root = nas_output_root / dataset_name
        try:
            _aggregate_dataset_publication(
                dataset_root, dataset_name, source_stems, indices,
                config_sha256=config_sha256, cache_sha256=cache_sha256,
                implementation_sha256=implementation_sha256,
                receipt_mode=receipt_mode, receipt_route=receipt_route,
                batch_plan=batch_plan,
                require_zero_filtered=require_zero_filtered)
            _verify_complete_dataset_receipt(
                dataset_root, dataset_name, source_stems, indices,
                config_sha256=config_sha256, cache_sha256=cache_sha256,
                implementation_sha256=implementation_sha256,
                receipt_mode=receipt_mode,
                require_zero_filtered=require_zero_filtered)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError,
                TypeError, ValueError) as exc:
            print(f"  [PUBLISH] {dataset_name} FAILED closed: {exc}")
            failed_set.add(dataset_name)
            failures.append(dataset_name)
            continue
        completed_set.add(dataset_name)
    return failures


# ═══════════════════════════════════════════════════════════════
# MFA 模型预提取 — 避免多 worker 竞争 + 匹配 MFA 内部 zip→flat 逻辑
# ═══════════════════════════════════════════════════════════════

def _ensure_mfa_model_extracted(models_dir: Path | None = None) -> bool:
    """Pre-extract MFA acoustic model so subprocess invocations find it ready.

    Mirrors MFA's own ``Archive.__init__`` (models.py:128-142):
      1. Extract ``mandarin_mfa.zip`` → ``mandarin_mfa_acoustic/``
      2. The zip stores files under ``mandarin_mfa/`` internally, so extraction
         creates ``mandarin_mfa_acoustic/mandarin_mfa/final.mdl``.
      3. MFA then *flattens*: moves files from the nested ``mandarin_mfa/`` up
         to ``mandarin_mfa_acoustic/`` and removes the empty subdirectory.
      4. kalpy validates that ``final.mdl`` lives directly in
         ``mandarin_mfa_acoustic/``.

    Called early in both batch and single-dataset flows.  Idempotent — skips
    if the sentinel ``final.mdl`` already exists flat.

    Returns True if the model is ready.
    """
    if models_dir is None:
        models_dir = PROJECT_ROOT / "models" / "mfa"

    acoustic_dir = models_dir / "extracted_models" / "acoustic" / "mandarin_mfa_acoustic"
    # MFA's kalpy validates flat: final.mdl directly inside mandarin_mfa_acoustic/
    sentinel = acoustic_dir / "final.mdl"

    if sentinel.exists():
        return True  # already correctly extracted

    zip_path = models_dir / "pretrained_models" / "acoustic" / "mandarin_mfa.zip"
    if not zip_path.exists():
        print("  WARNING: MFA acoustic model zip not found — will rely on MFA to download.")
        return False

    import zipfile as _zf

    # Clean up any stale / incorrectly-nested extraction
    if acoustic_dir.exists():
        shutil.rmtree(acoustic_dir, ignore_errors=True)

    print("  Pre-extracting MFA acoustic model (one-time)...")
    acoustic_dir.mkdir(parents=True, exist_ok=True)

    try:
        with _zf.ZipFile(zip_path) as _z:
            _z.extractall(acoustic_dir)
    except Exception as e:
        print(f"  ERROR extracting MFA model: {e}")
        return False

    # ── Flatten (exactly as MFA Archive.__init__ lines 136-142) ──
    # Zip internally: mandarin_mfa/final.mdl
    # After extract:  acoustic_dir/mandarin_mfa/final.mdl
    # After flatten:  acoustic_dir/final.mdl
    files = list(acoustic_dir.iterdir())
    if len(files) == 1 and files[0].is_dir():
        nested = files[0]
        for f in nested.iterdir():
            shutil.move(str(f), str(acoustic_dir / f.name))
        nested.rmdir()
    # ──────────────────────────────────────────────────────────

    if sentinel.exists():
        print("  MFA model ready.")
        return True

    # Last resort: if zip was flat, files should already be in place
    if (acoustic_dir / "final.alimdl").exists():
        print("  MFA model ready (zip was flat).")
        return True

    print("  WARNING: unexpected model extraction result — MFA may fail.")
    return False


# ═══════════════════════════════════════════════════════════════
# 远程文件系统检测 — 自动路由: NAS→流式 / 本地→直接
# ═══════════════════════════════════════════════════════════════

_REMOTE_FS_TYPES = frozenset({"cifs", "nfs", "nfs4", "smbfs", "fuse.sshfs",
                               "glusterfs", "cephfs", "afs"})


def _get_fs_type(path: Path) -> str:
    """Return filesystem type name for *path* (e.g. 'ext4', 'cifs', 'nfs4')."""
    try:
        # Use stat -f on Linux (avoids importing extra modules)
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    # Fallback: parse /proc/mounts
    try:
        path_str = str(path.resolve())
        best_match = ("", "")
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                if path_str.startswith(parts[1]) and len(parts[1]) > len(best_match[1]):
                    best_match = (parts[2], parts[1])
        return best_match[0]
    except Exception:
        pass
    return ""


def _is_remote_fs(path: Path) -> bool:
    """True if *path* is on a remote/network filesystem."""
    return _get_fs_type(path).lower() in _REMOTE_FS_TYPES


def _needs_streaming(data_dir: Path, ctc_dir: Path, local_work: Path | None) -> bool:
    """Determine whether streaming (prefetch+upload) is needed.

    Streaming is used when:
      1. At least one input path is on a remote filesystem, AND
      2. A local work directory is available.
    """
    if local_work is None:
        return False
    data_remote = _is_remote_fs(data_dir)
    ctc_remote = _is_remote_fs(ctc_dir) if ctc_dir != data_dir else data_remote
    return (data_remote or ctc_remote) and (local_work.exists() or local_work.parent.exists())


# ═══════════════════════════════════════════════════════════════
# 批次管理
# ═══════════════════════════════════════════════════════════════

class BatchManager:
    """Split stems into batches and track batch lifecycle."""

    def __init__(self, stems: list[str], batch_size: int,
                 nas_ctc_dir: Path, nas_audio_dir: Path,
                 local_base: Path,
                 layout_map: dict[str, str] | None = None,
                 wav_index: dict[str, Path] | None = None):
        self.stems = stems
        self.batch_size = batch_size
        self.nas_ctc_dir = nas_ctc_dir
        self.nas_audio_dir = nas_audio_dir
        self.local_base = local_base
        self.layout_map = layout_map or {}  # {stem: "flat"|"nested"}
        self.wav_index = wav_index or {}    # {stem: resolved_wav_path}
        self.batches: list[list[str]] = [
            stems[i:i + batch_size]
            for i in range(0, len(stems), batch_size)
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def batch_local_dir(self, batch_idx: int) -> Path:
        return self.local_base / f"batch_{batch_idx:04d}"

    def batch_audio_dir(self, batch_idx: int) -> Path:
        return self.batch_local_dir(batch_idx) / "audio"

    def batch_ctc_dir(self, batch_idx: int) -> Path:
        return self.batch_local_dir(batch_idx) / "ctc"

    def batch_output_dir(self, batch_idx: int) -> Path:
        return self.batch_local_dir(batch_idx) / "output"


# ═══════════════════════════════════════════════════════════════
# 三阶段流水线
# ═══════════════════════════════════════════════════════════════

class StreamingPipeline:
    """预取→处理→回传 三阶段并发流水线。

    背压: prefetch_queue maxsize=2 → 限制本地磁盘占用。
    """

    def __init__(self, batch_mgr: BatchManager,
                 pipeline_script: Path, config_path: Path,
                 mfa_python: Path, models_dir: Path,
                 nas_output_root: Path,
                 prefetch_buffer: int = 4,
                 upload_buffer: int = 4,
                 mfa_num_jobs: int = 0,
                 mfa_en_num_jobs: int = 0,
                 allow_overwrite: bool = True,
                 allow_force: bool = True):
        self.bm = batch_mgr
        self.pipeline_script = pipeline_script
        self.config_path = config_path
        self.mfa_python = mfa_python
        self.models_dir = models_dir
        self.nas_output_root = nas_output_root

        self.mfa_num_jobs = mfa_num_jobs
        self.mfa_en_num_jobs = mfa_en_num_jobs
        self.allow_overwrite = allow_overwrite
        self.allow_force = allow_force

        # Backpressure: prefetch N batches ahead to keep processing saturated
        # while bounding local NVMe usage; upload queue backpressure prevents
        # unbounded NVMe accumulation when NAS is slow.
        self.prefetch_queue: queue.Queue[int] = queue.Queue(
            maxsize=max(1, prefetch_buffer or 4))
        self.upload_queue: queue.Queue[int] = queue.Queue(
            maxsize=max(1, upload_buffer or 4))

        self.stats_lock = threading.Lock()
        self.stats: dict[str, int] = {
            "prefetched": 0, "processed": 0, "uploaded": 0,
            "prefetch_fail": 0, "process_fail": 0, "upload_fail": 0,
        }
        self._stop_event = threading.Event()

    # ── 预取线程 ─────────────────────────────────────────────

    def _prefetch_worker(self):
        """后台: NAS → 本地 SSD (并行文件拷贝)。"""
        import concurrent.futures as _cf

        for batch_idx in range(len(self.bm)):
            if self._stop_event.is_set():
                break

            stems = self.bm.batches[batch_idx]
            local_audio = self.bm.batch_audio_dir(batch_idx)
            local_ctc = self.bm.batch_ctc_dir(batch_idx)
            t0 = time.time()

            print(f"\n  [PREFETCH] batch {batch_idx+1}/{len(self.bm)} "
                  f"({len(stems)} stems) NAS → local ...")

            local_audio.mkdir(parents=True, exist_ok=True)
            local_ctc.mkdir(parents=True, exist_ok=True)

            # ── 并行拷贝: 音频 + CTC 文件 ──
            # 构建拷贝任务列表 (src, dst)，然后用线程池并发执行
            copy_tasks: list[tuple[Path, Path]] = []
            missing_audio = 0

            wav_index = self.bm.wav_index
            nas_audio_dir = self.bm.nas_audio_dir
            nas_ctc_dir = self.bm.nas_ctc_dir
            layout_map = self.bm.layout_map

            for stem in stems:
                # Audio: use pre-built wav_index (O(1), no CIFS)
                src_wav = wav_index.get(stem) if wav_index else None
                if src_wav is None:
                    src_wav = find_wav(nas_audio_dir, stem)
                if src_wav:
                    copy_tasks.append((src_wav, local_audio / f"{stem}.wav"))
                else:
                    missing_audio += 1

                # CTC files: use layout_map from discover_stems
                layout = layout_map.get(stem, "flat")
                ctc_src_base = nas_ctc_dir / stem if layout == "nested" else nas_ctc_dir
                for suffix in CTC_SUFFIXES:
                    copy_tasks.append(
                        (ctc_src_base / f"{stem}{suffix}",
                         local_ctc / f"{stem}{suffix}")
                    )
                ref_source = ctc_src_base / f"{stem}_ref.txt"
                if ref_source.is_file() and not ref_source.is_symlink():
                    copy_tasks.append((ref_source, local_ctc / ref_source.name))

            if missing_audio:
                print(f"    WARNING: audio not found for {missing_audio}/{len(stems)} stems")

            # ── 并行执行拷贝 (I/O-bound, 8 线程足够饱和 CIFS) ──
            n_workers = min(8, len(copy_tasks))
            copied = 0
            failed = 0
            with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(shutil.copy2, str(src), str(dst))
                    for src, dst in copy_tasks
                ]
                for fut in _cf.as_completed(futures):
                    try:
                        fut.result()
                        copied += 1
                    except Exception:
                        failed += 1

            if failed:
                print(f"    ERROR: {failed}/{len(copy_tasks)} file copies failed")

            ok = (missing_audio == 0 and failed == 0)

            # 写 manifest
            manifest = {"schema": "ctc-ready-manifest-v2",
                        "stems": stems, "n_stems": len(stems)}
            (local_ctc / "ctc_ready_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False))

            elapsed = time.time() - t0
            with self.stats_lock:
                if ok:
                    self.stats["prefetched"] += 1
                    print(f"  [PREFETCH] batch {batch_idx+1} done "
                          f"({elapsed:.1f}s, {len(stems)} stems)")
                else:
                    self.stats["prefetch_fail"] += 1
                    print(f"  [PREFETCH] batch {batch_idx+1} FAIL"
                          f" ({elapsed:.1f}s, {len(stems)} stems)"
                          f" — {missing_audio} missing audio, {failed} copy errors")

            if ok:
                self.prefetch_queue.put(batch_idx)

        self.prefetch_queue.put(None)  # Sentinel

    # ── 回传线程 ─────────────────────────────────────────────

    def _upload_worker(self):
        """后台: 本地 SSD → NAS (合并到数据集级目录)。"""
        while not self._stop_event.is_set():
            try:
                batch_idx = self.upload_queue.get(timeout=1)
            except queue.Empty:
                continue
            if batch_idx is None:
                break

            local_dir = self.bm.batch_local_dir(batch_idx)
            local_output = self.bm.batch_output_dir(batch_idx)
            local_filtered = local_dir / "workspace" / "filtered"
            # Merge into dataset-level dirs (not batch subdirs)
            nas_output = self.nas_output_root / "output"
            nas_filtered = self.nas_output_root / "filtered"

            print(f"\n  [UPLOAD] batch {batch_idx+1}/{len(self.bm)} → "
                  f"{self.nas_output_root} ...")
            t0 = time.time()
            ok = True

            try:
                if local_output.exists() and any(local_output.iterdir()):
                    if not _merge_to_nas(local_output, nas_output):
                        ok = False
                if local_filtered.exists() and any(local_filtered.iterdir()):
                    if not _merge_to_nas(local_filtered, nas_filtered):
                        ok = False
            except Exception as e:
                print(f"  [UPLOAD] batch {batch_idx+1} exception: {e}")
                ok = False

            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)

            elapsed = time.time() - t0
            with self.stats_lock:
                if ok:
                    self.stats["uploaded"] += 1
                    print(f"  [UPLOAD] batch {batch_idx+1} done ({elapsed:.1f}s)")
                else:
                    self.stats["upload_fail"] += 1
                    print(f"  [UPLOAD] batch {batch_idx+1} FAILED")

    # ── 处理单个批次 ─────────────────────────────────────────

    def _process_batch(self, batch_idx: int) -> bool:
        """本地处理一个批次 (调用 run_pipeline.py)。"""
        stems = self.bm.batches[batch_idx]
        local_dir = self.bm.batch_local_dir(batch_idx)
        local_audio = self.bm.batch_audio_dir(batch_idx)
        local_ctc = self.bm.batch_ctc_dir(batch_idx)
        local_output = self.bm.batch_output_dir(batch_idx)
        local_workspace = local_dir / "workspace"

        print(f"\n{'='*60}")
        print(f"  PROCESS batch {batch_idx+1}/{len(self.bm)} "
              f"({len(stems)} stems)")
        print(f"  Workspace: {local_workspace}")
        print(f"{'='*60}")

        cmd = [
            str(self.mfa_python),
            str(self.pipeline_script),
            "--config", str(self.config_path),
            "--mode", "ctc_ready",
            "--data-dir", str(local_audio),
            "--output-dir", str(local_output),
            "--workspace", str(local_workspace),
            "--ctc-ready", str(local_ctc),
            "--python", str(self.mfa_python),
        ]
        if self.allow_overwrite:
            cmd.append("--overwrite")
        if self.allow_force:
            cmd.append("--force")
        if self.mfa_num_jobs > 0:
            cmd += ["--mfa-jobs", str(self.mfa_num_jobs)]
        if self.mfa_en_num_jobs > 0:
            cmd += ["--mfa-en-jobs", str(self.mfa_en_num_jobs)]

        t0 = time.time()
        try:
            rc = subprocess.run(
                cmd,
                env=self._get_mfa_env(),
                timeout=7200,
                capture_output=False,
            ).returncode
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: batch {batch_idx+1}")
            rc = 1

        elapsed = time.time() - t0
        ok = (rc == 0)
        print(f"\n  PROCESS batch {batch_idx+1}: "
              f"{'OK' if ok else f'FAIL (rc={rc})'} ({elapsed:.1f}s)")
        return ok

    # ── 主循环 ──────────────────────────────────────────────

    def run(self) -> bool:
        """启动三阶段流水线。  Returns True if all batches processed successfully."""
        print(f"\n{'#'*60}")
        print(f"  Streaming Pipeline")
        print(f"  Batches: {len(self.bm)} × ~{self.bm.batch_size} stems")
        print(f"  Local work: {self.bm.local_base}")
        print(f"  NAS output: {self.nas_output_root}")
        print(f"{'#'*60}")

        total_batches = len(self.bm)

        prefetch_thread = threading.Thread(
            target=self._prefetch_worker, name="prefetch", daemon=True)
        upload_thread = threading.Thread(
            target=self._upload_worker, name="upload", daemon=True)

        prefetch_thread.start()
        upload_thread.start()

        completed = 0
        while completed < total_batches:
            batch_idx = self.prefetch_queue.get()
            if batch_idx is None:
                break

            ok = self._process_batch(batch_idx)
            with self.stats_lock:
                if ok:
                    self.stats["processed"] += 1
                else:
                    self.stats["process_fail"] += 1

            if ok:
                self.upload_queue.put(batch_idx)
            completed += 1

        self.upload_queue.put(None)
        upload_thread.join(timeout=600)
        prefetch_thread.join(timeout=60)

        all_ok = (self.stats["process_fail"] == 0
                  and self.stats["prefetch_fail"] == 0
                  and self.stats["upload_fail"] == 0)
        with self.stats_lock:
            print(f"\n{'#'*60}")
            print(f"  PIPELINE COMPLETE")
            print(f"  Prefetched: {self.stats['prefetched']}/{total_batches}")
            print(f"  Processed:  {self.stats['processed']}/{total_batches}")
            print(f"  Uploaded:   {self.stats['uploaded']}/{total_batches}")
            if self.stats['prefetch_fail']:
                print(f"  Prefetch failures: {self.stats['prefetch_fail']}")
            if self.stats['process_fail']:
                print(f"  Process failures:  {self.stats['process_fail']}")
            if self.stats['upload_fail']:
                print(f"  Upload failures:   {self.stats['upload_fail']}")
            print(f"{'#'*60}")
        return all_ok

    # ── Helpers ──────────────────────────────────────────────

    def _get_mfa_env(self) -> dict:
        return get_mfa_env(self.mfa_python, self.models_dir)


def _run_direct(args, data_dir: Path, ctc_dir: Path, output_dir: Path | None,
                mode: str | None = "ctc_ready"):
    """Pass-through to run_pipeline.py — data is local, no streaming needed."""
    if mode not in {"ctc_ready", "nvrasr_fallback"}:
        raise ValueError("unsupported direct producer mode")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(args.config),
        "--mode", mode,
        "--data-dir", str(data_dir),
    ]
    if mode == "ctc_ready":
        cmd += ["--ctc-ready", str(ctc_dir)]
    if output_dir:
        cmd += ["--output-dir", str(output_dir)]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.python:
        cmd += ["--python", args.python]
    print(f"  CMD: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  ERROR: run_pipeline.py returned {rc}")
        sys.exit(rc)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto-routing MFA pipeline — 自动识别路径类型选择最优模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (auto-detected by filesystem type):
  Remote fs (cifs/nfs) + --local-work  → STREAMING (prefetch → SSD → upload)
  Local fs                              → DIRECT    (delegates to run_pipeline.py)

Examples:
  # NAS → 自动流式
  python scripts/streaming_pipeline.py \\
      --data-dir //RS3621/.../dataset/wavs \\
      --ctc-ready //RS3621/.../ctc/wavs \\
      --local-work /ssd/mfa_work

  # 本地 → 自动直接模式
  python scripts/streaming_pipeline.py \\
      --data-dir /local/audio --ctc-ready /local/ctc

  # 批量
  python scripts/streaming_pipeline.py \\
      --batch-cache cache/batch_all.cache.json \\
      --local-work /ssd/mfa_work --batch-size 500
        """)
    # ── Unified input paths (same as run_pipeline.py) ──
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to audio WAV files.")
    parser.add_argument("--ctc-ready", type=str, default=None,
                        help="Path to CTC files (.TextGrid, .lab, etc.).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory.")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["full", "ctc_ready", "batch_ctc_ready", "nvrasr_fallback", "qwen3asr"],
                        help="Pipeline mode; qwen3asr is rejected by the streaming entry point.")

    # ── NAS paths (legacy, aliases for --data-dir / --ctc-ready / --output-dir) ──
    parser.add_argument("--nas-ctc", type=str, default=None,
                        help=argparse.SUPPRESS)  # alias for --ctc-ready
    parser.add_argument("--nas-audio", type=str, default=None,
                        help=argparse.SUPPRESS)  # alias for --data-dir
    parser.add_argument("--nas-output", type=str, default=None,
                        help=argparse.SUPPRESS)  # alias for --output-dir

    # ── Batch mode ──
    parser.add_argument("--batch-cache", type=Path, default=None,
                        help="Batch cache file (auto-detected if omitted).")

    # ── Streaming control ──
    parser.add_argument("--local-work", type=Path, default=None,
                        help="Local SSD workspace (auto: /ssd/mfa_work, /tmp/mfa_work).")
    parser.add_argument("--direct", action="store_true",
                        help="Force direct mode (skip streaming).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint, start from scratch.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Stems per batch in streaming mode (default: from config or 500).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit total stems (0=all).")
    parser.add_argument("--limit-datasets", type=int, default=0,
                        help="Limit number of datasets in batch mode (0=all).")
    parser.add_argument("--parallel-datasets", type=int, default=None,
                        help="Number of datasets to process in parallel (default: from config or 1).")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs to distribute workers across (default: auto-detect via torch). "
                             "Each worker is assigned --device cuda:{worker_id %% num_gpus}.")
    parser.add_argument("--mfa-jobs", type=int, default=None,
                        help="MFA num_jobs per worker (default: auto = min(4, cpu_count() // parallel)). "
                             "Set > 1 to utilize multi-core per batch. Requires pre-extracted model.")
    parser.add_argument("--mfa-en-jobs", type=int, default=None,
                        help="English MFA num_jobs per worker; capped by the same CPU budget.")
    parser.add_argument("--pipelined", action="store_true",
                        help="Enable pipelined GPU/CPU mode: NVASR and MFA run in parallel stages. "
                             "GPU workers process prealign, CPU workers process MFA alignment. "
                             "Keeps all GPUs busy while all CPU cores run MFA simultaneously.")
    parser.add_argument("--cpu-workers", type=int, default=0,
                        help="Number of CPU workers in pipelined mode (default: auto = cpu_count // 8).")
    parser.add_argument("--prefetch-buffer", type=int, default=0,
                        help="Max prefetched batches on local NVMe (0=auto: 4, 1=serial). "
                             "Larger values trade disk space for throughput.")
    parser.add_argument("--upload-buffer", type=int, default=0,
                        help="Max completed batches awaiting NAS upload (0=auto: 4). "
                             "Backpressure prevents NVMe exhaustion when NAS is slow.")

    # ── Staged execution mode ──
    parser.add_argument("--stage-all", action="store_true", default=True,
                        help="Stage all data to NVMe before processing (default). "
                             "Phase 1: NAS→NVMe, Phase 2: NVMe→NVMe (zero CIFS), "
                             "Phase 3: NVMe→NAS bulk upload.")
    parser.add_argument("--no-stage-all", "--streaming", action="store_false",
                        dest="stage_all",
                        help="Streaming mode: interleave prefetch/process/upload per batch. "
                             "Use when NVMe space is limited.")
    parser.add_argument("--force-stage", action="store_true",
                        help="Force stage-all even if NVMe space appears insufficient.")

    # ── Pipeline config ──
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--python", type=str, default=None,
                        help="MFA Python path (auto-detect).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Disable --overwrite in child processes (respect strict configs).")
    parser.add_argument("--no-force", action="store_true",
                        help="Disable --force in child processes (respect strict configs).")
    args = parser.parse_args()

    # ── Load config for streaming defaults ──
    import yaml as _yaml
    cfg = {}
    if args.config.exists():
        with open(args.config, 'r', encoding='utf-8') as _f:
            cfg = _yaml.safe_load(_f) or {}
    if args.mode == "qwen3asr" or cfg.get("mode") == "qwen3asr":
        print("ERROR: streaming_pipeline.py does not support mode=qwen3asr; use run_pipeline.py")
        return 1
    streaming_cfg = cfg.get("streaming", {})
    args._config = cfg  # stash for reuse by run_batch

    # --local-work: CLI > config > error
    # Supports single path (str) or list of paths for multi-NVMe setups
    if args.local_work is None:
        cfg_val = streaming_cfg.get("local_work", "")
        if cfg_val:
            if isinstance(cfg_val, list):
                # Multi-drive: resolve each path
                args.local_work = tuple(
                    Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
                    for p in cfg_val
                )
            else:
                p = Path(cfg_val)
                args.local_work = p if p.is_absolute() else PROJECT_ROOT / p
        else:
            parser.error("--local-work is required (or set 'streaming.local_work' in config)")

    # Normalize to tuple for uniform handling
    _lw = args.local_work
    if isinstance(_lw, (str, Path)):
        _lw = (_lw if isinstance(_lw, Path) else Path(_lw),)
    args._local_work_drives = tuple(
        p if p.is_absolute() else PROJECT_ROOT / p for p in _lw
    )

    # --prefetch-buffer / --upload-buffer: CLI > config > default 4
    if args.prefetch_buffer <= 0:
        args.prefetch_buffer = streaming_cfg.get("prefetch_buffer", 4)
    if args.upload_buffer <= 0:
        args.upload_buffer = streaming_cfg.get("upload_buffer", 4)

    # --batch-size: CLI > config > default 500
    if args.batch_size is None:
        cfg_val = streaming_cfg.get("batch_size", 0)
        args.batch_size = cfg_val if cfg_val > 0 else 500

    # --gpus: CLI > config > auto-detect
    if args.gpus is None:
        args.gpus = streaming_cfg.get("num_gpus", 0)
    if args.gpus <= 0:
        # Auto-detect GPU count
        try:
            import torch as _torch
            args.gpus = _torch.cuda.device_count()
        except ImportError:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10)
                args.gpus = len([l for l in result.stdout.splitlines() if l.strip()])
            except Exception:
                args.gpus = 1  # safe default
    if args.gpus < 1:
        args.gpus = 1
    print(f"  GPUs detected: {args.gpus}")

    # --batch-cache: CLI > config > auto-derive from config path (only if
    # no single-dataset paths are given, to avoid hijacking single-dataset mode)
    if args.batch_cache is None:
        cfg_val = streaming_cfg.get("batch_cache", "")
        if cfg_val:
            p = Path(cfg_val)
            args.batch_cache = p if p.is_absolute() else PROJECT_ROOT / p
        elif not (args.data_dir or args.ctc_ready or args.nas_ctc or args.nas_audio):
            derived = PROJECT_ROOT / "cache" / f"{args.config.stem}.cache.json"
            if derived.exists():
                args.batch_cache = derived

    # ── Unify legacy NAS args with standard args ──
    data_dir_arg = args.data_dir or args.nas_audio
    ctc_dir_arg = args.ctc_ready or args.nas_ctc
    output_dir_arg = args.output_dir or args.nas_output

    # ── Batch mode ──
    if args.batch_cache:
        ok = run_batch(args)
        if not ok:
            return 1
        return 0

    # ── Single-dataset mode ──
    if not data_dir_arg:
        parser.error("--data-dir is required (or --nas-audio for legacy mode)")

    ctc_dir_arg = ctc_dir_arg or data_dir_arg  # default CTC = same as audio

    # Resolve paths (UNC → Linux)
    data_dir = resolve_input_path(data_dir_arg)
    ctc_dir = resolve_input_path(ctc_dir_arg)
    output_dir = resolve_input_path(output_dir_arg) if output_dir_arg else None

    # Detect filesystem type
    data_fs = _get_fs_type(data_dir)
    ctc_fs = _get_fs_type(ctc_dir) if ctc_dir != data_dir else data_fs
    remote = (_get_fs_type(data_dir).lower() in _REMOTE_FS_TYPES
              or _get_fs_type(ctc_dir).lower() in _REMOTE_FS_TYPES)

    print(f"Data dir:  {data_dir}  [{data_fs}]")
    if ctc_dir != data_dir:
        print(f"CTC dir:   {ctc_dir}  [{ctc_fs}]")

    use_streaming = (not args.direct and args.local_work is not None and remote)

    if use_streaming:
        _staged = getattr(args, 'stage_all', True)
        _mode_label = "STAGED" if _staged else "STREAMING"
        print(f"Mode:      {_mode_label} (remote fs → prefetch to {args.local_work})")
        ok = run_single_dataset(
            nas_ctc=str(ctc_dir), nas_audio=str(data_dir),
            nas_output=str(output_dir or (ctc_dir.parent / "mfa_output")),
            config=args.config, local_work=args.local_work,
            batch_size=args.batch_size, limit=args.limit,
            python_path=args.python,
            prefetch_buffer=args.prefetch_buffer,
            upload_buffer=args.upload_buffer,
            staged=_staged,
            mfa_num_jobs=(args.mfa_jobs if args.mfa_jobs is not None
                          else cfg.get("mfa", {}).get("num_jobs", 0)),
            mfa_en_num_jobs=cfg.get("mfa_en", {}).get("num_jobs", 0),
            mode=_streaming_producer_mode(args.mode, cfg.get("mode")),
        )
        if not ok:
            return 1
    else:
        if remote:
            print("Mode:      DIRECT (remote fs, no --local-work; will be slow)")
            print("           Tip: add --local-work /ssd/mfa_work for streaming")
        else:
            print("Mode:      DIRECT (local fs)")
        _run_direct(
            args, data_dir, ctc_dir, output_dir,
            mode=_streaming_producer_mode(args.mode, cfg.get("mode")))
    return 0


def _streaming_producer_mode(
        requested_mode: str | None, config_mode: str | None = None,
) -> str | None:
    """Resolve the producer identity used by the composable batch route.

    Historically this wrapper treated every single-dataset invocation as
    ``ctc_ready``.  Preserve that compatibility for the legacy ``full`` and
    ``batch_ctc_ready`` entry modes, but never erase an explicit/configured
    raw fallback identity.  Unknown producer modes are rejected by callers.
    """
    mode = str(requested_mode or config_mode or "ctc_ready").strip()
    if mode == "nvrasr_fallback":
        return mode
    if mode in {"ctc_ready", "batch_ctc_ready", "full"}:
        return "ctc_ready"
    return None


def run_single_dataset(
    nas_ctc: str, nas_audio: str, nas_output: str,
    config: Path, local_work: Path,
    batch_size: int = 500, limit: int = 0,
    python_path: str | None = None,
    stems_override: list[str] | None = None,
    prefetch_buffer: int = 4,
    upload_buffer: int = 4,
    staged: bool = True,
    mfa_num_jobs: int = 0,
    mfa_en_num_jobs: int = 0,
    device: str = "",
    parallel_batches: int = 4,
    mode: str | None = None,
) -> bool:
    """Run pipeline for a single dataset.  Returns True on success.

    Args:
        staged: If True (default), use Stage All → Process All → Upload All.
                If False, use streaming (interleaved prefetch/process/upload).
    """
    import concurrent.futures as _cf3

    if mode is None:
        config_mode = None
        try:
            import yaml as _yaml_single
            payload = _yaml_single.safe_load(
                Path(config).read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                config_mode = payload.get("mode")
        except (OSError, UnicodeError, TypeError, ValueError):
            config_mode = None
        mode = _streaming_producer_mode(None, config_mode)
    else:
        mode = _streaming_producer_mode(mode)
    if mode not in {"ctc_ready", "nvrasr_fallback"}:
        print("ERROR: single-dataset route requires ctc_ready or "
              "nvrasr_fallback producer mode")
        return False

    # ── Ensure MFA model is pre-extracted before subprocess starts ──
    _ensure_mfa_model_extracted()

    # ── Resolve NAS paths (UNC → Linux translation) ──
    nas_ctc_dir = resolve_input_path(nas_ctc)
    nas_audio_dir = resolve_input_path(nas_audio)
    nas_output_root = resolve_input_path(nas_output)

    # CTC dir may not pre-exist for nvrasr_fallback mode (raw audio,
    # no prior NVASR run).  Create it so the pipeline can write CTC output.
    _has_pre_ctc = nas_ctc_dir.exists()
    if not _has_pre_ctc:
        if mode != "nvrasr_fallback":
            print(f"ERROR: CTC-ready input dir not found: {nas_ctc_dir}")
            return False
        nas_ctc_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Note: CTC dir created (nvrasr_fallback): {nas_ctc_dir}")
    if not nas_audio_dir.exists():
        print(f"ERROR: NAS audio dir not found: {nas_audio_dir}")
        print(f"  (translated from: {nas_audio})")
        return False

    print(f"\nNAS CTC:    {nas_ctc_dir}")
    print(f"NAS audio:  {nas_audio_dir}")
    print(f"NAS output: {nas_output_root}")

    # ── Discover stems (single scandir, O(1) set validation) ──
    print("\nDiscovering stems ...")
    if stems_override is not None:
        requested_stems = list(stems_override)
        stems = list(requested_stems)
        layout_map = {s: "nested" for s in stems}
        wav_index = {}
        for s in stems:
            w = find_wav(nas_audio_dir, s)
            if w:
                wav_index[s] = w
        stems = [s for s in stems if s in wav_index]
        print(f"  Using {len(stems)} stems (override)")
        if len(stems) != len(requested_stems):
            print(f"ERROR: stem override resolved {len(stems)}/"
                  f"{len(requested_stems)} WAV files")
            return False
    elif mode == "nvrasr_fallback":
        wav_index = build_file_index(nas_audio_dir, ".wav")
        stems = sorted(wav_index)
        layout_map = {
            stem: ("flat" if path.parent == nas_audio_dir else "nested")
            for stem, path in wav_index.items()
        }
    else:
        stems, _, layout_map, wav_index = discover_stems_separated(
            nas_ctc_dir, nas_audio_dir, require_all=True)
    if limit > 0:
        stems = stems[:limit]
    print(f"  Found {len(stems)} valid stems"
          + (f" (limited from discovery)" if limit > 0 else ""))

    if not stems:
        print("No valid stems found; no work remains.")
        return True

    # ── Find MFA Python ──
    if python_path:
        mfa_python = Path(python_path)
    else:
        mfa_python = find_mfa_python()
    if not mfa_python or not mfa_python.exists():
        print("ERROR: Cannot find MFA Python. Use --python PATH.")
        return False
    print(f"MFA Python: {mfa_python}")

    models_dir = PROJECT_ROOT / "models" / "mfa"
    local_work.mkdir(parents=True, exist_ok=True)

    batch_mgr = BatchManager(
        stems=stems,
        batch_size=batch_size,
        nas_ctc_dir=nas_ctc_dir,
        nas_audio_dir=nas_audio_dir,
        local_base=local_work,
        layout_map=layout_map,
        wav_index=wav_index,
    )
    resources = plan_streaming_resources(
        requested_cpu_workers=parallel_batches,
        requested_mfa_jobs=mfa_num_jobs if mfa_num_jobs > 0 else None,
        config_mfa_en_jobs=mfa_en_num_jobs,
        batch_size=batch_size,
        batch_count=len(batch_mgr),
    )
    parallel_batches = resources["cpu_workers"]
    mfa_num_jobs = resources["mfa_jobs_per_worker"]
    mfa_en_num_jobs = resources["mfa_en_jobs_per_worker"]

    if not staged and mode == "nvrasr_fallback":
        # The legacy interleaved class copies a pre-existing CTC bundle and
        # has no raw-ASR prefetch contract.  Route fallback through the
        # composable staged operations instead of silently relabeling it.
        print("  Note: nvrasr_fallback uses the composable staged route")
        staged = True

    if not staged:
        # ── Streaming mode (original behavior) ──
        pipeline = StreamingPipeline(
            batch_mgr=batch_mgr,
            pipeline_script=PROJECT_ROOT / "scripts" / "run_pipeline.py",
            config_path=config,
            mfa_python=mfa_python,
            models_dir=models_dir,
            nas_output_root=nas_output_root,
            prefetch_buffer=prefetch_buffer,
            upload_buffer=upload_buffer,
            mfa_num_jobs=mfa_num_jobs,
            mfa_en_num_jobs=mfa_en_num_jobs,
        )
        return pipeline.run()

    # ═══════════════════════════════════════════════════════════
    # Staged mode: Stage All → Process All → Upload All
    # ═══════════════════════════════════════════════════════════
    print(f"\n  Mode: STAGED — {len(batch_mgr)} batches, "
          f"{batch_size} stems/batch")

    ds = {"name": nas_output_root.name,
          "ctc_dir": str(nas_ctc_dir),
          "audio_dir": str(nas_audio_dir)}

    # ── Phase 1: Stage all batches NAS→NVMe ──
    print(f"\n  PHASE 1: Staging {len(batch_mgr)} batches NAS → NVMe ...")
    staged_dirs: dict[int, Path] = {}
    stage_failures: set[int] = set()
    text_index = (build_file_index(nas_audio_dir, ".txt")
                  if mode == "nvrasr_fallback" else None)
    t_stage = time.time()

    for bi in range(len(batch_mgr)):
        bstems = batch_mgr.batches[bi]
        try:
            local_dir, elapsed, missing = _stage_one_batch(
                ds=ds, batch_idx=bi, batch_stems=bstems,
                layout_map=layout_map, wav_index=wav_index,
                local_base=local_work, mode=mode,
                text_index=text_index,
            )
            if missing:
                print(f"  [STAGE] FAIL batch {bi}: {missing} unavailable inputs")
                stage_failures.add(bi)
            else:
                staged_dirs[bi] = local_dir
        except Exception as exc:
            print(f"  [STAGE] FAIL batch {bi}: {exc}")
            stage_failures.add(bi)

    print(f"  PHASE 1 DONE: {len(staged_dirs)}/{len(batch_mgr)} staged "
          f"({time.time() - t_stage:.0f}s)")

    # ── Phase 2: Process all batches NVMe→NVMe ──
    import concurrent.futures as _cf4
    print(f"\n  PHASE 2: Processing {len(staged_dirs)} batches on NVMe "
          f"(zero CIFS I/O, {parallel_batches} workers) ...")
    t_proc = time.time()
    proc_ok = 0
    proc_fail = 0
    processed_batches: set[int] = set()
    proc_lock = threading.Lock()

    def _proc_one(bi: int) -> bool:
        bstems = batch_mgr.batches[bi]
        return _process_one_batch(
            ds=ds, batch_idx=bi, batch_stems=bstems,
            local_base=local_work, config=config,
            mfa_python=mfa_python, models_dir=models_dir,
            nas_output_root=nas_output_root,
            batch_size=batch_size, python_path=python_path,
            mode=mode, device=device,
            restore_cache=False,
            persist_cache_on_failure=False,
            mfa_num_jobs=mfa_num_jobs,
            mfa_en_num_jobs=mfa_en_num_jobs,
        )

    n_proc = min(parallel_batches, len(staged_dirs))
    if n_proc <= 1:
        for bi in sorted(staged_dirs.keys()):
            try:
                ok = _proc_one(bi)
            except Exception as exc:
                print(f"  [PROC] CRASH batch {bi}: {exc}")
                ok = False
            if ok:
                proc_ok += 1
                processed_batches.add(bi)
            else:
                proc_fail += 1
    else:
        with _cf4.ThreadPoolExecutor(max_workers=n_proc) as pool:
            futures = {pool.submit(_proc_one, bi): bi for bi in staged_dirs}
            for fut in _cf4.as_completed(futures):
                try:
                    if fut.result():
                        with proc_lock:
                            proc_ok += 1
                            processed_batches.add(futures[fut])
                    else:
                        with proc_lock:
                            proc_fail += 1
                except Exception as exc:
                    bi = futures[fut]
                    print(f"  [PROC] CRASH batch {bi}: {exc}")
                    with proc_lock:
                        proc_fail += 1

    print(f"  PHASE 2 DONE: {proc_ok} OK, {proc_fail} FAIL "
          f"({time.time() - t_proc:.0f}s)")

    # ── Phase 3: Upload all results NVMe→NAS ──
    print(f"\n  PHASE 3: Uploading {proc_ok} batches NVMe → NAS ...")
    t_up = time.time()
    up_ok = 0
    up_fail = 0

    for bi in sorted(processed_batches):
        local_dir = staged_dirs[bi]
        if _upload_one_batch(
            local_dir=local_dir,
            nas_output_root=nas_output_root,
            ds_name=ds["name"],
            batch_idx=bi,
        ):
            up_ok += 1
        else:
            up_fail += 1
        _cleanup_one_batch_dir(local_dir)

    # Cleanup any remaining staged dirs
    for local_dir in staged_dirs.values():
        if local_dir.exists():
            _cleanup_one_batch_dir(local_dir)

    total_elapsed = time.time() - t_stage
    print(f"  PHASE 3 DONE: {up_ok} uploaded, {up_fail} FAIL "
          f"({time.time() - t_up:.0f}s)")
    print(f"  SINGLE DATASET TOTAL: {total_elapsed:.0f}s")

    expected_batches = len(batch_mgr)
    return (
        expected_batches > 0
        and not stage_failures
        and len(staged_dirs) == expected_batches
        and proc_fail == 0
        and proc_ok == expected_batches
        and len(processed_batches) == expected_batches
        and up_fail == 0
        and up_ok == expected_batches
    )


CHECKPOINT_SCHEMA = "streaming-checkpoint-v2"
BATCH_PROGRESS_SCHEMA = "streaming-batch-progress-v2"


def _checkpoint_path_identity(path: Path) -> dict[str, object]:
    """Return a bounded identity for a configured model artifact."""
    path = Path(path)
    identity: dict[str, object] = {"path": str(path.resolve(strict=False))}
    try:
        if path.is_file():
            identity.update({
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        elif path.is_dir():
            rows = []
            for child in sorted(path.rglob("*")):
                if child.is_file() and not child.is_symlink():
                    stat = child.stat()
                    rows.append((child.relative_to(path).as_posix(),
                                 stat.st_size, stat.st_mtime_ns))
            identity.update({"kind": "directory",
                             "tree_digest": stable_json_digest(rows)})
        else:
            identity["kind"] = "missing"
    except OSError as exc:
        identity.update({"kind": "unreadable", "error": str(exc)})
    return identity


def _checkpoint_model_identity(config: object) -> list[dict[str, object]]:
    """Collect model-like config paths without probing unrelated runtime data."""
    found: list[tuple[str, str]] = []

    def visit(value: object, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                child_prefix = f"{prefix}.{key_text}" if prefix else key_text
                if (isinstance(child, str)
                        and "model" in key_text.lower()):
                    found.append((child_prefix, child))
                else:
                    visit(child, child_prefix)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(config)
    identities = []
    for key, raw in sorted(set(found)):
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        identities.append({"config_key": key,
                           **_checkpoint_path_identity(path)})
    return identities


def _checkpoint_identity(cache: dict, args) -> dict[str, object]:
    """Build a stable resume contract for one batch-cache invocation."""
    datasets = []
    for dataset in cache.get("datasets", []):
        row = dict(dataset)
        if isinstance(row.get("stems"), list):
            row["stems"] = sorted(str(stem) for stem in row["stems"])
        datasets.append(row)
    datasets.sort(key=lambda row: (str(row.get("name", "")),
                                   stable_json_digest(row)))
    batch_size = max(1, int(getattr(args, "batch_size", 500) or 500))
    batch_ids = []
    for dataset in datasets:
        stems = dataset.get("stems")
        if not isinstance(stems, list):
            stems = []
        limit = max(0, int(getattr(args, "limit", 0) or 0))
        if limit:
            stems = stems[:limit]
        for batch_index in range(0, len(stems) or 1, batch_size):
            chunk = stems[batch_index:batch_index + batch_size]
            batch_ids.append({
                "dataset": str(dataset.get("name", "")),
                "batch_index": batch_index // batch_size,
                "stem_count": len(chunk),
                "stems_digest": stable_json_digest(chunk),
            })
    config = getattr(args, "_config", {}) or {}
    implementation_sha256 = getattr(args, "_implementation_sha256", "")
    if not implementation_sha256:
        implementation_sha256 = _implementation_sha256()
    cli = {
        "batch_size": batch_size,
        "limit": max(0, int(getattr(args, "limit", 0) or 0)),
        "parallel_datasets": getattr(args, "parallel_datasets", None),
        "mfa_jobs": getattr(args, "mfa_jobs", None),
        "mfa_en_jobs": getattr(args, "mfa_en_jobs", None),
        "gpus": getattr(args, "gpus", None),
        "pipelined": bool(getattr(args, "pipelined", False)),
    }
    return {
        "implementation_sha256": implementation_sha256,
        "input_manifest_digest": stable_json_digest(datasets),
        "batch_identities": batch_ids,
        "resolved_config_digest": stable_json_digest({
            "config": config, "cli": cli,
        }),
        "model_identity": _checkpoint_model_identity(config),
    }


def _validate_checkpoint_identity(stored: object,
                                  expected: dict[str, object] | None) -> None:
    if expected is None:
        return
    if stored != expected:
        raise ValueError(
            "checkpoint identity changed (input/config/model); "
            "recovery required, rerun with --no-resume"
        )


def _load_checkpoint(ckpt_path: Path,
                     expected_identity: dict[str, object] | None = None) -> set[str]:
    """Return completed datasets, failing closed on a stale/corrupt checkpoint."""
    if not ckpt_path.exists():
        return set()
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding='utf-8'))
        if not isinstance(ckpt, dict):
            raise ValueError("root must be an object")
        if ckpt.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                "legacy checkpoint requires recovery (run once with --no-resume)"
            )
        identity = ckpt.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("checkpoint identity is missing")
        _validate_checkpoint_identity(identity, expected_identity)
        completed = ckpt.get("completed", [])
        failed = ckpt.get("failed", [])
        if (not isinstance(completed, list) or not all(
                isinstance(name, str) and name for name in completed)
                or not isinstance(failed, list) or not all(
                    isinstance(name, str) and name for name in failed)):
            raise ValueError("completed/failed must be non-empty string lists")
        overlap = set(completed) & set(failed)
        if overlap:
            raise ValueError(f"completed/failed overlap: {sorted(overlap)[:3]}")
        return set(completed)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"checkpoint is corrupt or stale: {ckpt_path}: {exc}") from exc


def _save_checkpoint(ckpt_path: Path, completed: set[str], failed: set[str],
                     identity: dict[str, object]) -> None:
    """Atomically write checkpoint (write-then-rename)."""
    import datetime as _dt
    ckpt = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "completed": sorted(completed),
        "failed": sorted(failed),
    }
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    tmp.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(ckpt_path)


def _load_batch_progress(ckpt_path: Path) -> dict[str, dict]:
    """Return per-dataset batch progress, failing closed on stale state."""
    progress_path = ckpt_path.with_name(ckpt_path.stem + ".batch_progress.json")
    if not progress_path.exists():
        return {}
    try:
        payload = json.loads(progress_path.read_text(encoding='utf-8'))
        if (not isinstance(payload, dict)
                or payload.get("schema") != BATCH_PROGRESS_SCHEMA
                or not isinstance(payload.get("identity"), dict)
                or not isinstance(payload.get("datasets"), dict)):
            raise ValueError("legacy or malformed batch progress")
        return payload["datasets"]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"batch progress is corrupt or legacy: {progress_path}: {exc}; "
            "recovery required, rerun with --no-resume"
        ) from exc


def _save_batch_progress(ckpt_path: Path, ds_name: str,
                         done: int, fail: int, total: int,
                         identity: dict[str, object], *,
                         reset_on_identity_mismatch: bool = False) -> None:
    """Atomically write identity-bound per-dataset batch progress."""
    progress_path = ckpt_path.with_name(ckpt_path.stem + ".batch_progress.json")
    data = {"schema": BATCH_PROGRESS_SCHEMA,
            "identity": identity, "datasets": {}}
    if progress_path.exists():
        try:
            existing = json.loads(progress_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if not reset_on_identity_mismatch:
                raise RuntimeError(
                    "batch progress is corrupt; recovery required") from exc
            existing = None
        if (not isinstance(existing, dict)
                or existing.get("schema") != BATCH_PROGRESS_SCHEMA
                or existing.get("identity") != identity
                or not isinstance(existing.get("datasets"), dict)):
            if not reset_on_identity_mismatch:
                raise RuntimeError("batch progress identity changed; recovery required")
        else:
            data = existing
    data["datasets"][ds_name] = {"done": done, "fail": fail, "total": total}
    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tmp.replace(progress_path)


def _limit_stem_partitions(complete_stems: list[str],
                           incomplete_stems: list[str],
                           limit: int) -> tuple[list[str], list[str]]:
    """Apply a per-dataset limit before staged batch construction.

    Staged mode keeps CTC-ready stems ahead of fallback stems, matching its
    existing enqueue order.  Previously this path ignored ``--limit`` and a
    bounded validation could launch the entire frozen inventory.
    """
    limit = max(0, int(limit or 0))
    if not limit:
        return list(complete_stems), list(incomplete_stems)
    limited_complete = list(complete_stems[:limit])
    remaining = max(0, limit - len(limited_complete))
    limited_incomplete = list(incomplete_stems[:remaining])
    return limited_complete, limited_incomplete


def _execute_staged(
    args, all_batches: list, cache: dict, ckpt_path: Path,
    completed_set: set, failed_set: set,
    usable_drives: list, mfa_python: Path, models_dir: Path,
    parallel: int, ds_batch_tracker: dict,
    batch_size: int,
    mfa_num_jobs: int = 0,
    mfa_en_num_jobs: int = 0,
    allow_overwrite: bool = True,
    allow_force: bool = True,
    config_sha256: str = "", cache_sha256: str = "",
    implementation_sha256: str = "", require_zero_filtered: bool = False,
) -> tuple[int, list[str]]:
    """Three-phase staged execution: Stage All → Process All → Upload All.

    Phase 1: Copy ALL data NAS→NVMe (parallel, CIFS-optimized).
    Phase 2: Process ALL batches NVMe→NVMe (zero CIFS I/O).
    Phase 3: Upload ALL results NVMe→NAS (sequential per-dataset).

    Returns (ok_count, fail_list) for dataset-level accounting.
    """
    import concurrent.futures
    import queue as _queue

    total_batches = len(all_batches)
    checkpoint_identity = _checkpoint_identity(cache, args)
    trusted_batches: set[int] = set()
    nas_output_root = resolve_input_path(
        cache.get("output_root", "").rstrip("/"), PROJECT_ROOT)
    ckpt_lock = threading.Lock()

    # ═══════════════════════════════════════════════════════════
    # Phase 1 — STAGE ALL: NAS → NVMe
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 1: STAGE ALL — {total_batches} batches NAS → NVMe")
    print(f"{'='*60}")

    # Build staging task list with pre-assigned drives
    stage_tasks: list[tuple] = []  # (global_idx, batch_tuple, drive, local_base)
    staged_dirs: dict[int, Path] = {}  # global_idx → local_dir
    for gidx, item in enumerate(all_batches):
        drive = usable_drives[gidx % len(usable_drives)]
        local_base = drive / f"staged_worker_{gidx % parallel}"
        stage_tasks.append((gidx, item, local_base))
    # Shuffle so large datasets don't all land on same drive
    # (already interleaved by round-robin above)

    stage_failures: set[int] = set()
    stage_lock = threading.Lock()
    stage_completed = 0

    def stage_worker(wid: int) -> None:
        nonlocal stage_completed
        while True:
            with stage_lock:
                if not stage_tasks:
                    break
                gidx, (batch_mode, ds, batch_idx, batch_stems,
                       layout_map, wav_index, text_index), local_base = stage_tasks.pop(0)
            ds_name = ds["name"]
            try:
                dataset_root = nas_output_root / ds_name
                if _load_trusted_batch_evidence(
                        dataset_root, batch_idx, batch_stems,
                        config_sha256=config_sha256,
                        cache_sha256=cache_sha256,
                        implementation_sha256=implementation_sha256,
                        require_zero_filtered=require_zero_filtered) is not None:
                    trusted_batches.add(gidx)
                    print(f"  [STAGE] trusted evidence {ds_name}/{batch_idx:04d}; skip processing")
                    with stage_lock:
                        stage_completed += 1
                    continue
                local_dir, elapsed, missing = _stage_one_batch(
                    ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                    layout_map=layout_map, wav_index=wav_index,
                    local_base=local_base, mode=batch_mode, text_index=text_index,
                )
                if missing:
                    print(f"  [STAGE] FAIL {ds_name}/{batch_idx:04d}: "
                          f"{missing} missing audio files")
                    stage_failures.add(gidx)
                else:
                    staged_dirs[gidx] = local_dir
            except Exception as exc:
                print(f"  [STAGE] FAIL {ds_name}/{batch_idx:04d}: {exc}")
                import traceback as _tb
                _tb.print_exc()
                stage_failures.add(gidx)
            with stage_lock:
                stage_completed += 1
                if stage_completed % max(1, total_batches // 20) == 0:
                    print(f"  [STAGE] {stage_completed}/{total_batches} batches "
                          f"({stage_completed * 100 // total_batches}%)")

    n_stage_workers = min(parallel, 8, total_batches)
    t_stage_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_stage_workers) as pool:
        futures = [pool.submit(stage_worker, wid) for wid in range(n_stage_workers)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # propagate exceptions (though we catch internally)
    stage_elapsed = time.time() - t_stage_start

    n_staged = total_batches - len(stage_failures)
    print(f"\n  PHASE 1 DONE: {n_staged}/{total_batches} batches staged "
          f"({stage_elapsed:.0f}s)")
    if stage_failures:
        print(f"  WARNING: {len(stage_failures)} batches failed staging — "
              f"they will be skipped in Phase 2")

    # ═══════════════════════════════════════════════════════════
    # Phase 2 — PROCESS ALL: NVMe → NVMe (ZERO CIFS I/O)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 2: PROCESS ALL — {n_staged} batches on NVMe")
    print(f"  (Zero CIFS I/O — all reads/writes on local SSD)")
    print(f"{'='*60}")

    # Build process queue: only successfully-staged batches
    # Format: (global_idx, batch_mode, ds, batch_idx, batch_stems,
    #          layout_map, wav_index, text_index)
    proc_items: list[tuple] = []
    for gidx, (batch_mode, ds, batch_idx, batch_stems,
               layout_map, wav_index, text_index) in enumerate(all_batches):
        if gidx in trusted_batches:
            ds_batch_tracker[ds["name"]]["done"] += 1
            continue
        if gidx in stage_failures:
            ds_name = ds["name"]
            ds_batch_tracker[ds_name]["fail"] += 1
            continue
        proc_items.append((gidx, batch_mode, ds, batch_idx, batch_stems,
                           layout_map, wav_index, text_index))

    n_to_process = len(proc_items)
    proc_queue: _queue.Queue = _queue.Queue()
    for item in proc_items:
        proc_queue.put(item)

    def process_worker(wid: int) -> tuple[int, list[str]]:
        w_ok = 0
        w_fails: list[str] = []
        gpu_id = wid % args.gpus
        device_str = f"cuda:{gpu_id}"

        while True:
            try:
                (gidx, batch_mode, ds, batch_idx, batch_stems,
                 layout_map, wav_index, text_index) = proc_queue.get_nowait()
            except _queue.Empty:
                break

            # Derive local_base from the BATCH's gidx (not worker wid) so
            # staged files are found at the same path used during Phase 1.
            drive = usable_drives[gidx % len(usable_drives)]
            local_base = drive / f"staged_worker_{gidx % parallel}"

            ds_name = ds["name"]
            remaining = proc_queue.qsize()
            mode_tag = f" [{batch_mode}]" if batch_mode != "ctc_ready" else ""
            print(f"\n  [W{wid}:{device_str}] [{gidx+1}/{total_batches}]"
                  f" {ds_name}/{batch_idx:04d} ({len(batch_stems)} stems){mode_tag}"
                  f" [{remaining} left]")

            try:
                ok = _process_one_batch(
                    ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                    local_base=local_base, config=args.config,
                    mfa_python=mfa_python, models_dir=models_dir,
                    nas_output_root=nas_output_root,
                    batch_size=batch_size, python_path=args.python,
                    mode=batch_mode, device=device_str,
                    restore_cache=False,           # NO NAS I/O
                    persist_cache_on_failure=False, # upload in Phase 3
                    mfa_num_jobs=mfa_num_jobs,
                    mfa_en_num_jobs=mfa_en_num_jobs,
                    allow_overwrite=allow_overwrite,
                    allow_force=allow_force,
                )
            except Exception as _exc:
                print(f"  [W{wid}] CRASH {ds_name}/{batch_idx:04d}: {_exc}")
                import traceback as _tb
                _tb.print_exc()
                ok = False

            with ckpt_lock:
                tracker = ds_batch_tracker[ds_name]
                if ok:
                    tracker["done"] += 1
                else:
                    tracker["fail"] += 1
                _save_batch_progress(ckpt_path, ds_name,
                                     tracker["done"], tracker["fail"], tracker["total"],
                                     checkpoint_identity,
                                     reset_on_identity_mismatch=bool(
                                         getattr(args, "no_resume", False)))
                if tracker["done"] + tracker["fail"] >= tracker["total"]:
                    if tracker["fail"] == 0:
                        w_ok += 1
                    else:
                        w_fails.append(ds_name)
                    status = "DONE" if tracker["fail"] == 0 else "FAIL"
                    print(f"  [W{wid}] {ds_name} — {status} "
                          f"({tracker['done']}/{tracker['total']} batches)")

        return w_ok, w_fails

    t_proc_start = time.time()
    ok_count = 0
    fail_list: list[str] = []
    n_proc_workers = min(parallel, n_to_process)
    if n_proc_workers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_proc_workers) as pool:
            futures = [pool.submit(process_worker, wid) for wid in range(n_proc_workers)]
            for fut in concurrent.futures.as_completed(futures):
                w_ok, w_fails = fut.result()
                ok_count += w_ok
                fail_list.extend(w_fails)
    proc_elapsed = time.time() - t_proc_start

    # Record datasets that passed Phase 2 processing (NOT yet published).
    # Checkpoint is deferred until Phase 3 upload+merge succeeds.
    _phase2_ok: set[str] = set()
    _phase2_fail: set[str] = set()
    for ds_name in list(ds_batch_tracker.keys()):
        t = ds_batch_tracker[ds_name]
        if t["done"] + t["fail"] >= t["total"]:
            if t["fail"] == 0:
                _phase2_ok.add(ds_name)
            else:
                _phase2_fail.add(ds_name)
                failed_set.add(ds_name)
    # Do not write the dataset checkpoint here: batch completion is not a
    # publication result.  Only the final aggregate writes it below.
    if _phase2_fail:
        print(f"  PHASE 2: {len(_phase2_fail)} datasets FAILED processing,"
              f" {len(_phase2_ok)} datasets OK (pending upload)")
    print(f"\n  PHASE 2 DONE: {ok_count} datasets OK, "
          f"{n_to_process} batches processed ({proc_elapsed:.0f}s)")

    # ═══════════════════════════════════════════════════════════
    # Phase 3 — UPLOAD ALL: NVMe → NAS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PHASE 3: UPLOAD ALL — results NVMe → NAS")
    print(f"{'='*60}")

    # Group successfully-processed batches by dataset.
    # Failed batches were already cleaned up by _process_one_batch,
    # so they are naturally skipped by local_dir.exists() below.
    ds_upload_batches: dict[str, list[tuple[int, int, list[str], Path]]] = {}
    for gidx, (batch_mode, ds, batch_idx, batch_stems,
               layout_map, wav_index, text_index) in enumerate(all_batches):
        if gidx in stage_failures or gidx in trusted_batches:
            continue
        ds_name = ds["name"]
        local_dir = staged_dirs.get(gidx)
        if local_dir and local_dir.exists():
            ds_upload_batches.setdefault(ds_name, []).append(
                (gidx, batch_idx, batch_stems, local_dir))

    upload_failures: list[str] = []
    upload_lock = threading.Lock()
    failed_publish_dirs: set[Path] = set()
    upload_total = sum(len(v) for v in ds_upload_batches.values())
    upload_done = 0

    def upload_dataset(ds_name: str, batches: list[tuple[int, Path]]) -> bool:
        nonlocal upload_done
        ok = True
        for gidx, batch_idx, batch_stems, local_dir in batches:
            published = False
            try:
                published = _publish_batch_to_staging(
                    local_dir, nas_output_root / ds_name, batch_idx, batch_stems,
                    config_sha256=config_sha256, cache_sha256=cache_sha256,
                    implementation_sha256=implementation_sha256,
                    require_zero_filtered=require_zero_filtered)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"  [UPLOAD] {ds_name}/{batch_idx:04d} publication exception: {exc}")
            if not published:
                ok = False
                try:
                    _preserve_failed_batch(local_dir)
                except OSError:
                    # Keep the original path on the protected list if the
                    # forensic rename itself fails; the cleanup pass below
                    # must never delete a failed publication workspace.
                    with upload_lock:
                        failed_publish_dirs.add(local_dir)
            else:
                _cleanup_one_batch_dir(local_dir)
            with upload_lock:
                upload_done += 1
        return ok

    t_upload_start = time.time()
    n_upload_workers = min(parallel, 4, len(ds_upload_batches))
    if ds_upload_batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_upload_workers)) as pool:
            upload_futures = {
                pool.submit(upload_dataset, ds_name, batches): ds_name
                for ds_name, batches in ds_upload_batches.items()
            }
            for fut in concurrent.futures.as_completed(upload_futures):
                ds_name = upload_futures[fut]
                try:
                    if fut.result():
                        print(f"  [UPLOAD] {ds_name} — DONE")
                    else:
                        print(f"  [UPLOAD] {ds_name} — FAILED")
                        upload_failures.append(ds_name)
                except Exception as e:
                    print(f"  [UPLOAD] {ds_name} — CRASH: {e}")
                    upload_failures.append(ds_name)
    else:
        print(f"  No batches to upload.")

    upload_elapsed = time.time() - t_upload_start

    # ── Cleanup any remaining staged dirs ──
    for local_dir in staged_dirs.values():
        if local_dir in failed_publish_dirs:
            continue
        if local_dir.exists():
            _cleanup_one_batch_dir(local_dir)

    # Only the dataset-level aggregate and COMPLETE receipt may advance the
    # checkpoint.  Batch progress above is informational and never a resume
    # authority.
    plans: dict[str, list[tuple[int, list[str]]] ] = {}
    for _mode, ds, batch_idx, batch_stems, *_rest in all_batches:
        plans.setdefault(ds["name"], []).append((batch_idx, batch_stems))
    publish_failures = _finalize_dataset_publications(
        plans, ds_batch_tracker, nas_output_root, completed_set, failed_set,
                config_sha256=config_sha256, cache_sha256=cache_sha256,
                implementation_sha256=implementation_sha256,
                receipt_mode="staged", receipt_route=["stage", "process", "publish"],
                require_zero_filtered=require_zero_filtered)
    for name in upload_failures:
        if name not in publish_failures:
            failed_set.add(name)
            publish_failures.append(name)
    _save_checkpoint(ckpt_path, completed_set, failed_set, checkpoint_identity)

    total_elapsed = time.time() - t_stage_start
    print(f"\n  PHASE 3 DONE: {upload_done - len(upload_failures)}/{upload_total} "
          f"batches uploaded ({upload_elapsed:.0f}s)")
    print(f"  STAGED TOTAL: stage={stage_elapsed:.0f}s + "
          f"process={proc_elapsed:.0f}s + upload={upload_elapsed:.0f}s "
          f"= {total_elapsed:.0f}s")

    # Recalculate from datasets with verified COMPLETE receipts, not from
    # Phase 2 or per-batch upload status.
    ok_count = sum(1 for dataset_name in plans if dataset_name in completed_set)
    fail_list = list(set(failed_set) | set(publish_failures))

    return ok_count, fail_list


def _apply_batch_output_override(cache: dict, args) -> dict:
    """Apply an explicit batch output root without mutating the cache file.

    Batch caches freeze the source inventory, but a test/rerun must be able to
    publish into an isolated destination.  Keep the source and stem contract
    unchanged while rebinding each dataset's derived CTC/output namespace in
    memory.
    """
    config = getattr(args, "_config", {}) or {}
    raw_output = (getattr(args, "output_dir", None)
                  or getattr(args, "nas_output", None)
                  or config.get("output_dir"))
    if not raw_output:
        return cache
    import copy
    output_root = resolve_input_path(str(raw_output).rstrip("/"), PROJECT_ROOT)
    overridden = copy.deepcopy(cache)
    overridden["output_root"] = str(output_root)
    for dataset in overridden.get("datasets", []):
        name = str(dataset.get("name", ""))
        dataset["ctc_dir"] = str(output_root / name)
    return overridden


def run_batch(args) -> bool:
    """Iterate over all datasets from batch cache with checkpoint/resume support.

    Returns True when all datasets complete successfully, False otherwise.
    """
    import concurrent.futures

    cache_path = args.batch_cache
    if not cache_path.exists():
        print(f"ERROR: Batch cache not found: {cache_path}")
        print(f"  Run first: python scripts/run_pipeline.py --config configs/batch_all.yaml --scan-only")
        sys.exit(1)

    # Checkpoint file lives next to the cache file
    ckpt_path = cache_path.with_name(cache_path.stem + ".checkpoint.json")

    cached_override = getattr(args, "_batch_cache_data", None)
    if cached_override is not None:
        cache = cached_override
    else:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = _apply_batch_output_override(json.load(f), args)
        args._batch_cache_data = cache

    implementation_sha256 = _implementation_sha256()
    args._implementation_sha256 = implementation_sha256

    all_datasets = cache.get("datasets", [])
    if not all_datasets:
        print("ERROR: No datasets in cache!")
        sys.exit(1)

    checkpoint_identity = _checkpoint_identity(cache, args)

    # ── Resume: skip already-completed datasets ──
    completed_set: set[str] = set()
    failed_set: set[str] = set()
    if not getattr(args, 'no_resume', False):
        try:
            completed_set = _load_checkpoint(ckpt_path, checkpoint_identity)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return False
        if completed_set:
            # A dataset checkpoint is written only after final publication;
            # individual batch resume below remains evidence-gated.
            pending = [d for d in all_datasets if d["name"] not in completed_set]
            skipped = len(all_datasets) - len(pending)
            print(f"\n  Resume: {skipped} already completed, {len(pending)} remaining")
            all_datasets = pending
    if not all_datasets:
        print("All datasets already completed!")
        return True

    # Freeze and validate the source inventory before resolving output paths,
    # creating worker directories, or launching any child pipeline.
    try:
        _validate_frozen_cache_inventory(cache)
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: frozen cache inventory validation failed: {exc}")
        return False
    config_sha256 = _sha256_file(args.config)
    cache_sha256 = _sha256_file(cache_path)

    datasets = all_datasets[:args.limit_datasets] if args.limit_datasets > 0 else all_datasets

    # Resolve parallelism: CLI > config > default 1
    parallel = args.parallel_datasets
    _cfg = getattr(args, '_config', {})
    pipeline_cfg = _cfg.get("pipelined", {}) if _cfg else {}
    require_zero_filtered = bool(
        pipeline_cfg.get("require_zero_filtered", False))
    if parallel is None:
        parallel = _cfg.get("streaming", {}).get("parallel", 1) if _cfg else 1
    parallel = max(1, parallel)  # Keep user-specified value for batch-level scheduling

    # ── Resolve local work drives ──
    _drives = getattr(args, '_local_work_drives', (args.local_work,))
    # Validate at least one drive is usable
    usable_drives = []
    for d in _drives:
        d.parent.mkdir(parents=True, exist_ok=True) if not d.exists() else None
        d.mkdir(parents=True, exist_ok=True)
        usable_drives.append(d)
    if not usable_drives:
        print("ERROR: No usable local work drives!")
        sys.exit(1)

    # Plan once for ordinary and pipelined launch paths.  Each child pipeline
    # receives a job count that cannot collectively exceed the CPU budget.
    import os as _os
    cpu_count = _os.cpu_count() or 32
    config_mfa_jobs = _cfg.get("mfa", {}).get("num_jobs", 0) if _cfg else 0
    config_mfa_en_jobs = _cfg.get("mfa_en", {}).get("num_jobs", 0) if _cfg else 0
    _resource_plan = plan_streaming_resources(
        cpu_budget=cpu_count,
        requested_gpu_workers=args.gpus,
        requested_cpu_workers=(
            args.cpu_workers if args.pipelined else parallel),
        requested_mfa_jobs=args.mfa_jobs,
        requested_mfa_en_jobs=args.mfa_en_jobs,
        config_mfa_jobs=config_mfa_jobs,
        config_mfa_en_jobs=config_mfa_en_jobs,
        batch_size=args.batch_size,
        pipelined=args.pipelined,
    )
    if not args.pipelined:
        parallel = _resource_plan["cpu_workers"]
    _effective_mfa_jobs = _resource_plan["mfa_jobs_per_worker"]
    print(f"  MFA num_jobs/worker: {_effective_mfa_jobs}"
          f" (config={config_mfa_jobs}, parallel={parallel},"
          f" batch={args.batch_size}, CPUs={cpu_count})")
    # Memory estimate: each MFA Kaldi worker ≈ 500-1500 MB (model + features + lattice)
    _total_mfa_procs = parallel * _effective_mfa_jobs
    _est_gb_low = _total_mfa_procs * 0.5
    _est_gb_high = _total_mfa_procs * 1.5
    print(f"  Est. MFA memory: {_est_gb_low:.0f}-{_est_gb_high:.0f} GB"
          f" ({_total_mfa_procs} total: {parallel} workers x {_effective_mfa_jobs} jobs)")
    try:
        import psutil as _psutil
        _avail_gb = _psutil.virtual_memory().available / (1024**3)
        if _est_gb_high > _avail_gb * 0.7:
            _rec_jobs = max(1, int(_avail_gb * 0.7 / (1.5 * parallel)))
            print(f"  ⚠  WARNING: est {_est_gb_high:.0f} GB > 70% of {_avail_gb:.0f} GB RAM!")
            print(f"     Use --mfa-jobs {_rec_jobs} for safe memory (~{_rec_jobs * parallel * 1.5:.0f} GB)")
    except ImportError:
        pass
    _effective_mfa_en_jobs = _resource_plan["mfa_en_jobs_per_worker"]
    print(f"  MFA EN num_jobs/worker: {_effective_mfa_en_jobs}"
          f" (config={config_mfa_en_jobs})")
    # Update config for child processes
    if _cfg:
        _cfg.setdefault("mfa", {})["num_jobs"] = _effective_mfa_jobs
        _cfg.setdefault("mfa_en", {})["num_jobs"] = _effective_mfa_en_jobs
    args._effective_mfa_jobs = _effective_mfa_jobs
    args._effective_mfa_en_jobs = _effective_mfa_en_jobs

    # ── Resolve overwrite/force policy ──
    # CLI --no-overwrite/--no-force override config; default True for backward compat.
    args._allow_overwrite = (
        not getattr(args, 'no_overwrite', False)
        and _cfg.get("pipeline", {}).get("allow_overwrite", True) if _cfg else True
    )
    args._allow_force = (
        not getattr(args, 'no_force', False)
        and _cfg.get("pipeline", {}).get("allow_force", True) if _cfg else True
    )
    if not args._allow_overwrite:
        print("  --overwrite DISABLED (config or --no-overwrite)")
    if not args._allow_force:
        print("  --force DISABLED (config or --no-force)")

    print(f"\n{'#'*60}")
    print(f"  BATCH MODE: {len(datasets)} datasets from {cache_path}")
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  Parallel:    {parallel} concurrent workers")
    print(f"  Local work:  {len(usable_drives)} drive(s): {', '.join(str(d) for d in usable_drives)}")
    print(f"  MFA jobs/ds: {_effective_mfa_jobs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"{'#'*60}")

    # ── Ensure MFA acoustic model is pre-extracted (all modes) ──
    # Must happen before any worker/sequential run to avoid:
    #   - parallel race on shutil.unpack_archive → corrupt extraction
    #   - MFA subprocess extracting into double-nested path → FileNotFoundError
    _ensure_mfa_model_extracted()

    # ── Resolve MFA Python and models dir (needed by run_single_batch) ──
    if args.python:
        mfa_python = Path(args.python)
    else:
        mfa_python = find_mfa_python()
    if not mfa_python or not mfa_python.exists():
        print("ERROR: Cannot find MFA Python. Use --python PATH.")
        sys.exit(1)
    models_dir = PROJECT_ROOT / "models" / "mfa"
    print(f"MFA Python: {mfa_python}")

    if args.pipelined:
        # ``run_batch`` already proved this exact in-memory cache before any
        # output/work directory or GPU launch.  Pass the object identity to
        # the pipelined implementation so it can consume that proof without
        # repeating a full NAS metadata/hash scan.  Direct callers still take
        # the validation path below.
        return run_pipelined_batch(
            args, validated_frozen_cache=cache)

    # ── Batch-level parallel mode ──
    # Pre-discover stems for ALL datasets, split into batches, put ALL
    # individual batches into a shared queue.  Every worker processes
    # whatever batch is available — including batches from the same
    # dataset.  Keep this classifier/scheduler even with one worker: the old
    # dataset shortcut erased mixed/fallback producer identities and treated
    # WAV-only input as ctc_ready.
    import queue as _queue

    # Phase 1: pre-scan all datasets → build batch task list
    print(f"\n  Pre-scanning {len(datasets)} datasets ...")

    # Load batch-level progress for resume (which batches within each dataset are done)
    if not getattr(args, 'no_resume', False):
        try:
            _batch_progress = _load_batch_progress(ckpt_path)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return False
        progress_path = ckpt_path.with_name(ckpt_path.stem + ".batch_progress.json")
        if progress_path.exists():
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            if payload.get("identity") != checkpoint_identity:
                print("ERROR: batch progress identity changed; recovery required")
                return False
    else:
        _batch_progress = {}
    if _batch_progress:
        _total_skipped = sum(p.get("done", 0) for p in _batch_progress.values())
        print(f"  Batch progress: {_total_skipped} batches already done across"
              f" {len(_batch_progress)} datasets")

    # Load scan cache to avoid re-scanning on restart (expensive SMB find_wav calls)
    scan_cache_path = cache_path.with_name(cache_path.stem + ".scan.json")
    scan_cache: dict[str, dict] = {}
    if not getattr(args, 'no_resume', False) and scan_cache_path.exists():
        try:
            scan_cache = json.loads(scan_cache_path.read_text(encoding='utf-8'))
            hits = sum(1 for ds in datasets if ds["name"] in scan_cache)
            print(f"  Scan cache: {hits}/{len(datasets)} datasets cached")
        except Exception:
            scan_cache = {}
    scan_updated = False

    all_batches: list[tuple] = []  # (mode, ds, batch_idx, batch_stems, layout_map, wav_index, text_index)
    total_stems = 0
    total_incomplete = 0
    for ds_idx, ds in enumerate(datasets):
        ds_name = ds["name"]
        nas_ctc = resolve_input_path(ds.get("ctc_dir", ""))
        nas_audio = resolve_input_path(ds.get("audio_dir", ""))
        if not nas_audio.exists():
            print(f"  SKIP {ds_name}: audio dir not found")
            continue
        # CTC dir may not exist for nvrasr_fallback (raw audio, no prior NVASR run)
        if not nas_ctc.exists():
            nas_ctc.mkdir(parents=True, exist_ok=True)

        complete_stems: list[str] = []
        incomplete_stems: list[str] = []
        layout_map: dict[str, str] = {}
        wav_index: dict[str, Path] = {}
        incomplete_wav_index: dict[str, Path] = {}
        text_index: dict[str, Path] = {}

        # Support per-dataset stems override (for missing-files reprocessing)
        if "stems" in ds:
            all_stems = list(ds["stems"])
            layout_map = {s: "nested" for s in all_stems}  # assume nested layout

            # Check scan cache for pre-computed wav_index (avoids 38k SMB find_wav calls)
            cached = scan_cache.get(ds_name, {})
            if cached.get("stems") == all_stems:
                wav_index = {s: Path(p) for s, p in cached["wav_paths"].items()}
                # Cache also includes incomplete info if present
                if "incomplete_stems" in cached:
                    incomplete_stems = cached["incomplete_stems"]
                    incomplete_wav_index = {s: Path(p) for s, p
                                            in cached.get("incomplete_wav_paths", {}).items()}
                complete_stems = [s for s in all_stems if s in wav_index]
                print(f"  {ds_name}: {len(complete_stems)} stems (scan cache)"
                      + (f" + {len(incomplete_stems)} fallback" if incomplete_stems else ""))
            else:
                missing_ctc = 0
                ctc_files_flat, ctc_files_nested = build_ctc_presence(nas_ctc)
                for s in all_stems:
                    w = find_wav(nas_audio, s)
                    if not w:
                        continue
                    ctc_ok = all(f"{s}{suffix}" in ctc_files_flat
                                 or (s in ctc_files_nested
                                     and f"{s}{suffix}" in ctc_files_nested[s])
                                 for suffix in CTC_SUFFIXES)
                    if ctc_ok:
                        wav_index[s] = w
                        complete_stems.append(s)
                    else:
                        incomplete_wav_index[s] = w
                        incomplete_stems.append(s)
                        missing_ctc += 1
                # Save to scan cache
                scan_cache[ds_name] = {
                    "stems": all_stems,
                    "wav_paths": {s: str(p) for s, p in wav_index.items()},
                    "incomplete_stems": incomplete_stems,
                    "incomplete_wav_paths": {s: str(p) for s, p in incomplete_wav_index.items()},
                }
                scan_updated = True
                info = f"  {ds_name}: {len(complete_stems)} stems (scanned)"
                if missing_ctc:
                    info += f", {missing_ctc} incomplete → fallback"
                print(info)
        else:
            # Check if dataset has ANY pre-existing CTC output
            _ctc_flat, _ctc_nested = build_ctc_presence(nas_ctc)
            _has_ctc = bool(_ctc_flat or _ctc_nested)
            if _has_ctc:
                complete_stems, incomplete_stems, layout_map, wav_index = \
                    discover_stems_separated(nas_ctc, nas_audio, require_all=True)
            else:
                # Raw audio: no pre-existing CTC at all → discover from WAVs only
                complete_stems = []
                incomplete_stems = []
                layout_map = {}
                wav_index = {}
                incomplete_wav_index_nested = {}
                for entry in os.scandir(str(nas_audio)):
                    if entry.is_file() and entry.name.endswith(".wav"):
                        s = entry.name[:-4]
                        wav_index[s] = Path(entry.path)
                        incomplete_stems.append(s)
                        layout_map[s] = "flat"
                    elif entry.is_dir():
                        sub_wavs = list(Path(entry.path).glob("*.wav"))
                        if sub_wavs:
                            for sw in sub_wavs:
                                s = sw.stem
                                incomplete_wav_index_nested[s] = sw
                                incomplete_stems.append(s)
                                layout_map[s] = "nested"
                if incomplete_wav_index_nested:
                    wav_index.update(incomplete_wav_index_nested)
                incomplete_stems.sort()
                print(f"  {ds_name}: {len(incomplete_stems)} stems (raw audio, all → nvrasr_fallback)")
            # Build incomplete wav_index from wav_index (same stems)
            for s in incomplete_stems:
                if s in wav_index:
                    incomplete_wav_index[s] = wav_index[s]

        original_stem_count = len(complete_stems) + len(incomplete_stems)
        complete_stems, incomplete_stems = _limit_stem_partitions(
            complete_stems, incomplete_stems, args.limit)
        limited_stem_count = len(complete_stems) + len(incomplete_stems)
        if limited_stem_count < original_stem_count:
            print(f"  {ds_name}: limited to {limited_stem_count} stems")

        batch_size_eff = args.batch_size

        # ── Enqueue ctc_ready batches (complete stems) ──
        # Count-only progress is informational.  A batch is resumable only
        # through its exact trusted evidence and publication hashes below.
        _bp = _batch_progress.get(ds_name, {})
        if _bp.get("done", 0):
            print(f"  {ds_name}: ignoring count-only resume progress; "
                  f"checking trusted batch evidence")
        _dataset_batch_base = 0
        if complete_stems:
            batches_ctc = [complete_stems[i:i + batch_size_eff]
                           for i in range(0, len(complete_stems), batch_size_eff)]
            for batch_idx, batch_stems in enumerate(batches_ctc):
                all_batches.append(
                    ("ctc_ready", ds, batch_idx, batch_stems, layout_map, wav_index, None))
            total_stems += len(complete_stems)
            _info = f"  {ds_name}: {len(complete_stems)} stems → {len(batches_ctc)} ctc_ready batches"
            print(_info)
            _dataset_batch_base = len(batches_ctc)

        # ── Enqueue nvrasr_fallback batches (incomplete stems) ──
        if incomplete_stems:
            # Build text index for NVASR reference text
            text_index = build_file_index(nas_audio, ".txt")
            if not text_index:
                print(f"  {ds_name}: WARNING: {len(incomplete_stems)} fallback stems "
                      f"have no reference .txt — NVASR will use ASR-only")
            batches_fb = [incomplete_stems[i:i + batch_size_eff]
                          for i in range(0, len(incomplete_stems), batch_size_eff)]
            for batch_offset, batch_stems in enumerate(batches_fb):
                batch_idx = _dataset_batch_base + batch_offset
                all_batches.append(
                    ("nvrasr_fallback", ds, batch_idx, batch_stems,
                     layout_map, incomplete_wav_index, text_index))
            total_incomplete += len(incomplete_stems)
            _info = f"  {ds_name}: {len(incomplete_stems)} stems → {len(batches_fb)} nvrasr_fallback batches"
            print(_info)

        if not complete_stems and not incomplete_stems:
            print(f"  SKIP {ds_name}: no valid stems")
            continue

    # Persist scan cache for faster restart
    if scan_updated:
        try:
            scan_cache_path.write_text(
                json.dumps(scan_cache, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"  Scan cache saved: {scan_cache_path}")
        except Exception as e:
            print(f"  WARNING: Could not save scan cache: {e}")

    print(f"\n  Total: {len(all_batches)} batches ({total_stems} ctc_ready + {total_incomplete} fallback stems)")

    if not all_batches:
        print("No batches to process; no work remains.")
        return True

    # Track per-dataset completion: all batches of a dataset must
    # succeed before marking the dataset as DONE in checkpoint.
    total_batches = len(all_batches)
    ds_batch_tracker: dict[str, dict] = {}  # {ds_name: {"total": N, "done": n, "fail": n}}
    for ds_item in all_batches:
        # ds_item format: (mode, ds, batch_idx, batch_stems, layout_map, wav_index, text_index)
        ds_name = ds_item[1]["name"]
        if ds_name not in ds_batch_tracker:
            ds_batch_tracker[ds_name] = {"total": 0, "done": 0, "fail": 0}
        ds_batch_tracker[ds_name]["total"] += 1

    # ── NVMe space check for staged mode ──
    use_staging = getattr(args, 'stage_all', True)
    if use_staging:
        _est_gb = total_stems * 1.5 / 1024  # rough: 1.5 MB/stem (WAV + CTC overhead)
        try:
            _avail_gb = shutil.disk_usage(usable_drives[0]).free / (1024**3)
        except Exception:
            _avail_gb = float('inf')
        if _est_gb > _avail_gb * 0.7 and not getattr(args, 'force_stage', False):
            print(f"\n  ⚠  Estimated data ({_est_gb:.0f} GB) > 70% of NVMe free"
                  f" ({_avail_gb:.0f} GB)")
            print(f"     Falling back to streaming mode.")
            print(f"     Use --force-stage to override, or free up NVMe space.")
            use_staging = False

    if use_staging:
        # ── STAGED MODE: Stage All → Process All → Upload All ──
        ok_count, fail_list = _execute_staged(
            args=args, all_batches=all_batches, cache=cache,
            ckpt_path=ckpt_path, completed_set=completed_set,
            failed_set=failed_set, usable_drives=usable_drives,
            mfa_python=mfa_python, models_dir=models_dir,
            parallel=parallel, ds_batch_tracker=ds_batch_tracker,
            batch_size=args.batch_size,
            mfa_num_jobs=getattr(args, '_effective_mfa_jobs', 0),
            mfa_en_num_jobs=getattr(args, '_effective_mfa_en_jobs', 0),
            allow_overwrite=getattr(args, '_allow_overwrite', True),
            allow_force=getattr(args, '_allow_force', True),
            config_sha256=config_sha256, cache_sha256=cache_sha256,
            implementation_sha256=implementation_sha256,
            require_zero_filtered=require_zero_filtered,
        )
    else:
        # ── STREAMING MODE: interleave prefetch/process/upload per batch ──
        if not getattr(args, 'stage_all', True):
            print(f"\n  Mode: STREAMING (per-batch prefetch/process/upload)")
        else:
            print(f"\n  Mode: STREAMING (fallback — NVMe space constrained)")

        batch_queue: _queue.Queue = _queue.Queue()
        for gidx, item in enumerate(all_batches):
            batch_queue.put((gidx, item))

        ok_count = 0
        fail_list: list[str] = []
        ckpt_lock = threading.Lock()

        def worker(worker_id: int) -> tuple[int, list[str]]:
            """Pull individual batches from shared queue."""
            w_ok = 0
            w_fails: list[str] = []
            drive = usable_drives[worker_id % len(usable_drives)]
            local_base = drive / f"worker_{worker_id}"
            gpu_id = worker_id % args.gpus
            device_str = f"cuda:{gpu_id}"
            while True:
                try:
                    batch_global_idx, (batch_mode, ds, batch_idx, batch_stems,
                                       layout_map, wav_index, text_index) = batch_queue.get_nowait()
                except _queue.Empty:
                    break

                ds_name = ds["name"]
                nas_output_root = resolve_input_path(
                    cache.get("output_root", "").rstrip("/"), PROJECT_ROOT)
                batch_label = f"{ds_name}/{batch_idx:04d}"
                remaining = batch_queue.qsize()

                mode_tag = f" [{batch_mode}]" if batch_mode != "ctc_ready" else ""
                print(f"\n  [W{worker_id}:{device_str}] [{batch_global_idx+1}/{total_batches}]"
                      f" {batch_label} ({len(batch_stems)} stems){mode_tag}"
                      f" [{remaining} left]")

                try:
                    trusted = _load_trusted_batch_evidence(
                        nas_output_root / ds_name, batch_idx, batch_stems,
                        config_sha256=config_sha256, cache_sha256=cache_sha256,
                        implementation_sha256=implementation_sha256,
                        require_zero_filtered=require_zero_filtered)
                    if trusted is not None:
                        print(f"  [W{worker_id}] trusted evidence; skipping "
                              f"{batch_label}")
                        ok = True
                    else:
                        ok = run_single_batch(
                            ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                            layout_map=layout_map, wav_index=wav_index,
                            local_base=local_base, config=args.config,
                            mfa_python=mfa_python, models_dir=models_dir,
                            nas_output_root=nas_output_root,
                            batch_size=args.batch_size, python_path=args.python,
                            mode=batch_mode, text_index=text_index,
                            device=device_str,
                            mfa_num_jobs=getattr(args, '_effective_mfa_jobs', 0),
                            mfa_en_num_jobs=getattr(args, '_effective_mfa_en_jobs', 0),
                            config_sha256=config_sha256, cache_sha256=cache_sha256,
                            implementation_sha256=implementation_sha256,
                            require_zero_filtered=require_zero_filtered,
                        )
                except Exception as _exc:
                    print(f"  [W{worker_id}] CRASH processing {batch_label}: {_exc}")
                    import traceback as _tb
                    _tb.print_exc()
                    ok = False

                with ckpt_lock:
                    tracker = ds_batch_tracker[ds_name]
                    if ok:
                        tracker["done"] += 1
                    else:
                        tracker["fail"] += 1
                    _save_batch_progress(ckpt_path, ds_name,
                                         tracker["done"], tracker["fail"], tracker["total"],
                                         checkpoint_identity,
                                         reset_on_identity_mismatch=bool(
                                             getattr(args, "no_resume", False)))
                    if tracker["done"] + tracker["fail"] >= tracker["total"]:
                        if tracker["fail"] == 0:
                            w_ok += 1
                        else:
                            w_fails.append(ds_name)
                            failed_set.add(ds_name)
                        status = "DONE" if tracker["fail"] == 0 else "FAIL"
                        print(f"  [W{worker_id}] {ds_name} — {status} "
                              f"({tracker['done']}/{tracker['total']} batches)")

            # Preserve *.FAILED and other forensic evidence.  Successful
            # batches already remove their own local workspace.
            if local_base.exists():
                for child in local_base.iterdir():
                    if ".FAILED" not in child.name and ".STALE" not in child.name:
                        _cleanup_one_batch_dir(child)
            return w_ok, w_fails

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(worker, wid) for wid in range(parallel)]
            for fut in concurrent.futures.as_completed(futures):
                w_ok, w_fails = fut.result()
                ok_count += w_ok
                fail_list.extend(w_fails)

    plans: dict[str, list[tuple[int, list[str]]]] = {}
    for _mode, ds, batch_idx, batch_stems, *_rest in all_batches:
        plans.setdefault(ds["name"], []).append((batch_idx, batch_stems))
    if not use_staging:
        # `_execute_staged` already owns dataset aggregation, COMPLETE receipt
        # verification, and the checkpoint transaction.  Finalizing it again
        # here changes the receipt mode from ``staged`` to ``streaming`` and
        # conflicts with the valid receipt produced moments earlier.
        publish_failures = _finalize_dataset_publications(
            plans, ds_batch_tracker,
            resolve_input_path(cache.get("output_root", "").rstrip("/"), PROJECT_ROOT),
            completed_set, failed_set, config_sha256=config_sha256,
            cache_sha256=cache_sha256,
            implementation_sha256=implementation_sha256,
            receipt_mode="streaming",
            receipt_route=["stage", "process", "publish"],
            require_zero_filtered=require_zero_filtered)
        _save_checkpoint(ckpt_path, completed_set, failed_set, checkpoint_identity)
        fail_list = list(set(fail_list) | set(publish_failures))

    # ── Final summary ──
    all_ok = len(fail_list) == 0 and all(
        name in completed_set for name in plans)
    print(f"\n{'#'*60}")
    print(f"  BATCH COMPLETE: {ok_count}/{len(datasets)} OK"
          f"{' — ALL OK' if all_ok else ' — WITH FAILURES'}")
    if fail_list:
        print(f"  Failed: {', '.join(fail_list)}")
    print(f"{'#'*60}")
    return all_ok


def _save_progress(cache_path: Path, cache: dict, ds_name: str, ok: bool):
    """Append *ds_name* to completed_datasets and persist cache."""
    if ok:
        cache.setdefault("completed_datasets", []).append(ds_name)
        try:
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # non-critical


def _run_batch_sequential(args, datasets: list, cache: dict,
                         ckpt_path: Path, completed_set: set[str],
                         failed_set: set[str]) -> bool:
    """Compatibility dataset loop with checkpoint after each dataset."""
    ok_count = 0
    fail_list: list[str] = []
    checkpoint_identity = _checkpoint_identity(cache, args)
    # Use first available local drive
    _drives = getattr(args, '_local_work_drives', (args.local_work,))
    _first_drive = _drives[0] if _drives else args.local_work
    for i, ds in enumerate(datasets):
        ds_name = ds["name"]
        nas_ctc = ds.get("ctc_dir", "")
        nas_audio = ds.get("audio_dir", "")
        nas_output = cache.get("output_root", "").rstrip("/") + "/" + ds_name
        ds_local = _first_drive / ds_name

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(datasets)}] {ds_name}")
        print(f"  CTC:    {nas_ctc}")
        print(f"  Audio:  {nas_audio}")
        print(f"  Output: {nas_output}")
        print(f"{'='*60}")

        stems_ov = ds.get("stems", None)
        config_mode = (getattr(args, "_config", {}) or {}).get("mode")
        producer_mode = _streaming_producer_mode(
            ds.get("mode", cache.get("mode")), config_mode)
        if producer_mode is None:
            print(f"ERROR: {ds_name}: unsupported producer mode")
            ok = False
        else:
            ok = run_single_dataset(
                nas_ctc=nas_ctc, nas_audio=nas_audio,
                nas_output=nas_output, config=args.config,
                local_work=ds_local, batch_size=args.batch_size,
                limit=args.limit, python_path=args.python,
                stems_override=stems_ov,
                prefetch_buffer=args.prefetch_buffer,
                upload_buffer=args.upload_buffer,
                staged=getattr(args, 'stage_all', True),
                parallel_batches=1,  # sequential batch mode
                mode=producer_mode,
            )

        if ok:
            ok_count += 1
            failed_set.discard(ds_name)
            completed_set.add(ds_name)
            if ds_local.exists():
                shutil.rmtree(ds_local, ignore_errors=True)
        else:
            completed_set.discard(ds_name)
            failed_set.add(ds_name)
            fail_list.append(ds_name)
        _save_checkpoint(ckpt_path, completed_set, failed_set, checkpoint_identity)

        print(f"\n  [{i+1}/{len(datasets)}] {ds_name} — "
              f"{'DONE' if ok else 'FAILED'}")

    print(f"\n{'#'*60}")
    print(f"  BATCH COMPLETE: {ok_count}/{len(datasets)} OK")
    if fail_list:
        print(f"  Failed: {', '.join(fail_list)}")
    print(f"{'#'*60}")
    return not fail_list


# ═══════════════════════════════════════════════════════════════
# Pipelined mode — GPU (NVASR) and CPU (MFA) in parallel stages
# ═══════════════════════════════════════════════════════════════

_GPU_STAGING_COPY_ATTEMPTS = 3
_GPU_STAGING_RETRY_DELAY_S = 0.05


def _copy_gpu_staging_file(
    source: Path, target: Path, stem: str, *, use_link_or_copy: bool,
) -> bool:
    """Stage one GPU input file with bounded transient-NAS retries."""
    attempts = _GPU_STAGING_COPY_ATTEMPTS
    last_error = "copy operation returned false"
    for attempt in range(1, attempts + 1):
        try:
            if use_link_or_copy:
                copied = link_or_copy_file(source, target)
            else:
                copied = shutil.copy2(str(source), str(target))
            if copied is False:
                raise OSError("copy operation returned false")
            return True
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(_GPU_STAGING_RETRY_DELAY_S * attempt)

    print(
        f"  [GPU] staging copy failed stem={stem} source={source} "
        f"target={target} attempts={attempts} error={last_error}"
    )
    return False


def _run_gpu_phase(
    ds: dict, batch_idx: int, batch_stems: list[str],
    layout_map: dict, wav_index: dict, text_index: dict[str, Path] | None,
    local_base: Path, config: Path,
    mfa_python: Path, models_dir: Path,
    batch_size: int, python_path: str | None,
    device: str,
    nas_output_dir: Path | None = None,
    persist_cache: bool = True,
    allow_overwrite: bool = True,
    allow_force: bool = True,
) -> bool:
    """GPU phase: prefetch WAVs + NVASR prealign + normalize -> CTC output.

    Leaves the local workspace intact for the CPU phase to pick up.
    Returns True on success.
    """
    import concurrent.futures as _cf

    # A resumed run can assign the same numeric batch index to a different
    # dataset.  Isolate by dataset and quarantine any leftover GPU workspace
    # before staging new audio/CTC artifacts.
    local_dir = _batch_local_dir(local_base, batch_idx, ds.get("name"))
    if local_dir.exists() or local_dir.is_symlink():
        _quarantine_existing_path(local_dir, label="STALE")
    local_audio = local_dir / "audio"
    local_ctc = local_dir / "ctc"
    local_workspace = local_dir / "workspace"

    # ── Prefetch audio files ──
    local_audio.mkdir(parents=True, exist_ok=True)
    local_ctc.mkdir(parents=True, exist_ok=True)
    copy_tasks: list[tuple[Path, Path]] = []
    regular_audio_tasks: list[tuple[Path, Path]] = []
    missing_audio: list[str] = []
    for stem in batch_stems:
        src_wav = wav_index.get(stem) or find_wav(resolve_input_path(ds.get("audio_dir", "")), stem)
        if src_wav:
            # The downstream CTC/MFA receipt validator intentionally rejects
            # symlinked WAVs.  Keep text files cheap to stage, but materialize
            # audio as regular files on the local NVMe workspace.
            regular_audio_tasks.append((src_wav, local_audio / f"{stem}.wav"))
        else:
            missing_audio.append(stem)
        # Copy .txt if available (for reference text in NVASR)
        if text_index and stem in text_index:
            copy_tasks.append((text_index[stem], local_audio / f"{stem}.txt"))

    if missing_audio:
        print(f"  [GPU] Missing audio for {len(missing_audio)}/{len(batch_stems)} stems")
        return False
    all_tasks = len(copy_tasks) + len(regular_audio_tasks)
    n_workers = min(8, max(1, all_tasks // 100))
    failed_copies = 0
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_copy_gpu_staging_file, source, target, target.stem,
                        use_link_or_copy=False)
            for source, target in regular_audio_tasks
        ]
        futures += [
            pool.submit(_copy_gpu_staging_file, source, target, target.stem,
                        use_link_or_copy=True)
            for source, target in copy_tasks
        ]
        for f in _cf.as_completed(futures):
            try:
                if not f.result():
                    failed_copies += 1
            except Exception:
                failed_copies += 1
    if failed_copies:
        print(f"  [GPU] {failed_copies}/{all_tasks} staging copies failed")
        return False

    # ── Run NVASR prealign + normalize (GPU-intensive) ──
    cmd = [
        str(mfa_python),
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(config),
        "--mode", "nvrasr_fallback",
        "--data-dir", str(local_audio),
        "--workspace", str(local_workspace),
        "--python", str(mfa_python),
        "--stop-after", "normalize_en",
    ]
    if allow_overwrite:
        cmd.append("--overwrite")
    if allow_force:
        cmd.append("--force")
    if device:
        cmd += ["--device", "cuda:0"]  # CUDA_VISIBLE_DEVICES remaps this
    env = get_mfa_env(mfa_python, models_dir)
    _mfa_root = local_workspace / "mfa_root"
    _mfa_root.mkdir(parents=True, exist_ok=True)
    env["MFA_ROOT_DIR"] = str(_mfa_root)
    if device:
        gpu_idx = device.replace("cuda:", "")
        if gpu_idx.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_token(int(gpu_idx), env)

    # Producer success is a strict batch contract.  A complete-looking
    # subset is never promoted to CPU; retry the same requested batch instead.
    requested = list(batch_stems)
    expected = sorted(requested)
    last_rc = 1
    for attempt in range(1, 4):
        if attempt > 1:
            # Do not let artifacts from a failed attempt satisfy the next
            # attempt's exact-set check.  Quarantine them for forensics.
            for producer_dir in (local_workspace / "ctc_pretg",
                                  local_workspace / "ctc_pretg_adj"):
                if producer_dir.exists() or producer_dir.is_symlink():
                    _quarantine_existing_path(producer_dir,
                                               label=f"ATTEMPT{attempt - 1}")
        try:
            last_rc = subprocess.run(
                cmd, env=env, timeout=7200, capture_output=False).returncode
        except subprocess.TimeoutExpired:
            last_rc = 1

        _flatten_ctc_shards(
            local_workspace / "ctc_pretg", set(expected), CTC_SUFFIXES)
        exact = _validate_exact_ctc_bundle(
            local_workspace, local_audio, requested)
        if not exact:
            # A receipt with exact input/output stems but missing bindings can
            # be repaired once the producer artifacts themselves are exact.
            exact = _repair_complete_ctc_receipt(
                local_workspace, local_audio, requested)
        if exact:
            if persist_cache:
                _persist_ctc_adj_cache(local_workspace,
                                       resolve_input_path(ds.get("ctc_dir", "")))
                if nas_output_dir:
                    _persist_ctc_adj_cache(local_workspace, nas_output_dir)
            return True
        print(f"  [GPU] Attempt {attempt}/3 rejected: rc={last_rc}; "
              f"producer/receipt is not exact for {len(expected)} requested stems")

    # Preserve failed batch directory for forensic analysis, including all
    # quarantined attempt directories and the final producer state.
    _failed_dir = _preserve_failed_batch(local_dir)
    print(f"  [GPU] Preserved: {_failed_dir}")
    return False


def _ctc_ready_overwrite_args(*, allow_overwrite: bool,
                              sealed_ctc_raw: bool) -> list[str]:
    """Build ctc_ready overwrite arguments without reopening sealed raw CTC."""
    if allow_overwrite and not sealed_ctc_raw:
        return ["--overwrite"]
    return []


def _build_cpu_phase_command(
        *, mfa_python: Path, config: Path, local_audio: Path,
        local_ctc: Path, local_output: Path, local_workspace: Path,
        producer_mode: str, allow_overwrite: bool, allow_force: bool,
        sealed_ctc_raw: bool, skip_pad_silence: bool,
        mfa_num_jobs: int = 0, mfa_en_num_jobs: int = 0) -> list[str]:
    """Build the CPU command for the explicit GPU/CTC producer route.

    A fallback GPU producer already owns the raw CTC workspace and its mode
    fingerprint.  Resume at ``resample`` so the consumer binds that evidence
    directly; invoking ``ctc_ready`` would incorrectly re-enter ``link`` and
    recompute the effective mode.  A genuine ctc_ready batch retains the
    import route and its sealed-raw overwrite gate.
    """
    if producer_mode not in {"nvrasr_fallback", "ctc_ready"}:
        raise ValueError(f"unsupported CPU producer mode: {producer_mode}")
    cmd = [
        str(mfa_python),
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(config),
        "--mode", producer_mode,
        "--data-dir", str(local_audio),
        "--output-dir", str(local_output),
        "--workspace", str(local_workspace),
        "--python", str(mfa_python),
    ]
    if producer_mode == "nvrasr_fallback":
        cmd += ["--skip-to", "resample"]
        if allow_overwrite:
            # No link runs in this route; preserve overwrite semantics for
            # downstream derived MFA/postprocess output and reruns.
            cmd.append("--overwrite")
    else:
        cmd += ["--ctc-ready", str(local_ctc)]
        cmd.extend(_ctc_ready_overwrite_args(
            allow_overwrite=allow_overwrite,
            sealed_ctc_raw=sealed_ctc_raw,
        ))
    if allow_force:
        cmd.append("--force")
    if skip_pad_silence:
        cmd.append("--skip-pad_silence")
    if mfa_num_jobs > 0:
        cmd += ["--mfa-jobs", str(mfa_num_jobs)]
    if mfa_en_num_jobs > 0:
        cmd += ["--mfa-en-jobs", str(mfa_en_num_jobs)]
    return cmd


def _run_cpu_phase(
    ds: dict, batch_idx: int, batch_stems: list[str],
    local_base: Path, config: Path,
    mfa_python: Path, models_dir: Path,
    nas_output: Path,
    batch_size: int, python_path: str | None,
    mfa_num_jobs: int = 0,
    mfa_en_num_jobs: int = 0,
    allow_overwrite: bool = True,
    allow_force: bool = True,
    restore_cache: bool = True,
    config_sha256: str = "",
    cache_sha256: str = "",
    implementation_sha256: str = "",
    upload_lock=None,
    producer_mode: str = "ctc_ready",
    preserve_successful_workspaces: bool = False,
    require_zero_filtered: bool = False,
) -> bool:
    """CPU phase: read CTC from local workspace + run MFA align + postprocess.

    The local workspace must already contain CTC output from the GPU phase.
    Uploads final output to NAS and cleans up the local directory unless the
    pipelined retention setting explicitly preserves a successful workspace.
    """
    local_dir = _batch_local_dir(local_base, batch_idx, ds.get("name"))
    local_audio = local_dir / "audio"        # where GPU phase put WAVs
    local_ctc = local_dir / "ctc"
    local_workspace = local_dir / "workspace"
    local_output = local_dir / "output"

    if producer_mode not in {"nvrasr_fallback", "ctc_ready"}:
        raise ValueError(f"unsupported CPU producer mode: {producer_mode}")

    # True ctc_ready batches import CTC into a local ctc namespace.  A
    # fallback GPU producer already owns workspace/ctc_pretg; copying it into
    # ctc_ready would re-enter link and lose the producer mode identity.
    _ctc_sources = [
        local_workspace / "ctc_pretg_adj",
        local_workspace / "ctc_pretg",
    ]
    _sealed_ctc_raw = False
    if producer_mode == "ctc_ready":
        manifest = {"schema": "ctc-ready-manifest-v2",
                    "stems": batch_stems, "n_stems": len(batch_stems)}
        local_ctc.mkdir(parents=True, exist_ok=True)
        (local_ctc / "ctc_ready_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False))

        _linked = 0
        for _src in _ctc_sources:
            if not _src.exists():
                continue
            for _stem in batch_stems:
                # Flat layout: ctc_pretg/{stem}.TextGrid
                # Nested layout: ctc_pretg/{stem}/{stem}.TextGrid
                for _layout_dir in (_src, _src / _stem):
                    if not _layout_dir.exists():
                        continue
                    for _suffix in CTC_SUFFIXES:
                        _f = _layout_dir / f"{_stem}{_suffix}"
                        if not _f.is_file():
                            continue
                        _tgt = local_ctc / _f.name
                        if _tgt.exists():
                            continue
                        try:
                            shutil.copyfile(str(_f), str(_tgt))
                        except OSError:
                            continue
                        _linked += 1
            if _linked:
                break  # found files, stop searching lower-priority dirs
        if _linked:
            print(f"  [CPU] Linked {_linked} CTC files → ctc/")
        else:
            # Debug: list what's actually in the source dirs
            for _src in _ctc_sources:
                _n = len(list(_src.glob("*"))) if _src.exists() else -1
                print(f"  [CPU] {_src}: {'exists' if _src.exists() else 'MISSING'}"
                      f" ({_n} items)" if _n >= 0 else "")
            print(f"  WARNING: no CTC files found for {len(batch_stems)} stems "
                  f"— CPU phase will fail")

        # ctc_ready re-use is receipt-gated.  Carry the source receipt into
        # the import directory so link validates the exact audio/stem axis.
        for _src in _ctc_sources:
            _receipt = _src / ".ctc_run_receipt.json"
            if _receipt.is_file():
                try:
                    shutil.copyfile(str(_receipt), str(local_ctc / _receipt.name))
                except OSError as _exc:
                    print(f"  WARNING: could not stage CTC axis receipt: {_exc}")
                break

        for _src in _ctc_sources:
            _raw_manifest = _src / CTC_RAW_MANIFEST_NAME
            try:
                if _raw_manifest.is_file() and not _raw_manifest.is_symlink():
                    _raw_payload = json.loads(_raw_manifest.read_text(encoding="utf-8"))
                    if sorted(_raw_payload.get("stems", [])) == sorted(batch_stems):
                        shutil.copyfile(str(_raw_manifest),
                                        str(local_ctc / CTC_RAW_MANIFEST_NAME))
                        _sealed_ctc_raw = True
                        break
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue

    # ── Restore cached adjust output if available ──
    # A no-reference GPU phase has just produced the authoritative CTC axis
    # for this batch. Restoring ctc_pretg_adj from a shared NAS output tree
    # can silently import another run's stems/receipt, so callers may disable
    # this legacy optimization for isolated full-pipeline jobs.
    if restore_cache:
        _restore_ctc_adj_cache(local_workspace, nas_output)

    # ── Run MFA alignment + postprocess (CPU-intensive) ──
    # GPU phase already did pad_silence → skip to avoid double I/O.
    # Fallback resumes at resample; true ctc_ready resumes at link.
    cmd = _build_cpu_phase_command(
        mfa_python=mfa_python, config=config, local_audio=local_audio,
        local_ctc=local_ctc, local_output=local_output,
        local_workspace=local_workspace, producer_mode=producer_mode,
        allow_overwrite=allow_overwrite, allow_force=allow_force,
        sealed_ctc_raw=_sealed_ctc_raw,
        skip_pad_silence=(local_workspace / "padded_audio").exists(),
        mfa_num_jobs=mfa_num_jobs, mfa_en_num_jobs=mfa_en_num_jobs,
    )
    env = get_mfa_env(mfa_python, models_dir)
    _mfa_root = local_workspace / "mfa_root"
    _mfa_root.mkdir(parents=True, exist_ok=True)
    env["MFA_ROOT_DIR"] = str(_mfa_root)
    try:
        rc = subprocess.run(cmd, env=env, timeout=7200, capture_output=False).returncode
    except subprocess.TimeoutExpired:
        rc = 1

    if rc != 0:
        # Preserve a shared CTC cache only when this invocation explicitly
        # opted into shared cache restoration.  Isolated no-reference runs
        # must not mutate the dataset root after a downstream CPU failure.
        if restore_cache:
            _persist_ctc_adj_cache(local_workspace, nas_output)
        # Preserve failed batch directory for forensic analysis
        _failed_dir = _preserve_failed_batch(local_dir)
        print(f"  [CPU] Preserved: {_failed_dir}")
        return False

    # ── Publish isolated batch staging + evidence ──
    try:
        if not _publish_batch_to_staging(
                local_dir, nas_output, batch_idx, batch_stems,
                config_sha256=config_sha256, cache_sha256=cache_sha256,
                implementation_sha256=implementation_sha256,
                require_zero_filtered=require_zero_filtered):
            print(f"  [CPU] Batch publication failed — preserving {local_dir}")
            _preserve_failed_batch(local_dir)
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  [CPU] Batch publication exception: {exc}")
        _preserve_failed_batch(local_dir)
        return False

    # Cleanup is allowed only after staging file set/size/hash validation.
    # Failed publication and all earlier failure paths return before this
    # opt-in branch, so their forensic-preservation behavior is unchanged.
    if preserve_successful_workspaces:
        print(f"  [CPU] Successful workspace retained: {local_dir}")
    else:
        shutil.rmtree(local_dir, ignore_errors=True)
    return True


def run_pipelined_batch(
        args, *, validated_frozen_cache: dict | None = None) -> bool:
    """Pipelined GPU/CPU mode: NVASR and MFA run in parallel stages.

    GPU workers:  prefetch WAVs → NVASR prealign + normalize → CTC output
    CPU workers:  read CTC → MFA align + postprocess → upload to NAS

    Batches flow through two queues — GPU workers consume from gpu_queue
    and produce to cpu_queue, while CPU workers consume from cpu_queue.
    This keeps all 8 GPUs busy with NVASR while all CPU cores run MFA,
    without either resource waiting for the other.
    """
    import concurrent.futures
    import queue as _queue

    cache_path = args.batch_cache
    if not cache_path.exists():
        print(f"ERROR: Batch cache not found: {cache_path}")
        sys.exit(1)

    cached_override = getattr(args, "_batch_cache_data", None)
    if cached_override is not None:
        cache = cached_override
    else:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = _apply_batch_output_override(json.load(f), args)

    implementation_sha256 = _implementation_sha256()
    args._implementation_sha256 = implementation_sha256

    # A direct pipelined call must validate for itself.  The ordinary entry
    # point may hand off only the same cache object it already validated; an
    # arbitrary boolean or stale digest is deliberately insufficient.
    if validated_frozen_cache is not cache:
        try:
            _validate_frozen_cache_inventory(cache)
        except (OSError, TypeError, ValueError) as exc:
            print(f"ERROR: frozen cache inventory validation failed: {exc}")
            return False

    config_sha256 = _sha256_file(args.config)
    cache_sha256 = _sha256_file(cache_path)

    all_datasets = cache.get("datasets", [])
    if not all_datasets:
        print("ERROR: No datasets in cache!")
        sys.exit(1)

    # ── Config ──
    _cfg = getattr(args, '_config', {})
    pipeline_cfg = _cfg.get("pipelined", {}) if _cfg else {}
    require_zero_filtered = bool(
        pipeline_cfg.get("require_zero_filtered", False))
    preserve_successful_workspaces = bool(
        pipeline_cfg.get("preserve_successful_workspaces", False))
    n_gpu_workers = args.gpus  # 1 GPU per GPU worker; bounded after scanning.
    requested_cpu_workers = args.cpu_workers or pipeline_cfg.get("cpu_workers", 0)

    # ── Resume ──
    ckpt_path = cache_path.with_name(cache_path.stem + ".checkpoint.json")
    checkpoint_identity = _checkpoint_identity(cache, args)
    if getattr(args, "no_resume", False):
        completed_set = set()
    else:
        try:
            completed_set = _load_checkpoint(ckpt_path, checkpoint_identity)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return False
    failed_set: set[str] = set()

    # ── Resolve drives ──
    _drives = getattr(args, '_local_work_drives', (args.local_work,))
    usable_drives = list(_drives)

    # ── MFA Python ──
    if args.python:
        mfa_python = Path(args.python)
    else:
        mfa_python = find_mfa_python()
    models_dir = PROJECT_ROOT / "models" / "mfa"
    _ensure_mfa_model_extracted()
    nas_output_root = resolve_input_path(
        cache.get("output_root", "").rstrip("/"), PROJECT_ROOT)

    # ── Pre-scan datasets → build all batches ──
    print(f"\n  Pre-scanning {len(all_datasets)} datasets ...")
    all_gpu_batches: list[tuple] = []  # (ds, batch_idx, batch_stems, layout_map, wav_index, text_index)
    for ds in all_datasets:
        ds_name = ds["name"]
        if ds_name in completed_set:
            continue
        nas_ctc = resolve_input_path(ds.get("ctc_dir", ""))
        nas_audio = resolve_input_path(ds.get("audio_dir", ""))
        if not nas_audio.exists():
            print(f"  SKIP {ds_name}: audio dir not found")
            continue
        if not nas_ctc.exists():
            nas_ctc.mkdir(parents=True, exist_ok=True)

        # Discover stems — honour explicit stems override (test/limit mode)
        if "stems" in ds:
            all_stems = list(ds["stems"])
            layout_map = {s: "flat" for s in all_stems}
            wav_index = _index_wavs_for_stems(nas_audio, all_stems)
            all_stems = [s for s in all_stems if s in wav_index]
            print(f"  {ds_name}: {len(all_stems)} stems (from override)")
        else:
            # ``discover_stems_separated`` already performs the CTC presence
            # scan.  Avoid a second identical directory listing here; an empty
            # result still falls through to the raw-audio discovery path.
            complete_stems, incomplete_stems, layout_map, wav_index = \
                discover_stems_separated(nas_ctc, nas_audio, require_all=True)
            all_stems = complete_stems + incomplete_stems
            if not all_stems:
                # Raw audio: discover all WAVs (CTC dir empty or no .lab files found)
                all_stems = []
                layout_map = {}
                wav_index = {}
                for entry in sorted(os.scandir(str(nas_audio)), key=lambda e: e.name):
                    if entry.is_file() and entry.name.endswith(".wav"):
                        s = entry.name[:-4]
                        wav_index[s] = Path(entry.path)
                        all_stems.append(s)
                        layout_map[s] = "flat"
                print(f"  {ds_name}: {len(all_stems)} stems (raw → all need GPU phase)")

        # Apply limit
        if args.limit > 0 and args.limit < len(all_stems):
            all_stems = all_stems[:args.limit]
            print(f"  {ds_name}: limited to {len(all_stems)} stems")

        # Build text_index for reference text (optional)
        text_index: dict[str, Path] = {}
        for s in all_stems:
            txt = nas_audio / f"{s}.txt"
            if txt.exists():
                text_index[s] = txt

        # Split into batches
        bs = args.batch_size
        for bi in range(0, len(all_stems), bs):
            batch_stems = all_stems[bi:bi + bs]
            all_gpu_batches.append((ds, bi // bs, batch_stems,
                                     layout_map, wav_index, text_index))

    if not all_gpu_batches:
        print("No batches to process!")
        return True

    total_batches = len(all_gpu_batches)
    resource_plan = plan_streaming_resources(
        cpu_budget=os.cpu_count() or 1,
        requested_gpu_workers=n_gpu_workers,
        requested_cpu_workers=requested_cpu_workers,
        requested_mfa_jobs=getattr(args, "mfa_jobs", None),
        config_mfa_jobs=getattr(args, "_effective_mfa_jobs", 0),
        config_mfa_en_jobs=getattr(args, "_effective_mfa_en_jobs", 0),
        requested_mfa_en_jobs=getattr(args, "mfa_en_jobs", None),
        batch_size=args.batch_size,
        batch_count=total_batches,
        gpu_queue_size=getattr(args, "prefetch_buffer", 0),
        cpu_queue_size=getattr(args, "upload_buffer", 0),
        pipelined=True,
    )
    n_gpu_workers = resource_plan["gpu_workers"]
    n_cpu_workers = resource_plan["cpu_workers"]
    print(f"\n  Pipelined mode: {n_gpu_workers} GPU workers, {n_cpu_workers} CPU workers"
          f" (CPU budget {resource_plan['cpu_budget']}; "
          f"MFA jobs {resource_plan['mfa_jobs_per_worker']}/worker)")
    print(f"  Total batches: {total_batches}")
    print(f"{'#'*60}")

    # ── Queues ──
    # Both queues are bounded.  The feeder/consumer loops use short timeouts
    # so an upstream failure cannot leave a producer blocked forever.
    gpu_queue: _queue.Queue = _queue.Queue(maxsize=resource_plan["gpu_queue_size"])
    cpu_queue: _queue.Queue = _queue.Queue(maxsize=resource_plan["cpu_queue_size"])
    stop_event = threading.Event()
    failure_event = threading.Event()

    def put_with_stop(target: _queue.Queue, item: object) -> bool:
        while not stop_event.is_set():
            try:
                target.put(item, timeout=0.25)
                return True
            except _queue.Full:
                continue
        return False

    def get_with_stop(source: _queue.Queue):
        while not stop_event.is_set():
            try:
                return source.get(timeout=0.25)
            except _queue.Empty:
                continue
        return None

    # ── Tracking ──
    ok_count = 0
    trusted_ok_count = 0
    fail_list: list[str] = []
    ckpt_lock = threading.Lock()
    ds_tracker: dict[str, dict] = {}  # {ds_name: {total, done, fail}}
    for ds_item in all_gpu_batches:
        ds_name = ds_item[0]["name"]
        if ds_name not in ds_tracker:
            ds_tracker[ds_name] = {"total": 0, "done": 0, "fail": 0}
        ds_tracker[ds_name]["total"] += 1

    upload_lock = threading.Lock()
    active_cpu_batches: set[tuple[str, str, int]] = set()
    accounted_cpu_failures: set[tuple[str, str, int]] = set()
    accounted_gpu_failures: set[tuple[str, str, int]] = set()

    def cpu_batch_key(item: tuple) -> tuple[str, str, int]:
        ds, batch_idx, _batch_stems, local_base, _producer_mode = item
        return (str(Path(local_base).resolve()), str(ds["name"]), int(batch_idx))

    def preserve_worker_failure_workspace(
            local_base: Path, batch_idx: int, dataset_name: str, *,
            reason: str = "unexpected failure") -> None:
        """Move one failed/aborted workspace without replacing evidence."""
        local_dir = _batch_local_dir(local_base, batch_idx, dataset_name)
        if not (local_dir.exists() or local_dir.is_symlink()):
            return
        try:
            failed_dir = _preserve_failed_batch(local_dir)
            print(f"  [PIPELINE] Preserved {reason}: {failed_dir}")
        except Exception as preserve_exc:
            # The original exception remains the batch failure.  Keep the
            # worker alive long enough to account and signal the pipeline even
            # if a filesystem error prevents an additional preservation move.
            print(f"  [PIPELINE] Could not preserve {local_dir}: "
                  f"{type(preserve_exc).__name__}: {preserve_exc}")

    def account_gpu_batch_failure(
            ds: dict, batch_idx: int, local_base: Path) -> None:
        key = (str(Path(local_base).resolve()), str(ds["name"]), int(batch_idx))
        with ckpt_lock:
            if key in accounted_gpu_failures:
                return
            accounted_gpu_failures.add(key)
            tracker = ds_tracker[ds["name"]]
            tracker["fail"] += 1
            if tracker["done"] + tracker["fail"] >= tracker["total"]:
                failed_set.add(ds["name"])

    def account_unconsumed_cpu_batch(item: tuple) -> None:
        """Fail and preserve one queued batch that no CPU worker consumed."""
        if item is _CPU_SENTINEL or not isinstance(item, tuple):
            return
        ds, batch_idx, _batch_stems, local_base, _producer_mode = item
        key = cpu_batch_key(item)
        with ckpt_lock:
            if key in accounted_cpu_failures:
                return
            accounted_cpu_failures.add(key)
            tracker = ds_tracker[ds["name"]]
            tracker["fail"] += 1
            if tracker["done"] + tracker["fail"] >= tracker["total"]:
                failed_set.add(ds["name"])
            active = key in active_cpu_batches
        # A queued item is not owned by a running CPU worker.  Preserve only
        # those workspaces here; an active worker owns its own failure cleanup.
        if not active:
            local_dir = _batch_local_dir(local_base, batch_idx, ds["name"])
            if local_dir.exists() or local_dir.is_symlink():
                _preserve_failed_batch(local_dir)

    def drain_unconsumed_cpu_queue() -> None:
        while True:
            try:
                item = cpu_queue.get_nowait()
            except _queue.Empty:
                return
            account_unconsumed_cpu_batch(item)

    # ── GPU worker ──
    def gpu_worker(wid: int) -> None:
        nonlocal trusted_ok_count
        drive = usable_drives[wid % len(usable_drives)]
        local_base = drive / f"gpu_{wid}"
        gpu_id = wid % n_gpu_workers
        device_str = f"cuda:{gpu_id}"
        # This pipelined producer is explicitly the NVASR fallback route.
        # Carry the route state with the batch so the CPU consumer cannot
        # infer ctc_ready merely from files appearing in the workspace.
        producer_mode = "nvrasr_fallback"
        while True:
            item = get_with_stop(gpu_queue)
            if item is None:
                break
            if item is _GPU_SENTINEL:
                break
            ds, batch_idx, batch_stems, layout_map, wav_index, text_index = item
            ds_name = ds["name"]
            remaining = gpu_queue.qsize()
            print(f"\n  [GPU{device_str}] [{total_batches - remaining}/{total_batches}]"
                  f" {ds_name}/{batch_idx:04d} ({len(batch_stems)} stems)")
            try:
                if _load_trusted_batch_evidence(
                        nas_output_root / ds_name, batch_idx, batch_stems,
                        config_sha256=config_sha256,
                        cache_sha256=cache_sha256,
                        implementation_sha256=implementation_sha256,
                        require_zero_filtered=require_zero_filtered) is not None:
                    with ckpt_lock:
                        ds_tracker[ds_name]["done"] += 1
                        trusted_ok_count += 1
                    print(f"  [GPU{device_str}] trusted evidence; skip "
                          f"{ds_name}/{batch_idx:04d}")
                    continue

                ok = _run_gpu_phase(
                    ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                    layout_map=layout_map, wav_index=wav_index, text_index=text_index,
                    local_base=local_base, config=args.config,
                    mfa_python=mfa_python, models_dir=models_dir,
                    batch_size=args.batch_size, python_path=args.python,
                    device=device_str,
                    nas_output_dir=nas_output_root / ds_name,
                    persist_cache=bool(pipeline_cfg.get("restore_ctc_cache", True)),
                    allow_overwrite=getattr(args, '_allow_overwrite', True),
                    allow_force=getattr(args, '_allow_force', True),
                )
                if ok:
                    cpu_item = (ds, batch_idx, batch_stems,
                                local_base, producer_mode)
                    if put_with_stop(cpu_queue, cpu_item):
                        print(f"  [GPU{device_str}] {ds_name}/{batch_idx:04d} → CPU queue")
                    else:
                        # A downstream failure may arrive after this GPU phase
                        # completed but before its CPU item was enqueued.  It
                        # is still one failed batch and must not be lost.
                        preserve_worker_failure_workspace(
                            local_base, batch_idx, ds_name,
                            reason="downstream-stop GPU result")
                        account_gpu_batch_failure(ds, batch_idx, local_base)
                        failure_event.set()
                        stop_event.set()
                        break
                else:
                    account_gpu_batch_failure(ds, batch_idx, local_base)
                    failure_event.set()
                    stop_event.set()
            except Exception as exc:
                print(f"  [GPU{device_str}] unexpected batch exception "
                      f"{ds_name}/{batch_idx:04d}: {type(exc).__name__}: {exc}")
                preserve_worker_failure_workspace(
                    local_base, batch_idx, ds_name,
                    reason="unexpected GPU failure")
                account_gpu_batch_failure(ds, batch_idx, local_base)
                failure_event.set()
                stop_event.set()
                break

    _GPU_SENTINEL = object()
    _CPU_SENTINEL = object()  # signals CPU worker to exit

    def gpu_feeder() -> None:
        try:
            for item in all_gpu_batches:
                if not put_with_stop(gpu_queue, item):
                    return
            for _ in range(n_gpu_workers):
                if not put_with_stop(gpu_queue, _GPU_SENTINEL):
                    return
        except Exception as exc:
            print(f"  [GPU QUEUE] feeder failure: {exc}")
            failure_event.set()
            stop_event.set()

    # ── CPU worker ──
    def cpu_worker(wid: int) -> tuple[int, list[str]]:
        w_ok = 0
        w_fails: list[str] = []
        while True:
            item = get_with_stop(cpu_queue)
            if item is None:
                break
            if item is _CPU_SENTINEL:
                break
            ds, batch_idx, batch_stems, local_base, producer_mode = item
            ds_name = ds["name"]
            batch_key = cpu_batch_key(item)
            with ckpt_lock:
                active_cpu_batches.add(batch_key)
            nas_output = nas_output_root / ds_name
            remaining = cpu_queue.qsize()
            print(f"\n  [CPU{ wid}] [q:{remaining}]"
                  f" {ds_name}/{batch_idx:04d} ({len(batch_stems)} stems)")

            unexpected_error = None
            try:
                ok = _run_cpu_phase(
                    ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                    local_base=local_base, config=args.config,
                    mfa_python=mfa_python, models_dir=models_dir,
                    nas_output=nas_output,
                    batch_size=args.batch_size, python_path=args.python,
                    mfa_num_jobs=resource_plan["mfa_jobs_per_worker"],
                    mfa_en_num_jobs=resource_plan["mfa_en_jobs_per_worker"],
                    allow_overwrite=getattr(args, '_allow_overwrite', True),
                    allow_force=getattr(args, '_allow_force', True),
                    restore_cache=bool(pipeline_cfg.get("restore_ctc_cache", True)),
                    config_sha256=config_sha256,
                    cache_sha256=cache_sha256,
                    implementation_sha256=implementation_sha256,
                    upload_lock=upload_lock,
                    producer_mode=producer_mode,
                    preserve_successful_workspaces=preserve_successful_workspaces,
                    require_zero_filtered=require_zero_filtered,
                )
            except Exception as exc:
                unexpected_error = exc
                ok = False
                print(f"  [CPU{wid}] unexpected batch exception "
                      f"{ds_name}/{batch_idx:04d}: {type(exc).__name__}: {exc}")
                preserve_worker_failure_workspace(
                    local_base, batch_idx, ds_name,
                    reason="unexpected CPU failure")
            finally:
                with ckpt_lock:
                    active_cpu_batches.discard(batch_key)
            with ckpt_lock:
                tracker = ds_tracker[ds_name]
                if ok:
                    # Count successful batch consumers for the summary.  The
                    # dataset-level COMPLETE receipt is finalized below;
                    # keeping both counters separate avoids reporting 0/1
                    # after all per-batch CPU phases succeeded.
                    w_ok += 1
                    tracker["done"] += 1
                else:
                    if batch_key not in accounted_cpu_failures:
                        accounted_cpu_failures.add(batch_key)
                        tracker["fail"] += 1
                if tracker["done"] + tracker["fail"] >= tracker["total"]:
                    if tracker["fail"] != 0:
                        w_fails.append(ds_name)
                        failed_set.add(ds_name)
                    status = "DONE" if tracker["fail"] == 0 else "FAIL"
                    print(f"  [CPU{wid}] {ds_name} — {status} "
                          f"({tracker['done']}/{tracker['total']} batches)")
            if unexpected_error is not None or not ok:
                failure_event.set()
                stop_event.set()
                break
        return w_ok, w_fails

    # ── Launch both pools ──
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpu_workers) as gpu_pool, \
         concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu_workers) as cpu_pool:
        feeder = threading.Thread(target=gpu_feeder, name="pipelined-gpu-feeder", daemon=True)
        feeder.start()
        gpu_futures = [gpu_pool.submit(gpu_worker, wid) for wid in range(n_gpu_workers)]
        cpu_futures = [cpu_pool.submit(cpu_worker, wid) for wid in range(n_cpu_workers)]

        # Wait for GPU workers to finish producing.  A worker catches all
        # per-batch exceptions, but the root collector remains defensive so a
        # residual worker failure cannot cause premature CPU sentinel writes.
        try:
            for fut in concurrent.futures.as_completed(gpu_futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  [GPU] worker exception: {type(exc).__name__}: {exc}")
                    failure_event.set()
                    stop_event.set()
        finally:
            # Sentinel insertion is valid only after every producer future is
            # quiescent.  This wait is also the guard for an exception raised
            # by the root future collector itself.
            concurrent.futures.wait(gpu_futures)
            # Wake CPU workers after GPU production is complete.  On success
            # do not set stop_event: queued CPU work must drain before its
            # sentinels.  On failure, timed queue operations let everybody
            # exit even if a queue is full.
            _shutdown_pipelined_cpu_queue(
                cpu_queue, n_cpu_workers, stop_event, failure_event,
                _CPU_SENTINEL, drain_unconsumed_cpu_queue)
            feeder.join(timeout=5)

    ok_count, fail_list = _collect_pipelined_cpu_futures(
        cpu_futures, failure_event, stop_event, drain_unconsumed_cpu_queue)
    # Trusted evidence is a completed batch success even though it never
    # reaches a CPU future.  Count only explicit trusted skips and successful
    # CPU results; failed or unconsumed work remains outside this denominator.
    ok_count += trusted_ok_count

    # Dataset publication is the only point at which a dataset becomes
    # complete.  Batch success alone is insufficient: aggregate all evidence,
    # prove conservation, verify COMPLETE receipts, then checkpoint.
    plans: dict[str, list[tuple[int, list[str]]]] = {}
    for ds, batch_idx, batch_stems, *_rest in all_gpu_batches:
        plans.setdefault(ds["name"], []).append((batch_idx, batch_stems))
    publish_failures = _finalize_dataset_publications(
        plans, ds_tracker, nas_output_root, completed_set, failed_set,
        config_sha256=config_sha256, cache_sha256=cache_sha256,
        implementation_sha256=implementation_sha256,
        receipt_mode="pipelined",
        receipt_route=["gpu", "cpu", "dataset_publish"],
        require_zero_filtered=require_zero_filtered)
    for name in publish_failures:
        if name not in fail_list:
            fail_list.append(name)
    _save_checkpoint(ckpt_path, completed_set, failed_set, checkpoint_identity)

    all_ok = (len(fail_list) == 0 and not failure_event.is_set()
              and all(name in completed_set for name in ds_tracker))
    print(f"\n{'#'*60}")
    dataset_ok_count = sum(1 for name in ds_tracker if name in completed_set)
    print(f"  PIPELINED BATCH COMPLETE: {dataset_ok_count}/{len(ds_tracker)} datasets OK; "
          f"{ok_count}/{len(all_gpu_batches)} batches OK"
          f"{' — ALL OK' if all_ok else ' — WITH FAILURES'}")
    if fail_list:
        print(f"  Failed: {', '.join(fail_list)}")
    print(f"{'#'*60}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(main())
