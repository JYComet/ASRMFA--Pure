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
import json
import os
import queue
import shutil
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
    CTC_SUFFIXES,
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

def _persist_ctc_adj_cache(local_workspace: Path, nas_speaker: Path) -> None:
    """Upload ctc_pretg_adj to NAS if it exists — preserves expensive adjust output."""
    local_adj = local_workspace / "ctc_pretg_adj"
    if not local_adj.exists() or not any(local_adj.iterdir()):
        return
    nas_adj = nas_speaker / "ctc_pretg_adj"
    nas_adj.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        try:
            subprocess.run(
                [rsync, "-a",
                 str(local_adj) + "/", str(nas_adj) + "/"],
                capture_output=True, text=True, timeout=60)
        except Exception:
            pass  # non-critical: CTC cache upload failed, adjust will re-run next time
    else:
        try:
            for f in local_adj.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(local_adj)
                    tgt = nas_adj / rel
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(tgt))
        except Exception:
            pass


def _restore_ctc_adj_cache(local_workspace: Path, nas_speaker: Path) -> bool:
    """Download ctc_pretg_adj from NAS if cached — skip expensive adjust step."""
    nas_adj = nas_speaker / "ctc_pretg_adj"
    if not nas_adj.exists() or not any(nas_adj.iterdir()):
        return False
    local_adj = local_workspace / "ctc_pretg_adj"
    local_adj.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        try:
            rc = subprocess.run(
                [rsync, "-a", str(nas_adj) + "/", str(local_adj) + "/"],
                capture_output=True, text=True, timeout=60).returncode
            return rc == 0
        except (subprocess.TimeoutExpired, Exception):
            return False  # CIFS can be slow; skip restore, adjust will re-run
    else:
        try:
            for f in nas_adj.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(nas_adj)
                    tgt = local_adj / rel
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(f), str(tgt))
            return True
        except Exception:
            return False


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
    local_dir = local_base / f"batch_{batch_idx:04d}"
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
    missing_audio = 0

    for stem in batch_stems:
        src_wav = wav_index.get(stem)
        if src_wav is None:
            src_wav = find_wav(nas_audio_dir, stem)
        if src_wav:
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

    if not copy_tasks:
        elapsed = time.time() - t0
        print(f"  [STAGE {batch_idx:04d}] WARNING: no files to copy "
              f"(missing_audio={missing_audio})")
        return local_dir, elapsed, missing_audio

    n_workers = min(8, max(1, len(copy_tasks) // 100))
    failed_copies = 0
    with _cf2.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(link_or_copy_file, s, d) for s, d in copy_tasks]
        for f in _cf2.as_completed(futures):
            try:
                if not f.result():
                    failed_copies += 1
            except Exception:
                failed_copies += 1

    if failed_copies:
        print(f"  [STAGE {batch_idx:04d}] WARNING: {failed_copies}/{len(copy_tasks)} "
              f"copies failed (source files missing on NAS?)")

    # Write manifest for run_pipeline.py (ctc_ready only)
    if not is_fallback:
        manifest = {"stems": batch_stems, "n_stems": len(batch_stems)}
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
    local_dir = local_base / f"batch_{batch_idx:04d}"
    local_audio = local_dir / "audio"
    local_ctc = local_dir / "ctc"
    local_output = local_dir / "output"
    local_workspace = local_dir / "workspace"

    nas_output = nas_output_root / ds["name"]
    is_fallback = (mode == "nvrasr_fallback")

    # ── Restore cached adjust output (optional NAS read) ──
    if restore_cache:
        adj_cached = _restore_ctc_adj_cache(local_workspace, nas_output)
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
    if device:
        gpu_idx = device.replace("cuda:", "")
        if gpu_idx.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx

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

    upload_ok = _upload_one_batch(
        local_dir=local_dir, nas_output_root=nas_output_root,
        ds_name=ds["name"], batch_idx=batch_idx,
    )
    _cleanup_one_batch_dir(local_dir)

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

            if missing_audio:
                print(f"    WARNING: audio not found for {missing_audio}/{len(stems)} stems")

            # ── 并行执行拷贝 (I/O-bound, 8 线程足够饱和 CIFS) ──
            n_workers = min(8, len(copy_tasks))
            copied = 0
            failed = 0
            with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(link_or_copy_file, src, dst)
                    for src, dst in copy_tasks
                ]
                for fut in _cf.as_completed(futures):
                    try:
                        if fut.result():
                            copied += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

            if failed:
                print(f"    ERROR: {failed}/{len(copy_tasks)} file copies failed")

            ok = (missing_audio == 0 and failed == 0)

            # 写 manifest
            manifest = {"stems": stems, "n_stems": len(stems)}
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


def _run_direct(args, data_dir: Path, ctc_dir: Path, output_dir: Path | None):
    """Pass-through to run_pipeline.py — data is local, no streaming needed."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(args.config),
        "--mode", "ctc_ready",
        "--data-dir", str(data_dir),
        "--ctc-ready", str(ctc_dir),
    ]
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
            sys.exit(1)
        return

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
        )
        if not ok:
            sys.exit(1)
    else:
        if remote:
            print("Mode:      DIRECT (remote fs, no --local-work; will be slow)")
            print("           Tip: add --local-work /ssd/mfa_work for streaming")
        else:
            print("Mode:      DIRECT (local fs)")
        _run_direct(args, data_dir, ctc_dir, output_dir)


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
) -> bool:
    """Run pipeline for a single dataset.  Returns True on success.

    Args:
        staged: If True (default), use Stage All → Process All → Upload All.
                If False, use streaming (interleaved prefetch/process/upload).
    """
    import concurrent.futures as _cf3

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
        stems = list(stems_override)
        layout_map = {s: "nested" for s in stems}
        wav_index = {}
        for s in stems:
            w = find_wav(nas_audio_dir, s)
            if w:
                wav_index[s] = w
        stems = [s for s in stems if s in wav_index]
        print(f"  Using {len(stems)} stems (override)")
    else:
        stems, _, layout_map, wav_index = discover_stems_separated(
            nas_ctc_dir, nas_audio_dir, require_all=True)
    if limit > 0:
        stems = stems[:limit]
    print(f"  Found {len(stems)} valid stems"
          + (f" (limited from discovery)" if limit > 0 else ""))

    if not stems:
        print("ERROR: No valid stems found!")
        return False

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
    t_stage = time.time()

    for bi in range(len(batch_mgr)):
        bstems = batch_mgr.batches[bi]
        try:
            local_dir, elapsed, missing = _stage_one_batch(
                ds=ds, batch_idx=bi, batch_stems=bstems,
                layout_map=layout_map, wav_index=wav_index,
                local_base=local_work, mode="ctc_ready",
            )
            if missing:
                print(f"  [STAGE] FAIL batch {bi}: {missing} missing audio files")
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
    proc_lock = threading.Lock()

    def _proc_one(bi: int) -> bool:
        bstems = batch_mgr.batches[bi]
        return _process_one_batch(
            ds=ds, batch_idx=bi, batch_stems=bstems,
            local_base=local_work, config=config,
            mfa_python=mfa_python, models_dir=models_dir,
            nas_output_root=nas_output_root,
            batch_size=batch_size, python_path=python_path,
            mode="ctc_ready", device=device,
            restore_cache=False,
            persist_cache_on_failure=False,
            mfa_num_jobs=mfa_num_jobs,
            mfa_en_num_jobs=mfa_en_num_jobs,
        )

    n_proc = min(parallel_batches, len(staged_dirs))
    if n_proc <= 1:
        for bi in sorted(staged_dirs.keys()):
            ok = _proc_one(bi)
            if ok:
                proc_ok += 1
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

    for bi in sorted(staged_dirs.keys()):
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

    return proc_fail == 0 and up_fail == 0


def _load_checkpoint(ckpt_path: Path) -> set[str]:
    """Return set of completed dataset names from checkpoint."""
    if not ckpt_path.exists():
        return set()
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding='utf-8'))
        return set(ckpt.get("completed", []))
    except Exception:
        return set()


def _save_checkpoint(ckpt_path: Path, completed: set[str], failed: set[str]) -> None:
    """Atomically write checkpoint (write-then-rename)."""
    import datetime as _dt
    ckpt = {
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
    """Return per-dataset batch progress from .batch_progress.json.

    Returns {ds_name: {done: int, fail: int, total: int}}.
    """
    progress_path = ckpt_path.with_name(ckpt_path.stem + ".batch_progress.json")
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save_batch_progress(ckpt_path: Path, ds_name: str,
                         done: int, fail: int, total: int) -> None:
    """Atomically write per-dataset batch progress (non-blocking best-effort)."""
    progress_path = ckpt_path.with_name(ckpt_path.stem + ".batch_progress.json")
    try:
        data = {}
        if progress_path.exists():
            try:
                data = json.loads(progress_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        data[ds_name] = {"done": done, "fail": fail, "total": total}
        tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        tmp.replace(progress_path)
    except Exception:
        pass  # best-effort, don't block on I/O


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
                                     tracker["done"], tracker["fail"], tracker["total"])
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
    # Save interim progress (Phase 2 done, not yet Phase 3)
    _save_checkpoint(ckpt_path, completed_set, failed_set)
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
    ds_upload_batches: dict[str, list[tuple[int, Path]]] = {}  # {ds_name: [(gidx, local_dir), ...]}
    for gidx, (batch_mode, ds, batch_idx, batch_stems,
               layout_map, wav_index, text_index) in enumerate(all_batches):
        if gidx in stage_failures:
            continue
        ds_name = ds["name"]
        local_dir = staged_dirs.get(gidx)
        if local_dir and local_dir.exists():
            ds_upload_batches.setdefault(ds_name, []).append((gidx, local_dir))

    upload_failures: list[str] = []
    upload_lock = threading.Lock()
    upload_total = sum(len(v) for v in ds_upload_batches.values())
    upload_done = 0

    def upload_dataset(ds_name: str, batches: list[tuple[int, Path]]) -> bool:
        nonlocal upload_done
        ok = True
        for seq_idx, (gidx, local_dir) in enumerate(batches):
            if not _upload_one_batch(
                local_dir=local_dir,
                nas_output_root=nas_output_root,
                ds_name=ds_name,
                batch_idx=seq_idx,
            ):
                ok = False
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
        if local_dir.exists():
            _cleanup_one_batch_dir(local_dir)

    # Only mark datasets as completed if upload succeeded.
    # Datasets with any upload failure are NOT added to completed_set.
    for ds_name in _phase2_ok:
        if ds_name in upload_failures:
            failed_set.add(ds_name)
        else:
            completed_set.add(ds_name)
    _save_checkpoint(ckpt_path, completed_set, failed_set)

    total_elapsed = time.time() - t_stage_start
    print(f"\n  PHASE 3 DONE: {upload_done - len(upload_failures)}/{upload_total} "
          f"batches uploaded ({upload_elapsed:.0f}s)")
    print(f"  STAGED TOTAL: stage={stage_elapsed:.0f}s + "
          f"process={proc_elapsed:.0f}s + upload={upload_elapsed:.0f}s "
          f"= {total_elapsed:.0f}s")

    # Recalculate ok_count based on published datasets
    ok_count = len([d for d in _phase2_ok if d not in upload_failures])
    fail_list = list(failed_set)

    return ok_count, fail_list


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

    with open(cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    all_datasets = cache.get("datasets", [])
    if not all_datasets:
        print("ERROR: No datasets in cache!")
        sys.exit(1)

    # ── Resume: skip already-completed datasets ──
    completed_set: set[str] = set()
    failed_set: set[str] = set()
    if not getattr(args, 'no_resume', False):
        completed_set = _load_checkpoint(ckpt_path)
        if completed_set:
            pending = [d for d in all_datasets if d["name"] not in completed_set]
            skipped = len(all_datasets) - len(pending)
            print(f"\n  Resume: {skipped} already completed, {len(pending)} remaining")
            all_datasets = pending
    if not all_datasets:
        print("All datasets already completed!")
        return

    datasets = all_datasets[:args.limit_datasets] if args.limit_datasets > 0 else all_datasets

    # Resolve parallelism: CLI > config > default 1
    parallel = args.parallel_datasets
    _cfg = getattr(args, '_config', {})
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
        return run_pipelined_batch(args)

    if parallel <= 1:
        _run_batch_sequential(args, datasets, cache, ckpt_path, completed_set, failed_set)
        return

    # ── Batch-level parallel mode ──
    # Pre-discover stems for ALL datasets, split into batches, put ALL
    # individual batches into a shared queue.  Every worker processes
    # whatever batch is available — including batches from the same
    # dataset.  A 100k-stem dataset with 50 batches gets distributed
    # across all 8 workers instead of being stuck on 1 worker.
    import queue as _queue

    # Phase 1: pre-scan all datasets → build batch task list
    print(f"\n  Pre-scanning {len(datasets)} datasets ...")

    # Load batch-level progress for resume (which batches within each dataset are done)
    _batch_progress = _load_batch_progress(ckpt_path) if not getattr(args, 'no_resume', False) else {}
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

        batch_size_eff = args.batch_size

        # ── Enqueue ctc_ready batches (complete stems) ──
        _bp = _batch_progress.get(ds_name, {})
        _already_done = _bp.get("done", 0)
        if complete_stems:
            batches_ctc = [complete_stems[i:i + batch_size_eff]
                           for i in range(0, len(complete_stems), batch_size_eff)]
            _skipped_ctc = 0
            for batch_idx, batch_stems in enumerate(batches_ctc):
                if _already_done > 0:
                    _already_done -= 1
                    _skipped_ctc += 1
                    continue
                all_batches.append(
                    ("ctc_ready", ds, batch_idx, batch_stems, layout_map, wav_index, None))
            total_stems += len(complete_stems)
            _info = f"  {ds_name}: {len(complete_stems)} stems → {len(batches_ctc)} ctc_ready batches"
            if _skipped_ctc:
                _info += f" (skipped {_skipped_ctc} already done)"
            print(_info)

        # ── Enqueue nvrasr_fallback batches (incomplete stems) ──
        if incomplete_stems:
            # Build text index for NVASR reference text
            text_index = build_file_index(nas_audio, ".txt")
            if not text_index:
                print(f"  {ds_name}: WARNING: {len(incomplete_stems)} fallback stems "
                      f"have no reference .txt — NVASR will use ASR-only")
            batches_fb = [incomplete_stems[i:i + batch_size_eff]
                          for i in range(0, len(incomplete_stems), batch_size_eff)]
            _skipped_fb = 0
            for batch_idx, batch_stems in enumerate(batches_fb):
                if _already_done > 0:
                    _already_done -= 1
                    _skipped_fb += 1
                    continue
                all_batches.append(
                    ("nvrasr_fallback", ds, batch_idx, batch_stems,
                     layout_map, incomplete_wav_index, text_index))
            total_incomplete += len(incomplete_stems)
            _info = f"  {ds_name}: {len(incomplete_stems)} stems → {len(batches_fb)} nvrasr_fallback batches"
            if _skipped_fb:
                _info += f" (skipped {_skipped_fb} already done)"
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
        print("ERROR: No batches to process!")
        sys.exit(1)

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
                                         tracker["done"], tracker["fail"], tracker["total"])
                    if tracker["done"] + tracker["fail"] >= tracker["total"]:
                        if tracker["fail"] == 0:
                            w_ok += 1
                            completed_set.add(ds_name)
                        else:
                            w_fails.append(ds_name)
                            failed_set.add(ds_name)
                        _save_checkpoint(ckpt_path, completed_set, failed_set)
                        status = "DONE" if tracker["fail"] == 0 else "FAIL"
                        print(f"  [W{worker_id}] {ds_name} — {status} "
                              f"({tracker['done']}/{tracker['total']} batches)")

            if local_base.exists():
                shutil.rmtree(local_base, ignore_errors=True)
            return w_ok, w_fails

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(worker, wid) for wid in range(parallel)]
            for fut in concurrent.futures.as_completed(futures):
                w_ok, w_fails = fut.result()
                ok_count += w_ok
                fail_list.extend(w_fails)

    # ── Final summary ──
    all_ok = len(fail_list) == 0
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
                         failed_set: set[str]) -> None:
    """Sequential dataset loop with checkpoint after each dataset (used when parallel=1)."""
    ok_count = 0
    fail_list: list[str] = []
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
        )

        if ok:
            ok_count += 1
            completed_set.add(ds_name)
            if ds_local.exists():
                shutil.rmtree(ds_local, ignore_errors=True)
        else:
            failed_set.add(ds_name)
            fail_list.append(ds_name)
        _save_checkpoint(ckpt_path, completed_set, failed_set)

        print(f"\n  [{i+1}/{len(datasets)}] {ds_name} — "
              f"{'DONE' if ok else 'FAILED'}")

    print(f"\n{'#'*60}")
    print(f"  BATCH COMPLETE: {ok_count}/{len(datasets)} OK")
    if fail_list:
        print(f"  Failed: {', '.join(fail_list)}")
    print(f"{'#'*60}")


# ═══════════════════════════════════════════════════════════════
# Pipelined mode — GPU (NVASR) and CPU (MFA) in parallel stages
# ═══════════════════════════════════════════════════════════════

def _run_gpu_phase(
    ds: dict, batch_idx: int, batch_stems: list[str],
    layout_map: dict, wav_index: dict, text_index: dict[str, Path] | None,
    local_base: Path, config: Path,
    mfa_python: Path, models_dir: Path,
    batch_size: int, python_path: str | None,
    device: str,
    nas_output_dir: Path | None = None,
    allow_overwrite: bool = True,
    allow_force: bool = True,
) -> bool:
    """GPU phase: prefetch WAVs + NVASR prealign + normalize -> CTC output.

    Leaves the local workspace intact for the CPU phase to pick up.
    Returns True on success.
    """
    import concurrent.futures as _cf

    local_dir = local_base / f"batch_{batch_idx:04d}"
    local_audio = local_dir / "audio"
    local_ctc = local_dir / "ctc"
    local_workspace = local_dir / "workspace"

    # ── Prefetch audio files ──
    local_audio.mkdir(parents=True, exist_ok=True)
    local_ctc.mkdir(parents=True, exist_ok=True)
    copy_tasks: list[tuple[Path, Path]] = []
    missing_audio: list[str] = []
    for stem in batch_stems:
        src_wav = wav_index.get(stem) or find_wav(resolve_input_path(ds.get("audio_dir", "")), stem)
        if src_wav:
            copy_tasks.append((src_wav, local_audio / f"{stem}.wav"))
        else:
            missing_audio.append(stem)
        # Copy .txt if available (for reference text in NVASR)
        if text_index and stem in text_index:
            copy_tasks.append((text_index[stem], local_audio / f"{stem}.txt"))

    if missing_audio:
        print(f"  [GPU] Missing audio for {len(missing_audio)}/{len(batch_stems)} stems")
        return False
    n_workers = min(8, max(1, len(copy_tasks) // 100))
    failed_copies = 0
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(link_or_copy_file, s, d) for s, d in copy_tasks]
        for f in _cf.as_completed(futures):
            try:
                if not f.result():
                    failed_copies += 1
            except Exception:
                failed_copies += 1
    if failed_copies:
        print(f"  [GPU] {failed_copies}/{len(copy_tasks)} staging copies failed")
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
    if device:
        gpu_idx = device.replace("cuda:", "")
        if gpu_idx.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx

    try:
        rc = subprocess.run(cmd, env=env, timeout=7200, capture_output=False).returncode
    except subprocess.TimeoutExpired:
        rc = 1

    if rc != 0:
        # Preserve failed batch directory for forensic analysis
        _failed_dir = local_dir.with_name(local_dir.name + ".FAILED")
        if _failed_dir.exists():
            shutil.rmtree(_failed_dir, ignore_errors=True)
        shutil.move(str(local_dir), str(_failed_dir))
        print(f"  [GPU] Preserved: {_failed_dir}")
        return False

    # Persist CTC output to NAS for caching.
    # Save to BOTH ctc_dir (GPU phase default) and nas_output_dir (CPU phase reads from here),
    # so the restore in _run_cpu_phase always finds the cache regardless of which path it checks.
    _persist_ctc_adj_cache(local_workspace,
                           resolve_input_path(ds.get("ctc_dir", "")))
    if nas_output_dir:
        _persist_ctc_adj_cache(local_workspace, nas_output_dir)
    return True


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
) -> bool:
    """CPU phase: read CTC from local workspace + run MFA align + postprocess.

    The local workspace must already contain CTC output from the GPU phase.
    Uploads final output to NAS and cleans up the local directory.
    """
    local_dir = local_base / f"batch_{batch_idx:04d}"
    local_audio = local_dir / "audio"        # where GPU phase put WAVs
    local_ctc = local_dir / "ctc"
    local_workspace = local_dir / "workspace"
    local_output = local_dir / "output"

    # ── Prepare CTC manifest for ctc_ready mode ──
    manifest = {"stems": batch_stems, "n_stems": len(batch_stems)}
    local_ctc.mkdir(parents=True, exist_ok=True)
    (local_ctc / "ctc_ready_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False))

    # ── Link CTC output from GPU phase → CPU phase expects in local_ctc/ ──
    # GPU phase (nvrasr_fallback) writes to workspace/ctc_pretg/.
    # If adjust ran, also workspace/ctc_pretg_adj/ (better quality).
    # Search priority: ctc_pretg_adj > ctc_pretg > (already in local_ctc)
    _ctc_sources = [
        local_workspace / "ctc_pretg_adj",
        local_workspace / "ctc_pretg",
    ]
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
                        os.link(str(_f), str(_tgt))
                    except OSError:
                        shutil.copy2(str(_f), str(_tgt))
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

    # ── Restore cached adjust output if available ──
    _restore_ctc_adj_cache(local_workspace, nas_output)

    # ── Run MFA alignment + postprocess (CPU-intensive) ──
    cmd = [
        str(mfa_python),
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(config),
        "--mode", "ctc_ready",
        "--ctc-ready", str(local_ctc),
        "--data-dir", str(local_audio),
        "--output-dir", str(local_output),
        "--workspace", str(local_workspace),
        "--python", str(mfa_python),
    ]
    if allow_overwrite:
        cmd.append("--overwrite")
    if allow_force:
        cmd.append("--force")
    # GPU phase already did pad_silence → skip to avoid double I/O.
    # If GPU didn't run (e.g. standalone ctc_ready), padded_audio won't exist → keep it.
    if (local_workspace / "padded_audio").exists():
        cmd.append("--skip-pad_silence")
    if mfa_num_jobs > 0:
        cmd += ["--mfa-jobs", str(mfa_num_jobs)]
    if mfa_en_num_jobs > 0:
        cmd += ["--mfa-en-jobs", str(mfa_en_num_jobs)]
    env = get_mfa_env(mfa_python, models_dir)
    try:
        rc = subprocess.run(cmd, env=env, timeout=7200, capture_output=False).returncode
    except subprocess.TimeoutExpired:
        rc = 1

    if rc != 0:
        # Preserve CTC cache even on failure
        _persist_ctc_adj_cache(local_workspace, nas_output)
        # Preserve failed batch directory for forensic analysis
        _failed_dir = local_dir.with_name(local_dir.name + ".FAILED")
        if _failed_dir.exists():
            shutil.rmtree(_failed_dir, ignore_errors=True)
        shutil.move(str(local_dir), str(_failed_dir))
        print(f"  [CPU] Preserved: {_failed_dir}")
        return False

    # ── Upload results to NAS ──
    _upload_ok = True
    for local_src, nas_rel in [
        (local_output, nas_output / "output"),
        (local_workspace / "filtered", nas_output / "filtered"),
        (local_workspace / "ctc_pretg_adj", nas_output / "ctc_pretg_adj"),
    ]:
        if not local_src.exists() or not any(local_src.iterdir()):
            continue
        nas_rel.mkdir(parents=True, exist_ok=True)
        rsync = shutil.which("rsync")
        if rsync:
            try:
                rc = subprocess.run(
                    [rsync, "-a", str(local_src) + "/", str(nas_rel) + "/"],
                    capture_output=True, text=True, timeout=300).returncode
                if rc != 0:
                    print(f"  ERROR: rsync failed (rc={rc}) for {local_src} → {nas_rel}")
                    _upload_ok = False
            except subprocess.TimeoutExpired:
                print(f"  ERROR: rsync timed out for {local_src} → {nas_rel}")
                _upload_ok = False
            except Exception as e:
                print(f"  ERROR: rsync error for {local_src}: {e}")
                _upload_ok = False
        else:
            try:
                for f in local_src.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(local_src)
                        tgt = nas_rel / rel
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(f), str(tgt))
            except Exception as e:
                print(f"  ERROR: upload copy error for {local_src}: {e}")
                _upload_ok = False

    if not _upload_ok:
        # Preserve local directory for forensic analysis
        print(f"  [CPU] Upload failed — preserving {local_dir}")
        return False

    # Cleanup
    shutil.rmtree(local_dir, ignore_errors=True)
    return True


def run_pipelined_batch(args) -> bool:
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

    with open(cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    all_datasets = cache.get("datasets", [])
    if not all_datasets:
        print("ERROR: No datasets in cache!")
        sys.exit(1)

    # ── Config ──
    _cfg = getattr(args, '_config', {})
    pipeline_cfg = _cfg.get("pipelined", {}) if _cfg else {}
    n_gpu_workers = args.gpus  # 1 GPU per GPU worker; bounded after scanning.
    requested_cpu_workers = args.cpu_workers or pipeline_cfg.get("cpu_workers", 0)

    # ── Resume ──
    ckpt_path = cache_path.with_name(cache_path.stem + ".checkpoint.json")
    completed_set = _load_checkpoint(ckpt_path)
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
            all_gpu_batches.append((ds, len(all_gpu_batches), batch_stems,
                                     layout_map, wav_index, text_index))

    if not all_gpu_batches:
        print("No batches to process!")
        return

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
    fail_list: list[str] = []
    ckpt_lock = threading.Lock()
    ds_tracker: dict[str, dict] = {}  # {ds_name: {total, done, fail}}
    for ds_item in all_gpu_batches:
        ds_name = ds_item[0]["name"]
        if ds_name not in ds_tracker:
            ds_tracker[ds_name] = {"total": 0, "done": 0, "fail": 0}
        ds_tracker[ds_name]["total"] += 1

    nas_output_root = resolve_input_path(
        cache.get("output_root", "").rstrip("/"), PROJECT_ROOT)

    # ── GPU worker ──
    def gpu_worker(wid: int) -> None:
        drive = usable_drives[wid % len(usable_drives)]
        local_base = drive / f"gpu_{wid}"
        gpu_id = wid % n_gpu_workers
        device_str = f"cuda:{gpu_id}"
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

            ok = _run_gpu_phase(
                ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                layout_map=layout_map, wav_index=wav_index, text_index=text_index,
                local_base=local_base, config=args.config,
                mfa_python=mfa_python, models_dir=models_dir,
                batch_size=args.batch_size, python_path=args.python,
                device=device_str,
                nas_output_dir=nas_output_root / ds_name,
                allow_overwrite=getattr(args, '_allow_overwrite', True),
                allow_force=getattr(args, '_allow_force', True),
            )
            if ok:
                if put_with_stop(cpu_queue, (ds, batch_idx, batch_stems, local_base)):
                    print(f"  [GPU{device_str}] {ds_name}/{batch_idx:04d} → CPU queue")
                else:
                    failure_event.set()
                    break
            else:
                failure_event.set()
                stop_event.set()
                with ckpt_lock:
                    tracker = ds_tracker[ds_name]
                    tracker["fail"] += 1
                    if tracker["done"] + tracker["fail"] >= tracker["total"]:
                        failed_set.add(ds_name)
                        _save_checkpoint(ckpt_path, completed_set, failed_set)

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
            ds, batch_idx, batch_stems, local_base = item
            ds_name = ds["name"]
            nas_output = nas_output_root / ds_name
            remaining = cpu_queue.qsize()
            print(f"\n  [CPU{ wid}] [q:{remaining}]"
                  f" {ds_name}/{batch_idx:04d} ({len(batch_stems)} stems)")

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
            )
            with ckpt_lock:
                tracker = ds_tracker[ds_name]
                if ok:
                    tracker["done"] += 1
                else:
                    tracker["fail"] += 1
                if tracker["done"] + tracker["fail"] >= tracker["total"]:
                    if tracker["fail"] == 0:
                        w_ok += 1
                        completed_set.add(ds_name)
                    else:
                        w_fails.append(ds_name)
                        failed_set.add(ds_name)
                    _save_checkpoint(ckpt_path, completed_set, failed_set)
                    status = "DONE" if tracker["fail"] == 0 else "FAIL"
                    print(f"  [CPU{wid}] {ds_name} — {status} "
                          f"({tracker['done']}/{tracker['total']} batches)")
            if not ok:
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

        # Wait for GPU workers to finish producing
        try:
            for fut in concurrent.futures.as_completed(gpu_futures):
                fut.result()  # propagate exceptions
        finally:
            # Wake CPU workers after GPU production is complete.  On success
            # do not set stop_event: queued CPU work must drain before its
            # sentinels.  On failure, timed queue operations let everybody
            # exit even if a queue is full.
            if failure_event.is_set():
                stop_event.set()
                for _ in range(n_cpu_workers):
                    try:
                        cpu_queue.put_nowait(_CPU_SENTINEL)
                    except _queue.Full:
                        break
            else:
                for _ in range(n_cpu_workers):
                    while True:
                        try:
                            cpu_queue.put(_CPU_SENTINEL, timeout=0.25)
                            break
                        except _queue.Full:
                            continue
            feeder.join(timeout=5)

        for fut in concurrent.futures.as_completed(cpu_futures):
            w_ok, w_fails = fut.result()
            ok_count += w_ok
            fail_list.extend(w_fails)

    all_ok = len(fail_list) == 0 and not failure_event.is_set()
    print(f"\n{'#'*60}")
    print(f"  PIPELINED BATCH COMPLETE: {ok_count}/{len(all_datasets)} OK"
          f"{' — ALL OK' if all_ok else ' — WITH FAILURES'}")
    if fail_list:
        print(f"  Failed: {', '.join(fail_list)}")
    print(f"{'#'*60}")
    return all_ok


if __name__ == "__main__":
    main()
