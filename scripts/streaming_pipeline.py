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
        subprocess.run(
            [rsync, "-a", "--no-inc-recursive",
             str(local_adj) + "/", str(nas_adj) + "/"],
            capture_output=True, text=True, timeout=600)
    else:
        for f in local_adj.rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_adj)
                tgt = nas_adj / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(tgt))


def _restore_ctc_adj_cache(local_workspace: Path, nas_speaker: Path) -> bool:
    """Download ctc_pretg_adj from NAS if cached — skip expensive adjust step."""
    nas_adj = nas_speaker / "ctc_pretg_adj"
    if not nas_adj.exists() or not any(nas_adj.iterdir()):
        return False
    local_adj = local_workspace / "ctc_pretg_adj"
    local_adj.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        rc = subprocess.run(
            [rsync, "-a", "--no-inc-recursive",
             str(nas_adj) + "/", str(local_adj) + "/"],
            capture_output=True, text=True, timeout=600).returncode
        return rc == 0
    else:
        for f in nas_adj.rglob("*"):
            if f.is_file():
                rel = f.relative_to(nas_adj)
                tgt = local_adj / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(tgt))
        return True


def run_single_batch(
    ds: dict, batch_idx: int, batch_stems: list[str],
    layout_map: dict, wav_index: dict,
    local_base: Path, config: Path,
    mfa_python: Path, models_dir: Path,
    nas_output_root: Path,
    batch_size: int, python_path: str | None = None,
    mode: str = "ctc_ready",
    text_index: dict[str, Path] | None = None,
) -> bool:
    """Process a single batch (one set of stems) end-to-end.

    Prefetch → run_pipeline → upload → cleanup.  Synchronous; designed
    to be called from a ThreadPoolExecutor worker for batch-level parallelism.

    Args:
        mode: "ctc_ready" (pre-existing CTC, fast) or "nvrasr_fallback"
              (no/incomplete CTC, runs NVASR from scratch).
        text_index: {stem: txt_path} for NVASR reference text (fallback only).
    """
    local_dir = local_base / f"batch_{batch_idx:04d}"
    local_audio = local_dir / "audio"
    local_ctc = local_dir / "ctc"
    local_output = local_dir / "output"
    local_workspace = local_dir / "workspace"

    nas_ctc_dir = resolve_input_path(ds.get("ctc_dir", ""))
    nas_audio_dir = resolve_input_path(ds.get("audio_dir", ""))
    nas_output = nas_output_root / ds["name"]

    is_fallback = (mode == "nvrasr_fallback")

    # ── 1. Prefetch: NAS → local NVMe ──
    t0 = time.time()
    local_audio.mkdir(parents=True, exist_ok=True)

    import concurrent.futures as _cf2
    copy_tasks: list[tuple[Path, Path]] = []
    for stem in batch_stems:
        src_wav = wav_index.get(stem)
        if src_wav is None:
            src_wav = find_wav(nas_audio_dir, stem)
        if src_wav:
            copy_tasks.append((src_wav, local_audio / f"{stem}.wav"))
        if is_fallback:
            # Copy .txt reference text alongside audio for NVASR
            txt_src = None
            if text_index and stem in text_index:
                txt_src = text_index[stem]
            else:
                # Try flat, then nested layout
                for txt_path in (nas_audio_dir / f"{stem}.txt",
                                 nas_audio_dir / stem / f"{stem}.txt"):
                    if txt_path.exists():
                        txt_src = txt_path
                        break
            if txt_src:
                copy_tasks.append((txt_src, local_audio / f"{stem}.txt"))
        else:
            # ctc_ready: copy CTC files
            if not is_fallback:
                local_ctc.mkdir(parents=True, exist_ok=True)
            layout = layout_map.get(stem, "flat")
            ctc_base = nas_ctc_dir / stem if layout == "nested" else nas_ctc_dir
            for suffix in CTC_SUFFIXES:
                copy_tasks.append(
                    (ctc_base / f"{stem}{suffix}",
                     local_ctc / f"{stem}{suffix}")
                )

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
        print(f"  [BATCH {batch_idx:04d}] WARNING: {failed_copies}/{len(copy_tasks)} "
              f"prefetch copies failed (source files missing on NAS?)")

    # Write manifest for run_pipeline.py (ctc_ready only)
    if not is_fallback:
        manifest = {"stems": batch_stems, "n_stems": len(batch_stems)}
        (local_ctc / "ctc_ready_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False))

    prefetch_elapsed = time.time() - t0

    # ── 1.5 Restore cached adjust output (skip expensive adjust_ctc step) ──
    adj_cached = _restore_ctc_adj_cache(local_workspace, nas_output)
    if adj_cached:
        print(f"  [BATCH {batch_idx:04d}] Restored ctc_pretg_adj from NAS cache")

    # ── 2. Process: run_pipeline.py ──
    cmd = [
        str(mfa_python),
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(config),
        "--mode", mode,
        "--data-dir", str(local_audio),
        "--output-dir", str(local_output),
        "--workspace", str(local_workspace),
        "--python", str(mfa_python),
        "--overwrite", "--force",
    ]
    if is_fallback:
        # nvrasr_fallback: no --ctc-ready, NVASR generates CTC from scratch
        pass
    else:
        cmd += ["--ctc-ready", str(local_ctc)]
    t1 = time.time()
    try:
        rc = subprocess.run(
            cmd, env=get_mfa_env(mfa_python, models_dir),
            timeout=7200, capture_output=False,
        ).returncode
    except subprocess.TimeoutExpired:
        rc = 1
    process_elapsed = time.time() - t1

    if rc != 0:
        print(f"  [BATCH {batch_idx:04d}] {ds['name']} FAIL (rc={rc}) "
              f"(prefetch={prefetch_elapsed:.0f}s process={process_elapsed:.0f}s)")
        # Even on failure, preserve ctc_pretg_adj (expensive adjust output) to NAS
        _persist_ctc_adj_cache(local_workspace, nas_output)
        shutil.rmtree(local_dir, ignore_errors=True)
        return False

    # ── 3. Upload: local NVMe → NAS ──
    t2 = time.time()
    upload_ok = True
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
                rc_up = subprocess.run(
                    [rsync, "-a", "--no-inc-recursive",
                     str(local_src) + "/", str(nas_rel) + "/"],
                    capture_output=True, text=True, timeout=600).returncode
                if rc_up != 0:
                    print(f"    rsync warning: rc={rc_up} for {local_src}")
            except subprocess.TimeoutExpired:
                print(f"    rsync TIMEOUT for {local_src}, falling back to copy")
                rsync = None  # fall through to manual copy
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

    upload_elapsed = time.time() - t2

    # ── 4. Cleanup ──
    shutil.rmtree(local_dir, ignore_errors=True)

    total = time.time() - t0
    print(f"  [BATCH {batch_idx:04d}] {ds['name']} OK "
          f"(prefetch={prefetch_elapsed:.0f}s process={process_elapsed:.0f}s "
          f"upload={upload_elapsed:.0f}s total={total:.0f}s)")
    return True


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
                [rsync, "-a", "--no-inc-recursive",
                 str(src) + "/", str(dst) + "/"],
                capture_output=True, text=True, timeout=300).returncode
            if rc == 0:
                return True
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
                 nas_output_root: Path):
        self.bm = batch_mgr
        self.pipeline_script = pipeline_script
        self.config_path = config_path
        self.mfa_python = mfa_python
        self.models_dir = models_dir
        self.nas_output_root = nas_output_root

        self.prefetch_queue: queue.Queue[int] = queue.Queue(maxsize=2)
        self.upload_queue: queue.Queue[int] = queue.Queue()  # 无上限，不回堵主管线

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
                print(f"    WARNING: {failed}/{len(copy_tasks)} file copies failed")

            ok = (missing_audio == 0)

            # 写 manifest
            manifest = {"stems": stems, "n_stems": len(stems)}
            (local_ctc / "ctc_ready_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False))

            elapsed = time.time() - t0
            with self.stats_lock:
                self.stats["prefetched"] += 1
                print(f"  [PREFETCH] batch {batch_idx+1} done "
                      f"({elapsed:.1f}s, {len(stems)} stems)")

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
            "--overwrite",
            "--force",
        ]

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

            self.upload_queue.put(batch_idx)
            completed += 1

        self.upload_queue.put(None)
        upload_thread.join(timeout=600)
        prefetch_thread.join(timeout=60)

        all_ok = self.stats["process_fail"] == 0
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
    subprocess.run(cmd)


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

    # ── Pipeline config ──
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--python", type=str, default=None,
                        help="MFA Python path (auto-detect).")
    parser.add_argument("--overwrite", action="store_true")
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

    # --batch-size: CLI > config > default 500
    if args.batch_size is None:
        cfg_val = streaming_cfg.get("batch_size", 0)
        args.batch_size = cfg_val if cfg_val > 0 else 500

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
        run_batch(args)
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
        print(f"Mode:      STREAMING (remote fs → prefetch to {args.local_work})")
        ok = run_single_dataset(
            nas_ctc=str(ctc_dir), nas_audio=str(data_dir),
            nas_output=str(output_dir or (ctc_dir.parent / "mfa_output")),
            config=args.config, local_work=args.local_work,
            batch_size=args.batch_size, limit=args.limit,
            python_path=args.python,
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
) -> bool:
    """Run streaming pipeline for a single dataset.  Returns True on success."""
    # ── Ensure MFA model is pre-extracted before subprocess starts ──
    _ensure_mfa_model_extracted()

    # ── Resolve NAS paths (UNC → Linux translation) ──
    nas_ctc_dir = resolve_input_path(nas_ctc)
    nas_audio_dir = resolve_input_path(nas_audio)
    nas_output_root = resolve_input_path(nas_output)

    if not nas_ctc_dir.exists():
        print(f"ERROR: NAS CTC dir not found: {nas_ctc_dir}")
        print(f"  (translated from: {nas_ctc})")
        return False
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
        stems, layout_map, wav_index = discover_stems(nas_ctc_dir, nas_audio_dir, require_all=True)
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

    # ── Setup batch manager ──
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

    # ── Run ──
    pipeline = StreamingPipeline(
        batch_mgr=batch_mgr,
        pipeline_script=PROJECT_ROOT / "scripts" / "run_pipeline.py",
        config_path=config,
        mfa_python=mfa_python,
        models_dir=models_dir,
        nas_output_root=nas_output_root,
    )
    return pipeline.run()


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


def run_batch(args) -> None:
    """Iterate over all datasets from batch cache with checkpoint/resume support."""
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

    # Auto-adjust MFA num_jobs per dataset when running parallel:
    # Each dataset's MFA align spawns num_jobs processes internally.
    # With N parallel datasets, total MFA workers = N * num_jobs.
    # IMPORTANT: num_jobs > 1 causes a race condition inside MFA's
    # setup_acoustic_model() (os.rename without locking on phones.txt /
    # graphemes.txt).  We cap at 1 — batch-level parallelism already
    # provides enough throughput and avoids the race entirely.
    import os as _os
    cpu_count = _os.cpu_count() or 32
    _mfa_jobs = _cfg.get("mfa", {}).get("num_jobs", 32) if _cfg else 32
    if _mfa_jobs <= 0:
        _mfa_jobs = cpu_count
    # Force num_jobs=1: MFA's setup_acoustic_model() has a TOCTOU race
    # on phones.txt/graphemes.txt rename when multiple workers share a
    # temp directory. Batch-level parallelism (12 workers) already gives
    # enough throughput.
    _effective_mfa_jobs = 1
    if _effective_mfa_jobs < _mfa_jobs:
        print(f"  Note: Reducing MFA num_jobs from {_mfa_jobs} to {_effective_mfa_jobs} "
              f"(parallel={parallel}, CPUs={cpu_count})")
    # Update config for child processes
    if _cfg:
        _cfg.setdefault("mfa", {})["num_jobs"] = _effective_mfa_jobs
    args._effective_mfa_jobs = _effective_mfa_jobs

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
        if not nas_ctc.exists() or not nas_audio.exists():
            print(f"  SKIP {ds_name}: CTC or audio dir not found")
            continue

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
            complete_stems, incomplete_stems, layout_map, wav_index = \
                discover_stems_separated(nas_ctc, nas_audio, require_all=True)
            # Build incomplete wav_index from wav_index (same stems)
            for s in incomplete_stems:
                if s in wav_index:
                    incomplete_wav_index[s] = wav_index[s]

        batch_size_eff = args.batch_size

        # ── Enqueue ctc_ready batches (complete stems) ──
        if complete_stems:
            batches_ctc = [complete_stems[i:i + batch_size_eff]
                           for i in range(0, len(complete_stems), batch_size_eff)]
            for batch_idx, batch_stems in enumerate(batches_ctc):
                all_batches.append(
                    ("ctc_ready", ds, batch_idx, batch_stems, layout_map, wav_index, None))
            total_stems += len(complete_stems)
            print(f"  {ds_name}: {len(complete_stems)} stems → {len(batches_ctc)} ctc_ready batches")

        # ── Enqueue nvrasr_fallback batches (incomplete stems) ──
        if incomplete_stems:
            # Build text index for NVASR reference text
            text_index = build_file_index(nas_audio, ".txt")
            if not text_index:
                print(f"  {ds_name}: WARNING: {len(incomplete_stems)} fallback stems "
                      f"have no reference .txt — NVASR will use ASR-only")
            batches_fb = [incomplete_stems[i:i + batch_size_eff]
                          for i in range(0, len(incomplete_stems), batch_size_eff)]
            for batch_idx, batch_stems in enumerate(batches_fb):
                all_batches.append(
                    ("nvrasr_fallback", ds, batch_idx, batch_stems,
                     layout_map, incomplete_wav_index, text_index))
            total_incomplete += len(incomplete_stems)
            print(f"  {ds_name}: {len(incomplete_stems)} stems → {len(batches_fb)} nvrasr_fallback batches")

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

    # Phase 2: put all batches into shared queue
    batch_queue: _queue.Queue = _queue.Queue()
    global_batch_idx = 0
    for item in all_batches:
        batch_queue.put((global_batch_idx, item))
        global_batch_idx += 1
    total_batches = global_batch_idx

    # Track per-dataset completion: all batches of a dataset must
    # succeed before marking the dataset as DONE in checkpoint.
    ds_batch_tracker: dict[str, dict] = {}  # {ds_name: {"total": N, "done": n, "fail": n}}
    for ds_item in all_batches:
        # ds_item format: (mode, ds, batch_idx, batch_stems, layout_map, wav_index, text_index)
        ds_name = ds_item[1]["name"]
        if ds_name not in ds_batch_tracker:
            ds_batch_tracker[ds_name] = {"total": 0, "done": 0, "fail": 0}
        ds_batch_tracker[ds_name]["total"] += 1

    ok_count = 0
    fail_list: list[str] = []
    ckpt_lock = threading.Lock()

    def worker(worker_id: int) -> tuple[int, list[str]]:
        """Pull individual batches from shared queue."""
        w_ok = 0
        w_fails: list[str] = []
        drive = usable_drives[worker_id % len(usable_drives)]
        local_base = drive / f"worker_{worker_id}"
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
            print(f"\n  [W{worker_id}] [{batch_global_idx+1}/{total_batches}]"
                  f" {batch_label} ({len(batch_stems)} stems){mode_tag}"
                  f" [{remaining} left]")

            ok = run_single_batch(
                ds=ds, batch_idx=batch_idx, batch_stems=batch_stems,
                layout_map=layout_map, wav_index=wav_index,
                local_base=local_base, config=args.config,
                mfa_python=mfa_python, models_dir=models_dir,
                nas_output_root=nas_output_root,
                batch_size=args.batch_size, python_path=args.python,
                mode=batch_mode, text_index=text_index,
            )

            with ckpt_lock:
                tracker = ds_batch_tracker[ds_name]
                if ok:
                    tracker["done"] += 1
                else:
                    tracker["fail"] += 1
                # Dataset complete when all batches done
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

        # Cleanup worker dir
        if local_base.exists():
            shutil.rmtree(local_base, ignore_errors=True)
        return w_ok, w_fails

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(worker, wid) for wid in range(parallel)]
        for fut in concurrent.futures.as_completed(futures):
            w_ok, w_fails = fut.result()
            ok_count += w_ok
            fail_list.extend(w_fails)

    # end parallel

    print(f"\n{'#'*60}")
    print(f"  BATCH COMPLETE: {ok_count}/{len(datasets)} OK")
    if fail_list:
        print(f"  Failed: {', '.join(fail_list)}")
    print(f"{'#'*60}")


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


if __name__ == "__main__":
    main()
