#!/usr/bin/env python3
"""
Pre-MFA CTC anchor boundary adjustment using audio energy analysis.

在 MFA 前用音频能量修正 CTC 锚点边界:
- 句首 / 标点後词首: 检査是否多截取了静音 → 推後 start
- 句尾 / 标点前词尾: 检査是否有语音延续 → 延长 end; 或是否多留静音 → 缩短 end
- 同步调整标点位置

数据流:
  ctc_pretg/ (tokens.jsonl + punct.json + TextGrid + audio)
    -> energy-based boundary adjustment
    -> adjusted ctc_pretg/ (corrected anchors for MFA)
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audio_energy import (
    load_audio, frame_rms, word_rms,
    noise_floor_from_rms, global_noise_floor,
    speech_onset, speech_offset, median,
)


# ===== Speech boundary search (vectorised) =====

def _search_energy_rise(audio: np.ndarray, sr: int,
                        anchor_time: float, search_fwd_s: float,
                        noise_floor: float) -> float | None:
    """Search forward from *anchor_time* for sustained energy rise."""
    fs = max(1, int(0.005 * sr))          # 5 ms frames
    threshold = noise_floor * 3.0
    min_run = max(1, int(0.03 / 0.005))   # 6 frames @ 5 ms

    s = int(anchor_time * sr)
    e = min(len(audio), int((anchor_time + search_fwd_s) * sr))
    if e <= s + fs * min_run:
        return None

    rms, frame_dur = frame_rms(audio[s:e], sr, frame_ms=5.0)
    if len(rms) < min_run:
        return None

    onset = speech_onset(rms, 0, threshold, min_consecutive=min_run)
    if onset is None:
        return None
    t = anchor_time + onset * frame_dur
    return t if t > anchor_time + 0.015 else None


def _search_energy_fall(audio: np.ndarray, sr: int,
                        anchor_time: float, search_fwd_s: float,
                        noise_floor: float) -> float | None:
    """Search for energy fall: extend word end forward, or shorten backward."""
    fs = max(1, int(0.005 * sr))
    threshold = noise_floor * 3.0
    min_run = max(1, int(0.03 / 0.005))

    # First: check if anchor is already in silence → search backward
    check_s = int(max(0, anchor_time) * sr)
    check_e = min(len(audio), int((anchor_time + 0.05) * sr))
    if check_e > check_s + fs:
        check_rms, _ = frame_rms(audio[check_s:check_e], sr, frame_ms=10.0)
        if len(check_rms) >= 3 and np.all(check_rms[:3] < threshold):
            # Anchor in silence — search backward from anchor
            t_start = max(0, anchor_time - 0.4)
            t_end = min(len(audio) / sr, anchor_time + 0.05)
            s = int(t_start * sr); e = int(t_end * sr)
            if e <= s + fs:
                return None
            rms, frame_dur = frame_rms(audio[s:e], sr, frame_ms=10.0)
            n = len(rms)
            if n < 10:
                return None
            anchor_idx = int((anchor_time - t_start) / frame_dur)
            anchor_idx = min(anchor_idx, n - 1)
            # Search backward from anchor for last above-threshold
            search_end = max(0, anchor_idx - min_run)
            for i in range(search_end, 0, -1):
                if np.all(rms[i:i + min_run] > threshold):
                    t = t_start + (i + min_run) * frame_dur
                    if anchor_time - t > 0.03:
                        return t
                    break
            return None

    # Forward search for energy drop
    s = int(anchor_time * sr)
    e = min(len(audio), int((anchor_time + search_fwd_s) * sr))
    if e <= s + fs * min_run:
        return None

    rms, frame_dur = frame_rms(audio[s:e], sr, frame_ms=5.0)
    if len(rms) < min_run:
        return None

    below = np.where(rms < threshold)[0]
    for i in range(len(below) - min_run + 1):
        if below[i + min_run - 1] - below[i] == min_run - 1:
            t = anchor_time + below[i] * frame_dur
            if abs(t - anchor_time) > 0.015:
                return t
            break
    return None


# ===== Main adjustment =====

def adjust_boundaries(tokens: list[dict], punct: list[dict],
                      audio: np.ndarray, sr: int
                      ) -> tuple[list[dict], list[dict], dict]:
    stats = {"start_adj": 0, "end_extend": 0, "end_shorten": 0, "punct_adj": 0}

    def _is_nvv(w: str) -> bool:
        return bool(re.match(r'^[A-Z][A-Z0-9-]*[A-Z0-9]$', w))

    # Pre-compute global RMS once (reused for noise floor)
    full_rms, rms_frame_dur = frame_rms(audio, sr, frame_ms=20.0)
    nf = noise_floor_from_rms(full_rms, bottom_pct=0.15)

    # --- Part 1: word start boundaries (sentence start / after punctuation) ---
    for idx, tok in enumerate(tokens):
        if _is_nvv(tok["word"]):
            continue
        check = False
        if idx == 0:
            check = True
        else:
            prev = tokens[idx - 1]
            for p in punct:
                if prev["end_s"] - 0.03 <= p["start_s"] <= tok["start_s"] + 0.03:
                    check = True
                    break
        if not check:
            continue

        onset = _search_energy_rise(audio, sr, tok["start_s"], 0.40, nf)
        if onset is None or onset <= tok["start_s"] + 0.02:
            continue

        min_dur = 0.04
        new_start = min(onset, tok["end_s"] - min_dur)
        if new_start <= tok["start_s"] + 0.02:
            continue

        old_start = tok["start_s"]
        tok["start_s"] = round(new_start, 3)
        tok["start_ms"] = round(new_start * 1000, 1)

        if onset >= tok["end_s"]:
            pushed_end = onset + min_dur
            if idx + 1 < len(tokens):
                next_tok = tokens[idx + 1]
                if not _is_nvv(next_tok["word"]):
                    pushed_end = min(pushed_end, next_tok["start_s"] - 0.02)
            if pushed_end > tok["end_s"]:
                tok["end_s"] = round(pushed_end, 3)
                tok["end_ms"] = round(pushed_end * 1000, 1)

        stats["start_adj"] += 1

        for p in punct:
            if abs(p["end_s"] - old_start) < 0.03:
                p["end_s"] = round(new_start, 3)
                p["end_ms"] = round(new_start * 1000, 1)
                stats["punct_adj"] += 1

    # --- Part 2: word end boundaries (sentence end / before punctuation) ---
    for idx, tok in enumerate(tokens):
        if _is_nvv(tok["word"]):
            continue
        check = False
        next_tok = None
        if idx == len(tokens) - 1:
            check = True
        else:
            next_tok = tokens[idx + 1]
            for p in punct:
                if tok["end_s"] - 0.03 <= p["start_s"] <= next_tok["start_s"] + 0.03:
                    check = True
                    break
        if not check:
            continue

        offset = _search_energy_fall(audio, sr, tok["end_s"], 0.35, nf)
        if offset is None or abs(offset - tok["end_s"]) < 0.02:
            continue

        old_end = tok["end_s"]
        new_end = round(offset, 3)

        if new_end > old_end:
            if next_tok and not _is_nvv(next_tok["word"]):
                if new_end >= next_tok["start_s"] - 0.02:
                    new_end = next_tok["start_s"] - 0.02
            if new_end <= old_end + 0.02:
                continue
            tok["end_s"] = new_end
            tok["end_ms"] = round(new_end * 1000, 1)
            stats["end_extend"] += 1
            for p in punct:
                if abs(p["start_s"] - old_end) < 0.03:
                    p["start_s"] = new_end
                    p["start_ms"] = round(new_end * 1000, 1)
                    stats["punct_adj"] += 1
        elif new_end < old_end - 0.04:
            if new_end <= tok["start_s"] + 0.04:
                continue
            tok["end_s"] = new_end
            tok["end_ms"] = round(new_end * 1000, 1)
            stats["end_shorten"] += 1
            for p in punct:
                if abs(p["start_s"] - old_end) < 0.03:
                    p["start_s"] = new_end
                    p["start_ms"] = round(new_end * 1000, 1)
                    stats["punct_adj"] += 1

    return tokens, punct, stats


# ===== File processing =====

def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def rebuild_textgrid(orig_tg: Path, out_tg: Path,
                     tokens: list[dict], punct: list[dict]) -> None:
    tg_text = orig_tg.read_text(encoding="utf-8")
    m = re.search(r'^xmax = ([\d.]+)', tg_text, re.MULTILINE)
    duration_s = float(m.group(1)) if m else tokens[-1]["end_s"] + 1.0

    all_items = []
    for t in tokens:
        all_items.append({"text": t["word"], "start": t["start_s"], "end": t["end_s"]})
    for p in punct:
        all_items.append({"text": p["word"], "start": p["start_s"], "end": p["end_s"]})
    all_items.sort(key=lambda x: x["start"])

    intervals = []
    cursor = 0.0
    for i, item in enumerate(all_items):
        ws = item["start"]
        we = all_items[i + 1]["start"] if i + 1 < len(all_items) else item["end"]
        if ws > cursor + 0.005:
            intervals.append((cursor, ws, ""))
        intervals.append((ws, we, item["text"]))
        cursor = we
    if cursor < duration_s - 0.005:
        intervals.append((cursor, duration_s, ""))

    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        f"xmin = {_fmt(0)} ", f"xmax = {_fmt(duration_s)} ",
        "tiers? <exists> ", "size = 1 ", "item []: ",
        "    item [1]:", '        class = "IntervalTier" ',
        '        name = "words" ',
        f"        xmin = {_fmt(0)} ",
        f"        xmax = {_fmt(duration_s)} ",
        f"        intervals: size = {len(intervals)} ",
    ]
    for k, (s, e, txt) in enumerate(intervals, start=1):
        lines.extend([
            f"        intervals [{k}]:",
            f"            xmin = {_fmt(s)} ",
            f"            xmax = {_fmt(e)} ",
            f"            text = {_quote(txt)} ",
        ])
    out_tg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_one(stem: str, ctc_dir: Path, audio_dir: Path,
                out_dir: Path, blas_num_threads: int = 1) -> dict:
    """Process a single stem — safe for parallel execution.

    Each worker limits its own BLAS threads to *blas_num_threads* so
    that N concurrent processes don't create N × M BLAS threads and
    thrash the CPU caches.  The work is CPU-bound NumPy RMS + energy
    search; with ``OMP_NUM_THREADS=1``, N workers ≈ N× throughput.
    """
    # Pin BLAS threads inside this worker (inherited by child process)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = str(blas_num_threads)

    tokens_path = ctc_dir / f"{stem}_tokens.jsonl"
    punct_path = ctc_dir / f"{stem}_punct.json"
    wav_path = audio_dir / f"{stem}.wav"

    if not tokens_path.exists():
        return {"stem": stem, "error": "no tokens"}

    tokens = [json.loads(l) for l in
              tokens_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    punct = json.loads(punct_path.read_text(encoding="utf-8")) if punct_path.exists() else []
    audio, sr = load_audio(wav_path)

    adj_tokens, adj_punct, stats = adjust_boundaries(tokens, punct, audio, sr)

    # Guard: fix invalid intervals from CTC token overlap (e.g. NVV
    # overlapping adjacent word causes punct end_s < start_s).
    for p in adj_punct:
        if p["end_s"] <= p["start_s"]:
            p["end_s"] = p["start_s"] + 0.060

    # Dedup: remove ellipsis that overlaps with real punctuation (comma,
    # period, etc.).  Boundary adjustment can shift punct times and create
    # new overlaps that didn't exist in the raw CTC output.
    non_ellipsis = [p for p in adj_punct if p["word"] != "…"]
    ellipsis_only = [p for p in adj_punct if p["word"] == "…"]
    if non_ellipsis and ellipsis_only:
        kept_ellipsis = []
        for ep in ellipsis_only:
            overlap = any(
                nep["start_s"] < ep["end_s"] and nep["end_s"] > ep["start_s"]
                for nep in non_ellipsis)
            if not overlap:
                kept_ellipsis.append(ep)
        adj_punct = non_ellipsis + kept_ellipsis

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}_tokens.jsonl", "w", encoding="utf-8") as f:
        for t in adj_tokens:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(out_dir / f"{stem}_punct.json", "w", encoding="utf-8") as f:
        json.dump(adj_punct, f, ensure_ascii=False)

    orig_tg = ctc_dir / f"{stem}.TextGrid"
    if orig_tg.exists():
        rebuild_textgrid(orig_tg, out_dir / f"{stem}.TextGrid",
                        adj_tokens, adj_punct)

    for suffix in [".lab", "_text_cn.txt"]:
        src = ctc_dir / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, out_dir / f"{stem}{suffix}")

    stats["stem"] = stem
    return stats


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(
        description="Pre-MFA CTC anchor adjustment using audio energy")
    parser.add_argument("--ctc-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-stem adjustment details.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(set(
        p.stem.replace("_tokens", "")
        for p in args.ctc_dir.glob("*_tokens.jsonl")))
    if args.limit > 0:
        stems = stems[:args.limit]

    # Skip stems that already have adjusted output (NAS cache from previous run)
    existing = {p.stem for p in args.output_dir.glob("*.TextGrid")}
    if existing:
        new_stems = [s for s in stems if s not in existing]
        skipped = len(stems) - len(new_stems)
        if skipped:
            print(f"  Skipping {skipped}/{len(stems)} stems (already cached in output dir)")
        stems = new_stems

    if not stems:
        print("  All stems already have adjusted output. Nothing to do.")
        return 0

    import multiprocessing as mp
    import platform as _plat

    # ── Executor selection ──
    # Linux/macOS: ProcessPoolExecutor with fork — fast COW, true CPU parallelism.
    # Windows:      ThreadPoolExecutor — avoids per-worker spawn overhead
    #               (each worker would re-import numpy/scipy/soundfile, ~2-5 s).
    #               NumPy energy analysis releases the GIL, so threads are fine.
    if _plat.system() == "Windows":
        from concurrent.futures import ThreadPoolExecutor as _Pool, as_completed
        _use_initializer = False
        _exec_label = "ThreadPool"
    else:
        from concurrent.futures import ProcessPoolExecutor as _Pool, as_completed
        _use_initializer = True
        _exec_label = "ProcessPool"

    # Resource analysis for parallel processing:
    #   CPU  — frame_rms() + energy search are vectorized NumPy (no GIL).
    #          Each worker pins BLAS to 1 thread → N workers = N× throughput.
    #   I/O  — each worker reads a different {stem}.wav + .jsonl; no overlap.
    #   Mem  — each WAV is ~0.3-1 MB float32; N workers × 1 MB is negligible.
    #   Disk — on SMB/CIFS, concurrent reads may saturate network; use
    #          n_workers = min(cpu-1, 8) as a safe upper bound for SMB.
    n_cpu = mp.cpu_count() or 4
    n_workers = min(max(1, n_cpu - 1), len(stems))
    # Auto-detect local vs network filesystem for worker count
    # NVMe paths (pipeline local work dirs) → higher parallelism
    # SMB/CIFS/NFS paths → conservative cap to avoid network saturation
    _audio_path = str(args.audio_dir)
    _on_local = _audio_path.startswith("/mnt/nvme") or _audio_path.startswith("/dev/nvme")
    if _on_local:
        n_workers = min(n_workers, 32)   # local NVMe → up to 32 workers per batch
    else:
        n_workers = min(n_workers, 8)    # network FS → safe cap
    totals = {"start_adj": 0, "end_extend": 0, "end_shorten": 0,
              "punct_adj": 0, "files": 0}

    if n_workers <= 1 or len(stems) <= 2:
        # Sequential for tiny jobs — avoid process overhead
        for stem in stems:
            s = process_one(stem, args.ctc_dir, args.audio_dir, args.output_dir)
            totals["files"] += 1
            parts = []
            if s.get("start_adj", 0) > 0:
                parts.append(f"startx{s['start_adj']}")
            if s.get("end_extend", 0) > 0:
                parts.append(f"extendx{s['end_extend']}")
            if s.get("end_shorten", 0) > 0:
                parts.append(f"shortenx{s['end_shorten']}")
            for k in ["start_adj", "end_extend", "end_shorten", "punct_adj"]:
                totals[k] += s.get(k, 0)
            if args.verbose:
                print(f"  {stem}: {', '.join(parts) if parts else 'no changes'}")
            elif totals["files"] % 100 == 0:
                print(f"  ... {totals['files']}/{len(stems)} files adjusted", flush=True)
    else:
        print(f"  Parallel mode: {n_workers} workers for {len(stems)} files ({_exec_label}, BLAS=1 per worker)")
        with _Pool(max_workers=n_workers) as pool:
            futures = {
                pool.submit(process_one, stem, args.ctc_dir, args.audio_dir,
                            args.output_dir, 1): stem
                for stem in stems
            }
            for fut in as_completed(futures):
                stem = futures[fut]
                try:
                    s = fut.result()
                except Exception as e:
                    print(f"  FAIL {stem}: {e}")
                    continue
                totals["files"] += 1
                parts = []
                if s.get("start_adj", 0) > 0:
                    parts.append(f"startx{s['start_adj']}")
                if s.get("end_extend", 0) > 0:
                    parts.append(f"extendx{s['end_extend']}")
                if s.get("end_shorten", 0) > 0:
                    parts.append(f"shortenx{s['end_shorten']}")
                for k in ["start_adj", "end_extend", "end_shorten", "punct_adj"]:
                    totals[k] += s.get(k, 0)
                if args.verbose:
                    print(f"  {stem}: {', '.join(parts) if parts else 'no changes'}")
                # Progress heartbeat — every 100 files
                if totals["files"] % 100 == 0:
                    print(f"  ... {totals['files']}/{len(stems)} files adjusted", flush=True)

    print(f"\n{'='*50}")
    print(f"Total: {totals['files']} files")
    print(f"  Start adjustments:   {totals['start_adj']}")
    print(f"  End extended:        {totals['end_extend']}")
    print(f"  End shortened:       {totals['end_shorten']}")
    print(f"  Punct adjustments:   {totals['punct_adj']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
