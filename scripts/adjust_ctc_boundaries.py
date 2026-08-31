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
from pipeline_utils import (
    CTC_RAW_MANIFEST_NAME, CTC_WORK_RECEIPT_NAME,
    write_ctc_work_receipt, validate_ctc_work_receipt,
    find_wav,
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

    def _canonical_span(tok: dict) -> tuple[float, float] | None:
        unit = tok.get("canonical_unit")
        span = tok.get("canonical_span")
        if not isinstance(unit, dict) or not isinstance(span, (list, tuple)):
            return None
        if len(span) != 2 or span != list(unit.get("canonical_span", ())):
            return None
        return float(span[0]), float(span[1])

    def _write_start(tok: dict, value: float, source: str = "energy_start") -> None:
        canonical = _canonical_span(tok)
        if canonical is not None:
            value = canonical[0]
            if float(tok["end_s"]) < canonical[1]:
                tok["end_s"] = canonical[1]
                tok["end_ms"] = round(canonical[1] * 1000, 1)
        else:
            value = round(float(value), 3)
        tok["start_s"] = value
        tok["start_ms"] = round(value * 1000, 1)
        if canonical is not None:
            tok["processed_ctc_span"] = [value, float(tok["end_s"])]
            tok["processed_ctc_boundary_source"] = source

    def _write_end(tok: dict, value: float, source: str = "energy_end") -> None:
        canonical = _canonical_span(tok)
        if canonical is not None:
            value = max(float(value), canonical[1])
        else:
            value = round(float(value), 3)
        tok["end_s"] = value
        tok["end_ms"] = round(value * 1000, 1)
        if canonical is not None:
            tok["processed_ctc_span"] = [float(tok["start_s"]), value]
            tok["processed_ctc_boundary_source"] = source

    def _hard_boundary(tok: dict, next_tok: dict | None) -> float | None:
        """Return punctuation/long-pause boundary that energy must not cross."""
        candidates = [p["start_s"] for p in punct
                      if tok["end_s"] - 0.03 <= p["start_s"]
                      and (next_tok is None or p["start_s"] <= next_tok["start_s"] + 0.03)]
        if next_tok is not None and next_tok["start_s"] - tok["end_s"] >= 0.2 - 1e-9:
            candidates.append(next_tok["start_s"])
        return min(candidates) if candidates else None

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
        _write_start(tok, new_start)

        if onset >= tok["end_s"]:
            pushed_end = onset + min_dur
            if idx + 1 < len(tokens):
                next_tok = tokens[idx + 1]
                # Always clamp to next token's start regardless of type.
                # NVV boundaries are acoustically unreliable, but the NVV
                # still occupies time — content word cannot cross into it.
                pushed_end = min(pushed_end, next_tok["start_s"] - 0.02)
            if pushed_end > tok["end_s"]:
                _write_end(tok, pushed_end)

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
            check = _hard_boundary(tok, next_tok) is not None
        if not check:
            continue

        offset = _search_energy_fall(audio, sr, tok["end_s"], 0.35, nf)
        if offset is None or abs(offset - tok["end_s"]) < 0.02:
            continue

        old_end = tok["end_s"]
        new_end = round(offset, 3)
        hard_boundary = _hard_boundary(tok, next_tok)
        if hard_boundary is not None:
            new_end = min(new_end, hard_boundary)

        if new_end > old_end:
            if next_tok:
                # Always clamp to next token's start regardless of type.
                if new_end >= next_tok["start_s"] - 0.02:
                    new_end = next_tok["start_s"] - 0.02
            if new_end <= old_end + 0.02:
                continue
            _write_end(tok, new_end, "energy_end_hard_boundary"
                       if hard_boundary is not None else "energy_end")
            stats["end_extend"] += 1
            for p in punct:
                if abs(p["start_s"] - old_end) < 0.03:
                    p["start_s"] = new_end
                    p["start_ms"] = round(new_end * 1000, 1)
                    stats["punct_adj"] += 1
        elif new_end < old_end - 0.04:
            if new_end <= tok["start_s"] + 0.04:
                continue
            # Clamp to previous NVV's end: when the previous token is NVV,
            # content word's end should not retreat past the NVV's interval.
            if idx > 0 and _is_nvv(tokens[idx - 1]["word"]):
                prev_nvv_end = tokens[idx - 1]["end_s"]
                if new_end < prev_nvv_end + 0.02:
                    new_end = prev_nvv_end + 0.02
            _write_end(tok, new_end, "energy_end_hard_boundary"
                       if hard_boundary is not None else "energy_end")
            stats["end_shorten"] += 1
            for p in punct:
                if abs(p["start_s"] - old_end) < 0.03:
                    p["start_s"] = new_end
                    p["start_ms"] = round(new_end * 1000, 1)
                    stats["punct_adj"] += 1

    return tokens, punct, stats


def _protect_processed_english_geometry(tokens: list[dict]) -> None:
    """Restore immutable canonical floors after optional energy refinement."""
    for token in tokens:
        unit = token.get("canonical_unit")
        span = token.get("canonical_span")
        unit_span = unit.get("canonical_span") if isinstance(unit, dict) else None
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or span != list(unit_span or ())):
            continue
        canonical_start, canonical_end = float(span[0]), float(span[1])
        current_end = max(float(token.get("end_s", canonical_end)), canonical_end)
        token["start_s"] = canonical_start
        token["end_s"] = current_end
        token["start_ms"] = round(canonical_start * 1000, 1)
        token["end_ms"] = round(current_end * 1000, 1)
        token["processed_ctc_span"] = [canonical_start, current_end]
        if current_end > canonical_end:
            token.setdefault("processed_ctc_boundary_source", "energy_end")
        else:
            token["processed_ctc_boundary_source"] = (
                token.get("processed_ctc_boundary_source") or "canonical_end_floor")


def _record_adjusted_candidate_spans(tokens: list[dict],
                                     punct: list[dict]) -> None:
    """Seal adjusted spans while leaving candidate raw coordinates untouched."""
    for row in [*tokens, *punct]:
        if row.get("provenance_schema") != "nvasr-candidate-provenance-v1":
            continue
        start = row.get("start_s")
        end = row.get("end_s")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            continue
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            continue
        if not math.isfinite(float(start)) or not math.isfinite(float(end)):
            continue
        row["adjusted_span"] = [float(start), float(end)]
        row["adjusted_mapping_outcome"] = "unique"


def _processed_geometry_cache_complete(ctc_dir: Path,
                                       stems: set[str]) -> bool:
    """Return whether an adjusted cache has derived spans for all authority rows."""
    for stem in stems:
        token_path = ctc_dir / f"{stem}_tokens.jsonl"
        if not token_path.is_file() or token_path.is_symlink():
            return False
        try:
            rows = [json.loads(line) for line in
                    token_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        for row in rows:
            if not isinstance(row, dict) or not isinstance(
                    row.get("canonical_unit"), dict):
                continue
            span = row.get("processed_ctc_span")
            source = row.get("processed_ctc_boundary_source")
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, (int, float))
                               and not isinstance(value, bool)
                               and math.isfinite(float(value))
                               for value in span)
                    or span[1] <= span[0]
                    or not isinstance(source, str) or not source):
                return False
    return True


# ===== File processing =====

def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _read_pause_intervals(textgrid_path: Path) -> list[dict]:
    """Read explicit raw CTC pauses for processed-boundary ownership."""
    if not textgrid_path.is_file():
        return []
    text = textgrid_path.read_text(encoding="utf-8")
    pauses: list[dict] = []
    current_name = ""
    xmin = xmax = None
    in_interval = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("name = "):
            current_name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("intervals ["):
            in_interval = True
            xmin = xmax = None
        elif in_interval and line.startswith("xmin = "):
            xmin = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("xmax = "):
            xmax = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("text = "):
            label = line.split("=", 1)[1].strip().strip('"')
            if (current_name == "pauses" and label.strip() and xmin is not None
                    and xmax is not None and xmax > xmin):
                pauses.append({"start_s": xmin, "end_s": xmax})
            in_interval = False
    return pauses


def rebuild_textgrid(orig_tg: Path, out_tg: Path,
                     tokens: list[dict], punct: list[dict]) -> None:
    tg_text = orig_tg.read_text(encoding="utf-8")
    m = re.search(r'^xmax = ([\d.]+)', tg_text, re.MULTILINE)
    duration_s = float(m.group(1)) if m else tokens[-1]["end_s"] + 1.0

    # Preserve the explicit CTC pauses tier when present.  The words tier is
    # rebuilt from processed token geometry, but long pauses remain evidence
    # and must not disappear during adjustment.
    pause_intervals: list[tuple[float, float, str]] = []
    current_name = ""
    pending_start = pending_end = None
    in_interval = False
    for raw in tg_text.splitlines():
        line = raw.strip()
        if line.startswith('name = '):
            current_name = line.split('=', 1)[1].strip().strip('"')
        elif line.startswith('intervals ['):
            in_interval = True
            pending_start = pending_end = None
        elif in_interval and line.startswith('xmin = '):
            pending_start = float(line.split('=', 1)[1].strip())
        elif in_interval and line.startswith('xmax = '):
            pending_end = float(line.split('=', 1)[1].strip())
        elif in_interval and line.startswith('text = '):
            label = line.split('=', 1)[1].strip().strip('"')
            if (current_name == "pauses" and label.strip()
                    and pending_start is not None
                    and pending_end is not None and pending_end > pending_start):
                pause_intervals.append((pending_start, pending_end, label))
            in_interval = False

    events = [{"start": t["start_s"], "kind": "word"} for t in tokens]
    events += [{"start": p["start_s"], "kind": "punct"} for p in punct]
    events.sort(key=lambda x: (x["start"], x["kind"] == "word"))
    next_event_starts = [event["start"] for event in events]
    intervals = []
    cursor = 0.0
    for token in sorted(tokens, key=lambda item: item["start_s"]):
        ws = token["start_s"]
        following = [start for start in next_event_starts if start > ws + 0.001]
        we = min(token["end_s"], following[0]) if following else token["end_s"]
        we = max(ws, min(we, duration_s))
        if ws > cursor + 0.005:
            intervals.append((cursor, ws, ""))
        intervals.append((ws, we, token["word"]))
        cursor = we
    if cursor < duration_s - 0.005:
        intervals.append((cursor, duration_s, ""))

    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        f"xmin = {_fmt(0)} ", f"xmax = {_fmt(duration_s)} ",
        "tiers? <exists> ", f"size = {2 if pause_intervals else 1} ", "item []: ",
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
    if pause_intervals:
        lines.extend([
            "    item [2]:", '        class = "IntervalTier" ',
            '        name = "pauses" ', f"        xmin = {_fmt(0)} ",
            f"        xmax = {_fmt(duration_s)} ",
            f"        intervals: size = {len(pause_intervals)} ",
        ])
        for k, (s, e, txt) in enumerate(pause_intervals, start=1):
            lines.extend([
                f"        intervals [{k}]:",
                f"            xmin = {_fmt(s)} ",
                f"            xmax = {_fmt(e)} ",
                f"            text = {_quote(txt)} ",
            ])
    out_tg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_one(stem: str, ctc_dir: Path, audio_dir: Path,
                out_dir: Path, blas_num_threads: int = 1,
                apply_energy: bool = True) -> dict:
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
    wav_path = find_wav(audio_dir, stem)
    if wav_path is None:
        return {"stem": stem, "error": "no wav"}

    if not tokens_path.exists():
        return {"stem": stem, "error": "no tokens"}

    tokens = [json.loads(l) for l in
              tokens_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    punct = json.loads(punct_path.read_text(encoding="utf-8")) if punct_path.exists() else []
    if apply_energy:
        # Energy refinement needs samples.  Geometry-only mode must not load
        # every WAV merely to obtain its duration; the final English VAD, if
        # needed, is the only audio read in that mode.
        audio, sr = load_audio(wav_path)
        duration_s = len(audio) / sr
    else:
        import soundfile as sf
        audio = None
        sr = None
        duration_s = float(sf.info(str(wav_path)).duration)

    orig_tg = ctc_dir / f"{stem}.TextGrid"
    pauses = _read_pause_intervals(orig_tg)
    try:
        # Raw canonical spans are created by ctc_prealign and are immutable.
        # Resolve the mutable processed geometry only in the copied work root.
        # Keep this stage runnable in the MFA environment, which deliberately
        # does not install torch.  The CTC producer owns raw evidence, while
        # this torch-free helper owns the mutable processed geometry.
        from ctc_processed_geometry import resolve_processed_english_spans
        resolve_processed_english_spans(
            tokens, punct, pauses, duration_s, wav_path)
    except (OSError, ValueError, TypeError) as exc:
        return {"stem": stem, "error": f"processed English span: {exc}"}

    if apply_energy:
        adj_tokens, adj_punct, stats = adjust_boundaries(tokens, punct, audio, sr)
    else:
        adj_tokens, adj_punct = tokens, punct
        stats = {"start_adj": 0, "end_extend": 0, "end_shorten": 0, "punct_adj": 0}
    _protect_processed_english_geometry(adj_tokens)
    _record_adjusted_candidate_spans(adj_tokens, adj_punct)

    # Guard: fix invalid intervals where Part 1 and Part 2 independently
    # adjusted punct start_s / end_s into a crossing state (NVV between
    # content words obscures the punct position check).
    # Trust Part 2's start_s (acoustic evidence from energy fall) over
    # Part 1's end_s (which may have been pulled backward by NVV adjacency)
    # and shrink from the left rather than blindly extending the right.
    for p in adj_punct:
        if p["end_s"] <= p["start_s"]:
            p["start_s"] = round(p["end_s"] - 0.030, 3)
            if p["start_s"] < 0:
                p["start_s"] = 0.0
                p["end_s"] = 0.030

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

    if orig_tg.exists():
        rebuild_textgrid(orig_tg, out_dir / f"{stem}.TextGrid",
                        adj_tokens, adj_punct)

    for suffix in [".lab", "_text_cn.txt", "_text_raw.txt", "_ref.txt"]:
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
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run adjustment even if output files exist.")
    parser.add_argument("--raw-manifest", type=Path, default=None,
                        help="Immutable raw manifest used to bind the work receipt.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-stem adjustment details.")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Create processed CTC geometry without energy refinement.")
    args = parser.parse_args()

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    raw_manifest = args.raw_manifest or (args.ctc_dir / CTC_RAW_MANIFEST_NAME)
    if not raw_manifest.is_file() or raw_manifest.is_symlink():
        print(f"ERROR: raw manifest unavailable: {raw_manifest}")
        return 1

    # All writes land in a fresh directory.  The existing work tree is kept
    # intact until every selected stem has completed and the full namespace
    # has been verified.
    staging_dir = args.output_dir.parent / (
        f".{args.output_dir.name}.adjust-staging.{os.getpid()}")
    if staging_dir.exists() or staging_dir.is_symlink():
        print(f"ERROR: adjustment staging collision: {staging_dir}")
        return 1
    staging_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the complete current work namespace in staging.  This makes a
    # limited adjustment an atomic replacement of the whole work tree rather
    # than a partial tree that could lose unselected stems.
    for source in args.ctc_dir.iterdir():
        if source.name in {CTC_WORK_RECEIPT_NAME, CTC_RAW_MANIFEST_NAME}:
            continue
        if source.is_file() and not source.is_symlink():
            shutil.copyfile(source, staging_dir / source.name)
    # Retain already-adjusted stems when this run skips them.  The raw CTC
    # tree is the fallback namespace; the current output overlays it.
    if args.output_dir.is_dir() and not args.output_dir.is_symlink():
        for source in args.output_dir.iterdir():
            if source.name in {CTC_WORK_RECEIPT_NAME, CTC_RAW_MANIFEST_NAME}:
                continue
            if source.is_file() and not source.is_symlink():
                shutil.copyfile(source, staging_dir / source.name)
    output_dir_for_workers = staging_dir

    stems = sorted(set(
        p.stem.replace("_tokens", "")
        for p in args.ctc_dir.glob("*_tokens.jsonl")))
    if args.limit > 0:
        stems = stems[:args.limit]

    # Skip stems that already have adjusted output (unless --overwrite)
    if not args.overwrite:
        required_suffixes = (".TextGrid", ".lab", "_tokens.jsonl",
                             "_punct.json", "_text_cn.txt", "_text_raw.txt")
        existing = {
            stem for stem in stems
            if all((args.output_dir / f"{stem}{suffix}").is_file()
                   and not (args.output_dir / f"{stem}{suffix}").is_symlink()
                   for suffix in required_suffixes)
            and _processed_geometry_cache_complete(args.output_dir, {stem})
        }
        if existing:
            new_stems = [s for s in stems if s not in existing]
            skipped = len(stems) - len(new_stems)
            if skipped:
                print(f"  Skipping {skipped}/{len(stems)} stems (already cached in output dir)")
            stems = new_stems

    if not stems:
        print("  All stems already have adjusted output. Nothing to do.")
        shutil.rmtree(staging_dir, ignore_errors=True)
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
    expected = set(stems)
    completed_stems: set[str] = set()
    worker_errors: list[tuple[str, str]] = []

    if n_workers <= 1 or len(stems) <= 2:
        # Sequential for tiny jobs — avoid process overhead
        for stem in stems:
            try:
                s = process_one(stem, args.ctc_dir, args.audio_dir,
                                output_dir_for_workers,
                                apply_energy=not args.geometry_only)
            except Exception as exc:
                worker_errors.append((stem, f"worker exception: {exc}"))
                print(f"  FAIL {stem}: {exc}")
                continue
            if s.get("error"):
                worker_errors.append((stem, str(s["error"])))
                print(f"  FAIL {stem}: {s['error']}")
                continue
            completed_stems.add(stem)
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
                            output_dir_for_workers, 1,
                            not args.geometry_only): stem
                for stem in stems
            }
            for fut in as_completed(futures):
                stem = futures[fut]
                try:
                    s = fut.result()
                except Exception as e:
                    worker_errors.append((stem, f"worker exception: {e}"))
                    print(f"  FAIL {stem}: {e}")
                    continue
                if s.get("error"):
                    worker_errors.append((stem, str(s["error"])))
                    print(f"  FAIL {stem}: {s['error']}")
                    continue
                completed_stems.add(stem)
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

    if worker_errors or completed_stems != expected:
        missing = sorted(expected - completed_stems)
        print(f"ERROR: adjustment workers failed; missing={missing}")
        print(f"Staging preserved: {staging_dir}")
        return 1

    print(f"\n{'='*50}")
    print(f"Total: {totals['files']} files")
    print(f"  Start adjustments:   {totals['start_adj']}")
    print(f"  End extended:        {totals['end_extend']}")
    print(f"  End shortened:       {totals['end_shorten']}")
    print(f"  Punct adjustments:   {totals['punct_adj']}")
    actual = {p.stem for p in staging_dir.glob("*.TextGrid")}
    if not expected <= actual:
        print(f"ERROR: adjustment staging stem mismatch: missing={len(expected - actual)}")
        print(f"Staging preserved: {staging_dir}")
        return 1
    for stem in sorted(expected):
        for suffix in (".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
                       "_text_cn.txt", "_text_raw.txt"):
            path = staging_dir / f"{stem}{suffix}"
            if not path.is_file() or path.is_symlink():
                print(f"ERROR: adjustment staging artifact missing: {path}")
                print(f"Staging preserved: {staging_dir}")
                return 1

    # Capture lineage before moving the old work tree aside.  The old receipt
    # is the source of truth for all prior normalization/RIA/English stages.
    prior_ledger = []
    prior_receipt = args.ctc_dir / CTC_WORK_RECEIPT_NAME
    if prior_receipt.is_file() and not prior_receipt.is_symlink():
        try:
            prior_ledger = json.loads(prior_receipt.read_text(encoding="utf-8")).get(
                "transform_ledger", [])
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            print("ERROR: existing CTC work receipt is unreadable")
            print(f"Staging preserved: {staging_dir}")
            return 1
    ledger = [*prior_ledger, {
        "stage": "adjust", "status": "completed",
        "input_root": str(args.ctc_dir.resolve()),
        "atomic_staging": str(staging_dir),
    }]
    staged_receipt = write_ctc_work_receipt(
        staging_dir, raw_manifest, transform_ledger=ledger,
        work_root=args.output_dir)
    if validate_ctc_work_receipt(staging_dir, raw_manifest, staged_receipt):
        print("ERROR: adjustment staging work receipt failed validation")
        print(f"Staging preserved: {staging_dir}")
        return 1

    # Atomic directory publication.  An old work tree is renamed aside, never
    # deleted, so failed/restarted runs retain recoverable evidence.
    previous = None
    if args.output_dir.exists() or args.output_dir.is_symlink():
        previous = args.output_dir.with_name(
            f"{args.output_dir.name}.previous.{os.getpid()}")
        if previous.exists() or previous.is_symlink():
            print(f"ERROR: adjustment backup collision: {previous}")
            print(f"Staging preserved: {staging_dir}")
            return 1
        os.replace(args.output_dir, previous)
    os.replace(staging_dir, args.output_dir)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
