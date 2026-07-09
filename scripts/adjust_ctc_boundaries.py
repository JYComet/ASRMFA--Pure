#!/usr/bin/env python3
"""
Pre-MFA CTC anchor boundary adjustment using audio energy analysis.

在 MFA 前用音频能量修正 CTC 锚点边界:
- 句首 / 标点后词首: 检查是否多截取了静音 → 推后 start
- 句尾 / 标点前词尾: 检查是否有语音延续 → 延长 end; 或是否多留静音 → 缩短 end
- 同步调整标点位置

数据流:
  ctc_pretg/ (tokens.jsonl + punct.json + TextGrid + audio)
    -> energy-based boundary adjustment
    -> adjusted ctc_pretg/ (corrected anchors for MFA)
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path


# ===== Audio I/O =====

def load_audio(wav_path: Path) -> tuple[list[float], int]:
    import soundfile as sf
    audio, sr = sf.read(str(wav_path))
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return [float(s) for s in audio], sr


# ===== Energy analysis =====

def frame_rms(audio: list[float], sr: int, frame_ms: float = 10.0,
              hop_ms: float = 5.0) -> list[float]:
    fs = max(1, int(frame_ms / 1000.0 * sr))
    hs = max(1, int(hop_ms / 1000.0 * sr))
    if len(audio) < fs:
        return []
    return [math.sqrt(sum(s * s for s in audio[i:i + fs]) / fs + 1e-12)
            for i in range(0, len(audio) - fs + 1, hs)]


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def global_noise_floor(audio: list[float], sr: int) -> float:
    """Estimate global noise floor from quietest 15% of frames."""
    rms = frame_rms(audio, sr, frame_ms=20.0, hop_ms=10.0)
    if not rms:
        return 1e-6
    s = sorted(rms)
    bot = s[:max(1, int(len(s) * 0.15))]
    return max(median(bot), 1e-8)


# ===== Speech boundary search =====

def search_energy_transition(
    audio: list[float], sr: int,
    anchor_time: float,
    search_fwd_s: float,
    noise_floor: float,
    hop_ms: float = 5.0,
    direction: str = "rise",
) -> float | None:
    """
    Search around anchor_time for an energy transition.

    direction="rise": search forward to find where RMS rises (word start push later)
    direction="fall": search to find where RMS drops below threshold (word end)
      - If anchor is in speech, search forward for the drop
      - If anchor is already in silence, search backward for where speech ended
    """
    hs = max(1, int(hop_ms / 1000.0 * sr))
    fs = max(1, int(0.01 * sr))
    threshold = noise_floor * 3.0
    min_run = max(1, int(0.03 / (hop_ms / 1000.0)))

    if direction == "fall":
        # 先检查锚点处是否已经是静音 (语音可能已在前方结束)
        check_s = int(max(0, anchor_time) * sr)
        check_e = min(len(audio), int((anchor_time + 0.05) * sr))
        if check_e > check_s + fs:
            check_seg = audio[check_s:check_e]
            check_rms = frame_rms(check_seg, sr, frame_ms=10.0, hop_ms=hop_ms)
            if check_rms and all(v < threshold for v in check_rms[:min(3, len(check_rms))]):
                # 锚点已在静音中 → 向左搜索语音终点
                t_start = max(0, anchor_time - 0.4)
                t_end = min(len(audio) / sr, anchor_time + 0.05)
                s = int(t_start * sr); e = int(t_end * sr)
                if e <= s + fs: return None
                seg = audio[s:e]
                rms_vals = frame_rms(seg, sr, frame_ms=10.0, hop_ms=hop_ms)
                n = len(rms_vals)
                if n < 10: return None
                # 从锚点向左找最后一个高于阈值的帧
                anchor_idx = int((anchor_time - t_start) * 1000 / hop_ms)
                anchor_idx = min(anchor_idx, n - 1)
                last_above = None
                for i in range(anchor_idx - min_run, 0, -1):
                    if all(rms_vals[i + j] > threshold for j in range(min_run)):
                        last_above = i + min_run
                        break
                if last_above is not None:
                    t = t_start + last_above * hop_ms / 1000.0
                    if anchor_time - t > 0.03:
                        return t
                return None

    # 默认: 从 anchor 向前搜索
    t_start = anchor_time
    t_end = min(len(audio) / sr, anchor_time + search_fwd_s)
    s = int(t_start * sr)
    e = int(t_end * sr)
    if e <= s + fs:
        return None
    seg = audio[s:e]
    rms_vals = frame_rms(seg, sr, frame_ms=10.0, hop_ms=hop_ms)
    n = len(rms_vals)
    if n < 10:
        return None

    if direction == "rise":
        for i in range(0, n - min_run):
            if all(rms_vals[i + j] > threshold for j in range(min_run)):
                t = t_start + i * hop_ms / 1000.0
                if t > anchor_time + 0.015:
                    return t
    else:  # "fall"
        for i in range(0, n - min_run):
            if all(rms_vals[i + j] < threshold for j in range(min_run)):
                t = t_start + i * hop_ms / 1000.0
                if abs(t - anchor_time) > 0.015:
                    return t
    return None


# ===== Main adjustment =====

def adjust_boundaries(tokens: list[dict], punct: list[dict],
                      audio: list[float], sr: int) -> tuple[list[dict], list[dict], dict]:
    stats = {"start_adj": 0, "end_extend": 0, "end_shorten": 0, "punct_adj": 0}

    def _is_nvv(w: str) -> bool:
        return bool(re.match(r'^[A-Z][A-Z0-9-]*[A-Z0-9]$', w))

    nf = global_noise_floor(audio, sr)

    # --- Part 1: word start boundaries (sentence start / after punctuation) ---
    # 只对紧接标点/句首的普通词做 start 修正, NVV 和非首词不动
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

        onset = search_energy_transition(
            audio, sr, tok["start_s"], search_fwd_s=0.40,
            noise_floor=nf, direction="rise")
        if onset is None or onset <= tok["start_s"] + 0.02:
            continue

        min_dur = 0.04
        new_start = min(onset, tok["end_s"] - min_dur)
        if new_start <= tok["start_s"] + 0.02:
            continue

        old_start = tok["start_s"]
        tok["start_s"] = round(new_start, 3)
        tok["start_ms"] = round(new_start * 1000, 1)

        # 如果 onset 在 end 之后, 推后 end, 但不能越过下一个词的 start
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
    # 只对标点前/句尾的普通词做 end 修正, NVV 不动
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

        offset = search_energy_transition(
            audio, sr, tok["end_s"], search_fwd_s=0.35,
            noise_floor=nf, direction="fall")
        if offset is None or abs(offset - tok["end_s"]) < 0.02:
            continue

        old_end = tok["end_s"]
        new_end = round(offset, 3)

        if new_end > old_end:
            # Speech continues past the word boundary -> extend
            # 不能越过下一个非 NVV 词的 start
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
            # Energy already dropped before word end -> shorten
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
    """Rebuild TextGrid words tier with adjusted timestamps (no regex)."""
    tg_text = orig_tg.read_text(encoding="utf-8")

    # Extract overall duration from xmax line
    m = re.search(r'^xmax = ([\d.]+)', tg_text, re.MULTILINE)
    duration_s = float(m.group(1)) if m else tokens[-1]["end_s"] + 1.0

    # Build combined interval list sorted by start time
    def _is_nvv(w: str) -> bool:
        return bool(re.match(r'^[A-Z][A-Z0-9-]*[A-Z0-9]$', w))

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

    # Write entire TextGrid from scratch (reliable, no format fragility)
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
                out_dir: Path) -> dict:
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
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(set(
        p.stem.replace("_tokens", "")
        for p in args.ctc_dir.glob("*_tokens.jsonl")))
    if args.limit > 0:
        stems = stems[:args.limit]

    totals = {"start_adj": 0, "end_extend": 0, "end_shorten": 0,
              "punct_adj": 0, "files": 0}

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
        print(f"  {stem}: {', '.join(parts) if parts else 'no changes'}")

    print(f"\n{'='*50}")
    print(f"Total: {totals['files']} files")
    print(f"  Start adjustments:   {totals['start_adj']}")
    print(f"  End extended:        {totals['end_extend']}")
    print(f"  End shortened:       {totals['end_shorten']}")
    print(f"  Punct adjustments:   {totals['punct_adj']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
