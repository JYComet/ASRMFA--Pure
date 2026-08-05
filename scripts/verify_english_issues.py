#!/usr/bin/env python3
"""
Targeted verification script for English MFA alignment issues.

Checks each of the 6 reported issues against actual TextGrid output files.

Usage:
    python scripts/verify_english_issues.py --dir /mnt/Raw/新版合成英文数据对齐 \
        --en-phones-dir /mnt/nvme3/mfa_workspace/en_phones \
        --sample 400
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (
    is_english_token, is_nvv_token, is_silence, is_punct,
    is_cjk, is_pinyin_syllable, SILENCE_LABELS,
)
from postprocess_textgrids import parse_textgrid, tier_by_name, load_en_phones


def fmt_time(t: float) -> str:
    return f"{t:.3f}"


# ─── Issue ①: Text overlap ───────────────────────────────────────────

def check_overlaps(tg_path: Path) -> list[dict]:
    """Detect overlapping intervals in words tier."""
    tg = parse_textgrid(tg_path)
    words = tier_by_name(tg, "words")
    if words is None:
        return []

    overlaps = []
    intervals = words.intervals
    for i in range(len(intervals) - 1):
        cur = intervals[i]
        nxt = intervals[i + 1]
        if cur.xmax is None or nxt.xmin is None:
            continue
        overlap = cur.xmax - nxt.xmin
        if overlap > 0.0005:  # > 0.5ms
            cur_text = cur.text.strip()
            nxt_text = nxt.text.strip()
            if cur_text and nxt_text and not is_silence(cur_text) and not is_silence(nxt_text):
                overlaps.append({
                    "pos": i,
                    "cur": cur_text, "cur_range": f"{fmt_time(cur.xmin)}-{fmt_time(cur.xmax)}",
                    "nxt": nxt_text, "nxt_range": f"{fmt_time(nxt.xmin)}-{fmt_time(nxt.xmax)}",
                    "overlap_ms": round(overlap * 1000, 1),
                    "en_adjacent": is_english_token(cur_text) or is_english_token(nxt_text),
                })
    return overlaps


# ─── Issue ②: English phone boundary offset ──────────────────────────

def check_en_phone_boundaries(tg_path: Path) -> list[dict]:
    """Check if English token phone boundaries align with word boundaries."""
    tg = parse_textgrid(tg_path)
    words = tier_by_name(tg, "words")
    pp = tier_by_name(tg, "pinyin_phones") or tier_by_name(tg, "phones")
    if words is None or pp is None:
        return []

    offsets = []
    for w_iv in words.intervals:
        if not is_english_token(w_iv.text.strip()):
            continue
        # Find phones within this word
        word_phones = [p for p in pp.intervals
                       if p.xmax > w_iv.xmin + 0.001 and p.xmin < w_iv.xmax - 0.001
                       and not is_silence(p.text) and not is_punct(p.text)]
        if not word_phones:
            continue
        first_phone = word_phones[0]
        last_phone = word_phones[-1]
        start_offset = first_phone.xmin - w_iv.xmin
        end_offset = w_iv.xmax - last_phone.xmax
        if abs(start_offset) > 0.010 or abs(end_offset) > 0.010:
            offsets.append({
                "word": w_iv.text.strip(),
                "word_range": f"{fmt_time(w_iv.xmin)}-{fmt_time(w_iv.xmax)}",
                "phone_start_offset_ms": round(start_offset * 1000, 1),
                "phone_end_offset_ms": round(end_offset * 1000, 1),
                "n_phones": len(word_phones),
            })
    return offsets


# ─── Issue ③: English word insufficient phones ───────────────────────

def check_en_phone_count(tg_path: Path, en_data: list[dict] | None = None) -> list[dict]:
    """Check if English words have fewer phones than dictionary expects."""
    tg = parse_textgrid(tg_path)
    words = tier_by_name(tg, "words")
    pp = tier_by_name(tg, "pinyin_phones") or tier_by_name(tg, "phones")
    if words is None or pp is None:
        return []

    # Try to load CMUdict for expected phone counts
    try:
        from pipeline_utils import _load_cmudict
        cmu = _load_cmudict()
    except Exception:
        cmu = {}

    insufficient = []
    for w_iv in words.intervals:
        w_text = w_iv.text.strip().lower()
        if not is_english_token(w_text):
            continue
        word_phones = [p for p in pp.intervals
                       if p.xmax > w_iv.xmin + 0.001 and p.xmin < w_iv.xmax - 0.001
                       and not is_silence(p.text) and not is_punct(p.text)]
        actual_n = len(word_phones)

        # Check if any phone is a self-reference (the word text itself)
        has_self_ref = any(p.text.strip().lower() == w_text for p in word_phones)

        expected_n = len(cmu.get(w_text.rstrip('012'), [])) if cmu else None

        if actual_n <= 1 or has_self_ref:
            insufficient.append({
                "word": w_text,
                "word_range": f"{fmt_time(w_iv.xmin)}-{fmt_time(w_iv.xmax)}",
                "actual_phones": actual_n,
                "expected_phones": expected_n,
                "has_self_ref": has_self_ref,
                "phone_labels": [p.text for p in word_phones],
            })
    return insufficient


# ─── Issue ④: Abnormally long/short words ────────────────────────────

def check_abnormal_duration(tg_path: Path) -> list[dict]:
    """Detect Chinese words that are abnormally long (>3s) or short (<20ms)."""
    tg = parse_textgrid(tg_path)
    words = tier_by_name(tg, "words")
    if words is None:
        return []

    abnormal = []
    for w_iv in words.intervals:
        if is_silence(w_iv.text) or not w_iv.text.strip():
            continue
        dur = w_iv.duration
        w_text = w_iv.text.strip()
        # Skip English/NVV — different duration expectations
        if is_english_token(w_text) or is_nvv_token(w_text):
            continue
        if dur > 3.0:
            abnormal.append({
                "type": "too_long",
                "word": w_text,
                "duration_s": round(dur, 3),
                "range": f"{fmt_time(w_iv.xmin)}-{fmt_time(w_iv.xmax)}",
            })
        elif dur < 0.020 and is_pinyin_syllable(w_text):
            abnormal.append({
                "type": "too_short",
                "word": w_text,
                "duration_ms": round(dur * 1000, 1),
                "range": f"{fmt_time(w_iv.xmin)}-{fmt_time(w_iv.xmax)}",
            })
    return abnormal


# ─── Issue ⑤: Words tier gaps ────────────────────────────────────────

def check_tier_gaps(tg_path: Path) -> list[dict]:
    """Detect gaps between consecutive content intervals in words tier."""
    tg = parse_textgrid(tg_path)
    words = tier_by_name(tg, "words")
    if words is None:
        return []

    gaps = []
    intervals = words.intervals
    for i in range(len(intervals) - 1):
        cur = intervals[i]
        nxt = intervals[i + 1]
        if cur.xmax is None or nxt.xmin is None:
            continue
        gap = nxt.xmin - cur.xmax
        if gap > 0.005:  # > 5ms
            cur_text = cur.text.strip()
            nxt_text = nxt.text.strip()
            # Only count gaps between non-empty intervals
            if cur_text and nxt_text:
                # Classify the gap type
                cur_is_en = is_english_token(cur_text)
                nxt_is_en = is_english_token(nxt_text)
                cur_is_punct = is_punct(cur_text)
                nxt_is_punct = is_punct(nxt_text)
                cur_is_sil = is_silence(cur_text)
                nxt_is_sil = is_silence(nxt_text)

                if cur_is_sil or nxt_is_sil:
                    continue  # skip silence gaps — they're intentional

                if cur_is_en and nxt_is_en:
                    gap_type = "en→en"
                elif cur_is_en:
                    gap_type = "en→zh"
                elif nxt_is_en:
                    gap_type = "zh→en"
                elif cur_is_punct:
                    gap_type = "punct→zh"
                elif nxt_is_punct:
                    gap_type = "zh→punct"
                elif nxt_text == cur_text:
                    gap_type = "same_word"
                else:
                    gap_type = "zh→zh"

                gaps.append({
                    "type": gap_type,
                    "prev": cur_text,
                    "next": nxt_text,
                    "gap_ms": round(gap * 1000, 1),
                    "pos_range": f"{fmt_time(cur.xmax)}→{fmt_time(nxt.xmin)}",
                })
    return gaps


# ─── Issue ⑥: pinyin_phones discontinuity ────────────────────────────

def check_pp_continuity(tg_path: Path) -> list[dict]:
    """Detect gaps in pinyin_phones tier between consecutive intervals."""
    tg = parse_textgrid(tg_path)
    pp = tier_by_name(tg, "pinyin_phones") or tier_by_name(tg, "phones")
    if pp is None:
        return []

    gaps = []
    intervals = pp.intervals
    for i in range(len(intervals) - 1):
        cur = intervals[i]
        nxt = intervals[i + 1]
        if cur.xmax is None or nxt.xmin is None:
            continue
        gap = nxt.xmin - cur.xmax
        if gap > 0.005:  # > 5ms
            cur_text = cur.text.strip()
            nxt_text = nxt.text.strip()

            # Classify
            if is_silence(cur_text) or is_silence(nxt_text):
                continue  # silence-to-anything gaps are ok (they're labeled)

            # Classify the transition
            cur_is_en = cur_text.startswith("en:") or cur_text.isascii() and cur_text.isalpha() and len(cur_text) <= 3 and cur_text.isupper()
            nxt_is_en = nxt_text.startswith("en:") or nxt_text.isascii() and nxt_text.isalpha() and len(nxt_text) <= 3 and nxt_text.isupper()
            cur_is_punct = len(cur_text) == 1 and not cur_text.isalnum()
            nxt_is_punct = len(nxt_text) == 1 and not nxt_text.isalnum()

            if cur_is_en and nxt_is_en:
                gap_type = "en_ph→en_ph"
            elif cur_is_en:
                gap_type = "en_ph→zh_ph"
            elif nxt_is_en:
                gap_type = "zh_ph→en_ph"
            elif cur_is_punct:
                gap_type = "punct→zh_ph"
            elif nxt_is_punct:
                gap_type = "zh_ph→punct"
            else:
                gap_type = "zh_ph→zh_ph"

            gaps.append({
                "type": gap_type,
                "prev": cur_text,
                "next": nxt_text,
                "gap_ms": round(gap * 1000, 1),
            })
    return gaps


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify English MFA alignment issues in TextGrid output")
    parser.add_argument("--dir", type=Path, required=True,
                        help="Output directory with postprocessed TextGrids")
    parser.add_argument("--en-phones-dir", type=Path, default=None,
                        help="English MFA phones directory")
    parser.add_argument("--sample", type=int, default=0,
                        help="Limit to N random files (0=all)")
    parser.add_argument("--issue", type=str, default="all",
                        choices=["all", "overlap", "en_boundary", "en_phones",
                                 "duration", "gap", "pp_gap"],
                        help="Specific issue to check")
    args = parser.parse_args()

    tg_files = sorted(args.dir.rglob("*.TextGrid"))
    if not tg_files:
        print(f"No TextGrid files found in {args.dir}")
        return 1

    if args.sample > 0 and len(tg_files) > args.sample:
        import random
        random.seed(42)
        tg_files = random.sample(tg_files, args.sample)

    print(f"Checking {len(tg_files)} TextGrid files...")
    print(f"{'='*70}")

    all_issues = defaultdict(list)

    for i, tg_path in enumerate(tg_files):
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(tg_files)}")

        stem = tg_path.stem

        if args.issue in ("all", "overlap"):
            overlaps = check_overlaps(tg_path)
            if overlaps:
                all_issues["overlap"].append({"stem": stem, "count": len(overlaps), "details": overlaps})

        if args.issue in ("all", "en_boundary"):
            offsets = check_en_phone_boundaries(tg_path)
            if offsets:
                all_issues["en_boundary"].append({"stem": stem, "count": len(offsets), "details": offsets})

        if args.issue in ("all", "en_phones"):
            en_data = load_en_phones(stem, args.en_phones_dir)
            insufficient = check_en_phone_count(tg_path, en_data)
            if insufficient:
                all_issues["en_phones"].append({"stem": stem, "count": len(insufficient), "details": insufficient})

        if args.issue in ("all", "duration"):
            abnormal = check_abnormal_duration(tg_path)
            if abnormal:
                all_issues["duration"].append({"stem": stem, "count": len(abnormal), "details": abnormal})

        if args.issue in ("all", "gap"):
            gaps = check_tier_gaps(tg_path)
            if gaps:
                all_issues["gap"].append({"stem": stem, "count": len(gaps), "details": gaps})

        if args.issue in ("all", "pp_gap"):
            pp_gaps = check_pp_continuity(tg_path)
            if pp_gaps:
                all_issues["pp_gap"].append({"stem": stem, "count": len(pp_gaps), "details": pp_gaps})

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Results for {len(tg_files)} files:")
    print(f"{'='*70}")

    for issue_name, items in sorted(all_issues.items()):
        n_files = len(items)
        pct = n_files / len(tg_files) * 100
        total_instances = sum(it["count"] for it in items)
        print(f"\n  [{issue_name}] {n_files} files ({pct:.1f}%) — {total_instances} instances")

        # Show top patterns
        if items:
            # Aggregate by type
            type_counts = Counter()
            for item in items[:20]:  # show first 20 files' details
                for d in item["details"]:
                    if "type" in d:
                        type_counts[d["type"]] += 1
                    elif "en_adjacent" in d:
                        type_counts["en_adjacent" if d["en_adjacent"] else "zh↔zh"] += 1
            if type_counts:
                print(f"         Types: {dict(type_counts.most_common(10))}")

            # Show first 3 examples
            for item in items[:3]:
                stem = item["stem"]
                for d in item["details"][:2]:
                    print(f"         [{stem}] {d}")

    # ── Cross-issue correlations ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("Cross-issue correlation:")
    print(f"{'='*70}")

    gap_stems = {it["stem"] for it in all_issues.get("gap", [])}
    pp_gap_stems = {it["stem"] for it in all_issues.get("pp_gap", [])}
    overlap_stems = {it["stem"] for it in all_issues.get("overlap", [])}
    en_phone_stems = {it["stem"] for it in all_issues.get("en_phones", [])}

    print(f"  Files with BOTH gaps AND pp_gaps: {len(gap_stems & pp_gap_stems)}")
    print(f"  Files with pp_gaps but NO word gaps: {len(pp_gap_stems - gap_stems)}")
    print(f"  Files with word gaps but NO pp_gaps: {len(gap_stems - pp_gap_stems)}")
    print(f"  Files with overlaps AND en_phone issues: {len(overlap_stems & en_phone_stems)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
