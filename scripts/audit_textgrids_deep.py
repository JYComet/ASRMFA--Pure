#!/usr/bin/env python3
"""
Deep analysis of MFA TextGrid alignment — detects:
1. Hanzi-to-pinyin displacement errors (character pinyin doesn't match)
2. Position shift patterns (cascading displacement)
3. STT error propagation affecting alignment
4. Phone-level anomalies per word
"""

import os, re, sys, json
from collections import Counter, defaultdict
from pypinyin import lazy_pinyin, Style

SP_TOKEN_PAT = re.compile(r'^<sp\d+>$')
PUNCT_ONLY = re.compile(r'^[，。！？、；：""''…\-,.!?;:\'"\s\n\t\r]+$')
CHINESE_CHAR = re.compile(r'[一-鿿]')

def parse_textgrid(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return None

    result = {"filename": os.path.basename(filepath), "tiers": {}}
    items = re.split(r'\n\s*item\s*\[\d+\]:', content)

    for block in items[1:]:
        if 'IntervalTier' not in block:
            continue
        name_m = re.search(r'name\s*=\s*"(.*?)"', block)
        if not name_m:
            continue
        name = name_m.group(1)

        intervals = []
        interval_blocks = re.split(r'\n\s*intervals\s*\[\d+\]:', block)
        for ib in interval_blocks[1:]:
            xmin_m = re.search(r'xmin\s*=\s*([\d.]+)', ib)
            xmax_m = re.search(r'xmax\s*=\s*([\d.]+)', ib)
            text_m = re.search(r'text\s*=\s*"(.*?)"', ib)
            if xmin_m and xmax_m:
                intervals.append({
                    "xmin": float(xmin_m.group(1)),
                    "xmax": float(xmax_m.group(1)),
                    "text": text_m.group(1) if text_m else "",
                })

        result["tiers"][name] = intervals

    result["xmin"] = float(re.search(r'xmin\s*=\s*([\d.]+)', items[0]).group(1))
    result["xmax"] = float(re.search(r'xmax\s*=\s*([\d.]+)', items[0]).group(1))
    return result


def get_expected_pinyin(char):
    """Get expected pinyin (without tone) for a Chinese character."""
    if not CHINESE_CHAR.match(char):
        return None
    # For multi-char hanzi, get pinyin for each
    py_list = lazy_pinyin(char, style=Style.TONE3, neutral_tone_with_five=True)
    return py_list[0] if py_list else None


def normalize_pinyin(py):
    """Normalize pinyin for comparison: strip tone, lowercase."""
    if not py:
        return ""
    # Strip tone number at end
    py = re.sub(r'\d+$', '', py.strip().lower())
    return py


def analyze_file(filepath):
    """Deep analysis of a single file, returning issues found."""
    fname = os.path.basename(filepath)
    tg = parse_textgrid(filepath)
    if tg is None:
        return {"file": fname, "error": "parse_failed"}

    hanzi_tier = tg["tiers"].get("hanzi", [])
    words_tier = tg["tiers"].get("words", [])
    phones_tier = tg["tiers"].get("pinyin_phones", [])
    raw_text_tier = tg["tiers"].get("raw_text", [])

    # Build position-indexed data
    positions = []
    for i, (h, w) in enumerate(zip(hanzi_tier, words_tier)):
        h_text = h["text"].strip()
        w_text = w["text"].strip()
        is_chinese = bool(CHINESE_CHAR.search(h_text))
        is_punct = bool(PUNCT_ONLY.match(h_text))
        is_sp = bool(SP_TOKEN_PAT.match(h_text))

        expected_py = None
        expected_py_norm = None
        actual_py_norm = None

        if is_chinese:
            # Get expected pinyin for the Chinese character(s)
            expected_py = get_expected_pinyin(h_text)
            expected_py_norm = normalize_pinyin(expected_py)
            actual_py_norm = normalize_pinyin(w_text)

        positions.append({
            "idx": i,
            "hanzi": h_text,
            "word_pinyin": w_text,
            "is_chinese": is_chinese,
            "is_punct": is_punct,
            "is_sp": is_sp,
            "expected_py": expected_py,
            "expected_py_norm": expected_py_norm,
            "actual_py_norm": actual_py_norm,
            "h_xmin": h["xmin"],
            "h_xmax": h["xmax"],
            "match": expected_py_norm == actual_py_norm if is_chinese else None,
        })

    # ── Analysis ──

    # 1. Character-to-pinyin mismatch detection
    chinese_positions = [p for p in positions if p["is_chinese"]]
    total_chinese = len(chinese_positions)
    mismatches = [p for p in chinese_positions if p["match"] is False]
    match_count = sum(1 for p in chinese_positions if p["match"] is True)

    # 2. Displacement detection — find consecutive mismatch runs
    displacement_runs = []
    current_run = []
    for p in chinese_positions:
        if p["match"] is False:
            current_run.append(p)
        else:
            if len(current_run) >= 3:  # at least 3 consecutive mismatches = likely displacement
                displacement_runs.append(list(current_run))
            current_run = []
    if len(current_run) >= 3:
        displacement_runs.append(list(current_run))

    # 3. Try to characterize the displacement: for each run, check if
    # shifting left or right by N positions would fix it
    shift_analysis = []
    for run in displacement_runs:
        run_indices = [p["idx"] for p in run]
        # Check if the expected pinyin matches actual words at a shifted position
        # e.g., hanzi[i]'s expected pinyin matches word[i+1] → right-shift
        right_shift_matches = 0
        left_shift_matches = 0

        for p in run:
            i = p["idx"]
            # Check right shift: hanzi[i]'s expected pinyin == word[i+1]'s actual?
            if i + 1 < len(positions) and positions[i+1]["is_chinese"]:
                if p["expected_py_norm"] == positions[i+1]["actual_py_norm"]:
                    right_shift_matches += 1
            # Check left shift
            if i - 1 >= 0 and positions[i-1]["is_chinese"]:
                if p["expected_py_norm"] == positions[i-1]["actual_py_norm"]:
                    left_shift_matches += 1

        run_len = len(run)
        if right_shift_matches > run_len * 0.5:
            shift_type = "RIGHT (+1 position)"
        elif left_shift_matches > run_len * 0.5:
            shift_type = "LEFT (-1 position)"
        else:
            shift_type = "SCRAMBLED"

        start_hanzi = ''.join(p["hanzi"] for p in run[:5])
        start_expected = '/'.join(p["expected_py_norm"] for p in run[:5])
        start_actual = '/'.join(p["actual_py_norm"] for p in run[:5])

        shift_analysis.append({
            "start_idx": run[0]["idx"],
            "end_idx": run[-1]["idx"],
            "length": run_len,
            "shift_type": shift_type,
            "sample_hanzi": start_hanzi,
            "sample_expected": start_expected,
            "sample_actual": start_actual,
        })

    # 4. Find mismatched English/foreign word handling
    # Check if words like "AP", "Kpop", "Macbook" get proper handling
    english_like = []
    for p in positions:
        if not p["is_chinese"] and not p["is_punct"] and not p["is_sp"]:
            eng_match = re.match(r'^[A-Za-z0-9]+$', p["hanzi"])
            if eng_match and len(p["hanzi"]) > 1:
                # Check if word tier also has the same English
                if p["hanzi"].upper() != p["word_pinyin"].upper():
                    english_like.append({
                        "hanzi": p["hanzi"],
                        "word": p["word_pinyin"],
                        "idx": p["idx"],
                    })

    # 5. Check phone-level issues
    # For each word, count phones and check for anomalies
    phone_anomalies = []
    for w in words_tier:
        w_text = w["text"].strip()
        if not w_text or SP_TOKEN_PAT.match(w_text) or PUNCT_ONLY.match(w_text):
            continue
        # Find matching phone intervals
        w_phones = [p for p in phones_tier
                    if p["xmin"] >= w["xmin"] - 0.001
                    and p["xmax"] <= w["xmax"] + 0.001
                    and p["text"].strip()
                    and not PUNCT_ONLY.match(p["text"])]

        n_phones = len(w_phones)
        if n_phones == 0:
            phone_anomalies.append({"word": w_text, "phones": 0, "type": "no_phones"})
        elif n_phones > 8:
            phone_anomalies.append({"word": w_text, "phones": n_phones, "type": "too_many"})

    return {
        "file": fname,
        "duration": tg["xmax"] - tg["xmin"],
        "total_chinese_chars": total_chinese,
        "pinyin_matches": match_count,
        "pinyin_mismatches": len(mismatches),
        "mismatch_rate": round(len(mismatches) / max(total_chinese, 1), 4),
        "displacement_runs": shift_analysis,
        "num_displacement_runs": len(displacement_runs),
        "english_like_words": english_like,
        "phone_anomalies": phone_anomalies,
        "raw_text_preview": raw_text_tier[0]["text"][:80] if raw_text_tier else "",
    }


def batch_analyze(dirpath, max_files=None):
    files = sorted([f for f in os.listdir(dirpath) if f.endswith('.TextGrid')])
    if max_files:
        files = files[:max_files]

    results = []
    displacement_files = []
    mismatch_rate_dist = Counter()
    shift_type_dist = Counter()
    all_mismatch_rates = []

    print(f"Deep-analyzing {len(files)} files...")

    for i, fname in enumerate(files):
        if (i+1) % 1000 == 0:
            print(f"  Progress: {i+1}/{len(files)}")

        result = analyze_file(os.path.join(dirpath, fname))
        results.append(result)

        if result.get("pinyin_mismatches", 0) > 0:
            all_mismatch_rates.append(result["mismatch_rate"])
            rate_bucket = int(result["mismatch_rate"] * 20) * 5  # bucket into 5% groups
            bucket_label = f"{rate_bucket}-{rate_bucket+5}%"
            mismatch_rate_dist[bucket_label] += 1

        if result.get("num_displacement_runs", 0) > 0:
            displacement_files.append(result)
            for dr in result["displacement_runs"]:
                shift_type_dist[dr["shift_type"]] += 1

    # ── Report ──
    print(f"\n{'='*80}")
    print("DEEP ALIGNMENT ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"\nFiles analyzed: {len(files)}")

    # Mismatch rate distribution
    files_with_mismatches = [r for r in results if r.get("pinyin_mismatches", 0) > 0]
    print(f"\n── Character-to-Pinyin Mismatch ──")
    print(f"Files with mismatches: {len(files_with_mismatches)} ({len(files_with_mismatches)/max(len(files),1)*100:.1f}%)")
    print(f"Clean files (all match): {len(files) - len(files_with_mismatches)}")

    if all_mismatch_rates:
        avg = sum(all_mismatch_rates) / len(all_mismatch_rates)
        print(f"Average mismatch rate: {avg*100:.1f}%")
        print(f"\nMismatch rate distribution:")
        for bucket in sorted(mismatch_rate_dist.keys(), key=lambda x: int(x.split('-')[0])):
            print(f"  {bucket:10s}: {mismatch_rate_dist[bucket]:5d} files")

    # Displacement runs
    print(f"\n── Displacement Detection ──")
    print(f"Files with displacement: {len(displacement_files)} ({len(displacement_files)/max(len(files),1)*100:.1f}%)")
    print(f"Total displacement runs found: {sum(shift_type_dist.values())}")
    print(f"Shift type distribution:")
    for st, count in shift_type_dist.most_common():
        print(f"  {st:25s}: {count:5d} occurrences")

    # Displacement examples
    print(f"\n── Displacement Examples ──")
    disp_with_runs = sorted(displacement_files, key=lambda x: x.get("num_displacement_runs", 0), reverse=True)[:5]
    for dr in disp_with_runs:
        print(f"\n  File: {dr['file']}")
        print(f"  Duration: {dr['duration']:.1f}s, Chinese chars: {dr['total_chinese_chars']}")
        print(f"  Mismatches: {dr['pinyin_mismatches']}/{dr['total_chinese_chars']} ({dr['mismatch_rate']*100:.1f}%)")
        print(f"  Displacement runs: {dr['num_displacement_runs']}")
        for run in dr['displacement_runs'][:3]:
            print(f"    Run [{run['start_idx']}-{run['end_idx']}], length={run['length']}, type={run['shift_type']}")
            print(f"      Hanzi:    {run['sample_hanzi']}...")
            print(f"      Expected: {run['sample_expected']}...")
            print(f"      Actual:   {run['sample_actual']}...")
        if dr.get('raw_text_preview'):
            print(f"  Raw text: {dr['raw_text_preview']}...")

    # Files with very high mismatch rates
    severe = [r for r in results if r.get("mismatch_rate", 0) > 0.3]
    print(f"\n── Severe Cases (>30% mismatch rate): {len(severe)} files ──")
    for r in sorted(severe, key=lambda x: x["mismatch_rate"], reverse=True)[:5]:
        print(f"  {r['file']}: {r['mismatch_rate']*100:.1f}% mismatch, "
              f"{r['num_displacement_runs']} displacement runs")
        if r.get('raw_text_preview'):
            print(f"    Raw: {r['raw_text_preview']}...")

    # Phone anomalies summary
    phone_issue_files = [r for r in results if r.get("phone_anomalies")]
    print(f"\n── Phone Anomalies ──")
    print(f"Files with phone anomalies: {len(phone_issue_files)}")

    no_phone_count = sum(
        sum(1 for pa in r.get("phone_anomalies", []) if pa["type"] == "no_phones")
        for r in results
    )
    too_many_count = sum(
        sum(1 for pa in r.get("phone_anomalies", []) if pa["type"] == "too_many")
        for r in results
    )
    print(f"  Words with no phones: {no_phone_count}")
    print(f"  Words with >8 phones: {too_many_count}")

    # English-like word handling
    eng_files = [r for r in results if r.get("english_like_words")]
    print(f"\n── English Word Handling ──")
    print(f"Files with English-like words: {len(eng_files)}")

    # Collect unique English words
    eng_words = Counter()
    for r in results:
        for ew in r.get("english_like_words", []):
            eng_words[ew["hanzi"]] += 1
    print(f"  Top English-like words:")
    for word, count in eng_words.most_common(15):
        print(f"    {word}: {count} occurrences")

    # Write full JSON
    output_path = os.path.join(dirpath, "..", "deep_audit_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_files": len(files),
                "files_with_mismatches": len(files_with_mismatches),
                "files_with_displacement": len(displacement_files),
                "average_mismatch_rate": round(sum(all_mismatch_rates) / max(len(all_mismatch_rates), 1), 4),
                "shift_type_distribution": dict(shift_type_dist.most_common()),
                "severe_cases": len(severe),
                "phone_anomaly_files": len(phone_issue_files),
            },
            "severe_cases": [
                {"file": r["file"], "rate": r["mismatch_rate"], "runs": r["num_displacement_runs"],
                 "raw_text": r.get("raw_text_preview", "")}
                for r in sorted(severe, key=lambda x: x["mismatch_rate"], reverse=True)[:50]
            ],
            "displacement_files": [
                {"file": r["file"], "runs": r["num_displacement_runs"], "rate": r["mismatch_rate"],
                 "details": r["displacement_runs"]}
                for r in displacement_files[:100]
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nFull report written to: {output_path}")
    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/mnt/Raw/0805test"
    batch_analyze(target)
