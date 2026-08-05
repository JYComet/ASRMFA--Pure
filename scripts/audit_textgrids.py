#!/usr/bin/env python3
"""
Comprehensive audit of MFA TextGrid alignment results.
Checks for: missing words, missing phones, track misalignment,
displacement errors, special token leakage, duration anomalies,
empty intervals, punctuation issues, phone count mismatches,
and language mixing.
"""

import os
import re
import sys
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ── TextGrid Parsing ──────────────────────────────────────────────────────

def parse_textgrid(filepath: str) -> Optional[Dict]:
    """Parse a TextGrid file into a structured dict. Returns None on failure."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return None

    result = {"filename": os.path.basename(filepath), "tiers": {}}

    # Parse tier names
    tier_names = re.findall(r'name\s*=\s*"(.*?)"', content)

    # Parse intervals for each tier using a more robust approach
    # Split by item [N]: blocks
    items = re.split(r'\n\s*item\s*\[\d+\]:', content)

    # The first block before any item[] is the header
    tier_num = 0
    for block in items[1:]:  # skip header
        if 'IntervalTier' not in block:
            continue

        # Extract tier name
        name_match = re.search(r'name\s*=\s*"(.*?)"', block)
        if not name_match:
            continue
        name = name_match.group(1)

        # Extract xmin, xmax for the tier
        t_xmin = float(re.search(r'xmin\s*=\s*([\d.]+)', block).group(1))
        t_xmax = float(re.search(r'xmax\s*=\s*([\d.]+)', block).group(1))

        # Extract all intervals
        intervals = []
        interval_blocks = re.split(r'\n\s*intervals\s*\[\d+\]:', block)

        for ib in interval_blocks[1:]:  # skip size declaration
            xmin_m = re.search(r'xmin\s*=\s*([\d.]+)', ib)
            xmax_m = re.search(r'xmax\s*=\s*([\d.]+)', ib)
            text_m = re.search(r'text\s*=\s*"(.*?)"', ib)

            if xmin_m and xmax_m:
                intervals.append({
                    "xmin": float(xmin_m.group(1)),
                    "xmax": float(xmax_m.group(1)),
                    "text": text_m.group(1) if text_m else "",
                })

        result["tiers"][name] = {
            "xmin": t_xmin,
            "xmax": t_xmax,
            "intervals": intervals,
        }

    result["xmin"] = float(re.search(r'xmin\s*=\s*([\d.]+)', items[0]).group(1))
    result["xmax"] = float(re.search(r'xmax\s*=\s*([\d.]+)', items[0]).group(1))

    return result


# ── Character-to-Pinyin Knowledge ─────────────────────────────────────────

# This is deliberately incomplete — we detect mismatches by comparing
# pinyin sequence from the 'words' tier vs the 'pinyin' tier's sequence.

SP_TOKEN_PAT = re.compile(r'^<sp\d+>$')
PUNCT_PAT = re.compile(r'^[，。！？、；：""''…\-,.!?;:\'"\s]+$')
CHINESE_CHAR_PAT = re.compile(r'[一-鿿]')
ENGLISH_WORD_PAT = re.compile(r'^[A-Za-z0-9]+$')
RIA_PLACEHOLDER_PAT = re.compile(r'^RIA.*$')

# ── Audit Functions ───────────────────────────────────────────────────────

@dataclass
class Issue:
    file: str
    category: str  # e.g. "displacement", "missing_phones", "special_token", etc.
    severity: str  # "critical", "warning", "info"
    detail: str
    evidence: str = ""


def audit_file(filepath: str) -> List[Issue]:
    """Run all checks on a single TextGrid file."""
    issues = []
    fname = os.path.basename(filepath)

    tg = parse_textgrid(filepath)
    if tg is None:
        issues.append(Issue(fname, "parse_error", "critical", "Failed to parse TextGrid"))
        return issues

    tiers = tg["tiers"]
    expected_tiers = ["raw_text", "pinyin", "hanzi", "words", "pinyin_phones"]

    # ─── Check 1: Missing tiers ───
    missing_tiers = [t for t in expected_tiers if t not in tiers]
    if missing_tiers:
        issues.append(Issue(fname, "missing_tiers", "critical",
                           f"Missing tiers: {', '.join(missing_tiers)}"))
        # Can't continue if essential tiers are missing
        if "hanzi" in missing_tiers or "words" in missing_tiers:
            return issues

    hanzi = tiers.get("hanzi", {}).get("intervals", [])
    words = tiers.get("words", {}).get("intervals", [])
    phones = tiers.get("pinyin_phones", {}).get("intervals", [])
    raw_text = tiers.get("raw_text", {}).get("intervals", [])
    pinyin_tier = tiers.get("pinyin", {}).get("intervals", [])

    # ─── Check 2: Special token leakage ───
    sp_count = 0
    ria_count = 0
    for tier_name, tier_data in tiers.items():
        for interval in tier_data["intervals"]:
            text = interval["text"].strip()
            if SP_TOKEN_PAT.match(text):
                if tier_name in ("words", "pinyin_phones"):
                    sp_count += 1
            if RIA_PLACEHOLDER_PAT.match(text) and tier_name in ("words", "pinyin_phones"):
                ria_count += 1

    if sp_count > 0:
        issues.append(Issue(fname, "special_token_leak", "warning",
                           f"{sp_count} <spN> token(s) in words/phones tiers"))
    if ria_count > 0:
        issues.append(Issue(fname, "special_token_leak", "warning",
                           f"{ria_count} RIA placeholder(s) in words/phones tiers"))

    # ─── Check 3: Empty intervals ───
    empty_hanzi = sum(1 for iv in hanzi if iv["text"] == "")
    empty_words = sum(1 for iv in words if iv["text"] == "")
    empty_phones = sum(1 for iv in phones if iv["text"] == "")

    if empty_hanzi > 0:
        issues.append(Issue(fname, "empty_interval", "warning",
                           f"{empty_hanzi} empty intervals in hanzi tier"))
    if empty_words > 0:
        issues.append(Issue(fname, "empty_interval", "critical",
                           f"{empty_words} empty intervals in words tier"))
    if empty_phones > 0:
        issues.append(Issue(fname, "empty_interval", "critical",
                           f"{empty_phones} empty intervals in pinyin_phones tier"))

    # ─── Check 4: Hanzi vs Words count inconsistency ───
    # hanzi and words should have identical interval counts
    if hanzi and words:
        if len(hanzi) != len(words):
            issues.append(Issue(fname, "count_mismatch", "critical",
                               f"hanzi has {len(hanzi)} intervals, words has {len(words)}"))

    # ─── Check 5: Cross-tier time alignment ───
    # Check that hanzi intervals are contained within words intervals
    # And words intervals map to phone intervals

    # For each non-punct hanzi, find matching words and check time overlap
    if hanzi and words and len(hanzi) == len(words):
        time_mismatches = []
        for i, (h, w) in enumerate(zip(hanzi, words)):
            # Allow small floating point tolerance
            time_diff_xmin = abs(h["xmin"] - w["xmin"])
            time_diff_xmax = abs(h["xmax"] - w["xmax"])
            if time_diff_xmin > 0.001 or time_diff_xmax > 0.001:
                time_mismatches.append(i)

        if time_mismatches:
            issues.append(Issue(fname, "time_mismatch_hanzi_words", "critical",
                               f"Time boundary mismatch at {len(time_mismatches)} positions "
                               f"(indices: {time_mismatches[:5]}{'...' if len(time_mismatches) > 5 else ''})"))

    # Check words vs phones time alignment
    if words and phones:
        # Find phone intervals that fall within each word's time span
        words_without_phones = []
        for i, w in enumerate(words):
            w_text = w["text"].strip()
            if not w_text or SP_TOKEN_PAT.match(w_text) or PUNCT_PAT.match(w_text):
                continue
            # Find any phone interval overlapping this word
            has_phone = False
            for p in phones:
                if (p["xmin"] >= w["xmin"] - 0.001 and
                    p["xmax"] <= w["xmax"] + 0.001 and
                    p["text"].strip()):
                    has_phone = True
                    break
            if not has_phone:
                words_without_phones.append(i)

        if words_without_phones:
            # Only report if many are missing
            total_content = sum(1 for w in words
                              if w["text"].strip() and not SP_TOKEN_PAT.match(w["text"])
                              and not PUNCT_PAT.match(w["text"]))
            ratio = len(words_without_phones) / max(total_content, 1)
            if ratio > 0.1:
                issues.append(Issue(fname, "missing_phones", "critical",
                                   f"{len(words_without_phones)}/{total_content} words lack phone coverage "
                                   f"({ratio:.1%}, indices: {words_without_phones[:3]}...)"))

    # ─── Check 6: Displacement detection ───
    # Compare pinyin sequence from 'words' tier with the 'pinyin' tier
    # to detect shifted/offset alignments
    if pinyin_tier and words:
        raw_pinyin_text = pinyin_tier[0].get("text", "")
        # Extract pinyin tokens from the raw pinyin tier
        raw_pinyin_tokens = [t for t in raw_pinyin_text.split()
                           if t and not SP_TOKEN_PAT.match(t) and not PUNCT_PAT.match(t)]

        # Get non-punct, non-sp words
        word_tokens = [w["text"] for w in words
                      if w["text"] and not SP_TOKEN_PAT.match(w["text"])
                      and not PUNCT_PAT.match(w["text"])]

        # Count exact matches vs mismatches
        total_comparable = min(len(raw_pinyin_tokens), len(word_tokens))
        mismatches = 0
        mismatch_details = []
        for j in range(total_comparable):
            rp = raw_pinyin_tokens[j].strip()
            wt = word_tokens[j].strip()
            if rp and wt and rp != wt:
                mismatches += 1
                if len(mismatch_details) < 5:
                    mismatch_details.append(f"pos={j}: pinyin_tier=[{rp}] vs words=[{wt}]")

        if mismatches > 0:
            mismatch_ratio = mismatches / total_comparable
            severity = "critical" if mismatch_ratio > 0.05 else "warning"
            issues.append(Issue(fname, "pinyin_displacement", severity,
                               f"{mismatches}/{total_comparable} pinyin mismatches ({mismatch_ratio:.1%}) "
                               f"between pinyin tier and words tier",
                               "; ".join(mismatch_details)))

    # ─── Check 7: Duration anomalies ───
    # Check for extremely short intervals (< 5ms) in phones tier
    very_short = []
    for i, p in enumerate(phones):
        dur = p["xmax"] - p["xmin"]
        if dur < 0.005 and p["text"].strip() and not PUNCT_PAT.match(p["text"]):
            very_short.append((i, p["text"], dur))

    if very_short:
        short_count = len(very_short)
        shortest = min(very_short, key=lambda x: x[2])
        issues.append(Issue(fname, "duration_anomaly", "warning",
                           f"{short_count} phone(s) with duration < 5ms. "
                           f"Shortest: [{shortest[1]}]={shortest[2]*1000:.1f}ms at idx {shortest[0]}"))

    # Check for suspiciously long intervals (> 5s) in words tier
    very_long = []
    for i, w in enumerate(words):
        dur = w["xmax"] - w["xmin"]
        if dur > 5.0 and w["text"].strip() and not PUNCT_PAT.match(w["text"]):
            very_long.append((i, w["text"], dur))

    if very_long:
        issues.append(Issue(fname, "duration_anomaly", "warning",
                           f"{len(very_long)} word(s) with duration > 5s: "
                           f"{[(x[1], f'{x[2]:.1f}s') for x in very_long[:3]]}"))

    # ─── Check 8: Phone count per word ───
    # A Chinese syllable typically has 1-4 phones (initial + final, or just final).
    # Check for abnormally high phone counts per word (= possible runaway alignment)
    if words and phones:
        word_to_phone_count = []
        for i, w in enumerate(words):
            w_text = w["text"].strip()
            if not w_text or SP_TOKEN_PAT.match(w_text) or PUNCT_PAT.match(w_text):
                continue
            phone_count = sum(1 for p in phones
                            if p["xmin"] >= w["xmin"] - 0.001
                            and p["xmax"] <= w["xmax"] + 0.001
                            and p["text"].strip()
                            and not PUNCT_PAT.match(p["text"]))
            word_to_phone_count.append((w_text, phone_count))

        abnormal = [(w, c) for w, c in word_to_phone_count if c > 6 or c == 0]
        if abnormal:
            issues.append(Issue(fname, "phone_count_anomaly", "warning",
                               f"{len(abnormal)} word(s) with abnormal phone count: "
                               f"{[(w, c) for w, c in abnormal[:5]]}"))

    # ─── Check 9: Raw text quality (speech recognition errors) ───
    # Check if the raw_text contains known error markers
    # RIA is a known placeholder pattern in speech recognition errors
    if raw_text:
        rt = raw_text[0].get("text", "")
        ria_matches = re.findall(r'RIA\w*', rt)
        if ria_matches:
            issues.append(Issue(fname, "raw_text_quality", "info",
                               f"RIA placeholder(s) found: {ria_matches[:3]}"))
        # Check for unexpected repeated characters
        repeats = re.findall(r'(.)\1{5,}', rt)
        if repeats:
            issues.append(Issue(fname, "raw_text_quality", "info",
                               f"Repeated characters: {repeats}"))

    # ─── Check 10: Punctuation with phones ───
    # Punctuation marks should NOT have corresponding phone intervals
    punct_with_phones = []
    for w in words:
        if PUNCT_PAT.match(w["text"].strip()):
            matching_phones = [p for p in phones
                              if p["xmin"] >= w["xmin"] - 0.001
                              and p["xmax"] <= w["xmax"] + 0.001
                              and p["text"].strip()
                              and not PUNCT_PAT.match(p["text"])]
            if matching_phones:
                punct_with_phones.append((w["text"], len(matching_phones)))

    if punct_with_phones:
        issues.append(Issue(fname, "punct_with_phones", "warning",
                           f"{len(punct_with_phones)} punctuation mark(s) have assigned phones: "
                           f"{punct_with_phones[:3]}"))

    # ─── Check 11: Language mixing (English words in Chinese text) ───
    english_in_words = []
    for w in words:
        txt = w["text"].strip()
        if ENGLISH_WORD_PAT.match(txt) and len(txt) > 2:  # Skip short pinyin like "he"
            # This might be an English word - check its phone representation
            matching_phones = [p for p in phones
                              if p["xmin"] >= w["xmin"] - 0.001
                              and p["xmax"] <= w["xmax"] + 0.001]
            eng_phones = [p["text"] for p in matching_phones
                         if ":" in p["text"]]  # English ARPABET phones have :
            non_eng = [p["text"] for p in matching_phones
                      if ":" not in p["text"] and p["text"].strip()]
            english_in_words.append((txt, len(eng_phones), len(non_eng)))

    if english_in_words:
        issues.append(Issue(fname, "language_mixing", "info",
                           f"{len(english_in_words)} English-like word(s): "
                           f"{[(x[0], f'eng_phones={x[1]}', f'zh_phones={x[2]}') for x in english_in_words[:3]]}"))

    # ─── Check 12: Cross-tier total duration consistency ───
    total_durs = {}
    for name in expected_tiers:
        if name in tiers:
            td = tiers[name]
            total_durs[name] = td["xmax"] - td["xmin"]

    if total_durs:
        ref = max(total_durs.values())
        for name, d in total_durs.items():
            if abs(d - ref) > 0.01:
                issues.append(Issue(fname, "duration_inconsistency", "critical",
                                   f"Tier '{name}' duration {d:.3f}s differs from max {ref:.3f}s"))

    return issues


# ── Batch Analysis ────────────────────────────────────────────────────────

def audit_directory(dirpath: str, max_files: int = None):
    """Audit all TextGrid files in a directory."""
    files = sorted([f for f in os.listdir(dirpath) if f.endswith('.TextGrid')])
    if max_files:
        files = files[:max_files]

    all_issues = []
    issue_counts = Counter()
    severity_counts = Counter()
    files_with_issues = set()

    print(f"Auditing {len(files)} TextGrid files...")

    for i, fname in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(files)}")

        filepath = os.path.join(dirpath, fname)
        file_issues = audit_file(filepath)

        if file_issues:
            files_with_issues.add(fname)
            all_issues.extend(file_issues)
            for iss in file_issues:
                issue_counts[iss.category] += 1
                severity_counts[iss.severity] += 1

    print(f"\nDone. Processed {len(files)} files.\n")

    # ── Summary Report ──
    print("=" * 80)
    print("MFA ALIGNMENT AUDIT REPORT")
    print("=" * 80)
    print(f"\nDirectory: {dirpath}")
    print(f"Total files scanned: {len(files)}")
    print(f"Files with issues: {len(files_with_issues)} ({len(files_with_issues)/max(len(files),1)*100:.1f}%)")
    print(f"Clean files (no issues): {len(files) - len(files_with_issues)}")
    print(f"Total issues found: {len(all_issues)}")

    print(f"\n── Issue Severity Distribution ──")
    for sev in ["critical", "warning", "info"]:
        count = severity_counts.get(sev, 0)
        bar = "█" * min(count // max(1, len(all_issues) // 40), 40)
        print(f"  {sev:10s}: {count:5d}  {bar}")

    print(f"\n── Issue Category Breakdown ──")
    for cat, count in issue_counts.most_common():
        pct = count / max(len(all_issues), 1) * 100
        print(f"  {cat:30s}: {count:5d}  ({pct:5.1f}%)")

    # ── Category-specific details ──
    print(f"\n── Category Detail ──")

    # Displacement analysis
    disp_issues = [i for i in all_issues if i.category == "pinyin_displacement"]
    if disp_issues:
        critical_disps = [i for i in disp_issues if i.severity == "critical"]
        print(f"\n  Pinyin Displacement ({len(disp_issues)} files):")
        print(f"    Critical: {len(critical_disps)} files (>5% mismatch)")
        print(f"    Warning:  {len(disp_issues) - len(critical_disps)} files (≤5% mismatch)")
        if critical_disps:
            print(f"    Examples:")
            for d in critical_disps[:3]:
                print(f"      - {d.file}: {d.detail}")
                if d.evidence:
                    for line in d.evidence.split("; ")[:2]:
                        print(f"          {line}")

    # Duration anomalies
    dur_issues = [i for i in all_issues if i.category == "duration_anomaly"]
    if dur_issues:
        print(f"\n  Duration Anomalies ({len(dur_issues)} files):")
        for d in dur_issues[:5]:
            print(f"    - {d.file}: {d.detail}")

    # Missing phones
    mp_issues = [i for i in all_issues if i.category == "missing_phones"]
    if mp_issues:
        print(f"\n  Missing Phones ({len(mp_issues)} files):")
        for d in mp_issues[:5]:
            print(f"    - {d.file}: {d.detail}")

    # Time mismatches
    tm_issues = [i for i in all_issues if "time_mismatch" in i.category]
    if tm_issues:
        print(f"\n  Time Mismatches ({len(tm_issues)} files, hanzi-words boundary drift):")
        for d in tm_issues[:5]:
            print(f"    - {d.file}: {d.detail}")

    # Special token leakage
    st_issues = [i for i in all_issues if i.category == "special_token_leak"]
    if st_issues:
        print(f"\n  Special Token Leakage ({len(st_issues)} files):")
        for d in st_issues[:5]:
            print(f"    - {d.file}: {d.detail}")

    # Raw text quality
    rtq_issues = [i for i in all_issues if i.category == "raw_text_quality"]
    if rtq_issues:
        print(f"\n  Raw Text Quality ({len(rtq_issues)} files):")
        for d in rtq_issues[:5]:
            print(f"    - {d.file}: {d.detail}")

    # Empty intervals
    ei_issues = [i for i in all_issues if i.category == "empty_interval"]
    if ei_issues:
        print(f"\n  Empty Intervals ({len(ei_issues)} files):")
        critical_empty = [i for i in ei_issues if i.severity == "critical"]
        print(f"    Critical (words/phones tier): {len(critical_empty)} files")
        for d in critical_empty[:5]:
            print(f"    - {d.file}: {d.detail}")

    # Language mixing
    lm_issues = [i for i in all_issues if i.category == "language_mixing"]
    if lm_issues:
        print(f"\n  Language Mixing ({len(lm_issues)} files):")
        for d in lm_issues[:5]:
            print(f"    - {d.file}: {d.detail}")

    # ── Detailed examples ──
    print(f"\n── Detailed Issue Examples (3 worst files) ──")

    # Find files with most issues
    file_issue_count = Counter(i.file for i in all_issues)
    worst_files = file_issue_count.most_common(5)

    for fname, count in worst_files:
        file_issues = [i for i in all_issues if i.file == fname]
        print(f"\n  [{fname}] — {count} issues")
        for iss in file_issues:
            sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(iss.severity, "⚪")
            print(f"    {sev_icon} [{iss.severity}] [{iss.category}] {iss.detail}")
            if iss.evidence:
                for line in iss.evidence.split("; "):
                    print(f"        → {line}")

    # ── Summary stats ──
    print(f"\n── Statistical Summary ──")

    # Collect metrics
    all_durations = []
    all_phone_counts = []
    all_interval_spans = []

    for fname in files[:200]:  # sample first 200
        fp = os.path.join(dirpath, fname)
        tg = parse_textgrid(fp)
        if tg is None:
            continue
        all_durations.append(tg["xmax"] - tg["xmin"])
        phones_tier = tg["tiers"].get("pinyin_phones", {})
        if phones_tier:
            all_phone_counts.append(len(phones_tier.get("intervals", [])))
        hanzi_tier = tg["tiers"].get("hanzi", {})
        if hanzi_tier:
            for iv in hanzi_tier.get("intervals", []):
                all_interval_spans.append(iv["xmax"] - iv["xmin"])

    if all_durations:
        print(f"  Audio duration (sampled):")
        print(f"    Min:    {min(all_durations):.2f}s")
        print(f"    Max:    {max(all_durations):.2f}s")
        print(f"    Median: {sorted(all_durations)[len(all_durations)//2]:.2f}s")
        print(f"    Mean:   {sum(all_durations)/len(all_durations):.2f}s")

    if all_phone_counts:
        print(f"  Phone count per file (sampled):")
        print(f"    Min:    {min(all_phone_counts)}")
        print(f"    Max:    {max(all_phone_counts)}")
        print(f"    Mean:   {sum(all_phone_counts)/len(all_phone_counts):.1f}")

    if all_interval_spans:
        print(f"  Hanzi interval duration (sampled):")
        print(f"    Min:    {min(all_interval_spans)*1000:.1f}ms")
        print(f"    Max:    {max(all_interval_spans)*1000:.1f}ms")
        print(f"    P05:    {sorted(all_interval_spans)[len(all_interval_spans)//20]*1000:.1f}ms")
        print(f"    Median: {sorted(all_interval_spans)[len(all_interval_spans)//2]*1000:.1f}ms")

    # Write full JSON report
    output_path = os.path.join(dirpath, "..", "audit_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_files": len(files),
                "files_with_issues": len(files_with_issues),
                "clean_files": len(files) - len(files_with_issues),
                "total_issues": len(all_issues),
                "issue_counts": dict(issue_counts.most_common()),
                "severity_counts": dict(severity_counts),
            },
            "worst_files": [(f, c) for f, c in file_issue_count.most_common(20)],
            "issues": [
                {
                    "file": i.file,
                    "category": i.category,
                    "severity": i.severity,
                    "detail": i.detail,
                    "evidence": i.evidence,
                }
                for i in all_issues
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nFull report written to: {output_path}")

    return all_issues


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/mnt/Raw/0805test"
    audit_directory(target)
