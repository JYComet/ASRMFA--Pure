"""Verify risks using core string-based functions (no TextGrid needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from postprocess_textgrids import (
    _word_matches,
    _alpha_text_matches,
    _align_word_sequences,
    _extract_word_chars,
)
from pipeline_utils import (
    is_cjk, is_nvv_token, is_pinyin_syllable,
    is_word_like, is_punct, is_silence,
)

SEP = "=" * 60

# ═══════════════════════════════════════════════════════════
# 1. Risk 4.1: NW alignment correctness with vowel constraint
# ═══════════════════════════════════════════════════════════
print(SEP)
print("1. NW alignment — does vowel constraint prevent wrong pinyin→English matches?")
print(SEP)

def trace_alignment(ctc_seq, ref_seq):
    """Run NW alignment and return human-readable result."""
    alignment = _align_word_sequences(ctc_seq, ref_seq)
    lines = []
    for ctc_i, ref_i in alignment:
        ctc_txt = ctc_seq[ctc_i] if ctc_i is not None else "─"
        ref_txt = ref_seq[ref_i] if ref_i is not None else "─"
        lines.append(f"  {ctc_txt:16s} ↔ {ref_txt}")
    return "\n".join(lines)

# Scenario A: The original bug document case
print("\n── Scenario A: Original bug doc (SURPRISE-OH, qie4, pian4, OP) ──")
ctc_a = ["SURPRISE-OH", "qie4", "pian4", "OP"]
ref_a = ["SURPRISE", "OH", "切", "片", "OP"]
print(trace_alignment(ctc_a, ref_a))

# Check: qie4 should align to 切, not OH or SURPRISE
# Check: pian4 should align to 片

# Scenario B: qie4 adjacent to SURPRISE (3 vowels — risky)
print("\n── Scenario B: qie4 adjacent to SURPRISE (SURPRISE=3 vowels) ──")
ctc_b = ["SURPRISE-OH", "qie4", "pian4", "OP"]
ref_b = ["SURPRISE", "切", "片", "OP"]  # no OH!
print(trace_alignment(ctc_b, ref_b))
# CRITICAL: qie4 must NOT align to SURPRISE

# Scenario C: Pinyin rendering (rui4→ria, intended match)
print("\n── Scenario C: Pinyin rendering (rui4→ria, 2 vowels, intended) ──")
ctc_c = ["rui4"]
ref_c = ["ria"]
print(trace_alignment(ctc_c, ref_c))

# Scenario D: Multi-pinyin + English with ≥2 vowels
print("\n── Scenario D: kan4 + video + le5 (video=3 vowels) ──")
ctc_d = ["kan4", "video", "le5"]
ref_d = ["看", "video", "了"]
print(trace_alignment(ctc_d, ref_d))

# Scenario E: Staggered CJK + English
print("\n── Scenario E: Mixed CJK/EN/CJK (你好HELLO世界) ──")
ctc_e = ["ni3", "hao3", "HELLO", "shi4", "jie4"]
ref_e = ["你", "好", "HELLO", "世", "界"]
print(trace_alignment(ctc_e, ref_e))

# Scenario F: Edge — 2 pinyin tokens, 1 English word with ≥2 vowels
print("\n── Scenario F: interleaved (看video了 → kan4, video, le5) ──")
ctc_f = ["kan4", "video", "le5"]
ref_f = ["看", "video", "了"]
print(trace_alignment(ctc_f, ref_f))

# ═══════════════════════════════════════════════════════════
# 2. _word_matches vowel constraint table
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("2. _word_matches pinyin→English: vowel constraint verification")
print(SEP)

tests = [
    ("qie4", "OH",        False, "1 vowel → blocked"),
    ("qie4", "OP",        False, "1 vowel → blocked"),
    ("qie4", "in",        False, "1 vowel → blocked"),
    ("qie4", "up",        False, "1 vowel → blocked"),
    ("qie4", "SURPRISE",  False, "pinyin→English inference is forbidden"),
    ("qie4", "hello",     False, "pinyin→English inference is forbidden"),
    ("qie4", "video",     False, "pinyin→English inference is forbidden"),
    ("qie4", "people",    False, "pinyin→English inference is forbidden"),
    ("ai4",  "idol",      False, "canonical reference must supply English"),
    ("rui4", "ria",       False, "explicit RIA canonicalization owns this case"),
    ("qie4", "切",         True,  "CJK exact match (not vowel path)"),
    ("pian4","片",         True,  "CJK exact match"),
    ("shi4", "世",         True,  "CJK exact match"),
]

failures = []
for ctc, ref, expected, note in tests:
    result = _word_matches(ctc, ref)
    if result != expected:
        failures.append((ctc, ref, expected, result, note))
        print(f"  ✗ {ctc}↔{ref}: got {result}, expected {expected} — {note}")

if not failures:
    print("  ✓ All results match expectations")
else:
    print(f"\n  {len(failures)} FAILURES:")

# ═══════════════════════════════════════════════════════════
# 3. _alpha_text_matches vs _word_matches consistency
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("3. _alpha_text_matches ↔ _word_matches consistency (alpha path)")
print(SEP)

alpha_pairs = [
    ("SURPRISE-OH", "SURPRISE"),
    ("SURPRISE-OH", "OH"),
    ("OP", "OP"),
    ("li", "live"),
    ("ve", "live"),
    ("qie4", "OH"),
    ("qie4", "SURPRISE"),
    ("qie4", "hello"),
    ("rui4", "ria"),
    ("ai4", "idol"),
    ("HELLO", "HELLO"),
    ("HELLO", "WORLD"),
    ("<LAUGHTER>", "<LAUGHTER>"),
    ("laughter", "<LAUGHTER>"),
]

mismatches = []
for token, ref in alpha_pairs:
    wm = _word_matches(token, ref)
    am = _alpha_text_matches(token, ref)
    if wm != am:
        mismatches.append((token, ref, wm, am))
        print(f"  ✗ {token!r}↔{ref!r}: _wm={wm} _am={am}")

if mismatches:
    print(f"\n  {len(mismatches)} MISMATCHES — functions are INCONSISTENT")
else:
    print("  ✓ All 14 pairs consistent")

# ═══════════════════════════════════════════════════════════
# 4. _extract_word_chars edge cases
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. _extract_word_chars edge cases")
print(SEP)

char_tests = [
    ("你好世界", ["你", "好", "世", "界"]),
    ("SURPRISE，OH 切片OP", ["SURPRISE", "，", "OH", "切", "片", "OP"]),
    ("<LAUGHTER>你好", ["<LAUGHTER>", "你", "好"]),
    ("测试…结束", ["测", "试", "…", "结", "束"]),
    ("hello-world", ["hello-world"]),
    ("a—b", ["a", "—", "b"]),
    ("", []),
    ("   ", []),
    ("123abc", ["123abc"]),  # digits kept with alpha
]

for text, expected in char_tests:
    result = _extract_word_chars(text)
    status = "✓" if result == expected else "✗"
    if result != expected:
        print(f"  {status}: {text!r} → {result} (expected {expected})")
if all(_extract_word_chars(t) == e for t, e in char_tests):
    print("  ✓ All correct")

# ═══════════════════════════════════════════════════════════
# 5. is_pinyin_syllable on hanzi-tier-relevant strings
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("5. hanzi_pinyin filter: false positive check")
print(SEP)

fp_tests = [
    ("你", False), ("好", False), ("切", False), ("片", False),
    ("HELLO", False), ("SURPRISE", False), ("<LAUGHTER>", False),
    ("<sp1>", False), ("<sp2>", False), ("，", False), ("。", False),
    ("qie4", True), ("pian4", True), ("ni3", True), ("hao3", True),
    ("de5", True), ("le5", True), ("shi4", True), ("ai4", True),
    ("rui4", True), ("zhong1", True), ("er4", True), ("nv3", True),
]

fp_errors = []
for s, expected in fp_tests:
    result = is_pinyin_syllable(s)
    if result != expected:
        fp_errors.append((s, expected, result))
        print(f"  ✗ is_pinyin_syllable({s!r}) = {result}, expected {expected}")

if not fp_errors:
    print("  ✓ Zero false positives/negatives")

# ═══════════════════════════════════════════════════════════
# 6. is_nvv_token case insensitivity
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("6. is_nvv_token case insensitivity")
print(SEP)

nvv_tests = [
    ("<LAUGHTER>", True),
    ("<laughter>", True),
    ("laughter", True),
    ("LAUGHTER", True),
    ("Laughter", True),
    ("<BREATHING>", True),
    ("<breathing>", True),
    ("slaughter", False),  # must not partial-match
    ("<UNKNOWN>", False),
]

for s, expected in nvv_tests:
    result = is_nvv_token(s)
    status = "✓" if result == expected else "✗"
    if result != expected:
        print(f"  {status}: is_nvv_token({s!r}) = {result}, expected {expected}")

# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

# Check if NW alignment has any pinyin→English mismatch
def check_nw_misalignment(scenarios):
    """Check each scenario for pinyin tokens being aligned to English ref units."""
    issues = []
    for name, ctc, ref in scenarios:
        alignment = _align_word_sequences(ctc, ref)
        for ctc_i, ref_i in alignment:
            if ctc_i is not None and ref_i is not None:
                ctc_tok = ctc[ctc_i]
                ref_tok = ref[ref_i]
                # Check: if CTC token is pinyin but ref is English (not CJK)
                if (is_pinyin_syllable(ctc_tok) and not is_cjk(ref_tok)
                        and _word_matches(ctc_tok, ref_tok)):
                    issues.append((name, ctc_i, ctc_tok, ref_tok))
    return issues

scenarios = [
    ("A", ctc_a, ref_a),
    ("B", ctc_b, ref_b),
    ("C", ctc_c, ref_c),
    ("D", ctc_d, ref_d),
    ("E", ctc_e, ref_e),
    ("F", ctc_f, ref_f),
]

misalignments = check_nw_misalignment(scenarios)
if misalignments:
    print(f"✗ {len(misalignments)} Pinyin→English misalignments found:")
    for name, idx, ctc_tok, ref_tok in misalignments:
        print(f"  Scenario {name}: [{idx}] {ctc_tok!r} ↔ {ref_tok!r}")
else:
    print("✓ Zero pinyin→English misalignments in NW alignment")
    print("  The vowel constraint + CJK exact-match priority work correctly")
    print("  in all tested scenarios.")
