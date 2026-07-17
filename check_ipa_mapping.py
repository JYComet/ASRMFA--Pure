#!/usr/bin/env python3
"""Check all English IPA phones in en_phones output for ARPABET mapping completeness."""
import json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from pipeline_utils import _EN_IPA_TO_ARPABET, en_ipa_to_arpabet

en_dir = Path("output/test_en_mfa/en_phones")
all_phones = Counter()

for f in sorted(en_dir.glob("*.json")):
    data = json.loads(f.read_text(encoding="utf-8"))
    for seg in data:
        for p in seg.get("phones", []):
            phone = p["phone"].strip()
            all_phones[phone] += 1

# Use ascii() for ASCII-safe output to avoid GBK encoding issues
def safe_repr(s):
    return ascii(s)[1:-1]  # ascii() produces \uXXXX escapes

print(f"Total unique English phones found: {len(all_phones)}")

ok = missing = dropped = unknown = 0
issues = []

for phone, count in sorted(all_phones.items(), key=lambda x: -x[1]):
    result = en_ipa_to_arpabet(phone)

    if result == "":
        status = "DROPPED"
        arpabet = "(deleted)"
        dropped += count
    elif result == phone:
        arpabet = phone
        if phone in ("sil", "sp", "spn", "<eps>"):
            status = "PASS (silence)"
            ok += count
        elif phone.isascii() and phone.upper() == phone:
            status = "PASS (ARPABET)"
            ok += count
        else:
            status = "MISSING!"
            arpabet = "---NO MAP---"
            missing += count
            issues.append((phone, count, "no mapping found"))
    else:
        display = result[3:] if result.startswith("en:") else result
        arpabet = display
        status = "MAPPED"
        ok += count

    print(f"  {safe_repr(phone):24s} x{count:4d}  -> {arpabet:8s}  {status}")

print(f"\n  OK: {ok}, MISSING: {missing}, DROPPED: {dropped}")

if issues:
    print(f"\n!!! {len(issues)} MAPPING ISSUES FOUND:")
    for phone, count, reason in issues:
        print(f"  {safe_repr(phone)} (count={count}): {reason}")
    sys.exit(1)
else:
    print("All English IPA phones mapped successfully to ARPABET.")
    sys.exit(0)
