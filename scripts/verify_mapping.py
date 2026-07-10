#!/usr/bin/env python3
"""Verify IPA→pinyin reverse mapping for every dictionary entry."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Copy of the reverse mapping logic from postprocess_textgrids.py
IPA_CONSONANT_MAP = {
    'p': 'b', 'pʰ': 'p', 't': 'd', 'tʰ': 't', 'k': 'g', 'kʰ': 'k',
    'tɕ': 'j', 'tɕʰ': 'q', 'ʈʂ': 'zh', 'ʈʂʰ': 'ch', 'ts': 'z', 'tsʰ': 'c',
    'f': 'f', 's': 's', 'ɕ': 'x', 'ʂ': 'sh', 'x': 'h',
    'm': 'm', 'n': 'n', 'l': 'l', 'ɻ': 'r',
    'j': 'i', 'w': 'u', 'ɥ': 'v',
    'ŋ': 'ng', 'ʔ': '',
    'z̩': 'i0', 'ʐ̩': 'ir',
}

IPA_TONE_TO_DIGIT = {
    '˥˥': '1', '˥': '1', '˧˥': '2', '˨˩˦': '3', '˥˩': '4', '˩': '5',
}

IPA_VOWEL_BASE_MAP = {
    'a': 'a', 'o': 'o', 'ə': 'e', 'e': 'e',
    'i': 'i', 'u': 'u', 'y': 'v',
    'z̩': 'i0', 'ʐ̩': 'ir',
}

FINAL_DECOMPOSE = {
    'a': ['a'], 'o': ['o'], 'e': ['e'], 'e2': ['e'],
    'i': ['i'], 'u': ['u'], 'v': ['v'],
    'i0': ['i0'], 'u0': ['u0'], 'v0': ['v0'], 'ir': ['ir'],
    'ai': ['a', 'i'], 'ei': ['e', 'i'], 'ao': ['a', 'u'], 'ou': ['o', 'u'],
    'an': ['a', 'n'], 'en': ['e', 'n'], 'in': ['i', 'n'],
    'ang': ['a', 'ng'], 'eng': ['e', 'ng'], 'ing': ['i', 'ng'], 'ong': ['u', 'ng'],
    'ia': ['i', 'a'], 'ie': ['i', 'e'],
    'iao': ['i', 'a', 'u'], 'iu': ['i', 'o', 'u'], 'iou': ['i', 'o', 'u'],
    'ian': ['i', 'e', 'n'], 'iang': ['i', 'a', 'ng'], 'iong': ['i', 'u', 'ng'],
    'ua': ['u', 'a'], 'uo': ['u', 'o'],
    'uai': ['u', 'a', 'i'], 'ui': ['u', 'e', 'i'], 'uei': ['u', 'e', 'i'],
    'uan': ['u', 'a', 'n'], 'un': ['u', 'e', 'n'], 'uen': ['u', 'e', 'n'],
    'uang': ['u', 'a', 'ng'], 'ueng': ['u', 'e', 'ng'],
    've': ['v', 'e'], 'vn': ['v', 'n'], 'van': ['v', 'e', 'n'],
    'er': ['e', 'r'], 'io': ['i', 'o'],
    'n': ['n'], 'm': ['m'],
}

FINAL_TONE_INDEX = {
    'a': 0, 'o': 0, 'e': 0, 'e2': 0, 'i': 0, 'u': 0, 'v': 0,
    'i0': 0, 'u0': 0, 'v0': 0, 'ir': 0,
    'ai': 0, 'ei': 0, 'ao': 0, 'ou': 0,
    'an': 0, 'en': 0, 'in': 0,
    'ang': 0, 'eng': 0, 'ing': 0, 'ong': 0,
    'ia': 1, 'ie': 1, 'iao': 1, 'iu': 1, 'iou': 1,
    'ian': 1, 'iang': 1, 'iong': 1,
    'ua': 1, 'uo': 1, 'uai': 1, 'ui': 1, 'uei': 1,
    'uan': 1, 'un': 1, 'uen': 1,
    'uang': 1, 'ueng': 1,
    've': 1, 'vn': 0, 'van': 1,
    'er': 0, 'io': 1,
    'n': 0, 'm': 0,
}


def decompose_pinyin_phone(phone: str) -> list[str]:
    m = re.match(r'^(.+?)([1-5])$', phone)
    if not m:
        return [phone]
    base, tone = m.group(1), m.group(2)
    if base not in FINAL_DECOMPOSE:
        return [phone]
    components = FINAL_DECOMPOSE[base]
    tone_idx = FINAL_TONE_INDEX.get(base, 0)
    result = []
    for i, comp in enumerate(components):
        if i == tone_idx:
            result.append(comp + tone)
        else:
            result.append(comp)
    return result


def load_dict(path: Path) -> dict[str, list[str]]:
    d = {}
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1:]
    return d


def build_ipa_to_pinyin_map(pinyin_dict, ipa_dict):
    mapping = {}

    # 1. Static consonant map
    for ipa_p, py_p in IPA_CONSONANT_MAP.items():
        if py_p:
            mapping[ipa_p] = py_p

    # 2. Dict-based cross-reference with decomposition
    for token, pinyin_phones in pinyin_dict.items():
        ipa_phones = ipa_dict.get(token)
        if not ipa_phones:
            continue
        decomposed_py = []
        for phone in pinyin_phones:
            decomposed_py.extend(decompose_pinyin_phone(phone))
        if len(ipa_phones) == len(decomposed_py):
            for ipa_p, py_p in zip(ipa_phones, decomposed_py):
                if ipa_p not in mapping:
                    mapping[ipa_p] = py_p

    # 3. Vowel+tone mappings
    for base_ipa, base_py in IPA_VOWEL_BASE_MAP.items():
        for tone_ipa, tone_digit in IPA_TONE_TO_DIGIT.items():
            ipa_phone = base_ipa + tone_ipa
            py_phone = base_py + tone_digit
            if ipa_phone not in mapping:
                mapping[ipa_phone] = py_phone

    return mapping


def reverse_map_phones(ipa_phones, mapping):
    return [mapping.get(p, p) for p in ipa_phones]


def safe_print(*args, **kwargs):
    """Print safely on Windows terminals that can't handle IPA chars."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fall back to ASCII-safe representation
        for arg in args:
            try:
                print(str(arg).encode('ascii', errors='replace').decode('ascii'), **kwargs)
            except Exception:
                print(repr(arg))


def main():
    pinyin_path = PROJECT_ROOT / "dict" / "fullpinyin_enword.dict"
    ipa_path = PROJECT_ROOT / "dict" / "mfa_ipa.dict"

    pinyin_dict = load_dict(pinyin_path)
    ipa_dict = load_dict(ipa_path)
    mapping = build_ipa_to_pinyin_map(pinyin_dict, ipa_dict)

    print(f"Total IPA→pinyin mappings: {len(mapping)}")
    print(f"Pinyin dict entries: {len(pinyin_dict)}")
    print(f"IPA dict entries: {len(ipa_dict)}")
    print()

    # Verify each entry
    errors = []
    ok_count = 0
    no_ipa = 0

    for token, pinyin_phones in sorted(pinyin_dict.items()):
        ipa_phones = ipa_dict.get(token)
        if not ipa_phones:
            no_ipa += 1
            continue

        # Reverse-map IPA to pinyin components
        reverse_mapped = reverse_map_phones(ipa_phones, mapping)

        # Decompose the original pinyin for comparison
        decomposed_original = []
        for phone in pinyin_phones:
            decomposed_original.extend(decompose_pinyin_phone(phone))

        # Compare
        if reverse_mapped != decomposed_original:
            # Only report if this isn't an English/special entry
            if not any(re.match(r'^[A-Z]+\d$', p) for p in pinyin_phones):
                errors.append({
                    'token': token,
                    'pinyin_original': pinyin_phones,
                    'pinyin_decomposed': decomposed_original,
                    'ipa': ipa_phones,
                    'reverse_mapped': reverse_mapped,
                })
        else:
            ok_count += 1

    print(f"OK: {ok_count}")
    print(f"Missing IPA: {no_ipa}")
    print(f"Errors: {len(errors)}")
    print()

    if errors:
        print("=" * 80)
        print("MISMATCHES (reverse-mapped != decomposed original):")
        print("=" * 80)
        for e in errors:
            print(f"\nToken: {e['token']}")
            print(f"  Pinyin original:    {e['pinyin_original']}")
            print(f"  Pinyin decomposed:  {e['pinyin_decomposed']}")
            print(f"  IPA:                {e['ipa']}")
            print(f"  Reverse mapped:     {e['reverse_mapped']}")

    # Also check specific cases the user mentioned
    print("\n" + "=" * 80)
    print("SPECIFIC CASE CHECKS:")
    print("=" * 80)

    test_cases = [
        ('ai1', 'ai1 a˥ j', 'a1 i'),   # the user's example
        ('bai1', 'b ai1', 'b a1 i'),
        ('ao1', 'ao1 a˥ w', 'a1 u'),
        ('tian1', 't ian1', 't i e1 n'),
        ('biao1', 'b iao1', 'b i a1 u'),
        ('guai1', 'g uai1', 'g u a1 i'),
        ('liu1', 'l iou1', 'l i o1 u'),
        ('yue1', 've1', 'v e1'),
        ('ya1', 'ia1', 'i a1'),
        ('wa1', 'ua1', 'u a1'),
        ('er2', 'er2', 'e2 r'),
        ('yang1', 'iang1', 'i a1 ng'),
        ('zhi1', 'zh ir1', 'zh ir1'),
        ('zi1', 'z i01', 'z i01'),
        ('an1', 'an1 a˥ n', 'a1 n'),
        ('ji1', 'j i1', 'j i1'),
    ]

    for token, pinyin_str, expected_str in test_cases:
        ipa_phones = ipa_dict.get(token, [])
        pinyin_phones = pinyin_str.split()
        expected = expected_str.split()

        reverse_mapped = reverse_map_phones(ipa_phones, mapping)
        status = "OK" if reverse_mapped == expected else "MISMATCH"
        print(f"\n{status}: {token}")
        print(f"  IPA:            {ipa_phones}")
        print(f"  Reverse mapped: {reverse_mapped}")
        print(f"  Expected:       {expected}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
