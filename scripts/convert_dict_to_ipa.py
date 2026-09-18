#!/usr/bin/env python3
"""
Convert fullpinyin_enword.dict phone notation from pinyin (b a1) to MFA IPA (p a˥˥).

Keeps all tokens unchanged. Only remaps the phone columns.
"""

import re
import sys
from pathlib import Path

# Initial consonant mapping: pinyin → MFA IPA
INITIAL_MAP = {
    'b': 'p', 'p': 'pʰ', 'm': 'm', 'f': 'f',
    'd': 't', 't': 'tʰ', 'n': 'n', 'l': 'l',
    'g': 'k', 'k': 'kʰ', 'h': 'x',
    'j': 'tɕ', 'q': 'tɕʰ', 'x': 'ɕ',
    'zh': 'ʈʂ', 'ch': 'ʈʂʰ', 'sh': 'ʂ', 'r': 'ɻ',
    'z': 'ts', 'c': 'tsʰ', 's': 's',
    'y': 'j', 'w': 'w',
}

# Tone mapping: digit → Chao tone letters
TONE_MAP = {
    '1': '˥',
    '2': '˧˥',
    '3': '˨˩˦',
    '4': '˥˩',
    '5': '˩',
}

# Final (base, without tone) → list of IPA segments.
# The LAST vowel-like segment gets the tone.
# Format: list of strings. Segments ending in '*' get the tone appended.
FINAL_SEGMENTS = {
    # Simple finals
    'a':   ['a*'],
    'o':   ['o*'],
    'e':   ['ə*'],       # e→ə in most contexts
    'e2':  ['e*'],       # e in ye, ie contexts
    'i':   ['i*'],
    'u':   ['u*'],
    'v':   ['y*'],       # ü
    # i0 variants (used in yi, yin, ying, etc.)
    'i0':  ['i*'],
    'u0':  ['u*'],
    'v0':  ['y*'],
    # Compound finals
    'ai':   ['a*', 'j'],
    'ei':   ['e*', 'j'],
    'ao':   ['a*', 'w'],
    'ou':   ['o*', 'w'],
    # Nasal finals
    'an':   ['a*', 'n'],
    'en':   ['ə*', 'n'],
    'in':   ['i*', 'n'],
    'ang':  ['a*', 'ŋ'],
    'eng':  ['ə*', 'ŋ'],
    'ing':  ['i*', 'ŋ'],
    'ong':  ['u*', 'ŋ'],
    # i- medial
    'ia':   ['j', 'a*'],
    'ie':   ['j', 'e*'],
    'iao':  ['j', 'a*', 'w'],
    'iu':   ['j', 'o*', 'w'],
    'iou':  ['j', 'o*', 'w'],
    'ian':  ['j', 'e*', 'n'],
    'iang': ['j', 'a*', 'ŋ'],
    'iong': ['j', 'u*', 'ŋ'],
    # u- medial
    'ua':   ['w', 'a*'],
    'uo':   ['w', 'o*'],
    'uai':  ['w', 'a*', 'j'],
    'ui':   ['w', 'e*', 'j'],
    'uei':  ['w', 'e*', 'j'],
    'uan':  ['w', 'a*', 'n'],
    'un':   ['w', 'ə*', 'n'],
    'uen':  ['w', 'ə*', 'n'],
    'uang': ['w', 'a*', 'ŋ'],
    'ueng': ['w', 'ə*', 'ŋ'],
    # v- medial
    've':   ['ɥ', 'e*'],
    'vn':   ['y*', 'n'],
    'van':  ['ɥ', 'e*', 'n'],
    # Syllabic nasals (standalone nasal syllables: 嗯 n, 呣 m)
    'n':    ['n̩*'],
    'm':    ['m̩*'],
    # Special
    'er':   ['ə*', 'ɻ'],
    'io':   ['j', 'o*'],
}


def split_final(final_with_tone: str) -> tuple[str, str]:
    """Split 'a1' → ('a', '1'), 'uang3' → ('uang', '3'), 'i01' → ('i0', '1')."""
    m = re.match(r'^(.+?)([1-5])$', final_with_tone)
    if not m:
        raise ValueError(f"Cannot parse final: {final_with_tone}")
    return m.group(1), m.group(2)


def convert_final(final_with_tone: str) -> str:
    """Convert a final like 'a1' → 'a˥˥', 'ian3' → 'j e˨˩˦ n'."""
    base, tone_digit = split_final(final_with_tone)
    tone = TONE_MAP[tone_digit]

    if base not in FINAL_SEGMENTS:
        raise ValueError(f"Unknown final base: {base} (from {final_with_tone})")

    segs = FINAL_SEGMENTS[base]
    result_parts = []
    for seg in segs:
        if seg.endswith('*'):
            result_parts.append(seg[:-1] + tone)
        else:
            result_parts.append(seg)
    return ' '.join(result_parts)


def convert_entry(token: str, phones: list[str]) -> str | None:
    """Convert a dict entry. Returns the new line or None if not a Chinese entry."""
    if len(phones) == 0:
        return None

    # PAUSE tokens: use spn phone (compatible with acoustic model)
    if token in ('[PAUSE]', '<PAUSE>'):
        return f"{token} spn"

    # 1-phone entry: standalone final without initial (e.g., a1 a1)
    if len(phones) == 1 and re.search(r'[1-5]$', phones[0]):
        ipa_final = convert_final(phones[0])
        return f"{token} {ipa_final}"

    # 2-phone entry: initial + final_tone (e.g., ba1 b a1)
    if len(phones) == 2:
        initial = phones[0]
        final = phones[1]

        if initial in INITIAL_MAP and re.search(r'[1-5]$', final):
            ipa_initial = INITIAL_MAP[initial]

            # Special: apical vowel after dental sibilants (z/c/s + i0 → z̩)
            if initial in ('z', 'c', 's') and final.startswith('i0'):
                base, tone_digit = split_final(final)
                ipa_final = f"z̩{TONE_MAP[tone_digit]}"
            # Special: apical vowel after retroflex (zh/ch/sh/r + ir → ʐ̩)
            elif final.startswith('ir'):
                base, tone_digit = split_final(final)
                ipa_final = f"ʐ̩{TONE_MAP[tone_digit]}"
            else:
                ipa_final = convert_final(final)

            return f"{token} {ipa_initial} {ipa_final}"

    # English/ARPABET entries: keep as-is (or skip)
    # Check if phones look like ARPABET (uppercase + digits like IY1, AH0)
    if any(re.match(r'^[A-Z]+\d$', p) for p in phones):
        return None  # skip English ARPABET entries

    # Unknown format: keep as-is
    return f"{token} {' '.join(phones)}"


def convert_dict(input_path: Path, output_path: Path):
    """Convert the entire dictionary."""
    written = 0
    skipped = 0
    unknown = 0

    with open(input_path, 'r', encoding='utf-8-sig') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            token = parts[0]
            phones = parts[1:]

            result = convert_entry(token, phones)
            if result is None:
                skipped += 1
            else:
                fout.write(result + '\n')
                written += 1

        print(f"Converted: {written} entries written, {skipped} skipped (English/ARPABET)")
    print(f"Output: {output_path}")


def main():
    import argparse as _ap
    parser = _ap.ArgumentParser(description="Convert pinyin dict to MFA IPA format.")
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent.parent / "dict" / "fullpinyin_enword.dict",
                        help="Input pinyin dictionary")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "dict" / "mfa_ipa.dict",
                        help="Output IPA dictionary")
    args = parser.parse_args()
    if not args.input.exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)
    convert_dict(args.input, args.output)


if __name__ == "__main__":
    main()
