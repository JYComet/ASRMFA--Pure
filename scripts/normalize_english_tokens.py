#!/usr/bin/env python3
"""
Pre-processing: normalise English words in NVASR CTC output.

NVASR's SenseVoice tokenizer breaks OOV English words into Chinese pinyin
approximations (e.g. "ria"→"rui4"+"ya4", "live"→"li"+"ve").  This script
replaces those fragments with the canonical English spelling and merges
their timestamps, so downstream MFA alignment sees a single self-referential
token per English word.

Uses Needleman-Wunsch sequence alignment (same algorithm as
postprocess_textgrids.py) to find the optimal mapping between .lab tokens
and reference-text word units.
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

try:
    from pypinyin import lazy_pinyin, Style
except ModuleNotFoundError:
    raise SystemExit("pypinyin is not installed. Run: pip install pypinyin")

NVV_NAMES: set[str] = {
    "BREATHING", "LAUGHTER", "BURP", "COUGH", "CRYING", "GROAN",
    "HISS", "HUM", "SHH", "SIGH", "SNEEZE", "SNIFF", "SNORE",
    "TSK", "UHM", "WHISTLE", "YAWN",
    "QUESTION-YI", "QUESTION-EN", "QUESTION-OH", "QUESTION-AH",
    "QUESTION-EI", "QUESTION-HUH",
    "SURPRISE-OH", "SURPRISE-AH", "SURPRISE-WA", "SURPRISE-YO",
    "CONFIRMATION-EN", "DISSATISFACTION-HNN",
}


# ---------------------------------------------------------------------------
# Character classification (same as postprocess_textgrids)
# ---------------------------------------------------------------------------

def _is_cjk(ch: str) -> bool:
    return '一' <= ch <= '鿿'


def _is_alpha_group(s: str) -> bool:
    return s.isascii() and bool(s) and all(c.isalpha() or c == '-' for c in s)


def is_nvv_token(s: str) -> bool:
    return s.strip().strip('<>').upper() in NVV_NAMES


# ── Words to keep as single units ──────────────────────────────────────
# Read from dict/merge_words.dict.  Format: self-referential (word word).
# The NVASR tokenizer's phonetic approximations at these word positions
# are merged into the canonical form.  Add new words by editing the dict.
# ────────────────────────────────────────────────────────────────────────

def _load_merge_words() -> set[str]:
    """Load the set of words to keep un-split from dict/merge_words.dict."""
    dict_path = PROJECT_ROOT / "dict" / "merge_words.dict"
    if not dict_path.exists():
        return set()
    words: set[str] = set()
    with open(dict_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 1:
                words.add(parts[0])
    return words


MERGE_WORDS = _load_merge_words()


def is_english_token(token: str) -> bool:
    if not token or not token.isalpha():
        return False
    if not token.isascii():
        return False
    if is_nvv_token(token):
        return False
    if re.match(r'^[a-z]+[1-5]$', token):
        return False
    return True


def _is_word_like(s: str) -> bool:
    if not s:
        return False
    return _is_cjk(s) or s[0].isalpha() or s.isdigit() or is_nvv_token(s)


def _is_punct(s: str) -> bool:
    return bool(s.strip()) and not _is_word_like(s)


def _is_pinyin_token(tok: str) -> bool:
    return bool(re.match(r'^[a-z]+[1-5]$', tok))


def _extract_word_chars(text: str) -> list[str]:
    result = []
    buf = ""
    for c in text:
        if _is_cjk(c):
            if buf:
                result.append(buf)
                buf = ""
            result.append(c)
        elif c.isalpha() or c == '-':
            buf += c
        elif c.isdigit():
            buf += c
        else:
            if buf:
                result.append(buf)
                buf = ""
            if not c.isspace():
                result.append(c)
    if buf:
        result.append(buf)
    return result


def _pinyin_for_cjk(ch: str) -> str | None:
    try:
        py = lazy_pinyin(ch, style=Style.TONE3,
                        neutral_tone_with_five=True, errors="default")
        return py[0] if py else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sequence alignment (same as postprocess_textgrids._word_matches /
# _align_word_sequences)
# ---------------------------------------------------------------------------

def _token_matches_ref(tok: str, ref: str) -> bool:
    """Whether a .lab token could belong to a reference word unit."""
    t = tok.strip().lower()
    r = ref.lower()

    if _is_cjk(ref):
        try:
            py = lazy_pinyin(ref, style=Style.TONE3,
                            neutral_tone_with_five=True, errors="default")
            return py is not None and len(py) > 0 and py[0] == t
        except Exception:
            return False

    if not r.isascii():
        return False

    # Direct containment
    if t in r or r in t:
        return True

    # Single ASCII letter → fragment of English word
    if len(t) == 1 and t.isascii() and t.isalpha():
        return t in r

    # NVV token match
    t_clean = t.strip('<>'); r_clean = r.strip('<>')
    if t_clean in r_clean or r_clean in t_clean:
        return True

    # Pinyin syllable as phonetic rendering of English word — permissive
    # (DP global optimisation resolves ambiguities)
    if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha():
        return True

    return False


def _align_sequences(ctc_seq: list[str],
                     ref_seq: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch global alignment. Gap-first backtrack."""
    n, m = len(ctc_seq), len(ref_seq)
    INF = n + m + 10
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mc = 0 if _token_matches_ref(ctc_seq[i - 1], ref_seq[j - 1]) else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + mc)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            pairs.append((None, j - 1)); j -= 1
        else:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Core: normalise a single stem
# ---------------------------------------------------------------------------

def normalize_stem(txt_dir: Path, stem: str, dry_run: bool = False) -> bool:
    cn_path = txt_dir / f"{stem}_text_cn.txt"
    if not cn_path.exists():
        return False

    ref_text = cn_path.read_text(encoding="utf-8").strip()
    char_units = _extract_word_chars(ref_text)

    # Reference word units (punct filtered)
    ref_units: list[tuple[int, str]] = []
    for i, u in enumerate(char_units):
        if _is_word_like(u):
            ref_units.append((i, u))

    # English words in reference — only normalise those in MERGE_WORDS.
    en_ref_positions: dict[int, str] = {}  # ref_unit_idx → word
    for ri, (ci, u) in enumerate(ref_units):
        if u.lower() in MERGE_WORDS:
            en_ref_positions[ri] = u

    if not en_ref_positions:
        return False

    # Read .lab
    lab_path = txt_dir / f"{stem}.lab"
    if not lab_path.exists():
        return False
    lab_tokens = lab_path.read_text(encoding="utf-8").strip().split()

    # Read tokens.jsonl
    tokens_path = txt_dir / f"{stem}_tokens.jsonl"
    ctc_tokens: list[dict] = []
    if tokens_path.exists():
        for line in tokens_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                ctc_tokens.append(json.loads(line))

    # Align .lab tokens → reference word units
    ref_texts = [u for _, u in ref_units]
    aligned = _align_sequences(lab_tokens, ref_texts)

    # Build: ref_unit_idx → list of lab_indices (matched + following gaps)
    ref_to_lab: dict[int, list[int]] = {ri: [] for ri in en_ref_positions}
    lab_gap_indices: set[int] = set()
    for lab_i, ref_i in aligned:
        if lab_i is None:
            continue
        if ref_i is None:
            lab_gap_indices.add(lab_i)
        elif ref_i in en_ref_positions:
            ref_to_lab[ref_i].append(lab_i)

    # Merge adjacent gaps into the preceding English word's span
    for ri in sorted(ref_to_lab.keys()):
        if not ref_to_lab[ri]:
            continue
        last = ref_to_lab[ri][-1]
        # Absorb consecutive gaps after the last matched token
        g = last + 1
        while g in lab_gap_indices:
            ref_to_lab[ri].append(g)
            lab_gap_indices.discard(g)
            g += 1

    # Check if any English word needs normalisation.
    # Only replace when tokens are clearly fragments (single letters,
    # pinyin syllables) — never replace a complete English word that
    # just happens to differ from the reference (e.g. "life"→"live").
    changes: list[tuple[str, list[int]]] = []
    for ri, indices in sorted(ref_to_lab.items()):
        if not indices:
            continue
        indices.sort()
        en_word = en_ref_positions[ri]
        current = [lab_tokens[i] for i in indices]

        # Already correct
        if len(current) == 1 and current[0] == en_word:
            continue

        # Never merge an NVV token into an English word
        if any(is_nvv_token(t) for t in current):
            continue

        # Safety: only replace if tokens are clearly fragments of the target.
        # Pinyin fragments must share at least one letter with the English word
        # (phonetic plausibility).  e.g. "rui4" shares 'r','i' with "ria" ✓,
        # but "bu4" shares nothing with "ria" ✗.
        all_fragments = True
        en_lower = en_word.lower()
        for t in current:
            if len(t) == 1 and t.isascii() and t.isalpha():
                if t.lower() not in en_lower:
                    all_fragments = False; break
            elif _is_pinyin_token(t):
                base = t[:-1]  # strip tone digit
                if not any(c in en_lower for c in base):
                    all_fragments = False; break
            elif is_english_token(t) and t.lower() in en_lower:
                pass  # substring of target (e.g. "play" in "cosplay")
            else:
                all_fragments = False; break
        if not all_fragments:
            continue

        changes.append((en_word, indices))

    if not changes:
        return False

    if dry_run:
        for en_word, indices in changes:
            old = " + ".join(lab_tokens[i] for i in indices)
            print(f"  [{stem}] {old}  →  {en_word}  (indices {indices})")
        return False

    # Apply
    to_delete: set[int] = set()
    replacements: dict[int, tuple[str, float, float]] = {}
    for en_word, indices in changes:
        first, last = indices[0], indices[-1]
        s = ctc_tokens[first]["start_s"] if first < len(ctc_tokens) else 0.0
        e = ctc_tokens[last]["end_s"] if last < len(ctc_tokens) else 0.0
        replacements[first] = (en_word, s, e)
        for i in indices[1:]:
            to_delete.add(i)

    new_lab = []
    for i, t in enumerate(lab_tokens):
        if i in to_delete: continue
        new_lab.append(replacements[i][0] if i in replacements else t)

    new_ctc = []
    for i, ct in enumerate(ctc_tokens):
        if i in to_delete: continue
        if i in replacements:
            en_word, s, e = replacements[i]
            new_ctc.append({"word": en_word, "start_ms": round(s * 1000),
                           "end_ms": round(e * 1000), "start_s": s, "end_s": e, "type": "word"})
        else:
            new_ctc.append(ct)

    lab_path.write_text(" ".join(new_lab) + "\n", encoding="utf-8")
    tokens_path.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in new_ctc) + "\n",
        encoding="utf-8")

    for en_word, indices in changes:
        old = " + ".join(lab_tokens[i] for i in indices)
        print(f"  [{stem}] {old}  →  {en_word}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalise English-word tokens in NVASR CTC output")
    parser.add_argument("--txt-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    txt_dir = args.txt_dir
    if not txt_dir.exists():
        raise SystemExit(f"Directory not found: {txt_dir}")

    stems = set()
    for f in txt_dir.glob("*_text_cn.txt"):
        stems.add(f.name.replace("_text_cn.txt", ""))
    if not stems:
        for f in txt_dir.glob("*.lab"):
            if (txt_dir / f"{f.stem}_text_cn.txt").exists():
                stems.add(f.stem)

    changed = 0
    for stem in sorted(stems):
        if normalize_stem(txt_dir, stem, dry_run=args.dry_run):
            changed += 1

    if args.dry_run:
        print(f"\nWould normalise {changed}/{len(stems)} stems")
    else:
        print(f"\nNormalised {changed}/{len(stems)} stems")


if __name__ == "__main__":
    main()
