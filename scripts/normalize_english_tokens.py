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
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (
    NVV_NAMES,
    is_cjk, is_nvv_token, is_english_token, is_pinyin_syllable,
    is_word_like, is_punct, extract_word_chars,
)
from postprocess_textgrids import Interval, Tier, parse_textgrid, write_textgrid

# Lazy import — only needed for CJK pinyin fragments in mixed text.
# Pure English processing can proceed without pypinyin.
_pypinyin_available = False
_lazy_pinyin = None
_Style = None


def _ensure_pypinyin():
    global _pypinyin_available, _lazy_pinyin, _Style
    if not _pypinyin_available:
        try:
            from pypinyin import lazy_pinyin as _lp, Style as _st  # noqa: F811
            _lazy_pinyin = _lp
            _Style = _st
            _pypinyin_available = True
        except ModuleNotFoundError:
            print("  [normalize_en] pypinyin not installed — English-only mode")
            _pypinyin_available = False  # mark as tried
    return _pypinyin_available


# ---------------------------------------------------------------------------
# Character classification (same as postprocess_textgrids)
# ---------------------------------------------------------------------------


def _is_alpha_group(s: str) -> bool:
    return s.isascii() and bool(s) and all(c.isalpha() or c == '-' for c in s)


def _fragment_letters(token: str) -> str:
    """Return the alphabetic payload of an English/pinyin fragment."""
    t = token.strip().strip("<>").lower()
    if is_pinyin_syllable(t):
        return t[:-1]
    if t.isascii() and t.isalpha():
        return t
    return ""


def _tokens_plausibly_realise_reference(tokens: list[str], ref_word: str) -> bool:
    """Whether *tokens* can be safely collapsed to authoritative *ref_word*.

    In reference-text mode the spelling must come from ``*_ref.txt``.  This
    helper allows noisy CTC/ASR fragments such as ``li``+``ve`` to realise
    ``life`` without letting an unrelated word sequence consume the reference.
    """
    ref = "".join(ch for ch in ref_word.lower() if ch.isalpha())
    if not ref:
        return False

    pieces: list[str] = []
    for t in tokens:
        if is_nvv_token(t):
            return False
        piece = _fragment_letters(t)
        if not piece:
            return False
        pieces.append(piece)

    joined = "".join(pieces)
    if not joined:
        return False
    if joined == ref or joined in ref or ref in joined:
        return True

    # One or two ASR spelling substitutions are common for short English words
    # in a Chinese tokenizer (life/live, word/world).  Require a fairly close
    # string relationship so unrelated neighbouring English words are protected.
    return SequenceMatcher(None, joined, ref).ratio() >= 0.72



# ── English word detection ───────────────────────────────────────────
# Auto-detected from reference text: ASCII-alpha words of length >= 2.
# The NVASR tokenizer's phonetic approximations at these word positions
# are merged into the canonical form.
# ────────────────────────────────────────────────────────────────────────

# MERGE_WORDS whitelist removed — English words are now auto-detected
# from the reference text (ASCII-alpha, length >= 2).







def _pinyin_for_cjk(ch: str) -> str | None:
    if not _ensure_pypinyin():
        return None
    try:
        py = _lazy_pinyin(ch, style=_Style.TONE3,
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

    if is_cjk(ref):
        if _ensure_pypinyin():
            try:
                py = _lazy_pinyin(ref, style=_Style.TONE3,
                                  neutral_tone_with_five=True, errors="default")
                return py is not None and len(py) > 0 and py[0] == t
            except Exception:
                return False
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

    # Pinyin syllable as phonetic rendering of English word.
    # Only accept when the English reference has ≥2 vowels — prevents
    # short words like "OH"/"OP"/"in"/"Up" from matching pinyin syllables
    # (e.g. qie4↔OH cost 0 → NW gap-first picks OH over CJK 切).
    # Regression Case 10 & 31.
    if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha():
        vowel_count = sum(1 for ch in r if ch in 'aeiou')
        return vowel_count >= 2

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
# Fragment reclamation (Pass 2)
# ---------------------------------------------------------------------------


def _reclaim_fragments(lab_tokens: list[str],
                        ctc_tokens: list[dict]) -> tuple[list[str], list[dict], int]:
    """Merge orphan English fragments into adjacent English words (Pass 2).

    After NW alignment and merge (Pass 1), some English fragments may
    remain unmerged: single letters (e.g. "f" 60ms), 2-letter fragments
    (e.g. "OS" 420ms leftover from "SOS"+"OS"), or mixed-length fragments
    (e.g. "R"+"ia").  This pass scans for these orphans and absorbs them
    into adjacent longer English tokens, or merges consecutive fragments
    into a single token.

    Returns (new_lab, new_ctc, merged_count).
    Regression Case 31 Fix-1.
    """
    if len(lab_tokens) != len(ctc_tokens):
        return lab_tokens, ctc_tokens, 0

    n = len(lab_tokens)
    to_delete: set[int] = set()
    replacements: dict[int, tuple[str, float, float]] = {}

    for i in range(n):
        if i in to_delete:
            continue
        t = lab_tokens[i]
        if not (t.isascii() and t.isalpha() and 1 <= len(t) <= 2):
            continue
        # Skip already-correct single English words at correct duration
        dur = ctc_tokens[i]["end_s"] - ctc_tokens[i]["start_s"]
        if len(t) >= 3 and dur >= 0.080:
            continue

        # Look left: absorb into preceding English word
        if i > 0 and i - 1 not in to_delete:
            prev = lab_tokens[i - 1]
            if (prev.isascii() and prev.isalpha() and len(prev) >= 2
                    and t.lower() in prev.lower()):
                # Absorb fragment into previous word, extend its end time
                if i - 1 in replacements:
                    _, _, old_e = replacements[i - 1]
                    new_e = max(old_e, ctc_tokens[i]["end_s"])
                    replacements[i - 1] = (prev, ctc_tokens[i - 1]["start_s"], new_e)
                else:
                    new_e = max(ctc_tokens[i - 1]["end_s"], ctc_tokens[i]["end_s"])
                    replacements[i - 1] = (prev, ctc_tokens[i - 1]["start_s"], new_e)
                to_delete.add(i)
                continue

        # Look right: absorb into following English word
        if i + 1 < n and i + 1 not in to_delete:
            nxt = lab_tokens[i + 1]
            if (nxt.isascii() and nxt.isalpha() and len(nxt) >= 2
                    and t.lower() in nxt.lower()):
                if i + 1 in replacements:
                    _, old_s, _ = replacements[i + 1]
                    new_s = min(old_s, ctc_tokens[i]["start_s"])
                    replacements[i + 1] = (nxt, new_s, ctc_tokens[i + 1]["end_s"])
                else:
                    new_s = min(ctc_tokens[i + 1]["start_s"], ctc_tokens[i]["start_s"])
                    replacements[i + 1] = (nxt, new_s, ctc_tokens[i + 1]["end_s"])
                to_delete.add(i)
                continue

        # Look left: merge with adjacent fragment (symmetric)
        if i > 0 and i - 1 not in to_delete:
            prev = lab_tokens[i - 1]
            if (prev.isascii() and prev.isalpha() and 1 <= len(prev) <= 2
                    and i - 1 not in to_delete):
                merged = prev + t
                s = ctc_tokens[i - 1]["start_s"]
                e = ctc_tokens[i]["end_s"]
                replacements[i - 1] = (merged, s, e)
                to_delete.add(i)
                continue

        # Look right: merge with adjacent fragment
        if i + 1 < n and i + 1 not in to_delete:
            nxt = lab_tokens[i + 1]
            if (nxt.isascii() and nxt.isalpha() and 1 <= len(nxt) <= 2):
                merged = t + nxt
                s = ctc_tokens[i]["start_s"]
                e = ctc_tokens[i + 1]["end_s"]
                replacements[i] = (merged, s, e)
                to_delete.add(i + 1)

    if not to_delete and not replacements:
        return lab_tokens, ctc_tokens, 0

    new_lab = []
    for i, t in enumerate(lab_tokens):
        if i in to_delete:
            continue
        new_lab.append(replacements[i][0] if i in replacements else t)

    def merge_rows(rows: list[dict], first_index: int, word: str,
                   start: float, end: float) -> dict:
        """Retain the leftmost row and union all consumed source ordinals."""
        merged = dict(rows[0])
        ordinals: list[int] = []
        for offset, row in enumerate(rows):
            values = row.get("source_ctc_ordinals")
            if values is None and "source_ctc_ordinal" in row:
                values = [row["source_ctc_ordinal"]]
            if values is None:
                values = [first_index + offset]
            if (not isinstance(values, (list, tuple))
                    or any(not isinstance(value, int)
                           or isinstance(value, bool) or value < 0
                           for value in values)):
                raise ValueError("invalid source CTC ordinal metadata")
            ordinals.extend(values)
        merged.update({
            "word": word,
            "start_ms": round(start * 1000),
            "end_ms": round(end * 1000),
            "start_s": start,
            "end_s": end,
            "type": merged.get("type", "word"),
            "source_ctc_ordinals": sorted(set(ordinals)),
        })
        merged.pop("source_ctc_ordinal", None)
        return merged

    new_ctc = []
    for i, ct in enumerate(ctc_tokens):
        if i in to_delete:
            continue
        if i in replacements:
            word, s, e = replacements[i]
            consumed_indices = [i]
            left = i - 1
            while left in to_delete:
                consumed_indices.insert(0, left)
                left -= 1
            right = i + 1
            while right in to_delete:
                consumed_indices.append(right)
                right += 1
            consumed = [ctc_tokens[index] for index in consumed_indices]
            new_ctc.append(merge_rows(
                consumed, consumed_indices[0], word, s, e))
        else:
            new_ctc.append(ct)

    return new_lab, new_ctc, len(to_delete)


# ---------------------------------------------------------------------------
# Core: normalise a single stem
# ---------------------------------------------------------------------------

def rewrite_ctc_textgrid_words(tg_path: Path, tokens: list[dict]) -> None:
    """Structurally rewrite a normal CTC ``words`` tier, atomically.

    This intentionally accepts only a regular TextGrid with unique words and
    pauses tiers.  The historical malformed grammar is handled by the strict
    CTC-ready canonicalizer, never silently rewritten here.
    """
    original = parse_textgrid(tg_path)
    words = [tier for tier in original.tiers if tier.name == "words"]
    pauses = [tier for tier in original.tiers if tier.name == "pauses"]
    if len(words) != 1 or len(pauses) != 1:
        raise ValueError(f"Requires unique standard words/pauses tiers: {tg_path}")
    old_lexical = [iv for iv in words[0].intervals if iv.text.strip()]
    if not old_lexical or words[0].xmin != original.xmin or words[0].xmax != original.xmax:
        raise ValueError(f"Requires an intact standard words tier: {tg_path}")
    previous_end = original.xmin
    for index, interval in enumerate(words[0].intervals):
        if interval.xmax <= interval.xmin or interval.xmin + .003 < previous_end:
            raise ValueError(f"Requires valid standard word intervals ({index}): {tg_path}")
        previous_end = interval.xmax
    before_pauses = [(iv.xmin, iv.xmax, iv.text) for iv in pauses[0].intervals]
    cursor = original.xmin; intervals: list[Interval] = []
    for index, token in enumerate(tokens):
        try:
            start, end, text = float(token["start_s"]), float(token["end_s"]), str(token["word"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid token {index}") from exc
        if not (start >= cursor - .003 and end > start and end <= original.xmax + .003):
            raise ValueError(f"Invalid/non-monotonic token timing {index}")
        if start > cursor + 1e-9:
            intervals.append(Interval(cursor, start, ""))
        intervals.append(Interval(start, end, text)); cursor = end
    if cursor < original.xmax - 1e-9:
        intervals.append(Interval(cursor, original.xmax, ""))
    if not intervals and original.xmax > original.xmin:
        intervals.append(Interval(original.xmin, original.xmax, ""))
    words[0].xmin = original.xmin; words[0].xmax = original.xmax; words[0].intervals = intervals
    temporary = tg_path.with_name(tg_path.name + ".tmp")
    try:
        write_textgrid(original, temporary)
        rewritten = parse_textgrid(temporary)
        rewritten_words = [tier for tier in rewritten.tiers if tier.name == "words"]
        rewritten_pauses = [tier for tier in rewritten.tiers if tier.name == "pauses"]
        if len(rewritten_words) != 1 or len(rewritten_pauses) != 1:
            raise ValueError("rewritten TextGrid lost required tiers")
        lexical = [iv for iv in rewritten_words[0].intervals if iv.text.strip()]
        if len(lexical) != len(tokens): raise ValueError("rewritten words/token count mismatch")
        for index, (interval, token) in enumerate(zip(lexical, tokens)):
            if (interval.text != str(token["word"]) or abs(interval.xmin - float(token["start_s"])) > .003
                    or abs(interval.xmax - float(token["end_s"])) > .003):
                raise ValueError(f"rewritten words/token mismatch {index}")
        if [(iv.xmin, iv.xmax, iv.text) for iv in rewritten_pauses[0].intervals] != before_pauses:
            raise ValueError("rewritten pauses tier changed")
        os.replace(temporary, tg_path)
    finally:
        if temporary.exists(): temporary.unlink()


def normalize_stem(txt_dir: Path, stem: str, dry_run: bool = False) -> bool:
    # CTC output contains both the ASR diagnostic transcript and, when the
    # pipeline was run in reference-text mode, the authoritative source as
    # ``*_ref.txt``.  English fragment normalization must follow the same
    # source as .lab/TextGrid, never the potentially erroneous ASR result.
    ref_path = txt_dir / f"{stem}_ref.txt"
    cn_path = txt_dir / f"{stem}_text_cn.txt"
    reference_authoritative = False
    if ref_path.exists():
        ref_text = ref_path.read_text(encoding="utf-8-sig").strip()
        reference_authoritative = bool(ref_text)
    elif cn_path.exists():
        ref_text = cn_path.read_text(encoding="utf-8-sig").strip()
    else:
        return False
    char_units = extract_word_chars(ref_text)

    # Reference word units (punct filtered)
    ref_units: list[tuple[int, str]] = []
    for i, u in enumerate(char_units):
        if is_word_like(u):
            ref_units.append((i, u))

    # English words in reference — auto-detect ASCII-alpha words (len >= 2)
    # as candidates for fragment merging.
    # Exclude NVV tokens (BREATHING, LAUGHTER, etc.) — they have no acoustic
    # model and should never consume pinyin fragments.
    # Regression Case 31 Fix-3a (NVV guard).
    en_ref_positions: dict[int, str] = {}  # ref_unit_idx → word
    for ri, (ci, u) in enumerate(ref_units):
        if u.isascii() and u.isalpha() and len(u) >= 2 and not is_nvv_token(u):
            en_ref_positions[ri] = u

    if not en_ref_positions:
        return False

    # Read .lab
    lab_path = txt_dir / f"{stem}.lab"
    if not lab_path.exists():
        return False
    lab_tokens = lab_path.read_text(encoding="utf-8-sig").strip().split()

    # Read tokens.jsonl
    tokens_path = txt_dir / f"{stem}_tokens.jsonl"
    ctc_tokens: list[dict] = []
    if tokens_path.exists():
        for line in tokens_path.read_text(encoding="utf-8-sig").strip().split("\n"):
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
    # In reference-text mode, ref_text is authoritative: complete but wrong
    # ASR spellings (e.g. "live" for ref "life") must also be corrected.
    # Without *_ref.txt, keep the legacy conservative behaviour.
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

        if reference_authoritative:
            all_fragments = _tokens_plausibly_realise_reference(current, en_word)
        else:
            # Safety: only replace if tokens are clearly fragments of the
            # target.  Pinyin fragments must share at least one letter with
            # the English word (phonetic plausibility).  e.g. "rui4" shares
            # 'r','i' with "ria" ✓, but "bu4" shares nothing with "ria" ✗.
            all_fragments = True
            en_lower = en_word.lower()
            for t in current:
                if len(t) == 1 and t.isascii() and t.isalpha():
                    if t.lower() not in en_lower:
                        all_fragments = False; break
                elif is_pinyin_syllable(t):
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

    if dry_run:
        for en_word, indices in changes:
            old = " + ".join(lab_tokens[i] for i in indices)
            print(f"  [{stem}] {old}  →  {en_word}  (indices {indices})")
        return False

    # ── Apply Pass 1 merges ──
    if changes:
        def merge_rows(indices: list[int], word: str,
                       start: float, end: float) -> dict:
            """Retain leftmost evidence and union consumed source ordinals."""
            merged = dict(ctc_tokens[indices[0]])
            ordinals: list[int] = []
            for index in indices:
                row = ctc_tokens[index]
                values = row.get("source_ctc_ordinals")
                if values is None and "source_ctc_ordinal" in row:
                    values = [row["source_ctc_ordinal"]]
                if values is None:
                    values = [index]
                if (not isinstance(values, (list, tuple))
                        or any(not isinstance(value, int)
                               or isinstance(value, bool) or value < 0
                               for value in values)):
                    raise ValueError("invalid source CTC ordinal metadata")
                ordinals.extend(values)
            merged.update({
                "word": word,
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "start_s": start,
                "end_s": end,
                "type": merged.get("type", "word"),
                "source_ctc_ordinals": sorted(set(ordinals)),
            })
            merged.pop("source_ctc_ordinal", None)
            return merged

        to_delete: set[int] = set()
        replacements: dict[int, tuple[str, float, float]] = {}
        replacement_indices: dict[int, list[int]] = {}
        for en_word, indices in changes:
            first, last = indices[0], indices[-1]
            s = ctc_tokens[first]["start_s"] if first < len(ctc_tokens) else 0.0
            e = ctc_tokens[last]["end_s"] if last < len(ctc_tokens) else 0.0
            replacements[first] = (en_word, s, e)
            replacement_indices[first] = list(indices)
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
                new_ctc.append(merge_rows(
                    replacement_indices[i], en_word, s, e))
            else:
                new_ctc.append(ct)
    else:
        new_lab = list(lab_tokens)
        new_ctc = list(ctc_tokens)

    # ── Pass 2: Fragment reclamation ───────────────────────────────────
    # Legacy ASR mode may still self-reclaim adjacent fragments.  In
    # reference-text mode this is unsafe: it can synthesize "live" from
    # "li"+"ve" even when *_ref.txt says "life".  Pass 1 above already has
    # the reference word list, so any remaining orphan must not invent a new
    # spelling before MFA.
    if reference_authoritative:
        new_lab2, new_ctc2, frag_merged = new_lab, new_ctc, 0
    else:
        # Handles orphan fragments that Pass 1 missed: two short fragments
        # adjacent to each other (e.g. "f"+"an"→"fan") or fragments not
        # matched to any reference word by NW alignment.
        # Regression Case 31 Fix-1.
        new_lab2, new_ctc2, frag_merged = _reclaim_fragments(new_lab, new_ctc)

    if not changes and not frag_merged:
        return False

    lab_path.write_text(" ".join(new_lab2) + "\n", encoding="utf-8")
    tokens_path.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in new_ctc2) + "\n",
        encoding="utf-8")

    # Also update the CTC TextGrid anchors to match the corrected tokens.
    # MFA uses the TextGrid as word-boundary anchors; inconsistent anchors
    # (e.g. "rui4"+"ya4" in TextGrid vs "ria" in .lab) cause MFA to split
    # English words into fragments.
    tg_path = txt_dir / f"{stem}.TextGrid"
    if tg_path.exists():
        rewrite_ctc_textgrid_words(tg_path, new_ctc2)

    for en_word, indices in changes:
        old = " + ".join(lab_tokens[i] for i in indices)
        print(f"  [{stem}] {old}  →  {en_word}")

    if frag_merged:
        print(f"  [{stem}] fragment reclaim: {frag_merged} fragments absorbed")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auto_add_english_to_dict(txt_dir: Path, dict_path: Path) -> int:
    """Scan all .lab files for English tokens and add missing ones to MFA dict.

    English tokens (like "li", "ve", "A", "play") need self-referential
    entries in the MFA dictionary so MFA can treat them as CTC-only tokens
    (no acoustic model).  This mirrors the auto-add logic in ctc_prealign.py.
    """
    if not dict_path or not dict_path.exists():
        return 0

    # Collect English tokens from all .lab files
    english_tokens_found: set[str] = set()
    for lab_path in sorted(txt_dir.glob("*.lab")):
        tokens = lab_path.read_text(encoding="utf-8-sig").strip().split()
        for t in tokens:
            if is_english_token(t):
                english_tokens_found.add(t)

    if not english_tokens_found:
        return 0

    # Load existing dict keys
    existing: set[str] = set()
    with open(dict_path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                existing.add(line.split()[0])

    new_tokens = sorted(t for t in english_tokens_found if t not in existing)
    if new_tokens:
        with open(dict_path, 'a', encoding='utf-8') as f:
            for t in new_tokens:
                f.write(f"{t} {t}\n")
        print(f"  Added {len(new_tokens)} English tokens to MFA dict: {', '.join(new_tokens)}")
    else:
        print(f"  All {len(english_tokens_found)} English tokens already in MFA dict")

    return len(new_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalise English-word tokens in NVASR CTC output")
    parser.add_argument("--txt-dir", type=Path, required=True)
    parser.add_argument("--dict-path", type=Path, default=None,
                        help="MFA dictionary path (for auto-adding missing English tokens)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto: min(32, cpu_count))")
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

    stem_list = sorted(stems)
    if args.dry_run or len(stem_list) <= 4:
        # Serial: dry-run preview or too few stems to justify process overhead
        changed = 0
        errors = 0
        for stem in stem_list:
            try:
                if normalize_stem(txt_dir, stem, dry_run=args.dry_run):
                    changed += 1
            except Exception as e:
                errors += 1
                print(f"  [ERROR] {stem}: {e}")
    else:
        # Parallel: each stem is independent (separate .lab / .TextGrid / _tokens.jsonl)
        _max_w = min(32, os.cpu_count(), len(stem_list))
        n_workers = args.workers if args.workers > 0 else _max_w
        n_workers = min(n_workers, len(stem_list))  # don't exceed work items
        print(f"  Processing {len(stem_list)} stems with {n_workers} workers...")
        changed = 0
        errors = 0
        done = 0
        # Use fork on Linux for copy-on-write sharing of module globals; fall
        # back to the platform default on Windows, where fork is unavailable.
        mp = __import__('multiprocessing')
        ctx = (mp.get_context("fork") if "fork" in mp.get_all_start_methods()
               else mp.get_context())
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(normalize_stem, txt_dir, stem, False): stem
                for stem in stem_list
            }
            for future in as_completed(futures):
                stem = futures[future]
                done += 1
                try:
                    if future.result():
                        changed += 1
                except Exception as e:
                    errors += 1
                    print(f"  [ERROR] {stem}: {e}")
                if done % 500 == 0 or done == len(stem_list):
                    print(f"  [{done}/{len(stem_list)}] {changed} changed")

    if args.dry_run:
        print(f"\nWould normalise {changed}/{len(stems)} stems")
    else:
        print(f"\nNormalised {changed}/{len(stems)} stems")

    # Auto-add English tokens to MFA dictionary (safety net for ctc_ready mode)
    if args.dict_path and not args.dry_run:
        _auto_add_english_to_dict(txt_dir, args.dict_path)

    if errors:
        print(f"ERROR: {errors} stem(s) failed during English token normalization")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
