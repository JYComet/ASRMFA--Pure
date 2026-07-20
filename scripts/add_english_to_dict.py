#!/usr/bin/env python3
"""
Scan CTC .lab files under a root directory and add all missing English tokens
to the MFA dictionary as self-referential entries.

English tokens are ASCII-alpha tokens that are NOT:
  - NVV labels (BREATHING, QUESTION-YI, etc.)
  - Pinyin syllables with tone digits (rui4, ya4, etc.)

Each missing token is appended to the dict as ``token token``, so MFA
treats it as a CTC-only boundary (same as NVV tokens).

Usage:
  python scripts/add_english_to_dict.py \
      --root //RS3621/Research_TTS/Data/Raw/ASRNEW \
      --dict dict/mfa_ipa.dict

  # Dry-run: show what would be added without modifying the dict
  python scripts/add_english_to_dict.py --root <path> --dict <path> --dry-run
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (
    translate_path,
    resolve_input_path,
    is_english_token,
    is_nvv_token,
)


def collect_english_tokens(root: Path, progress_every: int = 2000) -> set[str]:
    """Walk *root* recursively and collect all English tokens from .lab files.

    Handles both flat layouts (``dir/{stem}.lab``) and nested layouts
    (``dir/{stem}/{stem}.lab`` / ``dir/{stem}/wavs/{stem}.lab``).

    Prints progress every *progress_every* .lab files to show the scan is alive.
    """
    english: set[str] = set()
    n_lab = 0
    n_tokens = 0

    for lab_path in root.rglob("*.lab"):
        if not lab_path.is_file():
            continue
        n_lab += 1
        if n_lab % progress_every == 0:
            print(f"  ... scanned {n_lab} .lab files, {len(english)} unique English tokens so far")
        try:
            text = lab_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        for token in text.split():
            n_tokens += 1
            if is_english_token(token):
                english.add(token)

    print(f"  Scanned {n_lab} .lab files ({n_tokens} tokens)")
    print(f"  Found {len(english)} unique English tokens")
    if english:
        print(f"  Tokens: {', '.join(sorted(english))}")
    return english


def load_dict_keys(dict_path: Path) -> set[str]:
    """Load the set of existing keys (first column) from *dict_path*."""
    keys: set[str] = set()
    if not dict_path.exists():
        print(f"  WARNING: Dict not found at {dict_path} — will create it")
        return keys
    with open(dict_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(line.split()[0])
    print(f"  Dict has {len(keys)} existing entries")
    return keys


def add_missing_tokens(dict_path: Path, new_tokens: set[str],
                       existing: set[str], dry_run: bool = False) -> int:
    """Append missing tokens to *dict_path* as self-referential entries.

    Comparison is case-insensitive to avoid duplicate entries that differ
    only in casing (MFA does case-insensitive dictionary lookup).

    Returns the number of tokens actually added.
    """
    # Case-insensitive existing set for comparison
    existing_lower: set[str] = {k.lower() for k in existing}

    to_add = sorted(t for t in new_tokens if t.lower() not in existing_lower)
    if not to_add:
        print("  All English tokens already in dict — nothing to add")
        return 0

    if dry_run:
        print(f"\n  Would add {len(to_add)} tokens: {', '.join(to_add)}")
        return len(to_add)

    with open(dict_path, 'a', encoding='utf-8') as f:
        for t in to_add:
            f.write(f"{t} {t}\n")
    print(f"\n  Added {len(to_add)} English tokens to MFA dict: {', '.join(to_add)}")
    return len(to_add)


def main():
    parser = argparse.ArgumentParser(
        description="Scan CTC .lab files and add missing English tokens to MFA dict")
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory to scan for .lab files "
             "(supports Windows UNC paths like //RS3621/... which are auto-translated)")
    parser.add_argument(
        "--dict", type=str, required=True,
        help="Path to MFA dictionary (e.g. dict/mfa_ipa.dict)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be added without modifying the dict")
    args = parser.parse_args()

    # ── Resolve paths (UNC → Linux translation) ──
    root = resolve_input_path(args.root, PROJECT_ROOT)
    dict_path = resolve_input_path(args.dict, PROJECT_ROOT)

    print(f"Root:      {root}")
    print(f"Dict:      {dict_path}")
    print(f"Mode:      {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print()

    if not root.exists():
        print(f"ERROR: Root directory not found: {root}")
        print(f"  (translated from: {args.root})")
        sys.exit(1)

    # ── Collect English tokens ──
    english_tokens = collect_english_tokens(root)
    if not english_tokens:
        print("  No English tokens found — nothing to do")
        return

    # ── Load existing dict ──
    existing = load_dict_keys(dict_path)

    # ── Add missing ──
    added = add_missing_tokens(dict_path, english_tokens, existing,
                               dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry-run complete. {added} token(s) would be added.")
    else:
        print(f"\nDone. {added} token(s) added to {dict_path}.")


if __name__ == "__main__":
    main()
