#!/usr/bin/env python3
"""
Prepare English TTS audio for MFA pipeline (no ASR mode).

Reads generated_scripts.jsonl, segments text into tokens,
creates .lab (pinyin) and _text_cn.txt files for MFA alignment.

Usage:
  python scripts/prepare_english_tts.py \
      --jsonl /mnt/project/Voxcpm/output/generated_scripts.jsonl \
      --audio-root /mnt/Raw/新版合成英文数据 \
      --output-dir output/hecheng_en_mfa/ctc_pretg
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from pypinyin import lazy_pinyin, Style
except ModuleNotFoundError:
    raise SystemExit("pypinyin is required. Run: pip install pypinyin")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def segment_text(text: str) -> list[tuple[str, str]]:
    """Segment text into (kind, token) pairs.

    Kinds: cjk, english, digit, punct
    - CJK chars: one per token
    - English words: contiguous ASCII letters
    - Digits: contiguous digit sequences
    - Punctuation/symbols: one per token
    """
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        # CJK character (including extension A)
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            tokens.append(('cjk', ch))
            i += 1
        # ASCII letters → group into word
        elif ch.isascii() and ch.isalpha():
            j = i
            while j < len(text) and text[j].isascii() and text[j].isalpha():
                j += 1
            tokens.append(('english', text[i:j]))
            i = j
        # Digits → group
        elif ch.isascii() and ch.isdigit():
            j = i
            while j < len(text) and text[j].isascii() and text[j].isdigit():
                j += 1
            tokens.append(('digit', text[i:j]))
            i = j
        # Punctuation / symbols / other
        else:
            tokens.append(('punct', ch))
            i += 1
    return tokens


def text_to_pinyin_tokens(text: str) -> list[str]:
    """Convert text to list of MFA-compatible tokens.

    - CJK chars → pinyin syllable (tone number, e.g. ni3)
    - English words → kept as-is
    - Digits → kept as-is
    - Punctuation → excluded (no acoustic realization in MFA)
    """
    segments = segment_text(text)
    result: list[str] = []
    for kind, token in segments:
        if kind == 'cjk':
            try:
                py = lazy_pinyin(token, style=Style.TONE3,
                                 neutral_tone_with_five=True,
                                 errors='default')
                result.append(py[0] if py else token)
            except Exception:
                result.append(token)
        elif kind in ('english', 'digit'):
            result.append(token)
        # Skip punctuation — not in MFA words tier
    return result


def make_lab_content(text: str) -> str:
    """Build .lab file content from raw text."""
    tokens = text_to_pinyin_tokens(text)
    return ' '.join(tokens)


def process_folders(
    audio_root: Path,
    jsonl_data: list[dict],
    output_dir: Path,
    limit: int = 0,
) -> dict:
    """Scan audio folders, create .lab and _text_cn.txt files.

    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "created": 0, "skipped": 0, "errors": 0}

    # Build num → text lookup from JSONL
    # JSONL line N (0-indexed) = audio number N (0-padded to 6 digits)
    print(f"JSONL has {len(jsonl_data)} entries")

    for folder_name in sorted(os.listdir(str(audio_root))):
        folder_path = audio_root / folder_name
        if not folder_path.is_dir():
            continue

        wav_files = sorted(folder_path.glob("*.wav"))
        if not wav_files:
            print(f"  SKIP {folder_name}: no .wav files")
            continue

        print(f"\n  {folder_name}: {len(wav_files)} .wav files")

        for wav_path in wav_files:
            stem = wav_path.stem  # e.g. "036000_弹幕互动_回应吐槽弹幕"
            stats["total"] += 1

            lab_path = output_dir / f"{stem}.lab"
            txt_path = output_dir / f"{stem}_text_cn.txt"
            raw_path = output_dir / f"{stem}_text_raw.txt"

            if lab_path.exists() and txt_path.exists():
                stats["skipped"] += 1
                continue

            # Extract audio number from filename prefix
            try:
                audio_num = int(stem.split('_')[0])
            except (ValueError, IndexError):
                print(f"    WARNING: cannot parse number from {stem}")
                stats["errors"] += 1
                continue

            if audio_num < 0 or audio_num >= len(jsonl_data):
                print(f"    WARNING: {stem} num={audio_num} out of range (0-{len(jsonl_data)-1})")
                stats["errors"] += 1
                continue

            entry = jsonl_data[audio_num]
            raw_text = entry.get("text", "").strip()

            if not raw_text:
                print(f"    WARNING: {stem} has empty text")
                stats["errors"] += 1
                continue

            # Build .lab content (pinyin tokens)
            lab_content = make_lab_content(raw_text)

            if not lab_content.strip():
                print(f"    WARNING: {stem} produced empty .lab from: {raw_text[:50]}...")
                stats["errors"] += 1
                continue

            # Write files
            try:
                lab_path.write_text(lab_content + "\n", encoding="utf-8")
                txt_path.write_text(raw_text + "\n", encoding="utf-8")
                raw_path.write_text(raw_text + "\n", encoding="utf-8")

                # Also write {stem}.txt alongside audio for NVASR reference-text mode
                if args.write_ref_txt:
                    ref_txt_path = wav_path.with_suffix('.txt')
                    if not ref_txt_path.exists() or args.overwrite:
                        ref_txt_path.write_text(raw_text + "\n", encoding="utf-8")

                stats["created"] += 1
            except OSError as e:
                print(f"    ERROR writing {stem}: {e}")
                stats["errors"] += 1
                continue

            if stats["created"] % 1000 == 0:
                print(f"    ... {stats['created']} created, {stats['skipped']} skipped")

            if limit > 0 and stats["created"] >= limit:
                print(f"    LIMIT reached ({limit})")
                break

        if limit > 0 and stats["created"] >= limit:
            break

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare English TTS audio for MFA pipeline (no ASR)")
    parser.add_argument("--jsonl", required=True,
                        help="Path to generated_scripts.jsonl")
    parser.add_argument("--audio-root", required=True,
                        help="Root directory containing speaker subfolders with .wav files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for .lab and _text_cn.txt files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files to create (0=all)")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Specific folders to process (default: all)")
    parser.add_argument("--write-ref-txt", action="store_true",
                        help="Also write {stem}.txt alongside each WAV for NVASR reference-text mode")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing reference .txt files")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    audio_root = Path(args.audio_root)
    output_dir = Path(args.output_dir)

    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)
    if not audio_root.exists():
        print(f"ERROR: Audio root not found: {audio_root}")
        sys.exit(1)

    # Load JSONL
    print(f"Loading JSONL: {jsonl_path}")
    jsonl_data = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jsonl_data.append(json.loads(line))

    print(f"Loaded {len(jsonl_data)} entries")

    # Filter folders if specified
    if args.folders:
        # Temporarily limit to specified folders
        import tempfile
        import shutil
        # Just validate folders exist
        for fn in args.folders:
            fp = audio_root / fn
            if not fp.is_dir():
                print(f"WARNING: folder not found: {fp}")

    stats = process_folders(audio_root, jsonl_data, output_dir, args.limit)

    print(f"\n{'='*50}")
    print(f"  Done! total={stats['total']}, created={stats['created']}, "
          f"skipped={stats['skipped']}, errors={stats['errors']}")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
