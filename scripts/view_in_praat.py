#!/usr/bin/env python3
"""
Select a TextGrid and open it with matching audio in Praat.

Usage:
  python scripts/view_in_praat.py                   # list output/ TextGrids
  python scripts/view_in_praat.py --dir filtered    # list filtered/ TextGrids
  python scripts/view_in_praat.py --dir aligned     # list raw MFA TextGrids
"""

import argparse
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_praat() -> Path | None:
    import shutil
    # Check project root, then PATH
    candidates = [PROJECT_ROOT / "Praat.exe"]
    found = shutil.which("Praat") or shutil.which("praat")
    if found:
        candidates.insert(0, Path(found))
    for c in candidates:
        if c.exists():
            return c
    return None


def find_wav(textgrid_path: Path, wav_dirs: list[Path]) -> Path | None:
    """Find matching wav by stem. Tries direct, clip subfolder, then segments/."""
    stem = textgrid_path.stem
    for d in wav_dirs:
        candidate = d / f"{stem}.wav"
        if candidate.exists():
            return candidate
    clip = re.sub(r"_seg\d+$", "", stem)
    if clip != stem:
        for d in wav_dirs:
            for sub in ("", "segments/"):
                candidate = d / clip / sub / f"{stem}.wav"
                if candidate.exists():
                    return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="View TextGrid + audio in Praat.")
    parser.add_argument("--dir", type=str, default="output",
                        help="Directory name or path (output/filtered/aligned)")
    parser.add_argument("--wav-dir", type=str, default=None,
                        help="Wav directory (default: corpus_clean/wav)")
    parser.add_argument("--praat", type=str, default=None, help="Path to Praat.exe")
    args = parser.parse_args()

    tg_dir = Path(args.dir)
    if not tg_dir.is_absolute():
        tg_dir = PROJECT_ROOT / args.dir
    if not tg_dir.exists():
        print(f"Directory not found: {tg_dir}")
        return

    # Search workspace first, then local corpus_clean
    ws = PROJECT_ROOT.parent / "workspace"
    wav_dirs = []
    if ws.exists():
        wav_dirs.append(ws / "corpus_clean" / "wav")
    wav_dirs.append(PROJECT_ROOT / "corpus_clean" / "wav")
    if args.wav_dir:
        wav_dirs.insert(0, Path(args.wav_dir))

    tg_files = sorted(tg_dir.glob("*.TextGrid"))
    if not tg_files:
        print(f"No .TextGrid files in {tg_dir}")
        return

    print(f"\n  TextGrid files in {tg_dir.name}/:\n")
    for i, f in enumerate(tg_files, 1):
        wav = find_wav(f, wav_dirs)
        print(f"  [{i:3d}] {'+wav' if wav else '-wav':5s}  {f.name}")

    print(f"\n  Enter number (q=quit): ", end="")
    try:
        choice = input().strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice.lower() in ("q", "quit", ""):
        return

    try:
        idx = int(choice) - 1
        tg_path = tg_files[idx]
    except (ValueError, IndexError):
        print(f"Invalid: {choice}")
        return

    wav_path = find_wav(tg_path, wav_dirs)
    if not wav_path:
        print(f"No matching wav for: {tg_path.name}")
        return

    praat = Path(args.praat) if args.praat else find_praat()
    if not praat:
        print("Praat.exe not found. Place it in the project root or use --praat PATH")
        return

    print(f"\n  TextGrid: {tg_path.name}")
    print(f"  Audio:    {wav_path.name}")
    subprocess.Popen([str(praat), "--open", str(wav_path), str(tg_path)])
    print("  Done.")


if __name__ == "__main__":
    main()
