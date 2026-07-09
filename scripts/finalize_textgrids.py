#!/usr/bin/env python3
"""
Final TextGrid cleanup — runs after all pipeline processing is complete.

Applies three transformations to every final TextGrid:
  1. Tier 1 (raw_text): prepend ``<sp1>`` if not already present.
  2. Tiers 2–5: rename the **first** ``<spN>`` interval (any N) to ``<sp1>``.
  3. All tiers: wrap bare NVV labels with angle brackets,
     e.g. ``BREATHING`` → ``<BREATHING>``.

Usage:
  python finalize_textgrids.py --input-dir output/ --output-dir finalized/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the TextGrid parser / writer from postprocess_textgrids
from postprocess_textgrids import parse_textgrid, write_textgrid, SILENCE_LABELS

# Full set of NVV token names (uppercase as they appear in TextGrids)
NVV_NAMES: set[str] = {
    "BREATHING", "LAUGHTER", "BURP", "COUGH", "CRYING", "GROAN",
    "HISS", "HUM", "SHH", "SIGH", "SNEEZE", "SNIFF", "SNORE",
    "TSK", "UHM", "WHISTLE", "YAWN",
    "QUESTION-YI", "QUESTION-EN", "QUESTION-OH", "QUESTION-AH",
    "QUESTION-EI", "QUESTION-HUH",
    "SURPRISE-OH", "SURPRISE-AH", "SURPRISE-WA", "SURPRISE-YO",
    "CONFIRMATION-EN", "DISSATISFACTION-HNN",
}


def is_nvv(text: str) -> bool:
    return text in NVV_NAMES


def is_silence(text: str) -> bool:
    """True for any ``<spN>`` silence label."""
    return (text.startswith("<sp") and text.endswith(">")
            and len(text) == 5 and text[3].isdigit())


def process_textgrid(input_path: Path, output_path: Path) -> bool:
    """Apply the three finalization passes. Returns True on success."""
    try:
        tg = parse_textgrid(input_path)
    except Exception as exc:
        print(f"  SKIP {input_path.name}: parse error ({exc})")
        return False

    n_tiers = len(tg.tiers)
    if n_tiers < 1:
        return False

    # ── Pass: wrap bare NVV names with < > in every tier ────────────────
    for tier in tg.tiers:
        for iv in tier.intervals:
            bare = iv.text
            if bare and is_nvv(bare):
                iv.text = f"<{bare}>"

    # ── Pass: tier 1 raw_text → prepend <sp1> ──────────────────────────
    raw = tg.tiers[0]
    if raw.intervals:
        first_text = raw.intervals[0].text.strip()
        if first_text and not first_text.startswith("<sp"):
            raw.intervals[0].text = f"<sp1>{first_text}"

    # ── Pass: tiers 2-5 → first <spN> → <sp1> ─────────────────────────
    for t_idx in range(1, min(n_tiers, 5)):
        tier = tg.tiers[t_idx]
        replaced = False
        for iv in tier.intervals:
            if is_silence(iv.text):
                iv.text = "<sp1>"
                replaced = True
                break
        if not replaced:
            # Also check if it's a bare sp0/sp1/sp2/sp3 without brackets
            for iv in tier.intervals:
                if iv.text in {"<eps>", "sil", "sp"}:
                    continue
                if iv.text.startswith("sp") and len(iv.text) == 3 and iv.text[2].isdigit():
                    iv.text = "<sp1>"
                    replaced = True
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_textgrid(tg, output_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Final TextGrid cleanup: <sp1> prefix, normalize first silence, "
                    "wrap NVV labels with <>.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing output TextGrids.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write final TextGrids.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    textgrids = sorted(input_dir.glob("*.TextGrid"))
    if not textgrids:
        print(f"  No .TextGrid files found in {input_dir}")
        return

    done = 0
    skipped = 0
    for tg_path in textgrids:
        out_path = output_dir / tg_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if process_textgrid(tg_path, out_path):
            done += 1

        # Show one sample of the transformation
        if done == 1 and not args.overwrite:
            pass  # suppress noisy first-time logging

    print(f"  Finalized {done} TextGrid(s){' (' + str(skipped) + ' skipped, use --overwrite)' if skipped else ''}"
          f" → {output_dir}")


if __name__ == "__main__":
    main()
