#!/usr/bin/env python3
"""
Final TextGrid cleanup — runs after all pipeline processing is complete.

Applies three transformations to every final TextGrid:
  1. Tier 1 (raw_text): prepend ``<sp1>`` if not already present.
  2. Tiers 2–5: rename the **first** ``<spN>`` interval (any N) to ``<sp1>``.
  3. All tiers: wrap bare NVV labels with angle brackets,
     e.g. ``BREATHING`` → ``<BREATHING>``.

Processes BOTH ``output/`` (passed QC) and ``filtered/`` (failed QC).
Filtered files are written to a ``filtered/`` subdirectory so you can
still review them with proper NVV bracketing and silence normalization.

Usage:
  python finalize_textgrids.py --input-dir output/ --filtered-dir filtered/ --output-dir finalized/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from postprocess_textgrids import parse_textgrid, write_textgrid

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
    return (text.startswith("<sp") and text.endswith(">")
            and len(text) == 5 and text[3].isdigit())


def process_textgrid(input_path: Path, output_path: Path) -> bool:
    try:
        tg = parse_textgrid(input_path)
    except Exception as exc:
        print(f"  SKIP {input_path.name}: parse error ({exc})")
        return False

    n_tiers = len(tg.tiers)
    if n_tiers < 1:
        return False

    # Pass: wrap bare NVV names with < > in every tier
    for tier in tg.tiers:
        for iv in tier.intervals:
            if iv.text and is_nvv(iv.text):
                iv.text = f"<{iv.text}>"

    # Pass: tier 1 → prepend <sp1>
    raw = tg.tiers[0]
    if raw.intervals:
        first_text = raw.intervals[0].text.strip()
        if first_text and not first_text.startswith("<sp"):
            raw.intervals[0].text = f"<sp1>{first_text}"

    # Pass: tiers 2-5 → first <spN> → <sp1>
    for t_idx in range(1, min(n_tiers, 5)):
        tier = tg.tiers[t_idx]
        for iv in tier.intervals:
            if is_silence(iv.text):
                iv.text = "<sp1>"
                break
        else:
            for iv in tier.intervals:
                if iv.text in {"<eps>", "sil", "sp"}:
                    continue
                if iv.text.startswith("sp") and len(iv.text) == 3 and iv.text[2].isdigit():
                    iv.text = "<sp1>"
                    break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_textgrid(tg, output_path)
    return True


def finalize_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> tuple[int, int]:
    """Process all TextGrids in *input_dir*, write to *output_dir*.
    Returns (done, skipped)."""
    if not input_dir.is_dir():
        return 0, 0
    textgrids = sorted(input_dir.glob("*.TextGrid"))
    if not textgrids:
        return 0, 0

    done = skipped = 0
    for tg_path in textgrids:
        out_path = output_dir / tg_path.name
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        if process_textgrid(tg_path, out_path):
            done += 1
    return done, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Final TextGrid cleanup: <sp1> prefix, normalize first silence, "
                    "wrap NVV labels with <>.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory with OK (passed-QC) TextGrids.")
    parser.add_argument("--filtered-dir", default=None,
                        help="Directory with filtered (failed-QC) TextGrids.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write finalized TextGrids.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    filtered_dir = Path(args.filtered_dir) if args.filtered_dir else None

    if not input_dir.is_dir() and not (filtered_dir and filtered_dir.is_dir()):
        print(f"ERROR: no input TextGrids found")
        sys.exit(1)

    total_done = 0
    total_skipped = 0

    # OK files → finalized/ root
    ok_out = output_dir
    d, s = finalize_dir(input_dir, ok_out, args.overwrite)
    total_done += d
    total_skipped += s
    if d or s:
        print(f"  OK:       {d} finalized" +
              (f" ({s} skipped, use --overwrite)" if s else "") +
              f" → {ok_out}")

    # Filtered files → finalized/filtered/
    if filtered_dir:
        filt_out = output_dir / "filtered"
        d, s = finalize_dir(filtered_dir, filt_out, args.overwrite)
        total_done += d
        total_skipped += s
        if d or s:
            print(f"  Filtered: {d} finalized" +
                  (f" ({s} skipped, use --overwrite)" if s else "") +
                  f" → {filt_out}")

    print(f"  Total finalized: {total_done}" +
          (f" ({total_skipped} skipped)" if total_skipped else ""))


if __name__ == "__main__":
    main()
