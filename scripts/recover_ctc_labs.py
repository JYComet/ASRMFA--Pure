#!/usr/bin/env python3
"""Recover corrupted MFA lab transcripts from validated CTC token bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (
    load_ctc_token_entries,
    read_ctc_textgrid_words,
    rebuild_lab_from_tokens,
    validate_ctc_transcript_bundle,
)

RECOVERY_SCHEMA = "ctc-lab-recovery-v2"
HISTORICAL_RECOVERY_SCHEMA = 1


def recover_directory(ctc_dir: Path, *, apply: bool) -> tuple[dict, int]:
    token_paths = sorted(ctc_dir.glob("*_tokens.jsonl"))
    rows: list[dict] = []
    recovered = unchanged = invalid = 0

    for tokens_path in token_paths:
        stem = tokens_path.name[:-len("_tokens.jsonl")]
        lab_path = ctc_dir / f"{stem}.lab"
        textgrid_path = ctc_dir / f"{stem}.TextGrid"
        row: dict = {"stem": stem, "status": "unknown"}
        try:
            token_words = [
                entry["word"].strip()
                for entry in load_ctc_token_entries(tokens_path)
            ]
            tg_words = read_ctc_textgrid_words(textgrid_path)
            if token_words != tg_words:
                raise ValueError(
                    f"TextGrid/tokens mismatch ({len(tg_words)} != "
                    f"{len(token_words)})"
                )
            lab_words = (
                lab_path.read_text(encoding="utf-8-sig").strip().split()
                if lab_path.exists() else []
            )
            if lab_words == token_words:
                row["status"] = "unchanged"
                unchanged += 1
            else:
                row.update({
                    "status": "would_recover" if not apply else "recovered",
                    "lab_count_before": len(lab_words),
                    "token_count": len(token_words),
                })
                if apply:
                    rebuild_lab_from_tokens(tokens_path, lab_path)
                    errors = validate_ctc_transcript_bundle(
                        ctc_dir, stem, _require_processed=False)
                    if errors:
                        raise ValueError("; ".join(errors))
                    # Invalidate stale normalization marker — the .lab
                    # content changed and downstream must re-validate.
                    _marker = ctc_dir / ".ctc_normalized"
                    if _marker.exists():
                        _marker.unlink()
                recovered += 1
        except (OSError, ValueError) as exc:
            row["status"] = "invalid"
            row["error"] = str(exc)
            invalid += 1
        rows.append(row)

    lab_stems = {path.stem for path in ctc_dir.glob("*.lab")}
    token_stems = {
        path.name[:-len("_tokens.jsonl")] for path in token_paths
    }
    missing_tokens = sorted(lab_stems - token_stems)
    for stem in missing_tokens:
        rows.append({
            "stem": stem,
            "status": "invalid",
            "error": "missing *_tokens.jsonl",
        })
    invalid += len(missing_tokens)

    summary = {
        "schema": RECOVERY_SCHEMA,
        "historical_schema": HISTORICAL_RECOVERY_SCHEMA,
        "mode": "apply" if apply else "dry_run",
        "ctc_dir": str(ctc_dir.resolve()),
        "token_bundles": len(token_paths),
        "recovered_or_would_recover": recovered,
        "unchanged": unchanged,
        "invalid": invalid,
        "rows": rows,
    }
    return summary, (1 if invalid else 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover .lab files only from validated *_tokens.jsonl + CTC "
            "TextGrid words.  Default is a read-only dry run."
        )
    )
    parser.add_argument("--ctc-dir", type=Path, required=True)
    parser.add_argument(
        "--apply", action="store_true",
        help="Atomically replace mismatching labs (use only in an isolated copy).",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if not args.ctc_dir.is_dir():
        print(f"ERROR: CTC directory not found: {args.ctc_dir}")
        return 1

    summary, rc = recover_directory(args.ctc_dir, apply=args.apply)
    manifest = args.manifest
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"CTC lab recovery ({summary['mode']}): "
        f"{summary['token_bundles']} bundles, "
        f"{summary['recovered_or_would_recover']} recover, "
        f"{summary['unchanged']} unchanged, {summary['invalid']} invalid"
    )
    if manifest is not None:
        print(f"Manifest: {manifest}")
    if rc:
        for row in summary["rows"]:
            if row["status"] == "invalid":
                print(f"  - {row['stem']}: {row['error']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
