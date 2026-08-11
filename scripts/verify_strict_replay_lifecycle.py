#!/usr/bin/env python3
"""CLI wrapper for strict-replay lifecycle v4.2.2 validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from verify_strict_replay_english_subset import _lifecycle_overall_failed, _verify_lifecycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict-replay lifecycle ownership/order")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = _verify_lifecycle(args.workspace, args.output)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if _lifecycle_overall_failed(args.output):
        print("ERROR: overall lifecycle verdict failed: strict_ok return_code != 0")
        return 1
    print(f"strict replay lifecycle verified: workspace={args.workspace} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
