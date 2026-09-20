"""Repair GAMEDATA reference-macro leaks in the published 0915ALL corpus.

The published corpus contains authority text that still carries localization
macros (`{NICKNAME}`, `{PLAYERAVATAR#SEXPRO[…]}`, `{RUBY#[S]…}`).  The stager
strips the braces but keeps the ASCII letters, so the macro name reaches the
aligner as a bare token and is G2P'd into nonsense.  ``reference_macro_slots``
now recognises those macros, but a branch that is audibly distinct still needs
evidence, which for this corpus is the voice track itself.

This driver is read-only by default.  ``--plan`` measures the corpus and writes
a manifest for review; nothing is staged or published without ``--apply``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reference_macro_resolutions import (  # noqa: E402
    SlotPlan, build_table, is_actionable, plan_manifest_row, plan_text,
    write_resolutions)

READ_ATTEMPTS = 4


def _read_text(path: str) -> str | None:
    """Read a NAS authority file, retrying the transient failures it emits."""
    for attempt in range(READ_ATTEMPTS):
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            if attempt == READ_ATTEMPTS - 1:
                return None
    return None


def load_inventory(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [item for item in payload["items"] if item.get("reference_path")]


def _scan_row(item: dict) -> dict:
    text = _read_text(item["reference_path"])
    if text is None:
        return {"run_stem": item["run_stem"], "state": "unreadable"}
    plans = plan_text(text, run_stem=item["run_stem"], game=item.get("game"))
    if not plans:
        return {"run_stem": item["run_stem"], "state": "clean"}
    return {
        "run_stem": item["run_stem"], "state": "affected",
        "game": item.get("game"), "speaker": item.get("speaker"),
        "actionable": is_actionable(plans),
        "slots": [plan_manifest_row(plan) for plan in plans],
    }


def scan(inventory: Path, cache: Path, workers: int) -> dict:
    """Scan every reference authority file once, resuming from ``cache``."""
    items = load_inventory(inventory)
    done: dict[str, dict] = {}
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["run_stem"]] = row
    pending = [item for item in items if item["run_stem"] not in done]
    print(f"inventory {len(items)}  cached {len(done)}  pending {len(pending)}",
          flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, row in enumerate(pool.map(_scan_row, pending)):
                done[row["run_stem"]] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if index % 5000 == 0:
                    handle.flush()
                    print(f"scanned {index}/{len(pending)}", flush=True)
    return done


def summarize(rows) -> dict:
    states = Counter(row["state"] for row in rows)
    affected = [row for row in rows if row["state"] == "affected"]
    macros = Counter(slot["macro"] for row in affected for slot in row["slots"])
    kinds = Counter(slot["kind"] for row in affected for slot in row["slots"])
    games = Counter(row.get("game") for row in affected)
    return {
        "scanned": len(rows), "states": dict(states),
        "affected": len(affected),
        "actionable": sum(1 for row in affected if row["actionable"]),
        "withheld": sum(1 for row in affected if not row["actionable"]),
        "macros": dict(macros.most_common()), "kinds": dict(kinds.most_common()),
        "games": dict(games.most_common()),
    }


def build_tables(affected) -> tuple[dict, list]:
    tables, withheld = {}, []
    for row in affected:
        if not row["actionable"]:
            withheld.append({"run_stem": row["run_stem"], "game": row.get("game"),
                             "slots": [s for s in row["slots"] if s["value"] is None]})
            continue
        try:
            tables[row["run_stem"]] = build_table(_Plans(row["slots"]))
        except ValueError as exc:
            withheld.append({"run_stem": row["run_stem"], "game": row.get("game"),
                             "slots": row["slots"], "error": str(exc)})
    return tables, withheld


class _Plans:
    """Rehydrate manifest rows into objects ``build_table`` accepts."""

    def __init__(self, slots):
        self._slots = slots

    def __iter__(self):
        for slot in self._slots:
            yield SlotPlan(
                run_stem=slot["run_stem"], literal=slot["literal"],
                macro=slot["macro"], kind=slot["kind"], value=slot["value"],
                alternatives=tuple(slot["alternatives"]), reason=slot["reason"],
                review_required=slot.get("review_required", False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--plan-root", required=True,
                        help="directory for scan cache and manifest")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.plan_root)
    cache = root / "scan" / "macro_scan.jsonl"
    if not args.summarize_only:
        scan(Path(args.inventory), cache, args.workers)
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = summarize(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    affected = [row for row in rows if row["state"] == "affected"]
    tables, withheld = build_tables(affected)
    (root / "plan").mkdir(parents=True, exist_ok=True)
    if tables:
        written = write_resolutions(root / "plan", tables)
        report["resolutions"] = written
    manifest = root / "plan" / "macro_repair_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in affected:
            if row["actionable"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    withheld_path = root / "plan" / "macro_repair_withheld.jsonl"
    with withheld_path.open("w", encoding="utf-8") as handle:
        for row in withheld:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["manifest"] = {"path": str(manifest), "rows": sum(
        1 for row in affected if row["actionable"])}
    report["withheld_manifest"] = {"path": str(withheld_path), "rows": len(withheld)}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
