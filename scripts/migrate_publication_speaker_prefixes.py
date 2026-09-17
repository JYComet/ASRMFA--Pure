#!/usr/bin/env python3
"""Move completed flat publications into the canonical game speaker namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from scripts.speaker_namespace import publication_speaker
except ModuleNotFoundError:
    from speaker_namespace import publication_speaker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _move_bound_file(source: Path, target: Path, digest: str | None,
                     *, apply: bool) -> str:
    if source.absolute() == target.absolute():
        return "unchanged"
    if target.exists():
        if not target.is_file() or (digest and _sha256(target) != digest):
            raise FileExistsError(f"conflicting namespaced target: {target}")
        if source.exists() and digest and _sha256(source) != digest:
            raise ValueError(f"source digest mismatch: {source}")
        if apply and source.exists():
            source.unlink()
        return "deduplicated"
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"missing publication source: {source}")
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    return "moved"


def migrate(run_root: Path, output_root: Path, *, apply: bool = False,
            workers: int = 32) -> dict:
    run_root, output_root = Path(run_root), Path(output_root)
    inventory = json.loads((run_root / "frozen_inventory.json").read_text(
        encoding="utf-8"))
    by_stem = {row["run_stem"]: row for row in inventory.get("items", [])}
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    complete = sorted(chunk for chunk, state in status.get("terminal", {}).items()
                      if state == "complete")
    counts = {"chunks": 0, "rows": 0, "moved": 0, "deduplicated": 0,
              "unchanged": 0, "renamed_speakers": 0}
    receipts = []
    moves = []
    for chunk_id in complete:
        receipt_path = run_root / "chunks" / chunk_id / "output_publication.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing completed publication receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipts.append((receipt_path, receipt))
        for row in receipt.get("replacements", []):
            stem = row.get("stem")
            item = by_stem.get(stem)
            if item is None:
                raise ValueError(f"unknown publication stem: {stem}")
            original = item.get("speaker") or "_default"
            speaker = publication_speaker(item.get("game"), original)
            kind_root = output_root / ("_filtered" if row.get("kind") == "filtered" else "")
            target = (kind_root / speaker / f"{stem}.TextGrid").absolute()
            if kind_root.absolute() not in target.parents:
                raise ValueError(f"namespaced target escapes output root: {target}")
            moves.append((row, Path(row.get("target", "")), target,
                          row.get("new_sha256")))
            counts["rows"] += 1
            if row.get("speaker") != speaker:
                counts["renamed_speakers"] += 1
            row["original_speaker"] = original
            row["speaker"] = speaker
            row["target"] = str(target)
            rollback = row.get("rollback")
            if rollback:
                rollback_root = Path(receipt["rollback_root"]) / (
                    "_filtered" if row.get("kind") == "filtered" else "")
                rollback_target = rollback_root / speaker / f"{stem}.TextGrid"
                moves.append((None, Path(rollback), rollback_target,
                              row.get("old_sha256")))
                row["rollback"] = str(rollback_target)
        receipt["speaker_namespace"] = "game-first-two-hanzi-pinyin-initials-v1"
        counts["chunks"] += 1
    def execute(move):
        _, source, target, digest = move
        return _move_bound_file(source, target, digest, apply=apply)
    with ThreadPoolExecutor(max_workers=max(1, workers),
                            thread_name_prefix="speaker-migrate") as pool:
        states = list(pool.map(execute, moves))
    for move, state in zip(moves, states):
        if move[0] is not None:
            counts[state] += 1
    if apply:
        for receipt_path, receipt in receipts:
            _write_json(receipt_path, receipt)
    return {"applied": apply, **counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(migrate(Path(args.run_root), Path(args.output_root),
                             apply=args.apply, workers=args.workers),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
