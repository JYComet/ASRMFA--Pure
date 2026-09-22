"""Retire GAMEDATA reference texts whose localization macros cannot be resolved.

A handful of authority files carry constructs whose value only exists at game
runtime -- ``{TEXTJOIN#54}`` is the player-chosen name of the pet, ``{scenevar(CCode)}``
is a scene variable, ``{REGEX#UNICODE[E001]}`` is a rich-text substitution.  The
spoken audio never contains the literal construct and there is no in-corpus
evidence that recovers it, so the text cannot be repaired.  The owner's decision
is to remove the reference text from the source corpus while keeping the audio,
which turns those items into no-reference (fallback) items on any future run.

The audio is never touched: a source ``.wav`` whose ``.txt`` is gone is still a
valid fallback item.  Nothing already published is modified.

Read-only by default.  ``--apply`` copies each file into ``--quarantine`` first,
fsyncs it, and only then unlinks the original, so ``--restore`` can put the
corpus back byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

UNRECOGNIZED = "unrecognized"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def load_targets(inventory: Path, withheld: Path) -> list[dict]:
    """Pair each unresolvable withheld row with its frozen source record."""
    items = {item["run_stem"]: item
             for item in json.loads(inventory.read_text(encoding="utf-8"))["items"]}
    targets = []
    for line in withheld.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not any(UNRECOGNIZED in slot.get("reason", "") for slot in row["slots"]):
            continue
        item = items.get(row["run_stem"])
        if item is None or not item.get("reference_path"):
            raise ValueError(f"withheld row has no frozen reference: {row['run_stem']}")
        targets.append({
            "run_stem": row["run_stem"], "game": row.get("game"),
            "speaker": item.get("speaker"),
            "reference_path": item["reference_path"],
            "reference_sha256": item.get("reference_sha256"),
            "audio_path": item.get("source_path"),
            "audio_sha256": item.get("audio_sha256"),
            "macros": [slot["literal"] for slot in row["slots"]
                       if UNRECOGNIZED in slot.get("reason", "")],
        })
    return sorted(targets, key=lambda row: row["run_stem"])


def plan_row(target: dict, quarantine: Path) -> dict:
    """Verify one target and describe what retirement would do to it."""
    source, audio = Path(target["reference_path"]), Path(target["audio_path"])
    stored = quarantine / f"{target['run_stem']}.txt"
    row = dict(target, quarantine_path=str(stored))
    if source.is_symlink() or (source.exists() and not source.is_file()):
        raise ValueError(f"unsafe source text: {source}")
    if audio.is_symlink() or not audio.is_file():
        raise ValueError(f"missing audio, refusing to retire {source}")
    if target["audio_sha256"] and _sha256(audio) != target["audio_sha256"]:
        raise ValueError(f"audio hash drift: {audio}")
    if source.is_file():
        digest = _sha256(source)
        if digest != target["reference_sha256"]:
            raise ValueError(f"reference hash drift: {source}")
        row["state"] = "to_retire"
        row["text_sha256"] = digest
        row["content"] = source.read_text(encoding="utf-8")
    elif stored.is_file() and _sha256(stored) == target["reference_sha256"]:
        row["state"] = "already_retired"
        row["text_sha256"] = target["reference_sha256"]
    else:
        raise ValueError(f"source text missing and not quarantined: {source}")
    return row


def retire(row: dict, quarantine: Path) -> None:
    """Quarantine the text, then unlink the original.  Audio is left alone."""
    if row["state"] != "to_retire":
        return
    quarantine.mkdir(parents=True, exist_ok=True)
    source, stored = Path(row["reference_path"]), Path(row["quarantine_path"])
    temporary = stored.with_name(stored.name + ".tmp")
    temporary.write_text(row["content"], encoding="utf-8")
    _fsync(temporary)
    if _sha256(temporary) != row["text_sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"quarantine copy digest mismatch: {source}")
    os.replace(temporary, stored)
    _fsync(stored)
    source.unlink()
    if source.exists():
        raise ValueError(f"source text survived unlink: {source}")
    if not Path(row["audio_path"]).is_file():
        raise ValueError(f"audio vanished during retirement: {row['audio_path']}")


def restore(row: dict) -> None:
    """Put a quarantined text back, refusing to overwrite anything."""
    source, stored = Path(row["reference_path"]), Path(row["quarantine_path"])
    if source.exists():
        raise ValueError(f"refusing to overwrite existing text: {source}")
    if not stored.is_file() or _sha256(stored) != row["text_sha256"]:
        raise ValueError(f"quarantine copy missing or drifted: {stored}")
    temporary = source.with_name(source.name + ".tmp")
    shutil.copyfile(stored, temporary)
    _fsync(temporary)
    os.replace(temporary, source)
    _fsync(source)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--withheld", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--apply", action="store_true",
                        help="without this the run only reports the plan")
    parser.add_argument("--restore", action="store_true",
                        help="copy quarantined texts back to the corpus")
    args = parser.parse_args(argv)

    quarantine, receipt = Path(args.quarantine), Path(args.receipt)
    targets = load_targets(Path(args.inventory), Path(args.withheld))
    rows = [plan_row(target, quarantine) for target in targets]
    print(json.dumps({"targets": len(rows),
                      "states": {state: sum(row["state"] == state for row in rows)
                                 for state in ("to_retire", "already_retired")}},
                     ensure_ascii=False))

    if args.restore:
        for row in rows:
            if Path(row["reference_path"]).exists():
                continue
            restore(row)
        print(json.dumps({"restored": len(rows)}, ensure_ascii=False))
        return 0

    if args.apply:
        for row in rows:
            retire(row, quarantine)
    payload = {
        "schema": "qwen3-reference-source-retirement-v1",
        "applied": bool(args.apply),
        "quarantine_root": str(quarantine),
        "counts": {"targets": len(rows),
                   "retired": sum(row["state"] == "to_retire" for row in rows),
                   "already_retired": sum(row["state"] == "already_retired" for row in rows)},
        "items": [{key: row[key] for key in
                   ("run_stem", "game", "speaker", "reference_path", "audio_path",
                    "text_sha256", "audio_sha256", "macros", "quarantine_path")}
                  for row in rows],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
