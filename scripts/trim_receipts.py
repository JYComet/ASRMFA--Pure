"""Content-bound receipts for the rebuildable audio preprocessing stage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RECEIPT_NAME = ".trim_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_audio(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"invalid trim input directory: {root}")
    paths = sorted((path for path in root.rglob("*")
                    if path.suffix.lower() == ".wav"), key=lambda p: p.as_posix())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("trim WAV inputs must be ordinary files")
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("trim input has duplicate WAV stems")
    if not paths:
        raise ValueError("trim input inventory is empty")
    return paths


def file_record(path: Path, root: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing trim artifact: {path}")
    return {"stem": path.stem, "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size, "sha256": sha256(path)}


def input_identity(root: Path, parameters: dict) -> dict:
    return {
        "schema": "trim-input-identity-v1", "input_root": str(root.resolve()),
        "parameters": parameters,
        "implementation": {name: sha256(Path(__file__).with_name(name))
                           for name in ("trim_silence_batch.py", "audio_utils.py", "trim_receipts.py")},
        "inputs": [file_record(path, root) for path in collect_audio(root)],
    }


def verify_complete(root: Path, identity: dict) -> dict:
    path = root / RECEIPT_NAME
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("trim output lacks a trusted receipt; use --overwrite or a fresh workspace")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "trim-stage-receipt-v1" or receipt.get("identity") != identity:
        raise ValueError("trim input/configuration changed; use --overwrite or a fresh workspace")
    if receipt.get("status") != "COMPLETE" or receipt.get("failures"):
        raise ValueError("trim stage is incomplete; use --overwrite to retry")
    observed = [file_record(path, root) for path in collect_audio(root)]
    if observed != receipt.get("outputs"):
        raise ValueError("trim output content or membership changed; use --overwrite to rebuild")
    if [row["path"] for row in observed] != [row["path"] for row in identity["inputs"]]:
        raise ValueError("trim output does not cover the frozen input inventory")
    return receipt


def write_receipt(root: Path, identity: dict, outputs: list[dict], failures: list[dict]) -> dict:
    receipt = {"schema": "trim-stage-receipt-v1",
               "status": "PARTIAL" if failures else "COMPLETE", "identity": identity,
               "inputs": identity["inputs"], "outputs": sorted(outputs, key=lambda row: row["path"]),
               "failures": sorted(failures, key=lambda row: row["stem"])}
    target = root / RECEIPT_NAME
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return receipt
