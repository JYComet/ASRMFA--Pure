"""Small journaled transaction helper for full-corpus file publication."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ValueError(f"unsafe directory component: {cursor}")
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_dir(directory.parent)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _within(path: Path, root: Path) -> bool:
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    return root == path or root in path.parents


def _restore(journal: dict, journal_path: Path, rollback_root: Path,
             allowed_roots: tuple[Path, ...]) -> None:
    if journal.get("schema") != "full-corpus-publication-transaction-v1":
        raise ValueError(f"unknown publication journal schema: {journal_path}")
    for entry in reversed(journal.get("entries", [])):
        target = Path(entry["target"])
        if not any(_within(target, root) for root in allowed_roots):
            raise ValueError(f"publication journal target escapes output root: {target}")
        if target.is_symlink():
            raise ValueError(f"symlink target blocks publication recovery: {target}")
        old_hash = entry.get("old_sha256")
        new_hash = entry["new_sha256"]
        if target.is_file():
            current_hash = _sha256(target)
        elif target.exists():
            raise ValueError(f"non-file target blocks publication recovery: {target}")
        else:
            current_hash = None
        backup = Path(entry["rollback"]) if entry.get("rollback") else None
        if backup is not None and (not _within(backup, rollback_root) or backup.is_symlink()):
            raise ValueError(f"unsafe publication rollback artifact: {backup}")
        if old_hash is None:
            if current_hash == new_hash:
                target.unlink()
                _fsync_dir(target.parent)
            elif current_hash is not None:
                raise ValueError(f"unexpected target contents during recovery: {target}")
        elif current_hash == old_hash:
            continue
        elif current_hash in {new_hash, None}:
            if backup is None or not backup.is_file() or _sha256(backup) != old_hash:
                raise ValueError(f"verified rollback copy missing during recovery: {target}")
            _mkdir_durable(target.parent)
            temporary = target.with_name(target.name + ".recover-" + journal["transaction_id"])
            shutil.copyfile(backup, temporary)
            _fsync_file(temporary)
            os.replace(temporary, target)
            _fsync_dir(target.parent)
        else:
            raise ValueError(f"unexpected target contents during recovery: {target}")
    for entry in journal.get("entries", []):
        temporary = Path(entry["temporary"])
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
    journal_path.unlink()
    _fsync_dir(journal_path.parent)


def recover_transactions(rollback_root: Path, allowed_roots) -> None:
    rollback_root = Path(rollback_root).resolve(strict=False)
    transaction_root = rollback_root / ".transactions"
    if transaction_root.is_symlink():
        raise ValueError(f"publication transaction directory is a symlink: {transaction_root}")
    if not transaction_root.exists():
        return
    roots = tuple(Path(root).resolve(strict=False) for root in allowed_roots)
    for journal_path in sorted(transaction_root.glob("*.json")):
        if journal_path.is_symlink() or not journal_path.is_file():
            raise ValueError(f"unsafe publication journal: {journal_path}")
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"unreadable publication journal: {journal_path}") from exc
        _restore(journal, journal_path, rollback_root, roots)


def publish_files(entries, rollback_root: Path, allowed_roots) -> list[dict]:
    """Prepare every artifact, commit replacements, or restore the whole group.

    Each entry has source, target, stem, kind, optional expected_sha256, and
    rollback_parts.  Returned rows keep the established receipt field names.
    """
    rollback_root = Path(rollback_root).resolve(strict=False)
    roots = tuple(Path(root).resolve(strict=False) for root in allowed_roots)
    recover_transactions(rollback_root, roots)
    rows = [dict(row) for row in entries]
    transaction_id = uuid.uuid4().hex
    transaction_root = rollback_root / ".transactions"
    _mkdir_durable(transaction_root)
    if transaction_root.is_symlink():
        raise ValueError(f"publication transaction directory is a symlink: {transaction_root}")
    journal_path = transaction_root / f"{transaction_id}.json"
    seen = set()
    prepared = []
    for index, row in enumerate(rows):
        source, target = Path(row["source"]), Path(row["target"])
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"unsafe publication source: {source}")
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"unsafe publication target: {target}")
        if not any(_within(target, root) for root in roots):
            raise ValueError(f"publication target escapes output root: {target}")
        canonical = target.resolve(strict=False)
        if canonical in seen:
            raise ValueError(f"duplicate publication target: {target}")
        seen.add(canonical)
        source_hash = _sha256(source)
        expected = row.get("expected_sha256")
        if expected and source_hash != expected:
            raise ValueError(f"staged publication source digest changed: {source}")
        old_hash = _sha256(target) if target.is_file() else None
        rollback = None
        if old_hash is not None:
            rollback_dir = rollback_root.joinpath(*row.get("rollback_parts", ()))
            historical = rollback_dir / target.name
            if not _within(historical, rollback_root):
                raise ValueError("rollback target escapes rollback root")
            if historical.is_symlink() or (historical.exists() and not historical.is_file()):
                raise ValueError(f"unsafe historical rollback artifact: {historical}")
            rollback = (rollback_dir / f"{target.name}.history-{transaction_id}"
                        if historical.exists() else historical)
        temporary = target.with_name(target.name + f".txn-{transaction_id}.new")
        backup_temporary = (rollback.with_name(rollback.name + f".txn-{transaction_id}.tmp")
                            if rollback is not None else None)
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f"publication temporary already exists: {temporary}")
        if backup_temporary is not None and (backup_temporary.exists() or backup_temporary.is_symlink()):
            raise ValueError(f"rollback temporary already exists: {backup_temporary}")
        prepared.append({**row, "source": str(source), "target": str(target),
                         "new_sha256": source_hash, "old_sha256": old_hash,
                         "rollback": str(rollback) if rollback else None,
                         "temporary": str(temporary),
                         "backup_temporary": str(backup_temporary) if backup_temporary else None})

    journal_written = False
    try:
        # Preserve historical rollback artifacts while giving each transaction
        # its own immutable snapshot name.
        for row in prepared:
            target, source = Path(row["target"]), Path(row["source"])
            _mkdir_durable(target.parent)
            temporary = Path(row["temporary"])
            shutil.copyfile(source, temporary)
            _fsync_file(temporary)
            if _sha256(temporary) != row["new_sha256"]:
                raise ValueError(f"publication copy digest mismatch: {source}")
            rollback = Path(row["rollback"]) if row["rollback"] else None
            if rollback is not None:
                _mkdir_durable(rollback.parent)
                backup_temp = Path(row["backup_temporary"])
                shutil.copyfile(target, backup_temp)
                _fsync_file(backup_temp)
                if _sha256(backup_temp) != row["old_sha256"]:
                    raise ValueError(f"rollback copy digest mismatch: {target}")
        for row in prepared:
            target = Path(row["target"])
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f"publication target changed during prepare: {target}")
            current_hash = _sha256(target) if target.is_file() else None
            if current_hash != row["old_sha256"]:
                raise ValueError(f"publication target changed during prepare: {target}")
        for row in prepared:
            rollback = Path(row["rollback"]) if row["rollback"] else None
            if rollback is not None:
                os.replace(Path(row["backup_temporary"]), rollback)
                _fsync_dir(rollback.parent)
        journal = {"schema": "full-corpus-publication-transaction-v1",
                   "transaction_id": transaction_id,
                   "entries": [{key: row[key] for key in (
                       "target", "rollback", "old_sha256", "new_sha256", "temporary")}
                               for row in prepared]}
        _atomic_json(journal_path, journal)
        journal_written = True
        for row in prepared:
            target, temporary = Path(row["target"]), Path(row["temporary"])
            os.replace(temporary, target)
            _fsync_dir(target.parent)
    except BaseException:
        if journal_written:
            _restore(journal, journal_path, rollback_root, roots)
        else:
            for row in prepared:
                for name in ("temporary", "backup_temporary"):
                    temporary = Path(row[name]) if row.get(name) else None
                    if temporary is not None and temporary.is_file() and not temporary.is_symlink():
                        temporary.unlink()
        raise
    journal_path.unlink()
    _fsync_dir(transaction_root)
    return [{"stem": row.get("stem"), "kind": row.get("kind"),
             **({"speaker": row["speaker"]} if row.get("speaker") is not None else {}),
             "target": row["target"], "rollback": row["rollback"],
             "old_sha256": row["old_sha256"], "new_sha256": row["new_sha256"]}
            for row in prepared]
