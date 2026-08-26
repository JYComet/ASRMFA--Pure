"""Manifest-bound CTC cache isolation for concurrent pipelined batches."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import streaming_pipeline as stream


def _workspace(root: Path, stem: str, identity: str) -> tuple[Path, Path]:
    workspace = root / stem
    raw = workspace / "ctc_pretg" / stream.CTC_RAW_MANIFEST_NAME
    adjusted = workspace / "ctc_pretg_adj"
    raw.parent.mkdir(parents=True)
    adjusted.mkdir(parents=True)
    raw.write_text(json.dumps({"identity": identity, "stems": [stem]}),
                   encoding="utf-8")
    (adjusted / f"{stem}.lab").write_text(stem, encoding="utf-8")
    (adjusted / stream.CTC_WORK_RECEIPT_NAME).write_text(
        json.dumps({
            "raw_manifest": {
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "identity": identity,
            },
            "work_root": str(adjusted),
        }),
        encoding="utf-8",
    )
    return workspace, raw


def test_persist_keeps_sibling_batch_bundles(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "validate_ctc_raw_manifest", lambda _path: [])
    monkeypatch.setattr(
        stream, "validate_ctc_work_receipt", lambda _path, _raw: [])
    nas = tmp_path / "nas"
    first, first_raw = _workspace(tmp_path / "local", "a", "identity-a")
    second, second_raw = _workspace(tmp_path / "local", "b", "identity-b")

    assert stream._persist_ctc_adj_cache(first, nas, first_raw)
    assert stream._persist_ctc_adj_cache(second, nas, second_raw)

    bundles = sorted((nas / "ctc_pretg_adj_batches").iterdir())
    assert len(bundles) == 2
    assert {path.name for bundle in bundles for path in bundle.glob("*.lab")} == {
        "a.lab", "b.lab"}
    assert not (nas / "ctc_pretg_adj").exists()


def test_restore_selects_only_matching_manifest_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "validate_ctc_raw_manifest", lambda _path: [])
    monkeypatch.setattr(
        stream, "validate_ctc_work_receipt", lambda _path, _raw: [])
    nas = tmp_path / "nas"
    first, first_raw = _workspace(tmp_path / "source", "a", "identity-a")
    second, second_raw = _workspace(tmp_path / "source", "b", "identity-b")
    assert stream._persist_ctc_adj_cache(first, nas, first_raw)
    assert stream._persist_ctc_adj_cache(second, nas, second_raw)

    restored, restored_raw = _workspace(
        tmp_path / "restore", "a", "identity-a")
    # Restore must replace this sentinel with only batch a's matching bundle.
    (restored / "ctc_pretg_adj" / "sentinel").write_text("x", encoding="utf-8")
    assert stream._restore_ctc_adj_cache(restored, nas, restored_raw)
    assert (restored / "ctc_pretg_adj" / "a.lab").is_file()
    assert not (restored / "ctc_pretg_adj" / "b.lab").exists()
