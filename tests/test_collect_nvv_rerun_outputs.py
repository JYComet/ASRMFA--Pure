"""Integration contracts for the private flat-output NVV collector."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

from collect_nvv_rerun_outputs import collect_nvv_rerun_outputs  # noqa: E402
from pipeline_utils import (  # noqa: E402
    make_pipeline_accounting_receipt,
    write_publish_manifest,
    write_pipeline_accounting_receipt,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _receipt(path: Path, manifest: Path, input_rows: list[dict[str, str]], **changes: object) -> Path:
    run_root = path.parent / "run"
    bound_inputs = [{
        **row,
        "source_wav_sha256": _bytes_sha256(f"wav:{row['stem']}".encode("ascii")),
        "target_sha256": _bytes_sha256(f"wav:{row['stem']}".encode("ascii")),
        "target_wav_path": str(run_root / "inputs" / row["game"] / f"{row['stem']}.wav"),
    } for row in input_rows]
    payload: dict[str, object] = {
        "schema": "gamedata-nvv-rerun-materialization-receipt-v1",
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "inputs": bound_inputs,
        "run_root": str(run_root),
    }
    payload.update(changes)
    return _write_json(path, payload)


def _flat_pair(root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stem}.TextGrid").write_text(f"grid:{stem}", encoding="utf-8")
    (root / f"{stem}.wav").write_bytes(f"wav:{stem}".encode("ascii"))


def _input_wavs(tmp_path: Path) -> Path:
    return tmp_path / "run" / "inputs" / "persona"


def _pipeline_receipt(root: Path, selected: list[str]) -> Path:
    produced = sorted(path.stem for path in root.glob("*.TextGrid")
                      if path.stem in selected)
    receipt = make_pipeline_accounting_receipt(
        source_stems=selected,
        eligible_stems=selected,
        exclusions={},
        output_stems=produced,
        filtered_stems=sorted(set(selected) - set(produced)),
        paths={"output": str(root)},
    )
    write_pipeline_accounting_receipt(root, receipt)
    return root / ".pipeline_run_receipt_v2.json"


def _call(tmp_path: Path, manifest: Path, receipt: Path, *, grids: Path, wavs: Path):
    selected = [json.loads(line)["stem"] for line in manifest.read_text(encoding="utf-8").splitlines()]
    pipeline_receipt = grids / ".pipeline_run_receipt_v2.json"
    if not pipeline_receipt.exists():
        pipeline_receipt = _pipeline_receipt(grids, selected)
    publish_manifest = grids / ".publish_manifest.json"
    if not publish_manifest.exists():
        write_publish_manifest(grids)
    return collect_nvv_rerun_outputs(
        manifest_path=manifest,
        materialization_receipt_path=receipt,
        flat_textgrid_roots={"persona": grids},
        flat_wav_roots={"persona": wavs},
        flat_textgrid_receipts={"persona": pipeline_receipt},
        textgrid_output_root=tmp_path / "collected-grids",
        wav_output_root=tmp_path / "collected-wavs",
        collection_receipt_path=tmp_path / "collection-receipt.json",
    )


def test_collects_only_complete_pairs_into_manifest_speaker_paths_and_records_missing(tmp_path: Path):
    # Removing either source half must make the corresponding key absent from
    # both collected trees while leaving a deterministic missing receipt row.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "complete"},
        {"game": "persona", "speaker": "bob", "stem": "missing"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "complete"},
        {"game": "persona", "speaker": "bob", "stem": "missing"},
    ])
    grids = tmp_path / "flat-grids"
    wavs = _input_wavs(tmp_path)
    _flat_pair(grids, "complete")
    grids.mkdir(exist_ok=True)
    (grids / "missing.TextGrid").write_text("unpaired", encoding="utf-8")
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / "complete.wav").write_bytes(b"wav:complete")

    result = _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)

    assert result["selected_count"] == 2
    assert result["collected_count"] == 1
    assert result["missing_count"] == 1
    assert (tmp_path / "collected-grids" / "persona" / "alice" / "complete.TextGrid").is_file()
    assert (tmp_path / "collected-wavs" / "persona" / "alice" / "complete.wav").is_file()
    assert not (tmp_path / "collected-grids" / "persona" / "bob" / "missing.TextGrid").exists()
    persisted = json.loads((tmp_path / "collection-receipt.json").read_text(encoding="utf-8"))
    assert persisted["missing"] == [{
        "game": "persona", "grid_present": True, "reason": "missing_flat_pair",
        "speaker": "bob", "stem": "missing", "wav_present": False,
    }]


def test_creates_empty_fresh_output_roots_when_every_selected_pair_is_missing(tmp_path: Path):
    # A downstream gate must be able to inspect an all-rejected rerun without
    # treating the collector's empty result as a missing staging directory.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "missing"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "missing"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    grids.mkdir()
    wavs.mkdir(parents=True, exist_ok=True)

    result = _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)

    assert result["collected_count"] == 0
    assert result["missing_count"] == 1
    assert (tmp_path / "collected-grids").is_dir()
    assert (tmp_path / "collected-wavs").is_dir()


def test_rejects_flat_output_stem_outside_selected_universe_before_writing_targets(tmp_path: Path):
    # Deleting this membership check would let a stale prior rerun leak into a
    # fresh publication candidate.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")
    (grids / "stale.TextGrid").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="outside selected universe"):
        _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)
    assert not (tmp_path / "collected-grids").exists()
    assert not (tmp_path / "collected-wavs").exists()


def test_rejects_extra_wav_even_when_it_is_in_the_flat_textgrid_root(tmp_path: Path):
    # Roots are inspected as media namespaces, not merely for the extension
    # this invocation intends to copy from each one.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    (grids / "orphan.wav").write_bytes(b"stale")
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")

    with pytest.raises(ValueError, match="outside selected universe"):
        _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)
    assert not (tmp_path / "collected-grids").exists()


def test_rejects_same_stem_wav_outside_the_receipt_bound_inputs_root(tmp_path: Path):
    # Stem equality alone is insufficient: a stale WAV from another rerun can
    # have the same name and even the same bytes.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids = tmp_path / "flat-grids"
    _flat_pair(grids, "listed")
    rogue_wavs = tmp_path / "other-run" / "inputs" / "persona"
    rogue_wavs.mkdir(parents=True)
    (rogue_wavs / "listed.wav").write_bytes(b"wav:listed")

    with pytest.raises(ValueError, match="receipt-bound inputs root"):
        collect_nvv_rerun_outputs(
            manifest_path=manifest,
            materialization_receipt_path=receipt,
            flat_textgrid_roots={"persona": grids},
            flat_wav_roots={"persona": rogue_wavs},
            flat_textgrid_receipts={"persona": _pipeline_receipt(grids, ["listed"])},
            textgrid_output_root=tmp_path / "collected-grids",
            wav_output_root=tmp_path / "collected-wavs",
            collection_receipt_path=tmp_path / "collection-receipt.json",
        )


def test_rejects_receipt_bound_wav_with_wrong_bytes(tmp_path: Path):
    # A path in inputs/game is not enough if the materialized bytes were
    # replaced between input freezing and collection.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True)
    (wavs / "listed.wav").write_bytes(b"replaced")

    with pytest.raises(ValueError, match="flat WAV hash mismatch"):
        _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)


def test_rejects_textgrid_bytes_mutated_after_same_root_publish_manifest(tmp_path: Path):
    # The v2 accounting receipt only proves stem membership.  A publisher
    # manifest must bind the TextGrid payload that the collector will copy.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")
    _pipeline_receipt(grids, ["listed"])
    write_publish_manifest(grids)
    (grids / "listed.TextGrid").write_text("grid:Xisted", encoding="utf-8")

    with pytest.raises(ValueError, match="publish manifest TextGrid hash mismatch"):
        _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)


def test_rejects_pipeline_receipt_not_bound_to_the_supplied_flat_textgrid_root(tmp_path: Path):
    # A valid receipt from an older output directory cannot authorize TextGrids
    # in a different directory merely because their stems match.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")
    old_root = tmp_path / "old-flat-grids"
    old_root.mkdir()
    old_receipt = _pipeline_receipt(old_root, ["listed"])

    with pytest.raises(ValueError, match="pipeline receipt path mismatch"):
        collect_nvv_rerun_outputs(
            manifest_path=manifest,
            materialization_receipt_path=receipt,
            flat_textgrid_roots={"persona": grids},
            flat_wav_roots={"persona": wavs},
            flat_textgrid_receipts={"persona": old_receipt},
            textgrid_output_root=tmp_path / "collected-grids",
            wav_output_root=tmp_path / "collected-wavs",
            collection_receipt_path=tmp_path / "collection-receipt.json",
        )


@pytest.mark.parametrize(
    ("receipt_changes", "match"),
    [
        ({"manifest_sha256": "0" * 64}, "manifest hash mismatch"),
        ({"inputs": [{"game": "persona", "speaker": "alice", "stem": "unknown"}]},
         "outside frozen manifest"),
    ],
)
def test_rejects_receipt_that_does_not_bind_selected_keys_to_frozen_manifest(
    tmp_path: Path, receipt_changes: dict[str, object], match: str,
):
    # A changed receipt binding must fail before source output is trusted.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"}], **receipt_changes)
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")
    _pipeline_receipt(grids, ["listed"])
    write_publish_manifest(grids)

    with pytest.raises(ValueError, match=match):
        _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)


def test_rejects_manifest_speaker_path_escape_before_creating_targets(tmp_path: Path):
    # Treating speaker as an ordinary path would permit writes outside the
    # classified output root.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "../escape", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "../escape", "stem": "listed"}])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")

    with pytest.raises(ValueError, match="path escape"):
        _call(tmp_path, manifest, receipt, grids=grids, wavs=wavs)
    assert not (tmp_path / "collected-grids").exists()


def test_rejects_nested_output_roots_before_creating_either_target(tmp_path: Path):
    # Nested roots could turn the two supposedly independent pair trees into
    # one mutable namespace when a collector is re-run.
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    receipt = _receipt(tmp_path / "materialization_receipt.json", manifest, [
        {"game": "persona", "speaker": "alice", "stem": "listed"},
    ])
    grids, wavs = tmp_path / "flat-grids", _input_wavs(tmp_path)
    _flat_pair(grids, "listed")
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / "listed.wav").write_bytes(b"wav:listed")
    _pipeline_receipt(grids, ["listed"])
    write_publish_manifest(grids)

    with pytest.raises(ValueError, match="must not be nested"):
        collect_nvv_rerun_outputs(
            manifest_path=manifest,
            materialization_receipt_path=receipt,
            flat_textgrid_roots={"persona": grids},
            flat_wav_roots={"persona": wavs},
            flat_textgrid_receipts={"persona": grids / ".pipeline_run_receipt_v2.json"},
            textgrid_output_root=tmp_path / "output",
            wav_output_root=tmp_path / "output" / "GAMESL",
            collection_receipt_path=tmp_path / "collection-receipt.json",
        )
    assert not (tmp_path / "output").exists()
