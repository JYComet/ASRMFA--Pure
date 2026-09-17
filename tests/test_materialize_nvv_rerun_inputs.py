import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from materialize_nvv_rerun_inputs import materialize_nvv_rerun_inputs  # noqa: E402
from run_pipeline import validate_config  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, name: str, payload: bytes = b"wav bytes") -> Path:
    source = tmp_path / "source" / f"{name}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload)
    return source


def _row(source: Path, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game": "persona",
        "speaker": "alice",
        "stem": "line",
        "source_wav_path": str(source),
        "source_wav_root": str(source.parent),
        "source_wav_sha256": _sha256(source),
        "asr_mode": "fallback",
        "reference_mode": "fallback",
        "expected_nvv_sequence": ["<BREATHING>"],
    }
    row.update(overrides)
    return row


def _manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")
    return path


def test_materializes_hash_bound_inputs_and_strict_private_yaml(tmp_path: Path):
    source = _source(tmp_path, "line", b"source payload")
    manifest = _manifest(tmp_path / "frozen.jsonl", [_row(source)])
    run_root = tmp_path / "fresh-run"

    receipt = materialize_nvv_rerun_inputs(
        manifest_path=manifest, run_root=run_root)

    target = run_root / "inputs" / "persona" / "line.wav"
    assert target.read_bytes() == source.read_bytes()
    assert _sha256(target) == _sha256(source)
    assert receipt["inputs"][0]["method"] in {"hardlink", "copy"}
    if receipt["inputs"][0]["method"] == "hardlink":
        assert target.stat().st_ino == source.stat().st_ino
    assert receipt["selected_count"] == 1
    assert receipt["selected_subset_sha256"]
    assert receipt["inputs"][0]["target_sha256"] == _sha256(source)
    config_path = run_root / "configs" / "persona.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["mode"] == "nvrasr_fallback"
    assert config["reference_mode"] == "fallback"
    assert config["data_dir"] == str(run_root / "inputs" / "persona")
    assert config["workspace"] == str(run_root / "workspaces" / "persona")
    assert config["output_dir"] == str(run_root / "staging" / "persona")
    assert config["output_staging"] is True
    assert config["ctc_prealign"]["nvv_enabled"] is True
    assert config["ctc_prealign"]["all_gpus"] is True
    assert config["postprocess"]["strict_ok"] is True
    assert config["postprocess"]["allow_filtered_integrity_failures"] is False
    assert receipt["configs"]["persona"]["sha256"] == _sha256(config_path)
    assert validate_config(config, "nvrasr_fallback") == []


def test_keys_file_selects_only_a_deduplicated_manifest_subset(tmp_path: Path):
    first = _source(tmp_path, "first", b"first")
    second = _source(tmp_path, "second", b"second")
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        _row(first, stem="first"), _row(second, stem="second"),
    ])
    keys = tmp_path / "canary.keys"
    keys.write_text("persona/alice/second\n", encoding="utf-8")

    receipt = materialize_nvv_rerun_inputs(
        manifest_path=manifest, run_root=tmp_path / "fresh-run", keys_file=keys)

    assert receipt["manifest_count"] == 2
    assert receipt["selected_count"] == 1
    assert (tmp_path / "fresh-run" / "inputs" / "persona" / "second.wav").is_file()
    assert not (tmp_path / "fresh-run" / "inputs" / "persona" / "first.wav").exists()


@pytest.mark.parametrize(
    ("row_overrides", "match"),
    [
        ({"source_wav_sha256": "0" * 64}, "hash mismatch"),
        ({"asr_mode": "reference"}, "fallback"),
        ({"reference_mode": "reference"}, "fallback"),
        ({"expected_nvv_sequence": []}, "expected_nvv_sequence"),
    ],
)
def test_rejects_invalid_or_nonfallback_manifest_rows(
    tmp_path: Path, row_overrides: dict[str, object], match: str,
):
    source = _source(tmp_path, "line")
    manifest = _manifest(tmp_path / "frozen.jsonl", [_row(source, **row_overrides)])
    run_root = tmp_path / "fresh-run"

    with pytest.raises(ValueError, match=match):
        materialize_nvv_rerun_inputs(manifest_path=manifest, run_root=run_root)
    assert not run_root.exists()


def test_rejects_stem_collision_with_different_source_before_creating_run_root(tmp_path: Path):
    first = _source(tmp_path, "first", b"first")
    second = _source(tmp_path, "second", b"second")
    manifest = _manifest(tmp_path / "frozen.jsonl", [
        _row(first, speaker="alice", stem="same"),
        _row(second, speaker="bob", stem="same"),
    ])
    run_root = tmp_path / "fresh-run"

    with pytest.raises(ValueError, match="target stem collision"):
        materialize_nvv_rerun_inputs(manifest_path=manifest, run_root=run_root)
    assert not run_root.exists()


def test_rejects_existing_or_symlinked_run_root(tmp_path: Path):
    source = _source(tmp_path, "line")
    manifest = _manifest(tmp_path / "frozen.jsonl", [_row(source)])
    run_root = tmp_path / "fresh-run"
    run_root.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        materialize_nvv_rerun_inputs(manifest_path=manifest, run_root=run_root)
    run_root.rmdir()
    os.symlink(tmp_path / "source", run_root)
    with pytest.raises(ValueError, match="already exists or is symlinked"):
        materialize_nvv_rerun_inputs(manifest_path=manifest, run_root=run_root)


def test_rejects_keys_outside_manifest_or_duplicate_keys(tmp_path: Path):
    source = _source(tmp_path, "line")
    manifest = _manifest(tmp_path / "frozen.jsonl", [_row(source)])
    keys = tmp_path / "bad.keys"
    keys.write_text("persona/alice/missing\npersona/alice/missing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        materialize_nvv_rerun_inputs(
            manifest_path=manifest, run_root=tmp_path / "fresh-run", keys_file=keys)


def test_rejects_source_path_escape_or_source_symlink(tmp_path: Path):
    source = _source(tmp_path, "line")
    escaped = _manifest(tmp_path / "escaped.jsonl", [_row(
        source, source_wav_path=str(source.parent / ".." / "source" / "line.wav"))])
    with pytest.raises(ValueError, match="path escape"):
        materialize_nvv_rerun_inputs(
            manifest_path=escaped, run_root=tmp_path / "escaped-run")

    linked_source = tmp_path / "linked.wav"
    os.symlink(source, linked_source)
    linked = _manifest(tmp_path / "linked.jsonl", [_row(
        linked_source, source_wav_sha256=_sha256(source))])
    with pytest.raises(ValueError, match="symlinked"):
        materialize_nvv_rerun_inputs(
            manifest_path=linked, run_root=tmp_path / "linked-run")


def test_rejects_missing_or_outside_protected_source_root(tmp_path: Path):
    source = _source(tmp_path, "line")
    missing_root = _manifest(tmp_path / "missing-root.jsonl", [_row(
        source, source_wav_root=None)])
    with pytest.raises(ValueError, match="source_wav_root"):
        materialize_nvv_rerun_inputs(
            manifest_path=missing_root, run_root=tmp_path / "missing-root-run")

    outside = _source(tmp_path / "outside", "other")
    external = _manifest(tmp_path / "outside-root.jsonl", [_row(
        outside, source_wav_root=str(source.parent))])
    with pytest.raises(ValueError, match="outside protected source root"):
        materialize_nvv_rerun_inputs(
            manifest_path=external, run_root=tmp_path / "outside-root-run")
