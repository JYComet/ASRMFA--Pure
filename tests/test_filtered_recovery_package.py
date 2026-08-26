from __future__ import annotations

import json
import shutil
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scripts.pipeline_utils import make_audio_transform_receipt
from scripts.filtered_recovery_package import (
    ARTIFACT_ONLY_COUNT,
    FROZEN_COUNT,
    MFA_AXIS_COUNT,
    PackageError,
    _find_one,
    _ordinary_file,
    validate_asset_scope,
    validate_frozen_selection,
    validate_parent_hashes,
    sha256,
    _validate_english_ledger,
    _validate_english_manifest,
)
from scripts.run_pipeline import _localize_filtered_recovery_transform_receipt


def _partition() -> tuple[list[str], list[str], list[str]]:
    frozen = [f"f{i:03d}" for i in range(FROZEN_COUNT)]
    accepted = [f"a{i:04d}" for i in range(1738)]
    axis = frozen[:MFA_AXIS_COUNT]
    return frozen, accepted, axis


def test_frozen_only_partition_is_exact_259_plus_3():
    frozen, accepted, axis = _partition()
    result = validate_frozen_selection(frozen, accepted, frozen, axis)
    assert len(result["artifact_only"]) == ARTIFACT_ONLY_COUNT
    assert len(result["axis_missing"]) == MFA_AXIS_COUNT


def test_rejects_accepted_stem_import_at_package_construction():
    frozen, accepted, axis = _partition()
    with pytest.raises(PackageError, match="accepted-stem"):
        validate_asset_scope([{"stem": accepted[0], "role": "audio"}], frozen, accepted)


def test_rejects_non_frozen_selection():
    frozen, accepted, axis = _partition()
    with pytest.raises(PackageError, match="complete frozen"):
        validate_frozen_selection(frozen, accepted, frozen[:-1], axis)


def test_rejects_non_262_package():
    frozen, accepted, axis = _partition()
    with pytest.raises(PackageError, match="exactly 262"):
        validate_frozen_selection(frozen[:-1], accepted, frozen[:-1], axis)


def test_rejects_ambiguous_glob_selection(tmp_path: Path):
    with pytest.raises(PackageError, match="glob"):
        _find_one(tmp_path, "strict_ok_runs/*/output/strict_ok_manifest.json")


def test_rejects_symlink_and_relative_escape(tmp_path: Path):
    root = tmp_path / "parent"
    root.mkdir()
    target = root / "receipt.json"
    target.write_text("{}", encoding="utf-8")
    (root / "alias.json").symlink_to(target)
    with pytest.raises(PackageError, match="symlink"):
        _ordinary_file(root / "alias.json", root, "receipt")
    with pytest.raises(PackageError, match="escapes"):
        _ordinary_file(root / "../receipt.json", root, "receipt")


def test_rejects_parent_hash_drift(tmp_path: Path):
    root = tmp_path / "parent"
    root.mkdir()
    artifact = root / "receipt.json"
    artifact.write_text("v1", encoding="utf-8")
    expected = {"receipt.json": sha256(artifact)}
    artifact.write_text("v2", encoding="utf-8")
    with pytest.raises(PackageError, match="hash changed"):
        validate_parent_hashes(root, expected)


def _v2_english_ledger(stem: str = "demo") -> dict:
    return {
        "schema": "strict-en-mfa-v2",
        "canonical_units": "canonical-english-units-v1",
        "stem": stem,
        "segments": [{
            "canonical_units": "canonical-english-units-v1",
            "status": "verified",
            "words": [{
                "canonical_binding": "canonical-english-units-v1",
                "unit_id": "en-u0000",
                "text": "K-Pop",
            }],
        }],
    }


def test_valid_v2_english_contract_requires_canonical_units():
    ledger = _validate_english_ledger(_v2_english_ledger(), "demo", "fixture ledger")
    assert ledger["schema"] == "strict-en-mfa-v2"
    manifest = _validate_english_manifest({
        "schema": "strict-en-mfa-v2",
        "canonical_units": "canonical-english-units-v1",
        "strict_provenance": True,
        "status": "no_english",
    }, "fixture manifest")
    assert manifest["status"] == "no_english"


@pytest.mark.parametrize("mutation", ["historical", "missing_binding", "tampered_word"])
def test_historical_or_tampered_english_contract_fails_closed(mutation: str):
    payload = _v2_english_ledger()
    if mutation == "historical":
        payload["schema"] = "strict-en-mfa-v1"
    elif mutation == "missing_binding":
        payload.pop("canonical_units")
    else:
        payload["segments"][0]["words"][0]["canonical_binding"] = "wrong"
    with pytest.raises(PackageError):
        _validate_english_ledger(payload, "demo", "fixture ledger")


def _write_wav(path: Path, *, sample_rate: int, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames)


def _transform_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path]:
    parent = tmp_path / "parent"
    tts = parent / "padded_audio" / "stem.wav"
    mfa = parent / "audio_16k" / "stem.wav"
    _write_wav(tts, sample_rate=48000, frames=480)
    _write_wav(mfa, sample_rate=16000, frames=160)
    source_receipt = parent / "audio_transform_receipts" / "stem.receipt.json"
    source_receipt.parent.mkdir(parents=True)
    source_receipt.write_text(json.dumps(make_audio_transform_receipt(tts, mfa)), encoding="utf-8")
    localized_tts = tmp_path / "imports" / "tts_audio" / "stem.wav"
    localized_mfa = tmp_path / "imports" / "mfa_audio" / "stem.wav"
    localized_tts.parent.mkdir(parents=True)
    localized_mfa.parent.mkdir(parents=True)
    shutil.copy2(tts, localized_tts)
    shutil.copy2(mfa, localized_mfa)
    (tmp_path / "imports" / "axis" / "audio_transform_receipts").mkdir(parents=True)
    row = {"stem": "stem", "path": str(mfa.resolve()), **_audio_metadata(mfa),
           "transform_receipt": str(source_receipt.resolve())}
    return row, parent, localized_tts, localized_mfa, source_receipt


def _audio_metadata(path: Path) -> dict:
    from scripts.pipeline_utils import _axis_audio_metadata
    return _axis_audio_metadata(path)


def test_filtered_recovery_localizes_valid_transform_receipt(tmp_path: Path):
    row, parent, localized_tts, localized_mfa, source_receipt = _transform_fixture(tmp_path)
    result = _localize_filtered_recovery_transform_receipt(
        row, "stem", parent_root=parent,
        parent_tts_audio=parent / "padded_audio" / "stem.wav",
        parent_mfa_audio=parent / "audio_16k" / "stem.wav",
        localized_tts_audio=localized_tts, localized_mfa_audio=localized_mfa,
        destination_dir=tmp_path / "imports" / "axis" / "audio_transform_receipts")
    localized = json.loads(Path(result["destination"]).read_text(encoding="utf-8"))
    original = json.loads(source_receipt.read_text(encoding="utf-8"))
    assert localized["input"]["path"] == str(localized_tts.resolve())
    assert localized["output"]["path"] == str(localized_mfa.resolve())
    assert {**localized["input"], "path": None} == {**original["input"], "path": None}
    assert {**localized["output"], "path": None} == {**original["output"], "path": None}
    assert localized["head_transform_s"] == original["head_transform_s"] == 0.0
    assert localized["tail_transform_s"] == original["tail_transform_s"] == 0.0
    assert localized["shift_s"] == original["shift_s"] == 0.0
    assert result["source_sha256"] == sha256(source_receipt)


def test_filtered_recovery_missing_transform_receipt_fails_closed(tmp_path: Path):
    row, parent, localized_tts, localized_mfa, _ = _transform_fixture(tmp_path)
    row.pop("transform_receipt")
    with pytest.raises(ValueError, match="transform receipt missing"):
        _localize_filtered_recovery_transform_receipt(
            row, "stem", parent_root=parent,
            parent_tts_audio=parent / "padded_audio" / "stem.wav",
            parent_mfa_audio=parent / "audio_16k" / "stem.wav",
            localized_tts_audio=localized_tts, localized_mfa_audio=localized_mfa,
            destination_dir=tmp_path / "imports" / "axis" / "audio_transform_receipts")


@pytest.mark.parametrize("mismatch, match", [
    ("row_path", "parent input-axis path mismatch"),
    ("row_hash", "parent input-axis sha256 mismatch"),
    ("receipt_hash", "transform receipt invalid"),
    ("receipt_path", "transform receipt invalid"),
])
def test_filtered_recovery_transform_metadata_path_hash_mismatch_fails_closed(
    tmp_path: Path, mismatch: str, match: str,
):
    row, parent, localized_tts, localized_mfa, source_receipt = _transform_fixture(tmp_path)
    if mismatch == "row_path":
        row["path"] = str(parent / "audio_16k" / "other.wav")
    elif mismatch == "row_hash":
        row["sha256"] = "0" * 64
    else:
        payload = json.loads(source_receipt.read_text(encoding="utf-8"))
        if mismatch == "receipt_hash":
            payload["output"]["sha256"] = "0" * 64
        else:
            payload["output"]["path"] = str(parent / "audio_16k" / "other.wav")
        source_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _localize_filtered_recovery_transform_receipt(
            row, "stem", parent_root=parent,
            parent_tts_audio=parent / "padded_audio" / "stem.wav",
            parent_mfa_audio=parent / "audio_16k" / "stem.wav",
            localized_tts_audio=localized_tts, localized_mfa_audio=localized_mfa,
            destination_dir=tmp_path / "imports" / "axis" / "audio_transform_receipts")
