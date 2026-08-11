from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path

from scripts import audit_strict_ok as audit
from scripts import postprocess_textgrids as post


def _wav(path: Path, seconds: float, rate: int = 16000) -> None:
    frames = int(round(seconds * rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * frames)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid(path: Path, xmax: float) -> None:
    path.write_text(
        f'''File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = {xmax}\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = {xmax}\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = {xmax}\n            text = "yi1"\n''', encoding="utf-8")


def _fixture(tmp_path: Path, *, aligned_xmax: float = 1.0,
             tts_seconds: float = 1.0):
    stem = "fixture"
    mfa = tmp_path / "mfa_axis_audio"
    tts = tmp_path / "tts_authoritative_audio"
    aligned = tmp_path / "aligned"
    mfa.mkdir(); tts.mkdir(); aligned.mkdir()
    mfa_wav = mfa / f"{stem}.wav"
    tts_wav = tts / f"{stem}.wav"
    _wav(mfa_wav, 1.0)
    _wav(tts_wav, tts_seconds)
    tg = aligned / f"{stem}.TextGrid"
    _grid(tg, aligned_xmax)
    row = {"stem": stem, "path": str(mfa_wav), "sha256": _sha(mfa_wav),
           "duration_s": 1.0, "sample_rate": 16000, "frames": 16000}
    input_axis = {"schema": post.MFA_INPUT_AXIS_SCHEMA,
                  "source_role": "mfa_axis_audio", "axis_root": str(mfa),
                  "stems": [stem], "stems_digest": post._axis_digest([stem]),
                  "audio": [row], "transform_receipts": [], "scale": 1.0}
    align_row = {"stem": stem, "path": str(tg), "sha256": _sha(tg),
                 "xmax": aligned_xmax, "audio_sha256": row["sha256"],
                 "audio_duration_s": 1.0}
    alignment = {"schema": post.MFA_ALIGNMENT_AXIS_SCHEMA,
                 "input_axis_schema": post.MFA_INPUT_AXIS_SCHEMA,
                 "input_axis_digest": post._axis_digest(input_axis),
                 "alignment_root": str(aligned), "stems": [stem],
                 "stems_digest": post._axis_digest([stem]),
                 "alignments": [align_row], "scale": 1.0}
    inp = tmp_path / "input.json"; aln = tmp_path / "alignment.json"
    inp.write_text(json.dumps(input_axis), encoding="utf-8")
    aln.write_text(json.dumps(alignment), encoding="utf-8")
    args = argparse.Namespace(mfa_input_axis_receipt=inp,
                              mfa_alignment_axis_receipt=aln,
                              mfa_axis_audio_root=mfa,
                              tts_authoritative_audio_root=tts,
                              textgrid_dir=aligned,
                              aligned_dir=aligned)
    return args, input_axis, alignment


def test_same_axis_positive(tmp_path):
    args, _, _ = _fixture(tmp_path)
    assert post._load_axis_contract(args) == ([], {"fixture": []})
    assert audit._axis_contract_reasons(args, {"fixture"}) == ([], {"fixture": []})


def test_tts_axis_mismatch_is_filterable(tmp_path):
    args, _, _ = _fixture(tmp_path, tts_seconds=1.1)
    errors, reasons = post._load_axis_contract(args)
    assert not errors
    assert reasons["fixture"] == ["tts_audio_axis_mismatch"]
    _, audit_reasons = audit._axis_contract_reasons(args, {"fixture"})
    assert audit_reasons["fixture"] == ["tts_audio_axis_mismatch"]


def test_alignment_axis_mismatch_is_filterable(tmp_path):
    args, _, _ = _fixture(tmp_path, aligned_xmax=1.01)
    errors, reasons = post._load_axis_contract(args)
    assert not errors
    assert reasons["fixture"] == ["mfa_alignment_axis_mismatch"]


def test_three_ms_boundary_is_accepted(tmp_path):
    args, _, _ = _fixture(tmp_path, aligned_xmax=1.003)
    errors, reasons = post._load_axis_contract(args)
    assert not errors
    assert reasons["fixture"] == []


def test_missing_receipt_is_infrastructure_failure(tmp_path):
    args, _, _ = _fixture(tmp_path)
    args.mfa_input_axis_receipt = None
    errors, _ = post._load_axis_contract(args)
    assert errors == ["axis_contract_receipts_missing"]
    errors, _ = audit._axis_contract_reasons(args, {"fixture"})
    assert errors == ["axis_contract_receipts_missing"]


def test_schema_digest_and_stem_tamper_are_infrastructure_failures(tmp_path):
    args, input_axis, _ = _fixture(tmp_path)
    input_axis["schema"] = "legacy-axis"
    input_axis["stems_digest"] = "0" * 64
    args.mfa_input_axis_receipt.write_text(json.dumps(input_axis), encoding="utf-8")
    errors, _ = post._load_axis_contract(args)
    assert "mfa_input_axis_schema_mismatch" in errors
    assert "axis_stem_conservation_invalid" in errors
    errors, _ = audit._axis_contract_reasons(args, {"fixture", "missing"})
    assert "axis_stem_conservation_invalid" in errors


def test_minimal_historical_axis_fixture_is_18_rejections():
    path = Path(__file__).parent / "fixtures" / "axis_forensics" / "axis_inventory_minimal.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert payload["schema"] == "mfa-axis-forensics-minimal-v1"
    assert len(rows) == 18
    assert {row["expected"]["tts_audio_axis_mismatch"] for row in rows} == {True}
    assert {row["expected"]["mfa_alignment_axis_mismatch"] for row in rows} == {False}
    assert {row["expected"]["candidate"] for row in rows} == {False}
    assert all(row["mfa_audio16"]["sha256"] != row["authoritative_audio"]["sha256"]
               for row in rows)
    assert all(abs(row["aligned_xmax"] - row["mfa_audio16"]["duration_s"]) <= 0.003
               for row in rows)
