"""Receipt-backed audio-axis cases shared by postprocess and strict audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import wave
from pathlib import Path

from scripts import audit_strict_ok as audit
from scripts import postprocess_textgrids as post


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav(path: Path, seconds: float, rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * int(seconds * rate))


def _grid(path: Path, xmax: float = 1.0) -> None:
    path.write_text(
        f'''File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = {xmax}\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = {xmax}\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = {xmax}\n            text = "yi1"\n''',
        encoding="utf-8")


def _meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        return {"sha256": _sha(path), "duration_s": handle.getnframes() / handle.getframerate(),
                "sample_rate": handle.getframerate(), "frames": handle.getnframes(),
                "channels": handle.getnchannels(), "sample_width": handle.getsampwidth()}


def _fixture(tmp_path: Path, *, missing: bool = False) -> tuple[argparse.Namespace, dict, dict]:
    tts, mfa, aligned = (tmp_path / "tts"), (tmp_path / "mfa"), (tmp_path / "aligned")
    tts.mkdir(); mfa.mkdir(); aligned.mkdir()
    stems = ["aligned"] + (["missing"] if missing else [])
    rows, alignment_rows, receipts = [], [], []
    for stem in stems:
        tts_wav, mfa_wav = tts / f"{stem}.wav", mfa / f"{stem}.wav"
        _wav(tts_wav, 1.0, 48000)
        _wav(mfa_wav, 1.0, 16000)
        rows.append({"stem": stem, "path": str(mfa_wav.resolve()), **_meta(mfa_wav)})
        transform = {"schema": "audio-transform-receipt-v1",
                     "input": {"path": str(tts_wav.resolve()), **_meta(tts_wav)},
                     "output": {"path": str(mfa_wav.resolve()), **_meta(mfa_wav)},
                     "head_transform_s": 0.0, "tail_transform_s": 0.0,
                     "shift_s": 0.0, "scale": 1.0}
        receipt = tmp_path / f"{stem}.transform.json"
        receipt.write_text(json.dumps(transform), encoding="utf-8")
        receipts.append(str(receipt.resolve()))
        if stem == "missing":
            alignment_rows.append({"stem": stem, "status": "missing_mfa_alignment"})
        else:
            grid = aligned / f"{stem}.TextGrid"
            _grid(grid)
            alignment_rows.append({"stem": stem, "status": "aligned", "path": str(grid.resolve()),
                                   "sha256": _sha(grid), "xmax": 1.0,
                                   "audio_sha256": _sha(mfa_wav), "audio_duration_s": 1.0})
    input_axis = {"schema": post.MFA_INPUT_AXIS_SCHEMA, "source_role": "mfa_axis_audio",
                  "axis_root": str(mfa.resolve()), "tts_authoritative_audio_root": str(tts.resolve()),
                  "stems": stems, "stems_digest": post._axis_digest(stems), "audio": rows,
                  "transform_receipts": receipts, "scale": 1.0}
    alignment = {"schema": post.MFA_ALIGNMENT_AXIS_V2_SCHEMA,
                 "input_axis_schema": post.MFA_INPUT_AXIS_SCHEMA,
                 "input_axis_digest": post._axis_digest(input_axis),
                 "alignment_root": str(aligned.resolve()), "stems": stems,
                 "stems_digest": post._axis_digest(stems), "alignments": alignment_rows,
                 "scale": 1.0,
                 "status_counts": {"aligned": 1, "missing_mfa_alignment": int(missing)}}
    input_path, alignment_path = tmp_path / "input.json", tmp_path / "alignment.json"
    input_path.write_text(json.dumps(input_axis), encoding="utf-8")
    alignment_path.write_text(json.dumps(alignment), encoding="utf-8")
    args = argparse.Namespace(mfa_input_axis_receipt=input_path,
                              mfa_alignment_axis_receipt=alignment_path,
                              mfa_axis_audio_root=mfa,
                              tts_authoritative_audio_root=tts,
                              textgrid_dir=aligned, aligned_dir=aligned)
    return args, input_axis, alignment


def test_unit_time_resample_receipt_reaches_both_consumers(tmp_path):
    args, _, _ = _fixture(tmp_path)
    assert post._load_axis_contract(args) == ([], {"aligned": []})
    assert audit._axis_contract_reasons(args, {"aligned"}) == ([], {"aligned": []})


def test_exact_identity_remains_a_valid_v1_style_axis(tmp_path):
    args, input_axis, alignment = _fixture(tmp_path)
    shutil.copy2(args.mfa_axis_audio_root / "aligned.wav",
                 args.tts_authoritative_audio_root / "aligned.wav")
    input_axis["transform_receipts"] = []
    alignment["input_axis_digest"] = post._axis_digest(input_axis)
    args.mfa_input_axis_receipt.write_text(json.dumps(input_axis), encoding="utf-8")
    args.mfa_alignment_axis_receipt.write_text(json.dumps(alignment), encoding="utf-8")
    assert post._load_axis_contract(args) == ([], {"aligned": []})
    assert audit._axis_contract_reasons(args, {"aligned"}) == ([], {"aligned": []})


def test_v2_missing_status_is_explicit_and_conserved(tmp_path):
    args, _, _ = _fixture(tmp_path, missing=True)
    assert post._load_axis_contract(args) == ([], {"aligned": [], "missing": ["missing_mfa_alignment"]})
    assert audit._axis_contract_reasons(args, {"aligned", "missing"}) == (
        [], {"aligned": [], "missing": ["missing_mfa_alignment"]})


def test_transform_tampering_or_missing_evidence_fails_closed(tmp_path):
    args, input_axis, _ = _fixture(tmp_path)
    receipt_path = Path(input_axis["transform_receipts"][0])
    transform = json.loads(receipt_path.read_text(encoding="utf-8"))
    transform["shift_s"] = 0.001
    receipt_path.write_text(json.dumps(transform), encoding="utf-8")
    for validator, expected in ((post._load_axis_contract, None),
                                (lambda a: audit._axis_contract_reasons(a, {"aligned"}), None)):
        errors, reasons = validator(args)
        assert not errors
        assert reasons["aligned"] == ["tts_audio_axis_mismatch"]

    input_axis["transform_receipts"] = []
    args.mfa_input_axis_receipt.write_text(json.dumps(input_axis), encoding="utf-8")
    assert post._load_axis_contract(args)[1]["aligned"] == ["tts_audio_axis_mismatch"]
    assert audit._axis_contract_reasons(args, {"aligned"})[1]["aligned"] == ["tts_audio_axis_mismatch"]


def test_transform_root_digest_and_status_tampering_are_rejected(tmp_path):
    args, input_axis, alignment = _fixture(tmp_path)
    input_axis["tts_authoritative_audio_root"] = str(tmp_path / "other-root")
    args.mfa_input_axis_receipt.write_text(json.dumps(input_axis), encoding="utf-8")
    assert "tts_authoritative_audio_root_binding_mismatch" in post._load_axis_contract(args)[0]
    assert "tts_authoritative_audio_root_binding_mismatch" in audit._axis_contract_reasons(args, {"aligned"})[0]

    input_axis["tts_authoritative_audio_root"] = str(args.tts_authoritative_audio_root.resolve())
    args.mfa_input_axis_receipt.write_text(json.dumps(input_axis), encoding="utf-8")
    alignment["input_axis_digest"] = "0" * 64
    args.mfa_alignment_axis_receipt.write_text(json.dumps(alignment), encoding="utf-8")
    assert "axis_receipt_digest_or_stem_mismatch" in post._load_axis_contract(args)[0]
    assert "axis_receipt_digest_or_stem_mismatch" in audit._axis_contract_reasons(args, {"aligned"})[0]

    alignment["input_axis_digest"] = post._axis_digest(input_axis)
    args.mfa_alignment_axis_receipt.write_text(json.dumps(alignment), encoding="utf-8")
    input_axis["source_role"] = "ctc_input_audio"
    args.mfa_input_axis_receipt.write_text(json.dumps(input_axis), encoding="utf-8")
    assert "mfa_input_axis_source_role_mismatch" in post._load_axis_contract(args)[0]
    assert "mfa_input_axis_source_role_mismatch" in audit._axis_contract_reasons(args, {"aligned"})[0]


def test_tier_desync_coalesces_hyphenated_english_without_weakening_cjk(tmp_path):
    hanzi = post.Tier("hanzi", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "你"), post.Interval(0.2, 0.5, "K-Pop"),
        post.Interval(0.5, 0.7, "好"),
    ])
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "ni3"), post.Interval(0.2, 0.35, "kp"),
        post.Interval(0.35, 0.5, "op"), post.Interval(0.5, 0.7, "hao3"),
    ])
    assert post._tier_desync_counts(hanzi, words) == (0, 3)

    malformed = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "kp"), post.Interval(0.2, 0.5, "op"),
        post.Interval(0.5, 0.7, "hao3"),
    ])
    mismatches, total = post._tier_desync_counts(hanzi, malformed)
    assert mismatches / total > 0.10


def test_reference_semantic_integrity_catches_english_position_swap(tmp_path):
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "di4"), post.Interval(0.2, 0.4, "tian1"),
        post.Interval(0.4, 0.6, "Noa"), post.Interval(0.6, 0.8, "shang4"),
    ])
    # CJK still reads 第天上, but the reference's ``N`` has been replaced by
    # ``Noa`` before the legitimate later Noa word.
    hanzi = post.Tier("hanzi", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "第"), post.Interval(0.2, 0.4, "Noa"),
        post.Interval(0.4, 0.6, "天"), post.Interval(0.6, 0.7, "，"),
        post.Interval(0.7, 0.8, "Noa"), post.Interval(0.8, 1.0, "上"),
    ])
    coverage, reasons = post.assess_reference_coverage(
        "第N天，Noa上", words, hanzi, reference_source="fixture")
    assert coverage["exact_cjk_sequence"] is True
    assert coverage["exact_semantic_sequence"] is False
    assert reasons == ["reference_semantic_sequence_mismatch"]


def test_final_semantic_veto_uses_finalized_sp1_labels(tmp_path):
    """A provisional `<sp2>` must not survive as a stale lexical veto.

    GPU1000 stem 000944 was semantically correct on disk, but its pre-write
    coverage check read `<sp2>` before finalization rewrote it to `<sp1>`.
    The final semantic contract must therefore be evaluated after that step.
    """
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.1, "<sp2>"), post.Interval(0.1, 1.0, "ni3"),
    ])
    hanzi = post.Tier("hanzi", 0.0, 1.0, [
        post.Interval(0.0, 0.1, "<sp2>"), post.Interval(0.1, 1.0, "你"),
    ])
    before, before_reasons = post.assess_reference_coverage("你", words, hanzi,
                                                              reference_source="fixture")
    assert before["exact_semantic_sequence"] is False
    assert "reference_semantic_sequence_mismatch" in before_reasons

    grid = post.TextGrid(0.0, 1.0, [
        post.Tier("raw_text", 0.0, 1.0, [post.Interval(0.0, 1.0, "你")]),
        post.Tier("pinyin", 0.0, 1.0, [post.Interval(0.0, 1.0, "ni3")]),
        hanzi, words,
        post.Tier("pinyin_phones", 0.0, 1.0, [
            post.Interval(0.0, 0.1, "<sp2>"), post.Interval(0.1, 1.0, "i3"),
        ]),
    ])
    post._finalize_textgrid(grid)
    final_hanzi = next(tier for tier in grid.tiers if tier.name == "hanzi")
    final_words = next(tier for tier in grid.tiers if tier.name == "words")
    after, after_reasons = post.assess_reference_coverage("你", final_words, final_hanzi,
                                                            reference_source="fixture")
    assert after["exact_semantic_sequence"] is True
    assert "reference_semantic_sequence_mismatch" not in after_reasons


def test_reference_semantic_integrity_matches_strict_spaced_english_tokens(tmp_path):
    words = post.Tier("words", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "zui4"), post.Interval(0.2, 0.4, "hou4"),
        post.Interval(0.4, 0.6, "all"), post.Interval(0.6, 0.8, "in"),
        post.Interval(0.8, 1.0, "le5"),
    ])
    hanzi = post.Tier("hanzi", 0.0, 1.0, [
        post.Interval(0.0, 0.2, "最"), post.Interval(0.2, 0.4, "后"),
        post.Interval(0.4, 0.6, "all"), post.Interval(0.6, 0.8, "in"),
        post.Interval(0.8, 1.0, "了"),
    ])
    coverage, reasons = post.assess_reference_coverage(
        "最后all in了", words, hanzi, reference_source="fixture")
    assert coverage["exact_semantic_sequence"] is True
    assert reasons == []
