#!/usr/bin/env python3
"""Verify an existing strict-ok v2 manifest without trusting its contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

POLICY_VERSION = "strict-ok-v3.2"
EN_PROVENANCE_SCHEMA = "strict-en-mfa-v1"


def _verify_pipeline_accounting(manifest: dict, expected: set[str], rejected: set[str]) -> list[str]:
    """Require a valid v2 receipt and bind its eligible set to strict output."""
    try:
        from pipeline_utils import (PIPELINE_ACCOUNTING_SCHEMA,
                                    read_pipeline_accounting_receipt,
                                    validate_pipeline_accounting_receipt)
        binding = manifest["pipeline_accounting_receipt"]
        if (not isinstance(binding, dict)
                or binding.get("schema") != PIPELINE_ACCOUNTING_SCHEMA):
            return ["pipeline_accounting_receipt_binding_missing"]
        receipt_path = Path(binding["path"])
        if receipt_path.is_symlink() or not receipt_path.is_file():
            return ["pipeline_accounting_receipt_not_regular"]
        receipt = read_pipeline_accounting_receipt(receipt_path)
        errors = validate_pipeline_accounting_receipt(receipt)
        if errors:
            return [f"pipeline_accounting_receipt_invalid:{error}" for error in errors]
        if _sha256(receipt_path) != binding.get("sha256"):
            return ["pipeline_accounting_receipt_hash_mismatch"]
        eligible = set(receipt["eligible"]["stems"])
        excluded = {row["stem"] for row in receipt.get("exclusions", [])}
        source = set(receipt["source"]["stems"])
        if eligible != expected:
            return ["pipeline_accounting_eligible_manifest_mismatch"]
        if excluded & (expected | rejected):
            return ["pipeline_accounting_exclusion_leaked_into_strict_set"]
        if source != eligible | excluded:
            return ["pipeline_accounting_source_conservation_mismatch"]
        if set(receipt["output"]["stems"]) | set(receipt["filtered"]["stems"]) != eligible:
            return ["pipeline_accounting_processed_set_mismatch"]
        return []
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"pipeline_accounting_receipt_failed:{exc}"]


def _safe_stem(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and value not in {".", ".."}
            and "\x00" not in value and Path(value).name == value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_regular_file(root: Path, value: object) -> Path | None:
    """Return a contained ordinary evidence file; reject links and escapes."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    target = root / path
    try:
        if target.is_symlink() or not target.is_file() or root.resolve() not in target.resolve().parents:
            return None
    except OSError:
        return None
    return target


def _entry_has_english(tg: Path) -> bool:
    try:
        from postprocess_textgrids import parse_textgrid
        from pipeline_utils import is_english_token
        parsed = parse_textgrid(tg)
        words = next(tier for tier in parsed.tiers if tier.name == "words")
        return any(is_english_token(interval.text.strip()) for interval in words.intervals)
    except Exception:
        # TextGrid integrity is checked independently by the auditor; a
        # verifier must never use parse failure to waive evidence requirements.
        return True


def _verify_english_evidence(root: Path, tg: Path, entry: dict) -> list[str]:
    errors: list[str] = []
    evidence = entry.get("english_provenance")
    needs_evidence = _entry_has_english(tg)
    if evidence is None:
        return ["english_provenance_evidence_missing"] if needs_evidence else []
    if not isinstance(evidence, dict) or evidence.get("schema") != EN_PROVENANCE_SCHEMA:
        return ["english_provenance_evidence_legacy_schema"]
    ledger = evidence.get("ledger")
    if not isinstance(ledger, dict):
        return ["english_provenance_ledger_missing"]
    ledger_path = _relative_regular_file(root, ledger.get("path"))
    if ledger_path is None or _sha256(ledger_path) != ledger.get("sha256"):
        errors.append("english_provenance_ledger_hash_mismatch")
    sources = evidence.get("source_textgrids")
    if not isinstance(sources, list) or (needs_evidence and not sources):
        errors.append("english_provenance_source_missing")
    else:
        for source in sources:
            if not isinstance(source, dict):
                errors.append("english_provenance_source_invalid")
                continue
            source_path = _relative_regular_file(root, source.get("path"))
            if source_path is None or _sha256(source_path) != source.get("sha256"):
                errors.append("english_provenance_source_hash_mismatch")
    return errors


def verify(path: Path, output_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"invalid_manifest:{exc}"]
    if manifest.get("policy_version") != POLICY_VERSION:
        errors.append("policy_version_mismatch")
    policy = manifest.get("english_provenance_policy")
    if not isinstance(policy, dict) or policy.get("schema") != EN_PROVENANCE_SCHEMA or policy.get("required") is not True:
        errors.append("english_provenance_policy_missing_or_legacy")
    if manifest.get("safe_empty"):
        errors.append("safe_empty_is_not_publishable")
    if manifest.get("global_reasons"):
        errors.append("manifest_has_global_reasons")
    root = output_dir or Path(manifest.get("output_dir", ""))
    if not root.is_dir():
        return errors + ["output_dir_missing"]
    expected_raw = manifest.get("expected_stems")
    if not isinstance(expected_raw, list):
        errors.append("invalid_expected_stems")
        expected_manifest: set[str] = set()
    else:
        expected_manifest = set()
        for stem in expected_raw:
            if not _safe_stem(stem):
                errors.append("unsafe_expected_stem")
            elif stem in expected_manifest:
                errors.append(f"duplicate_expected_stem:{stem}")
            else:
                expected_manifest.add(stem)
    entries = manifest.get("ok")
    if not isinstance(entries, list) or not entries:
        return errors + ["missing_ok_entries"]
    expected: set[str] = set()
    for entry in entries:
        try:
            stem = entry["stem"]
            if not _safe_stem(stem):
                errors.append("unsafe_ok_stem")
                continue
            if stem in expected:
                errors.append(f"duplicate_ok_stem:{stem}")
                continue
            expected.add(stem)
            tg = root / f"{stem}.TextGrid"
            if _sha256(tg) != entry["textgrid_sha256"]:
                errors.append(f"textgrid_hash_mismatch:{stem}")
            errors.extend(f"{error}:{stem}" for error in _verify_english_evidence(root, tg, entry))
            reference = entry["reference"]
            ref_path = Path(reference["path"])
            if _sha256(ref_path) != reference["sha256"]:
                errors.append(f"reference_hash_mismatch:{stem}")
        except (KeyError, OSError, TypeError):
            errors.append("invalid_ok_entry")
    rejected = manifest.get("rejected")
    rejected_stems: set[str] = set()
    if not isinstance(rejected, dict):
        errors.append("invalid_rejected")
    else:
        for stem, reasons in rejected.items():
            if not _safe_stem(stem):
                errors.append("unsafe_rejected_stem")
            elif stem in rejected_stems:
                errors.append(f"duplicate_rejected_stem:{stem}")
            else:
                rejected_stems.add(stem)
            if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
                errors.append("invalid_rejected_reasons")
    if expected & rejected_stems:
        errors.append("ok_rejected_overlap")
    if expected_manifest != expected | rejected_stems:
        errors.append("expected_ok_rejected_set_mismatch")
    errors.extend(_verify_pipeline_accounting(manifest, expected_manifest, rejected_stems))
    actual = {file.stem for file in root.glob("*.TextGrid")}
    if actual != expected:
        errors.append("output_textgrid_set_mismatch")
    return sorted(set(errors))


def _write_fixture(root: Path, stem: str = "demo") -> dict[str, Path]:
    """Build a tiny complete corpus without MFA/NVASR dependencies."""
    from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid

    output, filtered, ctc, refs, wavs, aligned, en_phones, en_aligned = [root / name for name in
        ("output", "filtered", "ctc", "refs", "wavs", "aligned", "en_phones", "en_aligned")]
    for directory in (output, filtered, ctc, refs, wavs, aligned, en_phones, en_aligned):
        directory.mkdir()
    (refs / f"{stem}.txt").write_text("你好!\n", encoding="utf-8")
    (ctc / f"{stem}.lab").write_text("ni3 hao3\n", encoding="utf-8")
    (ctc / f"{stem}_tokens.jsonl").write_text(
        "\n".join(json.dumps({"word": word, "start_s": start, "end_s": end})
                  for word, start, end in (("ni3", 0.0, 0.5), ("hao3", 0.5, 1.0))) + "\n",
        encoding="utf-8")
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.0, 0.5, "ni3"), Interval(0.5, 1.0, "hao3")]),
        Tier("pauses", 0.0, 1.0, []),
    ]), ctc / f"{stem}.TextGrid")
    with wave.open(str(wavs / f"{stem}.wav"), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    words = [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.45, "ni3"),
             Interval(0.45, 0.9, "hao3"), Interval(0.9, 1.0, "！")]
    hanzi = [Interval(iv.xmin, iv.xmax, text) for iv, text in zip(words, ("<sp1>", "你", "好", "！"))]
    phones = [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.45, "n"),
              Interval(0.45, 0.9, "h"), Interval(0.9, 1.0, "<sp1>")]
    final = TextGrid(0.0, 1.0, [
        Tier("raw_text", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1>你好！")]),
        Tier("pinyin", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1> ni3 hao3 ！")]),
        Tier("hanzi", 0.0, 1.0, hanzi), Tier("words", 0.0, 1.0, words),
        Tier("pinyin_phones", 0.0, 1.0, phones),
    ])
    write_textgrid(final, output / f"{stem}.TextGrid")
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.1, 0.45, "ni3"), Interval(0.45, 0.9, "hao3")]),
        Tier("phones", 0.0, 1.0, [Interval(0.1, 0.45, "n"), Interval(0.45, 0.9, "h")]),
    ]), aligned / f"{stem}.TextGrid")
    (output / "postprocess_report.jsonl").write_text(
        json.dumps({"stem": stem, "status": "ok", "warnings": []}) + "\n", encoding="utf-8")
    from pipeline_utils import write_pipeline_accounting_receipt
    write_pipeline_accounting_receipt(
        ctc, source_stems=[stem], eligible_stems=[stem], exclusions=[],
        output_stems=[stem], filtered_stems=[], run_id="fixture", mode="strict")
    en_manifest = en_phones / "en_alignment_manifest.json"
    en_manifest.write_text(json.dumps({
        "schema": EN_PROVENANCE_SCHEMA, "status": "no_english", "strict_provenance": True,
        "mfa": {}, "expected_segments": [], "produced_segments": [], "rejected_segments": [],
        "stem_ledgers": [], "counts": {"english_stems": 0, "english_segments": 0,
            "english_words": 0, "verified_words": 0, "rejected_words": 0}, "reason": "",
    }), encoding="utf-8")
    return {"output": output, "filtered": filtered, "ctc": ctc, "refs": refs, "wavs": wavs,
            "aligned": aligned, "en_phones": en_phones, "en_aligned": en_aligned,
            "en_manifest": en_manifest,
            "pipeline_receipt": ctc / ".pipeline_run_receipt_v2.json"}


def _write_english_unk_fixture(root: Path) -> dict[str, Path]:
    """A real English ``unk`` is valid only with authority + en: phones."""
    from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid

    paths = _write_fixture(root)
    (paths["refs"] / "demo.txt").write_text("unk\n", encoding="utf-8")
    (paths["ctc"] / "demo.lab").write_text("unk\n", encoding="utf-8")
    (paths["ctc"] / "demo_tokens.jsonl").write_text(
        json.dumps({"word": "unk", "start_s": 0.0, "end_s": 1.0}) + "\n", encoding="utf-8")
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "unk")]),
        Tier("pauses", 0.0, 1.0, []),
    ]), paths["ctc"] / "demo.TextGrid")
    words = [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "unk")]
    final = TextGrid(0.0, 1.0, [
        Tier("raw_text", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1>unk")]),
        Tier("pinyin", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1> unk")]),
        Tier("hanzi", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "unk")]),
        Tier("words", 0.0, 1.0, words),
        Tier("pinyin_phones", 0.0, 1.0, [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 1.0, "en:AH")]),
    ])
    write_textgrid(final, paths["output"] / "demo.TextGrid")
    # MFA may use spn for this English region; final English MFA en: phones
    # are authoritative for the final result.
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.1, 1.0, "unk")]),
        Tier("phones", 0.0, 1.0, [Interval(0.1, 1.0, "spn")]),
    ]), paths["aligned"] / "demo.TextGrid")
    # Strict source evidence is deliberately separate from the Chinese MFA
    # alignment fixture above.  The auditor must derive final en:AH timing
    # from this TextGrid, not from the ledger's copied phone list.
    source = paths["en_aligned"] / "demo_seg0.TextGrid"
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "unk")]),
        Tier("phones", 0.0, 1.0, [Interval(0.0, 1.0, "AH")]),
    ]), source)
    from audit_strict_ok import _sha256
    ledger = {
        "schema": EN_PROVENANCE_SCHEMA, "stem": "demo",
        "ctc_textgrid_sha256": _sha256(paths["ctc"] / "demo.TextGrid"),
        "segments": [{"segment_id": "demo:s0", "segment_ordinal": 0, "status": "verified",
            "reason": "", "mfa_textgrid": {"path": str(source), "sha256": _sha256(source)},
            "words": [{"word_id": "demo:s0:w0", "ctc_ordinal": 0, "ctc_text": "unk",
                "start": 0.0, "end": 1.0, "status": "verified", "reason": "",
                "mfa_word": {"ordinal": 0, "text": "unk", "start": 0.0, "end": 1.0},
                "phones": [{"ordinal": 0, "label": "AH", "start": 0.0, "end": 1.0,
                    "mfa_phone_ordinal": 0}], "provenance": "english_mfa_textgrid"}]}],
    }
    ledger_path = paths["en_phones"] / "demo_en_phones.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    paths["en_manifest"].write_text(json.dumps({
        "schema": EN_PROVENANCE_SCHEMA, "status": "success", "strict_provenance": True,
        "mfa": {"return_code": 0, "timed_out": False, "timeout_seconds": 1,
                 "command": ["fixture"], "acoustic_model_sha256": "a" * 64,
                 "dictionary_sha256": "b" * 64, "exception": ""},
        "expected_segments": ["demo:s0"], "produced_segments": ["demo:s0"],
        "rejected_segments": [], "stem_ledgers": [{"stem": "demo", "path": str(ledger_path),
            "sha256": _sha256(ledger_path)}], "counts": {"english_stems": 1,
            "english_segments": 1, "english_words": 1, "verified_words": 1,
            "rejected_words": 0}, "reason": "",
    }), encoding="utf-8")
    return paths


def _write_nvv_fixture(root: Path) -> dict[str, Path]:
    """Complete legal NVV + punctuation + sentence-initial sp1 fixture."""
    from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid

    paths = _write_fixture(root)
    (paths["refs"] / "demo.txt").write_text("<LAUGHTER>，\n", encoding="utf-8")
    (paths["ctc"] / "demo.lab").write_text("<LAUGHTER> ，\n", encoding="utf-8")
    (paths["ctc"] / "demo_tokens.jsonl").write_text(
        "\n".join(json.dumps({"word": word, "start_s": start, "end_s": end})
                  for word, start, end in (("<LAUGHTER>", 0.0, 0.8), ("，", 0.8, 1.0))) + "\n",
        encoding="utf-8")
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.0, 0.8, "<LAUGHTER>"), Interval(0.8, 1.0, "，")]),
        Tier("pauses", 0.0, 1.0, []),
    ]), paths["ctc"] / "demo.TextGrid")
    words = [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.8, "<LAUGHTER>"), Interval(0.8, 1.0, "，")]
    final = TextGrid(0.0, 1.0, [
        Tier("raw_text", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1><LAUGHTER>，")]),
        Tier("pinyin", 0.0, 1.0, [Interval(0.0, 1.0, "<sp1> <LAUGHTER> ，")]),
        Tier("hanzi", 0.0, 1.0, [Interval(iv.xmin, iv.xmax, iv.text) for iv in words]),
        Tier("words", 0.0, 1.0, words),
        Tier("pinyin_phones", 0.0, 1.0, [
            Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.8, "<LAUGHTER>"), Interval(0.8, 1.0, "<sp1>")]),
    ])
    write_textgrid(final, paths["output"] / "demo.TextGrid")
    write_textgrid(TextGrid(0.0, 1.0, [
        Tier("words", 0.0, 1.0, [Interval(0.1, 0.8, "<LAUGHTER>"), Interval(0.8, 1.0, "，")]),
        Tier("phones", 0.0, 1.0, [Interval(0.1, 0.8, "spn")]),
    ]), paths["aligned"] / "demo.TextGrid")
    return paths


def _self_test() -> int:
    from argparse import Namespace
    from audit_strict_ok import audit

    def audit_fixture(paths: dict[str, Path]):
        return audit(Namespace(
            output_dir=paths["output"], filtered_dir=paths["filtered"],
            ctc_dir=paths["ctc"], reference_dir=paths["refs"],
            wav_dir=paths["wavs"], aligned_dir=paths["aligned"],
            en_phones_dir=paths["en_phones"], en_aligned_dir=paths["en_aligned"],
            en_manifest=paths["en_manifest"],
            pipeline_receipt=paths["pipeline_receipt"],
            report=paths["output"] / "postprocess_report.jsonl", isolate=True,
        ))

    failures = 0
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        manifest, clean = audit_fixture(paths)
        manifest_path = paths["output"] / "strict_ok_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if not clean or manifest["safe_empty"] or verify(manifest_path, paths["output"]):
            print("FAIL strict-ok accepted fixture")
            failures += 1
        else:
            print("OK strict-ok accepted fixture")
        (paths["output"] / "demo.TextGrid").write_text("tampered\n", encoding="utf-8")
        if not verify(manifest_path, paths["output"]):
            print("FAIL strict-ok tamper detection")
            failures += 1
        else:
            print("OK strict-ok tamper detection")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        # A zero-length lexical interval is an invariant failure and must move,
        # not remain in output; the result is an explicit, nonpublishable empty.
        target = paths["output"] / "demo.TextGrid"
        target.write_text(target.read_text(encoding="utf-8").replace("xmax = 0.45", "xmax = 0.1", 1), encoding="utf-8")
        manifest, clean = audit_fixture(paths)
        if not manifest["safe_empty"] or (paths["output"] / "demo.TextGrid").exists() or not (paths["filtered"] / "demo.TextGrid").exists():
            print("FAIL strict-ok isolation/safe_empty")
            failures += 1
        else:
            print("OK strict-ok isolation/safe_empty")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        for directory, suffix in ((paths["ctc"], ".lab"), (paths["ctc"], "_tokens.jsonl"),
                                  (paths["ctc"], ".TextGrid"), (paths["refs"], ".txt"),
                                  (paths["wavs"], ".wav"), (paths["aligned"], ".TextGrid")):
            shutil.copy2(directory / f"demo{suffix}", directory / f"other{suffix}")
        shutil.copy2(paths["output"] / "demo.TextGrid", paths["filtered"] / "other.TextGrid")
        from pipeline_utils import write_pipeline_accounting_receipt
        write_pipeline_accounting_receipt(
            paths["ctc"], source_stems=["demo", "other"],
            eligible_stems=["demo", "other"], exclusions=[],
            output_stems=["demo"], filtered_stems=["other"], run_id="fixture",
            mode="strict")
        (paths["ctc"] / "other_tokens.jsonl").write_text(
            (paths["ctc"] / "other_tokens.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        (paths["output"] / "postprocess_report.jsonl").write_text(
            "\n".join((json.dumps({"stem": "demo", "status": "ok", "warnings": []}),
                        json.dumps({"stem": "other", "status": "filtered", "warnings": []}))) + "\n",
            encoding="utf-8")
        manifest, clean = audit_fixture(paths)
        manifest_path = paths["output"] / "strict_ok_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if not clean or verify(manifest_path, paths["output"]) or "other" not in manifest["rejected"]:
            print("FAIL prefiltered manifest partition compatibility")
            failures += 1
        else:
            print("OK prefiltered manifest partition compatibility")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        nested = paths["refs"] / "speaker_a"
        nested.mkdir()
        (paths["refs"] / "demo.txt").replace(nested / "demo.txt")
        manifest, clean = audit_fixture(paths)
        if not clean or manifest["safe_empty"]:
            print("FAIL nested exact reference")
            failures += 1
        else:
            print("OK nested exact reference")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        duplicate = paths["refs"] / "speaker_b"
        duplicate.mkdir()
        (duplicate / "demo.txt").write_text("你好!\n", encoding="utf-8")
        manifest, clean = audit_fixture(paths)
        if clean or "reference_basename_duplicate:demo" not in manifest["global_reasons"]:
            print("FAIL reference basename conflict")
            failures += 1
        else:
            print("OK reference basename conflict")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        (paths["refs"] / "demo_ref.txt").write_text("你好!\n", encoding="utf-8")
        manifest, clean = audit_fixture(paths)
        if clean or "reference_priority_conflict:demo" not in manifest["global_reasons"]:
            print("FAIL reference priority conflict")
            failures += 1
        else:
            print("OK reference priority conflict")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_english_unk_fixture(Path(td))
        manifest, clean = audit_fixture(paths)
        if not clean or manifest["safe_empty"]:
            print("FAIL real English unk with en phones")
            failures += 1
        else:
            print("OK real English unk with en phones")
        manifest_path = paths["output"] / "strict_ok_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        evidence = manifest["ok"][0].get("english_provenance", {}) if manifest["ok"] else {}
        if (not evidence or verify(manifest_path, paths["output"])
                or not (paths["output"] / evidence.get("ledger", {}).get("path", "")).is_file()):
            print("FAIL strict English evidence copied and verified")
            failures += 1
        else:
            print("OK strict English evidence copied and verified")
        # Published verification must reject both historical v2 manifests and
        # a changed self-contained evidence artifact.
        legacy = json.loads(json.dumps(manifest)); legacy["policy_version"] = "strict-ok-v2"
        manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
        old_rejected = verify(manifest_path, paths["output"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        evidence_path = paths["output"] / evidence["ledger"]["path"]
        evidence_path.write_text("tampered", encoding="utf-8")
        evidence_rejected = verify(manifest_path, paths["output"])
        if ("policy_version_mismatch" not in old_rejected
                or not any(item.startswith("english_provenance_ledger_hash_mismatch") for item in evidence_rejected)):
            print("FAIL legacy/evidence tamper rejection")
            failures += 1
        else:
            print("OK legacy/evidence tamper rejection")
        # Explicit MFA placeholder is never rescued by an English reference.
        target = paths["output"] / "demo.TextGrid"
        target.write_text(target.read_text(encoding="utf-8").replace('"unk"', '"<unk>"'), encoding="utf-8")
        manifest, _ = audit_fixture(paths)
        if "demo" not in manifest["rejected"] or not any("unknown" in reason for reason in manifest["rejected"]["demo"]):
            print("FAIL explicit <unk> rejection")
            failures += 1
        else:
            print("OK explicit <unk> rejection")
    with tempfile.TemporaryDirectory() as td:
        from argparse import Namespace
        from audit_strict_ok import _english_provenance_reasons, _load_english_manifest, _strict_parse
        paths = _write_english_unk_fixture(Path(td))
        raw = json.loads(paths["en_manifest"].read_text(encoding="utf-8"))
        raw["produced_segments"] = []
        raw["rejected_segments"] = [{"id": "demo:s0", "reason": "fixture_local_rejection"}]
        raw["counts"]["verified_words"] = 0; raw["counts"]["rejected_words"] = 1
        paths["en_manifest"].write_text(json.dumps(raw), encoding="utf-8")
        ns = Namespace(en_manifest=paths["en_manifest"], en_phones_dir=paths["en_phones"],
                       en_aligned_dir=paths["en_aligned"])
        loaded, errors = _load_english_manifest(ns)
        reasons, _ = _english_provenance_reasons(
            "demo", _strict_parse(paths["output"] / "demo.TextGrid"), paths["ctc"], ns, loaded)
        raw["mfa"]["return_code"] = 1
        paths["en_manifest"].write_text(json.dumps(raw), encoding="utf-8")
        invalid_success, invalid_errors = _load_english_manifest(ns)
        raw["mfa"]["return_code"] = 0; raw["mfa"]["exception"] = "forged exception"
        paths["en_manifest"].write_text(json.dumps(raw), encoding="utf-8")
        exception_success, exception_errors = _load_english_manifest(ns)
        if (not errors and reasons == ["english_segment_rejected"] and invalid_success is None and invalid_errors
                and exception_success is None and exception_errors):
            print("OK local English rejection stays local; invalid success MFA is global")
        else:
            print("FAIL local rejection/global MFA contract")
            failures += 1
    with tempfile.TemporaryDirectory() as td:
        from postprocess_textgrids import Interval, parse_textgrid, write_textgrid
        paths = _write_english_unk_fixture(Path(td))
        final_path = paths["output"] / "demo.TextGrid"
        final = parse_textgrid(final_path)
        final.tiers[4].intervals = [Interval(0.0, 0.05, "en:AA"),
                                    Interval(0.05, 0.1, "<sp1>"),
                                    Interval(0.1, 1.0, "en:AH")]
        write_textgrid(final, final_path)
        manifest, _ = audit_fixture(paths)
        if "final_sequence_mismatch" not in manifest["rejected"].get("demo", []):
            print("FAIL extra en phone outside English evidence")
            failures += 1
        else:
            print("OK extra en phone outside English evidence")
    with tempfile.TemporaryDirectory() as td:
        from argparse import Namespace
        from audit_strict_ok import _load_english_manifest, _safe_file_under
        paths = _write_fixture(Path(td))
        no_english = json.loads(paths["en_manifest"].read_text(encoding="utf-8"))
        no_english["counts"]["english_words"] = 1
        paths["en_manifest"].write_text(json.dumps(no_english), encoding="utf-8")
        ns = Namespace(en_manifest=paths["en_manifest"], en_phones_dir=paths["en_phones"],
                       en_aligned_dir=paths["en_aligned"])
        bad_no_english, no_english_errors = _load_english_manifest(ns)
        target = paths["en_phones"] / "real.json"; target.write_text("{}", encoding="utf-8")
        linked = paths["en_phones"] / "linked.json"; linked.symlink_to(target)
        try:
            _safe_file_under(paths["en_phones"], str(linked)); symlink_rejected = False
        except ValueError:
            symlink_rejected = True
        if bad_no_english is None and no_english_errors and symlink_rejected:
            print("OK no-English tamper and symlink evidence are rejected")
        else:
            print("FAIL no-English/symlink contract")
            failures += 1
    with tempfile.TemporaryDirectory() as td:
        paths = _write_english_unk_fixture(Path(td))
        report = paths["output"] / "postprocess_report.jsonl"
        report.write_text(report.read_text(encoding="utf-8") + "not-json\n", encoding="utf-8")
        manifest, clean = audit_fixture(paths)
        if clean or not manifest["global_reasons"] or (paths["output"] / "_provenance").exists():
            print("FAIL failed run leaves staged English evidence")
            failures += 1
        else:
            print("OK failed run cleans staged English evidence")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        old_evidence = paths["output"] / "_provenance" / "english"
        old_evidence.mkdir(parents=True)
        (old_evidence / "user-kept.txt").write_text("do not remove", encoding="utf-8")
        manifest, clean = audit_fixture(paths)
        if (clean or "english_provenance_evidence_collision" not in manifest["global_reasons"]
                or not (old_evidence / "user-kept.txt").is_file()):
            print("FAIL pre-existing evidence collision is global and preserved")
            failures += 1
        else:
            print("OK pre-existing evidence collision is global and preserved")
    with tempfile.TemporaryDirectory() as td:
        from postprocess_textgrids import Interval, parse_textgrid, write_textgrid
        paths = _write_english_unk_fixture(Path(td))
        final_path = paths["output"] / "demo.TextGrid"; final = parse_textgrid(final_path)
        final.tiers[4].intervals = [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.6, "en:AH"),
                                    Interval(0.6, 0.61, "<sp0>"), Interval(0.61, 1.0, "en:AH")]
        write_textgrid(final, final_path)
        manifest, _ = audit_fixture(paths)
        if "final_sequence_mismatch" not in manifest["rejected"].get("demo", []):
            print("FAIL silence inside English word is rejected")
            failures += 1
        else:
            print("OK silence inside English word is rejected")
    with tempfile.TemporaryDirectory() as td:
        from unittest.mock import patch
        import audit_strict_ok as strict_audit
        paths = _write_english_unk_fixture(Path(td))
        argv = ["audit_strict_ok.py", "--output-dir", str(paths["output"]),
                "--filtered-dir", str(paths["filtered"]), "--ctc-dir", str(paths["ctc"]),
                "--reference-dir", str(paths["refs"]), "--wav-dir", str(paths["wavs"]),
                "--aligned-dir", str(paths["aligned"]), "--en-phones-dir", str(paths["en_phones"]),
                "--en-aligned-dir", str(paths["en_aligned"]), "--en-manifest", str(paths["en_manifest"]),
                "--report", str(paths["output"] / "postprocess_report.jsonl")]
        with patch.object(sys, "argv", argv), patch.object(strict_audit, "_evidence_recheck", return_value=["fixture"]):
            rc = strict_audit.main()
        persisted = json.loads((paths["output"] / "strict_ok_manifest.json").read_text(encoding="utf-8"))
        if (rc != 1 or (paths["output"] / "_provenance" / "english").exists()
                or "fixture" not in persisted["global_reasons"]
                or any("english_provenance" in entry for entry in persisted["ok"])):
            print("FAIL post-commit evidence recheck rollback")
            failures += 1
        else:
            print("OK post-commit evidence recheck rollback")
    with tempfile.TemporaryDirectory() as td:
        from argparse import Namespace
        from audit_strict_ok import _english_provenance_reasons, _load_english_manifest, _strict_parse
        paths = _write_english_unk_fixture(Path(td))
        ledger_path = paths["en_phones"] / "demo_en_phones.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        duplicated = json.loads(json.dumps(ledger["segments"][0]))
        duplicated["segment_id"] = "demo:s1"; duplicated["segment_ordinal"] = 1
        duplicated["words"][0]["word_id"] = "demo:s1:w0"
        ledger["segments"].append(duplicated); ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        run = json.loads(paths["en_manifest"].read_text(encoding="utf-8"))
        run["expected_segments"].append("demo:s1"); run["produced_segments"].append("demo:s1")
        run["counts"].update({"english_segments": 2, "english_words": 2, "verified_words": 2})
        from audit_strict_ok import _sha256
        run["stem_ledgers"][0]["sha256"] = _sha256(ledger_path)
        paths["en_manifest"].write_text(json.dumps(run), encoding="utf-8")
        ns = Namespace(en_manifest=paths["en_manifest"], en_phones_dir=paths["en_phones"],
                       en_aligned_dir=paths["en_aligned"])
        loaded, errors = _load_english_manifest(ns)
        reasons, _ = _english_provenance_reasons("demo", _strict_parse(paths["output"] / "demo.TextGrid"),
                                                 paths["ctc"], ns, loaded)
        if not errors and reasons == ["source_textgrid_missing"]:
            print("OK duplicate segment cannot reuse another source TextGrid")
        else:
            print("FAIL duplicate source TextGrid contract")
            failures += 1
    with tempfile.TemporaryDirectory() as td:
        from postprocess_textgrids import Interval, TextGrid, Tier, parse_textgrid, write_textgrid
        from audit_strict_ok import _sha256
        paths = _write_english_unk_fixture(Path(td))
        source = paths["en_aligned"] / "demo_seg0.TextGrid"
        write_textgrid(TextGrid(0.0, 1.0, [
            Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "unk")]),
            Tier("phones", 0.0, 1.0, [Interval(0.0, 1 / 900, "AH"), Interval(1 / 900, 1.0, "L")]),
        ]), source)
        ledger_path = paths["en_phones"] / "demo_en_phones.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        segment = ledger["segments"][0]; segment["mfa_textgrid"]["sha256"] = _sha256(source)
        segment["words"][0]["phones"] = [
            {"ordinal": 0, "label": "AH", "start": 0.0, "end": 1 / 900, "mfa_phone_ordinal": 0},
            {"ordinal": 1, "label": "L", "start": 1 / 900, "end": 1.0, "mfa_phone_ordinal": 1},
        ]
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        run = json.loads(paths["en_manifest"].read_text(encoding="utf-8"))
        run["stem_ledgers"][0]["sha256"] = _sha256(ledger_path)
        paths["en_manifest"].write_text(json.dumps(run), encoding="utf-8")
        final_path = paths["output"] / "demo.TextGrid"; final = parse_textgrid(final_path)
        final.tiers[4].intervals = [Interval(0.0, 0.1, "<sp1>"), Interval(0.1, 0.101, "en:AH"),
                                    Interval(0.101, 1.0, "en:L")]
        write_textgrid(final, final_path)
        manifest, clean = audit_fixture(paths)
        if clean and not manifest["safe_empty"]:
            print("OK 1ms final English phone is independently accepted")
        else:
            print("FAIL 1ms final English phone contract")
            failures += 1
    with tempfile.TemporaryDirectory() as td:
        from audit_strict_ok import _aligned_reasons
        from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid
        root = Path(td)
        nvv = root / "nvv.TextGrid"
        pinyin = root / "pinyin.TextGrid"
        write_textgrid(TextGrid(0.0, 1.0, [
            Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "<LAUGHTER>")]),
            Tier("phones", 0.0, 1.0, [Interval(0.0, 1.0, "spn")]),
        ]), nvv)
        write_textgrid(TextGrid(0.0, 1.0, [
            Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "ni3")]),
            Tier("phones", 0.0, 1.0, [Interval(0.0, 1.0, "spn")]),
        ]), pinyin)
        if _aligned_reasons(nvv, "<LAUGHTER>") or "aligned_lexical_spn" not in _aligned_reasons(pinyin, "你"):
            print("FAIL owner-aware aligned spn")
            failures += 1
        else:
            print("OK owner-aware aligned spn")
    with tempfile.TemporaryDirectory() as td:
        from audit_strict_ok import _source_english_words
        from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid
        source = Path(td) / "source.TextGrid"
        write_textgrid(TextGrid(0.0, 1.0, [
            Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "hello")]),
            Tier("phones", 0.0, 1.0, [Interval(0.0, .001, "HH"), Interval(.001, 1.0, "AH")]),
        ]), source)
        short_positive = _source_english_words(source)
        write_textgrid(TextGrid(0.0, 1.0, [
            Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "hello")]),
            Tier("phones", 0.0, 1.0, [Interval(0.0, .7, "HH"), Interval(.699, 1.0, "AH")]),
        ]), source)
        try:
            _source_english_words(source)
            overlap_rejected = False
        except ValueError:
            overlap_rejected = True
        if len(short_positive) == 1 and overlap_rejected:
            print("OK strict source accepts short-positive and rejects 1ms overlap")
        else:
            print("FAIL strict source short/overlap contract")
            failures += 1
    with tempfile.TemporaryDirectory() as td:
        paths = _write_nvv_fixture(Path(td))
        manifest, clean = audit_fixture(paths)
        if (not clean or manifest["safe_empty"] or [entry["stem"] for entry in manifest["ok"]] != ["demo"]
                or not (paths["output"] / "demo.TextGrid").read_text(encoding="utf-8").count("<LAUGHTER>")):
            print("FAIL legal NVV/punct/sp1 fixture")
            failures += 1
        else:
            print("OK legal NVV/punct/sp1 fixture")
    from pipeline_utils import is_unknown_token, publish_output_versioned, write_publish_manifest
    from run_pipeline import NVASR_FALLBACK_STEP_ORDER, strict_run_paths
    if is_unknown_token("unk") or not is_unknown_token("<unk>"):
        print("FAIL explicit unknown classification")
        failures += 1
    else:
        print("OK explicit unknown classification")
    if (NVASR_FALLBACK_STEP_ORDER.index("normalize_en") + 1 != NVASR_FALLBACK_STEP_ORDER.index("resample")
            or NVASR_FALLBACK_STEP_ORDER.index("resample") + 1 != NVASR_FALLBACK_STEP_ORDER.index("adjust")):
        print("FAIL fallback resample ordering")
        failures += 1
    else:
        print("OK fallback resample ordering")
    out, filtered, target = strict_run_paths(Path("/tmp/work"), Path("/tmp/nas/out"), "run", False)
    if target is not None or out != Path("/tmp/work/strict_ok_runs/run/output") or filtered != Path("/tmp/work/strict_ok_runs/run/filtered"):
        print("FAIL no-output-staging path semantics")
        failures += 1
    else:
        print("OK no-output-staging path semantics")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        manifest, clean = audit_fixture(paths)
        manifest_path = paths["output"] / "strict_ok_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        destination = Path(td) / "published.runs" / "run"
        write_publish_manifest(paths["output"])
        if not clean or not publish_output_versioned(paths["output"], destination) or verify(destination / "strict_ok_manifest.json", destination):
            print("FAIL destination strict re-verification")
            failures += 1
        else:
            print("OK destination strict re-verification")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        manifest, clean = audit_fixture(paths)
        manifest_path = paths["output"] / "strict_ok_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        tampered = json.loads(json.dumps(manifest))
        tampered["ok"][0]["stem"] = "../escape"
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        unsafe = verify(manifest_path, paths["output"])
        tampered = json.loads(json.dumps(manifest))
        tampered["ok"].append(dict(tampered["ok"][0]))
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        duplicate = verify(manifest_path, paths["output"])
        tampered = json.loads(json.dumps(manifest))
        tampered["expected_stems"] = []
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        set_mismatch = verify(manifest_path, paths["output"])
        tampered = json.loads(json.dumps(manifest))
        tampered["rejected"]["demo"] = ["forged"]
        manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
        overlap = verify(manifest_path, paths["output"])
        if (not clean or "unsafe_ok_stem" not in unsafe or not any(error.startswith("duplicate_ok_stem") for error in duplicate)
                or "expected_ok_rejected_set_mismatch" not in set_mismatch or "ok_rejected_overlap" not in overlap):
            print("FAIL strict manifest partition tampering")
            failures += 1
        else:
            print("OK strict manifest partition tampering")
    with tempfile.TemporaryDirectory() as td:
        paths = _write_fixture(Path(td))
        manifest, clean = audit_fixture(paths)
        (paths["output"] / "strict_ok_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tone = paths["output"] / "tone_mapping.json"
        tone.write_text("{}", encoding="utf-8")
        write_publish_manifest(paths["output"])
        tone.write_text("[]", encoding="utf-8")  # same byte length, different digest
        destination = Path(td) / "tampered.runs" / "run"
        if not clean or publish_output_versioned(paths["output"], destination):
            print("FAIL same-length non-TextGrid publish tampering")
            failures += 1
        else:
            print("OK same-length non-TextGrid publish tampering")
    # Pipeline receipt publish-gate regressions: every invalid receipt must be
    # rejected before a destination target is created.
    for label, mutate in (
            ("missing", lambda path: path.unlink()),
            ("legacy", lambda path: path.write_text(json.dumps({"schema": "pipeline-run-receipt-v1"}), encoding="utf-8")),
            ("tampered", lambda path: path.write_text(path.read_text(encoding="utf-8").replace('"eligible"', '"eligible_tampered"', 1), encoding="utf-8")),
    ):
        with tempfile.TemporaryDirectory() as td:
            paths = _write_fixture(Path(td))
            manifest, clean = audit_fixture(paths)
            manifest_path = paths["output"] / "strict_ok_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            mutate(paths["pipeline_receipt"])
            errors = verify(manifest_path, paths["output"])
            target = Path(td) / "publish-target"
            if not errors or target.exists():
                print(f"FAIL publish gate {label} receipt/no-target")
                failures += 1
            else:
                print(f"OK publish gate {label} receipt/no-target")
    with tempfile.TemporaryDirectory() as td:
        from pipeline_utils import make_pipeline_accounting_receipt, write_pipeline_accounting_receipt
        root = Path(td)
        receipt_path = root / ".pipeline_run_receipt_v2.json"
        write_pipeline_accounting_receipt(
            receipt_path,
            make_pipeline_accounting_receipt(
                source_stems=["ok", "missing"], eligible_stems=["ok"],
                exclusions={"missing": "missing_reference"},
                output_stems=["ok"], filtered_stems=[]))
        binding = {"schema": "pipeline-run-receipt-v2", "path": str(receipt_path),
                   "sha256": _sha256(receipt_path)}
        accounting_manifest = {"pipeline_accounting_receipt": binding}
        if _verify_pipeline_accounting(accounting_manifest, {"ok"}, set()):
            print("FAIL exclusion-versus-rejection accounting separation")
            failures += 1
        else:
            print("OK exclusion-versus-rejection accounting separation")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict-ok manifest hashes and output set.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.manifest is None:
        return _self_test()
    errors = verify(args.manifest, args.output_dir)
    if errors:
        print("strict-ok verification failed: " + ", ".join(errors))
        return 1
    print("strict-ok manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
