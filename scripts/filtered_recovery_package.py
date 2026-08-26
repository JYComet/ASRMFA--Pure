"""Build and validate the frozen RIA-262 recovery package.

The recovery package is deliberately a manifest-only boundary.  It reads one
named strict run, records hashes for the evidence used by replay, and never
selects a run with a glob or writes below the parent workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "filtered-recovery-package-v2"
FROZEN_SCHEMA = "filtered-recovery-frozen-v2"
ACCEPTED_SCHEMA = "filtered-recovery-accepted-v2"
EVIDENCE_SCHEMA = "filtered-recovery-evidence-v3"
STRICT_ENGLISH_SCHEMA = "strict-en-mfa-v2"
HISTORICAL_STRICT_ENGLISH_SCHEMA = "strict-en-mfa-v1"
CANONICAL_ENGLISH_UNITS_SCHEMA = "canonical-english-units-v1"
PACKAGE_FILES = {
    "frozen_filtered.json", "accepted_manifest.json",
    "filtered_recovery_manifest.json", "evidence_receipt.json",
    "filtered_reason_ledger.jsonl", "package_receipt.json",
}
FROZEN_COUNT = 262
ARTIFACT_ONLY_COUNT = 259
MFA_AXIS_COUNT = 3
DEFAULT_STRICT_RUN = Path("strict_ok_runs/20260813T144340Z_1364496")


class PackageError(RuntimeError):
    """Raised for any unsafe, stale, ambiguous, or non-frozen package input."""


def _validate_english_ledger(payload: Any, stem: str, label: str) -> dict:
    """Accept only current-v2 English evidence with canonical-unit binding."""
    if not isinstance(payload, dict):
        raise PackageError(f"{label} is not an object")
    if payload.get("schema") != STRICT_ENGLISH_SCHEMA:
        raise PackageError(f"{label} schema is not {STRICT_ENGLISH_SCHEMA}")
    if payload.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
        raise PackageError(f"{label} canonical unit binding missing")
    if payload.get("stem") != stem:
        raise PackageError(f"{label} stem mismatch")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise PackageError(f"{label} segments missing")
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
            raise PackageError(f"{label} segment {index} canonical unit binding missing")
        words = segment.get("words")
        if not isinstance(words, list):
            raise PackageError(f"{label} segment {index} words missing")
        for word_index, word in enumerate(words):
            if not isinstance(word, dict) or word.get("canonical_binding") != CANONICAL_ENGLISH_UNITS_SCHEMA:
                raise PackageError(f"{label} word {index}:{word_index} canonical unit binding missing")
    return payload


def _validate_english_manifest(payload: Any, label: str) -> dict:
    """Accept only a successful current-v2 English producer manifest."""
    if not isinstance(payload, dict):
        raise PackageError(f"{label} is not an object")
    if payload.get("schema") != STRICT_ENGLISH_SCHEMA:
        raise PackageError(f"{label} schema is not {STRICT_ENGLISH_SCHEMA}")
    if payload.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
        raise PackageError(f"{label} canonical unit binding missing")
    if payload.get("strict_provenance") is not True:
        raise PackageError(f"{label} strict provenance missing")
    if payload.get("status") not in {"success", "no_english"}:
        raise PackageError(f"{label} status is not successful")
    return payload


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"invalid JSON: {path}: {exc}") from exc


def _ordinary_file(path: Path, root: Path, label: str) -> Path:
    """Resolve an explicitly named file without following a symlink escape."""
    raw = Path(path)
    if raw.is_absolute():
        candidate = raw
    else:
        candidate = root / raw
    if candidate.is_symlink():
        raise PackageError(f"{label} is a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise PackageError(f"{label} escapes parent: {candidate}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise PackageError(f"{label} is not an ordinary file: {candidate}")
    return resolved


def _ordinary_dir(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path))
    if raw.is_symlink() or not raw.is_dir():
        raise PackageError(f"{label} is not an ordinary directory: {raw}")
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PackageError(f"{label} has symlink ancestor: {cursor}")
    return raw.resolve(strict=True)


def _safe_rel(path: Path, root: Path, label: str = "path") -> str:
    try:
        rel = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise PackageError(f"{label} escapes parent: {path}") from exc
    if path.is_symlink() or rel.is_absolute() or ".." in rel.parts:
        raise PackageError(f"{label} is unsafe: {path}")
    return rel.as_posix()


def _find_one(root: Path, pattern: str, *, required: bool = True) -> Path | None:
    """Compatibility helper which intentionally forbids loose glob selection."""
    if any(ch in pattern for ch in "*?[]"):
        raise PackageError(f"ambiguous glob selection forbidden: {pattern}")
    candidate = root / pattern
    if candidate.is_file() and not candidate.is_symlink():
        return candidate
    if required:
        raise PackageError(f"explicit parent artifact missing: {candidate}")
    return None


def _stems(values: Any, label: str) -> list[str]:
    if not isinstance(values, list):
        raise PackageError(f"{label} must be a list")
    out = []
    for value in values:
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise PackageError(f"{label} contains malformed stem")
        out.append(value)
    if len(out) != len(set(out)):
        raise PackageError(f"{label} contains duplicate stems")
    return sorted(out)


def _report_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PackageError(f"cannot read report: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackageError(f"malformed report row: {exc}") from exc
        stem = row.get("stem") if isinstance(row, dict) else None
        if not isinstance(stem, str) or not stem or stem in rows:
            raise PackageError("report rows must have unique valid stems")
        row["_report_row_sha256"] = hashlib.sha256((line + "\n").encode()).hexdigest()
        rows[stem] = row
    return rows


def _digest_stems(stems: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(stems), ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def validate_frozen_selection(frozen: Any, accepted: Any, selected: Any,
                              axis_missing: Any) -> dict[str, list[str]]:
    """Validate the immutable 1738/262 partition and its 259+3 axis split."""
    frozen_stems = _stems(frozen, "frozen stems")
    accepted_stems = _stems(accepted, "accepted stems")
    selected_stems = _stems(selected, "selected stems")
    axis_stems = _stems(axis_missing, "MFA-axis stems")
    if len(frozen_stems) != FROZEN_COUNT:
        raise PackageError("frozen selection must contain exactly 262 stems")
    if len(accepted_stems) != 1738:
        raise PackageError("accepted selection must contain exactly 1738 stems")
    if selected_stems != frozen_stems:
        raise PackageError("recovery selection must be the complete frozen set")
    if set(frozen_stems) & set(accepted_stems):
        raise PackageError("recovery selection intersects accepted stems")
    if len(axis_stems) != MFA_AXIS_COUNT or not set(axis_stems) <= set(frozen_stems):
        raise PackageError("MFA-axis selection must be exactly three frozen stems")
    artifact_stems = sorted(set(frozen_stems) - set(axis_stems))
    if len(artifact_stems) != ARTIFACT_ONLY_COUNT:
        raise PackageError("artifact-only selection must contain exactly 259 stems")
    return {"frozen": frozen_stems, "accepted": accepted_stems,
            "selected": selected_stems, "axis_missing": axis_stems,
            "artifact_only": artifact_stems}


def validate_asset_scope(assets: Any, frozen: Any, accepted: Any) -> None:
    """Reject accepted or non-frozen assets before replay import construction."""
    frozen_set = set(_stems(frozen, "frozen stems"))
    accepted_set = set(_stems(accepted, "accepted stems"))
    if not isinstance(assets, list):
        raise PackageError("recovery assets must be a list")
    for row in assets:
        if not isinstance(row, dict):
            raise PackageError("recovery asset row malformed")
        stem = row.get("stem")
        if stem == "__parent__":
            continue
        if stem in accepted_set:
            raise PackageError("accepted-stem import attempted")
        if stem not in frozen_set:
            raise PackageError("recovery asset is outside frozen set")


def validate_parent_hashes(root: Path, expected: Any) -> dict[str, str]:
    """Re-hash an explicit parent map and reject traversal, symlink, or drift."""
    root = _ordinary_dir(root, "parent root")
    if not isinstance(expected, dict):
        raise PackageError("parent hash map missing")
    observed = {}
    for rel, digest in sorted(expected.items()):
        rel_path = Path(str(rel))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise PackageError(f"parent hash path unsafe: {rel}")
        path = _ordinary_file(root / rel_path, root, "parent hashed artifact")
        observed[str(rel)] = sha256(path)
        if observed[str(rel)] != digest:
            raise PackageError("parent artifact hash changed")
    return observed


def _explicit_layout(source_root: Path, strict_run: Path | None) -> dict[str, Path]:
    root = _ordinary_dir(source_root, "parent root")
    run_rel = strict_run or DEFAULT_STRICT_RUN
    run = Path(run_rel)
    if run.is_absolute():
        try:
            run.relative_to(root)
        except ValueError as exc:
            raise PackageError("strict run must be inside parent root") from exc
    else:
        if ".." in run.parts:
            raise PackageError("strict run traversal forbidden")
        run = root / run
    run = _ordinary_dir(run, "explicit strict run")
    output = _ordinary_dir(run / "output", "strict output")
    filtered = _ordinary_dir(run / "filtered", "strict filtered output")
    paths = {
        "strict_manifest": output / "strict_ok_manifest.json",
        "report": output / "postprocess_report.jsonl",
        "pipeline_receipt": output / ".pipeline_run_receipt_v2.json",
        "tone_mapping": output / "tone_mapping.json",
        "publish_manifest": output / ".publish_manifest.json",
        "ctc_receipt": root / "ctc_pretg/.ctc_run_receipt.json",
        "ctc_manifest": root / "ctc_pretg/manifest.json",
        "input_axis": root / "ctc_pretg/.mfa_input_axis_receipt.json",
        "alignment_axis": root / ".mfa_alignment_axis_receipt.json",
        "english_manifest": root / "en_phones/en_alignment_manifest.json",
    }
    for label, path in paths.items():
        paths[label] = _ordinary_file(path, root, label)
    paths.update({"root": root, "strict_run": run, "output": output, "filtered": filtered})
    return paths


def _declared_file(raw: Any, root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PackageError(f"{label} path missing")
    return _ordinary_file(Path(raw), root, label)


def _validate_receipts(paths: dict[str, Path], selected: list[str], frozen: list[str],
                      axis_missing: set[str]) -> dict[str, dict]:
    root = paths["root"]
    strict = _json(paths["strict_manifest"])
    pipeline = _json(paths["pipeline_receipt"])
    ctc = _json(paths["ctc_receipt"])
    input_axis = _json(paths["input_axis"])
    alignment = _json(paths["alignment_axis"])
    english = _validate_english_manifest(_json(paths["english_manifest"]),
                                         "strict English producer manifest")
    if pipeline.get("schema") != "pipeline-run-receipt-v2":
        raise PackageError("stale producer receipt: pipeline schema")
    if pipeline.get("mode") not in {"ctc_ready", "nvrasr_fallback", "full"}:
        raise PackageError("stale producer receipt: pipeline mode")
    if strict.get("pipeline_accounting_receipt", {}).get("sha256") not in {None, sha256(paths["pipeline_receipt"])}:
        raise PackageError("stale producer receipt: strict manifest accounting hash")
    declared_pipeline = strict.get("pipeline_accounting_receipt", {}).get("path")
    if declared_pipeline and Path(declared_pipeline).resolve() != paths["pipeline_receipt"]:
        raise PackageError("stale producer receipt: strict manifest accounting path")
    if ctc.get("schema") != "ctc-run-receipt-v2" or ctc.get("audio_axis_role") != "ctc_input_audio":
        raise PackageError("stale producer receipt: CTC anchor receipt")
    if input_axis.get("schema") != "mfa-input-axis-receipt-v1" \
            or input_axis.get("source_role") != "mfa_axis_audio" \
            or input_axis.get("tts_authoritative_audio_root") != str(root / "padded_audio"):
        raise PackageError("stale producer receipt: original-rate padded-audio axis")
    if alignment.get("schema") != "mfa-alignment-axis-receipt-v2":
        raise PackageError("stale producer receipt: MFA alignment axis")
    # Every producer universe must be the same sealed 2000-stem universe.
    if len(selected) != 2000:
        raise PackageError("parent strict denominator is not 2000")
    if _stems(strict.get("expected_stems"), "strict expected stems") != selected:
        raise PackageError("strict expected stems disagree with selected universe")
    if _stems(input_axis.get("stems"), "MFA input-axis stems") != selected \
            or _stems(alignment.get("stems"), "MFA alignment stems") != selected:
        raise PackageError("producer receipt stem universe drift")
    axis_rows = alignment.get("alignments")
    if not isinstance(axis_rows, list) or len(axis_rows) != 2000:
        raise PackageError("MFA alignment receipt rows are stale")
    actual_missing = {r.get("stem") for r in axis_rows
                      if isinstance(r, dict) and r.get("status") == "missing_mfa_alignment"}
    if actual_missing != axis_missing or len(actual_missing) != MFA_AXIS_COUNT:
        raise PackageError("MFA-axis exception set is not exactly three")
    if english.get("status") not in {"success", "no_english"}:
        raise PackageError("strict English producer is not successful")
    ledgers = english.get("stem_ledgers")
    if not isinstance(ledgers, list) or {r.get("stem") for r in ledgers if isinstance(r, dict)} != set(selected):
        raise PackageError("strict English ledger universe drift")
    ledger_by_stem = {r["stem"]: r for r in ledgers}
    for stem in frozen:
        row = ledger_by_stem.get(stem)
        if not isinstance(row, dict):
            raise PackageError(f"strict English ledger missing: {stem}")
        ledger = _declared_file(row.get("path"), root, f"English ledger {stem}")
        if row.get("sha256") != sha256(ledger):
            raise PackageError(f"stale producer receipt: English ledger {stem}")
    return {"strict": strict, "pipeline": pipeline, "ctc": ctc,
            "input_axis": input_axis, "alignment": alignment,
            "english": english, "ledger_by_stem": ledger_by_stem}


def _reconcile(source_root: Path, strict_run: Path | None = None) -> tuple[dict, dict, dict, dict]:
    paths = _explicit_layout(source_root, strict_run)
    root, filtered = paths["root"], paths["filtered"]
    strict = _json(paths["strict_manifest"])
    expected = _stems(strict.get("expected_stems"), "strict expected stems")
    accepted_rows = strict.get("ok")
    rejected = strict.get("rejected")
    if not isinstance(accepted_rows, list) or not isinstance(rejected, dict):
        raise PackageError("strict accepted/rejected sections malformed")
    accepted = _stems([r.get("stem") for r in accepted_rows if isinstance(r, dict)], "accepted stems")
    if len(accepted) != len(accepted_rows):
        raise PackageError("accepted rows malformed")
    frozen = _stems(list(rejected), "frozen rejected stems")
    if len(expected) != 2000 or set(accepted) | set(frozen) != set(expected):
        raise PackageError("exact 1738/262/2000 parent partition failed")
    filtered_names = []
    for item in filtered.iterdir():
        if item.is_symlink():
            raise PackageError(f"filtered output contains symlink: {item}")
        if item.is_file() and item.suffix == ".TextGrid":
            filtered_names.append(item.stem)
    if set(filtered_names) != set(frozen) or len(filtered_names) != FROZEN_COUNT:
        raise PackageError("strict filtered output is not exactly the frozen 262 set")
    rows = _report_rows(paths["report"])
    if set(rows) != set(expected):
        raise PackageError("postprocess report universe drift")
    axis = _json(paths["alignment_axis"])
    axis_missing = {r.get("stem") for r in axis.get("alignments", [])
                    if isinstance(r, dict) and r.get("status") == "missing_mfa_alignment"}
    if not axis_missing <= set(frozen):
        raise PackageError("MFA-axis exceptions are outside frozen set")
    selection = validate_frozen_selection(frozen, accepted, frozen, sorted(axis_missing))
    receipts = _validate_receipts(paths, expected, frozen, axis_missing)
    return paths, {"selected": expected, "accepted": accepted, "frozen": frozen,
                   "axis_missing": selection["axis_missing"]}, rows, receipts


def _asset_record(path: Path, root: Path, stem: str, role: str, *, digest: str | None = None) -> dict:
    return {"stem": stem, "role": role, "path": _safe_rel(path, root, role),
            "sha256": digest or sha256(path)}


def _build_assets(paths: dict[str, Path], partition: dict, receipts: dict, report: dict) -> list[dict]:
    root = paths["root"]
    ctc_root = root / "ctc_pretg"
    mfa_audio = root / "audio_16k"
    tts_audio = root / "padded_audio"
    aligned = root / "aligned"
    filtered = paths["filtered"]
    en_root = root / "en_phones"
    assets: list[dict] = []
    accepted = set(partition["accepted"])
    for stem in partition["frozen"]:
        if stem in accepted:
            raise PackageError(f"accepted-stem import attempted: {stem}")
        grid = _ordinary_file(filtered / f"{stem}.TextGrid", root, f"filtered artifact {stem}")
        text = grid.read_text(encoding="utf-8", errors="replace")
        axis_role = stem in set(partition["axis_missing"])
        if axis_role:
            category = "mfa_axis"
            if "<sp1>" in text:
                raise PackageError(f"MFA-axis entry unexpectedly has sentence-initial <sp1>: {stem}")
        else:
            category = "artifact_only"
            if "<sp1>" not in text:
                raise PackageError(f"artifact-only entry lacks sentence-initial <sp1>: {stem}")
        assets.append(_asset_record(grid, root, stem, f"{category}:filtered_textgrid"))
        required = {
            "ctc_anchor_lab": ctc_root / f"{stem}.lab",
            "ctc_anchor_textgrid": ctc_root / f"{stem}.TextGrid",
            "ctc_anchor_tokens": ctc_root / f"{stem}_tokens.jsonl",
            "reference_authority": ctc_root / f"{stem}_ref.txt",
            "punctuation_receipt": ctc_root / f"{stem}_punct.json",
            "mfa_axis_audio": mfa_audio / f"{stem}.wav",
            "tts_authoritative_padded_audio": tts_audio / f"{stem}.wav",
            "english_phone_receipt": en_root / f"{stem}_en_phones.json",
        }
        if not axis_role:
            required["mfa_aligned_textgrid"] = aligned / f"{stem}.TextGrid"
        for role, path in required.items():
            assets.append(_asset_record(_ordinary_file(path, root, role + f" {stem}"), root, stem, role))
        ledger_row = receipts["ledger_by_stem"][stem]
        english_phone = required["english_phone_receipt"]
        if ledger_row.get("sha256") != sha256(english_phone):
            raise PackageError(f"stale producer receipt: English phone receipt {stem}")
        ledger = _validate_english_ledger(
            _json(english_phone), stem, f"strict English ledger {stem}")
        for index, segment in enumerate(ledger.get("segments", [])):
            if not isinstance(segment, dict) or segment.get("status") != "verified":
                continue
            grid_ref = segment.get("mfa_textgrid")
            if not isinstance(grid_ref, dict):
                raise PackageError(f"strict English provenance grid missing: {stem}")
            grid = _declared_file(grid_ref.get("path"), root,
                                  f"English provenance grid {stem}:{index}")
            if grid_ref.get("sha256") != sha256(grid):
                raise PackageError(f"stale producer receipt: English provenance grid {stem}:{index}")
            assets.append(_asset_record(grid, root, stem,
                                        f"english_aligned_segment:{index}"))
        punct = _json(root / required["punctuation_receipt"].relative_to(root))
        if not isinstance(punct, (dict, list)):
            raise PackageError(f"punctuation receipt malformed: {stem}")
        report_row = report[stem]
        if not axis_role:
            if not isinstance(report_row.get("english_provenance"), dict):
                raise PackageError(f"strict English provenance missing from report: {stem}")
            if "ledger_sha256" not in report_row["english_provenance"]:
                raise PackageError(f"strict English provenance receipt missing ledger hash: {stem}")
    tone = _ordinary_file(paths["tone_mapping"], root, "NVV/tone receipt")
    assets.append(_asset_record(tone, root, "__parent__", "nvv_tone_receipt"))
    # The report itself is the frozen punctuation/NVV/sp1 accounting receipt;
    # the package also binds its bytes so a producer cannot be replaced later.
    assets.append(_asset_record(paths["report"], root, "__parent__", "postprocess_receipt"))
    return assets


def _write_once(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PackageError(f"package destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _fresh_package(path: Path) -> Path:
    raw = Path(os.path.abspath(path))
    if raw.exists() or raw.is_symlink():
        raise PackageError(f"package target must be fresh: {raw}")
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PackageError(f"package target has symlink ancestor: {cursor}")
    return raw


def prepare_package(source_root: Path, package_dir: Path, *, strict_run: Path | None = None) -> Path:
    paths, partition, report, receipts = _reconcile(source_root, strict_run)
    package = _fresh_package(package_dir)
    package.mkdir(parents=True)
    root = paths["root"]
    assets = _build_assets(paths, partition, receipts, report)
    validate_asset_scope(assets, partition["frozen"], partition["accepted"])
    global_paths = ["strict_manifest", "report", "pipeline_receipt", "tone_mapping",
                    "publish_manifest", "ctc_receipt", "ctc_manifest", "input_axis",
                    "alignment_axis", "english_manifest"]
    parent_hashes = {_safe_rel(paths[name], root, name): sha256(paths[name]) for name in global_paths}
    frozen = partition["frozen"]
    accepted = partition["accepted"]
    artifact_only = sorted(set(frozen) - set(partition["axis_missing"]))
    frozen_manifest = {
        "schema": FROZEN_SCHEMA, "count": FROZEN_COUNT, "stems": frozen,
        "artifact_only_count": ARTIFACT_ONLY_COUNT,
        "mfa_axis_count": MFA_AXIS_COUNT,
        "artifact_only_stems": artifact_only,
        "mfa_axis_stems": sorted(partition["axis_missing"]),
        "source": "explicit_strict_run_rejected_partition",
    }
    accepted_manifest = {
        "schema": ACCEPTED_SCHEMA, "count": len(accepted), "stems": accepted,
        "ok": [{"stem": stem} for stem in accepted],
        "output_dir": str(paths["output"]),
        "source": "strict_ok_manifest",
    }
    plan = {
        "schema": SCHEMA, "scope": "filtered_recovery_frozen_only",
        "stems": frozen, "source": FROZEN_COUNT, "eligible": FROZEN_COUNT,
        "exclusions": 0, "parent_root": str(root),
        "strict_run": _safe_rel(paths["strict_run"], root, "strict run"),
        "parent_artifact_sha256": parent_hashes,
        "accepted_import_forbidden": True,
        "assets": assets,
    }
    inner_mismatch = {
        "declared_sha256": sha256(paths["pipeline_receipt"]),
        "actual_sha256": sha256(paths["report"]),
    }
    if inner_mismatch["declared_sha256"] == inner_mismatch["actual_sha256"]:
        raise PackageError("inner producer receipt/report hashes unexpectedly equal")
    plan["declared_vs_actual_inner_receipt"] = inner_mismatch
    evidence = {
        "schema": EVIDENCE_SCHEMA, "source_root": str(root),
        "strict_run": _safe_rel(paths["strict_run"], root, "strict run"),
        "frozen_stems_sha256": _digest_stems(frozen),
        "parent_artifact_sha256": parent_hashes,
        "declared_vs_actual_inner_receipt": inner_mismatch,
        "required_receipts": {
            "ctc_anchors": _safe_rel(paths["ctc_receipt"], root),
            "original_rate_padded_audio_axis": _safe_rel(paths["input_axis"], root),
            "strict_english_provenance": _safe_rel(paths["english_manifest"], root),
            "reference_authority": "ctc_pretg/{stem}_ref.txt",
            "nvv_tone": _safe_rel(paths["tone_mapping"], root),
            "sentence_initial_sp1": "strict_run/filtered/{stem}.TextGrid",
            "punctuation": "ctc_pretg/{stem}_punct.json",
        },
        "partition": {"selected": len(partition["selected"]), "accepted": len(accepted),
                      "frozen": FROZEN_COUNT, "artifact_only": ARTIFACT_ONLY_COUNT,
                      "mfa_axis": MFA_AXIS_COUNT},
        "code_hashes": {"scripts/filtered_recovery_package.py": sha256(Path(__file__))},
    }
    ledger = []
    for stem in frozen:
        row = report[stem]
        ledger.append({"stem": stem, "status": "strict_rejected",
                       "disposition": "blocked_valid_rejection",
                       "filter_reasons": row.get("filter_reasons", []),
                       "report_row_sha256": row["_report_row_sha256"],
                       "axis_role": "mfa_axis" if stem in partition["axis_missing"] else "artifact_only"})
    _write_once(package / "frozen_filtered.json", frozen_manifest)
    _write_once(package / "accepted_manifest.json", accepted_manifest)
    _write_once(package / "filtered_recovery_manifest.json", plan)
    _write_once(package / "evidence_receipt.json", evidence)
    ledger_path = package / "filtered_reason_ledger.jsonl"
    ledger_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger), encoding="utf-8")
    ledger_path.chmod(0o444)
    receipt = {"schema": SCHEMA, "source_root": str(root), "strict_run": plan["strict_run"],
               "partition": evidence["partition"], "files": {}}
    for path in sorted(package.iterdir()):
        receipt["files"][path.name] = sha256(path)
    _write_once(package / "package_receipt.json", receipt)
    return package


def preflight_package(package_dir: Path) -> dict:
    package = _ordinary_dir(package_dir, "package directory")
    receipt = _json(package / "package_receipt.json")
    if receipt.get("schema") != SCHEMA:
        raise PackageError("package receipt schema mismatch")
    if set(receipt.get("files", {})) != PACKAGE_FILES - {"package_receipt.json"} \
            or {p.name for p in package.iterdir()} != PACKAGE_FILES:
        raise PackageError("package entries are not the exact expected set")
    for name, expected in receipt["files"].items():
        path = package / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise PackageError(f"package file drift: {name}")
    plan = _json(package / "filtered_recovery_manifest.json")
    frozen = _json(package / "frozen_filtered.json")
    accepted = _json(package / "accepted_manifest.json")
    stems = _stems(plan.get("stems"), "package selected stems")
    frozen_stems = _stems(frozen.get("stems"), "frozen stems")
    accepted_stems = _stems(accepted.get("stems"), "accepted stems")
    if plan.get("schema") != SCHEMA or plan.get("scope") != "filtered_recovery_frozen_only":
        raise PackageError("package is not frozen-only")
    if len(stems) != FROZEN_COUNT or stems != frozen_stems:
        raise PackageError("package is not exactly frozen 262")
    if frozen.get("count") != FROZEN_COUNT or frozen.get("artifact_only_count") != ARTIFACT_ONLY_COUNT \
            or frozen.get("mfa_axis_count") != MFA_AXIS_COUNT:
        raise PackageError("frozen package category counts are not 259+3")
    validate_frozen_selection(frozen_stems, accepted_stems, stems,
                              frozen.get("mfa_axis_stems"))
    source = _ordinary_dir(Path(str(receipt.get("source_root", ""))), "package parent")
    paths, partition, report, receipts = _reconcile(source, Path(str(receipt["strict_run"])))
    if set(partition["frozen"]) != set(stems) or set(partition["accepted"]) != set(accepted_stems):
        raise PackageError("package partition drift")
    if _digest_stems(stems) != _json(package / "evidence_receipt.json").get("frozen_stems_sha256"):
        raise PackageError("frozen stem digest drift")
    plan_hashes = plan.get("parent_artifact_sha256")
    if not isinstance(plan_hashes, dict):
        raise PackageError("parent hash map missing")
    observed = validate_parent_hashes(source, plan_hashes)
    evidence = _json(package / "evidence_receipt.json")
    if evidence.get("parent_artifact_sha256") != plan_hashes:
        raise PackageError("evidence parent hash map mismatch")
    validate_asset_scope(plan.get("assets", []), stems, accepted_stems)
    for asset in plan.get("assets", []):
        stem = asset.get("stem")
        if stem != "__parent__" and stem not in set(stems):
            raise PackageError("package asset is outside frozen set")
        if stem in set(accepted_stems):
            raise PackageError("accepted-stem asset import attempted")
        rel = Path(str(asset.get("path", "")))
        path = _ordinary_file(source / rel, source, "package asset")
        if sha256(path) != asset.get("sha256"):
            raise PackageError(f"package asset hash changed: {asset.get('role')}")
    code_hashes = evidence.get("code_hashes", {})
    if code_hashes.get("scripts/filtered_recovery_package.py") != sha256(Path(__file__)):
        raise PackageError("package producer code drift")
    return {"schema": "filtered-recovery-package-preflight-v2", "ok": True,
            "source_root": str(source), "strict_run": str(paths["strict_run"]),
            "selected": 2000, "accepted": len(accepted_stems), "frozen": FROZEN_COUNT,
            "artifact_only": ARTIFACT_ONLY_COUNT, "mfa_axis": MFA_AXIS_COUNT,
            "parent_artifact_sha256": plan_hashes}


def preflight_source(source_root: Path, *, strict_run: Path | None = None) -> dict:
    paths, partition, _, _ = _reconcile(source_root, strict_run)
    return {"schema": "filtered-recovery-source-preflight-v2", "ok": True,
            "source_root": str(paths["root"]), "strict_run": str(paths["strict_run"]),
            "selected": len(partition["selected"]), "accepted": len(partition["accepted"]),
            "frozen": len(partition["frozen"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--package-dir", type=Path, required=True)
    p.add_argument("--strict-run", type=Path, default=DEFAULT_STRICT_RUN)
    p = sub.add_parser("preflight")
    p.add_argument("--package-dir", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--strict-run", type=Path, default=DEFAULT_STRICT_RUN)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = prepare_package(args.source_root, args.package_dir, strict_run=args.strict_run)
            output = {"package": str(result)}
        elif args.package_dir is not None:
            output = preflight_package(args.package_dir)
        elif args.source_root is not None:
            output = preflight_source(args.source_root, strict_run=args.strict_run)
        else:
            raise PackageError("preflight requires --package-dir or --source-root")
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except PackageError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
