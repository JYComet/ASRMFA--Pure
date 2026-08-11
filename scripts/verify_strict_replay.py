#!/usr/bin/env python3
"""Independent verifier for ``strict-replay-import-v1`` receipts.

The verifier is intentionally read-only and fail-closed.  It checks the
canonical 96-slot identity, selected-only asset namespace, copy hashes/inode
independence, path safety, English provenance records, and the official v2
source-denominator receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from pipeline_utils import read_pipeline_accounting_receipt, validate_pipeline_accounting_receipt, stable_json_digest

CANONICAL_SHA256 = "d88b9ac874283dbc67dc38003fb78d872b799597ce940175a8301f78aa2c5bcf"
CANONICAL_SCHEMA = "mfa-quality-canonical-samples-v1"
RECEIPT_SCHEMA = "strict-replay-import-v1"
RECEIPT_SCHEMA_V21 = "strict-replay-import-v2.1"
CONFIG_PATH_SUFFIX = "configs/hecheng_ria_fresh.yaml"
CONFIG_SHA256 = "5d5ec4ca36460646f4cf9193641ab2fc1f5a4fa696633979304746b86b97e70b"
CATEGORIES = {"accepted", "missing_mfa", "pp_tier_gaps", "word_in_silence", "tier_discontinuity", "english_phone_deficit", "short_word", "english_provenance_rejected"}
RANGES = {"000000-017999", "018000-035999", "036000-053999"}
CTC_SUFFIXES = (".TextGrid", ".lab", "_tokens.jsonl", "_punct.json", "_text_cn.txt", "_text_raw.txt")


def _verify_v21(payload: dict, receipt_path: Path, output_dir: Path | None) -> list[str]:
    errors: list[str] = []
    paths = payload.get("paths", {})
    try:
        workspace = _safe_root(paths.get("workspace"), "workspace")
        output = _safe_root(paths.get("output"), "output")
        immutable = Path(paths.get("immutable_import", ""))
        if workspace == output or not immutable.is_absolute() or immutable != workspace / "strict_replay_import.json":
            errors.append("immutable workspace/output path contract mismatch")
        if receipt_path.resolve() != immutable.resolve() or receipt_path.resolve().parent != workspace:
            errors.append("receipt path is not workspace-owned immutable import")
        sidecar = workspace / "strict_replay_import.sha256"
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != _sha256(receipt_path):
            errors.append("immutable import sidecar mismatch")
        if output_dir is not None and output.resolve() != Path(output_dir).resolve():
            errors.append("output directory binding mismatch")
    except (KeyError, TypeError, ValueError, OSError):
        return ["v2.1 path contract invalid"]
    selected = payload.get("selection_slot_records")
    mapping = payload.get("slot_stem_mapping")
    canonical = payload.get("canonical", {})
    try:
        cpath = Path(canonical["path"])
        if canonical.get("schema") != CANONICAL_SCHEMA or canonical.get("sha256") != CANONICAL_SHA256 or not cpath.is_file() or _sha256(cpath) != CANONICAL_SHA256:
            errors.append("canonical identity/digest mismatch")
        cdata = json.loads(cpath.read_text(encoding="utf-8"))
        entries = cdata.get("entries", [])
        if isinstance(mapping, list) and (len(mapping) != 96 or [row.get("stem") for row in mapping] != [row.get("stem") for row in entries]):
            errors.append("slot_stem_mapping differs from canonical")
    except (KeyError, TypeError, OSError):
        errors.append("canonical manifest binding invalid")
    if not isinstance(selected, list) or len(selected) != 24 or payload.get("selection_slot_count") != 24 or payload.get("selection_slot_digest") != stable_json_digest(selected):
        errors.append("selection 24/count/digest mismatch")
    source = payload.get("source_stems", []); excluded = payload.get("excluded_stems", []); eligible = payload.get("eligible_stems", [])
    ex_records = payload.get("exclusion_records", [])
    for label, value, count, digest in (("source", source, payload.get("source_count"), payload.get("source_digest")), ("excluded", excluded, payload.get("excluded_count"), payload.get("excluded_digest")), ("eligible", eligible, payload.get("eligible_count"), payload.get("eligible_digest"))):
        if not isinstance(value, list) or value != sorted(value) or count != len(value) or digest != stable_json_digest(value):
            errors.append(f"{label} vector/count/digest mismatch")
    if not isinstance(ex_records, list) or ex_records != sorted(ex_records, key=lambda row: row.get("stem", "")) or payload.get("exclusion_count") != len(ex_records) or payload.get("exclusion_digest") != stable_json_digest(ex_records) or [row.get("stem") for row in ex_records] != excluded or any(not isinstance(row, dict) or set(row) != {"stem", "reason"} for row in ex_records):
        errors.append("exclusion records mismatch")
    if (len(source), len(excluded), len(eligible)) != (21, 3, 18):
        errors.append("source/exclusion/eligible denominator mismatch")
    roles = payload.get("dictionary_roles")
    names = {"chinese_mfa_dictionary", "pinyin_projection_dictionary", "english_pronunciation_dictionary"}
    if not isinstance(roles, dict) or set(roles) != names or payload.get("dictionary_roles_digest") != stable_json_digest(roles):
        errors.append("dictionary roles matrix/digest mismatch")
    for role in roles.values() if isinstance(roles, dict) else []:
        if not isinstance(role, dict) or set(role) != {"path", "sha256"} or not isinstance(role.get("path"), str) or not Path(role["path"]).is_absolute() or not Path(role["path"]).is_file() or Path(role["path"]).is_symlink() or _sha256(Path(role["path"])) != role.get("sha256"):
            errors.append("dictionary role path/hash invalid")
    english_path = workspace / "strict_replay_english_import.json"
    if not english_path.is_file() or english_path.is_symlink():
        errors.append("English v2.1 import missing/unsafe")
    else:
        try:
            english = json.loads(english_path.read_text(encoding="utf-8"))
            if english.get("schema") != "strict-replay-english-import-v2.1" or english.get("replay_import_manifest_sha256") != _sha256(receipt_path):
                errors.append("English v2.1 binding/schema mismatch")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("English v2.1 import unreadable")
    return sorted(set(errors))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_root(raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} missing")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not an ordinary directory")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} symlink ancestor: {cursor}")
    return path.resolve(strict=True)


def _safe_file(raw: object, root: Path, label: str) -> Path:
    if not isinstance(raw, str):
        raise ValueError(f"{label} path missing")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} path not absolute")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes root") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not an ordinary file")
    return path.resolve(strict=True)


def verify(receipt_path: Path, output_dir: Path | None = None) -> list[str]:
    receipt_path = Path(receipt_path)
    if output_dir is not None:
        output_dir = Path(output_dir)
    errors: list[str] = []
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"receipt unreadable: {exc}"]
    if isinstance(payload, dict) and payload.get("schema") == RECEIPT_SCHEMA_V21:
        return _verify_v21(payload, receipt_path, output_dir)
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        return ["receipt schema is not strict-replay-import-v2.1"]
    try:
        workspace = _safe_root(payload.get("paths", {}).get("workspace"), "workspace")
        output = _safe_root(payload.get("paths", {}).get("output"), "output")
        contract = payload.get("config_contract", {})
        if (not isinstance(contract, dict) or not str(contract.get("path", "")).endswith(CONFIG_PATH_SUFFIX)
                or contract.get("sha256") != CONFIG_SHA256 or not isinstance(contract.get("postprocess"), dict)):
            errors.append("config contract missing/mismatch")
        if output_dir is not None and output != output_dir.resolve():
            errors.append("output directory binding mismatch")
        if receipt_path.resolve().parent != output:
            errors.append("receipt is not in output directory")
        canonical = payload.get("canonical", {})
        cpath = _safe_file(canonical.get("path"), Path(canonical["path"]).parent.resolve(), "canonical manifest")
        if canonical.get("schema") != CANONICAL_SCHEMA or canonical.get("sha256") != CANONICAL_SHA256 or _sha256(cpath) != CANONICAL_SHA256:
            errors.append("canonical identity/digest mismatch")
        cdata = json.loads(cpath.read_text(encoding="utf-8"))
        entries = cdata.get("entries", [])
        if cdata.get("count") != 96 or len(entries) != 96:
            errors.append("canonical slot count is not 96")
        counts = {}
        for row in entries:
            key = (row.get("category"), row.get("range"))
            counts[key] = counts.get(key, 0) + 1
        if any(counts.get((cat, rng), 0) != 4 for cat in CATEGORIES for rng in RANGES):
            errors.append("canonical category-range slots are not 8x3x4")
        slots = payload.get("slots")
        if not isinstance(slots, list) or len(slots) != 96:
            errors.append("receipt slots missing/not 96")
        elif [s.get("stem") for s in slots] != [e.get("stem") for e in entries]:
            errors.append("receipt slot-to-stem mapping differs from canonical")
        selected = payload.get("selected_slot_records")
        if selected is None and isinstance(payload.get("selected_slots"), list):
            selected = payload.get("selected_slots")
        selected_count = payload.get("selected_slots") if isinstance(payload.get("selected_slots"), int) else len(selected or [])
        if not isinstance(selected, list) or len(selected) not in (24, 96):
            errors.append("selected_slots must be canonical 24-slot pilot or full 96")
            selected = []
        elif not {s.get("slot") for s in selected}.issubset({s.get("slot") for s in slots or []}):
            errors.append("selected_slots are not a canonical subset")
        if payload.get("source_manifest_slots") != 96 or payload.get("selected_slots_count") != len(selected) or selected_count != len(selected):
            errors.append("source/selected slot counts mismatch")
        if payload.get("pilot_selector_version") != "strict-replay-selector-v1":
            errors.append("pilot selector version mismatch")
        if len(selected) == 24:
            pairs = {(s.get("category"), s.get("range")) for s in selected}
            if len(pairs) != 24:
                errors.append("pilot selection is not exactly 8x3")
            for row in (payload.get("pilot_selector", {}).get("rank_rows", []) or []):
                stem = row.get("stem")
                if not isinstance(stem, str) or row.get("rank_hash") != hashlib.sha256(stem.encode("utf-8")).hexdigest():
                    errors.append("pilot rank/hash mismatch")
        slot_assets = payload.get("slot_assets")
        if (not isinstance(slot_assets, list) or len(slot_assets) != 96
                or [s.get("slot") for s in slot_assets] != [s.get("slot") for s in selected]
                or [s.get("stem") for s in slot_assets] != [s.get("stem") for s in selected]):
            if not (isinstance(slot_assets, list) and len(slot_assets) == len(selected)
                    and [s.get("slot") for s in slot_assets] == [s.get("slot") for s in selected]
                    and [s.get("stem") for s in slot_assets] == [s.get("stem") for s in selected]):
                errors.append("per-slot asset mapping missing/not selected canonical")
        stems = {s.get("stem") for s in selected}
        assets = payload.get("assets")
        if not isinstance(assets, list) or {a.get("stem") for a in assets} != stems:
            errors.append("asset namespace is not selected-only unique stems")
        for bundle in assets or []:
            stem = bundle.get("stem")
            if not isinstance(stem, str) or Path(stem).name != stem:
                errors.append("invalid asset stem")
                continue
            raw_assets = bundle.get("assets", {})
            for role in ("authoritative_wav", "authoritative_txt"):
                rec = raw_assets.get(role, {})
                src = _safe_file(rec.get("source"), _safe_root(payload["paths"]["source_root"], "source root"), role)
                cp = _safe_file(rec.get("copy"), workspace, role + " copy")
                if rec.get("sha256") != _sha256(src) or rec.get("sha256") != _sha256(cp) or os.path.samestat(src.stat(), cp.stat()):
                    errors.append(f"{stem} {role} hash/inode mismatch")
            for suffix in CTC_SUFFIXES:
                rec = raw_assets.get("ctc", {}).get(suffix, {})
                src = _safe_file(rec.get("source"), _safe_root(payload["paths"]["ctc_root"], "CTC root"), f"{stem} CTC")
                cp = _safe_file(rec.get("copy"), workspace, f"{stem} CTC copy")
                if rec.get("sha256") != _sha256(src) or rec.get("sha256") != _sha256(cp) or os.path.samestat(src.stat(), cp.stat()):
                    errors.append(f"{stem} CTC {suffix} hash/inode mismatch")
            aligned = bundle.get("aligned", {})
            if aligned.get("status") == "missing_mfa":
                if aligned.get("reason") != "missing_mfa_alignment":
                    errors.append(f"{stem} missing aligned reason invalid")
            else:
                src = _safe_file(aligned.get("source"), _safe_root(payload["paths"]["aligned_root"], "aligned root"), f"{stem} aligned")
                cp = _safe_file(aligned.get("copy"), workspace, f"{stem} aligned copy")
                if _sha256(src) != _sha256(cp) or os.path.samestat(src.stat(), cp.stat()):
                    errors.append(f"{stem} aligned hash/inode mismatch")
            english = bundle.get("english", {})
            for key, root_key in (("producer_manifest", "english_root"), ("ledger", "english_root")):
                rec = english.get(key, {})
                src = _safe_file(rec.get("source"), _safe_root(payload["paths"][root_key], "English root"), key)
                cp = _safe_file(rec.get("copy"), workspace, key + " copy")
                if _sha256(src) != _sha256(cp) or rec.get("sha256") != _sha256(src) or os.path.samestat(src.stat(), cp.stat()):
                    errors.append(f"{stem} {key} hash/inode mismatch")
            if english.get("validation", {}).get("valid") is not True:
                errors.append(f"{stem} English validation failed")
        missing = set(payload.get("missing_mfa_alignment", []))
        if not missing <= stems:
            errors.append("missing-MFA complement contains unknown stems")
        report = payload.get("report", {})
        if report.get("source") != len(stems) or report.get("eligible") != len(stems) - len(missing) or report.get("output") + report.get("filtered") != report.get("eligible"):
            errors.append("source/eligible/output/filtered/report conservation failed")
        accounting_path = output / ".pipeline_run_receipt_v2.json"
        accounting = read_pipeline_accounting_receipt(accounting_path)
        errors.extend(validate_pipeline_accounting_receipt(accounting))
        if set(accounting["source"]["stems"]) != stems or set(accounting["exclusions"][i]["stem"] for i in range(len(accounting["exclusions"]))) != missing:
            errors.append("accounting source/missing complement mismatch")
        if any(stage.get("return_code") != 0 or stage.get("reasons") for stage in payload.get("stages", [])):
            errors.append("stage return code/reasons fail-closed")
        if accounting.get("extra", {}).get("failed_steps"):
            errors.append("accounting failed_steps is non-empty")
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict-replay-import-v1 receipt")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    errors = verify(args.receipt, args.output_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"strict replay verified: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
