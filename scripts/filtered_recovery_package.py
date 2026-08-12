"""Safe preparation and read-only preflight for an exact filtered-replay package.

This module only snapshots evidence and validates a prepared package.  It never
executes MFA/CTC/postprocess and never writes under the source run root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

SCHEMA = "filtered-recovery-package-v1"
FROZEN_SCHEMA = "filtered-recovery-frozen-v1"
ACCEPTED_SCHEMA = "filtered-recovery-accepted-v1"
LEDGER_SCHEMA = "filtered-recovery-reason-ledger-v1"
EVIDENCE_SCHEMA = "filtered-recovery-evidence-v2"
PACKAGE_FILES = {"frozen_filtered.json", "accepted_manifest.json", "filtered_recovery_manifest.json",
                 "evidence_receipt.json", "filtered_reason_ledger.jsonl"}


class PackageError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageError(f"invalid JSON: {path}: {exc}") from exc


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
    if ".." in raw.parts:
        raise PackageError("package target traversal forbidden")
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PackageError(f"package target has symlink ancestor: {cursor}")
    return raw


def _source_root(path: Path) -> Path:
    root = Path(os.path.abspath(path))
    if not root.is_dir() or root.is_symlink():
        raise PackageError(f"source root is not an ordinary directory: {root}")
    return root.resolve(strict=True)


def _find_one(root: Path, pattern: str, *, required: bool = True) -> Path | None:
    matches = sorted(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())
    if len(matches) != 1 and required:
        raise PackageError(f"expected exactly one {pattern}, found {len(matches)}")
    return matches[0] if matches else None


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve(strict=True).relative_to(root.resolve(strict=True)))
    except (OSError, ValueError) as exc:
        raise PackageError(f"source artifact escapes root: {path}") from exc


def _source_artifacts(root: Path) -> dict[str, Path]:
    run_dirs = sorted(p.parent for p in root.glob("workspace/strict_ok_runs/*/output/strict_ok_manifest.json")
                      if p.is_file() and not p.is_symlink())
    matching = []
    for run_output in run_dirs:
        try:
            strict_probe = _json(run_output / "strict_ok_manifest.json")
            report_probe = _report_rows(run_output / "postprocess_report.jsonl")
            if len(strict_probe.get("ok", [])) == 794 and len(report_probe) == 1000:
                matching.append(run_output)
        except PackageError:
            continue
    if len(matching) == 1:
        run_output = matching[0]
        report_path = run_output / "postprocess_report.jsonl"
        strict_path = run_output / "strict_ok_manifest.json"
        pipeline_path = run_output / ".pipeline_run_receipt_v2.json"
    else:
        report_path = _find_one(root, "workspace/strict_ok_runs/*/output/postprocess_report.jsonl")
        strict_path = _find_one(root, "workspace/strict_ok_runs/*/output/strict_ok_manifest.json")
        pipeline_path = _find_one(root, "workspace/strict_ok_runs/*/output/.pipeline_run_receipt_v2.json")
    artifacts: dict[str, Path] = {}
    candidates = {
        "selected_manifest": root / "selected_manifest.json",
        "report": report_path,
        "strict_manifest": strict_path,
        "pipeline_receipt": pipeline_path,
        "alignment_axis": root / "workspace/.mfa_alignment_axis_receipt_recovered.json",
        "english_manifest": root / "workspace/en_phones/en_alignment_manifest.json",
        "config": root / "resolved_gpu1000_nvrasr_fallback.yaml",
    }
    if not candidates["alignment_axis"].is_file():
        candidates["alignment_axis"] = root / "workspace/.mfa_alignment_axis_receipt.json"
    for label, path in candidates.items():
        if path is None or not path.is_file() or path.is_symlink():
            raise PackageError(f"source artifact missing: {label}: {path}")
        artifacts[label] = path.resolve(strict=True)
    return artifacts


def _selected_stems(payload: dict) -> set[str]:
    rows = payload.get("samples")
    if not isinstance(rows, list):
        raise PackageError("selected manifest samples missing")
    stems = [r.get("stem") for r in rows if isinstance(r, dict)]
    if len(stems) != len(rows) or len(stems) != len(set(stems)) or any(not isinstance(s, str) or not s for s in stems):
        raise PackageError("selected manifest stems malformed or duplicated")
    return set(stems)


def _report_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PackageError(f"malformed report row: {exc}") from exc
        stem = row.get("stem") if isinstance(row, dict) else None
        if not isinstance(stem, str) or not stem or stem in rows:
            raise PackageError("report rows must have unique valid stems")
        row["_report_row_sha256"] = hashlib.sha256((line + "\n").encode("utf-8")).hexdigest()
        rows[stem] = row
    return rows


def _reconcile(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    artifacts = _source_artifacts(root)
    selected = _selected_stems(_json(artifacts["selected_manifest"]))
    strict = _json(artifacts["strict_manifest"])
    accepted_rows = strict.get("ok")
    rejected = strict.get("rejected")
    if not isinstance(accepted_rows, list) or not isinstance(rejected, dict):
        raise PackageError("strict manifest accepted/rejected sections malformed")
    accepted = [r.get("stem") if isinstance(r, dict) else None for r in accepted_rows]
    if any(not isinstance(s, str) for s in accepted) or len(accepted) != len(set(accepted)):
        raise PackageError("strict accepted rows malformed")
    rejected_stems = list(rejected)
    if len(rejected_stems) != len(set(rejected_stems)):
        raise PackageError("strict rejected keys duplicated")
    report = _report_rows(artifacts["report"])
    filtered = artifacts["report"].parent.parent / "filtered"
    filtered_stems = {p.stem for p in filtered.glob("*.TextGrid") if p.is_file() and not p.is_symlink()}
    report_filtered = {s for s, row in report.items() if str(row.get("status", "")).startswith("filtered")}
    strict_only = {s for s in rejected_stems if not str(report.get(s, {}).get("status", "")).startswith("filtered")}
    if len(strict_only) != 1 or any(str(report[s].get("status", "")).startswith("filtered") for s in strict_only):
        raise PackageError("strict-only rejected reconciliation requires one report-ok stem")
    frozen = report_filtered | strict_only
    if set(rejected_stems) != frozen or filtered_stems != frozen:
        raise PackageError("strict/report/filtered stem reconciliation mismatch")
    if len(selected) != 1000 or len(accepted) != 794 or len(frozen) != 206 or set(accepted) & frozen or set(accepted) | frozen != selected:
        raise PackageError("exact 794/206/1000 partition failed")
    return artifacts, {"selected": selected, "accepted": accepted, "frozen": sorted(frozen)}, report, strict


def prepare_package(source_root: Path, package_dir: Path) -> Path:
    source = _source_root(source_root)
    package = _fresh_package(package_dir)
    artifacts, partition, report, strict = _reconcile(source)
    package.mkdir(parents=True)
    frozen = partition["frozen"]
    accepted = sorted(partition["accepted"])
    frozen_manifest = {"schema": FROZEN_SCHEMA, "count": 206, "stems": frozen,
                      "source": "strict_rejected_plus_strict_only_report_ok"}
    strict_output_dir = str(Path(str(strict.get("output_dir", artifacts["strict_manifest"].parent))).resolve())
    accepted_manifest = {"schema": ACCEPTED_SCHEMA, "count": 794, "stems": accepted,
                        "ok": [{"stem": stem} for stem in accepted],
                        "output_dir": strict_output_dir, "source": "strict_ok_manifest"}
    declared_path, actual_path = artifacts["pipeline_receipt"], artifacts["report"]
    declared_hash, actual_hash = sha256(declared_path), sha256(actual_path)
    if declared_hash == actual_hash:
        raise PackageError("inner evidence hashes unexpectedly equal")
    rel_hashes = {_safe_rel(v, source): sha256(v) for v in artifacts.values()}
    mismatch = {"declared_sha256": declared_hash, "actual_sha256": actual_hash}
    plan = {"schema": "filtered-recovery-import-v1", "stems": frozen, "source": 206,
            "eligible": 206, "exclusions": 0,
            "declared_vs_actual_inner_receipt": mismatch,
            "inner_receipt_paths": {"declared": _safe_rel(declared_path, source),
                                    "actual": _safe_rel(actual_path, source)},
            "parent_artifact_sha256": rel_hashes}
    ledger_rows = []
    for stem in frozen:
        row = report.get(stem)
        report_reasons = row.get("filter_reasons", row.get("reasons", [])) if isinstance(row, dict) else []
        if not isinstance(report_reasons, list):
            report_reasons = [str(report_reasons)]
        strict_reasons = strict.get("rejected", {}).get(stem, [])
        if not isinstance(strict_reasons, list):
            strict_reasons = [str(strict_reasons)]
        reasons = list(dict.fromkeys([str(reason) for reason in report_reasons + strict_reasons]))
        ledger_rows.append({"stem": stem, "status": "strict_rejected", "disposition": "blocked_valid_rejection",
                            "report_reasons": report_reasons, "strict_reasons": strict_reasons,
                            "reasons": reasons, "report_row_sha256": row["_report_row_sha256"]})
    evidence = {"schema": "filtered-recovery-evidence-v1", "source_root": str(source),
                "source_artifacts": {k: {"path": _safe_rel(v, source), "sha256": sha256(v)} for k, v in artifacts.items()},
                "frozen_stems_sha256": hashlib.sha256(json.dumps(frozen, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "parent_artifact_sha256": rel_hashes,
                "declared_vs_actual_inner_receipt": mismatch,
                "partition": {"selected": 1000, "accepted": 794, "frozen": 206},
                "strict_manifest_sha256": sha256(artifacts["strict_manifest"]),
                "report_sha256": sha256(artifacts["report"]),
                "code_hashes": {name: sha256(Path(__file__).resolve().parent.parent / name)
                                for name in ("scripts/run_pipeline.py", "scripts/postprocess_textgrids.py",
                                             "scripts/audit_strict_ok.py", "scripts/filtered_recovery_package.py")}}
    _write_once(package / "frozen_filtered.json", frozen_manifest)
    _write_once(package / "accepted_manifest.json", accepted_manifest)
    _write_once(package / "filtered_recovery_manifest.json", plan)
    _write_once(package / "evidence_receipt.json", evidence)
    ledger = package / "filtered_reason_ledger.jsonl"
    ledger.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ledger_rows), encoding="utf-8")
    ledger.chmod(0o444)
    receipt = {"schema": SCHEMA, "source_root": str(source), "files": {}, "partition": {"selected": 1000, "accepted": 794, "frozen": 206}}
    for path in sorted(package.iterdir()):
        receipt["files"][path.name] = sha256(path)
    _write_once(package / "package_receipt.json", receipt)
    return package


def preflight_package(package_dir: Path) -> dict:
    package = Path(os.path.abspath(package_dir))
    if not package.is_dir() or package.is_symlink():
        raise PackageError("package directory missing or symlink")
    receipt_path = package / "package_receipt.json"
    receipt = _json(receipt_path)
    if receipt.get("schema") != SCHEMA:
        raise PackageError("package receipt schema mismatch")
    if set(receipt.get("files", {})) != PACKAGE_FILES or {p.name for p in package.iterdir()} != PACKAGE_FILES | {"package_receipt.json"}:
        raise PackageError("package entries are not the exact expected set")
    for name, expected in receipt.get("files", {}).items():
        path = package / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise PackageError(f"package file drift: {name}")
    source = _source_root(Path(receipt["source_root"]))
    artifacts, partition, report, strict = _reconcile(source)
    evidence = _json(package / "evidence_receipt.json")
    expected_labels = set(artifacts)
    if set(evidence.get("source_artifacts", {})) != expected_labels:
        raise PackageError("evidence artifact labels mismatch")
    expected_hashes = {_safe_rel(path, source): sha256(path) for path in artifacts.values()}
    if evidence.get("parent_artifact_sha256") != expected_hashes:
        raise PackageError("evidence parent artifact map mismatch")
    for label, row in evidence.get("source_artifacts", {}).items():
        rel = Path(str(row.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts or str(rel) not in expected_hashes:
            raise PackageError(f"evidence artifact path unsafe: {label}")
        if sha256(source / rel) != row["sha256"]:
            raise PackageError(f"source drift: {label}")
    frozen = _json(package / "frozen_filtered.json"); accepted = _json(package / "accepted_manifest.json")
    frozen_stems = frozen.get("stems", []); accepted_stems = accepted.get("stems", [])
    if (frozen.get("count") != 206 or accepted.get("count") != 794
            or not isinstance(frozen_stems, list) or not isinstance(accepted_stems, list)
            or any(not isinstance(s, str) for s in frozen_stems + accepted_stems)
            or len(frozen_stems) != len(set(frozen_stems)) or len(accepted_stems) != len(set(accepted_stems))
            or set(frozen_stems) != set(partition["frozen"])
            or set(accepted_stems) != set(partition["accepted"])):
        raise PackageError("prepared partition mismatch")
    for name, expected in evidence.get("code_hashes", {}).items():
        path = Path(__file__).resolve().parent.parent / name
        if not path.is_file() or sha256(path) != expected:
            raise PackageError(f"relevant code drift: {name}")
    return {"schema": "filtered-recovery-package-preflight-v1", "ok": True,
            "source_root": str(source), "selected": 1000, "accepted": 794, "frozen": 206,
            "source_artifacts": {k: str(v) for k, v in artifacts.items()}}


def preflight_source(source_root: Path) -> dict:
    """Read-only reconciliation of an existing source run (no package writes)."""
    source = _source_root(source_root)
    artifacts, partition, report, strict = _reconcile(source)
    ledger = []
    for stem in partition["frozen"]:
        row = report[stem]
        report_reasons = row.get("filter_reasons", row.get("reasons", []))
        if not isinstance(report_reasons, list): report_reasons = [str(report_reasons)]
        strict_reasons = strict.get("rejected", {}).get(stem, [])
        if not isinstance(strict_reasons, list): strict_reasons = [str(strict_reasons)]
        reasons = list(dict.fromkeys([str(reason) for reason in report_reasons + strict_reasons]))
        ledger.append({"stem": stem, "report_reasons": report_reasons,
                       "strict_reasons": strict_reasons, "reasons": reasons,
                       "disposition": "blocked_valid_rejection"})
    return {"schema": "filtered-recovery-source-preflight-v1", "ok": True,
            "source_root": str(source), "selected": 1000, "accepted": 794,
            "frozen": 206, "reason_ledger": ledger,
            "source_artifacts": {k: {"path": str(v), "sha256": sha256(v)} for k, v in artifacts.items()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--source-root", type=Path, required=True); p.add_argument("--package-dir", type=Path, required=True)
    p = sub.add_parser("preflight"); p.add_argument("--package-dir", type=Path); p.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.package_dir is None: raise PackageError("prepare requires --package-dir")
            result = prepare_package(args.source_root, args.package_dir)
        elif args.package_dir is not None:
            result = preflight_package(args.package_dir)
        elif args.source_root is not None:
            result = preflight_source(args.source_root)
        else:
            raise PackageError("preflight requires --package-dir or --source-root")
        print(json.dumps(result if isinstance(result, dict) else {"package": str(result)}, ensure_ascii=False, sort_keys=True))
        return 0
    except PackageError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
