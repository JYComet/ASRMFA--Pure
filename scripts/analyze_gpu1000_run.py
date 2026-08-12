#!/usr/bin/env python3
"""Fail-closed, non-publishing analysis of a GPU-1000 pipeline run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "pipeline-run-receipt-v2"


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL report, rejecting malformed rows instead of guessing."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}")
        rows.append(value)
    return rows


def build_filtered_root_cause_ledger(report_path: Path, output_path: Path | None = None,
                                     *, expected_filtered: int | None = None) -> dict[str, Any]:
    """Build a deterministic, non-exclusive ledger for filtered report rows.

    One stem may contribute several instances (the sum is therefore greater
    than the stem count).  Every instance carries the canonical report-row
    hash, subtype, examples and a conservative disposition.  No output is
    written by this function; callers may serialize the returned object.
    """
    rows = _jsonl_rows(report_path)
    filtered = [row for row in rows if str(row.get("status", "")).startswith("filtered")]
    if expected_filtered is not None and len(filtered) != expected_filtered:
        raise ValueError(f"filtered row count {len(filtered)} != {expected_filtered}")
    seen: set[str] = set(); instances: list[dict[str, Any]] = []
    taxonomy: dict[str, int] = {}
    trace_counts = {"semantic": 0, "mid_sp": 0, "displacement": 0, "warnings": 0}
    for row in filtered:
        stem = row.get("stem")
        if not isinstance(stem, str) or not stem or stem in seen:
            raise ValueError(f"filtered report stem missing or duplicated: {stem!r}")
        seen.add(stem)
        evidence_hash = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        reasons = row.get("filter_reasons", [])
        if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError(f"invalid filter_reasons for {stem}")
        # The report's reason list is the authoritative non-exclusive count.
        for reason in reasons:
            examples: list[str] = []
            detail = row.get(reason)
            if isinstance(detail, dict) and isinstance(detail.get("examples"), list):
                examples = [str(value) for value in detail["examples"]]
            elif reason == "warnings" and isinstance(row.get("warnings"), list):
                examples = [str(value) for value in row["warnings"]]
            disposition = "valid_rejection"
            if reason in {"reference_semantic_sequence_mismatch", "cjk_mismatch", "cjk_token_count_mismatch"}:
                disposition = "blocked"
            instance = {"stem": stem, "subtype": reason, "examples": examples,
                        "evidence_sha256": evidence_hash, "disposition": disposition}
            instances.append(instance); taxonomy[reason] = taxonomy.get(reason, 0) + 1
        # Trace semantic/mid-sp/displacement/warnings even where a future
        # report forgot to include a corresponding filter reason.
        coverage = row.get("reference_coverage", {})
        displacement = row.get("pinyin_displacement", {})
        traces = {
            "semantic": coverage.get("exact_semantic_sequence") is False,
            "mid_sp": "mid_sp" in row or "mid_sp" in reasons,
            "displacement": ("pinyin_displacement" in reasons or
                              (isinstance(displacement, dict) and
                               (displacement.get("displacement_runs", 0) > 0 or
                                displacement.get("mismatch_rate", 0) > 0))),
            "warnings": bool(row.get("warnings")) or "warnings" in reasons,
        }
        for subtype, present in traces.items():
            if present:
                trace_counts[subtype] += 1
    if output_path is not None:
        output_stems = {p.stem for p in output_path.glob("*.TextGrid") if not p.is_symlink()}
        if output_stems & seen:
            raise ValueError("filtered/output stem overlap")
    return {"schema": "filtered-root-cause-ledger-v1", "report": str(report_path.resolve()),
            "filtered_stems": sorted(seen), "filtered_count": len(seen),
            "instance_count": len(instances), "instances": instances,
            "taxonomy_counts": dict(sorted(taxonomy.items())), "trace_counts": trace_counts}


def _partition_sets(root: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    """Discover a run's output/filtered sets and selected manifest."""
    root = root.resolve()
    try: manifest = _read(root / "selected_manifest.json")
    except (OSError, json.JSONDecodeError): manifest = {}
    output = {p.stem for p in (root / "output").glob("*.TextGrid") if not p.is_symlink()}
    filtered = {p.stem for p in (root / "filtered").glob("*.TextGrid") if not p.is_symlink()}
    designated = None
    try:
        result = _read(root / "continuation_result.json")
        designated = Path(str(result.get("strict_receipt_path", ""))) if result.get("status") == "PASS_WITH_CONTINUATION" else None
    except (OSError, json.JSONDecodeError):
        pass
    continuation_present = designated is not None
    if designated and designated.is_file():
        resolved = designated.resolve(); strict_root = root / "workspace" / "strict_ok_runs"
        if (not resolved.is_relative_to(strict_root) or resolved.parent.parent.name.startswith("continuation_") is False
                or hashlib.sha256(resolved.read_bytes()).hexdigest() != result.get("strict_receipt_sha256")):
            designated = None
    if continuation_present and designated is None:
        return set(), set(), manifest if isinstance(manifest, dict) else {}
    if designated is not None:
        output = {p.stem for p in designated.parent.glob("*.TextGrid") if not p.is_symlink()}
        filtered = {p.stem for p in designated.parent.parent.joinpath("filtered").glob("*.TextGrid") if not p.is_symlink()}
    if not output and not filtered:
        receipts = sorted(root.glob("workspace/strict_ok_runs/*/output/.pipeline_run_receipt_v2.json"))
        if len(receipts) == 1:
            output = {p.stem for p in receipts[0].parent.glob("*.TextGrid") if not p.is_symlink()}
            filtered = {p.stem for p in receipts[0].parent.parent.joinpath("filtered").glob("*.TextGrid") if not p.is_symlink()}
    return output, filtered, manifest if isinstance(manifest, dict) else {}


def _continuation_lineage(root: Path, errors: list[str]) -> dict[str, Any]:
    """Validate a continuation receipt without treating it as a new upstream run."""
    path = root / "continuation_result.json"
    if not path.is_file():
        return {"present": False, "status": "NONE"}
    try:
        receipt = _read(path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"continuation_receipt_unreadable:{exc}"); return {"present": True, "status": "BLOCKED"}
    if not isinstance(receipt, dict) or receipt.get("schema") != "gpu1000-continuation-v1":
        errors.append("continuation_receipt_schema_invalid"); return {"present": True, "status": "BLOCKED"}
    if receipt.get("status") != "PASS_WITH_CONTINUATION": errors.append("continuation_status_not_pass")
    original = Path(str(receipt.get("original_root", ""))).resolve()
    original_receipt = original / "run_receipt.json"
    original_hash = hashlib.sha256(original_receipt.read_bytes()).hexdigest() if original_receipt.is_file() else ""
    if not original_receipt.is_file() or original_hash != receipt.get("original_run_receipt_sha256"):
        errors.append("continuation_original_receipt_binding_invalid")
    else:
        try:
            if _read(original_receipt).get("returncode") != 1:
                errors.append("continuation_original_rc_not1")
        except (OSError, json.JSONDecodeError):
            errors.append("continuation_original_receipt_unreadable")
    if receipt.get("scope_count") != 1 or not isinstance(receipt.get("scope"), dict): errors.append("continuation_scope_not_exact1")
    if receipt.get("merged_count") != 1000: errors.append("continuation_merged_count_invalid")
    axis = root / "workspace" / ".mfa_alignment_axis_receipt.json"
    if not axis.is_file() or hashlib.sha256(axis.read_bytes()).hexdigest() != receipt.get("axis_receipt_sha256"):
        errors.append("continuation_axis_binding_invalid")
    source_receipt = root / "workspace" / ".continuation_source_receipt.json"
    source_hash = hashlib.sha256(source_receipt.read_bytes()).hexdigest() if source_receipt.is_file() else ""
    if not source_receipt.is_file() or source_hash != receipt.get("source_receipt_sha256"):
        errors.append("continuation_source_binding_invalid")
    designated = receipt.get("strict_receipt_path")
    strict = [Path(designated)] if isinstance(designated, str) and Path(designated).is_file() else []
    if not strict:
        strict = sorted(root.glob("workspace/strict_ok_runs/*/output/.pipeline_run_receipt_v2.json"))
    if receipt.get("status") == "PASS_WITH_CONTINUATION":
        if len(strict) != 1 or hashlib.sha256(strict[0].read_bytes()).hexdigest() != receipt.get("strict_receipt_sha256"):
            errors.append("continuation_strict_binding_invalid")
    return {"present": True, "status": "PASS_WITH_CONTINUATION" if not any(e.startswith("continuation_") for e in errors) else "BLOCKED",
            "receipt": receipt}


def compare_acceptance_runs(parent_root: Path, future_root: Path) -> dict[str, Any]:
    """Fail-closed comparison of accepted-776 baseline against a future run."""
    errors: list[str] = []
    old_output, old_filtered, old_manifest = _partition_sets(parent_root)
    new_output, new_filtered, new_manifest = _partition_sets(future_root)
    if len(old_output) != 776:
        errors.append(f"parent_accepted_count:{len(old_output)}")
    if old_output & old_filtered or new_output & new_filtered:
        errors.append("partition_overlap")
    missing = old_output - new_output
    if missing:
        errors.append(f"old_accepted_missing:{len(missing)}")
    newly_filtered = new_filtered - old_filtered
    if newly_filtered:
        errors.append(f"new_filtered:{len(newly_filtered)}")
    old_samples = old_manifest.get("samples", [])
    new_samples = new_manifest.get("samples", [])
    old_identity = [{key: row.get(key) for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                    for row in old_samples if isinstance(row, dict)]
    new_identity = [{key: row.get(key) for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                    for row in new_samples if isinstance(row, dict)]
    if not old_identity or not new_identity or _digest(old_identity) != _digest(new_identity):
        errors.append("source_identity_drift")
    if new_manifest.get("count") != 1000 or new_manifest.get("run_label") != "full1000":
        errors.append("future_not_exact1000")
    selected = {row.get("stem") for row in new_samples if isinstance(row, dict) and isinstance(row.get("stem"), str)}
    plan = _plan_rows(future_root.resolve(), selected, 1000, errors)
    if len(plan) != 8 or any(len(row.get("stems", [])) != 125 for row in plan):
        errors.append("future_shards_not_exact_125")
    # Semantic integrity is hard: a future report with a hard semantic flag
    # cannot be accepted even if the set partition happens to conserve.
    for report_path in sorted(future_root.resolve().rglob("*.jsonl")):
        try: rows = _jsonl_rows(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for row in rows:
            coverage = row.get("reference_coverage", {})
            if row.get("hard_integrity_reasons") or (isinstance(coverage, dict) and coverage.get("exact_semantic_sequence") is False):
                errors.append(f"hard_semantic_drift:{row.get('stem', '?')}")
    return {"schema": "gpu1000-acceptance-comparison-v1", "ok": not errors,
            "errors": sorted(set(errors)), "parent_accepted": len(old_output),
            "future_output": len(new_output), "future_filtered": len(new_filtered),
            "old_accepted_missing": sorted(missing), "new_filtered": sorted(newly_filtered),
            "future_selected": len(selected), "future_shard_counts": [len(row.get("stems", [])) for row in plan]}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_filtered_recovery_logic(
    output_dir: Path, filtered_dir: Path, report_path: Path,
    frozen_stems: list[str], accepted_stems: list[str],
    *, expected_count: int | None = None,
) -> dict[str, Any]:
    """Independent audit for a quarantined all-filtered recovery replay."""
    errors: list[str] = []
    frozen, accepted = set(frozen_stems), set(accepted_stems)
    if expected_count is None:
        expected_count = len(frozen)
    if len(frozen) != expected_count or len(frozen_stems) != len(frozen):
        errors.append("frozen_set_count_or_duplicate")
    if frozen & accepted:
        errors.append("parent_accepted_intersection")
    try:
        rows = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        rows, errors = [], [f"report_unreadable:{exc}"]
    by_stem: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
            errors.append("report_row_malformed"); continue
        stem = row["stem"]
        if stem in by_stem: errors.append(f"report_duplicate:{stem}")
        by_stem[stem] = row
    if len(rows) != expected_count: errors.append(f"report_count:{len(rows)}")
    if set(by_stem) != frozen: errors.append("report_frozen_set_mismatch")
    output = {p.stem for p in output_dir.glob("*.TextGrid") if not p.is_symlink()}
    filtered = {p.stem for p in filtered_dir.glob("*.TextGrid") if not p.is_symlink()}
    if output & filtered: errors.append("output_filtered_overlap")
    if output | filtered != frozen: errors.append("output_filtered_frozen_union_mismatch")
    if output & accepted or filtered & accepted: errors.append("final_parent_accepted_intersection")
    taxonomy = {"ok": [], "filtered": []}
    for stem, row in by_stem.items():
        status = row.get("status")
        if status == "ok": taxonomy["ok"].append(stem)
        elif isinstance(status, str) and status.startswith("filtered"): taxonomy["filtered"].append(stem)
        else: errors.append(f"taxonomy_status_invalid:{stem}")
    if set(taxonomy["ok"]) != output: errors.append("report_output_taxonomy_mismatch")
    if set(taxonomy["filtered"]) != filtered: errors.append("report_filtered_taxonomy_mismatch")
    return {"ok": not errors, "errors": sorted(set(errors)),
            "source": expected_count, "eligible": expected_count, "exclusions": 0,
            "output": len(output), "filtered": len(filtered),
            "taxonomy": {key: len(value) for key, value in taxonomy.items()}}


def _write_new(path: Path, content: str) -> None:
    if path.exists(): raise RuntimeError(f"refusing to overwrite analysis output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content, encoding="utf-8")


def _shard_sizes(count: int) -> list[int]:
    base, remainder = divmod(count, 8)
    return [base + (1 if gpu < remainder else 0) for gpu in range(8)]


def _bucket(receipt: dict[str, Any], name: str, errors: list[str]) -> set[str]:
    raw = receipt.get(name)
    if not isinstance(raw, dict) or not isinstance(raw.get("stems"), list):
        errors.append(f"receipt_{name}_bucket_missing"); return set()
    stems = raw["stems"]
    if any(not isinstance(stem, str) or not stem or Path(stem).name != stem for stem in stems):
        errors.append(f"receipt_{name}_stems_invalid")
    if len(stems) != len(set(stems)): errors.append(f"receipt_{name}_stems_duplicate")
    ordered = sorted(stems)
    if raw.get("count") != len(stems): errors.append(f"receipt_{name}_count_mismatch")
    if raw.get("stems_digest") != _digest(ordered): errors.append(f"receipt_{name}_digest_mismatch")
    return set(stems)


def _validated_receipt(path: Path, errors: list[str], label: str) -> tuple[dict[str, Any], dict[str, set[str]]]:
    try: receipt = _read(path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}_receipt_unreadable:{exc}"); return {}, {}
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        errors.append(f"{label}_receipt_schema_invalid"); return receipt if isinstance(receipt, dict) else {}, {}
    buckets = {name: _bucket(receipt, name, errors) for name in ("source", "eligible", "output", "filtered")}
    exclusions = receipt.get("exclusions")
    if not isinstance(exclusions, list) or any(not isinstance(row, dict) or not isinstance(row.get("stem"), str)
                                               or not isinstance(row.get("reason"), str) or not row["reason"]
                                               for row in exclusions):
        errors.append(f"{label}_receipt_exclusions_invalid"); excluded: set[str] = set()
    else:
        excluded = {row["stem"] for row in exclusions}
        if len(excluded) != len(exclusions): errors.append(f"{label}_receipt_exclusions_duplicate")
    if buckets and buckets["source"] != buckets["eligible"] | excluded:
        errors.append(f"{label}_receipt_source_eligible_conservation_invalid")
    if buckets and (buckets["output"] & buckets["filtered"] or buckets["output"] | buckets["filtered"] != buckets["eligible"]):
        errors.append(f"{label}_receipt_output_filtered_conservation_invalid")
    return receipt, buckets


def _plan_rows(root: Path, universe: set[str], count: int, errors: list[str]) -> list[dict[str, Any]]:
    try: rows = _read(root / "shard_plan.json").get("shards")
    except (OSError, json.JSONDecodeError): rows = None
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("root_shard_plan_invalid"); return []
    wanted = _shard_sizes(count); seen: set[str] = set()
    for gpu, row in enumerate(rows):
        stems = row.get("stems") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("gpu") != gpu or not isinstance(stems, list) or len(stems) != wanted[gpu]:
            errors.append("root_shard_plan_invalid"); break
        if len(stems) != len(set(stems)): errors.append("root_shard_plan_overlap")
        seen.update(stems)
    if seen != universe: errors.append("root_shard_plan_universe_mismatch")
    return rows


def _quarantined_shards(workspace: Path, errors: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(workspace.glob("ctc_pretg.partial-*/_shard_gpu*/selected_stems.txt")):
        match = re.search(r"_shard_gpu(\d+)$", path.parent.name)
        if not match: continue
        stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.append({"shard_id": f"gpu{match.group(1)}", "stems": stems})
    if len(rows) != 8: errors.append("quarantined_shard_manifests_not_exactly_eight")
    return rows


def _ctc_shards(root: Path, workspace: Path, universe: set[str], count: int, errors: list[str]) -> list[dict[str, Any]]:
    receipt_path = workspace / "ctc_pretg" / ".pipeline_run_receipt_v2.json"
    receipt, buckets = _validated_receipt(receipt_path, errors, "ctc")
    if buckets.get("source") != universe or buckets.get("eligible") != universe:
        errors.append("ctc_receipt_selected_universe_mismatch")
    rows = receipt.get("shards") if isinstance(receipt, dict) else None
    if not isinstance(rows, list): rows = _quarantined_shards(workspace, errors)
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("ctc_shards_missing_or_ambiguous"); return []
    plan = _plan_rows(root, universe, count, errors); expected_sizes = _shard_sizes(count)
    seen: set[str] = set()
    canonical: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("stems"), list):
            errors.append("ctc_shard_row_invalid"); continue
        shard_id = row.get("shard_id", "")
        match = re.fullmatch(r"gpu(\d+)", str(shard_id))
        if not match or int(match.group(1)) != position:
            errors.append("ctc_shard_identifier_invalid"); continue
        stems = row["stems"]
        if row.get("count") != len(stems) or row.get("stems_digest") != _digest(sorted(stems)):
            errors.append(f"ctc_gpu{position}_shard_receipt_digest_invalid")
        if len(stems) != len(set(stems)) or len(stems) != expected_sizes[position]: errors.append(f"ctc_gpu{position}_shard_size_invalid")
        if seen & set(stems): errors.append("ctc_shard_overlap")
        seen.update(stems); canonical.append({"gpu": position, "stems": stems})
        if plan and set(stems) != set(plan[position]["stems"]): errors.append(f"ctc_gpu{position}_root_plan_mismatch")
    if seen != universe: errors.append("ctc_shard_union_selected_universe_mismatch")
    return canonical


def _strict_output(workspace: Path, universe: set[str], count: int, errors: list[str]) -> tuple[set[str], set[str], list[Any]]:
    designated = None
    try:
        result = _read(workspace.parent / "continuation_result.json")
        if result.get("status") == "PASS_WITH_CONTINUATION": designated = Path(str(result.get("strict_receipt_path", "")))
    except (OSError, json.JSONDecodeError): pass
    continuation_present = designated is not None
    if designated and designated.is_file():
        resolved = designated.resolve(); strict_root = workspace / "strict_ok_runs"
        if (not resolved.is_relative_to(strict_root) or not resolved.parent.parent.name.startswith("continuation_")
                or hashlib.sha256(resolved.read_bytes()).hexdigest() != result.get("strict_receipt_sha256")):
            errors.append("continuation_designated_strict_binding_invalid"); designated = None
    if continuation_present and designated is None:
        errors.append("continuation_designated_strict_binding_invalid")
        return set(), set(), []
    candidates = [designated] if designated and designated.is_file() else sorted(workspace.glob("strict_ok_runs/*/output/.pipeline_run_receipt_v2.json"))
    if len(candidates) != 1:
        errors.append("strict_output_receipt_not_exactly_one"); return set(), set(), []
    receipt_path = candidates[0]; output_dir = receipt_path.parent; filtered_dir = output_dir.parent / "filtered"
    manifest_path = output_dir / "strict_ok_manifest.json"
    receipt, buckets = _validated_receipt(receipt_path, errors, "strict")
    paths = receipt.get("paths", {}) if isinstance(receipt, dict) else {}
    if not isinstance(paths, dict) or paths.get("output") != str(output_dir.resolve()) or paths.get("filtered") != str(filtered_dir.resolve()):
        errors.append("strict_receipt_paths_not_bound_to_active_run")
    if buckets.get("source") != universe or buckets.get("eligible") != universe:
        errors.append("strict_receipt_selected_universe_mismatch")
    try: manifest = _read(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"strict_manifest_unreadable:{exc}"); return set(), set(), []
    if not isinstance(manifest, dict) or manifest.get("global_reasons") != []:
        errors.append("strict_manifest_global_reasons_nonempty_or_invalid")
    ok = manifest.get("ok") if isinstance(manifest, dict) else None
    rejected = manifest.get("rejected") if isinstance(manifest, dict) else None
    if not isinstance(ok, list) or not isinstance(rejected, dict):
        errors.append("strict_manifest_buckets_invalid"); return set(), set(), []
    ok_stems = {row.get("stem") for row in ok if isinstance(row, dict) and isinstance(row.get("stem"), str)}
    if len(ok_stems) != len(ok) or any(not isinstance(reasons, list) for reasons in rejected.values()): errors.append("strict_manifest_buckets_invalid")
    rejected_stems = set(rejected)
    if ok_stems != buckets.get("output", set()) or rejected_stems != buckets.get("filtered", set()): errors.append("strict_manifest_receipt_bucket_mismatch")
    if ok_stems & rejected_stems or ok_stems | rejected_stems != universe: errors.append("strict_manifest_selected_conservation_invalid")
    return ok_stems, rejected_stems, manifest.get("global_reasons", []) if isinstance(manifest, dict) else []


def _child_shard_execution(workspace: Path, plan: list[dict[str, Any]]) -> dict[str, bool]:
    """Accept quarantined child evidence only when it proves every planned GPU.

    This preserves auditability for old 15-second telemetry that can miss a
    short CTC child, without treating a parent shard *plan* as execution.
    """
    proven = {str(i): False for i in range(8)}
    for child in sorted(workspace.glob("ctc_pretg.partial-*/_shard_gpu*")):
        match = re.fullmatch(r"_shard_gpu(\d+)", child.name)
        if not match or not child.is_dir(): continue
        gpu = int(match.group(1)); key = str(gpu)
        if gpu not in range(8) or not (child / "selected_stems.txt").is_file(): continue
        stems = [line.strip() for line in (child / "selected_stems.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = plan[gpu].get("stems", []) if len(plan) == 8 else []
        summary = child / "summary.txt"; receipt_path = child / ".ctc_run_receipt.json"
        receipt_ok = False
        try:
            receipt = _read(receipt_path)
            output_stems = receipt.get("output_stems") if isinstance(receipt, dict) else None
            input_stems = receipt.get("input_stems") if isinstance(receipt, dict) else None
            argv = receipt.get("argv") if isinstance(receipt, dict) else None
            selected_path = str((child / "selected_stems.txt").resolve())
            child_path = str(child.resolve())
            argv_pairs = list(zip(argv, argv[1:])) if isinstance(argv, list) else []
            receipt_ok = (
                isinstance(receipt, dict) and receipt.get("schema") == "ctc-run-receipt-v2"
                and isinstance(output_stems, list) and output_stems == sorted(expected)
                and len(output_stems) == len(set(output_stems))
                and receipt.get("output_stems_digest") == _digest(sorted(expected))
                and isinstance(input_stems, list) and input_stems == sorted(expected)
                and receipt.get("input_stems_digest") == _digest(sorted(expected))
                and all(isinstance(value, str) for value in argv)
                and ("--stems-file", selected_path) in argv_pairs
                and ("--output-dir", child_path) in argv_pairs
            )
        except (OSError, json.JSONDecodeError, TypeError):
            receipt_ok = False
        if (set(stems) == set(expected) and len(stems) == len(expected) and receipt_ok and summary.is_file()
                and re.search(r"^Files:\s+(\d+)\s+total,\s+\1\s+OK,\s+0\s+failed$", summary.read_text(encoding="utf-8"), re.MULTILINE)):
            proven[key] = True
    return proven


def _telemetry(root: Path, workspace: Path, plan: list[dict[str, Any]], count: int,
               errors: list[str]) -> tuple[dict[str, int], dict[str, str]]:
    path = root / "nvidia-smi.telemetry.jsonl"; activity = {str(i): 0 for i in range(8)}
    source = {str(i): "none" for i in range(8)}
    if not path.is_file(): errors.append("missing_telemetry"); return activity, source
    try: rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError: errors.append("telemetry_invalid_jsonl"); return activity, source
    if not rows or any(not isinstance(row, dict) or "at_utc" not in row for row in rows): errors.append("telemetry_missing_timestamp")
    free: dict[str, list[int]] = {str(i): [] for i in range(8)}
    used: dict[str, list[int]] = {str(i): [] for i in range(8)}
    for row in rows:
        for gpu in row.get("gpus", []) if isinstance(row, dict) else []:
            index = gpu.get("index") if isinstance(gpu, dict) else None
            key = str(index)
            if key not in activity: continue
            pids = gpu.get("compute_pids", [])
            if isinstance(pids, list) and pids:
                activity[key] += 1; source[key] = "compute_pid"
            utilization = gpu.get("utilization_gpu_pct")
            if isinstance(utilization, int) and utilization > 0:
                activity[key] += 1; source[key] = "utilization"
            value = gpu.get("memory_free_mib")
            if isinstance(value, int): free[key].append(value)
            value = gpu.get("memory_used_mib")
            if isinstance(value, int): used[key].append(value)
    for key, values in free.items():
        # Minor driver bookkeeping fluctuations (e.g. 3 MiB) do not prove
        # execution. A material used/free delta does, even without a PID poll.
        if activity[key] == 0 and ((values and max(values) - min(values) >= 8)
                                 or (used[key] and max(used[key]) - min(used[key]) >= 8)):
            activity[key] = 1; source[key] = "memory_delta"
    child_proven = _child_shard_execution(workspace, plan)
    minimum = 2 if count == 1000 else 1
    for key, samples in activity.items():
        if samples < minimum:
            if child_proven[key]: source[key] = "quarantined_child_receipt"; activity[key] = minimum
            else: errors.append(f"gpu{key}_telemetry_activity_below_{minimum}")
    return activity, source


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve(); errors: list[str] = []
    continuation = _continuation_lineage(root, errors)
    try: manifest = _read(root / "selected_manifest.json")
    except (OSError, json.JSONDecodeError) as exc: raise RuntimeError(f"missing selection manifest: {exc}")
    samples = manifest.get("samples", []); selected = [row.get("stem") for row in samples if isinstance(row, dict)]
    count = manifest.get("count"); universe = set(selected)
    if not isinstance(count, int) or count != len(samples) or count != len(selected) or len(universe) != count:
        errors.append("selected_universe_invalid")
    if count != 1000 and not (isinstance(count, int) and 8 <= count <= 64): errors.append("selected_count_invalid")
    if count == 1000 and manifest.get("run_label") != "full1000": errors.append("full1000_run_label_invalid")
    if count != 1000 and manifest.get("run_label") != "confirmation_nonpublish": errors.append("confirmation_run_label_invalid")
    identity = [{key: row.get(key) for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                for row in samples if isinstance(row, dict)]
    if _digest(identity) != manifest.get("selection_digest"): errors.append("selection_digest_mismatch")
    run = _read(root / "run_receipt.json") if (root / "run_receipt.json").is_file() else {}
    if (not isinstance(run, dict) or run.get("returncode") != 0) and continuation.get("status") != "PASS_WITH_CONTINUATION":
        errors.append("pipeline_returncode_nonzero_or_missing")
    workspace = root / "workspace"
    shards = _ctc_shards(root, workspace, universe, count if isinstance(count, int) else 0, errors)
    output, filtered, reasons = _strict_output(workspace, universe, count if isinstance(count, int) else 0, errors)
    activity, activity_source = _telemetry(root, workspace, _plan_rows(root, universe, count if isinstance(count, int) else 0, errors), count if isinstance(count, int) else 0, errors)
    return {"schema": "gpu1000-analysis-v2", "root": str(root), "ok": not errors,
            "errors": sorted(set(errors)), "selected_count": len(universe), "run_label": manifest.get("run_label"),
            "shard_counts": {str(row["gpu"]): len(row["stems"]) for row in shards},
            "output_count": len(output), "filtered_count": len(filtered), "global_reasons": reasons,
            "gpu_activity_samples": activity, "gpu_activity_evidence": activity_source, "publication": "forbidden",
            "continuation": continuation}


def analyze_command(args: argparse.Namespace) -> int:
    try:
        report = analyze(args.root); json_out = args.json_out or args.root / "analysis.json"; md_out = args.markdown_out or args.root / "analysis.md"
        _write_new(json_out, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _write_new(md_out, "# GPU-1000 run analysis\n\n" + f"Status: {'PASS' if report['ok'] else 'FAIL'}\n\n" +
                   f"Selected: {report['selected_count']}; output: {report['output_count']}; filtered: {report['filtered_count']}\n\nErrors:\n" +
                   "\n".join(f"- {error}" for error in report["errors"]) + "\n")
        print(json.dumps(report, ensure_ascii=False)); return 0 if report["ok"] else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr); return 2


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--json-out", type=Path); parser.add_argument("--markdown-out", type=Path)
    return analyze_command(parser.parse_args())


if __name__ == "__main__": raise SystemExit(main())
