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


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    candidates = sorted(workspace.glob("strict_ok_runs/*/output/.pipeline_run_receipt_v2.json"))
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
    if not isinstance(run, dict) or run.get("returncode") != 0: errors.append("pipeline_returncode_nonzero_or_missing")
    workspace = root / "workspace"
    shards = _ctc_shards(root, workspace, universe, count if isinstance(count, int) else 0, errors)
    output, filtered, reasons = _strict_output(workspace, universe, count if isinstance(count, int) else 0, errors)
    activity, activity_source = _telemetry(root, workspace, _plan_rows(root, universe, count if isinstance(count, int) else 0, errors), count if isinstance(count, int) else 0, errors)
    return {"schema": "gpu1000-analysis-v2", "root": str(root), "ok": not errors,
            "errors": sorted(set(errors)), "selected_count": len(universe), "run_label": manifest.get("run_label"),
            "shard_counts": {str(row["gpu"]): len(row["stems"]) for row in shards},
            "output_count": len(output), "filtered_count": len(filtered), "global_reasons": reasons,
            "gpu_activity_samples": activity, "gpu_activity_evidence": activity_source, "publication": "forbidden"}


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
