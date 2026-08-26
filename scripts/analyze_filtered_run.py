#!/usr/bin/env python3
"""Build a reproducible taxonomy for one explicit strict filtered run.

The candidate axis is deliberately limited to ``<strict-run>/filtered``.  All
other evidence is anchored to the explicitly supplied workspace and strict
run; no other run is searched or consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STRICT_RUN = Path(
    "/mnt/nvme3/mfa_workspace_ria_xinzeng_20260813_hyphenfix/"
    "strict_ok_runs/20260813T144340Z_1364496"
)
DEFAULT_WORKSPACE = Path("/mnt/nvme3/mfa_workspace_ria_xinzeng_20260813_hyphenfix")
DEFAULT_OUTPUT_DIR = Path("/tmp/terra_ria262_w1")

SCHEMA = "ria-filtered-taxonomy-v1"
REPORT_NAME = "output/postprocess_report.jsonl"
RECEIPT_NAME = "output/.pipeline_run_receipt_v2.json"

EXPECTED_STEM_COUNT = 262
EXPECTED_SOURCE_COUNT = 2000
EXPECTED_OUTPUT_COUNT = 1738
EXPECTED_STATUS_COUNTS = {
    "filtered_short_word": 60,
    "filtered_mfa_unknown_source": 162,
    "filtered_mid_sp": 7,
    "filtered_words_tier_gaps": 18,
    "filtered_missing_mfa_alignment": 3,
    "filtered_short_word_mfa_unknown_source": 3,
    "filtered_overlapping_words": 2,
    "filtered_overlapping_words_words_tier_gaps": 1,
    "filtered_overlapping_words_short_word_pp_tier_gaps_words_tier_gaps": 1,
    "filtered_short_word_words_tier_gaps_mfa_unknown_source": 1,
    "filtered_words_tier_gaps_tier_discontinuity": 1,
    "filtered_short_word_words_tier_gaps": 1,
    "filtered_mid_sp_mfa_unknown_source": 1,
    "filtered_overlapping_words_words_tier_gaps_mfa_unknown_source": 1,
}
EXPECTED_REASON_OCCURRENCES = {
    "mfa_unknown_source": 168,
    "short_word": 66,
    "words_tier_gaps": 24,
    "mid_sp": 8,
    "overlapping_words": 5,
    "pp_tier_gaps": 1,
    "tier_discontinuity": 1,
    "missing_mfa_alignment": 3,
}
BASE_STATUS_COUNTS = {
    "filtered_short_word": 60,
    "filtered_mfa_unknown_source": 162,
    "filtered_mid_sp": 7,
    "filtered_words_tier_gaps": 18,
    "filtered_missing_mfa_alignment": 3,
}
EXPECTED_PRIMARY_PARTITIONS = {
    "unknown_gate": 168,
    "qc_audio": 91,
    "missing_mfa_axis": 3,
}
EXPECTED_MISSING_IDS = ["000009", "000960", "001844"]

CTC_FILES = {
    "textgrid": ".TextGrid",
    "lab": ".lab",
    "punct": "_punct.json",
    "reference": "_ref.txt",
    "text_cn": "_text_cn.txt",
    "text_raw": "_text_raw.txt",
    "tokens": "_tokens.jsonl",
}


class TaxonomyError(ValueError):
    """Raised when the explicit parent run cannot satisfy the contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(stem: object) -> bool:
    return (isinstance(stem, str) and bool(stem) and stem not in {".", ".."}
            and "\x00" not in stem and Path(stem).name == stem
            and "/" not in stem and "\\" not in stem)


def _required_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise TaxonomyError(f"{label} must be absolute: {path}")
    if path.is_symlink() or not path.is_dir():
        raise TaxonomyError(f"{label} is not a regular directory: {path}")
    return path.resolve(strict=True)


def resolve_anchored_artifact(root: Path, filename: str, *, required: bool = True) -> Path | None:
    """Resolve one regular child of ``root`` without allowing escapes/links."""
    root = _required_directory(root, "artifact root")
    if (not filename or Path(filename).name != filename or filename in {".", ".."}
            or "/" in filename or "\\" in filename or "\x00" in filename):
        raise TaxonomyError(f"unsafe artifact filename: {filename!r}")
    path = root / filename
    if not path.exists():
        if required:
            raise TaxonomyError(f"required artifact missing: {path}")
        return None
    if path.is_symlink() or not path.is_file():
        raise TaxonomyError(f"artifact is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    if resolved.parent != root:
        raise TaxonomyError(f"artifact escapes anchored root: {path}")
    return resolved


def _descriptor(root: Path, filename: str, *, required: bool) -> dict[str, Any]:
    path = resolve_anchored_artifact(root, filename, required=required)
    if path is None:
        return {"path": str((root / filename).absolute()), "exists": False,
                "regular_file": False, "size_bytes": None, "sha256": None}
    return {"path": str(path), "exists": True, "regular_file": True,
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaxonomyError(f"unable to read {label}: {exc}") from exc


def _read_report(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.is_symlink() or not path.is_file():
        raise TaxonomyError(f"report is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TaxonomyError(f"invalid report JSON at line {line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise TaxonomyError(f"report line {line_number} is not an object")
                row["_line_number"] = line_number
                rows.append(row)
    except OSError as exc:
        raise TaxonomyError(f"unable to read report: {exc}") from exc
    return rows, sha256_file(path)


def _candidate_stems(filtered_dir: Path) -> tuple[list[str], dict[str, Path]]:
    filtered_dir = _required_directory(filtered_dir, "filtered directory")
    stems: list[str] = []
    paths: dict[str, Path] = {}
    try:
        entries = sorted(filtered_dir.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise TaxonomyError(f"unable to list filtered directory: {exc}") from exc
    for path in entries:
        if path.suffix != ".TextGrid":
            continue
        if path.is_symlink() or not path.is_file():
            raise TaxonomyError(f"filtered candidate is not a regular file: {path}")
        stem = path.stem
        if not _safe_stem(stem):
            raise TaxonomyError(f"unsafe filtered stem: {stem!r}")
        if stem in paths:
            raise TaxonomyError(f"duplicate filtered stem: {stem}")
        paths[stem] = path.resolve(strict=True)
        stems.append(stem)
    if len(stems) != EXPECTED_STEM_COUNT:
        raise TaxonomyError(f"filtered candidate count mismatch: {len(stems)} != {EXPECTED_STEM_COUNT}")
    if len(set(stems)) != len(stems):
        raise TaxonomyError("duplicate filtered stems")
    return stems, paths


def _unique_report_stems(rows: Iterable[dict[str, Any]]) -> set[str]:
    stems: set[str] = set()
    for row in rows:
        stem = row.get("stem")
        if not _safe_stem(stem):
            raise TaxonomyError(f"unsafe or missing report stem: {stem!r}")
        if stem in stems:
            raise TaxonomyError(f"duplicate report stem: {stem}")
        stems.add(stem)
    return stems


def _validate_parent_receipt(receipt: dict[str, Any], candidate_stems: set[str]) -> None:
    if receipt.get("source_count") != EXPECTED_SOURCE_COUNT:
        raise TaxonomyError("parent source count mismatch")
    if receipt.get("output_count") != EXPECTED_OUTPUT_COUNT:
        raise TaxonomyError("parent output count mismatch")
    if receipt.get("filtered_count") != EXPECTED_STEM_COUNT:
        raise TaxonomyError("parent filtered count mismatch")
    filtered = receipt.get("filtered")
    if not isinstance(filtered, dict) or not isinstance(filtered.get("stems"), list):
        raise TaxonomyError("parent filtered stem ledger missing")
    stems = filtered["stems"]
    if len(stems) != len(set(stems)):
        raise TaxonomyError("duplicate stems in parent filtered ledger")
    if set(stems) != candidate_stems:
        raise TaxonomyError("parent filtered ledger does not match filtered directory")


def _validate_report(rows: list[dict[str, Any]], candidate_stems: set[str],
                    filtered_paths: dict[str, Path]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_SOURCE_COUNT:
        raise TaxonomyError(f"parent report row count mismatch: {len(rows)} != {EXPECTED_SOURCE_COUNT}")
    report_stems = _unique_report_stems(rows)
    if len(report_stems) != EXPECTED_SOURCE_COUNT:
        raise TaxonomyError("parent report stem count mismatch")
    selected: list[dict[str, Any]] = []
    for row in rows:
        stem = row["stem"]
        status = row.get("status")
        if stem not in candidate_stems:
            continue
        if not isinstance(status, str) or not status.startswith("filtered_"):
            raise TaxonomyError(f"candidate has non-filtered status: {stem}")
        reasons = row.get("filter_reasons")
        if (not isinstance(reasons, list) or not reasons
                or not all(isinstance(reason, str) and reason for reason in reasons)
                or len(reasons) != len(set(reasons))):
            raise TaxonomyError(f"invalid complete reason set: {stem}")
        expected_status = "filtered_" + "_".join(reasons)
        if status != expected_status:
            raise TaxonomyError(f"status/reason mismatch for {stem}: {status} != {expected_status}")
        output = row.get("output")
        expected_path = filtered_paths[stem]
        if not isinstance(output, str) or not Path(output).is_absolute():
            raise TaxonomyError(f"candidate report output path is not absolute: {stem}")
        try:
            output_path = Path(output).resolve(strict=True)
        except OSError as exc:
            raise TaxonomyError(f"candidate report output path is unreadable: {stem}") from exc
        if output_path != expected_path:
            raise TaxonomyError(f"candidate report output escapes filtered path: {stem}")
        selected.append(row)
    if len(selected) != EXPECTED_STEM_COUNT:
        raise TaxonomyError(f"filtered report row count mismatch: {len(selected)}")
    status_counts = Counter(row["status"] for row in selected)
    if dict(status_counts) != EXPECTED_STATUS_COUNTS:
        raise TaxonomyError(f"status group mismatch: {dict(status_counts)}")
    reason_counts = Counter(reason for row in selected for reason in row["filter_reasons"])
    if dict(reason_counts) != EXPECTED_REASON_OCCURRENCES:
        raise TaxonomyError(f"reason occurrence mismatch: {dict(reason_counts)}")
    compound_count = sum(count for status, count in status_counts.items()
                         if status not in BASE_STATUS_COUNTS)
    if compound_count != 12:
        raise TaxonomyError(f"compound/overlap stem count mismatch: {compound_count} != 12")
    return sorted(selected, key=lambda row: row["stem"])


def _primary_partition(reasons: list[str]) -> str:
    if "mfa_unknown_source" in reasons:
        return "unknown_gate"
    if "missing_mfa_alignment" in reasons:
        return "missing_mfa_axis"
    return "qc_audio"


def _repair_eligibility(partition: str) -> dict[str, Any]:
    if partition == "missing_mfa_axis":
        return {"eligible": False, "lane": "mfa_alignment_rerun",
                "basis": "missing_mfa_alignment"}
    if partition == "unknown_gate":
        return {"eligible": True, "lane": "unknown_source_repair",
                "basis": "mfa_unknown_source"}
    return {"eligible": True, "lane": "qc_audio_repair",
            "basis": "quality_control_or_audio_reason"}


def _artifact_evidence(stem: str, filtered_path: Path, workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ctc_root = workspace / "ctc_pretg"
    en_root = workspace / "en_phones"
    aligned_root = workspace / "aligned"
    padded_root = workspace / "padded_audio"
    availability: dict[str, Any] = {
        "filtered": {"textgrid": _descriptor(filtered_path.parent, filtered_path.name, required=True)},
        "ctc_pretg": {}, "en_phones": {}, "aligned": {}, "padded_audio": {},
    }
    for key, suffix in CTC_FILES.items():
        descriptor = _descriptor(ctc_root, stem + suffix, required=True)
        availability["ctc_pretg"][key] = descriptor
    availability["en_phones"]["json"] = _descriptor(en_root, stem + "_en_phones.json", required=True)
    availability["aligned"]["textgrid"] = _descriptor(aligned_root, stem + ".TextGrid", required=False)
    availability["padded_audio"]["wav"] = _descriptor(padded_root, stem + ".wav", required=True)
    hashes: dict[str, Any] = {}
    for category, values in availability.items():
        hashes[category] = {key: value["sha256"] for key, value in values.items()}
    return availability, hashes


def _make_record(row: dict[str, Any], filtered_path: Path, report_path: Path,
                 report_sha: str, workspace: Path) -> dict[str, Any]:
    stem = row["stem"]
    reasons = list(row["filter_reasons"])
    partition = _primary_partition(reasons)
    availability, artifact_hashes = _artifact_evidence(stem, filtered_path, workspace)
    row_for_evidence = {key: value for key, value in row.items() if key != "_line_number"}
    row_sha = sha256_bytes(_canonical(row_for_evidence))
    return {
        "stem": stem,
        "reasons": reasons,
        "primary_partition": partition,
        "report_evidence": {
            "path": str(report_path),
            "line_number": row["_line_number"],
            "status": row["status"],
            "row": row_for_evidence,
        },
        "artifact_availability": availability,
        "repair_eligibility": _repair_eligibility(partition),
        "evidence_hashes": {
            "report_file_sha256": report_sha,
            "report_row_sha256": row_sha,
            "artifacts": artifact_hashes,
        },
    }


def analyze(strict_run: Path = DEFAULT_STRICT_RUN,
            workspace: Path = DEFAULT_WORKSPACE,
            output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Validate the parent run and write the three W1 taxonomy artifacts."""
    strict_run = _required_directory(Path(strict_run), "strict run")
    workspace = _required_directory(Path(workspace), "workspace")
    if strict_run.parent != (workspace / "strict_ok_runs").resolve():
        raise TaxonomyError("strict run is not anchored under the supplied workspace strict_ok_runs")
    filtered_dir = strict_run / "filtered"
    report_path = strict_run / REPORT_NAME
    receipt_path = strict_run / RECEIPT_NAME
    stems, filtered_paths = _candidate_stems(filtered_dir)
    rows, report_sha = _read_report(report_path)
    receipt = _read_json(receipt_path, "pipeline receipt")
    if not isinstance(receipt, dict):
        raise TaxonomyError("pipeline receipt is not an object")
    _validate_parent_receipt(receipt, set(stems))
    selected = _validate_report(rows, set(stems), filtered_paths)
    records = [_make_record(row, filtered_paths[row["stem"]], report_path, report_sha, workspace)
               for row in selected]
    primary_counts = Counter(record["primary_partition"] for record in records)
    if dict(primary_counts) != EXPECTED_PRIMARY_PARTITIONS:
        raise TaxonomyError(f"primary partition mismatch: {dict(primary_counts)}")
    missing_ids = sorted(record["stem"][:6] for record in records
                         if "missing_mfa_alignment" in record["reasons"])
    if missing_ids != EXPECTED_MISSING_IDS:
        raise TaxonomyError(f"missing MFA stem mismatch: {missing_ids}")
    artifact_counts = Counter()
    for record in records:
        for category, values in record["artifact_availability"].items():
            for descriptor in values.values():
                artifact_counts[(category, descriptor["exists"])] += 1
    summary = {
        "schema": SCHEMA,
        "parent": {
            "strict_run": str(strict_run),
            "workspace": str(workspace),
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "filtered_dir": str(filtered_dir),
            "filtered_stem_count": len(stems),
        },
        "counts": {
            "stems": len(records),
            "unique_stems": len({record["stem"] for record in records}),
            "status_groups": dict(sorted(Counter(row["status"] for row in selected).items())),
            "exact_groups": dict(BASE_STATUS_COUNTS),
            "compound_overlap_stems": 12,
            "reason_occurrences": dict(sorted(
                Counter(reason for row in selected for reason in row["filter_reasons"]).items())),
            "primary_partitions": dict(sorted(primary_counts.items())),
            "missing_mfa_ids": missing_ids,
            "repair_eligibility": {
                "eligible": sum(record["repair_eligibility"]["eligible"] for record in records),
                "ineligible": sum(not record["repair_eligibility"]["eligible"] for record in records),
            },
            "artifact_availability": {
                f"{category}:{'present' if present else 'missing'}": count
                for (category, present), count in sorted(artifact_counts.items())
            },
        },
        "validation": {
            "passed": True,
            "expected_status_groups": dict(EXPECTED_STATUS_COUNTS),
            "expected_reason_occurrences": dict(EXPECTED_REASON_OCCURRENCES),
            "expected_primary_partitions": dict(EXPECTED_PRIMARY_PARTITIONS),
        },
    }
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        raise TaxonomyError(f"output directory must be absolute: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "taxonomy.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (output_dir / "missing_mfa_stems.txt").write_text(
        "".join(stem_id + "\n" for stem_id in missing_ids), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-run", type=Path, default=DEFAULT_STRICT_RUN)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        summary = analyze(**vars(build_parser().parse_args(argv)))
    except TaxonomyError as exc:
        print(f"analyze_filtered_run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
