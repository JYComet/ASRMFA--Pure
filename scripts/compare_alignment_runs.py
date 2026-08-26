#!/usr/bin/env python3
"""Fail-closed matched-stem comparison for v7/v9/v10 alignment runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EXPECTED_STEMS_SHA256 = (
    "e22f9d2ea067ef7e9bfa3abbe6fe36d594e8197a10eaddc02536c025be91357b"
)
EXPECTED_COUNT = 1000


class ComparisonError(RuntimeError):
    """Raised when comparison evidence is incomplete or inconsistent."""


def _digest(stems: list[str]) -> str:
    import hashlib

    payload = json.dumps(sorted(stems), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_output(root: Path) -> Path:
    root = Path(root)
    if (root / "strict_ok_manifest.json").is_file():
        return root
    candidates = sorted(root.glob("strict_ok_runs/*/output"))
    if len(candidates) != 1:
        raise ComparisonError(
            f"{root}: expected one explicit strict output, found {len(candidates)}; "
            "pass the strict_ok_runs/<run>/output directory")
    return candidates[0]


def _load_bundle(label: str, root: Path) -> dict:
    output = _strict_output(root)
    strict_path = output / "strict_ok_manifest.json"
    receipt_path = output / ".pipeline_run_receipt_v2.json"
    report_path = output / "postprocess_report.jsonl"
    for path in (strict_path, receipt_path, report_path):
        if not path.is_file():
            raise ComparisonError(f"{label}: missing required artifact: {path}")
    try:
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        report = [json.loads(line) for line in
                  report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{label}: unreadable evidence: {exc}") from exc
    expected = strict.get("expected_stems")
    if not isinstance(expected, list) or len(expected) != EXPECTED_COUNT:
        raise ComparisonError(f"{label}: strict expected_stems must contain 1000 stems")
    if len(set(expected)) != len(expected) or _digest(expected) != EXPECTED_STEMS_SHA256:
        raise ComparisonError(f"{label}: expected stem hash/count contract failed")
    if receipt.get("eligible", {}).get("stems") != sorted(expected):
        raise ComparisonError(f"{label}: accounting eligible stems mismatch")
    rows = {}
    for row in report:
        stem = row.get("stem") if isinstance(row, dict) else None
        if not isinstance(stem, str) or stem in rows:
            raise ComparisonError(f"{label}: postprocess report has malformed/duplicate stem")
        rows[stem] = row
    if set(rows) != set(expected):
        raise ComparisonError(f"{label}: postprocess report is not a complete 1000-stem set")
    ok = {row.get("stem") for row in strict.get("ok", []) if isinstance(row, dict)}
    rejected = strict.get("rejected", {})
    if not isinstance(rejected, dict) or ok & set(rejected):
        raise ComparisonError(f"{label}: strict partition is malformed or overlapping")
    if ok | set(rejected) != set(expected):
        raise ComparisonError(f"{label}: strict partition is incomplete")
    reasons = Counter()
    for row in rows.values():
        values = row.get("filter_reasons", row.get("reasons", []))
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            reasons.update(str(value) for value in values)
    return {"label": label, "root": str(root), "output": str(output),
            "expected": expected, "expected_hash": _digest(expected),
            "postprocess_output": sum(row.get("status") == "ok"
                                       for row in rows.values()),
            "postprocess_filtered": sum(row.get("status") != "ok"
                                         for row in rows.values()),
            "strict_ok": sorted(ok), "strict_filtered": sorted(rejected),
            "reasons": dict(sorted(reasons.items())), "report": rows}


def _canonical_shrinkage(bundle: dict) -> dict:
    """Count explicit processed/canonical end shrinkage in available JSON evidence."""
    output = Path(bundle["output"])
    roots = [output.parent.parent / "en_phones", output.parent / "en_phones",
             output.parents[2] / "en_phones" if len(output.parents) > 2 else output / "en_phones",
             Path(bundle["root"]) / "en_phones"]
    rows = []
    found_root = False
    saw_canonical_field = False
    for root in roots:
        if not root.is_dir():
            continue
        found_root = True
        for path in sorted(root.glob("*_en_phones.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise ComparisonError(f"{bundle['label']}: unreadable English evidence: {path}")
            stem = payload.get("stem", path.name[:-len("_en_phones.json")])
            for segment in payload.get("segments", []):
                for word in segment.get("words", []):
                    span = word.get("canonical_span")
                    saw_canonical_field = saw_canonical_field or (
                        isinstance(span, (list, tuple)) and len(span) == 2)
                    end = word.get("end")
                    if (isinstance(span, list) and len(span) == 2
                            and isinstance(end, (int, float))
                            and end < span[1] - 1e-6):
                        rows.append({"stem": stem, "word": word.get("text"),
                                     "processed_end": end,
                                     "canonical_end": span[1]})
        if rows:
            break
    if not found_root:
        raise ComparisonError(
            f"{bundle['label']}: missing canonical-shrinkage English evidence")
    if not saw_canonical_field:
        return {"available": False, "reason": "canonical_span_missing",
                "count": None, "stems": [], "rows": []}
    return {"available": True, "count": len(rows), "stems": sorted({row["stem"] for row in rows}),
            "rows": rows}


def compare_alignment_runs(v7: Path, v9: Path, v10: Path) -> dict:
    bundles = [_load_bundle("v7", v7), _load_bundle("v9", v9),
               _load_bundle("v10", v10)]
    hashes = {bundle["label"]: bundle["expected_hash"] for bundle in bundles}
    if len(set(hashes.values())) != 1:
        raise ComparisonError("v7/v9/v10 expected stem hashes differ")
    by_label = {bundle["label"]: bundle for bundle in bundles}
    expected = set(bundles[0]["expected"])
    status = {}
    for stem in sorted(expected):
        status[stem] = {
            label: {"postprocess": by_label[label]["report"][stem].get("status"),
                    "strict": stem in set(by_label[label]["strict_ok"]),
                    "reasons": by_label[label]["report"][stem].get(
                        "filter_reasons", by_label[label]["report"][stem].get("reasons", []))}
            for label in ("v7", "v9", "v10")}
    v7_ok, v9_ok, v10_ok = (set(by_label[label]["strict_ok"])
                            for label in ("v7", "v9", "v10"))
    return {
        "schema": "alignment-run-comparison-v1",
        "expected_stems": {"count": EXPECTED_COUNT, "sha256": hashes["v7"]},
        "runs": {bundle["label"]: {
            "postprocess": {"output": bundle["postprocess_output"],
                             "filtered": bundle["postprocess_filtered"]},
            "strict": {"ok": len(bundle["strict_ok"]),
                        "filtered": len(bundle["strict_filtered"])},
            "overlap_reasons": bundle["reasons"],
            "canonical_shrinkage": _canonical_shrinkage(bundle),
            "output": bundle["output"],
        } for bundle in bundles},
        "per_stem": status,
        "v7_pass_to_v10_fail": sorted(v7_ok - v10_ok),
        "v9_fail_to_v10_recovered": sorted((expected - v9_ok) & v10_ok),
        "v10_new_regressions": sorted((v7_ok & v9_ok) - v10_ok),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", required=True, type=Path)
    parser.add_argument("--v9", required=True, type=Path)
    parser.add_argument("--v10", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_alignment_runs(args.v7, args.v9, args.v10)
    except ComparisonError as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
