"""Fail-closed canary calculations for the JA/EN rollout gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence


DEFAULTS = {"total": 40, "min_accepted": 34, "min_bucket_accepted": 8, "mae_ms": 80.0, "p95_ms": 160.0,
            "buckets": ("ja_to_en", "en_to_ja", "no_pause", "short_english")}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values: return math.inf
    ordered = sorted(values); index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)); return ordered[index]


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_id(row: Mapping[str, Any]) -> str | None:
    for key in ("id", "uid", "stem", "row_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def evaluate_canary_gate(
    rows: Sequence[Mapping[str, Any]], *, gold: Mapping[str, Any] | None, target_id: str | None,
    proof: bool = False, thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the 40-row mixed gate without treating all rejects as success."""
    limits = dict(DEFAULTS); limits.update(thresholds or {})
    limits["buckets"] = DEFAULTS["buckets"]
    # Callers may tighten the gate, never weaken the approved floor.
    for key in ("total", "min_accepted", "min_bucket_accepted"):
        limits[key] = max(int(limits[key]), int(DEFAULTS[key]))
    for key in ("mae_ms", "p95_ms"):
        limits[key] = min(float(limits[key]), float(DEFAULTS[key]))
    reasons: list[str] = []
    if not isinstance(gold, Mapping) or not isinstance(gold.get("rows"), list) or not gold.get("rows"):
        return {"status": "BLOCKED", "target_id": target_id, "reasons": ["missing_gold"], "accepted": 0, "total": len(rows)}
    gold_id = gold.get("target_id")
    if not isinstance(target_id, str) or not isinstance(gold_id, str) or target_id != gold_id:
        reasons.append("gold_target_mismatch")
    if not proof or (not target_id or not target_id.startswith("gate-")):
        reasons.append("not_gate_proof_target")
    gold_rows = gold["rows"]
    gold_ids = [_row_id(row) if isinstance(row, Mapping) else None for row in gold_rows]
    result_ids = [_row_id(row) if isinstance(row, Mapping) else None for row in rows]
    if any(value is None for value in gold_ids + result_ids): reasons.append("row_id_missing")
    if len(set(value for value in gold_ids if value is not None)) != len(gold_ids): reasons.append("gold_id_duplicate")
    if len(set(value for value in result_ids if value is not None)) != len(result_ids): reasons.append("result_id_duplicate")
    if set(gold_ids) != set(result_ids): reasons.append("gold_result_set_mismatch")
    gold_by_id = {value: row for value, row in zip(gold_ids, gold_rows) if value is not None}
    gold_bucket_counts = Counter(str(row.get("bucket")) for row in gold_rows if isinstance(row, Mapping))
    for bucket in limits["buckets"]:
        if gold_bucket_counts.get(bucket, 0) != 10: reasons.append(f"gold_bucket_{bucket}_not_10")
    total = len(rows)
    if total != int(limits["total"]): reasons.append(f"expected_{limits['total']}_rows")
    accepted_rows = [row for row in rows if row.get("accepted") is True]
    accepted = len(accepted_rows)
    if accepted < int(limits["min_accepted"]): reasons.append("accepted_below_threshold")
    counts = Counter(str(row.get("bucket")) for row in accepted_rows)
    bucket_report = {bucket: counts.get(bucket, 0) for bucket in limits["buckets"]}
    result_bucket_counts = Counter(str(row.get("bucket")) for row in rows)
    for bucket in limits["buckets"]:
        if result_bucket_counts.get(bucket, 0) != 10: reasons.append(f"result_bucket_{bucket}_not_10")
    for bucket, count in bucket_report.items():
        if count < int(limits["min_bucket_accepted"]): reasons.append(f"bucket_{bucket}_below_threshold")
    if accepted == 0: reasons.append("all_rejected")
    metric_values: list[float] = []
    sample_rate = _finite_number(gold.get("sample_rate", 16000))
    if sample_rate is None or sample_rate <= 0: reasons.append("gold_sample_rate_invalid"); sample_rate = 16000.0
    for index, row in enumerate(rows):
        for flag in ("clipping", "overlap", "false_complete"):
            if row.get(flag) is True: reasons.append(f"{flag}@{index}")
        row_id = _row_id(row); gold_row = gold_by_id.get(row_id)
        if not isinstance(gold_row, Mapping):
            reasons.append(f"gold_row_missing@{index}"); continue
        expected_route = gold_row.get("gold_route", gold_row.get("route"))
        if expected_route is not None and row.get("route") is None:
            reasons.append(f"route_missing@{index}")
        elif expected_route is not None and row.get("route") != expected_route:
            reasons.append(f"route_error@{index}")
        if str(row.get("bucket")) != str(gold_row.get("bucket")):
            reasons.append(f"bucket_mismatch@{index}")
        for field, value in (("predicted_seam_sample", row.get("predicted_seam_sample")),
                             ("gold_seam_sample", gold_row.get("gold_seam_sample"))):
            if value is not None and _finite_number(value) is None:
                reasons.append(f"{field}_nonfinite@{index}")
        if row.get("accepted") is True:
            predicted = _finite_number(row.get("predicted_seam_sample", row.get("seam_sample")))
            expected = _finite_number(gold_row.get("gold_seam_sample", gold_row.get("seam_sample")))
            if predicted is None or expected is None:
                reasons.append(f"seam_sample_missing@{index}")
            else:
                metric_values.append(abs(predicted - expected) * 1000.0 / sample_rate)
    mae = sum(metric_values) / len(metric_values) if metric_values else math.inf
    p95 = _percentile(metric_values, .95)
    if mae > float(limits["mae_ms"]): reasons.append("seam_mae_above_threshold")
    if p95 > float(limits["p95_ms"]): reasons.append("seam_p95_above_threshold")
    if reasons: status = "BLOCKED" if any(reason in {"missing_gold", "gold_target_mismatch", "not_gate_proof_target"} for reason in reasons) else "REJECTED"
    else: status = "PASS"
    return {"status": status, "target_id": target_id, "gold_target_id": gold_id, "accepted": accepted,
            "total": total, "bucket_accepted": bucket_report, "seam_mae_ms": mae, "seam_p95_ms": p95,
            "reasons": reasons, "proof": bool(proof)}


def evaluate_reading_gate(
    rows: Sequence[Mapping[str, Any]], *, gold: Mapping[str, Any] | None, target_id: str | None,
    proof: bool = False,
) -> dict[str, Any]:
    """Evaluate the required 30 consistent and 30 ambiguous reading sets."""
    if not isinstance(gold, Mapping) or not isinstance(gold.get("rows"), list) or not gold.get("rows"):
        return {"status": "BLOCKED", "reasons": ["missing_gold"], "target_id": target_id}
    reasons = []
    if not isinstance(target_id, str) or target_id != gold.get("target_id"):
        reasons.append("gold_target_mismatch")
    if not proof or (not target_id or not target_id.startswith("gate-")):
        reasons.append("not_gate_proof_target")
    gold_rows = gold["rows"]
    gold_ids = [_row_id(row) if isinstance(row, Mapping) else None for row in gold_rows]
    result_ids = [_row_id(row) if isinstance(row, Mapping) else None for row in rows]
    if len(gold_rows) != 60: reasons.append("expected_60_gold_rows")
    if any(value is None for value in gold_ids + result_ids): reasons.append("row_id_missing")
    if len(set(value for value in gold_ids if value is not None)) != len(gold_ids): reasons.append("gold_id_duplicate")
    if len(set(value for value in result_ids if value is not None)) != len(result_ids): reasons.append("result_id_duplicate")
    if set(gold_ids) != set(result_ids): reasons.append("gold_result_set_mismatch")
    gold_by_id = {value: row for value, row in zip(gold_ids, gold_rows) if value is not None}
    for row in rows:
        if _row_id(row) not in gold_by_id: continue
        if row.get("accepted") is True:
            if row.get("reading") != gold_by_id[_row_id(row)].get("reading"):
                reasons.append(f"reading_mismatch@{_row_id(row)}")
            expected_route = gold_by_id[_row_id(row)].get("route")
            if expected_route is not None and row.get("route") != expected_route:
                reasons.append(f"route_mismatch@{_row_id(row)}")
    gold_set_counts = Counter(str(row.get("set")) for row in gold_rows if isinstance(row, Mapping))
    for name in ("consistent", "ambiguous"):
        if gold_set_counts.get(name, 0) != 30: reasons.append(f"gold_{name}_not_30")
    counts = Counter(str(gold_by_id[_row_id(row)].get("set")) for row in rows
                     if row.get("accepted") is True and _row_id(row) in gold_by_id)
    for name in ("consistent", "ambiguous"):
        if counts.get(name, 0) < 30: reasons.append(f"{name}_below_30")
    status = "BLOCKED" if any(reason in {"gold_target_mismatch", "not_gate_proof_target"} for reason in reasons) else ("PASS" if not reasons else "REJECTED")
    return {"status": status, "accepted": sum(counts.values()), "set_accepted": dict(counts), "reasons": reasons, "target_id": target_id}


def evaluate_pure_gate(rows: Sequence[Mapping[str, Any]], *, gold: Mapping[str, Any] | None, target_id: str | None, proof: bool = False) -> dict[str, Any]:
    if not isinstance(gold, Mapping) or not isinstance(gold.get("rows"), list) or len(gold["rows"]) != 20:
        return {"status": "BLOCKED", "reasons": ["missing_or_incomplete_gold"], "target_id": target_id}
    reasons: list[str] = []
    if target_id != gold.get("target_id"): reasons.append("gold_target_mismatch")
    if not proof or not isinstance(target_id, str) or not target_id.startswith("gate-"): reasons.append("not_gate_proof_target")
    gold_ids = [_row_id(row) if isinstance(row, Mapping) else None for row in gold["rows"]]
    result_ids = [_row_id(row) if isinstance(row, Mapping) else None for row in rows]
    if any(value is None for value in gold_ids + result_ids): reasons.append("row_id_missing")
    if len(set(value for value in gold_ids if value is not None)) != len(gold_ids): reasons.append("gold_id_duplicate")
    if len(set(value for value in result_ids if value is not None)) != len(result_ids): reasons.append("result_id_duplicate")
    if set(gold_ids) != set(result_ids): reasons.append("gold_result_set_mismatch")
    gold_by_id = {value: row for value, row in zip(gold_ids, gold["rows"]) if value is not None}
    accepted = [row for row in rows if row.get("accepted") is True]
    for row in accepted:
        expected = gold_by_id.get(_row_id(row), {}).get("route")
        if expected is not None and row.get("route") != expected: reasons.append("route_mismatch")
    routes = Counter(row.get("route") for row in accepted)
    if routes.get("ja", 0) < 10: reasons.append("pure_ja_below_10")
    if routes.get("en", 0) < 10: reasons.append("pure_en_below_10")
    status = "BLOCKED" if any(reason in {"gold_target_mismatch", "not_gate_proof_target", "missing_or_incomplete_gold"} for reason in reasons) else ("PASS" if not reasons else "REJECTED")
    return {"status": status, "accepted": len(accepted), "route_accepted": dict(routes), "reasons": reasons, "target_id": target_id}


calculate_canary_gate = evaluate_canary_gate
evaluate_mixed_gate = evaluate_canary_gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("rows", type=str); parser.add_argument("--gold", required=True); parser.add_argument("--target-id", required=True); parser.add_argument("--proof", action="store_true")
    args = parser.parse_args(argv)
    rows = json.loads(open(args.rows, encoding="utf-8").read()); gold = json.loads(open(args.gold, encoding="utf-8").read())
    report = evaluate_canary_gate(rows, gold=gold, target_id=args.target_id, proof=args.proof); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calculate_canary_gate", "evaluate_canary_gate", "evaluate_mixed_gate", "evaluate_pure_gate", "evaluate_reading_gate"]
