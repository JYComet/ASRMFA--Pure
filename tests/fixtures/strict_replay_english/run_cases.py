#!/usr/bin/env python3
"""Registry-driven synthetic fail-close matrix for replay interfaces."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUBSET = ROOT / "scripts" / "verify_strict_replay_english_subset.py"
LIFECYCLE = ROOT / "scripts" / "verify_strict_replay_lifecycle.py"
SUPPORTED_CASES = {
    "historical_v1_as_v2": "subset", "global_hash_mismatch": "subset", "canonical_external_stem": "subset",
    "import_receipt_hash_mismatch": "subset", "ledger_hash_mismatch": "subset", "canonical_path_mismatch": "subset",
    "membership_count_mismatch": "subset", "record_order_drift": "subset", "duplicate_record": "subset",
    "missing_record": "subset", "excluded_record": "subset", "later_cycle_path": "subset", "production_scope": "subset",
    "lifecycle_positive": "lifecycle", "lifecycle_existing_target_overwrite": "lifecycle",
    "lifecycle_stage_state_binding": "lifecycle", "lifecycle_upstream_downstream_cycle": "lifecycle",
    "lifecycle_tamper_one_byte": "lifecycle", "lifecycle_rc1_present": "lifecycle",
    "lifecycle_tamper_bad_path": "lifecycle", "lifecycle_tamper_bad_hash": "lifecycle",
    "lifecycle_tamper_symlink": "lifecycle", "lifecycle_tamper_json": "lifecycle",
    "lifecycle_missing_not_created": "lifecycle", "lifecycle_missing_path_exists": "lifecycle",
    "lifecycle_unreadable_valid": "lifecycle", "lifecycle_unreadable_false_claim": "lifecycle",
    "lifecycle_unsafe_valid": "lifecycle", "lifecycle_unsafe_false_claim": "lifecycle",
    "lifecycle_rc0_missing": "lifecycle", "lifecycle_unknown_status": "lifecycle",
    "lifecycle_unknown_reason": "lifecycle", "lifecycle_omitted_fields": "lifecycle",
    "lifecycle_missing_with_hash": "lifecycle", "lifecycle_present_no_hash": "lifecycle",
    "path_output_parent_import": "subset", "path_import_missing": "subset", "path_bad_basename": "subset",
    "path_bad_parent": "subset", "path_role_swap": "subset", "path_relative": "subset", "path_dotdot": "subset",
    "path_workspace_output_collision": "subset", "path_outside_workspace": "subset",
    "path_payload_actual_mismatch": "subset", "path_hash_change": "subset", "path_run_id_change": "subset",
    "path_membership_change": "subset", "path_late_output_binding": "subset", "path_production_replay_fields": "subset",
    "v21_parent_role_swap": "subset", "v21_parent_hash_mismatch": "subset", "v21_missing_binding": "subset",
    "v21_alias_binding": "subset", "v20_schema_rejected": "subset", "v21_mixed_encoding": "subset",
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _subset_payload(base: dict, out: Path, mutation: str) -> Path:
    payload = copy.deepcopy(base); payload.setdefault("paths", {})["output"] = str(out)
    payload["strict_replay_import_sha256"] = "fixture-bound"
    selected = payload.get("selected_slot_records", payload.get("selected_slots"))
    stems = sorted({row["stem"] for row in selected})
    subset = {"schema": "strict-replay-english-subset-v2", "import_receipt_sha256": "fixture-bound",
              "selected_stems": stems, "canonical_selected_stems_sha256": digest(stems),
              "english_stems": [], "ledgers": []}
    if mutation == "historical_v1_as_v2": subset["schema"] = "strict-en-mfa-v1"
    elif mutation == "global_hash_mismatch": subset["canonical_selected_stems_sha256"] = "0" * 64
    elif mutation == "canonical_external_stem": subset["english_stems"] = ["outside-canonical"]
    elif mutation == "import_receipt_hash_mismatch": subset["import_receipt_sha256"] = "0" * 64
    elif mutation == "ledger_hash_mismatch": subset["english_stems"] = [stems[0]]
    elif mutation == "canonical_path_mismatch": payload["canonical"]["path"] = "/tmp/not-the-canonical-manifest.json"
    elif mutation == "membership_count_mismatch": subset["selected_stems"] = stems + ["outside-canonical"]
    elif mutation == "record_order_drift": subset["selected_stems"] = list(reversed(stems)) + ["outside-canonical"]
    elif mutation == "missing_record": subset["english_stems"] = stems[:1]
    elif mutation == "excluded_record": subset["english_stems"] = ["outside-canonical"]
    elif mutation == "duplicate_record": subset["ledgers"] = [{"stem": stems[0], "path": "none", "sha256": "0" * 64}] * 2
    elif mutation == "production_scope": payload["scope"] = "full"
    elif mutation == "later_cycle_path": subset["later_artifact_path"] = "strict_replay_final_evidence.json"
    elif mutation.startswith("path_"):
        workspace = out.parent / "workspace"; workspace.mkdir(exist_ok=True)
        payload.setdefault("paths", {}).update({"workspace": str(workspace), "output": str(out), "immutable_import": str(out / "strict_replay_import.json")})
        if mutation == "path_import_missing": payload["paths"]["immutable_import"] = ""
        elif mutation == "path_bad_basename": payload["paths"]["immutable_import"] = str(workspace / "wrong.json")
        elif mutation == "path_bad_parent": payload["paths"]["immutable_import"] = str(workspace.parent / "strict_replay_import.json")
        elif mutation == "path_role_swap": payload["paths"]["immutable_import"] = str(out / "strict_replay_import.json")
        elif mutation == "path_relative": payload["paths"]["immutable_import"] = "strict_replay_import.json"
        elif mutation == "path_dotdot": payload["paths"]["immutable_import"] = str(workspace / ".." / "strict_replay_import.json")
        elif mutation == "path_workspace_output_collision": payload["paths"]["workspace"] = str(out)
        elif mutation == "path_outside_workspace": payload["paths"]["immutable_import"] = "/tmp/external/strict_replay_import.json"
        elif mutation == "path_payload_actual_mismatch": payload["paths"]["immutable_import"] = str(workspace / "other.json")
        elif mutation == "path_hash_change": payload["strict_replay_import_sha256"] = "0" * 64
        elif mutation == "path_run_id_change": payload["run_id"] = "wrong-run"
        elif mutation == "path_membership_change": payload["source_manifest_slots"] = 24
        elif mutation == "path_late_output_binding": subset["output"] = str(out / "filtered")
    elif mutation == "path_production_replay_fields": payload["scope"] = "production"
    elif mutation.startswith("v21_"):
        subset["parent_global_manifest"] = {"authoritative_source": {"path": "a"}, "workspace_copy": {"path": "a"}}
    if mutation == "v20_schema_rejected":
        subset["schema"] = "strict-replay-english-alignment-subset-v2"
    payload["english_subset"] = subset
    path = out / "strict_replay_import.json"; path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _lifecycle_fixture(root: Path, mutation: str) -> tuple[Path, Path]:
    workspace, output = root / "workspace", root / "output"; workspace.mkdir(parents=True); output.mkdir()
    imp = workspace / "strict_replay_import.json"; eng = workspace / "strict_replay_english_import.json"
    state = workspace / ".strict_replay_stage_state.json"; formal = output / ".pipeline_run_receipt_v2.json"
    strict = output / "strict_ok_manifest.json"; final = output / "strict_replay_final_evidence.json"
    imp.write_text(json.dumps({"schema": "strict-replay-import-v2.1", "paths": {"workspace": str(workspace), "output": str(output), "immutable_import": str(imp)}}), encoding="utf-8")
    eng.write_text(json.dumps({"schema": "strict-replay-english-import-v2.1"}), encoding="utf-8")
    (workspace / "strict_replay_import.sha256").write_text(hashlib.sha256(imp.read_bytes()).hexdigest() + "\n")
    state.write_text(json.dumps({"authoritative": False, "stages": []}), encoding="utf-8")
    formal_payload = {"mode": "strict_replay", "extra": {"strict_replay_receipt": str(imp)}}
    formal_payload["extra"]["strict_replay_evidence"] = {"import_manifest": str(imp), "import_sha256": hashlib.sha256(imp.read_bytes()).hexdigest(), "english_import": str(eng), "english_sha256": hashlib.sha256(eng.read_bytes()).hexdigest()}
    formal.write_text(json.dumps(formal_payload), encoding="utf-8")
    strict_payload = {"pipeline_accounting_receipt": {"path": str(formal), "sha256": hashlib.sha256(formal.read_bytes()).hexdigest()}, "strict_replay_evidence": {"formal_receipt": {"path": str(formal), "sha256": hashlib.sha256(formal.read_bytes()).hexdigest()}, "english_import": {"path": str(eng), "sha256": hashlib.sha256(eng.read_bytes()).hexdigest()}}}
    strict.write_text(json.dumps(strict_payload), encoding="utf-8")
    final_payload = {"schema": "strict-replay-final-evidence-v1", "authoritative": False, "import_sha256": hashlib.sha256(imp.read_bytes()).hexdigest(), "english_import_sha256": hashlib.sha256(eng.read_bytes()).hexdigest(), "formal_receipt_sha256": hashlib.sha256(formal.read_bytes()).hexdigest(), "strict_manifest_sha256": hashlib.sha256(strict.read_bytes()).hexdigest(), "strict_manifest_binding": {"status": "present", "expected_path": str(strict), "sha256": hashlib.sha256(strict.read_bytes()).hexdigest()}, "stage_results": [{"stage": "postprocess", "return_code": 0}, {"stage": "strict_ok", "return_code": 0}], "global_reasons": []}
    final.write_text(json.dumps(final_payload), encoding="utf-8")
    if mutation == "lifecycle_stage_state_binding": state.write_text(json.dumps({"authoritative": True}), encoding="utf-8")
    elif mutation == "lifecycle_upstream_downstream_cycle": imp.write_text(json.dumps({"schema": "strict-replay-import-v1", "next": str(final)}), encoding="utf-8")
    elif mutation == "lifecycle_tamper_one_byte": imp.write_text('{"schema":"strict-replay-import-v1","x":1}', encoding="utf-8")
    elif mutation == "lifecycle_existing_target_overwrite": final.write_text("overwrite", encoding="utf-8")
    elif mutation == "lifecycle_rc1_present":
        final_payload["stage_results"][-1]["return_code"] = 1; final_payload["global_reasons"] = ["strict_ok_failed"]
        final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_tamper_bad_path":
        final_payload["strict_manifest_binding"]["expected_path"] = str(output / "other.json"); final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_tamper_bad_hash":
        final_payload["strict_manifest_binding"]["sha256"] = "0" * 64; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_tamper_symlink":
        strict.unlink(); strict.symlink_to(imp); final_payload["strict_manifest_binding"]["status"] = "present"; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_tamper_json":
        strict.write_text("not json", encoding="utf-8"); final_payload["strict_manifest_binding"]["sha256"] = hashlib.sha256(strict.read_bytes()).hexdigest(); final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_missing_not_created":
        strict.unlink(); final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": None, "missing_reason": "not_created"}; final_payload["stage_results"][-1]["return_code"] = 1; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_missing_path_exists":
        final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": None, "missing_reason": "not_created"}; final_payload["stage_results"][-1]["return_code"] = 1; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation in {"lifecycle_unreadable_valid", "lifecycle_unreadable_false_claim"}:
        final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": None, "missing_reason": "unreadable"}; final_payload["stage_results"][-1]["return_code"] = 1; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation in {"lifecycle_unsafe_valid", "lifecycle_unsafe_false_claim"}:
        strict.unlink(); strict.symlink_to(imp); final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": None, "missing_reason": "unsafe_file_type"}; final_payload["stage_results"][-1]["return_code"] = 1; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_rc0_missing":
        strict.unlink(); final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": None, "missing_reason": "not_created"}; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_unknown_status":
        final_payload["strict_manifest_binding"]["status"] = "unknown"; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_unknown_reason":
        final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": None, "missing_reason": "unknown"}; final_payload["stage_results"][-1]["return_code"] = 1; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_omitted_fields":
        final_payload.pop("strict_manifest_binding"); final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_missing_with_hash":
        strict.unlink(); final_payload["strict_manifest_binding"] = {"status": "missing", "expected_path": str(strict), "sha256": "0" * 64, "missing_reason": "not_created"}; final_payload["stage_results"][-1]["return_code"] = 1; final.write_text(json.dumps(final_payload), encoding="utf-8")
    elif mutation == "lifecycle_present_no_hash":
        final_payload["strict_manifest_binding"].pop("sha256"); final.write_text(json.dumps(final_payload), encoding="utf-8")
    return workspace, output


def main() -> int:
    if len(sys.argv) != 2: return 2
    cases = json.loads((Path(__file__).parent / "cases.json").read_text())['cases']
    declared = [c.get("name") for c in cases]
    handlers = SUPPORTED_CASES.copy()
    if (len(declared) != len(set(declared)) or set(declared) != set(handlers)
            or any(c.get("expected_rc") not in (0, 1) or "expected_rc" not in c for c in cases)):
        print("FAIL registry declaration/handler coverage"); return 1
    base = json.loads(Path(sys.argv[1]).read_text()); failures = 0
    with tempfile.TemporaryDirectory(prefix="strict-replay-fixture-") as td:
        root = Path(td)
        for case in cases:
            name, expected, kind = case["name"], case["expected_rc"], handlers[case["name"]]
            try:
                if kind == "lifecycle":
                    ws, out = _lifecycle_fixture(root / name, name)
                    result = subprocess.run([sys.executable, str(LIFECYCLE), "--workspace", str(ws), "--output", str(out)], capture_output=True)
                else:
                    out = root / name; out.mkdir(); receipt = _subset_payload(base, out, name)
                    result = subprocess.run([sys.executable, str(SUBSET), str(receipt)], capture_output=True)
                actual = result.returncode
            except Exception:
                actual = 1
            status = "PASS" if actual == expected else "FAIL"
            print(f"{name} handler={name} kind={kind} actual={actual} expected={expected} {status}")
            failures += status == "FAIL"
    return int(failures)


if __name__ == "__main__": raise SystemExit(main())
