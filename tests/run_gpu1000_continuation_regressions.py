"""Stdlib smoke harness for continuation safety regressions (pytest-free)."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_gpu1000_continuation import (
    test_continuation_preflight_requires_exact_one_and_binds_999,
    test_discover_retry_plan_is_read_only_and_returns_original_path,
    test_continue_refuses_stale_v1_without_v2,
    test_v2_selection_and_tamper_are_bound,
    test_v2_accepts_empty_canonical_dirs_and_historical_strict,
    test_downstream_resume_success_and_tamper,
    test_downstream_resume_allows_orchestrator_delta_after_v2,
    test_axis_recovery_corrects_missing_fields_and_binds_new_resume,
    test_axis_recovery_rejects_grid_or_audio_tamper,
    test_axis_recovery_rejects_stale_old_resume_without_writes,
    test_mocked_downstream_route_uses_only_permitted_steps,
    test_direct_entrypoint_exposes_downstream_resume_commands,
    test_continue_after_mfa_is_atomic_new_root_and_original_immutable,
    test_continuation_rejects_downstream_namespace,
    test_composite_lineage_rejects_tampered_original_receipt,
    test_finalize_requires_single_strict_receipt_and_promotes_gate,
    test_stored_preflight_drift_blocks_continuation,
        test_dual_strict_designation_prefers_bound_continuation,
        test_singleton_batch_policy_is_narrow,
        test_canonical_nonempty_bootstrap_builder,
    test_singleton_batch_policy_is_narrow,
    test_canonical_nonempty_bootstrap_builder,
)
from test_gpu1000_singleton_mfa import (
    test_singleton_rc0_valid_grid_success_once,
    test_singleton_rc0_no_grid_or_invalid_grid_fails_without_rescue,
    test_singleton_generic_nonzero_fails_without_rescue,
    test_singleton_noalignments_gets_exactly_one_200_800_rescue,
)


def main() -> int:
    tests = [
        test_continuation_preflight_requires_exact_one_and_binds_999,
        test_discover_retry_plan_is_read_only_and_returns_original_path,
        test_continue_refuses_stale_v1_without_v2,
        test_v2_selection_and_tamper_are_bound,
        test_v2_accepts_empty_canonical_dirs_and_historical_strict,
        test_downstream_resume_success_and_tamper,
        test_downstream_resume_allows_orchestrator_delta_after_v2,
        test_axis_recovery_corrects_missing_fields_and_binds_new_resume,
        test_axis_recovery_rejects_grid_or_audio_tamper,
        test_axis_recovery_rejects_stale_old_resume_without_writes,
        test_mocked_downstream_route_uses_only_permitted_steps,
        test_direct_entrypoint_exposes_downstream_resume_commands,
        test_continue_after_mfa_is_atomic_new_root_and_original_immutable,
        test_continuation_rejects_downstream_namespace,
        test_composite_lineage_rejects_tampered_original_receipt,
    test_finalize_requires_single_strict_receipt_and_promotes_gate,
    test_stored_preflight_drift_blocks_continuation,
    test_dual_strict_designation_prefers_bound_continuation,
    test_singleton_batch_policy_is_narrow,
    test_canonical_nonempty_bootstrap_builder,
    test_singleton_rc0_valid_grid_success_once,
    test_singleton_rc0_no_grid_or_invalid_grid_fails_without_rescue,
    test_singleton_generic_nonzero_fails_without_rescue,
    test_singleton_noalignments_gets_exactly_one_200_800_rescue,
]
    for test in tests:
        with TemporaryDirectory() as directory:
            test(Path(directory)) if test.__code__.co_argcount else test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
