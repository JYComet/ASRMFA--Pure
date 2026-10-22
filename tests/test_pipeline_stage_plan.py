"""Ordinary parameter changes must not remove production stages."""
import copy
from argparse import Namespace

import pytest

from scripts import run_pipeline


def args(**values):
    return Namespace(step=None, skip_to=None, stop_after=None, scan_only=False, **values)


def test_parameters_keep_the_complete_0915all_processing_sequence():
    expected = ["trim", "resample", "prealign", "normalize_punct", "normalize",
                "normalize_ria", "normalize_en", "adjust", "align", "align_en", "postprocess"]
    for reference_mode, jobs, beam in (("authority", 1, 10), ("fallback", 40, 80)):
        cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
        cfg.update(reference_mode=reference_mode)
        cfg["mfa"].update(num_jobs=jobs, beam=beam)
        cfg["postprocess"]["strict_ok"] = False  # 0915ALL uses the outer publication gate.
        plan = run_pipeline.plan_pipeline_steps(args(), cfg, "full")
        assert plan["execution_steps"] == expected
        assert plan["invocation"] == "complete"


def test_optional_audit_does_not_mutate_later_pipeline_invocations():
    cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["postprocess"]["strict_ok"] = False
    run_pipeline.plan_pipeline_steps(args(), cfg, "full")
    cfg["postprocess"]["strict_ok"] = True
    assert run_pipeline.plan_pipeline_steps(args(), cfg, "full")["execution_steps"][-1] == "strict_ok"


def test_skip_flag_cannot_silently_punch_a_hole_in_normal_processing():
    with pytest.raises(ValueError, match="skip"):
        run_pipeline.plan_pipeline_steps(args(skip_normalize_en=True),
                                         run_pipeline.DEFAULT_CFG, "full")


def test_prepared_chunk_resume_retains_all_downstream_stages():
    request = args()
    request.skip_to = "align"
    cfg = copy.deepcopy(run_pipeline.DEFAULT_CFG)
    cfg["postprocess"]["strict_ok"] = False
    plan = run_pipeline.plan_pipeline_steps(request, cfg, "full")
    assert plan["execution_steps"] == ["align", "align_en", "postprocess"]
    assert plan["prerequisite_steps"] == ["trim", "resample", "prealign", "normalize_punct",
                                         "normalize", "normalize_ria", "normalize_en", "adjust"]
    assert plan["invocation"] == "resume"


def test_stop_after_outside_resume_range_is_rejected():
    request = args()
    request.skip_to, request.stop_after = "postprocess", "adjust"
    with pytest.raises(ValueError, match="stop-after"):
        run_pipeline.plan_pipeline_steps(request, run_pipeline.DEFAULT_CFG, "full")


def test_resume_cannot_treat_a_plan_as_proof_of_completed_prerequisites(tmp_path):
    request = args()
    request.skip_to = "align"
    plan = run_pipeline.plan_pipeline_steps(request, run_pipeline.DEFAULT_CFG, "full")
    ctx = {"workspace": tmp_path}
    with pytest.raises(ValueError, match="execution receipt"):
        run_pipeline.initialize_stage_execution(ctx, run_pipeline.DEFAULT_CFG, plan)
