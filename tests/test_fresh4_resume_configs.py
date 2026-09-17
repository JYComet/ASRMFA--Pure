"""Safety contracts for the fresh4 post-CTC resume configurations."""

import json
from pathlib import Path

import pytest
import yaml


RUN_ROOT = Path(
    "/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4"
)
CONFIG_ROOT = RUN_ROOT / "resume_configs"
COHORTS = (
    "reverse1999_reference",
    "reverse1999_asr",
    "persona_asr",
    "punishing_gray_raven_reference",
    "punishing_gray_raven_asr",
    "wuwa_new_reference",
    "wuwa_new_asr",
)


@pytest.mark.parametrize("name", COHORTS)
def test_resume_config_binds_canonical_ctc_and_disables_producer(name):
    path = CONFIG_ROOT / f"{name}.yaml"
    assert path.is_file(), f"missing run-local resume config: {path}"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["mode"] == "nvrasr_fallback"
    assert cfg["require_fresh_workspace"] is False
    assert cfg["ctc_pretg"].endswith("/ctc_raw")
    assert "/pipeline/" in cfg["ctc_pretg"]
    assert cfg["ctc_prealign"]["enabled"] is False
    assert cfg["ctc_prealign"]["all_gpus"] is False
    assert cfg["pad_silence"]["enabled"] is False
    assert cfg["mfa"]["native_anchor_fallback"] is True
    assert cfg["mfa"]["allow_partial"] is True
    assert cfg["mfa"]["min_output_ratio"] == 0.95
    assert cfg["workspace"].startswith(str(RUN_ROOT / "runtime" / "resume"))
    assert cfg["output_dir"].startswith(str(RUN_ROOT / "runtime" / "resume"))


def test_resume_command_manifest_starts_at_resample_without_ctc_or_gpu():
    path = CONFIG_ROOT / "commands.json"
    assert path.is_file(), f"missing resume command manifest: {path}"
    commands = json.loads(path.read_text(encoding="utf-8"))
    assert set(commands) == set(COHORTS)
    for name, command in commands.items():
        assert command["skip_to"] == "resample"
        assert command["config"] == str(CONFIG_ROOT / f"{name}.yaml")
        assert "ctc_prealign.py" not in " ".join(command["argv"])
        assert "--all-gpus" not in command["argv"]


@pytest.mark.parametrize("name", COHORTS)
def test_resume_config_has_sealed_ctc_accounting_denominator(name):
    """A resume may consume only a sealed producer bundle with v2 accounting."""
    cfg = yaml.safe_load(
        (CONFIG_ROOT / f"{name}.yaml").read_text(encoding="utf-8"))
    ctc_root = Path(cfg["ctc_pretg"])
    raw_receipt = json.loads(
        (ctc_root / ".ctc_run_receipt.json").read_text(encoding="utf-8"))
    accounting = json.loads(
        (ctc_root / ".pipeline_run_receipt_v2.json").read_text(encoding="utf-8"))
    assert raw_receipt.get("input_stems")
    assert raw_receipt.get("schema") == "ctc-run-receipt-v2"
    # The raw manifest is materialized by the first downstream bind when an
    # older completed CTC run predates the manifest contract.  Resuming must
    # still pin the producer/accounting receipts before any transform.
    assert (ctc_root / ".ctc_normalized").is_file()
    eligible = set(accounting["eligible"]["stems"])
    output = set(accounting["output"]["stems"])
    filtered = set(accounting["filtered"]["stems"])
    assert eligible == output | filtered
    exclusions = {
        row["stem"] for row in accounting.get("exclusions", [])
        if isinstance(row, dict) and "stem" in row
    }
    assert not (eligible & exclusions)
