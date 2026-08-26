"""Static dispatch gates for the isolated LAria v3 rerun packet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_CONFIG = ROOT / "configs/laria_v5_no_reference_strict_8gpu_20260825.yaml"
V1_CONFIG = ROOT / "configs/laria_v5_no_reference_8gpu_20260825.yaml"
V3_CONFIG = ROOT / "configs/laria_v5_no_reference_strict_8gpu_20260826_v3.yaml"
V2_CACHE = ROOT / "cache/laria_v5_no_reference_strict_8gpu_20260825.cache.json"
V3_CACHE = ROOT / "cache/laria_v5_no_reference_strict_8gpu_20260826_v3.cache.json"

EXPECTED_STEMS = 1055
MIN_OUTPUT = 842
V2_CACHE_ID = "cache/laria_v5_no_reference_strict_8gpu_20260825.cache.json"
V3_CACHE_ID = "cache/laria_v5_no_reference_strict_8gpu_20260826_v3.cache.json"
V3_CHECKPOINT_ID = (
    "cache/laria_v5_no_reference_strict_8gpu_20260826_v3.cache.checkpoint.json"
)
V2_OUTPUT_ID = "/mnt/Raw/0825/laria-v5-no-reference-8gpu-20260825-v2"
V2_WORKSPACE_ID = "/mnt/nvme3/mfa_workspace_laria_v5_0825"
V3_OUTPUT_ID = "/mnt/Raw/0825/laria-v5-no-reference-8gpu-20260826-v3"
V3_WORKSPACE_ID = "/mnt/nvme3/mfa_workspace_laria_v5_0826_v3"


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _frozen_stems() -> list[str]:
    """Read the existing v2 inventory without creating or mutating artifacts."""
    payload = json.loads(V2_CACHE.read_text(encoding="utf-8"))
    assert payload["schema"] == "streaming-dataset-cache-v1"
    assert payload["dataset_count"] == 1
    dataset = payload["datasets"][0]
    stems = dataset["stems"]
    inventory = dataset["source_inventory"]["files"]
    assert [row["stem"] for row in inventory] == stems
    return stems


def _policy_without_identity(config: dict) -> dict:
    clone = json.loads(json.dumps(config))
    clone.pop("output_dir", None)
    clone.pop("workspace", None)
    clone["streaming"].pop("batch_cache", None)
    clone["streaming"].pop("local_work", None)
    # v3 deliberately retains its run-bound CTC evidence after successful
    # local workspaces are cleaned; v2 did not.
    clone["pipelined"].pop("restore_ctc_cache", None)
    return clone


def _assert_complete_publication(evidence: dict, expected: set[str]) -> None:
    assert evidence.get("status") == "COMPLETE", "publication is not COMPLETE"
    output = set(evidence.get("output_stems", ()))
    filtered = set(evidence.get("filtered_stems", ()))
    assert not output & filtered, "output and filtered overlap"
    assert output | filtered == expected, "output/filtered is not exact expected union"

    shortfall = evidence.get("shortfall_evidence", {})
    assert isinstance(shortfall, dict)
    if len(output) >= MIN_OUTPUT:
        assert not shortfall, "shortfall evidence is only valid below the threshold"
        return

    missing = expected - output
    assert set(shortfall) == missing, "shortfall evidence is not per missing stem"
    for stem in missing:
        row = shortfall[stem]
        assert isinstance(row, dict), f"shortfall evidence missing for {stem}"
        assert row.get("hard_evidence") is True, (
            f"shortfall for {stem} lacks hard evidence"
        )


def test_v3_yaml_is_parseable_and_clones_v2_policy():
    v2 = _yaml(V2_CONFIG)
    v3 = _yaml(V3_CONFIG)
    assert _policy_without_identity(v3) == _policy_without_identity(v2)


def test_v3_config_passes_runtime_schema():
    from scripts import run_pipeline

    assert run_pipeline.validate_config(_yaml(V3_CONFIG), "nvrasr_fallback") == []


def test_v3_identity_isolated_from_v1_and_v2():
    v1 = _yaml(V1_CONFIG)
    v2 = _yaml(V2_CONFIG)
    v3 = _yaml(V3_CONFIG)
    assert v3["output_dir"] == V3_OUTPUT_ID
    assert v3["workspace"] == V3_WORKSPACE_ID
    assert v3["streaming"]["batch_cache"] == V3_CACHE_ID
    assert v3["streaming"]["batch_cache"] != V2_CACHE_ID
    assert v3["streaming"]["batch_cache"] != v1["streaming"]["batch_cache"]
    assert v3["output_dir"] != V2_OUTPUT_ID
    assert v3["output_dir"] != v1["output_dir"]
    assert v3["workspace"] != V2_WORKSPACE_ID
    assert v3["workspace"] != v1["workspace"]
    cache_path = Path(v3["streaming"]["batch_cache"])
    assert str(cache_path.with_name(cache_path.stem + ".checkpoint.json")) == (
        V3_CHECKPOINT_ID
    )
    assert V3_OUTPUT_ID not in {v2["output_dir"]}
    assert V3_WORKSPACE_ID not in {v2["workspace"]}
    assert all("0826_v3" in path for path in v3["streaming"]["local_work"])
    assert not any(
        path in set(v2["streaming"]["local_work"])
        for path in v3["streaming"]["local_work"]
    )


def test_frozen_inventory_is_exactly_1055_unique_stems():
    stems = _frozen_stems()
    assert stems == sorted(stems)
    assert len(stems) == EXPECTED_STEMS
    assert len(set(stems)) == EXPECTED_STEMS


def test_materialized_v3_cache_preserves_inventory_and_rebinds_output():
    source = json.loads(V2_CACHE.read_text(encoding="utf-8"))
    actual = json.loads(V3_CACHE.read_text(encoding="utf-8"))
    assert actual["source_count"] == EXPECTED_STEMS
    assert actual["datasets"][0]["stems"] == source["datasets"][0]["stems"]
    assert (actual["datasets"][0]["source_inventory"]
            == source["datasets"][0]["source_inventory"])
    assert actual["output_root"] == V3_OUTPUT_ID
    assert actual["datasets"][0]["ctc_dir"] == f"{V3_OUTPUT_ID}/LAria"
    assert "20260825-v2" not in V3_CACHE.read_text(encoding="utf-8")


def test_v3_keeps_eight_gpu_ctc_and_raw_evidence_policy():
    config = _yaml(V3_CONFIG)
    assert config["streaming"]["num_gpus"] == 8
    assert config["streaming"]["parallel"] == 8
    assert config["ctc_prealign"]["enabled"] is True
    assert config["ctc_prealign"]["nvv_enabled"] is True
    assert config["ctc_prealign"]["all_gpus"] is False
    assert config["pipelined"]["restore_ctc_cache"] is True


def test_complete_evidence_accepts_threshold_output_and_exact_accounting():
    expected = set(_frozen_stems())
    output = set(sorted(expected)[:MIN_OUTPUT])
    evidence = {
        "status": "COMPLETE",
        "output_stems": sorted(output),
        "filtered_stems": sorted(expected - output),
    }
    _assert_complete_publication(evidence, expected)


def test_complete_evidence_allows_only_hard_evidenced_shortfall():
    expected = set(_frozen_stems())
    output = set(sorted(expected)[: MIN_OUTPUT - 1])
    missing = expected - output
    evidence = {
        "status": "COMPLETE",
        "output_stems": sorted(output),
        "filtered_stems": sorted(missing),
        "shortfall_evidence": {
            stem: {"hard_evidence": True, "reason": "fixture"}
            for stem in sorted(missing)
        },
    }
    _assert_complete_publication(evidence, expected)


@pytest.mark.parametrize(
    "mutation",
    ["not_complete", "overlap", "missing_stem", "soft_shortfall"],
)
def test_complete_evidence_rejects_contract_breaks(mutation):
    expected = set(_frozen_stems())
    output = set(sorted(expected)[:MIN_OUTPUT])
    evidence = {
        "status": "COMPLETE",
        "output_stems": sorted(output),
        "filtered_stems": sorted(expected - output),
    }
    if mutation == "not_complete":
        evidence["status"] = "PARTIAL"
    elif mutation == "overlap":
        evidence["filtered_stems"] = sorted(expected - output) + [next(iter(output))]
    elif mutation == "missing_stem":
        evidence["filtered_stems"] = sorted(expected - output)[:-1]
    else:
        output = set(sorted(expected)[: MIN_OUTPUT - 1])
        evidence["output_stems"] = sorted(output)
        evidence["filtered_stems"] = sorted(expected - output)
        evidence["shortfall_evidence"] = {
            stem: {"hard_evidence": False} for stem in expected - output
        }
    with pytest.raises(AssertionError):
        _assert_complete_publication(evidence, expected)
