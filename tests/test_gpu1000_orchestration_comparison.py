from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_gpu1000_run as analyze


def _manifest(root: Path, stems: list[str]) -> None:
    samples = [{"speaker": "s", "stem": stem, "source_relative_wav": f"{stem}.wav",
                "source_relative_txt": f"{stem}.txt", "wav_sha256": stem + "w",
                "txt_sha256": stem + "t"} for stem in stems]
    identity = [{key: row[key] for key in ("speaker", "stem", "source_relative_wav",
                                            "source_relative_txt", "wav_sha256", "txt_sha256")}
                for row in samples]
    root.mkdir(parents=True, exist_ok=True)
    (root / "selected_manifest.json").write_text(json.dumps({
        "count": len(stems), "run_label": "full1000", "samples": samples,
        "selection_digest": analyze._digest(identity)}), encoding="utf-8")


def _run(root: Path, accepted: set[str], filtered: set[str], stems: list[str]) -> None:
    _manifest(root, stems)
    (root / "output").mkdir(); (root / "filtered").mkdir()
    for stem in accepted:
        (root / "output" / f"{stem}.TextGrid").write_text("x")
    for stem in filtered:
        (root / "filtered" / f"{stem}.TextGrid").write_text("x")
    shards = [{"gpu": gpu, "stems": stems[gpu * 125:(gpu + 1) * 125]}
              for gpu in range(8)]
    (root / "shard_plan.json").write_text(json.dumps({"shards": shards}), encoding="utf-8")


def test_comparison_blocks_old_accepted_missing_and_new_filtered(tmp_path):
    stems = [f"s{i}" for i in range(1000)]
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    _run(old_root, set(stems[:776]), set(stems[776:]), stems)
    _run(new_root, set(stems[:775]), set(stems[775:]), stems)
    report = analyze.compare_acceptance_runs(old_root, new_root)
    assert not report["ok"]
    assert any(error.startswith("old_accepted_missing:") for error in report["errors"])
    assert any(error.startswith("new_filtered:") for error in report["errors"])


def test_comparison_blocks_identity_and_shard_drift(tmp_path):
    stems = [f"s{i}" for i in range(1000)]
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    _run(old_root, set(stems[:776]), set(stems[776:]), stems)
    _run(new_root, set(stems), set(), stems)
    manifest = json.loads((new_root / "selected_manifest.json").read_text())
    manifest["samples"][0]["wav_sha256"] = "tampered"
    (new_root / "selected_manifest.json").write_text(json.dumps(manifest))
    plan = json.loads((new_root / "shard_plan.json").read_text())
    plan["shards"][0]["stems"] = plan["shards"][0]["stems"][:-1]
    (new_root / "shard_plan.json").write_text(json.dumps(plan))
    report = analyze.compare_acceptance_runs(old_root, new_root)
    assert not report["ok"]
    assert "source_identity_drift" in report["errors"]
    assert "future_shards_not_exact_125" in report["errors"]
