from scripts.ja_canary_gate import evaluate_canary_gate


def test_missing_gold_blocks_and_all_rejected_does_not_pass():
    blocked = evaluate_canary_gate([], gold=None, target_id="dev-1")
    assert blocked["status"] == "BLOCKED"
    rows = [{"id": f"row-{i}", "accepted": False, "bucket": ("ja_to_en", "en_to_ja", "no_pause", "short_english")[i // 10]} for i in range(40)]
    gold = {"target_id": "gate-1", "rows": [{"id": row["id"], "bucket": row["bucket"], "gold_seam_sample": 0} for row in rows]}
    rejected = evaluate_canary_gate(rows, gold=gold, target_id="gate-1", proof=True)
    assert rejected["status"] == "REJECTED"


def test_gate_requires_proof_target_and_bucket_minimums():
    rows = []
    for bucket in ("ja_to_en", "en_to_ja", "no_pause", "short_english"):
        rows.extend({"id": f"{bucket}-{i}", "accepted": True, "bucket": bucket, "route_error": False, "clipping": False, "overlap": False, "predicted_seam_sample": 320} for i in range(10))
    gold = {"target_id": "gate-1", "sample_rate": 16000, "rows": [{"id": row["id"], "bucket": row["bucket"], "gold_seam_sample": 0} for row in rows]}
    report = evaluate_canary_gate(rows, gold=gold, target_id="gate-1", proof=True)
    assert report["status"] == "PASS"


def test_nan_and_gold_target_mismatch_block_independent_calculation():
    rows = [{"id": "a", "accepted": True, "bucket": "ja_to_en", "predicted_seam_sample": float("nan")}]
    gold = {"target_id": "gate-a", "rows": [{"id": "a", "gold_seam_sample": 0}]}
    report = evaluate_canary_gate(rows, gold=gold, target_id="wrong", proof=True)
    assert report["status"] == "BLOCKED"
    assert "gold_target_mismatch" in report["reasons"]
