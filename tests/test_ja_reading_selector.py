from scripts.ja_asr_crossval import select_reading


def test_selector_precedence_manual_origin_consensus_medoid_none():
    analysis = {"canonical_sha256": "abc", "analysis_version": "v1"}
    manual = select_reading({"uid": "u", "reading_override": "さくら", "source_id": "src"}, [], analysis)
    assert manual["status"] == "manual_verified"
    origin = select_reading({"uid": "u", "source_id": "src", "origin_reading": "とうきょう", "origin_match": True}, [], analysis)
    assert origin["status"] == "origin_surface_confirmed"
    assert origin["shared_g2p_derivation"] is True
    assert origin["evidence_scope"] == "lexical_support"
    assert origin["acoustic_reading_proof"] is False
    consensus = select_reading({}, [
        {"family": "qwen", "provider": "qwen3-asr", "candidates": [{"reading": "さくら", "candidate_id": "c1"}]},
        {"family": "whisper", "provider": "whisper-large-v3", "candidates": [{"reading": "さくら", "candidate_id": "c2"}]},
    ], analysis)
    assert consensus["status"] == "asr_family_consensus"
    assert consensus["chosen_reading"] == "さくら"
    assert consensus["acoustic_reading_proof"] is True
    assert consensus["shared_g2p_derivation"] is False
    medoid = select_reading({}, [
        {"family": "qwen", "provider": "qwen3-asr", "candidates": [{"reading": "さくら"}]},
        {"family": "whisper", "provider": "whisper-large-v3", "candidates": [{"reading": "さく"}]},
    ], analysis)
    assert medoid["status"] == "asr_medoid"
    unresolved = select_reading({}, [], analysis)
    assert unresolved["status"] == "none"


def test_selector_requires_analysis_digest_and_does_not_vote_mfa():
    try:
        select_reading({}, [], {})
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("missing analysis digest must fail closed")


def test_shared_default_g2p_candidates_never_count_as_acoustic_consensus():
    analysis = {
        "canonical_sha256": "abc", "analysis_version": "v1", "analysis_digest": "analysis-1",
        "units": [{"token_id": "tok_000001", "candidate_id": "cand_000001",
                   "surface": "生", "read": "せい", "canonical_span": [0, 1],
                   "contextual_candidates": [{"candidate_id": "cand_000001", "reading": "せい",
                                                "provenance": "frontend_text_prediction"}]}],
    }
    results = [
        {"provider": "qwen3-asr", "family": "qwen", "status": "ok", "candidates": [{"token_id": "tok_000001", "reading": "せい", "is_same_surface_g2p": True}]},
        {"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "candidates": [{"token_id": "tok_000001", "reading": "せい", "is_same_surface_g2p": True}]},
    ]
    selected = select_reading({"uid": "u", "token_id": "tok_000001"}, results, analysis)
    assert selected["status"] == "asr_family_consensus"
    assert selected["chosen_reading"] == "せい"
    assert selected["shared_g2p_derivation"] is True
    assert selected["confidence"] == "lexical_family_consensus"
    assert selected["acoustic_reading_proof"] is False


def test_origin_requires_affirmative_lexical_support():
    analysis = {"canonical_sha256": "abc", "analysis_version": "v1", "contextual_reading": "さくら"}
    selected = select_reading({"uid": "u", "source_id": "s"}, [], analysis)
    assert selected["status"] == "none"


def test_unknown_or_non_asr_family_never_creates_a_vote():
    analysis = {"canonical_sha256": "abc", "analysis_version": "v1"}
    selected = select_reading({"uid": "u"}, [
        {"provider": "mfa", "family": "mfa", "status": "ok",
         "candidates": [{"reading": "さくら"}]},
        {"provider": "fixture-provider", "status": "ok",
         "candidates": [{"reading": "さくら"}]},
    ], analysis)
    assert selected["status"] == "none"
    assert selected["evidence"] == []


def test_per_token_overrides_keep_same_surface_occurrences_distinct(tmp_path):
    from scripts.ja_asr_crossval import reading_stage
    import json
    root = tmp_path / "run"; (root / "asr").mkdir(parents=True); (root / "frontend").mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "u", "text": "生生", "reading_overrides": {
        "tok_1": {"reading": "せい", "source_id": "gold-1"},
        "tok_2": {"reading": "なま", "source_id": "gold-2"}}}) + "\n", encoding="utf-8")
    analysis = {"schema": "ja-frontend-contract-v2", "uid": "u", "canonical_sha256": "a", "analysis_digest": "d", "profile": "p", "units": [
        {"token_id": "tok_1", "candidate_id": "cand_1", "surface": "生", "lexical_status": "lexical"},
        {"token_id": "tok_2", "candidate_id": "cand_2", "surface": "生", "lexical_status": "lexical"}]}
    (root / "frontend/frontend_analysis.json").write_text(json.dumps({"records": [analysis]}), encoding="utf-8")
    (root / "asr/asr_evidence.jsonl").write_text(json.dumps({"uid": "u", "providers": []}) + "\n", encoding="utf-8")
    result = reading_stage({"input_manifest": str(manifest)}, root / "reading")
    rows = [json.loads(line) for line in (root / "reading/locked_readings.jsonl").read_text().splitlines()]
    assert result.status == "COMPLETE"
    assert [(row["token_id"], row["chosen_reading"]) for row in rows] == [("tok_1", "せい"), ("tok_2", "なま")]


def test_english_candidate_consensus_binds_native_candidate_id(tmp_path):
    from scripts.ja_asr_crossval import reading_stage
    import json
    root = tmp_path / "run"; (root / "asr").mkdir(parents=True); (root / "frontend").mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "en", "text": "game"}) + "\n", encoding="utf-8")
    analysis = {"schema": "ja-frontend-contract-v2", "uid": "en", "canonical_sha256": "a", "analysis_digest": "d", "profile": "p", "units": [
        {"token_id": "tok_en", "candidate_id": "cand_en", "surface": "game", "language": "en", "lexical_status": "lexical"}]}
    (root / "frontend/frontend_analysis.json").write_text(json.dumps({"records": [analysis]}), encoding="utf-8")
    providers = [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "candidates": [{"token_id": "tok_en", "candidate_id": "cand_en", "reading": "game", "native_phones": ["G", "EY1", "M"]}]},
                 {"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "candidates": [{"token_id": "tok_en", "candidate_id": "cand_en", "reading": "game", "native_phones": ["G", "EY1", "M"]}]}]
    (root / "asr/asr_evidence.jsonl").write_text(json.dumps({"uid": "en", "providers": providers}) + "\n", encoding="utf-8")
    reading_stage({"input_manifest": str(manifest)}, root / "reading")
    row = json.loads((root / "reading/locked_readings.jsonl").read_text().strip())
    assert row["status"] == "asr_family_consensus"
    assert row["candidate_id"] == "cand_en"


def test_pure_english_single_dictionary_pronunciation_locks_without_asr_vote(tmp_path):
    from scripts.ja_asr_crossval import reading_stage
    import json
    root = tmp_path / "run"; (root / "asr").mkdir(parents=True); (root / "frontend").mkdir()
    manifest = tmp_path / "manifest.jsonl"; manifest.write_text(json.dumps({"uid": "en", "text": "game"}) + "\n")
    analysis = {"schema": "ja-frontend-contract-v2", "uid": "en", "canonical_sha256": "a", "analysis_digest": "d", "profile": "p", "units": [
        {"token_id": "tok_en", "candidate_id": "cand_en", "surface": "game", "language": "en", "lexical_status": "lexical",
         "contextual_candidates": [{"candidate_id": "cand_en", "reading": "game", "pronunciation": ["G", "EY1", "M"], "evidence_scope": "matching_arpa_dictionary"}]}]}
    (root / "frontend/frontend_analysis.json").write_text(json.dumps({"records": [analysis]}))
    (root / "asr/asr_evidence.jsonl").write_text(json.dumps({"uid": "en", "providers": []}) + "\n")
    reading_stage({"input_manifest": str(manifest)}, root / "reading")
    row = json.loads((root / "reading/locked_readings.jsonl").read_text().strip())
    assert row["status"] == "origin_surface_confirmed"
    assert row["candidate_id"] == "cand_en"
    assert row["native_phones"] == ["G", "EY1", "M"]
    assert row["chosen_pronunciation"] == ["G", "EY1", "M"]
    assert row["selection_basis"] == "dictionary_policy"
    assert row["evidence_scope"] == "dictionary_policy"
    assert row["acoustic_reading_proof"] is False


def test_stage_projects_exact_surface_asr_to_each_token_without_manual_candidates(tmp_path):
    from scripts.ja_asr_crossval import reading_stage
    import json
    root = tmp_path / "run"; (root / "asr").mkdir(parents=True); (root / "frontend").mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "ja", "text": "生生"}) + "\n", encoding="utf-8")
    analysis = {"schema": "ja-frontend-contract-v2", "uid": "ja", "canonical_text": "生生", "orig_text": "生生",
                "canonical_sha256": "a", "analysis_digest": "d", "profile": "p", "units": [
                    {"token_id": "tok_1", "candidate_id": "cand_1", "surface": "生", "read": "せい", "lexical_status": "lexical"},
                    {"token_id": "tok_2", "candidate_id": "cand_2", "surface": "生", "read": "なま", "lexical_status": "lexical"}]}
    (root / "frontend/frontend_analysis.json").write_text(json.dumps({"records": [analysis]}), encoding="utf-8")
    (root / "asr/asr_evidence.jsonl").write_text(json.dumps({"uid": "ja", "providers": [
        {"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "生生"}]}) + "\n", encoding="utf-8")
    result = reading_stage({"input_manifest": str(manifest)}, root / "reading")
    rows = [json.loads(line) for line in (root / "reading/locked_readings.jsonl").read_text().splitlines()]
    assert result.status == "COMPLETE"
    assert [(row["token_id"], row["status"], row["chosen_reading"]) for row in rows] == [
        ("tok_1", "origin_surface_confirmed", "せい"), ("tok_2", "origin_surface_confirmed", "なま")]
    for row in rows:
        assert row["shared_g2p_derivation"] is True
        assert row["evidence_scope"] == "lexical_support"
        assert row["acoustic_reading_proof"] is False
        assert row["evidence"][0]["source"] == "origin_lexical"
        assert row["evidence"][0]["provider"] == "qwen3-asr"


def test_stage_loads_occurrence_scoped_manual_overrides_and_rejects_conflicts(tmp_path):
    from scripts.ja_asr_crossval import reading_stage
    import json
    root = tmp_path / "run"; (root / "asr").mkdir(parents=True); (root / "frontend").mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "manual", "text": "生生", "reading_overrides": {
        "tok_2": {"reading": "なま", "source_id": "inline"}}}) + "\n", encoding="utf-8")
    overrides = tmp_path / "manual.json"
    overrides.write_text(json.dumps({"overrides": [
        {"uid": "manual", "token_id": "tok_1", "reading": "せい", "source_id": "gold-1"},
        {"uid": "manual", "token_id": "tok_2", "reading": "なま", "source_id": "gold-2"},
    ]}), encoding="utf-8")
    analysis = {"schema": "ja-frontend-contract-v2", "uid": "manual", "canonical_sha256": "a",
                "analysis_digest": "d", "profile": "p", "units": [
        {"token_id": "tok_1", "candidate_id": "cand_1", "surface": "生", "canonical_span": [0, 1], "lexical_status": "lexical"},
        {"token_id": "tok_2", "candidate_id": "cand_2", "surface": "生", "canonical_span": [1, 2], "lexical_status": "lexical"}]}
    (root / "frontend/frontend_analysis.json").write_text(json.dumps({"records": [analysis]}), encoding="utf-8")
    (root / "asr/asr_evidence.jsonl").write_text(json.dumps({"uid": "manual", "providers": []}) + "\n", encoding="utf-8")
    result = reading_stage({"input_manifest": str(manifest), "manual_overrides": str(overrides)}, root / "reading")
    rows = [json.loads(line) for line in (root / "reading/locked_readings.jsonl").read_text().splitlines()]
    assert result.status == "PARTIAL"
    assert rows[0]["status"] == "manual_verified"
    assert rows[0]["source_id"] == "gold-1"
    assert rows[1]["status"] == "blocked"
    assert rows[1]["error"]["code"] == "manual_override_conflict"


def test_stage_excludes_w2_morph_punctuation_from_locked_reading_set(tmp_path):
    from scripts.ja_asr_crossval import reading_stage
    import json
    root = tmp_path / "run"; (root / "asr").mkdir(parents=True); (root / "frontend").mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "punct", "text": "桜。"}) + "\n", encoding="utf-8")
    analysis = {"schema": "ja-frontend-contract-v2", "uid": "punct", "canonical_text": "桜。", "orig_text": "桜。", "canonical_sha256": "a",
                "analysis_digest": "d", "profile": "p", "units": [
        {"token_id": "tok_word", "candidate_id": "cand_word", "surface": "桜", "read": "さくら",
         "lexical_status": "lexical", "morph_punct": False},
        {"token_id": "tok_punct", "candidate_id": "cand_punct", "surface": "。", "read": "",
         "lexical_status": "lexical", "morph_punct": True}]}
    (root / "frontend/frontend_analysis.json").write_text(json.dumps({"records": [analysis]}), encoding="utf-8")
    (root / "asr/asr_evidence.jsonl").write_text(json.dumps({"uid": "punct", "providers": [
        {"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "桜。"}]}) + "\n", encoding="utf-8")
    result = reading_stage({"input_manifest": str(manifest)}, root / "reading")
    rows = [json.loads(line) for line in (root / "reading/locked_readings.jsonl").read_text().splitlines()]
    assert result.status == "COMPLETE"
    assert [row["token_id"] for row in rows] == ["tok_word"]
