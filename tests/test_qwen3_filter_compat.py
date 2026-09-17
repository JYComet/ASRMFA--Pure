"""Regression checks for Qwen transcripts passed through the shared filters."""

from scripts import postprocess_textgrids as post
import copy


def words(*labels):
    return post.Tier("words", 0.0, float(len(labels)), [
        post.Interval(float(i), float(i + 1), label)
        for i, label in enumerate(labels)])


def test_contextual_polyphones_project_back_to_original_hanzi():
    tier = words("wo3", "liao3", "jie3", "le5")
    alignment = post._fallback_cjk_alignment("我了解了。", tier)
    assert alignment["safe"] is True
    assert alignment["actual_to_source"] == {0: 0, 1: 1, 2: 2, 3: 3}
    assert [iv.text for iv in post._build_hanzi_tier(
        tier, "我了解了。").intervals] == ["我", "了", "解", "了"]


def test_wrong_syllable_still_cannot_consume_source_character():
    alignment = post._fallback_cjk_alignment("我了解了。", words(
        "wo3", "ma3", "jie3", "le5"))
    assert alignment["safe"] is False
    assert 1 not in alignment["actual_to_source"]


def test_contextual_mapping_retains_legacy_character_reading_compatibility():
    assert post._fallback_cjk_alignment(
        "了解", words("le5", "jie3"))["safe"] is True


def test_context_does_not_join_phrases_across_punctuation():
    alignment = post._fallback_cjk_alignment("重，复。", words("zhong4", "fu4"))
    assert alignment["safe"] is True
    assert alignment["actual_to_source"] == {0: 0, 1: 1}


def qwen_tokens():
    return [{"word": word, "unit": unit, "provider": "qwen3_hf",
             "lexical_timing_source": "qwen3_forced_aligner_hf",
             "start_s": float(i), "end_s": float(i + 1)}
            for i, (word, unit) in enumerate((("ni3", "你"), ("hao3", "好")))]


def _qwen_punct_entry(label, left, right):
    return {
        "schema": post.PUNCTUATION_EVIDENCE_SCHEMA,
        "word": label,
        "start_s": 1.0,
        "end_s": 1.1,
        "left_lexical_ordinal": left,
        "right_lexical_ordinal": right,
    }


def test_qwen_projection_inserts_anchored_final_mark_and_preserves_surface_spelling():
    tokens = qwen_tokens()
    projection = post._project_qwen_final_punctuation(
        "open-ai 你好",
        words("openai", "ni3", "，", "hao3"),
        [{**tokens[0], "unit": "openai", "word": "openai"}, *tokens],
        [_qwen_punct_entry("，", 1, 2)],
    )
    assert projection["status"] == "verified"
    assert projection["projected_text"] == "open-ai 你，好"
    assert projection["removed_source_punctuation"] == []


def test_qwen_projection_occurrence_aware_partial_deletion_is_ledgered():
    projection = post._project_qwen_final_punctuation(
        "你，！好", words("ni3", "！", "hao3"), qwen_tokens())
    assert projection["status"] == "verified"
    assert projection["projected_text"] == "你！好"
    assert projection["removed_source_punctuation"] == [{
        "label": "，", "source_boundary": 1,
        "reason": "absent_from_frozen_words",
    }]


def test_qwen_projection_rejects_ambiguous_lexical_mapping_and_keeps_source():
    tokens = qwen_tokens()
    tokens[1]["unit"] = "他"
    projection = post._project_qwen_final_punctuation(
        "你好，", words("ni3", "hao3"), tokens)
    assert projection["status"] == "rejected"
    assert projection["projected_text"] == "你好，"
    assert projection["removed_source_punctuation"] == []


def test_qwen_publication_rejection_is_reported_and_keeps_original_surface():
    tier = words("ni3", "hao3")
    raw = post.Tier("raw_text", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    py = post.Tier("pinyin", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    grid = post.TextGrid(0.0, 2.0, [raw, py, post.Tier("hanzi", 0.0, 2.0, []), tier])
    tokens = qwen_tokens()
    tokens[1]["unit"] = "他"
    post._freeze_processed_geometry(grid)
    post._commit_derived_barriers(post._make_derived_barrier_state(
        textgrid=grid, raw_text="你好，", reference_authoritative=True,
        pinyin_text="ni3 hao3", ctc_tokens=tokens))
    assert raw.intervals[0].text == "<sp1>你好，"
    transaction = grid._derived_publication_transaction
    assert transaction["qwen_projection"]["status"] == "rejected"
    reasons, _ = post._publication_contract_audit(
        tier, post.tier_by_name(grid, "hanzi"), None, None, "你好，", None,
        tokens, True, raw_text_tier=raw, pinyin_tier=py)
    assert "qwen_publication_projection_invalid" in reasons


def test_qwen_fallback_removed_mark_keeps_contextual_hanzi_mapping():
    tier = words("zhong4", "fu4")
    raw = post.Tier("raw_text", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    py = post.Tier("pinyin", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    grid = post.TextGrid(0.0, 2.0, [raw, py, post.Tier("hanzi", 0.0, 2.0, []), tier])
    tokens = [
        {"word": "zhong4", "unit": "重", "provider": "qwen3_hf",
         "lexical_timing_source": "qwen3_forced_aligner_hf",
         "start_s": 0.0, "end_s": 1.0},
        {"word": "fu4", "unit": "复", "provider": "qwen3_hf",
         "lexical_timing_source": "qwen3_forced_aligner_hf",
         "start_s": 1.0, "end_s": 2.0},
    ]
    ledger = post._fallback_punctuation_surface_ledger("重，复")
    post._freeze_processed_geometry(grid)
    post._commit_derived_barriers(post._make_derived_barrier_state(
        textgrid=grid, raw_text="重，复", fallback_surface_ledger=ledger,
        ctc_tokens=tokens))
    assert raw.intervals[0].text == "<sp1>重复"
    assert py.intervals[0].text == "<sp1> zhong4 fu4"
    assert [iv.text for iv in post.tier_by_name(grid, "hanzi").intervals] == [
        "重", "复"]


def test_zero_gap_punctuation_is_a_text_boundary_without_carving_words():
    tier = words("ni3", "hao3")
    before = copy.deepcopy(tier)
    ledger = post._fallback_punctuation_surface_ledger("你，好")
    points = post._qwen_punctuation_boundary_points(ledger, tier, qwen_tokens())
    assert [(p["label"], p["boundary"], p["time_s"]) for p in points] == [
        ("，", 1, 1.0)]
    assert tier == before
    assert post._render_pinyin_with_boundary_points(tier, points) == "<sp1> ni3 ， hao3"


def test_boundary_point_cannot_excuse_a_missing_positive_gap_owner():
    tokens = qwen_tokens()
    tokens[1]["start_s"] = 1.1
    assert post._qwen_punctuation_boundary_points(
        post._fallback_punctuation_surface_ledger("你，好"),
        words("ni3", "hao3"), tokens) == []


def test_boundary_point_requires_exact_qwen_source_and_final_lexical_sequence():
    ledger = post._fallback_punctuation_surface_ledger("你，好")
    for key, value in (("unit", "他"), ("provider", "nvasr")):
        tokens = qwen_tokens()
        tokens[0][key] = value
        assert post._qwen_punctuation_boundary_points(
            ledger, words("ni3", "hao3"), tokens) == []
    assert post._qwen_punctuation_boundary_points(
        ledger, words("hao3", "ni3"), qwen_tokens()) == []


def test_publication_rebuild_and_audit_drop_qwen_source_only_zero_gap_mark():
    tier = words("ni3", "hao3")
    raw = post.Tier("raw_text", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    py = post.Tier("pinyin", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    grid = post.TextGrid(0.0, 2.0, [raw, py, post.Tier("hanzi", 0.0, 2.0, []), tier])
    ledger = post._fallback_punctuation_surface_ledger("你，好")
    post._freeze_processed_geometry(grid)
    state = post._make_derived_barrier_state(
        textgrid=grid, raw_text="你，好", fallback_surface_ledger=ledger,
        ctc_tokens=qwen_tokens())
    post._commit_derived_barriers(state)
    assert raw.intervals[0].text == "<sp1>你好"
    assert py.intervals[0].text == "<sp1> ni3 hao3"
    reasons, details = post._publication_contract_audit(
        tier, post.tier_by_name(grid, "hanzi"), None, None, "你，好",
        None, qwen_tokens(), False, reference_mode="fallback",
        raw_text_tier=raw, pinyin_tier=py, fallback_surface_ledger=ledger)
    assert "fallback_punctuation_ownership_mismatch" not in reasons
    assert "pinyin_punctuation_sequence_mismatch" not in reasons
    assert "qwen_punctuation_boundary_points" not in details
    transaction = grid._derived_publication_transaction
    assert transaction["removed_source_punctuation_count"] == 1
    assert transaction["removed_source_punctuation"] == [{
        "label": "，", "source_boundary": 1,
        "reason": "absent_from_frozen_words",
    }]
    # A caller cannot turn a real lost interval owner into an accepted point.
    tokens = qwen_tokens()
    tokens[1]["start_s"] = 1.1
    reasons, _ = post._publication_contract_audit(
        tier, post.tier_by_name(grid, "hanzi"), None, None, "你，好",
        None, tokens, False, reference_mode="fallback",
        raw_text_tier=raw, pinyin_tier=py, fallback_surface_ledger=ledger)
    assert "fallback_punctuation_ownership_mismatch" in reasons


def test_qwen_reference_publication_drops_source_only_mark_without_geometry_change():
    tier = words("ni3", "hao3")
    raw = post.Tier("raw_text", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    py = post.Tier("pinyin", 0.0, 2.0, [post.Interval(0.0, 2.0, "stale")])
    grid = post.TextGrid(0.0, 2.0, [raw, py, post.Tier("hanzi", 0.0, 2.0, []), tier])
    before = [(iv.xmin, iv.xmax, iv.text) for iv in tier.intervals]
    post._freeze_processed_geometry(grid)
    post._rebuild_derived_from_frozen_words(
        grid, {}, {}, "你好，", reference_authoritative=True,
        pinyin_text="ni3 ， hao3", ctc_tokens=qwen_tokens())
    assert raw.intervals[0].text == "<sp1>你好"
    assert py.intervals[0].text == "<sp1> ni3 hao3"
    assert [(iv.xmin, iv.xmax, iv.text) for iv in tier.intervals] == before
    assert [iv.text for iv in post.tier_by_name(grid, "hanzi").intervals] == [
        "你", "好"]


def test_consecutive_source_marks_share_one_real_pause_owner():
    tier = post.Tier("words", 0.0, 3.0, [
        post.Interval(0.0, 1.0, "ni3"), post.Interval(1.0, 2.0, "hao3"),
        post.Interval(2.0, 3.0, "<sp2>")])
    ledger = post._fallback_punctuation_surface_ledger("你好……")
    injected, _ = post._inject_fallback_punctuation_gaps(
        tier, None, [], source_surface_ledger=ledger, ctc_tokens=qwen_tokens())
    assert [(iv.xmin, iv.xmax, iv.text) for iv in injected.intervals] == [
        (0.0, 1.0, "ni3"), (1.0, 2.0, "hao3"), (2.0, 3.0, "……")]
    hanzi = post._build_hanzi_tier(injected, "你好……")
    reasons, _ = post._publication_contract_audit(
        injected, hanzi, None, None, "你好……", None, qwen_tokens(), False,
        reference_mode="fallback", fallback_surface_ledger=ledger)
    assert "fallback_punctuation_ownership_mismatch" not in reasons
    projection = post._fallback_punctuation_projection("你好……", injected, qwen_tokens())
    assert projection["safe"] is True


def test_pinyin_reprojection_preserves_sentence_boundaries_and_model_times():
    from scripts.qwen3_prealign import reproject_pinyin_rows
    rows = [{"unit": u, "word": w, "provider": "qwen3_hf",
             "start_s": i * 0.2, "end_s": i * 0.2 + 0.1}
            for i, (u, w) in enumerate(zip("完成都认真", [
                "wan2", "cheng2", "du1", "ren4", "zhen1"]))]
    original = copy.deepcopy(rows)
    result = reproject_pinyin_rows(rows, "完成。都认真！")
    assert [r["word"] for r in result] == ["wan2", "cheng2", "dou1", "ren4", "zhen1"]
    assert rows == original
    assert [(r["start_s"], r["end_s"]) for r in result] == [
        (r["start_s"], r["end_s"]) for r in original]
    assert result[2]["pinyin_projection"]["original_word"] == "du1"
