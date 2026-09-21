import json
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from scripts.ja_en_schema import JAContractError, stable_digest
from scripts.ja_asr_crossval import select_reading
from scripts.ja_frontend import (
    DEFAULT_FRONTEND_OPTIONS,
    FrontendConfig,
    extract_contextual_accent_evidence,
    frontend_stage,
    probe_provider,
    reconstruct_locked_reading,
    run_frontend,
    bind_asr_candidates,
    project_blind_asr_candidates,
    unit_candidate_analysis,
)
from scripts.ja_text_layers import canonicalize_text


def _config():
    runtime = os.environ.get("JA_FRONTEND_PYTHON")
    if not runtime:
        if importlib.util.find_spec("pyopenjtalk") is None:
            pytest.skip("set JA_FRONTEND_PYTHON for the explicit real frontend bridge")
        runtime = sys.executable
    return {
        "provider": "pyopenjtalk-plus",
        "commit": "9e4bf25324ac135dfc81ca64aed2fa6a48b83304",
        "python": runtime,
        "normalize_mode": "NFKC",
        "use_vanilla": False,
        "use_sudachi_kanji_yomi": False,
        "predict_nani": False,
        "use_tsqyomi": False,
        "use_read_as_pron": False,
        "revert_long_vowels": False,
        "revert_yotsugana": False,
        "run_marine": False,
        "reject_unbound_spans": True,
    }


def _portable_config():
    return {
        "provider": "pyopenjtalk-plus", "commit": "9e4bf25324ac135dfc81ca64aed2fa6a48b83304",
        "normalize_mode": "NFKC", "use_vanilla": False, "use_sudachi_kanji_yomi": False,
        "predict_nani": False, "use_tsqyomi": False, "use_read_as_pron": False,
        "revert_long_vowels": False, "revert_yotsugana": False, "run_marine": False,
        "reject_unbound_spans": True,
    }


# These are literal full-context label rows captured from the pinned
# pyopenjtalk-plus runtime.  One representative label per mora is enough to
# exercise the public binding boundary without regenerating fixtures in-test.
_ACCENT_CASES = {
    "flat": (
        [{"string": "さくら", "read": "サクラ", "acc": 0, "mora_size": 3, "chain_flag": -1}],
        [
            "xx^sil-s+a=k/A:-2+1+3/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:3_3#0_0@1_1|1_3/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-3@1+1&1-1|1+3/J:xx_xx/K:1+1-3",
            "s^a-k+u=r/A:-1+2+2/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:3_3#0_0@1_1|1_3/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-3@1+1&1-1|1+3/J:xx_xx/K:1+1-3",
            "k^u-r+a=sil/A:0+3+1/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:3_3#0_0@1_1|1_3/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-3@1+1&1-1|1+3/J:xx_xx/K:1+1-3",
        ],
    ),
    "head": (
        [{"string": "みかん", "read": "ミカン", "acc": 1, "mora_size": 3, "chain_flag": -1}],
        [
            "xx^sil-m+i=k/A:0+1+3/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:3_1#0_0@1_1|1_3/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-3@1+1&1-1|1+3/J:xx_xx/K:1+1-3",
            "m^i-k+a=N/A:1+2+2/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:3_1#0_0@1_1|1_3/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-3@1+1&1-1|1+3/J:xx_xx/K:1+1-3",
            "k^a-N+sil=xx/A:2+3+1/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:3_1#0_0@1_1|1_3/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-3@1+1&1-1|1+3/J:xx_xx/K:1+1-3",
        ],
    ),
    "middle": (
        [{"string": "たべもの", "read": "タベモノ", "acc": 2, "mora_size": 4, "chain_flag": -1}],
        [
            "xx^sil-t+a=b/A:-1+1+4/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_2#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
            "t^a-b+e=m/A:0+2+3/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_2#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
            "b^e-m+o=n/A:1+3+2/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_2#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
            "m^o-n+o=sil/A:2+4+1/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_2#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
        ],
    ),
    "tail": (
        [{"string": "かみなり", "read": "カミナリ", "acc": 3, "mora_size": 4, "chain_flag": -1}],
        [
            "xx^sil-k+a=m/A:-2+1+4/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_3#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
            "k^a-m+i=n/A:-1+2+3/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_3#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
            "m^i-n+a=r/A:0+3+2/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_3#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
            "n^a-r+i=sil/A:1+4+1/B:xx-xx_xx/C:02_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_3#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
        ],
    ),
}


@pytest.mark.parametrize(("case", "expected"), [
    ("flat", ["L", "H", "H"]), ("head", ["H", "L", "L"]),
    ("middle", ["L", "H", "L", "L"]), ("tail", ["L", "H", "H", "L"]),
])
def test_literal_phrase_tone_fixture_shape(case, expected):
    rows, labels = _ACCENT_CASES[case]
    evidence = extract_contextual_accent_evidence(rows, labels)
    assert evidence["accent_phrases"][0]["expected_tones"] == expected


def test_literal_multiword_phrase_keeps_one_phrase_id_and_evidence_digest():
    rows = [
        {"string": "さくら", "read": "サクラ", "acc": 0, "mora_size": 3, "chain_flag": -1},
        {"string": "が", "read": "ガ", "acc": 0, "mora_size": 1, "chain_flag": 1},
    ]
    labels = [
        "xx^sil-s+a=k/A:-3+1+4/B:xx-xx_xx/C:02_xx+xx/D:13+xx_xx/E:xx_xx!xx_xx-xx/F:4_4#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
        "s^a-k+u=r/A:-2+2+3/B:xx-xx_xx/C:02_xx+xx/D:13+xx_xx/E:xx_xx!xx_xx-xx/F:4_4#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
        "k^u-r+a=g/A:-1+3+2/B:xx-xx_xx/C:02_xx+xx/D:13+xx_xx/E:xx_xx!xx_xx-xx/F:4_4#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
        "r^a-g+a=sil/A:0+4+1/B:02-xx_xx/C:13_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:4_4#0_0@1_1|1_4/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-4@1+1&1-1|1+4/J:xx_xx/K:1+1-4",
    ]
    result = extract_contextual_accent_evidence(rows, labels)
    assert [row["accent_phrase_id"] for row in result["moras"]] == ["ap0", "ap0", "ap0", "ap0"]
    assert result["provider_evidence_sha256"] == stable_digest({"njd_rows": rows, "full_context_labels": labels})


def test_label_cardinality_mismatch_is_rejected_without_guessing():
    rows, labels = _ACCENT_CASES["flat"]
    with pytest.raises(JAContractError) as exc:
        extract_contextual_accent_evidence(rows, labels[:-1])
    assert exc.value.code == "accent_phrase_unresolved"


def test_canonical_layer_preserves_original_and_maps_nfkc_offsets():
    layer = canonicalize_text("ﾊﾞ Ａ", "NFKC")
    assert layer["orig_text"] == "ﾊﾞ Ａ"
    assert layer["canonical_text"] == "バ A"
    assert layer["orig_to_canonical"]
    assert layer["canonical_to_orig"]
    assert layer["canonical_sha256"]


def test_nonempty_zero_span_mapping_is_rejected_instead_of_filtered():
    with pytest.raises(JAContractError) as exc:
        run_frontend("東京", _config(), provider=lambda text, **options: [{"char_span": [0, 0], "surface": "東京", "phonemes": ["t"], "read": "トウキョウ", "pron": "トーキョー", "mora_count": 4}])
    assert exc.value.code == "frontend_span_unbound"


def test_mapping_file_is_manual_route_policy_and_enters_identity(tmp_path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"entries": {"AI": {"language": "en", "pronunciation": [["AY1"]]}}}), encoding="utf-8")
    config = {"mapping_file": str(mapping), **_portable_config()}
    analysis = run_frontend("AI", config, provider=lambda text, **options: [{"char_span": [0, 2], "surface": "AI", "phonemes": ["e", "i"], "read": "エイ", "pron": "エイ", "mora_count": 2}])
    unit = analysis["units"][0]
    assert unit["language"] == "en"
    assert unit["route"] == "manual_lexicon"
    assert analysis["frontend_identity"]["mapping_file_sha256"]


def test_ruby_and_placeholder_markup_is_explicitly_unresolved():
    config = _portable_config()
    analysis = run_frontend("<ruby>今日</ruby>", config, provider=lambda text, **options: [{"char_span": [0, len(text)], "surface": text, "phonemes": ["k"], "read": "キョウ", "pron": "キョウ", "mora_count": 1}])
    assert analysis["units"][0]["route"] == "markup_unsupported"
    assert analysis["units"][0]["language"] == "unresolved"


def test_morph_punctuation_keeps_span_without_entering_asr_lexical_stream():
    analysis = run_frontend("桜が咲きました。 people", _config())
    punctuation = next(unit for unit in analysis["units"] if unit["surface"] == "。")
    assert punctuation["morph_punct"] is True
    assert punctuation["lexical_status"] == "unresolved"
    assert punctuation["canonical_span"] == [7, 8]
    bound = bind_asr_candidates(analysis, [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "さくらがさきました people"}])
    assert bound["asr_bindings"][0]["status"] == "bound"


def test_english_reconstruction_accepts_w1_native_phones_lock(tmp_path):
    dictionary = tmp_path / "en.dict"
    dictionary.write_text("people HH AH0 L OW1\n", encoding="utf-8")
    config = {**_portable_config(), "english_dictionary": str(dictionary)}
    analysis = run_frontend("people", config, provider=lambda text, **options: [{"char_span": [0, 6], "surface": "people", "phonemes": ["p", "i"], "read": "ピープル", "pron": "ピープル", "mora_count": 2}])
    unit = analysis["units"][0]
    request = {"canonical_sha256": analysis["canonical_sha256"], "analysis_digest": analysis["analysis_digest"], "analysis_version": analysis["analysis_version"], "frontend_profile": analysis["frontend_profile"], "frontend_options_digest": analysis["frontend_options_digest"], "locked_readings": [{"token_id": unit["token_id"], "candidate_id": unit["candidate_id"], "chosen_reading": "people", "native_phones": ["HH", "AH0", "L", "OW1"]}]}
    reconstructed = reconstruct_locked_reading(analysis, request, config)
    assert reconstructed["units"][0]["locked_native_phones"] == ["HH", "AH0", "L", "OW1"]


def test_real_frontend_bridge_returns_caller_bound_candidates():
    result = run_frontend("東京でgameをする", _config())
    assert result["schema"] == "ja-frontend-contract-v2"
    assert result["options"] == {**DEFAULT_FRONTEND_OPTIONS, "normalize_mode": "NFKC", "use_vanilla": False, "use_sudachi_kanji_yomi": False, "predict_nani": False}
    assert result["canonical_text"] == "東京でgameをする"
    assert result["units"]
    assert all(unit["orig_span"][1] > unit["orig_span"][0] for unit in result["units"] if unit["lexical_status"] == "lexical")
    tokyo = next(unit for unit in result["units"] if unit["surface"] == "東京")
    assert tokyo["orig_span"] == [0, 2]
    assert tokyo["canonical_span"] == [0, 2]
    assert tokyo["candidate_ids"] == [tokyo["candidate_id"]]


def test_pinned_frontend_accent_evidence_digest_is_deterministic():
    first = run_frontend("さくらが", _config())
    second = run_frontend("さくらが", _config())
    assert first["contextual_accent_evidence"]["provider_evidence_sha256"] == second["contextual_accent_evidence"]["provider_evidence_sha256"]
    assert first["units"][0]["accent_evidence_valid"] is True
    assert first["units"][0]["accent_evidence"]["provider_evidence_sha256"] == first["contextual_accent_evidence"]["provider_evidence_sha256"]


def test_locked_reading_requires_exact_digest_and_candidate_id():
    analysis = run_frontend("東京", _config())
    tokyo = analysis["units"][0]
    request = {
        "canonical_sha256": analysis["canonical_sha256"],
        "analysis_digest": analysis["analysis_digest"],
        "analysis_version": analysis["analysis_version"],
        "frontend_profile": analysis["frontend_profile"],
        "frontend_options_digest": analysis["frontend_options_digest"],
        "locked_readings": [{"token_id": tokyo["token_id"], "candidate_id": tokyo["candidate_id"], "chosen_reading": "とうきょう"}],
    }
    reconstructed = reconstruct_locked_reading(analysis, request, _config())
    assert reconstructed["schema"] == "ja-frontend-contract-v2"
    assert reconstructed["units"][0]["locked_reading"] == "とうきょう"
    assert reconstructed["units"][0]["accent_provenance"] == "invalidated_by_locked_reading"
    bad = dict(request, canonical_sha256="bad")
    with pytest.raises(JAContractError) as exc:
        reconstruct_locked_reading(analysis, bad, _config())
    assert exc.value.code == "frontend_representation_drift"


def test_locked_reading_rejects_missing_and_duplicate_lexical_units():
    analysis = run_frontend("東京で", _config())
    tokyo, de = [unit for unit in analysis["units"] if unit["lexical_status"] == "lexical"]
    base = {"canonical_sha256": analysis["canonical_sha256"], "analysis_digest": analysis["analysis_digest"], "analysis_version": analysis["analysis_version"], "frontend_profile": analysis["frontend_profile"], "frontend_options_digest": analysis["frontend_options_digest"]}
    with pytest.raises(JAContractError) as missing:
        reconstruct_locked_reading(analysis, {**base, "locked_readings": [{"token_id": tokyo["token_id"], "candidate_id": tokyo["candidate_id"], "chosen_reading": "とうきょう"}]}, _config())
    assert missing.value.code == "reading_ambiguous"
    duplicate = {**base, "locked_readings": [{"token_id": tokyo["token_id"], "candidate_id": tokyo["candidate_id"], "chosen_reading": "とうきょう"}, {"token_id": tokyo["token_id"], "candidate_id": tokyo["candidate_id"], "chosen_reading": "とうきょう"}, {"token_id": de["token_id"], "candidate_id": de["candidate_id"], "chosen_reading": "で"}]}
    with pytest.raises(JAContractError) as repeated:
        reconstruct_locked_reading(analysis, duplicate, _config())
    assert repeated.value.code == "reading_ambiguous"


def test_probe_provider_reports_capabilities():
    probe = probe_provider(_config())
    assert probe["provider"] == "pyopenjtalk-plus"
    assert {"g2p_mapping", "run_frontend_detailed", "make_phoneme_mapping", "extract_fullcontext"} <= set(probe["capabilities"])


def test_asr_transcript_binds_surface_and_kana_evidence_with_raw_offsets():
    analysis = run_frontend("東京で", _config())
    original_digest = analysis["analysis_digest"]
    bound = bind_asr_candidates(analysis, [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "とうきょうで"}])
    tokyo = next(unit for unit in bound["units"] if unit["caller_surface"] == "東京")
    assert tokyo["origin_lexical_status"] == "origin_surface_confirmed"
    assert tokyo["origin_kana_match"] is True
    evidence = tokyo["asr_candidates"][0]
    assert evidence["asr_orig_text"] == "とうきょうで"
    assert evidence["asr_canonical_span"] == [0, 5]
    assert evidence["asr_orig_span"] == [0, 5]
    assert evidence["source_canonical_span"] == [0, 2]
    assert evidence["source_orig_span"] == [0, 2]
    assert evidence["source_kana"] == tokyo["read"]
    view = unit_candidate_analysis(bound, tokyo["token_id"])
    assert view["origin_lexical_match"] is True
    assert bound["analysis_digest"] == original_digest
    assert bound["source_analysis_digest"] == original_digest
    assert bound["asr_analysis_digest"]
    assert view["analysis_digest"] == bound["analysis_digest"]


def test_asr_nonmatching_or_ambiguous_stream_stays_unresolved():
    analysis = run_frontend("東京", _config())
    mismatch = bind_asr_candidates(analysis, [{"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "asr_text": "京都"}])
    assert not mismatch["units"][0].get("asr_candidates")
    ambiguous = bind_asr_candidates(analysis, [{"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "asr_text": "東京東京"}])
    assert ambiguous["asr_bindings"][0]["status"] == "unresolved"
    assert not ambiguous["units"][0].get("asr_candidates")


def test_asr_family_consensus_is_lexical_support_and_does_not_lock_selection():
    analysis = run_frontend("東京", _config())
    bound = bind_asr_candidates(analysis, {"providers": [
        {"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "とうきょう"},
        {"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "asr_text": "とうきょう"},
    ]})
    unit = bound["units"][0]
    assert unit["asr_family_consensus"] is True
    assert unit["asr_consensus_families"] == ["qwen", "whisper"]
    assert unit["origin_lexical_status"] == "origin_surface_confirmed"
    assert "selected_reading" not in unit


def test_w1_blind_projection_hook_returns_token_bound_candidates():
    analysis = run_frontend("東京で", _config())
    providers = [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "とうきょうで"}]
    projection = project_blind_asr_candidates(analysis, providers, _config())
    tokyo = next(unit for unit in analysis["units"] if unit["caller_surface"] == "東京")
    assert projection[tokyo["token_id"]]["origin_match"] is True
    assert projection[tokyo["token_id"]]["candidates"][0]["canonical_span"] == [0, 5]


def test_nonexact_asr_kana_candidates_reach_w1_family_consensus():
    analysis = run_frontend("今日", _config())
    unit = next(value for value in analysis["units"] if value["lexical_status"] == "lexical")
    providers = [
        {"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "こんにち"},
        {"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "asr_text": "こんにち"},
        {"provider": "reazonspeech-nemo-v2", "family": "reazon", "status": "ok", "asr_text": "こんち"},
    ]
    projection = project_blind_asr_candidates(analysis, providers, _config())
    candidate = projection[unit["token_id"]]["candidates"][0]
    assert candidate["token_id"] == unit["token_id"]
    assert candidate["candidate_id"] == unit["candidate_id"]
    assert candidate["reading"] == "コンニチ"
    assert candidate["evidence_scope"] == "phonetic_transcription"
    selected = select_reading({"uid": "u", "token_id": unit["token_id"], "candidate_id": unit["candidate_id"]}, providers, unit_candidate_analysis(analysis, unit["token_id"]))
    # The W1 selector receives projected candidate rows through the helper;
    # direct selection on the un-enriched analysis remains unresolved.
    assert selected["status"] == "none"
    enriched = bind_asr_candidates(analysis, providers, _config())
    projected = project_blind_asr_candidates(enriched, providers, _config())
    token_candidates = projected[unit["token_id"]]["candidates"]
    vote_rows = [{"provider": row["provider"], "family": row["family"], "status": "ok", "candidates": [row]} for row in token_candidates]
    selected = select_reading({"uid": "u", "token_id": unit["token_id"], "candidate_id": unit["candidate_id"]}, vote_rows, unit_candidate_analysis(enriched, unit["token_id"]))
    assert selected["status"] == "asr_family_consensus"
    assert selected["chosen_reading"] == "コンニチ"


def test_asr_does_not_promote_english_oov_or_ai_surface():
    analysis = run_frontend("AI USB", _config())
    bound = bind_asr_candidates(analysis, [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "AI USB"}])
    assert all(not unit.get("asr_candidates") for unit in bound["units"])
    assert all(unit.get("origin_lexical_status") is None for unit in bound["units"])


def test_frontend_stage_binds_manifest_asr_to_each_occurrence(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "uid": "u1", "text": "東京で東京",
        "asr_results": [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "とうきょうでとうきょう"}],
    }], ensure_ascii=False), encoding="utf-8")
    config = {**_config(), "input_manifest": str(manifest)}
    stage_dir = tmp_path / "stages" / "frontend"
    result = frontend_stage(config, stage_dir)
    assert result.status == "COMPLETE"
    payload = json.loads((stage_dir / "frontend_analysis.json").read_text(encoding="utf-8"))
    units = [unit for unit in payload["records"][0]["units"] if unit["lexical_status"] == "lexical"]
    assert len(units) == 3
    assert all(unit["origin_surface_confirmed"] for unit in units)


def test_bound_origin_view_selects_contextual_reading_and_preserves_manual_occurrence_override():
    analysis = run_frontend("東京 東京", _config())
    providers = [{"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "とうきょう とうきょう"}]
    bound = bind_asr_candidates(analysis, providers)
    lexical = [unit for unit in bound["units"] if unit["lexical_status"] == "lexical"]
    assert len(lexical) == 2
    first = select_reading({"uid": "u", "token_id": lexical[0]["token_id"], "candidate_id": lexical[0]["candidate_id"]}, providers, unit_candidate_analysis(bound, lexical[0]["token_id"]))
    assert first["status"] == "origin_surface_confirmed"
    assert first["chosen_reading"] == lexical[0]["read"]
    manual = select_reading({"uid": "u", "token_id": lexical[1]["token_id"], "candidate_id": lexical[1]["candidate_id"], "source_id": "script", "reading_override": "なま"}, providers, unit_candidate_analysis(bound, lexical[1]["token_id"]))
    assert manual["status"] == "manual_verified"
    assert manual["chosen_reading"] == "なま"
