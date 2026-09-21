import json
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from scripts.ja_frontend import run_frontend
from scripts.ja_phone_adapter import (
    DEFAULT_DICTIONARY,
    DEFAULT_METADATA,
    build_alias_rows,
    openjtalk_to_semantic,
    semantic_to_japanese_mfa_v3,
    write_locked_alias_artifacts,
    load_japanese_mfa_inventory,
    semantic_stage,
)


def _analysis(text="東京 学校 こんにちは さくら"):
    runtime = os.environ.get("JA_FRONTEND_PYTHON")
    if not runtime:
        if importlib.util.find_spec("pyopenjtalk") is None:
            pytest.skip("set JA_FRONTEND_PYTHON for the explicit real frontend bridge")
        runtime = sys.executable
    return run_frontend(text, {
        "provider": "pyopenjtalk-plus",
        "commit": "9e4bf25324ac135dfc81ca64aed2fa6a48b83304",
        "python": runtime,
        "normalize_mode": "NFKC", "use_vanilla": False,
        "use_sudachi_kanji_yomi": False, "predict_nani": False,
        "use_tsqyomi": False, "use_read_as_pron": False,
        "revert_long_vowels": False, "revert_yotsugana": False,
        "run_marine": False, "reject_unbound_spans": True,
    })


def test_semantic_graph_preserves_many_to_many_mora_edges_and_golden_inventory():
    analysis = run_frontend("東京", _analysis_config())
    unit = analysis["units"][0]
    graph = openjtalk_to_semantic(unit)
    assert graph["schema"] == "ja-semantic-phone-graph-v1"
    target = semantic_to_japanese_mfa_v3(graph, load_japanese_mfa_inventory(Path(__file__).parent / "fixtures" / "ja_mfa_portable_metadata.json"), dictionary_path=Path(__file__).parent / "fixtures" / "ja_mfa_portable.dict")
    assert target["phones"] == ["t", "oː", "c", "oː"]
    assert any(edge["relation"] == "mora_phone" for edge in graph["edges"])
    assert any(len(edge["mora_ids"]) > 1 for edge in target["phone_nodes"] if edge.get("phone") == "oː")


def test_golden_japanese_rows_roundtrip_without_string_substitution():
    expected = {
        "さくら": ["s", "a", "k", "ɯ", "ɾ", "a"],
        "学校": ["ɡ", "a", "kː", "oː"],
        "こんにちは": ["k", "o", "ɲː", "i", "tɕ", "i", "w", "a"],
    }
    for text, phones in expected.items():
        unit = next(item for item in run_frontend(text, _analysis_config())["units"] if item["lexical_status"] == "lexical")
        result = semantic_to_japanese_mfa_v3(openjtalk_to_semantic(unit), load_japanese_mfa_inventory(Path(__file__).parent / "fixtures" / "ja_mfa_portable_metadata.json"), dictionary_path=Path(__file__).parent / "fixtures" / "ja_mfa_portable.dict")
        assert result["phones"] == phones


def test_occurrence_aliases_are_unique_and_dictionary_rows_are_locked(tmp_path):
    analysis = _analysis()
    rows = build_alias_rows(analysis, language="ja", inventory=load_japanese_mfa_inventory(Path(__file__).parent / "fixtures" / "ja_mfa_portable_metadata.json"), dictionary_path=Path(__file__).parent / "fixtures" / "ja_mfa_portable.dict")
    assert rows
    assert len({row["alias"] for row in rows}) == len(rows)
    dictionary = tmp_path / "locked.dict"
    alias_map = tmp_path / "alias_map.jsonl"
    report = write_locked_alias_artifacts(rows, dictionary, alias_map)
    assert report["aliases"] == [row["alias"] for row in rows]
    assert len(dictionary.read_text(encoding="utf-8").splitlines()) == len(rows)
    assert len(alias_map.read_text(encoding="utf-8").splitlines()) == len(rows)
    assert all(" " in line for line in dictionary.read_text(encoding="utf-8").splitlines())


def test_semantic_stage_pure_english_uses_only_bound_english_assets(tmp_path):
    fixture_dir = Path(__file__).parent / "fixtures"
    archive = tmp_path / "english.zip"
    archive.write_bytes(b"portable acoustic fixture")
    reconstruction = tmp_path / "stages" / "frontend" / "frontend_reconstruction.json"
    reconstruction.parent.mkdir(parents=True)
    reconstruction.write_text(json.dumps({"records": [{"uid": "en1", "units": [{
        "language": "en", "lexical_status": "lexical", "caller_surface": "hello", "surface": "hello",
        "read": "hello", "locked_phones": ["HH", "AH0", "L", "OW1"], "token_id": "tok0", "candidate_id": "cand0", "orig_span": [0, 5],
    }]}]}, ensure_ascii=False), encoding="utf-8")
    config = {"mfa": {
        "english_metadata": str(fixture_dir / "en_arpa_portable_metadata.json"),
        "english_acoustic": str(archive),
        "english_acoustic_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }}
    result = semantic_stage(config, tmp_path / "stages" / "semantic")
    assert result.status == "COMPLETE"
    rows = (tmp_path / "stages" / "semantic" / "alias_map.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["language"] == "en"


def _analysis_config():
    runtime = os.environ.get("JA_FRONTEND_PYTHON")
    if not runtime:
        if importlib.util.find_spec("pyopenjtalk") is None:
            pytest.skip("set JA_FRONTEND_PYTHON for the explicit real frontend bridge")
        runtime = sys.executable
    return {
        "provider": "pyopenjtalk-plus", "commit": "9e4bf25324ac135dfc81ca64aed2fa6a48b83304",
        "python": runtime, "normalize_mode": "NFKC",
        "use_vanilla": False, "use_sudachi_kanji_yomi": False, "predict_nani": False,
        "use_tsqyomi": False, "use_read_as_pron": False, "revert_long_vowels": False,
        "revert_yotsugana": False, "run_marine": False, "reject_unbound_spans": True,
    }
