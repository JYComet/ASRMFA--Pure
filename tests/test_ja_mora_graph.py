import importlib.util
import os
import sys
from pathlib import Path

import pytest

from scripts.ja_frontend import run_frontend
from scripts.ja_phone_adapter import load_japanese_mfa_inventory, openjtalk_to_semantic, semantic_to_japanese_mfa_v3


def test_long_vowel_keeps_many_to_many_mora_relation():
    runtime = os.environ.get("JA_FRONTEND_PYTHON")
    if not runtime:
        if importlib.util.find_spec("pyopenjtalk") is None:
            pytest.skip("set JA_FRONTEND_PYTHON for the explicit real frontend bridge")
        runtime = sys.executable
    config = {"provider": "pyopenjtalk-plus", "commit": "9e4bf25324ac135dfc81ca64aed2fa6a48b83304", "python": runtime, "normalize_mode": "NFKC", "use_vanilla": False, "use_sudachi_kanji_yomi": False, "predict_nani": False, "use_tsqyomi": False, "use_read_as_pron": False, "revert_long_vowels": False, "revert_yotsugana": False, "run_marine": False, "reject_unbound_spans": True}
    unit = run_frontend("東京", config)["units"][0]
    fixture_dir = Path(__file__).parent / "fixtures"
    target = semantic_to_japanese_mfa_v3(openjtalk_to_semantic(unit), load_japanese_mfa_inventory(fixture_dir / "ja_mfa_portable_metadata.json"), dictionary_path=fixture_dir / "ja_mfa_portable.dict")
    assert target["phones"] == ["t", "oː", "c", "oː"]
    assert any(len(node["mora_ids"]) > 1 for node in target["phone_nodes"] if node["phone"] == "oː")
