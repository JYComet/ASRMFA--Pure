from scripts.ja_en_schema import make_occurrence_alias
from pathlib import Path

from scripts.ja_phone_adapter import build_alias_rows, load_japanese_mfa_inventory


def test_alias_prefixes_are_ascii_and_per_occurrence():
    assert make_occurrence_alias("ja", 0) == "ju_000000"
    assert make_occurrence_alias("en", 1) == "eu_000001"
    fixture_dir = Path(__file__).parent / "fixtures"
    rows = build_alias_rows({"units": [{"language": "ja", "lexical_status": "lexical", "caller_surface": "さくら", "surface": "さくら", "read": "サクラ", "phones": ["s", "a", "k", "u", "r", "a"], "mora_count": 3, "token_id": "tok", "candidate_id": "cand", "orig_span": [0, 3]}]}, inventory=load_japanese_mfa_inventory(fixture_dir / "ja_mfa_portable_metadata.json"), dictionary_path=fixture_dir / "ja_mfa_portable.dict")
    assert rows[0]["alias"] == "ju_000000"
