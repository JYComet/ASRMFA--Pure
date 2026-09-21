from pathlib import Path

from scripts.ja_phone_adapter import load_japanese_mfa_inventory, route_language, split_mora


def test_inventory_is_audited_and_mora_split_is_not_phone_zip():
    inventory = load_japanese_mfa_inventory(Path(__file__).parent / "fixtures" / "ja_mfa_portable_metadata.json")
    assert inventory["coverage_status"] == "inventory_declared"
    assert split_mora("トウキョウ") == ["と", "う", "きょ", "う"]


def test_latin_route_stays_unresolved_without_policy():
    assert route_language({"caller_surface": "AI"}) == "unresolved"
    assert route_language({"caller_surface": "game", "clear_english_pronunciation": True}) == "en"
