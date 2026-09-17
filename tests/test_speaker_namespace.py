from scripts.speaker_namespace import game_initial_prefix, publication_speaker
from scripts.migrate_publication_speaker_prefixes import migrate

import hashlib
import json


def test_game_prefix_uses_first_two_hanzi_pinyin_initials():
    assert game_initial_prefix("崩坏三") == "BH"
    assert game_initial_prefix("崩铁") == "BT"
    assert game_initial_prefix("重返未来1999") == "ZF"


def test_publication_speaker_prefixes_game_speakers_without_duplication():
    assert publication_speaker("崩坏三", "姬子") == "BH姬子"
    assert publication_speaker("崩铁", "姬子") == "BT姬子"
    assert publication_speaker("鸣潮", "今汐") == "MC今汐"
    assert publication_speaker("尘白禁区", "CB恩雅") == "CB恩雅"
    assert publication_speaker("绝区零", "jqWise") == "jqWise"
    assert publication_speaker(None, "LAria") == "LAria"


def test_completed_publication_migration_moves_file_and_rewrites_receipt(tmp_path):
    run = tmp_path / "run"; output = tmp_path / "out"
    chunk = run / "chunks" / "reference-c"; chunk.mkdir(parents=True)
    source = output / "姬子" / "u1.TextGrid"
    source.parent.mkdir(parents=True); source.write_text("grid", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "frozen_inventory.json").write_text(json.dumps({"items": [{
        "run_stem": "u1", "game": "崩坏三", "speaker": "姬子"}]}))
    (run / "status.json").write_text(json.dumps({
        "terminal": {"reference-c": "complete"}}))
    receipt_path = chunk / "output_publication.json"
    receipt_path.write_text(json.dumps({"rollback_root": str(run / "rollback"),
        "replacements": [{"stem": "u1", "kind": "accepted", "speaker": "姬子",
                          "target": str(source), "rollback": None,
                          "old_sha256": None, "new_sha256": digest}]}))

    result = migrate(run, output, apply=True)

    target = output / "BH姬子" / "u1.TextGrid"
    receipt = json.loads(receipt_path.read_text())
    assert result["moved"] == 1
    assert target.read_text() == "grid"
    assert not source.exists()
    assert receipt["replacements"][0]["speaker"] == "BH姬子"
    assert receipt["replacements"][0]["target"] == str(target.resolve())

