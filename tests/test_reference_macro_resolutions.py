import pytest

from scripts.reference_macro_resolutions import (
    build_table, classify_slot, is_actionable, plan_text, read_resolutions,
    game_nickname_default, table_digest, transcript_has, write_resolutions,
)
from scripts.qwen3_timestamp_normalization import reference_macro_slots


def only_slot(text):
    slots = reference_macro_slots(text)
    assert len(slots) == 1
    return slots[0]


def test_transcript_match_ignores_punctuation_and_spacing():
    assert transcript_has("你好，旅行者！", "旅行者")
    assert transcript_has("他 说", "他")
    assert not transcript_has("你好，旅行者！", "开拓者")
    assert not transcript_has(None, "旅行者")


def test_homophone_branch_is_decided_by_policy_not_by_evidence():
    plan = classify_slot(
        only_slot("{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_FEMALE_PRONOUN_SHE]}"),
        run_stem="u1", game="原神", transcript=None)
    assert (plan.kind, plan.value) == ("policy", "他")
    assert not plan.review_required


def test_name_macro_falls_back_to_the_game_default_and_is_flagged_for_review():
    hero = classify_slot(only_slot("{NICKNAME}"), run_stem="u1", game="原神")
    assert (hero.kind, hero.value) == ("default", "旅行者")
    assert hero.review_required

    star = classify_slot(only_slot("{NICKNAME}"), run_stem="u2", game="崩铁")
    assert star.value == "开拓者"
    assert game_nickname_default("物华弥新") == "收藏家"
    assert game_nickname_default("未知游戏") is None


def test_confirmed_default_becomes_evidence_not_review():
    plan = classify_slot(only_slot("{NICKNAME}"), run_stem="u1", game="原神",
                         transcript="派蒙，旅行者你们来啦。")
    assert (plan.kind, plan.review_required) == ("evidence", False)


def test_unknown_game_cannot_default_a_name_macro():
    plan = classify_slot(only_slot("{NICKNAME}"), run_stem="u1", game="未知游戏")
    assert plan.kind == "unresolved" and plan.value is None


@pytest.mark.parametrize("transcript,expected", [
    ("哥哥，你回来啦", "哥哥"),
    ("姐姐，你回来啦", "姐姐"),
    ("你回来啦", None),                 # heard nothing that disambiguates
    ("哥哥和姐姐都来了", None),          # both branches heard -> ambiguous
])
def test_audibly_distinct_branch_needs_a_unique_transcript_match(transcript, expected):
    source = "{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_BROTHER|INFO_FEMALE_PRONOUN_SISTERA]}"
    plan = classify_slot(only_slot(source), run_stem="u1", game="原神",
                         transcript=transcript)
    assert plan.value == expected
    assert plan.kind == ("evidence" if expected else "unresolved")


def test_ruby_gloss_resolves_to_nothing_and_needs_no_evidence():
    plans = plan_text("杜麦{RUBY#[S]希望}尼。", run_stem="u1", game="原神")
    assert [p.value for p in plans] == [""]
    assert is_actionable(plans)


@pytest.mark.parametrize("source,reason", [
    ("{TEXTJOIN#54}", "unrecognized macro"),
    ("{SOMETHINGNEW#1}", "unrecognized macro"),
    ("{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_XIABOY|INFO_FEMALE_PRONOUN_XIAGIRL]}",
     "unrecognized macro labels"),
])
def test_unrecoverable_macros_are_withheld(source, reason):
    plan = classify_slot(only_slot(source), run_stem="u1", game="原神")
    assert plan.kind == "unresolved" and plan.value is None
    assert reason in plan.reason
    assert not is_actionable((plan,))


def test_item_is_actionable_only_when_every_slot_is_decided():
    assert not is_actionable(())
    mixed = plan_text("{NICKNAME}{TEXTJOIN#54}", run_stem="u1", game="原神")
    assert not is_actionable(mixed)
    assert is_actionable(plan_text("{NICKNAME}你好", run_stem="u1", game="原神"))


def test_build_table_rejects_a_contradiction_within_one_item():
    plans = plan_text("{NICKNAME}和{TEXTJOIN#54}", run_stem="u1", game="原神")
    assert build_table(plans) == {"{NICKNAME}": "旅行者"}
    from scripts.reference_macro_resolutions import SlotPlan
    clash = (SlotPlan("u1", "{NICKNAME}", "NICKNAME", "default", "旅行者", (), ""),
             SlotPlan("u1", "{NICKNAME}", "NICKNAME", "default", "开拓者", (), ""))
    with pytest.raises(ValueError, match="conflicting macro resolutions"):
        build_table(clash)


def test_resolutions_round_trip_and_detect_tampering(tmp_path):
    report = write_resolutions(tmp_path, {"u1": {"{NICKNAME}": "旅行者"}})
    tables, digest = read_resolutions(tmp_path)
    assert tables == {"u1": {"{NICKNAME}": "旅行者"}}
    assert digest == report["digest"]
    # Per-stem digests bind one chunk's tables; the run digest binds them all.
    assert table_digest(tables["u1"]) == table_digest({"{NICKNAME}": "旅行者"})

    path = tmp_path / "reference_macro_resolutions.json"
    payload = path.read_text(encoding="utf-8").replace("旅行者", "开拓者")
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_resolutions(tmp_path)
