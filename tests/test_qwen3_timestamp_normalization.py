import copy
import json

import pytest

from scripts.qwen3_timestamp_normalization import (
    normalize_qwen_input_text,
    normalize_qwen_reference_text,
    normalize_timestamps,
)


def rows(*spans):
    return [dict(unit=u, word=u, start_s=s, end_s=e, provider="qwen3_hf")
            for u, s, e in spans]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("……", "…"),
        ("你好………世界……", "你好…世界…"),
        ("「你」~（好）！", "你…好！"),
        ("你好,世界.!?", "你好，世界。！？"),
        ("don't open-ai foo_bar $50 50% + =", "don't open-ai foo_bar $50 50% + ="),
        ("'你好' 你’好 O’Neil", "你好 你好 O’Neil"),
        ('“你好” [Breathing] <UNKNOWN>', '你好 [Breathing] <UNKNOWN>'),
        ("... '…' …", "。。。 … …"),
        ("… …", "… …"),
    ],
)
def test_normalize_qwen_input_text_collapses_only_contiguous_unicode_ellipsis(text, expected):
    assert normalize_qwen_input_text(text) == expected


def test_repeated_ellipsis_is_collapsed_in_normalized_bundle_and_is_idempotent():
    first = normalize_timestamps(
        rows(("你", .5, .7), ("好", 1.2, 1.4)), "你……好。", 2
    )
    assert first[1] == "你…好。"
    assert [(pause["word"], pause["start_s"], pause["end_s"])
            for pause in first[2]] == [("…", .7, 1.2), ("。", 1.4, 2)]
    assert normalize_timestamps(first[0], first[1], 2) == first


@pytest.mark.parametrize(("source", "expected"), [
    ("<size=42>25亿！我们的1号嘉宾</size>", "二十五亿！我们的一号嘉宾"),
    ("AR214和Z7小队，1999年。", "AR二一四和Z七小队，一九九九年。"),
    ("概率低于0.0003%~", "概率低于零点零零零三…"),
    ("[Breathing]『你好』\\n<color=#00E1FFFF>世界</color>", "你好 世界"),
])
def test_reference_normalization_removes_game_markup_and_verbalizes_numbers(source, expected):
    assert normalize_qwen_reference_text(source) == expected


def test_reference_normalization_preserves_canonical_ellipsis_idempotently():
    text = "你…好……"
    assert normalize_qwen_reference_text(text) == "你…好…"
    assert normalize_qwen_reference_text(normalize_qwen_reference_text(text)) == "你…好…"


@pytest.mark.parametrize(("speaker", "expected"), [
    ("JQWise", "男主文本。"),
    ("JQBelle", "女主文本！"),
    ("JQSeed", "女主文本！"),
])
def test_reference_normalization_selects_one_conditional_gender_branch(speaker, expected):
    source = "{F#女主文本！}{M#男主文本。}"
    assert normalize_qwen_reference_text(source, speaker=speaker) == expected


@pytest.mark.parametrize(("source", "expected"), [
    # `{RUBY#[X]gloss}` is a ruby annotation: the surrounding text is the base
    # and the gloss is typeset above it, never spoken.
    ("杜麦{RUBY#[S]希望}尼，这就是他的古名。", "杜麦尼，这就是他的古名。"),
    ("「库塔{RUBY#[S]月之少女}尔」…我听说过你。", "库塔尔…我听说过你。"),
    ("花{RUBY#[D]   特拉洛坎}羽会的战士向来高傲", "花羽会的战士向来高傲"),
    ("把它转移到我的剑{RUBY#[S]钢铁的爪牙}里就行。", "把它转移到我的剑里就行。"),
    ("我的「古名」…「庇{RUBY#[D]奉献}笛」。", "我的古名…庇笛。"),
])
def test_reference_ruby_gloss_is_dropped_leaving_the_base_text(source, expected):
    assert normalize_qwen_reference_text(source) == expected


@pytest.mark.parametrize("branch", [
    "[INFO_MALE_PRONOUN_HE|INFO_FEMALE_PRONOUN_SHE]",
    "[INFO_FEMALE_PRONOUN_SHE|INFO_MALE_PRONOUN_HE]",
    # The corpus contains labels whose INFO_MALE_/INFO_FEMALE_ prefix
    # contradicts the pronoun it carries, so only the pronoun token may decide.
    "[INFO_MALE_PRONOUN_SHE|INFO_FEMALE_PRONOUN_HE]",
    "[INFO_MALE_PRONOUN_SHE|INFO_MALE_PRONOUN_HE]",
])
@pytest.mark.parametrize("macro", ["PLAYERAVATAR", "MATEAVATAR"])
def test_sexpro_homophone_branch_always_takes_the_fixed_policy(macro, branch):
    source = f"{{{macro}#SEXPRO{branch}}}好"
    assert normalize_qwen_reference_text(source) == "他好"


def test_sexpro_homophone_policy_cannot_be_overridden_by_a_table():
    source = "{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_FEMALE_PRONOUN_SHE]}好"
    assert normalize_qwen_reference_text(source, resolutions={source[:58]: "她"}) == "他好"


def test_macro_literal_spanning_a_gloss_does_not_swallow_its_neighbour():
    # `{PLAYERAVATAR#SEXPRO[…INFO_FEMALE}{NICKNAME}` must resolve as two macros.
    source = ("{NICKNAME}"
              "{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_HE|INFO_FEMALE_PRONOUN_SHE]}们认识的")
    assert normalize_qwen_reference_text(
        source, resolutions={"{NICKNAME}": "旅行者"}) == "旅行者他们认识的"


def test_audibly_distinct_branch_requires_evidence_and_honours_it():
    source = "{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_BROTHER|INFO_FEMALE_PRONOUN_SISTERA]}"
    with pytest.raises(ValueError, match="ambiguous branch"):
        normalize_qwen_reference_text(source)
    assert normalize_qwen_reference_text(source, resolutions={source: "哥哥"}) == "哥哥"
    with pytest.raises(ValueError, match="not one of its branches"):
        normalize_qwen_reference_text(source, resolutions={source: "姐姐的好朋友"})


def test_name_macro_requires_a_resolution_and_never_leaks_its_ascii_name():
    with pytest.raises(ValueError, match="no resolution"):
        normalize_qwen_reference_text("{NICKNAME}，好久不见。")
    assert normalize_qwen_reference_text(
        "{NICKNAME}，好久不见。", resolutions={"{NICKNAME}": "旅行者"}) == "旅行者，好久不见。"


@pytest.mark.parametrize("source", [
    "看来{TEXTJOIN#54}也已经准备好了。",                       # runtime join id
    "{PLAYERAVATAR#SEXPRO[INFO_MALE_PRONOUN_XIABOY|INFO_FEMALE_PRONOUN_XIAGIRL]}",
    "{REALNAME[ID(1)|HOSTONLY(true)]}",
    "{SOMETHINGNEW#1}",
])
def test_unrecognized_macros_fail_closed_instead_of_leaking(source):
    with pytest.raises(ValueError):
        normalize_qwen_reference_text(source)


def test_macro_free_text_is_byte_identical_with_or_without_a_table():
    for text in ("你好，世界。", "AR214和Z7小队，1999年。", "概率低于0.0003%~",
                 "「你」~（好）！", "don't open-ai foo_bar $50 50%"):
        assert (normalize_qwen_reference_text(text)
                == normalize_qwen_reference_text(text, resolutions={}))


def test_bare_aliases_are_inert_until_a_table_opts_in():
    # `player`/`TA` are ordinary words unless the run supplies a resolution, so
    # a normalizer call without a table cannot alter them.
    for text in ("player 你好", "TA 你好"):
        assert normalize_qwen_reference_text(text) == text
    assert normalize_qwen_reference_text(
        "TA 你好", resolutions={"alias:TA": "他"}) == "他 你好"
    with pytest.raises(ValueError, match="unresolved Qwen reference alias"):
        normalize_qwen_reference_text("TA 你好", resolutions={})


def test_unsafe_macro_resolution_is_rejected():
    with pytest.raises(ValueError, match="unsafe Qwen reference"):
        normalize_qwen_reference_text(
            "{NICKNAME}好", resolutions={"{NICKNAME}": "NICKNAME"})


def test_short_pause_extends_previous_word_and_long_pause_preserves_punctuation():
    original = rows(("你", 0.1, 0.3), ("好", 0.4, 0.6), ("啊", 1.1, 1.3))
    before = copy.deepcopy(original)
    fixed, text, pauses = normalize_timestamps(original, "你好，啊！", 1.4)
    assert original == before
    assert [(r["start_s"], r["end_s"]) for r in fixed] == [(.1, .4), (.4, .6), (1.1, 1.4)]
    assert text == "你好，啊！"
    assert [(p["word"], p["start_s"], p["end_s"]) for p in pauses] == [("，", .6, 1.1)]
    assert fixed[0]["timestamp_normalization"]["original_end_s"] == .3


def test_timestamp_normalization_clamps_quantized_tail_to_audio_axis():
    fixed, text, pauses = normalize_timestamps(
        rows(("你", .5, 1.04)), "你。", 1.0
    )
    assert fixed[0]["start_s"] == .5
    assert fixed[0]["end_s"] == 1.0
    assert fixed[0]["timestamp_normalization"]["original_end_s"] == 1.04
    assert text == "你。"
    assert pauses == []


@pytest.mark.parametrize("gap,label", [(.199999, "sp0"), (.2, "sp1"), (.499999, "sp1"), (.5, "sp2"), (1.499999, "sp2"), (1.5, "sp3")])
def test_pause_thresholds_use_serialized_microseconds(gap, label):
    fixed, text, pauses = normalize_timestamps(rows(("你", 0, .1), ("好", .1 + gap, .3 + gap)), "你好", .3 + gap)
    assert text == ("你好" if label == "sp0" else "你…好")
    if label == "sp0":
        assert fixed[0]["end_s"] == round(.1 + gap, 6)
    else:
        assert pauses[0]["pause_label"] == label


def test_edges_and_explicit_pause_tags_are_not_lexical_words():
    fixed, text, pauses = normalize_timestamps(rows(("Hello", .5, .7), ("world", .8, 1.0)), "<sp2>Hello<sp0> world<sp3>", 2.5)
    assert text == "Hello world…"
    assert len(fixed) == 2
    assert [(p["left_lexical_ordinal"], p["right_lexical_ordinal"]) for p in pauses] == [(1, None)]


@pytest.mark.parametrize("bad", [rows(("你", -.1, .2)), rows(("你", 0, float("nan"))), rows(("你", 0, 0)), rows(("你", 0, .6), ("好", .5, .7)), rows(("你", 0, 2))])
def test_invalid_timeline_is_rejected_without_fallback(bad):
    with pytest.raises(ValueError):
        normalize_timestamps(bad, "你好", 1)


def test_reapplying_normalization_preserves_evidence_and_geometry():
    first = normalize_timestamps(rows(("你", 0, .1), ("好", .2, .3)), "你好", .5)
    second = normalize_timestamps(first[0], first[1], .5)
    assert second == first


def test_main_defaults_to_qwen_without_changing_mfa_policy(tmp_path):
    from scripts.run_pipeline import load_config
    path = tmp_path / "config.yaml"
    path.write_text("mfa:\n  native_anchor_fallback: true\nctc_adjust:\n  enabled: true\n")
    cfg = load_config(path)
    assert cfg["ctc_prealign"]["provider"] == "qwen3_hf"
    assert cfg["mfa"]["native_anchor_fallback"] is True
    assert cfg["ctc_adjust"]["enabled"] is True
    assert cfg["postprocess"]["enable_text_correction"] is True


def test_legacy_main_config_migrates_runtime_without_mfa_changes(tmp_path, capsys):
    from scripts.run_pipeline import load_config, validate_config
    path = tmp_path / "config.yaml"
    path.write_text("ctc_prealign:\n  provider: nvasr\n  model_path: /old/Multilingual-NVASR\n  nvv_enabled: true\n  all_gpus: true\n  batch_size: 4\n  offset: 2\n  limit: 8\nmfa:\n  beam: 30\n")
    cfg = load_config(path)
    pc = cfg["ctc_prealign"]
    assert pc["provider"] == "qwen3_hf" and "Qwen3" in pc["model_path"]
    assert pc["offset"] == 2 and pc["limit"] == 8 and pc["batch_size"] == 4
    assert cfg["mfa"]["beam"] == 30 and pc["all_gpus"] is True
    assert "NVASR" in capsys.readouterr().out
    assert validate_config(cfg, "full") == []


def test_normalized_text_rejects_tampering():
    from scripts.qwen3_timestamp_normalization import normalized_bundle_text
    fixed, text, _ = normalize_timestamps(rows(("你", 0, .1)), "你", .5)
    assert normalized_bundle_text(fixed, text) == "你…"
    with pytest.raises(ValueError, match="evidence mismatch"):
        normalized_bundle_text(fixed, "你")


def test_parallel_inference_uses_qwen_workers_and_preserves_order():
    from pathlib import Path
    from scripts.qwen3_prealign import infer_qwen_items
    from scripts.qwen3_hf_backend import Qwen3HFSettings
    calls = []
    class Backend:
        last_language = "Chinese"
        def transcribe(self, audio, **kw):
            calls.append(("asr", audio.name))
            return "你"
        def align(self, audio, text, **kw):
            calls.append(("align", audio.name))
            return [dict(unit="你", start_s=0, end_s=.1)]
        def close(self):
            calls.append(("close", None))
    def factory(settings, **kw):
        calls.append(("device", settings.device))
        return Backend()
    settings = Qwen3HFSettings(Path("asr"), Path("aligner"), batch_size=2)
    result = list(infer_qwen_items([Path("a.wav"), Path("b.wav")], {"a": "你"},
                                  "auto", settings, devices=["cuda:0", "cuda:1"],
                                  backend_factory=factory))
    assert [wav.stem for wav, _ in result] == ["a", "b"]
    assert ("asr", "a.wav") not in calls and ("asr", "b.wav") in calls
    assert sum(kind == "close" for kind, _ in calls) == sum(kind == "device" for kind, _ in calls)


def test_mode_override_and_padded_legacy_provider_cannot_select_nvasr():
    from scripts.run_pipeline import migrate_main_prealign_config
    cfg = {"ctc_prealign": {"provider": " NVASR ", "model_path": "/custom/legacy-model"}}
    assert migrate_main_prealign_config(cfg, "full")["ctc_prealign"]["provider"] == "qwen3_hf"


def test_bad_config_section_is_left_for_schema_validation():
    from scripts.run_pipeline import migrate_main_prealign_config
    cfg = {"ctc_prealign": None}
    assert migrate_main_prealign_config(cfg, "full") == cfg


@pytest.mark.parametrize("leading", [.1, .3, .8, 2.0])
@pytest.mark.parametrize("tag", ["", "<sp0>", "<sp1>", "<sp2>", "<sp3>"])
def test_leading_silence_never_inserts_ellipsis(leading, tag):
    fixed, text, pauses = normalize_timestamps(rows(("你", leading, leading + .2)), tag + "你！", leading + .2)
    assert text == "你！"
    assert pauses == []
    assert fixed[0]["start_s"] == leading


@pytest.mark.parametrize("punctuation,normalized", [
    ("，", "，"), ("？！", "？！"), ("。", "。"), ("…", "…"),
    ("！ ”", "！ "), (", ", "， "),
])
def test_existing_punctuation_run_owns_long_pause(punctuation, normalized):
    first = normalize_timestamps(rows(("你", .5, .7), ("好", 1.2, 1.4)), "你" + punctuation + "好。", 2)
    fixed, text, pauses = first
    assert text == "你" + normalized + "好。"
    assert [(p["word"], p["start_s"], p["end_s"]) for p in pauses] == [("".join(normalized.split()), .7, 1.2), ("。", 1.4, 2)]
    assert normalize_timestamps(fixed, text, 2) == first


def test_explicit_tags_do_not_displace_existing_punctuation():
    fixed, text, pauses = normalize_timestamps(rows(("你", .5, .7), ("好", 1.2, 1.4)), "<sp2>你，<sp2>好！<sp3>", 2)
    assert text == "你，好！"
    assert [p["word"] for p in pauses] == ["，", "！"]


def test_unpunctuated_long_gap_adds_only_one_ellipsis_on_reapplication():
    first = normalize_timestamps(rows(("你", .5, .7), ("好", 1.2, 1.4)), "你好。", 2)
    assert first[1] == "你…好。"
    assert normalize_timestamps(first[0], first[1], 2) == first
