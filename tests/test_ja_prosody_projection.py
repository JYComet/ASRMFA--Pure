import copy

import pytest

from scripts.ja_en_schema import JAContractError, stable_digest
from scripts.ja_prosody import (
    build_prosody_alignment,
    mora_tones_from_phrase,
    resolve_mora_tones,
    validate_projected_phone,
)


def graph_for(reading, *, groups=None, kinds=None):
    kana = list(reading)
    digest = stable_digest(reading)
    groups = groups or [(index,) for index in range(len(kana))]
    kinds = kinds or ["regular"] * len(kana)
    basics = [{"basic_phone_id": f"bp-{index}", "mora_id": f"m-{index}",
               "realization": "observed"} for index in range(len(kana))]
    templates = []
    for index, group in enumerate(groups):
        templates.append({"native_phone_id": f"np-{index}", "native_phone": f"p{index}",
                          "language": "ja", "basic_phone_ids": [f"bp-{item}" for item in group],
                          "mora_ids": [f"m-{item}" for item in group], "transform": "identity"})
    return {"schema": "ja-semantic-phone-graph-v2", "uid": "u-prosody",
            "token_id": "tok-ja", "locked_reading": reading,
            "locked_reading_digest": digest,
            "mora_nodes": [{"mora_id": f"m-{index}", "kana": value, "kind": kinds[index],
                            "mora_index": index, "f0_observed": kinds[index] not in {"devoiced", "elided"}}
                           for index, value in enumerate(kana)],
            "basic_phone_nodes": basics, "native_phone_templates": templates}


def native_phone(index, *, mora_ids, basic_ids, language="ja", label=None, start=None, end=None):
    start = index * 160 if start is None else start
    end = start + 160 if end is None else end
    return {"phone_id": f"phone-{index}", "uid": "u-prosody", "language": language,
            "native_phone": label or f"p{index}", "mora_ids": list(mora_ids),
            "basic_phone_ids": list(basic_ids), "start_sample": start, "end_sample": end,
            "source_axis": {"start_sample": start, "end_sample": end},
            "alignment_axis": {"start_sample": start, "end_sample": end},
            "training_axis": {"start_sample": start, "end_sample": end}}


def alignment_for(graph):
    phones = []
    for index, template in enumerate(graph["native_phone_templates"]):
        phones.append(native_phone(index, mora_ids=template["mora_ids"], basic_ids=template["basic_phone_ids"],
                                  label=template["native_phone"]))
    return {"schema": "ja-en-alignment-v3", "uid": "u-prosody", "words": [],
            "languages": ["ja"], "native_phones": phones, "raw_mfa": {"phones": copy.deepcopy(phones)}}


def valid_frontend(reading, *, nucleus=0):
    digest = stable_digest(reading)
    evidence = {"adapter_version": "openjtalk-fullcontext-accent-v1",
                "provider_identity": {"provider": "test", "provider_revision": "r1"},
                "provider_evidence_sha256": "frontend-evidence",
                "unit_evidence_sha256": "unit-evidence",
                "accent_phrases": [{"accent_phrase_id": "ap0", "mora_count": len(reading),
                                    "nucleus": nucleus}]}
    return {"accent_evidence_valid": True, "locked_reading_digest": digest,
            "accent_evidence": evidence}


def closed_resource(graph, tones, *, entry="entry-1", source="manual"):
    return {"schema": "ja-tone-resource-v1", "version": 1, "source": source,
            "locked_reading_digest": graph["locked_reading_digest"], "mora_count": len(tones),
            "tones": tones, "entry_id": entry, "resource_path": f"/{source}.json",
            "resource_sha256": f"{source}-sha", "provider_revision": "r1",
            "adapter_version": "resource-adapter-v1", "evidence_digest": f"{source}-evidence"}


@pytest.mark.parametrize(("count", "nucleus", "tones"), [
    (3, 0, ["L", "H", "H"]), (3, 1, ["H", "L", "L"]),
    (4, 2, ["L", "H", "L", "L"]), (4, 3, ["L", "H", "H", "L"]),
])
def test_mora_tones_from_phrase(count, nucleus, tones):
    assert mora_tones_from_phrase(count, nucleus) == tones


def test_source_priority_is_not_voting():
    graph = graph_for("サク")
    rows = resolve_mora_tones(
        graph, valid_frontend("サク"),
        manual_overrides=closed_resource(graph, ["H", "L"]),
        accent_lexicon=closed_resource(graph, ["L", "H"], source="lexicon"),
    )
    assert [row["tone"] for row in rows] == ["H", "L"]
    assert {row["tone_source"] for row in rows} == {"manual_override"}
    assert all(row["overridden_sources"] == ["fixed_accent_lexicon", "contextual_frontend_prediction"] for row in rows)
    assert set(rows[0]["tone_provenance"]) == {
        "resource_path", "resource_sha256", "entry_id", "provider_revision",
        "locked_reading_digest", "adapter_version", "evidence_digest",
    }


def test_long_vowel_projects_vector_without_splitting_interval():
    graph = graph_for("コー", groups=[(0,), (1,)])
    alignment = alignment_for(graph)
    alignment["native_phones"][1] = native_phone(1, mora_ids=["m-0", "m-1"], basic_ids=["bp-0", "bp-1"],
                                                  label="oː", start=160, end=480)
    result = build_prosody_alignment(alignment, graph, valid_frontend("コー", nucleus=1))
    phone = next(row for row in result["native_phones"] if row["native_phone"] == "oː")
    assert phone["phone_kana"] == "コ|ー"
    assert phone["phone_tone"] == "H|L"
    assert (phone["start_sample"], phone["end_sample"]) == (160, 480)
    assert phone["duration_group"] == {"basic_phone_ids": ["bp-0", "bp-1"], "group_sum": 320,
                                       "internal_boundaries_known": False}


def test_projection_does_not_change_any_alignment_boundary_evidence():
    graph = graph_for("サク")
    alignment = alignment_for(graph)
    before = copy.deepcopy(alignment["native_phones"])
    result = build_prosody_alignment(alignment, graph, valid_frontend("サク"))
    for original, projected in zip(before, result["native_phones"], strict=True):
        for field in ("start_sample", "end_sample", "source_axis", "alignment_axis", "training_axis"):
            assert projected[field] == original[field]
    assert alignment["native_phones"] == before


@pytest.mark.parametrize(("reading", "groups"), [
    ("キット", [(0,), (1, 2)]), ("オンナ", [(0,), (1, 2)]),
    ("ウンメー", [(0,), (1, 2), (3,)]), ("グッズ", [(0,), (1, 2)]),
])
def test_cross_mora_native_phone_preserves_ordered_tone_vector(reading, groups):
    graph = graph_for(reading, groups=groups)
    result = build_prosody_alignment(alignment_for(graph), graph, valid_frontend(reading))
    cross = next(row for row in result["native_phones"] if len(row["mora_ids"]) > 1)
    assert cross["phone_kana"] == "|".join(reading[index] for index in
                                               [int(item.split("-")[1]) for item in cross["mora_ids"]])
    assert cross["phone_tone"].count("|") == 1
    assert cross["duration_group"]["internal_boundaries_known"] is False


def test_devoicing_and_elision_keep_textual_tone_without_fake_basic_timing():
    graph = graph_for("スキ", kinds=["devoiced", "elided"])
    graph["basic_phone_nodes"][1]["realization"] = "elided"
    graph["basic_phone_nodes"][1]["native_phone_id"] = None
    graph["native_phone_templates"] = graph["native_phone_templates"][:1]
    result = build_prosody_alignment(alignment_for(graph), graph, valid_frontend("スキ", nucleus=1))
    assert result["moras"][0]["tone"] == "H"
    assert result["moras"][0]["f0_observed"] is False
    assert result["basic_phones"][1]["tone"] == "L"
    assert "start_sample" not in result["basic_phones"][1]
    assert "duration" not in result["basic_phones"][1]


def test_final_sokuon_receives_a_mora_tone():
    graph = graph_for("アッ", kinds=["regular", "final_sokuon"])
    result = build_prosody_alignment(alignment_for(graph), graph, valid_frontend("アッ", nucleus=1))
    assert result["moras"][-1]["kind"] == "final_sokuon"
    assert result["moras"][-1]["tone"] == "L"


def test_projection_rejects_conflicting_single_value_for_cross_mora_phone():
    graph = graph_for("コー")
    moras = resolve_mora_tones(graph, valid_frontend("コー", nucleus=1))
    phone = native_phone(1, mora_ids=["m-0", "m-1"], basic_ids=["bp-0", "bp-1"], label="oː")
    phone.update(phone_kana="コ|ー", phone_tone="H")
    with pytest.raises(JAContractError, match="phone_tone_projection_lossy"):
        validate_projected_phone(phone, moras)


def test_english_never_receives_japanese_tone():
    alignment = {"schema": "ja-en-alignment-v3", "uid": "u-prosody", "words": [], "languages": ["en"],
                 "native_phones": [native_phone(0, mora_ids=[], basic_ids=[], language="en", label="AH")]}
    result = build_prosody_alignment(alignment, {"uid": "u-prosody", "mora_nodes": [], "basic_phone_nodes": []}, {})
    assert result["moras"] == []
    assert result["basic_phones"] == []
    assert {row["phone_tone"] for row in result["native_phones"]} == {"NA"}


def test_silence_breath_and_laughter_have_no_mora_and_use_na():
    alignment = {"schema": "ja-en-alignment-v3", "uid": "u-prosody", "words": [], "languages": ["event"],
                 "native_phones": [native_phone(index, mora_ids=[], basic_ids=[], language="event", label=label)
                                   for index, label in enumerate(["sil", "breath", "laugh"])]}
    result = build_prosody_alignment(alignment, {"uid": "u-prosody", "mora_nodes": [], "basic_phone_nodes": []}, {})
    assert result["moras"] == []
    assert all(row["mora_ids"] == [] and row["phone_tone"] == "NA" for row in result["native_phones"])


def test_phrase_mora_cardinality_mismatch_is_rejected():
    graph = graph_for("サクラ")
    frontend = valid_frontend("サクラ")
    frontend["accent_evidence"]["accent_phrases"][0]["mora_count"] = 2
    with pytest.raises(JAContractError, match="tone_cardinality_mismatch"):
        build_prosody_alignment(alignment_for(graph), graph, frontend)


def test_incomplete_known_resource_provenance_is_rejected():
    graph = graph_for("サク")
    override = closed_resource(graph, ["H", "L"])
    override.pop("resource_sha256")
    with pytest.raises(JAContractError, match="tone_provenance_missing"):
        resolve_mora_tones(graph, valid_frontend("サク"), override, None)


def test_digest_mismatch_cannot_use_known_resource():
    graph = graph_for("サク")
    override = closed_resource(graph, ["H", "L"])
    override["locked_reading_digest"] = "tampered"
    result = resolve_mora_tones(graph, valid_frontend("サク"), override, None)
    assert [row["tone"] for row in result] == ["L", "H"]
    assert {row["tone_source"] for row in result} == {"contextual_frontend_prediction"}
