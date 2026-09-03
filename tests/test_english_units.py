import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scripts.english_units import (
    MERGE_KIND_COMPOUND,
    MERGE_KIND_DIRECT,
    EnglishUnit,
    EnglishUnitError,
    canonicalize_english_token,
    is_english_fragment_token,
    merge_authority_fragment_group,
    merge_authority_units,
    parse_english_units,
    resolve_processed_english_token,
    validate_processed_english_token_binding,
)


def fragment(text, ordinal, start=None, end=None):
    item = {"text": text, "ordinal": ordinal}
    if start is not None:
        item.update(start=start, end=end)
    return item


def test_hyphenated_units_preserve_surface_and_have_stable_distinct_ids():
    units = parse_english_units("K-Pop V-Up hello K-Pop")

    assert [unit.surface_text for unit in units] == ["K-Pop", "V-Up", "hello", "K-Pop"]
    assert [unit.alignment_token for unit in units] == ["kpop", "vup", "hello", "kpop"]
    assert [unit.match_key for unit in units] == ["kpop", "vup", "hello", "kpop"]
    assert [unit.reference_ordinal for unit in units] == [0, 1, 2, 3]
    assert [unit.unit_id for unit in units] == ["en-u0000", "en-u0001", "en-u0002", "en-u0003"]
    assert units[0].merge_kind == MERGE_KIND_COMPOUND
    assert units[2].merge_kind == MERGE_KIND_DIRECT
    assert units[0].canonical_span == (0, 5)
    assert units[3].canonical_span == (17, 22)


def test_nvv_precedence_skips_nvv_and_keeps_other_units():
    units = parse_english_units("<LAUGHTER> LAUGHTER K-Pop")

    assert [unit.surface_text for unit in units] == ["K-Pop"]


def test_hyphenless_english_is_direct_and_canonicalization_is_strict():
    unit = parse_english_units("OpenAI")[0]

    assert unit.merge_kind == MERGE_KIND_DIRECT
    assert canonicalize_english_token("OpenAI") == "openai"
    with pytest.raises(EnglishUnitError) as error:
        canonicalize_english_token("open_ai")
    assert error.value.code == "invalid_english_syntax"
    with pytest.raises(EnglishUnitError) as error:
        canonicalize_english_token("open–ai")
    assert error.value.code == "invalid_english_syntax"
    with pytest.raises(EnglishUnitError) as error:
        canonicalize_english_token("rock'n'roll")
    assert error.value.code == "invalid_english_syntax"


def test_exact_contiguous_authority_merge_uses_source_ordinals_and_span():
    authority = parse_english_units("K-Pop")[0]
    merged = merge_authority_fragment_group(
        authority,
        [fragment("K", 7, 1.0, 1.2), fragment("Pop", 8, 1.2, 1.8)],
    )

    assert merged.surface_text == "K-Pop"
    assert merged.alignment_token == "kpop"
    assert merged.source_ctc_ordinals == (7, 8)
    assert merged.canonical_span == (1.0, 1.8)
    assert json.loads(merged.to_json())["source_ctc_ordinals"] == [7, 8]
    assert isinstance(merged, EnglishUnit)


@pytest.mark.parametrize(
    ("fragments", "code"),
    [
        ([fragment("K", 7)], "partial_fragment_match"),
        ([fragment("K", 7), fragment("Po", 8), fragment("p", 9), fragment("x", 10)],
         "extra_fragment"),
        ([fragment("Pop", 8), fragment("K", 7)], "noncontiguous_source_ctc_ordinals"),
        ([fragment("K", 7), fragment(",", 8), fragment("Pop", 9)], "punctuation_fragment"),
        ([fragment("K", 7), fragment("中", 8), fragment("Pop", 9)], "cjk_crossing"),
        ([fragment("K", 7), fragment("LAUGHTER", 8), fragment("Pop", 9)], "nvv_crossing"),
    ],
)
def test_authority_merge_rejects_non_exact_groups(fragments, code):
    with pytest.raises(EnglishUnitError) as error:
        merge_authority_fragment_group(parse_english_units("K-Pop")[0], fragments)
    assert error.value.code == code


def test_hyphenated_authority_accepts_visible_ctc_hyphen_separator():
    authority = parse_english_units("v-tuber")[0]
    merged = merge_authority_fragment_group(authority, [
        {"text": "v", "ordinal": 4, "start": 0.1, "end": 0.2},
        {"text": "-", "ordinal": 5, "start": 0.2, "end": 0.21},
        {"text": "tuber", "ordinal": 6, "start": 0.21, "end": 0.5},
    ])
    assert merged.surface_text == "v-tuber"
    assert merged.source_ctc_ordinals == (4, 5, 6)


def test_hyphenated_authority_tolerates_dropped_ctc_hyphen():
    authority = parse_english_units("K-Pop")[0]
    merged = merge_authority_fragment_group(authority, [
        {"text": "K", "ordinal": 7, "start": 0.1, "end": 0.2},
        {"text": "Pop", "ordinal": 9, "start": 0.2, "end": 0.5},
    ])
    assert merged.surface_text == "K-Pop"
    assert merged.source_ctc_ordinals == (7, 9)


def test_authority_merge_rejects_extra_units_and_reordered_unit_groups():
    authorities = parse_english_units("K-Pop V-Up")
    with pytest.raises(EnglishUnitError, match="authority_group_count_mismatch"):
        merge_authority_units(authorities, [[fragment("K", 0), fragment("Pop", 1)]])
    with pytest.raises(EnglishUnitError, match="source_ctc_ordinals_not_monotonic"):
        merge_authority_units(
            authorities,
            [[fragment("K", 2), fragment("Pop", 3)], [fragment("V", 1), fragment("Up", 2)]],
        )


def test_surface_casing_is_not_mutated():
    units = parse_english_units("k-POP v-Up")

    assert [unit.surface_text for unit in units] == ["k-POP", "v-Up"]
    assert [unit.alignment_token for unit in units] == ["kpop", "vup"]


def test_alpha_digit_units_keep_surface_identity_but_use_alpha_dictionary_key():
    units = parse_english_units("target1 target2 jin1 rui4 OK K-Pop")

    assert [unit.surface_text for unit in units] == [
        "target1", "target2", "OK", "K-Pop"
    ]
    assert [unit.alignment_token for unit in units] == [
        "target", "target", "ok", "kpop"
    ]
    assert [unit.unit_id for unit in units] == [
        "en-u0000", "en-u0001", "en-u0002", "en-u0003"
    ]
    assert canonicalize_english_token("target1") == "target"
    assert not is_english_fragment_token("jin1")
    assert not is_english_fragment_token("rui4")


def test_alpha_digit_ctc_suffix_is_final_and_exact():
    authority = parse_english_units("target1")[0]
    merged = merge_authority_fragment_group(
        authority, [fragment("target", 4), fragment("1", 5)])
    assert merged.surface_text == "target1"
    assert merged.alignment_token == "target"
    assert merged.source_ctc_ordinals == (4, 5)

    with pytest.raises(EnglishUnitError, match="numeric_suffix_not_final"):
        merge_authority_fragment_group(
            parse_english_units("target12")[0],
            [fragment("target", 0), fragment("1", 1), fragment("2", 2)])

    with pytest.raises(EnglishUnitError, match="pinyin"):
        merge_authority_fragment_group(
            authority, [fragment("jin1", 4), fragment("1", 5)])


def _strict_binding_fixture(surface, ordinals):
    unit = parse_english_units(surface)[0]
    canonical = unit.to_dict()
    canonical["source_ctc_ordinals"] = list(ordinals)
    token = {
        "type": "word", "word": unit.alignment_token,
        "surface_text": unit.surface_text,
        "source_ctc_ordinals": list(ordinals),
        "canonical_span": list(unit.canonical_span),
        "canonical_unit": canonical,
        "start_s": 0.1, "end_s": 0.5,
    }
    if any(right - left > 1 for left, right in zip(ordinals, ordinals[1:])):
        token["hyphen_separator_omitted"] = True
    record = {
        "ctc_ordinal": ordinals[0],
        "source_ctc_ordinals": list(ordinals),
        "ctc_text": unit.alignment_token,
        "unit_id": unit.unit_id,
        "alignment_token": unit.alignment_token,
        "canonical_span": list(unit.canonical_span),
    }
    return record, token


@pytest.mark.parametrize(
    ("surface", "ordinals"),
    [("K-Pop", [7, 9]), ("v-tuber", [18, 20, 21]),
     ("V-Up", [24, 26]), ("open-ai", [30, 32])],
)
def test_strict_noncontiguous_span_requires_exact_marked_processed_token(
        surface, ordinals):
    record, token = _strict_binding_fixture(surface, ordinals)
    assert resolve_processed_english_token([token], ordinals) is token
    validate_processed_english_token_binding(record, token)


def test_strict_binding_accepts_v10_surface_ctc_text_for_vtuber():
    record, token = _strict_binding_fixture("v-tuber", [18, 20, 21])
    record["ctc_text"] = "v-tuber"
    validate_processed_english_token_binding(record, token)


def test_strict_contiguous_span_remains_marker_free_and_nonhyphen_gaps_fail():
    record, token = _strict_binding_fixture("hello", [4, 5])
    token.pop("hyphen_separator_omitted", None)
    validate_processed_english_token_binding(record, None)

    record, token = _strict_binding_fixture("hello", [4, 6])
    token["hyphen_separator_omitted"] = True
    with pytest.raises(EnglishUnitError):
        validate_processed_english_token_binding(record, token)


@pytest.mark.parametrize("mutator", [
    lambda record, token: token.pop("hyphen_separator_omitted", None),
    lambda record, token: token.__setitem__("hyphen_separator_omitted", False),
    lambda record, token: token.__setitem__("source_ctc_ordinals", [7, 10]),
    lambda record, token: token["canonical_unit"].__setitem__("unit_id", "en-u9999"),
    lambda record, token: record.__setitem__("ctc_ordinal", 9),
])
def test_strict_noncontiguous_span_rejects_unbound_or_tampered_evidence(mutator):
    record, token = _strict_binding_fixture("K-Pop", [7, 9])
    mutator(record, token)
    resolved = resolve_processed_english_token([token], [7, 9])
    with pytest.raises(EnglishUnitError):
        validate_processed_english_token_binding(record, resolved)
