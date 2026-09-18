import sys
from pathlib import Path


from nvv_contract import (  # noqa: E402
    NVV_CONTRACT_SCHEMA,
    NVV_EXPECTED_SEQUENCE_REASON,
    audit_expected_nvv_sequence,
    audit_nvv_contract,
    build_nvv_contract,
    nvv_sequence,
)


TIERS = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")


def _tiers(*, raw="你 <BREATHING> 好", pinyin="ni3 <BREATHING> hao3",
           hanzi="你 <BREATHING> 好", words="ni3 <BREATHING> hao3",
           phones="n <BREATHING> h"):
    return dict(zip(TIERS, (raw, pinyin, hanzi, words, phones)))


def test_nvv_contract_preserves_order_and_repeated_occurrences():
    tiers = _tiers(raw="你 <BREATHING> <BREATHING> 好",
                   pinyin="ni3 <BREATHING> <BREATHING> hao3",
                   hanzi="你 <BREATHING> <BREATHING> 好",
                   words="ni3 <BREATHING> <BREATHING> hao3",
                   phones="n <BREATHING> <BREATHING> h")
    contract = build_nvv_contract(tiers)
    assert contract["schema"] == NVV_CONTRACT_SCHEMA
    assert contract["status"] == "verified"
    assert contract["sequences"]["words"] == ["<BREATHING>", "<BREATHING>"]
    assert audit_nvv_contract(tiers) == []


def test_nvv_contract_rejects_raw_only_and_sequence_mismatch():
    tiers = _tiers(pinyin="ni3 hao3", hanzi="你 好", words="ni3 hao3",
                   phones="n h")
    assert audit_nvv_contract(tiers) == ["cross_tier_nvv_sequence_mismatch"]

    tiers = _tiers(pinyin="ni3 <COUGH> hao3")
    assert audit_nvv_contract(tiers) == ["cross_tier_nvv_sequence_mismatch"]


def test_expected_nvv_sequence_requires_every_tier_to_match_repeated_occurrences():
    expected = ["<BREATHING>", "<BREATHING>", "<COUGH>"]
    tiers = _tiers(
        raw="你 <BREATHING> <BREATHING> <COUGH> 好",
        pinyin="ni3 <BREATHING> <BREATHING> <COUGH> hao3",
        hanzi="你 <BREATHING> <BREATHING> <COUGH> 好",
        words="ni3 <BREATHING> <BREATHING> <COUGH> hao3",
        phones="n <BREATHING> <BREATHING> <COUGH> h",
    )

    assert audit_expected_nvv_sequence(tiers, expected) == []


def test_expected_nvv_sequence_rejects_frozen_sequence_bypasses_and_mismatches():
    expected = ["<BREATHING>", "<COUGH>"]
    all_empty = _tiers(raw="你 好", pinyin="ni3 hao3", hanzi="你 好",
                       words="ni3 hao3", phones="n h")
    raw_only = _tiers(pinyin="ni3 hao3", hanzi="你 好", words="ni3 hao3",
                      phones="n h")
    wrong_order = _tiers(raw="你 <COUGH> <BREATHING> 好",
                         pinyin="ni3 <COUGH> <BREATHING> hao3",
                         hanzi="你 <COUGH> <BREATHING> 好",
                         words="ni3 <COUGH> <BREATHING> hao3",
                         phones="n <COUGH> <BREATHING> h")
    missing = _tiers(raw="你 <BREATHING> 好", pinyin="ni3 <BREATHING> hao3",
                     hanzi="你 <BREATHING> 好", words="ni3 <BREATHING> hao3",
                     phones="n <BREATHING> h")
    extra = _tiers(raw="你 <BREATHING> <COUGH> <LAUGHTER> 好",
                   pinyin="ni3 <BREATHING> <COUGH> <LAUGHTER> hao3",
                   hanzi="你 <BREATHING> <COUGH> <LAUGHTER> 好",
                   words="ni3 <BREATHING> <COUGH> <LAUGHTER> hao3",
                   phones="n <BREATHING> <COUGH> <LAUGHTER> h")

    for tiers in (all_empty, raw_only, wrong_order, missing, extra):
        assert audit_expected_nvv_sequence(tiers, expected) == [
            NVV_EXPECTED_SEQUENCE_REASON]


def test_expected_empty_sequence_keeps_all_empty_five_tier_data_valid():
    tiers = _tiers(raw="你 好", pinyin="ni3 hao3", hanzi="你 好",
                   words="ni3 hao3", phones="n h")

    assert audit_expected_nvv_sequence(tiers, []) == []


def test_nvv_sequence_only_treats_bare_canonical_uppercase_as_an_nvv():
    assert nvv_sequence("normal breathing exercises") == []
    assert nvv_sequence("BREATHING") == ["<BREATHING>"]
    assert nvv_sequence("<breathing> [Breathing]") == [
        "<BREATHING>", "<BREATHING>"]
