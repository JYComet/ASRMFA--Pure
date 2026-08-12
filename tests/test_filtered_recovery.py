from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pipeline_utils import is_pinyin_syllable
from scripts.postprocess_textgrids import Interval, Tier, TextGrid, detect_issues
from scripts.run_pipeline import (
    FILTERED_RECOVERY_SCHEMA,
    import_filtered_recovery_assets,
    validate_filtered_recovery_manifest,
    validate_filtered_recovery_partition,
)

FIXTURE_MISMATCH = {
    "declared_sha256": "9e6f2e7856971ee4e7dd07d9cdf3f1a017aabc46497d902ea4bd24f3bd4d05f2",
    "actual_sha256": "08c874b03104c260e29b4df4a4d7c22d28e107eaa443d3b66687639cdf8ee635",
}


def test_pinyin_classification_rejects_target_fragments_but_keeps_jin():
    assert is_pinyin_syllable("jin1")
    assert is_pinyin_syllable("JIN1")
    assert not is_pinyin_syllable("target1")
    assert not is_pinyin_syllable("target2")


def test_filtered_recovery_partition_is_exact_and_disjoint():
    frozen = [f"s{i:03d}" for i in range(224)]
    accepted = [f"a{i:03d}" for i in range(776)]
    partition = validate_filtered_recovery_partition(frozen, accepted, frozen[:10], frozen[10:])
    assert partition["source"] == partition["eligible"] == 224
    assert partition["exclusions"] == 0
    assert partition["output"] == 10 and partition["filtered"] == 214


def test_filtered_recovery_partition_derives_non224_denominator():
    frozen = ["f0", "f1", "f2"]
    accepted = ["a0"]
    partition = validate_filtered_recovery_partition(frozen, accepted, ["f0"], ["f1", "f2"])
    assert partition["source"] == partition["eligible"] == 3


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda f, a: (f + [f[0]], a), "duplicate"),
        (lambda f, a: (f + ["outside"], a), "union mismatch"),
        (lambda f, a: (f, a + [f[0]]), "intersection"),
    ],
)
def test_filtered_recovery_partition_rejects_bad_sets(mutator, message):
    frozen = [f"s{i:03d}" for i in range(224)]
    accepted = [f"a{i:03d}" for i in range(776)]
    bad_frozen, bad_accepted = mutator(frozen, accepted)
    with pytest.raises(ValueError, match=message):
        validate_filtered_recovery_partition(bad_frozen, bad_accepted, [], frozen)


def test_filtered_recovery_manifest_rejects_nonfrozen_and_wrong_mismatch():
    frozen = [f"s{i:03d}" for i in range(224)]
    accepted = [f"a{i:03d}" for i in range(776)]
    base = {
        "schema": FILTERED_RECOVERY_SCHEMA,
        "stems": frozen[:3], "source": 3, "eligible": 3, "exclusions": 0,
        "declared_vs_actual_inner_receipt": FIXTURE_MISMATCH,
    }
    assert validate_filtered_recovery_manifest(base, frozen, accepted)["count"] == 3
    bad = dict(base, stems=["outside"], source=1, eligible=1)
    with pytest.raises(ValueError, match="outside frozen"):
        validate_filtered_recovery_manifest(bad, frozen, accepted)
    bad = dict(base, declared_vs_actual_inner_receipt={"declared_sha256": "x", "actual_sha256": "y"})
    with pytest.raises(ValueError, match="known value"):
        validate_filtered_recovery_manifest(bad, frozen, accepted)


def test_filtered_recovery_import_records_copy_hashes_and_rejects_alias(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"recovery")
    rows = import_filtered_recovery_assets({"ctc": source}, tmp_path / "import", allowlist={"ctc"})
    assert rows[0]["sha256"]
    assert rows[0]["source_inode"] != rows[0]["destination_inode"]
    with pytest.raises(ValueError, match="fresh"):
        import_filtered_recovery_assets({"ctc": source}, tmp_path / "import", allowlist={"ctc"})


def test_word_in_silence_detector_explicit_disable_wins_over_strict_qc():
    words = Tier("words", 0.0, 1.0, [Interval(0.0, 1.0, "jin1")])
    phones = Tier("phones", 0.0, 1.0, [Interval(0.0, 1.0, "sil")])
    tg = TextGrid(0.0, 1.0, [words, phones])
    args = SimpleNamespace(filter_min_word_dur_sec=0.0, filter_min_word_sec=0.0,
                            filter_min_phone_coverage=0.0, filter_edge_gap_sec=99,
                            filter_long_word_sec=99, filter_flank_silence_sec=0.0,
                            filter_word_energy_ratio=2.0,
                            enable_word_in_silence_filter=False,
                            filter_short_phone=False, filter_short_phone_sec=0.015,
                            filter_long_consonant_sec=999, filter_long_vowel_sec=999)
    # Missing phone is an independent issue; the assertion is specifically
    # that no word_in_silence issue is emitted when the detector is disabled.
    issues = detect_issues(tg, args)
    assert all(row.get("rule") != "word_in_silence" for row in issues)
