import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid  # noqa: E402
from verify_nvv_rerun_manifest import verify_nvv_rerun_manifest  # noqa: E402


TIERS = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")
TEMPORAL_TIERS = ("hanzi", "words", "pinyin_phones")


def _labels(*, raw="<BREATHING> 你好", pinyin="<BREATHING> ni3 hao3",
            hanzi="<BREATHING> 你 好", words="<BREATHING> ni3 hao3",
            phones="<BREATHING> n i h ao"):
    return dict(zip(TIERS, (raw, pinyin, hanzi, words, phones)))


def _write_grid(root: Path, *, game: str = "persona", speaker: str = "Alice",
                stem: str = "line", labels: dict[str, str] | None = None,
                timed_intervals: dict[str, list[Interval]] | None = None) -> Path:
    labels = labels or _labels()
    timed_intervals = timed_intervals or {}
    tiers: list[Tier] = []
    for name in TIERS:
        intervals = timed_intervals.get(name)
        if intervals is None and name in TEMPORAL_TIERS and labels[name].startswith("<BREATHING>"):
            remainder = labels[name].removeprefix("<BREATHING>").strip()
            intervals = [
                Interval(0.0, 0.25, "<BREATHING>"),
                Interval(0.25, 1.0, remainder),
            ]
        if intervals is None:
            intervals = [Interval(0.0, 1.0, labels[name])]
        tiers.append(Tier(name, 0.0, 1.0, intervals))
    path = root / game / speaker / f"{stem}.TextGrid"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_textgrid(TextGrid(0.0, 1.0, tiers), path)
    return path


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ), encoding="utf-8")
    return path


def _row(*, game: str = "persona", speaker: str = "Alice", stem: str = "line",
         expected: list[str] | None = None) -> dict[str, object]:
    return {
        "game": game,
        "speaker": speaker,
        "stem": stem,
        "expected_nvv_sequence": expected if expected is not None else ["<BREATHING>"],
    }


def _outputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "report_path": tmp_path / "report.jsonl",
        "accepted_manifest_path": tmp_path / "accepted.jsonl",
        "rejected_manifest_path": tmp_path / "rejected.jsonl",
        "accepted_stems_path": tmp_path / "accepted.stems",
        "rejected_stems_path": tmp_path / "rejected.stems",
    }


def _verify(tmp_path: Path, root: Path, manifest: Path):
    return verify_nvv_rerun_manifest(
        manifest_path=manifest, textgrid_root=root, **_outputs(tmp_path))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_accepts_exact_frozen_sequence_and_writes_one_to_one_outputs(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root)
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row()])

    result = _verify(tmp_path, root, manifest)

    assert result == {"accepted": 1, "rejected": 0, "total": 1}
    output = _outputs(tmp_path)
    assert _read_jsonl(output["report_path"]) == [{
        "expected_nvv_sequence": ["<BREATHING>"],
        "game": "persona",
        "observed_nvv_sequences": {tier: ["<BREATHING>"] for tier in TIERS},
        "path": str(root / "persona" / "Alice" / "line.TextGrid"),
        "reasons": [],
        "speaker": "Alice",
        "status": "accepted",
        "stem": "line",
    }]
    assert _read_jsonl(output["accepted_manifest_path"]) == [_row()]
    assert output["rejected_manifest_path"].read_text(encoding="utf-8") == ""
    assert output["accepted_stems_path"].read_text(encoding="utf-8") == "persona/Alice/line\n"
    assert output["rejected_stems_path"].read_text(encoding="utf-8") == ""


def test_rejects_frozen_nvv_when_every_final_tier_is_empty(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root, labels=_labels(raw="你好", pinyin="ni3 hao3", hanzi="你 好",
                                     words="ni3 hao3", phones="n i h ao"))
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row()])

    assert _verify(tmp_path, root, manifest) == {"accepted": 0, "rejected": 1, "total": 1}
    report = _read_jsonl(_outputs(tmp_path)["report_path"])
    assert report[0]["status"] == "rejected"
    assert report[0]["reasons"] == ["frozen_expected_nvv_sequence_mismatch"]
    assert _read_jsonl(_outputs(tmp_path)["rejected_manifest_path"]) == [_row()]


def test_rejects_raw_only_nvv_even_when_frozen_sequence_matches_raw(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root, labels=_labels(pinyin="ni3 hao3", hanzi="你 好",
                                     words="ni3 hao3", phones="n i h ao"))
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row()])

    _verify(tmp_path, root, manifest)

    report = _read_jsonl(_outputs(tmp_path)["report_path"])
    assert report[0]["status"] == "rejected"
    assert report[0]["reasons"] == ["cross_tier_nvv_sequence_mismatch",
                                      "frozen_expected_nvv_sequence_mismatch"]


def test_missing_rerun_textgrid_is_a_rejected_row_and_preserves_the_manifest(tmp_path: Path):
    root = tmp_path / "rerun"
    root.mkdir()
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row(stem="missing")])

    assert _verify(tmp_path, root, manifest) == {"accepted": 0, "rejected": 1, "total": 1}

    output = _outputs(tmp_path)
    assert _read_jsonl(output["report_path"]) == [{
        "expected_nvv_sequence": ["<BREATHING>"],
        "game": "persona",
        "observed_nvv_sequences": {tier: [] for tier in TIERS},
        "path": str(root / "persona" / "Alice" / "missing.TextGrid"),
        "reasons": ["missing_rerun_textgrid"],
        "speaker": "Alice",
        "status": "rejected",
        "stem": "missing",
    }]
    assert _read_jsonl(output["rejected_manifest_path"]) == [_row(stem="missing")]
    assert output["rejected_stems_path"].read_text(encoding="utf-8") == (
        "persona/Alice/missing\n")


def test_multiple_missing_rows_mix_deterministically_with_accepted_and_rejected_rows(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root, stem="accepted")
    _write_grid(root, stem="raw-only", labels=_labels(
        pinyin="ni3 hao3", hanzi="你 好", words="ni3 hao3", phones="n i h ao"))
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [
        _row(stem="missing-z"),
        _row(stem="raw-only"),
        _row(stem="accepted"),
        _row(stem="missing-a"),
    ])

    assert _verify(tmp_path, root, manifest) == {"accepted": 1, "rejected": 3, "total": 4}

    output = _outputs(tmp_path)
    report = _read_jsonl(output["report_path"])
    assert [(row["stem"], row["status"], row["reasons"]) for row in report] == [
        ("accepted", "accepted", []),
        ("missing-a", "rejected", ["missing_rerun_textgrid"]),
        ("missing-z", "rejected", ["missing_rerun_textgrid"]),
        ("raw-only", "rejected", ["cross_tier_nvv_sequence_mismatch",
                                     "frozen_expected_nvv_sequence_mismatch"]),
    ]
    assert len(_read_jsonl(output["accepted_manifest_path"])) + len(
        _read_jsonl(output["rejected_manifest_path"])) == 4
    assert output["accepted_stems_path"].read_text(encoding="utf-8") == (
        "persona/Alice/accepted\n")
    assert output["rejected_stems_path"].read_text(encoding="utf-8") == (
        "persona/Alice/missing-a\npersona/Alice/missing-z\npersona/Alice/raw-only\n")


def test_rejects_identical_nvv_sequence_when_temporal_owners_are_shifted(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root, timed_intervals={
        "hanzi": [Interval(0.0, 0.25, "<BREATHING>"), Interval(0.25, 1.0, "你 好")],
        "words": [Interval(0.25, 0.50, "<BREATHING>"), Interval(0.50, 1.0, "ni3 hao3")],
        "pinyin_phones": [Interval(0.0, 0.25, "<BREATHING>"), Interval(0.25, 1.0, "n i h ao")],
    })
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row()])

    _verify(tmp_path, root, manifest)

    report = _read_jsonl(_outputs(tmp_path)["report_path"])
    assert report[0]["status"] == "rejected"
    assert report[0]["reasons"] == ["nvv_temporal_owner_mismatch"]


def test_rejects_punctuation_overlapping_an_otherwise_matching_nvv_owner(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root, timed_intervals={
        "hanzi": [Interval(0.0, 0.25, "<BREATHING>"), Interval(0.25, 1.0, "你 好")],
        "words": [
            Interval(0.0, 0.25, "<BREATHING>"),
            Interval(0.05, 0.20, "，"),
            Interval(0.25, 1.0, "ni3 hao3"),
        ],
        "pinyin_phones": [Interval(0.0, 0.25, "<BREATHING>"), Interval(0.25, 1.0, "n i h ao")],
    })
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row()])

    _verify(tmp_path, root, manifest)

    report = _read_jsonl(_outputs(tmp_path)["report_path"])
    assert report[0]["status"] == "rejected"
    assert report[0]["reasons"] == ["nvv_temporal_owner_mismatch"]


@pytest.mark.parametrize(
    ("rows", "setup", "match"),
    [
        ([_row(), _row()], lambda root: _write_grid(root), "duplicate manifest key"),
        ([_row(speaker="../escape")], lambda root: _write_grid(root), "path escape"),
    ],
)
def test_hard_fails_duplicate_escape_or_missing_manifest_target(
    tmp_path: Path, rows: list[dict[str, object]], setup, match: str,
):
    root = tmp_path / "rerun"
    root.mkdir()
    setup(root)
    manifest = _write_manifest(tmp_path / "frozen.jsonl", rows)

    with pytest.raises(ValueError, match=match):
        _verify(tmp_path, root, manifest)

    assert not _outputs(tmp_path)["report_path"].exists()


def test_hard_fails_textgrid_not_present_in_frozen_manifest(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root)
    _write_grid(root, stem="unlisted")
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [_row()])

    with pytest.raises(ValueError, match="not present in frozen manifest"):
        _verify(tmp_path, root, manifest)


def test_outputs_are_deterministically_sorted_by_game_speaker_and_stem(tmp_path: Path):
    root = tmp_path / "rerun"
    _write_grid(root, game="zzz", speaker="B", stem="z")
    _write_grid(root, game="persona", speaker="A", stem="a")
    manifest = _write_manifest(tmp_path / "frozen.jsonl", [
        _row(game="zzz", speaker="B", stem="z"),
        _row(game="persona", speaker="A", stem="a"),
    ])

    result = _verify(tmp_path, root, manifest)

    assert result == {"accepted": 2, "rejected": 0, "total": 2}
    output = _outputs(tmp_path)
    assert [row["stem"] for row in _read_jsonl(output["report_path"])] == ["a", "z"]
    assert output["accepted_stems_path"].read_text(encoding="utf-8") == (
        "persona/A/a\nzzz/B/z\n")
