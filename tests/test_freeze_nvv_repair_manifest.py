import hashlib
import json
import sys
from pathlib import Path

import pytest



from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid  # noqa: E402
import freeze_nvv_repair_manifest as manifest  # noqa: E402
from freeze_nvv_repair_manifest import (  # noqa: E402
    MANIFEST_SCHEMA,
    freeze_20260910_nvv_production_manifest,
    freeze_nvv_repair_manifest,
)


TIERS = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")
BROKEN_LEADING_BREATHING = {
    "raw_text": "<sp1><BREATHING>， 海灵教育出版社高爱数学。",
    "pinyin": "<sp1>， hai3 ling2 jiao4 yu4。",
    "hanzi": "， 海 灵 教 育。",
    "words": "， hai3 ling2 jiao4 yu4。",
    "pinyin_phones": "， h ai3 l ing2 j iao4。",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_grid(root: Path, *, game: str = "persona", speaker: str = "Alice",
                stem: str = "line", labels: dict[str, str] | None = None) -> Path:
    labels = labels or BROKEN_LEADING_BREATHING
    path = root / game / speaker / f"{stem}.TextGrid"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = TextGrid(0.0, 1.0, [
        Tier(name, 0.0, 1.0, [Interval(0.0, 1.0, labels[name])])
        for name in TIERS
    ])
    write_textgrid(grid, path)
    return path


def _write_wav(root: Path, *, game: str = "persona", speaker: str = "Alice",
               stem: str = "line") -> Path:
    path = root / game / speaker / f"{stem}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture wav payload")
    return path


def _write_source_wav(root: Path, *, name: str = "line") -> Path:
    path = root / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"actual original source wav payload")
    return path.resolve()


def _record(source: Path, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "game": "persona",
        "speaker": "Alice",
        "stem": "line",
        "source_wav_path": str(source),
        "asr_mode": "fallback",
        "reference_mode": "fallback",
    }
    record.update(overrides)
    return record


def test_freeze_manifest_records_real_broken_pattern_without_rejecting_it(tmp_path: Path):
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    source = _write_source_wav(tmp_path / "source")
    grid = _write_grid(grids)
    wav = _write_wav(wavs)

    first = freeze_nvv_repair_manifest(
        accepted_textgrid_root=grids,
        paired_wav_root=wavs,
        records=[_record(source)],
        run_id="20260910T000000Z-nvv-repair",
        allowed_source_roots=[source.parent],
    )
    second = freeze_nvv_repair_manifest(
        accepted_textgrid_root=grids,
        paired_wav_root=wavs,
        records=[_record(source)],
        run_id="20260910T000000Z-nvv-repair",
        allowed_source_roots=[source.parent],
    )

    assert first == second
    assert first == [{
        "schema": MANIFEST_SCHEMA,
        "run_id": "20260910T000000Z-nvv-repair",
        "game": "persona",
        "speaker": "Alice",
        "stem": "line",
        "old_textgrid_relative_path": "persona/Alice/line.TextGrid",
        "old_wav_relative_path": "persona/Alice/line.wav",
        "textgrid_sha256": _sha256(grid),
        "wav_sha256": _sha256(wav),
        "source_wav_path": str(source),
        "source_wav_root": str(source.parent),
        "source_wav_sha256": _sha256(source),
        "asr_mode": "fallback",
        "reference_mode": "fallback",
        "expected_nvv_sequence": ["<BREATHING>"],
        "expected_nvv_positions": [
            {"tier": "raw_text", "occurrence_ordinal": 0,
             "label": "<BREATHING>"},
        ],
        "observed_nvv_sequences": {
            "raw_text": ["<BREATHING>"],
            "pinyin": [], "hanzi": [], "words": [], "pinyin_phones": [],
        },
        "old_failure_class": "raw_only",
    }]


@pytest.mark.parametrize(
    ("labels", "has_wav", "source_exists", "match"),
    [
        ({name: "你好" for name in TIERS}, True, True, "no NVV"),
        (None, False, True, "paired WAV missing"),
        (None, True, False, "source WAV missing"),
    ],
)
def test_freeze_manifest_rejects_invalid_selected_records(
    tmp_path: Path, labels: dict[str, str] | None, has_wav: bool,
    source_exists: bool, match: str,
):
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    _write_grid(grids, labels=labels)
    wavs.mkdir(parents=True)
    if has_wav:
        _write_wav(wavs)
    source = ((tmp_path / "source" / "missing.wav").resolve()
              if not source_exists else _write_source_wav(tmp_path / "source"))
    source.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match=match):
        freeze_nvv_repair_manifest(
            accepted_textgrid_root=grids,
            paired_wav_root=wavs,
            records=[_record(source)],
            run_id="20260910T000000Z-nvv-repair",
            allowed_source_roots=[source.parent],
        )


def test_freeze_manifest_rejects_path_escape_and_duplicate_keys(tmp_path: Path):
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    source = _write_source_wav(tmp_path / "source")
    _write_grid(grids)
    _write_wav(wavs)

    with pytest.raises(ValueError, match="path escape"):
        freeze_nvv_repair_manifest(
            accepted_textgrid_root=grids,
            paired_wav_root=wavs,
            records=[_record(source, speaker="../escape")],
            run_id="20260910T000000Z-nvv-repair",
            allowed_source_roots=[source.parent],
        )
    with pytest.raises(ValueError, match="duplicate manifest key"):
        freeze_nvv_repair_manifest(
            accepted_textgrid_root=grids,
            paired_wav_root=wavs,
            records=[_record(source), _record(source)],
            run_id="20260910T000000Z-nvv-repair",
            allowed_source_roots=[source.parent],
        )


def test_freeze_manifest_rejects_claimed_source_hash_that_differs_from_real_file(
    tmp_path: Path,
):
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    source = _write_source_wav(tmp_path / "source")
    _write_grid(grids)
    _write_wav(wavs)

    with pytest.raises(ValueError, match="source WAV hash mismatch"):
        freeze_nvv_repair_manifest(
            accepted_textgrid_root=grids,
            paired_wav_root=wavs,
            records=[_record(source, source_wav_sha256="0" * 64)],
            run_id="20260910T000000Z-nvv-repair",
            allowed_source_roots=[source.parent],
        )


def test_freeze_manifest_scan_hashes_real_source_wav(tmp_path: Path):
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    source_root = tmp_path / "source"
    _write_grid(grids)
    _write_wav(wavs)
    source = _write_wav(source_root)
    _write_grid(grids, stem="not-nvv", labels={name: "你好" for name in TIERS})
    _write_wav(wavs, stem="not-nvv")
    _write_wav(source_root, stem="not-nvv")
    output = tmp_path / "manifest.jsonl"

    rows = freeze_nvv_repair_manifest(
        accepted_textgrid_root=grids,
        paired_wav_root=wavs,
        records=None,
        run_id="20260910T000000Z-nvv-repair",
        output_path=output,
        source_wav_root=source_root,
        asr_mode="fallback",
        reference_mode="fallback",
    )

    assert [row["stem"] for row in rows] == ["line"]
    assert rows[0]["source_wav_path"] == str(source.resolve())
    assert rows[0]["source_wav_root"] == str(source_root.resolve())
    assert rows[0]["source_wav_sha256"] == _sha256(source)
    assert output.read_text(encoding="utf-8") == json.dumps(
        rows[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def test_freeze_manifest_rejects_record_source_outside_vetted_root(tmp_path: Path):
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    source = _write_source_wav(tmp_path / "untrusted")
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    _write_grid(grids)
    _write_wav(wavs)

    with pytest.raises(ValueError, match="outside allowed source roots"):
        freeze_nvv_repair_manifest(
            accepted_textgrid_root=grids,
            paired_wav_root=wavs,
            records=[_record(source)],
            run_id="20260910T000000Z-nvv-repair",
            allowed_source_roots=[trusted],
        )


def _production_fixture(tmp_path: Path) -> dict[str, object]:
    grids = tmp_path / "aligned"
    wavs = tmp_path / "gamesl"
    baijing_source_root = tmp_path / "baijing-source"
    fresh4_staging_root = tmp_path / "fresh4-staging"
    _write_grid(grids, game="baijing", speaker="B", stem="lineb")
    _write_wav(wavs, game="baijing", speaker="B", stem="lineb")
    baijing_source = baijing_source_root / "lineb.wav"
    baijing_source.parent.mkdir()
    baijing_source.write_bytes(b"baijing original")
    receipts: dict[str, Path] = {}
    for game, speaker, stem in (
        ("persona", "P", "linep"),
        ("punishing_gray_raven", "G", "lineg"),
        ("reverse1999", "R", "liner"),
    ):
        _write_grid(grids, game=game, speaker=speaker, stem=stem)
        _write_wav(wavs, game=game, speaker=speaker, stem=stem)
        source = fresh4_staging_root / f"{game}-{stem}.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"{game} staged source".encode())
        receipt = tmp_path / f"{game}-rebuild.json"
        receipt.write_text(json.dumps({
            "game": game,
            "items": [{"stem": stem, "audio": str(source), "text_mode": "asr"}],
        }), encoding="utf-8")
        receipts[game] = receipt
    return {
        "accepted_textgrid_root": grids, "paired_wav_root": wavs,
        "baijing_source_wav_root": baijing_source_root,
        "fresh4_staging_root": fresh4_staging_root,
        "fresh4_receipts": receipts,
        "expected_game_counts": {
            "baijing": 1, "persona": 1, "punishing_gray_raven": 1,
            "reverse1999": 1,
        },
    }


def test_production_profile_builds_records_from_receipt_and_enforces_counts(
    tmp_path: Path,
):
    fixture = _production_fixture(tmp_path)

    rows = freeze_20260910_nvv_production_manifest(
        run_id="run",
        expected_total_occurrences=4,
        **fixture,
    )

    assert [row["game"] for row in rows] == [
        "baijing", "persona", "punishing_gray_raven", "reverse1999"]


def test_production_profile_hard_fails_when_expected_count_is_not_conserved(
    tmp_path: Path,
):
    fixture = _production_fixture(tmp_path)
    fixture["expected_game_counts"] = {
        "baijing": 2, "persona": 1, "punishing_gray_raven": 1,
        "reverse1999": 1,
    }

    with pytest.raises(ValueError, match="game count conservation failed"):
        freeze_20260910_nvv_production_manifest(
            run_id="run",
            expected_total_occurrences=4,
            **fixture,
        )


def test_production_profile_does_not_write_when_occurrence_count_drifts(
    tmp_path: Path,
):
    fixture = _production_fixture(tmp_path)
    output = tmp_path / "must-not-exist.jsonl"

    with pytest.raises(ValueError, match="NVV occurrence conservation failed"):
        freeze_20260910_nvv_production_manifest(
            run_id="run",
            output_path=output,
            expected_total_occurrences=5,
            **fixture,
        )

    assert not output.exists()


def test_production_profile_ignores_unrelated_game_but_rejects_affected_symlink(
    tmp_path: Path,
):
    fixture = _production_fixture(tmp_path)
    unrelated = Path(fixture["accepted_textgrid_root"]) / "unrelated"
    unrelated.mkdir()
    (unrelated / "do-not-follow").symlink_to(tmp_path / "missing")

    rows = freeze_20260910_nvv_production_manifest(
        run_id="run", expected_total_occurrences=4, **fixture)
    assert len(rows) == 4

    affected_link = Path(fixture["accepted_textgrid_root"]) / "persona" / "bad-link"
    affected_link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="game tree contains symlink"):
        freeze_20260910_nvv_production_manifest(
            run_id="run", expected_total_occurrences=4, **fixture)


def test_snapshot_production_uses_fixed_snapshot_authority_and_marks_baijing_wav_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    roots = {game: tmp_path / game for game in manifest.PRODUCTION_20260910_GAMES}
    baijing_source_root = tmp_path / "baijing-sources"
    fresh_source_root = tmp_path / "fresh-staging"
    receipts: dict[str, Path] = {}
    for game, root in roots.items():
        grid = _write_grid(root, game="nested", speaker="speaker", stem="line")
        grid.rename(root / "line.TextGrid")
        (root / "nested" / "speaker").rmdir()
        (root / "nested").rmdir()
        if game == "baijing":
            normal = _write_grid(root, game="nested", speaker="speaker", stem="normal",
                                 labels={name: "正常" for name in TIERS})
            normal.rename(root / "normal.TextGrid")
            (root / "nested" / "speaker").rmdir()
            (root / "nested").rmdir()
        source = ((baijing_source_root if game == "baijing" else fresh_source_root / game)
                  / "line.wav")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(game.encode())
        if game != "baijing":
            receipt = tmp_path / f"{game}.json"
            receipt.write_text(json.dumps({"game": game, "items": [{
                "stem": "line", "audio": str(source),
                "source": f"/original/{game}/Speaker/line.wav", "text_mode": "asr",
            }]}), encoding="utf-8")
            receipts[game] = receipt
    baijing_stage = tmp_path / "baijing-stage.json"
    baijing_stage.write_text(json.dumps({"line": {
        "audio": "/original/baijing/DistinctSpeaker/line.wav",
    }}), encoding="utf-8")
    monkeypatch.setattr(manifest, "SNAPSHOT_20260910", {
        "baijing": {"textgrid_root": roots["baijing"],
                    "source_wav_root": baijing_source_root,
                    "stage_manifest": baijing_stage},
        **{game: {"textgrid_root": roots[game], "receipt": receipts[game]}
           for game in receipts},
    })
    monkeypatch.setattr(manifest, "EXPECTED_20260910_GAME_COUNTS",
                        {game: 1 for game in roots})
    monkeypatch.setattr(manifest, "EXPECTED_20260910_OCCURRENCE_COUNT", 4)
    monkeypatch.setattr(manifest, "SNAPSHOT_FRESH4_STAGING_ROOT", fresh_source_root)
    gamesl = tmp_path / "old-gamesl"
    gamesl.mkdir()
    monkeypatch.setattr(manifest, "SNAPSHOT_FRESH4_GAMESL_ROOT", gamesl)

    rows = manifest.freeze_20260910_nvv_snapshot_manifest(run_id="snapshot-test")

    assert len(rows) == 4
    assert rows[0]["old_wav_snapshot_status"] == "unavailable_due_external_cleanup"
    assert rows[0]["speaker"] == "DistinctSpeaker"
    assert all(row["expected_nvv_sequence"] == ["<BREATHING>"] for row in rows)
    assert all("source_wav_root" in row for row in rows)
