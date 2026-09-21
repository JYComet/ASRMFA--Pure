from pathlib import Path

import pytest

from scripts.align_japanese_mfa import (
    build_locked_alias_dictionary,
    isolated_mfa_command,
    parse_raw_textgrid,
    validate_native_inventory,
    handle_align,
    map_raw_intervals_to_samples,
)


def _audio_receipt(uid: str = "u1") -> dict:
    return {
        "schema": "audio-transform-receipt-v2", "uid": uid,
        "source": {"path": "/fixture/source.wav", "sha256": "source", "sample_rate": 8000, "frames": 4000},
        "alignment": {"path": "/fixture/alignment.wav", "sha256": "alignment", "sample_rate": 16000, "frames": 8000},
        "train": {"path": "/fixture/train.wav", "sha256": "train", "sample_rate": 24000, "frames": 12000},
        "sample_transform": {"source_start": 10, "output_start": 0, "source_rate": 8000, "target_rate": 16000, "frame_policy": "round_half_up_v1"},
        "alignment_transform": {"source_start": 10, "output_start": 0, "source_rate": 8000, "target_rate": 16000, "frame_policy": "round_half_up_v1"},
        "train_transform": {"source_start": 0, "output_start": 0, "source_rate": 8000, "target_rate": 24000, "frame_policy": "round_half_up_v1"},
    }


def test_raw_interval_carries_token_and_receipt_projected_three_axes():
    rows = map_raw_intervals_to_samples(
        [{"raw_interval_id": 7, "xmin": 0.1, "xmax": 0.2, "text": "t"}],
        offset_sample=0, sample_rate=16000, ownership=(0, 8000), alias="ju_000000",
        language="ja", run_id="ja-run", unit_id="unit-1", token_id="tok-1",
        uid="u1",
        audio_receipt=_audio_receipt(),
    )
    phone = rows[0]
    assert phone["uid"] == "u1"
    assert phone["token_id"] == "tok-1"
    assert phone["alignment_axis"] == {"start_sample": 1600, "end_sample": 3200, "sample_rate": 16000,
                                       "artifact": {"path": "/fixture/alignment.wav", "sha256": "alignment"}}
    assert phone["source_axis"]["start_sample"] == 810
    assert phone["source_axis"]["end_sample"] == 1610
    assert phone["training_axis"]["start_sample"] == 2430
    assert phone["training_axis"]["end_sample"] == 4830
    assert phone["source_axis"]["transform"]["source_start"] == 10


def test_raw_seconds_and_receipt_projection_use_half_up_at_boundaries():
    rows = map_raw_intervals_to_samples(
        [{"raw_interval_id": 8, "xmin": "0.00003125", "xmax": "0.00009375", "text": "t"}],
        offset_sample=0, sample_rate=16000, ownership=(0, 8000), alias="ju_000000",
        language="ja", run_id="ja-run", unit_id="unit-1", token_id="tok-1",
        uid="u1",
        audio_receipt=_audio_receipt(),
    )
    phone = rows[0]
    assert (phone["start_sample"], phone["end_sample"]) == (1, 2)
    assert phone["source_axis"]["start_sample"] == 11
    assert phone["training_axis"]["start_sample"] == 33


def test_locked_dictionary_has_one_ascii_alias_row_per_occurrence(tmp_path: Path):
    rows = [
        {"alias": "ju_000000", "surface": "東京", "reading": "とうきょう", "pronunciation": ["t", "oː", "c", "oː"]},
        {"alias": "eu_000001", "surface": "game", "reading": "game", "pronunciation": ["G", "EY1", "M"]},
    ]
    dictionary = tmp_path / "locked.dict"
    alias_map = tmp_path / "alias_map.jsonl"
    result = build_locked_alias_dictionary(rows, dictionary, alias_map)
    assert result["aliases"] == ["ju_000000", "eu_000001"]
    assert dictionary.read_text(encoding="utf-8").splitlines() == [
        "ju_000000 t oː c oː", "eu_000001 G EY1 M"
    ]
    assert alias_map.read_text(encoding="utf-8").count("\n") == 2


def test_raw_textgrid_preserves_interval_ids_and_rejects_cardinality(tmp_path: Path):
    grid = tmp_path / "x.TextGrid"
    grid.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\n'
        'item []:\nitem [1]:\nclass = "IntervalTier"\nname = "words"\n'
        'xmin = 0\nxmax = 1\nintervals: size = 1\n'
        'intervals [1]:\nxmin = 0\nxmax = 1\ntext = "ju_000000"\n'
        'item [2]:\nclass = "IntervalTier"\nname = "phones"\n'
        'xmin = 0\nxmax = 1\nintervals: size = 2\n'
        'intervals [1]:\nxmin = 0\nxmax = 0.4\ntext = "t"\n'
        'intervals [2]:\nxmin = 0.4\nxmax = 1\ntext = "oː"\n', encoding="utf-8"
    )
    parsed = parse_raw_textgrid(grid, expected_aliases=["ju_000000"])
    assert [p["raw_interval_id"] for p in parsed["phones"]] == [1, 2]
    with pytest.raises(ValueError, match="cardinality"):
        parse_raw_textgrid(grid, expected_aliases=["ju_000000", "ju_000001"])


def test_inventory_check_rejects_foreign_language_phone():
    with pytest.raises(ValueError, match="inventory"):
        validate_native_inventory(["t", "EY1"], {"t", "oː"}, language="Japanese")


def test_mfa_command_has_fresh_root_no_tokenization_and_dither_zero(tmp_path: Path):
    command, env = isolated_mfa_command(
        corpus_dir=tmp_path / "corpus", dictionary=tmp_path / "locked.dict",
        acoustic_model=tmp_path / "japanese.zip", output_dir=tmp_path / "out",
        temporary_directory=tmp_path / "tmp", runtime_python=Path("/env/mfa/bin/python"),
    )
    assert "--no_tokenization" in command
    assert "--dither" in command and command[command.index("--dither") + 1] == "0.0"
    assert env["MFA_ROOT_DIR"] == str(tmp_path / "tmp" / "mfa_root")
    assert env["PATH"].split(":", 1)[0] == "/env/mfa/bin"


def _grid(path: Path, alias: str = "ju_000000") -> None:
    path.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n'
        'item [1]:\nclass = "IntervalTier"\nname = "words"\nxmin = 0\nxmax = 1\nintervals: size = 1\n'
        f'intervals [1]:\nxmin = 0\nxmax = 1\ntext = "{alias}"\n'
        'item [2]:\nclass = "IntervalTier"\nname = "phones"\nxmin = 0\nxmax = 1\nintervals: size = 1\n'
        'intervals [1]:\nxmin = 0\nxmax = 1\ntext = "t"\n', encoding="utf-8"
    )


def test_align_stage_runs_prepared_fixture_and_writes_strict_ledger(tmp_path: Path):
    grid = tmp_path / "fixture.TextGrid"; _grid(grid)
    def fake_runner(run, run_root, locked):
        return {"status": "COMPLETE", "textgrid": str(grid)}
    run = {"run_id": "ja_run_0000", "language": "ja", "unit_ids": ["u0"],
           "aliases": [{"alias": "ju_000000", "token_id": "tok-0", "surface": "東京", "reading": "とうきょう", "pronunciation": ["t"]}],
           "ownership_start_sample": 0, "ownership_end_sample": 16000,
           "context_start_sample": 0, "context_end_sample": 16000, "sample_rate": 16000,
           "audio_receipt": _audio_receipt()}
    stage = tmp_path / "stages" / "align"
    result = handle_align({"align": {"uid": "u1", "runs": [run], "mfa_runner": fake_runner}}, stage)
    assert result.status == "COMPLETE"
    payload = __import__("json").loads((stage / "strict_ja_mfa.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "strict-ja-mfa-v2"
    assert payload["runs"][0]["phones"][0]["raw_interval_id"] == 1
    assert payload["runs"][0]["phones"][0]["token_id"] == "tok-0"


def test_align_stage_rejects_word_alias_cardinality_mismatch(tmp_path: Path):
    grid = tmp_path / "bad.TextGrid"; _grid(grid, alias="ju_999999")
    def fake_runner(run, run_root, locked):
        return {"status": "COMPLETE", "textgrid": str(grid)}
    run = {"run_id": "ja_run_0000", "language": "ja", "unit_ids": ["u0"],
           "aliases": [{"alias": "ju_000000", "pronunciation": ["t"]}],
           "ownership_start_sample": 0, "ownership_end_sample": 16000,
           "context_start_sample": 0, "context_end_sample": 16000, "sample_rate": 16000}
    result = handle_align({"align": {"uid": "u1", "runs": [run], "mfa_runner": fake_runner}}, tmp_path / "stages" / "align")
    assert result.status == "REJECTED"
    assert (tmp_path / "stages" / "align" / "strict_ja_mfa.json").is_file()
    rejected = __import__("json").loads((tmp_path / "stages" / "align" / "strict_ja_mfa.json").read_text(encoding="utf-8"))
    assert rejected["ledger"]["rejected"] == ["u0"]
