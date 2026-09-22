from __future__ import annotations

import re
from pathlib import Path

from scripts.ja_tts_export import build_five_track_tiers, export_tts_artifacts, render_textgrid


def training_record_v2_fixture(tmp_path: Path, *, text: str = ' 「コー」 game ', phone_bounds: list[tuple[int, int]] | None = None) -> dict:
    bounds = phone_bounds or [(0, 4000), (4000, 8000), (8000, 12000), (12000, 16000)]
    japanese_text = text.rstrip() if "game" not in text else text[:text.index("game")]
    english_text = "game" if "game" in text else ""
    serialized_surface = japanese_text + english_text
    return {
        "schema": "tts-training-record-v2", "textgrid_schema": "five-track-textgrid-v1",
        "uid": "five-track", "sample_rate": 16000, "frame_count": 16000,
        "train_wav": {"path": str(tmp_path / "train.wav")}, "alignment_wav": {"path": str(tmp_path / "alignment.wav")},
        "words": [
            {"unit_id": "w0", "source_text": japanese_text, "kana": "コー", "language": "ja", "start_sample": bounds[0][0], "end_sample": bounds[1][1]},
            {"unit_id": "w1", "source_text": english_text, "kana": "", "language": "en", "start_sample": bounds[2][0], "end_sample": bounds[3][1]},
        ],
        "native_phones": [
            {"phone_id": "p0", "language": "ja", "native_phone": "k", "phone_kana": "コ", "phone_tone": "H", "start_sample": bounds[0][0], "end_sample": bounds[0][1]},
            {"phone_id": "p1", "language": "ja", "native_phone": "o", "phone_kana": "コ|ー", "phone_tone": "H|L", "start_sample": bounds[1][0], "end_sample": bounds[1][1]},
            {"phone_id": "p2", "language": "en", "native_phone": "G", "phone_kana": "", "phone_tone": "NA", "start_sample": bounds[2][0], "end_sample": bounds[2][1]},
            {"phone_id": "p3", "language": "en", "native_phone": "EY1", "phone_kana": "", "phone_tone": "NA", "start_sample": bounds[3][0], "end_sample": bounds[3][1]},
        ],
        "moras": [], "basic_phones": [], "duration_groups": [], "display_attachments": {"trailing": text[len(serialized_surface):]}, "quality_masks": {},
    }


def training_record_with_initial_gap(tmp_path: Path) -> dict:
    record = training_record_v2_fixture(tmp_path)
    record["native_phones"] = record["native_phones"][2:]
    record["native_phones"][0].update(phone_id="p2", start_sample=4000, end_sample=10000)
    record["native_phones"][1].update(phone_id="p3", start_sample=10000, end_sample=16000)
    return record


def parse_textgrid(path: Path) -> dict:
    return parse_textgrid_text(path.read_text(encoding="utf-8"))


def parse_textgrid_text(text: str) -> dict:
    tiers = []
    current = None
    interval = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("item [") and stripped != "item []:":
            current = {"name": "", "intervals": []}; tiers.append(current)
        elif current is not None and stripped.startswith("name = "):
            current["name"] = _quoted(stripped)
        elif current is not None and stripped.startswith("intervals ["):
            interval = {}; current["intervals"].append(interval)
        elif interval is not None and line.startswith("            ") and stripped.startswith("xmin = "):
            interval["start"] = round(float(stripped.split("=", 1)[1]) * 16000)
        elif interval is not None and line.startswith("            ") and stripped.startswith("xmax = "):
            interval["end"] = round(float(stripped.split("=", 1)[1]) * 16000)
        elif interval is not None and line.startswith("            ") and stripped.startswith("text = "):
            interval["text"] = _quoted(stripped)
    return {"tiers": tiers}


def _quoted(line: str) -> str:
    match = re.search(r'"(.*)"$', line)
    assert match
    return match.group(1).replace('""', '"')


def labels(grid: dict, tier_name: str) -> list[str]:
    return [row["text"] for row in next(t for t in grid["tiers"] if t["name"] == tier_name)["intervals"]]


def boundaries(grid: dict, tier_name: str) -> list[tuple[int, int]]:
    return [(row["start"], row["end"]) for row in next(t for t in grid["tiers"] if t["name"] == tier_name)["intervals"]]


def tier_is_continuous(tier: dict, start: int, end: int) -> bool:
    return bool(tier["intervals"]) and tier["intervals"][0]["start"] == start and tier["intervals"][-1]["end"] == end and all(left["end"] == right["start"] for left, right in zip(tier["intervals"], tier["intervals"][1:]))


def reconstruct_display(grid: dict, display_attachments: dict) -> str:
    return "".join(labels(grid, "original_text")) + display_attachments.get("trailing", "")


def authoritative_segments(record: dict, tier_name: str) -> list[dict]:
    return [row for row in next(t for t in build_five_track_tiers(record) if t["name"] == tier_name)["intervals"] if row.get("segment_id")]


def test_export_writes_exact_five_track_textgrid(tmp_path: Path):
    record = training_record_v2_fixture(tmp_path, text=' 「コー」 game ')
    paths = export_tts_artifacts(record, tmp_path / "out")
    parsed = parse_textgrid(paths["textgrid"])
    assert [tier["name"] for tier in parsed["tiers"]] == ["original_text", "kana", "mfa_phone", "phone_kana", "phone_tone"]
    assert labels(parsed, "phone_kana") == ["コ", "コ|ー", "", ""]
    assert labels(parsed, "phone_tone") == ["H", "H|L", "NA", "NA"]
    for tier_name in ("mfa_phone", "phone_kana", "phone_tone"):
        assert boundaries(parsed, tier_name) == boundaries(parsed, "mfa_phone")


def test_writer_preserves_whitespace_punctuation_quotes_and_one_sample_boundaries(tmp_path: Path):
    record = training_record_v2_fixture(tmp_path, text='  「声"」!  ', phone_bounds=[(0, 1), (1, 2), (2, 8000), (8000, 16000)])
    grid = parse_textgrid_text(render_textgrid(record))
    assert reconstruct_display(grid, record["display_attachments"]) == '  「声"」!  '
    assert all(tier_is_continuous(tier, 0, 16000) for tier in grid["tiers"])
    assert boundaries(grid, "mfa_phone")[:2] == [(0, 1), (1, 2)]
    assert labels(grid, "original_text")[0].endswith('「声"」!')


def test_empty_coverage_intervals_are_not_authoritative_phones(tmp_path: Path):
    record = training_record_with_initial_gap(tmp_path)
    grid = parse_textgrid_text(render_textgrid(record))
    assert len(authoritative_segments(record, "mfa_phone")) == 2
    assert labels(grid, "mfa_phone")[0] == ""
