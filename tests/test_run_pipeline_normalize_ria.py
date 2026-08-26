"""Regression tests for the normalize_ria step's boolean exit contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_pipeline
from scripts.postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid


def _token(word: str, start: float, end: float) -> dict:
    return {
        "word": word,
        "start_ms": round(start * 1000),
        "end_ms": round(end * 1000),
        "start_s": start,
        "end_s": end,
        "type": "word",
    }


def _write_bundle(ctc_dir: Path, stem: str, words: list[str], *, textgrid_words: list[str] | None = None) -> None:
    tokens = [_token(word, index / 10, (index + 1) / 10)
              for index, word in enumerate(words)]
    (ctc_dir / f"{stem}_tokens.jsonl").write_text(
        "".join(json.dumps(token, ensure_ascii=False) + "\n" for token in tokens),
        encoding="utf-8",
    )
    (ctc_dir / f"{stem}.lab").write_text(" ".join(words) + "\n", encoding="utf-8")
    grid_words = textgrid_words if textgrid_words is not None else words
    intervals = [Interval(index / 10, (index + 1) / 10, word)
                 for index, word in enumerate(grid_words)]
    write_textgrid(TextGrid(
        0.0,
        max(len(grid_words), 1) / 10,
        [
            Tier("words", 0.0, max(len(grid_words), 1) / 10, intervals),
            Tier("pauses", 0.0, max(len(grid_words), 1) / 10, []),
        ],
    ), ctc_dir / f"{stem}.TextGrid")


def _ctx(ctc_dir: Path, reference_mode: str = "auto") -> dict:
    return {"ctc_pretg": ctc_dir, "reference_mode": reference_mode}


def _make_zero_duration_token(ctc_dir: Path, stem: str) -> None:
    path = ctc_dir / f"{stem}_tokens.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["start_s"] = 13.47
    rows[-1]["end_s"] = 13.47
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize("allow_partial", [False, True])
def test_normalize_ria_changed_bundle_returns_zero_and_preserves_conversion(
    tmp_path, capsys, allow_partial
):
    _write_bundle(tmp_path, "demo", ["rui3", "ya4"])

    assert run_pipeline.step_normalize_ria(
        None,
        {"mfa": {"allow_partial": allow_partial}},
        Path("mfa-python"),
        _ctx(tmp_path),
    ) == 0

    tokens = [json.loads(line) for line in
              (tmp_path / "demo_tokens.jsonl").read_text(encoding="utf-8").splitlines()]
    assert tokens == [_token("ria", 0.0, 0.2)]
    assert (tmp_path / "demo.lab").read_text(encoding="utf-8") == "ria\n"
    assert run_pipeline.read_ctc_textgrid_words(tmp_path / "demo.TextGrid") == ["ria"]
    assert "1 synchronized bundle(s)" in capsys.readouterr().out


@pytest.mark.parametrize("allow_partial", [False, True])
def test_normalize_ria_no_change_is_zero_for_partial_modes(tmp_path, allow_partial):
    _write_bundle(tmp_path, "demo", ["ria"])
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
    }

    assert run_pipeline.step_normalize_ria(
        None,
        {"mfa": {"allow_partial": allow_partial}},
        Path("mfa-python"),
        _ctx(tmp_path),
    ) == 0

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("allow_partial", [False, True])
def test_normalize_ria_real_bundle_failure_respects_partial_mode(
    tmp_path, capsys, allow_partial
):
    _write_bundle(tmp_path, "demo", ["rui3", "ya4"])
    _make_zero_duration_token(tmp_path, "demo")
    ctx = _ctx(tmp_path, "fallback")

    assert run_pipeline.step_normalize_ria(
        None,
        {"mfa": {"allow_partial": allow_partial}},
        Path("mfa-python"),
        ctx,
    ) == (0 if allow_partial else 1)

    output = capsys.readouterr().out
    assert "RIA bundle(s) failed" in output
    assert "demo" in output
    assert ctx["ctc_normalize_ria_failures"][0][0] == "demo"
    assert "invalid interval 13.47..13.47" in ctx["ctc_normalize_ria_failures"][0][1]


@pytest.mark.parametrize("allow_partial", [False, True])
def test_normalize_ria_failure_ledger_is_per_stem(tmp_path, allow_partial):
    _write_bundle(tmp_path, "good", ["rui3", "ya4"])
    _write_bundle(tmp_path, "bad", ["rui3", "ya4"])
    _make_zero_duration_token(tmp_path, "bad")
    ctx = _ctx(tmp_path)

    assert run_pipeline.step_normalize_ria(
        None,
        {"mfa": {"allow_partial": allow_partial}},
        Path("mfa-python"),
        ctx,
    ) == (0 if allow_partial else 1)

    failures = ctx["ctc_normalize_ria_failures"]
    assert len(failures) == 1
    assert failures[0][0] == "bad"
    assert "invalid interval 13.47..13.47" in failures[0][1]


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"reference_mode": "authority"}, "authority"),
        ({"reference_mode": "fallback"}, "fallback"),
        ({"ctc_prealign": {"allow_missing_reference": False}}, "authority"),
        ({"ctc_prealign": {"allow_missing_reference": True}}, "fallback"),
    ],
)
def test_reference_mode_resolution_remains_backward_compatible(cfg, expected):
    assert run_pipeline.resolve_reference_mode(cfg) == expected
