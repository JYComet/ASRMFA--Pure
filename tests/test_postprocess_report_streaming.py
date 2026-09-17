"""Streaming contracts for the formal postprocess JSONL report reader."""

from pathlib import Path

from scripts import run_pipeline


class _PathCarrier:
    """An os.PathLike wrapper used to exercise path normalization."""

    def __init__(self, path: Path):
        self.path = path

    def __fspath__(self):
        return str(self.path)


class _LineOnlyReader:
    """Delegate iteration while rejecting whole-file reads."""

    def __init__(self, handle):
        self._handle = handle

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._handle.__exit__(*exc_info)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._handle)

    def read(self, *args, **kwargs):
        raise AssertionError("the report reader must not materialize the file")

    def readlines(self, *args, **kwargs):
        raise AssertionError("the report reader must not materialize the file")


def test_postprocess_report_reader_streams_utf8_jsonl_and_preserves_rows(
        tmp_path, monkeypatch, capsys):
    parent = (tmp_path / "deep parent with spaces" / "中文🙂" / ("a" * 64)
              / ("nested-" + "b" * 64) / "level.with.dots")
    report_path = parent / "report.multi.part.jsonl"
    report_path.parent.mkdir(parents=True)
    first = "stem.é🙂"
    second = "词.二"
    report_path.write_text(
        "\n"  # blank lines are ignored
        + '{"stem": "' + first + '"}\n'
        + '{"stem": "' + first + '"}\n'  # duplicates remain visible
        + '{"stem": "' + second + '"}\n'
        + '{not-json}\n'
        + '{"status": "missing stem"}\n'
        + "\n",
        encoding="utf-8",
    )

    original_open = Path.open

    def line_only_open(path, *args, **kwargs):
        return _LineOnlyReader(original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", line_only_open)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the report reader must not call read_text")),
    )

    expected = [first, first, second]
    for path_arg in (str(report_path), report_path, _PathCarrier(report_path)):
        assert run_pipeline._read_postprocess_report(path_arg) == (expected, 2)

    output = capsys.readouterr().out
    assert output.count("ERROR: invalid report row 5") == 3
    assert output.count("ERROR: invalid report row 6") == 3


def test_postprocess_report_reader_marks_missing_file_invalid(tmp_path, capsys):
    missing = tmp_path / "missing report.multi.part.jsonl"

    assert run_pipeline._read_postprocess_report(missing) == ([], 1)
    assert f"ERROR: missing postprocess report: {missing}" in capsys.readouterr().out
