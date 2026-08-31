"""Stdlib-mocked tests for the bounded singleton MFA continuation executor."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import gpu1000_orchestrate as tool


def _grid(path: Path, *, valid: bool = True) -> None:
    if not valid:
        path.write_text("not a TextGrid", encoding="utf-8")
        return
    path.write_text(
        '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "yi1"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "yi"
''', encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict, Path, Path, Path, Path]:
    root = tmp_path / "run"
    (root / "workspace").mkdir(parents=True)
    stem = "s0001"
    retry = root / "workspace" / "mfa_shards" / "run" / "retry_missing"
    retry.mkdir(parents=True)
    lab = retry / f"{stem}.lab"
    wav = retry / f"{stem}.wav"
    lab.write_text("0 1 yi", encoding="utf-8")
    wav.write_bytes(b"RIFF-mock-wav")
    (root / "resolved_gpu1000_nvrasr_fallback.yaml").write_text("mode: nvrasr_fallback\n", encoding="utf-8")
    fake_exe = root / "mfa-bin"
    fake_exe.write_bytes(b"mfa")
    (fake_exe.parent / "fstcompile").write_bytes(b"fstcompile")
    dictionary = root / "dict.dict"
    dictionary.write_text("yi Y I", encoding="utf-8")
    model = root / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"model")
    mfa_root = root / "proof-mfa-root"
    numba = root / "proof-numba"
    mfa_root.mkdir(); numba.mkdir()
    proof = {
        "command": [str(fake_exe), "align", "proof-corpus", str(dictionary), str(model),
                    "proof-output", "--audio_directory", "proof-audio",
                    "--temporary_directory", "proof-temp", "--beam", "20",
                    "--retry_beam", "80", "--dither", "0.0",
                    "--no_fine_tune", "--no_clean"],
        "mfa_executable": {"path": str(fake_exe), "sha256": tool.sha_file(fake_exe)},
        "dictionary": {"path": str(dictionary), "sha256": tool.sha_file(dictionary)},
        "model": {"path": str(model), "sha256": tool.sha_tree(model)},
        "environment": {"MFA_ROOT_DIR": str(mfa_root), "NUMBA_CACHE_DIR": str(numba),
                        "PATH_prefix": str(fake_exe.parent),
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
        "inputs": [{"stem": stem, "role": "anchor.lab", "source": str(lab),
                    "sha256": tool.sha_file(lab)},
                   {"stem": stem, "role": "mfa_axis_audio", "source": str(wav),
                    "sha256": tool.sha_file(wav)}],
    }
    scope = {"schema": tool.CONTINUATION_SCOPE_SCHEMA, "stems": [stem],
             "lab": str(lab), "wav": str(wav)}
    cfg = {"mfa": {"mfa_dict": str(dictionary), "acoustic_model": str(model)}}
    return root, scope, proof, fake_exe, dictionary, model, cfg


def _run_case(tmp_path: Path, *, first_rc: int = 0, first_stderr: str = "",
              first_grid: str = "valid", rescue_grid: str = "valid"):
    root, scope, proof, fake_exe, dictionary, model, cfg = _fixture(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        output = Path(argv[5])
        output.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            rc, stderr, grid_kind = first_rc, first_stderr, first_grid
        else:
            rc, stderr, grid_kind = 0, "", rescue_grid
        if grid_kind != "none":
            grid = output / scope["stems"][0] / f"{scope['stems'][0]}.TextGrid"
            grid.parent.mkdir(parents=True, exist_ok=True)
            _grid(grid, valid=grid_kind == "valid")
        return SimpleNamespace(returncode=rc, stdout="mock stdout", stderr=stderr)

    with patch.object(tool, "resolve_mfa_dependency", return_value=(str(fake_exe), "mock")), \
         patch.object(tool, "load_yaml", return_value=cfg), \
         patch.object(tool.subprocess, "run", side_effect=fake_run):
        result = None
        error = None
        try:
            result = tool._execute_singleton_mfa(root, scope, python=str(fake_exe), proof=proof)
        except Exception as exc:  # assertions below classify SafetyError cases
            error = exc
    receipt_path = root / "workspace" / "continuation_singleton_attempts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    return root, calls, result, error, receipt, proof, dictionary, model


def test_singleton_rc0_valid_grid_success_once(tmp_path: Path):
    root, calls, result, error, receipt, proof, dictionary, model = _run_case(tmp_path)
    assert error is None and result.is_file() and len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == proof["mfa_executable"]["path"]
    assert argv[1:2] == ["align"] and argv[3] == str(dictionary) and argv[4] == str(model)
    assert "--beam" in argv and argv[argv.index("--beam") + 1] == "20"
    assert "--retry_beam" in argv and argv[argv.index("--retry_beam") + 1] == "80"
    assert argv[argv.index("--dither") + 1] == "0.0"
    assert "--no_fine_tune" in argv and "--no_clean" in argv
    assert kwargs["env"]["MFA_ROOT_DIR"] != proof["environment"]["MFA_ROOT_DIR"]
    assert kwargs["env"]["NUMBA_CACHE_DIR"] != proof["environment"]["NUMBA_CACHE_DIR"]
    assert kwargs["env"]["MFA_ROOT_DIR"].startswith(str(root / "workspace"))
    assert kwargs["env"]["NUMBA_CACHE_DIR"].startswith(str(root / "workspace"))
    path_parts = kwargs["env"]["PATH"].split(os.pathsep)
    assert Path(path_parts[0]) == Path(proof["mfa_executable"]["path"]).parent.resolve()
    assert (Path(path_parts[0]) / "fstcompile").is_file()
    assert receipt["schema"] == "gpu1000-singleton-attempts-v1"
    assert len(receipt["attempts"]) == 1 and receipt["attempts"][0]["attempt"] == "20_80"
    assert receipt["attempts"][0]["returncode"] == 0
    assert receipt["dictionary_sha256"] == tool.sha_file(dictionary)
    assert receipt["model_sha256"] == tool.sha_tree(model)


def test_singleton_rc0_no_grid_or_invalid_grid_fails_without_rescue(tmp_path: Path):
    for kind in ("none", "invalid"):
        root, calls, result, error, receipt, *_ = _run_case(tmp_path / kind, first_grid=kind)
        assert result is None and isinstance(error, tool.SafetyError)
        assert len(calls) == 1 and len(receipt["attempts"]) == 1
        assert receipt["attempts"][0]["returncode"] == 0


def test_singleton_generic_nonzero_fails_without_rescue(tmp_path: Path):
    root, calls, result, error, receipt, *_ = _run_case(
        tmp_path, first_rc=2, first_stderr="GenericFailure", first_grid="none")
    assert result is None and isinstance(error, tool.SafetyError)
    assert len(calls) == 1 and len(receipt["attempts"]) == 1


def test_singleton_noalignments_gets_exactly_one_200_800_rescue(tmp_path: Path):
    root, calls, result, error, receipt, *_ = _run_case(
        tmp_path, first_rc=1, first_stderr="NoAlignmentsError: no path", first_grid="none")
    assert error is None and result.is_file() and len(calls) == 2
    first, second = calls
    assert first[0][first[0].index("--beam") + 1] == "20"
    assert first[0][first[0].index("--retry_beam") + 1] == "80"
    assert second[0][second[0].index("--beam") + 1] == "200"
    assert second[0][second[0].index("--retry_beam") + 1] == "800"
    assert receipt["attempts"][-1]["attempt"] == "200_800"
    assert len(receipt["attempts"]) == 2


def test_retry_argv_preserves_options_and_rebinds_only_namespaces(tmp_path: Path):
    root, scope, proof, _, dictionary, model, _ = _fixture(tmp_path)
    command = proof["command"]
    transformed = tool._transform_retry_command(
        command,
        corpus=root / "workspace" / "corpus",
        audio=root / "workspace" / "audio",
        output=root / "workspace" / "output",
        temp=root / "workspace" / "temp",
        proof=proof,
    )
    assert transformed[0:2] == command[0:2]
    assert transformed[3:5] == [str(dictionary), str(model)]
    assert transformed[transformed.index("--beam") + 1] == "20"
    assert transformed[transformed.index("--retry_beam") + 1] == "80"
    assert transformed[transformed.index("--audio_directory") + 1].endswith("/audio")
    assert transformed[transformed.index("--temporary_directory") + 1].endswith("/temp")
    assert transformed[-2:] == command[-2:]


def test_retry_argv_rejects_implicit_or_nonzero_dither(tmp_path: Path):
    root, _, proof, *_ = _fixture(tmp_path)
    for value in (None, "1.0"):
        command = list(proof["command"])
        index = command.index("--dither")
        if value is None:
            del command[index:index + 2]
        else:
            command[index + 1] = value
        try:
            tool._transform_retry_command(
                command, corpus=root / "workspace" / "corpus",
                audio=root / "workspace" / "audio",
                output=root / "workspace" / "output",
                temp=root / "workspace" / "temp", proof=proof)
        except tool.SafetyError as exc:
            assert "dither" in str(exc)
        else:
            raise AssertionError("nondeterministic MFA retry command was accepted")


def main() -> int:
    # Kept executable without pytest so the continuation harness can import
    # these cases and developers can also run this file directly.
    from tempfile import TemporaryDirectory
    tests = [test_singleton_rc0_valid_grid_success_once,
             test_singleton_rc0_no_grid_or_invalid_grid_fails_without_rescue,
             test_singleton_generic_nonzero_fails_without_rescue,
             test_singleton_noalignments_gets_exactly_one_200_800_rescue,
             test_retry_argv_preserves_options_and_rebinds_only_namespaces]
    for test in tests:
        with TemporaryDirectory() as directory:
            test(Path(directory))
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
