"""Integration contracts for the post-processing efficiency wiring.

These tests deliberately exercise the production entry points and seams rather
than only testing the individual cache/index helpers.  They are canaries for
the two easy failure modes of an optimisation refactor: implementing a helper
without wiring it into ``main``/``process_one``, and changing publication
ordering while retaining the old unbounded aggregation behaviour.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import postprocess_textgrids as post


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name]


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((item for item in call.keywords if item.arg == name), None)


def _constant(keyword: ast.keyword | None):
    return (keyword.value.value
            if keyword is not None and isinstance(keyword.value, ast.Constant)
            else None)


def test_main_is_wired_to_bounded_atomic_report_and_end_verification():
    source = inspect.getsource(post.main)
    tree = ast.parse(source)

    assert _calls(tree, "_run_bounded_postprocess"), (
        "main must publish through the bounded/atomic report runner")
    assert _calls(tree, "_verify_ctc_lifecycle_end"), (
        "main must verify the immutable CTC lifecycle after all workers finish")

    # The formal JSONL report must not be accumulated in one all-input list or
    # opened directly by main.  (The tone-reference JSON is a separate file.)
    assert "reports = []" not in source
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "open":
            continue
        mode = call.args[0] if call.args else None
        if isinstance(mode, ast.Constant) and mode.value == "w":
            receiver = call.func.value
            receiver_name = receiver.id if isinstance(receiver, ast.Name) else ""
            assert receiver_name not in {"rp", "report_path", "report_file"}, (
                "main must delegate formal report publication to the atomic writer")

    # A bounded runner should be visible in the production path, not merely
    # defined as an unused compatibility helper.
    assert any(call.lineno > 1 for call in _calls(tree, "_run_bounded_postprocess"))


def _lifecycle_context(tmp_path: Path, stem: str = "clip"):
    raw_dir = tmp_path / "raw evidence"
    work_dir = tmp_path / "work evidence"
    raw_dir.mkdir()
    work_dir.mkdir()
    raw_manifest = {"stems": [stem], "files": []}
    work_receipt = {"stems": [stem], "files": []}
    raw_bytes = (json.dumps(raw_manifest, ensure_ascii=False,
                            separators=(",", ":")) + "\n").encode()
    work_bytes = (json.dumps(work_receipt, ensure_ascii=False,
                             separators=(",", ":")) + "\n").encode()
    (raw_dir / ".ctc_raw_manifest.json").write_bytes(raw_bytes)
    (work_dir / ".ctc_work_receipt.json").write_bytes(work_bytes)
    return post.CtcLifecycleBatchContext(
        stems=(stem,), raw_manifest=raw_manifest, work_receipt=work_receipt,
        raw_dir=raw_dir, work_dir=work_dir,
        raw_manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        work_receipt_sha256=hashlib.sha256(work_bytes).hexdigest(),
        raw_manifest_bytes=raw_bytes, work_receipt_bytes=work_bytes,
    )


@pytest.mark.parametrize("which", ["raw", "work"])
def test_ctc_lifecycle_end_rejects_metadata_byte_replacement(tmp_path: Path,
                                                              which: str):
    context = _lifecycle_context(tmp_path)
    verifier = getattr(post, "_verify_ctc_lifecycle_end")

    # A byte-identical end state is accepted.
    assert verifier(context) is None

    if which == "raw":
        target = context.raw_dir / ".ctc_raw_manifest.json"
        target.write_bytes(target.read_bytes() + b" ")
    else:
        target = context.work_dir / ".ctc_work_receipt.json"
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="(raw manifest|work receipt|replaced)"):
        verifier(context)


def test_process_one_resets_frame_cache_context_on_early_failure(monkeypatch,
                                                                  tmp_path: Path):
    def fail_lifecycle(*_args, **_kwargs):
        raise RuntimeError("injected lifecycle failure")

    monkeypatch.setattr(post, "_load_ctc_lifecycle", fail_lifecycle)
    post._FRAME_CACHE_CONTEXT.set(None)
    args = SimpleNamespace(strict_ok=False, strict_en_provenance=False,
                           en_phones_dir=None)
    with pytest.raises(RuntimeError, match="injected lifecycle failure"):
        post.process_one(
            tmp_path / "name with spaces.TextGrid", tmp_path, tmp_path,
            tmp_path / "out", tmp_path / "filtered", args, {}, {})
    assert post._FRAME_CACHE_CONTEXT.get() is None


def test_batch_lifecycle_load_uses_preloaded_token_rows_without_jsonl_reread(
        monkeypatch, tmp_path: Path):
    """A batch context must hand workers parsed token rows exactly once."""
    context_type = post.CtcLifecycleBatchContext
    field_names = {field.name for field in context_type.__dataclass_fields__.values()}
    assert {"raw_token_rows", "work_token_rows"} <= field_names, (
        "batch preflight must expose parsed raw/work token rows to workers")

    stem = "clip"
    raw_row = {"word": "ni3", "start_s": 0.0, "end_s": 0.06}
    work_row = {"word": "ni3", "start_s": 0.0, "end_s": 0.06}
    base = _lifecycle_context(tmp_path, stem)
    context = post.CtcLifecycleBatchContext(
        stems=base.stems, raw_manifest=base.raw_manifest,
        work_receipt=base.work_receipt, raw_dir=base.raw_dir,
        work_dir=base.work_dir, raw_rows=base.raw_rows,
        work_rows=base.work_rows,
        raw_token_rows={(stem, "_tokens.jsonl"): (raw_row,)},
        work_token_rows={(stem, "_tokens.jsonl"): (work_row,)},
        raw_manifest_sha256=base.raw_manifest_sha256,
        work_receipt_sha256=base.work_receipt_sha256,
        raw_manifest_bytes=base.raw_manifest_bytes,
        work_receipt_bytes=base.work_receipt_bytes,
    )
    monkeypatch.setattr(
        post, "_verify_consumed_artifact_hash", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        post, "_nvasr_read_jsonl",
        lambda *_args, **_kwargs: pytest.fail("worker reread a token JSONL"))
    seen = {}

    def authority(*_args, **kwargs):
        seen["raw_rows"] = kwargs.get("raw_rows")
        seen["work_rows"] = kwargs.get("work_rows")
        return {"summary": {"status": "verified"},
                "ordered_raw_projections": []}

    monkeypatch.setattr(post, "_nvasr_build_producer_authority", authority)
    result = post._load_ctc_lifecycle(base.work_dir, stem, context)
    assert list(seen["raw_rows"]) == [raw_row]
    assert list(seen["work_rows"]) == [work_row]
    assert result["stem"] == stem


def test_report_failure_does_not_replace_existing_report(tmp_path: Path):
    report_path = tmp_path / "reports with spaces" / "postprocess_report.jsonl"
    report_path.parent.mkdir()
    original = '{"status":"old"}\n'
    report_path.write_text(original, encoding="utf-8")

    def worker(stem):
        if stem == "bad/name":
            raise RuntimeError("worker fault")
        return {"stem": stem, "status": "ok"}

    with pytest.raises(RuntimeError, match="worker fault"):
        post._run_bounded_postprocess(
            ["good", "bad/name", "later"], worker, report_path, workers=2)
    assert report_path.read_text(encoding="utf-8") == original


def test_frame_rms_call_sites_declare_local_or_global_alignment():
    audio_source = Path(post.__file__).with_name("audio_energy.py").read_text(
        encoding="utf-8")
    audio_tree = ast.parse(audio_source)
    for call in _calls(audio_tree, "frame_rms"):
        # Calls inside the cache-aware audio helpers are the contract points;
        # the function declaration itself is not a call and is not included.
        first = call.args[0] if call.args else None
        expected = "local" if isinstance(first, ast.Subscript) else "global"
        assert _constant(_keyword(call, "alignment")) == expected

    source = Path(post.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for call in _calls(tree, "_rms_frames_in_span"):
        assert _constant(_keyword(call, "alignment")) == "global", (
            "full-audio absolute-span RMS calls must declare global alignment")


def test_derived_barrier_has_a_production_call_site():
    source = Path(post.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_commit_derived_barriers"]
    assert definitions
    own = definitions[0]
    outside = [call for call in _calls(tree, "_commit_derived_barriers")
               if not (own.lineno <= call.lineno <= own.end_lineno)]
    assert outside, (
        "derived barrier helper must be called by the production publication path")
