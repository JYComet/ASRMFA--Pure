"""Contract tests for the planned postprocess batching and compatibility seams.

These tests intentionally describe the small, pure helper APIs used by the
postprocess consolidation.  They are red against the pre-consolidation
implementation: the helpers are not present yet, while the existing public
behaviour remains the oracle for differential assertions.
"""

from __future__ import annotations

import json
import hashlib
import ast
import inspect
from collections import Counter
from pathlib import Path

import pytest

from scripts import pipeline_utils
from scripts import postprocess_textgrids as post


def _api(name: str):
    """Resolve a planned helper so a missing contract fails explicitly."""
    return getattr(post, name)


def test_batch_preflight_validates_global_lifecycle_once_for_each_batch_size(
        monkeypatch, tmp_path):
    """R1: global raw/work validation is independent of logical stem count."""
    calls = Counter()

    def raw_validator(*args, **kwargs):
        calls["raw"] += 1
        return []

    def work_validator(*args, **kwargs):
        calls["work"] += 1
        return []

    monkeypatch.setattr(post, "validate_ctc_raw_manifest", raw_validator)
    monkeypatch.setattr(post, "validate_ctc_work_receipt", work_validator)

    for count in (1, 10, 100):
        calls.clear()
        stems = [f"stem-{index:03d}" for index in range(count)]
        raw_manifest = {
            "schema": pipeline_utils.CTC_RAW_MANIFEST_SCHEMA,
            "stems": stems,
            "files": [
                {"stem": stem, "suffix": suffix,
                 "name": f"{stem}{suffix}", "size": 1,
                 "sha256": "a" * 64}
                for stem in stems for suffix in pipeline_utils.CTC_SUFFIXES
            ],
        }
        work_receipt = {
            "schema": pipeline_utils.CTC_WORK_RECEIPT_SCHEMA,
            "stems": stems,
            "files": [
                {"stem": stem, "suffix": suffix,
                 "name": f"{stem}{suffix}", "size": 1,
                 "sha256": "b" * 64}
                for stem in stems for suffix in pipeline_utils.CTC_SUFFIXES
            ],
        }
        context = _api("_preflight_ctc_lifecycle_batch")(
            raw_manifest, work_receipt, tmp_path / "raw", tmp_path / "work")

        assert calls == {"raw": 1, "work": 1}
        # The preflight may index all rows, but must not perform per-stem
        # validation as a side effect of constructing the immutable context.
        assert tuple(context.stems) == tuple(stems)
        assert context.raw_rows["stem-000", ".TextGrid"]["sha256"] == "a" * 64


def test_consumed_sidecar_hashes_remain_per_stem_after_batch_preflight(
        monkeypatch, tmp_path):
    """R1/R2: preflight is global, while consumed evidence stays per stem."""
    consumed = []

    def hash_sidecar(path, expected_size, expected_sha256):
        consumed.append((Path(path).name, expected_size, expected_sha256))
        return True

    monkeypatch.setattr(post, "_verify_consumed_artifact_hash", hash_sidecar)
    stems = ["a", "b", "c"]
    context = _api("_preflight_ctc_lifecycle_batch")(
        {"stems": stems, "files": []}, {"stems": stems, "files": []},
        tmp_path / "raw", tmp_path / "work")
    for stem in stems:
        assert _api("_bind_consumed_stem_evidence")(
            context, stem, "_tokens.jsonl") is not None
    assert [name for name, *_ in consumed] == [
        "a_tokens.jsonl", "b_tokens.jsonl", "c_tokens.jsonl"]


def test_stem_artifact_bundle_reads_each_payload_once_and_copies_views(
        monkeypatch, tmp_path):
    """R2: one read/decode per artifact and independent raw/canonical rows."""
    counts = Counter()
    payloads = {
        "demo_tokens.jsonl": '{"word":"NI3","start_s":0,"end_s":1}\n',
        "demo_punct.json": '[{"label":"，","index":0}]',
        "demo.lab": "NI3\n",
        "demo_text_cn.txt": "你好，世界\n",
        "demo_ref.txt": "你好，世界\n",
    }

    for name, value in payloads.items():
        (tmp_path / name).write_text(value, encoding="utf-8")
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def counted_read_bytes(path, *args, **kwargs):
        if path.name in payloads:
            counts[(path.name, "bytes")] += 1
        return original_read_bytes(path, *args, **kwargs)

    def counted_read_text(path, *args, **kwargs):
        if path.name in payloads:
            counts[(path.name, "text")] += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(Path, "read_text", counted_read_text)
    expected = {
        name: {"size": len(value.encode("utf-8")),
               "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
        for name, value in payloads.items()
    }
    bundle = _api("_load_stem_artifact_bundle")(
        "demo", tmp_path, tmp_path, expected)
    assert bundle.raw_token_rows is not bundle.canonical_token_rows
    bundle.raw_token_rows[0]["word"] = "MUTATED"
    assert bundle.canonical_token_rows[0]["word"] == "NI3"
    def read_count(name):
        return sum(counts[(name, kind)] for kind in ("bytes", "text"))

    assert read_count("demo_tokens.jsonl") <= 1
    assert read_count("demo_punct.json") <= 1
    assert read_count("demo.lab") <= 1
    assert read_count("demo_text_cn.txt") <= 1
    assert read_count("demo_ref.txt") <= 1


def test_strict_manifest_context_parses_once_and_binds_ledgers_in_constant_time(
        monkeypatch, tmp_path):
    """R3: one global parse, O(1) stem lookup, independent ledger checks."""
    manifest_path = tmp_path / "en_alignment_manifest.json"
    stems = [f"stem-{index:03d}" for index in range(100)]
    segments = [f"{stem}:s0" for stem in stems]
    manifest_path.write_text(json.dumps({
        "schema": post.STRICT_EN_MFA_SCHEMA,
        "strict_provenance": True,
        "canonical_units": post.CANONICAL_UNITS_SCHEMA,
        "status": "success",
        "expected_segments": segments, "produced_segments": segments,
        "rejected_segments": [],
        "stem_ledgers": [
            {"stem": stem, "path": str(tmp_path / f"{stem}.json"),
             "sha256": "a" * 64}
            for stem in stems
        ],
    }), encoding="utf-8")
    parse_calls = Counter()
    original = json.loads

    def counted_loads(value, *args, **kwargs):
        parse_calls["manifest"] += 1
        return original(value, *args, **kwargs)

    monkeypatch.setattr(post.json, "loads", counted_loads)
    context = _api("_build_strict_en_manifest_context")(manifest_path)
    assert parse_calls["manifest"] == 1
    assert context.ledger_by_stem["stem-099"]["stem"] == "stem-099"
    assert context.lookup("stem-000") is context.ledger_by_stem["stem-000"]
    assert context.lookup("missing") is None
    assert context.validate_ledger("stem-000") is False


def test_pinyin_lookup_is_casefold_first_and_preserves_insertion_winner():
    """R4: casefold lookup must retain the first dictionary spelling."""
    dictionary = {
        "NI3": ["first", "winner"],
        "ni3": ["second", "must-not-win"],
        "Nǐ3": ["accented"],
    }
    lookup = _api("_build_pinyin_lookup")(dictionary)
    assert lookup["ni3"] == ["first", "winner"]
    assert lookup["nǐ3"] == ["accented"]


def test_source_phone_barrier_and_final_publication_rebuild_are_explicit(
        monkeypatch):
    """R5: lineage synchronization and publication are separate barriers."""
    events = []
    monkeypatch.setattr(post, "_reconcile_source_phone_lineage",
                        lambda *args, **kwargs: events.append("source") or True)
    monkeypatch.setattr(post, "_rebuild_derived_from_frozen_words",
                        lambda *args, **kwargs: events.append("publication"))
    state = _api("_make_derived_barrier_state")()
    _api("_commit_derived_barriers")(state)
    assert events == ["source", "publication"]
    assert state.publication_rebuild_count == 1
    assert state.source_phone_revision == state.words_revision


def _containment_oracle(phones, words):
    issues = []
    for phone_index, phone in enumerate(phones):
        owner = next((word_index for word_index, word in enumerate(words)
                      if word[0] <= phone[0] and phone[1] <= word[1]), None)
        if owner is None:
            issues.append({"phone_index": phone_index,
                           "reason": "phone_not_contained",
                           "phone": phone})
    return issues


def test_linear_containment_matches_quadratic_oracle_and_bounds_comparisons():
    """R7: ordered two-pointer containment is equivalent and linear."""
    words = [(index * 0.1, (index + 1) * 0.1)
             for index in range(100)]
    phones = [(start + 0.01, start + 0.02) for start, _ in words]
    phones.extend([(10.1, 10.11), (20.1, 20.11)])
    comparisons = []
    actual = _api("_phone_word_containment_linear")(
        phones, words, comparison_counter=comparisons)
    assert actual == _containment_oracle(phones, words)
    assert len(comparisons) <= 4 * (len(phones) + len(words))


def test_bounded_report_submission_and_atomic_failure_contract(tmp_path):
    """R8: bounded work and atomic report publication preserve old evidence."""
    report_path = tmp_path / "nested parent with spaces" / "post.part.report.jsonl"
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"stem":"old","status":"ok"}\n', encoding="utf-8")
    seen_pending = []
    stems = [f"stem.{index}.🙂" for index in range(17)]

    def worker(stem):
        return {"stem": stem, "status": "ok", "evidence": "原样-é"}

    result = _api("_run_bounded_postprocess")(
        stems, worker, report_path, workers=3,
        pending_observer=seen_pending.append)
    assert max(seen_pending) <= max(2 * 3, 4)
    rows = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
    assert [row["stem"] for row in rows] == stems
    assert len({row["stem"] for row in rows}) == len(stems)
    assert rows[0]["evidence"] == "原样-é"
    assert list(report_path.parent.glob(".*.tmp")) == []

    old = report_path.read_bytes()
    with pytest.raises(OSError):
        _api("_run_bounded_postprocess")(
            stems, lambda _stem: (_ for _ in ()).throw(OSError("worker")),
            report_path, workers=3)
    assert report_path.read_bytes() == old
    assert list(report_path.parent.glob(".*.tmp")) == []


def test_atomic_report_preserves_old_report_when_end_evidence_fails(tmp_path):
    """R8: end-of-batch evidence failure must not publish a partial report."""
    report_path = tmp_path / "report.jsonl"
    old = b'{"stem":"old","status":"ok"}\n'
    report_path.write_bytes(old)

    with pytest.raises(OSError):
        _api("_write_postprocess_report_atomic")(
            report_path,
            [{"stem": "new🙂", "status": "ok", "evidence": "原样-é"}],
            end_evidence=lambda: (_ for _ in ()).throw(OSError("end evidence")),
        )
    assert report_path.read_bytes() == old
    assert list(report_path.parent.glob(".*.tmp")) == []


@pytest.mark.parametrize("stem", [
    " plain stem ", "中文🙂", "e\u0301", "part.one.two", ".leading",
    "-leading", "nested/child", "a" * 120,
])
def test_path_and_stem_contract_preserves_exact_spelling(stem):
    """R2/R8: Unicode and punctuation are evidence, never display-normalized."""
    report = {"stem": stem, "path": f"C:\\evidence\\{stem}.TextGrid",
              "unc": f"\\\\server\\share\\{stem}.wav"}
    serialized = json.dumps(report, ensure_ascii=False)
    restored = json.loads(serialized)
    assert restored == report
    # NFC is intentionally not applied to evidence; this assertion covers
    # both already-normalized and combining-character spellings.
    assert restored["stem"] == stem


def test_windows_drive_and_unc_evidence_are_not_local_posix_paths(monkeypatch):
    """Pure path helper contract: Windows evidence strings stay opaque."""
    calls = []
    monkeypatch.setattr(Path, "exists", lambda self: calls.append(str(self)) or False)
    resolver = _api("_resolve_external_evidence_path")
    for value in (r"C:\evidence\clip.wav", r"\\server\share\clip.wav"):
        assert resolver(value, base_dir=Path("/tmp/local-root")) == value
    assert calls == []


def test_main_strict_manifest_is_fail_closed_before_worker_start():
    """R3: strict mode must not fall back to per-stem manifest reads."""
    source = Path(post.__file__).read_text(encoding="utf-8")
    assert "_build_strict_en_manifest_context" in source
    assert "strict_manifest.is_file()" in source
    assert "strict_manifest.is_symlink()" in source


def test_pre_freeze_sync_is_source_only_and_final_publication_is_unique():
    """R5: source barriers cannot publish derived tiers before the freeze."""
    source = Path(post.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    sync = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "_sync_derived_tiers")
    nested_names = [node.func.id for node in ast.walk(sync)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)]
    assert "_build_hanzi_tier" not in nested_names
    assert "build_pinyin_phones_tier" not in nested_names
    commits = [node for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "_rebuild_derived_from_frozen_words"]
    assert len(commits) == 1


def test_every_postprocess_sliced_rms_call_declares_local_alignment():
    """R6: sliced RMS calls must declare local alignment explicitly."""
    source = Path(post.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    def calls(name):
        return [node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == name]
    def alignment(call):
        for keyword in call.keywords:
            if keyword.arg == "alignment" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
        return None
    for call in calls("_frame_rms_vec"):
        first = call.args[0] if call.args else None
        if isinstance(first, ast.Subscript):
            assert alignment(call) == "local"
        elif isinstance(first, ast.Name) and first.id in {"audio", "wav_audio"}:
            assert alignment(call) == "global"
    for call in calls("_rms_frames_in_span"):
        assert alignment(call) == "global"


def test_main_guards_flat_textgrid_stem_collisions_before_workers():
    """Path compatibility: flat TextGrid names must fail closed."""
    source = inspect.getsource(post.main)
    assert "_reject_flat_stem_collisions" in source
    assert "tg_paths" in source
