#!/usr/bin/env python3
"""Synthetic operation-count benchmark for postprocess consolidation.

The benchmark never opens production corpora. It exercises the optimization
seams with temporary Unicode paths and fails if a linearity or bounded-memory
invariant is violated. It measures the candidate only, not a speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import postprocess_textgrids as post  # noqa: E402
from scripts.audio_energy import FrameRmsCache  # noqa: E402


SCHEMA = "postprocess-synthetic-benchmark-v1"


def _write_stem_fixture(root: Path, stem: str) -> None:
    payloads = {
        f"{stem}_tokens.jsonl": '{"word":"NI3","start_s":0,"end_s":1}\n',
        f"{stem}_punct.json": "[]\n",
        f"{stem}.lab": "NI3\n",
        f"{stem}_text_cn.txt": "你好\n",
        f"{stem}_ref.txt": "你好\n",
    }
    for name, content in payloads.items():
        (root / name).write_text(content, encoding="utf-8")


def _rss_snapshot() -> dict:
    """Return best-effort process peak RSS without hiding platform units."""
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform.startswith("linux"):
            return {"bytes": raw * 1024, "raw": raw, "raw_unit": "KiB"}
        return {"bytes": raw, "raw": raw, "raw_unit": "bytes"}
    except (ImportError, OSError, ValueError):
        return {"bytes": None, "raw": None, "raw_unit": "unavailable"}


def _one_run(stem_count: int, root: Path) -> dict:
    stems = [f"stem.{index:05d}.中🙂" for index in range(stem_count)]
    fixture_root = (root / "deep parent with spaces" / "中文🙂"
                    / "level.with.dots")
    fixture_root.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _write_stem_fixture(fixture_root, stem)

    validator_calls = Counter()

    def raw_validator(*_args, **_kwargs):
        validator_calls["raw"] += 1
        return []

    def work_validator(*_args, **_kwargs):
        validator_calls["work"] += 1
        return []

    manifest = {"stems": stems, "files": []}
    receipt = {"stems": stems, "files": []}
    physical_reads = Counter()
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent == fixture_root:
            physical_reads[os.fspath(candidate)] += 1
        return original_read_bytes(candidate, *args, **kwargs)

    comparisons: list[int] = []
    words = [(index * 0.1, (index + 1) * 0.1)
             for index in range(stem_count)]
    phones = [(start + 0.01, start + 0.02) for start, _ in words]
    pending: list[int] = []
    report_path = root / "report parent 中🙂" / "post.process.report.jsonl"
    audio = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    frame_cache = FrameRmsCache("synthetic.中🙂")
    barrier_counts = Counter()

    def counted_rebuild(*_args, **_kwargs):
        barrier_counts["publication"] += 1

    def counted_reconcile(*_args, **_kwargs):
        barrier_counts["source_phone"] += 1
        return True

    started = time.perf_counter()
    with mock.patch.object(post, "validate_ctc_raw_manifest", raw_validator), \
            mock.patch.object(post, "validate_ctc_work_receipt", work_validator), \
            mock.patch.object(Path, "read_bytes", counted_read_bytes), \
            mock.patch.object(post, "_rebuild_derived_from_frozen_words", counted_rebuild), \
            mock.patch.object(post, "_reconcile_source_phone_lineage", counted_reconcile):
        context = post._preflight_ctc_lifecycle_batch(
            manifest, receipt, root / "absent raw", root / "absent work")
        for stem in stems:
            post._load_stem_artifact_bundle(stem, fixture_root, fixture_root)
        post._phone_word_containment_linear(
            phones, words, comparison_counter=comparisons)
        post._build_pinyin_lookup({
            "NI3": ["first"], "ni3": ["second"], "Hao3": ["h", "ao3"]})
        frame_cache.global_frames(audio, 16000, frame_ms=10.0)
        frame_cache.global_frames(audio, 16000, frame_ms=10.0)
        segment = audio[80:8080]
        frame_cache.local_frames(segment, 16000, frame_ms=10.0)
        frame_cache.local_frames(segment, 16000, frame_ms=10.0)
        state = post._make_derived_barrier_state()
        post._commit_derived_barriers(state)
        dispatch = post._run_bounded_postprocess(
            stems,
            lambda stem: {"stem": stem, "status": "ok"},
            report_path,
            workers=min(4, max(1, stem_count)),
            pending_observer=pending.append,
            stream=True,
            end_evidence=lambda: post._verify_ctc_lifecycle_end(context),
        )
    wall_s = time.perf_counter() - started

    read_max = max(physical_reads.values(), default=0)
    with report_path.open(encoding="utf-8") as report_handle:
        row_count = sum(1 for line in report_handle if line.strip())
    operations = {
        "validator_calls": dict(validator_calls),
        "physical_artifact_reads": sum(physical_reads.values()),
        "max_reads_per_artifact": read_max,
        "frame_bank_computations": frame_cache.computation_count,
        "source_phone_barriers": barrier_counts["source_phone"],
        "publication_rebuilds": barrier_counts["publication"],
        "containment_comparisons": len(comparisons),
        "max_pending_futures": max(pending, default=0),
        "report_rows": row_count,
        "report_status_counts": dispatch["counts"],
    }
    worker_count = min(4, max(1, stem_count))
    limits = {
        "validator_calls_each": 1,
        "max_reads_per_artifact": 1,
        "frame_bank_computations": 2,
        "publication_rebuilds": 1,
        "containment_comparisons_max": 4 * (len(phones) + len(words)),
        "max_pending_futures": max(2 * worker_count, 4),
        "report_rows": stem_count,
    }
    ok = (
        validator_calls == {"raw": 1, "work": 1}
        and read_max <= limits["max_reads_per_artifact"]
        and frame_cache.computation_count == limits["frame_bank_computations"]
        and barrier_counts["publication"] == limits["publication_rebuilds"]
        and len(comparisons) <= limits["containment_comparisons_max"]
        and max(pending, default=0) <= limits["max_pending_futures"]
        and row_count == stem_count
        and dispatch["counts"] == {"ok": stem_count}
    )
    return {"wall_seconds": wall_s, "operations": operations,
            "limits": limits, "invariants_passed": ok}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-stems", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.synthetic_stems <= 0 or args.repeat <= 0:
        parser.error("--synthetic-stems and --repeat must be positive")

    tracemalloc.start()
    runs = []
    with tempfile.TemporaryDirectory(prefix="postprocess benchmark 中🙂 ") as temp:
        temp_root = Path(temp)
        for index in range(args.repeat):
            runs.append(_one_run(
                args.synthetic_stems, temp_root / f"repeat {index:02d}"))
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_times = [run["wall_seconds"] for run in runs]
    payload = {
        "schema": SCHEMA,
        "candidate_only": True,
        "synthetic_stems": args.synthetic_stems,
        "repeat": args.repeat,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wall_seconds": {
            "runs": wall_times,
            "median": statistics.median(wall_times),
        },
        "memory": {
            "tracemalloc_peak_bytes": traced_peak,
            "process_peak_rss": _rss_snapshot(),
            "note": "ru_maxrss is process-lifetime peak; tracemalloc covers Python allocations only",
        },
        "runs": runs,
        "all_invariants_passed": all(run["invariants_passed"] for run in runs),
        "rollout_note": (
            "This synthetic candidate-only benchmark does not prove a production speedup; "
            "compare baseline and candidate on representative authority and no-reference corpora."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["all_invariants_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
