"""Compare lifecycle/copy overhead using isolated synthetic inputs, without models.

Run with --baseline-scripts pointing at a pre-change scripts snapshot. The
same temporary paths and fixtures are used for both versions, in alternating
order. This is a microbenchmark, not a production throughput claim.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _probe(scripts: Path, root: Path, stems: int, batches: int) -> dict:
    sys.path.insert(0, str(scripts.resolve()))
    import run_pipeline as pipeline
    import streaming_pipeline as streaming

    utils = sys.modules[pipeline.materialize_ctc_work.__module__]
    root.mkdir(parents=True)
    raw, work = root / "raw", root / "work"
    raw.mkdir()
    for index in range(stems):
        stem = f"clip{index:04d}"
        for suffix in utils.CTC_SUFFIXES:
            payload = (json.dumps({"word": "ni3", "start_s": 0.0, "end_s": 0.1}) + "\n"
                       if suffix == "_tokens.jsonl" else "unchanged\n")
            (raw / (stem + suffix)).write_text(payload, encoding="utf-8")
    (raw / ".ctc_run_receipt.json").write_text("{}\n", encoding="utf-8")
    manifest = raw / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (raw / ".ctc_normalized").write_text(utils.make_ctc_normalization_marker(
        stems, hashlib.sha256(manifest.read_bytes()).hexdigest()), encoding="utf-8")
    utils.write_ctc_raw_manifest(raw)
    ctx = {"ctc_pretg": raw, "ctc_pretg_adj": work, "workspace": root,
           "data_dir": root, "config": {},
           "expected_stems": [f"clip{index:04d}" for index in range(stems)]}
    counts: Counter = Counter()

    def instrument(original):
        def counted(path):
            parent = Path(path).parent
            counts["raw" if parent == raw else "work" if parent == work else "other"] += 1
            return original(path)
        return counted

    with patch.object(utils, "_sha256_file", instrument(utils._sha256_file)), \
            patch.object(pipeline, "_sha256_file", instrument(pipeline._sha256_file)), \
            contextlib.redirect_stdout(io.StringIO()):
        started = time.perf_counter()
        results = [pipeline.STEPS[name][1](SimpleNamespace(overwrite=False), {},
                   Path("unused-python"), ctx)
                   for name in ("normalize_punct", "normalize", "normalize_ria", "normalize_en")]
        normalize_seconds = time.perf_counter() - started
    if results != [0, 0, 0, 0]:
        raise RuntimeError(f"normalization failed: {results}")
    receipt = json.loads((work / utils.CTC_WORK_RECEIPT_NAME).read_text())
    # Only implementation identities and their derived digest are expected
    # to differ between source revisions. All other receipt fields compare.
    receipt.pop("identity", None)
    receipt["fingerprints"].pop("producer", None)
    receipt["fingerprints"].pop("digest", None)
    work_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(work.iterdir()) if path.name != utils.CTC_WORK_RECEIPT_NAME}
    target = root / "publication"
    target_reads: Counter = Counter()
    original_hash = streaming._sha256_file

    def count_target(path):
        if Path(path).parent == target:
            target_reads[Path(path).name] += 1
        return original_hash(path)

    sources = []
    for index in range(batches):
        source = root / f"batch{index}"
        source.mkdir()
        (source / f"{index}.TextGrid").write_bytes(b"fixed artifact\n" * 1024)
        sources.append(source)
    with patch.object(streaming, "_sha256_file", count_target):
        started = time.perf_counter()
        for source in sources:
            streaming._copy_publication_tree(source, target)
        copy_seconds = time.perf_counter() - started
    published = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(target.iterdir())}
    return {"normalize_seconds": normalize_seconds, "normalize_hash_reads": dict(counts),
            "copy_seconds": copy_seconds, "target_hash_reads": sum(target_reads.values()),
            "receipt": receipt, "work_hashes": work_hashes, "published_hashes": published}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-scripts", type=Path)
    parser.add_argument("--candidate-scripts", type=Path,
                        default=Path(__file__).resolve().parents[1] / "scripts")
    parser.add_argument("--stems", type=int, default=100)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.stems, args.batches, args.repetitions) < 1:
        parser.error("counts must be positive")
    if args.probe:
        print(json.dumps(_probe(args.candidate_scripts, args.probe, args.stems, args.batches)))
        return 0
    if args.baseline_scripts is None:
        parser.error("--baseline-scripts is required")
    observations = {"baseline": [], "candidate": []}
    with tempfile.TemporaryDirectory(prefix="pipeline-efficiency-ab-") as temp:
        case = Path(temp) / "case"
        for index in range(args.repetitions):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            pair = {}
            for name in order:
                if case.exists():
                    shutil.rmtree(case)
                scripts = args.baseline_scripts if name == "baseline" else args.candidate_scripts
                command = [sys.executable, str(Path(__file__).resolve()), "--probe", str(case),
                           "--candidate-scripts", str(scripts), "--stems", str(args.stems),
                           "--batches", str(args.batches)]
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                pair[name] = json.loads(completed.stdout)
            for key in ("receipt", "work_hashes", "published_hashes"):
                if pair["baseline"][key] != pair["candidate"][key]:
                    raise AssertionError(f"A/B mismatch in {key}, repetition {index}")
            for name in pair:
                observations[name].append({key: value for key, value in pair[name].items()
                                           if key not in {"receipt", "work_hashes", "published_hashes"}})
    report = {"scope": "synthetic normalized-skip lifecycle and publication copy only",
              "stems": args.stems, "batches": args.batches, "repetitions": args.repetitions,
              "equivalence": "work bytes, published bytes and receipt fields match except code identity",
              "runs": observations,
              "medians": {name: {key: statistics.median(row[key] for row in rows)
                                  for key in ("normalize_seconds", "copy_seconds", "target_hash_reads")}
                          for name, rows in observations.items()}}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
