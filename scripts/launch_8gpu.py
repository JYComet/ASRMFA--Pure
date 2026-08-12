#!/usr/bin/env python3
"""Compatibility launcher for the former 8-GPU shard entry point.

The previous implementation split one strict CTC-ready run into eight
``run_pipeline.py`` shards.  That is not a safe execution model: shards can
compete for the same strict-run artifacts and have no single resource plan.
This compatibility entry point now launches *one* batch streaming pipeline in
pipelined mode.  ``streaming_pipeline.py`` owns the bounded GPU/CPU plan.

Only batch configurations are accepted.  In particular, this is deliberately
not an entry point for a strict ``mode: ctc_ready`` production run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "batch_all.yaml"
DEFAULT_NUM_GPUS = 8


def _load_batch_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the only configuration class this wrapper supports."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - streaming needs PyYAML too
        raise ValueError("PyYAML is required to validate the batch config") from exc

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"cannot read config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")

    mode = loaded.get("mode")
    strict_ready = loaded.get("strict_ctc_ready", {})
    if mode == "ctc_ready" or (
        isinstance(strict_ready, dict) and strict_ready.get("enabled")
    ):
        raise ValueError(
            "strict ctc_ready configs are old shard targets; run the strict "
            "workflow directly instead of launch_8gpu.py"
        )
    if mode != "batch_ctc_ready" or not isinstance(loaded.get("batch"), dict):
        raise ValueError(
            "launch_8gpu.py requires a batch config "
            "(mode: batch_ctc_ready with a batch: section)"
        )
    return loaded


def _gpu_selection(value: str) -> tuple[str, int]:
    """Return a CUDA visibility mask and worker count for legacy ``--gpus``."""
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("--gpus must contain at least one non-negative GPU ID")
    try:
        parsed = [int(gpu_id) for gpu_id in ids]
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of GPU IDs") from exc
    if any(gpu_id < 0 for gpu_id in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError("--gpus must contain unique non-negative GPU IDs")
    return ",".join(str(gpu_id) for gpu_id in parsed), len(parsed)


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Build one streaming command without creating shard state or processes."""
    _load_batch_config(args.config)

    env = os.environ.copy()
    gpu_workers = args.num_gpus
    if args.gpus is not None:
        mask, gpu_workers = _gpu_selection(args.gpus)
        env["CUDA_VISIBLE_DEVICES"] = mask

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "streaming_pipeline.py"),
        "--config", str(args.config),
        "--gpus", str(gpu_workers),
        "--pipelined",
    ]
    if args.local_work is not None:
        command.extend(("--local-work", str(args.local_work)))
    if args.mfa_python:
        command.extend(("--python", args.mfa_python))
    if args.mfa_jobs is not None:
        command.extend(("--mfa-jobs", str(args.mfa_jobs)))
    if args.mfa_en_jobs is not None:
        command.extend(("--mfa-en-jobs", str(args.mfa_en_jobs)))
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.parallel_datasets is not None:
        command.extend(("--parallel-datasets", str(args.parallel_datasets)))
    if args.cpu_workers is not None:
        command.extend(("--cpu-workers", str(args.cpu_workers)))
    if args.prefetch_buffer is not None:
        command.extend(("--prefetch-buffer", str(args.prefetch_buffer)))
    if args.upload_buffer is not None:
        command.extend(("--upload-buffer", str(args.upload_buffer)))
    if args.total_stems is not None:
        command.extend(("--limit", str(args.total_stems)))
    if args.no_resume:
        command.append("--no-resume")
    return command, env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility launcher: run one pipelined streaming batch, not "
            "eight independent strict-run shards."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Batch config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=DEFAULT_NUM_GPUS,
        help=f"Visible GPU worker count when --gpus is omitted (default: {DEFAULT_NUM_GPUS})",
    )
    parser.add_argument(
        "--gpus", type=str,
        help="Legacy GPU-ID selection, e.g. 0,1,2,3; sets CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--local-work", type=Path, help="Override streaming.local_work")
    parser.add_argument("--mfa-python", help="MFA Python passed to streaming_pipeline.py")
    parser.add_argument("--mfa-jobs", type=int, help="MFA jobs per worker (CPU-budget capped)")
    parser.add_argument("--mfa-en-jobs", type=int, help="English MFA jobs per worker (CPU-budget capped)")
    parser.add_argument("--batch-size", type=int, help="Stems per streaming batch")
    parser.add_argument("--parallel-datasets", type=int, help="Dataset parallelism")
    parser.add_argument("--cpu-workers", type=int, help="Pipelined CPU worker count")
    parser.add_argument("--prefetch-buffer", type=int, help="Pipelined GPU queue size")
    parser.add_argument("--upload-buffer", type=int, help="Pipelined CPU queue size")
    parser.add_argument(
        "--total-stems", type=int,
        help="Legacy compatibility alias for streaming_pipeline.py --limit",
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint state")
    parser.add_argument("--dry-run", action="store_true", help="Print one command without executing")
    parser.add_argument(
        "--sequential", action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.num_gpus < 1:
        parser.error("--num-gpus must be at least 1")
    if args.total_stems is not None and args.total_stems < 1:
        parser.error("--total-stems must be at least 1")
    if args.sequential:
        parser.error("--sequential is obsolete: this wrapper launches one streaming pipeline")
    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")

    try:
        command, env = build_command(args)
    except ValueError as exc:
        parser.error(str(exc))

    print("Compatibility mode: one pipelined streaming batch")
    print(f"Config: {args.config}")
    print(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '<inherited/all>')}")
    print("Command:")
    print("  " + " ".join(command))
    if args.dry_run:
        return 0

    return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
