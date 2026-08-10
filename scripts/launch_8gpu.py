#!/usr/bin/env python3
"""
8-GPU parallel launcher for English TTS MFA pipeline.

Splits the 54,000 stems into 8 shards, each processed by an independent
pipeline instance on a dedicated GPU (via CUDA_VISIBLE_DEVICES).

Architecture:
  GPU 0: stems [    0 ..  6749]  ─┐
  GPU 1: stems [ 6750 .. 13499]   │
  GPU 2: stems [13500 .. 20249]   │  Each runs full nvrasr_fallback pipeline:
  GPU 3: stems [20250 .. 26999]   │    prealign → pad_silence → resample →
  GPU 4: stems [27000 .. 33749]   │    align → align_en → postprocess
  GPU 5: stems [33750 .. 40499]   │
  GPU 6: stems [40500 .. 47249]   │  Outputs merge into shared NAS directory.
  GPU 7: stems [47250 .. 53999]  ─┘

Prerequisites:
  1. Reference text files ({stem}.txt) alongside audio files
     Run first: python scripts/prepare_english_tts.py --write-ref-txt ...
  2. MFA local root set up (/tmp/mfa_root or equivalent)
  3. configs/hecheng_english_mfa.yaml configured

Usage:
  # Dry-run (print commands only)
  python scripts/launch_8gpu.py --dry-run

  # Launch all 8 GPUs
  python scripts/launch_8gpu.py

  # Launch specific GPUs
  python scripts/launch_8gpu.py --gpus 0,1,2,3

  # Custom workspace root
  python scripts/launch_8gpu.py --local-work /ssd/mfa_work

  # Override MFA Python
  python scripts/launch_8gpu.py --mfa-python /path/to/python3

  # Limit total stems (testing)
  python scripts/launch_8gpu.py --total-stems 40
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_CONFIG = PROJECT_ROOT / "configs" / "hecheng_english_mfa.yaml"
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "hecheng_en_mfa"
DEFAULT_MFA_PYTHON = "/home/user/miniconda3/envs/mfa-dev/bin/python3"

TOTAL_STEMS = 54000
DEFAULT_NUM_GPUS = 8
SHARD_SIZE = (TOTAL_STEMS + DEFAULT_NUM_GPUS - 1) // DEFAULT_NUM_GPUS  # 6750


def write_shard_config(base_config: Path, workspace: Path, shard_id: int,
                       offset: int, limit: int, output_config: Path) -> None:
    """Generate a per-GPU config by adding offset/limit to the base config."""
    import yaml
    with open(base_config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    # Set shard-specific parameters
    cfg.setdefault("ctc_prealign", {})["offset"] = offset
    cfg.setdefault("ctc_prealign", {})["limit"] = limit

    # Per-GPU workspace (shared data_dir, separate intermediate files)
    cfg["workspace"] = str(workspace)

    with open(output_config, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def launch_shard(
    shard_id: int,
    offset: int,
    limit: int,
    workspace: Path,
    mfa_python: str,
    config_path: Path,
    dry_run: bool = False,
) -> subprocess.Popen | None:
    """Launch one pipeline instance for a shard on its assigned GPU."""

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_pipeline.py"),
        "--config", str(config_path),
        "--workspace", str(workspace),
        "--python", mfa_python,
        "--skip-normalize_punct", "--skip-normalize",
        "--skip-normalize_ria", "--skip-normalize_en",
        "--skip-adjust",
    ]
    # Respect config: only add --overwrite/--force if allowed
    _allow_overwrite = True
    _allow_force = True
    if config_path.exists():
        try:
            import yaml as _yaml
            _cfg = _yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
            _allow_overwrite = _cfg.get("pipeline", {}).get("allow_overwrite", True)
            _allow_force = _cfg.get("pipeline", {}).get("allow_force", True)
        except Exception:
            pass
    if _allow_overwrite:
        cmd.append("--overwrite")
    if _allow_force:
        cmd.append("--force")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(shard_id)

    print(f"\n{'='*60}")
    print(f"  GPU {shard_id}: stems [{offset} .. {offset + limit - 1}]"
          f" ({limit} files)")
    print(f"  Workspace: {workspace}")
    print(f"  Config:    {config_path}")
    print(f"  Cmd:       {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        return None

    # Redirect stdout/stderr to per-GPU log files
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = open(log_dir / f"gpu{shard_id}_stdout.log", 'w')
    stderr_log = open(log_dir / f"gpu{shard_id}_stderr.log", 'w')

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout_log,
        stderr=stderr_log,
        cwd=str(PROJECT_ROOT),
    )
    print(f"  PID: {proc.pid}")
    return proc


def main():
    parser = argparse.ArgumentParser(
        description="8-GPU parallel launcher for English TTS MFA pipeline")
    parser.add_argument("--config", type=Path, default=BASE_CONFIG,
                        help=f"Base config file (default: {BASE_CONFIG})")
    parser.add_argument("--local-work", type=Path, default=DEFAULT_WORKSPACE,
                        help=f"Workspace root (default: {DEFAULT_WORKSPACE})")
    parser.add_argument("--mfa-python", type=str, default=DEFAULT_MFA_PYTHON,
                        help=f"MFA Python path (default: {DEFAULT_MFA_PYTHON})")
    parser.add_argument("--total-stems", type=int, default=TOTAL_STEMS,
                        help=f"Total stems to split (default: {TOTAL_STEMS})")
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS,
                        help=f"Number of GPUs (default: {DEFAULT_NUM_GPUS})")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Specific GPU IDs to use, comma-separated"
                             " (default: 0..num_gpus-1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--sequential", action="store_true",
                        help="Run GPUs sequentially (for debugging)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    # Validate MFA Python
    mfa_py = Path(args.mfa_python)
    if not mfa_py.exists():
        print(f"WARNING: MFA Python not found: {args.mfa_python}")
        print(f"  (This is OK if running on a different machine)")

    # Determine GPU IDs
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))

    # Calculate shard boundaries
    num_shards = len(gpu_ids)
    shard_size = (args.total_stems + num_shards - 1) // num_shards

    print(f"Total stems:  {args.total_stems}")
    print(f"Num GPUs:     {num_shards} (IDs: {gpu_ids})")
    print(f"Shard size:   ~{shard_size} stems/GPU")
    print(f"Workspace:    {args.local_work}")
    print(f"Config:       {args.config}")
    print(f"Mode:         {'DRY RUN' if args.dry_run else 'LIVE'}")

    processes: list[subprocess.Popen] = []
    config_dir = args.local_work / "shard_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for i, gpu_id in enumerate(gpu_ids):
        offset = i * shard_size
        limit = min(shard_size, args.total_stems - offset)

        if limit <= 0:
            print(f"  GPU {gpu_id}: no stems to process, skipping")
            continue

        shard_workspace = args.local_work / f"gpu{gpu_id}"
        shard_config = config_dir / f"shard_gpu{gpu_id}.yaml"

        write_shard_config(args.config, shard_workspace, gpu_id,
                           offset, limit, shard_config)

        proc = launch_shard(
            shard_id=gpu_id,
            offset=offset,
            limit=limit,
            workspace=shard_workspace,
            mfa_python=args.mfa_python,
            config_path=shard_config,
            dry_run=args.dry_run,
        )

        if proc:
            processes.append((gpu_id, proc))
            if args.sequential:
                print(f"  Waiting for GPU {gpu_id} to complete...")
                proc.wait()
                print(f"  GPU {gpu_id} done (rc={proc.returncode})")

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN complete. {len(gpu_ids)} shard configs written to:")
        print(f"  {config_dir}/")
        print(f"{'='*60}")
        return

    if not args.sequential and processes:
        print(f"\n{'='*60}")
        print(f"  All {len(processes)} GPUs launched. Monitoring...")
        print(f"  Logs: {args.local_work}/gpuN/logs/")
        print(f"{'='*60}")

        # Wait for all processes
        failed = []
        for gpu_id, proc in processes:
            rc = proc.wait()
            if rc != 0:
                failed.append((gpu_id, rc))
                print(f"  GPU {gpu_id}: FAILED (rc={rc})")
            else:
                print(f"  GPU {gpu_id}: DONE")

        if failed:
            print(f"\n  FAILURES: {len(failed)} GPU(s)")
            for gpu_id, rc in failed:
                print(f"    GPU {gpu_id}: rc={rc}")
            return 1
        else:
            print(f"\n  ALL {len(processes)} GPUs completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
