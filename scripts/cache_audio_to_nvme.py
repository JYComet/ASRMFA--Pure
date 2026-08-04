#!/usr/bin/env python3
"""
Audio → NVMe cache manager for MFA pipeline.

Copies WAV files from NAS to local NVMe storage, preserving speaker
subdirectory structure.  The pipeline auto-detects this cache and uses
it as the audio source, eliminating NAS I/O contention.

Cache layout::

    /mnt/nvme3/mfa_audio_cache/
    ├── ria/
    │   ├── 036000_弹幕互动_回应吐槽弹幕.wav
    │   └── ...
    ├── 花礼/
    │   └── ...
    ├── 雪狐桑/
    │   └── ...
    └── cache_manifest.json

Usage::

    # One-time: populate the permanent NVMe cache
    python scripts/cache_audio_to_nvme.py \\
        --source /mnt/Raw/新版合成英文数据

    # Show cache status
    python scripts/cache_audio_to_nvme.py --status

    # Remove cache
    python scripts/cache_audio_to_nvme.py --remove

    # Custom cache location
    python scripts/cache_audio_to_nvme.py \\
        --source /mnt/Raw/新版合成英文数据 \\
        --cache /mnt/nvme4/my_cache
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_CACHE_ROOT = Path("/mnt/nvme3/mfa_audio_cache")
MANIFEST_NAME = "cache_manifest.json"


def scan_speaker_dirs(source: Path) -> dict[str, list[Path]]:
    """Scan source dir for speaker subdirectories containing .wav files.

    Returns ``{speaker_name: [wav_path, ...]}``.
    """
    speakers: dict[str, list[Path]] = {}
    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        sys.exit(1)

    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            wavs = sorted(entry.glob("*.wav"))
            if wavs:
                speakers[entry.name] = wavs
    return speakers


def build_manifest(cache_root: Path, speakers: dict[str, list[Path]],
                   source: Path) -> dict:
    """Build cache manifest metadata."""
    total_files = 0
    total_bytes = 0
    speaker_info = {}
    for name, wavs in speakers.items():
        n = len(wavs)
        size = sum(w.stat().st_size for w in wavs if w.exists())
        speaker_info[name] = {"files": n, "size_bytes": size}
        total_files += n
        total_bytes += size

    return {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source.resolve()),
        "cache_root": str(cache_root.resolve()),
        "total_files": total_files,
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / 1024**3, 2),
        "speakers": speaker_info,
    }


def copy_speaker(speaker_name: str, wavs: list[Path],
                 cache_root: Path, source: Path) -> tuple[int, int]:
    """Copy one speaker's WAVs to cache. Returns (copied, skipped)."""
    dest_dir = cache_root / speaker_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    n = len(wavs)
    total_bytes = 0
    t0 = time.time()

    for i, src in enumerate(wavs):
        dst = dest_dir / src.name
        if dst.exists():
            # Quick check: same size?
            if dst.stat().st_size == src.stat().st_size:
                skipped += 1
                if (i + 1) % 1000 == 0 or i == n - 1:
                    _print_progress(i + 1, n, t0, total_bytes, speaker_name)
                continue
        shutil.copy2(str(src), str(dst))
        total_bytes += src.stat().st_size
        copied += 1
        if (i + 1) % 1000 == 0 or i == n - 1:
            _print_progress(i + 1, n, t0, total_bytes, speaker_name)

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed * 1024 if elapsed > 0 else 0
    print(f"\r  {speaker_name}: {copied} copied, {skipped} skipped"
          f" | {gb:.1f} GB | {elapsed:.0f}s ({speed:.0f} MB/s)     ")
    return copied, skipped


def _print_progress(current: int, total: int, t0: float,
                    total_bytes: int, label: str = ""):
    pct = current / total * 100
    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed * 1024 if elapsed > 0 else 0
    print(f"\r  [{label}] {current}/{total} ({pct:.0f}%)"
          f" | {gb:.1f} GB | {speed:.0f} MB/s", end="")


def remove_cache(cache_root: Path) -> None:
    """Remove the entire cache directory."""
    if not cache_root.exists():
        print(f"Cache not found: {cache_root}")
        return
    print(f"Removing: {cache_root}")
    shutil.rmtree(str(cache_root), ignore_errors=True)
    if not cache_root.exists():
        print("  Done.")
    else:
        print(f"  WARNING: Some files could not be removed.")


def show_status(cache_root: Path) -> None:
    """Print cache status from manifest."""
    manifest_path = cache_root / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"No cache found at: {cache_root}")
        print(f"Run: python scripts/cache_audio_to_nvme.py --source <NAS_DIR>")
        return

    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"NVMe Audio Cache: {cache_root}")
    print(f"  Source:    {m['source']}")
    print(f"  Created:   {m['created']}")
    print(f"  Total:     {m['total_files']} files, {m['total_size_gb']} GB")
    print(f"  Speakers:")
    for name, info in m.get("speakers", {}).items():
        gb = info["size_bytes"] / 1024**3
        print(f"    {name}: {info['files']} files, {gb:.1f} GB")

    # Verify files actually exist
    missing = 0
    for name in m.get("speakers", {}):
        spk_dir = cache_root / name
        if spk_dir.exists():
            actual = len(list(spk_dir.glob("*.wav")))
            expected = m["speakers"][name]["files"]
            if actual != expected:
                print(f"    ⚠ {name}: expected {expected}, found {actual}")
                missing += expected - actual
        else:
            print(f"    ⚠ {name}: directory missing!")
            missing += m["speakers"][name]["files"]
    if missing == 0:
        print(f"  Status: ✓ complete")
    else:
        print(f"  Status: ⚠ {missing} files missing — re-run to repair")


def cmd_create(args: argparse.Namespace) -> int:
    """Create or update the NVMe audio cache."""
    source = Path(args.source)
    cache_root = Path(args.cache)

    print(f"Scanning: {source}")
    speakers = scan_speaker_dirs(source)
    if not speakers:
        print("ERROR: No speaker directories with .wav files found.")
        return 1

    total_wavs = sum(len(v) for v in speakers.values())
    print(f"Found {len(speakers)} speaker(s):")
    for name, wavs in speakers.items():
        size_gb = sum(w.stat().st_size for w in wavs) / 1024**3
        print(f"  {name}: {len(wavs)} WAVs, ~{size_gb:.1f} GB")
    print(f"Total: {total_wavs} WAVs")
    print(f"Cache: {cache_root}")

    # Check disk space
    cache_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(cache_root))
    needed = sum(
        sum(w.stat().st_size for w in wavs)
        for wavs in speakers.values()
    )
    if usage.free < needed:
        print(f"ERROR: Not enough space. Need {needed/1024**3:.1f} GB,"
              f" have {usage.free/1024**3:.1f} GB")
        return 1

    # Copy each speaker
    t_start = time.time()
    total_copied = 0
    total_skipped = 0
    for name, wavs in speakers.items():
        copied, skipped = copy_speaker(name, wavs, cache_root, source)
        total_copied += copied
        total_skipped += skipped

    # Write manifest
    manifest = build_manifest(cache_root, speakers, source)
    manifest_path = cache_root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")

    elapsed = time.time() - t_start
    print(f"\nDone: {total_copied} copied, {total_skipped} skipped"
          f" in {elapsed:.0f}s")
    print(f"Cache: {cache_root}")
    print(f"Manifest: {manifest_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Audio → NVMe cache manager for MFA pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Source data directory (NAS path)")
    parser.add_argument("--cache", type=str, default=str(DEFAULT_CACHE_ROOT),
                        help=f"Cache root directory (default: {DEFAULT_CACHE_ROOT})")
    parser.add_argument("--status", action="store_true",
                        help="Show cache status and exit")
    parser.add_argument("--remove", action="store_true",
                        help="Remove the cache directory and exit")
    args = parser.parse_args()

    cache_root = Path(args.cache)

    if args.remove:
        remove_cache(cache_root)
        return 0

    if args.status:
        show_status(cache_root)
        return 0

    if not args.source:
        print("ERROR: --source is required to create/populate the cache.")
        print("  python scripts/cache_audio_to_nvme.py --source /mnt/Raw/新版合成英文数据")
        return 1

    return cmd_create(args)


if __name__ == "__main__":
    sys.exit(main())
