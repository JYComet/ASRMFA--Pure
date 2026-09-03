#!/usr/bin/env python3
"""Stage GAMEDATA game audio + reference text from NAS to NVMe.

Flattens each game into a single directory of ``<stem>.wav`` (+ ``<stem>.txt``
for reference games).  Per-game rules are encoded in ``GAMES``:

- reference games : copy wav(+matching txt) pairs, flat, stem = basename
- 环行旅舍 (huanxing) : stem = ``<character>__<basename>`` because every
  character reuses the same generic line numbers (e.g. ``0001_签约_zh``).
- 异环 (yihuan) : convert ``.ogg`` -> ``.wav`` (PCM_16 mono) via ffmpeg.
- 白荆回廊 (baijing) : no-reference; drop ``*__*.wav`` content-hash duplicates.
- 重返未来1999 (reverse1999) : no-reference; wav only (ignore reference txt).

Idempotent: existing destination files are left untouched unless ``--overwrite``.
Writes a JSON manifest + orphan-audio/orphan-txt reports per game.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FFMPEG = "/home/user/miniconda3/envs/mfa-dev/bin/ffmpeg"

SRC_ROOT = Path("/mnt/nas/Research_TTS/Data/Raw/GAMEDATA")
DST_ROOT = Path("/mnt/nvme3/gamedata_20260903")

# codename -> (source folder, reference mode, flags)
GAMES = [
    ("genshin",      "原神",         True,  dict()),
    ("snowbreak",    "尘白禁区",     True,  dict()),
    ("yihuan",       "异环",         True,  dict(convert_ogg=True)),
    ("huanxing",     "环行旅舍",     True,  dict(char_prefix=True)),
    ("baijing",      "白荆回廊",     False, dict(drop_hash=True)),
    ("zhongmodi",    "终末地",       True,  dict()),
    ("zzz",          "绝区零",       True,  dict()),
    ("reverse1999",  "重返未来1999", False, dict()),
]


def convert_ogg_to_wav(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(src),
           "-ac", "1", "-c:a", "pcm_s16le", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0 or not tmp.exists():
        return False
    os.replace(tmp, dst)
    return True


def plan_game(src_dir: Path, ref: bool, flags: dict):
    """Enumerate (audio_src, txt_src|None, stem) items to stage."""
    char_prefix = flags.get("char_prefix", False)
    drop_hash = flags.get("drop_hash", False)
    convert_ogg = flags.get("convert_ogg", False)

    audio_exts = (".ogg",) if convert_ogg else (".wav",)
    if convert_ogg:
        audio_exts = (".wav", ".ogg")

    items = []
    for dirpath, _dirnames, filenames in os.walk(src_dir):
        d = Path(dirpath)
        char = d.name
        for fn in filenames:
            p = d / fn
            lower = fn.lower()
            if not (lower.endswith(".wav") or (convert_ogg and lower.endswith(".ogg"))):
                continue
            base = p.stem  # basename without extension
            if drop_hash and "__" in base:
                continue
            stem = f"{char}__{base}" if char_prefix else base
            # matching reference txt sits beside the audio, same basename
            txt = None
            if ref:
                cand = d / f"{base}.txt"
                if cand.is_file():
                    txt = cand
            items.append((p, txt, stem))
    return items


def stage_one(src_dir: Path, dst_dir: Path, ref: bool, flags: dict, overwrite: bool):
    items = plan_game(src_dir, ref, flags)
    convert_ogg = flags.get("convert_ogg", False)

    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    orphan_audio = []
    copied = 0
    skipped = 0
    errors = []

    def do_item(item):
        audio_src, txt_src, stem = item
        audio_dst = dst_dir / f"{stem}.wav"
        if audio_dst.exists() and not overwrite:
            return ("skip", stem, audio_src, txt_src)
        try:
            if convert_ogg and audio_src.suffix.lower() == ".ogg":
                if not convert_ogg_to_wav(audio_src, audio_dst):
                    return ("error", stem, audio_src, f"ffmpeg failed: {audio_src}")
            else:
                shutil.copy2(audio_src, audio_dst)
            if txt_src is not None:
                shutil.copy2(txt_src, dst_dir / f"{stem}.txt")
            return ("ok", stem, audio_src, txt_src)
        except OSError as e:
            return ("error", stem, audio_src, str(e))

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(do_item, it) for it in items]
        for fut in as_completed(futs):
            status, stem, audio_src, extra = fut.result()
            if status == "ok":
                copied += 1
                manifest[stem] = {
                    "audio": str(audio_src),
                    "txt": str(extra) if isinstance(extra, Path) else None,
                }
            elif status == "skip":
                skipped += 1
            else:
                errors.append({"stem": stem, "audio": str(audio_src), "err": extra})

    # orphan detection (ref mode): audio without txt, txt without audio
    orphan_txt = []
    if ref:
        audio_bases = {}
        for p, txt, stem in items:
            audio_bases[p.with_suffix("").name] = True
        for dirpath, _dn, filenames in os.walk(src_dir):
            for fn in filenames:
                if fn.lower().endswith(".txt"):
                    base = fn[:-4]
                    if base not in audio_bases:
                        orphan_txt.append(str(Path(dirpath) / fn))
        orphan_audio = [str(a) for a, t, _s in items if t is None]

    return {
        "copied": copied, "skipped": skipped, "total_audio": len(items),
        "errors": errors, "orphan_audio": orphan_audio, "orphan_txt": orphan_txt,
        "manifest": manifest,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", help="comma-separated codenames to stage")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    summary = {}
    for codename, src_name, ref, flags in GAMES:
        if only and codename not in only:
            continue
        src_dir = SRC_ROOT / src_name
        dst_dir = DST_ROOT / codename
        print(f"\n===== {src_name} -> {codename} (ref={ref}) =====")
        res = stage_one(src_dir, dst_dir, ref, flags, args.overwrite)
        summary[codename] = {
            "src": src_name, "ref": ref,
            "total_audio": res["total_audio"], "copied": res["copied"],
            "skipped": res["skipped"], "errors": len(res["errors"]),
            "orphan_audio": len(res["orphan_audio"]),
            "orphan_txt": len(res["orphan_txt"]),
        }
        # write manifest + reports
        (dst_dir / ".stage_manifest.json").write_text(
            json.dumps(res["manifest"], ensure_ascii=False, indent=1),
            encoding="utf-8")
        if res["errors"]:
            (dst_dir / ".stage_errors.json").write_text(
                json.dumps(res["errors"], ensure_ascii=False, indent=1),
                encoding="utf-8")
        if res["orphan_audio"]:
            (dst_dir / ".orphan_audio.txt").write_text(
                "\n".join(res["orphan_audio"]), encoding="utf-8")
        if res["orphan_txt"]:
            (dst_dir / ".orphan_txt.txt").write_text(
                "\n".join(res["orphan_txt"]), encoding="utf-8")
        print(f"  total_audio={res['total_audio']} copied={res['copied']} "
              f"skipped={res['skipped']} errors={len(res['errors'])} "
              f"orphan_audio={len(res['orphan_audio'])} orphan_txt={len(res['orphan_txt'])}")

    print("\n===== SUMMARY =====")
    for codename, s in summary.items():
        print(f"{codename:14s} src={s['src']:10s} ref={s['ref']!s:5s} "
              f"audio={s['total_audio']:6d} copied={s['copied']:6d} "
              f"err={s['errors']:4d} orphan_audio={s['orphan_audio']:5d} "
              f"orphan_txt={s['orphan_txt']:5d}")
    (DST_ROOT / ".stage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
