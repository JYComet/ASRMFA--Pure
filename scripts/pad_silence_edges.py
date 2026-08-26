#!/usr/bin/env python3
"""
Pad/trim head and tail silence to target duration, shifting CTC timestamps.

在 MFA 前统一音频首尾静音长度:
- 检测开头/结尾现有静音长度
- 过长则裁剪，不足则补零 → 统一到 target_silence_sec (默认 0.5s)
- 平移所有 CTC 时间戳 (TextGrid / tokens.jsonl / punct.json)
- 输出补全后的音频到 padded_audio_dir 和 output_audio_dir
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from trim_silence_batch import _detect_silence_at_beginning_vec, _detect_silence_at_end_vec
from pipeline_utils import find_wav

# Matches xmin = <float> / xmax = <float> in Praat TextGrid
_TG_TIME_RE = re.compile(r'^\s*(xmin|xmax)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$')


def load_exact_stems_file(path: Path) -> list[str]:
    """Load the immutable runner denominator without normalizing any line."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"stems file must be an ordinary file: {path}")
    try:
        stems = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError("stems file must be UTF-8") from exc
    if (not stems or any(not stem or stem.strip() != stem or Path(stem).name != stem
                         for stem in stems)):
        raise ValueError("stems file contains an empty, altered, or unsafe stem")
    if stems != sorted(stems) or len(stems) != len(set(stems)):
        raise ValueError("stems file must be sorted and unique")
    return stems


def validate_completion(expected_stems: list[str], results: list[dict],
                        padded_dir: Path, dry_run: bool,
                        output_dir: Path | None = None) -> list[str]:
    """Return fail-closed padding contract violations."""
    expected = set(expected_stems)
    issues: list[str] = []
    returned = [result.get("stem") for result in results if isinstance(result, dict)]
    returned_set = {stem for stem in returned if isinstance(stem, str)}
    if len(returned) != len(results) or len(returned) != len(returned_set):
        issues.append("worker results contain a missing/duplicate stem")
    if returned_set != expected:
        issues.append(
            f"worker result set mismatch: missing={len(expected - returned_set)}, "
            f"extra={len(returned_set - expected)}")
    errored = sorted(
        str(result.get("stem")) if isinstance(result, dict) else "<invalid-result>"
        for result in results
        if not isinstance(result, dict) or result.get("error")
    )
    if errored:
        issues.append(f"worker errors={len(errored)}: {errored[:10]}")
    if not dry_run:
        expected_names = {f"{stem}.wav" for stem in expected}
        if not padded_dir.is_dir():
            issues.append(f"padded output directory missing: {padded_dir}")
        else:
            entries = list(padded_dir.iterdir())
            ordinary_names = {entry.name for entry in entries
                              if entry.is_file() and not entry.is_symlink()}
            if (ordinary_names != expected_names or len(entries) != len(expected_names)
                    or any(entry.is_symlink() or not entry.is_file() for entry in entries)):
                issues.append(
                    f"padded WAV set mismatch: expected={len(expected_names)}, "
                    f"ordinary_files={len(ordinary_names)}, entries={len(entries)}")
        if output_dir is not None:
            if not output_dir.is_dir():
                issues.append(f"secondary audio output directory missing: {output_dir}")
            else:
                entries = list(output_dir.iterdir())
                ordinary_names = {entry.name for entry in entries
                                  if entry.is_file() and not entry.is_symlink()}
                if (ordinary_names != expected_names or len(entries) != len(expected_names)
                        or any(entry.is_symlink() or not entry.is_file() for entry in entries)):
                    issues.append("secondary padded WAV set mismatch")
    return issues


def shift_textgrid_timestamps(tg_path: Path, offset_s: float) -> None:
    """Shift all xmin/xmax values in a Praat TextGrid by *offset_s* seconds."""
    if offset_s == 0.0:
        return
    lines = tg_path.read_text(encoding='utf-8').splitlines()
    new_lines = []
    for line in lines:
        m = _TG_TIME_RE.match(line)
        if m:
            key, val = m.group(1), float(m.group(2))
            new_val = max(0.0, val + offset_s)
            # Preserve integer formatting if original was integer-like
            if val == int(val) and offset_s == int(offset_s):
                new_lines.append(f'{key} = {int(new_val)}')
            else:
                new_lines.append(f'{key} = {new_val:.6f}')
        else:
            new_lines.append(line)
    tg_path.write_text('\n'.join(new_lines), encoding='utf-8')


def shift_tokens_timestamps(tokens_path: Path, offset_s: float) -> None:
    """Shift start_s/end_s/start_ms/end_ms in _tokens.jsonl."""
    if offset_s == 0.0:
        return
    new_lines = []
    for line in tokens_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        tok = json.loads(line)
        tok['start_s'] = max(0.0, tok['start_s'] + offset_s)
        tok['end_s'] = max(0.0, tok['end_s'] + offset_s)
        tok['start_ms'] = max(0.0, tok['start_ms'] + offset_s * 1000)
        tok['end_ms'] = max(0.0, tok['end_ms'] + offset_s * 1000)
        new_lines.append(json.dumps(tok, ensure_ascii=False))
    tokens_path.write_text('\n'.join(new_lines), encoding='utf-8')


def shift_punct_timestamps(punct_path: Path, offset_s: float) -> None:
    """Shift start_s/end_s/start_ms/end_ms in _punct.json."""
    if offset_s == 0.0:
        return
    punct = json.loads(punct_path.read_text(encoding='utf-8'))
    for p in punct:
        p['start_s'] = max(0.0, p['start_s'] + offset_s)
        p['end_s'] = max(0.0, p['end_s'] + offset_s)
        p['start_ms'] = max(0.0, p['start_ms'] + offset_s * 1000)
        p['end_ms'] = max(0.0, p['end_ms'] + offset_s * 1000)
    punct_path.write_text(json.dumps(punct, ensure_ascii=False), encoding='utf-8')


def process_one(
    stem: str,
    audio_dir: Path,
    ctc_dir: Path,
    padded_audio_dir: Path,
    output_audio_dir: Path | None,
    target_silence_sec: float = 0.5,
    silence_threshold: float = 0.001,
    frame_length: int = 1024,
    dry_run: bool = False,
    wav_index: dict[str, str] | None = None,
    shift_ctc: bool = True,
) -> dict:
    """Process one stem: detect → pad/trim → shift timestamps → save."""

    if wav_index and stem in wav_index:
        wav_path = Path(wav_index[stem])
    else:
        wav_path = find_wav(audio_dir, stem)
    if wav_path is None:
        return {"stem": stem, "error": "no wav found"}

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)
    original_dur = len(audio) / sr

    # ── Detect existing edge silence ──
    head_sil = float(_detect_silence_at_beginning_vec(
        audio, sr, silence_threshold=silence_threshold, frame_length=frame_length))
    tail_sil = float(_detect_silence_at_end_vec(
        audio, sr, silence_threshold=silence_threshold, frame_length=frame_length))

    target_samples = int(target_silence_sec * sr)
    head_samples = int(head_sil * sr)

    # ── Head: pad or trim ──
    if head_sil > target_silence_sec + 0.001:
        # Too much head silence → trim
        trim = head_samples - target_samples
        audio = audio[trim:]
        time_offset = -head_sil + target_silence_sec
    elif head_sil < target_silence_sec - 0.001:
        # Not enough head silence → pad
        pad = target_samples - head_samples
        audio = np.concatenate([np.zeros(pad, dtype=np.float32), audio])
        time_offset = target_silence_sec - head_sil
    else:
        time_offset = 0.0

    # ── Tail: pad or trim ──
    # Re-detect tail on modified audio (head change may have shifted sample count)
    tail_sil_after = float(_detect_silence_at_end_vec(
        audio, sr, silence_threshold=silence_threshold, frame_length=frame_length))
    tail_samples = int(tail_sil_after * sr)

    if tail_sil_after > target_silence_sec + 0.001:
        trim = tail_samples - target_samples
        audio = audio[:max(0, len(audio) - trim)]
    elif tail_sil_after < target_silence_sec - 0.001:
        pad = target_samples - tail_samples
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

    new_dur = len(audio) / sr

    if not dry_run:
        # ── Save padded audio (flat layout) ──
        padded_audio_dir.mkdir(parents=True, exist_ok=True)
        out_wav = padded_audio_dir / f"{stem}.wav"
        sf.write(str(out_wav), audio, sr)

        # Also save to output dir if specified
        if output_audio_dir:
            output_audio_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_audio_dir / f"{stem}.wav"), audio, sr)

        # ── Shift CTC timestamps ──
        if shift_ctc and abs(time_offset) > 0.0001:
            # TextGrid
            tg_path = ctc_dir / f"{stem}.TextGrid"
            if tg_path.exists():
                shift_textgrid_timestamps(tg_path, time_offset)

            # tokens.jsonl
            tokens_path = ctc_dir / f"{stem}_tokens.jsonl"
            if tokens_path.exists():
                shift_tokens_timestamps(tokens_path, time_offset)

            # punct.json
            punct_path = ctc_dir / f"{stem}_punct.json"
            if punct_path.exists():
                shift_punct_timestamps(punct_path, time_offset)

    return {
        "stem": stem,
        "original_dur": round(original_dur, 3),
        "head_sil_before": round(head_sil, 3),
        "tail_sil_before": round(tail_sil, 3),
        "new_dur": round(new_dur, 3),
        "time_offset": round(time_offset, 3),
        "padded_wav": str(out_wav) if not dry_run else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pad/trim head and tail silence, shift CTC timestamps")
    parser.add_argument("--ctc-dir", required=True, help="CTC directory (ctc_pretg)")
    parser.add_argument("--audio-dir", required=True, help="Original audio directory")
    parser.add_argument("--padded-audio-dir", required=True,
                        help="Output directory for padded audio (used by downstream MFA)")
    parser.add_argument("--output-audio-dir", default=None,
                        help="Output directory for final padded audio (saved alongside results)")
    parser.add_argument("--target-silence-sec", type=float, default=0.5)
    parser.add_argument("--silence-threshold", type=float, default=0.001)
    parser.add_argument("--frame-length", type=int, default=1024)
    stem_group = parser.add_mutually_exclusive_group()
    stem_group.add_argument("--stem", default=None, help="Process a single stem")
    stem_group.add_argument("--stems-file", default=None,
                            help="Sorted UTF-8 denominator supplied by the strict runner")
    parser.add_argument("--wav-index", default=None, help="Pre-built wav_index.json (avoids slow glob on CIFS)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pre-ctc", action="store_true",
                        help="Consume a physical WAV denominator before CTC exists")
    args = parser.parse_args()

    # Load pre-built wav index if available
    wav_index: dict[str, str] | None = None
    if args.wav_index:
        wav_index = json.loads(Path(args.wav_index).read_text(encoding='utf-8'))
        print(f"Loaded wav index: {len(wav_index)} stems")

    ctc_dir = Path(args.ctc_dir)
    audio_dir = Path(args.audio_dir)
    padded_dir = Path(args.padded_audio_dir)
    output_dir = Path(args.output_audio_dir) if args.output_audio_dir else None

    # Discover stems from .lab files in ctc_dir
    if args.stems_file:
        try:
            stems = load_exact_stems_file(Path(args.stems_file))
        except ValueError as exc:
            parser.error(str(exc))
        actual_labs = {
            path.stem for path in ctc_dir.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix == ".lab"
        }
        unsafe_labs = [path for path in ctc_dir.iterdir()
                       if path.name.endswith(".lab")
                       and (path.is_symlink() or not path.is_file())]
        if (not args.pre_ctc) and (unsafe_labs or actual_labs != set(stems)):
            print("ERROR: CTC .lab set does not equal the supplied denominator")
            print(f"  missing={len(set(stems) - actual_labs)}, "
                  f"extra={len(actual_labs - set(stems))}, unsafe={len(unsafe_labs)}")
            return 1
    elif args.stem:
        stems = [args.stem]
    else:
        stems = []
        for p in ctc_dir.iterdir():
            if p.is_file() and p.suffix == '.lab':
                stems.append(p.stem)
        if not stems:
            # Nested: {stem}/{stem}.lab
            for p in ctc_dir.iterdir():
                if p.is_dir():
                    lab = p / f"{p.name}.lab"
                    if lab.exists():
                        stems.append(p.name)
        stems.sort()

    # ── Parallel processing with ProcessPoolExecutor ──
    # ThreadPoolExecutor hangs due to soundfile/libsndfile not being
    # fully thread-safe across worker threads.  ProcessPoolExecutor
    # gives each worker its own libsndfile context, avoiding deadlocks.
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not stems:
        print("ERROR: no stems to process")
        return 1
    _n_workers = min(_mp.cpu_count(), 64, len(stems))
    print(f"  并行 workers: {_n_workers}")

    results = []
    _done = 0
    _n = len(stems)
    if _n_workers <= 1 or _n <= 100:
        for stem in stems:
            try:
                r = process_one(stem=stem, audio_dir=audio_dir, ctc_dir=ctc_dir,
                                padded_audio_dir=padded_dir, output_audio_dir=output_dir,
                                target_silence_sec=args.target_silence_sec,
                                silence_threshold=args.silence_threshold,
                                frame_length=args.frame_length,
                                dry_run=args.dry_run, wav_index=wav_index,
                                shift_ctc=not args.pre_ctc)
            except Exception as exc:
                r = {"stem": stem, "error": f"worker exception: {exc}"}
            results.append(r)
            _done += 1
            if _done % 1000 == 0:
                print(f"  进度: {_done}/{_n}")
    else:
        with ProcessPoolExecutor(max_workers=_n_workers) as _pool:
            _futures = {}
            for s in stems:
                _fut = _pool.submit(
                    process_one, s, audio_dir, ctc_dir, padded_dir, output_dir,
                    args.target_silence_sec, args.silence_threshold,
                    args.frame_length, args.dry_run, wav_index, not args.pre_ctc)
                _futures[_fut] = s
            for _fut in as_completed(_futures):
                try:
                    r = _fut.result()
                except Exception as _e:
                    stem = _futures[_fut]
                    r = {"stem": stem, "error": f"worker exception: {_e}"}
                    print(f"  ERROR [{stem}]: {_e}")
                results.append(r)
                _done += 1
                if _done % 1000 == 0:
                    print(f"  进度: {_done}/{_n}")

    issues = validate_completion(stems, results, padded_dir, args.dry_run, output_dir)
    ok = sum(1 for r in results if "error" not in r)
    fail = len(results) - ok
    print(f"\nDone. ok={ok}, fail={fail}, total={len(results)}")
    for issue in issues:
        print(f"  ERROR: {issue}")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
