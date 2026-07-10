r"""
批量裁剪并规范化 WAV 静音。

这个脚本做什么：
- 递归读取输入目录中的 `.wav` 文件。
- 检测内部静音段，把超过阈值的静音裁剪到指定最大时长(1s)。
- 可选地把开头和结尾静音统一到目标长度，过长则裁短，过短则补零(0.5s)。
- 输出到指定目录，并保留输入目录下的相对路径结构。
- 支持多线程并行处理，适合在 MFA 对齐前统一音频格式和静音长度。

输入：
- `--input-dir`：原始 wav 文件夹。

输出：
- `--output-dir`：处理后的 wav 文件夹，文件名和相对路径与输入目录保持一致。

可选参数：
- `--max-silence-sec`：内部静音最长保留秒数，默认 `1.0`。
- `--sil-vol-threshold`：内部静音能量阈值，默认 `0.001`，越大越容易判为静音。
- `--sil-len-threshold`：至少多长的连续静音才会被当作静音段，默认 `0.08` 秒。
- `--normalize-edges`：是否规范化开头和结尾静音长度。
- `--target-edge-silence-sec`：开头/结尾目标静音长度，默认 `0.5` 秒。
- `--edge-silence-threshold`：检测开头/结尾静音的 RMS 阈值，默认 `0.001`。
- `--edge-frame-length`：检测开头/结尾静音时的帧长，默认 `1024`。
- `--workers`：并行线程数，默认按 CPU 数自动估计。

使用示例：
python scripts/trim_silence_batch.py --input-dir data\raw_wav --output-dir data\wav --max-silence-sec 1.0 --normalize-edges --target-edge-silence-sec 0.5 --edge-silence-threshold 0.001 --edge-frame-length 1024 --workers 8
"""

import argparse
import os
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import soundfile

try:
    from audio_utils import resample_audio
except ImportError:
    from .audio_utils import resample_audio


def _detect_silence_at_beginning_vec(
    wav: np.ndarray, sr: int, silence_threshold: float = 0.01, frame_length: int = 1024
) -> float:
    """Vectorised: find leading silence duration using reshape + argmax."""
    frame_count = len(wav) // frame_length
    if frame_count == 0:
        return 0.0
    frames = wav[:frame_count * frame_length].reshape(frame_count, frame_length)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    first_speech = np.argmax(rms >= silence_threshold)
    if first_speech == 0 and rms[0] >= silence_threshold:
        return 0.0
    if first_speech == 0 and np.all(rms < silence_threshold):
        return frame_count * frame_length / sr
    return float(first_speech * frame_length / sr)


def _detect_silence_at_end_vec(
    wav: np.ndarray, sr: int, silence_threshold: float = 0.01, frame_length: int = 1024
) -> float:
    """Vectorised: find trailing silence duration (reversed frame scan)."""
    frame_count = len(wav) // frame_length
    if frame_count == 0:
        return 0.0
    frames = wav[:frame_count * frame_length].reshape(frame_count, frame_length)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    # Scan reversed: find first speech frame from end
    reversed_rms = rms[::-1]
    first_speech_from_end = np.argmax(reversed_rms >= silence_threshold)
    if first_speech_from_end == 0 and reversed_rms[0] >= silence_threshold:
        return 0.0
    if first_speech_from_end == 0 and np.all(reversed_rms < silence_threshold):
        return frame_count * frame_length / sr
    return float(first_speech_from_end * frame_length / sr)

# Keep old names as aliases for compatibility
detect_silence_at_beginning = _detect_silence_at_beginning_vec
detect_silence_at_end = _detect_silence_at_end_vec


def normalize_start_end_silence(
    audio: np.ndarray, sr: int,
    target_silence_sec: float = 0.5,
    silence_threshold: float = 0.01,
    frame_length: int = 1024,
) -> np.ndarray:
    """Normalise leading/trailing silence to *target_silence_sec*."""
    beginning_silence = _detect_silence_at_beginning_vec(
        audio, sr, silence_threshold=silence_threshold, frame_length=frame_length
    )
    end_silence = _detect_silence_at_end_vec(
        audio, sr, silence_threshold=silence_threshold, frame_length=frame_length
    )

    target_samples = int(target_silence_sec * sr)
    beginning_samples = int(beginning_silence * sr)
    end_samples = int(end_silence * sr)

    # --- beginning ---
    if beginning_silence > target_silence_sec:
        audio = audio[beginning_samples - target_samples:]
    elif beginning_silence < target_silence_sec:
        pad = target_samples - beginning_samples
        audio = np.concatenate([np.zeros(pad, dtype=np.float32),
                                np.asarray(audio, dtype=np.float32)])
    else:
        audio = np.asarray(audio, dtype=np.float32)

    # --- end (offset by beginning padding, no re-scan needed) ---
    pad_offset = max(0, target_samples - beginning_samples) if beginning_silence < target_silence_sec else 0
    effective_end = end_silence
    if pad_offset > 0 and beginning_silence < target_silence_sec:
        # Audio was padded at front; end silence unchanged but frame alignment shifted
        effective_end = end_silence

    end_samples_actual = int(effective_end * sr)
    if effective_end > target_silence_sec:
        cut = end_samples_actual - target_samples
        audio = audio[:max(0, len(audio) - cut)]
    elif effective_end < target_silence_sec:
        pad = target_samples - end_samples_actual
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

    return audio


def search_silence_ranges(
    wav: np.ndarray, sr: int,
    sil_vol_threshold: float = 0.003,
    sil_len_threshold: float = 0.08,
) -> list[list[int]]:
    """Vectorised: find speech ranges by reshaping into 10ms frames."""
    wav_len = wav.shape[0]
    step = int(sr * 0.01)
    n_steps = max(0, wav_len // step)
    if n_steps == 0:
        return []

    # Vectorised: reshape into [n_steps, step], compute max abs per step
    frames = wav[:n_steps * step].reshape(n_steps, step)
    sil_labels = np.max(np.abs(frames), axis=1) < sil_vol_threshold

    spk_ranges = []
    spk_start = None
    min_spk_n = 10
    ext_n = 0

    for i in range(n_steps):
        if sil_labels[i]:
            if spk_start is not None and i - spk_start > min_spk_n:
                spk_ranges.append([max(0, spk_start - ext_n), min(n_steps, i + ext_n)])
            spk_start = None
        else:
            if spk_start is None:
                spk_start = i

    if spk_start is not None and n_steps - spk_start > 3:
        spk_ranges.append([max(0, spk_start - ext_n), n_steps])

    sil_ranges = []
    if len(spk_ranges) == 0:
        sil_ranges.append([0, wav_len])
        return sil_ranges

    sil_n = int(sil_len_threshold * sr / step)

    if spk_ranges[0][0] > sil_n:
        sil_ranges.append([0, spk_ranges[0][0]])

    for i in range(1, len(spk_ranges)):
        if spk_ranges[i][0] - spk_ranges[i - 1][1] > sil_n:
            sil_ranges.append([spk_ranges[i - 1][1], spk_ranges[i][0]])

    if n_steps - spk_ranges[-1][1] > sil_n:
        sil_ranges.append([spk_ranges[-1][1], n_steps])

    if not sil_ranges:
        return sil_ranges

    sil_ranges = [[r[0] * step, r[1] * step] for r in sil_ranges]
    sil_ranges[-1][1] = min(sil_ranges[-1][1], wav_len)
    return sil_ranges


def trim_excessive_silence(
    audio,
    sr,
    max_silence_sec=1.0,
    sil_vol_threshold=0.003,
    sil_len_threshold=0.08,
):
    sil_ranges = search_silence_ranges(
        audio,
        sr,
        sil_vol_threshold=sil_vol_threshold,
        sil_len_threshold=sil_len_threshold,
    )

    max_sil_samples = int(sr * max_silence_sec)
    new_segments = []
    current_pos = 0
    audio_len = len(audio)

    for s_i, e_i in sil_ranges:
        if s_i > current_pos:
            new_segments.append(audio[current_pos:s_i])

        sil_len = e_i - s_i
        if sil_len > 0:
            keep_len = min(sil_len, max_sil_samples)
            new_segments.append(audio[s_i:s_i + keep_len])

        current_pos = e_i

    if audio_len > current_pos:
        new_segments.append(audio[current_pos:audio_len])

    if not new_segments:
        return audio
    return np.concatenate(new_segments)


def process_one_file(
    input_wav: Path,
    input_root: Path,
    output_root: Path,
    max_silence_sec: float,
    sil_vol_threshold: float,
    sil_len_threshold: float,
    normalize_edges: bool,
    target_edge_silence_sec: float,
    edge_silence_threshold: float,
    edge_frame_length: int,
    target_sr: int | None = None,
):
    audio, sr = soundfile.read(str(input_wav))
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if target_sr and target_sr != sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    trimmed = trim_excessive_silence(
        audio=audio,
        sr=sr,
        max_silence_sec=max_silence_sec,
        sil_vol_threshold=sil_vol_threshold,
        sil_len_threshold=sil_len_threshold,
    )
    if normalize_edges:
        trimmed = normalize_start_end_silence(
            trimmed,
            sr,
            target_silence_sec=target_edge_silence_sec,
            silence_threshold=edge_silence_threshold,
            frame_length=edge_frame_length,
        )

    rel_path = input_wav.relative_to(input_root)
    output_path = output_root / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output_path), trimmed, sr)

    return str(input_wav), str(output_path), len(audio), len(trimmed), sr


def collect_wavs(input_root: Path):
    return [p for p in input_root.rglob("*.wav") if p.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description="Batch trim excessive silence for wav files (no segmentation)."
    )
    parser.add_argument("--input-dir", required=True, help="Input root directory")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--max-silence-sec", type=float, default=1.0)
    parser.add_argument("--sil-vol-threshold", type=float, default=0.001)
    parser.add_argument("--sil-len-threshold", type=float, default=0.08)
    parser.add_argument(
        "--normalize-edges",
        action="store_true",
        help="Normalize start/end silence to target duration",
    )
    parser.add_argument("--target-edge-silence-sec", type=float, default=0.5)
    parser.add_argument("--edge-silence-threshold", type=float, default=0.001)
    parser.add_argument("--edge-frame-length", type=int, default=1024)
    parser.add_argument("--target-sr", type=int, default=None,
                        help="Resample audio to target sample rate (e.g. 16000).")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 1) + 4)),
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    wav_files = collect_wavs(input_root)
    total = len(wav_files)
    if total == 0:
        print(f"No wav files found under: {input_root}")
        return

    print(f"Found {total} wav files. Start processing with workers={args.workers}")
    ok = 0
    fail = 0

    from concurrent.futures import ProcessPoolExecutor as _PPE
    with _PPE(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one_file,
                wav,
                input_root,
                output_root,
                args.max_silence_sec,
                args.sil_vol_threshold,
                args.sil_len_threshold,
                args.normalize_edges,
                args.target_edge_silence_sec,
                args.edge_silence_threshold,
                args.edge_frame_length,
                args.target_sr,
            )
            for wav in wav_files
        ]

        for idx, fut in enumerate(as_completed(futures), start=1):
            try:
                in_path, out_path, before_n, after_n, sr = fut.result()
                ok += 1
                print(
                    f"[{idx}/{total}] OK {in_path} -> {out_path} "
                    f"({before_n / sr:.2f}s -> {after_n / sr:.2f}s)"
                )
            except Exception as e:
                fail += 1
                print(f"[{idx}/{total}] FAIL {e}")

    print(f"Done. success={ok}, failed={fail}, total={total}")


if __name__ == "__main__":
    main()
