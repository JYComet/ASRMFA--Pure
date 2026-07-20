#!/usr/bin/env python3
"""Vectorised audio energy analysis — NumPy-based, shared across pipeline steps.

Replaces the per-file ``list[float]`` + pure-Python RMS loops in
``adjust_ctc_boundaries.py`` and ``postprocess_textgrids.py``.
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as float32 mono numpy array.  Returns (audio, sample_rate)."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0].copy()
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# Frame RMS (vectorised)
# ---------------------------------------------------------------------------

def frame_rms(audio: np.ndarray, sr: int, frame_ms: float = 5.0
              ) -> tuple[np.ndarray, float]:
    """Compute RMS energy per *frame_ms* frame.

    Returns ``(rms, frame_dur_s)`` where *rms* is a float32 array.
    """
    fs = max(1, int(frame_ms / 1000.0 * sr))
    n_frames = max(0, (len(audio) - fs) // fs + 1)
    if n_frames == 0:
        return np.array([], dtype=np.float32), 0.0

    frames = audio[:n_frames * fs].reshape(n_frames, fs)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return rms.astype(np.float32), fs / sr


def word_rms(audio: np.ndarray, sr: int, xmin: float, xmax: float) -> float:
    """Mean absolute amplitude in time slice *[xmin, xmax)*."""
    s = max(0, int(xmin * sr))
    e = min(len(audio), int(xmax * sr))
    if e <= s:
        return 0.0
    return float(np.mean(np.abs(audio[s:e])))


def segment_rms_above_ratio(audio: np.ndarray, sr: int,
                            start_s: float, end_s: float,
                            noise_floor: float, ratio: float) -> float:
    """Fraction of frames in *[start_s, end_s)* with RMS > *noise_floor* × *ratio*."""
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s * sr))
    if e <= s:
        return 0.0
    rms, _ = frame_rms(audio[s:e], sr, frame_ms=5.0)
    if len(rms) == 0:
        return 0.0
    return float(np.mean(rms > noise_floor * ratio))


# ---------------------------------------------------------------------------
# Noise floor (O(n) via partition — no sort)
# ---------------------------------------------------------------------------

def noise_floor_from_rms(rms: np.ndarray, bottom_pct: float = 0.10) -> float:
    """Estimate noise floor as the *bottom_pct* percentile of *rms*."""
    if len(rms) == 0:
        return 0.0
    k = max(1, int(len(rms) * bottom_pct))
    return float(np.partition(rms, k)[k])


def global_noise_floor(audio: np.ndarray, sr: int,
                       frame_ms: float = 5.0,
                       bottom_pct: float = 0.10) -> float:
    """Convenience: frame RMS → noise floor in one call."""
    rms, _ = frame_rms(audio, sr, frame_ms=frame_ms)
    return noise_floor_from_rms(rms, bottom_pct=bottom_pct)


# ---------------------------------------------------------------------------
# Speech onset / offset detection (vectorised)
# ---------------------------------------------------------------------------

def speech_onset(rms: np.ndarray, start_frame: int, threshold: float,
                 min_consecutive: int = 3) -> int | None:
    """Find first frame ≥ *threshold* with *min_consecutive* sustained frames.

    Returns frame index, or None.
    """
    above = np.where(rms[start_frame:] >= threshold)[0]
    if len(above) == 0:
        return None
    # Find first run of min_consecutive consecutive above-threshold frames
    diffs = np.diff(above)
    run_starts = np.where(np.concatenate(([True], diffs != 1)))[0]
    run_lens = np.diff(np.concatenate((run_starts, [len(above)])))
    for i in range(len(run_starts)):
        if run_lens[i] >= min_consecutive:
            return int(start_frame + above[run_starts[i]])
    return None


def speech_offset(rms: np.ndarray, start_frame: int, end_frame: int,
                  threshold: float, min_consecutive: int = 3) -> int | None:
    """Find last frame ≥ *threshold* (searching backwards from *end_frame*)."""
    segment = rms[start_frame:end_frame + 1]
    above = np.where(segment >= threshold)[0]
    if len(above) == 0:
        return None
    # Find last run of min_consecutive
    diffs = np.diff(above)
    run_ends = np.where(np.concatenate((diffs != 1, [True])))[0]
    run_lens = np.diff(np.concatenate(([-1], run_ends)))
    for i in range(len(run_ends) - 1, -1, -1):
        if run_lens[i] >= min_consecutive:
            return int(start_frame + above[run_ends[i]])
    return None


# ---------------------------------------------------------------------------
# Median (vectorised)
# ---------------------------------------------------------------------------

def median(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(values))
