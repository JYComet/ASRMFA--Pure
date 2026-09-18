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

def _frame_size(sr: int, frame_ms: float) -> int:
    """Return the sample count used by the existing frame-RMS semantics."""
    fs = max(1, int(frame_ms / 1000.0 * sr))
    return fs


def _frame_rms_uncached(audio: np.ndarray, sr: int, frame_ms: float = 5.0
                        ) -> tuple[np.ndarray, float]:
    """Compute RMS frames without consulting a cache."""
    audio = np.asarray(audio)
    fs = _frame_size(sr, frame_ms)
    n_frames = max(0, (len(audio) - fs) // fs + 1)
    if n_frames == 0:
        return np.array([], dtype=np.float32), 0.0

    frames = audio[:n_frames * fs].reshape(n_frames, fs)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return rms.astype(np.float32), fs / sr


def frame_rms(audio: np.ndarray, sr: int, frame_ms: float = 5.0,
              *, cache: "FrameRmsCache | None" = None,
              alignment: str = "global") -> tuple[np.ndarray, float]:
    """Compute RMS energy per *frame_ms* frame.

    Returns ``(rms, frame_dur_s)`` where *rms* is a float32 array.  Existing
    callers use the uncached path; ``cache`` is an opt-in per-stem cache and
    ``alignment`` distinguishes a full-audio bank from a sliced local one.
    """
    if cache is not None:
        return cache.frames(audio, sr, frame_ms=frame_ms,
                            alignment=alignment)
    return _frame_rms_uncached(audio, sr, frame_ms=frame_ms)


class FrameRmsCache:
    """Per-caller/stem cache for globally or locally aligned RMS frame banks.

    A cache instance is deliberately stateful and has no module-level state.
    The caller should create one instance per stem (or other independent
    owner) and pass it only within that scope.  The object contains only
    ordinary Python dictionaries and NumPy arrays, so it remains picklable
    for the Linux fork path.  Windows workers should receive independent
    instances rather than sharing one cache between threads.

    ``global_frames`` expects the complete audio array and aligns frames from
    sample zero.  ``local_frames`` expects a sliced segment and starts a new
    frame origin at that segment's first sample.  These modes intentionally
    have separate cache keys even when given the same array.
    """

    def __init__(self, stem_id=None):
        self.stem_id = stem_id
        self._banks = {}
        # Keep arrays alive for the lifetime of their identity-based keys so
        # Python cannot recycle an object id for a different audio array.
        self._audio_refs = {}
        self._computation_count = 0

    @property
    def computation_count(self) -> int:
        """Number of frame banks computed by this cache instance."""
        return self._computation_count

    @staticmethod
    def _audio_identity(audio: np.ndarray):
        array = np.asarray(audio)
        data_ptr = int(array.__array_interface__["data"][0])
        return (id(audio), data_ptr, tuple(array.shape),
                tuple(array.strides), array.dtype.str)

    def _key(self, audio: np.ndarray, sr: int, frame_ms: float,
             alignment: str):
        array = np.asarray(audio)
        return (self._audio_identity(audio), sr,
                _frame_size(sr, frame_ms), alignment), array

    def frames(self, audio: np.ndarray, sr: int, frame_ms: float = 5.0,
               *, alignment: str = "global") -> tuple[np.ndarray, float]:
        """Return a cached frame bank for the requested alignment mode."""
        if alignment not in {"global", "local"}:
            raise ValueError("alignment must be 'global' or 'local'")
        key, array = self._key(audio, sr, frame_ms, alignment)
        if key in self._banks:
            return self._banks[key]

        result = _frame_rms_uncached(array, sr, frame_ms=frame_ms)
        # Cached arrays are shared within one explicit owner scope; making
        # them read-only prevents a consumer from corrupting later results.
        result[0].setflags(write=False)
        self._banks[key] = result
        self._audio_refs[key] = array
        self._computation_count += 1
        return result

    def global_frames(self, audio: np.ndarray, sr: int,
                      frame_ms: float = 5.0) -> tuple[np.ndarray, float]:
        """Return frames aligned to the beginning of a full audio array."""
        return self.frames(audio, sr, frame_ms=frame_ms, alignment="global")

    def local_frames(self, audio: np.ndarray, sr: int,
                     frame_ms: float = 5.0) -> tuple[np.ndarray, float]:
        """Return frames aligned to the beginning of a sliced segment."""
        return self.frames(audio, sr, frame_ms=frame_ms, alignment="local")


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
    rms, _ = frame_rms(audio[s:e], sr, frame_ms=5.0,
                       alignment="local")
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
    rms, _ = frame_rms(audio, sr, frame_ms=frame_ms,
                       alignment="global")
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
