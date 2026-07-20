"""Shared audio processing utilities."""

import numpy as np
from math import gcd


def resample_audio(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio to target sample rate.

    Returns float32 array.  Avoids unnecessary float64 copies when the
    input is already float32 (scipy >= 1.6 supports float32 natively).
    """
    if src_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)

    # Ensure float64 for numerical stability in resampling filters
    a = audio.astype(np.float64, copy=False)

    if src_sr % target_sr == 0:
        from scipy.signal import decimate
        return decimate(a, src_sr // target_sr, ftype='fir').astype(np.float32, copy=False)

    from scipy.signal import resample_poly
    g = gcd(src_sr, target_sr)
    return resample_poly(a, target_sr // g, src_sr // g).astype(np.float32, copy=False)
