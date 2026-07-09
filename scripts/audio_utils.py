"""Shared audio processing utilities."""

import numpy as np


def resample_audio(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio to target sample rate.

    Uses decimate for integer-ratio downsampling, resample_poly otherwise.
    Returns float32 array.
    """
    if src_sr == target_sr:
        return audio.astype(np.float32)

    from scipy.signal import decimate, resample_poly

    if src_sr % target_sr == 0:
        return decimate(audio.astype('float64'), src_sr // target_sr,
                        ftype='iir').astype(np.float32)
    else:
        gcd = np.gcd(src_sr, target_sr)
        return resample_poly(audio.astype('float64'),
                             target_sr // gcd, src_sr // gcd).astype(np.float32)
