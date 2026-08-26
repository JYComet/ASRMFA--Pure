"""Torch-free resolution of mutable English CTC geometry.

The CTC producer keeps its short, fixed-frame spans as immutable evidence.
This module derives the separate processed span used by later MFA and
TextGrid stages, so those stages can run in the MFA environment without
importing the torch-backed CTC producer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


FRAME_MS = 60
GEOMETRY_TOLERANCE_S = 0.003
RMS_FRAME_S = 0.005
SUSTAINED_SILENCE_S = 0.030
ACTIVE_FLOOR_MIN = 0.001
NOISE_FLOOR_RATIO = 0.10


class ProcessedGeometryError(ValueError):
    """Structured fail-closed error for invalid processed CTC geometry."""

    def __init__(self, reason_code: str, message: str, **context: object):
        self.reason_code = reason_code
        self.context = {"reason_code": reason_code, **context}
        self.details = self.context
        super().__init__(
            json.dumps({"message": message, **self.context},
                       ensure_ascii=False, sort_keys=True, default=str))


class CanonicalNextOwnerConflict(ProcessedGeometryError):
    """The immutable canonical span overlaps the next lexical owner."""

    def __init__(self, *, canonical_span: list[float],
                 next_lexical_start: float, candidate_start: float,
                 row: dict):
        unit = row.get("canonical_unit")
        unit_context = {
            "word": row.get("word"),
            "surface_text": row.get("surface_text"),
            "reference_identity": row.get("reference_identity"),
            "reference_ordinal": row.get("reference_ordinal"),
            "source_ctc_ordinals": row.get("source_ctc_ordinals"),
            "next_lexical_word": row.get("next_lexical_word"),
        }
        if isinstance(unit, dict):
            unit_context["canonical_unit_id"] = unit.get("unit_id")
        stem_context = (row.get("stem") or row.get("source_stem")
                        or row.get("reference_identity"))
        super().__init__(
            "canonical_next_owner_conflict",
            "canonical English span crosses the next lexical owner",
            canonical_span=canonical_span,
            candidate_start=candidate_start,
            next_lexical_start=next_lexical_start,
            source="ctc_processed_geometry",
            unit_context=unit_context,
            stem_context=stem_context,
        )


def _load_local_rms_profile(wav_path: str | Path | None) -> dict | None:
    """Decode one WAV into a deterministic, globally aligned 5 ms profile."""
    if wav_path is None:
        return None
    try:
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            return None
        if getattr(audio, "ndim", 0) > 1:
            audio = audio[:, 0]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if len(audio) == 0 or not np.all(np.isfinite(audio)):
            return None
        frame_samples = max(1, int(round(RMS_FRAME_S * sample_rate)))
        frame_count = len(audio) // frame_samples
        if frame_count <= 0:
            return None
        frames = audio[:frame_count * frame_samples].reshape(
            frame_count, frame_samples)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        if len(rms) == 0 or not np.all(np.isfinite(rms)):
            return None
        partition_index = min(
            len(rms) - 1, max(1, int(len(rms) * NOISE_FLOOR_RATIO)))
        noise_floor = float(np.partition(rms, partition_index)[partition_index])
        frame_s = frame_samples / sample_rate
        return {
            "rms": rms,
            "frame_s": frame_s,
            "noise_floor": noise_floor,
            "threshold": max(3.0 * noise_floor, ACTIVE_FLOOR_MIN),
            "audio_end": len(audio) / sample_rate,
        }
    except Exception:
        return None


def _first_sustained_silence(
        profile: dict, start_s: float, end_s: float) -> tuple[float, bool] | None:
    """Return the first 30 ms low-energy onset and whether it starts at onset."""
    try:
        import numpy as np

        rms = profile["rms"]
        frame_s = float(profile["frame_s"])
        threshold = float(profile["threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    if frame_s <= 0 or end_s <= start_s:
        return None
    first = max(0, int(math.ceil(start_s / frame_s - 1e-9)))
    stop = min(len(rms), int(math.floor(end_s / frame_s + 1e-9)))
    required = max(1, int(math.ceil(SUSTAINED_SILENCE_S / frame_s - 1e-9)))
    if stop - first < required:
        return None
    below = np.asarray(rms[first:stop]) < threshold
    run = 0
    for offset, is_below in enumerate(below):
        run = run + 1 if bool(is_below) else 0
        if run >= required:
            onset_index = first + offset - required + 1
            onset = round(onset_index * frame_s, 9)
            return onset, onset_index == first
    return None


def _profile_speech_end(profile: dict, search_from_s: float) -> float | None:
    """Mirror the legacy final-word VAD using the already decoded profile."""
    try:
        import numpy as np

        rms = profile["rms"]
        frame_s = float(profile["frame_s"])
    except (KeyError, TypeError, ValueError):
        return None
    first = max(0, int(math.ceil(search_from_s / frame_s - 1e-9)))
    segment = np.asarray(rms[first:])
    if frame_s <= 0 or len(segment) == 0:
        return None
    threshold = float(np.max(segment)) * 0.05
    active = np.flatnonzero(segment > threshold)
    if len(active) == 0:
        return None
    return round((first + int(active[-1])) * frame_s, 3)


def _vad_speech_end(wav_path: str, search_from_s: float) -> float | None:
    """Find the last energetic frame after ``search_from_s``."""
    try:
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        frame_len = int(sample_rate * 0.01)
        hop = frame_len // 2
        if frame_len <= 0:
            return None
        start_sample = int(search_from_s * sample_rate)
        if start_sample >= len(audio):
            return None
        segment = audio[start_sample:]
        frame_count = (len(segment) - frame_len) // hop
        if frame_count <= 0:
            return None
        # A strided view avoids the old Python loop over every 10 ms frame.
        frames = np.lib.stride_tricks.sliding_window_view(
            segment, frame_len)[::hop][:frame_count]
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        if len(rms) == 0:
            return None
        threshold = np.max(rms) * 0.05
        last_speech_frame = len(rms) - 1
        for index in range(len(rms) - 1, -1, -1):
            if rms[index] > threshold:
                last_speech_frame = index
                break
        return round(search_from_s + (last_speech_frame * hop) / sample_rate, 3)
    except Exception:
        return None


def resolve_processed_english_spans(
        words: list[dict], punct: list[dict], pauses: list[dict],
        duration_s: float, wav_path: str | Path | None = None) -> list[dict]:
    """Resolve safe English geometry without changing raw CTC evidence.

    ``canonical_span`` remains the raw CTC evidence.  The mutable processed
    span keeps the raw start and resolves its right edge against the next
    lexical token, punctuation, a long pause, or a validated VAD end for a
    true sentence-final token.
    """
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid audio duration for processed English spans")
    rms_profile = _load_local_rms_profile(wav_path)
    readable_audio_end = (min(duration, float(rms_profile["audio_end"]))
                          if rms_profile is not None else duration)

    def _number(value: object, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {label} in processed English span") from exc
        if not math.isfinite(result):
            raise ValueError(f"non-finite {label} in processed English span")
        return result

    def _raw_span(row: dict) -> tuple[float, float]:
        span = row.get("canonical_span")
        unit = row.get("canonical_unit")
        unit_span = unit.get("canonical_span") if isinstance(unit, dict) else None
        if (not isinstance(span, (list, tuple)) or len(span) != 2
                or not isinstance(unit, dict)
                or not isinstance(unit_span, (list, tuple)) or len(unit_span) != 2
                or span != list(unit_span)):
            raise ValueError("canonical English raw span missing or owner conflict")
        start = _number(span[0], "canonical start")
        end = _number(span[1], "canonical end")
        if start < 0 or end <= start or end > duration + 1e-6:
            raise ValueError("canonical English raw span outside audio axis")
        return start, end

    punctuation: list[tuple[float, float]] = []
    for item in punct:
        try:
            if isinstance(item, dict):
                start_value = item.get("start_s", item.get("start"))
                end_value = item.get("end_s", item.get("end", start_value))
            else:
                start_value, end_value = item[0], item[1]
            start = _number(start_value, "punctuation start")
            end = _number(end_value, "punctuation end")
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if 0 <= start < duration and end > start:
            punctuation.append((start, min(end, duration)))
    punctuation.sort()

    long_pauses: list[tuple[float, float]] = []
    for item in pauses:
        try:
            if isinstance(item, dict):
                start_value = item.get("start_s")
                end_value = item.get("end_s")
                if start_value is None:
                    start_value = float(item.get("start_ms", 0)) / 1000
                if end_value is None:
                    end_value = float(item.get("end_ms", 0)) / 1000
            else:
                start_value, end_value = item[0], item[1]
            start = _number(start_value, "pause start")
            end = _number(end_value, "pause end")
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if end - start >= 0.2 - 1e-9 and 0 <= start < duration and end > start:
            long_pauses.append((start, min(end, duration)))
    long_pauses.sort()

    lexical_indexes = [
        index for index, row in enumerate(words)
        if isinstance(row.get("word"), str) and row.get("word", "").strip()
    ]
    for index, row in enumerate(words):
        if not isinstance(row.get("canonical_unit"), dict):
            continue
        raw_start, raw_end = _raw_span(row)
        if raw_end > readable_audio_end + 1e-6:
            raise ValueError("canonical English raw span outside readable audio")
        next_start = None
        next_candidate = None
        for candidate_index in lexical_indexes:
            if candidate_index <= index:
                continue
            candidate = words[candidate_index]
            candidate_start = _number(
                candidate.get("start", candidate.get("start_s")),
                "next lexical start")
            next_start = candidate_start
            next_candidate = candidate
            break

        if (next_start is not None
                and raw_end > next_start + GEOMETRY_TOLERANCE_S):
            raise CanonicalNextOwnerConflict(
                canonical_span=[raw_start, raw_end],
                next_lexical_start=next_start,
                candidate_start=next_start,
                row={**row, "next_lexical_word":
                     next_candidate.get("word") if next_candidate else None},
            )

        search_end = min(next_start if next_start is not None else duration,
                         readable_audio_end)
        punctuation_starts = [start for start, _ in punctuation
                              if raw_end - 1e-6 <= start < search_end - 1e-6]
        punctuation_start = min(punctuation_starts) if punctuation_starts else None
        pause_candidates = [(start, end) for start, end in long_pauses
                            if raw_end - 1e-6 <= start < search_end - 1e-6]
        declared_pause = min(pause_candidates) if pause_candidates else None

        processed_end = None
        source = None
        # Punctuation is an unconditional owner boundary.  A declared pause is
        # considered only when it precedes punctuation; readable audio must
        # prove its onset or a later sustained-silence onset inside that pause.
        if (punctuation_start is not None
                and (declared_pause is None
                     or punctuation_start <= declared_pause[0] + 1e-9)):
            processed_end = punctuation_start
            source = "raw_end_punctuation"
        elif declared_pause is not None:
            pause_start, pause_end = declared_pause
            if rms_profile is None:
                # No usable acoustic evidence: retain the historical,
                # fail-closed declared-pause boundary.
                processed_end = pause_start
                source = "raw_end_long_pause"
            else:
                acoustic_limit = min(
                    pause_end, search_end, duration,
                    punctuation_start if punctuation_start is not None else duration,
                    float(rms_profile.get("audio_end", duration)))
                silence = _first_sustained_silence(
                    rms_profile, pause_start, acoustic_limit)
                if silence is not None:
                    silence_onset, starts_at_declared_onset = silence
                    if starts_at_declared_onset:
                        processed_end = pause_start
                        source = "raw_end_long_pause"
                    else:
                        processed_end = silence_onset
                        source = "energy_end_hard_boundary"
                # An acoustically false pause is not a boundary.  Leave the
                # candidate unset so punctuation/next lexical ownership below
                # remains authoritative.

        if processed_end is None and punctuation_start is not None:
            processed_end = punctuation_start
            source = "raw_end_punctuation"
        elif processed_end is None and next_start is not None:
            processed_end = min(next_start, duration, readable_audio_end)
            source = "next_lexical_token_start"
        elif processed_end is None:
            vad_end = (_profile_speech_end(rms_profile, raw_start)
                       if rms_profile is not None else None)
            if vad_end is None and wav_path is not None and rms_profile is None:
                vad_end = _vad_speech_end(str(wav_path), raw_start)
            if (vad_end is not None and math.isfinite(float(vad_end))
                    and raw_end < float(vad_end) <= duration + 1e-6):
                processed_end = min(float(vad_end), duration)
                source = "vad_speech_end"
            else:
                processed_end = raw_end
                source = "raw_end_fallback"

        assert source is not None
        processed_end = max(processed_end, raw_end)
        if processed_end <= raw_start:
            raise ValueError("processed English span has no positive owner interval")
        processed_span = [raw_start, min(processed_end, duration, readable_audio_end)]
        row["processed_ctc_span"] = processed_span
        row["processed_ctc_boundary_source"] = source
        row["start"] = processed_span[0]
        row["end"] = processed_span[1]
        if "start_s" in row or "end_s" in row:
            row["start_s"] = processed_span[0]
            row["end_s"] = processed_span[1]
            if "start_ms" in row:
                row["start_ms"] = round(processed_span[0] * 1000, 1)
            if "end_ms" in row:
                row["end_ms"] = round(processed_span[1] * 1000, 1)
    return words
