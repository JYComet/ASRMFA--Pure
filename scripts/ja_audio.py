#!/usr/bin/env python3
"""Conservative source audio inventory and integer sample-axis transforms."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any, Mapping

import numpy as np
try:
    import scipy
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - dependency absence is reported by callers
    scipy = None
    resample_poly = None


class AudioContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise AudioContractError(f"source audio missing or symlinked: {candidate}")
    try:
        with wave.open(str(candidate), "rb") as handle:
            compression = handle.getcomptype()
            info = {
                "path": str(candidate),
                "sha256": _sha256(candidate),
                "sample_rate": handle.getframerate(),
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "frames": handle.getnframes(),
                "duration_samples": handle.getnframes(),
                "duration_seconds": handle.getnframes() / handle.getframerate(),
                "compression": compression,
            }
    except (OSError, wave.Error) as exc:
        raise AudioContractError(f"invalid WAV: {candidate}: {exc}") from exc
    if info["compression"] != "NONE" or info["sample_rate"] <= 0 or info["channels"] <= 0:
        raise AudioContractError(f"unsupported WAV header: {candidate}")
    if info["sample_width"] not in {1, 2, 3, 4}:
        raise AudioContractError(f"unsupported PCM width: {info['sample_width']}")
    return info


def _decode_pcm(path: Path) -> tuple[np.ndarray, int]:
    info = inspect_wav(path)
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(info["frames"])
    width = info["sample_width"]
    if width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        scale = 128.0
    elif width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        scale = 32768.0
    elif width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (bytes_[:, 0].astype(np.int32) | (bytes_[:, 1].astype(np.int32) << 8) |
                  (bytes_[:, 2].astype(np.int32) << 16)).astype(np.int32)
        values[values & 0x800000 != 0] -= 1 << 24
        values = values.astype(np.float64)
        scale = float(1 << 23)
    else:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        scale = float(1 << 31)
    matrix = values.reshape(info["frames"], info["channels"])
    return np.clip(matrix / scale, -1.0, 1.0), info["sample_rate"]


def _resample_with_details(signal: np.ndarray, source_rate: int,
                           target_rate: int) -> tuple[np.ndarray, dict[str, int]]:
    if source_rate == target_rate:
        return signal, {"raw_resample_frames": int(signal.shape[0]),
                        "crop_after_frames": 0, "pad_after_frames": 0}
    if signal.size == 0:
        return np.zeros(0, dtype=np.float64), {
            "raw_resample_frames": 0, "crop_after_frames": 0, "pad_after_frames": 0,
        }
    # Positive integer half-up rounding makes the sample axis replayable
    # without depending on a language's floating-point ``round`` semantics.
    target_frames = int((signal.shape[0] * target_rate + source_rate // 2) // source_rate)
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float64), {
            "raw_resample_frames": 0, "crop_after_frames": 0, "pad_after_frames": 0,
        }
    if resample_poly is None:
        raise AudioContractError("scipy.signal.resample_poly is required for antialiased resampling")
    divisor = int(np.gcd(source_rate, target_rate))
    output = np.asarray(resample_poly(
        signal, target_rate // divisor, source_rate // divisor,
        window=("kaiser", 5.0), padtype="constant",
    ), dtype=np.float64)
    raw_frames = int(output.shape[0])
    # scipy's filter support can differ by one sample across versions; the
    # declared integer axis is authoritative and therefore gets explicit pad
    # or crop at the end.
    pad_after = max(0, target_frames - raw_frames)
    crop_after = max(0, raw_frames - target_frames)
    if pad_after:
        output = np.pad(output, (0, pad_after))
    return output[:target_frames], {
        "raw_resample_frames": raw_frames,
        "crop_after_frames": crop_after,
        "pad_after_frames": pad_after,
    }


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Compatibility wrapper retaining the old private helper's array API."""
    return _resample_with_details(signal, source_rate, target_rate)[0]


def _write_pcm16(path: Path, signal: np.ndarray, sample_rate: int) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal, -1.0, 1.0)
    # Round-to-nearest with an explicit integer scale is stable across numpy
    # versions; a source of exact zero remains exact zero.
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def transform_audio(source: str | Path, *, sample_rate: int = 16000,
                    trim_start_frames: int = 0, trim_end_frames: int = 0) -> dict[str, Any]:
    """Decode, downmix and resample without implicit silence deletion.

    Trimming is opt-in and only removes explicitly requested edge frames.  No
    energy threshold is consulted, preserving weak Japanese closures/vowels.
    """
    if type(sample_rate) is not int or sample_rate <= 0:
        raise AudioContractError("sample_rate must be a positive integer")
    if type(trim_start_frames) is not int or type(trim_end_frames) is not int or trim_start_frames < 0 or trim_end_frames < 0:
        raise AudioContractError("edge trims must be non-negative integers")
    source_path = Path(source).expanduser().absolute()
    source_info = inspect_wav(source_path)
    matrix, source_rate = _decode_pcm(source_path)
    mono = matrix.mean(axis=1)
    end = len(mono) - trim_end_frames if trim_end_frames else len(mono)
    if trim_start_frames + trim_end_frames > len(mono):
        raise AudioContractError("edge trims remove the complete source")
    cropped = mono[trim_start_frames:end]
    output, resample_details = _resample_with_details(cropped, source_rate, sample_rate)
    if source_rate == sample_rate:
        method = "pcm16_reencode_v1"
        resample_fields: dict[str, Any] = {
            "scipy_version": None,
            "up": 1,
            "down": 1,
            "window": None,
            "padtype": None,
        }
    else:
        divisor = int(np.gcd(source_rate, sample_rate))
        method = "scipy_resample_poly_v1"
        resample_fields = {
            "scipy_version": getattr(scipy, "__version__", None),
            "up": int(sample_rate // divisor),
            "down": int(source_rate // divisor),
            "window": {"name": "kaiser", "beta": 5.0},
            "padtype": "constant",
        }
    target_frames = int(output.shape[0])
    predicted_output_header = {
        "sample_rate": int(sample_rate), "channels": 1, "sample_width": 2,
        "frames": target_frames, "duration_samples": target_frames,
    }
    transform = {
        "source_start": int(trim_start_frames),
        "source_end": int(source_info["frames"] - trim_end_frames),
        "output_start": 0,
        "output_frames": target_frames,
        "source_rate": int(source_rate),
        "target_rate": int(sample_rate),
        "rate_ratio_num": int(sample_rate),
        "rate_ratio_den": int(source_rate),
        "frame_policy": "identity_v1" if source_rate == sample_rate else "round_half_up_v1",
        "method": method,
        "dtype": "float64",
        "downmix": "mean",
        "downmix_method": "mean_float64_v1",
        "source_header": dict(source_info),
        "output_header": predicted_output_header,
        "crop_start_frames": int(trim_start_frames),
        "crop_end_frames": int(trim_end_frames),
        "raw_resample_frames": int(resample_details["raw_resample_frames"]),
        "crop_after_frames": int(resample_details["crop_after_frames"]),
        "pad_before_frames": 0,
        "pad_after_frames": int(resample_details["pad_after_frames"]),
        "crop": {"start_frames": int(trim_start_frames), "end_frames": int(trim_end_frames),
                 "after_resample_frames": int(resample_details["crop_after_frames"])},
        "pad": {"before_frames": 0, "after_frames": int(resample_details["pad_after_frames"])},
        **resample_fields,
    }
    return {
        "source": source_info,
        "source_frames": int(source_info["frames"]),
        "output_frames": int(output.shape[0]),
        "sample_rate": sample_rate,
        "trimmed_frames": int(trim_start_frames + trim_end_frames),
        "signal": output,
        "sample_transform": transform,
    }


def prepare_alignment_wav(source: str | Path, target: str | Path, *, sample_rate: int = 16000,
                          trim_start_frames: int = 0, trim_end_frames: int = 0) -> dict[str, Any]:
    source_info = inspect_wav(source)
    if (source_info["sample_rate"] == sample_rate and source_info["channels"] == 1 and
            source_info["sample_width"] == 2 and trim_start_frames == 0 and trim_end_frames == 0):
        target_path = Path(target).expanduser().absolute()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(source), "rb") as inp:
            raw = inp.readframes(inp.getnframes())
        with wave.open(str(target_path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(sample_rate); out.writeframes(raw)
        output_info = inspect_wav(target_path)
        transform = {"source_start": 0, "source_end": source_info["frames"], "output_start": 0,
                     "output_frames": source_info["frames"], "source_rate": sample_rate,
                     "target_rate": sample_rate, "rate_ratio_num": 1, "rate_ratio_den": 1,
                     "frame_policy": "identity_v1",
                     "method": "pcm16_copy_v1", "dtype": "int16",
                     "downmix": "identity", "downmix_method": "identity_pcm16_v1",
                     "source_header": dict(source_info),
                     "output_header": output_info, "crop_start_frames": 0,
                     "crop_end_frames": 0, "raw_resample_frames": source_info["frames"],
                     "crop_after_frames": 0, "pad_before_frames": 0, "pad_after_frames": 0,
                     "crop": {"start_frames": 0, "end_frames": 0, "after_resample_frames": 0},
                     "pad": {"before_frames": 0, "after_frames": 0},
                     "scipy_version": None, "up": 1, "down": 1,
                     "window": None, "padtype": None,
                     "output_sha256": output_info["sha256"]}
        return {"source": source_info, "source_frames": source_info["frames"], "output_frames": source_info["frames"],
                "sample_rate": sample_rate, "trimmed_frames": 0, "output": output_info,
                "sample_transform": transform}
    transformed = transform_audio(source, sample_rate=sample_rate,
                                   trim_start_frames=trim_start_frames,
                                   trim_end_frames=trim_end_frames)
    target_path = Path(target).expanduser().absolute()
    _write_pcm16(target_path, transformed.pop("signal"), sample_rate)
    output_info = inspect_wav(target_path)
    transformed["output"] = output_info
    transformed["sample_transform"]["output_header"] = output_info
    transformed["sample_transform"]["output_sha256"] = output_info["sha256"]
    return transformed


def prepare_training_wav(source: str | Path, target: str | Path) -> dict[str, Any]:
    source_info = inspect_wav(source)
    if source_info["channels"] == 1 and source_info["sample_width"] == 2:
        target_path = Path(target).expanduser().absolute()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(source), "rb") as inp:
            raw = inp.readframes(inp.getnframes())
        with wave.open(str(target_path), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(source_info["sample_rate"]); out.writeframes(raw)
        output = inspect_wav(target_path)
        transform = {"source_start": 0, "source_end": source_info["frames"], "output_start": 0,
                     "output_frames": source_info["frames"], "source_rate": source_info["sample_rate"],
                     "target_rate": source_info["sample_rate"], "rate_ratio_num": 1, "rate_ratio_den": 1,
                     "frame_policy": "identity_v1",
                     "method": "pcm16_copy_v1", "dtype": "int16",
                     "downmix": "identity", "downmix_method": "identity_pcm16_v1",
                     "source_header": dict(source_info),
                     "output_header": output, "crop_start_frames": 0,
                     "crop_end_frames": 0, "raw_resample_frames": source_info["frames"],
                     "crop_after_frames": 0, "pad_before_frames": 0, "pad_after_frames": 0,
                     "crop": {"start_frames": 0, "end_frames": 0, "after_resample_frames": 0},
                     "pad": {"before_frames": 0, "after_frames": 0},
                     "scipy_version": None, "up": 1, "down": 1,
                     "window": None, "padtype": None,
                     "output_sha256": output["sha256"]}
        return {"source": source_info, "source_frames": source_info["frames"], "output_frames": source_info["frames"],
                "sample_rate": source_info["sample_rate"], "trimmed_frames": 0, "output": output,
                "sample_transform": transform}
    transformed = transform_audio(source, sample_rate=source_info["sample_rate"])
    signal = transformed.pop("signal")
    target_path = Path(target).expanduser().absolute()
    _write_pcm16(target_path, signal, source_info["sample_rate"])
    transformed["output"] = inspect_wav(target_path)
    transformed["sample_transform"]["output_header"] = transformed["output"]
    transformed["sample_transform"]["output_sha256"] = transformed["output"]["sha256"]
    return transformed


def make_audio_receipt(uid: str, source: str | Path, train: str | Path, alignment: str | Path,
                       *, alignment_transform: Mapping[str, Any],
                       train_transform: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source_info, train_info, alignment_info = map(inspect_wav, (source, train, alignment))
    alignment_transform_value = dict(alignment_transform)
    train_transform_value = dict(train_transform or alignment_transform)
    return {
        "schema": "audio-transform-receipt-v2", "uid": uid,
        "source": source_info, "train": train_info, "alignment": alignment_info,
        "sample_transform": alignment_transform_value,
        "alignment_transform": dict(alignment_transform_value),
        "train_transform": train_transform_value,
    }


__all__ = ["AudioContractError", "inspect_wav", "transform_audio", "prepare_alignment_wav",
           "prepare_training_wav", "make_audio_receipt"]


def _manifest_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = Path(str(config["input_manifest"])).expanduser().absolute()
    if manifest.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload.get("items", payload) if isinstance(payload, Mapping) else payload


def _safe_stem(uid: str, ordinal: int) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uid).strip("._")
    suffix = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:10]
    return f"{(cleaned[:68] or f'item_{ordinal:06d}')}__{suffix}"


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def inventory_stage(config: Mapping[str, Any], stage_dir: Path):
    """Stage 0 source inventory handler; source files remain external inputs."""
    try:
        from .ja_en_schema import StageResult, atomic_write_json, make_receipt
    except ImportError:
        from ja_en_schema import StageResult, atomic_write_json, make_receipt
    rows = _manifest_rows(config)
    inventory: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        uid = str(row.get("uid", row.get("id", f"item_{ordinal:06d}")))
        source_value = row.get("source_wav", row.get("wav", row.get("audio")))
        item: dict[str, Any] = {"uid": uid, "ordinal": ordinal, "source_wav": source_value,
                                "orig_text": row.get("orig_text", row.get("text")), "status": "verified"}
        try:
            if not isinstance(source_value, str):
                raise AudioContractError("manifest row has no source_wav/wav/audio")
            item["audio"] = inspect_wav(source_value)
        except AudioContractError as exc:
            item["status"] = "rejected"
            item["error"] = {"code": "source_audio_invalid", "message": str(exc)}
        text = item.get("orig_text")
        if isinstance(text, str):
            item["orig_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        inventory.append(item)
    output = stage_dir / "source_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, {"schema": "ja-source-inventory-v1", "items": inventory}, workspace=stage_dir.parent.parent)
    status = "COMPLETE" if all(row["status"] == "verified" for row in inventory) else "PARTIAL"
    receipt = make_receipt(stage="inventory", status=status, inputs={"manifest": str(config["input_manifest"])}, outputs=[output])
    receipt_path = stage_dir / "receipt.json"
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="inventory", status=status, receipt_path=str(receipt_path))


def audio_stage(config: Mapping[str, Any], stage_dir: Path):
    """Stage 1 handler producing ``train__*.wav`` and ``alignment__*.wav``."""
    try:
        from .ja_en_schema import StageResult, atomic_write_json, make_receipt
    except ImportError:
        from ja_en_schema import StageResult, atomic_write_json, make_receipt
    rows = _manifest_rows(config)
    stage_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    statuses: list[str] = []
    for ordinal, row in enumerate(rows):
        uid = str(row.get("uid", row.get("id", f"item_{ordinal:06d}")))
        stem = _safe_stem(uid, ordinal)
        source = row.get("source_wav", row.get("wav", row.get("audio")))
        try:
            if not isinstance(source, str):
                raise AudioContractError("manifest row has no source WAV")
            train = stage_dir / f"train__{stem}.wav"
            alignment = stage_dir / f"alignment__{stem}.wav"
            train_info = prepare_training_wav(source, train)
            align_info = prepare_alignment_wav(source, alignment)
            receipt = make_audio_receipt(
                uid, source, train, alignment,
                train_transform=train_info.get("sample_transform", {}),
                alignment_transform=align_info["sample_transform"],
            )
            receipt["status"] = "verified"
            statuses.append("verified")
        except (AudioContractError, OSError, ValueError) as exc:
            receipt = {"schema": "audio-transform-receipt-v2", "uid": uid, "status": "rejected",
                       "source": {"path": source}, "train": {}, "alignment": {},
                       "sample_transform": {}, "alignment_transform": {}, "train_transform": {},
                       "error": {"code": "source_audio_invalid", "message": str(exc)}}
            statuses.append("rejected")
        receipts.append(receipt)
    receipt_file = stage_dir / "audio_transform_receipts.jsonl"
    _write_jsonl(receipt_file, receipts)
    status = "COMPLETE" if statuses and all(value == "verified" for value in statuses) else "PARTIAL"
    receipt = make_receipt(stage="audio", status=status, inputs={"manifest": str(config["input_manifest"])},
                           outputs=[p for p in stage_dir.iterdir() if p.is_file() and p.name != "receipt.json"])
    receipt_path = stage_dir / "receipt.json"
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="audio", status=status, receipt_path=str(receipt_path))


def register_stages(registrar: Any) -> None:
    """Register only this module's inventory/audio namespaces with the core."""
    registrar("inventory", inventory_stage, output_namespace="inventory")
    registrar("audio", audio_stage, output_namespace="audio")
