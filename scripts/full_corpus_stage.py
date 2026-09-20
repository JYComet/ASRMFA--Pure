"""Private flat staging and verified GAMEDATA/Wuthering Waves audio publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf

try:
    from scripts.full_corpus_inventory import InventoryItem
    from scripts.qwen3_timestamp_normalization import normalize_qwen_reference_text
except ModuleNotFoundError:  # direct script execution from repository root
    from full_corpus_inventory import InventoryItem
    from qwen3_timestamp_normalization import normalize_qwen_reference_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    except OSError:
        pass


def _safe_target(root: Path, *parts: str) -> Path:
    root = Path(root).resolve()
    candidate = root.joinpath(*parts)
    cursor = root
    for part in parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlink path component is not allowed: {cursor}")
    target = candidate.resolve(strict=False)
    if root not in target.parents:
        raise ValueError(f"target escapes root: {target}")
    return target


def _normalize_audio(source: Path, target: Path) -> dict:
    # Keep the staging transform local so the inventory/stager can run as a
    # standalone script (the legacy finalizer imports its module peers by
    # top-level name).  The detector follows the established RMS contract.
    audio, rate = sf.read(str(source), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    if rate <= 0 or not np.isfinite(audio).all() or not np.any(np.abs(audio) > 1e-7):
        raise ValueError("source audio has invalid rate, non-finite samples, or no speech")
    window = max(1, round(.01 * rate))
    def edge_silence(values):
        if len(values) < window:
            return 0
        cumulative = np.concatenate(([0.0], np.cumsum(values.astype(np.float64) ** 2)))
        means = (cumulative[window:] - cumulative[:-window]) / window
        active = np.flatnonzero(means >= .001 ** 2)
        return int(active[0] + window) if len(active) else len(values)
    head = edge_silence(audio)
    target_samples = round(.5 * rate)
    if head > target_samples:
        audio = audio[head - target_samples:]
    elif head < target_samples:
        audio = np.concatenate((np.zeros(target_samples - head, dtype=np.float32), audio))
    tail = edge_silence(audio[::-1])
    if tail > target_samples:
        audio = audio[:len(audio) - (tail - target_samples)]
    elif tail < target_samples:
        audio = np.concatenate((audio, np.zeros(target_samples - tail, dtype=np.float32)))
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, rate, subtype="PCM_16", format="WAV")
    return {"source": str(source), "target": str(target),
            "sample_rate": int(rate), "duration": len(audio) / rate}


def verify_padded_wav(path: Path, target_seconds: float = .5) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing staged WAV: {path}")
    info = sf.info(str(path))
    if info.channels != 1 or info.subtype != "PCM_16":
        raise ValueError("staged WAV must be mono PCM16")
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not np.any(np.abs(audio) > 1e-7):
        raise ValueError("staged WAV is empty, non-finite, or silent")
    edge = max(1024 / rate, .03)
    n = max(1, round(target_seconds * rate))
    def detected(values):
        window = max(1, round(.01 * rate))
        if len(values) < window:
            return 0.0
        cumulative = np.concatenate(([0.0], np.cumsum(values.astype(np.float64) ** 2)))
        means = (cumulative[window:] - cumulative[:-window]) / window
        active = np.flatnonzero(means >= .001 ** 2)
        return (float(active[0] + window) / rate) if len(active) else len(values) / rate
    head_detected, tail_detected = detected(audio), detected(audio[::-1])
    quiet_n = max(1, n - max(1, round(.01 * rate)))
    head = float(np.sqrt(np.mean(np.square(audio[:quiet_n])))) if len(audio) >= quiet_n else 1.0
    tail = float(np.sqrt(np.mean(np.square(audio[-quiet_n:])))) if len(audio) >= quiet_n else 1.0
    duration = len(audio) / rate
    return {
        "sample_rate": int(rate), "channels": int(info.channels),
        "subtype": info.subtype, "duration": duration,
        "head_rms": head, "tail_rms": tail,
        "head_detected_silence": head_detected, "tail_detected_silence": tail_detected,
        "head_ok": abs(head_detected - target_seconds) <= edge and head <= .001,
        "tail_ok": abs(tail_detected - target_seconds) <= edge and tail <= .001,
        "sha256": _sha256(path),
    }


@dataclass(frozen=True)
class StageReceipt:
    run_stem: str
    source_id: str
    game: str | None
    speaker: str
    pipeline_wav: Path
    gamesl_wav: Path | None
    pipeline_wav_sha256: str
    gamesl_wav_sha256: str | None
    text_path: Path | None
    text_sha256: str | None
    normalized_text: str | None
    source_sha256: str | None = None
    # Bound to the resolution table that produced ``normalized_text``.  A stage
    # receipt carrying the wrong digest is stale and must be re-normalized from
    # source, which is how the macro repair invalidates frozen artifacts.
    macro_resolution_digest: str | None = None


def stage_item(item: InventoryItem, chunk_input: Path,
               gamesl_stage: Path | None = None, *,
               resolutions: Mapping[str, str] | None = None,
               resolution_digest: str | None = None) -> StageReceipt:
    chunk_input = Path(chunk_input)
    chunk_input.mkdir(parents=True, exist_ok=True)
    pipeline = chunk_input / f"{item.run_stem}.wav"
    receipt_path = chunk_input / f"{item.run_stem}.stage_receipt.json"
    source_hash = _sha256(item.source_path)
    if source_hash != item.audio_sha256:
        raise ValueError(f"frozen source hash drift: {item.source_path}")
    if item.reference_path is not None:
        reference_hash = _sha256(item.reference_path)
        if reference_hash != item.reference_sha256:
            raise ValueError(f"frozen reference hash drift: {item.reference_path}")
        expected_normalized_text = normalize_qwen_reference_text(
            item.reference_path.read_text(encoding="utf-8"), speaker=item.speaker,
            resolutions=resolutions)
    else:
        if resolutions is not None:
            raise ValueError("resolution table supplied for a fallback item")
        expected_normalized_text = None
    if pipeline.exists() and receipt_path.is_file():
        try:
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            gamesl_saved = Path(saved["gamesl_wav"]) if saved.get("gamesl_wav") else None
            text_saved = Path(saved["text_path"]) if saved.get("text_path") else None
            if saved.get("source_sha256") != source_hash or saved.get("pipeline_wav_sha256") != _sha256(pipeline):
                raise ValueError("stage receipt/source digest mismatch")
            if gamesl_saved is not None and (not gamesl_saved.is_file() or saved.get("gamesl_wav_sha256") != _sha256(gamesl_saved)):
                raise ValueError("GAMESL stage receipt digest mismatch")
            if item.reference_path is not None:
                if text_saved is None:
                    text_saved = chunk_input / f"{item.run_stem}.txt"
                if (not text_saved.is_file()
                        or saved.get("text_sha256") != _sha256(text_saved)
                        or text_saved.read_text(encoding="utf-8") != expected_normalized_text
                        or saved.get("normalized_text") != expected_normalized_text
                        or saved.get("macro_resolution_digest") != resolution_digest):
                    temporary_text = text_saved.with_name(text_saved.name + ".tmp")
                    temporary_text.write_text(expected_normalized_text, encoding="utf-8")
                    with temporary_text.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary_text, text_saved)
                    _fsync_parent(text_saved)
                    saved["text_path"] = str(text_saved)
                    saved["text_sha256"] = _sha256(text_saved)
                    saved["normalized_text"] = expected_normalized_text
                    saved["macro_resolution_digest"] = resolution_digest
                    temporary_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
                    temporary_receipt.write_text(
                        json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
                    with temporary_receipt.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary_receipt, receipt_path)
                    _fsync_parent(receipt_path)
            elif text_saved is not None or saved.get("text_sha256") or saved.get("normalized_text"):
                raise ValueError("fallback stage receipt unexpectedly contains reference text")
            return StageReceipt(item.run_stem, item.source_id, item.game, item.speaker,
                                pipeline, gamesl_saved, saved["pipeline_wav_sha256"],
                                saved.get("gamesl_wav_sha256"),
                                text_saved,
                                saved.get("text_sha256"), saved.get("normalized_text"),
                                source_hash, saved.get("macro_resolution_digest"))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid stage resume receipt: {receipt_path}") from exc
    if pipeline.exists() or pipeline.is_symlink():
        raise ValueError(f"stale pipeline staging target exists: {pipeline}")
    gamesl = None
    if item.needs_gamesl_padding:
        if gamesl_stage is None:
            raise ValueError("gamesl_stage is required for padded sources")
        gamesl = _safe_target(gamesl_stage, item.game or "_default", item.speaker,
                              f"{item.run_stem}.wav")
        if gamesl.exists() or gamesl.is_symlink():
            raise ValueError(f"stale GAMESL staging target exists: {gamesl}")
        gamesl.parent.mkdir(parents=True, exist_ok=True)
        temporary = gamesl.with_name(gamesl.name + ".tmp")
        _normalize_audio(item.source_path, temporary)
        edge_receipt = verify_padded_wav(temporary)
        if not edge_receipt["head_ok"] or not edge_receipt["tail_ok"]:
            raise ValueError(
                f"padded edge verification failed for {item.run_stem}: "
                f"head_ok={edge_receipt['head_ok']} tail_ok={edge_receipt['tail_ok']}"
            )
        os.replace(temporary, gamesl)
        _fsync_parent(gamesl)
        temporary_pipeline = pipeline.with_name(pipeline.name + ".tmp")
        shutil.copyfile(gamesl, temporary_pipeline)
        os.replace(temporary_pipeline, pipeline)
        _fsync_parent(pipeline)
    else:
        temporary_pipeline = pipeline.with_name(pipeline.name + ".tmp")
        source_info = sf.info(str(item.source_path))
        if source_info.format == "WAV" and source_info.channels == 1:
            # Keep the original sample representation (including LAria FLOAT)
            # when it already satisfies the pipeline-axis contract.
            shutil.copyfile(item.source_path, temporary_pipeline)
            with temporary_pipeline.open("rb") as handle:
                os.fsync(handle.fileno())
        else:
            audio, rate = sf.read(str(item.source_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            audio = np.asarray(audio, dtype=np.float32)
            if rate <= 0 or not np.isfinite(audio).all() or not np.any(np.abs(audio) > 1e-7):
                raise ValueError("v5 source audio is invalid, non-finite, or silent")
            sf.write(str(temporary_pipeline), audio, rate, subtype="FLOAT", format="WAV")
        os.replace(temporary_pipeline, pipeline)
        _fsync_parent(pipeline)
    pipeline_hash = _sha256(pipeline)
    gamesl_hash = _sha256(gamesl) if gamesl else None
    if gamesl_hash is not None and gamesl_hash != pipeline_hash:
        raise ValueError("pipeline and GAMESL staged WAV hashes differ")

    text_path = None
    text_hash = None
    normalized_text = None
    if item.reference_path is not None:
        normalized_text = expected_normalized_text
        text_path = chunk_input / f"{item.run_stem}.txt"
        if not normalized_text:
            raise ValueError("normalized authority text is empty")
        temporary_text = text_path.with_name(text_path.name + ".tmp")
        temporary_text.write_text(normalized_text, encoding="utf-8")
        with temporary_text.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_text, text_path)
        _fsync_parent(text_path)
        text_hash = _sha256(text_path)
    receipt = StageReceipt(item.run_stem, item.source_id, item.game, item.speaker,
                        pipeline, gamesl, pipeline_hash, gamesl_hash,
                        text_path, text_hash, normalized_text)
    receipt = StageReceipt(
        receipt.run_stem, receipt.source_id, receipt.game, receipt.speaker,
        receipt.pipeline_wav, receipt.gamesl_wav, receipt.pipeline_wav_sha256,
        receipt.gamesl_wav_sha256, receipt.text_path, receipt.text_sha256,
        receipt.normalized_text, source_hash, resolution_digest)
    payload = {"run_stem": receipt.run_stem, "source_sha256": source_hash,
               "pipeline_wav": str(receipt.pipeline_wav), "pipeline_wav_sha256": receipt.pipeline_wav_sha256,
               "gamesl_wav": str(receipt.gamesl_wav) if receipt.gamesl_wav else None,
               "gamesl_wav_sha256": receipt.gamesl_wav_sha256,
               "text_path": str(receipt.text_path) if receipt.text_path else None,
               "text_sha256": receipt.text_sha256, "normalized_text": receipt.normalized_text,
               "macro_resolution_digest": resolution_digest}
    temporary_receipt = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary_receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with temporary_receipt.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary_receipt, receipt_path)
    _fsync_parent(receipt_path)
    return receipt


def publish_gamesl(receipts, gamesl_root: Path, rollback_root: Path) -> dict:
    gamesl_root, rollback_root = Path(gamesl_root), Path(rollback_root)
    replacements = []
    for receipt in receipts:
        if receipt.gamesl_wav is None:
            continue
        target = _safe_target(gamesl_root, receipt.game or "_default",
                              receipt.speaker, receipt.gamesl_wav.name)
        source = Path(receipt.gamesl_wav)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"unsafe staged GAMESL source: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        old_hash = None
        rollback = None
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"unsafe GAMESL target: {target}")
            old_hash = _sha256(target)
            rollback = _safe_target(rollback_root, receipt.game or "_default",
                                    receipt.speaker, target.name)
            if rollback.exists() or rollback.is_symlink():
                raise ValueError(f"rollback artifact already exists: {rollback}")
            rollback.parent.mkdir(parents=True, exist_ok=True)
            temporary_rollback = rollback.with_name(rollback.name + ".tmp")
            shutil.copyfile(target, temporary_rollback)
            with temporary_rollback.open("rb") as handle:
                os.fsync(handle.fileno())
            if _sha256(temporary_rollback) != old_hash:
                temporary_rollback.unlink(missing_ok=True)
                raise ValueError("rollback copy digest mismatch")
            os.replace(temporary_rollback, rollback)
        temporary = target.with_name(target.name + ".tmp")
        try:
            shutil.copyfile(source, temporary)
            if _sha256(temporary) != receipt.gamesl_wav_sha256:
                raise ValueError("staged GAMESL digest changed")
            os.replace(temporary, target)
            _fsync_parent(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            if rollback is not None and rollback.exists():
                # The target may already have been replaced before a later
                # durability/hash operation failed.  Restore unconditionally
                # so callers never observe a half-installed new target.
                target.unlink(missing_ok=True)
                shutil.copyfile(rollback, temporary)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                _fsync_parent(target)
            raise
        replacements.append({"stem": receipt.run_stem, "kind": "gamesl", "target": str(target), "rollback": str(rollback) if rollback else None,
                             "old_sha256": old_hash, "new_sha256": _sha256(target)})
    return {"replaced_count": sum(row["old_sha256"] is not None for row in replacements),
            "published_count": len(replacements), "replacements": replacements,
            "rollback_root": str(rollback_root)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--private-check", default="")
    parser.parse_args()
    print(json.dumps({"private_check": True, "public_writes": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
