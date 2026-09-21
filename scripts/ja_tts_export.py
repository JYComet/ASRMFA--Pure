"""Deterministic TTS artifacts for the Japanese/English pipeline.

This module consumes already verified alignment dictionaries.  It does not
run MFA, infer readings, or measure F0.  The sample axis in the alignment is
the authority; decimal TextGrid times are a serialization of that axis.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .ja_en_schema import (JAContractError, StageResult, atomic_write_bytes,
                               atomic_write_json, canonical_json, make_receipt,
                               sha256_file, validate_record)
except ImportError:  # pragma: no cover
    from ja_en_schema import (JAContractError, StageResult, atomic_write_bytes,
                              atomic_write_json, canonical_json, make_receipt,
                              sha256_file, validate_record)


def _int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer sample")
    return value


def _span(row: Mapping[str, Any], sample_rate: int) -> tuple[int, int]:
    if "start_sample" in row or "end_sample" in row:
        start, end = _int(row.get("start_sample"), "start_sample"), _int(row.get("end_sample"), "end_sample")
    else:
        try:
            start = int(round(float(row["start"]) * sample_rate))
            end = int(round(float(row["end"]) * sample_rate))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("alignment row lacks sample or second span") from exc
    if start < 0 or end <= start:
        raise ValueError(f"invalid sample span {start}:{end}")
    return start, end


def _file_info(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"audio path is not a regular file: {candidate}")
    with wave.open(str(candidate), "rb") as handle:
        info = {"path": str(candidate), "sample_rate": handle.getframerate(),
                "channels": handle.getnchannels(), "sample_width": handle.getsampwidth(),
                "frames": handle.getnframes()}
    info["sha256"] = sha256_file(candidate)
    return info


def build_quality_masks(prosody: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build explicit, independent masks for text accent and measured F0."""
    prosody = prosody or {}
    accent = prosody.get("accent_predicted")
    f0 = prosody.get("f0_measured")
    accent_known = accent is not None
    f0_known = False
    accent_provenance = accent.get("provenance") if isinstance(accent, Mapping) else None
    f0_provenance = f0.get("provenance") if isinstance(f0, Mapping) else None
    if accent_known and not accent_provenance:
        raise ValueError("accent_predicted requires text provenance")
    if f0 is not None:
        if not isinstance(f0, Mapping) or not f0_provenance:
            raise ValueError("f0_measured requires measurement provenance")
        source_kind = f0.get("source_kind") or (f0_provenance.get("source_kind") if isinstance(f0_provenance, Mapping) else None)
        if source_kind not in {"audio_measurement", "acoustic_measurement", "measurement"}:
            raise ValueError("f0_measured provenance must identify an audio measurement")
        values = f0.get("values", f0.get("hz"))
        if isinstance(values, (list, tuple)):
            for value in values:
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                    raise ValueError("f0_measured values must be finite or null")
            f0_known = any(value is not None for value in values)
        elif values is not None:
            if not isinstance(values, (int, float)) or not math.isfinite(float(values)):
                raise ValueError("f0_measured value must be finite or null")
            f0_known = True
    return {
        "accent_predicted_known_mask": bool(accent_known),
        "accent_known_mask": bool(accent_known),
        "f0_known_mask": bool(f0_known),
        "accent_predicted_provenance": accent_provenance,
        "f0_measured_provenance": f0_provenance,
        "unvoiced_mask": list(prosody.get("unvoiced_mask", [])),
        "estimated_mora_timing_mask": list(prosody.get("estimated_mora_timing_mask", [])),
    }


def _normalise_words(alignment: Mapping[str, Any], sample_rate: int) -> list[dict[str, Any]]:
    words = []
    for index, source in enumerate(alignment.get("words", [])):
        row = dict(source)
        start, end = _span(row, sample_rate)
        language = row.get("language")
        if language not in {"ja", "en"}:
            raise ValueError(f"word {index} has unknown language")
        words.append({**row, "unit_id": str(row.get("unit_id", f"unit_{index:06d}")),
                      "text": str(row.get("text", "")), "language": language,
                      "start_sample": start, "end_sample": end})
    return words


def _normalise_phones(alignment: Mapping[str, Any], sample_rate: int, words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = -1
    word_ids = {str(row["unit_id"]) for row in words}
    for index, source in enumerate(alignment.get("phones", [])):
        row = dict(source)
        start, end = _span(row, sample_rate)
        if start < previous:
            raise ValueError("phones are not monotonic on the integer sample axis")
        language = row.get("language")
        if language not in {"ja", "en"}:
            raise ValueError(f"phone {index} has unknown language")
        unit_id = str(row.get("unit_id", ""))
        if unit_id and unit_id not in word_ids:
            raise ValueError(f"phone {index} references unknown unit")
        native = row.get("native_phone", row.get("phone"))
        if not isinstance(native, str) or not native:
            raise ValueError(f"phone {index} has no native label")
        phone_id = str(row.get("phone_id", f"phone_{index:06d}"))
        result.append({**row, "phone_id": phone_id, "unit_id": unit_id,
                       "native_phone": native, "phone": str(row.get("phone", native)),
                       "language": language, "start_sample": start, "end_sample": end,
                       "duration_samples": end - start,
                       "duration_seconds": (end - start) / sample_rate,
                       "mora_ids": list(row.get("mora_ids", []))})
        previous = end
    if not result:
        raise ValueError("alignment contains no phones")
    return result


def build_training_record(
    alignment: Mapping[str, Any], *, train_wav: str | Path, alignment_wav: str | Path,
    speaker: str | None = None, text_layers: Mapping[str, Any] | None = None,
    audio_receipt: Mapping[str, Any] | None = None, sample_rate: int | None = None,
    prosody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an alignment-v2 object into one training-record-v1 object."""
    uid = alignment.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("alignment uid is required")
    train_info, align_info = _file_info(train_wav), _file_info(alignment_wav)
    rate = sample_rate or int(align_info["sample_rate"])
    if rate <= 0 or align_info["sample_rate"] != rate:
        raise ValueError("alignment sample rate does not match requested axis")
    words = _normalise_words(alignment, rate)
    if not words:
        raise ValueError("alignment words are required; ownership cannot be inferred")
    phones = _normalise_phones(alignment, rate, words)
    train_rate = int(train_info["sample_rate"])
    for phone in phones:
        # Keep both axes explicit.  The alignment axis remains authoritative;
        # train coordinates are the deterministic half-up projection used by
        # the audio receipt and are never inferred from wall-clock floats.
        phone["train_start_sample"] = int((phone["start_sample"] * train_rate + rate // 2) // rate)
        phone["train_end_sample"] = int((phone["end_sample"] * train_rate + rate // 2) // rate)
    graph = dict(alignment.get("mora_graph") or {"moras": [], "relations": []})
    known_phones = {row["phone_id"] for row in phones}
    known_moras = {str(row.get("mora_id")) for row in graph.get("moras", [])}
    for relation in graph.get("relations", []):
        if relation.get("phone_id") not in known_phones or relation.get("mora_id") not in known_moras:
            raise ValueError("mora graph relation references an unknown node")
    prosody_payload = prosody or alignment.get("prosody") or {}
    record = {
        "schema": "tts-training-record-v1", "uid": uid,
        "train_wav": {**train_info, "path": str(Path(train_wav).expanduser().absolute())},
        "alignment_wav": {**align_info, "path": str(Path(alignment_wav).expanduser().absolute())},
        "sample_rate": rate, "train_sample_rate": train_rate, "speaker": speaker if speaker is not None else alignment.get("speaker"),
        "words": words,
        "text_layers": dict(text_layers or alignment.get("text_layers") or {}),
        "selected_reading": alignment.get("selected_reading"),
        "selected_readings": dict(alignment.get("selected_readings") or {}),
        "phones": phones, "durations": [row["duration_samples"] for row in phones],
        "mora_graph": graph, "accent_predicted": prosody_payload.get("accent_predicted"),
        "f0_measured": prosody_payload.get("f0_measured"),
        "quality_masks": build_quality_masks(prosody_payload),
        "source_alignment_schema": alignment.get("schema"),
        "provenance": {"raw_interval_ids": [row.get("raw_interval_id") for row in phones],
                       "aliases": sorted({row["alias"] for row in phones if row.get("alias")})},
    }
    if audio_receipt is None:
        raise ValueError("authoritative audio-transform receipt is required")
    if not isinstance(audio_receipt, Mapping) or audio_receipt.get("schema") != "audio-transform-receipt-v2" or not all(key in audio_receipt for key in ("source", "train", "alignment", "sample_transform")):
        raise ValueError("audio-transform receipt schema/source/train/alignment fields are required")
    # Bind the receipt to the files consumed by this record.  A producer cannot
    # claim a valid transform while quietly swapping the training/alignment
    # WAVs after the upstream audio stage completed.
    for receipt_key, actual_info in (("train", train_info), ("alignment", align_info)):
        declared = audio_receipt.get(receipt_key)
        if not isinstance(declared, Mapping):
            raise ValueError(f"audio-transform receipt {receipt_key} metadata is required")
        for field in ("path", "sha256", "frames", "sample_rate", "channels", "sample_width"):
            if declared.get(field) != actual_info.get(field):
                raise ValueError(f"audio-transform receipt {receipt_key}.{field} differs from file")
    source = audio_receipt.get("source")
    transform = audio_receipt.get("sample_transform")
    if not isinstance(source, Mapping) or not isinstance(transform, Mapping):
        raise ValueError("audio-transform source and sample_transform metadata are required")
    source_path = source.get("path")
    if not source_path or not Path(str(source_path)).expanduser().is_file():
        raise ValueError("audio-transform source path is missing")
    source_info = _file_info(str(source_path))
    for field in ("path", "sha256", "frames", "sample_rate", "channels", "sample_width"):
        if source.get(field) != source_info.get(field):
            raise ValueError(f"audio-transform source.{field} differs from file")
    source_rate = int(source_info["sample_rate"])
    sample_transform = audio_receipt.get("sample_transform")
    source_offset = int(sample_transform.get("source_start", 0))
    for phone in phones:
        phone["source_start_sample"] = source_offset + int((phone["start_sample"] * source_rate + rate // 2) // rate)
        phone["source_end_sample"] = source_offset + int((phone["end_sample"] * source_rate + rate // 2) // rate)
    record["source_sample_rate"] = source_rate
    record["audio_transform"] = dict(audio_receipt)
    for field in ("locked_aliases", "native_inventory", "raw_mfa", "reading_evidence", "partition"):
        if field not in alignment:
            raise ValueError(f"authoritative {field} is required")
        record[field] = alignment[field]
    # Preserve the full unit ledger alongside the three outcome buckets.  The
    # verifier uses this authoritative join key; a UID is an utterance and is
    # never itself a unit partition.
    if isinstance(record.get("partition"), Mapping):
        partition = dict(record["partition"])
        if not isinstance(partition.get("expected_unit_ids"), list):
            partition["expected_unit_ids"] = [str(word["unit_id"]) for word in words]
        record["partition"] = partition
    for field in ("semantic_graph_path", "semantic_graph_sha256", "semantic_alias_map_path", "semantic_alias_map_sha256", "locked_dict_path", "locked_dict_sha256", "native_inventory_path", "native_inventory_sha256"):
        if field in alignment:
            record[field] = alignment[field]
    reading_evidence = record["reading_evidence"]
    selected_map = record.get("selected_readings") if isinstance(record.get("selected_readings"), Mapping) else {}
    if not isinstance(reading_evidence, Mapping):
        raise ValueError("authoritative locked reading evidence is required")
    if selected_map:
        if not isinstance(reading_evidence.get("locks"), list) or not reading_evidence.get("locks"):
            raise ValueError("multi-token records require reading_evidence.locks")
    elif not record.get("selected_reading") or reading_evidence.get("selected_reading") != record.get("selected_reading"):
        raise ValueError("authoritative locked reading evidence is required")
    validate_record(record, "tts-training-record-v1")
    return record


def _quote(value: Any) -> str:
    return str(value).replace('"', '""')


def _tier(name: str, intervals: Sequence[tuple[int, int, str]], sample_rate: int) -> str:
    xmax = max((end for _, end, _ in intervals), default=0) / sample_rate
    lines = ["    item [0]:", "        class = \"IntervalTier\"", f"        name = \"{_quote(name)}\"",
             "        xmin = 0", f"        xmax = {xmax:.9f}", f"        intervals: size = {len(intervals)}"]
    for index, (start, end, text) in enumerate(intervals, 1):
        lines += [f"        intervals [{index}]:", f"            xmin = {start / sample_rate:.9f}",
                  f"            xmax = {end / sample_rate:.9f}", f"            text = \"{_quote(text)}\""]
    return "\n".join(lines)


def render_textgrid(record: Mapping[str, Any]) -> str:
    rate = int(record.get("sample_rate", 16000))
    words = record.get("words") or record.get("textgrid_words") or []
    if not words:
        raise ValueError("words are required for TextGrid ownership")
        words = list(grouped.values())
    word_intervals = [(int(w["start_sample"]), int(w["end_sample"]), str(w.get("text", ""))) for w in words]
    phone_intervals = [(int(p["start_sample"]), int(p["end_sample"]), f"{p['language']}:{p['native_phone']}") for p in record["phones"]]
    language_intervals = [(int(w["start_sample"]), int(w["end_sample"]), str(w["language"])) for w in words]
    end = max((x[1] for x in word_intervals + phone_intervals), default=0) / rate
    header = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {end:.9f}", "tiers? <exists>", "size = 3", "item []:"]
    tiers = [_tier("words", word_intervals, rate), _tier("phones", phone_intervals, rate), _tier("language", language_intervals, rate)]
    # Replace the helper's placeholder index with the canonical tier index.
    return "\n".join(header + [tier.replace("item [0]:", f"item [{i}]:", 1) for i, tier in enumerate(tiers, 1)]) + "\n"


def export_tts_artifacts(record: Mapping[str, Any], output_dir: str | Path, *, stem: str | None = None) -> dict[str, Path]:
    output = Path(output_dir).expanduser().absolute()
    if output.is_symlink():
        raise ValueError("TTS output namespace must not be symlinked")
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError("TTS output namespace must be a regular directory")
    uid = stem or str(record["uid"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", uid) or uid in {".", ".."}:
        raise ValueError("uid/stem is not a safe artifact filename")
    validate_record(record, "tts-training-record-v1")
    jsonl = output / "tts_training_records.jsonl"
    existing: list[str] = []
    if jsonl.exists():
        existing = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("uid") != record["uid"]]
    payload = "\n".join(existing + [canonical_json(record).decode("utf-8").rstrip("\n")]) + "\n"
    atomic_write_bytes(jsonl, payload.encode("utf-8"))
    textgrid = output / f"{uid}.TextGrid"
    atomic_write_bytes(textgrid, render_textgrid(record).encode("utf-8"))
    return {"jsonl": jsonl, "textgrid": textgrid}


# Stable descriptive aliases used by downstream stage owners.
export_training_record = export_tts_artifacts
write_textgrid = render_textgrid


def handle_tts(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    receipt_path = stage_dir / "receipt.json"
    workspace = stage_dir.parent.parent
    settings = config.get("tts", {}) if isinstance(config.get("tts", {}), Mapping) else {}
    candidates = [settings.get("alignment_jsonl"), settings.get("alignment_artifact")]
    source = next((Path(value).expanduser().absolute() for value in candidates if value and Path(value).is_file()), None)
    if source is None:
        receipt = make_receipt(stage="tts", status="BLOCKED", params={"implementation": "tts-training-record-v1"},
                               errors=[{"code": "publish_blocked", "message": "prepared ja-en alignment JSONL is required", "code_path": "tts.alignment_jsonl"}])
        atomic_write_json(receipt_path, receipt, workspace=workspace)
        return StageResult(stage="tts", status="BLOCKED", receipt_path=str(receipt_path))
    rows = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip(): rows.append(json.loads(line))
        outputs = []
        for row in rows:
            alignment = row.get("alignment", row)
            # The pipeline bridge always declares this stage input.  Keep the
            # standalone legacy helper usable for Task 7's fixture migration.
            if isinstance(config.get("stage_inputs"), Mapping) and alignment.get("schema") != "ja-prosody-alignment-v1":
                raise ValueError("TTS requires ja-prosody-alignment-v1 input")
            if alignment.get("schema") == "ja-prosody-alignment-v1":
                # Transitional v1 exporter adapter: consume only authoritative
                # native rows; do not consult a legacy `phones` field.
                alignment = {**alignment, "phones": list(alignment.get("native_phones", []))}
            train_wav = row.get("train_wav") or alignment.get("train_wav")
            alignment_wav = row.get("alignment_wav") or alignment.get("alignment_wav")
            if not train_wav or not alignment_wav:
                raise ValueError("alignment row must bind train_wav and alignment_wav")
            record = build_training_record(alignment, train_wav=train_wav, alignment_wav=alignment_wav,
                                            speaker=row.get("speaker", alignment.get("speaker")),
                                            text_layers=row.get("text_layers", alignment.get("text_layers")),
                                            audio_receipt=row.get("audio_receipt"))
            produced = export_tts_artifacts(record, stage_dir)
            outputs.extend(produced.values())
        receipt = make_receipt(stage="tts", status="COMPLETE", inputs={"artifacts": [{"path": str(source), "exists": True, "size": source.stat().st_size, "sha256": sha256_file(source)}]}, outputs=sorted({str(p) for p in outputs}), params={"implementation": "tts-training-record-v1"})
        atomic_write_json(receipt_path, receipt, workspace=workspace)
        return StageResult(stage="tts", status="COMPLETE", receipt_path=str(receipt_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        receipt = make_receipt(stage="tts", status="REJECTED", inputs={"artifacts": []}, params={"implementation": "tts-training-record-v1"}, errors=[{"code": "tts_invalid", "message": str(exc)}])
    atomic_write_json(receipt_path, receipt, workspace=workspace)
    return StageResult(stage="tts", status="REJECTED", receipt_path=str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("tts", handle_tts, output_namespace="tts")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("record", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(argv); export_tts_artifacts(json.loads(args.record.read_text(encoding="utf-8")), args.output); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_quality_masks", "build_training_record", "export_tts_artifacts", "export_training_record", "handle_tts", "register_stages", "render_textgrid", "write_textgrid"]
