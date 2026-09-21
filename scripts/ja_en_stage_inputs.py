"""Receipt-bound, UID-scoped inputs for the Japanese/English pipeline.

This module is deliberately a bridge between upstream frozen artifacts and the
runtime stages.  It does not infer a route from MFA output and it never turns
an observed phone list into an expected alias list.  Every returned object is
serializable so the core runner can persist and resume a UID independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .ja_en_schema import (
        JAContractError,
        atomic_write_bytes,
        atomic_write_json,
        artifact_record,
        canonical_json,
        reject_symlink,
        sha256_file,
        validate_exact_partition,
        validate_manifest,
        validate_receipt,
    )
except ImportError:  # pragma: no cover
    from ja_en_schema import (
        JAContractError,
        atomic_write_bytes,
        atomic_write_json,
        artifact_record,
        canonical_json,
        reject_symlink,
        sha256_file,
        validate_exact_partition,
        validate_manifest,
        validate_receipt,
    )


ANCHOR_REQUEST_SCHEMA = "ja-en-anchor-request-v1"
ALIGNMENT_REQUEST_SCHEMA = "ja-en-alignment-request-v1"
MERGE_REQUEST_SCHEMA = "ja-en-merge-request-v1"
TTS_ROW_SCHEMA = "tts-training-record-v1"


def _fail(code: str, message: str, path: str = "$") -> None:
    raise JAContractError(code, message, path)


def _workspace(path: str | os.PathLike[str]) -> Path:
    root = reject_symlink(path, code="manifest_symlink").absolute()
    if root.exists() and not root.is_dir():
        _fail("manifest_path_invalid", "workspace is not a directory", str(root))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _output(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except ValueError:
        _fail("manifest_path_invalid", "generated path escapes workspace", str(candidate))
    current = candidate.parent
    while True:
        if current.is_symlink():
            _fail("manifest_symlink", "generated parent is a symlink", str(current))
        if current == root or current.parent == current:
            break
        current = current.parent
    return candidate


def _json(path: str | os.PathLike[str]) -> Any:
    candidate = reject_symlink(path, code="receipt_output_symlink")
    if not candidate.is_file():
        _fail("receipt_missing", "JSON artifact is missing", str(candidate))
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("receipt_invalid", str(exc), str(candidate))


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        _fail("manifest_path_invalid", "path is required", field)
    candidate = reject_symlink(value, code="receipt_output_symlink").absolute()
    if not candidate.is_file():
        _fail("receipt_missing", "referenced file is missing", str(candidate))
    return candidate


def _file_ref(path: str | os.PathLike[str]) -> dict[str, Any]:
    candidate = _path(path, "$.path")
    return artifact_record(candidate)


def _verify_ref(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        _fail("receipt_invalid", "file reference is required", field)
    actual = _file_ref(value["path"])
    for key in ("path", "sha256"):
        if value.get(key) != actual.get(key):
            _fail("receipt_hash_mismatch", f"file reference differs at {key}", field)
    if "size" in value and value["size"] != actual["size"]:
        _fail("receipt_hash_mismatch", "file size differs", field)
    return actual


def _records(path: Path) -> list[dict[str, Any]]:
    candidate = reject_symlink(path, code="receipt_output_symlink")
    if not candidate.is_file():
        return []
    raw = candidate.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(data, Mapping):
        for container in ("items", "records", "graphs"):
            if isinstance(data.get(container), list):
                data = data[container]
                break
        else:
            if data.get("uid") is not None:
                data = [data]
    if not isinstance(data, list):
        # Stage outputs are commonly JSONL even when their filename is .json.
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        data = rows
    return [dict(row) for row in data if isinstance(row, Mapping)]


def _stage_index(config: Mapping[str, Any], root: Path, section: str, names: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    section_value = config.get(section) or {}
    candidates: list[Path] = []
    if isinstance(section_value, Mapping):
        for key in ("path", "receipt_path", "receipts_path", "locked_readings_path", "contracts_path", "graphs_path", "output"):
            value = section_value.get(key)
            if value:
                candidates.append(Path(str(value)).expanduser())
    for name in names:
        candidates.extend([root / "stages" / section / name, root / name])
    result: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        for row in _records(candidate):
            uid = row.get("uid") or row.get("id")
            if uid is not None:
                result.setdefault(str(uid), []).append(row)
    return result


def _load_manifest(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    source = config.get("input_manifest")
    if isinstance(source, (str, os.PathLike)):
        source_path = reject_symlink(source, code="receipt_output_symlink")
        if source_path.suffix.lower() == ".jsonl":
            data = _records(source_path)
        else:
            try:
                data = _json(source)
            except JAContractError as exc:
                if exc.code != "receipt_invalid":
                    raise
                data = _records(source_path)
    else:
        data = source
    rows = validate_manifest(data)
    audio = _stage_index(config, root, "audio", ("audio_transform_receipts.jsonl", "audio_transform_receipts.json", "receipts.jsonl"))
    reading = _stage_index(config, root, "reading", ("locked_readings.jsonl", "locked_readings.json"))
    frontend = _stage_index(config, root, "frontend", ("frontend_contracts.jsonl", "frontend_contracts.json", "frontend_reconstruction.json", "analysis.jsonl"))
    semantic = _stage_index(config, root, "semantic", ("semantic_graphs.jsonl", "semantic_graph.json", "alias_map.jsonl", "graphs.jsonl"))
    enriched: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row.get("uid") or row.get("id"))
        merged = dict(row)
        audio_rows = audio.get(uid, [])
        reading_rows = reading.get(uid, [])
        frontend_rows = frontend.get(uid, [])
        semantic_rows = semantic.get(uid, [])
        for source in audio_rows:
            if source.get("schema") == "audio-transform-receipt-v2":
                merged["audio_receipt"] = source
            elif source.get("audio_receipt"):
                merged["audio_receipt"] = source["audio_receipt"]
        if reading_rows:
            locked = []
            for row in reading_rows:
                if row.get("schema") == "ja-reading-selection-v2" or row.get("selected_reading"):
                    locked.append(row)
                elif isinstance(row.get("reading_lock"), Mapping):
                    locked.append(dict(row["reading_lock"]))
            if len(locked) == 1:
                merged["reading_lock"] = locked[0]
            elif locked:
                statuses = {str(row.get("status")) for row in locked}
                merged["reading_lock"] = {"schema": "ja-reading-selection-v2", "uid": uid,
                                           "status": "manual_verified" if statuses <= {"manual_verified", "origin_surface_confirmed", "COMPLETE", "VERIFIED", "LOCKED"} else "REJECTED",
                                           "selected_reading": "".join(str(row.get("selected_reading") or row.get("chosen_reading") or "") for row in locked),
                                           "candidates": [], "tokens": locked}
        units: list[dict[str, Any]] = []
        for source in frontend_rows:
            if source.get("units") and isinstance(source["units"], list):
                units.extend(dict(unit) for unit in source["units"]
                             if isinstance(unit, Mapping)
                             and (not ("lexical_status" in unit) or unit.get("lexical_status") == "lexical")
                             and not unit.get("morph_punct") and not unit.get("morph_ignored"))
            if source.get("lexical_units") and isinstance(source["lexical_units"], list):
                units.extend(dict(unit) for unit in source["lexical_units"]
                             if isinstance(unit, Mapping)
                             and (not ("lexical_status" in unit) or unit.get("lexical_status") == "lexical")
                             and not unit.get("morph_punct") and not unit.get("morph_ignored"))
            if "frontend" not in merged:
                merged["frontend"] = source
            if "canonical_spoken_text" not in merged:
                merged["canonical_spoken_text"] = source.get("canonical_text") or source.get("caller_text")
        alias_rows = [row for row in semantic_rows if row.get("alias")]
        aliases = {str(row.get("token_id")): row for row in alias_rows if row.get("token_id")}
        if units:
            for unit in units:
                token_id = str(unit.get("token_id") or unit.get("unit_id") or "")
                alias = aliases.get(token_id)
                if alias:
                    unit.setdefault("alias", alias.get("alias"))
                    unit.setdefault("pronunciation", alias.get("pronunciation") or alias.get("native_phones"))
                    unit.setdefault("reading", alias.get("reading"))
                unit.setdefault("unit_id", token_id)
                unit.setdefault("text", unit.get("caller_surface") or unit.get("surface") or "")
                unit.setdefault("char_span", unit.get("canonical_span"))
            merged["lexical_units"] = units
        graphs = [source for source in semantic_rows if source.get("schema") in {"ja-semantic-phone-graph-v1", "ja-semantic-phone-graph-v2"}]
        if graphs:
            merged["semantic_graphs"] = graphs
        for source in semantic_rows:
            if source.get("schema") in {"ja-semantic-phone-graph-v1", "ja-semantic-phone-graph-v2"}:
                merged.setdefault("semantic_graph", source)
        if alias_rows:
            merged["alias_rows"] = alias_rows
        enriched.append(merged)
    return enriched


def _audio_info(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            info = {"path": str(path), "sample_rate": handle.getframerate(),
                    "channels": handle.getnchannels(), "sample_width": handle.getsampwidth(),
                    "frames": handle.getnframes()}
    except (OSError, wave.Error) as exc:
        _fail("source_audio_invalid", str(exc), str(path))
    info["sha256"] = sha256_file(path)
    return info


def _audio_receipt(row: Mapping[str, Any], uid: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = row.get("audio_receipt") or row.get("audio_transform_receipt")
    if isinstance(receipt, (str, os.PathLike)):
        receipt = _json(receipt)
    if not isinstance(receipt, Mapping) or receipt.get("schema") != "audio-transform-receipt-v2":
        _fail("receipt_invalid", "audio-transform receipt is required", f"$.{uid}.audio_receipt")
    if receipt.get("uid") not in {None, uid}:
        _fail("receipt_identity_drift", "audio receipt UID differs", f"$.{uid}.audio_receipt.uid")
    for key in ("source", "train", "alignment", "sample_transform"):
        if not isinstance(receipt.get(key), Mapping):
            _fail("receipt_invalid", f"audio receipt lacks {key}", f"$.{uid}.audio_receipt.{key}")
    alignment = _verify_ref(receipt["alignment"], f"$.{uid}.audio_receipt.alignment")
    info = _audio_info(Path(alignment["path"]))
    if (info["sample_rate"], info["channels"], info["sample_width"]) != (16000, 1, 2):
        _fail("source_audio_invalid", "alignment audio must be 16 kHz mono PCM16", f"$.{uid}.alignment")
    return dict(receipt), info


def _reading(row: Mapping[str, Any], uid: str) -> dict[str, Any]:
    value = row.get("reading_lock") or row.get("reading_receipt")
    if isinstance(value, (str, os.PathLike)):
        value = _json(value)
    if not isinstance(value, Mapping):
        _fail("reading_unresolved", "locked reading receipt is required", f"$.{uid}.reading_lock")
    result = dict(value)
    if result.get("uid") not in {None, uid} or result.get("status") not in {"COMPLETE", "VERIFIED", "LOCKED", "manual_verified", "origin_surface_confirmed"}:
        _fail("reading_unresolved", "reading is not a locked completion", f"$.{uid}.reading_lock")
    if not isinstance(result.get("selected_reading"), str) or not result["selected_reading"]:
        _fail("reading_unresolved", "selected reading is missing", f"$.{uid}.reading_lock")
    return result


def _units(row: Mapping[str, Any], uid: str, text: str) -> list[dict[str, Any]]:
    units = row.get("lexical_units") or row.get("units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)) or not units:
        _fail("anchor_invalid", "frozen lexical units are required", f"$.{uid}.lexical_units")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous_end = 0
    for index, source in enumerate(units):
        if not isinstance(source, Mapping):
            _fail("anchor_invalid", "unit must be an object", f"$.{uid}.lexical_units[{index}]")
        unit = dict(source)
        unit_id = str(unit.get("unit_id") or f"{uid}:unit_{index:04d}")
        language = unit.get("language") or unit.get("route")
        if language not in {"ja", "en"}:
            _fail("anchor_invalid", "unit language must be ja or en", f"$.{uid}.lexical_units[{index}]")
        span = unit.get("char_span") or unit.get("canonical_span")
        if not isinstance(span, Sequence) or len(span) != 2 or type(span[0]) is not int or type(span[1]) is not int:
            _fail("anchor_invalid", "exact canonical character span is required", f"$.{uid}.lexical_units[{index}]")
        start, end = span
        if start < previous_end or end <= start or end > len(text) or text[start:end] != str(unit.get("text", "")):
            _fail("anchor_invalid", "unit span is not an exact canonical text slice", f"$.{uid}.lexical_units[{index}]")
        if unit_id in seen:
            _fail("anchor_invalid", "duplicate lexical unit ID", f"$.{uid}.lexical_units[{index}]")
        seen.add(unit_id)
        alias = unit.get("alias")
        pronunciation = unit.get("pronunciation")
        if not isinstance(alias, str) or not alias.isascii() or not isinstance(pronunciation, list) or not pronunciation:
            _fail("dictionary_roundtrip_failed", "unit needs a locked ASCII alias and pronunciation", f"$.{uid}.lexical_units[{index}]")
        if any(alias == prior.get("alias") for prior in result):
            _fail("anchor_invalid", "duplicate occurrence alias", f"$.{uid}.lexical_units[{index}].alias")
        unit.update({"unit_id": unit_id, "language": language, "route": language,
                     "char_span": [start, end], "text": text[start:end], "alias": alias,
                     "pronunciation": list(pronunciation)})
        result.append(unit)
        previous_end = end
    return result


def _route(units: Sequence[Mapping[str, Any]], row: Mapping[str, Any], uid: str) -> str:
    languages = [str(unit["language"]) for unit in units]
    expected = "mixed" if len(set(languages)) > 1 else languages[0]
    declared = row.get("route") or row.get("language_route")
    if declared is not None and declared != expected:
        _fail("anchor_invalid", "declared route differs from frozen unit routes", f"$.{uid}.route")
    return expected


def _write(root: Path, relative: str, payload: Mapping[str, Any]) -> str:
    path = _output(root, relative)
    atomic_write_json(path, payload, workspace=root)
    return str(path)


def _prepare_anchor_requests_impl(config: Mapping[str, Any], workspace: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Materialize one receipt-bound Qwen request for every manifest UID."""
    root = _workspace(workspace)
    qwen_value = (config.get("asr") or {}).get("qwen_forced_aligner")
    qwen = dict(qwen_value) if isinstance(qwen_value, Mapping) else {"model": qwen_value}
    model = qwen.get("model") or (config.get("asr") or {}).get("qwen_model")
    if not model:
        _fail("runtime_capability_missing", "Qwen forced-aligner identity is required", "$.asr")
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    global_aliases: set[str] = set()
    for row in _load_manifest(config, root):
        uid = str(row.get("uid") or row.get("id"))
        if uid in seen:
            _fail("manifest_duplicate_id", "duplicate UID", f"$.{uid}")
        seen.add(uid)
        receipt, alignment = _audio_receipt(row, uid)
        text = row.get("canonical_spoken_text") or row.get("text") or row.get("orig_text")
        if not isinstance(text, str) or not text:
            _fail("anchor_invalid", "canonical spoken text is required", f"$.{uid}.text")
        units = _units(row, uid, text)
        aliases = {str(unit["alias"]) for unit in units}
        duplicate_aliases = aliases.intersection(global_aliases)
        if duplicate_aliases:
            _fail("anchor_invalid", f"occurrence aliases repeat across UIDs: {sorted(duplicate_aliases)}", f"$.{uid}.lexical_units")
        global_aliases.update(aliases)
        request = {
            "schema": ANCHOR_REQUEST_SCHEMA, "uid": uid,
            "audio": alignment, "audio_receipt": receipt,
            "canonical_spoken_text": text, "lexical_units": units,
            "route": _route(units, row, uid), "reading_lock": _reading(row, uid),
            "qwen": {"model": str(model), "device": qwen.get("device", "cpu"), "dtype": qwen.get("dtype", "float32")},
            "source_receipt": row.get("source_receipt") or row.get("asr_receipt"),
            "frontend": dict(row.get("frontend") or {}), "semantic_graph": row.get("semantic_graph"),
            "semantic_graphs": list(row.get("semantic_graphs") or []),
            "speaker": row.get("speaker"),
        }
        request["request_digest"] = hashlib.sha256(canonical_json(request)).hexdigest()
        _write(root, f"stages/anchors/requests/{uid}.json", request)
        requests.append(request)
    _write(root, "stages/anchors/requests.json", {"schema": ANCHOR_REQUEST_SCHEMA, "uids": [r["uid"] for r in requests]})
    return requests


def _crop_wav(source: Path, destination: Path, start: int, end: int) -> dict[str, Any]:
    with wave.open(str(source), "rb") as inp:
        if (inp.getframerate(), inp.getnchannels(), inp.getsampwidth()) != (16000, 1, 2):
            _fail("source_audio_invalid", "crop source must be 16 kHz mono PCM16", str(source))
        total = inp.getnframes()
        if start < 0 or end <= start or end > total:
            _fail("alignment_invalid", "crop is outside source sample axis", str(destination))
        inp.setpos(start)
        payload = inp.readframes(end - start)
    reject_symlink(destination, code="manifest_symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(payload)
    return _audio_info(destination)


def _runs(anchor: Mapping[str, Any], root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = list(anchor.get("lexical_units") or anchor.get("units") or [])
    audio = anchor.get("audio")
    if isinstance(audio, Mapping):
        audio_path = Path(str(audio["path"]))
        total = int(audio.get("frames") or anchor.get("total_samples") or 0)
    else:
        audio_path = _path(audio, "$.audio")
        total = int(anchor.get("total_samples") or _audio_info(audio_path)["frames"])
    current_audio = _audio_info(audio_path)
    expected_audio = audio if isinstance(audio, Mapping) else {}
    if (expected_audio.get("sha256") and current_audio.get("sha256") != expected_audio.get("sha256")) or current_audio.get("frames") != total:
        _fail("receipt_hash_mismatch", "anchor audio changed before crop materialization", f"$.{anchor['uid']}.audio")
    padding = int(round(float((config.get("align") or {}).get("padding_ms", 100)) * 16))
    by_id = {str(unit["unit_id"]): unit for unit in units}
    planned = list(anchor.get("runs") or [])
    groups: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    if planned:
        for planned_run in planned:
            group = [by_id[str(unit_id)] for unit_id in planned_run.get("unit_ids", []) if str(unit_id) in by_id]
            if not group:
                _fail("alignment_invalid", "anchor run has no frozen units", f"$.{anchor['uid']}.runs")
            groups.append((planned_run, group))
    else:
        grouped: list[list[Mapping[str, Any]]] = []
        for unit in units:
            if not grouped or grouped[-1][0]["language"] != unit["language"]:
                grouped.append([unit])
            else:
                grouped[-1].append(unit)
        groups = [({}, group) for group in grouped]
    result: list[dict[str, Any]] = []
    uid = str(anchor["uid"])
    for index, (planned_run, group) in enumerate(groups):
        language = str(group[0]["language"])
        owner_start = int(planned_run.get("ownership_start_sample", 0 if len(groups) == 1 else group[0].get("start_sample", group[0]["char_span"][0] * 100)))
        owner_end = int(planned_run.get("ownership_end_sample", total if len(groups) == 1 else group[-1].get("end_sample", group[-1]["char_span"][1] * 100)))
        context_start = int(planned_run.get("context_start_sample", max(0, owner_start - padding)))
        context_end = int(planned_run.get("context_end_sample", min(total, owner_end + padding)))
        run_id = str(planned_run.get("run_id") or f"{uid}:run_{index:04d}:{language}")
        directory = _output(root, "stages", "align", "requests", uid)
        crop_path = directory / f"{index:04d}-{language}.wav"
        crop = _crop_wav(audio_path, crop_path, context_start, context_end)
        aliases = [{"alias": unit["alias"], "unit_id": unit["unit_id"], "token_id": unit.get("token_id", unit["unit_id"]),
                    "language": language, "pronunciation": list(unit["pronunciation"])} for unit in group]
        lab_path = directory / f"{index:04d}-{language}.lab"
        dict_path = directory / f"{index:04d}-{language}.dict"
        lab = "\n".join(f"{a['alias']}\t{a['alias']}" for a in aliases) + "\n"
        dictionary = "\n".join(f"{a['alias']} {' '.join(a['pronunciation'])}" for a in aliases) + "\n"
        atomic_write_bytes(lab_path, lab.encode("utf-8"), workspace=root)
        atomic_write_bytes(dict_path, dictionary.encode("utf-8"), workspace=root)
        result.append({"run_id": run_id, "language": language, "unit_ids": [u["unit_id"] for u in group],
                       "aliases": aliases, "crop": crop, "context_start_sample": context_start,
                       "context_end_sample": context_end, "global_offset_sample": context_start,
                       "ownership_start_sample": owner_start, "ownership_end_sample": owner_end,
                       "lab": artifact_record(lab_path), "locked_dictionary": artifact_record(dict_path),
                       "source_audio": dict(expected_audio) if expected_audio else _audio_info(audio_path)})
    return result


def _prepare_alignment_requests_impl(config: Mapping[str, Any], workspace: str | os.PathLike[str], anchor_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Materialize actual PCM16 crops and language runs from frozen anchors."""
    root = _workspace(workspace)
    manifest_rows = {str(row.get("uid") or row.get("id")): row for row in _load_manifest(config, root)}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for anchor in anchor_plans:
        if not isinstance(anchor, Mapping) or anchor.get("uid") in seen:
            _fail("manifest_duplicate_id", "duplicate or malformed anchor UID")
        uid = str(anchor["uid"]); seen.add(uid)
        if anchor.get("schema") not in {None, ANCHOR_REQUEST_SCHEMA, "ja-en-alignment-plan-v2"}:
            _fail("schema_invalid", "unexpected anchor request schema", f"$.{uid}.schema")
        normalized = dict(anchor)
        source_row = manifest_rows.get(uid)
        if source_row is None:
            _fail("manifest_invalid", "anchor UID is absent from manifest", f"$.{uid}")
        if "lexical_units" not in normalized and "units" in normalized:
            normalized["lexical_units"] = normalized["units"]
        if "audio_receipt" not in normalized:
            normalized["audio_receipt"] = source_row.get("audio_receipt")
        if "reading_lock" not in normalized:
            normalized["reading_lock"] = source_row.get("reading_lock")
        if "audio" not in normalized:
            receipt = source_row.get("audio_receipt") or {}
            normalized["audio"] = (receipt.get("alignment") or {}).get("path") if isinstance(receipt, Mapping) else None
            normalized["audio"] = normalized["audio"] or source_row.get("alignment_wav") or source_row.get("wav")
        source_units = {str(unit.get("unit_id") or unit.get("token_id")): unit for unit in source_row.get("lexical_units", []) if isinstance(unit, Mapping)}
        normalized["lexical_units"] = [{**source_units.get(str(unit.get("unit_id") or unit.get("token_id")), {}), **dict(unit)} for unit in normalized.get("lexical_units", [])]
        if not isinstance(normalized.get("audio"), Mapping):
            normalized["audio"] = _audio_info(_path(normalized["audio"], f"$.{uid}.audio"))
        if "route" not in normalized:
            normalized["route"] = _route(normalized["lexical_units"], source_row, uid)
        if not normalized.get("audio_receipt") or not normalized.get("reading_lock"):
            _fail("receipt_invalid", "anchor plan lacks audio or reading receipt", f"$.{uid}")
        runs = _runs(normalized, root, config)
        request = {"schema": ALIGNMENT_REQUEST_SCHEMA, "uid": uid,
                   "source_anchor": {"uid": uid, "digest": anchor.get("request_digest") or hashlib.sha256(canonical_json(anchor)).hexdigest()},
                   "audio": dict(normalized["audio"]), "audio_receipt": dict(normalized["audio_receipt"]),
                   "route": normalized["route"], "runs": runs,
                   "reading_lock": dict(normalized["reading_lock"]), "source_receipt": normalized.get("source_receipt"),
                   "frontend": dict(normalized.get("frontend") or source_row.get("frontend") or {}), "semantic_graph": normalized.get("semantic_graph") or source_row.get("semantic_graph"),
                   "semantic_graphs": list(normalized.get("semantic_graphs") or source_row.get("semantic_graphs") or []),
                   "qwen": dict(normalized.get("qwen") or {})}
        _write(root, f"stages/align/requests/{uid}.json", request)
        result.append(request)
    _write(root, "stages/align/requests.json", {"schema": ALIGNMENT_REQUEST_SCHEMA, "uids": [r["uid"] for r in result]})
    return result


def _prepare_merge_requests_impl(config: Mapping[str, Any], workspace: str | os.PathLike[str], anchor_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create pre-MFA merge contracts from frozen run aliases and ownership."""
    root = _workspace(workspace)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_padding = int(round(float((config.get("merge") or {}).get("initial_padding_ms", 40)) * 16))
    initial_padding = base_padding * 2
    for request in anchor_plans:
        if not isinstance(request, Mapping) or not request.get("uid"):
            _fail("schema_invalid", "alignment request must contain UID")
        uid = str(request["uid"])
        if uid in seen:
            _fail("manifest_duplicate_id", "duplicate merge UID", f"$.{uid}")
        seen.add(uid)
        runs = request.get("runs")
        if not isinstance(runs, Sequence) or not runs:
            _fail("alignment_invalid", "language runs are required", f"$.{uid}.runs")
        aliases: list[dict[str, Any]] = []
        expected_languages: dict[str, str] = {}
        ownership: dict[str, dict[str, int]] = {}
        raw_ledger: dict[str, Any] = {}
        inventory: dict[str, Any] = {}
        for run in runs:
            if not isinstance(run, Mapping) or run.get("language") not in {"ja", "en"}:
                _fail("alignment_invalid", "run language is required", f"$.{uid}.runs")
            run_id = str(run.get("run_id"))
            ownership[run_id] = {"start_sample": int(run["ownership_start_sample"]), "end_sample": int(run["ownership_end_sample"])}
            run_aliases = run.get("aliases")
            if not isinstance(run_aliases, Sequence) or not run_aliases:
                _fail("dictionary_roundtrip_failed", "run locked aliases are required", f"$.{uid}.{run_id}")
            for alias in run_aliases:
                if not isinstance(alias, Mapping) or not alias.get("alias") or alias.get("language") != run["language"]:
                    _fail("dictionary_roundtrip_failed", "run alias language/identity is invalid", f"$.{uid}.{run_id}")
                alias_row = dict(alias); aliases.append(alias_row)
                expected_languages[str(alias["alias"])] = str(alias["language"])
            raw_ledger[run_id] = {"status": "PENDING_MFA", "expected_aliases": [a["alias"] for a in run_aliases],
                                  "raw_textgrid": None, "verified": [], "rejected": [], "unresolved": []}
            inventory[run_id] = {"language": run["language"], "status": "REQUIRED_FROM_MODEL_ASSET",
                                 "expected_aliases": [a["alias"] for a in run_aliases]}
        if len({a["alias"] for a in aliases}) != len(aliases):
            _fail("dictionary_roundtrip_failed", "occurrence aliases must be globally unique per UID", f"$.{uid}.aliases")
        expected_units = [{"alias": a["alias"], "unit_id": a.get("unit_id"), "token_id": a.get("token_id"), "language": a["language"],
                           "pronunciation": list(a["pronunciation"])} for a in aliases]
        rerun = {"kind": "two_sided_context_rerun_v1", "initial_padding_samples": initial_padding,
                 "left": {"run_ids": [str(runs[0]["run_id"])], "padding_samples": initial_padding * 2},
                 "right": {"run_ids": [str(runs[-1]["run_id"])], "padding_samples": initial_padding * 2},
                 "max_attempts": 2, "ownership": ownership}
        merge = {"schema": MERGE_REQUEST_SCHEMA, "uid": uid,
                 "expected_languages": expected_languages, "expected_aliases": [a["alias"] for a in aliases],
                 "expected_units": expected_units, "locked_aliases": expected_units, "ownership": ownership,
                 "seams": list(request.get("seams") or []), "raw_ledger": raw_ledger,
                 "native_inventory": inventory, "runs": list(runs),
                 "source_receipt": request.get("source_receipt") or request.get("audio_receipt"),
                 "source_alignment": dict(request.get("audio") or {}), "reading_lock": request.get("reading_lock"),
                 "rerun_plan": rerun, "initial_padding_samples": initial_padding,
                 "audio_receipt": request.get("audio_receipt"),
                 "route": request.get("route"), "frontend": request.get("frontend"),
                 "semantic_graph": request.get("semantic_graph"), "semantic_graphs": list(request.get("semantic_graphs") or []), "qwen": request.get("qwen")}
        _write(root, f"stages/merge/requests/{uid}.json", merge)
        result.append(merge)
    _write(root, "stages/merge/requests.json", {"schema": MERGE_REQUEST_SCHEMA, "uids": [r["uid"] for r in result]})
    return result


def _blocked(schema: str, uid: str, exc: Exception) -> dict[str, Any]:
    error = exc.as_dict() if isinstance(exc, JAContractError) else {"code": "publish_blocked", "message": str(exc)}
    return {"schema": schema, "uid": uid, "status": "REJECTED", "errors": [error]}


def prepare_anchor_requests(config: Mapping[str, Any], workspace: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return all UID requests, retaining a rejected UID as an explicit row."""
    try:
        return _prepare_anchor_requests_impl(config, workspace)
    except Exception:
        root = _workspace(workspace)
        rows = _load_manifest(config, root)
        result: list[dict[str, Any]] = []
        for row in rows:
            uid = str(row.get("uid") or row.get("id"))
            subconfig = dict(config); subconfig["input_manifest"] = [row]
            try:
                result.extend(_prepare_anchor_requests_impl(subconfig, root))
            except Exception as exc:
                result.append(_blocked(ANCHOR_REQUEST_SCHEMA, uid, exc))
        return result


def prepare_alignment_requests(config: Mapping[str, Any], workspace: str | os.PathLike[str], anchor_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        return _prepare_alignment_requests_impl(config, workspace, anchor_plans)
    except Exception:
        result: list[dict[str, Any]] = []
        for plan in anchor_plans:
            uid = str(plan.get("uid") or "") if isinstance(plan, Mapping) else ""
            try:
                result.extend(_prepare_alignment_requests_impl(config, workspace, [plan]))
            except Exception as exc:
                result.append(_blocked(ALIGNMENT_REQUEST_SCHEMA, uid, exc))
        return result


def prepare_merge_requests(config: Mapping[str, Any], workspace: str | os.PathLike[str], anchor_plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        return _prepare_merge_requests_impl(config, workspace, anchor_plans)
    except Exception:
        result: list[dict[str, Any]] = []
        for plan in anchor_plans:
            uid = str(plan.get("uid") or "") if isinstance(plan, Mapping) else ""
            try:
                result.extend(_prepare_merge_requests_impl(config, workspace, [plan]))
            except Exception as exc:
                result.append(_blocked(MERGE_REQUEST_SCHEMA, uid, exc))
        return result


def _validate_phone_identity(alignment: Mapping[str, Any], uid: str, aliases: set[str]) -> None:
    phones = alignment.get("native_phones")
    if not isinstance(phones, Sequence) or not phones:
        _fail("publish_blocked", "verified native phones are required", f"$.{uid}.native_phones")
    seen: set[str] = set()
    for index, phone in enumerate(phones):
        if not isinstance(phone, Mapping) or phone.get("uid", uid) != uid:
            _fail("publish_blocked", "phone crosses UID boundary", f"$.{uid}.native_phones[{index}]")
        alias = phone.get("alias")
        if alias not in aliases:
            _fail("publish_blocked", "phone alias was not frozen upstream", f"$.{uid}.native_phones[{index}]")
        phone_id = str(phone.get("phone_id") or "")
        if not phone_id or phone_id in seen:
            _fail("publish_blocked", "phone IDs must be unique and namespaced", f"$.{uid}.native_phones[{index}]")
        seen.add(phone_id)


def _validate_raw_mfa(raw_mfa: Mapping[str, Any], uid: str) -> None:
    runs = raw_mfa.get("runs")
    if not isinstance(runs, Sequence) or not runs:
        _fail("publish_blocked", "raw MFA run ledger is required", f"$.{uid}.raw_mfa.runs")
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or not run.get("run_id"):
            _fail("publish_blocked", "raw MFA run identity is required", f"$.{uid}.raw_mfa.runs[{index}]")
        textgrid = run.get("raw_textgrid") or run.get("textgrid")
        if not isinstance(textgrid, Mapping):
            _fail("publish_blocked", "raw TextGrid reference is required", f"$.{uid}.raw_mfa.runs[{index}]")
        _verify_ref(textgrid, f"$.{uid}.raw_mfa.runs[{index}].raw_textgrid")


def _stage_artifact(root: Path, section: str, name: str) -> dict[str, Any] | None:
    path = root / "stages" / section / name
    return artifact_record(path) if path.is_file() and not path.is_symlink() else None


def _strict_ledger(root: Path, uid: str, language: str) -> dict[str, Any] | None:
    align_root = root / "stages" / "align"
    candidates = [align_root / f"strict_{language}_mfa.json"]
    candidates.extend(path for path in align_root.rglob(f"strict_{language}_mfa.json") if not path.is_symlink())
    candidates.extend(path for path in align_root.rglob(f"strict_{language}*.json") if not path.is_symlink())
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        payload = _json(path)
        if isinstance(payload, Mapping) and payload.get("uid") == uid:
            result = dict(payload)
            result["artifact"] = artifact_record(path)
            return result
    return None


def _inventory_ref(config: Mapping[str, Any], language: str) -> dict[str, Any] | None:
    mfa = config.get("mfa") or {}
    key = "japanese_metadata" if language == "ja" else "english_metadata"
    value = mfa.get(key)
    if not value:
        return None
    path = _path(value, f"$.mfa.{key}")
    payload = _json(path)
    return {"language": language, "phones": list(payload.get("phones") or []),
            "asset": _file_ref(path), "metadata": payload}


def _mora_graph(manifest_row: Mapping[str, Any], uid: str) -> dict[str, Any]:
    graphs = list(manifest_row.get("semantic_graphs") or [])
    if not graphs and isinstance(manifest_row.get("semantic_graph"), Mapping):
        graphs = [manifest_row["semantic_graph"]]
    if not graphs:
        return {"moras": [], "relations": []}
    moras: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for graph in graphs:
        token_id = graph.get("token_id") or next((u.get("unit_id") for u in manifest_row.get("lexical_units", []) if u.get("language") == "ja"), "")
        for node in graph.get("mora_nodes", []):
            if isinstance(node, Mapping):
                item = dict(node); item["mora_id"] = f"{uid}:{token_id}:{node.get('id')}"
                moras.append(item)
        for edge in graph.get("edges", []):
            if isinstance(edge, Mapping) and edge.get("relation") == "mora_phone":
                relations.append({"mora_id": f"{uid}:{token_id}:{edge.get('mora_id')}",
                                  "semantic_phone_id": f"{uid}:{token_id}:{edge.get('phone_id')}"})
    return {"moras": moras, "relations": relations, "source": graphs}


def _enrich_merged_alignment(config: Mapping[str, Any], root: Path, alignment: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    """Join actual merge output with W1/W2 and strict-MFA receipts by UID/token."""
    uid = str(alignment["uid"])
    enriched = dict(alignment)
    units = list(manifest_row.get("lexical_units") or [])
    if "locked_aliases" not in enriched:
        enriched["locked_aliases"] = [{"alias": u["alias"], "unit_id": u["unit_id"], "language": u["language"],
                                        "pronunciation": list(u["pronunciation"]), "token_id": u.get("token_id", u["unit_id"])}
                                       for u in units if u.get("alias") and u.get("pronunciation")]
    ledgers = {language: _strict_ledger(root, uid, language) for language in ("ja", "en")}
    ledgers = {language: ledger for language, ledger in ledgers.items() if ledger is not None}
    if "raw_mfa" not in enriched:
        raw_runs: list[dict[str, Any]] = []
        for language, ledger in ledgers.items():
            for run in ledger.get("runs", []):
                phones = list(run.get("phones") or [])
                ref = None
                if phones and phones[0].get("raw_artifact_path"):
                    ref = {"path": phones[0]["raw_artifact_path"], "sha256": phones[0].get("raw_artifact_sha256")}
                elif run.get("textgrid"):
                    ref = _file_ref(run["textgrid"])
                if ref:
                    raw_runs.append({"run_id": run.get("run_id"), "language": language, "raw_textgrid": ref,
                                     "phones": phones, "ownership_start_sample": run.get("ownership_start_sample"),
                                     "ownership_end_sample": run.get("ownership_end_sample"),
                                     "context_start_sample": run.get("context_start_sample"),
                                     "context_end_sample": run.get("context_end_sample"),
                                     "ledger": ledger.get("ledger")})
        assets: dict[str, Any] = {}
        mfa = config.get("mfa") or {}
        for language in {str(u.get("language")) for u in units}:
            acoustic_key = "japanese_acoustic" if language == "ja" else "english_acoustic"
            dictionary_key = "japanese_dictionary" if language == "ja" else "english_dictionary"
            for kind, key in (("acoustic_model", acoustic_key), ("dictionary", dictionary_key)):
                value = mfa.get(key)
                if value and Path(str(value)).is_file() and not Path(str(value)).is_symlink():
                    assets[f"{language}:{kind}"] = _file_ref(value)
        for run in raw_runs:
            language = str(run.get("language"))
            run["model_assets"] = {key: value for key, value in assets.items() if key.startswith(f"{language}:")}
        enriched["raw_mfa"] = {"runs": raw_runs, "ledgers": ledgers, "model_assets": assets}
    if "native_inventory" not in enriched:
        inventories = {language: _inventory_ref(config, language) for language in {u.get("language") for u in units}}
        enriched["native_inventory"] = {language: value for language, value in inventories.items() if value is not None}
    if "reading_evidence" not in enriched:
        lock = manifest_row.get("reading_lock") or {}
        tokens = list(lock.get("tokens") or [lock]) if isinstance(lock, Mapping) else []
        selected = {str(item.get("token_id")): item.get("selected_reading") or item.get("chosen_reading")
                    for item in tokens if item.get("token_id")}
        enriched["reading_evidence"] = {"locks": tokens, "selected_readings": selected,
                                         "selected_reading": next(iter(selected.values())) if len(selected) == 1 else selected}
        enriched["selected_reading"] = enriched["reading_evidence"]["selected_reading"]
        enriched["selected_readings"] = selected
    enriched.setdefault("mora_graph", _mora_graph(manifest_row, uid))
    # MFA-native rows are already bound to semantic templates by the merge
    # stage.  Reconstruct relations from those bound IDs only; a semantic node
    # may never acquire a timing interval through list position or label.
    actual_phones = list(enriched.get("native_phones") or [])
    if actual_phones:
        relations: list[dict[str, Any]] = []
        for phone in actual_phones:
            token_id = phone.get("token_id")
            if not isinstance(token_id, str) or not token_id:
                _fail("native_basic_mapping_ambiguous", "bound native phone token identity is required", f"$.{uid}.native_phones")
            for mora_id in phone.get("mora_ids", []):
                relations.append({"mora_id": f"{uid}:{token_id}:{mora_id}", "phone_id": phone.get("phone_id")})
        graph = dict(enriched["mora_graph"])
        graph["relations"] = relations
        enriched["mora_graph"] = graph
    enriched.setdefault("frontend", manifest_row.get("frontend") or {})
    enriched.setdefault("model_ids", {"qwen": (config.get("asr") or {}).get("qwen_forced_aligner"),
                                       "mfa": (config.get("mfa") or {}).get("japanese_acoustic")})
    enriched.setdefault("dict_ids", {"ja": (config.get("mfa") or {}).get("japanese_dictionary"),
                                      "en": (config.get("mfa") or {}).get("english_dictionary")})
    if "partition" not in enriched:
        expected_units = [str(u["unit_id"]) for u in units]
        verified: list[str] = []
        rejected: list[str] = []
        unresolved: list[str] = []
        for ledger in ledgers.values():
            data = ledger.get("ledger") or {}
            verified.extend(str(x) for x in data.get("verified", []))
            rejected.extend(str(x) for x in data.get("rejected", []))
            unresolved.extend(str(x) for x in data.get("unresolved", []))
        enriched["partition"] = {"verified": verified, "rejected": rejected, "unresolved": unresolved}
        if set(verified + rejected + unresolved) != set(expected_units):
            # Preserve the exact ledger; the final validator will reject rather
            # than replacing a missing unit with an observed phone alias.
            enriched["partition"] = {"verified": verified, "rejected": rejected, "unresolved": sorted(set(expected_units) - set(verified + rejected + unresolved))}
    if "audio_receipt" not in enriched:
        enriched["audio_receipt"] = manifest_row.get("audio_receipt")
    if "alignment_wav" not in enriched and isinstance(enriched.get("audio_receipt"), Mapping):
        enriched["alignment_wav"] = enriched["audio_receipt"].get("alignment")
    if "train_wav" not in enriched and isinstance(enriched.get("audio_receipt"), Mapping):
        enriched["train_wav"] = enriched["audio_receipt"].get("train")
    enriched.setdefault("source_receipt", _json(root / "stages" / "align" / "receipt.json") if (root / "stages" / "align" / "receipt.json").is_file() else None)
    enriched.setdefault("seams", list(alignment.get("seams") or []))
    # The frozen pronunciation is the join key across semantic aliases and
    # native MFA phones.  A count-only or set-only match would permit reordered
    # phones to pass, so compare the ordered sequences exactly.
    semantic_aliases = {str(row.get("alias")): row for row in manifest_row.get("alias_rows", []) if isinstance(row, Mapping) and row.get("alias")}
    observed = {}
    for phone in enriched.get("native_phones", []):
        if isinstance(phone, Mapping) and phone.get("alias"):
            observed.setdefault(str(phone["alias"]), []).append(str(phone.get("native_phone") or phone.get("phone")))
    for alias_row in enriched.get("locked_aliases", []):
        alias = str(alias_row.get("alias"))
        expected = list(alias_row.get("pronunciation") or [])
        semantic = semantic_aliases.get(alias)
        semantic_phones = list((semantic or {}).get("native_phones") or (semantic or {}).get("pronunciation") or expected)
        actual = observed.get(alias, [])
        if actual and (expected != semantic_phones or actual != expected):
            _fail("mfa_inventory_mismatch", "locked, semantic, and raw native phone order differ", f"$.{uid}.aliases.{alias}")
    return enriched


def assemble_tts_rows(config: Mapping[str, Any], workspace: str | os.PathLike[str], merged_alignments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Assemble authoritative post-merge TTS records, preserving provenance."""
    root = _workspace(workspace)
    manifest = {str(row.get("uid") or row.get("id")): row for row in _load_manifest(config, root)}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alignment in merged_alignments:
        if not isinstance(alignment, Mapping):
            _fail("publish_blocked", "merged alignment must be an object")
        uid = str(alignment.get("uid") or "")
        if not uid or uid in seen:
            _fail("publish_blocked", "missing or duplicate merged UID", f"$.{uid}")
        seen.add(uid)
        if uid not in manifest:
            _fail("publish_blocked", "merged UID is absent from manifest", f"$.{uid}")
        alignment = _enrich_merged_alignment(config, root, alignment, manifest[uid])
        if alignment.get("partition", {}).get("unresolved") or alignment.get("status") in {"PARTIAL", "UNRESOLVED", "REJECTED"}:
            _fail("publish_blocked", "partial or unresolved alignment cannot publish", f"$.{uid}.partition")
        locked = alignment.get("locked_aliases")
        if not isinstance(locked, Sequence) or not locked:
            _fail("publish_blocked", "locked alias rows are required", f"$.{uid}.locked_aliases")
        aliases = {str(row.get("alias")) for row in locked if isinstance(row, Mapping)}
        if None in aliases or "None" in aliases:
            _fail("publish_blocked", "invalid locked alias", f"$.{uid}.locked_aliases")
        partition = alignment.get("partition")
        if not isinstance(partition, Mapping):
            _fail("partition_not_exact", "final partition is required", f"$.{uid}.partition")
        partition_values = {str(k): list(v) for k, v in partition.items()}
        unit_ids = {str(row.get("unit_id")) for row in locked if row.get("unit_id")}
        partition_expected = unit_ids if set().union(*(set(values) for values in partition_values.values())) <= unit_ids else aliases
        validate_exact_partition(sorted(partition_expected), partition_values)
        _validate_phone_identity(alignment, uid, aliases)
        for field in ("native_inventory", "raw_mfa", "reading_evidence", "mora_graph", "frontend", "model_ids", "dict_ids", "seams"):
            if field not in alignment:
                _fail("publish_blocked", f"authoritative {field} evidence is required", f"$.{uid}.{field}")
        _validate_raw_mfa(alignment["raw_mfa"], uid)
        if not isinstance(alignment["native_inventory"], Mapping) or not alignment["native_inventory"]:
            _fail("publish_blocked", "native model inventory is required", f"$.{uid}.native_inventory")
        reading = _reading(manifest[uid], uid)
        if alignment.get("selected_reading") != reading.get("selected_reading") or alignment.get("reading_evidence", {}).get("selected_reading") != reading.get("selected_reading"):
            _fail("reading_unresolved", "post-merge reading differs from lock", f"$.{uid}.reading")
        audio_receipt = alignment.get("audio_receipt") or manifest[uid].get("audio_receipt")
        if isinstance(audio_receipt, (str, os.PathLike)):
            audio_receipt = _json(audio_receipt)
        if not isinstance(audio_receipt, Mapping):
            _fail("receipt_invalid", "post-merge audio receipt is required", f"$.{uid}.audio_receipt")
        _, audio = _audio_receipt({"audio_receipt": audio_receipt}, uid)
        train_ref = _verify_ref(alignment.get("train_wav") or audio_receipt["train"], f"$.{uid}.train_wav")
        align_ref = _verify_ref(alignment.get("alignment_wav") or audio_receipt["alignment"], f"$.{uid}.alignment_wav")
        if train_ref["path"] != audio_receipt["train"].get("path") or align_ref["path"] != audio_receipt["alignment"].get("path"):
            _fail("receipt_hash_mismatch", "TTS audio differs from transform receipt", f"$.{uid}.audio")
        source_receipt = alignment.get("source_receipt")
        if source_receipt is None:
            _fail("publish_blocked", "source stage receipt is required", f"$.{uid}.source_receipt")
        validate_receipt(source_receipt)
        try:
            from .ja_tts_export import build_training_record
        except ImportError:  # pragma: no cover
            from ja_tts_export import build_training_record
        enriched = dict(alignment)
        enriched["uid"] = uid
        record = build_training_record(enriched, train_wav=train_ref["path"], alignment_wav=align_ref["path"],
                                       speaker=manifest[uid].get("speaker"),
                                       text_layers=alignment.get("frontend") or manifest[uid].get("frontend"),
                                       audio_receipt=dict(audio_receipt), sample_rate=16000)
        record.update({"locked_aliases": list(locked), "native_inventory": alignment["native_inventory"],
                       "raw_mfa": alignment["raw_mfa"], "reading_evidence": alignment["reading_evidence"],
                       "partition": dict(partition), "frontend": alignment["frontend"],
                       "model_ids": alignment["model_ids"], "dict_ids": alignment["dict_ids"],
                       "seams": list(alignment["seams"]), "source_receipt": source_receipt,
                       "expected_aliases": sorted(aliases)})
        _write(root, f"stages/tts/rows/{uid}.json", record)
        result.append(record)
    lines = b"".join(canonical_json(row) for row in result)
    atomic_write_bytes(_output(root, "stages/tts/tts_training_records.jsonl"), lines, workspace=root)
    return result
