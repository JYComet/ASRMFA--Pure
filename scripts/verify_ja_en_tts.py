"""Independent verifier for JA/EN TTS artifacts.

The verifier has its own JSONL/TextGrid parsing and recomputes file hashes,
sample durations, language namespaces and mora relations.  Producer receipt
status is evidence to inspect, never a reason to accept a malformed record.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import wave
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None

try:
    from .ja_en_schema import PRODUCTION_STAGES, SCHEMAS, sha256_file
except ImportError:  # pragma: no cover
    from ja_en_schema import PRODUCTION_STAGES, SCHEMAS, sha256_file


def _error(code: str, message: str, path: str | None = None) -> dict[str, Any]:
    row = {"code": code, "message": message}
    if path is not None:
        row["path"] = path
    return row


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, errors = [], []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [_error("receipt_missing", str(exc), str(path))]
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(_error("schema_invalid", str(exc), f"{path}:{index}"))
    return rows, errors


def _verify_audio(info: Mapping[str, Any], label: str, errors: list[dict[str, Any]]) -> None:
    path = info.get("path") if isinstance(info, Mapping) else None
    if not isinstance(path, str):
        errors.append(_error("source_audio_invalid", f"{label} has no path")); return
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        errors.append(_error("source_audio_invalid", f"{label} is missing or symlinked", path)); return
    expected = info.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(_error("receipt_invalid", f"{label} sha256 is mandatory", path))
    elif expected != sha256_file(candidate):
        errors.append(_error("receipt_hash_mismatch", f"{label} hash differs", path))
    try:
        with wave.open(str(candidate), "rb") as handle:
            frames = handle.getnframes(); rate = handle.getframerate()
            channels = handle.getnchannels(); width = handle.getsampwidth()
            compression = handle.getcomptype()
        if any(info.get(key) is None for key in ("frames", "sample_rate", "channels", "sample_width")):
            errors.append(_error("receipt_invalid", f"{label} frame/rate/channel/width metadata is mandatory", path))
        if info.get("frames") is not None and int(info["frames"]) != frames:
            errors.append(_error("audio_transform_invalid", f"{label} frame count differs", path))
        if info.get("sample_rate") is not None and int(info["sample_rate"]) != rate:
            errors.append(_error("audio_transform_invalid", f"{label} sample rate differs", path))
        if info.get("channels") is not None and int(info["channels"]) != channels:
            errors.append(_error("audio_transform_invalid", f"{label} channel count differs", path))
        if info.get("sample_width") is not None and int(info["sample_width"]) != width:
            errors.append(_error("audio_transform_invalid", f"{label} sample width differs", path))
        if compression != "NONE":
            errors.append(_error("source_audio_invalid", f"{label} is compressed rather than PCM", path))
    except (OSError, wave.Error, TypeError, ValueError, OverflowError) as exc:
        errors.append(_error("source_audio_invalid", str(exc), path))


def _wav_payload(path: str) -> tuple[dict[str, int], bytes]:
    with wave.open(path, "rb") as handle:
        info = {"sample_rate": handle.getframerate(), "channels": handle.getnchannels(), "sample_width": handle.getsampwidth(), "frames": handle.getnframes()}
        return info, handle.readframes(info["frames"])


def _decode_pcm(payload: bytes, info: Mapping[str, int]) -> Any:
    if np is None:
        raise RuntimeError("numpy is required for independent audio replay")
    width, channels, frames = info["sample_width"], info["channels"], info["frames"]
    if width == 1:
        values = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0; scale = 128.0
    elif width == 2:
        values = np.frombuffer(payload, dtype="<i2").astype(np.float64); scale = 32768.0
    elif width == 3:
        bytes_ = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        values = (bytes_[:, 0].astype(np.int32) | (bytes_[:, 1].astype(np.int32) << 8) | (bytes_[:, 2].astype(np.int32) << 16)).astype(np.int32)
        values[values & 0x800000 != 0] -= 1 << 24; values = values.astype(np.float64); scale = float(1 << 23)
    elif width == 4:
        values = np.frombuffer(payload, dtype="<i4").astype(np.float64); scale = float(1 << 31)
    else:
        raise ValueError("unsupported PCM width")
    return np.clip(values.reshape(frames, channels) / scale, -1.0, 1.0)


def _half_up_frames(source_frames: int, source_rate: int, target_rate: int) -> int:
    return int((source_frames * target_rate + source_rate // 2) // source_rate)


def _declared_path(value: Any, root: Path | None) -> Path | None:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, (str, Path)) or not str(value):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ((root or Path.cwd()) / path).absolute()


def _verify_inventory_artifact(record: Mapping[str, Any], phones: Sequence[Mapping[str, Any]],
                               root: Path | None, errors: list[dict[str, Any]]) -> None:
    """Re-read model archive metadata when an alignment binds it.

    The compact ``native_inventory`` map is retained for backwards-compatible
    dev fixtures, but a production alignment must also bind the exact archive
    metadata and digest.  This prevents a producer from defining its own
    inventory in the training JSONL.
    """
    def inventory_values(payload: Any, language: str) -> set[str]:
        if isinstance(payload, Mapping):
            for key in ("phones", "inventory", "phone_inventory", "phone_set"):
                if key in payload:
                    payload = payload[key]; break
            if isinstance(payload, Mapping):
                payload = payload.get(language, payload.get("phones", []))
        if not isinstance(payload, list): return set()
        values: set[str] = set()
        for item in payload:
            if isinstance(item, str): values.add(item)
            elif isinstance(item, Mapping):
                value = item.get("phone", item.get("label", item.get("symbol")))
                if isinstance(value, str): values.add(value)
        return values

    def read_archive(path: Path, language: str) -> set[str]:
        if not zipfile.is_zipfile(path):
            raise ValueError("MFA acoustic inventory must be a ZIP archive")
        candidates = [name for name in zipfile.ZipFile(path).namelist()
                      if Path(name).name.lower() in {"meta.json", "metadata.json", "inventory.json", "meta.yaml", "metadata.yaml", "meta.yml", "metadata.yml"}]
        if not candidates: raise ValueError("acoustic archive has no metadata inventory")
        with zipfile.ZipFile(path) as archive:
            for name in candidates:
                raw = archive.read(name).decode("utf-8")
                if name.lower().endswith("json"):
                    payload = json.loads(raw)
                else:
                    try:
                        import yaml
                    except ImportError as exc:
                        raise ValueError("PyYAML is required for MFA metadata.yaml") from exc
                    payload = yaml.safe_load(raw)
                values = inventory_values(payload, language)
                if values: return values
        raise ValueError(f"acoustic archive metadata has no {language} phone inventory")

    raw = record.get("raw_mfa")
    runs = raw.get("runs", []) if isinstance(raw, Mapping) else []
    if not isinstance(runs, list) or not runs:
        return
    declared_inventory = record.get("native_inventory")
    assets = raw.get("model_assets", {}) if isinstance(raw, Mapping) else {}

    def asset_for(language: str) -> Mapping[str, Any] | None:
        """Resolve the W3 language asset without searching the workspace."""
        value = assets.get(f"{language}:acoustic_model") if isinstance(assets, Mapping) else None
        if value is None and isinstance(assets, Mapping):
            lang = assets.get(language)
            value = lang.get("acoustic_model") if isinstance(lang, Mapping) else None
        return value if isinstance(value, Mapping) else None

    for run in runs:
        if not isinstance(run, Mapping):
            errors.append(_error("mfa_inventory_mismatch", "raw MFA run is not an object")); continue
        language = str(run.get("language", ""))
        asset = asset_for(language)
        path_value = (run.get("native_inventory_path") or run.get("inventory_path") or
                      (asset.get("path") if asset else None))
        digest = (run.get("native_inventory_sha256") or run.get("inventory_sha256") or
                  (asset.get("sha256") if asset else None))
        path = _declared_path(path_value, root)
        if path is None or not isinstance(digest, str):
            errors.append(_error("mfa_inventory_mismatch", "each MFA run needs acoustic archive path and hash", str(path) if path else None)); continue
        if path.is_symlink() or not path.is_file():
            errors.append(_error("mfa_inventory_mismatch", "bound native inventory archive is missing", str(path))); continue
        if digest != sha256_file(path):
            errors.append(_error("receipt_hash_mismatch", "native inventory archive digest differs", str(path)))
        try:
            allowed = read_archive(path, language)
            run_id = str(run.get("run_id", ""))
            run_phones = [phone for phone in phones
                          if (phone.get("run_id") is None or str(phone.get("run_id")) == run_id)
                          and phone.get("language") == language]
            # Model archives generally contain a superset inventory.  The
            # independent relation is therefore per-run coverage: every
            # exported native label must be present in the bound archive.
            required = {str(phone.get("native_phone")) for phone in run_phones}
            if not required.issubset(allowed):
                errors.append(_error("mfa_inventory_mismatch", "record phone is absent from acoustic archive", str(path)))
            for phone in phones:
                if phone.get("run_id") is not None and str(phone.get("run_id")) != run_id: continue
                if phone.get("language") != language: continue
                if str(phone.get("native_phone")) not in allowed:
                    errors.append(_error("mfa_inventory_mismatch", "phone is outside bound acoustic inventory", str(path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(_error("mfa_inventory_mismatch", str(exc), str(path)))


def _verify_locked_dictionary(record: Mapping[str, Any], phones: Sequence[Mapping[str, Any]],
                              root: Path | None, errors: list[dict[str, Any]]) -> None:
    """Check each run's dictionary against only that run's aliases."""
    raw = record.get("raw_mfa")
    runs = raw.get("runs", []) if isinstance(raw, Mapping) else []
    if not isinstance(runs, list) or not runs: return
    assets = raw.get("model_assets", {}) if isinstance(raw, Mapping) else {}

    def dictionary_asset(language: str) -> Mapping[str, Any] | None:
        value = assets.get(f"{language}:dictionary") if isinstance(assets, Mapping) else None
        if value is None and isinstance(assets, Mapping):
            lang = assets.get(language)
            value = lang.get("dictionary") if isinstance(lang, Mapping) else None
        return value if isinstance(value, Mapping) else None

    seen_runs: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping): continue
        run_id = str(run.get("run_id", ""))
        if run_id in seen_runs: errors.append(_error("dictionary_roundtrip_failed", "duplicate MFA run id", run_id))
        seen_runs.add(run_id)
        asset = dictionary_asset(str(run.get("language", "")))
        dictionary = _declared_path(run.get("locked_dict_path") or run.get("dictionary_path") or
                                    (asset.get("path") if asset else None), root)
        digest = (run.get("locked_dict_sha256") or run.get("dictionary_sha256") or
                  (asset.get("sha256") if asset else None))
        if dictionary is None or not isinstance(digest, str):
            errors.append(_error("dictionary_roundtrip_failed", "each MFA run needs locked dictionary path and hash", run_id)); continue
        if dictionary.is_symlink() or not dictionary.is_file():
            errors.append(_error("dictionary_roundtrip_failed", "locked dictionary is missing", str(dictionary))); continue
        if digest != sha256_file(dictionary):
            errors.append(_error("receipt_hash_mismatch", "locked dictionary digest differs", str(dictionary)))
        run_phones = [phone for phone in phones if str(phone.get("run_id", "")) == run_id]
        expected: dict[str, list[str]] = {}
        for alias in run.get("aliases", []) if isinstance(run.get("aliases"), list) else []:
            if isinstance(alias, Mapping) and alias.get("alias"):
                expected[str(alias["alias"])] = [str(value) for value in alias.get("pronunciation", [])]
        for phone in run_phones:
            if isinstance(phone.get("alias"), str) and str(phone["alias"]) not in expected:
                expected.setdefault(str(phone["alias"]), []).append(str(phone.get("native_phone")))
        observed: dict[str, list[str]] = {}
        try:
            for line_no, line in enumerate(dictionary.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if not fields or fields[0].startswith("#"): continue
                alias, pronunciation = fields[0], fields[1:]
                if alias in observed or not pronunciation:
                    errors.append(_error("dictionary_roundtrip_failed", "locked dictionary has duplicate/empty alias", f"{dictionary}:{line_no}")); continue
                observed[alias] = pronunciation
            if observed != expected:
                errors.append(_error("dictionary_roundtrip_failed", "run dictionary does not exactly match its aliases", str(dictionary)))
        except OSError as exc:
            errors.append(_error("dictionary_roundtrip_failed", str(exc), str(dictionary)))


def _verify_reading_artifact(record: Mapping[str, Any], root: Path | None,
                             errors: list[dict[str, Any]]) -> None:
    evidence = record.get("reading_evidence")
    if not isinstance(evidence, Mapping):
        return
    locks = evidence.get("locks")
    selected_map = record.get("selected_readings")
    strict = isinstance(locks, list) or isinstance(selected_map, Mapping) and bool(selected_map)
    if not strict:
        # Legacy single-token development records are checked by the caller's
        # release blocker.  They must never be used as a production evidence
        # substitute, but retaining this branch keeps old offline fixtures
        # inspectable.
        return
    if not isinstance(locks, list) or not locks:
        errors.append(_error("reading_unresolved", "reading_evidence.locks is required for every token")); return

    def ref(name: str, *aliases: str) -> tuple[Path | None, str | None]:
        value: Any = evidence.get(name)
        if value is None:
            for alias in aliases:
                value = evidence.get(alias)
                if value is not None: break
        if value is None and isinstance(evidence.get("sources"), Mapping):
            value = evidence["sources"].get(name)
        digest = None
        if isinstance(value, Mapping):
            digest = value.get("sha256") or value.get("sha256_digest")
        else:
            digest = evidence.get(f"{name}_sha256")
        return _declared_path(value, root), digest

    lock_path, lock_hash = ref("locked_readings_path", "locks_path")
    reconstruction_path, reconstruction_hash = ref("reconstruction_path", "frontend_reconstruction_path")
    analysis_path, analysis_hash = ref("analysis_path", "frontend_analysis_path", "candidate_analysis_path")
    alias_path, alias_hash = ref("semantic_alias_map_path", "alias_map_path")
    refs = (("locked_readings", lock_path, lock_hash), ("reconstruction", reconstruction_path, reconstruction_hash),
            ("analysis", analysis_path, analysis_hash), ("alias_map", alias_path, alias_hash))
    for name, path, digest in refs:
        if path is None or path.is_symlink() or not path.is_file():
            errors.append(_error("reading_unresolved", f"{name} artifact path/hash is required", str(path) if path else name)); continue
        if not isinstance(digest, str) or digest != sha256_file(path):
            errors.append(_error("receipt_hash_mismatch", f"{name} artifact digest is missing or differs", str(path)))

    expected_uid = str(record.get("uid", ""))
    expected_keys = {(expected_uid, str(lock.get("token_id")), str(lock.get("candidate_id")))
                    for lock in locks if isinstance(lock, Mapping)}
    if len(expected_keys) != len(locks) or any(not isinstance(lock, Mapping) or not lock.get("token_id") or not lock.get("candidate_id") for lock in locks):
        errors.append(_error("reading_unresolved", "reading locks must be unique (uid, token_id, candidate_id) rows"))

    def read_rows(path: Path | None, *, nested: bool = True) -> list[Mapping[str, Any]]:
        if path is None or not path.is_file(): return []
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text) if path.suffix.lower() == ".json" else None
            raw_values: list[Any] = []
            if value is not None:
                raw_values = value.get("records", value.get("rows", value.get("items", []))) if isinstance(value, Mapping) else value
                if isinstance(raw_values, Mapping): raw_values = [raw_values]
            else:
                raw_values = [json.loads(line) for line in text.splitlines() if line.strip()]
            result: list[Mapping[str, Any]] = []
            for item in raw_values if isinstance(raw_values, list) else []:
                if isinstance(item, Mapping):
                    result.append(item)
                    if nested:
                        for key in ("units", "locks", "candidates", "rows"):
                            child = item.get(key)
                            if isinstance(child, list):
                                for x in child:
                                    if not isinstance(x, Mapping): continue
                                    # W2 reconstruction stores uid on the
                                    # record and token/candidate on units.
                                    # Preserve that binding while flattening;
                                    # never match a unit by position or text.
                                    child_row = dict(x)
                                    if child_row.get("uid") is None and item.get("uid") is not None:
                                        child_row["uid"] = item["uid"]
                                    if child_row.get("candidate_id") is None:
                                        child_row["candidate_id"] = child_row.get("locked_candidate_id")
                                    result.append(child_row)
            return result
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(_error("reading_unresolved", str(exc), str(path))); return []

    lock_rows = read_rows(lock_path)
    reconstruction_rows = read_rows(reconstruction_path)
    analysis_rows = read_rows(analysis_path)
    alias_rows = read_rows(alias_path, nested=False)
    def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (str(row.get("uid", "")), str(row.get("token_id", "")), str(row.get("candidate_id", "")))
    def value(row: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in row: return row[name]
        return None
    by_lock = {key(row): row for row in lock_rows if row.get("uid") is not None and row.get("token_id") is not None}
    by_reconstruction = {key(row): row for row in reconstruction_rows if row.get("uid") is not None and row.get("token_id") is not None}
    by_analysis = {key(row): row for row in analysis_rows if row.get("uid") is not None and row.get("token_id") is not None}
    by_alias = {key(row): row for row in alias_rows if row.get("uid") is not None and row.get("token_id") is not None}
    phones_by_unit: dict[str, list[str]] = {}
    for phone in record.get("phones", []):
        if isinstance(phone, Mapping): phones_by_unit.setdefault(str(phone.get("unit_id")), []).append(str(phone.get("native_phone")))
    for lock in locks:
        if not isinstance(lock, Mapping): continue
        k = (expected_uid, str(lock.get("token_id")), str(lock.get("candidate_id")))
        locked = by_lock.get(k)
        if locked is None:
            errors.append(_error("reading_unresolved", "locked_readings.jsonl lacks exact token/candidate row", str(k))); continue
        selected = lock.get("chosen_reading", lock.get("selected_reading"))
        for row_name, row in (("locked", locked), ("reconstruction", by_reconstruction.get(k)), ("analysis", by_analysis.get(k)), ("alias", by_alias.get(k))):
            if row is None:
                errors.append(_error("reading_unresolved", f"{row_name} artifact lacks exact token/candidate row", str(k))); continue
            if row_name != "alias" and value(row, "chosen_reading", "selected_reading") not in {None, selected}:
                errors.append(_error("reading_unresolved", f"{row_name} chosen reading differs", str(k)))
            if row_name in {"locked", "reconstruction", "analysis"}:
                for field, aliases in (("analysis_digest", ("analysis_digest",)), ("canonical_sha256", ("canonical_sha256", "canonicalSHA"))):
                    expected = value(lock, *aliases)
                    actual = value(row, *aliases)
                    if expected is not None and actual != expected: errors.append(_error("reading_unresolved", f"{row_name} {field} differs", str(k)))
            if row_name == "alias":
                if value(row, "reading", "chosen_reading", "selected_reading") not in {None, selected}:
                    errors.append(_error("reading_unresolved", "semantic alias reading differs", str(k)))
                pronunciation = row.get("pronunciation")
                expected_phones = phones_by_unit.get(str(lock.get("token_id")), [])
                if isinstance(pronunciation, list) and expected_phones and [str(v) for v in pronunciation] != expected_phones:
                    errors.append(_error("reading_unresolved", "semantic alias pronunciation differs", str(k)))


def _verify_semantic_artifact(record: Mapping[str, Any], phones: Sequence[Mapping[str, Any]],
                              root: Path | None, errors: list[dict[str, Any]]) -> None:
    path = _declared_path(record.get("semantic_graph_path"), root)
    digest = record.get("semantic_graph_sha256")
    has_japanese = any(str(phone.get("language")) == "ja" for phone in phones if isinstance(phone, Mapping))
    if path is None:
        if has_japanese or record.get("selected_readings"):
            errors.append(_error("semantic_relation_ambiguous", "semantic graph path/hash is required"))
        return
    if path.is_symlink() or not path.is_file():
        errors.append(_error("semantic_relation_ambiguous", "semantic graph artifact is missing", str(path))); return
    if not isinstance(digest, str) or digest != sha256_file(path):
        errors.append(_error("receipt_hash_mismatch", "semantic graph digest differs", str(path)))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        graphs = payload.get("graphs", []) if isinstance(payload, Mapping) else []
        if not isinstance(graphs, list) or not graphs:
            raise ValueError("semantic graph list is empty")
        if not all(isinstance(graph, Mapping) for graph in graphs): raise ValueError("semantic graph entries are not objects")
        wanted = {str(lock.get("candidate_id")) for lock in record.get("reading_evidence", {}).get("locks", []) if isinstance(lock, Mapping)}
        selected_graphs = [graph for graph in graphs
                           if (graph.get("uid") is not None and str(graph.get("uid")) == str(record.get("uid")))
                           or (graph.get("uid") is None and (not wanted or str(graph.get("candidate_id")) in wanted))]
        if wanted and {str(graph.get("candidate_id")) for graph in selected_graphs} != wanted:
            errors.append(_error("semantic_relation_ambiguous", "semantic graph lacks a locked candidate", str(path)))
        if any(graph.get("uid") is None and wanted and str(graph.get("candidate_id")) not in wanted for graph in graphs):
            errors.append(_error("semantic_relation_ambiguous", "semantic graph contains an unbound candidate", str(path)))
        expected_by_unit: dict[str, list[str]] = {}
        for phone in phones:
            if isinstance(phone, Mapping): expected_by_unit.setdefault(str(phone.get("unit_id")), []).append(str(phone.get("native_phone")))
        for graph in selected_graphs:
            node_rows = graph.get("semantic_phone_nodes") or graph.get("phone_nodes") or []
            labels = [str(node.get("phone", node.get("native_phone"))) for node in node_rows if isinstance(node, Mapping)]
            graph_token = str(graph.get("token_id", ""))
            expected = expected_by_unit.get(graph_token, []) if graph_token else []
            target = graph.get("target") if isinstance(graph.get("target"), Mapping) else {}
            target_phones = target.get("phones") if isinstance(target.get("phones"), list) else graph.get("phones")
            if isinstance(target_phones, list):
                labels = [str(value) for value in target_phones]
            if expected and labels != expected:
                errors.append(_error("semantic_relation_ambiguous", "semantic target phone sequence differs from exported token", str(path)))
            elif not expected and labels and not any(set(values).issubset(set(labels)) for values in expected_by_unit.values()):
                errors.append(_error("semantic_relation_ambiguous", "semantic graph does not cover exported native phones", str(path)))
            semantic_ids = {str(node.get("id")) for node in (graph.get("nodes", []) + node_rows) if isinstance(node, Mapping)}
            mora_ids = {str(node.get("id")) for node in graph.get("mora_nodes", []) if isinstance(node, Mapping)}
            edges = graph.get("edges", [])
            mora_edges = [edge for edge in edges if isinstance(edge, Mapping) and edge.get("relation") == "mora_phone"]
            if has_japanese and not mora_edges:
                errors.append(_error("mora_phone_relation_unresolved", "semantic graph has no mora_phone edges", str(path)))
            if mora_edges and (not semantic_ids or not mora_ids):
                errors.append(_error("mora_phone_relation_unresolved", "semantic mora edge references an empty node set", str(path)))
            for edge in mora_edges:
                if str(edge.get("phone_id")) not in semantic_ids or str(edge.get("mora_id")) not in mora_ids:
                    errors.append(_error("mora_phone_relation_unresolved", "semantic mora edge references unknown node", str(path)))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(_error("semantic_relation_ambiguous", str(exc), str(path)))


def _replay_pcm16_transform(source_path: str, output_path: str, recipe: Mapping[str, Any]) -> bool:
    source_info, source_payload = _wav_payload(source_path)
    output_info, output_payload = _wav_payload(output_path)
    method = recipe.get("method")
    if method in {"identity_fixture_v1", "pcm16_copy_v1"}:
        return source_info == output_info and source_payload == output_payload
    source_signal = _decode_pcm(source_payload, source_info).mean(axis=1)
    start = int(recipe.get("source_start", 0)); end = int(recipe.get("source_end", source_info["frames"]))
    if start < 0 or end < start or end > source_info["frames"]:
        raise ValueError("invalid source crop")
    cropped = source_signal[start:end]
    target_rate = int(recipe.get("target_rate", recipe.get("sample_rate", output_info["sample_rate"])))
    target_frames = int(recipe.get("output_frames", output_info["frames"]))
    theoretical_frames = _half_up_frames(end - start, source_info["sample_rate"], target_rate)
    if target_frames != theoretical_frames:
        return False
    if source_info["sample_rate"] == target_rate:
        expected = cropped
    else:
        if resample_poly is None: raise RuntimeError("scipy.signal.resample_poly is unavailable")
        divisor = math.gcd(source_info["sample_rate"], target_rate)
        if recipe.get("up") is not None and int(recipe["up"]) != target_rate // divisor: return False
        if recipe.get("down") is not None and int(recipe["down"]) != source_info["sample_rate"] // divisor: return False
        expected = np.asarray(resample_poly(cropped, target_rate // divisor, source_info["sample_rate"] // divisor,
                                            window=("kaiser", 5.0), padtype="constant"), dtype=np.float64)
    if target_frames <= 0:
        target_frames = _half_up_frames(len(cropped), source_info["sample_rate"], target_rate)
    if expected.shape[0] < target_frames: expected = np.pad(expected, (0, target_frames - expected.shape[0]))
    expected = expected[:target_frames]
    expected_payload = np.rint(np.clip(expected, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return output_info["sample_rate"] == target_rate and output_info["channels"] == 1 and output_info["sample_width"] == 2 and output_info["frames"] == target_frames and output_payload == expected_payload


def _verify_raw_runs(record: Mapping[str, Any], phones: Sequence[Mapping[str, Any]],
                     root: Path | None, errors: list[dict[str, Any]]) -> None:
    raw = record.get("raw_mfa")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("runs"), list):
        return
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for phone in phones:
        if phone.get("run_id") is None: continue
        interval_value = phone.get("raw_interval_index", phone.get("interval_index", phone.get("raw_interval_id")))
        key = (str(phone.get("run_id")), str(interval_value))
        if key in by_key: errors.append(_error("alignment_invalid", "duplicate raw MFA interval identity", f"{key[0]}/{key[1]}"))
        by_key[key] = phone
    seen: set[tuple[str, str]] = set()
    for run in raw["runs"]:
        if not isinstance(run, Mapping):
            errors.append(_error("alignment_invalid", "raw MFA run is not an object")); continue
        textgrid_ref = run.get("raw_textgrid") or run.get("raw_artifact_path") or run.get("textgrid_path")
        path = _declared_path(textgrid_ref, root)
        if path is None or path.is_symlink() or not path.is_file():
            errors.append(_error("receipt_output_missing", "raw MFA artifact is missing", str(path) if path else None)); continue
        expected_hash = (run.get("raw_artifact_sha256") or run.get("sha256") or
                         (textgrid_ref.get("sha256") if isinstance(textgrid_ref, Mapping) else None))
        if not isinstance(expected_hash, str) or expected_hash != sha256_file(path):
            errors.append(_error("receipt_hash_mismatch", "raw MFA artifact digest differs", str(path)))
        try:
            tiers = _textgrid_parse(path)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(_error("alignment_invalid", str(exc), str(path))); continue
        tier_name = str(run.get("raw_tier", run.get("tier", "phones")))
        intervals = tiers.get(tier_name, [])
        if not intervals:
            errors.append(_error("alignment_invalid", f"raw MFA tier {tier_name!r} is empty or absent", str(path))); continue
        rate = int(run.get("sample_rate", record.get("sample_rate", 16000)))
        offset = int(run.get("crop_offset_sample", run.get("offset_sample", 0)))
        words = tiers.get(str(run.get("raw_word_tier", run.get("word_tier", "words"))), [])
        ownership_start = int(run.get("ownership_start_sample", -2**63)); ownership_end = int(run.get("ownership_end_sample", 2**63 - 1))
        ordered_raw: dict[str, list[str]] = {}
        for index, interval in enumerate(intervals):
            interval_id = str(interval.get("interval_index", index + 1))
            key = (str(run.get("run_id", "")), interval_id)
            phone = by_key.get(key)
            if phone is None:
                if interval.get("text") in {"sil", "sp", "spn", "<eps>"}: continue
                errors.append(_error("alignment_invalid", "raw MFA interval has no exported phone ownership", f"{path}:{tier_name}[{interval_id}]")); continue
            seen.add(key)
            start = offset + int(round(float(interval["xmin"]) * rate))
            end = offset + int(round(float(interval["xmax"]) * rate))
            expected_label = str(phone.get("native_phone", ""))
            if interval.get("text") != expected_label:
                errors.append(_error("mfa_inventory_mismatch", "raw MFA phone label differs from record", f"{path}:{tier_name}[{interval_id}]"))
            if (start, end) != (phone.get("start_sample"), phone.get("end_sample")):
                errors.append(_error("alignment_invalid", "raw MFA sample coordinates differ from record", f"{path}:{tier_name}[{interval_id}]"))
            if start < ownership_start or end > ownership_end:
                errors.append(_error("alignment_invalid", "raw MFA interval escapes run ownership", f"{path}:{tier_name}[{interval_id}]"))
            if phone.get("raw_interval_index", phone.get("interval_index", phone.get("raw_interval_id"))) not in {int(interval_id) if interval_id.isdigit() else interval_id, interval_id}:
                errors.append(_error("alignment_invalid", "raw MFA interval identity differs from record", f"{path}:{tier_name}[{interval_id}]"))
            containing = [word.get("text") for word in words if word.get("text") and word.get("xmin", 0) <= interval["xmin"] and interval["xmax"] <= word.get("xmax", 0)]
            if len(containing) != 1:
                errors.append(_error("alignment_invalid", "raw MFA phone is not owned by exactly one raw word", f"{path}:{tier_name}[{interval_id}]"))
            elif phone.get("alias") not in {containing[0], str(containing[0])}:
                errors.append(_error("alignment_invalid", "raw MFA word ownership alias differs from phone", f"{path}:{tier_name}[{interval_id}]"))
            if len(containing) == 1:
                ordered_raw.setdefault(str(containing[0]), []).append(str(interval.get("text")))
            phone_artifact = _declared_path(phone.get("raw_artifact_path"), root)
            if phone_artifact is not None and phone_artifact != path.absolute():
                errors.append(_error("alignment_invalid", "phone raw artifact binding differs from run", str(path)))
        expected_aliases = {str(item.get("alias")): [str(value) for value in item.get("pronunciation", [])]
                           for item in run.get("aliases", []) if isinstance(item, Mapping) and item.get("alias")}
        for alias, pronunciation in expected_aliases.items():
            if ordered_raw.get(alias, []) != pronunciation:
                errors.append(_error("dictionary_roundtrip_failed", "semantic alias pronunciation differs from ordered raw MFA phones", str(path)))
    missing = set(by_key) - seen
    for run_id, interval_id in sorted(missing):
        errors.append(_error("alignment_invalid", f"phone raw interval {run_id}/{interval_id} is not represented in raw MFA"))


def verify_training_record(record: Mapping[str, Any], *, root: Path | None = None) -> list[dict[str, Any]]:
    """Return independent contract errors for one tts-training-record-v1."""
    errors: list[dict[str, Any]] = []
    if record.get("schema") != "tts-training-record-v1":
        errors.append(_error("schema_unknown", "unexpected training record schema")); return errors
    uid = record.get("uid")
    if not isinstance(uid, str) or not uid:
        errors.append(_error("schema_invalid", "uid is required"))
    sample_rate = record.get("sample_rate")
    if type(sample_rate) is not int or sample_rate <= 0:
        errors.append(_error("schema_invalid", "sample_rate must be a positive integer")); sample_rate = 16000
    for key in ("train_wav", "alignment_wav"):
        if isinstance(record.get(key), Mapping): _verify_audio(record[key], key, errors)
        else: errors.append(_error("source_audio_invalid", f"{key} metadata is missing"))
    transform = record.get("audio_transform")
    if transform is None:
        errors.append(_error("audio_transform_invalid", "audio_transform is required"))
    else:
        if not isinstance(transform, Mapping):
            errors.append(_error("audio_transform_invalid", "audio_transform must be an object"))
        else:
            source_frames = transform.get("source_frames")
            alignment_frames = transform.get("output_frames")
            actual_source = record.get("train_wav", {}).get("frames") if isinstance(record.get("train_wav"), Mapping) else None
            actual_alignment = record.get("alignment_wav", {}).get("frames") if isinstance(record.get("alignment_wav"), Mapping) else None
            if source_frames is not None and actual_source is not None and int(source_frames) != int(actual_source):
                errors.append(_error("audio_transform_invalid", "source frame transform disagrees with audio metadata"))
            if alignment_frames is not None and actual_alignment is not None and int(alignment_frames) != int(actual_alignment):
                errors.append(_error("audio_transform_invalid", "alignment frame transform disagrees with audio metadata"))
            sample_transform = transform.get("sample_transform")
            method = sample_transform.get("method") if isinstance(sample_transform, Mapping) else None
            if method not in {"identity_fixture_v1", "pcm16_copy_v1", "pcm16_reencode_v1", "linear_integer_axis_v1", "scipy_resample_poly_v1"}:
                errors.append(_error("audio_transform_invalid", "unsupported or missing audio transform recipe"))
            source_path = transform.get("source_path")
            if not source_path and isinstance(transform.get("source"), Mapping):
                source_path = transform["source"].get("path")
                if not transform.get("source_sha256"):
                    transform = {**transform, "source_sha256": transform["source"].get("sha256")}
            if source_path:
                source = Path(str(source_path)).expanduser()
                if source.is_symlink() or not source.is_file():
                    errors.append(_error("source_audio_invalid", "transform source is missing or symlinked", str(source)))
                else:
                    source_hash = transform.get("source_sha256")
                    if not isinstance(source_hash, str) or len(source_hash) != 64:
                        errors.append(_error("receipt_invalid", "transform source hash is mandatory", str(source)))
                    elif source_hash != sha256_file(source):
                        errors.append(_error("receipt_hash_mismatch", "transform source hash differs", str(source)))
                    source_info = None
                    try:
                        source_info, _ = _wav_payload(str(source))
                        declared_source = transform.get("source") if isinstance(transform.get("source"), Mapping) else {}
                        for field in ("frames", "sample_rate", "channels", "sample_width"):
                            declared = transform.get(f"source_{field}", declared_source.get(field))
                            if declared is not None and int(declared) != int(source_info[field]):
                                errors.append(_error("audio_transform_invalid", f"transform source {field} differs", str(source)))
                    except (OSError, wave.Error, ValueError, TypeError) as exc:
                        errors.append(_error("source_audio_invalid", str(exc), str(source)))
                    declared_header = sample_transform.get("source_header") if isinstance(sample_transform, Mapping) else None
                    if source_info is not None and isinstance(declared_header, Mapping):
                        for field in ("sample_rate", "channels", "sample_width", "frames"):
                            if declared_header.get(field) != source_info.get(field):
                                errors.append(_error("audio_transform_invalid", f"sample transform source header {field} differs", str(source)))
            declared_output_header = sample_transform.get("output_header") if isinstance(sample_transform, Mapping) else None
            if isinstance(declared_output_header, Mapping):
                try:
                    actual_output_info, _ = _wav_payload(str(record["alignment_wav"]["path"]))
                    for field in ("sample_rate", "channels", "sample_width", "frames"):
                        if declared_output_header.get(field) != actual_output_info.get(field):
                            errors.append(_error("audio_transform_invalid", f"sample transform output header {field} differs"))
                except (OSError, wave.Error) as exc:
                    errors.append(_error("audio_transform_invalid", str(exc)))
            if isinstance(sample_transform, Mapping) and sample_transform.get("output_sha256"):
                actual_alignment_path = Path(str(record["alignment_wav"]["path"])).expanduser()
                if sample_transform.get("output_sha256") != sha256_file(actual_alignment_path):
                    errors.append(_error("receipt_hash_mismatch", "sample transform output hash differs", str(actual_alignment_path)))
            train_transform = transform.get("train_transform")
            if source_path and isinstance(train_transform, Mapping):
                try:
                    if not _replay_pcm16_transform(str(source_path), str(record["train_wav"]["path"]), train_transform):
                        errors.append(_error("audio_transform_invalid", "independent source-to-training audio replay differs"))
                except (OSError, wave.Error, ValueError, RuntimeError, TypeError) as exc:
                    errors.append(_error("audio_transform_invalid", str(exc)))
            if method in {"identity_fixture_v1", "pcm16_copy_v1"}:
                try:
                    train_info, train_bytes = _wav_payload(str(record["train_wav"]["path"]))
                    align_info, align_bytes = _wav_payload(str(record["alignment_wav"]["path"]))
                    if train_info != align_info or train_bytes != align_bytes:
                        errors.append(_error("audio_transform_invalid", "identity transform output is not PCM-bit-exact"))
                except (OSError, wave.Error) as exc:
                    errors.append(_error("source_audio_invalid", str(exc)))
            elif method == "linear_integer_axis_v1":
                try:
                    if not source_path:
                        raise ValueError("linear transform source is not bound")
                    source_info, source_payload = _wav_payload(str(source_path))
                    align_info, align_payload = _wav_payload(str(record["alignment_wav"]["path"]))
                    source_matrix = _decode_pcm(source_payload, source_info).mean(axis=1)
                    start = int(sample_transform.get("source_start", 0)); end = int(sample_transform.get("source_end", source_info["frames"]))
                    if start < 0 or end < start or end > source_info["frames"]:
                        raise ValueError("invalid source crop")
                    cropped = source_matrix[start:end]
                    target_rate = int(transform.get("sample_rate", align_info["sample_rate"]))
                    target_frames = int(sample_transform.get("output_frames", _half_up_frames(len(cropped), source_info["sample_rate"], target_rate)))
                    theoretical_frames = _half_up_frames(end - start, source_info["sample_rate"], target_rate)
                    if target_frames != theoretical_frames:
                        raise ValueError("declared output frames do not match half-up frame policy")
                    if source_info["sample_rate"] == target_rate:
                        expected_signal = cropped
                    elif resample_poly is not None:
                        # W1's first receipt revision called this recipe
                        # linear_integer_axis_v1 while its implementation was
                        # already scipy.signal.resample_poly.  Replay the
                        # actual frozen transform for both names.
                        divisor = math.gcd(source_info["sample_rate"], target_rate)
                        expected_signal = np.asarray(resample_poly(cropped, target_rate // divisor, source_info["sample_rate"] // divisor,
                                                                   window=("kaiser", 5.0), padtype="constant"), dtype=np.float64)
                        if expected_signal.shape[0] < target_frames:
                            expected_signal = np.pad(expected_signal, (0, target_frames - expected_signal.shape[0]))
                        expected_signal = expected_signal[:target_frames]
                    else:
                        raise RuntimeError("scipy.signal.resample_poly is unavailable for W1 transform replay")
                    expected_pcm = np.rint(np.clip(expected_signal, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    if align_info["sample_rate"] != target_rate or align_info["sample_width"] != 2 or align_info["channels"] != 1 or align_payload != expected_pcm:
                        errors.append(_error("audio_transform_invalid", "independent linear audio replay differs from alignment WAV"))
                except (OSError, wave.Error, ValueError, RuntimeError) as exc:
                    errors.append(_error("audio_transform_invalid", str(exc)))
            elif method == "scipy_resample_poly_v1":
                try:
                    if resample_poly is None:
                        raise RuntimeError("scipy.signal.resample_poly is unavailable")
                    if not source_path:
                        raise ValueError("scipy transform source is not bound")
                    source_info, source_payload = _wav_payload(str(source_path))
                    align_info, align_payload = _wav_payload(str(record["alignment_wav"]["path"]))
                    source_matrix = _decode_pcm(source_payload, source_info)
                    source_rate = int(sample_transform.get("source_rate", source_info["sample_rate"]))
                    target_rate = int(sample_transform.get("target_rate", transform.get("sample_rate", align_info["sample_rate"])))
                    if source_info["sample_rate"] != source_rate:
                        raise ValueError("source rate metadata disagrees with WAV header")
                    start = int(sample_transform.get("source_start", 0)); end = int(sample_transform.get("source_end", source_info["frames"]))
                    if start < 0 or end < start or end > source_info["frames"]:
                        raise ValueError("invalid source crop")
                    cropped = source_matrix[start:end].mean(axis=1)
                    target_frames = int(sample_transform.get("output_frames", _half_up_frames(len(cropped), source_rate, target_rate)))
                    theoretical_frames = _half_up_frames(end - start, source_rate, target_rate)
                    if target_frames != theoretical_frames:
                        raise ValueError("declared output frames do not match half-up frame policy")
                    divisor = math.gcd(source_rate, target_rate)
                    if sample_transform.get("up") is not None and int(sample_transform["up"]) != target_rate // divisor:
                        raise ValueError("resample up factor differs")
                    if sample_transform.get("down") is not None and int(sample_transform["down"]) != source_rate // divisor:
                        raise ValueError("resample down factor differs")
                    expected_signal = np.asarray(resample_poly(cropped, target_rate // divisor, source_rate // divisor,
                                                               window=("kaiser", 5.0), padtype="constant"), dtype=np.float64)
                    if expected_signal.shape[0] < target_frames:
                        expected_signal = np.pad(expected_signal, (0, target_frames - expected_signal.shape[0]))
                    expected_signal = expected_signal[:target_frames]
                    expected_pcm = np.rint(np.clip(expected_signal, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                    if (align_info["sample_rate"] != target_rate or align_info["sample_width"] != 2 or
                            align_info["channels"] != 1 or align_info["frames"] != target_frames or
                            align_payload != expected_pcm):
                        errors.append(_error("audio_transform_invalid", "independent scipy resample replay differs from alignment WAV"))
                except (OSError, wave.Error, ValueError, RuntimeError, TypeError) as exc:
                    errors.append(_error("audio_transform_invalid", str(exc)))
    phones = record.get("phones")
    durations = record.get("durations")
    if not isinstance(phones, list) or not phones:
        errors.append(_error("tts_invalid", "phones must be non-empty")); phones = []
    if not isinstance(durations, list) or len(durations) != len(phones):
        errors.append(_error("tts_invalid", "durations cardinality differs from phones")); durations = []
    previous = -1
    phone_ids: set[str] = set()
    for index, phone in enumerate(phones):
        if not isinstance(phone, Mapping):
            errors.append(_error("tts_invalid", "phone is not an object", f"phones[{index}]")); continue
        start, end = phone.get("start_sample"), phone.get("end_sample")
        alignment_frames = record.get("alignment_wav", {}).get("frames") if isinstance(record.get("alignment_wav"), Mapping) else None
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or start < previous or (alignment_frames is not None and isinstance(end, int) and end > int(alignment_frames)):
            errors.append(_error("alignment_invalid", "phone sample span is invalid or non-monotonic", f"phones[{index}]"))
        if isinstance(start, int) and isinstance(end, int) and index < len(durations) and durations[index] != end - start:
            errors.append(_error("alignment_invalid", "duration does not equal sample span", f"phones[{index}]"))
        train_rate = record.get("train_sample_rate")
        train_frames = record.get("train_wav", {}).get("frames") if isinstance(record.get("train_wav"), Mapping) else None
        train_start, train_end = phone.get("train_start_sample"), phone.get("train_end_sample")
        if "train_sample_rate" in record and (type(train_rate) is not int or train_rate <= 0 or type(train_start) is not int or type(train_end) is not int):
            errors.append(_error("audio_transform_invalid", "phone lacks train sample-axis projection", f"phones[{index}]"))
        elif "train_sample_rate" in record and (train_start != int((start * train_rate + sample_rate // 2) // sample_rate) or train_end != int((end * train_rate + sample_rate // 2) // sample_rate) or train_end <= train_start or (train_frames is not None and train_end > int(train_frames))):
            errors.append(_error("audio_transform_invalid", "train sample-axis projection differs", f"phones[{index}]"))
        source_rate = record.get("source_sample_rate")
        source_frames = record.get("audio_transform", {}).get("source_frames") if isinstance(record.get("audio_transform"), Mapping) else None
        source_start = phone.get("source_start_sample"); source_end = phone.get("source_end_sample")
        transform_recipe = record.get("audio_transform", {}).get("sample_transform", {}) if isinstance(record.get("audio_transform"), Mapping) else {}
        source_offset = int(transform_recipe.get("source_start", 0)) if isinstance(transform_recipe, Mapping) else 0
        if "source_sample_rate" in record and (type(source_rate) is not int or type(source_start) is not int or type(source_end) is not int):
            errors.append(_error("audio_transform_invalid", "phone lacks source sample-axis projection", f"phones[{index}]"))
        elif "source_sample_rate" in record and (source_start != source_offset + int((start * source_rate + sample_rate // 2) // sample_rate) or source_end != source_offset + int((end * source_rate + sample_rate // 2) // sample_rate) or source_end <= source_start or (source_frames is not None and source_end > source_offset + int(source_frames))):
            errors.append(_error("audio_transform_invalid", "source sample-axis projection differs", f"phones[{index}]"))
        previous = end if isinstance(end, int) else previous
        language = phone.get("language")
        native = phone.get("native_phone")
        if language not in {"ja", "en"} or not isinstance(native, str) or not native:
            errors.append(_error("mfa_inventory_mismatch", "native phone/language is invalid", f"phones[{index}]"))
        pid = phone.get("phone_id")
        if not isinstance(pid, str) or pid in phone_ids:
            errors.append(_error("tts_invalid", "phone_id is missing or duplicated", f"phones[{index}]"))
        else: phone_ids.add(pid)
        if not isinstance(phone.get("alias"), str) or not phone.get("alias"):
            errors.append(_error("dictionary_roundtrip_failed", "every phone must bind a locked occurrence alias", f"phones[{index}]"))
        if phone.get("raw_interval_id") is None or not phone.get("unit_id"):
            errors.append(_error("alignment_invalid", "phone lacks raw MFA interval or ownership unit", f"phones[{index}]"))
    aliases = record.get("locked_aliases")
    if not isinstance(aliases, list) or not aliases:
        errors.append(_error("dictionary_roundtrip_failed", "locked_aliases are required")); aliases = []
    alias_rows: dict[str, Mapping[str, Any]] = {}
    for index, alias in enumerate(aliases):
        if not isinstance(alias, Mapping) or not isinstance(alias.get("alias"), str) or alias.get("alias") in alias_rows:
            errors.append(_error("dictionary_roundtrip_failed", "alias rows must be unique", f"locked_aliases[{index}]")); continue
        if not alias.get("pronunciation"):
            errors.append(_error("dictionary_roundtrip_failed", "alias pronunciation is empty", f"locked_aliases[{index}]"))
        alias_rows[str(alias["alias"])] = alias
    phone_aliases = {str(phone.get("alias")) for phone in phones if isinstance(phone, Mapping) and phone.get("alias")}
    if phone_aliases != set(alias_rows):
        errors.append(_error("dictionary_roundtrip_failed", "phone aliases and locked aliases differ"))
    inventory = record.get("native_inventory")
    if not isinstance(inventory, Mapping):
        errors.append(_error("mfa_inventory_mismatch", "native_inventory is required"))
    else:
        for phone in phones:
            if phone.get("native_phone") not in set(inventory.get(phone.get("language"), [])):
                errors.append(_error("mfa_inventory_mismatch", "phone is outside declared native inventory"))
    _verify_inventory_artifact(record, [phone for phone in phones if isinstance(phone, Mapping)], root, errors)
    _verify_locked_dictionary(record, [phone for phone in phones if isinstance(phone, Mapping)], root, errors)
    raw_mfa = record.get("raw_mfa")
    raw_run_mode = isinstance(raw_mfa, Mapping) and isinstance(raw_mfa.get("runs"), list)
    raw_phone_rows = raw_mfa.get("phones", []) if isinstance(raw_mfa, Mapping) else []
    if not isinstance(raw_mfa, Mapping) or (not raw_run_mode and len(raw_phone_rows) != len(phones)):
        errors.append(_error("alignment_invalid", "raw MFA projection cardinality differs from phones"))
    elif raw_run_mode:
        if not raw_mfa.get("runs"):
            errors.append(_error("alignment_invalid", "raw MFA run ledger is empty"))
    elif not raw_mfa.get("textgrid_path"):
        errors.append(_error("alignment_invalid", "raw MFA TextGrid binding is missing"))
    else:
        raw_grid = Path(str(raw_mfa["textgrid_path"])).expanduser()
        if raw_grid.is_symlink() or not raw_grid.is_file():
            errors.append(_error("receipt_output_missing", "raw MFA TextGrid is missing", str(raw_grid)))
        else:
            raw_intervals = _textgrid_intervals(raw_grid)
            raw_phones = [item for item in raw_intervals if not item[2] in {"words", "phones", "language"}]
            if len(raw_phones) < len(phones):
                errors.append(_error("alignment_invalid", "raw MFA TextGrid phone cardinality is below TTS phones", str(raw_grid)))
    _verify_raw_runs(record, [phone for phone in phones if isinstance(phone, Mapping)], root, errors)
    reading = record.get("reading_evidence")
    selected_map = record.get("selected_readings") if isinstance(record.get("selected_readings"), Mapping) else {}
    if not isinstance(reading, Mapping):
        errors.append(_error("reading_unresolved", "reading evidence is not bound to selected reading"))
    elif selected_map:
        locks = reading.get("locks")
        if not isinstance(locks, list) or len(locks) != len(selected_map):
            errors.append(_error("reading_unresolved", "multi-token reading evidence locks are incomplete"))
        else:
            lock_keys = {(str(lock.get("uid")), str(lock.get("token_id")), str(lock.get("candidate_id"))) for lock in locks if isinstance(lock, Mapping)}
            if len(lock_keys) != len(locks): errors.append(_error("reading_unresolved", "reading locks are duplicated or malformed"))
            for lock in locks:
                if not isinstance(lock, Mapping) or selected_map.get(str(lock.get("token_id"))) != lock.get("chosen_reading", lock.get("selected_reading")):
                    errors.append(_error("reading_unresolved", "selected_readings differs from locked reading row"))
    elif not isinstance(record.get("selected_reading"), str) or not record.get("selected_reading") or reading.get("selected_reading") != record.get("selected_reading"):
        errors.append(_error("reading_unresolved", "authoritative locked reading evidence is required"))
    _verify_reading_artifact(record, root, errors)
    _verify_semantic_artifact(record, [phone for phone in phones if isinstance(phone, Mapping)], root, errors)
    partition = record.get("partition")
    if not isinstance(partition, Mapping):
        errors.append(_error("partition_not_exact", "record partition is required"))
    else:
        buckets: dict[str, list[str]] = {}
        for bucket in ("verified", "rejected", "unresolved"):
            values = partition.get(bucket)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                errors.append(_error("partition_not_exact", f"partition.{bucket} must be a list of unit IDs")); values = []
            buckets[bucket] = list(values)
        sets = {bucket: set(values) for bucket, values in buckets.items()}
        if sum(len(values) for values in sets.values()) != len(set().union(*sets.values())):
            errors.append(_error("partition_not_exact", "partition buckets overlap", uid))
        expected_values = partition.get("expected_unit_ids")
        if not isinstance(expected_values, list):
            expected_values = [word.get("unit_id") for word in record.get("words", [])
                              if isinstance(word, Mapping) and isinstance(word.get("unit_id"), str)]
            if not expected_values:
                expected_values = [unit_id for run in (record.get("raw_mfa", {}).get("runs", []) if isinstance(record.get("raw_mfa"), Mapping) else [])
                                   for unit_id in (run.get("unit_ids", []) if isinstance(run, Mapping) and isinstance(run.get("unit_ids"), list) else [])]
        expected = {str(value) for value in expected_values if isinstance(value, str) and value}
        if not expected:
            errors.append(_error("partition_not_exact", "partition expected_unit_ids or authoritative unit ledger is required", uid))
        elif set().union(*sets.values()) != expected:
            errors.append(_error("partition_not_exact", "partition buckets do not exactly cover expected unit IDs", uid))
    graph = record.get("mora_graph")
    has_japanese = any(isinstance(phone, Mapping) and phone.get("language") == "ja" for phone in phones)
    if not isinstance(graph, Mapping):
        if has_japanese:
            errors.append(_error("mora_phone_relation_unresolved", "mora_graph is missing"))
        graph = {}
    elif has_japanese and (not graph.get("moras") or not graph.get("relations")):
        errors.append(_error("mora_phone_relation_unresolved", "mora_graph is missing"))
    mora_ids = {str(row.get("mora_id")) for row in graph.get("moras", []) if isinstance(row, Mapping)}
    for index, relation in enumerate(graph.get("relations", []) if isinstance(graph.get("relations", []), list) else []):
        if not isinstance(relation, Mapping) or relation.get("phone_id") not in phone_ids or relation.get("mora_id") not in mora_ids:
            errors.append(_error("mora_phone_relation_unresolved", "mora relation references unknown node", f"mora_graph.relations[{index}]"))
    related_moras = {relation.get("mora_id") for relation in graph.get("relations", []) if isinstance(relation, Mapping)}
    related_phones = {relation.get("phone_id") for relation in graph.get("relations", []) if isinstance(relation, Mapping)}
    required_mora_phones = {str(phone.get("phone_id")) for phone in phones if isinstance(phone, Mapping) and phone.get("language") == "ja"}
    if has_japanese and (related_moras != mora_ids or related_phones != required_mora_phones):
        errors.append(_error("mora_phone_relation_unresolved", "mora graph is not bidirectionally covering phones and moras"))
    if not has_japanese and (mora_ids or related_moras or related_phones):
        errors.append(_error("mora_phone_relation_unresolved", "pure English record must not claim Japanese mora relations"))
    masks = record.get("quality_masks")
    if not isinstance(masks, Mapping):
        errors.append(_error("tts_invalid", "quality_masks is missing"))
    else:
        if masks.get("accent_predicted_known_mask") and not masks.get("accent_predicted_provenance"):
            errors.append(_error("tts_invalid", "accent prediction mask lacks provenance"))
        if masks.get("f0_known_mask") and not masks.get("f0_measured_provenance"):
            errors.append(_error("tts_invalid", "measured F0 mask lacks provenance"))
        if record.get("f0_measured") is None and masks.get("f0_known_mask"):
            errors.append(_error("tts_invalid", "unknown F0 cannot have a known mask"))
    if record.get("accent_predicted") is not None and not isinstance(record.get("accent_predicted"), Mapping):
        errors.append(_error("tts_invalid", "accent_predicted must retain provenance object"))
    if isinstance(record.get("accent_predicted"), Mapping):
        if not record["accent_predicted"].get("provenance"):
            errors.append(_error("tts_invalid", "accent_predicted provenance is missing"))
        source_kind = record["accent_predicted"].get("source_kind")
        if source_kind is not None and source_kind not in {"text_prediction", "text_frontend", "frontend_prediction"}:
            errors.append(_error("tts_invalid", "accent_predicted must identify a text prediction source"))
    if record.get("f0_measured") is not None:
        f0 = record.get("f0_measured")
        if not isinstance(f0, Mapping) or not f0.get("provenance"):
            errors.append(_error("tts_invalid", "f0_measured must retain measurement provenance"))
        elif (f0.get("source_kind") or (f0.get("provenance", {}).get("source_kind") if isinstance(f0.get("provenance"), Mapping) else None)) not in {"audio_measurement", "acoustic_measurement", "measurement"}:
            errors.append(_error("tts_invalid", "f0_measured provenance is not an audio measurement"))
        values = f0.get("values", f0.get("hz")) if isinstance(f0, Mapping) else None
        if isinstance(values, list):
            if masks.get("f0_known_mask") and not any(value is not None for value in values):
                errors.append(_error("tts_invalid", "F0 known mask has no measured values"))
            for index, value in enumerate(values):
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                    errors.append(_error("tts_invalid", "f0_measured values must be finite or null", f"f0_measured[{index}]"))
    return errors


def _quoted_value(line: str, field: str) -> str | None:
    """Read a Praat quoted value, including doubled embedded quotes."""
    match = re.match(rf"^\s*{re.escape(field)}\s*=\s*\"(.*)\"\s*$", line)
    if not match:
        return None
    value = match.group(1)
    return value.replace('""', '"')


def _textgrid_parse(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse the interval tiers we consume without trusting tier counts.

    This deliberately does not use the producer's parser.  It accepts Praat's
    doubled quote escape and rejects non-finite or reversed numeric bounds.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    tiers: dict[str, list[dict[str, Any]]] = {}
    current_name: str | None = None
    current_tier_index = 0
    pending: dict[str, Any] | None = None
    interval_index = 0
    for raw in lines:
        line = raw.strip()
        item = re.match(r"item\s*\[(\d+)\]:", line)
        if item:
            current_tier_index = int(item.group(1))
            current_name = None; pending = None; interval_index = 0
            continue
        if current_name is None:
            name = _quoted_value(line, "name")
            if name is not None:
                current_name = name; tiers.setdefault(name, [])
            continue
        interval = re.match(r"intervals\s*\[(\d+)\]:", line)
        if interval:
            interval_index = int(interval.group(1))
            pending = {"tier": current_name, "tier_index": current_tier_index,
                       "interval_index": interval_index}
            tiers[current_name].append(pending)
            continue
        if pending is None:
            continue
        for field in ("xmin", "xmax"):
            number = re.match(rf"^{field}\s*=\s*(.+?)\s*$", line)
            if number:
                try:
                    value = float(number.group(1))
                except ValueError as exc:
                    raise ValueError(f"invalid TextGrid {field}: {number.group(1)!r}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"non-finite TextGrid {field}")
                pending[field] = value
                break
        else:
            text = _quoted_value(line, "text")
            if text is not None:
                pending["text"] = text
    for tier, intervals in tiers.items():
        for row in intervals:
            if not all(field in row for field in ("xmin", "xmax", "text")):
                raise ValueError(f"incomplete TextGrid interval in tier {tier!r}")
            if row["xmax"] < row["xmin"]:
                raise ValueError(f"reversed TextGrid interval in tier {tier!r}")
    return tiers


def _textgrid_tiers(path: Path) -> set[str]:
    return set(_textgrid_parse(path))


def _textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    return [(row["xmin"], row["xmax"], row["text"])
            for rows in _textgrid_parse(path).values() for row in rows]


def _verify_receipt_files(root: Path, payload: Mapping[str, Any], errors: list[dict[str, Any]]) -> None:
    outputs = payload.get("outputs", [])
    if not isinstance(outputs, list):
        errors.append(_error("receipt_invalid", "receipt outputs must be a list")); return
    declared: set[Path] = {root / "receipt.json", root / ".ja_en_pipeline_receipt.json", root / ".ja_en_run_identity.json",
                           root / ".ja_en_pipeline_cache.json", root / ".ja_en_stage_cache.json", root / ".ja_en.lock",
                           root / "canary_gate.json", root / "reading_gate.json",
                           root / "pure_gate.json", root / "supply_chain_lock.json", root / "gold.json", root / "alias_map.jsonl"}
    for index, row in enumerate(outputs):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            errors.append(_error("receipt_invalid", "receipt output path is missing", f"outputs[{index}]")); continue
        candidate = Path(row["path"]).expanduser()
        if not candidate.is_absolute(): candidate = root / candidate
        candidate = candidate.absolute(); declared.add(candidate)
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(_error("receipt_invalid", "receipt output escapes workspace", str(candidate))); continue
        if candidate.is_symlink():
            errors.append(_error("receipt_output_symlink", "receipt output is symlinked", str(candidate))); continue
        if row.get("exists") is not True or not candidate.is_file():
            errors.append(_error("receipt_output_missing", "receipt output is missing", str(candidate))); continue
        if row.get("size") is None or row.get("sha256") is None:
            errors.append(_error("receipt_invalid", "receipt output size and sha256 are mandatory", str(candidate))); continue
        if int(row["size"]) != candidate.stat().st_size:
            errors.append(_error("receipt_hash_mismatch", "receipt output size differs", str(candidate)))
        if row["sha256"] != sha256_file(candidate):
            errors.append(_error("receipt_hash_mismatch", "receipt output hash differs", str(candidate)))
    inputs = payload.get("inputs", {})
    artifacts = inputs.get("artifacts", []) if isinstance(inputs, Mapping) else []
    if not isinstance(artifacts, list):
        errors.append(_error("receipt_invalid", "receipt input artifacts must be a list")); artifacts = []
    for index, row in enumerate(artifacts):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            errors.append(_error("receipt_invalid", "receipt input path is missing", f"inputs.artifacts[{index}]")); continue
        candidate = Path(row["path"]).expanduser(); candidate = candidate if candidate.is_absolute() else root / candidate
        candidate = candidate.absolute()
        if candidate.is_symlink() or not candidate.is_file():
            errors.append(_error("receipt_output_missing", "receipt input artifact is missing", str(candidate))); continue
        if row.get("size") is None or row.get("sha256") is None or int(row["size"]) != candidate.stat().st_size or row["sha256"] != sha256_file(candidate):
            errors.append(_error("receipt_hash_mismatch", "receipt input artifact hash/size differs", str(candidate)))
    try:
        for candidate in root.rglob("*"):
            if candidate.is_file() and not candidate.is_symlink() and "stages" not in candidate.relative_to(root).parts and candidate not in declared:
                errors.append(_error("resume_extra_file", "unlisted output file", str(candidate)))
    except OSError as exc:
        errors.append(_error("receipt_invalid", str(exc), str(root)))


def _verify_manifest_partition(root: Path, payload: Mapping[str, Any] | None,
                               rows: Sequence[Mapping[str, Any]], errors: list[dict[str, Any]]) -> None:
    """Compare exported UIDs and optional unit/run axes to the frozen manifest."""
    if not isinstance(payload, Mapping):
        return
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), Mapping) else {}
    value = inputs.get("manifest") or inputs.get("input_manifest") or payload.get("manifest")
    if value is None and isinstance(inputs.get("artifacts"), list):
        for artifact in inputs["artifacts"]:
            if isinstance(artifact, Mapping) and (str(artifact.get("kind", "")).lower() in {"manifest", "input_manifest"} or Path(str(artifact.get("path", ""))).name.endswith(("manifest.jsonl", "manifest.json"))):
                value = artifact.get("path"); break
    if value is None and inputs.get("artifacts"):
        errors.append(_error("partition_not_exact", "run receipt does not bind an input manifest"))
        return
    path = _declared_path(value, root)
    if path is None:
        return
    if path.is_symlink() or not path.is_file():
        errors.append(_error("partition_not_exact", "declared input manifest is missing", str(path))); return
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            expected = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
            expected = document.get("items", document) if isinstance(document, Mapping) else document
        expected_uids = {str(item.get("uid", item.get("id"))) for item in expected if isinstance(item, Mapping)}
        actual_uids: set[str] = set()
        for row in rows:
            partition = row.get("partition") if isinstance(row, Mapping) else None
            if isinstance(partition, Mapping):
                buckets = [set(str(value) for value in partition.get(bucket, [])) for bucket in ("verified", "rejected", "unresolved")]
                if sum(len(bucket) for bucket in buckets) != len(set().union(*buckets)):
                    errors.append(_error("partition_not_exact", "record partition buckets overlap", str(row.get("uid"))))
                actual_uids.update(str(value) for bucket in ("verified", "rejected", "unresolved") for value in partition.get(bucket, []))
            else:
                actual_uids.add(str(row.get("uid")))
        if expected_uids != actual_uids:
            errors.append(_error("partition_not_exact", "manifest UID partition differs from TTS outputs", str(path)))
        expected_units = {str(item.get("unit_id")) for item in expected if isinstance(item, Mapping) and item.get("unit_id") is not None}
        if not expected_units:
            alias_path = root / "stages" / "semantic" / "alias_map.jsonl"
            if alias_path.is_file() and not alias_path.is_symlink():
                alias_rows, _ = _read_jsonl(alias_path)
                expected_units = {str(item.get("token_id", item.get("unit_id"))) for item in alias_rows if item.get("token_id", item.get("unit_id")) is not None}
        actual_units: set[str] = set()
        for row in rows:
            partition = row.get("partition") if isinstance(row, Mapping) else None
            if isinstance(partition, Mapping):
                actual_units.update(str(value) for bucket in ("verified", "rejected", "unresolved")
                                     for value in partition.get(bucket, []) if isinstance(value, str))
            else:
                actual_units.update(str(word.get("unit_id")) for word in row.get("words", [])
                                    if isinstance(word, Mapping) and word.get("unit_id") is not None)
        if expected_units and expected_units != actual_units:
            errors.append(_error("partition_not_exact", "manifest unit partition differs from TTS ownership", str(path)))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        errors.append(_error("partition_not_exact", str(exc), str(path)))


def _external_gate_inputs(root: Path, gate_payload: Mapping[str, Any], gate_name: str,
                          errors: list[dict[str, Any]], declared_paths: set[Path] | None = None) -> tuple[Any, list[Mapping[str, Any]]]:
    """Load gate gold/results from declared files, never from a PASS payload."""
    stem = gate_name.removesuffix("_gate.json")
    value = gate_payload.get("gold_path") or gate_payload.get("gold_file")
    if value is None:
        for candidate in (root / f"{stem}_gold.json", root / "gold.json"):
            if candidate.is_file() and not candidate.is_symlink():
                value = candidate; break
    gold_path = _declared_path(value, root)
    gold: Any = None
    if gold_path is None or gold_path.is_symlink() or not gold_path.is_file():
        errors.append(_error("publish_blocked", "gate is missing an external gold artifact", str(gold_path) if gold_path else str(root / f"{stem}_gold.json")))
    else:
        try:
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            expected_hash = gate_payload.get("gold_sha256")
            if not isinstance(expected_hash, str) or expected_hash != sha256_file(gold_path):
                errors.append(_error("publish_blocked", "gate gold digest is missing or differs", str(gold_path)))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(_error("publish_blocked", str(exc), str(gold_path)))
    result_value = gate_payload.get("results_path")
    result_path = _declared_path(result_value, root)
    if result_path is not None:
        try:
            if result_path.is_symlink() or not result_path.is_file():
                raise OSError("gate result artifact is missing or symlinked")
            try:
                result_path.relative_to(root)
            except ValueError as exc:
                raise OSError("gate result artifact escapes workspace") from exc
            if declared_paths is not None and result_path not in declared_paths:
                errors.append(_error("publish_blocked", "gate results artifact is not bound by the run receipt", str(result_path)))
            expected_result_hash = gate_payload.get("results_sha256")
            if not isinstance(expected_result_hash, str) or expected_result_hash != sha256_file(result_path):
                errors.append(_error("publish_blocked", "gate results digest is missing or differs", str(result_path)))
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            result_rows = result_payload.get("rows", result_payload) if isinstance(result_payload, Mapping) else result_payload
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(_error("publish_blocked", str(exc), str(result_path))); result_rows = []
    else:
        errors.append(_error("publish_blocked", "gate must bind actual result rows by results_path"))
        result_rows = []
    if not isinstance(result_rows, list) or any(not isinstance(row, Mapping) for row in result_rows):
        errors.append(_error("publish_blocked", "gate result rows are missing or malformed")); result_rows = []
    return gold, list(result_rows)


def verify_workspace(workspace: str | Path) -> dict[str, Any]:
    """Verify a fresh output directory and return a machine-readable report."""
    root = Path(workspace).expanduser().absolute(); errors: list[dict[str, Any]] = []
    if not root.is_dir() or root.is_symlink():
        return {"ok": False, "status": "REJECTED", "checked": 0, "errors": [_error("manifest_path_invalid", "workspace is not a regular directory", str(root))]}
    for path in root.rglob("*"):
        if path.is_symlink(): errors.append(_error("manifest_symlink", "symlink in verifier scope", str(path)))
    jsonl_paths = sorted(root.rglob("tts_training_records.jsonl"))
    if not jsonl_paths:
        errors.append(_error("receipt_missing", "tts_training_records.jsonl is missing", str(root)))
    elif len(jsonl_paths) != 1:
        errors.append(_error("partition_not_exact", "training JSONL output set must contain exactly one file", str(root)))
    rows: list[dict[str, Any]] = []
    for path in jsonl_paths:
        current, parse_errors = _read_jsonl(path); rows.extend(current); errors.extend(parse_errors)
    if not rows:
        errors.append(_error("tts_invalid", "training JSONL contains no records", str(root)))
    seen_uid: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("uid") in seen_uid: errors.append(_error("schema_invalid", "duplicate uid", f"row[{index}]"))
        seen_uid.add(row.get("uid"))
        errors.extend(verify_training_record(row, root=root))
        # Legacy offline fixtures can still be inspected, but a production
        # publication needs the upstream evidence references below.  These are
        # release blockers until W3's stage-input bridge supplies them.
        raw_assets = row.get("raw_mfa", {}).get("model_assets", {}) if isinstance(row.get("raw_mfa"), Mapping) else {}
        if not (row.get("native_inventory_path") or row.get("inventory_path") or
                (isinstance(row.get("native_inventory"), Mapping) and row["native_inventory"].get("path")) or
                (isinstance(raw_assets, Mapping) and any("acoustic_model" in str(key) for key in raw_assets))):
            errors.append(_error("publish_blocked", "native acoustic inventory archive binding is missing", f"row[{index}]"))
        raw_binding = row.get("raw_mfa")
        if not isinstance(raw_binding, Mapping) or not isinstance(raw_binding.get("runs"), list) or not raw_binding.get("runs"):
            errors.append(_error("publish_blocked", "per-run raw MFA artifact bindings are missing", f"row[{index}]"))
        reading_binding = row.get("reading_evidence")
        if not isinstance(reading_binding, Mapping) or not all(
                reading_binding.get(key) or (isinstance(reading_binding.get("sources"), Mapping) and reading_binding["sources"].get(key))
                for key in ("locked_readings_path", "reconstruction_path", "analysis_path", "semantic_alias_map_path")):
            errors.append(_error("publish_blocked", "external reading analysis binding is missing", f"row[{index}]"))
        if "train_sample_rate" not in row or any(not isinstance(phone, Mapping) or "train_start_sample" not in phone or "train_end_sample" not in phone for phone in row.get("phones", [])):
            errors.append(_error("publish_blocked", "source-to-training sample-axis projections are missing", f"row[{index}]"))
        if "source_sample_rate" not in row or any(not isinstance(phone, Mapping) or "source_start_sample" not in phone or "source_end_sample" not in phone for phone in row.get("phones", [])):
            errors.append(_error("publish_blocked", "source sample-axis projections are missing", f"row[{index}]"))
        candidate_grids = []
        explicit_grid = row.get("textgrid_path")
        if explicit_grid:
            path = Path(str(explicit_grid)).expanduser(); candidate_grids = [path if path.is_absolute() else root / path]
        else:
            candidate_grids = sorted(root.rglob(f"{row.get('uid')}.TextGrid"))
        if len(candidate_grids) != 1 or not candidate_grids[0].is_file() or candidate_grids[0].is_symlink(): errors.append(_error("receipt_output_missing", "exactly one TextGrid is required", str(candidate_grids[0] if candidate_grids else root)))
        else:
            grid = candidate_grids[0]
            try:
                parsed_tiers = _textgrid_parse(grid)
                if not {"words", "phones", "language"}.issubset(parsed_tiers):
                    errors.append(_error("tts_invalid", "TextGrid lacks required tier", str(grid)))
                rate = int(row.get("sample_rate", 16000))
                phone_intervals = parsed_tiers.get("phones", [])
                expected_phones = row.get("phones", [])
                if len(phone_intervals) != len(expected_phones):
                    errors.append(_error("tts_invalid", "TextGrid phone cardinality differs from JSONL", str(grid)))
                for phone, actual in zip(expected_phones, phone_intervals):
                    expected_label = f"{phone.get('language')}:{phone.get('native_phone')}"
                    if (actual.get("text") != expected_label or
                            round(actual["xmin"] * rate) != phone.get("start_sample") or
                            round(actual["xmax"] * rate) != phone.get("end_sample")):
                        errors.append(_error("tts_invalid", "TextGrid native phone or integer timing differs from JSONL", str(grid)))
                word_intervals = parsed_tiers.get("words", [])
                language_intervals = parsed_tiers.get("language", [])
                expected_words = row.get("words", [])
                if len(word_intervals) != len(expected_words) or len(language_intervals) != len(expected_words):
                    errors.append(_error("tts_invalid", "TextGrid word/language cardinality differs from JSONL", str(grid)))
                for word, actual_word, actual_language in zip(expected_words, word_intervals, language_intervals):
                    start = round(actual_word["xmin"] * rate); end = round(actual_word["xmax"] * rate)
                    if (start, end, str(actual_word.get("text", ""))) != (word.get("start_sample"), word.get("end_sample"), str(word.get("text", ""))):
                        errors.append(_error("tts_invalid", "TextGrid word timing/text differs from JSONL", str(grid)))
                    if (round(actual_language["xmin"] * rate), round(actual_language["xmax"] * rate), str(actual_language.get("text", ""))) != (word.get("start_sample"), word.get("end_sample"), str(word.get("language", ""))):
                        errors.append(_error("tts_invalid", "TextGrid language ownership differs from JSONL", str(grid)))
            except (OSError, UnicodeError, ValueError) as exc: errors.append(_error("tts_invalid", str(exc), str(grid)))
    declared_alias_values = [row.get("semantic_alias_map_path") or row.get("alias_map_path") for row in rows]
    declared_alias_values = [value for value in declared_alias_values if value]
    alias_map_candidates = []
    if declared_alias_values:
        alias_map_candidates = [_declared_path(declared_alias_values[0], root)]
    else:
        semantic_alias = root / "stages" / "semantic" / "alias_map.jsonl"
        # Root alias_map is retained solely for old offline fixtures.  A real
        # run must bind the semantic namespace explicitly in its records.
        alias_map_candidates = [semantic_alias if semantic_alias.is_file() else root / "alias_map.jsonl"]
    alias_map_candidates = [path for path in alias_map_candidates if path is not None and path.is_file() and not path.is_symlink()]
    if alias_map_candidates:
        alias_path = alias_map_candidates[0]
        declared_alias_hashes = {str(row.get("semantic_alias_map_sha256")) for row in rows if row.get("semantic_alias_map_sha256")}
        if declared_alias_hashes and (len(declared_alias_hashes) != 1 or next(iter(declared_alias_hashes)) != sha256_file(alias_path)):
            errors.append(_error("receipt_hash_mismatch", "semantic alias map digest differs", str(alias_path)))
        alias_rows, alias_parse_errors = _read_jsonl(alias_path); errors.extend(alias_parse_errors)
        alias_seen: set[str] = set()
        map_pron: dict[str, list[str]] = {}
        for index, alias in enumerate(alias_rows):
            name = alias.get("alias")
            if not isinstance(name, str) or name in alias_seen or not isinstance(alias.get("pronunciation"), list) or not alias["pronunciation"]:
                errors.append(_error("dictionary_roundtrip_failed", "alias map rows must be unique and non-empty", f"{alias_path}:{index + 1}")); continue
            alias_seen.add(name); map_pron[name] = [str(phone) for phone in alias["pronunciation"]]
            if alias.get("uid") is not None and str(alias.get("uid")) not in {str(row.get("uid")) for row in rows}:
                errors.append(_error("dictionary_roundtrip_failed", "alias map UID is outside this manifest", f"{alias_path}:{index + 1}"))
        expected_aliases: set[str] = set()
        expected_pron: dict[str, list[str]] = {}
        expected_alias_uids: dict[str, set[str]] = {}
        for row in rows:
            for phone in row.get("phones", []):
                if isinstance(phone, Mapping) and phone.get("alias"):
                    name = str(phone["alias"]); expected_aliases.add(name); expected_pron.setdefault(name, []).append(str(phone.get("native_phone"))); expected_alias_uids.setdefault(name, set()).add(str(row.get("uid")))
        if alias_seen != expected_aliases:
            errors.append(_error("dictionary_roundtrip_failed", "alias map coverage differs from training phones", str(alias_path)))
        for name in expected_aliases:
            if map_pron.get(name) != expected_pron.get(name):
                errors.append(_error("dictionary_roundtrip_failed", "alias map pronunciation differs from raw phone sequence", str(alias_path)))
        for alias in alias_rows:
            if alias.get("alias") in expected_alias_uids and alias.get("uid") is not None and str(alias.get("uid")) not in expected_alias_uids[str(alias["alias"])]:
                errors.append(_error("dictionary_roundtrip_failed", "alias map UID does not bind the owning record", str(alias_path)))
    canonical_receipt = root / "receipt.json"
    legacy_receipt = root / ".ja_en_pipeline_receipt.json"
    if legacy_receipt.is_file() or legacy_receipt.is_symlink():
        errors.append(_error("receipt_invalid", "legacy root receipt is not the canonical receipt.json", str(legacy_receipt)))
    receipt = canonical_receipt
    receipt_payload: Mapping[str, Any] | None = None
    if not receipt.is_file() or receipt.is_symlink():
        errors.append(_error("receipt_missing", "mandatory run receipt is missing", str(receipt)))
    else:
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            receipt_payload = payload if isinstance(payload, Mapping) else None
            if isinstance(payload, Mapping):
                _verify_receipt_files(root, payload, errors)
                _verify_manifest_partition(root, payload, rows, errors)
            if not isinstance(payload, Mapping) or payload.get("schema") != "ja-pipeline-receipt-v1":
                errors.append(_error("receipt_invalid", "run receipt schema is not ja-pipeline-receipt-v1", str(receipt)))
            payload_status = payload.get("status") if isinstance(payload, Mapping) else None
            if payload_status not in {"PENDING", "RUNNING", "PARTIAL", "COMPLETE", "REJECTED", "BLOCKED"}:
                errors.append(_error("receipt_invalid", "run receipt has an unknown status", str(receipt)))
            gate = root / "canary_gate.json"
            mixed_required = any(isinstance(row, Mapping) and {phone.get("language") for phone in row.get("phones", []) if isinstance(phone, Mapping)} == {"ja", "en"} for row in rows)
            if not gate.is_file() and mixed_required:
                errors.append(_error("publish_blocked", "mixed output requires an independent canary gate", str(gate)))
            elif gate.is_file():
                try:
                    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
                    try:
                        from .ja_canary_gate import evaluate_canary_gate
                    except ImportError:  # direct script execution
                        from ja_canary_gate import evaluate_canary_gate
                    declared = {Path(str(item.get("path"))).expanduser().absolute() for item in payload.get("outputs", []) if isinstance(item, Mapping) and item.get("path")}
                    bound_gold, result_rows = _external_gate_inputs(root, gate_payload, "canary_gate.json", errors, declared)
                    recomputed = evaluate_canary_gate(result_rows, gold=bound_gold, target_id=gate_payload.get("target_id"), proof=bool(gate_payload.get("proof", False)))
                    if recomputed.get("status") != "PASS" or gate_payload.get("status") != recomputed.get("status"):
                        errors.append(_error("publish_blocked", "canary status is not an independently recomputed PASS", str(gate)))
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    errors.append(_error("receipt_invalid", str(exc), str(gate)))
            if payload.get("status") == "COMPLETE" and errors:
                errors.append(_error("publish_blocked", "COMPLETE status conflicts with independent verification"))
        except (OSError, json.JSONDecodeError) as exc: errors.append(_error("receipt_invalid", str(exc), str(receipt)))
    try:
        from .ja_canary_gate import evaluate_pure_gate, evaluate_reading_gate
    except ImportError:  # direct script execution
        from ja_canary_gate import evaluate_pure_gate, evaluate_reading_gate
    for filename, calculator in (("reading_gate.json", evaluate_reading_gate), ("pure_gate.json", evaluate_pure_gate)):
        gate_path = root / filename
        if not gate_path.is_file() or gate_path.is_symlink():
            errors.append(_error("publish_blocked", f"mandatory {filename} is missing", str(gate_path))); continue
        try:
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            declared = {Path(str(item.get("path"))).expanduser().absolute() for item in (receipt_payload or {}).get("outputs", []) if isinstance(item, Mapping) and item.get("path")}
            bound_gold, result_rows = _external_gate_inputs(root, gate_payload, filename, errors, declared)
            recomputed = calculator(result_rows, gold=bound_gold, target_id=gate_payload.get("target_id"), proof=bool(gate_payload.get("proof", False)))
            if recomputed.get("status") != "PASS" or gate_payload.get("status") != recomputed.get("status"):
                errors.append(_error("publish_blocked", f"{filename} is not an independently recomputed PASS", str(gate_path)))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(_error("receipt_invalid", str(exc), str(gate_path)))
    stage_root = root / "stages"
    if stage_root.is_dir():
        stage_dirs = [path for path in stage_root.iterdir() if path.is_dir() and not path.is_symlink()]
        if not stage_dirs:
            errors.append(_error("receipt_missing", "stage receipt DAG is empty", str(stage_root)))
        expected_stage_names = set()
        if isinstance(receipt_payload, Mapping) and isinstance(receipt_payload.get("inputs"), Mapping) and isinstance(receipt_payload["inputs"].get("stages"), list):
            expected_stage_names = {str(name) for name in receipt_payload["inputs"]["stages"]}
            actual_stage_names = {path.name for path in stage_dirs}
            if actual_stage_names != expected_stage_names:
                errors.append(_error("partition_not_exact", "stage receipt set differs from declared DAG", str(stage_root)))
        actual_stage_names = {path.name for path in stage_dirs}
        missing_production = set(PRODUCTION_STAGES) - actual_stage_names
        for missing in sorted(missing_production):
            errors.append(_error("publish_blocked", f"required production stage receipt is missing: {missing}", str(stage_root / missing)))
        for stage_dir in stage_dirs:
            stage_receipt = stage_dir / "receipt.json"
            if not stage_receipt.is_file() or stage_receipt.is_symlink():
                errors.append(_error("receipt_missing", "stage receipt is missing", str(stage_receipt))); continue
            try:
                stage_payload = json.loads(stage_receipt.read_text(encoding="utf-8"))
                if not isinstance(stage_payload, Mapping) or stage_payload.get("schema") != "ja-pipeline-receipt-v1":
                    errors.append(_error("receipt_invalid", "stage receipt schema is invalid", str(stage_receipt)))
                if isinstance(stage_payload, Mapping):
                    if stage_payload.get("stage") != stage_dir.name:
                        errors.append(_error("receipt_invalid", "stage receipt name does not match its DAG namespace", str(stage_receipt)))
                    if stage_payload.get("status") not in {"PENDING", "RUNNING", "PARTIAL", "COMPLETE", "REJECTED", "BLOCKED"}:
                        errors.append(_error("receipt_invalid", "stage receipt status is invalid", str(stage_receipt)))
                    elif stage_dir.name != "verify" and stage_payload.get("status") != "COMPLETE":
                        errors.append(_error("publish_blocked", "required production stage is not COMPLETE", str(stage_receipt)))
                    declared_stage = {stage_receipt.absolute()}
                    for output in stage_payload.get("outputs", []):
                        if isinstance(output, Mapping) and output.get("path"):
                            output_path = Path(str(output["path"])).expanduser(); output_path = output_path if output_path.is_absolute() else root / output_path
                            output_path = output_path.absolute()
                            try:
                                output_path.relative_to(root)
                            except ValueError:
                                errors.append(_error("receipt_invalid", "stage output escapes workspace", str(output_path))); continue
                            declared_stage.add(output_path.absolute())
                            if output.get("exists") is not True or not output_path.is_file() or output_path.is_symlink():
                                errors.append(_error("receipt_output_missing", "stage output is missing", str(output_path)))
                            elif output.get("size") is None or output.get("sha256") is None:
                                errors.append(_error("receipt_invalid", "stage output size and sha256 are mandatory", str(output_path)))
                            elif int(output["size"]) != output_path.stat().st_size or output["sha256"] != sha256_file(output_path):
                                errors.append(_error("receipt_hash_mismatch", "stage output hash differs", str(output_path)))
                    stage_inputs = stage_payload.get("inputs", {})
                    input_artifacts = stage_inputs.get("artifacts", []) if isinstance(stage_inputs, Mapping) else []
                    if not isinstance(input_artifacts, list):
                        errors.append(_error("receipt_invalid", "stage input artifacts must be a list", str(stage_receipt)))
                        input_artifacts = []
                    for artifact in input_artifacts:
                        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
                            errors.append(_error("receipt_invalid", "stage input path is missing", str(stage_receipt))); continue
                        input_path = _declared_path(artifact["path"], root)
                        if input_path is None or input_path.is_symlink() or not input_path.is_file():
                            errors.append(_error("receipt_output_missing", "stage input artifact is missing", str(input_path))); continue
                        if artifact.get("size") is None or artifact.get("sha256") is None or int(artifact["size"]) != input_path.stat().st_size or artifact["sha256"] != sha256_file(input_path):
                            errors.append(_error("receipt_hash_mismatch", "stage input artifact hash differs", str(input_path)))
                    for actual in stage_dir.rglob("*"):
                        if actual.is_file() and not actual.is_symlink() and actual.absolute() not in declared_stage:
                            errors.append(_error("resume_extra_file", "unlisted stage output", str(actual)))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(_error("receipt_invalid", str(exc), str(stage_receipt)))
    else:
        errors.append(_error("receipt_missing", "stage receipt DAG is missing", str(stage_root)))
    lock = root / "supply_chain_lock.json"
    # The active run configuration is authoritative for lock selection.  A
    # convenient root filename is only a fallback for offline fixtures.
    configured_lock: Any = None
    if isinstance(receipt_payload, Mapping) and isinstance(receipt_payload.get("inputs"), Mapping):
        configured_lock = (receipt_payload["inputs"].get("supply_chain_lock") or
                           receipt_payload["inputs"].get("lock_path"))
        config_value = receipt_payload["inputs"].get("config")
        if configured_lock is None and isinstance(config_value, Mapping):
            configured_lock = config_value.get("supply_chain_lock") or config_value.get("supply_chain_lock_path")
    identity = root / ".ja_en_run_identity.json"
    if configured_lock is None and identity.is_file() and not identity.is_symlink():
        try:
            identity_payload = json.loads(identity.read_text(encoding="utf-8"))
            identity_config = identity_payload.get("identity", {}).get("config", {}) if isinstance(identity_payload.get("identity"), Mapping) else {}
            configured_lock = (identity_config.get("supply_chain_lock") or identity_payload.get("supply_chain_lock") or identity_payload.get("lock_path"))
        except (OSError, json.JSONDecodeError):
            errors.append(_error("receipt_invalid", "run identity cache is not valid JSON", str(identity)))
    configured_path = _declared_path(configured_lock, root)
    if configured_path is not None:
        lock = configured_path
    if (configured_path is None and (not lock.is_file() or lock.is_symlink())) and isinstance(receipt_payload, Mapping):
        for artifact in receipt_payload.get("inputs", {}).get("artifacts", []) if isinstance(receipt_payload.get("inputs"), Mapping) else []:
            candidate = Path(str(artifact.get("path", ""))).expanduser()
            if candidate.name == "supply_chain_lock.json" and candidate.is_file() and not candidate.is_symlink(): lock = candidate; break
    if not lock.is_file() or lock.is_symlink():
        errors.append(_error("supply_chain_invalid", "mandatory supply-chain lock is missing", str(lock)))
    else:
        try:
            lock_payload = json.loads(lock.read_text(encoding="utf-8"))
            try:
                from .verify_ja_supply_chain import load_lock, verify_lock
            except ImportError:  # direct script execution
                from verify_ja_supply_chain import load_lock, verify_lock
            active_config = None
            if isinstance(receipt_payload, Mapping) and isinstance(receipt_payload.get("inputs"), Mapping):
                candidate_config = receipt_payload["inputs"].get("config")
                active_config = candidate_config if isinstance(candidate_config, Mapping) else None
            if active_config is None and identity.is_file() and not identity.is_symlink():
                try:
                    identity_payload = json.loads(identity.read_text(encoding="utf-8"))
                    candidate_config = identity_payload.get("identity", {}).get("config", {})
                    active_config = candidate_config if isinstance(candidate_config, Mapping) else None
                except (OSError, json.JSONDecodeError):
                    pass
            verify_lock(load_lock(lock), strict=True, base_dir=lock.parent, active_config=active_config)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(_error("supply_chain_invalid", str(exc), str(lock)))
        except Exception as exc:
            errors.append(_error("supply_chain_license_unknown", str(exc), str(lock)))
    gold = root / "gold.json"
    if (not gold.is_file() or gold.is_symlink()) and isinstance(receipt_payload, Mapping):
        for artifact in receipt_payload.get("inputs", {}).get("artifacts", []) if isinstance(receipt_payload.get("inputs"), Mapping) else []:
            candidate = Path(str(artifact.get("path", ""))).expanduser()
            if candidate.name == "gold.json" and candidate.is_file() and not candidate.is_symlink(): gold = candidate; break
    if not gold.is_file() or gold.is_symlink():
        errors.append(_error("publish_blocked", "human gold binding is missing", str(gold)))
    alias_map = root / "stages" / "semantic" / "alias_map.jsonl"
    if not alias_map.is_file() and rows:
        declared_alias = rows[0].get("semantic_alias_map_path") or rows[0].get("alias_map_path")
        declared_path = _declared_path(declared_alias, root)
        if declared_path is not None: alias_map = declared_path
    if not alias_map.is_file() and (root / "alias_map.jsonl").is_file():
        alias_map = root / "alias_map.jsonl"
    if (not alias_map.is_file() or alias_map.is_symlink()) and isinstance(receipt_payload, Mapping):
        for artifact in receipt_payload.get("inputs", {}).get("artifacts", []) if isinstance(receipt_payload.get("inputs"), Mapping) else []:
            candidate = Path(str(artifact.get("path", ""))).expanduser()
            if candidate.name == "alias_map.jsonl" and candidate.is_file() and not candidate.is_symlink(): alias_map = candidate; break
    if (not alias_map.is_file() or alias_map.is_symlink()) and rows:
        errors.append(_error("dictionary_roundtrip_failed", "locked alias map artifact is missing", str(alias_map)))
    release_codes = {"publish_blocked", "supply_chain_invalid", "supply_chain_license_unknown", "model_asset_missing", "dictionary_asset_missing", "julius_diagnostic_unavailable"}
    integrity_errors = [error for error in errors if error.get("code") not in release_codes]
    release_blockers = [error for error in errors if error.get("code") in release_codes]
    release_ready = not errors
    status = "COMPLETE" if release_ready else ("BLOCKED" if not integrity_errors else "REJECTED")
    return {"ok": not integrity_errors, "integrity_ok": not integrity_errors, "release_ready": release_ready,
            "status": status, "checked": len(rows), "errors": errors, "integrity_errors": integrity_errors,
            "release_blockers": release_blockers, "uids": sorted(seen_uid), "production_write_back": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--workspace", required=True, type=Path); parser.add_argument("--integrity-only", action="store_true", help="return zero for data-integrity validation even when release gates are blocked")
    args = parser.parse_args(argv); report = verify_workspace(args.workspace); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if (report["ok"] if args.integrity_only else report["release_ready"]) else 1


verify_record = verify_training_record
verify_run = verify_workspace
independent_verify = verify_workspace


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["independent_verify", "verify_record", "verify_run", "verify_training_record", "verify_workspace"]
