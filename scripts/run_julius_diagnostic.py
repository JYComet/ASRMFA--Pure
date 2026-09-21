"""Strict, optional Julius/JATTS diagnostic adapter.

Julius is a comparison backend only.  It consumes a frozen alignment WAV and
a locked semantic representation and writes below its diagnostic stage.  It
never writes MFA, TTS, reading, or cache artifacts.  A configured JATTS
workflow is required; this module never synthesizes a raw ``-palign`` call.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.machinery
import json
import math
import re
import shlex
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .ja_en_schema import StageResult, atomic_write_json, make_receipt, sha256_file, stable_digest
except ImportError:  # pragma: no cover
    from ja_en_schema import StageResult, atomic_write_json, make_receipt, sha256_file, stable_digest


JULIUS4SEG_COMMIT = "e14beae2940fd5a6ac5a9d2afc249eac6fac4a50"
JULIUS4SEG_SOURCE_SHA256 = "066bd90c23c4fb8683fb9e7ddcf66fc6156b12d452570034c06395b301ca74a3"
JATTS_COMMIT = "a5a8cd0b9a92caa065b1b6cc4cf8e805e56e4c61"
_ASSET_NAMES = ("binary", "model", "dictionary", "converter", "jatts")
_UID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PHONE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")


def _regular_file(value: str | Path | None, name: str) -> Path:
    if not value:
        raise ValueError(f"{name} asset is required")
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} asset must be a regular non-symlink file: {path}")
    return path


def _safe_root(path: str | Path) -> Path:
    root_input = Path(path).expanduser()
    root_absolute = root_input.absolute()
    current = Path(root_absolute.anchor or "/")
    for part in root_absolute.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"diagnostic stage parent is symlinked: {current}")
    root = root_absolute.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"diagnostic stage is symlinked: {root}")
    return root


def _inside(path: str | Path, root: Path) -> Path:
    raw = Path(path).expanduser()
    candidate = root / raw if not raw.is_absolute() else raw
    candidate = candidate.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"diagnostic output escapes stage directory: {candidate}") from exc
    current = root_resolved
    target_parent = candidate.parent
    try:
        relative_parts = target_parent.relative_to(root_resolved).parts
    except ValueError:
        relative_parts = ()
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"diagnostic output contains symlink: {current}")
    return candidate


def _audit_tree(root: Path) -> None:
    real_root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"workflow emitted symlink: {path}")
        try:
            path.resolve().relative_to(real_root)
        except ValueError as exc:
            raise ValueError(f"workflow output escapes stage directory: {path}") from exc


def validate_pcm16(path: str | Path, *, expected_rate: int = 16000) -> dict[str, Any]:
    candidate = _regular_file(path, "audio")
    try:
        with wave.open(str(candidate), "rb") as handle:
            info = {
                "path": str(candidate),
                "sample_rate": handle.getframerate(),
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "frames": handle.getnframes(),
                "compression": handle.getcomptype(),
            }
    except (OSError, wave.Error) as exc:
        raise ValueError(f"invalid WAV: {candidate}") from exc
    if info["sample_rate"] != expected_rate or info["channels"] != 1 or info["sample_width"] != 2 or info["compression"] != "NONE":
        raise ValueError("Julius requires 16000 Hz mono PCM16 WAV")
    info["sha256"] = sha256_file(candidate)
    return info


def _ticks_to_seconds(value: str, timebase: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("lab time must be finite")
    base = timebase.lower()
    if base in {"100ns", "julius", "10mhz"}:
        return number / 10_000_000.0
    if base in {"ms", "milliseconds"}:
        return number / 1000.0
    if base in {"s", "sec", "seconds"}:
        return number
    raise ValueError(f"unsupported lab timebase: {timebase}")


def parse_lab(content_or_path: str | Path, *, sample_rate: int = 16000, timebase: str = "seconds") -> list[dict[str, Any]]:
    """Parse JATTS ``.lab`` rows into integer sample spans."""

    if isinstance(content_or_path, Path):
        text = content_or_path.read_text(encoding="utf-8")
    elif isinstance(content_or_path, str) and "\n" not in content_or_path:
        try:
            candidate = Path(content_or_path)
            text = candidate.read_text(encoding="utf-8") if candidate.is_file() else content_or_path
        except OSError:
            text = content_or_path
    else:
        text = str(content_or_path)
    result: list[dict[str, Any]] = []
    previous_end = 0
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        fields = re.split(r"\s+", line)
        if len(fields) != 3:
            raise ValueError(f"invalid .lab row {line_number}")
        try:
            start_s = _ticks_to_seconds(fields[0], timebase)
            end_s = _ticks_to_seconds(fields[1], timebase)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid .lab time at row {line_number}") from exc
        phone = fields[2]
        if not phone or not _PHONE_RE.fullmatch(phone):
            raise ValueError(f"invalid .lab phone at row {line_number}")
        try:
            start = int(round(start_s * sample_rate))
            end = int(round(end_s * sample_rate))
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"invalid .lab time at row {line_number}") from exc
        if start < 0 or start < previous_end or end <= start:
            raise ValueError(f"non-monotonic .lab row {line_number}")
        result.append({"interval_id": len(result) + 1, "phone": phone, "start_s": start_s, "end_s": end_s, "start_sample": start, "end_sample": end, "duration_samples": end - start})
        previous_end = end
    return result


def _load_conv2julius(source: Path) -> Callable[[str], Any]:
    module_name = f"_julius_converter_{hashlib.sha256(str(source).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        loader = importlib.machinery.SourceFileLoader(module_name, str(source))
        spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:
        raise ValueError(f"unable to load pinned converter source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "conv2julius", None)
    if not callable(fn):
        raise ValueError("pinned converter has no callable conv2julius")
    return fn


def _semantic_rows(semantic_phones: Sequence[Mapping[str, Any]] | None, semantic_graph: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if semantic_graph is not None:
        if not isinstance(semantic_graph, Mapping):
            raise ValueError("semantic graph must be an object")
        nodes = semantic_graph.get("semantic_phone_nodes")
        if not isinstance(nodes, list):
            nodes = [node for node in semantic_graph.get("nodes", []) if isinstance(node, Mapping) and node.get("kind") == "semantic_phone"]
        if not nodes:
            raise ValueError("semantic graph has no semantic phone nodes")
        return [node for node in nodes if isinstance(node, Mapping)]
    if not isinstance(semantic_phones, Sequence) or isinstance(semantic_phones, (str, bytes)) or not semantic_phones:
        raise ValueError("selected semantic phone graph is required")
    if not all(isinstance(row, Mapping) for row in semantic_phones):
        raise ValueError("semantic phone rows must be objects")
    return list(semantic_phones)


def _hiragana_reading(reading: str) -> tuple[str, dict[str, Any]]:
    """Normalize katakana code points only; never perform kanji G2P."""

    normalized_chars: list[str] = []
    for char in reading:
        if "ァ" <= char <= "ヺ":
            normalized_chars.append(chr(ord(char) - 0x60))
        else:
            normalized_chars.append(char)
    normalized = "".join(normalized_chars)
    if not normalized or any(not ("ぁ" <= char <= "ゖ" or char in "ー・") for char in normalized):
        raise ValueError("locked reading must be hiragana/katakana; hidden G2P is forbidden")
    return normalized, {"method": "katakana_to_hiragana_codepoint", "source_reading": reading, "converter_reading": normalized}


def _openjtalk_event_to_julius(events: Sequence[str]) -> list[str]:
    """Translate locked OpenJTalk event symbols to conv2julius units.

    This is a source-event mapping, not a G2P fallback.  It is deliberately
    limited to the symbols emitted by the pinned frontend graph and fails on
    unknown events.
    """

    result: list[str] = []
    index = 0
    compounds = {"k y": "ky", "g y": "gy", "n y": "ny", "h y": "hy", "m y": "my", "r y": "ry", "b y": "by", "p y": "py", "s h": "sh", "c h": "ch", "t s": "ts"}
    direct = {"a", "i", "u", "e", "o", "k", "g", "s", "z", "t", "d", "n", "h", "f", "b", "p", "m", "r", "y", "w", "j", "N", "q", "cl", "sh", "ch", "ts", "ky", "gy", "ny", "hy", "my", "ry", "by", "py", "I", "U", ":"}
    while index < len(events):
        event = str(events[index])
        if index + 1 < len(events):
            pair = f"{event} {events[index + 1]}"
            if pair in compounds:
                result.append(compounds[pair]); index += 2; continue
        if event in direct:
            result.append({"I": "i", "U": "u", "cl": "q"}.get(event, event)); index += 1; continue
        if event.endswith(":") and event[:-1] in {"a", "i", "u", "e", "o"}:
            result.append(event); index += 1; continue
        raise ValueError(f"unsupported OpenJTalk source event {event!r}")
    return result


def _map_source_group_to_actual(events: Sequence[str], actual: Sequence[str], cursor: int) -> tuple[list[str], int]:
    """Map one W2 source-event group onto the pinned converter output.

    OpenJTalk represents the native ``とう``/``こう`` long-vowel relation as
    ``o o``; pinned julius4seg represents that same locked reading as ``o u``.
    This is accepted only for an explicit two-vowel source group and is kept
    in provenance by the caller.
    """

    if len(events) == 2 and events[0] == events[1] and events[0] in {"a", "i", "u", "e", "o"}:
        vowel = events[0]
        if cursor < len(actual) and actual[cursor] == f"{vowel}:":
            return [actual[cursor]], cursor + 1
        allowed_second = {vowel}
        if vowel == "o":
            allowed_second.add("u")
        elif vowel == "e":
            allowed_second.add("i")
        if cursor + 2 <= len(actual) and actual[cursor] == vowel and actual[cursor + 1] in allowed_second:
            return [actual[cursor], actual[cursor + 1]], cursor + 2
        raise ValueError("long-vowel source group does not match pinned Julius representation")
    expected = _openjtalk_event_to_julius(events)
    if list(actual[cursor:cursor + len(expected)]) != expected:
        raise ValueError("source event group does not match pinned Julius representation")
    return expected, cursor + len(expected)


def convert_reading_to_julius(
    reading: str,
    *,
    semantic_phones: Sequence[Mapping[str, Any]] | None = None,
    semantic_graph: Mapping[str, Any] | None = None,
    converter: Callable[..., Any] | None = None,
    converter_path: str | Path | None = None,
    converter_commit: str | None = None,
    mapping_provenance: Mapping[str, Any] | None = None,
    selected_unit_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bridge locked semantic phones through pinned ``conv2julius``."""

    if not isinstance(reading, str) or not reading:
        raise ValueError("selected locked reading is required")
    if converter_commit != JULIUS4SEG_COMMIT:
        raise ValueError("exact julius4seg converter commit is required")
    if not isinstance(mapping_provenance, Mapping) or mapping_provenance.get("method") != "conv2julius":
        raise ValueError("conv2julius mapping provenance is required")
    if mapping_provenance.get("converter_commit") != JULIUS4SEG_COMMIT:
        raise ValueError("mapping provenance converter commit mismatch")
    rows = _semantic_rows(semantic_phones, semantic_graph)
    expected: list[str] = []
    node_ids: list[str] = []
    unit_ids: list[str] = []
    for index, row in enumerate(rows):
        node_id = row.get("id", row.get("node_id"))
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"semantic node {index} has no stable id")
        raw_explicit = row.get("julius_phones", row.get("julius_phone"))
        explicit = raw_explicit
        if explicit is None:
            explicit = []
        if isinstance(explicit, str):
            explicit = [explicit]
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise ValueError(f"semantic node {node_id} has no explicit Julius mapping")
        if raw_explicit is not None and not explicit:
            raise ValueError(f"semantic node {node_id} has empty Julius mapping")
        for phone in explicit:
            if not isinstance(phone, str) or not _PHONE_RE.fullmatch(phone):
                raise ValueError(f"unsupported or ambiguous Julius mapping at {node_id}")
            expected.append(phone)
        node_ids.append(node_id)
        unit_id = row.get("unit_id", row.get("token_id"))
        if unit_id is not None:
            if not isinstance(unit_id, str):
                raise ValueError("semantic unit IDs are ambiguous")
            if unit_id not in unit_ids:
                unit_ids.append(unit_id)
    if selected_unit_ids is not None:
        selected = [str(value) for value in selected_unit_ids]
        if not selected or len(selected) != len(set(selected)) or (unit_ids and set(selected) != set(unit_ids)):
            raise ValueError("selected unit IDs do not match semantic graph")
        unit_ids = selected
    if not unit_ids:
        raise ValueError("selected semantic unit IDs are required")
    if converter is None:
        if converter_path is None:
            raise ValueError("pinned converter source is required")
        converter = _load_conv2julius(_regular_file(converter_path, "converter"))
    converter_reading, normalization = _hiragana_reading(reading)
    converted = converter(converter_reading)
    if isinstance(converted, Mapping):
        converted = converted.get("phones") or converted.get("julius")
    if not isinstance(converted, str):
        raise ValueError("conv2julius must return a phone string")
    actual = converted.split()
    # Graphs produced by the current frontend expose OpenJTalk source event
    # groups, while older fixtures may already carry explicit Julius units.
    # Derive the former deterministically and retain the group provenance.
    source_by_id: dict[str, str] = {}
    if isinstance(semantic_graph, Mapping):
        for source in semantic_graph.get("frontend_phone_nodes", []):
            if isinstance(source, Mapping) and isinstance(source.get("id"), str) and isinstance(source.get("phone"), str):
                source_by_id[source["id"]] = source["phone"]
    derived: list[str] = []
    derived_groups: list[dict[str, Any]] = []
    cursor = 0
    used_source_groups = False
    for row in rows:
        explicit = row.get("julius_phones", row.get("julius_phone"))
        if explicit is None:
            source_ids = row.get("source_openjtalk_ids")
            if not isinstance(source_ids, Sequence) or isinstance(source_ids, (str, bytes)) or not source_ids or not source_by_id:
                raise ValueError(f"semantic node {row.get('id')} has no source event group for Julius mapping")
            source_events = [source_by_id.get(str(source_id)) for source_id in source_ids]
            if any(event is None for event in source_events):
                raise ValueError("semantic source event is missing")
            group, cursor = _map_source_group_to_actual([str(event) for event in source_events], actual, cursor)
            used_source_groups = True
            derived.extend(group)
            derived_groups.append({"node_id": row.get("id"), "source_openjtalk_ids": list(source_ids), "julius_phones": group})
        else:
            if isinstance(explicit, str):
                explicit = [explicit]
            derived.extend(str(phone) for phone in explicit)
    expected = derived
    if (used_source_groups and cursor != len(actual)) or actual != expected:
        raise ValueError(f"converter/semantic phone sequence mismatch: converter={actual!r}, semantic={expected!r}")
    provenance = dict(mapping_provenance)
    provenance["reading_normalization"] = normalization
    if derived_groups:
        provenance["source_event_groups"] = derived_groups
    return {"reading": reading, "converter_reading": converter_reading, "reading_normalization": normalization, "phones": actual, "converter": "conv2julius", "converter_commit": converter_commit, "mapping_provenance": provenance, "semantic_node_ids": node_ids, "selected_unit_ids": unit_ids, "semantic_graph_digest": stable_digest(semantic_graph) if semantic_graph is not None else stable_digest(rows)}


def _asset_audit(*, binary: str | Path | None, model: str | Path | None, dictionary: str | Path | None, converter_path: str | Path | None, jatts_path: str | Path | None, asset_hashes: Mapping[str, str] | None, asset_licenses: Mapping[str, str] | None, allow_converter_fixture: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    values = {"binary": binary, "model": model, "dictionary": dictionary, "converter": converter_path, "jatts": jatts_path}
    if not isinstance(asset_hashes, Mapping) or not isinstance(asset_licenses, Mapping):
        raise ValueError("hash and license evidence for every Julius asset is required")
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name in _ASSET_NAMES:
        path = _regular_file(values[name], name)
        actual = sha256_file(path)
        expected = asset_hashes.get(name)
        if not isinstance(expected, str) or expected.lower() != actual:
            raise ValueError(f"{name} asset hash mismatch")
        if name == "converter" and actual != JULIUS4SEG_SOURCE_SHA256 and not allow_converter_fixture:
            raise ValueError("converter source is not the audited pinned julius4seg file")
        license_status = asset_licenses.get(name)
        if isinstance(license_status, Mapping):
            license_status = license_status.get("status")
        if license_status not in {"approved", "reviewed"}:
            raise ValueError(f"{name} asset license is not approved/reviewed")
        records[name] = {"path": str(path), "sha256": actual, "license_status": license_status}
        paths[name] = path
    return records, paths


def _command_from_workflow(workflow: Any, *, root: Path, input_dir: Path, audio: Path, reading_file: Path, lab: Path, uid: str, assets: Mapping[str, Path] | None = None) -> list[str]:
    if workflow is None:
        raise ValueError("audited JATTS workflow command is required")
    if isinstance(workflow, Mapping):
        workflow = workflow.get("command") or workflow.get("segment_command")
    if isinstance(workflow, str):
        command = shlex.split(workflow)
    elif isinstance(workflow, Sequence) and not isinstance(workflow, (str, bytes)):
        command = [str(part) for part in workflow]
    else:
        raise ValueError("JATTS workflow command must be a string or argv list")
    if not command:
        raise ValueError("JATTS workflow command is empty")
    values = {"stage_dir": str(root), "input_dir": str(input_dir), "audio": str(audio), "reading_file": str(reading_file), "lab": str(lab), "uid": uid}
    values.update({name: str(path) for name, path in (assets or {}).items()})
    result = [part.format(**values) for part in command]
    # External absolute arguments may be legitimate inputs (Julius binary,
    # acoustic model, and dictionary).  Only values expanded from an output
    # placeholder are constrained to this stage.
    output_markers = ("{stage_dir}", "{input_dir}", "{audio}", "{reading_file}", "{lab}")
    for template, part in zip(command, result):
        if any(marker in template for marker in output_markers):
            _inside(part, root)
    return result


def _commands_from_workflow(workflow: Any, *, root: Path, input_dir: Path, audio: Path, reading_file: Path, lab: Path, uid: str, assets: Mapping[str, Path] | None = None) -> list[list[str]]:
    """Expand either one configured command or an audited prepare/segment/post chain."""

    if isinstance(workflow, Mapping) and not (workflow.get("command") or workflow.get("segment_command")):
        commands: list[list[str]] = []
        for name in ("prepare", "segment", "post"):
            if workflow.get(name) is not None:
                commands.append(_command_from_workflow(workflow[name], root=root, input_dir=input_dir, audio=audio, reading_file=reading_file, lab=lab, uid=uid, assets=assets))
        if commands:
            return commands
    return [_command_from_workflow(workflow, root=root, input_dir=input_dir, audio=audio, reading_file=reading_file, lab=lab, uid=uid, assets=assets)]


def _write_diagnostic(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = _safe_root(output_dir)
    atomic_write_json(output_dir / "julius_diagnostic.json", dict(payload))
    return dict(payload)


def run_julius_diagnostic(audio: str | Path, reading: str, output_dir: str | Path, *, enabled: bool = False, binary: str | Path | None = None, model: str | Path | None = None, dictionary: str | Path | None = None, converter_path: str | Path | None = None, jatts_path: str | Path | None = None, converter_source: str | Path | None = None, jatts_reference: str | Path | None = None, lab: str | Path | None = None, semantic_phones: Sequence[Mapping[str, Any]] | None = None, semantic_graph: Mapping[str, Any] | None = None, selected_unit_ids: Sequence[str] | None = None, mapping_provenance: Mapping[str, Any] | None = None, converter_commit: str | None = None, jatts_reference_commit: str | None = None, runner: Callable[..., Any] | None = None, converter: Callable[..., Any] | None = None, asset_hashes: Mapping[str, str] | None = None, asset_licenses: Mapping[str, str] | None = None, workflow: Any = None, timeout_seconds: int = 120, uid: str | None = None) -> dict[str, Any]:
    out = _safe_root(output_dir)
    audio_id = uid or Path(audio).stem
    base = {"schema": "julius-diagnostic-v1", "uid": audio_id, "production_write_back": False}
    if not enabled:
        return _write_diagnostic(out, {**base, "status": "diagnostic_unavailable", "reason": "disabled"})
    if not isinstance(audio_id, str) or not _UID_RE.fullmatch(audio_id):
        return _write_diagnostic(out, {**base, "status": "diagnostic_unavailable", "reason": "invalid uid"})
    converter_path = converter_path or converter_source
    jatts_path = jatts_path or jatts_reference
    if converter_commit != JULIUS4SEG_COMMIT or jatts_reference_commit != JATTS_COMMIT:
        return _write_diagnostic(out, {**base, "status": "diagnostic_unavailable", "reason": "exact Julius4seg/JATTS commits are required", "required_commits": {"converter": JULIUS4SEG_COMMIT, "jatts": JATTS_COMMIT}})
    try:
        audio_info = validate_pcm16(audio)
        assets, paths = _asset_audit(binary=binary, model=model, dictionary=dictionary, converter_path=converter_path, jatts_path=jatts_path, asset_hashes=asset_hashes, asset_licenses=asset_licenses, allow_converter_fixture=converter is not None)
        representation = convert_reading_to_julius(reading, semantic_phones=semantic_phones, semantic_graph=semantic_graph, converter=converter, converter_path=paths["converter"], converter_commit=converter_commit, mapping_provenance=mapping_provenance, selected_unit_ids=selected_unit_ids)
        root = out / "workflow"
        if root.exists() and any(root.rglob("*")):
            raise ValueError("workflow stage must be fresh; stale Julius outputs are forbidden")
        input_dir = root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        staged_audio = input_dir / f"{audio_id}.wav"
        shutil.copyfile(audio_info["path"], staged_audio)
        if sha256_file(staged_audio) != audio_info["sha256"]:
            raise ValueError("staged audio hash changed")
        reading_file = input_dir / f"{audio_id}.txt"
        reading_file.write_text(reading, encoding="utf-8")
        lab_path = _inside(lab, root) if lab is not None else root / f"{audio_id}.lab"
        commands = _commands_from_workflow(workflow, root=root, input_dir=input_dir, audio=staged_audio, reading_file=reading_file, lab=lab_path, uid=audio_id, assets=paths)
        bound = {str(arg) for command in commands for arg in command}
        for asset_name in ("binary", "model", "dictionary", "converter", "jatts"):
            if str(paths[asset_name]) not in bound:
                raise ValueError(f"configured workflow does not bind audited {asset_name} asset")
        execute = runner or subprocess.run
        command_records: list[dict[str, Any]] = []
        completed: Any = None
        for command in commands:
            completed = execute(command, cwd=str(root), check=True, capture_output=True, text=True, timeout=timeout_seconds)
            command_records.append({"argv": command, "stdout": getattr(completed, "stdout", "") or "", "stderr": getattr(completed, "stderr", "") or ""})
        _audit_tree(out)
        if not lab_path.is_file():
            candidates = sorted(root.rglob("*.lab"))
            if len(candidates) == 1:
                lab_path = candidates[0]
            else:
                raise ValueError("JATTS workflow did not produce exactly one .lab")
        intervals = parse_lab(lab_path, sample_rate=16000, timebase="seconds")
        if not intervals:
            raise ValueError("Julius .lab has no intervals")
        if intervals[-1]["end_sample"] > audio_info["frames"]:
            raise ValueError("Julius .lab exceeds frozen WAV")
        emitted = [row["phone"] for row in intervals if row["phone"] not in {"silB", "silE"}]
        if emitted != representation["phones"]:
            raise ValueError(f"Julius phone sequence mismatch: emitted={emitted!r}, expected={representation['phones']!r}")
        result = {**base, "status": "COMPLETE", "frozen_wav_hash": audio_info["sha256"], "audio": audio_info, "selected_reading": reading, "selected_reading_digest": stable_digest(reading), "selected_unit_ids": representation["selected_unit_ids"], "semantic_graph_digest": representation["semantic_graph_digest"], "representation": representation, "representation_diff": {"expected_phones": representation["phones"], "emitted_phones": emitted, "match": True}, "raw_intervals": intervals, "asset_ids": assets, "commands": command_records, "lab_path": str(lab_path), "production_write_back": False}
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        return _write_diagnostic(out, {**base, "status": "diagnostic_unavailable", "reason": str(exc), "production_write_back": False})
    return _write_diagnostic(out, result)


def _load_rows(source: Any) -> list[Mapping[str, Any]]:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        rows = list(source)
    elif source:
        path = _regular_file(source, "prepared input")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        return []
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("prepared items must be objects")
    return rows


def handle_julius(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    stage_dir = _safe_root(stage_dir)
    settings = config.get("julius_diagnostic", {}) or {}
    receipt_path = stage_dir / "receipt.json"
    if not bool(settings.get("enabled", False)):
        _write_diagnostic(stage_dir, {"schema": "julius-diagnostic-v1", "uid": "stage", "status": "diagnostic_unavailable", "reason": "disabled", "production_write_back": False})
        receipt = make_receipt(stage="julius", status="BLOCKED", outputs=[stage_dir / "julius_diagnostic.json"], params={"implementation": "julius-diagnostic-v1", "enabled": False}, errors=[{"code": "julius_diagnostic_unavailable", "message": "Julius diagnostic is disabled"}])
        atomic_write_json(receipt_path, receipt)
        return StageResult(stage="julius", status="BLOCKED", receipt_path=str(receipt_path))
    try:
        rows = _load_rows(settings.get("prepared_items") or settings.get("prepared_jsonl") or settings.get("input_jsonl"))
        if not rows and settings.get("audio") and settings.get("selected_reading", settings.get("reading")):
            rows = [{"uid": Path(str(settings["audio"])).stem, "audio": settings["audio"], "selected_reading": settings.get("selected_reading", settings.get("reading")), "semantic_phones": settings.get("semantic_phones"), "semantic_graph": settings.get("semantic_graph"), "selected_unit_ids": settings.get("selected_unit_ids"), "mapping_provenance": settings.get("mapping_provenance")}]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rows = [{"uid": "stage", "error": str(exc)}]
    results: list[dict[str, Any]] = []
    for row in rows:
        uid = row.get("uid") or row.get("id")
        audio = row.get("audio") or row.get("alignment_wav") or row.get("wav")
        reading = row.get("selected_reading") or row.get("reading")
        if not isinstance(uid, str) or not _UID_RE.fullmatch(uid) or not audio or not isinstance(reading, str):
            results.append({"schema": "julius-diagnostic-v1", "uid": str(uid or "unknown"), "status": "diagnostic_unavailable", "reason": row.get("error", "uid, frozen audio, and locked reading are required"), "production_write_back": False})
            continue
        item_dir = stage_dir / "items" / uid
        results.append(run_julius_diagnostic(audio, reading, item_dir, enabled=True, uid=uid, binary=settings.get("binary"), model=settings.get("model"), dictionary=settings.get("dictionary"), converter_path=settings.get("converter_path", settings.get("converter_source", settings.get("converter"))), jatts_path=settings.get("jatts_path", settings.get("jatts_reference", settings.get("jatts"))), semantic_phones=row.get("semantic_phones"), semantic_graph=row.get("semantic_graph"), selected_unit_ids=row.get("selected_unit_ids"), mapping_provenance=row.get("mapping_provenance", settings.get("mapping_provenance")), converter_commit=settings.get("converter_commit"), jatts_reference_commit=settings.get("jatts_reference_commit"), asset_hashes=settings.get("asset_hashes"), asset_licenses=settings.get("asset_licenses"), workflow=row.get("workflow", settings.get("workflow", settings.get("workflow_command"))), runner=settings.get("runner"), timeout_seconds=int(settings.get("timeout_seconds", 120))))
    summary = stage_dir / "diagnostic_receipt.json"
    complete = bool(results) and all(item.get("status") == "COMPLETE" for item in results)
    atomic_write_json(summary, {"schema": "julius-diagnostic-v1", "status": "COMPLETE" if complete else "diagnostic_unavailable", "production_write_back": False, "results": results})
    _audit_tree(stage_dir)
    outputs = [summary] + sorted(path for path in stage_dir.rglob("*") if path.is_file() and path not in {summary, receipt_path})
    receipt = make_receipt(stage="julius", status="COMPLETE" if complete else "PARTIAL", outputs=outputs, params={"implementation": "julius-diagnostic-v1", "enabled": True}, commands=[str(item.get("commands", [])) for item in results], errors=[] if complete else [{"code": "julius_diagnostic_unavailable", "message": "prepared Julius input/assets are unavailable or failed"}])
    atomic_write_json(receipt_path, receipt)
    return StageResult(stage="julius", status="COMPLETE" if complete else "PARTIAL", receipt_path=str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("julius", handle_julius, output_namespace="julius")


parse_julius_lab = parse_lab
run_diagnostic = run_julius_diagnostic


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("reading")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, help="JSON diagnostic configuration")
    parser.add_argument("--workspace", type=Path, help="run workspace; enabled output is constrained to workspace/diagnostics/julius")
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--binary")
    parser.add_argument("--model")
    parser.add_argument("--dictionary")
    parser.add_argument("--converter-path")
    parser.add_argument("--jatts-path")
    parser.add_argument("--converter-commit", default=JULIUS4SEG_COMMIT)
    parser.add_argument("--jatts-reference-commit", default=JATTS_COMMIT)
    parser.add_argument("--semantic-json", type=Path, help="JSON file containing semantic_phones or semantic_graph")
    parser.add_argument("--mapping-provenance-json", type=Path)
    parser.add_argument("--asset-hashes-json", type=Path)
    parser.add_argument("--asset-licenses-json", type=Path)
    parser.add_argument("--selected-unit-id", action="append", dest="selected_unit_ids")
    parser.add_argument("--workflow", nargs="+")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, Mapping):
        raise SystemExit("--config must contain a JSON object")
    args.enabled = bool(args.enabled or config.get("enabled", False))
    workspace = args.workspace or (Path(config["workspace"]) if config.get("workspace") else None)
    if args.enabled:
        if workspace is None:
            raise SystemExit("enabled CLI runs require --workspace or config.workspace")
        diagnostic_root = workspace.expanduser().absolute() / "diagnostics" / "julius"
        args.output = _inside(args.output, diagnostic_root)
    args.binary = args.binary or config.get("binary")
    args.model = args.model or config.get("model")
    args.dictionary = args.dictionary or config.get("dictionary")
    args.converter_path = args.converter_path or config.get("converter_path", config.get("converter"))
    args.jatts_path = args.jatts_path or config.get("jatts_path", config.get("jatts"))
    args.workflow = args.workflow or config.get("workflow", config.get("workflow_command"))
    def load_json(path: Path | None) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path else None
    semantic_payload = load_json(args.semantic_json) or config.get("semantic", {}) or {}
    mapping_provenance = load_json(args.mapping_provenance_json) or config.get("mapping_provenance")
    asset_hashes = load_json(args.asset_hashes_json) or config.get("asset_hashes")
    asset_licenses = load_json(args.asset_licenses_json) or config.get("asset_licenses")
    selected_unit_ids = args.selected_unit_ids or config.get("selected_unit_ids")
    result = run_julius_diagnostic(args.audio, args.reading, args.output, enabled=args.enabled, binary=args.binary, model=args.model, dictionary=args.dictionary, converter_path=args.converter_path, jatts_path=args.jatts_path, semantic_phones=semantic_payload.get("semantic_phones"), semantic_graph=semantic_payload.get("semantic_graph", semantic_payload if semantic_payload.get("nodes") else None), selected_unit_ids=selected_unit_ids, mapping_provenance=mapping_provenance, converter_commit=config.get("converter_commit", args.converter_commit), jatts_reference_commit=config.get("jatts_reference_commit", args.jatts_reference_commit), asset_hashes=asset_hashes, asset_licenses=asset_licenses, workflow=args.workflow)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JATTS_COMMIT", "JULIUS4SEG_COMMIT", "JULIUS4SEG_SOURCE_SHA256", "convert_reading_to_julius", "handle_julius", "parse_julius_lab", "parse_lab", "register_stages", "run_diagnostic", "run_julius_diagnostic", "validate_pcm16"]
