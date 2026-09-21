"""Pinned pyopenjtalk-plus bridge and the W2 frontend contract.

The bridge is deliberately a subprocess boundary.  A run configured with the
frontend venv therefore cannot accidentally import a second pyopenjtalk
distribution from the pipeline interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .ja_en_schema import (
        JAContractError,
        StageResult,
        atomic_write_json,
        artifact_record,
        canonical_json,
        make_receipt,
        sha256_file,
        stable_digest,
    )
    from .ja_text_layers import canonical_span_to_orig, canonicalize_text, validate_monotonic_spans
except ImportError:  # direct script execution
    from ja_en_schema import JAContractError, StageResult, atomic_write_json, artifact_record, canonical_json, make_receipt, sha256_file, stable_digest
    from ja_text_layers import canonical_span_to_orig, canonicalize_text, validate_monotonic_spans


PYOPENJTALK_COMMIT = "9e4bf25324ac135dfc81ca64aed2fa6a48b83304"
DEFAULT_RUNTIME = ""
DEFAULT_FRONTEND_OPTIONS = {
    "run_marine": False,
    "use_vanilla": False,
    "use_tsqyomi": False,
    "use_sudachi_kanji_yomi": False,
    "predict_nani": False,
    "normalize_mode": "NFKC",
    "use_read_as_pron": False,
    "revert_long_vowels": False,
    "revert_yotsugana": False,
}
ANALYSIS_VERSION = "ja-frontend-analysis-v2"
_CAPABILITIES = ("g2p_mapping", "run_frontend_detailed", "make_phoneme_mapping", "extract_fullcontext")


@dataclass(frozen=True)
class FrontendConfig:
    provider: str = "pyopenjtalk-plus"
    commit: str = PYOPENJTALK_COMMIT
    runtime_python: str = DEFAULT_RUNTIME
    options: Mapping[str, Any] = None  # type: ignore[assignment]
    reject_unbound_spans: bool = True
    manual_lexicon: Mapping[str, Any] = None  # type: ignore[assignment]
    mapping_file: str | None = None
    english_dictionary: str | None = None
    english_metadata: str | None = None
    build_receipt: str | None = None
    require_build_receipt: bool = False

    def __post_init__(self) -> None:
        values = dict(DEFAULT_FRONTEND_OPTIONS)
        values.update(dict(self.options or {}))
        for key in DEFAULT_FRONTEND_OPTIONS:
            if key not in values or (key != "normalize_mode" and type(values[key]) is not bool):
                raise JAContractError("config_malformed", f"frontend option {key} must be explicit")
        if values["normalize_mode"] not in {"None", "NFC", "NFKC"}:
            raise JAContractError("config_malformed", "normalize_mode must be None, NFC, or NFKC")
        if self.reject_unbound_spans is not True:
            raise JAContractError("config_malformed", "reject_unbound_spans must be true")
        object.__setattr__(self, "options", values)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "FrontendConfig":
        source = config or {}
        parent = source
        if isinstance(source.get("frontend"), Mapping):
            source = source["frontend"]
        values = dict(DEFAULT_FRONTEND_OPTIONS)
        values.update({key: source[key] for key in DEFAULT_FRONTEND_OPTIONS if key in source})
        runtime = source.get("runtime_python", source.get("python", os.environ.get("JA_FRONTEND_PYTHON", DEFAULT_RUNTIME)))
        mfa = parent.get("mfa", {}) if isinstance(parent.get("mfa"), Mapping) else (source.get("mfa", {}) if isinstance(source.get("mfa"), Mapping) else {})
        mapping_file_value = source.get("mapping_file", parent.get("mapping_file"))
        mapping_file = str(mapping_file_value) if mapping_file_value else None
        file_lexicon: Mapping[str, Any] = {}
        if mapping_file:
            mapping_path = Path(mapping_file).expanduser()
            if not mapping_path.is_file():
                raise JAContractError("dictionary_asset_missing", "configured frontend mapping file does not exist", str(mapping_path))
            try:
                loaded = json.loads(mapping_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise JAContractError("config_malformed", "frontend mapping file must be valid JSON", str(mapping_path)) from exc
            if isinstance(loaded, Mapping) and isinstance(loaded.get("entries", loaded.get("mappings", loaded)), Mapping):
                file_lexicon = dict(loaded.get("entries", loaded.get("mappings", loaded)))
            else:
                raise JAContractError("config_malformed", "frontend mapping file must be an object or contain entries/mappings", str(mapping_path))
        inline_lexicon = source.get("manual_lexicon", parent.get("manual_lexicon", {}))
        merged_lexicon = {**file_lexicon, **(dict(inline_lexicon) if isinstance(inline_lexicon, Mapping) else {})}
        return cls(
            provider=str(source.get("provider", "pyopenjtalk-plus")),
            commit=str(source.get("commit", PYOPENJTALK_COMMIT)),
            runtime_python=str(runtime),
            options=values,
            reject_unbound_spans=bool(source.get("reject_unbound_spans", True)),
            manual_lexicon=merged_lexicon,
            mapping_file=mapping_file,
            english_dictionary=str(source.get("english_dictionary", mfa.get("english_dictionary"))) if source.get("english_dictionary", mfa.get("english_dictionary")) else None,
            english_metadata=str(source.get("english_metadata", mfa.get("english_metadata"))) if source.get("english_metadata", mfa.get("english_metadata")) else None,
            build_receipt=str(source.get("build_receipt")) if source.get("build_receipt") else None,
            require_build_receipt=bool(source.get("require_build_receipt", False)),
        )

    def identity(self) -> dict[str, Any]:
        mapping_digest = sha256_file(self.mapping_file) if self.mapping_file and Path(self.mapping_file).is_file() else None
        return {"provider": self.provider, "commit": self.commit, "runtime_python": self.runtime_python, "options": dict(self.options), "manual_lexicon": dict(self.manual_lexicon or {}), "mapping_file": self.mapping_file, "mapping_file_sha256": mapping_digest, "english_dictionary": self.english_dictionary, "english_metadata": self.english_metadata, "build_receipt": self.build_receipt, "require_build_receipt": self.require_build_receipt}


def _worker_code() -> str:
    return r'''
import importlib.metadata as md
import inspect
import json
import hashlib
from pathlib import Path
import sys
import pyopenjtalk

request = json.load(sys.stdin)
dists = md.packages_distributions().get("pyopenjtalk", [])
if not dists:
    dists = ["unknown"]
capabilities = {name: str(inspect.signature(getattr(pyopenjtalk, name))) for name in ("g2p_mapping", "run_frontend_detailed", "make_phoneme_mapping", "extract_fullcontext") if hasattr(pyopenjtalk, name)}
distribution_files = []
try:
    distribution = md.distribution("pyopenjtalk-plus")
    for entry in distribution.files or ():
        path = Path(distribution.locate_file(entry))
        if path.is_file():
            distribution_files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size})
except Exception:
    distribution_files = []
distribution_files.sort(key=lambda row: (row["path"], row["size"], row["sha256"]))
if request.get("op") == "probe":
    print(json.dumps({"provider": "pyopenjtalk-plus", "distributions": dists, "version": md.version("pyopenjtalk-plus"), "module": pyopenjtalk.__file__, "capabilities": capabilities, "distribution_files": distribution_files}, ensure_ascii=False))
    raise SystemExit(0)
options = dict(request["options"])
if request["op"] == "mapping":
    njd, morphs = pyopenjtalk.run_frontend_detailed(text=request["text"], **options)
    rows = pyopenjtalk.g2p_mapping(text=request["text"], **options)
    labels = pyopenjtalk.extract_fullcontext(text=request["text"], **options)
elif request["op"] == "reading":
    rows = pyopenjtalk.g2p_mapping(text=request["text"], **options)
else:
    raise ValueError("unknown bridge operation")
print(json.dumps({"provider": "pyopenjtalk-plus", "distributions": dists, "version": md.version("pyopenjtalk-plus"), "module": pyopenjtalk.__file__, "capabilities": capabilities, "distribution_files": distribution_files, "rows": rows, "morphs": morphs if request["op"] == "mapping" else [], "njd_rows": njd if request["op"] == "mapping" else [], "full_context_labels": labels if request["op"] == "mapping" else []}, ensure_ascii=False))
'''


def _run_bridge(runtime_python: str, request: Mapping[str, Any]) -> dict[str, Any]:
    runtime = Path(runtime_python).expanduser()
    if not runtime.is_file():
        raise JAContractError("frontend_capability_missing", "configured frontend Python does not exist", str(runtime))
    try:
        completed = subprocess.run(
            [str(runtime), "-c", _worker_code()],
            input=canonical_json(request).decode("utf-8"), text=True, capture_output=True, check=False,
        )
    except OSError as exc:
        raise JAContractError("frontend_capability_missing", str(exc), str(runtime)) from exc
    if completed.returncode:
        detail = completed.stderr.strip()[-2000:] or completed.stdout.strip()[-2000:]
        code = "frontend_capability_missing" if "ModuleNotFoundError" in detail or "ImportError" in detail else "provider_ambiguous"
        raise JAContractError(code, detail or "frontend worker failed", str(runtime))
    try:
        # pyopenjtalk may print optional-model warnings to stdout during
        # import.  The worker emits exactly one JSON object; parse the last
        # line that starts that object rather than treating warnings as data.
        output = completed.stdout.strip()
        json_lines = [line for line in output.splitlines() if line.lstrip().startswith("{")]
        payload = json.loads(json_lines[-1] if json_lines else output)
    except json.JSONDecodeError as exc:
        raise JAContractError("frontend_capability_missing", "frontend worker returned invalid JSON", str(runtime)) from exc
    if not isinstance(payload, dict):
        raise JAContractError("frontend_capability_missing", "frontend worker response is not an object")
    return payload


def probe_provider(config: Mapping[str, Any] | FrontendConfig | None = None) -> dict[str, Any]:
    settings = config if isinstance(config, FrontendConfig) else FrontendConfig.from_mapping(config)
    if settings.provider != "pyopenjtalk-plus":
        raise JAContractError("frontend_provider_ambiguous", f"unsupported provider {settings.provider!r}")
    payload = _run_bridge(settings.runtime_python, {"op": "probe"})
    distributions = [str(value).lower().replace("_", "-") for value in payload.get("distributions", [])]
    if len(distributions) != 1 or distributions[0] != "pyopenjtalk-plus":
        raise JAContractError("frontend_provider_ambiguous", f"namespace providers are {distributions!r}")
    if settings.commit != PYOPENJTALK_COMMIT:
        raise JAContractError("frontend_provider_ambiguous", "frontend commit is not the pinned commit")
    missing = sorted(set(_CAPABILITIES) - set(payload.get("capabilities", {})))
    if missing:
        raise JAContractError("frontend_capability_missing", f"missing API capabilities: {missing!r}")
    provenance: dict[str, Any] = {"status": "unverified"}
    if settings.build_receipt:
        receipt_path = Path(settings.build_receipt).expanduser()
        if not receipt_path.is_file():
            raise JAContractError("frontend_capability_missing", "frontend build receipt is missing", str(receipt_path))
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JAContractError("frontend_capability_missing", "frontend build receipt is invalid", str(receipt_path)) from exc
        if receipt.get("source_commit") != settings.commit:
            raise JAContractError("frontend_representation_drift", "frontend build receipt commit mismatch", str(receipt_path))
        wheel_rows = [row for row in receipt.get("artifacts", []) if str(row.get("path", "")).endswith(".whl")]
        if not wheel_rows or not all(row.get("sha256") and row.get("size") for row in wheel_rows):
            raise JAContractError("frontend_capability_missing", "frontend build receipt lacks wheel hash/size", str(receipt_path))
        for wheel in wheel_rows:
            wheel_path = Path(str(wheel.get("path", ""))).expanduser()
            if wheel_path.is_file():
                if wheel_path.stat().st_size != wheel["size"] or hashlib.sha256(wheel_path.read_bytes()).hexdigest() != wheel["sha256"]:
                    raise JAContractError("frontend_capability_missing", "frontend wheel receipt hash mismatch", str(wheel_path))
        expected_files = receipt.get("installed_distribution_files")
        actual_files = payload.get("distribution_files")
        if not isinstance(expected_files, list) or not expected_files or not isinstance(actual_files, list):
            raise JAContractError("frontend_capability_missing", "frontend build receipt lacks installed distribution file binding", str(receipt_path))
        expected_by_path = {str(row.get("path")): row for row in expected_files if isinstance(row, Mapping)}
        actual_by_path = {str(row.get("path")): row for row in actual_files if isinstance(row, Mapping)}
        if set(expected_by_path) != set(actual_by_path):
            raise JAContractError("frontend_representation_drift", "installed frontend distribution files differ from build receipt", str(receipt_path))
        for path_text, expected in expected_by_path.items():
            actual = actual_by_path[path_text]
            if actual.get("sha256") != expected.get("sha256") or actual.get("size") != expected.get("size"):
                raise JAContractError("frontend_representation_drift", "installed frontend distribution file hash mismatch", path_text)
        provenance = {"status": "verified", "receipt": str(receipt_path), "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(), "wheel": wheel_rows}
    elif settings.require_build_receipt:
        raise JAContractError("frontend_capability_missing", "production frontend requires build receipt")
    payload["expected_commit"] = PYOPENJTALK_COMMIT
    payload["source_commit"] = settings.commit
    payload["build_provenance"] = provenance
    payload["options"] = dict(settings.options)
    return payload


def _span_row(row: Mapping[str, Any], canonical_length: int, index: int) -> tuple[int, int]:
    span = row.get("char_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise JAContractError("frontend_span_unbound", "mapping has no valid char_span", f"$.units[{index}].char_span")
    start, end = int(span[0]), int(span[1])
    if start == 0 and end == 0 and row.get("surface"):
        raise JAContractError("frontend_span_unbound", "non-empty mapping is unbound", f"$.units[{index}].char_span")
    if start < 0 or end <= start or end > canonical_length:
        raise JAContractError("frontend_span_unbound", "span is outside canonical caller text", f"$.units[{index}].char_span")
    return start, end


_LABEL_A_RE = re.compile(r"/A:([-+]?\d+)\+(\d+)\+(\d+)")
_LABEL_F_RE = re.compile(r"/F:(\d+)_(\d+)#[^@]*@(\d+)_(\d+)\|(\d+)_(\d+)")


def _accent_error(message: str, path: str | None = None) -> JAContractError:
    return JAContractError("accent_phrase_unresolved", message, path)


def _accent_row_int(row: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name in row:
            try:
                return int(row[name])
            except (TypeError, ValueError) as exc:
                raise _accent_error(f"invalid {name} in NJD row") from exc
    raise _accent_error(f"NJD row lacks {'/'.join(names)}")


def _accent_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if current and _accent_row_int(row, "chain_flag") != 1:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def _mora_tones(mora_count: int, nucleus: int) -> list[str]:
    if mora_count < 1 or nucleus < 0 or nucleus > mora_count:
        raise _accent_error("invalid phrase nucleus")
    if nucleus == 1:
        return ["H"] + ["L"] * (mora_count - 1)
    tones = ["L"] + ["H"] * (mora_count - 1)
    if 1 < nucleus < mora_count:
        tones[nucleus:] = ["L"] * (mora_count - nucleus)
    return tones


def _parsed_label_moras(labels: Sequence[str]) -> list[dict[str, Any]]:
    """Collapse phone labels to their explicit, one-based mora positions."""
    phrases: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise _accent_error("full-context label is not text", f"$.full_context_labels[{index}]")
        a_match = _LABEL_A_RE.search(label)
        if a_match is None:
            continue
        f_match = _LABEL_F_RE.search(label)
        if f_match is None:
            raise _accent_error("mora label has no parseable F feature", f"$.full_context_labels[{index}]")
        offset, position, remaining = (int(value) for value in a_match.groups())
        mora_count, encoded_nucleus, phrase_index, phrase_remaining, breath_position, breath_remaining = (
            int(value) for value in f_match.groups()
        )
        if position < 1 or remaining < 1 or position + remaining - 1 != mora_count:
            raise _accent_error("A/F mora cardinality disagrees", f"$.full_context_labels[{index}]")
        if offset != position - encoded_nucleus:
            raise _accent_error("A/F nucleus position disagrees", f"$.full_context_labels[{index}]")
        descriptor = (mora_count, encoded_nucleus, phrase_index, phrase_remaining, breath_position, breath_remaining)
        if current and descriptor != current[-1]["label_descriptor"]:
            # A descriptor changes only at a phrase boundary.  The next mora
            # must restart at one; otherwise the label stream is ambiguous.
            if position != 1:
                raise _accent_error("full-context phrase descriptor changes mid-phrase", f"$.full_context_labels[{index}]")
            phrases.append(current)
            current = []
        if current and position == current[-1]["mora_index_in_phrase"]:
            current[-1]["label_indices"].append(index)
            continue
        if current and position != current[-1]["mora_index_in_phrase"] + 1:
            raise _accent_error("full-context mora positions are not monotonic", f"$.full_context_labels[{index}]")
        if not current and position != 1:
            raise _accent_error("full-context phrase does not start at mora one", f"$.full_context_labels[{index}]")
        current.append({
            "mora_index_in_phrase": position,
            "mora_count": mora_count,
            "encoded_nucleus": encoded_nucleus,
            "label_descriptor": descriptor,
            "label_indices": [index],
        })
    if current:
        phrases.append(current)
    return [{"moras": phrase, "descriptor": phrase[0]["label_descriptor"]} for phrase in phrases]


def _bind_full_context_moras(groups: Sequence[Sequence[Mapping[str, Any]]], labels: Sequence[str]) -> list[dict[str, Any]]:
    expected: list[tuple[int, int]] = []
    for group in groups:
        mora_count = sum(_accent_row_int(row, "mora_size", "mora_count") for row in group)
        if mora_count == 0:
            continue
        nucleus = _accent_row_int(group[0], "acc", "accent_nucleus")
        if nucleus < 0 or nucleus > mora_count:
            raise _accent_error("NJD phrase nucleus is outside its mora count")
        expected.append((mora_count, nucleus))
    parsed = _parsed_label_moras(labels)
    if len(parsed) != len(expected):
        raise _accent_error("NJD and full-context accent phrase counts differ")
    result: list[dict[str, Any]] = []
    for phrase_number, ((mora_count, nucleus), label_phrase) in enumerate(zip(expected, parsed)):
        label_moras = label_phrase["moras"]
        descriptor = label_phrase["descriptor"]
        if len(label_moras) != mora_count or descriptor[0] != mora_count:
            raise _accent_error("NJD and full-context mora cardinalities differ")
        encoded_nucleus = mora_count if nucleus == 0 else nucleus
        if descriptor[1] != encoded_nucleus:
            raise _accent_error("NJD and full-context nuclei differ")
        for expected_position, label_mora in enumerate(label_moras, start=1):
            if label_mora["mora_index_in_phrase"] != expected_position:
                raise _accent_error("full-context mora sequence is not contiguous")
            result.append({
                "accent_phrase_id": f"ap{phrase_number}",
                "mora_index_in_phrase": expected_position,
                "mora_count": mora_count,
                "nucleus": nucleus,
                "downstep_after_mora": nucleus if nucleus else None,
                "phrase_start": expected_position == 1,
                "phrase_end": expected_position == mora_count,
                "label_indices": label_mora["label_indices"],
            })
    return result


def _phrase_summaries(moras: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    by_phrase: dict[str, list[Mapping[str, Any]]] = {}
    for mora in moras:
        by_phrase.setdefault(str(mora["accent_phrase_id"]), []).append(mora)
    for phrase_id, phrase_moras in by_phrase.items():
        first = phrase_moras[0]
        mora_count = int(first["mora_count"])
        nucleus = int(first["nucleus"])
        if len(phrase_moras) != mora_count:
            raise _accent_error("bound phrase has incomplete mora sequence")
        summaries.append({
            "accent_phrase_id": phrase_id,
            "mora_count": mora_count,
            "nucleus": nucleus,
            "downstep_after_mora": nucleus if nucleus else None,
            "expected_tones": _mora_tones(mora_count, nucleus),
        })
    return summaries


def _provider_evidence_digest(evidence: Mapping[str, Any]) -> str:
    return stable_digest({
        "adapter_version": evidence["adapter_version"],
        "njd_rows": evidence["njd_rows"],
        "full_context_labels": evidence["full_context_labels"],
        "provider_identity": evidence["provider_identity"],
        "accent_phrases": evidence["accent_phrases"],
        "moras": evidence["moras"],
    })


def _unit_evidence_digest(evidence: Mapping[str, Any]) -> str:
    return stable_digest({key: value for key, value in evidence.items() if key != "unit_evidence_sha256"})


def _scoped_unit_evidence(root_evidence: Mapping[str, Any], moras: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = {row["accent_phrase_id"]: row for row in root_evidence["accent_phrases"]}
    phrase_ids = {row["accent_phrase_id"] for row in moras}
    evidence = {
        "adapter_version": root_evidence["adapter_version"],
        "provider_evidence_sha256": root_evidence["provider_evidence_sha256"],
        "provider_identity": root_evidence["provider_identity"],
        "njd_rows": root_evidence["njd_rows"],
        "full_context_labels": root_evidence["full_context_labels"],
        "accent_phrases": [summaries[phrase_id] for phrase_id in sorted(phrase_ids)],
        "moras": list(moras),
    }
    evidence["unit_evidence_sha256"] = _unit_evidence_digest(evidence)
    return evidence


def extract_contextual_accent_evidence(
    njd_rows: Sequence[Mapping[str, Any]], labels: Sequence[str], provider_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize pinned OpenJTalk phrase evidence without re-predicting text."""
    rows = [dict(row) for row in njd_rows]
    copied_labels = list(labels)
    moras = _bind_full_context_moras(_accent_groups(rows), copied_labels)
    evidence = {
        "adapter_version": "openjtalk-fullcontext-accent-v1",
        "accent_phrases": _phrase_summaries(moras),
        "moras": moras,
        "njd_rows": rows,
        "full_context_labels": copied_labels,
        "provider_identity": dict(provider_identity or {}),
    }
    evidence["provider_evidence_sha256"] = _provider_evidence_digest(evidence)
    return evidence


def _attach_contextual_accent_evidence(units: list[dict[str, Any]], evidence: Mapping[str, Any] | None) -> None:
    cursor = 0
    all_moras = list(evidence.get("moras", [])) if evidence else []
    for unit in units:
        contextual_reading = str(unit.get("read", ""))
        unit["contextual_reading"] = contextual_reading
        unit["contextual_reading_digest"] = stable_digest(contextual_reading)
        unit["locked_reading_digest"] = unit["contextual_reading_digest"]
        count = int(unit.get("mora_count", 0) or 0)
        unit_moras = all_moras[cursor:cursor + count]
        cursor += count
        unit_evidence = None if evidence is None else _scoped_unit_evidence(evidence, unit_moras)
        unit["accent_evidence"] = unit_evidence
        unit["accent_evidence_valid"] = evidence is not None
        if evidence is None:
            unit["accent_evidence_invalid_reason"] = "contextual_accent_unavailable"
    if evidence is not None and cursor != len(all_moras):
        raise _accent_error("frontend units and full-context mora cardinalities differ")


def validate_frontend_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the validity gate for contextual evidence after a reading lock."""
    payload = dict(contract)
    contextual_evidence = payload.get("contextual_accent_evidence")
    evidence_digest_matches = (
        isinstance(contextual_evidence, Mapping)
        and all(key in contextual_evidence for key in (
            "adapter_version", "njd_rows", "full_context_labels", "provider_identity",
            "accent_phrases", "moras", "provider_evidence_sha256",
        ))
        and contextual_evidence["provider_evidence_sha256"] == _provider_evidence_digest(contextual_evidence)
    )
    expected_unit_evidence: list[dict[str, Any] | None] = []
    if evidence_digest_matches:
        cursor = 0
        for unit in contract.get("units", []):
            try:
                mora_count = int(unit.get("mora_count", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                expected_unit_evidence = []
                break
            if mora_count < 0:
                expected_unit_evidence = []
                break
            unit_moras = contextual_evidence["moras"][cursor:cursor + mora_count]
            if len(unit_moras) != mora_count:
                expected_unit_evidence = []
                break
            try:
                expected_unit_evidence.append(_scoped_unit_evidence(contextual_evidence, unit_moras))
            except (KeyError, TypeError):
                expected_unit_evidence = []
                break
            cursor += mora_count
        if cursor != len(contextual_evidence["moras"]):
            expected_unit_evidence = []
    checked_units: list[dict[str, Any]] = []
    for index, unit in enumerate(contract.get("units", [])):
        checked = dict(unit)
        contextual_reading = str(checked.get("contextual_reading", checked.get("read", "")))
        locked_reading = str(checked.get("locked_reading", contextual_reading))
        expected_contextual_digest = stable_digest(contextual_reading)
        expected_locked_digest = stable_digest(locked_reading)
        contextual_digest_matches = checked.get("contextual_reading_digest", expected_contextual_digest) == expected_contextual_digest
        locked_digest_matches = checked.get("locked_reading_digest", expected_locked_digest) == expected_locked_digest
        checked["contextual_reading_digest"] = expected_contextual_digest
        checked["locked_reading_digest"] = expected_locked_digest
        unit_evidence = checked.get("accent_evidence")
        unit_evidence_matches = (
            isinstance(unit_evidence, Mapping)
            and "unit_evidence_sha256" in unit_evidence
            and unit_evidence["unit_evidence_sha256"] == _unit_evidence_digest(unit_evidence)
            and index < len(expected_unit_evidence)
            and unit_evidence == expected_unit_evidence[index]
        )
        if not contextual_digest_matches:
            checked["accent_evidence_valid"] = False
            checked["accent_evidence_invalid_reason"] = "contextual_reading_digest_mismatch"
        elif not locked_digest_matches:
            checked["accent_evidence_valid"] = False
            checked["accent_evidence_invalid_reason"] = "locked_reading_digest_mismatch"
        elif expected_locked_digest != expected_contextual_digest:
            checked["accent_evidence_valid"] = False
            checked["accent_evidence_invalid_reason"] = "locked_reading_changed"
        elif unit_evidence is None:
            checked["accent_evidence_valid"] = False
            checked["accent_evidence_invalid_reason"] = "contextual_accent_unavailable"
        elif not evidence_digest_matches or not unit_evidence_matches:
            checked["accent_evidence_valid"] = False
            checked["accent_evidence_invalid_reason"] = "accent_evidence_digest_mismatch"
        else:
            checked["accent_evidence_valid"] = True
            checked.pop("accent_evidence_invalid_reason", None)
        checked_units.append(checked)
    payload["units"] = checked_units
    return payload


def _load_arpa_pronunciations(path: str | None) -> dict[str, list[list[str]]]:
    if not path:
        return {}
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise JAContractError("dictionary_asset_missing", "configured English ARPA dictionary does not exist", str(candidate))
    result: dict[str, list[list[str]]] = {}
    for line in candidate.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        word = fields[0].lower()
        pronunciation = fields[1:]
        if len(pronunciation) >= 5 and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value) for value in pronunciation[:4]):
            pronunciation = pronunciation[4:]
        result.setdefault(word, []).append(pronunciation)
    return result


def _route_candidate(surface: str, caller_surface: str, settings: FrontendConfig, arpa: Mapping[str, list[list[str]]]) -> tuple[str, str, list[list[str]], str]:
    manual = (settings.manual_lexicon or {}).get(caller_surface, (settings.manual_lexicon or {}).get(surface))
    if isinstance(manual, Mapping):
        language = str(manual.get("language", "unresolved"))
        pronunciations = manual.get("pronunciation", manual.get("pronunciations", []))
        if isinstance(pronunciations, str):
            pronunciations = [pronunciations.split()]
        return language, "manual_lexicon", [list(item) for item in pronunciations or []], "manual"
    if isinstance(manual, str):
        return "ja", "manual_lexicon", [[manual]], "manual"
    if _contains_unsupported_markup(caller_surface):
        return "unresolved", "markup_unsupported", [], "unresolved"
    if caller_surface.isascii() and caller_surface.isalpha():
        if caller_surface.upper() in {"AI", "USB"}:
            return "unresolved", "latin_policy_required", [], "unresolved"
        pronunciations = arpa.get(caller_surface.lower(), [])
        if pronunciations:
            return "en", "matching_arpa_dictionary", pronunciations, "dictionary"
        return "unresolved", "english_oov_unresolved", [], "unresolved"
    return "ja", "verified_ja_frontend", [], "frontend_text_prediction"


def _contains_unsupported_markup(text: str) -> bool:
    lowered = text.lower()
    return ("<ruby" in lowered or "</ruby" in lowered or "${" in text or
            "《" in text or "》" in text or "｜" in text or
            bool(re.search(r"\{[^{}]+\}|<<[^<>]+>>", text)))


def _candidate_unit(row: Mapping[str, Any], index: int, layer: Mapping[str, Any], settings: FrontendConfig, arpa: Mapping[str, list[list[str]]], normalized_morph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    start, end = _span_row(row, len(layer["canonical_text"]), index)
    phones = list(row.get("phonemes") or [])
    ignored = bool(row.get("is_ignored", False))
    morph_punct = str(row.get("pos", "")) == "記号" or str(row.get("pos_group1", "")) == "記号"
    empty = not phones
    lexical_status = "ignored" if ignored or morph_punct else ("empty" if empty else "lexical")
    candidate_id = f"cand_{index:06d}"
    token_id = f"tok_{index:06d}"
    caller_surface = layer["canonical_text"][start:end]
    language, route, english_prons, route_evidence = _route_candidate(str(row.get("surface", "")), caller_surface, settings, arpa)
    if morph_punct:
        language, route, english_prons, route_evidence = "unresolved", "morph_punctuation", [], "non_lexical"
    contextual_candidates = [{
        "candidate_id": candidate_id,
        "reading": row.get("read", ""),
        "pronunciation": row.get("pron", ""),
        "phones": phones,
        "evidence_scope": "frontend_text_prediction",
        "provenance": "pyopenjtalk-plus",
    }]
    for pronunciation in english_prons:
        contextual_candidates.append({
            "candidate_id": f"{candidate_id}_en",
            "reading": caller_surface,
            "pronunciation": pronunciation,
            "phones": pronunciation,
            "evidence_scope": route,
            "provenance": settings.english_dictionary,
        })
    return {
        "token_id": token_id,
        "candidate_id": candidate_id,
        "candidate_ids": [candidate_id] + ([f"{candidate_id}_en"] if english_prons else []),
        "contextual_candidate_ids": [candidate_id] + ([f"{candidate_id}_en"] if english_prons else []),
        "contextual_candidates": contextual_candidates,
        "surface": row.get("surface", ""),
        "caller_surface": caller_surface,
        "orig": row.get("orig", row.get("surface", "")),
        "read": row.get("read", ""),
        "pron": row.get("pron", ""),
        "phones": phones,
        "canonical_span": [start, end],
        "orig_span": canonical_span_to_orig(layer, [start, end]),
        "domains": {"canonical": "canonical_text", "original": "orig_text"},
        "mecab_normalized_span": list(normalized_morph.get("char_span", [])) if normalized_morph and normalized_morph.get("char_span") else None,
        "mecab_normalized_span_domain": "mecab_normalized_text",
        "mecab_surface": normalized_morph.get("surface") if normalized_morph else None,
        "language": language,
        "route": route,
        "route_evidence": route_evidence,
        "english_pronunciations": english_prons,
        "english_dictionary": settings.english_dictionary,
        "pos": row.get("pos", ""),
        "pos_group1": row.get("pos_group1", ""),
        "pos_group2": row.get("pos_group2", ""),
        "pos_group3": row.get("pos_group3", ""),
        "mora_count": int(row.get("mora_count", 0) or 0),
        "accent_nucleus": int(row.get("accent_nucleus", 0) or 0),
        "morph_ignored": bool(row.get("is_ignored", False)),
        "morph_punct": morph_punct,
        "phoneme_empty": empty,
        "lexical_status": lexical_status if language != "unresolved" else "unresolved",
        "is_unknown": bool(row.get("is_unknown", False)),
        "source": {"provider": "pyopenjtalk-plus", "frontend_evidence": "text_prediction"},
    }


def _normalized_distribution_files(provider_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact, canonicalized installed provider file records.

    A build receipt is useful provenance, but it cannot identify an installed
    frontend by itself.  These records are collected inside the pinned bridge
    process and bind this evidence to its currently installed package bytes.
    """
    raw_files = provider_info.get("distribution_files")
    if not isinstance(raw_files, list) or not raw_files:
        raise JAContractError("frontend_capability_missing", "frontend accent identity lacks installed distribution file hashes")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(raw_files):
        if not isinstance(row, Mapping):
            raise JAContractError("frontend_capability_missing", "frontend distribution file record is malformed", f"$.distribution_files[{index}]")
        path = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if (not isinstance(path, str) or not path or type(size) is not int or size < 0
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
            raise JAContractError("frontend_capability_missing", "frontend distribution file record is malformed", f"$.distribution_files[{index}]")
        normalized.append({"path": path, "size": size, "sha256": digest.lower()})
    normalized.sort(key=lambda row: (row["path"], row["size"], row["sha256"]))
    if len({row["path"] for row in normalized}) != len(normalized):
        raise JAContractError("frontend_capability_missing", "frontend distribution file records have duplicate paths")
    return normalized


def _accent_provider_identity(settings: FrontendConfig, provider_info: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(settings.options)
    enabled_features = [name for name in ("run_marine", "use_tsqyomi", "predict_nani") if options[name]]
    provider_revision = provider_info.get("version")
    build_provenance = provider_info.get("build_provenance")
    if (not isinstance(provider_revision, str) or not provider_revision
            or not isinstance(build_provenance, Mapping)
            or build_provenance.get("status") not in {"verified", "unverified"}):
        raise JAContractError("frontend_capability_missing", "frontend accent identity lacks provider revision/build provenance")
    installed_distribution_files = _normalized_distribution_files(provider_info)
    installed_distribution_digest = stable_digest(installed_distribution_files)
    common_identity = {
        "provider_revision": provider_revision,
        "frontend_commit": settings.commit,
        "build_provenance_digest": stable_digest(build_provenance),
        "options_digest": stable_digest(options),
        "installed_distribution_digest": installed_distribution_digest,
    }
    if enabled_features:
        learned_identities = provider_info.get("learned_model_identities")
        if not isinstance(learned_identities, Mapping):
            raise JAContractError("frontend_capability_missing", "enabled learned frontend behavior lacks reopenable model identity")
        feature_assets: dict[str, dict[str, Any]] = {}
        for feature_name in enabled_features:
            asset = learned_identities.get(feature_name)
            if not isinstance(asset, Mapping):
                raise JAContractError("frontend_capability_missing", f"enabled learned feature {feature_name} lacks model asset identity")
            model_name = asset.get("model_name")
            model_revision = asset.get("model_revision")
            checkpoint = asset.get("checkpoint")
            artifact_sha256 = asset.get("artifact_sha256")
            if (not isinstance(model_name, str) or not model_name
                    or not (isinstance(model_revision, str) and model_revision
                            or isinstance(checkpoint, str) and checkpoint)
                    or not isinstance(artifact_sha256, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", artifact_sha256)):
                raise JAContractError("frontend_capability_missing", f"enabled learned feature {feature_name} has incomplete model asset identity")
            if any(asset.get(key) != value for key, value in common_identity.items()):
                raise JAContractError("frontend_capability_missing", f"enabled learned feature {feature_name} identity disagrees with installed frontend")
            feature_assets[feature_name] = {
                "model_name": model_name,
                **({"model_revision": model_revision} if isinstance(model_revision, str) and model_revision else {}),
                **({"checkpoint": checkpoint} if isinstance(checkpoint, str) and checkpoint else {}),
                "artifact_sha256": artifact_sha256.lower(),
                **common_identity,
            }
        model_identity = {
            "mode": "learned",
            "enabled_features": enabled_features,
            **common_identity,
            "feature_assets": feature_assets,
        }
    else:
        model_identity = {"mode": "rule_based", **common_identity}
    return {
        "provider": provider_info.get("provider", settings.provider),
        "provider_revision": provider_revision,
        "frontend_commit": settings.commit,
        "options": options,
        "options_digest": stable_digest(options),
        "build_provenance": dict(build_provenance),
        "build_provenance_digest": stable_digest(build_provenance),
        "installed_distribution_files": installed_distribution_files,
        "installed_distribution_digest": installed_distribution_digest,
        "model_identity": model_identity,
    }


def run_frontend(caller_text: str, config: Mapping[str, Any] | FrontendConfig | None = None, provider: Callable[..., Any] | None = None) -> dict[str, Any]:
    settings = config if isinstance(config, FrontendConfig) else FrontendConfig.from_mapping(config)
    layer = canonicalize_text(caller_text, settings.options["normalize_mode"])
    if provider is None:
        probe = probe_provider(settings)
        bridge = _run_bridge(settings.runtime_python, {"op": "mapping", "text": layer["canonical_text"], "options": {**settings.options, "normalize_mode": "None"}})
        rows = bridge.get("rows", [])
        provider_info = {key: bridge.get(key) for key in ("provider", "version", "module", "distributions", "capabilities", "distribution_files")}
        provider_info["build_provenance"] = probe.get("build_provenance")
        if _normalized_distribution_files(provider_info) != _normalized_distribution_files(probe):
            raise JAContractError("frontend_representation_drift", "probe and mapping bridge installed distribution files differ")
        accent_evidence = extract_contextual_accent_evidence(
            bridge.get("njd_rows", []), bridge.get("full_context_labels", []),
            _accent_provider_identity(settings, provider_info),
        )
    else:
        rows = provider(layer["canonical_text"], **{**settings.options, "normalize_mode": "None"})
        accent_evidence = None
        provider_info = {"provider": settings.provider, "version": "injected", "module": "injected", "distributions": [settings.provider], "capabilities": list(_CAPABILITIES)}
    if not isinstance(rows, list) or not rows:
        raise JAContractError("frontend_capability_missing", "frontend returned no mapping")
    # A provider may emit empty boundary/silence rows with ``(0, 0)``.  Drop
    # only genuinely empty rows; a non-empty lexical row at that span must be
    # rejected so it cannot disappear from the caller-bound analysis.
    mapped_rows = [(source_index, row) for source_index, row in enumerate(rows)
                   if not (row.get("char_span") == [0, 0] and not row.get("surface") and not row.get("phonemes"))]
    spans = [_span_row(row, len(layer["canonical_text"]), index) for index, (_, row) in enumerate(mapped_rows)]
    try:
        validate_monotonic_spans(spans, len(layer["canonical_text"]))
    except ValueError as exc:
        raise JAContractError("frontend_span_unbound", str(exc)) from exc
    arpa = _load_arpa_pronunciations(settings.english_dictionary)
    morphs = bridge.get("morphs", []) if provider is None else []
    units = [_candidate_unit(row, index, layer, settings, arpa, morphs[source_index] if source_index < len(morphs) else None)
             for index, (source_index, row) in enumerate(mapped_rows)]
    _attach_contextual_accent_evidence(units, accent_evidence)
    if _contains_unsupported_markup(caller_text):
        for unit in units:
            unit["language"] = "unresolved"
            unit["route"] = "markup_unsupported"
            unit["route_evidence"] = "unresolved"
            unit["lexical_status"] = "unresolved" if not unit.get("morph_ignored") else unit["lexical_status"]
            unit["markup_provenance"] = {"status": "unsupported", "source_domain": "caller_text"}
    payload = {
        "schema": "ja-frontend-contract-v2",
        "contract_version": "ja-frontend-contract-v2",
        "caller_text": caller_text,
        "orig_text": caller_text,
        "canonical_text": layer["canonical_text"],
        "text_layer": layer,
        "text_layer_digest": layer["text_layer_digest"],
        "canonical_sha256": layer["canonical_sha256"],
        "canonicalSHA": layer["canonical_sha256"],
        "normalization": {"mode": settings.options["normalize_mode"], "version": layer["version"], "lossy": layer["normalization"]["lossy"]},
        "normalization_version": layer["version"],
        "profile": "ja-en-frontend-v2",
        "frontend_commit": settings.commit,
        "frontend_identity": {**settings.identity(), "provider_info": provider_info},
        "options": dict(settings.options),
        "units": units,
        "tokenIDs": [unit["token_id"] for unit in units],
        "canonical_spans": [unit["canonical_span"] for unit in units],
        "languages": [unit["language"] for unit in units],
        "contextual_candidate_ids": [unit["contextual_candidate_ids"] for unit in units],
        "contextual_readings": [unit["read"] for unit in units],
        "contextual_phones": [unit["phones"] for unit in units],
        "contextual_accent_evidence": accent_evidence,
        "provenance": {"frontend": provider_info, "text_layer": layer["text_layer_digest"]},
        "analysis_provenance": "frontend_candidates_only_no_locked_reading",
    }
    payload["analysis_version"] = ANALYSIS_VERSION
    payload["frontend_profile"] = "ja-en-frontend-v2"
    payload["frontend_options_digest"] = stable_digest(dict(settings.options))
    payload["route_partition"] = {
        "ja": [unit["token_id"] for unit in units if unit["language"] == "ja"],
        "en": [unit["token_id"] for unit in units if unit["language"] == "en"],
        "unresolved": [unit["token_id"] for unit in units if unit["language"] == "unresolved"],
        "ignored": [unit["token_id"] for unit in units if unit["lexical_status"] in {"ignored", "empty"}],
    }
    payload["analysis_digest"] = stable_digest(payload)
    return validate_frontend_contract(payload)


def _reading_request_rows(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = request.get("locked_readings", request.get("readings", []))
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    if not isinstance(rows, list):
        raise JAContractError("reading_ambiguous", "locked_readings must be a list")
    return rows


def reconstruct_locked_reading(analysis: Mapping[str, Any], request: Mapping[str, Any], config: Mapping[str, Any] | FrontendConfig | None = None) -> dict[str, Any]:
    if analysis.get("schema") != "ja-frontend-contract-v2":
        raise JAContractError("schema_invalid", "analysis is not a frontend contract")
    if request.get("canonical_sha256", request.get("canonicalSHA")) != analysis.get("canonical_sha256") or request.get("analysis_digest", request.get("analysisDigest")) != analysis.get("analysis_digest"):
        raise JAContractError("frontend_representation_drift", "locked reading request digest does not match analysis")
    if request.get("analysis_version", request.get("analysisVersion")) != analysis.get("analysis_version"):
        raise JAContractError("frontend_representation_drift", "locked reading request version does not match analysis")
    requested_profile = request.get("frontend_profile", request.get("profile"))
    if requested_profile != analysis.get("frontend_profile"):
        raise JAContractError("frontend_representation_drift", "locked reading request profile does not match analysis")
    if request.get("frontend_options_digest", request.get("frontendOptionsDigest")) != analysis.get("frontend_options_digest"):
        raise JAContractError("frontend_representation_drift", "locked reading request options do not match analysis")
    settings = config if isinstance(config, FrontendConfig) else FrontendConfig.from_mapping(config)
    by_token = {unit["token_id"]: unit for unit in analysis.get("units", [])}
    by_candidate: dict[str, Mapping[str, Any]] = {}
    for unit in analysis.get("units", []):
        for candidate_id in unit.get("candidate_ids", [unit.get("candidate_id")]):
            by_candidate[str(candidate_id)] = unit
    requests = _reading_request_rows(request)
    if not requests:
        raise JAContractError("reading_ambiguous", "no locked reading rows")
    lock_by_token: dict[str, Mapping[str, Any]] = {}
    lock_by_candidate: set[str] = set()
    for index, row in enumerate(requests):
        token_id = row.get("token_id", row.get("tokenID"))
        candidate_id = row.get("candidate_id", row.get("candidateID"))
        reading = row.get("chosen_reading", row.get("chosenReading", row.get("reading")))
        if token_id not in by_token or candidate_id not in by_candidate or by_candidate[candidate_id]["token_id"] != token_id:
            raise JAContractError("frontend_representation_drift", "locked token/candidate id is unknown", f"$.locked_readings[{index}]")
        if not isinstance(reading, str) or not reading:
            raise JAContractError("reading_ambiguous", "chosen_reading is required", f"$.locked_readings[{index}]")
        if token_id in lock_by_token or candidate_id in lock_by_candidate:
            raise JAContractError("reading_ambiguous", "duplicate locked token/candidate id", f"$.locked_readings[{index}]")
        lock_by_token[token_id] = row
        lock_by_candidate.add(candidate_id)
    lexical_units = [unit for unit in analysis.get("units", []) if unit.get("lexical_status") == "lexical"]
    missing = [unit.get("token_id") for unit in lexical_units if unit.get("token_id") not in lock_by_token]
    if missing:
        raise JAContractError("reading_ambiguous", f"missing locked readings for lexical units: {missing!r}", "$.locked_readings")
    # Rebuild from the locked reading itself.  The surface candidate is never
    # sent back through default G2P, so a surface heteronym cannot overwrite a
    # selector decision.
    result_units: list[dict[str, Any]] = []
    for unit in analysis["units"]:
        row = lock_by_token.get(unit["token_id"])
        if row is None:
            result_units.append(dict(unit))
            continue
        reading = str(row.get("chosen_reading", row.get("reading")))
        selected_candidate_id = str(row.get("candidate_id"))
        if unit.get("language") == "en":
            # W1's frozen selector names dictionary output ``native_phones``;
            # accept that explicit spelling at this boundary as equivalent to
            # chosen_pronunciation, while retaining the locked native phones.
            pronunciation = row.get("chosen_pronunciation", row.get("pronunciation", row.get("native_phones", row.get("phones", []))))
            if isinstance(pronunciation, str):
                pronunciation = pronunciation.split()
            if not isinstance(pronunciation, list) or not pronunciation:
                raise JAContractError("reading_ambiguous", "English locked occurrence needs chosen_pronunciation", f"$.units[{unit['token_id']}]")
            updated = dict(unit)
            updated.update({
                "locked_reading": reading,
                "locked_english_phones": list(pronunciation),
                "locked_phones": list(pronunciation),
                "locked_native_phones": list(pronunciation),
                "locked_pronunciation": list(pronunciation),
                "locked_candidate_id": selected_candidate_id,
                "reading_provenance": row.get("evidence", row.get("provenance", "locked_reading_request")),
                "accent_prediction_valid": False,
                "locked_reading_digest": stable_digest(reading),
            })
            result_units.append(updated)
            continue
        rebuilt = run_frontend(reading, settings)
        lexical = [candidate for candidate in rebuilt["units"] if candidate["lexical_status"] == "lexical"]
        if not lexical:
            raise JAContractError("dictionary_roundtrip_failed", "locked reading produced no lexical phones", f"$.units[{unit['token_id']}]")
        locked_phones = [phone for candidate in lexical for phone in candidate["phones"]]
        locked_mora = sum(int(candidate.get("mora_count", 0) or 0) for candidate in lexical)
        updated = dict(unit)
        updated.update({
            "locked_reading": reading,
            "locked_openjtalk_phones": locked_phones,
            "locked_phones": locked_phones,
            "locked_mora_count": locked_mora,
            "locked_mora": [f"locked_mora_{index:04d}" for index in range(locked_mora)],
            "locked_candidate_id": selected_candidate_id,
            "reading_provenance": row.get("evidence", row.get("provenance", "locked_reading_request")),
            "accent_provenance": "regenerated_from_locked_reading" if reading != unit.get("read") else "frontend_text_prediction",
            "accent_prediction_valid": reading == unit.get("read"),
            "locked_reading_digest": stable_digest(reading),
        })
        if reading != unit.get("read"):
            updated["accent_provenance"] = "invalidated_by_locked_reading"
        result_units.append(updated)
    payload = dict(analysis)
    payload["units"] = result_units
    payload["analysis_provenance"] = "reconstructed_from_locked_reading_request"
    payload["locked_reading_request_digest"] = stable_digest(request)
    checked = validate_frontend_contract(payload)
    checked["reconstruction_digest"] = stable_digest({
        "canonical_sha256": checked["canonical_sha256"], "units": checked["units"],
    })
    return checked


def reconstruct_locked_readings(analysis: Mapping[str, Any], request: Mapping[str, Any], config: Mapping[str, Any] | FrontendConfig | None = None) -> dict[str, Any]:
    """Plural API alias used by W1 workers processing one utterance record."""
    return reconstruct_locked_reading(analysis, request, config)


def _kana(text: str) -> str:
    return "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヺ" else char for char in text)


def _compact_text(text: str) -> tuple[str, list[int]]:
    """Return lexical characters and their canonical-text offsets."""
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        chars.append(char)
        offsets.append(index)
    return "".join(chars), offsets


def bind_asr_candidates(analysis: Mapping[str, Any], asr_results: Sequence[Mapping[str, Any]] | Mapping[str, Any], frontend_config: Mapping[str, Any] | FrontendConfig | None = None) -> dict[str, Any]:
    """Bind raw ASR transcripts to W2 units without re-normalizing downstream.

    Matching is monotonic over compact lexical streams.  A provider transcript
    that cannot consume the exact stream, or has more than one possible match,
    remains unresolved.  The function only adds lexical evidence; it never
    changes a selected/locked reading.
    """
    if analysis.get("schema") != "ja-frontend-contract-v2":
        raise JAContractError("schema_invalid", "ASR binding needs a frontend contract")
    if not isinstance(analysis.get("analysis_digest"), str) or not analysis.get("analysis_version"):
        raise JAContractError("frontend_representation_drift", "candidate analysis digest/version is required")
    units = [dict(unit) for unit in analysis.get("units", [])]
    if isinstance(asr_results, Mapping):
        # Provider workers commonly write either a list directly or a receipt
        # with ``providers``/``results``.  Accept both while retaining each
        # provider's raw transcript unchanged in the binding receipt.
        asr_results = asr_results.get("providers", asr_results.get("results", ()))
    if not isinstance(asr_results, Sequence) or isinstance(asr_results, (str, bytes)):
        raise JAContractError("frontend_span_unbound", "ASR provider results must be a sequence")
    normalize_mode = str(analysis.get("options", {}).get("normalize_mode", "NFKC"))
    bindings: list[dict[str, Any]] = []
    for provider_index, provider in enumerate(asr_results):
        raw_text = provider.get("asr_text", provider.get("text", provider.get("transcript")))
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        asr_layer = canonicalize_text(raw_text, normalize_mode)
        asr_stream, asr_offsets = _compact_text(asr_layer["canonical_text"])
        cursor = 0
        provider_bindings: list[dict[str, Any]] = []
        provider_ambiguous = False
        for unit in units:
            if unit.get("lexical_status") not in {"lexical", "unresolved"}:
                continue
            if unit.get("language") != "ja" or unit.get("route") != "verified_ja_frontend":
                # Consume an exact non-Japanese surface as an alignment
                # anchor, while deliberately emitting no reading candidate.
                # This lets mixed utterances retain Japanese lexical evidence
                # without turning English/OOV/AI/USB text into a W2 reading.
                anchor_surface, _ = _compact_text(str(unit.get("caller_surface", unit.get("surface", ""))))
                if anchor_surface and asr_stream.startswith(anchor_surface, cursor):
                    cursor += len(anchor_surface)
                elif anchor_surface:
                    provider_ambiguous = True
                continue
            # Blind ASR can provide lexical support for a verified Japanese
            # occurrence.  It cannot turn an unresolved ASCII token (AI/USB,
            # an OOV name, or a borrowing without policy) into an English or
            # Japanese reading merely because the surface was echoed.
            source_surface = str(unit.get("caller_surface", unit.get("surface", "")))
            source_surface, _ = _compact_text(source_surface)
            source_reading = _kana(str(unit.get("read", "")))
            source_reading, _ = _compact_text(source_reading)
            choices: list[tuple[str, str]] = []
            if source_surface and asr_stream.startswith(source_surface, cursor):
                choices.append(("surface", source_surface))
            if source_reading and asr_stream.startswith(source_reading, cursor):
                choices.append(("kana", source_reading))
            if not choices:
                # A unique later occurrence may be bound only when it is the
                # sole candidate; otherwise the token is deliberately left
                # unresolved rather than substring-guessing its span.
                later = [
                    (kind, value, position)
                    for kind, value in (("surface", source_surface), ("kana", source_reading))
                    if value
                    for position in range(cursor + 1, len(asr_stream))
                    if asr_stream.startswith(value, position)
                ]
                if len(later) != 1:
                    provider_ambiguous = provider_ambiguous or bool(later)
                    continue
                kind, value, match_start = later[0]
            else:
                # Surface is preferred when both domains happen to match.
                kind, value = choices[0]
                match_start = cursor
            match_end = match_start + len(value)
            canonical_span = [asr_offsets[match_start], asr_offsets[match_end - 1] + 1]
            candidate_id = f"asr_{provider.get('provider', provider_index)}_{unit.get('token_id')}"
            evidence = {
                "candidate_id": candidate_id,
                "token_id": unit.get("token_id"),
                "provider": provider.get("provider", f"provider_{provider_index}"),
                "family": provider.get("family"),
                "raw_asr_text": raw_text,
                "asr_orig_text": asr_layer["orig_text"],
                "asr_canonical_text": asr_layer["canonical_text"],
                "asr_text_layer_digest": asr_layer["text_layer_digest"],
                "asr_canonical_span": canonical_span,
                "asr_orig_span": canonical_span_to_orig(asr_layer, canonical_span),
                "source_surface": unit.get("caller_surface", unit.get("surface", "")),
                "source_kana": unit.get("read", ""),
                "source_canonical_span": list(unit.get("canonical_span", [])),
                "source_orig_span": list(unit.get("orig_span", [])),
                "match_type": kind,
                "matched_text": value,
                "matched_kana": _kana(value) if kind == "kana" else _kana(str(unit.get("read", ""))),
                "contextual_reading": unit.get("read") if unit.get("language") == "ja" else unit.get("caller_surface"),
                "evidence_scope": "lexical_support",
                "provenance": "raw_asr_text_exact_stream",
                "is_same_surface_g2p": bool(provider.get("is_same_surface_g2p", False)),
                "shared_g2p_derivation": bool(provider.get("shared_g2p_derivation", False)),
            }
            provider_bindings.append(evidence)
            cursor = match_end
        if cursor != len(asr_stream):
            provider_ambiguous = True
        # Optional configured frontend projection handles a transcript whose
        # orthography is not directly surface/kana aligned. It emits a
        # token-bound candidate, never an origin confirmation. A one-token
        # utterance is safe without anchors; multi-token candidates require a
        # unique one-to-one gap between exact monotonic anchors.
        if provider_ambiguous and frontend_config is not None:
            try:
                projected = run_frontend(raw_text, frontend_config)
                source_units = [unit for unit in units if unit.get("lexical_status") == "lexical" and unit.get("language") == "ja" and unit.get("route") == "verified_ja_frontend"]
                projected_units = [unit for unit in projected.get("units", []) if unit.get("lexical_status") == "lexical" and unit.get("language") == "ja" and unit.get("route") == "verified_ja_frontend"]
                anchors: list[tuple[int, int]] = []
                projected_cursor = 0
                for source_index, source_unit in enumerate(source_units):
                    matches = [projected_index for projected_index in range(projected_cursor, len(projected_units))
                               if (str(source_unit.get("caller_surface", source_unit.get("surface", ""))) == str(projected_units[projected_index].get("surface", ""))
                                   or _kana(str(source_unit.get("read", ""))) == _kana(str(projected_units[projected_index].get("read", ""))))]
                    if len(matches) == 1:
                        anchors.append((source_index, matches[0]))
                        projected_cursor = matches[0] + 1
                gaps: list[tuple[int, int, int, int]] = []
                boundaries = [(-1, -1), *anchors, (len(source_units), len(projected_units))]
                for (left_source, left_projected), (right_source, right_projected) in zip(boundaries, boundaries[1:]):
                    if right_source - left_source - 1 == 1 and right_projected - left_projected - 1 == 1:
                        gaps.append((left_source + 1, right_source, left_projected + 1, right_projected))
                if len(source_units) == 1 and len(projected_units) == 1 and not anchors:
                    gaps = [(0, 1, 0, 1)]
                if gaps:
                    provider_bindings = []
                    for source_start, source_end, projected_start, projected_end in gaps:
                        source_unit = source_units[source_start]
                        projected_unit = projected_units[projected_start]
                        span = list(projected_unit.get("canonical_span", []))
                        if len(span) != 2 or span[1] <= span[0]:
                            raise JAContractError("frontend_span_unbound", "projected ASR unit has no non-empty span")
                        evidence = {
                            "candidate_id": f"asr_{provider.get('provider', provider_index)}_{source_unit.get('token_id')}_candidate",
                            "token_id": source_unit.get("token_id"),
                            "provider": provider.get("provider", f"provider_{provider_index}"),
                            "family": provider.get("family"),
                            "raw_asr_text": raw_text,
                            "asr_orig_text": asr_layer["orig_text"],
                            "asr_canonical_text": asr_layer["canonical_text"],
                            "asr_text_layer_digest": asr_layer["text_layer_digest"],
                            "asr_canonical_span": span,
                            "asr_orig_span": canonical_span_to_orig(asr_layer, span),
                            "source_surface": source_unit.get("caller_surface", source_unit.get("surface", "")),
                            "source_kana": source_unit.get("read", ""),
                            "source_canonical_span": list(source_unit.get("canonical_span", [])),
                            "source_orig_span": list(source_unit.get("orig_span", [])),
                            "match_type": "kana_candidate",
                            "matched_text": asr_layer["canonical_text"][span[0]:span[1]],
                            "matched_kana": _kana(str(projected_unit.get("read", ""))),
                            "contextual_reading": projected_unit.get("read"),
                            "candidate_phones": list(projected_unit.get("phones", [])),
                            "evidence_scope": "phonetic_transcription",
                            "provenance": "raw_asr_text_frontend_projection",
                            "asr_frontend_analysis_digest": projected.get("analysis_digest"),
                            "is_same_surface_g2p": False,
                            "shared_g2p_derivation": False,
                        }
                        provider_bindings.append(evidence)
                    provider_ambiguous = False
                    unresolved_candidates = []
            except (JAContractError, OSError, ValueError):
                # A failed optional projection is deliberately unresolved.
                pass
        if provider_ambiguous:
            unresolved_candidates = provider_bindings
            provider_bindings = []
        else:
            unresolved_candidates = []
        bindings.append({
            "provider": provider.get("provider", f"provider_{provider_index}"),
            "family": provider.get("family"),
            "raw_asr_text": raw_text,
            "asr_text_layer": asr_layer,
            "status": "unresolved" if provider_ambiguous else "bound",
            "candidates": provider_bindings,
            "unresolved_candidates": unresolved_candidates,
        })
        for evidence in provider_bindings:
            token_id = evidence["token_id"]
            for unit in units:
                if unit.get("token_id") != token_id:
                    continue
                unit.setdefault("asr_candidates", []).append(evidence)
                unit.setdefault("asr_candidate_ids", []).append(evidence["candidate_id"])
                if evidence["match_type"] in {"surface", "kana"}:
                    unit["origin_lexical_match"] = True
                    unit["origin_surface_match"] = unit.get("origin_surface_match", False) or evidence["match_type"] == "surface"
                    unit["origin_kana_match"] = unit.get("origin_kana_match", False) or evidence["match_type"] == "kana"
                    unit["origin_surface_confirmed"] = True
                    unit["origin_lexical_status"] = "origin_surface_confirmed"
                    unit["contextual_reading"] = evidence["contextual_reading"]
                break
    # A family can contribute at most one consensus vote.  Keep this summary
    # beside the occurrence so W1 can pass a unit view to its frozen selector
    # without reimplementing span matching or normalisation.
    for unit in units:
        family_readings: dict[str, set[str]] = {}
        for evidence in unit.get("asr_candidates", []):
            family = str(evidence.get("family") or evidence.get("provider") or "").strip()
            reading = str(evidence.get("contextual_reading") or "").strip()
            if family and reading:
                family_readings.setdefault(reading, set()).add(family)
        consensus = next(((reading, families) for reading, families in family_readings.items() if len(families) >= 2), None)
        if consensus is not None:
            reading, families = consensus
            unit["asr_family_consensus"] = True
            unit["asr_consensus_reading"] = reading
            unit["asr_consensus_families"] = sorted(families)
        else:
            unit["asr_family_consensus"] = False
    enriched = dict(analysis)
    enriched["units"] = units
    enriched["asr_bindings"] = bindings
    enriched["asr_binding_digest"] = stable_digest(bindings)
    # The W2 candidate-analysis identity is the lock boundary consumed by W1
    # and reconstruction. ASR evidence is a derived namespace and gets its
    # own digest; it must never rewrite the original analysis_digest.
    enriched["source_analysis_digest"] = analysis.get("analysis_digest")
    enriched["asr_analysis_digest"] = stable_digest({key: value for key, value in enriched.items() if key not in {"analysis_digest", "asr_analysis_digest"}})
    return enriched


def unit_candidate_analysis(analysis: Mapping[str, Any], token_id: str) -> dict[str, Any]:
    """Project one bound unit into the selector's frozen analysis view."""
    unit = next((value for value in analysis.get("units", []) if value.get("token_id") == token_id), None)
    if unit is None:
        raise JAContractError("frontend_span_unbound", f"unknown token_id {token_id!r}")
    view = dict(analysis)
    view.update({
        "token_id": token_id,
        "contextual_reading": unit.get("contextual_reading"),
        "origin_reading": unit.get("contextual_reading"),
        "origin_surface_match": bool(unit.get("origin_surface_match")),
        "origin_kana_match": bool(unit.get("origin_kana_match")),
        "origin_lexical_match": bool(unit.get("origin_lexical_match")),
        "origin_lexical_status": unit.get("origin_lexical_status"),
        "asr_family_consensus": bool(unit.get("asr_family_consensus")),
        "asr_consensus_reading": unit.get("asr_consensus_reading"),
        "asr_consensus_families": list(unit.get("asr_consensus_families", [])),
        "candidate_ids": unit.get("candidate_ids", []),
        "asr_candidate_ids": unit.get("asr_candidate_ids", []),
    })
    return view


def project_blind_asr_candidates(analysis: Mapping[str, Any], providers: Sequence[Mapping[str, Any]] | Mapping[str, Any], frontend_config: Mapping[str, Any] | FrontendConfig | None = None) -> dict[str, dict[str, Any]]:
    """Expose W1's projection hook using W2-bound occurrence evidence.

    ``frontend_config`` is accepted for the shared W1/W2 hook contract.  Raw
    transcript normalisation and caller span binding always come from this
    module's analysis identity; the selector must not create a second text
    layer from that config.
    """
    bound = bind_asr_candidates(analysis, providers, frontend_config=frontend_config)
    projected: dict[str, dict[str, Any]] = {}
    for unit in bound.get("units", []):
        evidence = [dict(row) for row in unit.get("asr_candidates", []) if isinstance(row, Mapping)]
        if not evidence:
            continue
        candidates = []
        for row in evidence:
            candidates.append({
                # The selector binds this candidate to the original W2
                # occurrence. ``evidence_candidate_id`` remains unique per
                # provider while candidate_id is the occurrence contract used
                # by the current selector's allow-list.
                "token_id": unit.get("token_id"),
                "candidate_id": unit.get("candidate_id"),
                "evidence_candidate_id": row.get("candidate_id"),
                "reading": row.get("contextual_reading"),
                "phones": list(row.get("candidate_phones", [])),
                "origin_reading": row.get("contextual_reading"),
                "provider": row.get("provider"),
                "family": row.get("family"),
                "match_kind": row.get("match_type"),
                "canonical_span": row.get("asr_canonical_span"),
                "orig_span": row.get("asr_orig_span"),
                "provenance": row.get("provenance"),
                "evidence_scope": row.get("evidence_scope"),
                "is_same_surface_g2p": row.get("is_same_surface_g2p", False),
                "shared_g2p_derivation": row.get("shared_g2p_derivation", False),
            })
        projected[str(unit.get("token_id"))] = {
            "origin_reading": unit.get("contextual_reading"),
            "origin_match": bool(unit.get("origin_lexical_match")),
            "match_kind": ("surface" if unit.get("origin_surface_match") else ("kana" if unit.get("origin_kana_match") else "kana_candidate")),
            "provider": evidence[0].get("provider"),
            "family": evidence[0].get("family"),
            "lexical_support": True,
            "candidates": candidates,
            "candidate_ids": list(unit.get("asr_candidate_ids", [])),
            "canonical_span": list(unit.get("canonical_span", [])),
            "orig_span": list(unit.get("orig_span", [])),
            "analysis_digest": bound.get("analysis_digest"),
            "analysis_version": bound.get("analysis_version"),
            "frontend_profile": bound.get("frontend_profile"),
        }
    return projected


project_asr_candidates = project_blind_asr_candidates
analyze_blind_asr = project_blind_asr_candidates


def _manifest_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(config["input_manifest"])).expanduser()
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("items", value) if isinstance(value, Mapping) else value
    return rows


def _write_immutable_json(path: Path, payload: Mapping[str, Any], workspace: Path) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JAContractError("frontend_representation_drift", "existing frontend artifact is not valid JSON", str(path)) from exc
        if existing != payload:
            raise JAContractError("frontend_representation_drift", "frontend artifact is immutable and differs", str(path))
        return
    atomic_write_json(path, payload, workspace=workspace)


def _locked_requests(config: Mapping[str, Any], stage_dir: Path) -> dict[str, dict[str, Any]]:
    frontend_cfg = config.get("frontend", {}) if isinstance(config.get("frontend"), Mapping) else {}
    configured = frontend_cfg.get("locked_readings_path", config.get("locked_readings_path"))
    candidates = [Path(str(configured))] if configured else [stage_dir.parent / "reading" / "locked_readings.json", stage_dir.parent / "reading" / "locked_readings.jsonl"]
    path = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
    if path is None:
        return {}
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            uid = str(row.get("uid", ""))
            grouped.setdefault(uid, {key: row[key] for key in ("canonical_sha256", "analysis_digest", "analysis_version", "frontend_profile", "frontend_options_digest") if key in row})
            grouped[uid].setdefault("locked_readings", []).append(row)
        return grouped
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, Mapping) else payload
    if isinstance(records, Mapping):
        return {str(uid): dict(request) for uid, request in records.items()}
    return {str(row.get("uid", "")): dict(row) for row in records if isinstance(row, Mapping)}


def _asr_results_by_uid(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load optional raw ASR receipts for the frontend stage.

    The frontend stage may run after a provider stage, but W2 owns the only
    text normalisation and span binding.  This helper accepts an explicit
    config path or per-manifest ``asr_results``; it never discovers files by
    scanning a workspace.
    """
    frontend_cfg = config.get("frontend", {}) if isinstance(config.get("frontend"), Mapping) else {}
    configured = frontend_cfg.get("asr_results_path", config.get("asr_results_path"))
    if not configured:
        return {}
    path = Path(str(configured)).expanduser()
    if not path.is_file():
        raise JAContractError("frontend_span_unbound", f"configured ASR results path is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload.get("results", payload)) if isinstance(payload, Mapping) else payload
    if isinstance(records, Mapping):
        return {str(uid): value for uid, value in records.items()}
    if isinstance(records, list):
        return {str(row.get("uid", row.get("id", ""))): row.get("providers", row.get("asr_results", row.get("results", [])))
                for row in records if isinstance(row, Mapping) and row.get("uid", row.get("id")) is not None}
    raise JAContractError("frontend_span_unbound", "configured ASR results must be a list or record map")


def frontend_stage(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    stage_dir.mkdir(parents=True, exist_ok=True)
    settings = FrontendConfig.from_mapping(config)
    analyses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        configured_asr = _asr_results_by_uid(config)
    except JAContractError as exc:
        receipt_path = stage_dir / "receipt.json"
        receipt = make_receipt(stage="frontend", status="BLOCKED", params={"frontend": settings.identity()}, errors=[exc.as_dict()])
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult("frontend", "BLOCKED", str(receipt_path))
    try:
        probe = probe_provider(settings)
    except JAContractError as exc:
        receipt_path = stage_dir / "receipt.json"
        receipt = make_receipt(stage="frontend", status="BLOCKED", params={"frontend": settings.identity()}, errors=[exc.as_dict()])
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult("frontend", "BLOCKED", str(receipt_path))
    for row in _manifest_rows(config):
        uid = str(row.get("uid", row.get("id", "")))
        text = row.get("text", row.get("script", ""))
        try:
            analysis = run_frontend(str(text), settings)
            analysis["uid"] = uid
            raw_asr = row.get("asr_results", row.get("providers"))
            if raw_asr is None:
                raw_asr = configured_asr.get(uid)
            if raw_asr:
                analysis = bind_asr_candidates(analysis, raw_asr)
                analysis["uid"] = uid
            analyses.append(analysis)
        except JAContractError as exc:
            errors.append({"uid": uid, **exc.as_dict()})
    output = stage_dir / "frontend_analysis.json"
    payload = {"schema": "ja-frontend-contract-v2", "frontend_identity": settings.identity(), "provider_probe": probe, "records": analyses, "errors": errors}
    _write_immutable_json(output, payload, stage_dir.parent.parent)
    outputs: list[Path] = [output]
    requests = _locked_requests(config, stage_dir)
    if requests and analyses:
        reconstructions: list[dict[str, Any]] = []
        reconstruction_errors: list[dict[str, Any]] = []
        for analysis in analyses:
            request = requests.get(str(analysis.get("uid")))
            if request is None:
                reconstruction_errors.append({"uid": analysis.get("uid"), "code": "reading_ambiguous", "message": "missing LockedReadings request"})
                continue
            try:
                reconstructions.append(reconstruct_locked_reading(analysis, request, settings))
            except JAContractError as exc:
                reconstruction_errors.append({"uid": analysis.get("uid"), **exc.as_dict()})
        reconstruction_path = stage_dir / "frontend_reconstruction.json"
        _write_immutable_json(reconstruction_path, {"schema": "ja-frontend-contract-v2", "records": reconstructions, "errors": reconstruction_errors}, stage_dir.parent.parent)
        outputs.append(reconstruction_path)
        errors.extend(reconstruction_errors)
    receipt_path = stage_dir / "receipt.json"
    status = "COMPLETE" if analyses and not errors else ("PARTIAL" if analyses else "BLOCKED")
    receipt = make_receipt(stage="frontend", status=status, outputs=outputs, params={"frontend": settings.identity()}, tools=["pyopenjtalk-plus"], errors=errors)
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult("frontend", status, str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("frontend", frontend_stage, output_namespace="frontend")


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--text")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe_provider(), ensure_ascii=False, indent=2))
    elif args.text is not None:
        print(json.dumps(run_frontend(args.text), ensure_ascii=False, indent=2))
    else:
        parser.error("--probe or --text is required")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "DEFAULT_FRONTEND_OPTIONS", "FrontendConfig", "FrontendCandidateAnalysis", "frontend_stage", "probe_provider",
    "bind_asr_candidates", "extract_contextual_accent_evidence", "reconstruct_locked_reading", "reconstruct_locked_readings", "register_stages", "run_frontend", "unit_candidate_analysis", "validate_frontend_contract",
    "project_blind_asr_candidates", "project_asr_candidates", "analyze_blind_asr",
]


FrontendCandidateAnalysis = dict
