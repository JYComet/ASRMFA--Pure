#!/usr/bin/env python3
"""Provider-neutral Japanese ASR cross-validation and reading selection.

Only the provider worker knows how to load a model.  This module keeps raw
provider evidence and applies a family-capped, deterministic reading policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ASR_PROFILES = {
    "qwen_only_dev": ("qwen",),
    "baseline_3family": ("qwen", "whisper", "reazon"),
    "extended_5model": ("qwen", "whisper", "reazon", "kotoba", "reazon_k2"),
}
PROFILE_PROVIDERS = {
    "qwen_only_dev": ("qwen3-asr",),
    "baseline_3family": ("qwen3-asr", "whisper-large-v3", "reazonspeech-nemo-v2"),
    "extended_5model": ("qwen3-asr", "whisper-large-v3", "reazonspeech-nemo-v2", "kotoba-v2", "reazonspeech-k2-v2"),
}
FAMILY_BY_PROVIDER = {
    "qwen3-asr": "qwen", "whisper-large-v3": "whisper", "kotoba-v2": "whisper",
    "reazonspeech-nemo-v2": "reazon", "reazonspeech-k2-v2": "reazon",
}


@dataclass(frozen=True)
class ASRProfile:
    name: str
    families: tuple[str, ...]
    production: bool


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    family: str
    status: str
    asr_text: str | None = None
    raw_stdout: str = ""
    runtime: str | None = None
    model_revision: str | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "family": self.family, "status": self.status,
            "asr_text": self.asr_text, "raw_stdout": self.raw_stdout,
            "runtime": self.runtime, "model_revision": self.model_revision,
            "candidates": [dict(row) for row in self.candidates],
            "error": self.error,
        }


def validate_profile(name: str, *, available_families: set[str] | None = None) -> ASRProfile:
    if name not in ASR_PROFILES:
        raise ValueError(f"unsupported ASR profile: {name}")
    families = ASR_PROFILES[name]
    if available_families is not None and name == "baseline_3family" and not set(families).issubset(available_families):
        raise ValueError("baseline_3family requires three ASR families")
    return ASRProfile(name=name, families=families, production=name != "qwen_only_dev")


def _parse_provider_output(stdout: str) -> tuple[str, tuple[dict[str, Any], ...]]:
    text = stdout.strip()
    if not text:
        raise ValueError("provider returned empty output")
    try:
        payload = json.loads(text.splitlines()[-1])
    except json.JSONDecodeError:
        return text, ()
    if isinstance(payload, Mapping):
        value = payload.get("asr_text", payload.get("text", payload.get("transcript")))
        if isinstance(value, str) and value.strip():
            candidates = payload.get("candidates", ())
            candidate_rows = tuple(dict(item) for item in candidates if isinstance(item, Mapping)) if isinstance(candidates, Sequence) else ()
            return value.strip(), candidate_rows
    raise ValueError("provider JSON must contain non-empty text/asr_text/transcript")


def run_provider_command(command: Sequence[str] | str, wav: str | Path, *, family: str,
                         provider: str, timeout_s: float = 300.0, runtime: str | None = None,
                         model_revision: str | None = None) -> ProviderResult:
    """Run a local provider protocol: command receives only the WAV path.

    No transcript, reading, prompt or user text is appended to the command.
    The worker may return JSON ``{"text": ...}`` or plain text on stdout.
    """
    argv = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    if not argv:
        return ProviderResult(provider, family, "failed", runtime=runtime, model_revision=model_revision,
                              error={"reason": "empty_command"})
    argv = [*argv, "--wav", str(Path(wav).expanduser().absolute())]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              check=False, env=os.environ.copy())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProviderResult(provider, family, "failed", runtime=runtime, model_revision=model_revision,
                              error={"reason": type(exc).__name__, "message": str(exc)})
    if proc.returncode:
        return ProviderResult(provider, family, "failed", raw_stdout=proc.stdout[-8192:], runtime=runtime,
                              model_revision=model_revision,
                              error={"reason": "provider_exit", "returncode": proc.returncode,
                                     "stderr": proc.stderr[-8192:]})
    try:
        asr_text, candidates = _parse_provider_output(proc.stdout)
    except ValueError as exc:
        return ProviderResult(provider, family, "failed", raw_stdout=proc.stdout[-8192:], runtime=runtime,
                              model_revision=model_revision,
                              error={"reason": "invalid_output", "message": str(exc)})
    return ProviderResult(provider, family, "ok", asr_text=asr_text, raw_stdout=proc.stdout[-8192:],
                          runtime=runtime, model_revision=model_revision, candidates=candidates)


def _candidate_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = result.get("candidates")
    if isinstance(candidates, str):
        return [{"reading": candidates}]
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        rows = []
        for item in candidates:
            if isinstance(item, str):
                rows.append({"reading": item})
            elif isinstance(item, Mapping):
                reading = item.get("reading") or item.get("kana") or item.get("kana_candidate")
                if isinstance(reading, str) and reading:
                    row = dict(item); row.setdefault("reading", reading); rows.append(row)
        return rows
    candidate = result.get("candidate")
    if isinstance(candidate, str) and candidate:
        return [{"reading": candidate}]
    if isinstance(candidate, Mapping):
        reading = candidate.get("reading") or candidate.get("kana") or candidate.get("kana_candidate")
        if isinstance(reading, str) and reading:
            row = dict(candidate); row.setdefault("reading", reading); return [row]
    # Raw utterance text is retained for audit but is never a token reading
    # candidate. W2 must provide span-bound kana/phonetic projections.
    return []


def _analysis_unit(analysis: Mapping[str, Any], token_id: str | None) -> Mapping[str, Any] | None:
    if not token_id or not isinstance(analysis.get("units"), list):
        return None
    return next((unit for unit in analysis["units"] if isinstance(unit, Mapping) and unit.get("token_id") == token_id), None)


def family_vote_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Return candidate -> distinct family set; one provider per family counts."""
    votes: dict[str, set[str]] = {}
    seen_family: set[tuple[str, str]] = set()
    for row in rows:
        family = str(row.get("family", "")).strip()
        provider = str(row.get("provider", "")).strip()
        if family not in {"qwen", "whisper", "reazon"} or provider not in {
                "qwen3-asr", "whisper-large-v3", "kotoba-v2",
                "reazonspeech-nemo-v2", "reazonspeech-k2-v2"}:
            continue
        if not family or str(row.get("status", "ok")) not in {"ok", "success", "complete"}:
            continue
        for candidate in _candidate_rows(row):
            reading = str(candidate.get("reading", "")).strip()
            if not reading:
                continue
            key = (family, reading)
            if key in seen_family:
                continue
            seen_family.add(key)
            votes.setdefault(reading, set()).add(family)
    return votes


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def _analysis_digest(analysis: Mapping[str, Any]) -> tuple[str, str]:
    digest = analysis.get("canonical_sha256", analysis.get("canonicalSHA", analysis.get("canonical_text_sha256")))
    version = (analysis.get("analysis_version") or analysis.get("version") or
               analysis.get("normalization_version") or analysis.get("profile") or analysis.get("schema"))
    if not isinstance(digest, str) or not digest or not isinstance(version, str) or not version:
        raise ValueError("candidate analysis requires canonical SHA and analysis version")
    return digest, version


def _projection_evidence(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("_projection_evidence")
    return value if isinstance(value, Mapping) else {}


def _lexical_evidence(row: Mapping[str, Any], *, source_id: Any = None) -> dict[str, Any]:
    projection = _projection_evidence(row)
    evidence: dict[str, Any] = {
        "source": "origin_lexical",
        "evidence_scope": "lexical_support",
        "surface_match": True,
    }
    resolved_source_id = source_id if source_id is not None else row.get("source_id")
    if resolved_source_id is not None:
        evidence["source_id"] = resolved_source_id
    for key in ("provider", "family", "raw_asr_text", "raw_asr_span", "artifact_path", "artifact_sha256",
                "raw_stdout_sha256", "match_kind", "canonical_span", "orig_span"):
        value = projection.get(key)
        if value is not None:
            evidence[key] = value
    provider_evidence = projection.get("provider_evidence")
    if isinstance(provider_evidence, list):
        evidence["provider_evidence"] = [dict(item) for item in provider_evidence if isinstance(item, Mapping)]
    return evidence


def select_reading(row: Mapping[str, Any], asr_results: Sequence[Mapping[str, Any]],
                   analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fixed manual → origin → family consensus → medoid → none policy."""
    digest, version = _analysis_digest(analysis)
    analysis_digest = analysis.get("analysis_digest", analysis.get("analysis_version_digest"))
    if isinstance(analysis.get("units"), list) and not isinstance(analysis_digest, str):
        raise ValueError("candidate analysis requires analysis_digest")
    uid = str(row.get("uid", ""))
    bound_unit = _analysis_unit(analysis, str(row.get("token_id")))
    bound_candidate_id = row.get("candidate_id") or (bound_unit.get("candidate_id") if bound_unit else None)
    base = {"schema": "ja-reading-selection-v2", "uid": uid,
            "canonical_sha256": digest, "analysis_version": version,
            "canonicalSHA": digest,
            "analysis_profile": analysis.get("profile", analysis.get("analysis_profile")),
            "analysis_version_digest": analysis.get("version_digest", analysis.get("analysis_digest")),
            "analysis_digest": analysis_digest,
            "token_id": row.get("token_id", analysis.get("token_id")),
            "selector_version": "ja-reading-selector-v1", "candidates": []}
    override = row.get("reading_override")
    if isinstance(override, str) and override and isinstance(row.get("source_id"), str) and row["source_id"]:
        return {**base, "status": "manual_verified", "selected_reading": override,
                "chosen_reading": override, "candidate_id": bound_candidate_id, "source_id": row["source_id"],
                "confidence": "manual", "evidence_scope": "manual_override", "acoustic_reading_proof": False,
                "evidence": [{"source": "manual", "source_id": row["source_id"],
                              "evidence_scope": "manual_override"}]}
    if bound_unit and str(bound_unit.get("language", "")) == "en":
        dictionary_candidates = [candidate for candidate in bound_unit.get("contextual_candidates", [])
                                 if isinstance(candidate, Mapping) and candidate.get("pronunciation")
                                 and str(candidate.get("evidence_scope", "")).endswith("dictionary")]
        requested_id = row.get("english_candidate_id") or row.get("pronunciation_candidate_id")
        if isinstance(row.get("pronunciation_override"), Mapping):
            requested_id = row["pronunciation_override"].get("candidate_id")
        chosen_candidate = next((candidate for candidate in dictionary_candidates if candidate.get("candidate_id") == requested_id), None) if requested_id else (dictionary_candidates[0] if len(dictionary_candidates) == 1 else None)
        if chosen_candidate is not None:
            reading = str(chosen_candidate.get("reading", bound_unit.get("surface", "")))
            phones = list(chosen_candidate.get("pronunciation") or chosen_candidate.get("phones") or [])
            return {**base, "status": "origin_surface_confirmed", "selected_reading": reading,
                    "chosen_reading": reading, "candidate_id": chosen_candidate.get("candidate_id"),
                    "native_phones": phones, "chosen_pronunciation": phones, "pronunciation": phones,
                    "confidence": "dictionary_policy",
                    "selection_basis": "dictionary_policy", "evidence_scope": "dictionary_policy",
                    "shared_g2p_derivation": False, "acoustic_reading_proof": False,
                    "evidence": [{"source": "english_dictionary", "candidate_id": chosen_candidate.get("candidate_id"),
                                  "evidence_scope": "dictionary_policy"}]} 
    origin = (analysis.get("contextual_reading") or analysis.get("origin_reading") or
              row.get("origin_reading") or row.get("origin_lexical_reading"))
    origin_match = any(analysis.get(key) is True for key in ("origin_surface_match", "origin_kana_match", "origin_lexical_match")) or row.get("origin_match") is True
    if isinstance(origin, str) and origin and origin_match:
        projection = _projection_evidence(row)
        return {**base, "status": "origin_surface_confirmed", "selected_reading": origin,
                "chosen_reading": origin, "confidence": "lexical_support",
                "candidate_id": bound_candidate_id,
                "shared_g2p_derivation": True, "evidence_scope": "lexical_support",
                "acoustic_reading_proof": False, "selection_basis": "origin_lexical",
                "evidence": [_lexical_evidence(row, source_id=projection.get("source_id"))]}
    token_id = row.get("token_id")
    units = analysis.get("units") if isinstance(analysis.get("units"), list) else []
    if token_id and units:
        unit = _analysis_unit(analysis, str(token_id))
        allowed_ids = set()
        if unit:
            allowed_ids.update(str(value) for value in (unit.get("candidate_ids") or []))
            allowed_ids.update(str(value) for value in (unit.get("asr_candidate_ids") or []))
        asr_results = [dict(result, candidates=[candidate for candidate in _candidate_rows(result)
                           if candidate.get("token_id") == token_id
                           and (not allowed_ids or not candidate.get("candidate_id") or str(candidate.get("candidate_id")) in allowed_ids)]) for result in asr_results]
    votes = family_vote_rows(asr_results)
    candidate_rows = sorted(votes)
    base["candidates"] = [{"reading": candidate, "families": sorted(votes[candidate])} for candidate in candidate_rows]
    consensus = [candidate for candidate in candidate_rows if len(votes[candidate]) >= 2]
    if consensus:
        chosen = sorted(consensus)[0]
        selected_ids = [c.get("candidate_id") for result in asr_results if result.get("family") in votes[chosen]
                        for c in _candidate_rows(result) if c.get("reading") == chosen and c.get("candidate_id")]
        chosen_candidates = [c for result in asr_results if result.get("family") in votes[chosen]
                             for c in _candidate_rows(result) if c.get("reading") == chosen]
        shared = bool(chosen_candidates) and all(
            bool(c.get("is_same_surface_g2p") or c.get("shared_default_g2p") or
                 c.get("shared_g2p_derivation")) for c in chosen_candidates)
        consensus_scope = "lexical_family_support" if shared else "acoustic_candidate"
        return {**base, "status": "asr_family_consensus", "selected_reading": chosen,
                "chosen_reading": chosen,
                "confidence": "lexical_family_consensus" if shared else "family_consensus",
                "candidate_id": sorted(selected_ids)[0] if selected_ids else None,
                "shared_g2p_derivation": shared, "evidence_scope": consensus_scope,
                "acoustic_reading_proof": not shared,
                "evidence": [{"families": sorted(votes[chosen]), "evidence_scope": consensus_scope,
                              "acoustic_reading_proof": not shared}]}
    if candidate_rows:
        medoid = min(candidate_rows, key=lambda candidate: (sum(_edit_distance(candidate, other) for other in candidate_rows), candidate))
        return {**base, "status": "asr_medoid", "selected_reading": None, "chosen_reading": medoid,
                "candidate_id": None, "confidence": "diagnostic_only", "production_verified": False,
                "evidence_scope": "diagnostic_only", "acoustic_reading_proof": False,
                "evidence": [{"diagnostic": "medoid", "families": sorted(votes[medoid])}]}
    return {**base, "status": "none", "selected_reading": None, "chosen_reading": None,
            "candidate_id": None, "confidence": "unresolved", "production_verified": False, "evidence": []}


def provider_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    section = config.get("asr", {}) or {}
    profile = validate_profile(str(section.get("profile", "qwen_only_dev")))
    specs: list[dict[str, Any]] = []
    configured = section.get("providers", {}) or {}
    root = Path(__file__).resolve().parent
    for provider in PROFILE_PROVIDERS[profile.name]:
        family = FAMILY_BY_PROVIDER[provider]
        raw = configured.get(provider, {}) if isinstance(configured, Mapping) else {}
        raw = raw if isinstance(raw, Mapping) else {}
        runtime = raw.get("runtime") or section.get(f"{provider}_runtime") or section.get(f"{family}_runtime") or os.environ.get("JA_ASR_RUNTIME")
        model = raw.get("model") or section.get(f"{provider}_model") or section.get(f"{family}_model") or os.environ.get(f"JA_ASR_{family.upper()}_MODEL")
        if provider == "qwen3-asr":
            model = model or section.get("qwen_model") or os.environ.get("JA_ASR_QWEN_MODEL")
        command = raw.get("command") or section.get(f"{provider}_command") or os.environ.get(f"JA_ASR_{family.upper()}_COMMAND")
        if not command and model:
            runtime = runtime or sys.executable
            command = [runtime, str(root / "ja_asr_provider_worker.py"), "--provider", provider, "--model", str(model)]
            language = raw.get("language") or section.get(f"{provider}_language") or section.get("language")
            if language:
                command.extend(["--language", str(language)])
            if section.get("device"):
                command.extend(["--device", str(section["device"])])
            if section.get("dtype"):
                command.extend(["--dtype", str(section["dtype"])])
        specs.append({"provider": provider, "family": family, "command": command,
                      "runtime": runtime, "model_revision": raw.get("revision") or section.get(f"{provider}_revision"),
                      "model": model})
    return specs


__all__ = ["ASR_PROFILES", "PROFILE_PROVIDERS", "ASRProfile", "ProviderResult", "validate_profile", "run_provider_command",
           "family_vote_rows", "select_reading", "provider_specs"]


def _manifest_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = Path(str(config["input_manifest"])).expanduser().absolute()
    if manifest.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload.get("items", payload) if isinstance(payload, Mapping) else payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _failed_spec(spec: Mapping[str, Any], reason: str) -> ProviderResult:
    return ProviderResult(str(spec["provider"]), str(spec["family"]), "failed",
                          runtime=spec.get("runtime"), model_revision=spec.get("model_revision"),
                          error={"reason": reason})


def asr_stage(config: Mapping[str, Any], stage_dir: Path):
    try:
        from .ja_en_schema import StageResult, atomic_write_json, make_receipt
    except ImportError:
        from ja_en_schema import StageResult, atomic_write_json, make_receipt
    rows = _manifest_rows(config)
    audio_receipts = _jsonl(stage_dir.parent / "audio" / "audio_transform_receipts.jsonl")
    audio_by_uid = {str(row.get("uid")): row for row in audio_receipts}
    specs = provider_specs(config)
    evidence: list[dict[str, Any]] = []
    complete_rows = 0
    profile = validate_profile(str(config.get("asr", {}).get("profile", "qwen_only_dev")))
    for row in rows:
        uid = str(row.get("uid", row.get("id", "")))
        audio_row = audio_by_uid.get(uid, {})
        wav = audio_row.get("alignment", {}).get("path")
        if not isinstance(wav, str):
            # Keep a stable failed record when Stage 1 was not run.
            wav = str(row.get("source_wav", row.get("wav", row.get("audio", ""))))
        providers: list[dict[str, Any]] = []
        for spec in specs:
            command = spec.get("command")
            if not command:
                result = _failed_spec(spec, "provider_command_missing")
            else:
                result = run_provider_command(command, wav, family=str(spec["family"]), provider=str(spec["provider"]),
                                              runtime=spec.get("runtime"), model_revision=spec.get("model_revision"))
            item = result.as_dict()
            if result.status == "ok":
                candidates = list(result.candidates)
                if candidates:
                    item["candidates"] = candidates
            providers.append(item)
        observed = {str(item.get("family")) for item in providers if item.get("status") == "ok"}
        required = {str(spec["family"]) for spec in specs}
        row_status = "COMPLETE" if required.issubset(observed) else ("PARTIAL" if observed else "BLOCKED")
        if row_status == "COMPLETE":
            complete_rows += 1
        evidence.append({"schema": "ja-asr-evidence-v2", "uid": uid,
                         "source_wav": row.get("source_wav", row.get("wav", row.get("audio"))),
                         "orig_text": row.get("orig_text", row.get("text")), "providers": providers,
                         "family_vote_policy": "one_per_family", "production_profile": config.get("asr", {}).get("profile"),
                         "status": row_status, "required_families": sorted(required), "observed_families": sorted(observed)})
    output = stage_dir / "asr_evidence.jsonl"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, evidence)
    status = "COMPLETE" if complete_rows == len(rows) and rows else ("PARTIAL" if complete_rows else "BLOCKED")
    receipt = make_receipt(stage="asr", status=status, inputs={"profile": config.get("asr", {}).get("profile")}, outputs=[output])
    receipt_path = stage_dir / "receipt.json"
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="asr", status=status, receipt_path=str(receipt_path))


def _analysis_for_row(row: Mapping[str, Any], *, asr_results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidate = row.get("candidate_analysis")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    # W2 normally supplies this file.  A row may carry its digest in a
    # manifest for a small integration test, but no normalization is performed.
    digest = row.get("canonical_sha256", row.get("canonicalSHA"))
    if isinstance(digest, str) and digest:
        return {"canonical_sha256": digest, "analysis_version": str(row.get("analysis_version", "w2-v1"))}
    return None


def _manual_overrides(config: Mapping[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, tuple[int, int]], dict[str, Any]], set[str]]:
    """Load occurrence-scoped manual readings; reject ambiguous declarations."""
    source = config.get("manual_overrides")
    if not source:
        return {}, {}, set()
    path = Path(str(source)).expanduser()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"manual_overrides file is missing or symlinked: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            records = payload.get("overrides", payload.get("items", []))
            if isinstance(records, Mapping):
                records = [dict(value, uid=key) for key, value in records.items() if isinstance(value, Mapping)]
        else:
            records = payload
    if not isinstance(records, list):
        raise ValueError("manual_overrides must contain an overrides/items list")
    by_token: dict[tuple[str, str], dict[str, Any]] = {}
    by_span: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
    invalid: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            invalid.add("<invalid>")
            continue
        uid = str(record.get("uid", record.get("id", "")))
        reading = record.get("reading", record.get("chosen_reading"))
        source_id = record.get("source_id")
        token_id = record.get("token_id")
        span = record.get("canonical_span", record.get("span"))
        if not uid or not isinstance(reading, str) or not reading or not isinstance(source_id, str) or not source_id:
            invalid.add(uid or "<invalid>")
            continue
        value = {"reading": reading, "source_id": source_id}
        if isinstance(token_id, str) and token_id:
            key = (uid, token_id)
            if key in by_token and by_token[key] != value:
                invalid.add(uid)
            by_token[key] = value
        elif isinstance(span, (list, tuple)) and len(span) == 2 and all(type(item) is int for item in span):
            key = (uid, (int(span[0]), int(span[1])))
            if key in by_span and by_span[key] != value:
                invalid.add(uid)
            by_span[key] = value
        else:
            invalid.add(uid or "<invalid>")
    return by_token, by_span, invalid


def project_blind_asr_candidates(analysis: Mapping[str, Any], providers: Sequence[Mapping[str, Any]],
                                 frontend_config: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Project W2 spans/readings onto raw ASR evidence without re-normalizing.

    W2 may expose a richer ``project_blind_asr_candidates`` helper; this
    fallback consumes only its canonical text, token spans and contextual
    readings.  Exact utterance surface or exact W2 reading-stream matches are
    lexical support, never acoustic votes.
    """
    external = None
    try:
        from . import ja_frontend  # type: ignore
        external = next((getattr(ja_frontend, name, None) for name in
                         ("project_blind_asr_candidates", "project_asr_candidates", "analyze_blind_asr" )
                         if callable(getattr(ja_frontend, name, None))), None)
    except ImportError:
        external = None
    # The shared W2 hook is authoritative when the complete analysis identity
    # is present. Legacy unit fixtures without ``analysis_version`` use the
    # deterministic local projection below and never mutate W2 text layers.
    if callable(external) and analysis.get("analysis_version"):
        try:
            projected = external(analysis, providers, frontend_config)
        except TypeError:
            projected = external(analysis, providers)
        if isinstance(projected, Mapping):
            normalized_projection: dict[str, dict[str, Any]] = {}
            for key, value in projected.items():
                if not isinstance(value, Mapping):
                    continue
                token_id = str(key)
                entry = dict(value)
                candidate_ids = entry.get("candidate_ids") if isinstance(entry.get("candidate_ids"), list) else []
                candidates = []
                for index, candidate in enumerate(entry.get("candidates", []) if isinstance(entry.get("candidates"), list) else []):
                    if not isinstance(candidate, Mapping):
                        continue
                    item = dict(candidate)
                    item.setdefault("token_id", token_id)
                    item.setdefault("candidate_id", entry.get("candidate_id") or (candidate_ids[index] if index < len(candidate_ids) else None))
                    candidates.append(item)
                entry["candidates"] = candidates
                normalized_projection[token_id] = entry
            return normalized_projection
    units = [unit for unit in analysis.get("units", [])
             if isinstance(unit, Mapping) and unit.get("lexical_status", "lexical") == "lexical"
             and not bool(unit.get("morph_punct"))]
    if not units:
        return {}
    exact_surface = {str(value) for value in (analysis.get("canonical_text"), analysis.get("orig_text")) if isinstance(value, str)}
    reading_stream = "".join(str(unit.get("read", "")) for unit in units)
    result: dict[str, dict[str, Any]] = {}
    for provider in providers:
        if provider.get("status") not in {"ok", "success", "complete"} or not isinstance(provider.get("asr_text"), str):
            continue
        transcript = str(provider["asr_text"])
        match_kind = "surface" if transcript in exact_surface else ("kana" if reading_stream and transcript == reading_stream else None)
        if match_kind is None:
            continue
        for unit in units:
            token_id = str(unit.get("token_id"))
            reading = str(unit.get("read", ""))
            if not reading:
                continue
            provider_evidence = {
                "provider": provider.get("provider"),
                "family": provider.get("family"),
                "raw_asr_text": transcript,
                "raw_asr_span": provider.get("raw_asr_span") or [0, len(transcript)],
                "artifact_path": provider.get("artifact_path") or provider.get("raw_artifact_path"),
                "artifact_sha256": provider.get("artifact_sha256") or provider.get("raw_artifact_sha256"),
                "raw_stdout_sha256": provider.get("raw_stdout_sha256"),
            }
            candidate = {
                "token_id": token_id,
                "candidate_id": unit.get("candidate_id"),
                "reading": reading,
                "origin_reading": reading,
                "provider": provider.get("provider"),
                "family": provider.get("family"),
                "match_kind": match_kind,
                "canonical_span": unit.get("canonical_span"),
                "orig_span": unit.get("orig_span"),
                "provenance": "w2_exact_text_projection",
                "evidence_scope": "lexical_support",
                "lexical_support": True,
                "shared_g2p_derivation": True,
                "acoustic_reading_proof": False,
                "raw_asr_span": provider_evidence["raw_asr_span"],
                "artifact_path": provider_evidence["artifact_path"],
                "artifact_sha256": provider_evidence["artifact_sha256"],
            }
            entry = result.setdefault(token_id, {
                "origin_reading": reading, "origin_match": True,
                "match_kind": match_kind, "provider": provider.get("provider"),
                "family": provider.get("family"), "lexical_support": True,
                "raw_asr_text": transcript,
                "shared_g2p_derivation": True, "evidence_scope": "lexical_support",
                "acoustic_reading_proof": False, "candidates": [], "provider_evidence": [],
                "candidate_id": unit.get("candidate_id"),
                "canonical_span": unit.get("canonical_span"), "orig_span": unit.get("orig_span"),
            })
            entry["candidates"].append(candidate)
            entry["provider_evidence"].append(provider_evidence)
    return result


def reading_stage(config: Mapping[str, Any], stage_dir: Path):
    try:
        from .ja_en_schema import StageResult, atomic_write_json, make_receipt
    except ImportError:
        from ja_en_schema import StageResult, atomic_write_json, make_receipt
    rows = _manifest_rows(config)
    manual_by_token, manual_by_span, invalid_manual_uids = _manual_overrides(config)
    evidence = {str(row.get("uid")): row for row in _jsonl(stage_dir.parent / "asr" / "asr_evidence.jsonl")}
    frontend_path = stage_dir.parent / "frontend" / "frontend_analysis.json"
    frontend_records: dict[str, Mapping[str, Any]] = {}
    if frontend_path.is_file():
        payload = json.loads(frontend_path.read_text(encoding="utf-8"))
        frontend_records = {str(record.get("uid")): record for record in payload.get("records", []) if isinstance(record, Mapping)}
    selections: list[dict[str, Any]] = []
    for row in rows:
        uid = str(row.get("uid", row.get("id", "")))
        current = evidence.get(uid, {"providers": []})
        asr_results = current.get("providers", [])
        analysis = frontend_records.get(uid) or _analysis_for_row(row, asr_results=asr_results)
        if analysis is None:
            selections.append({"schema": "ja-reading-selection-v2", "uid": uid,
                               "status": "blocked", "selected_reading": None,
                               "candidates": [], "error": {"code": "candidate_analysis_missing",
                                                              "message": "W2 candidate analysis is required"}})
            continue
        units = analysis.get("units") if isinstance(analysis.get("units"), list) else []
        if units:
            frontend_config = config.get("frontend_config") or config.get("frontend")
            if not isinstance(frontend_config, Mapping):
                frontend_config = None
            blind_projection = project_blind_asr_candidates(analysis, asr_results, frontend_config)
            lexical_units = [unit for unit in units if unit.get("lexical_status", "lexical") == "lexical"
                             and unit.get("surface") and not bool(unit.get("morph_punct"))]
            if uid in invalid_manual_uids:
                selections.extend({"schema": "ja-reading-selection-v2", "uid": uid, "token_id": unit.get("token_id"),
                                   "status": "blocked", "selected_reading": None, "candidates": [],
                                   "error": {"code": "manual_override_invalid",
                                              "message": "manual override needs uid, token_id or canonical_span, reading, and source_id"}}
                                  for unit in lexical_units)
                continue
            overrides = row.get("reading_overrides", {})
            if not isinstance(overrides, Mapping):
                overrides = {}
            if row.get("reading_override") and len(lexical_units) != 1:
                selections.extend({"schema": "ja-reading-selection-v2", "uid": uid, "token_id": unit.get("token_id"),
                                   "status": "blocked", "selected_reading": None, "candidates": [],
                                   "error": {"code": "reading_ambiguous", "message": "utterance override needs exactly one lexical unit"}}
                                  for unit in lexical_units)
                continue
            for unit in lexical_units:
                token_id = unit.get("token_id")
                inline_override = overrides.get(token_id, overrides.get(str(unit.get("canonical_span"))))
                external_override = manual_by_token.get((uid, str(token_id)))
                span_value = unit.get("canonical_span")
                if external_override is None and isinstance(span_value, (list, tuple)) and len(span_value) == 2 and all(type(item) is int for item in span_value):
                    external_override = manual_by_span.get((uid, (int(span_value[0]), int(span_value[1]))))
                if inline_override is not None and external_override is not None and inline_override != external_override:
                    selections.append({"schema": "ja-reading-selection-v2", "uid": uid, "token_id": token_id,
                                       "status": "blocked", "selected_reading": None, "candidates": [],
                                       "error": {"code": "manual_override_conflict",
                                                  "message": "inline and manual_overrides readings disagree"}})
                    continue
                override = external_override if external_override is not None else inline_override
                unit_row = {**row, "token_id": token_id, "candidate_id": unit.get("candidate_id")}
                projected = blind_projection.get(str(token_id))
                token_asr_results = asr_results
                if projected:
                    projection_evidence = dict(projected)
                    projected_candidates = projected.get("candidates", [])
                    if isinstance(projected_candidates, list):
                        valid_candidates = [candidate for candidate in projected_candidates if isinstance(candidate, Mapping)]
                        if valid_candidates:
                            first_candidate = valid_candidates[0]
                            for key in ("provider", "family", "raw_asr_span", "artifact_path", "artifact_sha256",
                                        "raw_stdout_sha256", "match_kind", "canonical_span", "orig_span"):
                                if projection_evidence.get(key) is None and first_candidate.get(key) is not None:
                                    projection_evidence[key] = first_candidate.get(key)
                            if projection_evidence.get("raw_asr_span") is None:
                                projection_evidence["raw_asr_span"] = first_candidate.get("asr_orig_span") or first_candidate.get("orig_span")
                            provider_name = projection_evidence.get("provider")
                            provider_row = next((item for item in asr_results
                                                 if isinstance(item, Mapping) and item.get("provider") == provider_name), None)
                            if isinstance(provider_row, Mapping):
                                for key in ("raw_asr_text", "asr_text", "artifact_path", "raw_artifact_path", "artifact_sha256", "raw_artifact_sha256",
                                            "raw_stdout_sha256"):
                                    target_key = ("raw_asr_text" if key == "asr_text" else
                                                  "artifact_path" if key == "raw_artifact_path" else
                                                  "artifact_sha256" if key == "raw_artifact_sha256" else key)
                                    if projection_evidence.get(target_key) is None and provider_row.get(key) is not None:
                                        projection_evidence[target_key] = provider_row.get(key)
                            projection_evidence.setdefault("provider_evidence", [
                                {key: candidate.get(key) for key in ("provider", "family", "raw_asr_span",
                                 "artifact_path", "artifact_sha256", "raw_stdout_sha256") if candidate.get(key) is not None}
                                for candidate in valid_candidates])
                    unit_row["_projection_evidence"] = projection_evidence
                    projected_ids = projected.get("candidate_ids") if isinstance(projected.get("candidate_ids"), list) else []
                    if projected.get("candidate_id") or projected_ids:
                        unit_row["candidate_id"] = projected.get("candidate_id") or projected_ids[0]
                    if projected.get("origin_match") is True and projected.get("origin_reading"):
                        unit_row.update({"origin_reading": projected["origin_reading"], "origin_match": True,
                                         "shared_g2p_derivation": True,
                                         "evidence_scope": "lexical_support",
                                         "acoustic_reading_proof": False})
                    projection_candidates = projected.get("candidates", [])
                    if isinstance(projection_candidates, list):
                        normalized_candidates = []
                        for candidate in projection_candidates:
                            if not isinstance(candidate, Mapping):
                                continue
                            normalized = dict(candidate)
                            normalized.setdefault("token_id", token_id)
                            candidate_ids = projected.get("candidate_ids") if isinstance(projected.get("candidate_ids"), list) else []
                            normalized.setdefault("candidate_id", projected.get("candidate_id") or (candidate_ids[0] if candidate_ids else None))
                            if normalized.get("raw_asr_span") is None:
                                normalized["raw_asr_span"] = normalized.get("asr_orig_span") or normalized.get("orig_span")
                            normalized_candidates.append(normalized)
                        token_asr_results = list(asr_results) + [
                            {"provider": candidate.get("provider"),
                             "family": candidate.get("family"), "status": "ok",
                             "candidates": [candidate]}
                            for candidate in normalized_candidates]
                if isinstance(override, Mapping):
                    unit_row["reading_override"] = override.get("reading")
                    unit_row["source_id"] = override.get("source_id")
                elif isinstance(override, str):
                    unit_row["reading_override"] = override
                selections.append(select_reading(unit_row, token_asr_results, analysis))
        else:
            selections.append(select_reading(row, asr_results, analysis))
    stage_dir.mkdir(parents=True, exist_ok=True)
    output = stage_dir / "locked_readings.jsonl"
    _write_jsonl(output, selections)
    production_statuses = {"manual_verified", "origin_surface_confirmed", "asr_family_consensus"}
    status = "COMPLETE" if selections and all(item.get("status") in production_statuses for item in selections) else ("PARTIAL" if selections else "BLOCKED")
    receipt = make_receipt(stage="reading", status=status, inputs={"evidence": str(stage_dir.parent / "asr" / "asr_evidence.jsonl")}, outputs=[output])
    receipt_path = stage_dir / "receipt.json"
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="reading", status=status, receipt_path=str(receipt_path))


def register_stages(registrar: Any) -> None:
    registrar("asr", asr_stage, output_namespace="asr")
    registrar("reading", reading_stage, output_namespace="reading")
