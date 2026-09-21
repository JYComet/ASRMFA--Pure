"""Pure Japanese textual-tone resolution and native-phone projection.

This module deliberately has no stage or runtime dependencies.  Its inputs are
already locked artifacts: semantic-v2 supplies ordered mora/basic ownership and
alignment-v3 supplies native MFA intervals.  The code never estimates F0 or
creates sub-phone boundaries.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:  # Support both ``python -m scripts.ja_prosody`` and direct imports.
    from .ja_en_schema import JAContractError, StageResult, atomic_write_bytes, atomic_write_json, canonical_json, make_receipt, sha256_file
except ImportError:  # pragma: no cover - script invocation compatibility
    from ja_en_schema import JAContractError, StageResult, atomic_write_bytes, atomic_write_json, canonical_json, make_receipt, sha256_file


TONE_VALUES = frozenset({"H", "L", "UNK"})
_EVENT_LABELS = frozenset({"sil", "sp", "pau", "breath", "laugh", "laughter"})
_RESOURCE_SCHEMA = "ja-tone-resource-v1"
_RESOURCE_VERSION = 1


def mora_tones_from_phrase(mora_count: int, nucleus: int) -> list[str]:
    """Return the OpenJTalk-style textual H/L contour for one accent phrase."""
    if not isinstance(mora_count, int) or isinstance(mora_count, bool) or mora_count < 1:
        raise JAContractError("accent_phrase_unresolved", "invalid phrase mora count")
    if not isinstance(nucleus, int) or isinstance(nucleus, bool) or nucleus < 0 or nucleus > mora_count:
        raise JAContractError("accent_phrase_unresolved", "invalid phrase nucleus")
    if nucleus == 1:
        return ["H"] + ["L"] * (mora_count - 1)
    tones = ["L"] + ["H"] * (mora_count - 1)
    if 1 < nucleus < mora_count:
        for index in range(nucleus, mora_count):
            tones[index] = "L"
    return tones


def _mapping(value: Any, code: str, message: str, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JAContractError(code, message, path)
    return value


def _locked_digest(graph: Mapping[str, Any]) -> str:
    digest = graph.get("locked_reading_digest")
    if not isinstance(digest, str) or not digest:
        raise JAContractError("tone_provenance_missing", "semantic graph has no locked reading digest", "$.locked_reading_digest")
    return digest


def _resource_matches(resource: Mapping[str, Any] | None, digest: str) -> bool:
    return isinstance(resource, Mapping) and resource.get("locked_reading_digest") == digest


def _resource_provenance(name: str, resource: Mapping[str, Any], digest: str, mora_count: int) -> tuple[list[str], dict[str, Any]]:
    """Validate the minimal closed manual/lexicon resource shape.

    A path+hash pair makes an external resource reopenable.  Tests and other
    pure callers may instead bind embedded canonical content with a digest.
    """
    if resource.get("schema") != _RESOURCE_SCHEMA or resource.get("version") != _RESOURCE_VERSION:
        raise JAContractError("tone_provenance_missing", "tone resource schema/version is incomplete", f"$.{name}")
    if resource.get("locked_reading_digest") != digest:
        raise JAContractError("tone_provenance_missing", "tone resource locked reading digest differs", f"$.{name}.locked_reading_digest")
    tones = resource.get("tones")
    cardinality = resource.get("cardinality", resource.get("mora_count"))
    if not isinstance(tones, list) or not isinstance(cardinality, int) or isinstance(cardinality, bool):
        raise JAContractError("tone_provenance_missing", "tone resource needs tones and cardinality", f"$.{name}")
    if resource.get("mora_count") is not None and resource.get("cardinality") is not None and resource["mora_count"] != resource["cardinality"]:
        raise JAContractError("tone_cardinality_mismatch", "tone resource cardinalities disagree", f"$.{name}")
    if cardinality != mora_count or len(tones) != mora_count:
        raise JAContractError("tone_cardinality_mismatch", "tone resource does not cover semantic morae", f"$.{name}")
    if any(tone not in {"H", "L"} for tone in tones):
        raise JAContractError("tone_provenance_missing", "known tone resource contains invalid tone", f"$.{name}.tones")
    entry_id = resource.get("entry_id", resource.get("entry_identity"))
    adapter_version = resource.get("adapter_version")
    provider_revision = resource.get("provider_revision")
    evidence_digest = resource.get("evidence_digest")
    resource_path = resource.get("resource_path")
    resource_hash = resource.get("resource_sha256", resource.get("sha256"))
    embedded = resource.get("embedded_canonical_digest")
    if (not all(isinstance(value, str) and value for value in (entry_id, adapter_version, provider_revision, evidence_digest))
            or not ((isinstance(resource_path, str) and resource_path and isinstance(resource_hash, str) and resource_hash)
                    or (isinstance(embedded, str) and embedded))):
        raise JAContractError("tone_provenance_missing", "tone resource lacks reopenable provenance", f"$.{name}")
    return list(tones), {
        "resource_path": resource_path if resource_path else "embedded",
        "resource_sha256": resource_hash if resource_hash else embedded,
        "entry_id": entry_id,
        "provider_revision": provider_revision,
        "locked_reading_digest": digest,
        "adapter_version": adapter_version,
        "evidence_digest": evidence_digest,
    }


def _frontend_provenance(frontend: Mapping[str, Any], digest: str, nodes: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    if not frontend.get("accent_evidence_valid") or frontend.get("locked_reading_digest") != digest:
        raise JAContractError("accent_phrase_unresolved", "contextual frontend evidence is not valid for the locked reading", "$.frontend")
    evidence = _mapping(frontend.get("accent_evidence"), "tone_provenance_missing", "frontend accent evidence is missing", "$.frontend.accent_evidence")
    phrases = evidence.get("accent_phrases")
    if not isinstance(phrases, list) or not phrases:
        raise JAContractError("accent_phrase_unresolved", "frontend accent phrases are missing", "$.frontend.accent_evidence.accent_phrases")
    by_phrase: dict[str, list[str]] = {}
    phrase_metadata: dict[str, tuple[int, int]] = {}
    for index, phrase in enumerate(phrases):
        phrase = _mapping(phrase, "accent_phrase_unresolved", "frontend phrase is invalid", f"$.frontend.accent_evidence.accent_phrases[{index}]")
        count, nucleus = phrase.get("mora_count"), phrase.get("nucleus")
        if not isinstance(count, int) or not isinstance(nucleus, int):
            raise JAContractError("accent_phrase_unresolved", "frontend phrase has no count/nucleus", f"$.frontend.accent_evidence.accent_phrases[{index}]")
        phrase_tones = mora_tones_from_phrase(count, nucleus)
        expected = phrase.get("expected_tones")
        if expected is not None and expected != phrase_tones:
            raise JAContractError("accent_phrase_unresolved", "frontend phrase tones do not match its nucleus", f"$.frontend.accent_evidence.accent_phrases[{index}]")
        phrase_id = phrase.get("accent_phrase_id")
        if not isinstance(phrase_id, str) or not phrase_id or phrase_id in by_phrase:
            raise JAContractError("accent_phrase_unresolved", "frontend phrase identity is invalid", f"$.frontend.accent_evidence.accent_phrases[{index}]")
        by_phrase[phrase_id] = phrase_tones
        phrase_metadata[phrase_id] = (count, nucleus)
    evidence_moras = evidence.get("moras")
    if not isinstance(evidence_moras, list):
        raise JAContractError("accent_phrase_unresolved", "frontend mora positions are missing", "$.frontend.accent_evidence.moras")
    evidence_positions: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in evidence_moras:
        row = _mapping(row, "accent_phrase_unresolved", "frontend mora position is invalid", "$.frontend.accent_evidence.moras")
        phrase_id, position = row.get("accent_phrase_id"), row.get("mora_index_in_phrase")
        if not isinstance(phrase_id, str) or not isinstance(position, int) or phrase_id not in by_phrase or (phrase_id, position) in evidence_positions:
            raise JAContractError("accent_phrase_unresolved", "frontend mora position is invalid", "$.frontend.accent_evidence.moras")
        if position < 1 or position > len(by_phrase[phrase_id]):
            raise JAContractError("tone_cardinality_mismatch", "frontend mora position is outside phrase cardinality", "$.frontend.accent_evidence.moras")
        if (row.get("mora_count"), row.get("nucleus")) != phrase_metadata[phrase_id]:
            raise JAContractError("tone_cardinality_mismatch", "frontend mora metadata differs from phrase summary", "$.frontend.accent_evidence.moras")
        evidence_positions[(phrase_id, position)] = row
    semantic_positions: list[tuple[str, int]] = []
    for node in nodes:
        phrase_id, position = node.get("accent_phrase_id"), node.get("mora_index_in_phrase")
        if not isinstance(phrase_id, str) or not isinstance(position, int) or phrase_id not in by_phrase:
            raise JAContractError("tone_cardinality_mismatch", "semantic mora has no explicit frontend phrase position", "$.mora_nodes")
        semantic_positions.append((phrase_id, position))
    if len(set(semantic_positions)) != len(semantic_positions) or set(semantic_positions) != set(evidence_positions):
        raise JAContractError("tone_cardinality_mismatch", "scoped frontend mora positions differ from semantic mora ownership", "$.mora_nodes")
    tones: list[str] = []
    for node in nodes:
        phrase_id, position = node.get("accent_phrase_id"), node.get("mora_index_in_phrase")
        evidence_row = evidence_positions.get((phrase_id, position))
        if not isinstance(phrase_id, str) or not isinstance(position, int) or phrase_id not in by_phrase or evidence_row is None:
            raise JAContractError("tone_cardinality_mismatch", "semantic mora has no explicit frontend phrase position", "$.mora_nodes")
        tones.append(by_phrase[phrase_id][position - 1])
    identity = _mapping(evidence.get("provider_identity"), "tone_provenance_missing", "frontend provider identity is missing", "$.frontend.accent_evidence.provider_identity")
    provider_revision = identity.get("provider_revision")
    adapter_version = evidence.get("adapter_version")
    provider_digest = evidence.get("provider_evidence_sha256")
    unit_digest = evidence.get("unit_evidence_sha256")
    if not all(isinstance(value, str) and value for value in (provider_revision, adapter_version, provider_digest, unit_digest)):
        raise JAContractError("tone_provenance_missing", "frontend evidence lacks reopenable provenance", "$.frontend.accent_evidence")
    return tones, {
        "resource_path": f"frontend:{identity.get('provider', 'unknown')}",
        "resource_sha256": provider_digest,
        "entry_id": "|".join(str(phrase.get("accent_phrase_id", index)) for index, phrase in enumerate(phrases)),
        "provider_revision": provider_revision,
        "locked_reading_digest": digest,
        "adapter_version": adapter_version,
        "evidence_digest": unit_digest,
    }


def resolve_mora_tones(
    graph: Mapping[str, Any], frontend: Mapping[str, Any],
    manual_overrides: Mapping[str, Any] | None = None, accent_lexicon: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve one deterministic textual tone per semantic mora.

    Source order is fixed and never performs a majority vote: manual override,
    fixed lexicon, then authenticated contextual frontend evidence, then UNK.
    """
    digest = _locked_digest(graph)
    nodes = graph.get("mora_nodes")
    if not isinstance(nodes, list):
        raise JAContractError("tone_cardinality_mismatch", "semantic graph mora nodes are missing", "$.mora_nodes")
    candidates: list[dict[str, Any]] = []
    if _resource_matches(manual_overrides, digest):
        candidate_tones, candidate_provenance = _resource_provenance("manual_overrides", manual_overrides, digest, len(nodes))
        candidates.append({"source": "manual_override", "tones": candidate_tones, "provenance": candidate_provenance})
    if _resource_matches(accent_lexicon, digest):
        candidate_tones, candidate_provenance = _resource_provenance("accent_lexicon", accent_lexicon, digest, len(nodes))
        candidates.append({"source": "fixed_accent_lexicon", "tones": candidate_tones, "provenance": candidate_provenance})
    if isinstance(frontend, Mapping) and frontend.get("accent_evidence_valid") and frontend.get("locked_reading_digest") == digest:
        candidate_tones, candidate_provenance = _frontend_provenance(frontend, digest, nodes)
        candidates.append({"source": "contextual_frontend_prediction", "tones": candidate_tones, "provenance": candidate_provenance})
    chosen = candidates[0] if candidates else None
    source = chosen["source"] if chosen else "unknown"
    tones = list(chosen["tones"]) if chosen else ["UNK"] * len(nodes)
    provenance = chosen["provenance"] if chosen else None
    overridden = [{**copy.deepcopy(candidate), "overridden_by": source} for candidate in candidates[1:]]
    result: list[dict[str, Any]] = []
    for index, node_value in enumerate(nodes):
        node = _mapping(node_value, "tone_cardinality_mismatch", "semantic mora is invalid", f"$.mora_nodes[{index}]")
        mora_id, kana = node.get("mora_id"), node.get("kana")
        if not isinstance(mora_id, str) or not mora_id or not isinstance(kana, str):
            raise JAContractError("tone_cardinality_mismatch", "semantic mora identity is invalid", f"$.mora_nodes[{index}]")
        tone = tones[index]
        result.append({
            "mora_id": mora_id, "kana": kana, "kind": node.get("kind", "regular"),
            "mora_index": node.get("mora_index", index), "tone": tone, "tone_known": tone in {"H", "L"},
            # Textual accent is retained even where acoustic F0 is unobservable.
            "tone_source": source, "f0_observed": bool(node.get("f0_observed", False)),
            "tone_provenance": copy.deepcopy(provenance), "overridden_sources": copy.deepcopy(overridden),
        })
    return result


def _is_non_mora_phone(phone: Mapping[str, Any]) -> bool:
    return phone.get("language") != "ja" or str(phone.get("native_phone", "")).lower() in _EVENT_LABELS


def _display_for_native(phone: Mapping[str, Any], mora_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    if _is_non_mora_phone(phone):
        if phone.get("mora_ids") or phone.get("basic_phone_ids"):
            raise JAContractError("phone_tone_projection_lossy", "non-Japanese/event phone has semantic ownership", "$.native_phones")
        return "", "NA"
    ids = phone.get("mora_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) and item in mora_by_id for item in ids):
        raise JAContractError("phone_tone_projection_lossy", "Japanese phone has no ordered semantic morae", "$.native_phones")
    ordered = [mora_by_id[item] for item in ids]
    return "|".join(str(row["kana"]) for row in ordered), "|".join(str(row["tone"]) for row in ordered)


def _duration_group(phone: Mapping[str, Any]) -> dict[str, Any] | None:
    basic_ids = phone.get("basic_phone_ids")
    if not isinstance(basic_ids, list) or len(basic_ids) <= 1:
        return None
    start, end = phone.get("start_sample"), phone.get("end_sample")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise JAContractError("phone_tone_projection_lossy", "multi-basic native phone has invalid interval", "$.native_phones")
    phone_id = phone.get("phone_id")
    if not isinstance(phone_id, str) or not phone_id:
        raise JAContractError("phone_tone_projection_lossy", "multi-basic native phone has no identity", "$.native_phones")
    return {"duration_group_id": f"duration-group-{phone_id}", "native_phone_id": phone_id,
            "basic_phone_ids": list(basic_ids), "total_duration_samples": end - start,
            "internal_boundaries_known": False, "boundary_source": "unknown_inside_mfa_interval",
            "duration_loss_mode": "group_sum"}


def _assert_template_authority(phone: Mapping[str, Any], template: Mapping[str, Any]) -> None:
    fields = ("native_phone", "mora_ids", "basic_phone_ids", "transform")
    if any(phone.get(field) != template.get(field) for field in fields):
        raise JAContractError("phone_tone_projection_lossy", "alignment phone differs from semantic native template", "$.native_phones")
    for field in ("token_id", "alias"):
        if template.get(field) is not None and phone.get(field) != template.get(field):
            raise JAContractError("phone_tone_projection_lossy", "alignment phone identity differs from semantic native template", "$.native_phones")


def validate_projected_phone(phone: Mapping[str, Any], mora_rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject a serialized phone row that collapses ordered mora tone data."""
    mora_by_id = {row.get("mora_id"): row for row in mora_rows if isinstance(row, Mapping)}
    expected_kana, expected_tone = _display_for_native(phone, mora_by_id)
    if phone.get("phone_kana") != expected_kana or phone.get("phone_tone") != expected_tone:
        raise JAContractError("phone_tone_projection_lossy", "projected phone tone/kana is not the ordered mora vector", "$.native_phones")
    expected_group = _duration_group(phone)
    if expected_group is not None and phone.get("duration_group_id") != expected_group["duration_group_id"]:
        raise JAContractError("phone_tone_projection_lossy", "multi-basic duration group is incomplete", "$.native_phones")


def project_native_phone_tones(
    alignment: Mapping[str, Any], graph: Mapping[str, Any], mora_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach ordered display vectors without changing alignment timing evidence."""
    phones = alignment.get("native_phones")
    if not isinstance(phones, list):
        raise JAContractError("phone_tone_projection_lossy", "alignment native phones are missing", "$.native_phones")
    templates = graph.get("native_phone_templates", [])
    if not isinstance(templates, list):
        raise JAContractError("phone_tone_projection_lossy", "semantic native templates are missing", "$.native_phone_templates")
    template_by_id = {item.get("native_phone_id"): item for item in templates if isinstance(item, Mapping)}
    mora_by_id = {row.get("mora_id"): row for row in mora_rows if isinstance(row, Mapping)}
    projected: list[dict[str, Any]] = []
    for index, raw in enumerate(phones):
        phone = _mapping(raw, "phone_tone_projection_lossy", "native phone is invalid", f"$.native_phones[{index}]")
        if phone.get("language") == "ja":
            template_id = phone.get("native_phone_template_id")
            template = template_by_id.get(template_id) if isinstance(template_id, str) else (templates[index] if index < len(templates) else None)
            if not isinstance(template, Mapping):
                raise JAContractError("phone_tone_projection_lossy", "Japanese phone has no semantic native template", f"$.native_phones[{index}]")
            _assert_template_authority(phone, template)
        kana, tone = _display_for_native(phone, mora_by_id)
        row = copy.deepcopy(dict(phone))
        row["phone_kana"], row["phone_tone"] = kana, tone
        group = _duration_group(phone)
        if group is not None:
            row["duration_group_id"] = group["duration_group_id"]
        validate_projected_phone(row, mora_rows)
        projected.append(row)
    return projected


def build_prosody_alignment(
    alignment: Mapping[str, Any], graph: Mapping[str, Any], frontend: Mapping[str, Any],
    manual_overrides: Mapping[str, Any] | None = None, accent_lexicon: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the pure v1 prosody view from immutable semantic/alignment inputs."""
    mora_rows = resolve_mora_tones(graph, frontend, manual_overrides, accent_lexicon) if graph.get("mora_nodes") else []
    by_id = {row["mora_id"]: row for row in mora_rows}
    basic_rows: list[dict[str, Any]] = []
    basics = graph.get("basic_phone_nodes", [])
    if not isinstance(basics, list):
        raise JAContractError("tone_cardinality_mismatch", "semantic basic phones are invalid", "$.basic_phone_nodes")
    for index, raw in enumerate(basics):
        basic = _mapping(raw, "tone_cardinality_mismatch", "semantic basic phone is invalid", f"$.basic_phone_nodes[{index}]")
        mora_id = basic.get("mora_id")
        if mora_id not in by_id:
            raise JAContractError("tone_cardinality_mismatch", "basic phone mora ownership is unresolved", f"$.basic_phone_nodes[{index}]")
        mora = by_id[mora_id]
        row = copy.deepcopy(dict(basic))
        row.update({"tone": mora["tone"], "tone_known": mora["tone_known"], "tone_source": mora["tone_source"],
                    "f0_observed": mora["f0_observed"]})
        basic_rows.append(row)
    native_rows = project_native_phone_tones(alignment, graph, mora_rows)
    duration_groups = [group for phone in native_rows if (group := _duration_group(phone)) is not None]
    sources: list[dict[str, Any]] = []
    for row in mora_rows:
        provenance = row["tone_provenance"]
        if provenance is not None and provenance not in sources:
            sources.append(copy.deepcopy(provenance))
    return {
        "schema": "ja-prosody-alignment-v1", "uid": alignment.get("uid", graph.get("uid")),
        "words": copy.deepcopy(alignment.get("words", [])), "moras": mora_rows,
        "basic_phones": basic_rows, "native_phones": native_rows, "duration_groups": duration_groups, "tone_sources": sources,
    }


def handle_prosody(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    """Build all independent prosody rows and preserve a fail-closed UID ledger."""
    workspace = stage_dir.parent.parent
    receipt_path = stage_dir / "receipt.json"
    settings = config.get("prosody") if isinstance(config.get("prosody"), Mapping) else {}
    source_value = settings.get("alignment_jsonl")
    if not isinstance(source_value, str):
        receipt = make_receipt(stage="prosody", status="BLOCKED", params={"implementation": "ja-mora-tone-v1"},
                               errors=[{"code": "publish_blocked", "message": "prosody.alignment_jsonl is required"}])
        atomic_write_json(receipt_path, receipt, workspace=workspace)
        return StageResult("prosody", "BLOCKED", str(receipt_path))
    source = Path(source_value).expanduser().absolute()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        if not source.is_file() or source.is_symlink():
            raise JAContractError("publish_blocked", "prosody alignment source is unavailable", str(source))
        source_text = source.read_text(encoding="utf-8")
        try:
            source_payload = json.loads(source_text)
        except json.JSONDecodeError:
            source_payload = None
        if isinstance(source_payload, Mapping) and isinstance(source_payload.get("alignments"), list):
            source_rows = list(source_payload["alignments"])
        elif isinstance(source_payload, Mapping):
            source_rows = [dict(source_payload.get("alignment", source_payload))]
        elif isinstance(source_payload, list):
            source_rows = source_payload
        else:
            # JSONL remains the normal bridge artifact.
            source_rows = [json.loads(line) for line in source_text.splitlines() if line.strip()]
        try:
            from .ja_en_stage_inputs import assemble_prosody_rows
        except ImportError:  # pragma: no cover
            from ja_en_stage_inputs import assemble_prosody_rows
        for alignment in source_rows:
            uid = str(alignment.get("uid", "")) if isinstance(alignment, Mapping) else ""
            try:
                rows.extend(assemble_prosody_rows(config, workspace, [alignment]))
            except JAContractError as error:
                failures.append({"uid": uid, **error.as_dict()})
            except Exception as error:  # a bad UID must not erase other work
                failures.append({"uid": uid, "code": "verifier_failed", "message": f"{type(error).__name__}: {error}"})
        expected_uids = {str(row.get("uid", "")) for row in source_rows if isinstance(row, Mapping) and row.get("uid")}
        blocked_uids: set[str] = set()
        upstream_errors = workspace / "stages" / "merge" / "uid_errors.json"
        if upstream_errors.is_file() and not upstream_errors.is_symlink():
            ledger = json.loads(upstream_errors.read_text(encoding="utf-8"))
            if isinstance(ledger, Mapping):
                expected_uids.update(str(value) for value in ledger.get("expected_uids", []) if value)
                blocked_uids.update(str(value) for value in ledger.get("blocked_uids", []) if value)
            for error in ledger.get("errors", []) if isinstance(ledger, Mapping) else []:
                if isinstance(error, Mapping) and error.get("uid"):
                    failures.append({"uid": str(error["uid"]), **{key: value for key, value in error.items() if key != "uid"}})
                    blocked_uids.add(str(error["uid"]))
        failed_uids = {str(item.get("uid", "")) for item in failures if item.get("uid")}
        for uid in sorted(blocked_uids - failed_uids):
            failures.append({"uid": uid, "code": "publish_blocked", "message": "upstream merge UID is blocked"})
        output = stage_dir / "prosody_alignments.jsonl"
        atomic_write_bytes(output, b"".join(canonical_json(row) for row in rows), workspace=workspace)
        if failures:
            atomic_write_json(stage_dir / "uid_errors.json", {
                "schema": "ja-en-uid-error-ledger-v1", "stage": "prosody",
                "expected_uids": sorted(expected_uids),
                "blocked_uids": sorted({str(item["uid"]) for item in failures if item.get("uid")}), "errors": failures,
            }, workspace=workspace)
        status = "COMPLETE" if not failures else "PARTIAL"
        receipt = make_receipt(stage="prosody", status=status,
                               inputs={"artifacts": [{"path": str(source), "sha256": sha256_file(source)}]},
                               outputs=[output], params={"implementation": "ja-mora-tone-v1"}, errors=failures)
    except JAContractError as error:
        receipt = make_receipt(stage="prosody", status="REJECTED", params={"implementation": "ja-mora-tone-v1"},
                               errors=[error.as_dict()])
        status = "REJECTED"
    atomic_write_json(receipt_path, receipt, workspace=workspace)
    return StageResult("prosody", status, str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("prosody", handle_prosody, output_namespace="prosody")


__all__ = [
    "TONE_VALUES", "build_prosody_alignment", "mora_tones_from_phrase", "project_native_phone_tones",
    "resolve_mora_tones", "validate_projected_phone", "handle_prosody", "register_stages",
]
