"""Strict integer-sample merge for isolated Japanese and English MFA runs."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable, Mapping, Sequence


MERGED_ALIGNMENT_FILENAME = "ja_en_alignment.json"
MERGED_ALIGNMENT_JSONL_FILENAME = "ja_en_alignment.jsonl"
MERGE_REQUIRED_PHONE_KEYS = frozenset({"alias", "language", "phone", "start_sample", "end_sample"})

try:
    from .ja_en_schema import JAContractError, StageResult, atomic_write_json, make_receipt
except ImportError:  # pragma: no cover
    from ja_en_schema import JAContractError, StageResult, atomic_write_json, make_receipt


class MergeRejected(ValueError):
    """A raw MFA interval cannot be published on the global sample axis."""


def _native_label(phone: Mapping[str, Any]) -> str:
    """Return the literal native MFA label, rejecting unqualified labels."""
    label = phone.get("native_phone")
    if not isinstance(label, str) or not label:
        raise JAContractError("native_basic_mapping_ambiguous", "raw MFA native label is required", "$.native_phones")
    return label


def _locked_aliases(alignment: Mapping[str, Any], uid: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index the locked dictionary evidence by its stable composite identity."""
    rows = alignment.get("locked_aliases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise JAContractError("native_basic_mapping_ambiguous", "locked alias evidence is required", "$.locked_aliases")
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise JAContractError("native_basic_mapping_ambiguous", "locked alias row is invalid", f"$.locked_aliases[{index}]")
        alias, token_id = source.get("alias"), source.get("token_id")
        pronunciation = source.get("pronunciation")
        if not isinstance(alias, str) or not alias or not isinstance(token_id, str) or not token_id:
            raise JAContractError("native_basic_mapping_ambiguous", "locked alias/token identity is required", f"$.locked_aliases[{index}]")
        if not isinstance(pronunciation, Sequence) or isinstance(pronunciation, (str, bytes)) or not pronunciation or not all(isinstance(phone, str) and phone for phone in pronunciation):
            raise JAContractError("native_basic_mapping_ambiguous", "locked pronunciation is required", f"$.locked_aliases[{index}]")
        key = (uid, token_id, alias)
        if key in indexed:
            raise JAContractError("native_basic_mapping_ambiguous", "locked alias identity is duplicated", f"$.locked_aliases[{index}]")
        indexed[key] = {**dict(source), "pronunciation": list(pronunciation)}
    return indexed


def _axis_boundaries(phone: Mapping[str, Any], path: str) -> None:
    """Require declared axes; coordinates are copied, never inferred or equated."""
    for axis in ("source_axis", "alignment_axis", "training_axis"):
        if phone.get(axis) is None:
            raise JAContractError("alignment_invalid", f"{axis} evidence is required", path)
    for boundary in ("start_sample", "end_sample"):
        if type(phone.get(boundary)) is not int:
            raise JAContractError("alignment_invalid", f"{boundary} must be an integer", path)
    if phone["start_sample"] < 0 or phone["end_sample"] <= phone["start_sample"]:
        raise JAContractError("alignment_invalid", "native interval bounds are invalid", path)
    if phone.get("raw_interval_id") is None or not isinstance(phone.get("run_id"), str) or not phone["run_id"]:
        raise JAContractError("alignment_invalid", "raw interval provenance is required", path)


def bind_native_phone_graph(alignment: Mapping[str, Any], semantic_graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bind raw native MFA intervals to locked semantic templates without fallbacks.

    The only match key is ``(uid, token_id, alias)`` and repeated labels are
    paired by their pronunciation occurrence.  Native labels merely validate
    the already-selected template; they are never a search key.
    """
    uid = alignment.get("uid")
    if not isinstance(uid, str) or not uid:
        raise JAContractError("native_basic_mapping_ambiguous", "alignment UID is required", "$.uid")
    raw_phones = alignment.get("native_phones")
    if not isinstance(raw_phones, Sequence) or isinstance(raw_phones, (str, bytes)) or not raw_phones:
        raise JAContractError("native_basic_mapping_ambiguous", "raw native MFA phones are required", "$.native_phones")
    locks = _locked_aliases(alignment, uid)
    templates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for graph_index, graph in enumerate(semantic_graphs):
        if not isinstance(graph, Mapping) or graph.get("uid") != uid:
            raise JAContractError("native_basic_mapping_ambiguous", "semantic graph UID differs from alignment", f"$.semantic_graphs[{graph_index}]")
        graph_token = graph.get("token_id")
        for template_index, source in enumerate(graph.get("native_phone_templates") or []):
            if not isinstance(source, Mapping):
                raise JAContractError("native_basic_mapping_ambiguous", "semantic native template is invalid", f"$.semantic_graphs[{graph_index}].native_phone_templates[{template_index}]")
            template = dict(source)
            token_id = template.get("token_id") or graph_token
            if not isinstance(token_id, str) or not token_id or (template.get("token_id") and graph_token and template["token_id"] != graph_token):
                raise JAContractError("native_basic_mapping_ambiguous", "semantic template token identity is invalid", f"$.semantic_graphs[{graph_index}].native_phone_templates[{template_index}]")
            alias = template.get("alias") or graph.get("alias")
            if alias is None:
                matching = [key[2] for key in locks if key[:2] == (uid, token_id)]
                if len(matching) != 1:
                    raise JAContractError("native_basic_mapping_ambiguous", "semantic template has no unique locked alias", f"$.semantic_graphs[{graph_index}].native_phone_templates[{template_index}]")
                alias = matching[0]
            key = (uid, token_id, alias)
            if key not in locks or not isinstance(template.get("native_phone"), str) or not template["native_phone"]:
                raise JAContractError("native_basic_mapping_ambiguous", "semantic template does not bind locked identity", f"$.semantic_graphs[{graph_index}].native_phone_templates[{template_index}]")
            template["alias"] = alias
            templates.setdefault(key, []).append(template)
    for key, locked in locks.items():
        candidates = templates.get(key, [])
        if [template.get("native_phone") for template in candidates] != locked["pronunciation"]:
            raise JAContractError("native_basic_mapping_ambiguous", "semantic templates differ from locked pronunciation", "$.semantic_graphs")

    ordered = sorted((dict(phone) for phone in raw_phones), key=lambda phone: (phone.get("start_sample", -1), phone.get("end_sample", -1), str(phone.get("run_id", "")), str(phone.get("raw_interval_id", ""))))
    occurrences: dict[tuple[str, str, str], int] = {}
    native: list[dict[str, Any]] = []
    for index, phone in enumerate(ordered):
        path = f"$.native_phones[{index}]"
        _axis_boundaries(phone, path)
        token_id, alias = phone.get("token_id"), phone.get("alias")
        if phone.get("uid", uid) != uid or not isinstance(token_id, str) or not token_id or not isinstance(alias, str) or not alias:
            raise JAContractError("native_basic_mapping_ambiguous", "raw phone composite identity is invalid", path)
        key = (uid, token_id, alias)
        candidates = templates.get(key, [])
        occurrence = occurrences.get(key, 0)
        if occurrence >= len(candidates):
            raise JAContractError("native_basic_mapping_ambiguous", "no ordered semantic template", path)
        template = candidates[occurrence]
        if _native_label(phone) != template["native_phone"]:
            raise JAContractError("native_basic_mapping_ambiguous", "raw native label differs from locked template occurrence", path)
        occurrences[key] = occurrence + 1
        native.append({**phone, "uid": uid, "native_phone": _native_label(phone),
                       "basic_phone_ids": list(template.get("basic_phone_ids") or []),
                       "mora_ids": list(template.get("mora_ids") or []),
                       "transform": template.get("transform"),
                       "native_phone_template_id": template.get("native_phone_id"),
                       "boundary_source": "mfa_native_interval"})
    for key, candidates in templates.items():
        if occurrences.get(key, 0) != len(candidates):
            raise JAContractError("native_basic_mapping_ambiguous", "raw MFA intervals do not cover locked pronunciation", "$.native_phones")
    raw_mfa = dict(alignment.get("raw_mfa") or {})
    raw_mfa["phones"] = [dict(phone) for phone in ordered]
    return {**dict(alignment), "schema": "ja-en-alignment-v3", "native_phones": native, "raw_mfa": raw_mfa}


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate retry publications while retaining conflicts."""
    result: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (row.get("language"), row.get("alias"), row.get("raw_interval_id"), row.get("phone"), row.get("start_sample"), row.get("end_sample"))
        if key not in seen:
            seen[key] = row
            result.append(row)
    return result


def _sample(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if type(value) is not int or value < 0:
        raise MergeRejected(f"invalid integer {key}")
    return value


def merge_global_sample_axis(
    japanese_intervals: Sequence[Mapping[str, Any]],
    english_intervals: Sequence[Mapping[str, Any]],
    *,
    ownership: tuple[int, int],
    sample_rate: int,
    expected_languages: Mapping[str, str],
    reject_edge_touch: bool = False,
    run_ownership: Mapping[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Validate and merge raw intervals without clipping, scaling or edits."""
    owner_start, owner_end = ownership
    if type(owner_start) is not int or type(owner_end) is not int or owner_start < 0 or owner_end <= owner_start:
        raise MergeRejected("invalid ownership range")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise MergeRejected("invalid sample rate")
    all_rows: list[dict[str, Any]] = []
    seen_aliases: set[str] = set()
    for language, rows in (("ja", japanese_intervals), ("en", english_intervals)):
        prefix = "ja:" if language == "ja" else "en:"
        for row in rows:
            alias = row.get("alias")
            if not isinstance(alias, str) or expected_languages.get(alias) != language:
                raise MergeRejected(f"cardinality/language mismatch for alias {alias!r}")
            phone = row.get("phone", row.get("label"))
            if not isinstance(phone, str) or not phone.startswith(prefix):
                raise MergeRejected(f"native inventory namespace mismatch for {language}: {phone!r}")
            start = _sample(row, "start_sample")
            end = _sample(row, "end_sample")
            if end <= start:
                raise MergeRejected("non-positive phone interval")
            if start < owner_start or end > owner_end:
                raise MergeRejected("ownership crossing interval")
            row_run = row.get("run_id")
            local_owner = (run_ownership or {}).get(str(row_run)) if row_run is not None else None
            if local_owner is None and row_run is not None and row.get("ownership_start_sample") is not None:
                local_owner = (row["ownership_start_sample"], row["ownership_end_sample"])
            if local_owner is not None:
                local_start, local_end = local_owner
                if start < local_start or end > local_end:
                    raise MergeRejected(f"run ownership crossing interval: {row_run}")
            crop_start = row.get("crop_start_sample", row.get("context_start_sample", owner_start))
            crop_end = row.get("crop_end_sample", row.get("context_end_sample", owner_end))
            if reject_edge_touch and (start == crop_start or end == crop_end):
                raise MergeRejected("phone touches padded crop edge")
            all_rows.append({**dict(row), "language": language, "phone": phone, "start_sample": start, "end_sample": end})
            seen_aliases.add(alias)
    expected = set(expected_languages)
    if seen_aliases != expected:
        raise MergeRejected(f"cardinality mismatch: expected {sorted(expected)!r}, got {sorted(seen_aliases)!r}")
    all_rows.sort(key=lambda row: (row["start_sample"], row["end_sample"], row["language"], row["alias"], row.get("raw_interval_id", 0)))
    previous_end = owner_start
    for row in all_rows:
        if row["start_sample"] < previous_end:
            raise MergeRejected("overlap on global sample axis")
        previous_end = row["end_sample"]
    return all_rows


def retry_seam_both_sides(
    *,
    seam_id: str,
    initial_padding_samples: int,
    max_retries: int,
    rerun: Callable[[str, int], Any],
    validate: Callable[[Any], bool],
) -> dict[str, Any]:
    """Rerun both seam sides with growing context, then reject if still bad."""
    if type(initial_padding_samples) is not int or initial_padding_samples < 0:
        raise ValueError("initial_padding_samples must be non-negative")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_retries):
        # The first retry must widen context beyond the failed crop.
        padding = initial_padding_samples * (2 ** (attempt + 1)) if initial_padding_samples else 0
        left = rerun("left", padding)
        right = rerun("right", padding)
        ok = bool(validate({"left": left, "right": right, "padding_samples": padding, "attempt": attempt + 1}))
        attempts.append({"attempt": attempt + 1, "padding_samples": padding, "left": left, "right": right, "valid": ok})
        if ok:
            return {"seam_id": seam_id, "status": "VERIFIED", "attempts": attempts, "padding_samples": padding}
    return {"seam_id": seam_id, "status": "REJECTED", "attempts": attempts, "reason": "seam_retry_exhausted"}


def merge_with_seam_retries(
    japanese_intervals: Sequence[Mapping[str, Any]],
    english_intervals: Sequence[Mapping[str, Any]],
    *,
    ownership: tuple[int, int],
    sample_rate: int,
    expected_languages: Mapping[str, str],
    seam_id: str,
    initial_padding_samples: int,
    max_retries: int,
    rerun: Callable[[str, int], tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
    reject_edge_touch: bool = True,
    run_ownership: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Validate raw merge and orchestrate actual two-sided reruns on failure."""
    try:
        merged = merge_global_sample_axis(japanese_intervals, english_intervals, ownership=ownership, sample_rate=sample_rate, expected_languages=expected_languages, reject_edge_touch=reject_edge_touch, run_ownership=run_ownership)
        return {"status": "VERIFIED", "intervals": merged, "retries": []}
    except MergeRejected as initial_error:
        accepted_merged: list[dict[str, Any]] | None = None

        def rerun_side(side: str, padding: int) -> Any:
            ja, en = rerun(side, padding)
            return {"japanese": list(ja), "english": list(en)}

        def validate_pair(pair: Mapping[str, Any]) -> bool:
            nonlocal accepted_merged
            try:
                left_ja = _dedupe_rows(pair["left"]["japanese"])
                right_ja = _dedupe_rows(pair["right"]["japanese"])
                left_en = _dedupe_rows(pair["left"]["english"])
                right_en = _dedupe_rows(pair["right"]["english"])
                accepted_merged = merge_global_sample_axis(_dedupe_rows(left_ja + right_ja), _dedupe_rows(left_en + right_en), ownership=ownership, sample_rate=sample_rate, expected_languages=expected_languages, reject_edge_touch=reject_edge_touch, run_ownership=run_ownership)
                return True
            except MergeRejected:
                return False

        retry = retry_seam_both_sides(seam_id=seam_id, initial_padding_samples=initial_padding_samples, max_retries=max_retries, rerun=rerun_side, validate=validate_pair)
        if retry.get("status") == "VERIFIED" and accepted_merged is not None:
            retry["intervals"] = accepted_merged
        retry["initial_error"] = str(initial_error)
        return retry


def handle_merge(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    """Merge prepared language ledgers and publish one verified alignment."""
    receipt_path = stage_dir / "receipt.json"
    raw = config.get("merge")
    if not isinstance(raw, Mapping):
        raw = (config.get("stage_inputs") or {}).get("merge") if isinstance(config.get("stage_inputs"), Mapping) else None
    if raw is None and (stage_dir / "input.json").is_file():
        raw = json.loads((stage_dir / "input.json").read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        receipt = make_receipt(stage="merge", status="BLOCKED", params={"implementation": "strict-integer-sample-ja-en-v2"}, errors=[{"code": "publish_blocked", "message": "raw MFA ledgers and ownership are required"}])
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult(stage="merge", status="BLOCKED", receipt_path=str(receipt_path))

    def load_rows(value: Any, namespace: str) -> list[dict[str, Any]]:
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.suffix == ".jsonl":
                return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping) and "runs" in payload:
                value = [phone for run in payload["runs"] for phone in run.get("phones", [])]
            else:
                value = payload.get("phones", payload.get("intervals", payload)) if isinstance(payload, Mapping) else payload
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("merge intervals must be a sequence or JSON path")
        rows = [dict(row) for row in value]
        for row in rows:
            phone = row.get("phone", row.get("native_phone"))
            if isinstance(phone, str) and not phone.startswith(("ja:", "en:")):
                row["phone"] = namespace + phone
            # ``phone`` is the raw MFA tier label in historical ledgers.  It
            # may be normalised with its language namespace, but never used to
            # select a semantic template.
            if not row.get("native_phone") and isinstance(phone, str):
                row["native_phone"] = phone.removeprefix(namespace)
        return rows

    try:
        japanese = load_rows(raw.get("japanese_intervals", raw.get("ja_intervals", raw.get("japanese_ledger"))), "ja:")
        english = load_rows(raw.get("english_intervals", raw.get("en_intervals", raw.get("english_ledger"))), "en:")
        expected = raw.get("expected_languages")
        if not isinstance(expected, Mapping) or not expected:
            raise MergeRejected("expected_languages alias map is required; observed phones cannot define cardinality")
        ownership = (int(raw["ownership_start_sample"]), int(raw["ownership_end_sample"])) if "ownership_start_sample" in raw else tuple(raw["ownership"])
        common = dict(ownership=ownership, sample_rate=int(raw.get("sample_rate", 16000)), expected_languages=expected, reject_edge_touch=bool(raw.get("reject_edge_touch", False)), run_ownership=raw.get("run_ownership"))
        try:
            merged = merge_global_sample_axis(japanese, english, **common)
            result = {"status": "VERIFIED", "intervals": merged, "retries": []}
        except MergeRejected as initial:
            rerun = raw.get("rerun")
            if not callable(rerun) and isinstance(raw.get("rerun_plan"), Mapping):
                plan = raw["rerun_plan"]
                def rerun(side: str, padding: int) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]:
                    side_plan = plan.get(side)
                    if not isinstance(side_plan, Mapping):
                        raise MergeRejected(f"missing serialized rerun plan for {side}")
                    ja_rows = load_rows(side_plan.get("japanese_intervals", side_plan.get("ja_intervals")), "ja:")
                    en_rows = load_rows(side_plan.get("english_intervals", side_plan.get("en_intervals")), "en:")
                    return ja_rows, en_rows
            if not callable(rerun):
                raise initial
            result = merge_with_seam_retries(japanese, english, ownership=ownership, sample_rate=int(raw.get("sample_rate", 16000)), expected_languages=expected, seam_id=str(raw.get("seam_id", "seam_0000")), initial_padding_samples=int(raw.get("initial_padding_samples", 0)), max_retries=int(raw.get("max_retries", 0)), rerun=rerun, reject_edge_touch=bool(raw.get("reject_edge_touch", True)), run_ownership=raw.get("run_ownership"))
        if result.get("status") != "VERIFIED" or not isinstance(result.get("intervals"), list):
            raise MergeRejected(str(result.get("reason", "merge rejected")))
        uid = str(raw.get("uid", config.get("uid", "")))
        semantic_graphs = raw.get("semantic_graphs")
        if not isinstance(semantic_graphs, Sequence) or isinstance(semantic_graphs, (str, bytes)):
            semantic_graph = raw.get("semantic_graph")
            semantic_graphs = [semantic_graph] if isinstance(semantic_graph, Mapping) else []
        locked_aliases = raw.get("locked_aliases", raw.get("expected_units"))
        raw_mfa = dict(raw.get("raw_mfa") or {})
        if "runs" not in raw_mfa and isinstance(raw.get("runs"), Sequence):
            raw_mfa["runs"] = list(raw["runs"])
        seed = {
            "schema": "ja-en-alignment-v2", "uid": uid,
            "words": list(raw.get("words", [])), "native_phones": result["intervals"],
            "locked_aliases": locked_aliases, "raw_mfa": raw_mfa,
            "languages": list(raw.get("languages", [{"language": "ja", "unit_ids": sorted({row["alias"] for row in japanese})}, {"language": "en", "unit_ids": sorted({row["alias"] for row in english})}])),
            "seams": list(raw.get("seams", [])), "retry": result.get("retries", []),
        }
        payload = bind_native_phone_graph(seed, list(semantic_graphs))
        output = stage_dir / MERGED_ALIGNMENT_FILENAME
        output_jsonl = stage_dir / MERGED_ALIGNMENT_JSONL_FILENAME
        atomic_write_json(output, payload, workspace=stage_dir.parent.parent)
        atomic_write_json(output_jsonl, {"alignment": payload, **payload}, workspace=stage_dir.parent.parent)
        receipt = make_receipt(stage="merge", status="COMPLETE", inputs={"uid": payload["uid"], "phone_count": len(payload["native_phones"])}, outputs=[output, output_jsonl], params={"implementation": "strict-integer-sample-ja-en-v3", "retry_count": len(result.get("retries", []))})
    except Exception as exc:
        receipt = make_receipt(stage="merge", status="REJECTED", params={"implementation": "strict-integer-sample-ja-en-v3"}, errors=[{"code": "seam_rejected", "message": str(exc)}])
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="merge", status=receipt["status"], receipt_path=str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("merge", handle_merge, output_namespace="merge")


__all__ = [
    "MergeRejected", "bind_native_phone_graph", "handle_merge", "merge_global_sample_axis", "merge_with_seam_retries",
    "register_stages", "retry_seam_both_sides", "MERGED_ALIGNMENT_FILENAME", "MERGED_ALIGNMENT_JSONL_FILENAME", "MERGE_REQUIRED_PHONE_KEYS",
]
