"""Fail-closed chunk result classification and atomic speaker publication."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

try:
    from scripts.speaker_namespace import publication_speaker
except ModuleNotFoundError:
    from speaker_namespace import publication_speaker


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


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


@dataclass(frozen=True)
class PublicationEntry:
    stem: str
    source: Path
    target: Path
    speaker: str
    source_id: str | None = None
    game: str | None = None
    text_mode: str | None = None
    kind: str = "accepted"

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass(frozen=True)
class ChunkPublication:
    accepted: tuple[PublicationEntry, ...]
    filtered: tuple[PublicationEntry, ...]
    failed: tuple[dict, ...]
    input_stems: tuple[str, ...]
    accepted_stems: tuple[str, ...]
    filtered_stems: tuple[str, ...]
    failed_stems: tuple[str, ...]
    chunk_id: str = ""


def _inventory_map(inventory):
    items = inventory.get("items", []) if isinstance(inventory, dict) else getattr(inventory, "items", inventory)
    return {_get(item, "run_stem"): item for item in items}


def _grid_tiers(path: Path):
    try:
        try:
            from scripts.postprocess_textgrids import parse_textgrid
        except ModuleNotFoundError:
            # ``full_corpus_orchestrator.py`` is also supported as a direct
            # script, in which case its directory (rather than the repo root)
            # is on sys.path.
            from postprocess_textgrids import parse_textgrid
        grid = parse_textgrid(path)
        return grid, {tier.name for tier in grid.tiers}
    except Exception as exc:
        raise ValueError(f"invalid TextGrid {path}: {exc}") from exc


def _validate_grid(path: Path, item, require_final_contract: bool = True) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing TextGrid: {path}")
    grid, tiers = _grid_tiers(path)
    expected = ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")
    if require_final_contract and tuple(tier.name for tier in grid.tiers) != expected:
        raise ValueError(f"TextGrid tiers differ from final contract: {path}")
    if not grid.tiers:
        raise ValueError(f"filtered TextGrid has no diagnostic tier: {path}")
    audio = _get(item, "pipeline_wav") or _get(item, "audio_path")
    if not audio or not Path(audio).is_file() or Path(audio).is_symlink():
        raise ValueError(f"required pipeline WAV missing: {path}")
    import soundfile as sf
    try:
        info = sf.info(str(audio))
        if info.format != "WAV" or info.channels != 1:
            raise ValueError(f"invalid pipeline WAV: {audio}")
        # GAMEDATA/Wuwa are materialized as mono PCM16; v5 deliberately keeps
        # the source sample format and only needs the same time axis.
        if _get(item, "needs_gamesl_padding", False) and info.subtype != "PCM_16":
            raise ValueError(f"padded pipeline WAV must be PCM16: {audio}")
        duration = info.duration
    except Exception as exc:
        raise ValueError(f"unreadable pipeline WAV: {audio}: {exc}") from exc
    if abs(float(grid.xmax) - duration) > .01:
        raise ValueError(f"TextGrid/audio axis mismatch: {path}")
    for tier in grid.tiers:
        if abs(tier.xmin) > .01 or abs(tier.xmax - duration) > .01 or not tier.intervals:
            raise ValueError(f"TextGrid tier domain/coverage invalid: {path}")


def validate_chunk_result(chunk, inventory) -> ChunkPublication:
    imap = _inventory_map(inventory)
    chunk_id = _get(chunk, "chunk_id", "")
    rows = _get(chunk, "items", None)
    if rows is None and isinstance(chunk, dict):
        rows = chunk.get("input", [])
    rows = list(rows or [])
    stems = [_get(row, "run_stem") if not isinstance(row, str) else row for row in rows]
    if len(stems) != len(set(stems)):
        raise ValueError("duplicate stems in chunk input")
    if set(stems) - set(imap):
        raise ValueError("chunk contains unknown frozen stem")
    evidence = _get(chunk, "evidence", None)
    if evidence is None:
        # The old qwen_receipt/punctuation_projection booleans were caller
        # assertions and are intentionally no longer accepted as proof.
        raise ValueError("sealed producer and final punctuation evidence is required")
    evidence_partitions = _validate_evidence(evidence, stems=stems, items=rows)
    output_root = Path(_get(chunk, "output_root", ""))
    filtered_root = Path(_get(chunk, "filtered_root", ""))
    accepted, filtered, failed = [], [], []
    explicit_failed = _get(chunk, "failed", {}) or {}
    if isinstance(explicit_failed, list):
        explicit_failed = {str(x): "failed" for x in explicit_failed}
    for stem in stems:
        item = imap[stem]
        paths = []
        if output_root:
            paths.append((output_root / f"{stem}.TextGrid", "accepted"))
        if filtered_root:
            paths.append((filtered_root / f"{stem}.TextGrid", "filtered"))
        present = [(path, kind) for path, kind in paths if path.exists()]
        if len(present) > 1:
            raise ValueError(f"stem appears in multiple terminal buckets: {stem}")
        if present:
            path, kind = present[0]
            if stem not in evidence_partitions[kind]:
                raise ValueError(f"physical {kind} result differs from sealed accounting: {stem}")
            _validate_grid(path, item, require_final_contract=(kind == "accepted"))
            public_speaker = publication_speaker(
                _get(item, "game"), _get(item, "speaker") or "_default")
            row = PublicationEntry(
                stem, path, Path(public_speaker) / path.name,
                public_speaker, _get(item, "source_id"),
                _get(item, "game"), _get(item, "text_mode"), kind)
            (accepted if kind == "accepted" else filtered).append(row)
        elif stem in explicit_failed:
            if stem not in evidence_partitions["producer_filtered"]:
                raise ValueError(f"explicit failure has no sealed producer filter evidence: {stem}")
            failed.append({"stem": stem, "reason": explicit_failed[stem],
                           "speaker": publication_speaker(
                               _get(item, "game"), _get(item, "speaker") or "_default"),
                           "source_id": _get(item, "source_id"),
                           "game": _get(item, "game"), "text_mode": _get(item, "text_mode")})
        else:
            raise ValueError(f"frozen stem has no terminal result: {stem}")
    return ChunkPublication(tuple(accepted), tuple(filtered), tuple(failed),
                             tuple(stems), tuple(row["stem"] for row in accepted),
                             tuple(row["stem"] for row in filtered),
                             tuple(row["stem"] for row in failed), chunk_id)


def _validate_evidence(evidence: dict, stems=None, items=None) -> dict[str, set[str]]:
    """Validate paths and digests of sealed pipeline evidence.

    The publisher accepts only artifacts written by run_pipeline.  A caller
    supplied ``verified: true`` flag is never treated as evidence.
    """
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a mapping of sealed artifact paths")
    required = ("qwen_manifest", "qwen_identity", "raw_timestamps",
                "accounting", "postprocess_report", "punctuation_evidence")
    for key in required:
        if key not in evidence:
            raise ValueError(f"missing sealed evidence: {key}")
    paths = []
    for key in required:
        value = evidence[key]
        if key == "raw_timestamps":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("raw timestamp receipts are missing")
            paths.extend(value)
        else:
            paths.append(value)
    declared_digests = evidence.get("sha256")
    if not isinstance(declared_digests, dict):
        raise ValueError("sealed evidence digests are required")
    for raw in paths:
        path = Path(raw)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sealed evidence path is not a regular file: {path}")
        declared = declared_digests.get(str(raw))
        if not declared or declared != _sha256(path):
            raise ValueError(f"sealed evidence digest mismatch: {path}")
    identity = Path(evidence["qwen_identity"])
    try:
        payload = json.loads(identity.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid Qwen identity artifact: {identity}") from exc
    blob = json.dumps(payload, ensure_ascii=False).lower()
    required_identity = ("schema", "runtime", "models", "settings", "inputs", "identity_digest")
    if any(not payload.get(key) for key in required_identity):
        raise ValueError("incomplete Qwen identity evidence")
    if payload.get("provider") != "qwen3_hf" or "nvasr" in blob:
        raise ValueError("sealed producer identity is not Qwen3-only")
    manifest = Path(evidence["qwen_manifest"])
    try:
        rows = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid Qwen manifest artifact: {manifest}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError("Qwen manifest must contain timestamp rows")
    for row in rows:
        if not isinstance(row, dict) or row.get("provider") != "qwen3_hf":
            raise ValueError("Qwen manifest row is not Qwen3 evidence")
        if row.get("timestamp_normalization") != "qwen3-timestamp-normalization-v2":
            raise ValueError("Qwen timestamp normalization receipt is missing")
        if row.get("lexical_timing_source") != "qwen3_forced_aligner_hf":
            raise ValueError("Qwen forced aligner receipt is missing")
    for key in ("qwen_manifest", "accounting"):
        try:
            json.loads(Path(evidence[key]).read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid sealed JSON evidence: {evidence[key]}") from exc
    accounting = json.loads(Path(evidence["accounting"]).read_text(encoding="utf-8"))
    expected = set(stems or [])
    if accounting.get("schema") != "pipeline-run-receipt-v2" or accounting.get("run_health") != "healthy" or accounting.get("silent_loss") != 0:
        raise ValueError("final accounting health evidence is invalid")
    eligible = set(accounting.get("eligible", {}).get("stems", []))
    output = set(accounting.get("output", {}).get("stems", []))
    filtered = set(accounting.get("filtered", {}).get("stems", []))
    if expected and (eligible != expected or output & filtered or output | filtered != expected):
        raise ValueError("final accounting denominator mismatch")
    mode = None
    extra = accounting.get("extra", {})
    if extra.get("reference_mode") in {"authority", "reference"}:
        mode = "reference"
    elif extra.get("reference_mode") == "fallback":
        mode = "fallback"
    report_rows = {}
    try:
        with Path(evidence["postprocess_report"]).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    stem = row.get("stem") or row.get("run_stem")
                    if stem:
                        report_rows[stem] = row
    except Exception as exc:
        raise ValueError("postprocess evidence is unreadable") from exc
    producer_output = set(evidence.get("output_stems", []))
    producer_filtered = set(evidence.get("filtered_stems", []))
    if producer_output or producer_filtered:
        if (producer_output & producer_filtered
                or producer_output | producer_filtered != expected
                or not producer_filtered <= filtered):
            raise ValueError("sealed producer accounting denominator mismatch")
    for stem in output:
        row = report_rows.get(stem)
        if not row or row.get("publication_contract", {}).get("status") != "verified" or row.get("hard_integrity_reasons") != []:
            raise ValueError(f"postprocess publication evidence is invalid: {stem}")
        row_mode = row.get("reference_mode") or mode
        if row_mode in {"fallback", "no_reference"}:
            if not row.get("fallback_transcript") or not row.get("fallback_punctuation_projection"):
                raise ValueError(f"fallback transcript/projection evidence missing: {stem}")
        else:
            details = row.get("publication_contract", {}).get("details", row)
            if not row.get("timestamp_normalized_transcript") and not details.get("timestamp_normalized_transcript"):
                raise ValueError(f"authority timestamp evidence missing: {stem}")
            if not details.get("reference_punctuation_projection"):
                raise ValueError(f"authority punctuation evidence missing: {stem}")
    for stem in filtered - producer_filtered:
        row = report_rows.get(stem)
        status = str(row.get("status", "")) if row else ""
        reasons = row.get("filter_reasons", []) if row else []
        if not row or (not status.startswith("filtered") and not reasons):
            raise ValueError(f"postprocess filter evidence is invalid: {stem}")
    return {"accepted": output, "filtered": filtered,
            "producer_filtered": producer_filtered}


def _pub_rows(publication):
    if isinstance(publication, ChunkPublication):
        return list(publication.accepted) + list(publication.filtered)
    rows = []
    for kind in ("accepted", "filtered"):
        for row in publication.get(kind, []):
            if isinstance(row, PublicationEntry):
                rows.append(row)
            else:
                rows.append(PublicationEntry(
                    row["stem"] if "stem" in row else Path(row["path"]).stem,
                    Path(row["path"]), Path(row.get("speaker", "_default")) / Path(row["path"]).name,
                    row.get("speaker", "_default"), row.get("source_id"), row.get("game"),
                    row.get("text_mode"), kind))
    return rows


def publish_chunk(publication, output_root: Path, rollback_root: Path | None = None) -> dict:
    output_root, rollback_root = Path(output_root), Path(rollback_root or Path(output_root) / "_rollback")
    replacements = []
    for row in _pub_rows(publication):
        source = Path(row.source)
        speaker = str(row.speaker or "_default")
        if not speaker or Path(speaker).name != speaker or speaker in {".", ".."}:
            raise ValueError("unsafe publication speaker")
        publish_root = output_root / ("_filtered" if row.kind == "filtered" else "")
        target = (publish_root / speaker / source.name).resolve(strict=False)
        if publish_root.resolve() not in target.parents:
            raise ValueError("publication target escapes output root")
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"unsafe publication source: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        old_hash, rollback = None, None
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"unsafe publication target: {target}")
            old_hash = _sha256(target)
            rollback = (rollback_root / ("_filtered" if row.kind == "filtered" else "") /
                        speaker / target.name).resolve(strict=False)
            if rollback_root.resolve() not in rollback.parents:
                raise ValueError("rollback target escapes root")
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
            os.replace(temporary, target)
            _fsync_parent(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            if rollback is not None and rollback.exists():
                target.unlink(missing_ok=True)
                shutil.copyfile(rollback, temporary)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                _fsync_parent(target)
            raise
        replacements.append({"stem": row.stem, "kind": row.kind, "speaker": speaker,
                             "target": str(target), "rollback": str(rollback) if rollback else None,
                             "old_sha256": old_hash, "new_sha256": _sha256(target)})
    return {"published_count": len(replacements),
            "replaced_count": sum(row["old_sha256"] is not None for row in replacements),
            "replacements": replacements, "rollback_root": str(rollback_root),
            "speaker_namespace": "game-first-two-hanzi-pinyin-initials-v1"}


def build_final_report(inventory, publications, *, config_digest: str | None = None,
                       publication_receipts=None, status=None) -> dict:
    imap = _inventory_map(inventory)
    rows = []
    replacements = list(publication_receipts or [])
    for publication in publications:
        rows.extend(_pub_rows(publication))
        if isinstance(publication, ChunkPublication):
            rows.extend(publication.failed)
    terminal = []
    for publication in publications:
        if isinstance(publication, ChunkPublication):
            terminal.extend(publication.accepted_stems)
            terminal.extend(publication.filtered_stems)
            terminal.extend(publication.failed_stems)
    included = set(imap)
    if len(terminal) != len(set(terminal)) or set(terminal) != included:
        raise ValueError("terminal publication buckets do not conserve frozen inputs")
    counts = Counter()
    grouped = Counter()
    durations = Counter()
    reasons = Counter()
    for stem, item in imap.items():
        counts["input"] += 1
        key = (_get(item, "source_id"), _get(item, "game"), _get(item, "speaker"), _get(item, "text_mode"))
        grouped[key] += 1
        durations[key] += float(_get(item, "duration_seconds", 0.0) or 0.0)
    for row in rows:
        if isinstance(row, PublicationEntry):
            counts[row.kind if row.kind in {"accepted", "filtered"} else "failed"] += 1
        else:
            counts["failed"] += 1
            reasons[str(row.get("reason", "unknown"))] += 1
    excluded = inventory.get("excluded", []) if isinstance(inventory, dict) else list(getattr(inventory, "excluded", ()))
    invalid = inventory.get("invalid", []) if isinstance(inventory, dict) else list(getattr(inventory, "invalid", ()))
    counts["excluded"] = len(excluded); counts["invalid"] = len(invalid)
    return {"schema": "qwen3-0915all-report-v1", "counts": dict(counts),
            "excluded": [{"source_id": _get(row, "source_id"), "stem": _get(row, "original_stem"),
                           "reason": _get(row, "reason")} for row in excluded],
            "invalid": [{"source_id": _get(row, "source_id"), "stem": _get(row, "original_stem"),
                         "reason": _get(row, "reason")} for row in invalid],
            "terminal_stems": sorted(set(terminal)),
            "groups": [{"source_id": key[0], "game": key[1], "speaker": key[2],
                         "text_mode": key[3], "input": count,
                         "duration_seconds": durations[key]}
                        for key, count in sorted(grouped.items(), key=lambda pair: str(pair[0]))],
            "reasons": dict(reasons),
            "replacements": replacements,
            "status": status,
            "config_sha256": config_digest}
