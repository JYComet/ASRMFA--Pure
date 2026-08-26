#!/usr/bin/env python3
"""Fail-closed verifier for the strict-replay English current-v2 subset.

The production importer owns the frozen ``strict-replay-import-v1`` receipt;
this verifier only reads that receipt and the copied English ledgers.  It does
not run MFA/CTC and it does not accept a historical ``strict-en-mfa-v1`` file
as evidence for the current-v2 subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
try:
    from pipeline_utils import is_english_token, stable_json_digest
except ImportError:  # pragma: no cover - direct fixture import fallback
    stable_json_digest = lambda value: hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    is_english_token = lambda value: bool(value) and value.isascii() and value.isalpha()

CANONICAL_SCHEMA = "mfa-quality-canonical-samples-v1"
CANONICAL_SHA256 = "d88b9ac874283dbc67dc38003fb78d872b799597ce940175a8301f78aa2c5bcf"
IMPORT_SCHEMA = "strict-replay-import-v1"
CURRENT_V2_SCHEMAS = {
    "strict-replay-english-subset-v2",
    "strict-en-mfa-current-v2",
    "strict-en-mfa-v2",
}
ENGLISH_IMPORT_SCHEMA = "strict-replay-english-import-v1"
ENGLISH_IMPORT_V2_SCHEMA = "strict-replay-english-import-v2"
ENGLISH_IMPORT_V21_SCHEMA = "strict-replay-english-import-v2.1"
REPLAY_V2_SCHEMA = "strict-replay-import-v2"
REPLAY_V21_SCHEMA = "strict-replay-import-v2.1"
ALIGNMENT_SUBSET_V2_SCHEMA = "strict-replay-english-alignment-subset-v2"
ALIGNMENT_SUBSET_V21_SCHEMA = "strict-replay-english-alignment-subset-v2.1"
PRODUCER_REVISION = "strict-replay-english-producer-v4.2.1"
FINAL_EVIDENCE_SCHEMA = "strict-replay-final-evidence-v1"
STRICT_ENGLISH_SCHEMA = "strict-en-mfa-v2"
HISTORICAL_STRICT_ENGLISH_SCHEMA = "strict-en-mfa-v1"
CANONICAL_ENGLISH_UNITS_SCHEMA = "canonical-english-units-v1"
HISTORICAL_SCHEMAS = {"strict-en-mfa-v1", "strict-replay-english-subset-v1"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_stem(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and "\x00" not in value
            and value not in {".", ".."} and Path(value).name == value)


def _ordinary_file(raw: object, roots: tuple[Path, ...]) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("path missing")
    candidate = Path(raw)
    if not candidate.is_absolute():
        # Relative evidence is resolved only against the explicitly supplied
        # output roots, never against the process cwd.
        for root in roots:
            probe = root / candidate
            if probe.is_file() and not probe.is_symlink():
                candidate = probe
                break
        else:
            raise ValueError("relative evidence path missing")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("evidence is not an ordinary file")
    resolved = candidate.resolve(strict=True)
    if roots and not any(resolved == root.resolve() or root.resolve() in resolved.parents
                         for root in roots if root.exists()):
        raise ValueError("evidence escapes configured roots")
    return resolved


def _absolute_role(raw: object, label: str, errors: list[str]) -> Path | None:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute() or ".." in Path(raw).parts:
        errors.append(f"{label} path is not normalized absolute")
        return None
    return Path(raw)


def _selected_pairs(receipt: dict, errors: list[str]) -> tuple[list[dict], set[str]]:
    slots = receipt.get("slots")
    if not isinstance(slots, list) or len(slots) != 96:
        errors.append("canonical slot mapping missing/not-96")
        slots = []
    selected = receipt.get("selected_slot_records")
    if selected is None:
        selected = receipt.get("selected_slots")
    if not isinstance(selected, list) or len(selected) not in (24, 96):
        errors.append("selected slots are not canonical 24-pilot/96-full records")
        selected = []
    slot_pairs = {(row.get("slot"), row.get("stem")) for row in slots
                  if isinstance(row, dict)}
    pairs = [(row.get("slot"), row.get("stem")) for row in selected
             if isinstance(row, dict)]
    if len(pairs) != len(selected) or len(pairs) != len(set(pairs)):
        errors.append("selected slot mapping has duplicates/malformed rows")
    if any(pair not in slot_pairs for pair in pairs):
        errors.append("selected slot is outside canonical mapping")
    if receipt.get("source_manifest_slots") != 96:
        errors.append("source manifest is not canonical 96")
    if receipt.get("selected_slots_count") != len(selected):
        errors.append("selected slot count mismatch")
    stems = {stem for _, stem in pairs if _safe_stem(stem)}
    if any(not _safe_stem(stem) for _, stem in pairs):
        errors.append("selected slot contains unsafe stem")
    if len(selected) == 24:
        selector = receipt.get("pilot_selector", {})
        if (receipt.get("pilot_selector_version") != "strict-replay-selector-v1"
                or not isinstance(selector, dict) or selector.get("pilot") is not True):
            errors.append("24-slot selection is not explicit strict-replay pilot")
    return selected, stems


def _load_receipt(path: Path, errors: list[str]) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"receipt unreadable:{exc}")
        return None
    if not isinstance(payload, dict) or payload.get("schema") != IMPORT_SCHEMA:
        errors.append("receipt schema is not strict-replay-import-v1")
        return None
    return payload


def _load_subset(receipt: dict, errors: list[str]) -> dict | None:
    subset = receipt.get("english_subset")
    if subset is None:
        subset = receipt.get("english_provenance_subset")
    if subset is None:
        subset = receipt.get("english_provenance")
    if not isinstance(subset, dict):
        errors.append("current-v2 English subset missing")
        return None
    schema = subset.get("schema")
    if schema in HISTORICAL_SCHEMAS or not isinstance(schema, str) or schema not in CURRENT_V2_SCHEMAS:
        errors.append("historical-v1-as-v2 or unknown English subset schema")
        return None
    return subset


def _v2_english_ledger_errors(payload: object, label: str, *,
                              stem: str | None = None,
                              require_segments: bool = False) -> list[str]:
    """Return contract failures for current English ledger evidence.

    The direct verifier must not treat a v1 ledger wrapped in a v2 import as
    fresh evidence.  Segment/word bindings are checked when a real ledger
    payload is available; metadata-only direct fixtures can still exercise
    the import boundary without manufacturing MFA output.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{label} is not an object"]
    if payload.get("schema") != STRICT_ENGLISH_SCHEMA:
        errors.append(f"{label} schema is not {STRICT_ENGLISH_SCHEMA}")
    if payload.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
        errors.append(f"{label} canonical unit binding missing")
    if stem is not None and payload.get("stem") not in (None, stem):
        errors.append(f"{label} stem mismatch")
    segments = payload.get("segments")
    if segments is None and not require_segments:
        return errors
    if not isinstance(segments, list):
        errors.append(f"{label} segments missing")
        return errors
    for index, segment in enumerate(segments):
        if (not isinstance(segment, dict)
                or segment.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA):
            errors.append(f"{label} segment {index} canonical unit binding missing")
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            errors.append(f"{label} segment {index} words missing")
            continue
        for word_index, word in enumerate(words):
            if (not isinstance(word, dict)
                    or word.get("canonical_binding") != CANONICAL_ENGLISH_UNITS_SCHEMA):
                errors.append(f"{label} word {index}:{word_index} canonical unit binding missing")
    return errors


def _v2_english_manifest_errors(payload: object, label: str) -> list[str]:
    """Return contract failures for the current English parent manifest."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{label} is not an object"]
    if payload.get("schema") != STRICT_ENGLISH_SCHEMA:
        errors.append(f"{label} schema is not {STRICT_ENGLISH_SCHEMA}")
    if payload.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
        errors.append(f"{label} canonical unit binding missing")
    if payload.get("strict_provenance") is not True:
        errors.append(f"{label} strict provenance missing")
    return errors


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _replay_accounting_scope(replay_path: Path, formal_path: Path | None,
                             english_path: Path, final_path: Path | None,
                             errors: list[str], *, active_subset_path: Path,
                             active_parent_path: Path,
                             require_final: bool = True) -> dict | None:
    """Require explicit formal/final receipts; never search siblings."""
    if formal_path is None:
        errors.append("formal receipt path is required (no sibling search)")
        return None
    accounting_path = Path(formal_path)
    if not accounting_path.is_absolute() or ".." in accounting_path.parts:
        errors.append("formal receipt path is not normalized absolute")
        return None
    if accounting_path.is_symlink() or not accounting_path.is_file():
        errors.append("formal receipt missing/non-regular")
        return None
    try:
        accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("strict replay accounting receipt missing/unreadable")
        return None
    if accounting.get("mode") != "strict_replay":
        errors.append("English import is not bound to strict_replay scope")
    bound = accounting.get("extra", {}).get("strict_replay_receipt")
    if not isinstance(bound, str) or Path(bound).resolve() != replay_path.resolve():
        errors.append("strict replay accounting/import binding mismatch")
    evidence = accounting.get("extra", {}).get("strict_replay_evidence", {})
    if evidence.get("english_import") != str(english_path.resolve()):
        errors.append("formal receipt/English import path binding mismatch")
    elif evidence.get("english_sha256") != _sha256(english_path):
        errors.append("formal receipt/English import hash mismatch")
    if evidence.get("english_subset") != str(active_subset_path.resolve()):
        errors.append("formal receipt/English subset path binding mismatch")
    elif evidence.get("english_subset_sha256") != _sha256(active_subset_path):
        errors.append("formal receipt/English subset hash mismatch")
    if evidence.get("parent_english_manifest") != str(active_parent_path.resolve()):
        errors.append("formal receipt/parent English path binding mismatch")
    elif evidence.get("parent_english_sha256") != _sha256(active_parent_path):
        errors.append("formal receipt/parent English hash mismatch")
    if final_path is None and require_final:
        errors.append("final evidence path is required (no sibling search)")
    elif final_path is not None:
        final_path = Path(final_path)
        if (not final_path.is_absolute() or ".." in final_path.parts
                or final_path.is_symlink() or not final_path.is_file()):
            errors.append("final evidence missing/non-regular/unsafe")
        else:
            try:
                final = json.loads(final_path.read_text(encoding="utf-8"))
                if final.get("english_import_sha256") != _sha256(english_path):
                    errors.append("final evidence/English import hash mismatch")
                if final.get("parent_english_sha256") != _sha256(active_parent_path):
                    errors.append("final evidence/parent English hash mismatch")
                if final.get("english_subset_sha256") != _sha256(active_subset_path):
                    errors.append("final evidence/English subset hash mismatch")
                if final.get("formal_receipt_sha256") != _sha256(accounting_path):
                    errors.append("final evidence/formal receipt hash mismatch")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                errors.append("final evidence unreadable")
    return accounting


def _english_required_stems(replay: dict, errors: list[str]) -> tuple[list[str], list[str], list[str]]:
    assets = replay.get("assets")
    if not isinstance(assets, list):
        errors.append("replay assets missing")
        return [], [], []
    source: list[str] = []
    missing = replay.get("missing_mfa_alignment", [])
    if not isinstance(missing, list) or any(not _safe_stem(item) for item in missing):
        errors.append("replay missing alignment membership malformed")
        missing = []
    for bundle in assets:
        stem = bundle.get("stem") if isinstance(bundle, dict) else None
        if not _safe_stem(stem) or stem in source:
            errors.append("replay asset membership duplicate/malformed")
            continue
        source.append(stem)
    source = sorted(source)
    excluded = sorted(set(missing))
    eligible = sorted(set(source) - set(excluded))
    required: list[str] = []
    for bundle in assets:
        stem = bundle.get("stem") if isinstance(bundle, dict) else None
        if stem not in eligible:
            continue
        try:
            text_rec = bundle["assets"]["authoritative_txt"]
            text_path = Path(text_rec["copy"]).resolve(strict=True)
            text = text_path.read_text(encoding="utf-8")
            lexical = [token for token in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
                       if is_english_token(token)]
            if lexical:
                required.append(stem)
        except (KeyError, OSError, TypeError, ValueError):
            errors.append(f"English-required authority unreadable:{stem}")
    return source, eligible, sorted(required)


def _verify_english_import_v21(path: Path, *, replay_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"v2.1 English import unreadable:{exc}"]
    exact = {"schema", "scope", "english_schema", "canonical_units", "run_id", "timestamp_utc", "canonical_manifest_path", "canonical_manifest_sha256",
             "replay_import_manifest_path", "replay_import_manifest_sha256", "selection_slot_records", "selection_slot_count", "selection_slot_digest",
             "source_stems", "source_count", "source_digest", "exclusion_records", "exclusion_count", "exclusion_digest", "excluded_stems", "excluded_count", "excluded_digest",
             "eligible_stems", "eligible_count", "eligible_digest", "english_required_stems", "english_required_count", "english_required_digest",
             "english_entries_stems", "english_entries_count", "english_entries_digest", "producer_revision", "config_sha256", "dictionary_roles", "dictionary_roles_digest",
             "parent_global_manifest", "english_alignment_subset_path", "english_alignment_subset_sha256", "parent_english_manifest_path", "parent_english_manifest_sha256", "records"}
    if not isinstance(payload, dict) or set(payload) != exact:
        return ["v2.1 English import exact field set mismatch"]
    if payload["schema"] != ENGLISH_IMPORT_V21_SCHEMA or payload["scope"] != "strict_replay":
        errors.append("v2.1 schema/scope mismatch")
    if payload["english_schema"] != STRICT_ENGLISH_SCHEMA:
        errors.append("v2.1 English provenance schema mismatch")
    if payload["canonical_units"] != CANONICAL_ENGLISH_UNITS_SCHEMA:
        errors.append("v2.1 canonical English units binding missing")
    if any(key.startswith("selected_") and not key.startswith("selection_slot_") for key in payload):
        errors.append("selected legacy field present")
    if any(key in payload for key in ("dictionary_sha256", "dictionary_path")):
        errors.append("singular dictionary field present")
    replay_path = replay_path or Path(str(payload["replay_import_manifest_path"]))
    if not replay_path.is_file() or replay_path.is_symlink() or replay_path.name != "strict_replay_import.json":
        return errors + ["v2.1 replay import path invalid"]
    if payload["replay_import_manifest_sha256"] != _sha256(replay_path):
        errors.append("v2.1 replay import hash mismatch")
    try: replay = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError): replay = {}
    if replay.get("schema") != REPLAY_V21_SCHEMA:
        errors.append("v2.1 replay schema mismatch")
    selected = replay.get("selection_slot_records", [])
    if (selected != payload["selection_slot_records"] or len(selected) != 24
            or payload["selection_slot_count"] != 24 or payload["selection_slot_digest"] != stable_json_digest(selected)):
        errors.append("v2.1 selection vector mismatch")
    # v2.1 has one canonical slot identity field.  A legacy ``slots`` field
    # is deliberately not an acceptable fallback: accepting it would allow
    # an older receipt to masquerade as the active import contract.
    slot_mapping = replay.get("slot_stem_mapping")
    if (not isinstance(slot_mapping, list) or len(slot_mapping) != 96
            or any(not isinstance(row, dict)
                   or set(row) != {"slot", "stem"}
                   for row in slot_mapping)):
        errors.append("slot_stem_mapping missing/not-96")
        slot_mapping = []
    pairs = {(row.get("slot"), row.get("stem")) for row in slot_mapping
             if isinstance(row, dict)}
    selected_pairs = {(row.get("slot"), row.get("stem"))
                      for row in selected if isinstance(row, dict)}
    if (len(selected_pairs) != len(selected)
            or any((row.get("slot"), row.get("stem")) not in pairs
                   for row in selected if isinstance(row, dict))):
        errors.append("v2.1 selection canonical membership mismatch")
    source = sorted(row.get("stem") for row in replay.get("assets", []) if isinstance(row, dict))
    excluded = payload["excluded_stems"]
    exclusion_records = payload["exclusion_records"]
    if (excluded != sorted(excluded) or payload["excluded_count"] != len(excluded)
            or payload["excluded_digest"] != stable_json_digest(excluded)
            or exclusion_records != sorted(exclusion_records, key=lambda row: row.get("stem", ""))
            or payload["exclusion_count"] != len(exclusion_records)
            or payload["exclusion_digest"] != stable_json_digest(exclusion_records)
            or [row.get("stem") for row in exclusion_records] != excluded
            or any(not isinstance(row, dict) or set(row) != {"stem", "reason"} or not isinstance(row.get("reason"), str) for row in exclusion_records)):
        errors.append("excluded stems/exclusion records mismatch")
    eligible = sorted(set(source) - set(excluded))
    for label, actual, declared, count, digest_value in (("source", source, payload["source_stems"], payload["source_count"], payload["source_digest"]), ("eligible", eligible, payload["eligible_stems"], payload["eligible_count"], payload["eligible_digest"])):
        if actual != declared or count != len(actual) or digest_value != stable_json_digest(actual):
            errors.append(f"{label} vector mismatch")
    for label in ("english_required", "english_entries"):
        stems = payload[f"{label}_stems"]
        if stems != sorted(stems) or payload[f"{label}_count"] != len(stems) or payload[f"{label}_digest"] != stable_json_digest(stems):
            errors.append(f"{label} vector mismatch")
    if (len(selected), len(source), len(excluded), len(eligible), payload["english_required_count"], payload["english_entries_count"]) != (24, 21, 3, 18, 18, 18):
        errors.append("v2.1 frozen denominator mismatch")
    roles = payload["dictionary_roles"]
    role_names = {"chinese_mfa_dictionary", "pinyin_projection_dictionary",
                  "english_pronunciation_dictionary"}
    if (not isinstance(roles, dict) or set(roles) != role_names
            or payload["dictionary_roles_digest"] != stable_json_digest(roles)):
        errors.append("dictionary roles matrix/digest mismatch")
    else:
        for role in roles.values():
            if (not isinstance(role, dict) or set(role) != {"path", "sha256"}
                    or not Path(role.get("path", "")).is_absolute()
                    or role.get("path") in {r.get("path") for r in roles.values()
                                              if isinstance(r, dict) and r is not role}
                    or role.get("path") and Path(role["path"]).is_symlink()
                    or not re.fullmatch(r"[0-9a-f]{64}", role.get("sha256", ""))
                    or not Path(role.get("path", "")).is_file()
                    or _sha256(Path(role["path"])) != role["sha256"]):
                errors.append("dictionary role path/hash/alias invalid")
    parent = payload["parent_global_manifest"]
    if set(parent) != {"authoritative_source", "workspace_copy", "content_identity_sha256"}:
        errors.append("parent_global_manifest role set mismatch")
    else:
        roles_seen = []
        for name in ("authoritative_source", "workspace_copy"):
            role = parent[name]; roles_seen.append(role.get("path"))
            if (set(role) != {"path", "sha256", "immutable_import_path", "immutable_import_sha256"}
                    or not Path(role["path"]).is_absolute() or Path(role["path"]).is_symlink()
                    or role["path"] != role["immutable_import_path"] or role["sha256"] != role["immutable_import_sha256"]
                    or _sha256(Path(role["path"])) != role["sha256"]):
                errors.append(f"parent_global_manifest {name} binding invalid")
        if roles_seen[0] == roles_seen[1] or parent["content_identity_sha256"] != parent["authoritative_source"]["sha256"] or parent["authoritative_source"]["sha256"] != parent["workspace_copy"]["sha256"]:
            errors.append("parent_global_manifest role swap/hash mismatch")
    subset_path = Path(payload["english_alignment_subset_path"]); parent_path = Path(payload["parent_english_manifest_path"])
    if not subset_path.is_file() or subset_path.is_symlink() or payload["english_alignment_subset_sha256"] != _sha256(subset_path) or not parent_path.is_file() or parent_path.is_symlink() or payload["parent_english_manifest_sha256"] != _sha256(parent_path):
        errors.append("subset/parent path hash invalid")
    try:
        subset = json.loads(subset_path.read_text(encoding="utf-8")); parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        if subset.get("schema") != ALIGNMENT_SUBSET_V21_SCHEMA:
            errors.append("subset/parent schema mismatch")
        errors.extend(_v2_english_manifest_errors(parent_payload, "parent English manifest"))
        pg = subset.get("parent_global_manifest", {})
        if pg.get("authoritative_source", {}).get("path") == pg.get("workspace_copy", {}).get("path"): errors.append("parent roles swapped")
    except (OSError, ValueError, json.JSONDecodeError): errors.append("subset/parent JSON unreadable")
    records = payload["records"]
    if [r.get("stem") for r in records if isinstance(r, dict)] != payload["english_entries_stems"] or len(records) != 18:
        errors.append("records membership/order mismatch")
    for record in records:
        if not isinstance(record, dict):
            errors.append("v2.1 record malformed")
            continue
        stem = record.get("stem")
        if record.get("status") != "english_required":
            errors.append(f"v2.1 record status invalid:{stem}")
        if record.get("schema") != STRICT_ENGLISH_SCHEMA:
            errors.append(f"v2.1 record schema invalid:{stem}")
        if record.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
            errors.append(f"v2.1 record canonical unit binding missing:{stem}")
        ledger = record.get("ledger")
        errors.extend(_v2_english_ledger_errors(
            ledger, f"v2.1 ledger metadata:{stem}", stem=stem))
    return sorted(set(errors))


def _verify_english_import_v2(path: Path, *, replay_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"v2 English import unreadable:{exc}"]
    exact = {
        "schema", "scope", "english_schema", "canonical_units", "run_id", "timestamp_utc", "canonical_manifest_path", "canonical_manifest_sha256",
        "replay_import_manifest_path", "replay_import_manifest_sha256", "selection_slot_records", "selection_slot_count", "selection_slot_digest",
        "source_stems", "source_count", "source_digest", "exclusion_records", "exclusion_stems", "exclusion_count", "exclusion_digest",
        "eligible_stems", "eligible_count", "eligible_digest", "english_required_stems", "english_required_count", "english_required_digest",
        "english_entries_stems", "english_entries_count", "english_entries_digest", "producer_revision", "config_sha256",
        "dictionary_roles", "dictionary_roles_digest", "english_alignment_subset_path", "english_alignment_subset_sha256",
        "parent_english_manifest_path", "parent_english_manifest_sha256", "records",
    }
    if isinstance(payload, dict) and payload.get("schema") == ENGLISH_IMPORT_V21_SCHEMA:
        return _verify_english_import_v21(path, replay_path=replay_path)
    if not isinstance(payload, dict) or set(payload) != exact:
        errors.append("v2 English import exact field set mismatch")
        if not isinstance(payload, dict): return errors
    if payload.get("schema") != ENGLISH_IMPORT_V2_SCHEMA or payload.get("scope") != "strict_replay":
        errors.append("v2 English import schema/scope mismatch")
    if payload.get("english_schema") != STRICT_ENGLISH_SCHEMA:
        errors.append("v2 English provenance schema mismatch")
    if payload.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
        errors.append("v2 canonical English units binding missing")
    if payload.get("producer_revision") != PRODUCER_REVISION:
        errors.append("v2 producer revision mismatch")
    if any(key.startswith("selected_") and key not in {"selection_slot_records", "selection_slot_count", "selection_slot_digest"} for key in payload):
        errors.append("selected-prefixed legacy field present")
    if any(key in payload for key in ("dictionary_sha256", "dictionary_path")):
        errors.append("singular dictionary field present")
    replay_raw = payload.get("replay_import_manifest_path")
    replay_path = replay_path or (Path(replay_raw) if isinstance(replay_raw, str) else Path(""))
    if not replay_path.is_file() or replay_path.is_symlink() or replay_path.name != "strict_replay_import.json":
        errors.append("v2 replay import path invalid")
        return errors
    if payload.get("replay_import_manifest_sha256") != _sha256(replay_path):
        errors.append("v2 replay import hash mismatch")
    try: replay = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError): replay = {}
    if replay.get("schema") != REPLAY_V2_SCHEMA:
        errors.append("v2 replay import schema mismatch")
    rpaths = replay.get("paths", {}) if isinstance(replay, dict) else {}
    if isinstance(rpaths, dict):
        ws = _absolute_role(rpaths.get("workspace"), "workspace", errors)
        out = _absolute_role(rpaths.get("output"), "output", errors)
        imm = _absolute_role(rpaths.get("immutable_import"), "immutable_import", errors)
        if ws is not None and out is not None and ws == out:
            errors.append("v2 replay workspace/output collision")
        if ws is not None and imm != ws / "strict_replay_import.json":
            errors.append("v2 replay immutable path mismatch")
        if imm is not None and imm != replay_path:
            errors.append("v2 replay immutable actual path mismatch")
    else:
        errors.append("v2 replay paths missing")
    selected = replay.get("selection_slot_records")
    if (not isinstance(selected, list) or len(selected) != 24
            or payload.get("selection_slot_count") != 24
            or payload.get("selection_slot_records") != selected
            or payload.get("selection_slot_digest") != stable_json_digest(selected)):
        errors.append("selection vector/count/digest mismatch")
    slots = replay.get("slot_stem_mapping", [])
    if not isinstance(slots, list) or len(slots) != 96:
        errors.append("slot_stem_mapping missing/not-96")
        slots = []
    slot_pairs = {(row.get("slot"), row.get("stem")) for row in slots if isinstance(row, dict)}
    if (not isinstance(selected, list) or len({(row.get("slot"), row.get("stem")) for row in selected if isinstance(row, dict)}) != len(selected)
            or any((row.get("slot"), row.get("stem")) not in slot_pairs for row in selected if isinstance(row, dict))):
        errors.append("selection canonical membership/duplicate mismatch")
    source = sorted(row.get("stem") for row in replay.get("assets", []) if isinstance(row, dict))
    missing = sorted(replay.get("missing_mfa_alignment", []))
    eligible = sorted(set(source) - set(missing))
    vectors = (("source", source, payload.get("source_stems"), payload.get("source_count"), payload.get("source_digest")),
               ("exclusion", missing, payload.get("exclusion_stems"), payload.get("exclusion_count"), payload.get("exclusion_digest")),
               ("eligible", eligible, payload.get("eligible_stems"), payload.get("eligible_count"), payload.get("eligible_digest")))
    for label, actual, declared, count, digest_value in vectors:
        if declared != actual or count != len(actual) or digest_value != stable_json_digest(actual):
            errors.append(f"{label} vector/count/digest mismatch")
    required = payload.get("english_required_stems")
    entries_stems = payload.get("english_entries_stems")
    for label, value, count, digest_value in (("english_required", required, payload.get("english_required_count"), payload.get("english_required_digest")), ("english_entries", entries_stems, payload.get("english_entries_count"), payload.get("english_entries_digest"))):
        if not isinstance(value, list) or value != sorted(value) or count != len(value) or digest_value != stable_json_digest(value):
            errors.append(f"{label} vector/count/digest mismatch")
    if not (len(source) == 21 and len(missing) == 3 and len(eligible) == 18 and len(required or []) == 18 and len(entries_stems or []) == 18):
        errors.append("frozen 21/3/18/18/18 denominator mismatch")
    roles = payload.get("dictionary_roles")
    role_names = {"chinese_mfa_dictionary", "pinyin_projection_dictionary", "english_pronunciation_dictionary"}
    if not isinstance(roles, dict) or set(roles) != role_names or payload.get("dictionary_roles_digest") != stable_json_digest(roles):
        errors.append("dictionary_roles matrix/digest mismatch")
    else:
        role_paths = set()
        for role in roles.values():
            if (not isinstance(role, dict) or set(role) != {"path", "sha256"}
                    or not isinstance(role["path"], str) or not re.fullmatch(r"[0-9a-f]{64}", role["sha256"])):
                errors.append("dictionary role malformed")
                continue
            role_path = Path(role["path"])
            if not role_path.is_absolute() or role_path in role_paths or role_path.is_symlink() or not role_path.is_file():
                errors.append("dictionary role path invalid/duplicate")
            else:
                role_paths.add(role_path)
                if _sha256(role_path) != role["sha256"]:
                    errors.append("dictionary role hash mismatch")
    subset_path = Path(str(payload.get("english_alignment_subset_path", "")))
    parent_path = Path(str(payload.get("parent_english_manifest_path", "")))
    if not subset_path.is_file() or subset_path.is_symlink() or payload.get("english_alignment_subset_sha256") != _sha256(subset_path):
        errors.append("English alignment subset path/hash invalid")
    if not parent_path.is_file() or parent_path.is_symlink() or payload.get("parent_english_manifest_sha256") != _sha256(parent_path):
        errors.append("parent English manifest path/hash invalid")
    try:
        subset = json.loads(subset_path.read_text(encoding="utf-8")); parent = json.loads(parent_path.read_text(encoding="utf-8"))
        if subset.get("schema") != ALIGNMENT_SUBSET_V2_SCHEMA:
            errors.append("parent/subset schema mismatch")
        errors.extend(_v2_english_manifest_errors(parent, "parent English manifest"))
        if (subset.get("parent_manifest_path") != str(parent_path)
                or subset.get("parent_manifest_sha256") != payload.get("parent_english_manifest_sha256")
                or subset.get("parent_manifest_copy_sha256") != payload.get("parent_english_manifest_sha256")):
            errors.append("parent manifest hash/path identity mismatch")
        if (subset.get("parent_manifest_copy_path") != str(parent_path)
                or not parent_path.is_file() or parent_path.is_symlink()):
            errors.append("parent manifest copied path mismatch")
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("parent/subset JSON unreadable")
    records = payload.get("records")
    if not isinstance(records, list) or [r.get("stem") for r in records if isinstance(r, dict)] != entries_stems:
        errors.append("v2 records order/membership mismatch")
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict) or record.get("stem") not in set(required or []) or record.get("status") != "english_required":
            errors.append("v2 record non-English/missing/excluded")
            continue
        stem = record.get("stem")
        if record.get("schema") != STRICT_ENGLISH_SCHEMA:
            errors.append(f"v2 record schema invalid:{stem}")
        if record.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
            errors.append(f"v2 record canonical unit binding missing:{stem}")
        workspace = _absolute_role(rpaths.get("workspace"), "workspace", errors) if isinstance(rpaths, dict) else None
        ledger = record.get("ledger")
        if not isinstance(ledger, dict):
            errors.append(f"v2 ledger metadata missing:{stem}")
            continue
        if ledger.get("schema") != STRICT_ENGLISH_SCHEMA:
            errors.append(f"v2 ledger schema invalid:{stem}")
        if ledger.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
            errors.append(f"v2 ledger canonical unit binding missing:{stem}")
        try:
            if workspace is None or Path(ledger["path"]).is_absolute() or ".." in Path(ledger["path"]).parts:
                raise ValueError("ledger path role")
            ledger_path = workspace / ledger["path"]
            if ledger_path.is_symlink() or not ledger_path.is_file() or not _within(workspace, ledger_path) or _sha256(ledger_path) != ledger.get("sha256"):
                raise ValueError("ledger hash/path")
            source_records = record.get("source_textgrids")
            if not isinstance(source_records, list) or not source_records:
                raise ValueError("source TextGrid evidence missing")
            for source in source_records:
                rel = Path(source["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("source path role")
                source_path = workspace / rel
                if source_path.is_symlink() or not source_path.is_file() or not _within(workspace, source_path) or _sha256(source_path) != source.get("sha256"):
                    raise ValueError("source TextGrid hash/path")
            ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            errors.extend(_v2_english_ledger_errors(
                ledger_payload, f"v2 ledger payload:{stem}",
                stem=stem, require_segments=True))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"v2 record evidence invalid:{record.get('stem')}:{exc}")
    return sorted(set(errors))


def _verify_english_import_historical_negative(
    path: Path, *, replay_path: Path | None = None,
    formal_path: Path | None = None,
    final_path: Path | None = None,
    subset_path: Path | None = None,
    parent_path: Path | None = None,
    subset_sha256: str | None = None,
    parent_sha256: str | None = None,
    require_final: bool = True,
    config_path: Path | None = None,
    dictionary_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"English import unreadable:{exc}"]
    if not isinstance(payload, dict) or payload.get("schema") != ENGLISH_IMPORT_SCHEMA:
        return ["schema is not strict-replay-english-import-v1"]
    required_keys = {
        "schema", "scope", "run_id", "timestamp_utc", "canonical_manifest_path",
        "canonical_manifest_sha256", "replay_import_manifest_path",
        "replay_import_manifest_sha256", "source_count", "source_membership_digest",
        "eligible_count", "eligible_membership_digest", "exclusion_count",
        "exclusion_membership_digest", "english_required_count",
        "english_required_membership_digest", "producer_revision", "config_sha256",
        "dictionary_sha256", "dictionary_roles", "records",
        "english_alignment_subset_path", "english_alignment_subset_sha256",
        "parent_english_manifest_path", "parent_english_manifest_sha256",
    }
    if set(payload) != required_keys:
        errors.append("English import top-level schema incomplete/extra fields")
    if payload.get("scope") != "strict_replay":
        errors.append("English import scope is not strict_replay")
    if payload.get("producer_revision") != PRODUCER_REVISION:
        errors.append("producer revision mismatch")
    replay_raw = payload.get("replay_import_manifest_path")
    replay_path = replay_path or (Path(replay_raw) if isinstance(replay_raw, str) else Path(""))
    if not replay_path.is_absolute() or ".." in replay_path.parts:
        errors.append("replay import manifest path is not normalized absolute")
    if replay_path.name != "strict_replay_import.json":
        errors.append("replay import binding is not the frozen import manifest")
    if any(part in {"strict_ok_manifest.json", "filtered", "report.json"}
           for part in replay_path.parts):
        errors.append("replay import binding references later-cycle artifact")
    try:
        replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_hash = _sha256(replay_path)
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("replay import manifest missing/unreadable")
        return sorted(set(errors))
    if payload.get("replay_import_manifest_sha256") != replay_hash:
        errors.append("replay import manifest hash mismatch")
    if replay_payload.get("schema") != IMPORT_SCHEMA:
        errors.append("replay import manifest schema mismatch")
    replay_paths = replay_payload.get("paths")
    if not isinstance(replay_paths, dict):
        errors.append("replay paths missing")
        replay_paths = {}
    workspace_path = _absolute_role(replay_paths.get("workspace"), "workspace", errors)
    output_path = _absolute_role(replay_paths.get("output"), "output", errors)
    immutable_raw = replay_paths.get("immutable_import")
    immutable_path = _absolute_role(immutable_raw, "immutable_import", errors)
    if workspace_path is not None and output_path is not None:
        if workspace_path == output_path:
            errors.append("workspace/output role collision")
        expected_import = workspace_path / "strict_replay_import.json"
        if immutable_path != expected_import:
            errors.append("immutable import path is not exact workspace child")
    if immutable_path is not None:
        if replay_path.resolve() != immutable_path:
            errors.append("replay path/payload immutable import mismatch")
        if immutable_path.name != "strict_replay_import.json" or not immutable_path.is_file() or immutable_path.is_symlink():
            errors.append("immutable import is missing/unsafe/bad basename")
        elif replay_path.exists() and not os.path.samestat(immutable_path.stat(), replay_path.stat()):
            errors.append("immutable import opened-path alias mismatch")
    if workspace_path is not None and path.resolve() != workspace_path / "strict_replay_english_import.json":
        errors.append("English import path is not exact workspace child")
    expected_subset = workspace_path / "strict_replay_english_alignment_subset.json" if workspace_path else None
    expected_parent = workspace_path / "en_phones" / "en_alignment_manifest.json" if workspace_path else None
    subset_path = subset_path or (Path(payload.get("english_alignment_subset_path", ""))
                                  if payload.get("english_alignment_subset_path") else None)
    parent_path = parent_path or (Path(payload.get("parent_english_manifest_path", ""))
                                  if payload.get("parent_english_manifest_path") else None)
    if subset_path is None or parent_path is None:
        errors.append("English subset/parent paths are required")
    else:
        if expected_subset is not None and subset_path.resolve() != expected_subset:
            errors.append("English subset path is not exact workspace child")
        if expected_parent is not None and parent_path.resolve() != expected_parent:
            errors.append("parent English manifest path role mismatch")
        try:
            subset_hash = _sha256(subset_path)
            parent_hash = _sha256(parent_path)
            if payload.get("english_alignment_subset_sha256") != subset_hash:
                errors.append("English subset hash mismatch")
            if subset_sha256 is not None and subset_sha256 != subset_hash:
                errors.append("English subset CLI hash mismatch")
            if payload.get("parent_english_manifest_sha256") != parent_hash:
                errors.append("parent English manifest hash mismatch")
            if parent_sha256 is not None and parent_sha256 != parent_hash:
                errors.append("parent English CLI hash mismatch")
            subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))
            if subset_payload.get("schema") != "strict-replay-english-alignment-subset-v1":
                errors.append("English subset schema mismatch")
            if subset_payload.get("parent_manifest_sha256") != parent_hash:
                errors.append("English subset/parent hash binding mismatch")
            selected_stems = sorted({row.get("stem") for row in replay_payload.get("selected_slot_records", [])
                                     if isinstance(row, dict) and isinstance(row.get("stem"), str)})
            if subset_payload.get("selected_stems") != selected_stems:
                errors.append("English subset selected membership mismatch")
            expected_segments = subset_payload.get("expected_segments", [])
            produced_segments = subset_payload.get("produced_segments", [])
            rejected_segments = subset_payload.get("rejected_segments", [])
            digests = subset_payload.get("digests", {})
            if subset_payload.get("counts", {}).get("english_stems") != len(selected_stems):
                errors.append("English subset stem count mismatch")
            if digests.get("selected_stems") != _stable_sha256(selected_stems):
                errors.append("English subset selected digest mismatch")
            if digests.get("expected_segments") != _stable_sha256(sorted(expected_segments)):
                errors.append("English subset expected digest mismatch")
            if digests.get("produced_segments") != _stable_sha256(sorted(produced_segments)):
                errors.append("English subset produced digest mismatch")
            if digests.get("rejected_segments") != _stable_sha256(sorted(
                    row.get("id") for row in rejected_segments if isinstance(row, dict))):
                errors.append("English subset rejected digest mismatch")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            errors.append("English subset/parent evidence unreadable")
    if formal_path is not None and output_path is not None:
        formal_resolved = Path(formal_path)
        if (not formal_resolved.is_absolute() or formal_resolved.parent.resolve() != output_path.resolve()
                or formal_resolved.name != ".pipeline_run_receipt_v2.json"):
            errors.append("formal receipt path is not exact output child")
    if subset_path is None or parent_path is None:
        # Development error: the caller must pass the exact paths it just
        # validated; helper defaults/sibling inference are forbidden.
        return sorted(set(errors + ["English subset/parent paths are required"]))
    _replay_accounting_scope(
        replay_path, formal_path, path, final_path, errors,
        active_subset_path=subset_path, active_parent_path=parent_path,
        require_final=require_final)
    if formal_path is not None and subset_path is not None and parent_path is not None:
        try:
            formal_payload = json.loads(Path(formal_path).read_text(encoding="utf-8"))
            evidence = formal_payload.get("extra", {}).get("strict_replay_evidence", {})
            if evidence.get("english_subset") != str(subset_path.resolve()):
                errors.append("formal receipt/subset path binding mismatch")
            elif evidence.get("english_subset_sha256") != _sha256(subset_path):
                errors.append("formal receipt/subset hash mismatch")
            if evidence.get("parent_english_manifest") != str(parent_path.resolve()):
                errors.append("formal receipt/parent path binding mismatch")
            elif evidence.get("parent_english_sha256") != _sha256(parent_path):
                errors.append("formal receipt/parent hash mismatch")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            errors.append("formal receipt subset/parent evidence unreadable")

    canonical_path = Path(str(payload.get("canonical_manifest_path", "")))
    try:
        if (payload.get("canonical_manifest_sha256") != CANONICAL_SHA256
                or _sha256(canonical_path) != CANONICAL_SHA256):
            errors.append("canonical manifest hash mismatch")
        canonical_data = json.loads(canonical_path.read_text(encoding="utf-8"))
        if canonical_data.get("schema") != CANONICAL_SCHEMA or canonical_data.get("count") != 96:
            errors.append("canonical manifest schema/count mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("canonical manifest missing/unreadable")

    source, eligible, required = _english_required_stems(replay_payload, errors)
    excluded = sorted(set(source) - set(eligible))
    digest_checks = {
        "source": (payload.get("source_count"), payload.get("source_membership_digest"), source),
        "eligible": (payload.get("eligible_count"), payload.get("eligible_membership_digest"), eligible),
        "exclusion": (payload.get("exclusion_count"), payload.get("exclusion_membership_digest"), excluded),
        "english_required": (payload.get("english_required_count"), payload.get("english_required_membership_digest"), required),
    }
    for label, (count, declared, members) in digest_checks.items():
        if count != len(members) or declared != stable_json_digest(sorted(members)):
            errors.append(f"{label} membership count/digest mismatch")

    if config_path is not None:
        try:
            if payload.get("config_sha256") != _sha256(Path(config_path).resolve(strict=True)):
                errors.append("config hash mismatch")
        except OSError:
            errors.append("config path unreadable")
    elif not isinstance(payload.get("config_sha256"), str) or len(payload["config_sha256"]) != 64:
        errors.append("config hash malformed")
    if dictionary_path is not None:
        try:
            english_hash = _sha256(Path(dictionary_path).resolve(strict=True))
            if payload.get("dictionary_sha256") != english_hash:
                errors.append("dictionary hash mismatch")
            roles = payload.get("dictionary_roles")
            if not isinstance(roles, dict):
                errors.append("dictionary role matrix missing")
            else:
                role = roles.get("english_pronunciation_dictionary")
                if (not isinstance(role, dict) or role.get("sha256") != english_hash
                        or role.get("path") != str(Path(dictionary_path).resolve())):
                    errors.append("English dictionary role mismatch")
                for name in ("chinese_mfa_dictionary", "pinyin_projection_dictionary"):
                    other = roles.get(name)
                    if other is not None and (not isinstance(other, dict)
                                              or not isinstance(other.get("sha256"), str)
                                              or len(other["sha256"]) != 64):
                        errors.append(f"{name} role fingerprint malformed")
        except OSError:
            errors.append("dictionary path unreadable")
    elif payload.get("dictionary_sha256") is not None and (
            not isinstance(payload.get("dictionary_sha256"), str)
            or len(payload["dictionary_sha256"]) != 64):
        errors.append("dictionary hash malformed")
    roles = payload.get("dictionary_roles")
    if not isinstance(roles, dict):
        errors.append("dictionary role matrix missing")
    elif any(name not in roles for name in ("chinese_mfa_dictionary",
                                            "pinyin_projection_dictionary",
                                            "english_pronunciation_dictionary")):
        errors.append("dictionary role matrix incomplete")

    workspace_raw = replay_payload.get("paths", {}).get("workspace")
    workspace = Path(workspace_raw) if isinstance(workspace_raw, str) else Path("")
    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("records missing/not-list")
        records = []
    record_stems = [row.get("stem") for row in records if isinstance(row, dict)]
    if record_stems != sorted(record_stems) or len(record_stems) != len(set(record_stems)):
        errors.append("records order/duplicate drift")
    if record_stems != required:
        errors.append("records membership differs from English-required set")
    for record in records:
        if not isinstance(record, dict):
            errors.append("record malformed")
            continue
        stem = record.get("stem")
        if stem not in set(required) or record.get("status") != "english_required":
            errors.append(f"record missing/excluded/non-English:{stem}")
        ledger = record.get("ledger")
        try:
            if not isinstance(ledger, dict) or Path(ledger["path"]).is_absolute() or ".." in Path(ledger["path"]).parts:
                raise ValueError("ledger path not relative")
            ledger_path = _ordinary_file(str(workspace / ledger["path"]), (workspace,))
            if (_sha256(ledger_path) != ledger.get("sha256")
                    or ledger.get("schema") != "strict-en-mfa-v1"):
                raise ValueError("ledger hash/schema")
            ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            if ledger_payload.get("schema") != ledger.get("schema") or ledger_payload.get("stem") != stem:
                raise ValueError("ledger payload identity")
            source_records = record.get("source_textgrids")
            if not isinstance(source_records, list) or not source_records:
                raise ValueError("source TextGrid evidence missing")
            for source in source_records:
                rel = Path(source["path"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("source TextGrid path not relative")
                source_path = _ordinary_file(str(workspace / rel), (workspace,))
                if _sha256(source_path) != source.get("sha256"):
                    raise ValueError("source TextGrid hash")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"record evidence invalid:{stem}:{exc}")
        for value in json.dumps(record, ensure_ascii=False).split('"'):
            if any(token in value for token in ("strict_ok_manifest", "filtered", "report.json", "output/")):
                errors.append(f"record anti-cycle path:{stem}")
                break
    return sorted(set(errors))


def _verify_lifecycle(workspace: Path, output: Path) -> list[str]:
    """Validate v4.2.2 write-once ownership and downstream binding order."""
    errors: list[str] = []
    workspace = Path(workspace); output = Path(output)
    imp = workspace / "strict_replay_import.json"
    eng = workspace / "strict_replay_english_import.json"
    state = workspace / ".strict_replay_stage_state.json"
    formal = output / ".pipeline_run_receipt_v2.json"
    strict = output / "strict_ok_manifest.json"
    final = output / "strict_replay_final_evidence.json"
    for path, label in ((imp, "import"), (eng, "English import"), (state, "stage state"),
                        (formal, "formal receipt"), (final, "final evidence")):
        if not path.is_file() or path.is_symlink():
            errors.append(f"{label} missing/non-regular")
    if imp.is_file():
        sidecar = workspace / "strict_replay_import.sha256"
        if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != _sha256(imp):
            errors.append("import sidecar/tamper hash mismatch")
    # Final evidence is the downstream decision point. Load it first and
    # safely; missing/unsafe final evidence must not trigger speculative reads
    # of a path it may claim.
    try:
        final_payload = json.loads(final.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return errors + ["final evidence unreadable"]
    try:
        import_payload = json.loads(imp.read_text(encoding="utf-8"))
        english_payload = json.loads(eng.read_text(encoding="utf-8"))
        state_payload = json.loads(state.read_text(encoding="utf-8"))
        formal_payload = json.loads(formal.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return errors + ["lifecycle JSON unreadable"]
    replay_paths = import_payload.get("paths") if isinstance(import_payload, dict) else None
    if not isinstance(replay_paths, dict):
        errors.append("lifecycle import paths missing")
    else:
        payload_workspace = _absolute_role(replay_paths.get("workspace"), "workspace", errors)
        payload_output = _absolute_role(replay_paths.get("output"), "output", errors)
        payload_import = _absolute_role(replay_paths.get("immutable_import"), "immutable_import", errors)
        if payload_workspace != workspace or payload_output != output:
            errors.append("lifecycle payload/context path mismatch")
        if payload_workspace == payload_output:
            errors.append("lifecycle workspace/output role collision")
        if payload_import != imp:
            errors.append("lifecycle immutable import path mismatch")
    stage_results = final_payload.get("stage_results")
    strict_rcs = [row.get("return_code") for row in stage_results
                  if isinstance(row, dict) and row.get("stage") == "strict_ok"] if isinstance(stage_results, list) else []
    strict_rc = strict_rcs[-1] if strict_rcs and type(strict_rcs[-1]) is int else None
    binding = final_payload.get("strict_manifest_binding")
    if not isinstance(binding, dict):
        errors.append("strict manifest binding missing")
        binding = {}
    expected_strict = str(strict)
    status = binding.get("status")
    if status not in {"present", "missing"}:
        errors.append("strict manifest binding status invalid")
    if binding.get("expected_path") != expected_strict:
        errors.append("strict manifest binding expected path mismatch")
    declared_sha = binding.get("sha256")
    if status == "present":
        if strict_rc is not None and strict_rc != 0:
            # A present manifest after rc=1 is legal, but must still be safe.
            pass
        if not isinstance(declared_sha, str) or re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None:
            errors.append("strict manifest present hash malformed")
        elif strict.is_symlink() or not strict.is_file():
            errors.append("strict manifest present file unsafe/missing")
        else:
            try:
                json.loads(strict.read_text(encoding="utf-8"))
                if _sha256(strict) != declared_sha:
                    errors.append("strict manifest present hash mismatch")
            except (OSError, ValueError, json.JSONDecodeError):
                errors.append("strict manifest present JSON unreadable")
    elif status == "missing":
        if strict_rc in (None, 0):
            errors.append("strict manifest missing with successful/unknown strict rc")
        if declared_sha is not None:
            errors.append("strict manifest missing hash must be null")
        if binding.get("missing_reason") not in {"not_created", "unreadable", "unsafe_file_type"}:
            errors.append("strict manifest missing reason invalid")
        if binding.get("missing_reason") == "not_created" and strict.exists():
            errors.append("strict manifest not_created claim but path exists")
        if binding.get("missing_reason") == "unsafe_file_type":
            if not strict.exists() or (strict.is_file() and not strict.is_symlink()):
                errors.append("strict manifest unsafe_file_type claim unproven")
        if binding.get("missing_reason") == "unreadable":
            try:
                strict.read_bytes()
            except OSError:
                pass
            else:
                errors.append("strict manifest unreadable claim unproven")
    strict_payload = {}
    if status == "present" and strict.is_file() and not strict.is_symlink():
        try:
            strict_payload = json.loads(strict.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            strict_payload = {}
    if import_payload.get("schema") != REPLAY_V21_SCHEMA:
        errors.append("lifecycle import schema mismatch")
    if english_payload.get("schema") != ENGLISH_IMPORT_V21_SCHEMA:
        errors.append("lifecycle English schema mismatch")
    if state_payload.get("authoritative") is not False:
        errors.append("stage state is authoritative/bound")
    if state_payload.get("evidence") or state_payload.get("hashes"):
        errors.append("stage state carries trust evidence")
    if formal_payload.get("mode") != "strict_replay":
        errors.append("formal receipt is not strict_replay")
    evidence = formal_payload.get("extra", {}).get("strict_replay_evidence")
    if not isinstance(evidence, dict):
        errors.append("formal receipt strict_replay_evidence missing")
        evidence = {}
    expected = {
        "import_manifest": (imp, evidence.get("import_sha256")),
        "english_import": (eng, evidence.get("english_sha256")),
    }
    for key, (path, declared) in expected.items():
        if evidence.get(key) != str(path) or declared != _sha256(path):
            errors.append(f"formal receipt {key} binding/hash mismatch")
    if formal_payload.get("extra", {}).get("strict_replay_receipt") != str(imp):
        errors.append("formal receipt import path ambiguity")
    report = formal_payload.get("derived", {})
    if report and (report.get("output_count", report.get("output", 0))
                   + report.get("filtered_count", report.get("filtered", 0))
                   != report.get("eligible_count", report.get("eligible", 0))):
        errors.append("formal receipt output/filtered conservation mismatch")
    strict_binding = strict_payload.get("pipeline_accounting_receipt", {})
    if strict_binding.get("path") != str(formal) or strict_binding.get("sha256") != _sha256(formal):
        errors.append("strict manifest formal receipt binding mismatch")
    strict_english = strict_payload.get("strict_replay_english_import", strict_payload.get("english_import"))
    if strict_english is None:
        strict_english = strict_payload.get("strict_replay_evidence", {}).get("english_import")
    strict_formal = strict_payload.get("strict_replay_evidence", {}).get("formal_receipt")
    if strict_formal is None:
        errors.append("strict manifest formal evidence binding missing")
    elif (strict_formal.get("path") != str(formal)
          or strict_formal.get("sha256") != _sha256(formal)):
        errors.append("strict manifest formal evidence binding mismatch")
    if strict_english is None:
        errors.append("strict manifest English binding missing")
    elif (strict_english.get("path") != str(eng)
          or strict_english.get("sha256") != _sha256(eng)):
        errors.append("strict manifest English binding mismatch")
    if final_payload.get("schema") != FINAL_EVIDENCE_SCHEMA or final_payload.get("authoritative") is not False:
        errors.append("final evidence schema/authority mismatch")
    final_bindings = {
        "import_sha256": _sha256(imp), "english_import_sha256": _sha256(eng),
        "formal_receipt_sha256": _sha256(formal),
    }
    for key, value in final_bindings.items():
        if final_payload.get(key) != value:
            errors.append(f"final evidence {key} mismatch")
    if status == "present" and strict.is_file() and final_payload.get("strict_manifest_sha256") != _sha256(strict):
        errors.append("final evidence strict manifest hash mismatch")
    stage_results = final_payload.get("stage_results")
    if not isinstance(stage_results, list):
        errors.append("final evidence stage results missing")
    else:
        names = [row.get("stage") for row in stage_results if isinstance(row, dict)]
        if names[:2] != ["postprocess", "strict_ok"]:
            errors.append("final evidence stage ordering mismatch")
        if any(not isinstance(row, dict) or type(row.get("return_code")) is not int
               for row in stage_results):
            errors.append("final evidence stage return code malformed")
        if final_payload.get("global_reasons", []) and not any(
                row.get("return_code") != 0 for row in stage_results if isinstance(row, dict)):
            errors.append("final evidence global reasons without failed stage")
    if imp.resolve() == eng.resolve() or formal.resolve() == final.resolve():
        errors.append("lifecycle equivalent producer overwrite")
    for source, target in ((imp, formal), (eng, formal), (formal, strict), (strict, final)):
        try:
            if os.path.samestat(source.stat(), target.stat()):
                errors.append("lifecycle artifact inode alias")
        except OSError:
            pass
    # Upstream artifacts may bind only to source/canonical/import inputs.  The
    # import's report is legal only as scalar accounting, never as an artifact
    # object carrying paths, hashes, or nested downstream evidence.
    report_value = import_payload.get("report")
    if report_value is not None:
        if not isinstance(report_value, dict) or any(
                isinstance(value, (dict, list))
                or (isinstance(value, int) and value < 0)
                or (isinstance(value, str) and value not in {"healthy", "failed", "strict_replay"})
                for value in report_value.values()):
            errors.append("import report violates scalar allowlist")
    for label, payload in (("import", import_payload), ("English", english_payload)):
        text = json.dumps(payload, ensure_ascii=False)
        forbidden = ("strict_ok_manifest", "strict_replay_final_evidence",
                     ".pipeline_run_receipt_v2", "strict_replay_stage_state")
        if any(token in text for token in forbidden):
            errors.append(f"{label} upstream/downstream cycle binding")
        if label == "English":
            keys = []
            def collect(value: object) -> None:
                if isinstance(value, dict):
                    keys.extend(str(key).lower() for key in value)
                    for child in value.values(): collect(child)
                elif isinstance(value, list):
                    for child in value: collect(child)
            collect(payload)
            if any(any(word in key for word in ("formal", "output", "filtered", "report",
                                                "strict_manifest", "final", "stage_state"))
                   for key in keys):
                errors.append("English ancestor downstream binding")
        downstream_hashes = {_sha256(formal), _sha256(strict), _sha256(final)}
        if any(value in downstream_hashes for value in re.findall(r"[0-9a-f]{64}", text)):
            errors.append(f"{label} aliases downstream artifact hash")
    return sorted(set(errors))


def _lifecycle_overall_failed(output: Path) -> bool:
    """Overall CLI verdict: any strict_ok nonzero return is a failure."""
    try:
        payload = json.loads((Path(output) / "strict_replay_final_evidence.json").read_text(encoding="utf-8"))
        rows = payload.get("stage_results", [])
        return any(row.get("stage") == "strict_ok" and row.get("return_code") != 0
                   for row in rows if isinstance(row, dict))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return True


def verify(receipt_path: Path, output_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    receipt_path = Path(receipt_path)
    try:
        initial = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        initial = None
    if isinstance(initial, dict) and initial.get("schema") == ENGLISH_IMPORT_SCHEMA:
        return _verify_english_import_historical_negative(receipt_path)
    receipt = _load_receipt(receipt_path, errors)
    if receipt is None:
        return sorted(set(errors))
    if "scope" in receipt and receipt.get("scope") != "strict_replay":
        errors.append("strict replay receipt scope is not strict_replay")
        if any(key in receipt for key in ("paths", "immutable_import", "replay_import_manifest_path", "english_subset")):
            errors.append("production receipt carries replay path fields")
    if isinstance(receipt.get("paths"), dict) and "immutable_import" in receipt["paths"]:
        paths = receipt["paths"]
        ws = _absolute_role(paths.get("workspace"), "workspace", errors)
        out = _absolute_role(paths.get("output"), "output", errors)
        imm = _absolute_role(paths.get("immutable_import"), "immutable_import", errors)
        if ws is not None and out is not None and ws == out:
            errors.append("workspace/output role collision")
        if ws is not None and imm != ws / "strict_replay_import.json":
            errors.append("immutable import exact path mismatch")
        if imm is not None and receipt_path.resolve() != imm:
            errors.append("payload/import actual path mismatch")
    paths = receipt.get("paths", {})
    output = Path(output_dir) if output_dir is not None else Path(paths.get("output", ""))
    if not output.is_dir():
        errors.append("output directory missing")
    if receipt_path.resolve().parent != output.resolve():
        errors.append("receipt/output binding mismatch")
    sidecar = output / "strict_replay_import.sha256"
    if sidecar.is_file():
        try:
            if sidecar.read_text(encoding="ascii").strip() != _sha256(receipt_path):
                errors.append("strict replay import sidecar hash mismatch")
        except OSError:
            errors.append("strict replay import sidecar unreadable")
    canonical = receipt.get("canonical", {})
    cpath = Path(canonical.get("path", "")) if isinstance(canonical, dict) else Path("")
    try:
        cdata = json.loads(cpath.read_text(encoding="utf-8"))
        if (canonical.get("schema") != CANONICAL_SCHEMA
                or canonical.get("sha256") != CANONICAL_SHA256
                or _sha256(cpath) != CANONICAL_SHA256
                or cdata.get("count") != 96
                or not isinstance(cdata.get("entries"), list)
                or len(cdata["entries"]) != 96):
            errors.append("canonical identity/hash/slot count mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("canonical manifest unreadable")
        cdata = {}
    selected, selected_stems = _selected_pairs(receipt, errors)
    subset = _load_subset(receipt, errors)
    if subset is None:
        return sorted(set(errors))
    subset_text = json.dumps(subset, ensure_ascii=False)
    if any(token in subset_text for token in ("strict_replay_final_evidence", "strict_ok_manifest",
                                              ".pipeline_run_receipt_v2", '"filtered"', '"report"')):
        errors.append("English subset downstream anti-cycle binding")
    if "parent_global_manifest" in subset:
        pg = subset.get("parent_global_manifest")
        if (not isinstance(pg, dict) or not isinstance(pg.get("authoritative_source"), dict)
                or not isinstance(pg.get("workspace_copy"), dict)
                or pg["authoritative_source"].get("path") == pg["workspace_copy"].get("path")):
            errors.append("parent_global_manifest role binding invalid")

    # Bind the subset to this exact import receipt and to the canonical pilot
    # membership.  Both spellings are accepted for compatibility, but at
    # least one authoritative digest is mandatory.
    receipt_digest = _sha256(receipt_path)
    # Producers may bind the subset to the immutable import digest sidecar
    # (or a top-level pre-subset digest) to avoid a self-referential JSON hash.
    bound_digest = receipt.get("strict_replay_import_sha256")
    if not isinstance(bound_digest, str) and sidecar.is_file():
        try:
            bound_digest = sidecar.read_text(encoding="ascii").strip()
        except OSError:
            bound_digest = None
    import_digest = (subset.get("import_receipt_sha256")
                     or subset.get("strict_replay_receipt_sha256")
                     or subset.get("receipt_sha256"))
    if import_digest != (bound_digest or receipt_digest):
        errors.append("English subset/import receipt hash mismatch")
    expected_stems = sorted(selected_stems)
    subset_stems = subset.get("selected_stems", subset.get("stems"))
    if not isinstance(subset_stems, list) or any(not _safe_stem(item) for item in subset_stems):
        errors.append("English subset membership missing/unsafe")
        subset_stems = []
    if sorted(set(subset_stems)) != expected_stems:
        errors.append("English subset membership is not canonical selected subset")
    membership_hash = (subset.get("canonical_selected_stems_sha256")
                       or subset.get("selected_stems_sha256")
                       or subset.get("membership_sha256"))
    if not isinstance(membership_hash, str) or membership_hash != _stable_sha256(expected_stems):
        errors.append("English subset global membership hash mismatch")

    records = subset.get("ledgers", subset.get("entries", subset.get("records")))
    if not isinstance(records, list):
        errors.append("English subset ledger records missing")
        records = []
    seen: set[str] = set()
    english_root_raw = paths.get("english_root")
    roots = (output, Path(english_root_raw)) if isinstance(english_root_raw, str) and english_root_raw else (output,)
    for record in records:
        if not isinstance(record, dict):
            errors.append("English ledger record malformed")
            continue
        stem = record.get("stem")
        if not _safe_stem(stem) or stem in seen:
            errors.append("English ledger duplicate/unsafe stem")
            continue
        seen.add(stem)
        if stem not in selected_stems:
            errors.append(f"English ledger stem outside canonical subset:{stem}")
        schema = record.get("schema")
        if schema != STRICT_ENGLISH_SCHEMA:
            if schema == HISTORICAL_STRICT_ENGLISH_SCHEMA:
                errors.append(f"historical-v1 ledger used as current-v2:{stem}")
            else:
                errors.append(f"current-v2 ledger schema invalid:{stem}")
        if record.get("canonical_units") != CANONICAL_ENGLISH_UNITS_SCHEMA:
            errors.append(f"current-v2 ledger canonical unit binding missing:{stem}")
        source = record.get("path", record.get("source"))
        try:
            ledger_path = _ordinary_file(source, roots)
            declared = record.get("sha256")
            if not isinstance(declared, str) or _sha256(ledger_path) != declared:
                errors.append(f"English ledger hash mismatch:{stem}")
            ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            errors.extend(_v2_english_ledger_errors(
                ledger_payload, f"English ledger payload:{stem}",
                stem=stem, require_segments=True))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            errors.append(f"English ledger missing/unreadable:{stem}")
    # A current-v2 English subset may contain only the selected stems that
    # actually have English evidence.  If the producer records that narrower
    # set explicitly, require exact equality; otherwise require only the
    # canonical-subset upper bound (never a canonical-external stem).
    english_stems = subset.get("english_stems")
    if english_stems is not None:
        if (not isinstance(english_stems, list)
                or any(not _safe_stem(item) for item in english_stems)
                or set(english_stems) != seen):
            errors.append("English ledger authoritative membership incomplete")
    elif not seen <= selected_stems:
        errors.append("English ledger authoritative membership incomplete")
    english_hash = subset.get("english_stems_sha256")
    if english_hash is not None:
        if english_hash != _stable_sha256(sorted(seen)):
            errors.append("English ledger membership hash mismatch")

    if receipt.get("global_reasons"):
        errors.append("strict replay receipt has global reasons")
    return sorted(set(errors))


def verify_english_import_active(
    path: Path,
    *,
    replay_path: Path | None = None,
    formal_path: Path | None = None,
    final_path: Path | None = None,
    subset_path: Path | None = None,
    parent_path: Path | None = None,
    subset_sha256: str | None = None,
    parent_sha256: str | None = None,
    require_final: bool = False,
    config_path: Path | None = None,
    dictionary_path: Path | None = None,
) -> list[str]:
    """Verify the active English import contract.

    This is the sole public entry point for current consumers.  It accepts
    only the v2.1 import schema, runs the exact/vector/roles verifier, and
    checks every explicitly supplied path/hash binding without discovering
    sibling artifacts.  The historical
    ``_verify_english_import_historical_negative`` helper is
    intentionally kept private for negative-compatibility fixtures only.
    """
    errors: list[str] = []

    def resolve_file(raw: Path | None, label: str) -> Path | None:
        if raw is None:
            return None
        try:
            candidate = Path(raw)
            if (not candidate.is_absolute() or ".." in candidate.parts
                    or candidate.is_symlink() or not candidate.is_file()):
                raise ValueError("not a normalized ordinary file")
            return candidate.resolve(strict=True)
        except (OSError, TypeError, ValueError):
            errors.append(f"{label} explicit binding invalid")
            return None

    import_path = resolve_file(Path(path), "English import")
    if import_path is None:
        return sorted(set(errors))
    try:
        payload = json.loads(import_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"v2.1 English import unreadable:{exc}"]
    if not isinstance(payload, dict) or payload.get("schema") != ENGLISH_IMPORT_V21_SCHEMA:
        return ["active English consumer requires strict-replay-english-import-v2.1"]

    replay_bound = resolve_file(replay_path, "replay import")
    subset_bound = resolve_file(subset_path, "English subset")
    parent_bound = resolve_file(parent_path, "parent English manifest")
    formal_bound = resolve_file(formal_path, "formal receipt")
    final_bound = resolve_file(final_path, "final evidence")
    config_bound = resolve_file(config_path, "config")
    dictionary_bound = resolve_file(dictionary_path, "dictionary")
    try:
        errors.extend(_verify_english_import_v21(import_path,
                                                 replay_path=replay_bound))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        # Malformed v2.1 values must be a verifier failure, never an audit
        # process exception or an accidental acceptance.
        errors.append(f"v2.1 import value validation failed:{exc}")

    def compare_path(arg: Path | None, key: str, label: str) -> None:
        if arg is None:
            return
        declared = payload.get(key)
        if declared != str(arg):
            errors.append(f"{label} explicit binding mismatch")

    def compare_hash(arg: Path | None, declared_key: str, explicit_hash: str | None,
                     label: str) -> None:
        if arg is None:
            return
        actual = _sha256(arg)
        if payload.get(declared_key) != actual:
            errors.append(f"{label} hash binding mismatch")
        if explicit_hash is not None and explicit_hash != actual:
            errors.append(f"{label} explicit hash mismatch")

    compare_path(replay_bound, "replay_import_manifest_path", "replay import")
    compare_path(subset_bound, "english_alignment_subset_path", "English subset")
    compare_path(parent_bound, "parent_english_manifest_path", "parent English manifest")
    compare_hash(subset_bound, "english_alignment_subset_sha256", subset_sha256,
                 "English subset")
    compare_hash(parent_bound, "parent_english_manifest_sha256", parent_sha256,
                 "parent English manifest")
    if config_bound is not None and payload.get("config_sha256") != _sha256(config_bound):
        errors.append("config hash binding mismatch")
    if dictionary_bound is not None:
        role = payload.get("dictionary_roles", {}).get(
            "english_pronunciation_dictionary", {})
        if (role.get("path") != str(dictionary_bound)
                or role.get("sha256") != _sha256(dictionary_bound)):
            errors.append("dictionary explicit binding mismatch")
    # When the caller supplies the formal accounting receipt, validate its
    # explicit evidence DAG as well.  This is intentionally opt-in: active
    # consumers may verify an import before postprocess creates a final
    # evidence artifact, but must never silently discover one beside it.
    replay_actual = replay_bound
    if replay_actual is None:
        replay_raw = payload.get("replay_import_manifest_path")
        replay_actual = resolve_file(Path(replay_raw), "replay import") \
            if isinstance(replay_raw, str) else None
    subset_actual = subset_bound
    if subset_actual is None:
        subset_raw = payload.get("english_alignment_subset_path")
        subset_actual = resolve_file(Path(subset_raw), "English subset") \
            if isinstance(subset_raw, str) else None
    parent_actual = parent_bound
    if parent_actual is None:
        parent_raw = payload.get("parent_english_manifest_path")
        parent_actual = resolve_file(Path(parent_raw), "parent English manifest") \
            if isinstance(parent_raw, str) else None
    if formal_bound is not None and replay_actual and subset_actual and parent_actual:
        _replay_accounting_scope(
            replay_actual, formal_bound, import_path, final_bound, errors,
            active_subset_path=subset_actual, active_parent_path=parent_actual,
            require_final=require_final)
    if require_final and formal_bound is None:
        errors.append("formal receipt path is required")
    if require_final and final_bound is None:
        errors.append("final evidence path is required")
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict-replay English current-v2 subset")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replay-import", type=Path, default=None,
                        help="strict_replay_import.json when verifying producer v4.2.1 output")
    parser.add_argument("--formal-receipt", type=Path, default=None,
                        help="explicit output/.pipeline_run_receipt_v2.json (required for English import)")
    parser.add_argument("--final-evidence", type=Path, default=None,
                        help="explicit output/strict_replay_final_evidence.json (required for English import)")
    parser.add_argument("--subset", type=Path, default=None,
                        help="explicit strict_replay_english_alignment_subset.json")
    parser.add_argument("--parent-manifest", type=Path, default=None,
                        help="explicit copied parent-global English manifest")
    parser.add_argument("--subset-sha256", default=None)
    parser.add_argument("--parent-sha256", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dictionary", type=Path, default=None)
    parser.add_argument("--lifecycle-workspace", type=Path, default=None)
    parser.add_argument("--lifecycle-output", type=Path, default=None)
    args = parser.parse_args()
    lifecycle_mode = args.lifecycle_workspace is not None or args.lifecycle_output is not None
    if args.lifecycle_workspace is not None or args.lifecycle_output is not None:
        if args.lifecycle_workspace is None or args.lifecycle_output is None:
            errors = ["both --lifecycle-workspace and --lifecycle-output are required"]
        else:
            errors = _verify_lifecycle(args.lifecycle_workspace, args.lifecycle_output)
    else:
        try:
            schema = json.loads(args.receipt.read_text(encoding="utf-8")).get("schema")
        except (OSError, ValueError, json.JSONDecodeError):
            schema = None
        if schema == ENGLISH_IMPORT_V21_SCHEMA:
            errors = verify_english_import_active(
                args.receipt,
                replay_path=args.replay_import,
                formal_path=args.formal_receipt,
                final_path=args.final_evidence,
                subset_path=args.subset,
                parent_path=args.parent_manifest,
                subset_sha256=args.subset_sha256,
                parent_sha256=args.parent_sha256,
                require_final=False,
                config_path=args.config,
                dictionary_path=args.dictionary,
            )
        else:
            errors = ["active English consumer requires strict-replay-english-import-v2.1"]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if lifecycle_mode and _lifecycle_overall_failed(args.lifecycle_output):
        print("ERROR: overall lifecycle verdict failed: strict_ok return_code != 0")
        return 1
    print(f"strict replay English subset verified: {args.receipt}")
    return 0


# Stable import name for synthetic fixture tests.
verify_subset = verify


if __name__ == "__main__":
    raise SystemExit(main())
