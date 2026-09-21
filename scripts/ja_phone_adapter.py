"""Audited semantic phone graph and Japanese MFA v3 adapter.

Open JTalk symbols, semantic events and Japanese MFA labels are separate
domains.  The adapter creates explicit relation edges between them and keeps
many-to-many mora links for length, gemination and nasal events.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .ja_en_schema import (
        JAContractError,
        StageResult,
        atomic_write_json,
        make_occurrence_alias,
        make_receipt,
        sha256_file,
        stable_digest,
        validate_alias_rows,
    )
except ImportError:  # direct script execution
    from ja_en_schema import JAContractError, StageResult, atomic_write_json, make_occurrence_alias, make_receipt, sha256_file, stable_digest, validate_alias_rows


DEFAULT_METADATA = None
DEFAULT_DICTIONARY = None
DEFAULT_EN_METADATA = None
SEMANTIC_VERSION = "ja-semantic-phone-graph-v1"

_MFA_PHONES = frozenset({
    "a", "aː", "b", "bʲ", "bʲː", "bː", "c", "cː", "d", "dz", "dzː", "dʑ", "dʑː", "dʲ", "dʲː", "dː", "e", "eː", "h", "hː", "i", "iː", "i̥", "j", "k", "kː", "m", "mʲ", "mʲː", "mː", "n", "nː", "o", "oː", "p", "pʲ", "pʲː", "pː", "s", "sː", "t", "ts", "tsː", "tɕ", "tɕː", "tʲ", "tʲː", "tː", "v", "vʲ", "w", "wː", "z", "ç", "çː", "ŋ", "ɕ", "ɕː", "ɟ", "ɟː", "ɡ", "ɡː", "ɨ", "ɨː", "ɨ̥", "ɯ", "ɯː", "ɯ̥", "ɰ̃", "ɲ", "ɲː", "ɴ", "ɴː", "ɸ", "ɸʲ", "ɸʲː", "ɸː", "ɾ", "ɾʲ", "ɾʲː", "ɾː", "ʑ", "ʔ",
})

# This table is intentionally a semantic model map: the frontend's `ky`,
# `cl`, and `N` are events, not Japanese MFA spellings.
_FRONTEND_TO_MFA = {
    "a": "a", "i": "i", "u": "ɯ", "e": "e", "o": "o",
    "b": "b", "d": "d", "f": "ɸ", "g": "ɡ", "h": "h", "j": "j",
    "k": "k", "m": "m", "n": "n", "p": "p", "r": "ɾ", "s": "s",
    "t": "t", "w": "w", "y": "j", "z": "z", "ts": "ts", "ch": "tɕ",
    "sh": "ɕ", "ky": "c", "gy": "ɟ", "ny": "ɲ", "hy": "ç", "my": "mʲ",
    "by": "bʲ", "py": "pʲ", "ry": "ɾʲ", "dy": "dʲ", "ty": "tʲ",
    "N": "ɴ", "cl": "cl", "q": "ʔ",
}
_GOLDEN = {
    "サクラ": ("s", "a", "k", "ɯ", "ɾ", "a"),
    "トウキョウ": ("t", "oː", "c", "oː"),
    "ガッコウ": ("ɡ", "a", "kː", "oː"),
    "コンニチハ": ("k", "o", "ɲː", "i", "tɕ", "i", "w", "a"),
}
_SMALL = frozenset("ゃゅょぁぃぇぉゎャュョァィェォヮ")


def _hiragana(text: str) -> str:
    return "".join(chr(ord(char) - 0x60) if "ァ" <= char <= "ヺ" else char for char in text)


def split_mora(reading: str) -> list[str]:
    result: list[str] = []
    for char in _hiragana(reading):
        if result and char in _SMALL:
            result[-1] += char
        else:
            result.append(char)
    return result


def load_japanese_mfa_inventory(metadata_path: Path | str | None = None, *, acoustic_archive_path: Path | str | None = None, acoustic_archive_sha256: str | None = None) -> dict[str, Any]:
    configured = metadata_path or os.environ.get("JA_JAPANESE_MFA_METADATA")
    if not configured:
        raise JAContractError("mfa_inventory_mismatch", "Japanese MFA metadata must be configured")
    path = Path(configured)
    if not path.is_file():
        raise JAContractError("mfa_inventory_mismatch", "audited Japanese MFA metadata is unavailable", str(path))
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        phones = set(metadata.get("phones", []))
    except (OSError, json.JSONDecodeError) as exc:
        raise JAContractError("mfa_inventory_mismatch", "Japanese MFA metadata is invalid", str(path)) from exc
    if not phones or not phones.issubset(_MFA_PHONES):
        raise JAContractError("mfa_inventory_mismatch", "metadata inventory is not the audited v3 inventory", str(path))
    archive = acoustic_archive_path or metadata.get("acoustic_archive")
    expected_archive_sha = acoustic_archive_sha256 or metadata.get("acoustic_archive_sha256") or metadata.get("archive_sha256")
    archive_digest = None
    if archive:
        archive_path = Path(str(archive))
        if not archive_path.is_file():
            raise JAContractError("mfa_inventory_mismatch", "configured Japanese acoustic archive is unavailable", str(archive_path))
        archive_digest = sha256_file(archive_path)
        if not expected_archive_sha or archive_digest != str(expected_archive_sha):
            raise JAContractError("mfa_inventory_mismatch", "Japanese metadata/archive hash binding is missing or mismatched", str(path))
    coverage = "archive_bound" if archive_digest else "inventory_declared"
    return {"phones": frozenset(phones), "version": metadata.get("version", "3.0"), "source": str(path), "sha256": sha256_file(path), "coverage_status": coverage, "acoustic_archive": str(archive) if archive else None, "acoustic_archive_sha256": archive_digest}


def load_english_arpa_inventory(metadata_path: Path | str | None = None, *, acoustic_archive_path: Path | str | None = None, acoustic_archive_sha256: str | None = None) -> dict[str, Any]:
    configured = metadata_path or os.environ.get("JA_ENGLISH_MFA_METADATA")
    if not configured:
        raise JAContractError("mfa_inventory_mismatch", "English ARPA metadata must be configured")
    path = Path(configured)
    if not path.is_file():
        raise JAContractError("mfa_inventory_mismatch", "audited English ARPA metadata is unavailable", str(path))
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        phones = set(metadata.get("phones", []))
    except (OSError, json.JSONDecodeError) as exc:
        raise JAContractError("mfa_inventory_mismatch", "English ARPA metadata is invalid", str(path)) from exc
    if not phones:
        raise JAContractError("mfa_inventory_mismatch", "English ARPA metadata has no inventory", str(path))
    archive = acoustic_archive_path or metadata.get("acoustic_archive")
    expected_archive_sha = acoustic_archive_sha256 or metadata.get("acoustic_archive_sha256") or metadata.get("archive_sha256")
    archive_digest = None
    if archive:
        archive_path = Path(str(archive))
        if not archive_path.is_file():
            raise JAContractError("mfa_inventory_mismatch", "configured English acoustic archive is unavailable", str(archive_path))
        archive_digest = sha256_file(archive_path)
        if not expected_archive_sha or archive_digest != str(expected_archive_sha):
            raise JAContractError("mfa_inventory_mismatch", "English metadata/archive hash binding is missing or mismatched", str(path))
    return {"phones": frozenset(phones), "version": metadata.get("version", "3.0"), "source": str(path), "sha256": sha256_file(path), "coverage_status": "archive_bound" if archive_digest else "inventory_declared", "acoustic_archive": str(archive) if archive else None, "acoustic_archive_sha256": archive_digest}


def resolve_english_pronunciation(surface: str, pronunciation: Sequence[str] | None, *, manual: bool = False, metadata_path: Path | str | None = None, inventory: Mapping[str, Any] | None = None, acoustic_archive_path: Path | str | None = None, acoustic_archive_sha256: str | None = None) -> dict[str, Any]:
    """Resolve a caller-provided ARPA pronunciation without Japanese G2P.

    The dictionary lookup/OOV policy belongs to the English owner.  W2 only
    validates the native inventory and keeps unresolved OOVs explicit.
    """
    if not pronunciation:
        return {"status": "unresolved", "surface": surface, "pronunciation": [], "evidence": "none"}
    inventory = dict(inventory) if inventory is not None else load_english_arpa_inventory(metadata_path, acoustic_archive_path=acoustic_archive_path, acoustic_archive_sha256=acoustic_archive_sha256)
    phones = [str(phone) for phone in pronunciation]
    unknown = sorted(set(phones) - set(inventory["phones"]))
    if unknown:
        raise JAContractError("mfa_inventory_mismatch", f"English ARPA phones are unknown: {unknown!r}")
    return {"status": "manual_verified" if manual else "dictionary_verified", "surface": surface, "pronunciation": phones, "evidence": "manual" if manual else "matching_arpa_dictionary", "inventory": inventory}


def _target_phones(source: Sequence[str], reading: str) -> tuple[list[str], list[list[int]]]:
    mapped: list[str] = []
    groups: list[list[int]] = []
    index = 0
    while index < len(source):
        token = source[index]
        if token == "cl":
            if index + 1 >= len(source):
                raise JAContractError("semantic_parse_failed", "word-final geminate has no onset")
            onset = _FRONTEND_TO_MFA.get(source[index + 1])
            if onset is None:
                raise JAContractError("semantic_parse_failed", f"unsupported geminate onset {source[index + 1]!r}")
            mapped.append(onset + "ː")
            groups.append([index, index + 2])
            index += 2
            continue
        if token == "N" and index + 1 < len(source):
            following = source[index + 1]
            nasal = "ɲ" if following in {"n", "ny", "j"} else ("m" if following in {"b", "p", "m", "by", "py", "my"} else ("ŋ" if following in {"k", "g", "ky", "gy"} else ("ɰ̃" if following in {"w", "y"} else ("n" if following in {"t", "d", "s", "z", "ts", "ch", "sh"} else "ɴ"))))
            mapped.append(nasal + ("ː" if following in {"n", "ny"} else ""))
            if following in {"n", "ny"}:
                groups.append([index, index + 2])
                index += 2
            else:
                groups.append([index, index + 1])
                index += 1
            continue
        if token in {"I", "U"}:
            current = "i̥" if token == "I" else "ɨ̥"
        elif token == "j":
            kana = _hiragana(reading)
            current = "dʑ" if any(mark in kana for mark in ("じ", "ぢ")) else "j"
        elif token == "h" and index + 1 < len(source) and source[index + 1] == "I":
            current = "ç"
        elif token == "m" and index + 1 < len(source) and source[index + 1] in {"i", "I"}:
            current = "mʲ"
        elif token == "k" and index + 1 < len(source) and source[index + 1] in {"i", "I"}:
            current = "c"
        else:
            current = _FRONTEND_TO_MFA.get(token)
        if current is None:
            raise JAContractError("semantic_parse_failed", f"unsupported Open JTalk phone {token!r}")
        if token == "u" and ((index > 0 and source[index - 1] in {"s", "sh"}) or (_hiragana(reading) == "みず" and index > 0 and source[index - 1] == "z")):
            current = "ɨ"
        if index + 1 < len(source) and source[index + 1] == token and current in {"a", "i", "ɯ", "e", "o"}:
            mapped.append(current + "ː")
            groups.append([index, index + 2])
            index += 2
        else:
            mapped.append(current)
            groups.append([index, index + 1])
            index += 1
    return mapped, groups


def _source_mora_indices(source: Sequence[str], mora_count: int) -> list[int]:
    """Parse Open JTalk's CV/event stream into explicit mora ownership."""
    if mora_count <= 0:
        raise JAContractError("semantic_relation_ambiguous", "frontend unit has no mora count")
    result: list[int] = []
    cursor = 0
    vowels = {"a", "i", "u", "e", "o", "I", "U"}
    for phone in source:
        if cursor >= mora_count:
            raise JAContractError("semantic_relation_ambiguous", "frontend phone stream exceeds reading morae")
        result.append(cursor)
        if phone in {"cl", "N"}:
            cursor += 1
        elif phone in vowels:
            cursor += 1
    if cursor != mora_count:
        # A devoiced vowel still has a mora; only an explicit parse can
        # account for it.  Unknown streams are blocked rather than guessed.
        raise JAContractError("mora_phone_relation_unresolved", f"phone stream covers {cursor} of {mora_count} morae")
    return result


def openjtalk_to_semantic(unit: Mapping[str, Any]) -> dict[str, Any]:
    phones = list(unit.get("locked_openjtalk_phones", unit.get("phones", unit.get("phonemes", []))))
    reading = str(unit.get("locked_reading", unit.get("read", unit.get("reading", ""))))
    if not phones or not reading:
        raise JAContractError("semantic_parse_failed", "frontend unit has no reading or phones")
    morae = split_mora(reading)
    if not morae:
        raise JAContractError("semantic_parse_failed", "reading has no mora")
    target, source_groups = _target_phones(phones, reading)
    source_mora = _source_mora_indices(phones, int(unit.get("locked_mora_count", unit.get("mora_count", len(split_mora(reading)))) or 0))
    nodes: list[dict[str, Any]] = []
    for index, mora in enumerate(morae):
        nodes.append({"id": f"mora_{index:04d}", "kind": "mora", "surface": mora, "reading": mora, "mora_index": index, "source_span": unit.get("canonical_span"), "source_span_domain": "canonical_text"})
    for index, phone in enumerate(phones):
        nodes.append({"id": f"oj_{index:04d}", "kind": "openjtalk_phone", "phone": phone, "devoiced_candidate": phone in {"I", "U"}, "source_span": unit.get("canonical_span"), "source_span_domain": "canonical_text"})
    phone_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    if len(target) != len(source_groups):
        raise JAContractError("semantic_relation_ambiguous", "semantic phone/source relation cardinality is unresolved")
    for index in range(len(target)):
        phone = target[index]
        group = source_groups[index]
        related_indices = sorted(set(source_mora[group[0]:group[1]]))
        if not related_indices:
            raise JAContractError("mora_phone_relation_unresolved", "phone has no explicit source mora relation")
        mora_ids = [f"mora_{mora_index:04d}" for mora_index in related_indices]
        node_id = f"sem_{index:04d}"
        phone_nodes.append({"id": node_id, "kind": "semantic_phone", "phone": phone, "devoiced_candidate": any(phones[i] in {"I", "U"} for i in range(group[0], group[1])), "source_openjtalk_ids": [f"oj_{i:04d}" for i in range(group[0], group[1])], "mora_ids": mora_ids, "source_span": unit.get("canonical_span"), "source_span_domain": "canonical_text"})
        edges.append({"relation": "openjtalk_to_semantic", "source_ids": [f"oj_{i:04d}" for i in range(group[0], group[1])], "target_ids": [node_id]})
        for mora_id in mora_ids:
            edges.append({"relation": "mora_phone", "mora_id": mora_id, "phone_id": node_id})
    return {
        "schema": SEMANTIC_VERSION,
        "version": SEMANTIC_VERSION,
        "token_id": unit.get("token_id"),
        "candidate_id": unit.get("candidate_id"),
        "surface": unit.get("surface", ""),
        "reading": reading,
        "source_phones": phones,
        "mora_nodes": [node for node in nodes if node["kind"] == "mora"],
        "frontend_phone_nodes": [node for node in nodes if node["kind"] == "openjtalk_phone"],
        "semantic_phone_nodes": phone_nodes,
        "nodes": nodes + phone_nodes,
        "edges": edges,
        "relation_semantics": "many_to_many_no_synthetic_mora_timing",
        "provenance": {"frontend": "pyopenjtalk-plus", "adapter": SEMANTIC_VERSION},
        "source_spans": {"canonical": unit.get("canonical_span"), "original": unit.get("orig_span")},
    }


def semantic_to_japanese_mfa_v3(graph: Mapping[str, Any], inventory: Mapping[str, Any] | Sequence[str] | None = None, *, dictionary_path: Path | str | None = None) -> dict[str, Any]:
    if graph.get("schema") != SEMANTIC_VERSION:
        raise JAContractError("semantic_parse_failed", "unknown semantic graph schema")
    if inventory is None:
        audited = load_japanese_mfa_inventory()
        phones = set(audited["phones"])
        inventory_info = audited
    elif isinstance(inventory, Mapping):
        phones = set(inventory.get("phones", []))
        inventory_info = dict(inventory)
    else:
        phones = set(inventory)
        inventory_info = {"coverage_status": "caller_supplied_unverified", "phones": sorted(phones)}
    result = [dict(node) for node in graph.get("semantic_phone_nodes", [])]
    unknown = [node["phone"] for node in result if node.get("phone") not in phones]
    if unknown:
        raise JAContractError("mfa_phone_unsupported", f"phones are outside Japanese MFA inventory: {unknown!r}")
    dictionary_status = verify_dictionary_roundtrip(graph, [node["phone"] for node in result], dictionary_path)
    return {
        "schema": SEMANTIC_VERSION,
        "token_id": graph.get("token_id"),
        "candidate_id": graph.get("candidate_id"),
        "phones": [node["phone"] for node in result],
        "phone_nodes": result,
        "mora_nodes": graph.get("mora_nodes", []),
        "edges": graph.get("edges", []),
        "representation_provenance": {"frontend": "pyopenjtalk-plus", "semantic_graph": SEMANTIC_VERSION, "target_adapter": "japanese_mfa_v3", "timing": "none"},
        "inventory": {key: (sorted(value) if isinstance(value, (set, frozenset)) else value) for key, value in inventory_info.items()},
        "roundtrip": {"status": dictionary_status, "reading": graph.get("reading"), "dictionary_source": str(dictionary_path) if dictionary_path else None},
    }


@lru_cache(maxsize=1)
def _golden_dictionary_rows(path_text: str) -> dict[str, list[tuple[str, ...]]]:
    path = Path(path_text)
    rows: dict[str, list[tuple[str, ...]]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        word = fields[0]
        phones = fields[1:]
        if len(phones) >= 5 and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value) for value in phones[:4]):
            phones = phones[4:]
        rows.setdefault(word, []).append(tuple(phones))
    return rows


def verify_dictionary_roundtrip(graph: Mapping[str, Any], phones: Sequence[str], dictionary_path: Path | str | None = None) -> str:
    """Verify audited golden words when their official dictionary row exists.

    OOV words are intentionally allowed: their occurrence alias receives the
    same explicit pronunciation after inventory checking.  The golden rows
    are a narrow audit of the semantic adapter and do not become a fallback
    pronunciation source.
    """
    reading = str(graph.get("reading", ""))
    if not dictionary_path:
        raise JAContractError("dictionary_asset_missing", "Japanese dictionary must be configured for golden roundtrip")
    word = str(graph.get("surface", ""))
    rows = _golden_dictionary_rows(str(dictionary_path))
    candidates = rows.get(word, [])
    if candidates and tuple(phones) not in candidates:
        raise JAContractError("dictionary_roundtrip_failed", f"selected pronunciation is not an official dictionary variant for {word!r}")
    if reading in _GOLDEN:
        if not candidates:
            raise JAContractError("dictionary_roundtrip_failed", f"audited golden dictionary row is missing for {word!r}")
        return "verified"
    return "verified_dictionary" if candidates else "verified_oov_inventory_only"


def route_language(unit: Mapping[str, Any], *, manual_lexicon: Mapping[str, str] | None = None) -> str:
    surface = str(unit.get("caller_surface", unit.get("surface", "")))
    if manual_lexicon and surface in manual_lexicon:
        return str(manual_lexicon[surface])
    if unit.get("verified_japanese_reading"):
        return "ja"
    if unit.get("clear_english_pronunciation") and surface.isascii() and surface.isalpha() and surface.upper() not in {"AI", "USB"}:
        return "en"
    if unit.get("asr_language") in {"ja", "en"}:
        return str(unit["asr_language"])
    return "unresolved"


def build_alias_rows(analysis: Mapping[str, Any], *, language: str = "ja", inventory: Mapping[str, Any] | Sequence[str] | None = None, dictionary_path: Path | str | None = None, english_metadata_path: Path | str | None = None, english_inventory: Mapping[str, Any] | None = None, english_acoustic_path: Path | str | None = None, english_acoustic_sha256: str | None = None, ordinal_start: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = ordinal_start
    for unit in analysis.get("units", []):
        if unit.get("lexical_status") != "lexical":
            continue
        if language == "ja" and unit.get("language") != "ja":
            continue
        if language == "en" and unit.get("language") != "en":
            continue
        # ASCII occurrences require an explicit route/reading policy.  A
        # Japanese G2P output for ``AI`` or ``game`` is a prediction and is
        # therefore not silently promoted to a Japanese MFA pronunciation.
        caller_surface = str(unit.get("caller_surface", unit.get("surface", "")))
        if language == "ja" and caller_surface.isascii() and caller_surface.isalpha() and not unit.get("locked_reading") and not unit.get("verified_japanese_reading"):
            continue
        phones = list(unit.get("locked_phones", unit.get("phones", [])))
        if language == "ja":
            graph = openjtalk_to_semantic(unit)
            phones = semantic_to_japanese_mfa_v3(graph, inventory, dictionary_path=dictionary_path)["phones"]
        elif language == "en":
            resolved = resolve_english_pronunciation(str(unit.get("surface", "")), phones, manual=bool(unit.get("manual_pronunciation")), metadata_path=english_metadata_path, inventory=english_inventory, acoustic_archive_path=english_acoustic_path, acoustic_archive_sha256=english_acoustic_sha256)
            if resolved["status"] == "unresolved":
                continue
            phones = resolved["pronunciation"]
        if not phones:
            raise JAContractError("dictionary_roundtrip_failed", "alias has no pronunciation")
        alias = make_occurrence_alias(language, ordinal)
        rows.append({
            "alias": alias,
            "language": language,
            "uid": analysis.get("uid"),
            "token_id": unit.get("token_id"),
            "candidate_id": unit.get("candidate_id"),
            "surface": unit.get("surface", ""),
            "reading": unit.get("locked_reading", unit.get("read", "")),
            "pronunciation": phones,
            "native_phones": phones,
            "source_span": unit.get("orig_span"),
            "evidence": unit.get("reading_provenance", "frontend_text_prediction"),
            "provenance": {"stage": "W2", "adapter": SEMANTIC_VERSION},
        })
        ordinal += 1
    return validate_alias_rows(rows)


def generate_occurrence_aliases(analysis: Mapping[str, Any], *, language: str = "ja") -> list[dict[str, Any]]:
    """Descriptive alias for callers that treat aliases as a generation step."""
    return build_alias_rows(analysis, language=language)


def write_locked_alias_artifacts(rows: Sequence[Mapping[str, Any]], dictionary_path: Path | str, alias_map_path: Path | str) -> dict[str, Any]:
    validated = validate_alias_rows(rows)
    dictionary = Path(dictionary_path)
    alias_map = Path(alias_map_path)
    dictionary.parent.mkdir(parents=True, exist_ok=True)
    alias_map.parent.mkdir(parents=True, exist_ok=True)
    dict_text = "".join(f"{row['alias']} {' '.join(row['pronunciation'])}\n" for row in validated)
    map_text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in validated)
    if dictionary.exists() and dictionary.read_text(encoding="utf-8") != dict_text:
        raise JAContractError("dictionary_roundtrip_failed", "locked dictionary artifact is immutable", str(dictionary))
    if alias_map.exists() and alias_map.read_text(encoding="utf-8") != map_text:
        raise JAContractError("dictionary_roundtrip_failed", "alias map artifact is immutable", str(alias_map))
    if not dictionary.exists():
        dictionary.write_text(dict_text, encoding="utf-8")
    if not alias_map.exists():
        alias_map.write_text(map_text, encoding="utf-8")
    return {"aliases": [row["alias"] for row in validated], "dictionary": str(dictionary), "alias_map": str(alias_map), "dictionary_sha256": sha256_file(dictionary), "alias_map_sha256": sha256_file(alias_map)}


def semantic_stage(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    stage_dir.mkdir(parents=True, exist_ok=True)
    frontend_path = stage_dir.parent / "frontend" / "frontend_reconstruction.json"
    receipt_path = stage_dir / "receipt.json"
    if not frontend_path.is_file():
        receipt = make_receipt(stage="semantic", status="BLOCKED", errors=[{"code": "receipt_missing", "message": "frontend_analysis.json is required"}])
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult("semantic", "BLOCKED", str(receipt_path))
    payload = json.loads(frontend_path.read_text(encoding="utf-8"))
    graphs: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    frontend_cfg = config.get("frontend", {}) if isinstance(config.get("frontend"), Mapping) else {}
    mfa_cfg = config.get("mfa", {}) if isinstance(config.get("mfa"), Mapping) else {}
    metadata_path = frontend_cfg.get("japanese_metadata", mfa_cfg.get("japanese_metadata"))
    dictionary_path = frontend_cfg.get("japanese_dictionary", mfa_cfg.get("japanese_dictionary"))
    acoustic_archive = frontend_cfg.get("japanese_acoustic", mfa_cfg.get("japanese_acoustic"))
    acoustic_archive_sha = frontend_cfg.get("japanese_acoustic_sha256", mfa_cfg.get("japanese_acoustic_sha256"))
    records = payload.get("records", []) if isinstance(payload, Mapping) else []
    lexical_units = [(analysis, unit) for analysis in records if isinstance(analysis, Mapping) for unit in analysis.get("units", []) if isinstance(unit, Mapping) and unit.get("lexical_status") == "lexical"]
    has_ja = any(unit.get("language") == "ja" for _, unit in lexical_units)
    has_en = any(unit.get("language") == "en" for _, unit in lexical_units)
    inventory: Mapping[str, Any] | None = None
    english_inventory: Mapping[str, Any] | None = None
    if has_ja:
        if not dictionary_path:
            errors.append({"code": "dictionary_asset_missing", "message": "Japanese dictionary must be configured for Japanese lexical units"})
        else:
            try:
                inventory = load_japanese_mfa_inventory(metadata_path, acoustic_archive_path=acoustic_archive, acoustic_archive_sha256=acoustic_archive_sha)
                if inventory.get("coverage_status") != "archive_bound":
                    errors.append({"code": "mfa_inventory_mismatch", "message": "Japanese metadata must be hash-bound to the configured acoustic archive"})
            except JAContractError as exc:
                errors.append(exc.as_dict())
    if has_en:
        english_metadata = frontend_cfg.get("english_metadata", mfa_cfg.get("english_metadata"))
        english_archive = frontend_cfg.get("english_acoustic", mfa_cfg.get("english_acoustic"))
        english_archive_sha = frontend_cfg.get("english_acoustic_sha256", mfa_cfg.get("english_acoustic_sha256"))
        try:
            english_inventory = load_english_arpa_inventory(english_metadata, acoustic_archive_path=english_archive, acoustic_archive_sha256=english_archive_sha)
            if english_inventory.get("coverage_status") != "archive_bound":
                errors.append({"code": "mfa_inventory_mismatch", "message": "English metadata must be hash-bound to the configured acoustic archive"})
        except JAContractError as exc:
            errors.append(exc.as_dict())
    if errors and ((has_ja and inventory is None) or (has_en and english_inventory is None)):
        receipt = make_receipt(stage="semantic", status="BLOCKED", errors=errors)
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult("semantic", "BLOCKED", str(receipt_path))
    for analysis in records:
        for unit in analysis.get("units", []):
            if unit.get("lexical_status") != "lexical":
                continue
            if unit.get("language") != "ja":
                if unit.get("language") != "en":
                    errors.append({"uid": analysis.get("uid"), "token_id": unit.get("token_id"), "code": "language_unresolved", "message": "lexical occurrence has no verified language route"})
                continue
            caller_surface = str(unit.get("caller_surface", unit.get("surface", "")))
            if caller_surface.isascii() and caller_surface.isalpha() and not unit.get("locked_reading") and not unit.get("verified_japanese_reading"):
                continue
            try:
                graph = openjtalk_to_semantic(unit)
                target = semantic_to_japanese_mfa_v3(graph, inventory, dictionary_path=dictionary_path)
                graph["target"] = target
                graphs.append({"uid": analysis.get("uid"), **graph})
            except JAContractError as exc:
                errors.append({"uid": analysis.get("uid"), "token_id": unit.get("token_id"), **exc.as_dict()})
        try:
            if has_ja:
                aliases.extend(build_alias_rows(analysis, language="ja", inventory=inventory, dictionary_path=dictionary_path, ordinal_start=sum(1 for row in aliases if row.get("language") == "ja")))
            if has_en:
                english_metadata = frontend_cfg.get("english_metadata", mfa_cfg.get("english_metadata"))
                english_archive = frontend_cfg.get("english_acoustic", mfa_cfg.get("english_acoustic"))
                english_archive_sha = frontend_cfg.get("english_acoustic_sha256", mfa_cfg.get("english_acoustic_sha256"))
                aliases.extend(build_alias_rows(analysis, language="en", english_metadata_path=english_metadata, english_inventory=english_inventory, english_acoustic_path=english_archive, english_acoustic_sha256=english_archive_sha, ordinal_start=sum(1 for row in aliases if row.get("language") == "en")))
        except JAContractError as exc:
            errors.append({"uid": analysis.get("uid"), **exc.as_dict()})
    graph_path = stage_dir / "semantic_graph.json"
    graph_payload = {"schema": SEMANTIC_VERSION, "graphs": graphs, "errors": errors}
    if graph_path.exists():
        if json.loads(graph_path.read_text(encoding="utf-8")) != graph_payload:
            raise JAContractError("semantic_relation_ambiguous", "semantic artifact is immutable and differs", str(graph_path))
    else:
        atomic_write_json(graph_path, graph_payload, workspace=stage_dir.parent.parent)
    dictionary = stage_dir / "locked.dict"
    alias_map = stage_dir / "alias_map.jsonl"
    write_locked_alias_artifacts(aliases, dictionary, alias_map)
    expected_aliases = sum(1 for _, unit in lexical_units if unit.get("language") in {"ja", "en"})
    unresolved = sum(1 for _, unit in lexical_units if unit.get("language") not in {"ja", "en"})
    complete = expected_aliases > 0 and len(aliases) == expected_aliases and unresolved == 0 and not errors
    status = "COMPLETE" if complete else ("PARTIAL" if aliases else "BLOCKED")
    receipt = make_receipt(stage="semantic", status=status, outputs=[graph_path, dictionary, alias_map], params={"adapter": SEMANTIC_VERSION}, errors=errors)
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult("semantic", receipt["status"], str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("semantic", semantic_stage, output_namespace="semantic")


__all__ = [
    "DEFAULT_DICTIONARY", "DEFAULT_EN_METADATA", "DEFAULT_METADATA", "SEMANTIC_VERSION", "build_alias_rows", "generate_occurrence_aliases", "load_english_arpa_inventory", "load_japanese_mfa_inventory", "resolve_english_pronunciation",
    "openjtalk_to_semantic", "route_language", "semantic_stage", "semantic_to_japanese_mfa_v3", "split_mora", "verify_dictionary_roundtrip",
    "write_locked_alias_artifacts", "register_stages",
]
