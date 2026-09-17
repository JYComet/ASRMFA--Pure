"""Read-only inventory and stable namespace for the 0915ALL corpus.

The inventory is deliberately independent of the pipeline.  It hashes source
files while scanning, records the source-relative identity, and never writes
under an input root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
_EXCLUDED_WUWA = {"其它语音 - Others", "带变量语音 - Placeholder"}
_V5_EXACT = {"xuehusang", "xiaoyuan", "wumi", "mieli"}
_V5_PREFIXES = ("gs", "hk", "sr", "ww")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration(path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames / info.samplerate) if info.samplerate else 0.0
    except Exception as exc:
        raise ValueError(f"unreadable or invalid audio: {exc}") from exc


def stable_run_stem(source_id: str, game: str | None, speaker: str,
                    relative_path: str) -> str:
    identity = [source_id, game or "", speaker,
                unicodedata.normalize("NFC", relative_path)]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return "u" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _safe_component(value: str) -> bool:
    return bool(value and value not in {".", ".."}
                and Path(value).name == value and "\x00" not in value)


def _reference(audio: Path, suffix: str) -> tuple[Path | None, str | None]:
    candidate = audio.with_suffix(suffix)
    if candidate.is_symlink():
        raise ValueError(f"reference symlink is not allowed: {candidate}")
    if candidate.is_file() and candidate.stat().st_size:
        try:
            if not candidate.read_text(encoding="utf-8").strip():
                return None, None
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"unreadable reference text: {exc}") from exc
        return candidate, _sha256(candidate)
    return None, None


@dataclass(frozen=True)
class InventoryItem:
    source_id: str
    game: str | None
    speaker: str
    source_path: Path
    source_relative_path: str
    original_stem: str
    run_stem: str
    audio_sha256: str
    audio_bytes: int
    duration_seconds: float
    reference_path: Path | None
    reference_sha256: str | None
    reference_suffix: str | None
    text_mode: Literal["reference", "fallback"]
    needs_gamesl_padding: bool


@dataclass(frozen=True)
class InventoryExcluded:
    source_id: str
    source_path: Path
    source_relative_path: str
    original_stem: str
    reason: str


@dataclass(frozen=True)
class InventoryInvalid:
    source_id: str
    source_path: Path
    source_relative_path: str
    original_stem: str
    reason: str


@dataclass(frozen=True)
class InventorySnapshot:
    items: tuple[InventoryItem, ...]
    excluded: tuple[InventoryExcluded, ...]
    invalid: tuple[InventoryInvalid, ...] = ()
    schema: str = "qwen3-0915all-inventory-v1"


def _files(root: Path, prune_names=()):
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"source root is not a real directory: {root}")
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs.sort(); files.sort()
        current = Path(directory)
        dirs[:] = [name for name in dirs if name not in set(prune_names)]
        for name in list(dirs):
            path = current / name
            if path.is_symlink():
                raise ValueError(f"source symlink is not allowed: {path}")
        for name in files:
            path = current / name
            if path.is_symlink():
                raise ValueError(f"source symlink is not allowed: {path}")
            if path.suffix.lower() in _AUDIO_SUFFIXES:
                yield path


def _make_item(source_id: str, game: str | None, speaker: str,
               root: Path, audio: Path, ref_suffix: str,
               needs_padding: bool) -> InventoryItem:
    if not _safe_component(speaker):
        raise ValueError(f"unsafe speaker: {speaker!r}")
    rel = audio.relative_to(root).as_posix()
    ref, ref_hash = _reference(audio, ref_suffix)
    return InventoryItem(
        source_id=source_id, game=game, speaker=speaker,
        source_path=audio, source_relative_path=rel,
        original_stem=audio.stem, run_stem=stable_run_stem(source_id, game, speaker, rel),
        audio_sha256=_sha256(audio), audio_bytes=audio.stat().st_size,
        duration_seconds=_duration(audio), reference_path=ref,
        reference_sha256=ref_hash, reference_suffix=ref_suffix if ref else None,
        text_mode="reference" if ref else "fallback",
        needs_gamesl_padding=needs_padding,
    )


def _source_root(config: dict, key: str) -> Path:
    roots = config.get("sources", config)
    value = roots.get(key)
    if value is None:
        raise ValueError(f"missing source root: {key}")
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError(f"source root must be absolute: {key}={root}")
    return root


def _scan_gamedata(root: Path):
    for audio in _files(root):
        rel = audio.relative_to(root).parts
        game = rel[0] if rel else "_default"
        speaker = rel[1] if len(rel) > 2 else "_default"
        yield _make_item("gamedata", game, speaker, root, audio, ".txt", True), None


def _scan_wuwa(root: Path):
    for audio in _files(root):
        rel = audio.relative_to(root).parts
        excluded = next((part for part in rel if part in _EXCLUDED_WUWA), None)
        if excluded:
            yield None, InventoryExcluded("wuwa", audio, audio.relative_to(root).as_posix(),
                                          audio.stem, "excluded_category")
            continue
        speaker = rel[0] if rel else "_default"
        yield _make_item("wuwa", "鸣潮", speaker, root, audio, ".lab", True), None


def _v5_allowed(name: str, selection: dict | None = None) -> bool:
    selection = selection or {}
    exact = {str(value).lower() for value in selection.get("v5_0707_exact", _V5_EXACT)}
    prefixes = tuple(str(value).lower() for value in selection.get("v5_0707_prefixes", _V5_PREFIXES))
    keep = set(selection.get("v5_0707_keep", ["合成ria"]))
    lower = name.lower()
    return (name in keep or
            (not _CJK.search(name) and
             (lower in exact or lower.startswith(prefixes) or bool(name))))


def _scan_v5(root: Path):
    for audio in _files(root):
        rel = audio.relative_to(root).parts
        speaker = rel[0] if rel else "_default"
        if not _v5_allowed(speaker):
            yield None, InventoryExcluded("v5_0707", audio, audio.relative_to(root).as_posix(),
                                          audio.stem, "excluded_speaker")
            continue
        yield _make_item("v5_0707", None, speaker, root, audio, ".txt", False), None


def scan_sources(config: dict) -> InventorySnapshot:
    items: list[InventoryItem] = []
    excluded: list[InventoryExcluded] = []
    invalid: list[InventoryInvalid] = []
    sample_limit = int(config.get("_sample_only", 0) or 0)
    selection = config.get("selection", {}) or {}
    wuwa_excluded = set(selection.get("wuwa_excluded", _EXCLUDED_WUWA))
    def inspect(candidate):
        key, game, speaker, root, audio, suffix, padded = candidate
        try:
            return _make_item(key, game, speaker, root, audio, suffix, padded), None
        except (OSError, ValueError) as exc:
            return None, InventoryInvalid(key, audio, audio.relative_to(root).as_posix(),
                                          audio.stem, str(exc))

    workers = max(1, int(config.get("inventory_workers", min(16, os.cpu_count() or 1))))
    batch_size = max(workers, int(config.get("inventory_batch_size", 4096)))
    candidates = []
    # Alternate deterministic directory traversal with bounded parallel NAS
    # reads. This avoids retaining/submitting the entire corpus at once.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inventory") as pool:
        def flush():
            for item, bad in pool.map(inspect, candidates):
                if item is not None:
                    items.append(item)
                if bad is not None:
                    invalid.append(bad)
            candidates.clear()

        for _scanner, key in ((_scan_gamedata, "gamedata"),
                              (_scan_v5, "v5_0707"), (_scan_wuwa, "wuwa")):
            source_root = _source_root(config, key)
            included_seen = 0
            for audio in _files(source_root):
                rel = audio.relative_to(source_root).parts
                if key == "gamedata":
                    game = rel[0] if rel else "_default"; speaker = rel[1] if len(rel) > 2 else "_default"; suffix = ".txt"; padded = True
                elif key == "wuwa":
                    excluded_name = next((part for part in rel if part in wuwa_excluded), None)
                    if excluded_name:
                        excluded.append(InventoryExcluded("wuwa", audio, audio.relative_to(source_root).as_posix(), audio.stem, "excluded_category")); continue
                    game = "鸣潮"; speaker = rel[0] if rel else "_default"; suffix = ".lab"; padded = True
                else:
                    speaker = rel[0] if rel else "_default"
                    if not _v5_allowed(speaker, selection):
                        excluded.append(InventoryExcluded("v5_0707", audio, audio.relative_to(source_root).as_posix(), audio.stem, "excluded_speaker")); continue
                    game = None; suffix = ".txt"; padded = False
                candidates.append((key, game, speaker, source_root, audio, suffix, padded))
                included_seen += 1
                if len(candidates) >= batch_size:
                    flush()
                if sample_limit and included_seen >= sample_limit:
                    break
        if candidates:
            flush()
    seen: dict[str, str] = {}
    for item in items:
        identity = json.dumps([item.source_id, item.game or "", item.speaker,
                               unicodedata.normalize("NFC", item.source_relative_path)],
                              ensure_ascii=False, separators=(",", ":"))
        prior = seen.setdefault(item.run_stem, identity)
        if prior != identity:
            raise ValueError(f"stable run stem collision: {item.run_stem}")
    return InventorySnapshot(tuple(sorted(items, key=lambda x: x.run_stem)),
                             tuple(sorted(excluded, key=lambda x: (x.source_id, x.source_relative_path))),
                             tuple(sorted(invalid, key=lambda x: (x.source_id, x.source_relative_path))))


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


def write_frozen_inventory(snapshot: InventorySnapshot, path: Path) -> dict:
    payload = {
        "schema": snapshot.schema,
        "items": [_json_value(asdict(item)) for item in snapshot.items],
        "excluded": [_json_value(asdict(item)) for item in snapshot.excluded],
        "invalid": [_json_value(asdict(item)) for item in snapshot.invalid],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"frozen inventory already exists with different bytes: {path}")
        return payload
    temporary.write_bytes(content)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    try:
        fd = os.open(path.parent, os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-only", type=int, default=0)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.sample_only:
        config = dict(config)
        config["_sample_only"] = args.sample_only
    snapshot = scan_sources(config)
    rows = snapshot.items
    print(json.dumps({"schema": snapshot.schema, "items": [_json_value(asdict(x)) for x in rows],
                      "excluded_count": len(snapshot.excluded)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
