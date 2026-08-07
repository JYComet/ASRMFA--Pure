#!/usr/bin/env python3
"""Fail-closed preparation for the fixed strict English CTC-ready run.

``inspect`` and ``verify-ready`` are read-only.  ``prepare`` requires a fresh
run root and copies only regular files.  ``finalize`` validates the *complete*
rerun output before copying any rerun bundle, so an interrupted/bad rerun can
never produce a partly ready CTC set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_utils import CTC_SUFFIXES, load_ctc_token_entries, validate_ctc_transcript_bundle
from postprocess_textgrids import Interval, TextGrid, Tier, parse_textgrid, write_textgrid

SCHEMA = "hecheng-english-ctc-ready-v3"
TRANSFORM_SCHEMA = "ctc-textgrid-canonical-v1"
TRANSFORMATION_VERSION = "1"
LEGACY_PARSER_SIGNATURE = "legacy-zero-based-no-word-domain-early-item2-v1"
TIMING_TOLERANCE_S = .003
RUN_ROOT = Path("/mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v3_1")
SOURCE = Path("/mnt/Raw/新版合成英文数据")
LEGACY = Path("/mnt/nvme3/mfa_workspace/ctc_pretg")
DICT_SOURCE = PROJECT_ROOT / "dict" / "mfa_ipa.dict"
MISSING_REFS = {
    "024198_杂谈互动_数据里程牌庆祝",
    "036000_弹幕互动_回应吐槽弹幕",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def inside(path: Path, root: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes run root: {path}")
    return resolved


def require_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")


def index_unique(root: Path, suffix: str) -> tuple[dict[str, Path], dict[str, list[str]]]:
    result: dict[str, Path] = {}; duplicate: dict[str, list[str]] = {}
    for current, _dirs, names in os.walk(root):
        for name in names:
            if not name.lower().endswith(suffix):
                continue
            path = Path(current) / name; stem = name[:-len(suffix)]
            if stem in result:
                duplicate.setdefault(stem, [str(result[stem])]).append(str(path))
            else:
                result[stem] = path
    for stem in duplicate:
        result.pop(stem, None)
    return result, duplicate


def ctc_path(root: Path, stem: str, suffix: str) -> Path:
    return root / f"{stem}{suffix}"


def _wav_duration(path: Path) -> float:
    require_regular(path)
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise ValueError("WAV has invalid sample rate")
        return handle.getnframes() / rate


def _valid_intervals(intervals, *, xmin: float, xmax: float, name: str) -> None:
    previous_end = xmin
    for index, interval in enumerate(intervals):
        if not all(math.isfinite(value) for value in (interval.xmin, interval.xmax)) or interval.xmax <= interval.xmin:
            raise ValueError(f"{name}[{index}] invalid interval")
        if interval.xmin < xmin - TIMING_TOLERANCE_S or interval.xmax > xmax + TIMING_TOLERANCE_S:
            raise ValueError(f"{name}[{index}] outside TextGrid domain")
        if interval.xmin + 1e-6 < previous_end:
            raise ValueError(f"{name}[{index}] non-monotonic interval")
        previous_end = interval.xmax


def _standard_textgrid(path: Path, tokens: list[dict], wav_path: Path) -> dict:
    """Validate the only normal TextGrid grammar accepted for direct copying."""
    tg = parse_textgrid(path)
    if not all(math.isfinite(value) for value in (tg.xmin, tg.xmax)) or tg.xmax <= tg.xmin:
        raise ValueError("TextGrid has invalid domain")
    duration = _wav_duration(wav_path)
    if tg.xmin < -TIMING_TOLERANCE_S or tg.xmax > duration + TIMING_TOLERANCE_S:
        raise ValueError("TextGrid domain outside WAV duration")
    if len(tg.tiers) != 2 or [tier.name for tier in tg.tiers] != ["words", "pauses"]:
        raise ValueError("requires exactly two ordered tiers: words, pauses")
    words_tier, pauses_tier = tg.tiers
    if ((words_tier.xmin, words_tier.xmax) != (tg.xmin, tg.xmax)
            or (pauses_tier.xmin, pauses_tier.xmax) != (tg.xmin, tg.xmax)):
        raise ValueError("tier domain differs from TextGrid domain")
    _valid_intervals(words_tier.intervals, xmin=tg.xmin, xmax=tg.xmax, name="words")
    _valid_intervals(pauses_tier.intervals, xmin=tg.xmin, xmax=tg.xmax, name="pauses")
    words = [iv for iv in words_tier.intervals if iv.text.strip()]
    if len(words) != len(tokens):
        raise ValueError("words/tokens count mismatch")
    for index, (word, token) in enumerate(zip(words, tokens)):
        if word.text.strip() != str(token["word"]).strip():
            raise ValueError(f"words[{index}] token text mismatch")
        # CTC token end times are canonical anchors; normal standard TextGrids
        # may legitimately contain a different interval end.
        if abs(word.xmin - float(token["start_s"])) > TIMING_TOLERANCE_S:
            raise ValueError(f"words[{index}] token start mismatch")
    return {"kind": "standard", "textgrid": tg, "tokens": tokens}


def _quoted(line: str, field: str) -> str:
    if not line.startswith(field + " = "):
        raise ValueError(f"expected {field}")
    value = line.split("=", 1)[1].strip()
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise ValueError(f"invalid quoted {field}")
    return value[1:-1].replace('""', '"')


def _number(line: str, field: str) -> float:
    if not line.startswith(field + " = "):
        raise ValueError(f"expected {field}")
    value = float(line.split("=", 1)[1].strip())
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def _parse_exact_known_malformed(path: Path, tokens: list[dict], wav_path: Path) -> dict:
    """Parse *only* the historical zero-based/early-item[2] writer grammar."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    index = 0
    def take(expected: str) -> str:
        nonlocal index
        if index >= len(lines) or lines[index] != expected:
            raise ValueError(f"expected exact malformed grammar: {expected}")
        index += 1; return expected
    def take_prefix(prefix: str) -> str:
        nonlocal index
        if index >= len(lines) or not lines[index].startswith(prefix):
            raise ValueError(f"expected exact malformed grammar: {prefix}")
        value = lines[index]; index += 1; return value
    take('File type = "ooTextFile"'); take('Object class = "TextGrid"')
    xmin = _number(take_prefix("xmin = "), "xmin"); xmax = _number(take_prefix("xmax = "), "xmax")
    take("tiers? <exists>"); take("size = 2"); take("item []:"); take("item [1]:")
    take('class = "IntervalTier"'); _quoted(take_prefix("name = "), "name")
    if _quoted(lines[index - 1], "name") != "words": raise ValueError("first tier is not words")
    # The observed broken writer omitted words-tier xmin/xmax entirely.
    # Presence of either would be a different grammar and must be rerun.
    size_line = take_prefix("intervals: size = ")
    try: word_count = int(size_line.split("=", 1)[1].strip())
    except ValueError as exc: raise ValueError("invalid declared words count") from exc
    if word_count < 0: raise ValueError("negative declared words count")
    take("item [2]:"); take('class = "IntervalTier"')
    words: list[Interval] = []
    for ordinal in range(word_count):
        take(f"intervals [{ordinal}]:")
        words.append(Interval(_number(take_prefix("xmin = "), "xmin"), _number(take_prefix("xmax = "), "xmax"), _quoted(take_prefix("text = "), "text")))
    if _quoted(take_prefix("name = "), "name") != "pauses": raise ValueError("second tier is not pauses")
    pxmin = _number(take_prefix("xmin = "), "xmin"); pxmax = _number(take_prefix("xmax = "), "xmax")
    size_line = take_prefix("intervals: size = ")
    try: pause_count = int(size_line.split("=", 1)[1].strip())
    except ValueError as exc: raise ValueError("invalid declared pauses count") from exc
    if pause_count < 0: raise ValueError("negative declared pauses count")
    pauses: list[Interval] = []
    for ordinal in range(1, pause_count + 1):
        take(f"intervals [{ordinal}]:")
        pauses.append(Interval(_number(take_prefix("xmin = "), "xmin"), _number(take_prefix("xmax = "), "xmax"), _quoted(take_prefix("text = "), "text")))
    if index != len(lines): raise ValueError("unexpected malformed TextGrid tail")
    if (pxmin, pxmax) != (xmin, xmax): raise ValueError("pause tier domain differs from TextGrid domain")
    duration = _wav_duration(wav_path)
    if xmin < -TIMING_TOLERANCE_S or xmax <= xmin or xmax > duration + TIMING_TOLERANCE_S: raise ValueError("TextGrid domain outside WAV duration")
    _valid_intervals(words, xmin=xmin, xmax=xmax, name="words"); _valid_intervals(pauses, xmin=xmin, xmax=xmax, name="pauses")
    if len(words) != len(tokens): raise ValueError("words/tokens count mismatch")
    for ordinal, (word, token) in enumerate(zip(words, tokens)):
        if word.text.strip() != str(token["word"]).strip(): raise ValueError(f"words[{ordinal}] token text mismatch")
        if (abs(word.xmin - float(token["start_s"])) > TIMING_TOLERANCE_S or abs(word.xmax - float(token["end_s"])) > TIMING_TOLERANCE_S): raise ValueError(f"words[{ordinal}] token interval mismatch")
    return {"kind": "canonicalize", "parser_signature": LEGACY_PARSER_SIGNATURE,
            "xmin": xmin, "xmax": xmax, "words": words, "pauses": pauses, "tokens": tokens}


def classify_ctc_bundle(root: Path, stem: str, wav_path: Path) -> tuple[str | None, list[str], dict | None]:
    """Shared transcript validation is always the first fail-closed gate."""
    errors: list[str] = []
    for suffix in CTC_SUFFIXES:
        try:
            require_regular(ctc_path(root, stem, suffix))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        return None, errors, None
    errors.extend(validate_ctc_transcript_bundle(root, stem))
    if errors:
        return None, errors, None
    try:
        tokens = load_ctc_token_entries(ctc_path(root, stem, "_tokens.jsonl"))
        return "standard", [], _standard_textgrid(ctc_path(root, stem, ".TextGrid"), tokens, wav_path)
    except Exception as exc:
        standard_error = str(exc)
    try:
        tokens = load_ctc_token_entries(ctc_path(root, stem, "_tokens.jsonl"))
        return "canonicalize", [], _parse_exact_known_malformed(ctc_path(root, stem, ".TextGrid"), tokens, wav_path)
    except Exception as exc:
        return None, [f"standard: {standard_error}", f"malformed: {exc}"], None


def validate_ctc_bundle(root: Path, stem: str, wav_path: Path) -> list[str]:
    category, errors, _parsed = classify_ctc_bundle(root, stem, wav_path)
    return [] if category else errors


def inspect(source_dir: Path, legacy_ctc: Path) -> dict:
    if not source_dir.is_dir() or source_dir.is_symlink() or not legacy_ctc.is_dir() or legacy_ctc.is_symlink():
        raise ValueError("source-dir and legacy-ctc must be non-symlink directories")
    wavs, wav_dup = index_unique(source_dir, ".wav")
    texts, text_dup = index_unique(source_dir, ".txt")
    if wav_dup or text_dup:
        raise ValueError("duplicate source stems: " + ", ".join(sorted(set(wav_dup) | set(text_dup))[:20]))
    wav_stems, txt_stems = set(wavs), set(texts)
    authoritative = sorted(wav_stems & txt_stems)
    rerun: dict[str, list[str]] = {}; standard: list[str] = []; canonicalize: list[str] = []
    for stem in authoritative:
        category, issues, _parsed = classify_ctc_bundle(legacy_ctc, stem, wavs[stem])
        if not category:
            rerun[stem] = issues
        elif category == "standard":
            standard.append(stem)
        else:
            canonicalize.append(stem)
    payload = {"schema": SCHEMA, "source_dir": str(source_dir.resolve()), "legacy_ctc": str(legacy_ctc.resolve()),
               "wav_count": len(wavs), "txt_count": len(texts), "authoritative_stems": authoritative,
               "wav_paths": {stem: str(path.resolve()) for stem, path in sorted(wavs.items())},
               "txt_paths": {stem: str(path.resolve()) for stem, path in sorted(texts.items())},
               "missing_reference": sorted(wav_stems - txt_stems), "txt_only": sorted(txt_stems - wav_stems),
               "legacy_standard": standard, "legacy_canonicalize": canonicalize,
               # Kept as an explicit compatibility alias for consumers of v2
               # reports; it means direct-copy standard bundles only.
               "legacy_valid": standard, "needs_rerun": sorted(rerun), "needs_rerun_reasons": rerun}
    payload["inventory_sha256"] = stable_hash(payload)
    return payload


def enforce_counts(report: dict, args) -> None:
    if not args.require_expected_counts:
        return
    actual = (report["wav_count"], report["txt_count"], len(report["authoritative_stems"]),
              len(report["missing_reference"]), len(report["txt_only"]), len(report["legacy_standard"]),
              len(report["legacy_canonicalize"]), len(report["needs_rerun"]))
    expected = (args.expected_wavs, args.expected_txts, args.expected_authoritative,
                args.expected_missing_refs, args.expected_txt_only, args.expected_standard,
                args.expected_canonicalize, args.expected_rerun)
    if actual != expected or set(report["missing_reference"]) != set(args.expected_missing_stems):
        raise ValueError(f"unexpected source inventory actual={actual}, expected={expected}, "
                         f"missing={report['missing_reference']}")


def copy_new(src: Path, dst: Path, run_root: Path, *, stem: str, kind: str) -> dict:
    require_regular(src); inside(dst, run_root)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"target exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.parent.is_symlink():
        raise ValueError(f"target parent symlink: {dst.parent}")
    source = {"source_path": str(src.resolve()), "source_size": src.stat().st_size, "source_sha256": sha256(src)}
    shutil.copy2(src, dst); require_regular(dst); inside(dst, run_root)
    result = {"stem": stem, "kind": kind, **source, "destination_path": str(dst.resolve()),
              "destination_size": dst.stat().st_size, "destination_sha256": sha256(dst)}
    if result["source_size"] != result["destination_size"] or result["source_sha256"] != result["destination_sha256"]:
        raise ValueError(f"copy hash mismatch: {src}")
    return result


def _canonical_words(xmin: float, xmax: float, legacy_words: list[Interval], tokens: list[dict]) -> tuple[list[Interval], list[dict]]:
    """Build a continuous words tier and exact token-to-output interval map."""
    words: list[Interval] = []; mapping: list[dict] = []; cursor = xmin
    if len(legacy_words) != len(tokens):
        raise ValueError("canonical words/tokens count mismatch")
    for token_index, (legacy_word, token) in enumerate(zip(legacy_words, tokens)):
        # The legacy TextGrid is the boundary source.  Token timing was only
        # validated for agreement in the exact parser; never replace its end.
        start, end = legacy_word.xmin, legacy_word.xmax
        if start < cursor - TIMING_TOLERANCE_S or end <= start:
            raise ValueError("canonical token intervals are not monotonic")
        if start > cursor + 1e-9:
            words.append(Interval(cursor, start, ""))
        words.append(Interval(start, end, legacy_word.text))
        mapping.append({"token_index": token_index, "canonical_word_ordinal": len(words) - 1,
                        "textgrid_interval_index": len(words)})
        cursor = end
    if cursor < xmax - 1e-9:
        words.append(Interval(cursor, xmax, ""))
    if not words and xmax > xmin:
        words.append(Interval(xmin, xmax, ""))
    return words, mapping


def canonicalize_legacy_textgrid(src: Path, dst: Path, parsed: dict, run_root: Path, *, stem: str, token_path: Path, wav_path: Path) -> dict:
    """Create a fresh, standard TextGrid from the exact accepted old grammar."""
    require_regular(src); require_regular(token_path); require_regular(wav_path); inside(dst, run_root)
    if dst.exists() or dst.is_symlink(): raise FileExistsError(f"target exists: {dst}")
    words, mapping = _canonical_words(parsed["xmin"], parsed["xmax"], parsed["words"], parsed["tokens"])
    target = TextGrid(parsed["xmin"], parsed["xmax"], [
        Tier("words", parsed["xmin"], parsed["xmax"], words),
        Tier("pauses", parsed["xmin"], parsed["xmax"], parsed["pauses"]),
    ])
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.name + ".tmp")
    write_textgrid(target, temporary); os.replace(temporary, dst)
    # Re-read the destination with the common parser before accepting it.
    _standard_textgrid(dst, parsed["tokens"], wav_path)
    def evidence(path: Path) -> dict:
        return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}
    return {"stem": stem, "kind": "legacy_canonical_textgrid", "transform_schema": TRANSFORM_SCHEMA,
            "transformation_version": TRANSFORMATION_VERSION, "parser_signature": parsed["parser_signature"],
            "word_count": len(parsed["words"]), "pause_count": len(parsed["pauses"]),
            "source_textgrid": evidence(src), "source_tokens": evidence(token_path), "source_audio": evidence(wav_path),
            "destination_textgrid": evidence(dst), "token_interval_mapping": mapping}


def verify_evidence_files(items: list[dict], run_root: Path) -> None:
    for item in items:
        source = Path(item["source_path"]); path = Path(item["destination_path"])
        require_regular(source); inside(path, run_root); require_regular(path)
        if (source.stat().st_size != item["source_size"] or sha256(source) != item["source_sha256"]
                or path.stat().st_size != item["destination_size"] or sha256(path) != item["destination_sha256"]
                or item["source_size"] != item["destination_size"]
                or item["source_sha256"] != item["destination_sha256"]):
            raise ValueError(f"prepared evidence tampered: {path}")


def _expected_copy_map(args, report: dict) -> dict[tuple[str, str, str, str], None]:
    """Exact immutable prepare-copy mapping, excluding no optional artifact."""
    root = args.run_root.resolve(); legacy = Path(report["legacy_ctc"])
    result: dict[tuple[str, str, str, str], None] = {}
    def add(kind: str, stem: str, src: Path, dst: Path) -> None:
        result[(kind, stem, str(src.resolve()), str(dst.resolve()))] = None
    for stem in report["legacy_standard"]:
        for suffix in CTC_SUFFIXES:
            add("legacy_standard", stem, ctc_path(legacy, stem, suffix), ctc_path(root / "ctc_ready", stem, suffix))
    for stem in report["legacy_canonicalize"]:
        for suffix in CTC_SUFFIXES:
            if suffix != ".TextGrid":
                add("legacy_canonical_copy", stem, ctc_path(legacy, stem, suffix), ctc_path(root / "ctc_ready", stem, suffix))
    for stem in report["authoritative_stems"]:
        add("audio_view", stem, Path(report["wav_paths"][stem]), root / "audio_view" / f"{stem}.wav")
        add("reference_view", stem, Path(report["txt_paths"][stem]), root / "reference_view" / f"{stem}.txt")
    for stem in report["needs_rerun"]:
        add("rerun_audio", stem, Path(report["wav_paths"][stem]), root / "ctc_rerun_audio" / f"{stem}.wav")
        add("rerun_pinyin", stem, Path(report["txt_paths"][stem]), root / "ctc_rerun_pinyin" / f"{stem}.txt")
    add("run_local_dict", "", args.dictionary_source, root / "dict" / "mfa_ipa.dict")
    return result


def verify_prepared_mapping(items: list[dict], args, report: dict) -> None:
    actual = {(item.get("kind"), item.get("stem"), item.get("source_path"), item.get("destination_path"))
              for item in items if isinstance(item, dict)}
    expected = set(_expected_copy_map(args, report))
    if len(actual) != len(items) or actual != expected:
        raise ValueError("prepared file mapping is missing, extra, duplicate, or misdirected")
    verify_evidence_files(items, args.run_root)


def _expected_transform_map(args, report: dict) -> set[tuple[str, str, str, str, str]]:
    legacy = Path(report["legacy_ctc"]); root = args.run_root.resolve()
    return {("legacy_canonical_textgrid", stem, str(ctc_path(legacy, stem, ".TextGrid").resolve()),
             str(ctc_path(legacy, stem, "_tokens.jsonl").resolve()),
             str((root / "ctc_ready" / f"{stem}.TextGrid").resolve()))
            for stem in report["legacy_canonicalize"]}


def verify_transforms(items: list[dict], args, report: dict) -> None:
    actual = {(item.get("kind"), item.get("stem"), item.get("source_textgrid", {}).get("path"),
               item.get("source_tokens", {}).get("path"), item.get("destination_textgrid", {}).get("path"))
              for item in items if isinstance(item, dict)}
    expected = _expected_transform_map(args, report)
    if len(actual) != len(items) or actual != expected:
        raise ValueError("canonical transform mapping is missing, extra, duplicate, or misdirected")
    for item in items:
        if (item.get("transform_schema") != TRANSFORM_SCHEMA
                or item.get("transformation_version") != TRANSFORMATION_VERSION
                or item.get("parser_signature") != LEGACY_PARSER_SIGNATURE
                or not isinstance(item.get("word_count"), int) or not isinstance(item.get("pause_count"), int)
                or not isinstance(item.get("token_interval_mapping"), list)):
            raise ValueError("canonical transform schema/mapping invalid")
        for key in ("source_textgrid", "source_tokens", "source_audio", "destination_textgrid"):
            entry = item.get(key)
            if not isinstance(entry, dict): raise ValueError(f"canonical transform missing {key}")
            path = Path(entry.get("path", "")); require_regular(path)
            if key == "destination_textgrid": inside(path, args.run_root)
            if path.stat().st_size != entry.get("size") or sha256(path) != entry.get("sha256"):
                raise ValueError(f"canonical transform evidence tampered: {key}")
        stem = item["stem"]; tokens = load_ctc_token_entries(ctc_path(Path(report["legacy_ctc"]), stem, "_tokens.jsonl"))
        source = _parse_exact_known_malformed(Path(item["source_textgrid"]["path"]), tokens, Path(item["source_audio"]["path"]))
        if (item["word_count"] != len(source["words"]) or item["pause_count"] != len(source["pauses"])
                or item["parser_signature"] != source["parser_signature"]):
            raise ValueError("canonical transform parser/count evidence invalid")
        parsed = _standard_textgrid(Path(item["destination_textgrid"]["path"]), tokens, Path(item["source_audio"]["path"]))
        words = parsed["textgrid"].tiers[[tier.name for tier in parsed["textgrid"].tiers].index("words")].intervals
        lexical = [index for index, interval in enumerate(words) if interval.text.strip()]
        expected_mapping = [{"token_index": index, "canonical_word_ordinal": ordinal,
                             "textgrid_interval_index": ordinal + 1} for index, ordinal in enumerate(lexical)]
        if item["token_interval_mapping"] != expected_mapping:
            raise ValueError("canonical transform token interval mapping invalid")
        lexical_intervals = [interval for interval in words if interval.text.strip()]
        if [(interval.xmin, interval.xmax, interval.text) for interval in lexical_intervals] != [
                (interval.xmin, interval.xmax, interval.text) for interval in source["words"]]:
            raise ValueError("canonical transform changed legacy lexical boundaries")


def exact_file_stems(root: Path, suffix: str, run_root: Path) -> set[str]:
    inside(root, run_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid artifact directory: {root}")
    stems: set[str] = set()
    for path in root.glob(f"*{suffix}"):
        require_regular(path); inside(path, run_root); stems.add(path.name[:-len(suffix)])
    return stems


def bundle_stems(root: Path, run_root: Path) -> set[str]:
    found: set[str] = set()
    for suffix in CTC_SUFFIXES:
        found |= exact_file_stems(root, suffix, run_root)
    return found


def render_rerun_command(args) -> list[str]:
    # ctc_prealign has no --stems-file.  Its isolated audio directory is the
    # selector, so one explicit GPU processes exactly the rerun stem set.
    return [args.asr_python, "scripts/ctc_prealign.py", "--data-dir", str(args.run_root / "reference_view"),
            "--audio-dir", str(args.run_root / "ctc_rerun_audio"), "--pinyin-dir",
            str(args.run_root / "ctc_rerun_pinyin"), "--output-dir", str(args.run_root / "ctc_rerun_output"),
            "--model-path", args.asr_model, "--dict-path", str(args.run_root / "dict" / "mfa_ipa.dict"),
            "--device", args.asr_device, "--no-dict-update"]


def prepare(args, report: dict) -> Path:
    enforce_counts(report, args)
    if args.run_root.exists():
        raise FileExistsError(f"fresh run-root required: {args.run_root}")
    args.run_root.mkdir(parents=True); inside(args.run_root, args.run_root)
    wavs, _ = index_unique(args.source_dir, ".wav"); texts, _ = index_unique(args.source_dir, ".txt")
    items: list[dict] = []; transforms: list[dict] = []
    ctc_ready = args.run_root / "ctc_ready"
    for stem in report["legacy_standard"]:
        for suffix in CTC_SUFFIXES:
            items.append(copy_new(ctc_path(args.legacy_ctc, stem, suffix), ctc_path(ctc_ready, stem, suffix),
                                  args.run_root, stem=stem, kind="legacy_standard"))
    for stem in report["legacy_canonicalize"]:
        _category, errors, parsed = classify_ctc_bundle(args.legacy_ctc, stem, wavs[stem])
        if errors or not parsed: raise ValueError(f"canonical source changed: {stem}")
        for suffix in CTC_SUFFIXES:
            if suffix != ".TextGrid":
                items.append(copy_new(ctc_path(args.legacy_ctc, stem, suffix), ctc_path(ctc_ready, stem, suffix),
                                      args.run_root, stem=stem, kind="legacy_canonical_copy"))
        transforms.append(canonicalize_legacy_textgrid(ctc_path(args.legacy_ctc, stem, ".TextGrid"),
                          ctc_path(ctc_ready, stem, ".TextGrid"), parsed, args.run_root, stem=stem,
                          token_path=ctc_path(args.legacy_ctc, stem, "_tokens.jsonl"), wav_path=wavs[stem]))
    for stem in report["authoritative_stems"]:
        items.append(copy_new(wavs[stem], args.run_root / "audio_view" / f"{stem}.wav", args.run_root,
                              stem=stem, kind="audio_view"))
        items.append(copy_new(texts[stem], args.run_root / "reference_view" / f"{stem}.txt", args.run_root,
                              stem=stem, kind="reference_view"))
    for stem in report["needs_rerun"]:
        items.append(copy_new(wavs[stem], args.run_root / "ctc_rerun_audio" / f"{stem}.wav", args.run_root,
                              stem=stem, kind="rerun_audio"))
        items.append(copy_new(texts[stem], args.run_root / "ctc_rerun_pinyin" / f"{stem}.txt", args.run_root,
                              stem=stem, kind="rerun_pinyin"))
    items.append(copy_new(args.dictionary_source, args.run_root / "dict" / "mfa_ipa.dict", args.run_root,
                          stem="", kind="run_local_dict"))
    rerun_stems = args.run_root / "rerun_stems.txt"; inside(rerun_stems, args.run_root)
    rerun_stems.write_text("\n".join(report["needs_rerun"]) + ("\n" if report["needs_rerun"] else ""), encoding="utf-8")
    manifest = {"schema": SCHEMA, "state": "awaiting_rerun", "inventory_sha256": report["inventory_sha256"],
                "authoritative_stems": report["authoritative_stems"], "legacy_standard": report["legacy_standard"],
                "legacy_canonicalize": report["legacy_canonicalize"], "legacy_valid": report["legacy_valid"],
                "needs_rerun": report["needs_rerun"], "missing_reference": report["missing_reference"],
                "txt_only": report["txt_only"], "prepared_files": items,
                "prepared_files_sha256": stable_hash(items), "rerun_stems_path": str(rerun_stems.resolve()),
                "rerun_stems_size": rerun_stems.stat().st_size, "rerun_stems_sha256": sha256(rerun_stems),
                "transforms": transforms, "transforms_sha256": stable_hash(transforms),
                "category_counts": {"standard_copy": len(report["legacy_standard"]),
                                    "canonicalized_legacy": len(report["legacy_canonicalize"]), "rerun": len(report["needs_rerun"])},
                "dictionary_source": {"path": str(args.dictionary_source.resolve()), "size": args.dictionary_source.stat().st_size,
                                      "sha256": sha256(args.dictionary_source)},
                "source_dictionary": {"path": str((args.run_root / "dict" / "mfa_ipa.dict").resolve()),
                                      "size": (args.run_root / "dict" / "mfa_ipa.dict").stat().st_size,
                                      "sha256": sha256(args.run_root / "dict" / "mfa_ipa.dict")},
                "rerun_command": render_rerun_command(args)}
    path = args.run_root / "prepare_manifest.json"; path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_prepare(args, report: dict) -> tuple[dict, Path]:
    path = args.run_root / "prepare_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("missing/corrupt prepare manifest") from exc
    rerun_stems = args.run_root / "rerun_stems.txt"
    expected_text = "\n".join(report["needs_rerun"]) + ("\n" if report["needs_rerun"] else "")
    if (manifest.get("schema") != SCHEMA or manifest.get("state") != "awaiting_rerun"
            or manifest.get("inventory_sha256") != report["inventory_sha256"]
            or manifest.get("prepared_files_sha256") != stable_hash(manifest.get("prepared_files", []))
            or set(manifest.get("authoritative_stems", [])) != set(report["authoritative_stems"])
            or set(manifest.get("legacy_standard", [])) != set(report["legacy_standard"])
            or set(manifest.get("legacy_canonicalize", [])) != set(report["legacy_canonicalize"])
            or set(manifest.get("needs_rerun", [])) != set(report["needs_rerun"])
            or set(manifest.get("missing_reference", [])) != set(report["missing_reference"])
            or set(manifest.get("txt_only", [])) != set(report["txt_only"])
            or manifest.get("rerun_command") != render_rerun_command(args)
            or manifest.get("transforms_sha256") != stable_hash(manifest.get("transforms", []))
            or manifest.get("category_counts") != {"standard_copy": len(report["legacy_standard"]), "canonicalized_legacy": len(report["legacy_canonicalize"]), "rerun": len(report["needs_rerun"])}
            or manifest.get("rerun_stems_path") != str(rerun_stems.resolve())):
        raise ValueError("prepare manifest state binding invalid")
    require_regular(rerun_stems); inside(rerun_stems, args.run_root)
    if (rerun_stems.read_text(encoding="utf-8") != expected_text
            or rerun_stems.stat().st_size != manifest.get("rerun_stems_size")
            or sha256(rerun_stems) != manifest.get("rerun_stems_sha256")):
        raise ValueError("rerun_stems evidence invalid")
    dictionary_source = manifest.get("dictionary_source")
    if (not isinstance(dictionary_source, dict) or dictionary_source.get("path") != str(args.dictionary_source.resolve())
            or dictionary_source.get("size") != args.dictionary_source.stat().st_size
            or dictionary_source.get("sha256") != sha256(args.dictionary_source)):
        raise ValueError("dictionary source evidence invalid")
    source_dictionary = manifest.get("source_dictionary"); run_dict = args.run_root / "dict" / "mfa_ipa.dict"
    if (not isinstance(source_dictionary, dict) or source_dictionary.get("path") != str(run_dict.resolve())
            or source_dictionary.get("size") != run_dict.stat().st_size or source_dictionary.get("sha256") != sha256(run_dict)):
        raise ValueError("run-local dictionary evidence invalid")
    verify_prepared_mapping(manifest["prepared_files"], args, report)
    verify_transforms(manifest.get("transforms", []), args, report)
    return manifest, path


def _verify_pre_finalize_sets(args, report: dict) -> None:
    root = args.run_root; authority, rerun = map(set, (report["authoritative_stems"], report["needs_rerun"]))
    valid = set(report["legacy_standard"]) | set(report["legacy_canonicalize"])
    if exact_file_stems(root / "audio_view", ".wav", root) != authority:
        raise ValueError("audio_view stem set is not exact")
    if exact_file_stems(root / "reference_view", ".txt", root) != authority:
        raise ValueError("reference_view stem set is not exact")
    if exact_file_stems(root / "ctc_rerun_audio", ".wav", root) != rerun:
        raise ValueError("rerun_audio stem set is not exact")
    if exact_file_stems(root / "ctc_rerun_pinyin", ".txt", root) != rerun:
        raise ValueError("rerun_pinyin stem set is not exact")
    if bundle_stems(root / "ctc_ready", root) != valid:
        raise ValueError("ctc_ready legacy stem set is not exact")


def finalize(args, report: dict) -> Path:
    enforce_counts(report, args); manifest, manifest_path = load_prepare(args, report)
    if manifest.get("state") != "awaiting_rerun":
        raise ValueError("prepare manifest is not awaiting_rerun")
    _verify_pre_finalize_sets(args, report)
    rerun_out = args.rerun_ctc or args.run_root / "ctc_rerun_output"; inside(rerun_out, args.run_root)
    rerun = set(report["needs_rerun"])
    # Prevalidate the entire source set before writing any CTC target file.
    if bundle_stems(rerun_out, args.run_root) != rerun:
        raise ValueError("rerun CTC stem set is not exact")
    for stem in sorted(rerun):
        issues = validate_ctc_bundle(rerun_out, stem, args.run_root / "ctc_rerun_audio" / f"{stem}.wav")
        if issues:
            raise ValueError(f"rerun CTC invalid {stem}: {issues[0]}")
    ctc_ready = args.run_root / "ctc_ready"
    rerun_items: list[dict] = []
    for stem in sorted(rerun):
        for suffix in CTC_SUFFIXES:
            rerun_items.append(copy_new(ctc_path(rerun_out, stem, suffix), ctc_path(ctc_ready, stem, suffix),
                                         args.run_root, stem=stem, kind="rerun_ctc"))
    expected = set(report["authoritative_stems"])
    if bundle_stems(ctc_ready, args.run_root) != expected:
        raise ValueError("final ctc_ready stem set is not exact")
    for stem in sorted(expected):
        issues = validate_ctc_bundle(ctc_ready, stem, args.run_root / "audio_view" / f"{stem}.wav")
        if issues:
            raise ValueError(f"final CTC invalid {stem}: {issues[0]}")
    artifacts: dict[str, dict] = {}
    for stem in sorted(expected):
        audio = args.run_root / "audio_view" / f"{stem}.wav"; require_regular(audio)
        reference = args.run_root / "reference_view" / f"{stem}.txt"; require_regular(reference)
        lane = ("standard_copy" if stem in report["legacy_standard"] else "canonicalized_legacy"
                if stem in report["legacy_canonicalize"] else "rerun")
        artifacts[stem] = {"origin_lane": lane,
                           "audio": {"path": str(audio.resolve()), "size": audio.stat().st_size, "sha256": sha256(audio)},
                           "reference": {"path": str(reference.resolve()), "size": reference.stat().st_size, "sha256": sha256(reference)},
                           "ctc": {suffix: {"path": str(ctc_path(ctc_ready, stem, suffix).resolve()),
                                             "size": ctc_path(ctc_ready, stem, suffix).stat().st_size,
                                             "sha256": sha256(ctc_path(ctc_ready, stem, suffix))}
                                   for suffix in CTC_SUFFIXES}}
    evidence = {"schema": SCHEMA, "state": "ready", "prepare_manifest_sha256": sha256(manifest_path),
                "inventory_sha256": report["inventory_sha256"], "stems": sorted(expected),
                "stem_count": len(expected), "missing_reference": report["missing_reference"],
                "roots": {"run": str(args.run_root.resolve()), "ctc_ready": str(ctc_ready.resolve()),
                          "audio_view": str((args.run_root / "audio_view").resolve()),
                          "reference_view": str((args.run_root / "reference_view").resolve())},
                "artifacts": artifacts, "prepared_files_sha256": manifest["prepared_files_sha256"],
                "transforms_sha256": manifest["transforms_sha256"], "category_counts": manifest["category_counts"],
                "source_dictionary": manifest["source_dictionary"],
                "rerun_files": rerun_items, "rerun_files_sha256": stable_hash(rerun_items)}
    path = args.run_root / "ctc_ready_evidence.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"target exists: {path}")
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def verify_ready(args, report: dict) -> None:
    enforce_counts(report, args); manifest, manifest_path = load_prepare(args, report)
    evidence_path = args.run_root / "ctc_ready_evidence.json"
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("missing/corrupt ready evidence") from exc
    expected = set(report["authoritative_stems"])
    if (evidence.get("schema") != SCHEMA or evidence.get("state") != "ready"
            or evidence.get("prepare_manifest_sha256") != sha256(manifest_path)
            or evidence.get("inventory_sha256") != report["inventory_sha256"]
            or set(evidence.get("stems", [])) != expected
            or evidence.get("stem_count") != len(expected)
            or evidence.get("missing_reference") != report["missing_reference"]
            or evidence.get("roots") != {"run": str(args.run_root.resolve()), "ctc_ready": str((args.run_root / "ctc_ready").resolve()), "audio_view": str((args.run_root / "audio_view").resolve()), "reference_view": str((args.run_root / "reference_view").resolve())}
            or evidence.get("prepared_files_sha256") != manifest.get("prepared_files_sha256")
            or evidence.get("transforms_sha256") != manifest.get("transforms_sha256")
            or evidence.get("category_counts") != manifest.get("category_counts")
            or evidence.get("source_dictionary") != manifest.get("source_dictionary")
            or evidence.get("rerun_files_sha256") != stable_hash(evidence.get("rerun_files", []))):
        raise ValueError("ready evidence state binding invalid")
    verify_prepared_mapping(manifest["prepared_files"], args, report)
    rerun = set(report["needs_rerun"])
    if exact_file_stems(args.run_root / "audio_view", ".wav", args.run_root) != expected:
        raise ValueError("ready audio_view stem set is not exact")
    if exact_file_stems(args.run_root / "reference_view", ".txt", args.run_root) != expected:
        raise ValueError("ready reference_view stem set is not exact")
    if exact_file_stems(args.run_root / "ctc_rerun_audio", ".wav", args.run_root) != rerun:
        raise ValueError("ready rerun_audio stem set is not exact")
    if exact_file_stems(args.run_root / "ctc_rerun_pinyin", ".txt", args.run_root) != rerun:
        raise ValueError("ready rerun_pinyin stem set is not exact")
    if bundle_stems(args.run_root / "ctc_ready", args.run_root) != expected:
        raise ValueError("ready ctc stem set is not exact")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("ready artifact stem set is not exact")
    canonical_root = args.run_root.resolve()
    for stem in expected:
        issues = validate_ctc_bundle(args.run_root / "ctc_ready", stem, args.run_root / "audio_view" / f"{stem}.wav")
        if issues:
            raise ValueError(f"ready CTC invalid {stem}: {issues[0]}")
        item = artifacts.get(stem)
        lane = ("standard_copy" if stem in report["legacy_standard"] else "canonicalized_legacy"
                if stem in report["legacy_canonicalize"] else "rerun")
        if not isinstance(item, dict) or set(item) != {"origin_lane", "audio", "reference", "ctc"} or item.get("origin_lane") != lane or not isinstance(item.get("ctc"), dict):
            raise ValueError(f"ready evidence stem missing: {stem}")
        if set(item["ctc"]) != set(CTC_SUFFIXES):
            raise ValueError(f"ready CTC evidence suffixes invalid: {stem}")
        expected_entries = [(item["audio"], canonical_root / "audio_view" / f"{stem}.wav"),
                            (item["reference"], canonical_root / "reference_view" / f"{stem}.txt")]
        expected_entries += [(item["ctc"][suffix], canonical_root / "ctc_ready" / f"{stem}{suffix}")
                             for suffix in CTC_SUFFIXES]
        for entry, expected_path in expected_entries:
            if not isinstance(entry, dict) or entry.get("path") != str(expected_path):
                raise ValueError(f"ready artifact path misdirected: {stem}")
            path = Path(entry["path"]); inside(path, args.run_root); require_regular(path)
            if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"ready artifact tampered: {path}")
    rerun_out = args.rerun_ctc or args.run_root / "ctc_rerun_output"; inside(rerun_out, args.run_root)
    expected_rerun = {("rerun_ctc", stem, str(ctc_path(rerun_out, stem, suffix).resolve()),
                       str(ctc_path(canonical_root / "ctc_ready", stem, suffix).resolve()))
                      for stem in rerun for suffix in CTC_SUFFIXES}
    rerun_items = evidence.get("rerun_files")
    if not isinstance(rerun_items, list):
        raise ValueError("ready rerun evidence missing")
    actual_rerun = {(item.get("kind"), item.get("stem"), item.get("source_path"), item.get("destination_path"))
                    for item in rerun_items if isinstance(item, dict)}
    if len(actual_rerun) != len(rerun_items) or actual_rerun != expected_rerun:
        raise ValueError("ready rerun evidence missing, extra, or misdirected")
    verify_evidence_files(rerun_items, args.run_root)


def _v3_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="inspect", choices=("inspect", "prepare", "finalize", "verify-ready"))
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT); parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--legacy-ctc", type=Path, default=LEGACY); parser.add_argument("--rerun-ctc", type=Path)
    parser.add_argument("--dictionary-source", type=Path, default=DICT_SOURCE)
    parser.add_argument("--require-expected-counts", action="store_true")
    parser.add_argument("--expected-wavs", type=int, default=54000); parser.add_argument("--expected-txts", type=int, default=53998)
    parser.add_argument("--expected-authoritative", type=int, default=53998); parser.add_argument("--expected-missing-refs", type=int, default=2)
    parser.add_argument("--expected-txt-only", type=int, default=0); parser.add_argument("--expected-missing-stems", nargs="*", default=sorted(MISSING_REFS))
    parser.add_argument("--expected-standard", type=int, default=7204)
    parser.add_argument("--expected-canonicalize", type=int, default=46586)
    parser.add_argument("--expected-rerun", type=int, default=208)
    parser.add_argument("--asr-python", default="/home/user/miniconda3/envs/asr/bin/python"); parser.add_argument("--asr-model", default="/mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR")
    parser.add_argument("--asr-device", default="cuda:0")
    args = parser.parse_args()
    try:
        report = inspect(args.source_dir, args.legacy_ctc)
        if args.action == "inspect": enforce_counts(report, args); print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.action == "prepare": print(prepare(args, report))
        elif args.action == "finalize": print(finalize(args, report))
        else: verify_ready(args, report); print("ready evidence verified")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1
    return 0


# v4 production path -------------------------------------------------------
# Kept separate from the preceding v3 diagnostic code so no legacy CTC parser
# or copy path is reachable from the production entry point.
V4_SCHEMA = "hecheng-english-ctc-ready-v4"
V4_SIGNATURE = "ctc-ready-independent-v1"
V4_AXIS = "authoritative_wav"
V4_ACTION = "acoustic_rerun"
V4_REASON = "legacy_audio_provenance_unbound"
V4_MISSING = ["024198_杂谈互动_数据里程牌庆祝", "036000_弹幕互动_回应吐槽弹幕"]
V4_SUFFIXES = tuple(CTC_SUFFIXES) + ("_ref.txt",)
V4_NVV_MODE = "reference_only"
V4_ASR_NVV_BIAS = False
V4_CONTENT_AUTHORITY = "reference"
TOL = .003
DOMAIN_TOL = .000001

def _v4_sorted(values, name):
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be exact sorted unique")

def _v4_index(root, suffix):
    found={}; duplicate=[]
    for base, dirs, names in os.walk(root):
        if any((Path(base) / d).is_symlink() for d in dirs): raise ValueError(f"source symlink directory: {base}")
        for name in names:
            if name.lower().endswith(suffix):
                path=Path(base)/name; stem=name[:-len(suffix)]
                if path.is_symlink(): raise ValueError(f"source symlink: {path}")
                if stem in found: duplicate.append(stem); found.pop(stem, None)
                elif stem not in duplicate: found[stem]=path
    return found, sorted(set(duplicate))

def _v4_wav(path):
    require_regular(path)
    with wave.open(str(path), "rb") as h: frames, rate, channels=h.getnframes(),h.getframerate(),h.getnchannels()
    if min(frames,rate,channels)<=0: raise ValueError(f"invalid WAV: {path}")
    return {"frames":frames,"sample_rate":rate,"channels":channels,"duration_s":frames/rate}

def _v4_evidence(path, wav=False):
    require_regular(path); result={"path":str(path.resolve()),"size":path.stat().st_size,"sha256":sha256(path)}
    if wav: result["wav"]=_v4_wav(path)
    return result

def inspect(source_dir):
    if not source_dir.is_dir() or source_dir.is_symlink(): raise ValueError("source-dir must be a non-symlink directory")
    wavs, wd=_v4_index(source_dir,".wav"); txts, td=_v4_index(source_dir,".txt")
    if wd or td: raise ValueError("duplicate source stems")
    ws,ts=set(wavs),set(txts); stems=sorted(ws&ts); missing=sorted(ws-ts); txt_only=sorted(ts-ws)
    for x,n in ((stems,"stems"),(missing,"missing"),(txt_only,"txt_only")): _v4_sorted(x,n)
    taxonomy=[{"stem":s,"reason":V4_REASON,"action":V4_ACTION} for s in stems]
    r={"schema":V4_SCHEMA,"source_dir":str(source_dir.resolve()),"wav_count":len(wavs),"txt_count":len(txts),"stem_count":len(stems),"authoritative_stems":stems,"missing_reference":missing,"txt_only":txt_only,"wav_paths":{s:str(wavs[s].resolve()) for s in stems},"txt_paths":{s:str(txts[s].resolve()) for s in stems},"final_audio_axis":V4_AXIS,"padding_policy":"forbidden","action_counts":{V4_ACTION:len(stems)},"taxonomy":taxonomy,"taxonomy_sha256":stable_hash(taxonomy)}
    r["inventory_sha256"]=stable_hash(r); return r

def enforce_counts(report,args):
    if not args.require_expected_counts: return
    actual=(report["wav_count"],report["txt_count"],len(report["authoritative_stems"]),len(report["missing_reference"]),len(report["txt_only"]))
    expected=(args.expected_wavs,args.expected_txts,args.expected_authoritative,args.expected_missing_refs,args.expected_txt_only)
    if actual!=expected or report["missing_reference"] != sorted(args.expected_missing_stems): raise ValueError(f"unexpected source inventory {actual}")

def _v4_copy(src,dst,root,stem,kind,wav=False):
    require_regular(src); inside(dst,root)
    if dst.exists() or dst.is_symlink(): raise FileExistsError(f"target exists: {dst}")
    dst.parent.mkdir(parents=True,exist_ok=True); source=_v4_evidence(src,wav); shutil.copy2(src,dst); dest=_v4_evidence(dst,wav)
    if source["size"]!=dest["size"] or source["sha256"]!=dest["sha256"] or (wav and source["wav"]!=dest["wav"]) or os.path.samestat(src.stat(),dst.stat()): raise ValueError("copy mismatch or inode alias")
    return {"kind":kind,"stem":stem,"source":source,"destination":dest}

def _v4_expected_copies(args,r):
    root=args.run_root.resolve(); out=[]
    for s in r["authoritative_stems"]:
        out += [("audio_view",s,r["wav_paths"][s],str((root/"audio_view"/(s+".wav")).resolve())),("reference_view",s,r["txt_paths"][s],str((root/"reference_view"/(s+".txt")).resolve()))]
    return out+[("run_local_dict","",str(args.dictionary_source.resolve()),str((root/"dict"/"mfa_ipa.dict").resolve()))]

def _v4_verify_copies(items,args,r):
    actual=[(x.get("kind"),x.get("stem"),x.get("source",{}).get("path"),x.get("destination",{}).get("path")) for x in items]
    if actual != _v4_expected_copies(args,r): raise ValueError("prepared copy mapping not exact")
    for x in items:
        wav=x["kind"]=="audio_view"
        source=Path(x["source"]["path"]); destination=Path(x["destination"]["path"])
        if _v4_evidence(source,wav)!=x["source"] or _v4_evidence(destination,wav)!=x["destination"] or os.path.samestat(source.stat(),destination.stat()): raise ValueError("copy evidence tampered/alias")

def render_rerun_command(args):
    command = [args.asr_python,"scripts/ctc_prealign.py","--data-dir",str(args.run_root/"reference_view"),"--audio-dir",str(args.run_root/"audio_view"),"--pinyin-dir",str(args.run_root/"reference_view"),"--output-dir",str(args.run_root/"ctc_rerun_output"),"--model-path",args.asr_model,"--dict-path",str(args.run_root/"dict"/"mfa_ipa.dict"),"--all-gpus","--no-dict-update","--require-fresh-output","--no-nvv"]
    if command.count("--no-nvv") != 1:
        raise ValueError("reference-only rerun command must contain --no-nvv exactly once")
    return command

def prepare(args,r):
    enforce_counts(r,args)
    if args.run_root.exists(): raise FileExistsError(f"fresh run-root required: {args.run_root}")
    args.run_root.mkdir(parents=True); inside(args.run_root,args.run_root); items=[]
    for s in r["authoritative_stems"]:
        items.append(_v4_copy(Path(r["wav_paths"][s]),args.run_root/"audio_view"/(s+".wav"),args.run_root,s,"audio_view",True)); items.append(_v4_copy(Path(r["txt_paths"][s]),args.run_root/"reference_view"/(s+".txt"),args.run_root,s,"reference_view"))
    items.append(_v4_copy(args.dictionary_source,args.run_root/"dict"/"mfa_ipa.dict",args.run_root,"","run_local_dict"))
    m={"schema":V4_SCHEMA,"state":"awaiting_acoustic_rerun","inventory_sha256":r["inventory_sha256"],"stem_count":len(r["authoritative_stems"]),"authoritative_stems":r["authoritative_stems"],"missing_reference":r["missing_reference"],"txt_only":r["txt_only"],"final_audio_axis":V4_AXIS,"padding_policy":"forbidden","action_counts":r["action_counts"],"taxonomy":r["taxonomy"],"taxonomy_sha256":r["taxonomy_sha256"],"prepared_files":items,"prepared_files_sha256":stable_hash(items),"source_dictionary":_v4_evidence(args.dictionary_source),"run_local_dictionary":_v4_evidence(args.run_root/"dict"/"mfa_ipa.dict"),"rerun_command":render_rerun_command(args),"nvv_mode":V4_NVV_MODE,"asr_nvv_bias":V4_ASR_NVV_BIAS,"content_authority":V4_CONTENT_AUTHORITY}
    p=args.run_root/"prepare_manifest.json"; temporary=p.with_name(p.name+".tmp"); temporary.write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,p); return p

def _v4_load(args,r):
    p=args.run_root/"prepare_manifest.json"
    try:m=json.loads(p.read_text(encoding="utf-8"))
    except Exception as e: raise ValueError("missing/corrupt prepare manifest") from e
    binds={"schema":V4_SCHEMA,"state":"awaiting_acoustic_rerun","inventory_sha256":r["inventory_sha256"],"stem_count":len(r["authoritative_stems"]),"authoritative_stems":r["authoritative_stems"],"missing_reference":r["missing_reference"],"txt_only":r["txt_only"],"final_audio_axis":V4_AXIS,"padding_policy":"forbidden","action_counts":r["action_counts"],"taxonomy":r["taxonomy"],"taxonomy_sha256":r["taxonomy_sha256"],"rerun_command":render_rerun_command(args),"nvv_mode":V4_NVV_MODE,"asr_nvv_bias":V4_ASR_NVV_BIAS,"content_authority":V4_CONTENT_AUTHORITY}
    expected_keys=set(binds)|{"prepared_files","prepared_files_sha256","source_dictionary","run_local_dictionary"}
    if set(m)!=expected_keys or any(m.get(k)!=v for k,v in binds.items()) or m.get("prepared_files_sha256")!=stable_hash(m.get("prepared_files",[])): raise ValueError("prepare binding invalid")
    source_dict=_v4_evidence(args.dictionary_source); run_dict=_v4_evidence(args.run_root/"dict"/"mfa_ipa.dict")
    if m.get("source_dictionary")!=source_dict or m.get("run_local_dictionary")!=run_dict or (source_dict["size"],source_dict["sha256"])!=(run_dict["size"],run_dict["sha256"]): raise ValueError("dictionary evidence invalid")
    _v4_verify_copies(m["prepared_files"],args,r); return m,p

def _v4_json(path,punct,duration):
    try: data=json.loads(path.read_text(encoding="utf-8-sig")) if punct else [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    except Exception as e: raise ValueError("invalid timing JSON") from e
    if not isinstance(data,list): raise ValueError("timing JSON not list")
    last_start=last_end=-math.inf
    for i,x in enumerate(data):
        try:a,b,am,bm=map(float,(x["start_s"],x["end_s"],x["start_ms"],x["end_ms"]))
        except Exception as e: raise ValueError(f"timing {i} fields") from e
        if not isinstance(x,dict) or (not punct and not str(x.get("word","")).strip()) or not all(math.isfinite(z) for z in (a,b,am,bm)) or b<=a or a<-DOMAIN_TOL or b>duration+DOMAIN_TOL or abs(a*1000-am)>.51 or abs(b*1000-bm)>.51 or (not punct and (a+DOMAIN_TOL<last_start or b+DOMAIN_TOL<last_end)): raise ValueError(f"timing {i} invalid")
        if not punct: last_start,last_end=a,b
    return data

def validate_standard_bundle(root,stem,audio):
    for suf in V4_SUFFIXES: require_regular(root/(stem+suf))
    shared=validate_ctc_transcript_bundle(root,stem)
    if shared: raise ValueError(shared[0])
    meta=_v4_wav(audio); tok=_v4_json(root/(stem+"_tokens.jsonl"),False,meta["duration_s"]); _v4_json(root/(stem+"_punct.json"),True,meta["duration_s"]); tg=parse_textgrid(root/(stem+".TextGrid"))
    if len(tg.tiers)!=2 or [x.name for x in tg.tiers] != ["words","pauses"] or abs(tg.xmin)>DOMAIN_TOL or abs(tg.xmax-meta["duration_s"])>DOMAIN_TOL: raise ValueError("not standard final-axis TextGrid")
    for tier in tg.tiers:
        if abs(tier.xmin-tg.xmin)>DOMAIN_TOL or abs(tier.xmax-tg.xmax)>DOMAIN_TOL: raise ValueError("tier domain")
        last=tg.xmin
        for iv in tier.intervals:
            if not all(math.isfinite(z) for z in (iv.xmin,iv.xmax)) or iv.xmax<=iv.xmin or iv.xmin<-DOMAIN_TOL or iv.xmax>meta["duration_s"]+DOMAIN_TOL or iv.xmin+DOMAIN_TOL<last: raise ValueError("TextGrid interval")
            last=iv.xmax
    words=[x for x in tg.tiers[0].intervals if x.text.strip()]
    if len(words)!=len(tok) or any(x.text.strip()!=str(y["word"]).strip() or abs(x.xmin-float(y["start_s"]))>TOL for x,y in zip(words,tok)): raise ValueError("TextGrid/token mismatch")

def _v4_bundle_stems(root,run_root):
    inside(root,run_root); found=set()
    for suf in V4_SUFFIXES:
        for p in root.glob("*"+suf): require_regular(p); found.add(p.name[:-len(suf)])
    return sorted(found)

def _v4_rerun_namespace(root,run_root,stems):
    inside(root,run_root); allowed={s+suf for s in stems for suf in V4_SUFFIXES}|{"manifest.json","summary.txt",".ctc_normalized"}; actual=[]
    for p in root.iterdir():
        if p.is_symlink() or not p.is_file() or p.name not in allowed: raise ValueError(f"unexpected rerun artifact: {p.name}")
        actual.append(p.name)
    if sorted(actual)!=sorted(allowed): raise ValueError("rerun metadata/files are not exact")

def _v4_rerun_manifest(root,run_root,stems):
    try: entries=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    except Exception as e: raise ValueError("invalid rerun manifest") from e
    if not isinstance(entries,list): raise ValueError("rerun manifest not list")
    seen=[]
    for entry in entries:
        if not isinstance(entry,dict): raise ValueError("rerun manifest entry")
        audio=Path(entry.get("audio","")); stem=audio.stem; seen.append(stem)
        if str(audio.resolve()) != str((run_root/"audio_view"/(stem+".wav")).resolve()) or entry.get("textgrid") != str((root/(stem+".TextGrid")).resolve()) or entry.get("lab") != str((root/(stem+".lab")).resolve()): raise ValueError("rerun manifest paths")
        meta=_v4_wav(audio); tokens=_v4_json(root/(stem+"_tokens.jsonl"),False,meta["duration_s"]); tg=parse_textgrid(root/(stem+".TextGrid")); words=[x for x in tg.tiers[0].intervals if x.text.strip()]
        words_meta=entry.get("_words")
        if not isinstance(words_meta,list) or entry.get("n_words")!=len(tokens) or len(words_meta)!=len(tokens) or not math.isfinite(float(entry.get("duration_s",math.nan))) or abs(float(entry["duration_s"])-meta["duration_s"])>DOMAIN_TOL: raise ValueError("rerun manifest timing/count")
        for w,t,iv in zip(words_meta,tokens,words):
            if not isinstance(w,dict) or str(w.get("word","")).strip()!=str(t["word"]).strip() or abs(float(w.get("start",math.nan))-float(t["start_s"]))>TOL or abs(iv.xmin-float(t["start_s"]))>TOL: raise ValueError("rerun manifest words")
    if seen!=stems or len(seen)!=len(set(seen)): raise ValueError("rerun manifest stem coverage")

def finalize(args,r):
    enforce_counts(r,args);m,mpath=_v4_load(args,r); stems=r["authoritative_stems"]; rerun=args.run_root/"ctc_rerun_output"; inside(rerun,args.run_root); _v4_rerun_namespace(rerun,args.run_root,stems); _v4_rerun_manifest(rerun,args.run_root,stems)
    if _v4_bundle_stems(rerun,args.run_root)!=stems: raise ValueError("rerun namespace not exact")
    for s in stems:
        validate_standard_bundle(rerun,s,args.run_root/"audio_view"/(s+".wav"))
        rerun_ref_path=rerun/(s+"_ref.txt"); reference_path=args.run_root/"reference_view"/(s+".txt"); require_regular(rerun_ref_path)
        rerun_ref=_v4_evidence(rerun_ref_path); reference=_v4_evidence(reference_path)
        if (rerun_ref["size"],rerun_ref["sha256"]) != (reference["size"],reference["sha256"]) or os.path.samestat(rerun_ref_path.stat(),reference_path.stat()): raise ValueError("rerun _ref.txt differs from authoritative reference")
    ready=args.run_root/"ctc_ready"; ready.mkdir(); copied=[]
    for s in stems:
        for suf in V4_SUFFIXES: copied.append(_v4_copy(rerun/(s+suf),ready/(s+suf),args.run_root,s,"rerun_ctc"))
    if _v4_bundle_stems(ready,args.run_root)!=stems: raise ValueError("ready namespace not exact")
    art={}
    for s in stems:
        audio=args.run_root/"audio_view"/(s+".wav"); ref=args.run_root/"reference_view"/(s+".txt"); validate_standard_bundle(ready,s,audio)
        art[s]={"origin_action":V4_ACTION,"audio":_v4_evidence(audio,True),"reference":_v4_evidence(ref),"authoritative_audio":_v4_evidence(Path(r["wav_paths"][s]),True),"authoritative_reference":_v4_evidence(Path(r["txt_paths"][s])),"ctc":{suf:_v4_evidence(ready/(s+suf)) for suf in V4_SUFFIXES}}
    e={"schema":V4_SCHEMA,"state":"ready","independent_verifier_signature":V4_SIGNATURE,"prepare_manifest_sha256":sha256(mpath),"inventory_sha256":r["inventory_sha256"],"stem_count":len(stems),"authoritative_stems":stems,"missing_reference":r["missing_reference"],"txt_only":r["txt_only"],"final_audio_axis":V4_AXIS,"padding_policy":"forbidden","action_counts":r["action_counts"],"taxonomy":r["taxonomy"],"taxonomy_sha256":r["taxonomy_sha256"],"nvv_mode":V4_NVV_MODE,"asr_nvv_bias":V4_ASR_NVV_BIAS,"content_authority":V4_CONTENT_AUTHORITY,"roots":{"run":str(args.run_root.resolve()),"audio_view":str((args.run_root/"audio_view").resolve()),"reference_view":str((args.run_root/"reference_view").resolve()),"ctc_ready":str(ready.resolve())},"source_dictionary":m["source_dictionary"],"run_local_dictionary":m["run_local_dictionary"],"artifacts":art,"rerun_files":copied,"rerun_files_sha256":stable_hash(copied)}
    out=args.run_root/"ctc_ready_evidence.json"; temporary=out.with_name(out.name+".tmp"); temporary.write_text(json.dumps(e,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,out)
    from verify_hecheng_english_ctc_ready_v4 import verify
    verify(args.run_root,args.source_dir,args.dictionary_source,
           args.asr_python,args.asr_model)
    return out

def verify_ready(args,r):
    from verify_hecheng_english_ctc_ready_v4 import verify
    enforce_counts(r,args); verify(args.run_root,args.source_dir,args.dictionary_source,
                                  args.asr_python,args.asr_model)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",nargs="?",default="inspect",choices=("inspect","prepare","finalize","verify-ready")); p.add_argument("--run-root",type=Path,default=Path("/mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0")); p.add_argument("--source-dir",type=Path,default=SOURCE); p.add_argument("--dictionary-source",type=Path,default=DICT_SOURCE); p.add_argument("--require-expected-counts",action="store_true"); p.add_argument("--expected-wavs",type=int,default=54000);p.add_argument("--expected-txts",type=int,default=53998);p.add_argument("--expected-authoritative",type=int,default=53998);p.add_argument("--expected-missing-refs",type=int,default=2);p.add_argument("--expected-txt-only",type=int,default=0);p.add_argument("--expected-missing-stems",nargs="*",default=V4_MISSING);p.add_argument("--asr-python",default="/home/user/miniconda3/envs/asr/bin/python");p.add_argument("--asr-model",default="/mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR");a=p.parse_args()
    try:
        r=inspect(a.source_dir)
        if a.action=="inspect":enforce_counts(r,a);print(json.dumps(r,ensure_ascii=False,indent=2))
        elif a.action=="prepare":print(prepare(a,r))
        elif a.action=="finalize":print(finalize(a,r))
        else:verify_ready(a,r);print("ready evidence verified")
    except Exception as e: print(f"ERROR: {e}",file=sys.stderr);return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())
