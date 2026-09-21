"""Isolated Japanese MFA v3 runner and strict raw TextGrid validation.

The runner writes occurrence aliases into a run-local dictionary so MFA never
chooses among alternate readings.  Japanese and English callers use separate
roots, dictionaries, ledgers and output directories.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


JA_LEDGER_FILENAME = "strict_ja_mfa.json"
EN_LEDGER_FILENAME = "strict_en_mfa.json"
ALIAS_MAP_FILENAME = "alias_map.jsonl"
LOCKED_DICTIONARY_FILENAME = "locked.dict"
RUN_REQUIRED_KEYS = frozenset({"run_id", "language", "unit_ids", "ownership_start_sample", "ownership_end_sample", "context_start_sample", "context_end_sample"})
PHONE_REQUIRED_KEYS = frozenset({"alias", "language", "phone", "native_phone", "raw_interval_id", "start_sample", "end_sample"})

try:
    from .ja_en_schema import (
        JAContractError, StageResult, atomic_write_json, make_receipt,
        validate_alias_rows, validate_occurrence_alias, validate_exact_partition, sha256_file,
    )
except ImportError:  # pragma: no cover
    from ja_en_schema import (
        JAContractError, StageResult, atomic_write_json, make_receipt,
        validate_alias_rows, validate_occurrence_alias, validate_exact_partition, sha256_file,
    )


def _ensure_fresh_directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"fresh MFA directory is a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_locked_alias_dictionary(
    rows: Sequence[Mapping[str, Any]], dictionary_path: Path, alias_map_path: Path,
) -> dict[str, Any]:
    """Write one and only one pronunciation row for each occurrence alias."""
    validated = validate_alias_rows(rows)
    seen: set[str] = set()
    dict_lines: list[str] = []
    map_lines: list[str] = []
    for row in validated:
        alias = row["alias"]
        if alias in seen:
            raise ValueError(f"duplicate alias: {alias}")
        seen.add(alias)
        pronunciation = row["pronunciation"]
        if not all(isinstance(phone, str) and phone and not any(ch.isspace() for ch in phone) for phone in pronunciation):
            raise ValueError(f"invalid pronunciation for {alias}")
        dict_lines.append(f"{alias} {' '.join(pronunciation)}")
        map_lines.append(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    dictionary_path = Path(dictionary_path)
    alias_map_path = Path(alias_map_path)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    alias_map_path.parent.mkdir(parents=True, exist_ok=True)
    dictionary_path.write_text("\n".join(dict_lines) + ("\n" if dict_lines else ""), encoding="utf-8")
    alias_map_path.write_text("\n".join(map_lines) + ("\n" if map_lines else ""), encoding="utf-8")
    return {"aliases": [row["alias"] for row in validated], "dictionary": str(dictionary_path), "alias_map": str(alias_map_path)}


_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ITEM_RE = re.compile(r"^\s*item \[(\d+)\]:\s*$")
_INTERVAL_RE = re.compile(r"^\s*intervals \[(\d+)\]:\s*$")
_KV_RE = re.compile(r"^\s*(xmin|xmax|name|text)\s*=\s*(.*?)\s*$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    return value


def parse_raw_textgrid(
    path: Path,
    *,
    expected_aliases: Sequence[str] | None = None,
    expected_phone_count: int | None = None,
    native_inventory: set[str] | frozenset[str] | None = None,
    language: str = "Japanese",
) -> dict[str, Any]:
    """Parse long TextGrid while retaining tier and raw interval IDs."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tiers: list[dict[str, Any]] = []
    current_tier: dict[str, Any] | None = None
    current_interval: dict[str, Any] | None = None
    tier_index = 0
    top_xmin = top_xmax = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        item_match = _ITEM_RE.match(stripped)
        if item_match:
            if current_interval is not None and current_tier is not None:
                current_tier["intervals"].append(current_interval)
                current_interval = None
            if current_tier is not None:
                tiers.append(current_tier)
            tier_index = int(item_match.group(1))
            current_tier = {"raw_tier_id": tier_index, "name": None, "xmin": None, "xmax": None, "intervals": []}
            continue
        interval_match = _INTERVAL_RE.match(stripped)
        if interval_match and current_tier is not None:
            if current_interval is not None:
                current_tier["intervals"].append(current_interval)
            current_interval = {"raw_interval_id": int(interval_match.group(1)), "xmin": None, "xmax": None, "text": ""}
            continue
        match = _KV_RE.match(stripped)
        if not match:
            continue
        key, value = match.groups()
        if current_interval is not None and key in {"xmin", "xmax", "text"}:
            current_interval[key] = _unquote(value)
        elif current_tier is not None and key in {"name", "xmin", "xmax"}:
            current_tier[key] = _unquote(value) if key == "name" else float(value)
        elif key == "xmin":
            top_xmin = float(value)
        elif key == "xmax":
            top_xmax = float(value)
    if current_interval is not None and current_tier is not None:
        current_tier["intervals"].append(current_interval)
    if current_tier is not None:
        tiers.append(current_tier)
    if not tiers:
        raise ValueError("TextGrid contains no tiers")
    for tier in tiers:
        last = None
        seen_interval_ids: set[int] = set()
        for interval in tier["intervals"]:
            try:
                start = float(interval["xmin"])
                end = float(interval["xmax"])
            except (TypeError, ValueError) as exc:
                raise ValueError("TextGrid interval has invalid bounds") from exc
            if end <= start:
                raise ValueError("TextGrid interval has non-positive duration")
            interval_id = interval["raw_interval_id"]
            if interval_id in seen_interval_ids:
                raise ValueError("TextGrid has duplicate raw interval ID")
            seen_interval_ids.add(interval_id)
            if last is not None and start < last:
                raise ValueError("TextGrid intervals are not monotonic")
            interval["xmin"] = start
            interval["xmax"] = end
            last = end
    by_name = {str(tier.get("name")): tier["intervals"] for tier in tiers}
    words = by_name.get("words", [])
    if expected_aliases is not None and not words:
        raise ValueError("TextGrid cardinality mismatch: words tier is missing")
    if expected_aliases is not None and words:
        expected = list(expected_aliases)
        observed = [interval["text"] for interval in words if interval["text"] and interval["text"] not in {"<eps>", "sil", "sp", "spn"}]
        if observed != expected:
            raise ValueError(f"TextGrid word cardinality/alias mismatch: expected {expected!r}, got {observed!r}")
    phones = by_name.get("phones", [])
    if not phones:
        raise ValueError("TextGrid has no phones tier")
    if expected_phone_count is not None and len(phones) != expected_phone_count:
        raise ValueError(f"TextGrid phone cardinality mismatch: expected {expected_phone_count}, got {len(phones)}")
    if native_inventory is not None:
        validate_native_inventory([str(interval["text"]) for interval in phones if interval["text"]], native_inventory, language=language)
    return {"path": str(path), "xmin": top_xmin, "xmax": top_xmax, "tiers": tiers, "phones": phones, "words": words}


def load_inventory_metadata(path: Path) -> set[str]:
    """Load the model-native phone list from an MFA metadata receipt."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    phones = payload.get("phones") if isinstance(payload, Mapping) else None
    if not isinstance(phones, list) or not phones or not all(isinstance(phone, str) and phone for phone in phones):
        raise ValueError("MFA metadata has no valid phones inventory")
    return set(phones)


def validate_native_inventory(phones: Sequence[str], inventory: set[str] | frozenset[str], *, language: str) -> None:
    unknown = sorted(set(phones) - set(inventory) - {"sil", "sp", "spn"})
    if unknown:
        raise ValueError(f"{language} phone inventory mismatch: {unknown!r}")
    if language not in {"Japanese", "English"}:
        raise ValueError(f"unknown MFA language: {language}")


def build_strict_ledger(
    *, uid: str, language: str, runs: Sequence[Mapping[str, Any]],
    expected_unit_ids: Sequence[str], verified: Sequence[str],
    rejected: Sequence[str], unresolved: Sequence[str],
) -> dict[str, Any]:
    """Build a fail-closed per-language ledger with exact unit partition."""
    if language not in {"ja", "en"}:
        raise ValueError("strict ledger language must be ja or en")
    partition = validate_exact_partition(expected_unit_ids, {"verified": verified, "rejected": rejected, "unresolved": unresolved})
    normalized_runs = [dict(run) for run in runs]
    return {
        "schema": "strict-ja-mfa-v2" if language == "ja" else "strict-en-mfa-v2",
        "uid": uid,
        "language": language,
        "runs": normalized_runs,
        "ledger": partition,
    }


def map_raw_intervals_to_samples(
    intervals: Sequence[Mapping[str, Any]], *, offset_sample: int,
    sample_rate: int, ownership: tuple[int, int], alias: str,
    language: str = "ja",
    run_id: str | None = None,
    raw_artifact_path: str | None = None,
    raw_artifact_sha256: str | None = None,
    tier: str = "phones",
    interval_index: int | None = None,
    unit_id: str | None = None,
) -> list[dict[str, Any]]:
    """Map raw MFA seconds to the immutable global sample axis."""
    if type(offset_sample) is not int or offset_sample < 0 or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("invalid sample transform")
    owner_start, owner_end = ownership
    if language not in {"ja", "en"}:
        raise ValueError("language must be ja or en")
    result: list[dict[str, Any]] = []
    for local_index, interval in enumerate(intervals):
        try:
            start = offset_sample + int(round(float(interval["xmin"]) * sample_rate))
            end = offset_sample + int(round(float(interval["xmax"]) * sample_rate))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid raw MFA interval") from exc
        if end <= start or start < owner_start or end > owner_end:
            raise ValueError("raw MFA interval crosses ownership")
        label = str(interval.get("text", ""))
        result.append({
            "alias": alias,
            "language": language,
            "phone": label,
            "native_phone": label,
            "raw_interval_id": interval.get("raw_interval_id"),
            "raw_interval_index": interval_index if interval_index is not None else local_index,
            "raw_artifact_path": raw_artifact_path,
            "raw_artifact_sha256": raw_artifact_sha256,
            "raw_tier": tier,
            "run_id": run_id,
            "crop_offset_sample": offset_sample,
            "unit_id": unit_id,
            "start_sample": start,
            "end_sample": end,
        })
    return result


def isolated_mfa_command(
    *, corpus_dir: Path, dictionary: Path, acoustic_model: Path, output_dir: Path,
    temporary_directory: Path, runtime_python: Path, num_jobs: int = 1,
    dither: float = 0.0,
) -> tuple[list[str], dict[str, str]]:
    """Build the safe MFA command with an isolated root and no tokenization."""
    for path in (corpus_dir, dictionary, acoustic_model, output_dir, temporary_directory):
        if Path(path).is_symlink():
            raise ValueError(f"MFA path is a symlink: {path}")
    root = Path(temporary_directory) / "mfa_root"
    numba = Path(temporary_directory) / "numba_cache"
    _ensure_fresh_directory(Path(temporary_directory))
    _ensure_fresh_directory(root)
    _ensure_fresh_directory(numba)
    command = [
        str(runtime_python), "-m", "montreal_forced_aligner", "align",
        str(corpus_dir), str(dictionary), str(acoustic_model), str(output_dir),
        "--temporary_directory", str(temporary_directory), "--single_speaker",
        "--no_tokenization", "--no_textgrid_cleanup", "--num_jobs", str(int(num_jobs)),
        "--dither", str(float(dither)), "--overwrite", "--output_format", "long_textgrid",
    ]
    env = os.environ.copy()
    runtime_bin = Path(runtime_python).expanduser().absolute().parent
    # Calling the environment's interpreter directly avoids the activation
    # hook that resets MFA_ROOT_DIR.  MFA's helper binaries (fstcompile,
    # sox, etc.) still need the same env/bin at the front of PATH.
    env.update({
        "MFA_ROOT_DIR": str(root),
        "NUMBA_CACHE_DIR": str(numba),
        "PATH": str(runtime_bin) + os.pathsep + env.get("PATH", ""),
        "CONDA_PREFIX": str(runtime_bin.parent),
    })
    return command, env


def run_isolated_mfa(
    *, corpus_dir: Path, dictionary: Path, acoustic_model: Path, output_dir: Path,
    temporary_directory: Path, runtime_python: Path, timeout: int = 1800,
    num_jobs: int = 1, dither: float = 0.0,
) -> dict[str, Any]:
    command, env = isolated_mfa_command(
        corpus_dir=corpus_dir, dictionary=dictionary, acoustic_model=acoustic_model,
        output_dir=output_dir, temporary_directory=temporary_directory,
        runtime_python=runtime_python, num_jobs=num_jobs, dither=dither,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(temporary_directory) / "mfa.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            status = "COMPLETE" if completed.returncode == 0 else "REJECTED"
            return {"status": status, "return_code": completed.returncode, "command": command, "log_path": str(log_path), "output_dir": str(output_dir), "environment": {"MFA_ROOT_DIR": env["MFA_ROOT_DIR"], "NUMBA_CACHE_DIR": env["NUMBA_CACHE_DIR"]}}
        except subprocess.TimeoutExpired:
            return {"status": "REJECTED", "return_code": "timeout", "command": command, "log_path": str(log_path), "output_dir": str(output_dir), "environment": {"MFA_ROOT_DIR": env["MFA_ROOT_DIR"], "NUMBA_CACHE_DIR": env["NUMBA_CACHE_DIR"]}}


def run_language_mfa(language: str, **kwargs: Any) -> dict[str, Any]:
    """Shared isolated launcher used by both new JA and JAEN English runs."""
    if language not in {"ja", "en", "Japanese", "English"}:
        raise ValueError("language must be ja/en or Japanese/English")
    return run_isolated_mfa(**kwargs)


def run_japanese_mfa(**kwargs: Any) -> dict[str, Any]:
    return run_language_mfa("ja", **kwargs)


def run_english_mfa(**kwargs: Any) -> dict[str, Any]:
    return run_language_mfa("en", **kwargs)


def handle_align(config: Mapping[str, Any], stage_dir: Path) -> StageResult:
    """Execute prepared JA/EN language runs and publish a strict ledger.

    Input contract: ``align.runs`` is a list of run mappings containing
    ``run_id``, ``language``, ``unit_ids``, ``aliases`` and isolated MFA paths
    (``corpus_dir``, ``dictionary``, ``acoustic_model``, ``output_dir``,
    ``temporary_directory``, ``runtime_python``).  ``mfa_runner`` may be
    injected for deterministic fixture tests; production invokes MFA directly.
    """
    receipt_path = stage_dir / "receipt.json"
    raw = config.get("align")
    if not isinstance(raw, Mapping):
        raw = (config.get("stage_inputs") or {}).get("align") if isinstance(config.get("stage_inputs"), Mapping) else None
    if raw is None and (stage_dir / "input.json").is_file():
        raw = json.loads((stage_dir / "input.json").read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("runs"), Sequence):
        receipt = make_receipt(stage="align", status="BLOCKED", params={"implementation": "isolated-ja-en-mfa-v3"}, errors=[{"code": "publish_blocked", "message": "alignment inputs/runtime must be supplied explicitly"}])
        atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
        return StageResult(stage="align", status="BLOCKED", receipt_path=str(receipt_path))
    try:
        mfa_runner = raw.get("mfa_runner") or config.get("mfa_runner")
        ledger_runs: dict[str, list[dict[str, Any]]] = {"ja": [], "en": []}
        ledger_expected: dict[str, list[str]] = {"ja": [], "en": []}
        output_paths: list[Path] = []
        commands: list[str] = []
        run_metadata: list[dict[str, Any]] = []
        for run in raw["runs"]:
            if not isinstance(run, Mapping):
                raise ValueError("alignment run must be a mapping")
            run_id = str(run["run_id"])
            language = str(run.get("language", raw.get("language", "ja")))
            if language not in {"ja", "en"}:
                raise ValueError(f"unsupported run language: {language}")
            run_root = stage_dir / "runs" / run_id
            run_root.mkdir(parents=True, exist_ok=True)
            rows = run.get("aliases", run.get("alias_rows", []))
            if isinstance(rows, (str, Path)):
                rows = [json.loads(line) for line in Path(rows).read_text(encoding="utf-8").splitlines() if line.strip()]
            if not isinstance(rows, Sequence):
                raise ValueError(f"{run_id}: aliases must be a sequence")
            expected_prefix = "ju_" if language == "ja" else "eu_"
            if any(not str(row.get("alias", "")).startswith(expected_prefix) for row in rows):
                raise ValueError(f"{run_id}: alias language namespace mismatch")
            locked = run_root / LOCKED_DICTIONARY_FILENAME
            alias_map = run_root / ALIAS_MAP_FILENAME
            build_locked_alias_dictionary(rows, locked, alias_map)
            if callable(mfa_runner):
                result = mfa_runner(run, run_root, locked)
            else:
                required = ("corpus_dir", "acoustic_model", "output_dir", "temporary_directory", "runtime_python")
                missing = [key for key in required if not run.get(key)]
                if missing:
                    raise ValueError(f"{run_id}: missing MFA inputs {missing}")
                source_corpus = Path(str(run["corpus_dir"]))
                local_corpus = run_root / "corpus"
                local_output = run_root / "output"
                # MFA creates internal symlinks in its corpus database/cache;
                # keep that scratch tree outside the strict stage namespace
                # and bind its path in the receipt instead of treating it as
                # a publishable stage output.
                local_temp = Path(str(run["temporary_directory"])).expanduser().absolute() / f"{run_id}.mfa-work"
                if local_temp.exists() or local_temp.is_symlink():
                    raise ValueError(f"{run_id}: MFA temporary directory is not fresh")
                if source_corpus.is_symlink() or not source_corpus.is_dir():
                    raise ValueError(f"{run_id}: corpus_dir is missing or symlink")
                local_corpus.mkdir(parents=True, exist_ok=False)
                for source in sorted(source_corpus.iterdir()):
                    if source.is_symlink() or not source.is_file():
                        raise ValueError(f"{run_id}: corpus contains invalid artifact {source.name}")
                    shutil.copyfile(source, local_corpus / source.name)
                result = run_isolated_mfa(
                    corpus_dir=local_corpus, dictionary=locked,
                    acoustic_model=Path(str(run["acoustic_model"])), output_dir=local_output,
                    temporary_directory=local_temp, runtime_python=Path(str(run["runtime_python"])),
                    timeout=int(run.get("timeout", raw.get("timeout", 1800))), num_jobs=int(run.get("num_jobs", 1)), dither=0.0,
                )
            if not isinstance(result, Mapping) or result.get("status") not in {"COMPLETE", "success", "VERIFIED"}:
                raise ValueError(f"{run_id}: MFA failed: {dict(result) if isinstance(result, Mapping) else result}")
            if isinstance(result.get("command"), Sequence):
                commands.append(" ".join(str(part) for part in result["command"]))
            run_metadata.append({
                "run_id": run_id, "language": language,
                "acoustic_model": str(run.get("acoustic_model", "")),
                "acoustic_model_sha256": sha256_file(Path(str(run["acoustic_model"]))) if run.get("acoustic_model") and Path(str(run["acoustic_model"])).is_file() else None,
                "dictionary": str(locked), "dictionary_sha256": sha256_file(locked),
                "inventory_path": str(run.get("native_inventory_path", "")),
                "inventory_sha256": sha256_file(Path(str(run["native_inventory_path"]))) if run.get("native_inventory_path") and Path(str(run["native_inventory_path"])).is_file() else None,
                "temporary_directory": str(run.get("temporary_directory", "")),
                "command": list(result.get("command", [])),
                "environment": dict(result.get("environment", {})),
            })
            grid_raw = result.get("textgrid") or run.get("textgrid")
            if not grid_raw:
                output = Path(str(result.get("output_dir", run.get("output_dir", run_root / "output"))))
                candidates = sorted(output.glob("*.TextGrid"))
                if len(candidates) != 1:
                    raise ValueError(f"{run_id}: expected exactly one TextGrid, got {len(candidates)}")
                grid_raw = candidates[0]
            expected_aliases = [str(row["alias"]) for row in rows]
            alias_to_unit = {str(row["alias"]): str(row.get("unit_id", row.get("token_id", row["alias"]))) for row in rows}
            alias_to_pron = {str(row["alias"]): [str(phone) for phone in row["pronunciation"]] for row in rows}
            inventory = run.get("native_inventory")
            if run.get("native_inventory_path"):
                inventory = load_inventory_metadata(Path(str(run["native_inventory_path"])))
            if not callable(mfa_runner) and not inventory:
                raise ValueError(f"{run_id}: native model metadata inventory is required")
            parsed = parse_raw_textgrid(Path(str(grid_raw)), expected_aliases=expected_aliases, expected_phone_count=run.get("expected_phone_count"), native_inventory=inventory, language="Japanese" if language == "ja" else "English")
            offset = int(run.get("offset_sample", 0))
            ownership = (int(run.get("ownership_start_sample", 0)), int(run.get("ownership_end_sample", run.get("total_samples", 2**63 - 1))))
            words = parsed.get("words", [])
            phones: list[dict[str, Any]] = []
            silence_intervals: list[dict[str, Any]] = []
            grid_sha256 = sha256_file(Path(str(grid_raw)))
            for interval_index, interval in enumerate(parsed["phones"]):
                if interval["text"] in {"sil", "sp", "spn", "<eps>"}:
                    silence_intervals.append({**interval, "raw_artifact_path": str(Path(str(grid_raw)).resolve()), "raw_artifact_sha256": grid_sha256, "raw_tier": "phones", "run_id": run_id, "crop_offset_sample": offset})
                    continue
                containing = [word["text"] for word in words if word["xmin"] <= interval["xmin"] and interval["xmax"] <= word["xmax"] and word["text"] and word["text"] not in {"<eps>", "sil", "sp", "spn"}]
                if len(containing) != 1:
                    raise ValueError(f"{run_id}: phone cannot be assigned to one alias")
                mapped = map_raw_intervals_to_samples([interval], offset_sample=offset, sample_rate=int(run.get("sample_rate", 16000)), ownership=ownership, alias=containing[0], language=language, run_id=run_id, raw_artifact_path=str(Path(str(grid_raw)).resolve()), raw_artifact_sha256=grid_sha256, interval_index=interval_index, unit_id=alias_to_unit.get(containing[0]))
                for phone in mapped:
                    phone["phone_id"] = f"{raw.get('uid', config.get('uid', ''))}:{run_id}:{phone['alias']}:p{len(phones):06d}"
                phones.extend(mapped)
            observed_by_alias: dict[str, list[str]] = {}
            for phone in phones:
                observed_by_alias.setdefault(str(phone["alias"]), []).append(str(phone["native_phone"]))
            for alias, pronunciation in alias_to_pron.items():
                if observed_by_alias.get(alias, []) != pronunciation:
                    raise ValueError(f"{run_id}: MFA phone sequence does not equal locked pronunciation for {alias}")
            unit_ids = [str(value) for value in run.get("unit_ids", expected_aliases)]
            missing_run_keys = sorted(RUN_REQUIRED_KEYS - set(run))
            if missing_run_keys:
                raise ValueError(f"{run_id}: missing run contract keys {missing_run_keys}")
            ledger = build_strict_ledger(uid=str(raw.get("uid", config.get("uid", ""))), language=language, runs=[{key: run[key] for key in RUN_REQUIRED_KEYS if key in run}], expected_unit_ids=unit_ids, verified=unit_ids, rejected=[], unresolved=[])
            ledger["runs"][0]["phones"] = phones
            ledger["runs"][0]["silence_intervals"] = silence_intervals
            ledger["runs"][0]["textgrid"] = str(grid_raw)
            ledger_runs[language].extend(ledger["runs"])
            ledger_expected[language].extend(unit_ids)
            output_paths.extend([locked, alias_map])
            # Bind every run-local corpus, raw TextGrid, log and MFA scratch
            # artifact in the receipt so resume cannot silently ignore it.
            for artifact in sorted(run_root.rglob("*")):
                if artifact.is_symlink():
                    raise ValueError(f"{run_id}: run artifact is a symlink: {artifact}")
                if artifact.is_file():
                    output_paths.append(artifact)
        all_ledgers: dict[str, dict[str, Any]] = {}
        for language in ("ja", "en"):
            if not ledger_runs[language]:
                continue
            all_ledgers[language] = build_strict_ledger(uid=str(raw.get("uid", config.get("uid", ""))), language=language, runs=ledger_runs[language], expected_unit_ids=ledger_expected[language], verified=ledger_expected[language], rejected=[], unresolved=[])
        for language, ledger in all_ledgers.items():
            path = stage_dir / (JA_LEDGER_FILENAME if language == "ja" else EN_LEDGER_FILENAME)
            atomic_write_json(path, ledger, workspace=stage_dir.parent.parent)
            output_paths.append(path)
        output_paths = list(dict.fromkeys(output_paths))
        receipt = make_receipt(stage="align", status="COMPLETE", inputs={"languages": sorted(all_ledgers), "run_count": len(raw["runs"]), "runs": run_metadata}, outputs=output_paths, params={"implementation": "isolated-ja-en-mfa-v3", "dither": 0.0}, commands=commands)
    except Exception as exc:
        rejected_outputs: list[Path] = []
        if isinstance(raw, Mapping) and isinstance(raw.get("runs"), Sequence):
            grouped: dict[str, list[dict[str, Any]]] = {"ja": [], "en": []}
            grouped_expected: dict[str, list[str]] = {"ja": [], "en": []}
            for source_run in raw["runs"]:
                if not isinstance(source_run, Mapping):
                    continue
                lang = str(source_run.get("language", raw.get("language", "ja")))
                if lang not in grouped:
                    continue
                units = [str(value) for value in source_run.get("unit_ids", [row.get("alias") for row in source_run.get("aliases", [])])]
                grouped_expected[lang].extend(units)
                rejected_run = {key: source_run.get(key, 0 if key.endswith("_sample") else []) for key in RUN_REQUIRED_KEYS}
                rejected_run.update({"run_id": str(source_run.get("run_id", "unknown")), "language": lang, "unit_ids": units, "error": str(exc)})
                grouped[lang].append(rejected_run)
            for lang in ("ja", "en"):
                if not grouped[lang]:
                    continue
                rejected_ledger = build_strict_ledger(uid=str(raw.get("uid", config.get("uid", ""))), language=lang, runs=grouped[lang], expected_unit_ids=grouped_expected[lang], verified=[], rejected=grouped_expected[lang], unresolved=[])
                rejected_ledger["error"] = str(exc)
                ledger_path = stage_dir / (JA_LEDGER_FILENAME if lang == "ja" else EN_LEDGER_FILENAME)
                atomic_write_json(ledger_path, rejected_ledger, workspace=stage_dir.parent.parent)
                rejected_outputs.append(ledger_path)
        for artifact in sorted((stage_dir / "runs").rglob("*")) if (stage_dir / "runs").exists() else []:
            if artifact.is_file() and not artifact.is_symlink():
                rejected_outputs.append(artifact)
        rejected_outputs = list(dict.fromkeys(rejected_outputs))
        receipt = make_receipt(stage="align", status="REJECTED", outputs=rejected_outputs, params={"implementation": "isolated-ja-en-mfa-v3"}, errors=[{"code": "alignment_invalid", "message": str(exc)}])
    atomic_write_json(receipt_path, receipt, workspace=stage_dir.parent.parent)
    return StageResult(stage="align", status=receipt["status"], receipt_path=str(receipt_path))


def register_stages(registrar: Callable[..., Any]) -> None:
    registrar("align", handle_align, output_namespace="align")


__all__ = [
    "build_locked_alias_dictionary", "build_strict_ledger", "handle_align", "isolated_mfa_command", "load_inventory_metadata", "map_raw_intervals_to_samples", "parse_raw_textgrid",
    "register_stages", "run_english_mfa", "run_isolated_mfa", "run_japanese_mfa", "run_language_mfa", "validate_native_inventory",
    "JA_LEDGER_FILENAME", "EN_LEDGER_FILENAME", "ALIAS_MAP_FILENAME", "LOCKED_DICTIONARY_FILENAME", "RUN_REQUIRED_KEYS", "PHONE_REQUIRED_KEYS",
]
