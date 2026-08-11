#!/usr/bin/env python3
"""Prepare and execute the isolated, non-publishing GPU-1000 MFA run.

This deliberately is a small wrapper around ``run_pipeline.py``.  It is not a
second pipeline and it never makes a source tree writable, publishes output,
or retries a run.  The real ``run`` command is intentionally a separately
authorised operation; the other commands are useful with temporary fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - reported by command
    yaml = None

PROJECT = Path(__file__).resolve().parent.parent
SPEAKERS = ("ria", "花礼", "雪狐桑")
QUOTAS = {"ria": 334, "花礼": 333, "雪狐桑": 333}
SEED = "gpu1000-v1"
SELECTOR = "gpu1000-balanced-hash-v1"
PRELIMINARY_SELECTION_DIGEST = "2114f03a1d2ff4b85660c06b76f657c8c1a3414709c9ccc3754fe2c223139335"
ENV_ALLOWLIST = ("PATH", "HOME", "CUDA_VISIBLE_DEVICES", "CONDA_PREFIX",
                 "LD_LIBRARY_PATH", "PYTHONPATH", "LANG", "LC_ALL")
TELEMETRY_INTERVAL_SECONDS = 2


class SafetyError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_tree(path: Path) -> str:
    if path.is_file():
        return sha_file(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(child.relative_to(path)).encode("utf-8")); h.update(b"\0")
        h.update(sha_file(child).encode("ascii")); h.update(b"\n")
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_once(path: Path, value: Any, *, text: bool = False) -> None:
    if path.exists():
        raise SafetyError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if text else json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o444)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def require_new_root(root: Path) -> None:
    if root.exists():
        raise SafetyError(f"prepare target must not exist: {root}")
    if root.is_symlink():
        raise SafetyError(f"prepare target may not be a symlink: {root}")


def source_samples(source: Path, quotas: dict[str, int], explicit_stems: list[str] | None = None) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Freeze eligible sources and record, rather than hide, bad source entries.

    A lone WAV must not make a 54k cache unusable: it is an explicit source
    exclusion.  A duplicate *eligible* global identity is different -- there
    is no authoritative choice, so preparation fails before it creates a root.
    """
    candidates: dict[str, list[dict[str, str]]] = {}
    exclusions: list[dict[str, str]] = []
    for speaker in SPEAKERS:
        folder = source / speaker
        if not folder.is_dir():
            raise SafetyError(f"missing speaker source directory: {folder}")
        rows: list[dict[str, str]] = []
        wavs = sorted(folder.glob("*.wav"), key=lambda p: (nfc(p.name), str(p)))
        paired_txt: set[Path] = set()
        for wav in wavs:
            stem = nfc(wav.stem)
            if wav.stem != stem or wav.name != nfc(wav.name):
                exclusions.append({"speaker": speaker, "reason": "non_nfc_wav_name", "path": str(wav)})
                continue
            if not wav.is_file() or wav.is_symlink():
                exclusions.append({"speaker": speaker, "reason": "nonordinary_wav", "path": str(wav)})
                continue
            txt = wav.with_suffix(".txt")
            paired_txt.add(txt)
            if txt.is_symlink() or (txt.exists() and not txt.is_file()):
                exclusions.append({"speaker": speaker, "reason": "nonordinary_sibling_txt", "path": str(txt)})
                continue
            if not txt.exists():
                exclusions.append({"speaker": speaker, "reason": "missing_sibling_txt", "path": str(wav)})
                continue
            if txt.stem != stem or txt.name != nfc(txt.name):
                exclusions.append({"speaker": speaker, "reason": "non_nfc_or_mismatched_sibling_txt", "path": str(txt)})
                continue
            rows.append({"speaker": speaker, "stem": stem, "wav": str(wav), "txt": str(txt),
                         "rank": hashlib.sha256(f"{SEED}\0{speaker}\0{stem}".encode("utf-8")).hexdigest()})
        for txt in sorted(folder.glob("*.txt"), key=lambda p: (nfc(p.name), str(p))):
            if txt not in paired_txt:
                exclusions.append({"speaker": speaker, "reason": "orphan_txt", "path": str(txt)})
        candidates[speaker] = rows
    by_stem: dict[str, list[dict[str, str]]] = {}
    for rows in candidates.values():
        for row in rows: by_stem.setdefault(row["stem"], []).append(row)
    ambiguous = {stem for stem, rows in by_stem.items() if len(rows) > 1}
    for stem in sorted(ambiguous):
        for row in sorted(by_stem[stem], key=lambda r: (r["speaker"], r["wav"])):
            exclusions.append({"speaker": row["speaker"], "reason": "duplicate_global_eligible_stem",
                               "path": row["wav"], "stem": stem})
    if ambiguous:
        raise SafetyError("duplicate global eligible stem authority ambiguity: " + ", ".join(sorted(ambiguous)))
    result: dict[str, list[dict[str, str]]] = {}
    for speaker in SPEAKERS:
        rows = candidates[speaker]
        if explicit_stems is None:
            if len(rows) < quotas[speaker]:
                raise SafetyError(f"{speaker} has {len(rows)} eligible samples; need {quotas[speaker]}")
            result[speaker] = sorted(rows, key=lambda r: (r["rank"], r["stem"]))[:quotas[speaker]]
    if explicit_stems is not None:
        if len(explicit_stems) != len(set(explicit_stems)):
            raise SafetyError("explicit selected-stems list contains duplicates")
        available = {row["stem"]: row for rows in candidates.values() for row in rows}
        missing = sorted(set(explicit_stems) - set(available))
        if missing: raise SafetyError("explicit selected stem is not eligible: " + ", ".join(missing[:10]))
        result = {speaker: [] for speaker in SPEAKERS}
        for stem in explicit_stems: result[available[stem]["speaker"]].append(available[stem])
    return result, sorted(exclusions, key=lambda r: (r["speaker"], r["reason"], r["path"]))


def balanced_quotas(count: int) -> dict[str, int]:
    base, remainder = divmod(count, len(SPEAKERS))
    return {speaker: base + (1 if index < remainder else 0) for index, speaker in enumerate(SPEAKERS)}


def shard_sizes(count: int) -> list[int]:
    base, remainder = divmod(count, 8)
    return [base + (1 if gpu < remainder else 0) for gpu in range(8)]


def load_explicit_stems(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
        stems = value.get("stems") if isinstance(value, dict) else value
    except json.JSONDecodeError:
        stems = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(stems, list) or not all(isinstance(stem, str) and stem for stem in stems):
        raise SafetyError("selected-stems must be a JSON list/{stems:list} or nonempty newline list")
    return [nfc(stem) for stem in stems]


def git_evidence() -> dict[str, Any]:
    def out(*argv: str) -> str:
        try:
            return subprocess.run(argv, cwd=PROJECT, check=True, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"
    return {"head": out("git", "rev-parse", "HEAD").strip(),
            "status_porcelain": out("git", "status", "--porcelain"),
            "diff": out("git", "diff", "--binary"),
            "untracked": out("git", "ls-files", "--others", "--exclude-standard")}


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SafetyError("PyYAML is required")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SafetyError(f"config is not a mapping: {path}")
    return data


def materialize_config(template: Path, root: Path) -> dict[str, Any]:
    cfg = load_yaml(template)
    cfg["mode"] = "nvrasr_fallback"
    cfg["data_dir"] = str(root / "input")
    cfg["workspace"] = str(root / "workspace")
    cfg["output_dir"] = str(root / "configured_output_never_publish")
    cfg["output_staging"] = False
    cfg["use_cache"] = False
    ctc = cfg.setdefault("ctc_prealign", {})
    ctc.update({"enabled": True, "all_gpus": True, "limit": 0})
    if "/mnt/Raw" in json.dumps(cfg, ensure_ascii=False):
        raise SafetyError("resolved config contains forbidden /mnt/Raw")
    return cfg


def prepare(args: argparse.Namespace) -> int:
    root, source, template = args.root.resolve(), args.source.resolve(), args.config.resolve()
    require_new_root(root)
    explicit_stems = load_explicit_stems(args.selected_stems) if getattr(args, "selected_stems", None) else None
    count = args.count if getattr(args, "count", None) is not None else (len(explicit_stems) if explicit_stems is not None else 1000)
    if count != 1000 and not 8 <= count <= 64:
        raise SafetyError("confirmation count must be 8..64; full run count is exactly 1000")
    if explicit_stems is not None and len(explicit_stems) != count:
        raise SafetyError("--count does not match explicit selected-stems length")
    quotas = balanced_quotas(count) if explicit_stems is None else {speaker: 0 for speaker in SPEAKERS}
    picked, exclusions = source_samples(source, quotas, explicit_stems)
    selected = [row for speaker in SPEAKERS for row in picked[speaker]]
    # A sorted identity order is the contract used by sharding and analysis.
    selected.sort(key=lambda r: (r["speaker"], r["stem"]))
    if len(selected) != count or len({row["stem"] for row in selected}) != count:
        raise SafetyError("selected identity count is not globally unique")
    names = [f"{row['stem']}{suffix}" for row in selected for suffix in (".wav", ".txt")]
    if len(names) != len(set(names)) or any(name != nfc(name) for name in names):
        raise SafetyError("selected destination filename collision or NFC collision")
    root.mkdir(parents=True)
    copied: list[dict[str, str]] = []
    for row in selected:
        destination = root / "input"; destination.mkdir(parents=True, exist_ok=True)
        source_relative = str(Path(row["wav"]).relative_to(source))
        copied_row = {"speaker": row["speaker"], "stem": row["stem"], "source_relative_wav": source_relative,
                      "source_relative_txt": str(Path(row["txt"]).relative_to(source))}
        for kind in ("wav", "txt"):
            src, dst = Path(row[kind]), destination / Path(row[kind]).name
            shutil.copy2(src, dst)  # ordinary copy: links are rejected above
            if dst.is_symlink() or sha_file(src) != sha_file(dst):
                raise SafetyError(f"copy integrity failure: {src} -> {dst}")
            copied_row[f"{kind}_destination"] = str(dst)
            copied_row[f"{kind}_sha256"] = sha_file(dst)
        copied.append(copied_row)
    cfg = materialize_config(template, root)
    config_path = root / "resolved_gpu1000_nvrasr_fallback.yaml"
    if yaml is None: raise SafetyError("PyYAML is required")
    write_once(config_path, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=True), text=True)
    identity = [{key: row[key] for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                for row in copied]
    manifest = {"schema": "gpu1000-selection-v1", "selector": SELECTOR, "seed": SEED,
                "preliminary_selection_digest": PRELIMINARY_SELECTION_DIGEST,
                "source": str(source), "quotas": {speaker: len(picked[speaker]) for speaker in SPEAKERS},
                "count": len(copied), "run_label": "full1000" if count == 1000 else "confirmation_nonpublish",
                "samples": copied, "selection_digest": digest(identity)}
    write_once(root / "selected_manifest.json", manifest)
    write_once(root / "source_inventory.json", {"schema": "gpu1000-source-inventory-v1", "source": str(source),
                                                  "eligible_counts": {speaker: len(picked[speaker]) for speaker in SPEAKERS},
                                                  "exclusions": exclusions, "exclusions_digest": digest(exclusions)})
    write_once(root / "selected_stems.json", {"stems": [r["stem"] for r in copied], "digest": digest(identity)})
    # ctc_prealign discovers the flat directory by sorted WAV pathname.  The
    # selection contract may retain speaker grouping, but the shard plan must
    # mirror that consumer order exactly or its all-GPU receipt cannot bind.
    ctc_order = sorted(copied, key=lambda row: Path(row["wav_destination"]).name)
    offset = 0; shards = []
    for gpu, size in enumerate(shard_sizes(count)):
        shards.append({"gpu": gpu, "stems": [r["stem"] for r in ctc_order[offset:offset + size]]}); offset += size
    write_once(root / "shard_plan.json", {"schema": "gpu1000-shard-plan-v1", "shards": shards,
                                           "digest": digest(shards)})
    evidence = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "git": git_evidence(),
                "config_template": str(template), "config_template_sha256": sha_file(template),
                "resolved_config_sha256": sha_file(config_path),
                "run_pipeline_sha256": sha_file(PROJECT / "scripts/run_pipeline.py"),
                "ctc_prealign_sha256": sha_file(PROJECT / "scripts/ctc_prealign.py"),
                "model_and_dictionary": provenance_paths(cfg),
                "source_exclusions": exclusions, "source_exclusions_digest": digest(exclusions)}
    write_once(root / "prepare_evidence.json", evidence)
    print(json.dumps({"root": str(root), "count": len(copied), "selection_digest": manifest["selection_digest"]}))
    return 0


def provenance_paths(cfg: dict[str, Any]) -> list[dict[str, str]]:
    raw = [cfg.get("models_dir"), cfg.get("mfa_dict"), cfg.get("pinyin_dict"),
           cfg.get("ctc_prealign", {}).get("model_path"), cfg.get("mfa_en", {}).get("dictionary"),
           cfg.get("mfa_en", {}).get("acoustic_model"), cfg.get("mfa_en", {}).get("g2p_model")]
    result = []
    for value in raw:
        if value:
            p = Path(str(value))
            result.append({"path": str(p), "exists": str(p.exists()),
                           "sha256": sha_tree(p) if p.exists() else ""})
    return result


def state(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name in ("selected_manifest.json", "prepare_evidence.json", "source_inventory.json", "shard_plan.json",
                 "resolved_gpu1000_nvrasr_fallback.yaml"):
        if not (root / name).is_file(): raise SafetyError(f"not an exclusive prepared root; missing {name}")
    return read_json(root / "selected_manifest.json"), read_json(root / "prepare_evidence.json"), read_json(root / "shard_plan.json")


def validate_prepared(root: Path) -> list[str]:
    manifest, evidence, shards = state(root)
    errors: list[str] = []
    samples = manifest.get("samples", [])
    count = manifest.get("count")
    if not isinstance(count, int) or count != len(samples) or (count != 1000 and not 8 <= count <= 64): errors.append("selection_count_invalid")
    if len({r.get("stem") for r in samples}) != count: errors.append("selected_stems_not_globally_unique")
    if count == 1000 and manifest.get("quotas") != QUOTAS: errors.append("full_selection_quota_invalid")
    if count != 1000 and manifest.get("run_label") != "confirmation_nonpublish": errors.append("confirmation_run_not_labeled_nonpublish")
    identity = [{key: row.get(key) for key in ("speaker", "stem", "source_relative_wav", "source_relative_txt", "wav_sha256", "txt_sha256")}
                for row in samples]
    if digest(identity) != manifest.get("selection_digest"): errors.append("selection_manifest_digest_mismatch")
    wanted = {r["stem"]: r for r in samples}
    input_dir = root / "input"
    expected_names = {f"{stem}{suffix}" for stem in wanted for suffix in (".wav", ".txt")}
    actual_names = {entry.name for entry in input_dir.iterdir()} if input_dir.is_dir() else set()
    if actual_names != expected_names or len(actual_names) != count * 2:
        errors.append("flat_input_namespace_not_exact")
    for entry in input_dir.iterdir() if input_dir.is_dir() else ():
        if entry.is_dir() or entry.is_symlink() or not entry.is_file(): errors.append(f"flat_input_nonordinary_entry:{entry.name}")
    for stem, row in wanted.items():
        for suffix, key in ((".wav", "wav_sha256"), (".txt", "txt_sha256")):
            file = input_dir / f"{stem}{suffix}"
            if (not file.is_file() or file.is_symlink() or str(file) != row.get(key.replace("_sha256", "_destination"))
                    or sha_file(file) != row.get(key)):
                errors.append(f"input_hash_mismatch:{stem}{suffix}")
    shard_rows = shards.get("shards", [])
    members = [stem for shard in shard_rows for stem in shard.get("stems", [])]
    expected_sizes = shard_sizes(count) if isinstance(count, int) else []
    if (len(shard_rows) != 8 or [len(s.get("stems", [])) for s in shard_rows] != expected_sizes
            or set(members) != set(wanted) or len(members) != len(set(members))):
        errors.append("shard_plan_not_eight_disjoint_expected_sizes")
    if sha_file(root / "resolved_gpu1000_nvrasr_fallback.yaml") != evidence.get("resolved_config_sha256"):
        errors.append("resolved_config_hash_mismatch")
    for name, path in (("run_pipeline_sha256", PROJECT / "scripts/run_pipeline.py"),
                       ("ctc_prealign_sha256", PROJECT / "scripts/ctc_prealign.py")):
        if sha_file(path) != evidence.get(name): errors.append(f"code_hash_mismatch:{path.name}")
    cfg = load_yaml(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    if (cfg.get("data_dir") != str(root / "input") or cfg.get("workspace") != str(root / "workspace")
            or cfg.get("output_dir") != str(root / "configured_output_never_publish")
            or cfg.get("output_staging") is not False or cfg.get("use_cache") is not False):
        errors.append("resolved_config_safety_values_invalid")
    ctc = cfg.get("ctc_prealign", {})
    if not (ctc.get("enabled") is True and ctc.get("all_gpus") is True and ctc.get("limit") == 0):
        errors.append("resolved_ctc_all_gpus_limit_invalid")
    if (root / "workspace").exists() or (root / "configured_output_never_publish").exists():
        errors.append("prepared_root_not_exclusive_workspace_or_output_exists")
    if "/mnt/Raw" in (root / "resolved_gpu1000_nvrasr_fallback.yaml").read_text(encoding="utf-8"):
        errors.append("forbidden_publish_path")
    return errors


def _command_error(label: str, completed: subprocess.CompletedProcess[str]) -> SafetyError:
    """Keep diagnostic evidence useful without embedding unbounded command output."""
    stderr = (completed.stderr or "").strip()[:2000]
    stdout = (completed.stdout or "").strip()[:2000]
    return SafetyError(f"{label} rc={completed.returncode} stderr={stderr!r} stdout={stdout!r}")


def gpu_snapshot(fake: str | None = None) -> list[dict[str, Any]]:
    if fake:
        data = json.loads(Path(fake).read_text(encoding="utf-8"))
        return data["gpus"] if isinstance(data, dict) else data
    gpu_proc = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu",
                               "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    if gpu_proc.returncode: raise _command_error("nvidia-smi gpu query failed", gpu_proc)
    rows: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    for line in gpu_proc.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 5: raise SafetyError(f"nvidia-smi gpu query malformed row: {line!r}")
        row = {"index": int(values[0]), "uuid": values[1], "memory_free_mib": int(values[2]),
               "memory_used_mib": int(values[3]),
               "utilization_gpu_pct": None if values[4] in ("", "N/A", "[Not Supported]") else int(values[4]),
               "compute_pids": []}
        rows.append(row); by_uuid[values[1]] = row
    apps_proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if apps_proc.returncode: raise _command_error("nvidia-smi compute-app query failed", apps_proc)
    for line in apps_proc.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if not line.strip(): continue
        if len(values) != 2: raise SafetyError(f"nvidia-smi compute-app query malformed row: {line!r}")
        pid, uuid = values
        if uuid not in by_uuid: raise SafetyError(f"nvidia-smi compute-app references unknown gpu uuid: {uuid!r}")
        by_uuid[uuid]["compute_pids"].append(pid)
    return rows


def resolve_mfa_dependency(python: str) -> tuple[str | None, str]:
    """Resolve MFA in the exact interpreter environment, never ambient PATH."""
    executable = Path(python).resolve()
    sibling = executable.parent / "mfa"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling), "sibling_executable"
    probe = subprocess.run([str(executable), "-c", "import montreal_forced_aligner"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if probe.returncode == 0:
        return f"module:{executable}:montreal_forced_aligner", "module_import"
    return None, f"missing sibling {sibling}; module probe {_command_error('mfa module import failed', probe)}"


def preflight(args: argparse.Namespace) -> int:
    root = args.root.resolve(); errors = validate_prepared(root)
    cfg = load_yaml(root / "resolved_gpu1000_nvrasr_fallback.yaml")
    if args.fake_gpus:
        mfa_path, mfa_resolution = None, "not_checked_fake_mode"
    else:
        mfa_path, mfa_resolution = resolve_mfa_dependency(getattr(args, "python", sys.executable))
    if not args.fake_gpus:
        for record in provenance_paths(cfg):
            if record["exists"] != "True": errors.append(f"missing_dependency:{record['path']}")
        if mfa_path is None: errors.append("missing_dependency:mfa")
    free = shutil.disk_usage(root).free
    if free < 100 * 1024**3: errors.append("insufficient_disk_less_than_100GiB")
    try:
        gpus = gpu_snapshot(args.fake_gpus)
        indices = sorted(g.get("index") for g in gpus)
        if indices != list(range(8)): errors.append("need_exactly_gpus_0_through_7")
        for gpu in gpus:
            if int(gpu.get("memory_free_mib", 0)) < 40 * 1024: errors.append(f"gpu{gpu.get('index')}_free_memory_below_40GiB")
            if gpu.get("compute_pids"): errors.append(f"gpu{gpu.get('index')}_has_compute_pids")
    except (ValueError, OSError, SafetyError, json.JSONDecodeError) as exc: errors.append(f"gpu_probe_failed:{exc}")
    receipt = {"schema": "gpu1000-preflight-v1", "at_utc": datetime.now(timezone.utc).isoformat(),
               "root": str(root), "errors": errors, "gpus": gpus if 'gpus' in locals() else [],
               "mfa_interpreter": str(Path(getattr(args, "python", sys.executable)).resolve()),
               "resolved_mfa_path": mfa_path, "mfa_resolution": mfa_resolution}
    write_once(root / "preflight_receipt.json", receipt)
    if errors: raise SafetyError("preflight failed: " + "; ".join(errors))
    print(json.dumps(receipt, ensure_ascii=False)); return 0


def safe_env() -> dict[str, str]:
    return {key: os.environ[key] for key in ENV_ALLOWLIST if key in os.environ}


def telemetry_loop(path: Path, stop: threading.Event, fake: str | None) -> None:
    while not stop.is_set():
        try: snapshot: dict[str, Any] = {"at_utc": datetime.now(timezone.utc).isoformat(), "gpus": gpu_snapshot(fake)}
        except Exception as exc: snapshot = {"at_utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
        with path.open("a", encoding="utf-8") as out: out.write(json.dumps(snapshot) + "\n")
        stop.wait(TELEMETRY_INTERVAL_SECONDS)


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if (root / "run_receipt.json").exists() or (root / "run_started.json").exists():
        raise SafetyError("same run root cannot be run twice")
    if not (root / "preflight_receipt.json").is_file(): raise SafetyError("run requires successful preflight receipt")
    if read_json(root / "preflight_receipt.json").get("errors"): raise SafetyError("run requires successful preflight")
    errors = validate_prepared(root)
    if errors: raise SafetyError("prepared state changed: " + "; ".join(errors))
    config = root / "resolved_gpu1000_nvrasr_fallback.yaml"
    if "/mnt/Raw" in config.read_text(encoding="utf-8"): raise SafetyError("refusing /mnt/Raw publication")
    argv = [args.python, str(PROJECT / "scripts/run_pipeline.py"), "--config", str(config), "--python", args.python,
            "--no-cache", "--no-output-staging", "--validate"]
    write_once(root / "run_started.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
                                            "environment": safe_env(), "telemetry_interval_seconds": TELEMETRY_INTERVAL_SECONDS})
    if args.dry_run:
        write_once(root / "run_receipt.json", {"schema": "gpu1000-run-v1", "dry_run": True, "argv": argv,
                                                "returncode": 0, "started_at_utc": datetime.now(timezone.utc).isoformat(), "ended_at_utc": datetime.now(timezone.utc).isoformat()})
        return 0
    stdout, stderr, telemetry = root / "pipeline.stdout.log", root / "pipeline.stderr.log", root / "nvidia-smi.telemetry.jsonl"
    stop = threading.Event(); watcher = threading.Thread(target=telemetry_loop, args=(telemetry, stop, args.fake_gpus), daemon=True)
    started = datetime.now(timezone.utc).isoformat(); watcher.start()
    try:
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            completed = subprocess.run(argv, cwd=PROJECT, env=safe_env(), stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False)
    finally:
        stop.set(); watcher.join(timeout=20)
    write_once(root / "run_receipt.json", {"schema": "gpu1000-run-v1", "argv": argv, "environment": safe_env(), "returncode": completed.returncode,
                                             "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                             "stdout_sha256": sha_file(stdout), "stderr_sha256": sha_file(stderr), "telemetry_sha256": sha_file(telemetry)})
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--root", type=Path, required=True); p.add_argument("--source", type=Path, default=Path("/mnt/nvme3/mfa_audio_cache_ria")); p.add_argument("--config", type=Path, default=PROJECT / "configs/gpu1000_nvrasr_fallback.yaml"); p.add_argument("--count", type=int); p.add_argument("--selected-stems", type=Path); p.set_defaults(func=prepare)
    p = sub.add_parser("preflight"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable); p.add_argument("--fake-gpus", type=str); p.set_defaults(func=preflight)
    p = sub.add_parser("run"); p.add_argument("--root", type=Path, required=True); p.add_argument("--python", default=sys.executable); p.add_argument("--fake-gpus", type=str); p.add_argument("--dry-run", action="store_true"); p.set_defaults(func=run)
    p = sub.add_parser("analyze"); p.add_argument("--root", type=Path, required=True); p.add_argument("--json-out", type=Path); p.add_argument("--markdown-out", type=Path); p.set_defaults(func=lambda a: __import__("analyze_gpu1000_run").analyze_command(a))
    args = parser.parse_args()
    try: return args.func(args)
    except SafetyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
