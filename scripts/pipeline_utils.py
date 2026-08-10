#!/usr/bin/env python3
"""
共享工具 — 路径翻译、文件发现、MFA 环境。

被 run_pipeline.py 和 streaming_pipeline.py 共同导入。
"""

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════
# UNC -> Linux 路径翻译
# ═══════════════════════════════════════════════════════════════

_WIN_UNC_MAP: dict[str, str] = {}


def _detect_smb_mounts() -> dict[str, str]:
    """Parse /proc/mounts for CIFS mounts -> UNC->linux mapping."""
    mapping: dict[str, str] = {}
    if platform.system() == "Windows":
        return mapping
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            dev, mnt, fstype = parts[0], parts[1], parts[2]
            if fstype != "cifs":
                continue
            dev_path = dev.replace("//", "", 1)
            if dev_path.startswith("192.168."):
                unc = f"//{dev_path}"
                mapping[unc] = mnt
                mapping[unc.replace("/", "\\")] = mnt
        for unc, mnt in list(mapping.items()):
            clean = unc.replace("\\", "/")
            if "192.168.102.202/Research_TTS" in clean:
                parts_after = clean.split("Research_TTS", 1)
                if len(parts_after) > 1:
                    suffix = parts_after[1]
                    mapping[f"//RS3621/Research_TTS{suffix}"] = mnt
                    mapping[f"\\\\RS3621\\Research_TTS{suffix.replace('/', chr(92))}"] = mnt
    except Exception:
        pass
    return mapping


_WIN_UNC_MAP = _detect_smb_mounts()


def translate_path(path_str: str) -> str:
    """Convert Windows UNC -> Linux mount path."""
    if not path_str or platform.system() == "Windows":
        return path_str
    normalized = path_str.replace("\\", "/")
    for unc_raw, linux_mnt in sorted(_WIN_UNC_MAP.items(),
                                     key=lambda x: -len(x[0])):
        unc_norm = unc_raw.replace("\\", "/")
        if normalized.startswith(unc_norm):
            rest = normalized[len(unc_norm):].lstrip("/")
            return f"{linux_mnt}/{rest}" if rest else linux_mnt
    return path_str


def resolve_input_path(raw: str, base: Path = PROJECT_ROOT) -> Path:
    """Translate UNC + resolve relative -> absolute Path."""
    if not raw:
        return base
    translated = translate_path(raw)
    p = Path(translated)
    return p if p.is_absolute() else (base / p)


# ═══════════════════════════════════════════════════════════════
# MFA Python 发现
# ═══════════════════════════════════════════════════════════════

def find_mfa_python(cfg_python: str = "") -> Optional[Path]:
    """Auto-detect Python with MFA installed.

    Checks (in order): explicit config path -> ``mfa`` on PATH ->
    common conda environments (Linux & Windows).
    """
    if cfg_python:
        p = Path(cfg_python)
        if p.exists():
            return p

    # Try config/env-sourced Python (mfa on PATH)
    mfa_on_path = shutil.which("mfa")
    if mfa_on_path:
        parent = Path(mfa_on_path).parent
        py = parent / ("python.exe" if os.name == "nt" else "python3")
        if py.exists():
            return py

    # Search common conda envs
    home = Path.home()
    is_win = os.name == "nt"
    conda_roots = [
        home / "miniconda3",
        home / "anaconda3",
        home / "opt" / "miniconda3",
        home / "opt" / "anaconda3",
        Path("/opt/conda"),
        Path("/usr/local/anaconda3"),
    ]
    env_names = ["mfa_chinese", "mfa_mandarin", "mfa", "mfa-dev", "asr"]

    for conda_root in conda_roots:
        for env_name in env_names:
            env_dir = conda_root / "envs" / env_name
            py_bin = env_dir / ("python.exe" if is_win else "bin/python3")
            if py_bin.exists():
                return py_bin
            py_bin = env_dir / ("python.exe" if is_win else "bin/python")
            if py_bin.exists():
                return py_bin

    return None


def get_mfa_env(mfa_python: Path, models_dir: Path,
                 blas_num_threads: str = "1") -> dict[str, str]:
    """Build environment dict for MFA subprocess calls.

    *blas_num_threads* controls BLAS threading per Kaldi worker.
    Default ``"1"`` prevents oversubscription: with N MFA workers on
    an M-core machine, each OpenBLAS/MKL call would otherwise spawn
    M threads internally, creating N×M total threads and catastrophic
    contention.  Single-threaded BLAS + process-level parallelism
    (MFA's ``--num_jobs``) gives near-linear scaling.
    """
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(models_dir)
    # Pin BLAS threads per worker — critical for multi-core scaling
    for ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[ev] = blas_num_threads
    lib_bin = mfa_python.parent / "Library" / "bin"
    paths = [str(mfa_python.parent)]
    if lib_bin.exists():
        paths.append(str(lib_bin))
    if "PATH" in env:
        paths.append(env["PATH"])
    env["PATH"] = os.pathsep.join(paths)
    return env


# ═══════════════════════════════════════════════════════════════
# 文件索引 — 单次 scandir + set 查找 (避免逐文件 exists())
# ═══════════════════════════════════════════════════════════════

# CTC 输出文件的 6 种后缀
CTC_SUFFIXES: list[str] = [
    ".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
    "_text_cn.txt", "_text_raw.txt",
]


def build_ctc_presence(ctc_dir: Path) -> "tuple[set[str], dict[str, set[str]]]":
    """单次 os.scandir -> O(1) 文件名查找。

    Returns:
        flat_names:   顶层文件名集合
        nested_names: {子目录名: {该子目录内文件名}}
    """
    flat_names: set[str] = set()
    nested_names: dict[str, set[str]] = {}

    try:
        with os.scandir(str(ctc_dir)) as it:
            for entry in it:
                if entry.is_file():
                    flat_names.add(entry.name)
                elif entry.is_dir():
                    sub = set()
                    try:
                        with os.scandir(entry.path) as it2:
                            for e2 in it2:
                                if e2.is_file():
                                    sub.add(e2.name)
                    except OSError:
                        pass
                    if sub:
                        nested_names[entry.name] = sub
    except OSError:
        pass

    return flat_names, nested_names


def build_file_index(root: Path, suffix: str) -> dict[str, Path]:
    """{stem: path} index — single scandir, no rglob."""
    index: dict[str, Path] = {}
    try:
        with os.scandir(str(root)) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(suffix):
                    stem = entry.name[:-len(suffix)]
                    if stem not in index:
                        index[stem] = Path(entry.path)
    except OSError:
        pass
    # Try one level of subdirectories
    if not index:
        try:
            with os.scandir(str(root)) as it:
                for entry in it:
                    if entry.is_dir():
                        try:
                            with os.scandir(entry.path) as it2:
                                for e2 in it2:
                                    if e2.is_file() and e2.name.endswith(suffix):
                                        stem = e2.name[:-len(suffix)]
                                        if stem not in index:
                                            index[stem] = Path(e2.path)
                        except OSError:
                            pass
        except OSError:
            pass

    # Deeply-nested fallback (e.g. session/clip/stem.wav) — use find(1)
    if not index:
        try:
            import subprocess as _sp
            result = _sp.run(
                ["find", str(root), "-name", f"*{suffix}", "-type", "f"],
                capture_output=True, text=True, timeout=120)
            for line in result.stdout.splitlines():
                stem = Path(line).stem
                if stem and stem not in index:
                    index[stem] = Path(line)
        except Exception:
            pass

    return index


def build_flat_file_names(root: Path, suffix: str) -> set[str]:
    """Return matching top-level file names with one directory scan.

    This deliberately does not recurse or fall back to ``find``: callers use
    it for flat-layout manifests where a nested file must not satisfy the
    manifest's ``{stem}{suffix}`` contract.  ``DirEntry.is_file`` preserves
    the previous ``Path.is_file`` symlink-following behavior while reducing
    one metadata syscall per expected stem to a single ``scandir`` pass.
    """
    names: set[str] = set()
    try:
        with os.scandir(str(root)) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(suffix):
                    names.add(entry.name[:-len(suffix)])
    except OSError:
        pass
    return names


def count_files_fast(dirpath: Path, suffix: str, max_count: int = 10000) -> int:
    """Count files ending with *suffix*, bailing at *max_count*."""
    n = 0
    try:
        with os.scandir(str(dirpath)) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(suffix):
                    n += 1
                    if n >= max_count:
                        return n
    except OSError:
        pass
    return n


def find_wav(audio_dir: Path, stem: str) -> Optional[Path]:
    """Find {stem}.wav — flat -> nested -> zero-padded -> glob fallback."""
    wav = audio_dir / f"{stem}.wav"
    if wav.exists():
        return wav
    wav = audio_dir / stem / f"{stem}.wav"
    if wav.exists():
        return wav
    if stem.isdigit():
        for width in (5, 6, 7, 8):
            wav = audio_dir / f"{stem.zfill(width)}.wav"
            if wav.exists():
                return wav
    candidates = list(audio_dir.glob(f"**/{stem}.wav"))
    if candidates:
        return candidates[0]
    m = re.search(r'(\d+)$', stem)
    if m:
        candidates = list(audio_dir.glob(f"**/*{m.group(1)}.wav"))
        candidates = [c for c in candidates
                      if re.search(rf'(?<!\d){m.group(1)}(?!\d)', c.stem)]
        return candidates[0] if candidates else None
    return None


def link_or_copy_file(src: Path, dst: Path) -> bool:
    """Best-effort: hard-link -> symlink -> copy."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    for fn in (os.link, os.symlink):
        try:
            fn(str(src), str(dst))
            return True
        except OSError:
            pass
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except OSError:
        return False


def discover_stems(ctc_dir: Path, audio_dir: Path,
                   require_all: bool = True) -> "tuple[list[str], dict[str, str]]":
    """Discover valid stems with layout info — avoids per-file exists() on SMB.

    Returns:
        stems:      sorted list of valid stems
        layout_map: {stem: "flat"|"nested"} — which CTC layout each stem uses
    """
    flat_names, nested_names = build_ctc_presence(ctc_dir)

    # Build audio index (single scandir)
    audio_index: set[str] = set()
    try:
        with os.scandir(str(audio_dir)) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".wav"):
                    audio_index.add(entry.name[:-4])
        if not audio_index:
            try:
                with os.scandir(str(audio_dir)) as it:
                    for entry in it:
                        if entry.is_dir():
                            try:
                                with os.scandir(entry.path) as it2:
                                    for e2 in it2:
                                        if e2.is_file() and e2.name.endswith(".wav"):
                                            audio_index.add(e2.name[:-4])
                            except OSError:
                                pass
            except OSError:
                pass
    except OSError:
        pass

    # Deeply-nested fallback (e.g. session/clip/stem.wav) — use find(1) which
    # is orders of magnitude faster than Python rglob over CIFS.
    if not audio_index:
        try:
            import subprocess as _sp
            result = _sp.run(
                ["find", str(audio_dir), "-name", "*.wav", "-type", "f"],
                capture_output=True, text=True, timeout=120)
            for line in result.stdout.splitlines():
                stem = Path(line).stem
                if stem:
                    audio_index.add(stem)
        except Exception:
            pass

    # Collect candidates
    candidate_stems: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fname in flat_names:
        if fname.endswith(".lab"):
            stem = fname[:-4]
            if stem not in seen:
                candidate_stems.append((stem, "flat"))
                seen.add(stem)
    for dirname, sub_files in nested_names.items():
        if f"{dirname}.lab" in sub_files:
            if dirname not in seen:
                candidate_stems.append((dirname, "nested"))
                seen.add(dirname)

    # Validate
    valid: list[str] = []
    layout_map: dict[str, str] = {}
    for stem, layout in candidate_stems:
        if stem not in audio_index:
            if find_wav(audio_dir, stem) is None:
                continue
        if require_all:
            if layout == "flat":
                ok = all(f"{stem}{suffix}" in flat_names
                         for suffix in CTC_SUFFIXES)
            else:
                ok = all(f"{stem}{suffix}" in nested_names.get(stem, set())
                         for suffix in CTC_SUFFIXES)
            if not ok:
                continue
        valid.append(stem)
        layout_map[stem] = layout

    valid.sort()
    return valid, layout_map


def discover_stems_separated(ctc_dir: Path, audio_dir: Path,
                             require_all: bool = True) -> "tuple[list[str], list[str], dict[str, str], dict[str, Path]]":
    """Like discover_stems() but returns (complete, incomplete) separately.

    Incomplete stems are those with audio + a .lab file but missing ≥1 CTC suffix.
    Stems without any .lab file are excluded entirely (never processed by NVASR).

    Returns:
        complete_stems:   sorted list of stems with all CTC files + audio
        incomplete_stems: sorted list of stems with audio + .lab but missing ≥1 CTC suffix
        layout_map:       {stem: "flat"|"nested"}
        wav_index:        {stem: resolved_wav_path}
    """
    flat_names, nested_names = build_ctc_presence(ctc_dir)

    # Build audio index (single scandir)
    audio_index: set[str] = set()
    try:
        with os.scandir(str(audio_dir)) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".wav"):
                    audio_index.add(entry.name[:-4])
        if not audio_index:
            try:
                with os.scandir(str(audio_dir)) as it:
                    for entry in it:
                        if entry.is_dir():
                            try:
                                with os.scandir(entry.path) as it2:
                                    for e2 in it2:
                                        if e2.is_file() and e2.name.endswith(".wav"):
                                            audio_index.add(e2.name[:-4])
                            except OSError:
                                pass
            except OSError:
                pass
    except OSError:
        pass

    # Deeply-nested fallback (e.g. session/clip/stem.wav) — use find(1) which
    # is orders of magnitude faster than Python rglob over CIFS.
    if not audio_index:
        try:
            import subprocess as _sp
            result = _sp.run(
                ["find", str(audio_dir), "-name", "*.wav", "-type", "f"],
                capture_output=True, text=True, timeout=120)
            for line in result.stdout.splitlines():
                stem = Path(line).stem
                if stem:
                    audio_index.add(stem)
        except Exception:
            pass

    # Collect candidates
    candidate_stems: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fname in flat_names:
        if fname.endswith(".lab"):
            stem = fname[:-4]
            if stem not in seen:
                candidate_stems.append((stem, "flat"))
                seen.add(stem)
    for dirname, sub_files in nested_names.items():
        if f"{dirname}.lab" in sub_files:
            if dirname not in seen:
                candidate_stems.append((dirname, "nested"))
                seen.add(dirname)

    # Validate — split into complete and incomplete
    complete_stems: list[str] = []
    incomplete_stems: list[str] = []
    layout_map: dict[str, str] = {}
    wav_index: dict[str, Path] = {}
    for stem, layout in candidate_stems:
        # Must have audio
        if stem not in audio_index:
            wav_path = find_wav(audio_dir, stem)
            if wav_path is None:
                continue
            wav_index[stem] = wav_path
        else:
            # We only know the stem exists, resolve the full path
            wav_path = find_wav(audio_dir, stem)
            if wav_path is None:
                continue
            wav_index[stem] = wav_path

        if require_all:
            if layout == "flat":
                all_ok = all(f"{stem}{suffix}" in flat_names
                             for suffix in CTC_SUFFIXES)
            else:
                all_ok = all(f"{stem}{suffix}" in nested_names.get(stem, set())
                             for suffix in CTC_SUFFIXES)
            if all_ok:
                complete_stems.append(stem)
            else:
                incomplete_stems.append(stem)
        else:
            complete_stems.append(stem)

        layout_map[stem] = layout

    complete_stems.sort()
    incomplete_stems.sort()
    return complete_stems, incomplete_stems, layout_map, wav_index


# ═══════════════════════════════════════════════════════════════
# 数据传输 (rsync/cp)
# ═══════════════════════════════════════════════════════════════

def _has_rsync() -> bool:
    return shutil.which("rsync") is not None


def copy_tree_fast(src: Path, dst: Path) -> bool:
    """rsync -a or shutil.copytree fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _has_rsync():
        rc = subprocess.run(
            ["rsync", "-a", "--no-inc-recursive",
             str(src) + "/", str(dst) + "/"],
            capture_output=True, text=True, timeout=600).returncode
        if rc == 0:
            return True
    try:
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(str(src), str(dst), symlinks=True, dirs_exist_ok=True)
        return True
    except Exception as e:
        print(f"  Copy failed: {e}")
        return False


def sync_tree_back(src: Path, dst: Path) -> bool:
    """Sync local -> NAS with 3 retries + exponential backoff."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        if _has_rsync():
            rc = subprocess.run(
                ["rsync", "-a", "--remove-source-files",
                 "--no-inc-recursive",
                 str(src) + "/", str(dst) + "/"],
                capture_output=True, text=True, timeout=600).returncode
            if rc == 0:
                return True
        else:
            try:
                for f in src.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(src)
                        target = dst / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(f), str(target))
                for d in sorted(src.rglob("*"), reverse=True):
                    if d.is_dir() and not any(d.iterdir()):
                        d.rmdir()
                return True
            except Exception as e:
                print(f"  Upload attempt {attempt+1} failed: {e}")
        import time
        time.sleep(2 ** attempt)
    return False


def write_publish_manifest(src: Path) -> Path:
    """Write a manifest for one run-specific output staging directory."""
    src = src.resolve()
    payload = []
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.name == ".publish_manifest.json":
            continue
        rel = path.relative_to(src).as_posix()
        payload.append({
            "path": rel,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    manifest = {
        "schema": 2,
        "source": str(src),
        "files": payload,
    }
    manifest_path = src / ".publish_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_publish_accounting(src: Path, strict_manifest_path: Path) -> list[str]:
    """Validate frozen v2 accounting and strict membership before publishing."""
    try:
        manifest = json.loads(strict_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"strict manifest unreadable: {exc}"]
    binding = manifest.get("pipeline_accounting_receipt")
    if not isinstance(binding, dict):
        return ["pipeline accounting receipt binding missing"]
    if binding.get("schema") != PIPELINE_ACCOUNTING_SCHEMA:
        return ["pipeline accounting receipt must be v2"]
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return ["pipeline accounting receipt path missing"]
    receipt_path = Path(raw_path)
    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            return ["pipeline accounting receipt is not a regular file"]
        receipt = read_pipeline_accounting_receipt(receipt_path)
        if binding.get("sha256") != _sha256_file(receipt_path):
            return ["pipeline accounting receipt hash mismatch"]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"pipeline accounting receipt invalid: {exc}"]
    errors = validate_pipeline_accounting_receipt(receipt)
    if errors:
        return [f"pipeline accounting receipt invalid: {error}" for error in errors]
    try:
        expected_raw = manifest["expected_stems"]
        ok_raw = manifest["ok"]
        rejected_raw = manifest["rejected"]
        expected = set(_accounting_stems(expected_raw, "strict expected_stems"))
        strict_ok = {entry["stem"] for entry in ok_raw}
        strict_rejected = set(rejected_raw)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"strict accounting evidence malformed: {exc}"]
    eligible = set(receipt["eligible"]["stems"])
    output = set(receipt["output"]["stems"])
    filtered = set(receipt["filtered"]["stems"])
    if expected != eligible:
        return ["strict expected stems do not equal receipt eligible stems"]
    if strict_ok != output or strict_rejected != filtered:
        return ["strict output/rejected sets do not equal receipt output/filtered"]
    return []


def publish_output_versioned(src: Path, dst: Path) -> bool:
    """Publish validated staging to a new, empty versioned destination.

    The function never deletes or overwrites a pre-existing result directory.
    Callers must select a fresh versioned destination for every publication.
    The source is retained so a failed upload remains recoverable.
    """
    try:
        src = src.resolve(strict=True)
        dst = dst.resolve(strict=False)
    except OSError as exc:
        print(f"  Publish path resolution failed: {exc}")
        return False
    if src == dst or not dst.is_absolute() or len(dst.parts) < 3:
        print(f"  Refusing unsafe publish target: {dst}")
        return False
    if not dst.parent.name.endswith(".runs"):
        print(f"  Refusing non-versioned strict publish target: {dst}")
        return False
    accounting_errors = _validate_publish_accounting(src, src / "strict_ok_manifest.json")
    if accounting_errors:
        print("  Refusing publish: " + "; ".join(accounting_errors))
        return False
    try:
        from verify_strict_ok import verify as _verify_strict_ok
        strict_errors = _verify_strict_ok(src / "strict_ok_manifest.json", src)
    except Exception as exc:
        print(f"  Strict manifest verification could not run: {exc}")
        return False
    if strict_errors:
        print("  Refusing publish: strict manifest invalid: " + ", ".join(strict_errors))
        return False
    if dst.exists():
        # Even an empty pre-created directory is not a fresh run target.  This
        # prevents a retry from silently reusing a version identifier.
        print(f"  Refusing existing publish target: {dst}")
        return False

    manifest_path = src / ".publish_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema"] != 2:
            raise ValueError("publish manifest schema must be 2")
        entries = manifest["files"]
        expected_payload = {
            entry["path"]: (int(entry["size"]), str(entry["sha256"]))
            for entry in entries
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"  Invalid publish manifest {manifest_path}: {exc}")
        return False

    for rel in expected_payload:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            print(f"  Unsafe path in publish manifest: {rel}")
            return False
    actual_payload = {
        path.relative_to(src).as_posix(): (path.stat().st_size, _sha256_file(path))
        for path in src.rglob("*")
        if path.is_file() and path.name != ".publish_manifest.json"
    }
    if actual_payload != expected_payload:
        print("  Publish manifest does not match staging size/hash contents")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if _has_rsync():
        result = subprocess.run(
            ["rsync", "-a", "--no-inc-recursive",
             str(src) + "/", str(dst) + "/"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  Mirror publish failed (rsync rc={result.returncode}): "
                  f"{result.stderr[-1000:]}")
            return False
    else:
        try:
            dst.mkdir(parents=True, exist_ok=True)
            for source in src.rglob("*"):
                if source.is_file():
                    rel = source.relative_to(src)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        except OSError as exc:
            print(f"  Versioned publish failed: {exc}")
            return False

    expected_files = set(expected_payload) | {".publish_manifest.json"}
    published_files = {
        path.relative_to(dst).as_posix()
        for path in dst.rglob("*") if path.is_file()
    }
    if published_files != expected_files:
        print("  Published file set does not match the run manifest")
        return False
    for rel, (size, digest) in expected_payload.items():
        try:
            if (dst / rel).stat().st_size != size:
                print(f"  Published size mismatch: {rel}")
                return False
            if _sha256_file(dst / rel) != digest:
                print(f"  Published SHA-256 mismatch: {rel}")
                return False
        except OSError:
            return False
    try:
        destination_errors = _verify_strict_ok(dst / "strict_ok_manifest.json", dst)
    except Exception as exc:
        print(f"  Published strict manifest verification could not run: {exc}")
        return False
    if destination_errors:
        print("  Published strict manifest invalid: " + ", ".join(destination_errors))
        return False
    return True


# ═══════════════════════════════════════════════════════════════
# Shared constants — canonical definitions used across the pipeline.
# Edit HERE when adding/changing NVV names, IPA mappings, etc.
# ═══════════════════════════════════════════════════════════════

import re as _re

# Versioned contract for CTC-side transcript normalization.  A marker from an
# older schema must never make a newer pipeline skip validation/recovery.
CTC_NORMALIZATION_MARKER = "reference-authority-v3-safe-transcript\n"

# v4 marker embeds content identity (stem count + manifest digest) so a
# marker leftover from a different run or tampered data cannot be mistaken
# for a valid normalization certificate.
_CTC_MARKER_V4_HEADER = "reference-authority-v4-safe-transcript"


def make_ctc_normalization_marker(stem_count: int, manifest_sha256: str) -> str:
    """Build a v4 marker that binds content identity to the certificate."""
    return (
        f"{_CTC_MARKER_V4_HEADER}\n"
        f"stems={stem_count}\n"
        f"manifest_sha256={manifest_sha256}\n"
    )


def parse_ctc_normalization_marker(text: str) -> dict | None:
    """Extract content identity from a v4 marker.

    Returns a dict with keys ``stems`` (int) and ``manifest_sha256`` (str),
    or ``None`` when the marker is missing, unparseable, or from an older
    schema version.
    """
    lines = text.strip().split("\n")
    if not lines or lines[0] != _CTC_MARKER_V4_HEADER:
        return None
    info: dict = {}
    for line in lines[1:]:
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    if "stems" not in info or "manifest_sha256" not in info:
        return None
    try:
        info["stems"] = int(info["stems"])
    except ValueError:
        return None
    return info

# ── Silence / pause tokens ──────────────────────────────────────
SILENCE_LABELS: set[str] = {"<eps>", "<sil>", "sil", "<sp0>", "<sp1>", "<sp2>", "<sp3>", "spn"}


def is_silence(text: str) -> bool:
    """Check if *text* is a silence / pause token."""
    t = text.strip()
    return t in SILENCE_LABELS or t.startswith("<sp") or t in ("", "<eps>")


# ── English phone prefix for mixed-language tier output ──────────
EN_PHONE_PREFIX: str = "en:"

# ── NVV (Non-Verbal Vocalisation) names ─────────────────────────
NVV_NAMES: set[str] = {
    "BREATHING", "LAUGHTER", "BURP", "COUGH", "CRYING", "GROAN",
    "HISS", "HUM", "SHH", "SIGH", "SNEEZE", "SNIFF", "SNORE",
    "TSK", "UHM", "WHISTLE", "YAWN",
    "QUESTION-YI", "QUESTION-EN", "QUESTION-OH", "QUESTION-AH",
    "QUESTION-EI", "QUESTION-HUH",
    "SURPRISE-OH", "SURPRISE-AH", "SURPRISE-WA", "SURPRISE-YO",
    "CONFIRMATION-EN", "DISSATISFACTION-HNN",
}

NVV_TO_MFA: dict[str, str] = {
    "Breathing": "BREATHING", "Laughter": "LAUGHTER", "Burp": "BURP",
    "Cough": "COUGH", "Crying": "CRYING", "Groan": "GROAN", "Hiss": "HISS",
    "Hum": "HUM", "Shh": "SHH", "Sigh": "SIGH", "Sneeze": "SNEEZE",
    "Sniff": "SNIFF", "Snore": "SNORE", "Tsk": "TSK", "Uhm": "UHM",
    "Whistle": "WHISTLE", "Yawn": "YAWN",
    "Question-yi": "QUESTION-YI", "Question-en": "QUESTION-EN",
    "Question-oh": "QUESTION-OH", "Question-ah": "QUESTION-AH",
    "Question-ei": "QUESTION-EI", "Question-huh": "QUESTION-HUH",
    "Surprise-oh": "SURPRISE-OH", "Surprise-ah": "SURPRISE-AH",
    "Surprise-wa": "SURPRISE-WA", "Surprise-yo": "SURPRISE-YO",
    "Confirmation-en": "CONFIRMATION-EN",
    "Dissatisfaction-hnn": "DISSATISFACTION-HNN",
    "Pause": "PAUSE",
}

# ── Chinese initials (consonant phones without tone) ────────────
CHINESE_INITIALS_SET: set[str] = {
    "p", "pʰ", "t", "tʰ", "k", "kʰ",
    "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ", "ts", "tsʰ",
    "f", "s", "ɕ", "ʂ", "x",
    "m", "n", "l", "ɻ",
    "j", "w", "ɥ",
    "ŋ", "ʔ",
}

# ── IPA -> pinyin mapping tables ─────────────────────────────────
IPA_CONSONANT_MAP: dict[str, str] = {
    'p': 'b', 'pʰ': 'p', 't': 'd', 'tʰ': 't', 'k': 'g', 'kʰ': 'k',
    'tɕ': 'j', 'tɕʰ': 'q', 'ʈʂ': 'zh', 'ʈʂʰ': 'ch', 'ts': 'z', 'tsʰ': 'c',
    'f': 'f', 's': 's', 'ɕ': 'x', 'ʂ': 'sh', 'x': 'h',
    'm': 'm', 'n': 'n', 'l': 'l', 'ɻ': 'r',
    'j': 'i', 'w': 'u', 'ɥ': 'v',
    'ŋ': 'ng', 'ʔ': '',
    'z̩': 'i0', 'ʐ̩': 'ir',
}

IPA_TONE_TO_DIGIT: dict[str, str] = {
    '˥˥': '1', '˥': '1', '˧˥': '2', '˨˩˦': '3', '˥˩': '4', '˩': '5',
}

IPA_VOWEL_BASE_MAP: dict[str, str] = {
    'a': 'a', 'o': 'o', 'ə': 'e', 'e': 'e',
    'i': 'i', 'u': 'u', 'y': 'v',
    'z̩': 'i0', 'ʐ̩': 'ir',
}

TONE_MARK_CHARS: set[str] = set('˥˧˨˩˦')

FINAL_DECOMPOSE: dict[str, list[str]] = {
    'a': ['a'], 'o': ['o'], 'e': ['e'], 'e2': ['e'],
    'i': ['i'], 'u': ['u'], 'v': ['v'],
    'i0': ['i0'], 'u0': ['u0'], 'v0': ['v0'], 'ir': ['ir'],
    'ai': ['a', 'i'], 'ei': ['e', 'i'], 'ao': ['a', 'u'], 'ou': ['o', 'u'],
    'an': ['a', 'n'], 'en': ['e', 'n'], 'in': ['i', 'n'],
    'ang': ['a', 'ng'], 'eng': ['e', 'ng'], 'ing': ['i', 'ng'], 'ong': ['u', 'ng'],
    'ia': ['i', 'a'], 'ie': ['i', 'e'],
    'iao': ['i', 'a', 'u'], 'iu': ['i', 'o', 'u'], 'iou': ['i', 'o', 'u'],
    'ian': ['i', 'e', 'n'], 'iang': ['i', 'a', 'ng'], 'iong': ['i', 'u', 'ng'],
    'ua': ['u', 'a'], 'uo': ['u', 'o'],
    'uai': ['u', 'a', 'i'], 'ui': ['u', 'e', 'i'], 'uei': ['u', 'e', 'i'],
    'uan': ['u', 'a', 'n'], 'un': ['u', 'e', 'n'], 'uen': ['u', 'e', 'n'],
    'uang': ['u', 'a', 'ng'], 'ueng': ['u', 'e', 'ng'],
    've': ['v', 'e'], 'vn': ['v', 'n'], 'van': ['v', 'e', 'n'],
    'er': ['e', 'r'], 'io': ['i', 'o'],
    'n': ['n'], 'm': ['m'],
}

FINAL_TONE_INDEX: dict[str, int] = {
    'a': 0, 'o': 0, 'e': 0, 'e2': 0, 'i': 0, 'u': 0, 'v': 0,
    'i0': 0, 'u0': 0, 'v0': 0, 'ir': 0,
    'ai': 0, 'ei': 0, 'ao': 0, 'ou': 0,
    'an': 0, 'en': 0, 'in': 0,
    'ang': 0, 'eng': 0, 'ing': 0, 'ong': 0,
    'ia': 1, 'ie': 1, 'iao': 1, 'iu': 1, 'iou': 1,
    'ian': 1, 'iang': 1, 'iong': 1,
    'ua': 1, 'uo': 1, 'uai': 1, 'ui': 1, 'uei': 1,
    'uan': 1, 'un': 1, 'uen': 1,
    'uang': 1, 'ueng': 1,
    've': 1, 'vn': 0, 'van': 1,
    'er': 0, 'io': 1,
    'n': 0, 'm': 0,
}

# ── CJK short function words (often compressed by MFA) ─────────
CHINESE_SHORT_WORDS: set[str] = {
    "的", "了", "着", "呢", "吗", "吧", "啊", "嘛", "呀", "哦",
    "是", "在", "个", "和", "就", "也", "都", "不", "没",
    "de5", "le5", "zhe5", "ne5", "ma5", "ba5", "a5", "ya5",
}


# ═══════════════════════════════════════════════════════════════
# ASR 后处理 — 标点规范化 + ria 音译还原
# (ctc_prealign.py 和 run_pipeline.py 共享, 单一真相源)
# ═══════════════════════════════════════════════════════════════

# ASCII→CJK 标点映射 (逐条即时处理, 替代批量后处理扫描)
_ASCII_TO_CJK_PUNCT: dict[str, str] = {
    ",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "：",
}
_ASCII_TO_CJK_TABLE = str.maketrans(_ASCII_TO_CJK_PUNCT)

# 白名单 CJK 标点 — 非白名单 CJK 标点将被替换为 ，
_NORM_ALLOWED_PUNCT = frozenset("，。！？、；：…")

# ria 中文音译变体 → 拉丁原文
# SenseVoice 有时将英文名 "ria" 识别为近音 CJK 组合
RIA_VARIANTS: dict[str, str] = {
    "瑞娅": "ria",
    "瑞亚": "ria",
    "瑞雅": "ria",
    "瑞啊": "ria",
}


def replace_ria_variants(text: str) -> str:
    """将文本中的中文 ria 音译变体替换为拉丁 ria."""
    for variant, replacement in RIA_VARIANTS.items():
        text = text.replace(variant, replacement)
    return text


def normalize_punct_inline(text: str) -> str:
    """逐条标点规范化: ASCII→CJK + 相邻标点合并 + 非白名单→，."""
    # Phase 1: ASCII → CJK
    text = text.translate(_ASCII_TO_CJK_TABLE)

    # Phase 2: non-whitelist CJK punct/symbol → ，
    chars: list[str] = []
    for ch in text:
        o = ord(ch)
        if ((0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)
                and ch not in _NORM_ALLOWED_PUNCT
                and ch != ' ' and not ('a' <= ch.lower() <= 'z')
                and not ch.isdigit()):
            chars.append('，')
            continue
        chars.append(ch)
    text = ''.join(chars)

    # Phase 3: adjacent punct merge
    merged: list[str] = []
    for ch in text:
        if merged and ch in _NORM_ALLOWED_PUNCT and merged[-1] in _NORM_ALLOWED_PUNCT:
            continue
        merged.append(ch)
    return ''.join(merged)


# ═══════════════════════════════════════════════════════════════
# Character / token classification helpers
# ═══════════════════════════════════════════════════════════════

def is_cjk(ch: str) -> bool:
    """True if *ch* is a single CJK Unified Ideograph character."""
    return ('一' <= ch <= '鿿') or ('㐀' <= ch <= '䶿')


def is_nvv_token(token: str) -> bool:
    """Check if *token* is an NVV label (BREATHING, QUESTION-YI, etc.)."""
    return token.strip().strip('<>').upper() in NVV_NAMES


def is_unknown_token(token: str) -> bool:
    """True only for explicit MFA unknown placeholders.

    A bare ``unk`` can be real English lexical content.  Its validity needs
    authority and English-phone context, which the strict auditor supplies.
    """
    return token.strip().lower() in {"<unk>", "[bracketed]"}


def is_english_token(token: str) -> bool:
    """Token is an English lexical token, including hyphenated spellings."""
    if not token or not token.isalpha():
        # Hyphens are lexical inside forms such as ``K-Pop`` and ``V-up``;
        # reject leading/trailing/repeated hyphens so punctuation is not
        # accidentally classified as an English word.
        if (not token or token[0] == '-' or token[-1] == '-'
                or '--' in token
                or not all(ch.isascii() and (ch.isalpha() or ch == '-')
                           for ch in token)
                or not any(ch.isalpha() for ch in token)):
            return False
    if not token.isascii():
        return False
    if is_unknown_token(token):
        return False
    if is_nvv_token(token):
        return False
    if _re.match(r'^[a-z]+[1-5]$', token):
        return False
    return True


def is_pinyin_syllable(token: str) -> bool:
    """True for Chinese pinyin syllable with tone digit (e.g. jin1, ya4)."""
    return bool(_re.match(r'^[a-z]+[1-5]$', token))


def is_word_like(s: str) -> bool:
    """True for CJK chars, pinyin syllables, English words, digits, NVV labels."""
    if not s:
        return False
    return (is_unknown_token(s) or is_cjk(s) or s[0].isalpha()
            or s.isdigit() or is_nvv_token(s))


def is_punct(s: str) -> bool:
    """True if *s* is a non-word token (punctuation / symbol)."""
    return bool(s.strip()) and not is_word_like(s)


# ── CTC transcript bundle integrity ────────────────────────────

_MIXED_PINYIN_TONE_RE = _re.compile(r"^[a-z]+[一二三四五]$")
_NUMERAL_PROTECTED_RE = _re.compile(
    r"(\[[^\]]+\]|<[^>]+>|(?<![A-Za-z0-9])[a-z]+[1-5](?![A-Za-z0-9])"
    r"|[A-Z][A-Z0-9-]*[A-Z0-9])"
)


def normalize_reference_numerals(text: str, transform) -> str:
    """Apply a numeral transform while preserving lexical control tokens.

    This helper is for human/reference text, never for an already-tokenized
    MFA lab transcript.  Pinyin tone digits, bracketed/NVV labels and uppercase
    identifiers are protected defensively.
    """
    parts = _NUMERAL_PROTECTED_RE.split(text)
    for index, part in enumerate(parts):
        if not part or _NUMERAL_PROTECTED_RE.fullmatch(part):
            continue
        try:
            parts[index] = transform(part, "an2cn")
        except Exception:
            # cn2an can reject mixed strings.  Preserve the source rather than
            # changing a transcript on a best-effort guess.
            parts[index] = part
    return "".join(parts)


def load_ctc_token_entries(tokens_path: Path) -> list[dict]:
    """Load and validate one CTC tokens JSONL file.

    Token order and timing are part of the CTC/MFA hand-off contract.  The
    validator deliberately permits adjacent-token overlap, but starts and ends
    must each be monotonic and every duration must be positive.
    """
    if not tokens_path.exists():
        raise FileNotFoundError(f"Missing CTC tokens: {tokens_path}")

    entries: list[dict] = []
    prev_start = -1.0
    prev_end = -1.0
    for line_no, line in enumerate(
            tokens_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{tokens_path.name}:{line_no}: invalid JSON: {exc}"
            ) from exc
        word = entry.get("word")
        if not isinstance(word, str) or not word.strip():
            raise ValueError(
                f"{tokens_path.name}:{line_no}: missing/non-string word"
            )
        try:
            start = float(entry["start_s"])
            end = float(entry["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{tokens_path.name}:{line_no}: invalid start_s/end_s"
            ) from exc
        if start < 0 or end <= start:
            raise ValueError(
                f"{tokens_path.name}:{line_no}: invalid interval {start}..{end}"
            )
        if start + 1e-6 < prev_start or end + 1e-6 < prev_end:
            raise ValueError(
                f"{tokens_path.name}:{line_no}: non-monotonic interval"
            )
        prev_start, prev_end = start, end
        entries.append(entry)

    if not entries:
        raise ValueError(f"No CTC tokens in {tokens_path}")
    return entries


def read_ctc_textgrid_words(textgrid_path: Path) -> list[str]:
    """Read the lexical words tier from an NVASR CTC TextGrid."""
    if not textgrid_path.exists():
        raise FileNotFoundError(f"Missing CTC TextGrid: {textgrid_path}")
    content = textgrid_path.read_text(encoding="utf-8-sig")
    words_match = _re.search(
        r'^\s*name\s*=\s*"words"\s*$', content, _re.MULTILINE)
    if words_match is None:
        raise ValueError(f"Missing words tier in {textgrid_path}")
    pauses_match = _re.search(
        r'^\s*name\s*=\s*"pauses"\s*$',
        content[words_match.end():],
        _re.MULTILINE,
    )
    end = (words_match.end() + pauses_match.start()
           if pauses_match is not None else len(content))
    segment = content[words_match.end():end]
    words = [m.group(1).replace('""', '"') for m in _re.finditer(
        r'^\s*text\s*=\s*"(.*)"\s*$', segment, _re.MULTILINE
    ) if m.group(1)]
    if not words:
        raise ValueError(f"Empty words tier in {textgrid_path}")
    return words


def rebuild_lab_from_tokens(tokens_path: Path, lab_path: Path) -> list[str]:
    """Atomically rebuild an MFA lab transcript from validated CTC words."""
    words = [entry["word"].strip()
             for entry in load_ctc_token_entries(tokens_path)]
    lab_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = lab_path.with_name(f".{lab_path.name}.tmp")
    tmp_path.write_text(" ".join(words) + "\n", encoding="utf-8")
    tmp_path.replace(lab_path)
    return words


def validate_ctc_transcript_bundle(ctc_dir: Path, stem: str) -> list[str]:
    """Return contract violations for lab/tokens/CTC words of stem."""
    lab_path = ctc_dir / f"{stem}.lab"
    tokens_path = ctc_dir / f"{stem}_tokens.jsonl"
    textgrid_path = ctc_dir / f"{stem}.TextGrid"
    errors: list[str] = []

    try:
        token_words = [entry["word"].strip()
                       for entry in load_ctc_token_entries(tokens_path)]
    except (OSError, ValueError) as exc:
        return [str(exc)]

    if not lab_path.exists():
        errors.append(f"Missing MFA transcript: {lab_path}")
        lab_words: list[str] = []
    else:
        try:
            lab_words = lab_path.read_text(
                encoding="utf-8-sig").strip().split()
        except OSError as exc:
            errors.append(f"Cannot read {lab_path}: {exc}")
            lab_words = []
        if lab_words != token_words:
            errors.append(
                f"lab/tokens mismatch ({len(lab_words)} != {len(token_words)})"
            )

    try:
        tg_words = read_ctc_textgrid_words(textgrid_path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    else:
        if tg_words != token_words:
            errors.append(
                f"TextGrid/tokens mismatch ({len(tg_words)} != {len(token_words)})"
            )

    contaminated = [word for word in lab_words
                    if _MIXED_PINYIN_TONE_RE.fullmatch(word)]
    if contaminated:
        errors.append(
            "tone digit converted to CJK numeral: " + ", ".join(contaminated[:5])
        )
    return errors


# ── English MFA phone classification ─────────────────────────────

_ENGLISH_VOWELS: set[str] = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
    'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX',
}
_ENGLISH_CONSONANTS: set[str] = {
    'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'ENG', 'F', 'G',
    'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 'S',
    'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH',
}
_ENGLISH_SILENCE_PHONES: set[str] = {'sil', 'sp', 'spn', '<eps>'}


def is_english_phone(phone: str) -> bool:
    """Check if *phone* is an MFA English phone (ARPABET-based, with optional stress)."""
    p = phone.strip().rstrip('012')
    return p in _ENGLISH_VOWELS or p in _ENGLISH_CONSONANTS or p in _ENGLISH_SILENCE_PHONES


def is_english_vowel_phone(phone: str) -> bool:
    """Check if *phone* is an English vowel (MFA ARPABET-based)."""
    p = phone.strip().rstrip('012')
    return p in _ENGLISH_VOWELS


def is_english_consonant_phone(phone: str) -> bool:
    """Check if *phone* is an English consonant (MFA ARPABET-based)."""
    p = phone.strip().rstrip('012')
    return p in _ENGLISH_CONSONANTS


# ── English IPA -> ARPABET mapping (legacy compat) ─────────────────
# When using the ARPABET-native english_us_arpa model, the mapping is
# a no-op (ARPABET phones pass through unchanged).  The table is kept
# for backward compatibility with english_mfa (IPA-based) output.
# Vowels default to stress level 0; stress can be overridden with a
# lexicon lookup in postprocessing.

_EN_IPA_TO_ARPABET: dict[str, str] = {
    # ── Stops ──
    "p": "P", "pʰ": "P", "pʲ": "P", "pʷ": "P",
    "b": "B", "bʲ": "B",
    "t": "T", "tʰ": "T", "tʲ": "T", "tʷ": "T", "t̪": "T",
    "d": "D", "dʲ": "D", "d̪": "D",
    "k": "K", "kʰ": "K", "kʷ": "K", "kp": "K",
    "ɡ": "G", "g": "G",
    "ʔ": "",  # glottal stop -> dropped
    "c": "K", "cʰ": "K", "cʷ": "K",
    "ɟ": "G", "ɟʷ": "G",
    "ʈ": "T", "ʈʰ": "T", "ʈʲ": "T", "ʈʷ": "T",

    # ── Affricates ──
    "tʃ": "CH",
    "dʒ": "JH",

    # ── Fricatives ──
    "f": "F", "fʲ": "F", "fʷ": "F",
    "v": "V", "vʲ": "V",
    "θ": "TH",
    "ð": "DH",
    "s": "S",
    "z": "Z",
    "ʃ": "SH",
    "ʒ": "ZH",
    "h": "HH",
    "ç": "HH",
    "ɦ": "HH",

    # ── Nasals ──
    "m": "M", "mʲ": "M", "m̩": "M",
    "n": "N", "n̩": "N",
    "ŋ": "NG",
    "ɱ": "M",
    "ɲ": "N",
    "ɳ": "N",

    # ── Liquids ──
    "l": "L",
    "ɫ": "L",
    "ɹ": "R",
    "ɻ": "R",
    "ɾ": "R",

    # ── Glides ──
    "j": "Y",
    "w": "W",
    "ʋ": "W",
    "ʎ": "Y",

    # ── Vowels (monophthongs) -> stress-0 by default ──
    "i": "IY0", "iː": "IY0",
    "ɪ": "IH0",
    "e": "EY0", "eː": "EY0",
    "ɛ": "EH0", "ɛ̃": "EH0",
    "æ": "AE0",
    "a": "AA0", "aː": "AA0",
    "ɑ": "AA0",
    "ɒ": "AA0",
    "ɔ": "AO0",
    "o": "OW0", "oː": "OW0",
    "ʊ": "UH0",
    "u": "UW0", "uː": "UW0",
    "ə": "AH0",
    "ʌ": "AH0",
    "ɜ": "ER0",
    "ɝ": "ER0",
    "ɐ": "AH0",
    "ɨ": "IH0",
    "ʉ": "UW0", "ʉː": "UW0",
    "ɤ": "AH0",

    # ── Diphthongs ──
    "aj": "AY0",
    "aw": "AW0",
    "ɔj": "OY0",
    "ej": "EY0",
    "ow": "OW0",
    "əw": "OW0",
}

# Tracks unexpected IPA→ARPABET mapping hits when using the ARPA model.
# With english_us_arpa, this set should remain empty (all phones are no-op pass-through).
_en_ipa_mapping_hits: set[str] = set()


def en_ipa_to_arpabet(phone: str) -> str:
    """Map a single MFA English IPA phone to ARPABET.

    Returns the ARPABET equivalent, or the original phone unchanged
    if it cannot be mapped (silence / spn / unrecognised).

    With the ARPABET-native english_us_arpa model, this is normally
    a no-op.  When an IPA→ARPABET conversion actually fires, the
    mapping is recorded to :data:`_en_ipa_mapping_hits` for later
    diagnostics.
    """
    p = phone.strip()
    if not p:
        return p
    # Already ARPABET or silence — pass through
    if p in ("sil", "sp", "spn", "<eps>"):
        return p
    if p.startswith("en:"):
        inner = p[3:]
        mapped = _EN_IPA_TO_ARPABET.get(inner, inner)
        if mapped != inner:
            _en_ipa_mapping_hits.add(f"{inner}→{mapped}")
        if mapped == "":
            return ""  # explicitly dropped (glottal stop)
        return f"en:{mapped}" if mapped else f"en:{inner}"
    mapped = _EN_IPA_TO_ARPABET.get(p, p)
    if mapped != p:
        _en_ipa_mapping_hits.add(f"{p}→{mapped}")
    if mapped == "":
        return ""  # explicitly dropped (glottal stop)
    return mapped


def report_en_ipa_mappings() -> int:
    """Log IPA→ARPABET conversion hits and return the count.

    When the ARPABET-native model is working correctly the count is 0.
    Non-zero means some IPA phones were unexpectedly converted.
    """
    if _en_ipa_mapping_hits:
        print(f"  IPA→ARPABET mapping triggered ({len(_en_ipa_mapping_hits)} unique): "
              f"{', '.join(sorted(_en_ipa_mapping_hits)[:20])}"
              f"{'…' if len(_en_ipa_mapping_hits) > 20 else ''}")
    return len(_en_ipa_mapping_hits)


# ── Sequence alignment (Needleman-Wunsch) ──────────────────────────

def align_sequences(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Needleman-Wunsch global alignment of two token sequences.

    Returns list of (index_in_a, index_in_b) for matched pairs.
    Unmatched tokens are omitted.  Used by _snap_to_ctc and stress mapping.
    """
    import numpy as _np
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    dp = _np.full((n + 1, m + 1), 9999, dtype=_np.int32)
    dp[0, 0] = 0
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if a[i - 1] == b[j - 1] else 1
        if dp[i, j] == dp[i - 1, j - 1] + cost:
            if cost == 0:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ── CMUdict stress lookup ──────────────────────────────────────────

import threading as _threading

_cmudict: dict[str, list[str]] | None = None
_cmudict_lock = _threading.Lock()


def _load_cmudict() -> dict[str, list[str]]:
    """Lazy-load CMU Pronouncing Dictionary (ARPABET with stress)."""
    global _cmudict
    if _cmudict is not None:
        return _cmudict

    with _cmudict_lock:
        if _cmudict is not None:
            return _cmudict

        result: dict[str, list[str]] = {}
        # Try nltk first
        try:
            import nltk
            entries = nltk.corpus.cmudict.entries()
            for word, phones in entries:
                word_lower = word.lower()
                if word_lower not in result:
                    result[word_lower] = list(phones)
            if result:
                import sys
                print(f"  CMUdict loaded via nltk: {len(result)} entries", file=sys.stderr)
        except Exception:
            pass

        # Fallback: try local cmudict file
        if not result:
            for path in [
                "dict/cmudict.dict",
                "/usr/share/cmudict/cmudict.dict",
            ]:
                try:
                    p = Path(__file__).parent.parent / path if not Path(path).is_absolute() else Path(path)
                    if p.exists():
                        for line in p.read_text(encoding="latin-1").splitlines():
                            line = line.strip()
                            if not line or line.startswith(";;;"):
                                continue
                            parts = line.split()
                            if len(parts) >= 2:
                                word = parts[0].split("(")[0].lower()
                                phones = [p for p in parts[1:] if p]
                                if word not in result:
                                    result[word] = phones
                        import sys
                        print(f"  CMUdict loaded from {p}: {len(result)} entries", file=sys.stderr)
                        break
                except Exception:
                    pass

        if not result:
            import sys
            print("  CMUdict not available — English ARPABET stress will default to 0", file=sys.stderr)

        _cmudict = result
        return _cmudict


def apply_arpabet_stress(arpabet_phones: list[str], word: str) -> list[str]:
    """Apply CMUdict stress markers to unstressed ARPABET phones.

    Looks up *word* in CMUdict to get the canonical ARPABET pronunciation
    with stress (e.g. HH AH0 L OW1).  Maps stress digits onto the
    aligned unstressed phones by position.

    When CMUdict is unavailable or the word is unknown, returns the
    input phones unchanged.
    """
    if not arpabet_phones:
        return arpabet_phones

    cmu = _load_cmudict()
    if not cmu:
        return arpabet_phones  # CMUdict not available — stress stays 0

    canonical = cmu.get(word.lower())
    if not canonical:
        return arpabet_phones

    # Extract stress pattern from canonical: [0=unstressed, 1=primary, 2=secondary]
    stress_pattern = []
    for p in canonical:
        s = p[-1]
        stress_pattern.append(int(s) if s in "012" else 0)

    # Map stress to aligned phones (without stress digits)
    aligned = [p.rstrip("012") for p in arpabet_phones]
    canonical_no_stress = [p.rstrip("012") for p in canonical]

    # Align aligned phones to canonical via Needleman-Wunsch
    pairs = align_sequences(aligned, canonical_no_stress)

    # Build stress mapping: aligned_pos -> canonical_stress
    stress_map: dict[int, int] = {}
    for ai, ci in pairs:
        stress_map[ai] = stress_pattern[ci]

    # Apply stress (only to vowels; ARPABET consonants carry no stress digit)
    result = list(arpabet_phones)
    for idx, stress in stress_map.items():
        base = result[idx].rstrip("012")
        if base in _ENGLISH_VOWELS:
            result[idx] = f"{base}{stress}"

    return result


def extract_word_chars(text: str) -> list[str]:
    """Split *text* into word-like units (CJK chars, alpha groups, punct)."""
    result: list[str] = []
    buf: str = ""
    for c in text:
        if is_cjk(c):
            if buf:
                result.append(buf)
                buf = ""
            result.append(c)
        elif c.isalpha() or c == '-':
            buf += c
        elif c.isdigit():
            buf += c
        else:
            if buf:
                result.append(buf)
                buf = ""
            if not c.isspace():
                result.append(c)
    if buf:
        result.append(buf)
    return result


# ═══════════════════════════════════════════════════════════════════
# Model tree digest — Case 99 provenance (R5)
# ═══════════════════════════════════════════════════════════════════

def compute_model_tree_digest(model_dir: Path) -> tuple[str, list[dict]]:
    """Compute a deterministic content digest of an ASR model directory.

    Walks every regular file in *model_dir* (sorted by relative path),
    records its size and SHA-256, and feeds both the relative path and
    content hash into a rolling tree digest.

    Symlinks, non-regular files, and path-escape attempts are rejected.

    Returns:
        (tree_digest_hex, file_manifest) where *file_manifest* is a list
        of ``{"relpath": str, "size": int, "sha256": str}`` sorted by
        ``relpath``.
    """
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError(f"model tree root is not a regular directory: {model_dir}")
    digest = hashlib.sha256()
    file_manifest: list[dict] = []
    for child in sorted(p for p in model_dir.rglob("*") if p.is_file()):
        if child.is_symlink():
            raise ValueError(f"symlink not allowed in model tree: {child}")
        rel = child.relative_to(model_dir).as_posix()
        if rel.startswith("..") or rel.startswith("/"):
            raise ValueError(f"model tree path escape: {rel}")
        file_sha = _sha256_file(child)
        file_manifest.append({
            "relpath": rel,
            "size": child.stat().st_size,
            "sha256": file_sha,
        })
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), file_manifest


# ═══════════════════════════════════════════════════════════════════
# CTC run receipt — Case 99 provenance (R5)
# ═══════════════════════════════════════════════════════════════════

_CTC_RUN_RECEIPT_SCHEMA = "ctc-run-receipt-v1"
_SHARD_RECEIPT_SCHEMA = "ctc-shard-receipt-v1"


def _stable_json_digest(value: object) -> str:
    """Deterministic SHA-256 of a JSON-serialisable value."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


def write_ctc_run_receipt(
    output_dir: Path,
    actual_argv: list[str],
    asr_python: str,
    model_path: Path,
    model_tree_digest: str,
    model_file_manifest: list[dict],
    dict_path: Path,
    dict_digest: str,
    input_stems: list[str],
    output_stems: list[str],
) -> dict:
    """Atomically write a CTC run receipt binding provenance evidence.

    The receipt proves that a specific CTC process loaded a specific model
    tree, dictionary, and input stem set, and produced an exact output stem
    set.  It is the on-disk counterpart of the prepare-time frozen model
    identity.

    Returns the receipt dict that was written.
    """
    import time as _time  # local to avoid shadowing
    input_sorted = sorted(input_stems)
    output_sorted = sorted(output_stems)
    receipt: dict = {
        "schema": _CTC_RUN_RECEIPT_SCHEMA,
        "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "argv": list(actual_argv),
        "asr_python": str(asr_python),
        "model": {
            "path": str(model_path.resolve()),
            "tree_digest": model_tree_digest,
            "files": model_file_manifest,
        },
        "dictionary": {
            "path": str(dict_path.resolve()),
            "digest": dict_digest,
        },
        "input_stems": input_sorted,
        "input_stems_digest": _stable_json_digest(input_sorted),
        "output_stems": output_sorted,
        "output_stems_digest": _stable_json_digest(output_sorted),
    }
    receipt_path = output_dir / ".ctc_run_receipt.json"
    tmp = receipt_path.with_name(".ctc_run_receipt.json.tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, receipt_path)
    return receipt


def write_ctc_shard_receipt(
    shard_dir: Path,
    gpu_id: int,
    model_tree_digest: str,
    dict_digest: str,
    stems: list[str],
    parent_argv: list[str],
) -> dict:
    """Atomically write a per-GPU-shard receipt for all-GPU provenance.

    Every shard must record the same model/dict identity so the parent
    can cross-check before merging artifacts.
    """
    stems_sorted = sorted(stems)
    receipt: dict = {
        "schema": _SHARD_RECEIPT_SCHEMA,
        "gpu_id": gpu_id,
        "model_tree_digest": model_tree_digest,
        "dict_digest": dict_digest,
        "stems": stems_sorted,
        "stems_digest": _stable_json_digest(stems_sorted),
        "parent_argv": list(parent_argv),
    }
    receipt_path = shard_dir / ".shard_receipt.json"
    tmp = receipt_path.with_name(".shard_receipt.json.tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, receipt_path)
    return receipt


def validate_ctc_receipts_same_identity(
    shard_receipts: list[dict],
    parent_model_tree_digest: str,
    parent_dict_digest: str,
) -> list[str]:
    """Check that all shard receipts share the same model/dict identity.

    Returns a list of error strings (empty = all consistent).
    """
    errors: list[str] = []
    for i, r in enumerate(shard_receipts):
        if r.get("model_tree_digest") != parent_model_tree_digest:
            errors.append(
                f"shard {i} model_tree_digest {r.get('model_tree_digest')!r} "
                f"!= parent {parent_model_tree_digest!r}"
            )
        if r.get("dict_digest") != parent_dict_digest:
            errors.append(
                f"shard {i} dict_digest {r.get('dict_digest')!r} "
                f"!= parent {parent_dict_digest!r}"
            )
    return errors


# ═══════════════════════════════════════════════════════════════════
# Strict MFA TextGrid validator — Cases 76/83 (R7)
# ═══════════════════════════════════════════════════════════════════

_TG_TIMING_TOLERANCE_S = 0.003  # matches TIMING_TOLERANCE_S in prepare/v4 verifier


def validate_strict_mfa_textgrid(
    path: Path,
    wav_duration_s: float | None = None,
) -> list[str]:
    """Validate a long-format MFA TextGrid with structural and domain checks.

    This replaces the old substring match (``"words" in content``) with a
    proper grammar-aware parser.  Every violation is returned as a
    diagnostic string; an empty list means the file is structurally valid.

    Checks performed:
      - Grammar header (``File type = "ooTextFile"``, ``Object class = "TextGrid"``)
      - Finite, strictly-increasing global xmin/xmax
      - ``tiers? <exists>`` header and matching tier count
      - Unique tier names, each ``"IntervalTier"`` with in-domain xmin/xmax
      - Interval count matches declaration
      - Every interval is finite, positive-duration, monotonic within its tier
      - WAV domain (when *wav_duration_s* is provided)
      - Expected tiers ``words`` and ``phones`` exist with at least one
        non-empty interval each
    """
    errors: list[str] = []

    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [f"cannot read TextGrid: {exc}"]

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ["empty TextGrid file"]

    i = 0
    n = len(lines)

    def _peek() -> str:
        return lines[i] if i < n else "<eof>"

    def _take(expected: str) -> None:
        nonlocal i
        if i >= n:
            raise ValueError(f"expected {expected!r}, got <eof>")
        if lines[i] != expected:
            raise ValueError(f"expected {expected!r}, got {lines[i]!r}")
        i += 1

    def _pref(prefix: str) -> str:
        nonlocal i
        if i >= n:
            raise ValueError(f"expected line starting with {prefix!r}, got <eof>")
        line = lines[i]
        if not line.startswith(prefix):
            raise ValueError(f"expected line starting with {prefix!r}, got {line!r}")
        i += 1
        return line

    def _num(line: str, key: str) -> float:
        if not line.startswith(key + " = "):
            raise ValueError(f"expected {key!r}, got {line!r}")
        val = float(line.split("=", 1)[1])
        if not math.isfinite(val):
            raise ValueError(f"non-finite {key}: {val}")
        return val

    def _unquote(val: str) -> str:
        val = val.strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        return val.replace('""', '"')

    try:
        # Grammar header
        _take('File type = "ooTextFile"')
        _take('Object class = "TextGrid"')

        # Global domain
        gxmin = _num(_pref("xmin = "), "xmin")
        gxmax = _num(_pref("xmax = "), "xmax")
        if gxmax <= gxmin:
            errors.append(f"global xmax {gxmax} <= xmin {gxmin}")

        _take("tiers? <exists>")
        size_line = _pref("size = ")
        declared_tiers = int(size_line.split("=", 1)[1])
        if declared_tiers < 1:
            errors.append(f"declared {declared_tiers} tiers, need >= 1")

        _take("item []:")

        tier_names: list[str] = []
        tiers_data: list[dict] = []

        for ti in range(1, declared_tiers + 1):
            _take(f"item [{ti}]:")
            _take('class = "IntervalTier"')

            name_line = _pref("name = ")
            name = _unquote(name_line.split("=", 1)[1].strip())
            if name in tier_names:
                errors.append(f"duplicate tier name: {name!r}")
            tier_names.append(name)

            txmin = _num(_pref("xmin = "), "xmin")
            txmax = _num(_pref("xmax = "), "xmax")
            if not (gxmin - _TG_TIMING_TOLERANCE_S <= txmin <= gxmax + _TG_TIMING_TOLERANCE_S):
                errors.append(
                    f"tier {name!r} xmin {txmin} outside global "
                    f"[{gxmin}, {gxmax}]"
                )
            if not (gxmin - _TG_TIMING_TOLERANCE_S <= txmax <= gxmax + _TG_TIMING_TOLERANCE_S):
                errors.append(
                    f"tier {name!r} xmax {txmax} outside global "
                    f"[{gxmin}, {gxmax}]"
                )

            iv_size_line = _pref("intervals: size = ")
            declared_ivs = int(iv_size_line.split("=", 1)[1])
            if declared_ivs < 0:
                errors.append(f"tier {name!r}: negative interval count {declared_ivs}")

            intervals: list[tuple[float, float, str]] = []
            for ji in range(1, declared_ivs + 1):
                _take(f"intervals [{ji}]:")
                iv_xmin = _num(_pref("xmin = "), "xmin")
                iv_xmax = _num(_pref("xmax = "), "xmax")
                text_line = _pref("text = ")
                iv_text = _unquote(text_line.split("=", 1)[1].strip())
                if not math.isfinite(iv_xmin) or not math.isfinite(iv_xmax):
                    errors.append(
                        f"tier {name!r} interval {ji}: non-finite boundary"
                    )
                if iv_xmax <= iv_xmin:
                    errors.append(
                        f"tier {name!r} interval {ji}: xmax {iv_xmax} <= xmin {iv_xmin}"
                    )
                if intervals:
                    prev_end = intervals[-1][1]
                    if iv_xmin + _TG_TIMING_TOLERANCE_S < prev_end:
                        errors.append(
                            f"tier {name!r} interval {ji}: xmin {iv_xmin} "
                            f"< previous xmax {prev_end} (non-monotonic)"
                        )
                intervals.append((iv_xmin, iv_xmax, iv_text))

            tiers_data.append({
                "name": name,
                "xmin": txmin,
                "xmax": txmax,
                "intervals": intervals,
            })

        # No trailing content
        if i != n:
            errors.append(f"unexpected trailing content at line {i}: {_peek()!r}")

        # Expected tiers: words and phones with non-empty intervals
        tier_map = {td["name"]: td for td in tiers_data}
        for required in ("words", "phones"):
            if required not in tier_map:
                errors.append(f"missing required tier: {required!r}")
                continue
            non_empty = [iv for iv in tier_map[required]["intervals"] if iv[2].strip()]
            if not non_empty:
                errors.append(f"tier {required!r} has no non-empty intervals")

        # WAV domain check
        if wav_duration_s is not None:
            if gxmax > wav_duration_s + _TG_TIMING_TOLERANCE_S:
                errors.append(
                    f"TextGrid xmax {gxmax} exceeds WAV duration {wav_duration_s}"
                )
            for td in tiers_data:
                for iv in td["intervals"]:
                    if iv[1] > wav_duration_s + _TG_TIMING_TOLERANCE_S:
                        errors.append(
                            f"tier {td['name']!r} interval end {iv[1]} "
                            f"exceeds WAV duration {wav_duration_s}"
                        )
                    if iv[0] < -_TG_TIMING_TOLERANCE_S:
                        errors.append(
                            f"tier {td['name']!r} interval start {iv[0]} negative"
                        )

    except ValueError as exc:
        errors.append(str(exc))
    except OSError as exc:
        errors.append(f"I/O error: {exc}")

    return errors


# ═══════════════════════════════════════════════════════════════════
# Pipeline run receipt — R1/R7 denominator & output path contract
# ═══════════════════════════════════════════════════════════════════

_PIPELINE_RECEIPT_SCHEMA = "pipeline-run-receipt-v1"


def make_pipeline_run_id() -> str:
    """Generate a unique, sortable run ID for output isolation."""
    import time as _time
    return f"{_time.strftime('%Y%m%dT%H%M%SZ', _time.gmtime())}_{os.getpid()}"


def write_pipeline_run_receipt(
    output_dir: Path,
    run_id: str,
    mode: str,
    route: list[str],
    input_stems: list[str],
    output_stems: list[str],
    filtered_stems: list[str],
    failed_steps: list[str],
    actual_output_path: Path,
    actual_filtered_path: Path,
    extra: dict | None = None,
) -> dict:
    """Atomically write a pipeline run receipt binding all output paths.

    The receipt proves what stems went in, what came out, what was filtered,
    and what failed — forming the denominator-conservation proof (R1).

    Returns the receipt dict.
    """
    import time as _time
    receipt: dict = {
        "schema": _PIPELINE_RECEIPT_SCHEMA,
        "run_id": run_id,
        "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "mode": mode,
        "route": route,
        "input_stems": sorted(input_stems),
        "input_count": len(input_stems),
        "output_stems": sorted(output_stems),
        "output_count": len(output_stems),
        "filtered_stems": sorted(filtered_stems),
        "filtered_count": len(filtered_stems),
        "failed_steps": failed_steps,
        "paths": {
            "output": str(actual_output_path),
            "filtered": str(actual_filtered_path),
        },
    }
    if extra:
        receipt["extra"] = extra
    receipt_path = output_dir / ".pipeline_run_receipt.json"
    tmp = receipt_path.with_name(".pipeline_run_receipt.json.tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, receipt_path)
    return receipt


# ═══════════════════════════════════════════════════════════════════
# Source-denominator accounting contract (v2)
# ═══════════════════════════════════════════════════════════════════

# v1 receipts contain only counts inferred from whichever directories happened
# to exist at the end of a run.  They are deliberately not upgraded here: a
# complete source denominator requires frozen stem evidence and explicit
# exclusions.  Consumers that need a resumable/strict receipt must use v2.
PIPELINE_ACCOUNTING_SCHEMA = "pipeline-run-receipt-v2"
PIPELINE_ACCOUNTING_RECEIPT_NAME = ".pipeline_run_receipt_v2.json"
_ACCOUNTING_REASONS_DISALLOWED = {
    "filtered", "quality_rejection", "processed_quality_rejection",
}


def _accounting_stems(value: object, field: str) -> list[str]:
    """Normalize and validate a stem sequence without silently deduplicating."""
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field} must be a stem sequence")
    stems = list(value)
    if any(not isinstance(stem, str) or not stem or Path(stem).name != stem
           for stem in stems):
        raise ValueError(f"{field} contains invalid stem")
    if len(stems) != len(set(stems)):
        raise ValueError(f"{field} contains duplicate stems")
    return sorted(stems)


def _accounting_digest(stems: list[str]) -> str:
    return _stable_json_digest(stems)


def _normalize_exclusions(exclusions: object) -> list[dict[str, str]]:
    """Accept mapping, ``(stem, reason)`` pairs, or row dictionaries."""
    if exclusions is None:
        return []
    rows: list[dict[str, str]] = []
    if isinstance(exclusions, dict):
        iterable = exclusions.items()
    else:
        iterable = exclusions
    try:
        for item in iterable:  # type: ignore[union-attr]
            if isinstance(item, dict):
                stem, reason = item.get("stem"), item.get("reason")
                extra = set(item) - {"stem", "reason"}
                if extra:
                    raise ValueError(f"exclusion has unknown fields: {sorted(extra)}")
            else:
                try:
                    stem, reason = item
                except (TypeError, ValueError) as exc:
                    raise ValueError("exclusions must contain stem/reason pairs") from exc
            if (not isinstance(stem, str) or not stem
                    or Path(stem).name != stem):
                raise ValueError(f"invalid exclusion stem: {stem!r}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"invalid exclusion reason for {stem!r}")
            reason = reason.strip()
            if reason in _ACCOUNTING_REASONS_DISALLOWED:
                raise ValueError(
                    f"processed quality rejection belongs in filtered, not exclusions: {stem}")
            rows.append({"stem": stem, "reason": reason})
    except TypeError as exc:
        raise ValueError("exclusions must be a mapping or sequence") from exc
    if len({row["stem"] for row in rows}) != len(rows):
        raise ValueError("exclusions contain duplicate stems")
    return sorted(rows, key=lambda row: row["stem"])


def _normalize_shards(shards: object) -> list[dict[str, object]]:
    if not isinstance(shards, list):
        raise ValueError("shards must be a list")
    normalized: list[dict[str, object]] = []
    ids: set[str] = set()
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"shard {index} must be an object")
        shard_id = shard.get("shard_id", str(index))
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError(f"shard {index} has invalid shard_id")
        if shard_id in ids:
            raise ValueError(f"duplicate shard_id: {shard_id}")
        ids.add(shard_id)
        stems = _accounting_stems(shard.get("stems"), f"shard {index}")
        normalized.append({
            "shard_id": shard_id,
            "count": len(stems),
            "stems": stems,
            "stems_digest": _accounting_digest(stems),
        })
    return normalized


def classify_receipt_accounting(receipt: dict) -> dict[str, object]:
    """Return derived loss/health fields after validating a v2 receipt.

    ``silent_loss`` is the number of source stems that are neither explicitly
    excluded nor present in output/filtered.  A healthy run has zero silent
    loss and no accounting violations.
    """
    errors = validate_pipeline_accounting_receipt(receipt, check_derived=False)
    source_raw = receipt.get("source")
    eligible_raw = receipt.get("eligible")
    output_raw = receipt.get("output")
    filtered_raw = receipt.get("filtered")
    source = set(source_raw.get("stems", [])) if isinstance(source_raw, dict) else set()
    eligible = (set(eligible_raw.get("stems", []))
                if isinstance(eligible_raw, dict) else set())
    output = set(output_raw.get("stems", [])) if isinstance(output_raw, dict) else set()
    filtered = (set(filtered_raw.get("stems", []))
                if isinstance(filtered_raw, dict) else set())
    exclusions_raw = receipt.get("exclusions", [])
    excluded = {row.get("stem") for row in exclusions_raw
                if isinstance(row, dict)}
    silent = len(source - eligible - excluded)
    health = "healthy" if not errors and silent == 0 else "accounting_error"
    return {
        "silent_loss": silent,
        "run_health": health,
        "source_count": len(source),
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
        "output_count": len(output),
        "filtered_count": len(filtered),
    }


def make_pipeline_accounting_receipt(
    source_stems: list[str],
    eligible_stems: list[str],
    exclusions: object,
    output_stems: list[str],
    filtered_stems: list[str],
    *,
    run_id: str = "",
    mode: str = "",
    route: list[str] | None = None,
    paths: dict[str, str] | None = None,
    shards: list[dict] | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a machine-readable v2 source-denominator receipt.

    Stems are frozen evidence, not inferred from counts.  Validation is run
    before returning so callers cannot write an internally inconsistent
    receipt.  ``missing_reference`` is an ordinary explicit exclusion reason;
    processed quality failures must be represented by ``filtered_stems``.
    """
    source = _accounting_stems(source_stems, "source_stems")
    eligible = _accounting_stems(eligible_stems, "eligible_stems")
    output = _accounting_stems(output_stems, "output_stems")
    filtered = _accounting_stems(filtered_stems, "filtered_stems")
    exclusion_rows = _normalize_exclusions(exclusions)
    exclusions_by_reason: dict[str, list[str]] = {}
    for row in exclusion_rows:
        exclusions_by_reason.setdefault(row["reason"], []).append(row["stem"])
    def bucket(stems: list[str]) -> dict[str, object]:
        return {"count": len(stems), "stems": stems,
                "stems_digest": _accounting_digest(stems)}
    receipt: dict[str, object] = {
        "schema": PIPELINE_ACCOUNTING_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "route": list(route or []),
        "source": bucket(source),
        "eligible": bucket(eligible),
        "exclusions": exclusion_rows,
        "exclusions_by_reason": {
            reason: sorted(stems)
            for reason, stems in sorted(exclusions_by_reason.items())
        },
        "output": bucket(output),
        "filtered": bucket(filtered),
        "paths": dict(paths or {}),
    }
    if shards is not None:
        receipt["shards"] = _normalize_shards(shards)
    if extra:
        receipt["extra"] = dict(extra)
    derived = classify_receipt_accounting(receipt)
    receipt["derived"] = derived
    # Flat count aliases make the contract convenient for shell/audit callers.
    receipt.update(derived)
    errors = validate_pipeline_accounting_receipt(receipt)
    if errors:
        raise ValueError("invalid accounting receipt: " + "; ".join(errors))
    return receipt


def validate_pipeline_accounting_receipt(
    receipt: dict, *, check_derived: bool = True,
    expected_shard_stems: list[str] | None = None,
) -> list[str]:
    """Return diagnostics for a v2 receipt (empty means valid).

    Validation rejects duplicate/cross-shard stems, overlap between accounting
    buckets, stale digest evidence, and any source stem that disappears.  A v1
    receipt is always rejected; callers may inspect it separately but must not
    promote it by inference.
    """
    errors: list[str] = []
    if not isinstance(receipt, dict):
        return ["receipt must be an object"]
    if receipt.get("schema") != PIPELINE_ACCOUNTING_SCHEMA:
        return ["receipt schema is not pipeline-run-receipt-v2"]
    buckets: dict[str, set[str]] = {}
    for name in ("source", "eligible", "output", "filtered"):
        raw = receipt.get(name)
        if not isinstance(raw, dict):
            errors.append(f"{name} bucket missing")
            continue
        try:
            stems = _accounting_stems(raw.get("stems"), name)
        except ValueError as exc:
            errors.append(str(exc)); continue
        buckets[name] = set(stems)
        if raw.get("count") != len(stems):
            errors.append(f"{name} count mismatch")
        if raw.get("stems_digest") != _accounting_digest(stems):
            errors.append(f"{name} stems digest mismatch")
    exclusions = receipt.get("exclusions")
    try:
        rows = _normalize_exclusions(exclusions)
    except ValueError as exc:
        errors.append(str(exc)); rows = []
    excluded = {row["stem"] for row in rows}
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["reason"], []).append(row["stem"])
    grouped = {reason: sorted(stems)
               for reason, stems in sorted(grouped.items())}
    if "exclusions_by_reason" in receipt:
        raw_grouped = receipt.get("exclusions_by_reason")
        if raw_grouped != grouped:
            errors.append("exclusions_by_reason mismatch")
    if "source" in buckets and not excluded <= buckets["source"]:
        errors.append("exclusion stem absent from source")
    if "source" in buckets and "eligible" in buckets:
        if buckets["eligible"] & excluded:
            errors.append("eligible/exclusion overlap")
        if buckets["source"] != buckets["eligible"] | excluded:
            errors.append("source != eligible disjoint-union exclusions")
    if "eligible" in buckets and "output" in buckets and "filtered" in buckets:
        if buckets["output"] & buckets["filtered"]:
            errors.append("output/filtered overlap")
        if buckets["output"] | buckets["filtered"] != buckets["eligible"]:
            errors.append("eligible != output disjoint-union filtered")
    shards = receipt.get("shards")
    if shards is not None:
        if not isinstance(shards, list):
            errors.append("shards must be a list")
        else:
            union: set[str] = set()
            shard_ids: set[str] = set()
            for index, shard in enumerate(shards):
                if not isinstance(shard, dict):
                    errors.append(f"shard {index} must be an object"); continue
                try:
                    shard_stems = set(_accounting_stems(shard.get("stems"),
                                                        f"shard {index}"))
                except ValueError as exc:
                    errors.append(str(exc)); continue
                shard_id = shard.get("shard_id", str(index))
                if not isinstance(shard_id, str) or not shard_id:
                    errors.append(f"shard {index} has invalid shard_id")
                elif shard_id in shard_ids:
                    errors.append(f"duplicate shard_id: {shard_id}")
                else:
                    shard_ids.add(shard_id)
                duplicate = union & shard_stems
                if duplicate:
                    errors.append(f"cross-shard duplicate stems: {sorted(duplicate)}")
                union |= shard_stems
                if "count" not in shard or shard.get("count") != len(shard_stems):
                    errors.append(f"shard {index} count mismatch")
                digest = shard.get("stems_digest")
                if digest != _accounting_digest(sorted(shard_stems)):
                    errors.append(f"shard {index} stems digest mismatch")
            if expected_shard_stems is not None:
                try:
                    expected = set(_accounting_stems(expected_shard_stems,
                                                     "expected_shard_stems"))
                except ValueError as exc:
                    errors.append(str(exc)); expected = set()
            else:
                expected = buckets.get("eligible", set())
            if union != expected:
                errors.append("shard union does not exactly match expected stems")
    if check_derived:
        expected = classify_receipt_accounting(receipt)
        if receipt.get("derived") != expected:
            errors.append("derived accounting mismatch")
        for key, value in expected.items():
            if receipt.get(key) != value:
                errors.append(f"flat derived field mismatch: {key}")
    return errors


def write_pipeline_accounting_receipt(
    path: Path, receipt: dict | None = None, **kwargs: object,
) -> dict:
    """Atomically write a v2 receipt and return its canonical payload.

    ``path`` may be a receipt filename or an output directory.  Passing no
    *receipt* builds one from the keyword arguments accepted by
    :func:`make_pipeline_accounting_receipt`.
    """
    if receipt is None:
        receipt = make_pipeline_accounting_receipt(**kwargs)  # type: ignore[arg-type]
    errors = validate_pipeline_accounting_receipt(receipt)
    if errors:
        raise ValueError("invalid accounting receipt: " + "; ".join(errors))
    target = (path / PIPELINE_ACCOUNTING_RECEIPT_NAME
              if path.is_dir() or path.suffix == "" else path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, target)
    return receipt


def read_pipeline_accounting_receipt(
    path: Path, *, allow_legacy: bool = False,
) -> dict:
    """Read a v2 receipt; v1 is rejected unless explicitly requested."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != PIPELINE_ACCOUNTING_SCHEMA:
        if allow_legacy and payload.get("schema") == _PIPELINE_RECEIPT_SCHEMA:
            return payload
        raise ValueError("legacy or unknown receipt schema; v1 cannot be promoted")
    errors = validate_pipeline_accounting_receipt(payload)
    if errors:
        raise ValueError("invalid accounting receipt: " + "; ".join(errors))
    return payload


# Explicit aliases used by lightweight verifiers and downstream runners.
make_receipt_accounting = make_pipeline_accounting_receipt
validate_receipt_accounting = validate_pipeline_accounting_receipt
read_receipt_accounting = read_pipeline_accounting_receipt
write_receipt_accounting = write_pipeline_accounting_receipt
