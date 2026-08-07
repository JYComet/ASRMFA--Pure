#!/usr/bin/env python3
"""
Complete Chinese MFA forced alignment pipeline.

Full mode:  trim -> resample -> prealign -> normalize -> adjust -> validate -> align -> postprocess
ctc_ready:  link -> normalize_punct -> normalize -> normalize_en -> resample -> adjust -> align -> postprocess
            (skip trim/prealign — use pre-existing NVASR CTC output)

Usage:
  # Full pipeline
  python scripts/run_pipeline.py --config configs/my_task.yaml

  # ctc_ready mode — audio already trimmed + NVASR CTC already run
  python scripts/run_pipeline.py --ctc-ready E:/path/to/ctc_output --data-dir E:/path/to/audio

  # Single step / partial run
  python scripts/run_pipeline.py --step align
  python scripts/run_pipeline.py --skip-to align
  python scripts/run_pipeline.py --config my_config.yaml
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    print("ERROR: pyyaml is required. Run: pip install pyyaml")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

# ── Shared pipeline utilities (canonical implementations in pipeline_utils.py) ──
sys.path.insert(0, str(SCRIPTS_DIR))
from pipeline_utils import (
    CTC_NORMALIZATION_MARKER, parse_ctc_normalization_marker,
    find_mfa_python, get_mfa_env,
    build_ctc_presence, build_file_index, count_files_fast, find_wav,
    is_punct, is_word_like,
    load_ctc_token_entries, normalize_reference_numerals,
    read_ctc_textgrid_words, rebuild_lab_from_tokens,
    validate_ctc_transcript_bundle,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-platform path translation — Windows UNC ↔ Linux SMB mount
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-detected mapping: Windows UNC -> Linux mount point
# Built at import time from /proc/mounts
_WIN_UNC_MAP: dict[str, str] = {}

def _detect_smb_mounts() -> dict[str, str]:
    """Parse /proc/mounts for CIFS/SMB mounts; derive UNC->linux mapping."""
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
            # dev = //server/share/path...
            # Extract the share path
            dev_path = dev.replace("//", "", 1)  # server/share/path...
            if dev_path.startswith("192.168."):
                # IP-based — match by network path convention
                # Build possible UNC variants
                server_share = dev_path
                # e.g. "192.168.102.202/Research_TTS/Data/Raw" -> map from multiple patterns
                # Store as-is
                unc = f"//{server_share}"
                mapping[unc] = mnt
                # Also store with backslash variant
                mapping[unc.replace("/", "\\")] = mnt

        # Build RS3621 mapping if we can find //192.168.102.202/Research_TTS
        # This is the SMB server behind the DNS alias RS3621
        for unc, mnt in list(mapping.items()):
            clean = unc.replace("\\", "/")
            if "192.168.102.202/Research_TTS" in clean:
                # Map RS3621 aliases
                parts_after = clean.split("Research_TTS", 1)
                if len(parts_after) > 1:
                    suffix = parts_after[1]
                    mapping[f"//RS3621/Research_TTS{suffix}"] = mnt
                    _win_suf = suffix.replace("/", "\\")
                    mapping[f"\\\\RS3621\\Research_TTS{_win_suf}"] = mnt
    except Exception:
        pass
    return mapping

_WIN_UNC_MAP = _detect_smb_mounts()


def translate_path(path_str: str) -> str:
    """Convert Windows UNC paths to Linux mount paths.

    On Windows, returns the path unchanged.
    On Linux, translates ``\\\\RS3621\\...`` -> ``/mnt/Raw/...`` etc.

    Also handles mixed-separator paths from config files.
    """
    if not path_str or platform.system() == "Windows":
        return path_str

    # Normalise: backslash -> forward slash for comparison
    normalized = path_str.replace("\\", "/")

    # Try exact match first, then longest-prefix match
    for unc_raw, linux_mnt in sorted(_WIN_UNC_MAP.items(),
                                     key=lambda x: -len(x[0])):
        unc_norm = unc_raw.replace("\\", "/")
        if normalized.startswith(unc_norm):
            rest = normalized[len(unc_norm):]
            # Remove leading slash if present (UNC path might have it)
            rest = rest.lstrip("/")
            result = f"{linux_mnt}/{rest}" if rest else linux_mnt
            return result

    return path_str


def resolve_input_path(raw: str, base: Path = PROJECT_ROOT) -> Path:
    """Resolve *raw* path with UNC->Linux translation + relative resolution.

    - Empty / None -> returns base
    - Windows UNC -> translated to Linux mount, then returned as Path
    - Absolute path (already translated) -> returned as-is
    - Relative path -> resolved against *base*
    """
    if not raw:
        return base
    translated = translate_path(raw)
    p = Path(translated)
    if p.is_absolute():
        return p
    return base / p


# ---------------------------------------------------------------------------
# Built-in defaults — task configs only need to specify what differs
# ---------------------------------------------------------------------------

DEFAULT_CFG: dict = {
    "mode": "full",               # "full" | "ctc_ready"
    "workspace": "workspace",
    "data_dir": "data_dir",
    "nvme_cache": "",              # NVMe 音频缓存路径 (空=自动检测 /mnt/nvme3/mfa_audio_cache)
    "txt_suffix": "",
    "audio_dir": "audio",
    "pinyin_dir": "pinyin",
    "aligned_dir": "aligned",
    "output_dir": "output",
    "filtered_dir": "filtered",
    "validate_dir": "validate",
    "temp_dir": "temp",
    "ctc_pretg": "ctc_pretg",
    "ctc_pretg_adj": "ctc_pretg_adj",
    "models_dir": "models/mfa",
    "acoustic_model": "mandarin_mfa",
    "mfa_dict": "dict/mfa_ipa.dict",
    "pinyin_dict": "dict/fullpinyin_enword.dict",
    "python_path": "",
    "keep_16k_audio": True,
    "ctc_ready": {
        "ctc_dir": "",             # pre-existing NVASR CTC output dir
        "text_dir": "",            # optional reference .txt dir (defaults to data_dir)
        "require_all": True,       # skip stems missing any of the 6 CTC files
        "stem_range": None,        # optional [start, end] inclusive range filter
        "stems": None,             # optional explicit list of stems to process
        "stem_prefix": "",         # prepended to numeric stems (e.g., "合成ria_")
    },
    "trim": {
        "max_silence_sec": 1.0,
        "sil_vol_threshold": 0.005,
        "sil_len_threshold": 0.08,
        "normalize_edges": True,
        "target_edge_silence_sec": 0.5,
        "edge_silence_threshold": 0.001,
        "edge_frame_length": 1024,
        "target_sr": None,
        "workers": 8,
    },
    "pad_silence": {
        "enabled": True,              # set false to skip edge silence padding entirely
        "target_edge_silence_sec": 0.5,
        "silence_threshold": 0.001,
        "frame_length": 1024,
        "output_audio": False,        # write padded WAVs to output/ (default off)
    },
    "prepare": {"copy_wav": False, "keep_punctuation": True},
    "ctc_prealign": {
        "enabled": True,
        "model_path": "/mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR",
        "device": "cuda:0",
        "python": "/home/user/miniconda3/envs/asr/bin/python",
        "limit": 0,
        "timeout": 3600,
        "nvv_enabled": True,           # NVV 标签检测 (默认启用)
    },
    "ctc_adjust": {"enabled": True, "limit": 0},
    "mfa": {
        "num_jobs": 0,               # 0 = auto (os.cpu_count()); on high-core machines,
                                     # set explicitly to avoid oversubscription.
                                     # Shards = min(8, cpu//4, stems//200); ~2 cores/shard for MFCC.
        "single_speaker": True,
        "output_format": "long_textgrid",
        "clean": False,              # keep feature cache for faster re-runs
        "no_tokenization": True,
        "skip_validate": True,       # MFA align internally validates; standalone validate is redundant
        "fine_tune": False,          # DISABLED: adjust_ctc_boundaries already refines anchors (Regression Case 16)
        "fine_tune_boundary_tolerance": 0.02,  # only used when fine_tune: true
        "beam": 20,                  # Viterbi beam width
        "retry_beam": 80,            # beam width for retry on failure
    },
    "mfa_en": {
        "enabled": True,
        "num_jobs": 4,
        "normalize_workers": 0,       # 0=auto: min(32, cpu_count)
        "corpus_workers": 0,          # 0=auto: min(16, cpu_count)
        "padding_ms": 50,
        "min_segment_dur_ms": 200,
        "max_gap_merge_s": 0.35,
        "beam": 10,
        "retry_beam": 40,
        "acoustic_model": "pretrained_models/acoustic/english_us_arpa.zip",
        "dictionary": "dict/cmudict.dict",
        "g2p_model": "pretrained_models/g2p/english_us_arpa.zip",
        "timeout": 1800,
        "g2p_timeout": 300,
        "strict_provenance": True,
        "fine_tune": False,
    },
    "postprocess": {
        "strict_ok": True,
        "merge_silence": True,
        "min_sil_merge_sec": 0.2,
        "fix_short_word": True,
        "short_word_max_sec": 0.25,
        "flank_silence_sec": 0.4,
        "short_word_search_window": 0.5,
        "detect_bgm": True,
        "bgm_noise_floor_ratio": 2.0,
        "bgm_min_sil_dur": 0.3,
        "bgm_speech_ratio": 1.0,
        "bgm_min_energy": 0.01,
        "bgm_max_threshold": 0.05,
        "filter_suspicious": True,
        "filter_short_phone_sec": 0.015,
        "filter_long_word_sec": 1.0,
        "filter_min_word_sec": 0.15,
        "filter_min_word_dur_sec": 0.02,
        "filter_word_energy_ratio": 2.0,
        "enable_word_in_silence_filter": False,  # 默认关闭 word_in_silence 过滤
        "filter_min_phone_coverage": 0.35,
        "filter_edge_gap_sec": 0.25,
        "filter_flank_silence_sec": 0.4,
        "filter_long_consonant_sec": 999.0,
        "filter_long_vowel_sec": 999.0,
        "filter_short_phone_en_sec": 0.010,
        "filter_long_vowel_en_sec": 0.500,
        "filter_long_consonant_en_sec": 1.000,
        "filter_min_en_phone_coverage": 0.25,
        "enable_text_correction": True,
        "handle_unexpected_sil": True,
        "workers": 0,            # 0 = auto (os.cpu_count())
    },
    "output_spec": {
        "trim": ["audio/**/*.wav"],
        "prealign": [
            "ctc_pretg/*.TextGrid", "ctc_pretg/*.lab",
            "ctc_pretg/*_tokens.jsonl", "ctc_pretg/*_text_cn.txt",
            "ctc_pretg/manifest.json", "ctc_pretg/summary.txt",
        ],
        "adjust": [
            "ctc_pretg_adj/*.TextGrid", "ctc_pretg_adj/*.lab",
            "ctc_pretg_adj/*_tokens.jsonl", "ctc_pretg_adj/*_text_cn.txt",
        ],
        "align": ["aligned/*.TextGrid"],
        "postprocess": [
            "output/*.TextGrid", "output/tone_mapping.json",
            "output/postprocess_report.jsonl", "filtered/*.TextGrid",
        ],
    },
    "streaming": {
        "prefetch_buffer": 4,     # max prefetched batches on local NVMe
        "upload_buffer": 4,       # max completed batches awaiting NAS upload
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.  Returns a new dict."""
    import copy
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load config file and merge with built-in defaults.

    Task configs only need ``workspace`` and ``data_dir`` — everything
    else inherits sensible defaults from :data:`DEFAULT_CFG`.
    """
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CFG, user_cfg)


def resolve_path(base: Path, value: str | None) -> Path | None:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


def strict_run_paths(workspace: Path, configured_output: Path, run_id: str,
                     publish_enabled: bool) -> tuple[Path, Path, Path | None]:
    """Return private strict output/filter paths and optional publish target.

    Keeping this pure makes ``--no-output-staging`` testable: strict local
    isolation is always enabled, while NAS publication is opt-in only.
    """
    run_root = workspace / "strict_ok_runs" / run_id
    publish_target = (
        configured_output.parent / f"{configured_output.name}.runs" / run_id
        if publish_enabled else None
    )
    return run_root / "output", run_root / "filtered", publish_target


def resolve_num_jobs(cfg_val: int, n_stems: int = 0) -> int:
    """Resolve *num_jobs* config value for MFA ``--num_jobs``.

    *  ``0`` or negative → ``os.cpu_count()``
    *  *n_stems* optional hint: caps at ``n_stems`` (can't parallelize
       beyond utterances).
    *  BLAS threading is pinned to 1 per worker by :func:`get_mfa_env`,
       so ``num_jobs = cpu_count`` → each worker uses 1 BLAS thread →
       near-linear scaling without oversubscription.
    """
    if cfg_val <= 0:
        import multiprocessing as mp
        return mp.cpu_count()
    if n_stems > 0:
        return min(cfg_val, n_stems)
    return cfg_val


# ---------------------------------------------------------------------------
# MFA environment — imported from pipeline_utils (find_mfa_python, get_mfa_env)
# ---------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════════════════
# NVMe audio cache — transparent NAS → local SSD acceleration
# ═══════════════════════════════════════════════════════════════════════════════

_NVME_CACHE_ROOT = Path("/mnt/nvme3/mfa_audio_cache")
_NVME_MANIFEST_NAME = "cache_manifest.json"
_TEMP_CACHE_ROOT = Path("/tmp/mfa_audio_cache")


def _resolve_nvme_cache(data_dir: Path,
                        nvme_override: str | None = None,
                        auto_cache: bool = False,
                        ) -> tuple[Path | None, bool]:
    """Detect and optionally create an NVMe-local mirror of the audio data.

    Resolution order:
    1. If *nvme_override* is given, use that path verbatim.
    2. If ``/mnt/nvme3/mfa_audio_cache/cache_manifest.json`` exists and its
       ``source`` field matches *data_dir*, use it (permanent, keep).
    3. If *auto_cache* is True, create a temp mirror under
       ``/tmp/mfa_audio_cache/``, copy speaker subdirectories,
       and mark for auto-cleanup.
    4. Otherwise, print setup instructions and return None (fall back to NAS).

    Returns ``(cache_dir, is_temp)`` where *is_temp* means the caller should
    delete the directory when done.
    """
    import json as _json

    # 1. Explicit override (from --nvme-cache or config nvme_cache)
    if nvme_override:
        _p = Path(nvme_override)
        if _p.exists():
            # Verify it has the right structure before trusting it
            if (_p / _NVME_MANIFEST_NAME).exists() or any(
                d.is_dir() and any(d.glob("*.wav"))
                for d in _p.iterdir()
            ):
                return _p, False
        print(f"  NVMe cache not ready: {_p}")

    # 2. Permanent cache check
    _manifest = _NVME_CACHE_ROOT / _NVME_MANIFEST_NAME
    if _manifest.exists():
        try:
            _m = _json.loads(_manifest.read_text(encoding="utf-8"))
            _m_src = Path(_m.get("source", ""))
            if _m_src == data_dir.resolve():
                if any(
                    (d.is_dir() and any(d.glob("*.wav")))
                    for d in _NVME_CACHE_ROOT.iterdir()
                ):
                    return _NVME_CACHE_ROOT, False
        except Exception:
            pass

    # 3. Auto-create temp cache (only with explicit --auto-cache flag)
    if auto_cache:
        print(f"  Creating temp audio cache: {_TEMP_CACHE_ROOT}")
        _t0 = time.time()
        _copied = 0
        _bytes_copied = 0
        try:
            _TEMP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            for _entry in sorted(data_dir.iterdir()):
                if not _entry.is_dir():
                    continue
                _wavs = sorted(_entry.glob("*.wav"))
                if not _wavs:
                    continue
                _speaker_dst = _TEMP_CACHE_ROOT / _entry.name
                _speaker_dst.mkdir(exist_ok=True)
                _n = len(_wavs)
                for _i, _src in enumerate(_wavs):
                    _dst = _speaker_dst / _src.name
                    if not _dst.exists():
                        import shutil as _shutil
                        _shutil.copy2(str(_src), str(_dst))
                        _copied += 1
                        _bytes_copied += _src.stat().st_size
                    if (_i + 1) % 1000 == 0 or _i == _n - 1:
                        _pct = (_i + 1) / _n * 100
                        _elapsed = time.time() - _t0
                        _gb = _bytes_copied / 1024**3
                        _spd = _gb / _elapsed * 1024 if _elapsed > 0 else 0
                        print(f"\r    [{_entry.name}] {_i+1}/{_n} ({_pct:.0f}%)"
                              f" | {_gb:.1f} GB | {_spd:.0f} MB/s", end="")
                print()
            _elapsed = time.time() - _t0
            _gb = _bytes_copied / 1024**3
            print(f"  Temp cache ready: {_copied} files ({_gb:.1f} GB)"
                  f" in {_elapsed:.0f}s")
            return _TEMP_CACHE_ROOT, True
        except Exception as _e:
            print(f"  WARNING: Temp cache creation failed: {_e}")
            return None, False

    # 4. No cache found — print instructions, fall back to NAS
    print(f"  ═══════════════════════════════════════════════════════════")
    print(f"  NVMe audio cache not found.")
    print(f"  Run once to eliminate NAS I/O (~30 min, one-time):")
    print(f"    python scripts/cache_audio_to_nvme.py \\")
    print(f"        --source {data_dir}")
    print(f"  Or use --auto-cache to create a temp cache for this run.")
    print(f"  Falling back to NAS audio (slower).")
    print(f"  ═══════════════════════════════════════════════════════════")
    return None, False


def _cleanup_nvme_cache(cache_dir: Path, is_temp: bool) -> None:
    """Remove temp cache directory. No-op for permanent caches."""
    if not is_temp or not cache_dir or not cache_dir.exists():
        return
    import shutil as _shutil
    print(f"Cleaning temp audio cache: {cache_dir}")
    _shutil.rmtree(str(cache_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_python(script: Path, script_args: list[str], mfa_python: Path,
               models_dir: Path, desc: str = "", timeout: int = 86400) -> int:
    cmd = [str(mfa_python), str(script)] + script_args
    print(f"\n{'='*60}\n  {desc or script.name}\n  {' '.join(cmd)}\n{'='*60}\n")
    try:
        result = subprocess.run(cmd, env=get_mfa_env(mfa_python, models_dir),
                                timeout=timeout, capture_output=False)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s: {desc or script.name}")
        return 1


def run_mfa(mfa_args: list[str], mfa_python: Path, models_dir: Path,
            desc: str = "", timeout: int | None = None) -> int:
    print(f"\n{'='*60}\n  {desc or 'MFA: ' + ' '.join(mfa_args)}\n{'='*60}\n")
    try:
        return subprocess.run(
            [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa"] + mfa_args,
            env=get_mfa_env(mfa_python, models_dir), timeout=timeout,
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s: {desc or 'MFA'}")
        return 1


# ---------------------------------------------------------------------------
# Pipeline steps — all take (args, cfg, mfa_python, ctx)
# ctx = {data_dir, audio_dir, pinyin_dir, mfa_audio_dir, aligned_dir,
#        output_dir, filtered_dir, validate_dir, models_dir, temp_dir,
#        mfa_dict, ctc_pretg}
# ---------------------------------------------------------------------------

def step_trim_silence(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    tc = cfg["trim"]
    wav_out = ctx["audio_dir"]
    if wav_out.exists() and any(wav_out.iterdir()) and not args.force:
        print(f"  Output exists: {wav_out}. Use --force to re-run.")
        return 0
    trim_args = [
        "--input-dir", str(ctx["data_dir"]), "--output-dir", str(wav_out),
        "--max-silence-sec", str(tc["max_silence_sec"]),
        "--sil-vol-threshold", str(tc["sil_vol_threshold"]),
        "--sil-len-threshold", str(tc["sil_len_threshold"]),
        "--workers", str(tc["workers"]),
    ]
    if tc.get("normalize_edges"):
        trim_args += [
            "--normalize-edges",
            "--target-edge-silence-sec", str(tc["target_edge_silence_sec"]),
            "--edge-silence-threshold", str(tc["edge_silence_threshold"]),
            "--edge-frame-length", str(tc["edge_frame_length"]),
        ]
    if tc.get("target_sr"):
        trim_args += ["--target-sr", str(int(tc["target_sr"]))]
    return run_python(SCRIPTS_DIR / "trim_silence_batch.py", trim_args, mfa_python,
                      ctx["models_dir"],
                      "Step 1: Audio Preprocessing")


def _resample_one(wav_path: Path, audio_dir: Path, out_dir: Path,
                  target_sr: int, overwrite: bool) -> tuple[str, bool, str]:
    """Worker for parallel resample (module-level, pickleable)."""
    import shutil
    import struct
    import soundfile as sf
    sys.path.insert(0, str(SCRIPTS_DIR))
    from audio_utils import resample_audio

    rel = wav_path.relative_to(audio_dir)
    out = out_dir / rel
    if out.exists() and not overwrite:
        return (str(rel), False, "skipped")

    # Fast path: read sample rate from WAV header (44 bytes) instead of full
    # sf.info() — saves ~5-15ms per file on SMB/CIFS mounts
    def _read_sr_fast(p: Path) -> int:
        """Read sample rate from WAV header only."""
        try:
            with open(str(p), 'rb') as fh:
                header = fh.read(44)
            if len(header) >= 40 and header[:4] == b'RIFF':
                return struct.unpack_from('<I', header, 24)[0]
        except Exception:
            pass
        # Fallback: soundfile (handles non-standard headers, FLAC, etc.)
        try:
            return sf.info(str(p)).samplerate
        except Exception:
            return 0

    sr = _read_sr_fast(wav_path)
    if sr == target_sr:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(str(wav_path), str(out))
            return (str(rel), True, "linked")
        except OSError:
            shutil.copy2(str(wav_path), str(out))
            return (str(rel), True, "copied")

    out.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(wav_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
    sf.write(str(out), audio, target_sr, subtype='PCM_16')
    return (str(rel), True, "resampled")


def step_resample_for_mfa(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Resample trimmed audio to 16kHz for MFA (parallelised).

    Uses ThreadPoolExecutor because the work is I/O-bound (file read/write,
    hard-link, copy) and any CPU work (scipy resample) releases the GIL.
    This avoids ProcessPoolExecutor's per-worker spawn overhead on Windows
    (each worker imports numpy/scipy/soundfile from scratch, ~2-5 s each).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp

    audio_dir = ctx["audio_dir"]
    mfa_audio_dir = ctx["mfa_audio_dir"]
    target_sr = 16000
    overwrite = args.overwrite

    expected_stems = tuple(ctx.get("expected_stems", ()))

    # Use scandir for fast flat-listing (common case)
    wavs: list[Path] = []

    if expected_stems and ctx.get("strict_ready"):
        evidence = ctx.get("strict_ready_evidence")
        if (not isinstance(evidence, dict) or audio_dir != evidence.get("_audio_root")):
            print("  ERROR: strict resample input escaped evidenced audio_view")
            return 1
        for stem in expected_stems:
            candidate = audio_dir / f"{stem}.wav"
            record = evidence["_artifacts"][stem]["audio"]
            try:
                resolved = _strict_regular_file(candidate, audio_dir)
            except ValueError as exc:
                print(f"  ERROR: {exc}")
                return 1
            if resolved != record["path"] or candidate.stat().st_size != record["size"]:
                print(f"  ERROR: strict resample WAV evidence mismatch: {stem}")
                return 1
            wavs.append(candidate)
        print(f"  Selected {len(wavs)} evidenced WAVs from the immutable audio_view")
    elif expected_stems:
        expected_names = {f"{stem}.wav" for stem in expected_stems}
        try:
            entries = list(audio_dir.iterdir())
        except OSError as exc:
            print(f"  ERROR: cannot enumerate strict audio input: {exc}")
            return 1
        ordinary_names = {entry.name for entry in entries
                          if entry.is_file() and not entry.is_symlink()}
        if (ordinary_names != expected_names or len(entries) != len(expected_names)
                or any(entry.is_symlink() or not entry.is_file() for entry in entries)):
            print("  ERROR: resample audio set differs from frozen denominator")
            print(f"    missing={len(expected_names - ordinary_names)}, "
                  f"extra={len(ordinary_names - expected_names)}")
            return 1
        wavs = [audio_dir / f"{stem}.wav" for stem in expected_stems]

    # Fast path: read stems from ctc_ready manifest (no directory scan)
    manifest_path = ctx.get("ctc_pretg", Path()) / "ctc_ready_manifest.json"
    if not expected_stems and manifest_path.exists():
        import json as _json
        try:
            manifest = _json.loads(manifest_path.read_text())
            stems = manifest.get("stems", [])
            if stems:
                wavs = []
                missing = 0
                for s in stems:
                    w = find_wav(audio_dir, s)
                    if w:
                        wavs.append(w)
                    else:
                        missing += 1
                if missing:
                    print(f"  Warning: {missing}/{len(stems)} WAVs not found"
                          f" (mangled filenames)")
                print(f"  Found {len(wavs)} WAVs from manifest"
                      f" (skipping directory scan)")
        except Exception:
            pass

    if not wavs:
        try:
            with os.scandir(str(audio_dir)) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".wav"):
                        wavs.append(Path(entry.path))
        except OSError:
            pass
        if not wavs:
            wavs = list(audio_dir.rglob("*.wav"))  # fallback to recursive

    if not wavs:
        print("  No WAVs found in audio dir.")
        return 1

    # Fast count via scandir (avoid rglob on SMB)
    existing_count = 0
    if mfa_audio_dir.exists():
        try:
            with os.scandir(str(mfa_audio_dir)) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".wav"):
                        existing_count += 1
        except OSError:
            pass
    if existing_count >= len(wavs) and not overwrite:
        print(f"  {existing_count} resampled WAVs already exist. Use --overwrite to redo.")
        return 0

    mfa_audio_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(resolve_num_jobs(cfg.get("mfa", {}).get("num_jobs", 0)),
                    len(wavs), mp.cpu_count())
    done = skipped = 0
    actions: dict[str, int] = {}

    if n_workers <= 1 or len(wavs) <= 4:
        # Sequential for small jobs
        for wav in wavs:
            _, ok, action = _resample_one(wav, audio_dir, mfa_audio_dir,
                                          target_sr, overwrite)
            if ok:
                done += 1
            else:
                skipped += 1
            actions[action] = actions.get(action, 0) + 1
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_resample_one, w, audio_dir, mfa_audio_dir,
                            target_sr, overwrite): w
                for w in wavs
            }
            for fut in as_completed(futures):
                _, ok, action = fut.result()
                if ok:
                    done += 1
                else:
                    skipped += 1
                actions[action] = actions.get(action, 0) + 1

    parts = [f"{done} done"]
    for action, n in sorted(actions.items()):
        parts.append(f"{n} {action}")
    print(f"  Resampled to {target_sr}Hz -> {mfa_audio_dir}  ({', '.join(parts)})")
    if expected_stems:
        entries = list(mfa_audio_dir.iterdir())
        actual_names = {entry.name for entry in entries
                        if entry.is_file() and not entry.is_symlink()}
        expected_names = {f"{stem}.wav" for stem in expected_stems}
        if (actual_names != expected_names or len(entries) != len(expected_names)
                or any(entry.is_symlink() or not entry.is_file() for entry in entries)):
            print("  ERROR: resampled WAV set differs from frozen denominator")
            return 1
    return 0 if done > 0 else 1


def step_mfa_validate(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """MFA validate — uses NVASR output (.lab) as corpus."""
    mc = cfg["mfa"]
    corpus_dir = ctx["ctc_pretg"]
    if not corpus_dir.exists() or not list(corpus_dir.glob("*.lab")):
        # Fallback to pinyin_dir if ctc_pretg has no .lab files
        corpus_dir = ctx["pinyin_dir"]
        if not list(corpus_dir.glob("*.txt")):
            print("ERROR: No .lab files in ctc_pretg/ or .txt files in pinyin_dir.")
            return 1
    # Use pre-extracted directory if available — avoids MFA Archive.__init__
    # deleting and re-extracting the zip (which races with parallel workers).
    extracted = ctx["models_dir"] / "extracted_models" / "acoustic" / f"{cfg['acoustic_model']}_acoustic"
    acoustic_model_arg = str(extracted) if extracted.is_dir() else cfg["acoustic_model"]

    mfa_args = [
        "validate", str(corpus_dir), str(ctx["mfa_dict"]),
        "--acoustic_model_path", acoustic_model_arg,
        "--audio_directory", str(ctx["mfa_audio_dir"]),
        "--temporary_directory", str(ctx["temp_dir"]),
        "--num_jobs", str(resolve_num_jobs(mc.get("num_jobs", 0))),
        "--overwrite",
    ]
    if mc.get("clean"):
        mfa_args.append("--clean")
    if mc.get("single_speaker"):
        mfa_args.append("--single_speaker")
    return run_mfa(mfa_args, mfa_python, ctx["models_dir"], "Step 5: MFA Validate")


def step_prealign(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Run NVASR CTC forced alignment -> produce MFA anchor TextGrids."""
    pc = cfg.get("ctc_prealign", {})
    if not pc.get("enabled", False):
        print("  CTC prealign disabled in config (ctc_prealign.enabled=false). Skipping.")
        return 0

    ctc_out = ctx["ctc_pretg"]
    if ctc_out.exists() and any(ctc_out.glob("*.TextGrid")) and not args.overwrite:
        print(f"  CTC TextGrids exist: {ctc_out}. Use --overwrite to re-run.")
        return 0

    # NVASR needs funasr+torch — use dedicated Python, not MFA's
    nvras_py = pc.get("python", "")
    if not nvras_py:
        nvras_py = sys.executable  # fallback: the Python running this pipeline
    nvras_py_path = Path(nvras_py)
    if not nvras_py_path.exists():
        print(f"ERROR: NVASR Python not found: {nvras_py}")
        print(f"  Set ctc_prealign.python in config.yaml to a Python with funasr+torch installed.")
        return 1
    print(f"  NVASR Python: {nvras_py_path}")

    prealign_args = [
        "--data-dir", str(ctx["data_dir"]),
        "--pinyin-dir", str(ctx["pinyin_dir"]),
        "--audio-dir", str(ctx["audio_dir"]),
        "--output-dir", str(ctc_out),
        "--model-path", str(resolve_path(PROJECT_ROOT,
                                        pc.get("model_path", "models/Multilingual-NVASR"))),
        "--device", getattr(args, "device", None) or pc.get("device", "cuda:0"),
        "--dict-path", str(ctx["mfa_dict"]),
    ]
    if pc.get("nvv_bias", 0) > 0:
        prealign_args += ["--nvv-bias", str(pc["nvv_bias"])]
    if not pc.get("nvv_enabled", True):
        prealign_args.append("--no-nvv")
    if pc.get("limit", 0) > 0:
        prealign_args += ["--limit", str(pc["limit"])]
    if pc.get("offset", 0) > 0:
        prealign_args += ["--offset", str(pc["offset"])]
    if args.overwrite:
        prealign_args.append("--overwrite")

    # Use run_python with the NVASR Python, not mfa_python
    return run_python(SCRIPTS_DIR / "ctc_prealign.py", prealign_args, nvras_py_path,
                      ctx["models_dir"], "Step 4: CTC Pre-alignment (NVASR -> MFA anchors)",
                      timeout=pc.get("timeout", 3600))


def _skip_if_ctc_normalized(ctx: dict) -> bool:
    """Return True if ctc_prealign already ran normalize_* steps.

    A v3 (string-only) marker is accepted for backward compatibility with
    data written by older pipeline versions.  A v4 marker additionally
    validates that its embedded stem count and manifest digest match the
    current on-disk state — a mismatch means the data was modified after
    normalization and must be re-validated.
    """
    _marker = ctx.get("ctc_pretg", Path()) / ".ctc_normalized"
    try:
        if not _marker.exists():
            return False
        _text = _marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False

    # v3 legacy marker: exact string match
    if _text == CTC_NORMALIZATION_MARKER:
        print(f"  Skipping: ctc_prealign already normalized (v3 marker: {_marker})")
        return True

    # v4 content-identity marker: parse and validate
    _info = parse_ctc_normalization_marker(_text)
    if _info is not None:
        _ctc_dir = ctx.get("ctc_pretg", Path())
        _actual_labs = len(list(_ctc_dir.glob("*.lab")))
        if _info["stems"] != _actual_labs:
            print(f"  Re-running normalization: stem count mismatch "
                  f"(marker={_info['stems']}, actual={_actual_labs})")
            return False
        _manifest = _ctc_dir / "manifest.json"
        if _manifest.exists():
            _actual_digest = hashlib.sha256(
                _manifest.read_bytes()).hexdigest()
            if _info["manifest_sha256"] != _actual_digest:
                print(f"  Re-running normalization: manifest digest mismatch")
                return False
        print(f"  Skipping: ctc_prealign already normalized "
              f"(v4 marker, {_info['stems']} stems)")
        return True

    if _marker.exists():
        print(f"  Re-running normalization: unparseable marker ({_marker})")
    return False


def step_normalize_punct(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Normalize punctuation in CTC output text and sync with punct.json anchors.

    1. ASCII -> CJK equivalents (existing)
    2. Non-whitelist punctuation -> ，(fullwidth comma)
    3. Merge adjacent punctuation — no two puncts side by side;
       timestamps in _punct.json are merged to span the combined range.
    """
    if _skip_if_ctc_normalized(ctx):
        return 0
    import json

    ALLOWED_PUNCT = frozenset("，。！？、；：…")
    ASCII_MAP = {
        ",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "：",
    }
    ctc_dir = ctx["ctc_pretg"]
    count = 0
    missing = 0

    for txt_file in sorted(ctc_dir.glob("*_text_cn.txt")):
        stem = txt_file.stem.replace("_text_cn", "")

        try:
            text = txt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            missing += 1
            print(f"  WARNING: Skipping {txt_file.name} — file missing "
                  f"(symlink target gone?)")
            continue

        # --- load CTC punctuation anchors (time-aligned) ---
        punct_file = ctc_dir / f"{stem}_punct.json"
        punct_entries: list[dict] = []
        if punct_file.exists():
            try:
                punct_entries = json.loads(punct_file.read_text())
            except Exception:
                pass

        # === Phase 1 — ASCII -> CJK ===
        text = text.translate(str.maketrans(ASCII_MAP))
        for p in punct_entries:
            w = p.get("word", "")
            if w in ASCII_MAP:
                p["word"] = ASCII_MAP[w]

        # === Phase 2 — classify each character ===
        # '-' between two ASCII letters is part of an NVV token
        # (e.g. SURPRISE-OH, QUESTION-EI) — never treat as punct.
        # Regression Case 17-E / Case 23.
        char_info: list[tuple[str, bool | None, str]] = []
        for i, ch in enumerate(text):
            is_hyphen_in_nvv = (
                ch == '-'
                and i > 0 and i + 1 < len(text)
                and text[i - 1].isascii() and text[i - 1].isalpha()
                and text[i + 1].isascii() and text[i + 1].isalpha()
            )
            if is_punct(ch) and not is_hyphen_in_nvv:
                char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
            else:
                char_info.append(("other", None, ch))

        # Build map from punct ordinal -> punct_entries index
        pidx_map: dict[int, int] = {}
        pi = 0
        for ci, (kind, _, _) in enumerate(char_info):
            if kind == "punct":
                pidx_map[ci] = pi
                pi += 1

        # Mark entries that will be deleted (merged away)
        for p in punct_entries:
            p["_merge_del"] = False

        # === Phase 3 — replace abnormal + merge adjacent ===
        new_chars: list[str] = []
        i = 0
        punct_seq = 0  # ordinal among punct characters so far

        while i < len(char_info):
            kind, is_allowed, ch = char_info[i]
            if kind != "punct":
                new_chars.append(ch)
                i += 1
                continue

            # Collect consecutive punctuation characters
            group: list[tuple[int, bool, str]] = []  # (char_index, is_allowed, char)
            j = i
            while j < len(char_info) and char_info[j][0] == "punct":
                group.append((j, char_info[j][1], char_info[j][2]))
                j += 1

            # ---- single punctuation ----
            if len(group) == 1:
                _, ia, ch = group[0]
                if ia:
                    new_chars.append(ch)
                else:
                    new_chars.append("，")
                    if punct_seq < len(punct_entries):
                        punct_entries[punct_seq]["word"] = "，"
                punct_seq += 1
                i = j
                continue

            # ---- N adjacent punctuation -> merge into one ， ----
            new_chars.append("，")

            first_seq = punct_seq
            last_seq = punct_seq + len(group) - 1

            if first_seq < len(punct_entries) and last_seq < len(punct_entries):
                first = punct_entries[first_seq]
                last = punct_entries[last_seq]

                first["word"] = "，"
                first["end_ms"] = last["end_ms"]
                first["end_s"] = last["end_s"]

                for k in range(first_seq + 1, last_seq + 1):
                    if k < len(punct_entries):
                        punct_entries[k]["_merge_del"] = True

            punct_seq += len(group)
            i = j

        # === Phase 4 — write back ===
        new_text = "".join(new_chars)
        new_punct = [p for p in punct_entries if not p.pop("_merge_del", False)]

        changed = new_text != text or len(new_punct) != len(punct_entries)

        if changed:
            txt_file.write_text(new_text + "\n", encoding="utf-8")
            if punct_file.exists() or new_punct:
                punct_file.write_text(
                    json.dumps(new_punct, ensure_ascii=False), encoding="utf-8")
            count += 1

    if missing:
        print(f"  WARNING: {missing} _text_cn.txt file(s) not found, skipped")
    print(f"  Normalized punctuation in {count} files")
    return 0


def step_normalize_text(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Normalize numerals in human text and verify/recover CTC transcripts.

    MFA lab files are token sequences.  Applying cn2an to them corrupts
    pinyin tone suffixes (for example rui4 -> rui四), so a lab is only
    recovered from its validated tokens JSONL sequence.
    """
    if _skip_if_ctc_normalized(ctx):
        return 0
    try:
        import cn2an
    except ImportError:
        cn2an = None
        print("  cn2an not installed; validating CTC bundles without text numeral conversion.")
    ctc_dir = ctx["ctc_pretg"]
    text_changed = 0
    lab_recovered = 0
    failures: list[tuple[str, str]] = []

    # Human/ASR text may contain true Arabic numerals.  Protect pinyin/NVV
    # tokens even here, but never apply this transform to an MFA lab.
    for txt_file in sorted(ctc_dir.glob("*_text_cn.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            print(f"  WARNING: Skipping {txt_file.name} — file missing (symlink target gone?)")
            continue
        normalized = (
            normalize_reference_numerals(text, cn2an.transform)
            if cn2an is not None else text
        )
        if normalized != text:
            txt_file.write_text(normalized + "\n", encoding="utf-8")
            text_changed += 1

    # tokens + CTC words must agree before tokens are allowed to repair a lab.
    for tokens_path in sorted(ctc_dir.glob("*_tokens.jsonl")):
        stem = tokens_path.name[:-len("_tokens.jsonl")]
        lab_path = ctc_dir / f"{stem}.lab"
        textgrid_path = ctc_dir / f"{stem}.TextGrid"
        try:
            token_words = [
                entry["word"].strip()
                for entry in load_ctc_token_entries(tokens_path)
            ]
            tg_words = read_ctc_textgrid_words(textgrid_path)
            if token_words != tg_words:
                raise ValueError(
                    f"TextGrid/tokens mismatch ({len(tg_words)} != "
                    f"{len(token_words)})"
                )
            current_words = (
                lab_path.read_text(encoding="utf-8-sig").strip().split()
                if lab_path.exists() else []
            )
            if current_words != token_words:
                rebuild_lab_from_tokens(tokens_path, lab_path)
                lab_recovered += 1
            bundle_errors = validate_ctc_transcript_bundle(ctc_dir, stem)
            if bundle_errors:
                raise ValueError("; ".join(bundle_errors))
        except (OSError, ValueError) as exc:
            failures.append((stem, str(exc)))

    lab_stems = {p.stem for p in ctc_dir.glob("*.lab")}
    token_stems = {
        p.name[:-len("_tokens.jsonl")]
        for p in ctc_dir.glob("*_tokens.jsonl")
    }
    missing_tokens = sorted(lab_stems - token_stems)
    if missing_tokens:
        failures.extend((stem, "missing *_tokens.jsonl")
                        for stem in missing_tokens)

    print(f"  Numeral normalization: {text_changed} human text file(s) changed")
    print(f"  CTC transcript recovery: {lab_recovered} lab file(s) rebuilt from tokens")
    if failures:
        print(f"  ERROR: {len(failures)} invalid CTC transcript bundle(s)")
        for stem, reason in failures[:20]:
            print(f"    - {stem}: {reason}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        return 1
    return 0

def step_normalize_ria(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Merge legacy ria fragments across lab, tokens and CTC TextGrid.

    Safety net for old CTC output.  New data is handled inline by
    ctc_prealign (align_text gets CJK→ria before tokenizer).
    Does NOT modify _text_cn.txt / _text_raw.txt (ASR archive).
    """
    if _skip_if_ctc_normalized(ctx):
        return 0
    import json, re
    from normalize_english_tokens import rewrite_ctc_textgrid_words

    ctc_dir = ctx["ctc_pretg"]
    if not ctc_dir or not ctc_dir.exists():
        return 0

    changed_count = 0
    failures: list[tuple[str, str]] = []

    for tokens_path in sorted(ctc_dir.rglob("*_tokens.jsonl")):
        stem = tokens_path.name[:-len("_tokens.jsonl")]
        lab_file = tokens_path.with_name(f"{stem}.lab")
        tg_path = tokens_path.with_name(f"{stem}.TextGrid")
        try:
            entries = load_ctc_token_entries(tokens_path)
            new_entries: list[dict] = []
            index = 0
            changed = False
            while index < len(entries):
                current = entries[index]
                word = current["word"]
                if (re.fullmatch(r"rui[0-5]", word)
                        and index + 1 < len(entries)
                        and re.fullmatch(
                            r"(?:ya|a)[0-5]", entries[index + 1]["word"])):
                    following = entries[index + 1]
                    new_entries.append({
                        "word": "ria",
                        "start_ms": current["start_ms"],
                        "end_ms": following["end_ms"],
                        "start_s": current["start_s"],
                        "end_s": following["end_s"],
                        "type": current.get("type", "word"),
                    })
                    index += 2
                    changed = True
                else:
                    new_entries.append(current)
                    index += 1
            if not changed:
                continue

            if not tg_path.exists():
                raise FileNotFoundError(f"missing CTC TextGrid: {tg_path}")
            tokens_tmp = tokens_path.with_name(f".{tokens_path.name}.tmp")
            lab_tmp = lab_file.with_name(f".{lab_file.name}.tmp")
            tokens_tmp.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False)
                          for row in new_entries) + "\n",
                encoding="utf-8",
            )
            lab_tmp.write_text(
                " ".join(row["word"] for row in new_entries) + "\n",
                encoding="utf-8",
            )
            rewrite_ctc_textgrid_words(tg_path, new_entries)
            tokens_tmp.replace(tokens_path)
            lab_tmp.replace(lab_file)
            errors = validate_ctc_transcript_bundle(ctc_dir, stem)
            if errors:
                raise ValueError("; ".join(errors))
            changed_count += 1
        except (OSError, ValueError, KeyError) as exc:
            failures.append((stem, str(exc)))

    if changed_count:
        print(f"  [normalize_ria] {changed_count} synchronized bundle(s)")
    if failures:
        print(f"  ERROR: {len(failures)} RIA bundle(s) failed")
        for stem, reason in failures[:20]:
            print(f"    - {stem}: {reason}")
        return 1
    return 0


def step_normalize_en(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Normalise English-word phonetic fragments in .lab and _tokens.jsonl.

    NVASR tokenizer breaks OOV English words into pinyin approximations
    (e.g. "ria"->"rui4"+"ya4").  This step merges them back into the
    canonical spelling before MFA alignment.
    """
    if _skip_if_ctc_normalized(ctx):
        return 0
    ctc_dir = ctx["ctc_pretg"]
    if not ctc_dir or not ctc_dir.exists():
        return 0

    script = SCRIPTS_DIR / "normalize_english_tokens.py"
    norm_en_args = ["--txt-dir", str(ctc_dir)]
    en_cfg = cfg.get("mfa_en", {})
    nw = en_cfg.get("normalize_workers", 0)
    if nw > 0:
        norm_en_args += ["--workers", str(nw)]
    if ctx.get("mfa_dict"):
        norm_en_args += ["--dict-path", str(ctx["mfa_dict"])]
    rc = run_python(
        script, norm_en_args,
        mfa_python, ctx["models_dir"],
        "Step 2b: Normalise English tokens")
    if rc != 0:
        return rc

    invalid: list[tuple[str, list[str]]] = []
    for lab_path in sorted(ctc_dir.glob("*.lab")):
        errors = validate_ctc_transcript_bundle(ctc_dir, lab_path.stem)
        if errors:
            invalid.append((lab_path.stem, errors))
    if invalid:
        print(f"  ERROR: {len(invalid)} CTC bundle(s) invalid after normalize_en")
        for stem, errors in invalid[:20]:
            print(f"    - {stem}: {'; '.join(errors)}")
        if len(invalid) > 20:
            print(f"    ... and {len(invalid) - 20} more")
        return 1
    print(f"  CTC bundle validation: {len(list(ctc_dir.glob('*.lab')))} OK")
    return 0


def step_adjust_ctc(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Run energy-based CTC anchor boundary adjustment before MFA."""
    ac = cfg.get("ctc_adjust", {})
    if not ac.get("enabled", True):
        print("  CTC adjust disabled in config (ctc_adjust.enabled=false). Skipping.")
        ctx["ctc_pretg_adj"] = ctx["ctc_pretg"]
        return 0

    ctc_in = ctx["ctc_pretg"]
    ctc_out = ctx["ctc_pretg_adj"]

    if ctc_out.exists() and any(ctc_out.glob("*.TextGrid")) and not args.overwrite:
        print(f"  Adjusted CTC anchors exist: {ctc_out}. Use --overwrite to re-run.")
        ctx["ctc_pretg_adj"] = ctc_out
        return 0

    adjust_args = [
        "--ctc-dir", str(ctc_in),
        "--audio-dir", str(ctx["audio_dir"]),
        "--output-dir", str(ctc_out),
    ]
    if ac.get("limit", 0) > 0:
        adjust_args += ["--limit", str(ac["limit"])]
    if args.overwrite:
        adjust_args.append("--overwrite")

    rc = run_python(SCRIPTS_DIR / "adjust_ctc_boundaries.py", adjust_args, mfa_python,
                    ctx["models_dir"], "Step 5: Adjust CTC boundaries (energy-based)")

    if rc == 0:
        ctx["ctc_pretg_adj"] = ctc_out
    return rc


def _run_mfa_sharded(
    stems: list[str],
    corpus_dir: Path,
    audio_dir: Path,
    anchors_dir: Path | None,
    dict_path: Path,
    acoustic_model: str,
    aligned_dir: Path,
    workspace: Path,
    mfa_python: Path,
    models_dir: Path,
    num_jobs: int,
    single_speaker: bool,
    no_tokenization: bool,
    beam: int,
    retry_beam: int,
    boost_silence: float,
    clean: bool,
    overwrite: bool,
    output_format: str = "long_textgrid",
    timeout: int | None = None,
    desc: str = "MFA Align",
    **extra_args,
) -> int:
    """Run MFA align in parallel shards to accelerate MFCC extraction.

    MFA's MFCC phase uses limited internal parallelism (~2 cores).  By
    splitting the corpus into N independent shards, each with its own
    MFA instance, we achieve N× MFCC throughput.

    CMVN: with --single_speaker, each shard computes its own per-speaker
    statistics.  For TTS synthetic data (same voice), subset statistics
    are near-identical to global — negligible alignment impact.
    """
    import multiprocessing as _mp

    _n = len(stems)
    _n_shards = min(8, _mp.cpu_count() // 4, max(1, _n // 200))
    if _n_shards <= 1:
        return None  # signal: caller should run single MFA

    _per_shard = (_n + _n_shards - 1) // _n_shards
    _jobs_per_shard = max(1, num_jobs // _n_shards)
    print(f"  MFA sharding: {_n_shards}× ({_per_shard} stems,"
          f" {_jobs_per_shard} jobs each)")

    # ── Prepare shard directories ──
    _run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    _run_id = f"{_run_id}_{os.getpid()}"
    _shard_root = workspace / "mfa_shards" / _run_id
    _log_dir = workspace / "mfa_logs" / _run_id
    _shard_root.mkdir(parents=True, exist_ok=False)
    _log_dir.mkdir(parents=True, exist_ok=False)
    _shard_dirs: list[Path] = []
    _shard_stems: list[list[str]] = []
    _t0 = time.time()

    # Phase 1: create directories + collect symlink pairs (sequential, cheap)
    _link_tasks: list[tuple[Path, Path]] = []  # (src, dst)
    _input_errors: list[str] = []

    for _si in range(_n_shards):
        _ss = stems[_si * _per_shard : (_si + 1) * _per_shard]
        if not _ss:
            break
        _shard_stems.append(_ss)
        _sd = _shard_root / f"shard_{_si:02d}"
        _sd.mkdir(parents=True, exist_ok=True)
        _shard_dirs.append(_sd)

        for _sub in ("corpus", "audio", "anchors", "output", "temp"):
            (_sd / _sub).mkdir(parents=True, exist_ok=True)

        for _stem in _ss:
            _src = corpus_dir / f"{_stem}.lab"
            if _src.exists():
                _link_tasks.append((_src, _sd / "corpus" / f"{_stem}.lab"))
            else:
                _input_errors.append(f"{_stem}: missing lab")
            _src = audio_dir / f"{_stem}.wav"
            if _src.exists():
                _link_tasks.append((_src, _sd / "audio" / f"{_stem}.wav"))
            else:
                _input_errors.append(f"{_stem}: missing wav")
            if anchors_dir:
                _src = anchors_dir / f"{_stem}.TextGrid"
                if _src.exists():
                    _link_tasks.append((_src, _sd / "anchors" / f"{_stem}.TextGrid"))
                else:
                    _input_errors.append(f"{_stem}: missing anchor")

    if _input_errors:
        print(f"  ERROR: {len(_input_errors)} missing MFA shard input(s)")
        for error in _input_errors[:20]:
            print(f"    - {error}")
        print(f"  Shard workspace retained: {_shard_root}")
        return 1

    # Phase 2: parallel symlink creation (each symlink is an independent
    # filesystem metadata operation — threads are the right fit)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _total_links = 0
    _link_failures = 0
    _link_failure_details: list[str] = []
    _n_threads = min(32, len(_link_tasks))
    with ThreadPoolExecutor(max_workers=_n_threads) as _executor:
        _futures = {
            _executor.submit(os.symlink, str(_src), str(_dst)): (_src, _dst)
            for _src, _dst in _link_tasks
        }
        for _future in as_completed(_futures):
            _src, _dst = _futures[_future]
            try:
                _future.result()
                _total_links += 1
            except FileExistsError:
                _total_links += 1  # already linked (e.g. re-run)
            except Exception as exc:
                _link_failures += 1
                if len(_link_failure_details) < 20:
                    _link_failure_details.append(f"{_src} -> {_dst}: {exc}")

    _elapsed = time.time() - _t0
    _msg = f"  Symlinked {_total_links} files in {_elapsed:.1f}s"
    if _link_failures:
        _msg += f" ({_link_failures} failures)"
    print(_msg)
    if _link_failures:
        print("  ERROR: shard input links are incomplete")
        for detail in _link_failure_details:
            print(f"    - {detail}")
        print(f"  Shard workspace retained: {_shard_root}")
        return 1

    # ── Launch parallel MFA instances ──
    _procs: list[tuple] = []
    for _si, _ss in enumerate(_shard_stems):
        _sd = _shard_dirs[_si]
        _mfa_args = [
            "align", str(_sd / "corpus"), str(dict_path),
            acoustic_model, str(_sd / "output"),
            "--audio_directory", str(_sd / "audio"),
            "--temporary_directory", str(_sd / "temp"),
            "--output_format", output_format,
            "--num_jobs", str(_jobs_per_shard),
            "--overwrite", "--no_textgrid_cleanup",
        ]
        if anchors_dir:
            _mfa_args += ["--textgrid_directory", str(_sd / "anchors")]
        if single_speaker:
            _mfa_args.append("--single_speaker")
        if no_tokenization:
            _mfa_args.append("--no_tokenization")
        if clean:
            _mfa_args.append("--clean")
        _mfa_args += ["--beam", str(beam)]
        _mfa_args += ["--retry_beam", str(retry_beam)]
        _mfa_args += ["--boost_silence", str(boost_silence)]
        for _k, _v in extra_args.items():
            if _v is not None:
                _mfa_args += [f"--{_k}", str(_v)]

        _cmd = [str(mfa_python), "-m",
                "montreal_forced_aligner.command_line.mfa"] + _mfa_args
        print(f"  [shard {_si}/{_n_shards}] {len(_ss)} stems,"
              f" {_jobs_per_shard} jobs")
        _log_path = _log_dir / f"shard_{_si:02d}.log"
        _log_handle = _log_path.open("w", encoding="utf-8")
        _proc = subprocess.Popen(
            _cmd,
            env=get_mfa_env(mfa_python, models_dir),
            stdout=_log_handle,
            stderr=subprocess.STDOUT,
        )
        _procs.append((
            _si, _proc, _sd, _log_handle, _log_path,
            set(_ss), time.time(),
        ))

    # ── Wait for all shards ──
    _failed: list[int] = []
    _return_codes: dict[int, int | str] = {}
    for (_si, _proc, _sd, _log_handle, _log_path,
         _expected, _started_at) in _procs:
        try:
            if timeout:
                _remaining = max(1.0, float(timeout) - (time.time() - _started_at))
                _rc = _proc.wait(timeout=_remaining)
            else:
                _rc = _proc.wait()
        except subprocess.TimeoutExpired:
            _return_codes[_si] = "timeout"
            _failed.append(_si)
            print(f"  [shard {_si}] TIMEOUT after {timeout}s")
            _proc.terminate()
            try:
                _proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _proc.kill()
                _proc.wait()
            _rc = _proc.returncode if _proc.returncode is not None else -9
        finally:
            _log_handle.close()
        _return_codes.setdefault(_si, _rc)
        if _rc != 0:
            if _si not in _failed:
                _failed.append(_si)
            print(f"  [shard {_si}] FAILED (rc={_rc}, log={_log_path})")
        else:
            print(f"  [shard {_si}] DONE (log={_log_path})")

    # rc=0 is not sufficient: every shard must emit exactly its own stems and
    # each TextGrid must expose both MFA tiers.
    import json as _json
    _manifest_rows: list[dict] = []
    _all_missing: list[str] = []
    _all_extra: list[str] = []
    _all_invalid: list[str] = []
    for (_si, _proc, _sd, _log_handle, _log_path,
         _expected, _started_at) in _procs:
        _tg_paths = sorted((_sd / "output").glob("*.TextGrid"))
        _produced = {path.stem for path in _tg_paths}
        _missing = sorted(_expected - _produced)
        _extra = sorted(_produced - _expected)
        _invalid: list[str] = []
        for _tg in _tg_paths:
            try:
                _content = _tg.read_text(encoding="utf-8-sig")
                if ('name = "words"' not in _content
                        or 'name = "phones"' not in _content):
                    _invalid.append(_tg.stem)
            except OSError:
                _invalid.append(_tg.stem)
        _all_missing.extend(_missing)
        _all_extra.extend(_extra)
        _all_invalid.extend(_invalid)
        if _missing or _extra or _invalid:
            if _si not in _failed:
                _failed.append(_si)
        _manifest_rows.append({
            "shard": _si,
            "return_code": _return_codes.get(_si),
            "log": str(_log_path),
            "expected_count": len(_expected),
            "produced_count": len(_produced),
            "missing": _missing,
            "extra": _extra,
            "invalid": _invalid,
        })

    _manifest_path = _log_dir / "mfa_output_manifest.json"
    _manifest_path.write_text(_json.dumps({
        "schema": 1,
        "run_id": _run_id,
        "expected_total": len(stems),
        "shards": _manifest_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _missing_path = _log_dir / "mfa_missing_stems.json"
    _missing_path.write_text(_json.dumps({
        "missing": sorted(set(_all_missing)),
        "extra": sorted(set(_all_extra)),
        "invalid": sorted(set(_all_invalid)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if _failed:
        print(f"  ERROR: {len(set(_failed))}/{_n_shards} shards failed "
              f"or produced an incomplete set")
        print(f"  Manifest: {_manifest_path}")
        print(f"  Shard workspace retained: {_shard_root}")
        return 1

    # ── Merge aligned TextGrids ──
    _merged = 0
    for _sd in _shard_dirs:
        for _tg in (_sd / "output").glob("*.TextGrid"):
            _dest = aligned_dir / _tg.name
            if overwrite or not _dest.exists():
                import shutil as _shutil
                _shutil.copy2(str(_tg), str(_dest))
                _merged += 1

    _aligned_now = {path.stem for path in aligned_dir.glob("*.TextGrid")}
    _expected_all = set(stems)
    if _aligned_now != _expected_all:
        _missing_after = sorted(_expected_all - _aligned_now)
        _extra_after = sorted(_aligned_now - _expected_all)
        print("  ERROR: merged aligned set is inconsistent")
        print(f"    missing ({len(_missing_after)}): {_missing_after[:10]}")
        print(f"    extra ({len(_extra_after)}): {_extra_after[:10]}")
        print(f"  Shard workspace retained: {_shard_root}")
        return 1

    # ── Cleanup successful shard data; persistent logs/manifests remain ──
    import shutil as _shutil
    _shutil.rmtree(str(_shard_root), ignore_errors=True)

    print(f"  Sharded MFA: {_merged} TextGrids merged,"
          f" {_n_shards} shards completed")
    print(f"  MFA manifest: {_manifest_path}")
    return 0


def step_mfa_align(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """MFA align — uses NVASR .lab as corpus + CTC TextGrid as anchors.

    NVASR produces both the transcript (.lab) and the word boundaries (TextGrid)
    from the same ASR text.  This guarantees 100% word matching between corpus and
    anchors, so MFA uses every CTC word boundary for phone-level refinement.
    """
    mc = cfg["mfa"]
    # Use adjusted CTC if available (must exist AND contain .lab files),
    # fall back to raw CTC output
    ctc_dir = ctx.get("ctc_pretg_adj", ctx["ctc_pretg"])
    if not ctc_dir.exists() or not any(ctc_dir.glob("*.lab")):
        ctc_dir = ctx["ctc_pretg"]

    # Check for NVASR corpus (.lab files)
    use_nvasr_corpus = ctc_dir.exists() and any(ctc_dir.glob("*.lab"))
    corpus_dir = ctc_dir if use_nvasr_corpus else ctx["pinyin_dir"]

    # Clean temp dir when overwriting — only remove alignment DB, keep feature cache
    import shutil
    if args.overwrite:
        # Only clean alignment outputs, preserve MFCC feature cache in temp_dir
        if ctx["aligned_dir"].exists():
            shutil.rmtree(ctx["aligned_dir"], ignore_errors=True)
            ctx["aligned_dir"].mkdir(parents=True, exist_ok=True)
        # Remove stale MFA sqlite DBs (they reference old alignment state)
        if ctx["temp_dir"].exists():
            for db_file in ctx["temp_dir"].glob("*.db"):
                try:
                    db_file.unlink(missing_ok=True)
                except Exception:
                    pass

    if not list(corpus_dir.glob("*.lab" if use_nvasr_corpus else "*.txt")):
        print("ERROR: No corpus files found.")
        return 1

    # Check for CTC anchors
    use_anchors = ctc_dir.exists() and any(ctc_dir.glob("*.TextGrid"))

    if use_nvasr_corpus:
        print(f"  NVASR corpus: {ctc_dir} (.lab files from ASR text)")
    if use_anchors:
        print(f"  CTC anchors:  {ctc_dir}")
        print(f"  Transcript and anchors from SAME source -> 100% word match")

    # Use pre-extracted directory if available — avoids MFA Archive.__init__
    # deleting and re-extracting the zip (which races with parallel workers).
    extracted_acoustic = ctx["models_dir"] / "extracted_models" / "acoustic" / f"{cfg['acoustic_model']}_acoustic"
    acoustic_model_arg2 = str(extracted_acoustic) if extracted_acoustic.is_dir() else cfg["acoustic_model"]

    mfa_args = [
        "align", str(corpus_dir), str(ctx["mfa_dict"]),
        acoustic_model_arg2, str(ctx["aligned_dir"]),
        "--audio_directory", str(ctx["mfa_audio_dir"]),
        "--temporary_directory", str(ctx["temp_dir"]),
        "--output_format", mc.get("output_format", "long_textgrid"),
        "--num_jobs", str(resolve_num_jobs(mc.get("num_jobs", 0))),
        "--overwrite", "--no_textgrid_cleanup",
    ]
    if use_anchors:
        mfa_args += ["--textgrid_directory", str(ctc_dir)]
    if mc.get("clean"):
        mfa_args.append("--clean")
    if mc.get("single_speaker"):
        mfa_args.append("--single_speaker")
    if mc.get("no_tokenization"):
        mfa_args.append("--no_tokenization")

    # ── Kaldi alignment parameters ──
    # beam: Viterbi beam width (default 10). Wider = more paths explored, fewer failures.
    mfa_args += ["--beam", str(mc.get("beam", 20))]
    # retry_beam: beam width for retry on failure (default 40). Wider = more rescue attempts.
    mfa_args += ["--retry_beam", str(mc.get("retry_beam", 80))]
    # boost_silence: silence probability multiplier in HMM (default 1.0). >1 -> prefer silence.
    mfa_args += ["--boost_silence", str(mc.get("boost_silence", 1.0))]
    # acoustic_scale: weight of acoustic vs transition scores (default 0.1). Lower = looser constraints.
    if mc.get("acoustic_scale") is not None:
        mfa_args += ["--acoustic_scale", str(mc["acoustic_scale"])]
    # transition_scale: weight of transition probabilities (default 1.0).
    if mc.get("transition_scale") is not None:
        mfa_args += ["--transition_scale", str(mc["transition_scale"])]

    # ── Fine-tune: allows CTC anchor boundaries to float during a refinement pass ──
    # DISABLED by default: adjust_ctc_boundaries already refines anchors via
    # energy analysis; MFA fine_tune floats boundaries toward its acoustic
    # model (trained on clean speech) and degrades alignment on NVV, BGM,
    # English tokens, and short function words.  See Regression Case 16.
    # NOTE: --fine_tune is a FLAG (presence = on).  MFA 3.3.9
    # pretrained.py:92 has a typo bug (fine_tune overwritten by
    # fine_tune_boundary_tolerance) — patched in this env.
    # See Regression Case 16 for why fine_tune defaults to off.
    if mc.get("fine_tune", False):
        mfa_args.append("--fine_tune")
        fine_tune_tolerance = mc.get("fine_tune_boundary_tolerance", 0.02)
        if fine_tune_tolerance is not None and fine_tune_tolerance > 0:
            mfa_args += ["--fine_tune_boundary_tolerance", str(fine_tune_tolerance)]

    # ── Try sharded MFA (parallel MFCC extraction) ──
    _mfa_audio_dir = ctx["mfa_audio_dir"]
    _stems = sorted(p.stem for p in corpus_dir.glob("*.lab"))
    _frozen = set(ctx.get("expected_stems", ()))
    if _frozen and (set(_stems) != _frozen or len(_stems) != len(_frozen)):
        print("  ERROR: MFA corpus differs from frozen strict denominator")
        print(f"    missing ({len(_frozen - set(_stems))}): "
              f"{sorted(_frozen - set(_stems))[:10]}")
        print(f"    extra ({len(set(_stems) - _frozen)}): "
              f"{sorted(set(_stems) - _frozen)[:10]}")
        return 1
    _missing_audio = [
        stem for stem in _stems
        if not (_mfa_audio_dir / f"{stem}.wav").exists()
    ]
    _missing_anchors = [
        stem for stem in _stems
        if use_anchors and not (ctc_dir / f"{stem}.TextGrid").exists()
    ]
    if _missing_audio or _missing_anchors:
        print("  ERROR: MFA input set is incomplete")
        if _missing_audio:
            print(f"    missing audio ({len(_missing_audio)}): {_missing_audio[:10]}")
        if _missing_anchors:
            print(f"    missing anchors ({len(_missing_anchors)}): "
                  f"{_missing_anchors[:10]}")
        return 1
    if ctx.get("strict_ready") and (ctx["aligned_dir"].exists()
                                    or ctx["aligned_dir"].is_symlink()):
        print(f"  ERROR: strict aligned target preexists: {ctx['aligned_dir']}")
        return 1
    ctx["aligned_dir"].mkdir(parents=True, exist_ok=not ctx.get("strict_ready", False))
    ctx["temp_dir"].mkdir(parents=True, exist_ok=True)
    _extra = {}
    if mc.get("acoustic_scale") is not None:
        _extra["acoustic_scale"] = mc["acoustic_scale"]
    if mc.get("transition_scale") is not None:
        _extra["transition_scale"] = mc["transition_scale"]

    _rc = _run_mfa_sharded(
        stems=_stems,
        corpus_dir=corpus_dir,
        audio_dir=_mfa_audio_dir,
        anchors_dir=ctc_dir if use_anchors else None,
        dict_path=ctx["mfa_dict"],
        acoustic_model=acoustic_model_arg2,
        aligned_dir=ctx["aligned_dir"],
        workspace=ctx["workspace"],
        mfa_python=mfa_python,
        models_dir=ctx["models_dir"],
        num_jobs=resolve_num_jobs(mc.get("num_jobs", 0)),
        single_speaker=mc.get("single_speaker", False),
        no_tokenization=mc.get("no_tokenization", False),
        beam=mc.get("beam", 20),
        retry_beam=mc.get("retry_beam", 80),
        boost_silence=mc.get("boost_silence", 1.0),
        clean=mc.get("clean", False),
        overwrite=args.overwrite,
        output_format=mc.get("output_format", "long_textgrid"),
        timeout=mc.get("timeout"),
        desc="Step 6: MFA Align (sharded)",
        **_extra,
    )
    if _rc is not None:
        return _rc
    # Fallback: single MFA instance
    _rc = run_mfa(
        mfa_args, mfa_python, ctx["models_dir"],
        "Step 6: MFA Align"
        + (" (NVASR corpus + CTC anchors)"
           if use_nvasr_corpus and use_anchors else ""),
        timeout=mc.get("timeout"),
    )
    if _rc != 0:
        return _rc
    _produced = {path.stem for path in ctx["aligned_dir"].glob("*.TextGrid")}
    _expected = set(_stems)
    _missing = sorted(_expected - _produced)
    _extra = sorted(_produced - _expected)
    if _missing or _extra:
        print("  ERROR: single-process MFA output set is incomplete")
        if _missing:
            print(f"    missing ({len(_missing)}): {_missing[:10]}")
        if _extra:
            print(f"    extra ({len(_extra)}): {_extra[:10]}")
        return 1
    print(f"  MFA output set: {len(_produced)}/{len(_expected)} complete")
    return 0


def step_mfa_align_en(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """English MFA alignment — processes English word segments with english_us_arpa.

    Extracts English word audio from CTC boundaries, runs MFA with the
    english_us_arpa acoustic model (ARPABET phone set), and writes per-stem
    *_en_phones.json files.
    When no English words are found in the corpus, this step is a no-op.
    """
    en_cfg = cfg.get("mfa_en", {})
    if not en_cfg.get("enabled", True):
        print("  English MFA: disabled (mfa_en.enabled=false)")
        return 0

    ctc_dir = ctx.get("ctc_pretg_adj", ctx["ctc_pretg"])
    if not ctc_dir.exists() or not any(ctc_dir.iterdir()):
        ctc_dir = ctx["ctc_pretg"]
    audio_dir = ctx["mfa_audio_dir"]
    output_dir = ctx["workspace"] / "en_phones"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve English model paths
    en_acoustic = en_cfg.get("acoustic_model", str(PROJECT_ROOT / "pretrained_models" / "acoustic" / "english_us_arpa.zip"))
    en_dict = en_cfg.get("dictionary", str(PROJECT_ROOT / "dict" / "cmudict.dict"))
    en_g2p = en_cfg.get("g2p_model", str(PROJECT_ROOT / "pretrained_models" / "g2p" / "english_us_arpa.zip"))

    # Resolve relative paths, with fallback to PROJECT_ROOT.parent/pretrained_models
    for val, key in [(en_acoustic, "acoustic_model"), (en_dict, "dictionary"), (en_g2p, "g2p_model")]:
        p = Path(val)
        if not p.is_absolute():
            resolved = PROJECT_ROOT / val
            # Fallback: pretrained_models may live one level above the repo
            # (e.g. /mnt/project/MFA_Pause/pretrained_models/ instead of
            #  /mnt/project/MFA_Pause/repo/pretrained_models/)
            if not resolved.exists() and "pretrained_models" in val:
                _parent_resolved = PROJECT_ROOT.parent / val
                if _parent_resolved.exists():
                    resolved = _parent_resolved
            if key == "acoustic_model":
                en_acoustic = str(resolved)
            elif key == "dictionary":
                en_dict = str(resolved)
            else:
                en_g2p = str(resolved)

    temp_dir = ctx["temp_dir"] / "en_mfa"
    temp_dir.mkdir(parents=True, exist_ok=True)

    align_en_args = [
        "--ctc-dir", str(ctc_dir),
        "--audio-dir", str(audio_dir),
        "--output-dir", str(output_dir),
        "--acoustic-model", en_acoustic,
        "--dictionary", en_dict,
        "--g2p-model", en_g2p,
        "--temp-dir", str(temp_dir),
        "--num-jobs", str(resolve_num_jobs(en_cfg.get("num_jobs", 4))),
        "--padding-ms", str(en_cfg.get("padding_ms", 50)),
        "--min-segment-dur-ms", str(en_cfg.get("min_segment_dur_ms", 200)),
        "--max-gap-merge-s", str(en_cfg.get("max_gap_merge_s", 0.35)),
        "--beam", str(en_cfg.get("beam", 10)),
        "--retry-beam", str(en_cfg.get("retry_beam", 40)),
        "--timeout", str(en_cfg.get("timeout", 1800)),
        "--g2p-timeout", str(en_cfg.get("g2p_timeout", 300)),
    ]
    if en_cfg.get("strict_provenance", True):
        align_en_args.append("--strict-provenance")
    if en_cfg.get("fine_tune", False):
        align_en_args.append("--fine-tune")
    cw = en_cfg.get("corpus_workers", 0)
    if cw > 0:
        align_en_args += ["--corpus-workers", str(cw)]
    if args.python:
        align_en_args += ["--python", str(mfa_python)]

    script = SCRIPTS_DIR / "align_english_mfa.py"
    outer_timeout = (en_cfg.get("timeout", 1800)
                     + en_cfg.get("g2p_timeout", 300)
                     + en_cfg.get("preparation_timeout_margin", 900))
    return run_python(script, align_en_args, mfa_python, ctx["models_dir"],
                      desc="English MFA Alignment", timeout=outer_timeout)


def step_postprocess(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Post-process MFA aligned TextGrids.

    NVV and punctuation are self-referential in the MFA dictionary,
    so MFA preserves them natively — no post-injection needed.
    """
    pc = cfg["postprocess"]
    ctc_dir = ctx.get("ctc_pretg_adj", ctx["ctc_pretg"])  # use adjusted if available
    if not ctc_dir.exists() or not any(ctc_dir.glob("*.lab")):
        ctc_dir = ctx["ctc_pretg"]
    aligned_dir = ctx["aligned_dir"]

    # Postprocess must use the corpus denominator, not merely whichever aligned
    # files happen to exist.  Otherwise a partial MFA run silently disappears
    # from the report (the 0805 run lost 139 stems this way).
    expected_stems = set(ctx.get("expected_stems", ()))
    if not expected_stems:
        corpus_stems = {p.stem for p in ctc_dir.glob("*.lab")}
        audio_stems = {p.stem for p in ctx["mfa_audio_dir"].glob("*.wav")}
        if corpus_stems and audio_stems:
            expected_stems = corpus_stems | audio_stems
            print(f"  (using lab+audio union as denominator: {len(expected_stems)} stems)")
        else:
            expected_stems = corpus_stems or audio_stems
    corpus_stems = {p.stem for p in ctc_dir.glob("*.lab")}
    audio_stems = {p.stem for p in ctx["mfa_audio_dir"].glob("*.wav")}
    aligned_stems = {p.stem for p in aligned_dir.glob("*.TextGrid")}
    missing_audio = sorted(expected_stems - audio_stems)
    missing_aligned = sorted(expected_stems - aligned_stems)
    unexpected_aligned = sorted(aligned_stems - expected_stems)
    if (not expected_stems or corpus_stems != expected_stems
            or missing_audio or missing_aligned or unexpected_aligned):
        print("  ERROR: postprocess input set is incomplete/inconsistent")
        print(f"    expected labs:      {len(expected_stems)}")
        if corpus_stems != expected_stems:
            print(f"    corpus mismatch: missing={len(expected_stems - corpus_stems)}, "
                  f"extra={len(corpus_stems - expected_stems)}")
        print(f"    available audio:    {len(audio_stems)}")
        print(f"    aligned TextGrids:  {len(aligned_stems)}")
        if missing_audio:
            print(f"    missing audio ({len(missing_audio)}): {missing_audio[:10]}")
        if missing_aligned:
            print(f"    missing aligned ({len(missing_aligned)}): {missing_aligned[:10]}")
        if unexpected_aligned:
            print(f"    unexpected aligned ({len(unexpected_aligned)}): "
                  f"{unexpected_aligned[:10]}")
        return 1

    pp_args = [
        "--txt-dir", str(ctc_dir),
        "--textgrid-dir", str(aligned_dir),
        "--output-dir", str(ctx["output_dir"]),
        "--filtered-dir", str(ctx["filtered_dir"]),
        "--wav-dir", str(ctx["mfa_audio_dir"]),
        "--raw-text-dir", str(ctx.get("raw_text_dir", ctx["data_dir"])),
        "--original-txt-dir", str(ctx.get("raw_text_dir", ctx["data_dir"])),
        "--pinyin-dict", str(resolve_path(PROJECT_ROOT, cfg.get("pinyin_dict", "dict/fullpinyin_enword.dict"))),
        "--ipa-dict", str(ctx.get(
            "mfa_dict", resolve_path(PROJECT_ROOT, cfg.get("mfa_dict", "dict/mfa_ipa.dict")))),
        "--en-phones-dir", str(ctx["workspace"] / "en_phones"),
        "--tone-ref", str(ctx["output_dir"] / "tone_mapping.json"),
    ]
    # Silence merge
    if pc.get("merge_silence", True):
        pp_args += ["--merge-max-sil-sec", str(pc.get("min_sil_merge_sec", 0.2))]
    else:
        pp_args.append("--no-merge-silence")
    # Short word fix
    if pc.get("fix_short_word", True):
        pp_args += ["--fix-short-word-sec", str(pc.get("short_word_max_sec", 0.25))]
        pp_args += ["--fix-min-silence-sec", str(pc.get("flank_silence_sec", 0.4))]
        pp_args += ["--fix-search-sec", str(pc.get("short_word_search_window", 0.5))]
    else:
        pp_args.append("--no-fix-short-word")
    # BGM detection
    if pc.get("detect_bgm", True):
        pp_args += ["--bgm-noise-floor-ratio", str(pc.get("bgm_noise_floor_ratio", 2.0))]
        pp_args += ["--bgm-min-sil-dur", str(pc.get("bgm_min_sil_dur", 0.3))]
        pp_args += ["--bgm-speech-ratio", str(pc.get("bgm_speech_ratio", 1.0))]
        pp_args += ["--bgm-min-energy", str(pc.get("bgm_min_energy", 0.01))]
        pp_args += ["--bgm-max-threshold", str(pc.get("bgm_max_threshold", 0.05))]
    else:
        pp_args.append("--no-detect-bgm")
    # Quality filters
    if pc.get("filter_suspicious", True):
        if pc.get("filter_short_phone", True):
            pp_args += ["--filter-short-phone-sec", str(pc.get("filter_short_phone_sec", 0.015))]
        else:
            pp_args.append("--no-filter-short-phone")
        pp_args += ["--filter-long-word-sec", str(pc.get("filter_long_word_sec", 1.0))]
        pp_args += ["--filter-min-word-sec", str(pc.get("filter_min_word_sec", 0.15))]
        pp_args += ["--filter-min-word-dur-sec", str(pc.get("filter_min_word_dur_sec", 0.02))]
        if pc.get("enable_word_in_silence_filter", False):
            pp_args += ["--filter-word-energy-ratio", str(pc.get("filter_word_energy_ratio", 2.0))]
        else:
            pp_args += ["--filter-word-energy-ratio", "0"]
        pp_args += ["--filter-min-phone-coverage", str(pc.get("filter_min_phone_coverage", 0.35))]
        pp_args += ["--filter-edge-gap-sec", str(pc.get("filter_edge_gap_sec", 0.25))]
        pp_args += ["--filter-flank-silence-sec", str(pc.get("filter_flank_silence_sec", 0.4))]
        pp_args += ["--filter-long-consonant-sec", str(pc.get("filter_long_consonant_sec", 999.0))]
        pp_args += ["--filter-long-vowel-sec", str(pc.get("filter_long_vowel_sec", 999.0))]
        pp_args += ["--filter-short-phone-en-sec", str(pc.get("filter_short_phone_en_sec", 0.010))]
        pp_args += ["--filter-long-vowel-en-sec", str(pc.get("filter_long_vowel_en_sec", 0.500))]
        pp_args += ["--filter-long-consonant-en-sec", str(pc.get("filter_long_consonant_en_sec", 1.000))]
        pp_args += ["--filter-min-en-phone-coverage", str(pc.get("filter_min_en_phone_coverage", 0.25))]
    else:
        pp_args.append("--no-filter-suspicious")
    if pc.get("strict_ok", True):
        pp_args.append("--strict-ok")
    # Text correction & unexpected silence handling
    if not pc.get("enable_text_correction", True):
        pp_args.append("--no-enable-text-correction")
    if not pc.get("handle_unexpected_sil", True):
        pp_args.append("--no-handle-unexpected-sil")
    if pc.get("workers", 0) > 0:
        pp_args += ["--workers", str(pc["workers"])]
    if args.overwrite:
        pp_args.append("--overwrite")
    rc = run_python(SCRIPTS_DIR / "postprocess_textgrids.py", pp_args, mfa_python,
                    ctx["models_dir"], "Step 7: Post-processing")
    if rc != 0:
        return rc

    # Validate the complete publication set before the caller can sync it.
    import json as _json
    output_stems = {p.stem for p in ctx["output_dir"].glob("*.TextGrid")}
    filtered_stems = {p.stem for p in ctx["filtered_dir"].glob("*.TextGrid")}
    overlap = sorted(output_stems & filtered_stems)
    combined = output_stems | filtered_stems
    missing_result = sorted(expected_stems - combined)
    unexpected_result = sorted(combined - expected_stems)

    report_path = ctx["output_dir"] / "postprocess_report.jsonl"
    report_stems: list[str] = []
    report_invalid = 0
    if report_path.exists():
        for line_no, line in enumerate(
                report_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = _json.loads(line)
                report_stems.append(row["stem"])
            except (KeyError, _json.JSONDecodeError):
                report_invalid += 1
                print(f"  ERROR: invalid report row {line_no}")
    else:
        report_invalid = 1
        print(f"  ERROR: missing postprocess report: {report_path}")

    tone_path = ctx["output_dir"] / "tone_mapping.json"
    tone_valid = False
    try:
        tone_data = _json.loads(tone_path.read_text(encoding="utf-8"))
        tone_valid = isinstance(tone_data, dict) and bool(tone_data)
    except (OSError, _json.JSONDecodeError):
        pass

    report_set = set(report_stems)
    contract_failed = (
        bool(overlap or missing_result or unexpected_result or report_invalid)
        or report_set != expected_stems
        or len(report_stems) != len(report_set)
        or not tone_valid
    )
    if contract_failed:
        print("  ERROR: postprocess output contract failed")
        if overlap:
            print(f"    output/filtered overlap ({len(overlap)}): {overlap[:10]}")
        if missing_result:
            print(f"    missing result ({len(missing_result)}): {missing_result[:10]}")
        if unexpected_result:
            print(f"    unexpected result ({len(unexpected_result)}): "
                  f"{unexpected_result[:10]}")
        if report_set != expected_stems or len(report_stems) != len(report_set):
            print(f"    report stems: {len(report_set)} unique / "
                  f"{len(report_stems)} rows / {len(expected_stems)} expected")
        if not tone_valid:
            print(f"    invalid/missing tone mapping: {tone_path}")
        return 1

    print(f"  Postprocess contract: {len(expected_stems)} stems, "
          f"{len(output_stems)} output, {len(filtered_stems)} filtered")
    return 0

def step_strict_ok(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Re-audit final candidates before a strict-ok manifest may exist."""
    ctc_dir = ctx.get("ctc_pretg_adj", ctx["ctc_pretg"])
    if not ctc_dir.exists() or not any(ctc_dir.glob("*.lab")):
        ctc_dir = ctx["ctc_pretg"]
    manifest = ctx["output_dir"] / "strict_ok_manifest.json"
    strict_args = [
        "--output-dir", str(ctx["output_dir"]),
        "--filtered-dir", str(ctx["filtered_dir"]),
        "--ctc-dir", str(ctc_dir),
        "--reference-dir", str(ctx["raw_text_dir"]),
        "--wav-dir", str(ctx["mfa_audio_dir"]),
        "--aligned-dir", str(ctx["aligned_dir"]),
        "--en-phones-dir", str(ctx["workspace"] / "en_phones"),
        "--en-aligned-dir", str(ctx["temp_dir"] / "en_mfa" / "en_aligned"),
        "--en-manifest", str(ctx["workspace"] / "en_phones" / "en_alignment_manifest.json"),
        "--report", str(ctx["output_dir"] / "postprocess_report.jsonl"),
        "--manifest", str(manifest),
    ]
    return run_python(SCRIPTS_DIR / "audit_strict_ok.py", strict_args, mfa_python,
                      ctx["models_dir"], "Step 8: strict-ok independent audit")


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_step_output(step_name: str, workspace: Path, spec: dict) -> list[str]:
    """Check that expected output files exist for *step_name*.

    Returns a list of failure descriptions (empty = all OK).
    """
    patterns = spec.get(step_name, [])
    if not patterns:
        return []

    failures: list[str] = []
    for pattern in patterns:
        matches = list(workspace.glob(pattern))
        if not matches:
            failures.append(f"  MISSING: {pattern}")
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scan cache -- persist directory scan results to skip re-scanning
# ---------------------------------------------------------------------------

CACHE_VERSION = 1


def _get_cache_dir(config_path: Path, cache_dir_override: str | None = None) -> Path:
    """Resolve the cache directory for *config_path*."""
    if cache_dir_override:
        p = Path(cache_dir_override)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return PROJECT_ROOT / "cache"


def _get_cache_path(config_path: Path, cache_dir: Path) -> Path:
    """Cache file path for *config_path* (e.g. ``batch_all.cache.json``)."""
    return cache_dir / f"{config_path.stem}.cache.json"


def load_scan_cache(cache_path: Path) -> dict | None:
    """Load scan cache if it exists and version matches.  Returns None on miss."""
    if not cache_path.exists():
        return None
    try:
        import json as _j
        data = _j.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            print(f"  Cache version mismatch ({data.get('version')} != {CACHE_VERSION}), ignoring.")
            return None
        print(f"  Loaded scan cache: {cache_path}")
        return data
    except Exception as e:
        print(f"  Failed to load cache {cache_path}: {e}")
        return None


def save_scan_cache(cache_path: Path, cache_data: dict) -> None:
    """Persist scan cache to disk (creates parent directory if needed)."""
    import json as _j
    import datetime as _dt
    cache_data.setdefault("version", CACHE_VERSION)
    cache_data["scanned_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        _j.dumps(cache_data, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  Scan cache saved: {cache_path}")


# ---------------------------------------------------------------------------
# File-index helpers — imported from pipeline_utils
#   build_file_index, build_ctc_presence, count_files_fast, find_wav
# ---------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path) -> bool:
    """Link *src* -> *dst* with least-cost strategy.

    Strategy (tried in order):
      1. os.symlink  — works cross-device, near-zero I/O
      2. os.link     — same-device hard link (instant, zero space)
      3. shutil.copy2 — fallback when both fail

    Returns True on success, False if *src* does not exist.
    """
    if not src.exists():
        return False
    if src.resolve() == dst.resolve():
        return True    # same file — nothing to do
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(str(src), str(dst))
        return True
    except OSError:
        pass
    try:
        os.link(str(src), str(dst))
        return True
    except OSError:
        pass
    import shutil, time as _t
    for _i in range(3):
        try:
            shutil.copy2(str(src), str(dst))
            return True
        except (OSError, FileNotFoundError):
            if _i < 2:
                _t.sleep(0.3)
                dst.parent.mkdir(parents=True, exist_ok=True)
    return False


# ---------------------------------------------------------------------------
# Step: link (ctc_ready mode) — validate pre-existing NVASR output
# ---------------------------------------------------------------------------

_CTC_SUFFIXES = [
    ".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
    "_text_cn.txt", "_text_raw.txt",
]

STRICT_READY_SCHEMA = "hecheng-english-ctc-ready-v4"
STRICT_READY_COUNT = 53998
STRICT_READY_ACTION = "acoustic_rerun"
STRICT_READY_FINAL_AUDIO_AXIS = "authoritative_wav"
STRICT_READY_PADDING_POLICY = "forbidden"
STRICT_READY_VERIFIER_SIGNATURE = "ctc-ready-independent-v1"
STRICT_READY_AUTHORITATIVE_SOURCE = Path("/mnt/Raw/新版合成英文数据")
STRICT_READY_SOURCE_DICTIONARY = PROJECT_ROOT / "dict" / "mfa_ipa.dict"
STRICT_READY_MISSING_REFERENCES = [
    "024198_杂谈互动_数据里程牌庆祝",
    "036000_弹幕互动_回应吐槽弹幕",
]
STRICT_READY_VERIFY_HOOK = None  # synthetic tests may inject a read-only verifier

_STRICT_READY_TOP_LEVEL_KEYS = {
    "schema", "state", "independent_verifier_signature",
    "prepare_manifest_sha256", "inventory_sha256", "authoritative_stems",
    "stem_count", "missing_reference", "txt_only", "final_audio_axis",
    "padding_policy", "action_counts", "taxonomy", "taxonomy_sha256",
    "roots", "source_dictionary", "run_local_dictionary", "artifacts",
    "rerun_files", "rerun_files_sha256",
}


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _stable_json_sha256(value: object) -> str:
    """Match the v4 preparation/verifier canonical JSON digest."""
    import hashlib
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_symlink_components(path: Path) -> Path:
    """Return an absolute lexical path after rejecting every symlink component."""
    raw = Path(os.path.abspath(path))
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"symlink directory component forbidden: {cursor}")
    return raw


def _strict_directory(path: Path) -> Path:
    """Return a canonical directory only when no path component is a symlink."""
    raw = _reject_symlink_components(path)
    if not raw.is_dir():
        raise ValueError(f"ordinary directory required: {raw}")
    return raw.resolve(strict=True)


def _strict_regular_file(path: Path, root: Path) -> Path:
    """Resolve an ordinary file below a trusted root; reject all symlinks."""
    root_abs = _strict_directory(root)
    candidate = Path(os.path.abspath(path))
    try:
        rel = candidate.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {path}") from exc
    cursor = root_abs
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"symlink component forbidden: {cursor}")
    if not candidate.is_file():
        raise ValueError(f"ordinary file required: {candidate}")
    return candidate.resolve(strict=True)


def _validate_exact_regular_namespace(root: Path, expected_names: set[str], label: str) -> None:
    root = _strict_directory(root)
    entries = list(root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{label} namespace contains non-ordinary entries")
    actual = {entry.name for entry in entries}
    if actual != expected_names or len(actual) != len(entries):
        raise ValueError(f"{label} namespace is not exact")


def _strict_ready_mode(cfg: dict) -> bool:
    return isinstance(cfg.get("ctc_ready", {}).get("expected_ready_evidence"), dict)


def validate_strict_ready_invocation(args, cfg: dict, mode: str, use_cache: bool) -> Path:
    """Reject every CLI/config route that could bypass a fresh strict import."""
    if not _strict_ready_mode(cfg):
        return Path()
    cr = cfg.get("ctc_ready", {})
    pin = cr.get("expected_ready_evidence", {})
    pin_hash = pin.get("sha256")
    taxonomy_hash = pin.get("taxonomy_sha256")
    if not _is_sha256_hex(pin_hash) or not _is_sha256_hex(taxonomy_hash):
        raise ValueError(
            "strict v4 evidence SHA256 and taxonomy SHA256 must be finalized")
    forbidden_flags = ["force", "overwrite", "use_cache", "auto_cache", "scan_only",
                       "output_staging"]
    active = [name for name in forbidden_flags if getattr(args, name, False)]
    active += [f"skip_{name}" for name in STEPS if getattr(args, f"skip_{name}", False)]
    if (mode != "ctc_ready" or active or use_cache
            or getattr(args, "step", None) or getattr(args, "skip_to", None)
            or getattr(args, "stop_after", None)
            or getattr(args, "ctc_ready", None) or getattr(args, "data_dir", None)
            or getattr(args, "nvme_cache", None) or getattr(args, "output_dir", None)
            or getattr(args, "cache_dir", None)
            or getattr(args, "dataset_offset", 0) or getattr(args, "dataset_limit", 0)
            or cr.get("stems") is not None or cr.get("stem_range") is not None
            or cr.get("require_all") is not True or cr.get("isolate_copy") is not True
            or cr.get("expected_count") != STRICT_READY_COUNT
            or cr.get("require_fresh_workspace") is not True
            or Path(str(cr.get("authoritative_source_dir", "")))
            != STRICT_READY_AUTHORITATIVE_SOURCE
            or Path(str(cr.get("source_dictionary", "")))
            != STRICT_READY_SOURCE_DICTIONARY
            or cfg.get("runtime_mfa_dict") != "runtime/mfa_ipa.dict"
            or cfg.get("disable_nvme_cache") is not True
            or cfg.get("use_cache") is not False
            or cfg.get("output_staging") is not False
            or cfg.get("pad_silence", {}).get("enabled") is not False
            or cfg.get("ctc_adjust", {}).get("enabled") is not True
            or cfg.get("ctc_adjust", {}).get("limit", 0) != 0
            or cfg.get("postprocess", {}).get("strict_ok") is not True
            or cfg.get("mfa_en", {}).get("enabled") is not True
            or cfg.get("mfa_en", {}).get("strict_provenance") is not True
            or set(pin) != {"path", "sha256", "schema", "state",
                            "taxonomy_sha256", "independent_verifier_signature"}
            or pin.get("schema") != STRICT_READY_SCHEMA or pin.get("state") != "ready"
            or pin.get("independent_verifier_signature") != STRICT_READY_VERIFIER_SIGNATURE
            or not Path(str(pin.get("path", ""))).is_absolute()
            or not _is_sha256_hex(pin_hash) or not _is_sha256_hex(taxonomy_hash)):
        raise ValueError(f"strict CTC-ready invocation/config bypass rejected: {active}")
    raw = getattr(args, "workspace", None)
    if raw:
        workspace = Path(raw); workspace = workspace if workspace.is_absolute() else PROJECT_ROOT / workspace
    else:
        workspace = Path(cfg.get("workspace", "default"))
        if not workspace.is_absolute():
            workspace = PROJECT_ROOT / "output" / workspace
    workspace = _reject_symlink_components(workspace)
    if workspace.exists() or workspace.is_symlink():
        raise ValueError(f"strict workspace must not preexist: {workspace}")
    return workspace


def _strict_hash_record(record: object, expected_path: Path, trusted_root: Path,
                        *, wav: bool = False) -> dict:
    expected_keys = {"path", "size", "sha256"}
    if wav:
        expected_keys.add("wav")
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError(f"artifact evidence incomplete: {expected_path}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"artifact path must be absolute: {expected_path}")
    path = _strict_regular_file(Path(raw_path), trusted_root)
    if path != expected_path.resolve(strict=True):
        raise ValueError(f"artifact path redirected: {path}")
    if (type(record["size"]) is not int or record["size"] <= 0
            or path.stat().st_size != record["size"]):
        raise ValueError(f"artifact size mismatch: {path}")
    if not _is_sha256_hex(record["sha256"]):
        raise ValueError(f"artifact hash invalid: {path}")
    normalized = {"path": path, "size": record["size"], "sha256": record["sha256"]}
    if wav:
        import math
        metadata = record["wav"]
        if not isinstance(metadata, dict) or set(metadata) != {
                "frames", "sample_rate", "channels", "duration_s"}:
            raise ValueError(f"WAV evidence incomplete: {path}")
        if (any(type(metadata[key]) is not int or metadata[key] <= 0
                for key in ("frames", "sample_rate", "channels"))
                or isinstance(metadata["duration_s"], bool)
                or not isinstance(metadata["duration_s"], (int, float))
                or not math.isfinite(float(metadata["duration_s"]))
                or abs(float(metadata["duration_s"])
                       - metadata["frames"] / metadata["sample_rate"]) > 1e-9):
            raise ValueError(f"WAV metadata invalid: {path}")
        normalized["wav"] = dict(metadata)
    return normalized


def _strict_external_hash_record(record: object, trusted_root: Path | None = None,
                                 *, wav: bool = False) -> dict:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("authoritative artifact evidence incomplete")
    path = Path(record["path"])
    if not path.is_absolute():
        raise ValueError(f"authoritative artifact path must be absolute: {path}")
    return _strict_hash_record(
        record, path, trusted_root if trusted_root is not None else path.parent,
        wav=wav)


def _same_artifact_content(left: dict, right: dict, *, wav: bool = False) -> bool:
    keys = ["size", "sha256"] + (["wav"] if wav else [])
    return all(left.get(key) == right.get(key) for key in keys)


def load_and_validate_ready_evidence(cfg: dict) -> dict:
    """Validate the pinned v4 rerun lineage before importing any CTC file."""
    cr = cfg.get("ctc_ready", {})
    pin = cr.get("expected_ready_evidence", {})
    if (set(pin) != {"path", "sha256", "schema", "state", "taxonomy_sha256",
                    "independent_verifier_signature"}
            or not _is_sha256_hex(pin.get("sha256"))
            or not _is_sha256_hex(pin.get("taxonomy_sha256"))):
        raise ValueError("ready evidence/taxonomy SHA256 pins are not finalized")
    if (pin.get("schema") != STRICT_READY_SCHEMA or pin.get("state") != "ready"
            or pin.get("independent_verifier_signature")
            != STRICT_READY_VERIFIER_SIGNATURE):
        raise ValueError("configured ready evidence contract invalid")

    evidence_path = Path(str(pin.get("path", "")))
    if not evidence_path.is_absolute():
        raise ValueError("ready evidence path must be absolute")
    run_root = _strict_directory(evidence_path.parent)
    evidence_path = _strict_regular_file(evidence_path, run_root)
    if evidence_path != (run_root / "ctc_ready_evidence.json").resolve(strict=True):
        raise ValueError("ready evidence filename/root invalid")
    if _sha256_file(evidence_path) != pin["sha256"]:
        raise ValueError("ready evidence pinned hash mismatch")
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _STRICT_READY_TOP_LEVEL_KEYS:
        raise ValueError("ready evidence top-level namespace invalid")
    if (payload["schema"] != STRICT_READY_SCHEMA or payload["state"] != "ready"
            or payload["independent_verifier_signature"]
            != STRICT_READY_VERIFIER_SIGNATURE):
        raise ValueError("ready evidence schema/state/signature mismatch")

    stems = payload["authoritative_stems"]
    expected_count = cr.get("expected_count")
    if (expected_count != STRICT_READY_COUNT or payload["stem_count"] != STRICT_READY_COUNT
            or not isinstance(stems, list) or len(stems) != STRICT_READY_COUNT
            or stems != sorted(stems) or len(set(stems)) != len(stems)
            or not all(isinstance(stem, str) and stem and Path(stem).name == stem
                       for stem in stems)):
        raise ValueError("ready evidence stem denominator invalid")
    if (payload["missing_reference"] != STRICT_READY_MISSING_REFERENCES
            or payload["txt_only"] != []):
        raise ValueError("ready evidence named source exclusions invalid")
    expected_actions = {STRICT_READY_ACTION: STRICT_READY_COUNT}
    if payload["action_counts"] != expected_actions:
        raise ValueError("ready evidence action counts invalid")
    if (payload["final_audio_axis"] != STRICT_READY_FINAL_AUDIO_AXIS
            or payload["padding_policy"] != STRICT_READY_PADDING_POLICY):
        raise ValueError("ready evidence audio-axis/padding contract invalid")

    taxonomy = payload["taxonomy"]
    if (not isinstance(taxonomy, list) or len(taxonomy) != STRICT_READY_COUNT
            or any(not isinstance(row, dict)
                   or set(row) != {"stem", "reason", "action"}
                   or row["stem"] != stem or row["action"] != STRICT_READY_ACTION
                   or not isinstance(row["reason"], str) or not row["reason"]
                   for stem, row in zip(stems, taxonomy))):
        raise ValueError("ready evidence taxonomy invalid")
    taxonomy_sha = _stable_json_sha256(taxonomy)
    if (payload["taxonomy_sha256"] != taxonomy_sha
            or pin["taxonomy_sha256"] != taxonomy_sha):
        raise ValueError("ready evidence taxonomy digest mismatch")
    for field in ("prepare_manifest_sha256", "inventory_sha256",
                  "rerun_files_sha256"):
        if not _is_sha256_hex(payload[field]):
            raise ValueError(f"ready evidence {field} invalid")
    prepare_manifest = _strict_regular_file(run_root / "prepare_manifest.json", run_root)
    if _sha256_file(prepare_manifest) != payload["prepare_manifest_sha256"]:
        raise ValueError("ready evidence prepare-manifest binding invalid")

    ctc_root = _strict_directory(resolve_input_path(cr.get("ctc_dir", ""), PROJECT_ROOT))
    audio_root = _strict_directory(resolve_input_path(cfg.get("data_dir", ""), PROJECT_ROOT))
    reference_root = _strict_directory(
        resolve_input_path(cr.get("text_dir", ""), PROJECT_ROOT))
    authoritative_source_root = _strict_directory(
        resolve_input_path(cr.get("authoritative_source_dir", ""), PROJECT_ROOT))
    if authoritative_source_root != _strict_directory(STRICT_READY_AUTHORITATIVE_SOURCE):
        raise ValueError("configured authoritative source root invalid")
    configured_dict = resolve_path(PROJECT_ROOT, cfg.get("mfa_dict", "dict/mfa_ipa.dict"))
    configured_source_dict = resolve_path(
        PROJECT_ROOT, cr.get("source_dictionary", ""))
    if configured_dict is None or configured_source_dict is None:
        raise ValueError("source/run-local dictionary path missing")
    if configured_source_dict != STRICT_READY_SOURCE_DICTIONARY:
        raise ValueError("configured source dictionary invalid")
    roots = payload["roots"]
    expected_roots = {
        "run": str(run_root), "ctc_ready": str(ctc_root),
        "audio_view": str(audio_root), "reference_view": str(reference_root),
    }
    if not isinstance(roots, dict) or set(roots) != set(expected_roots) or roots != expected_roots:
        raise ValueError("ready evidence canonical roots invalid")

    artifacts = payload["artifacts"]
    if (not isinstance(artifacts, dict) or len(artifacts) != len(stems)
            or set(artifacts) != set(stems)):
        raise ValueError("ready evidence artifact stem set/order invalid")
    normalized: dict[str, dict] = {}
    action_counts = {STRICT_READY_ACTION: 0}
    for stem in stems:
        item = artifacts[stem]
        if (not isinstance(item, dict) or set(item) != {
                "origin_action", "audio", "reference", "authoritative_audio",
                "authoritative_reference", "ctc"}):
            raise ValueError(f"ready evidence artifact incomplete: {stem}")
        if item["origin_action"] != STRICT_READY_ACTION:
            raise ValueError(f"ready evidence origin action invalid: {stem}")
        action_counts[STRICT_READY_ACTION] += 1
        ctc = item["ctc"]
        if not isinstance(ctc, dict) or set(ctc) != set(_CTC_SUFFIXES):
            raise ValueError(f"ready evidence CTC suffix set invalid: {stem}")
        audio = _strict_hash_record(
            item["audio"], audio_root / f"{stem}.wav", run_root, wav=True)
        reference = _strict_hash_record(
            item["reference"], reference_root / f"{stem}.txt", run_root)
        authoritative_audio = _strict_external_hash_record(
            item["authoritative_audio"], authoritative_source_root, wav=True)
        authoritative_reference = _strict_external_hash_record(
            item["authoritative_reference"], authoritative_source_root)
        if (authoritative_audio["path"].stem != stem
                or authoritative_audio["path"].suffix.lower() != ".wav"
                or authoritative_reference["path"].stem != stem
                or authoritative_reference["path"].suffix.lower() != ".txt"):
            raise ValueError(f"authoritative source stem path mismatch: {stem}")
        if not _same_artifact_content(audio, authoritative_audio, wav=True):
            raise ValueError(f"ready audio is not an authoritative byte copy: {stem}")
        if not _same_artifact_content(reference, authoritative_reference):
            raise ValueError(f"ready reference is not an authoritative byte copy: {stem}")
        if (os.path.samestat(audio["path"].stat(), authoritative_audio["path"].stat())
                or os.path.samestat(reference["path"].stat(),
                                    authoritative_reference["path"].stat())):
            raise ValueError(f"ready authority copy is an inode alias: {stem}")
        normalized[stem] = {
            "origin_action": item["origin_action"],
            "audio": audio, "reference": reference,
            "authoritative_audio": authoritative_audio,
            "authoritative_reference": authoritative_reference,
            "ctc": {
                suffix: _strict_hash_record(
                    ctc[suffix], ctc_root / f"{stem}{suffix}", run_root)
                for suffix in _CTC_SUFFIXES
            },
        }
    if action_counts != expected_actions:
        raise ValueError("ready evidence per-stem action counts invalid")

    source_dictionary = _strict_hash_record(
        payload["source_dictionary"], configured_source_dict,
        configured_source_dict.parent)
    dictionary = _strict_hash_record(
        payload["run_local_dictionary"], configured_dict, run_root)
    if not _same_artifact_content(source_dictionary, dictionary):
        raise ValueError("run-local dictionary is not an authoritative byte copy")
    if os.path.samestat(source_dictionary["path"].stat(), dictionary["path"].stat()):
        raise ValueError("run-local dictionary is an inode alias")

    rerun_files = payload["rerun_files"]
    if (not isinstance(rerun_files, list)
            or len(rerun_files) != STRICT_READY_COUNT * len(_CTC_SUFFIXES)
            or _stable_json_sha256(rerun_files) != payload["rerun_files_sha256"]):
        raise ValueError("ready evidence rerun-file mapping invalid")
    rerun_root = run_root / "ctc_rerun_output"
    for index, copy_record in enumerate(rerun_files):
        stem = stems[index // len(_CTC_SUFFIXES)]
        suffix = _CTC_SUFFIXES[index % len(_CTC_SUFFIXES)]
        if (not isinstance(copy_record, dict)
                or set(copy_record) != {"kind", "stem", "source", "destination"}
                or copy_record["kind"] != "rerun_ctc" or copy_record["stem"] != stem
                or copy_record["destination"] != artifacts[stem]["ctc"][suffix]):
            raise ValueError(f"ready rerun copy mapping invalid: {stem}{suffix}")
        source = _strict_hash_record(
            copy_record["source"], rerun_root / f"{stem}{suffix}", run_root)
        if not _same_artifact_content(source, normalized[stem]["ctc"][suffix]):
            raise ValueError(f"ready CTC is not a rerun byte copy: {stem}{suffix}")
        if os.path.samestat(
                source["path"].stat(), normalized[stem]["ctc"][suffix]["path"].stat()):
            raise ValueError(f"ready CTC is a rerun inode alias: {stem}{suffix}")

    _validate_exact_regular_namespace(
        audio_root, {f"{stem}.wav" for stem in stems}, "ready audio")
    _validate_exact_regular_namespace(
        reference_root, {f"{stem}.txt" for stem in stems}, "ready reference")
    expected_ctc_names = {
        f"{stem}{suffix}" for stem in stems for suffix in _CTC_SUFFIXES}
    _validate_exact_regular_namespace(ctc_root, expected_ctc_names, "ready CTC")
    payload.update({
        "_path": evidence_path, "_sha256": pin["sha256"],
        "_run_root": run_root, "_audio_root": audio_root,
        "_reference_root": reference_root, "_ctc_root": ctc_root,
        "_authoritative_source_root": authoritative_source_root,
        "_artifacts": normalized, "_dictionary": dictionary,
        "_source_dictionary": source_dictionary,
    })
    return payload


def load_strict_stem_selection(path: Path | None, evidence_stems: list[str]) -> tuple[list[str], dict]:
    if path is None:
        return list(evidence_stems), {"scope": "full", "path": None, "sha256": None}
    selector = _strict_regular_file(path, path.parent)
    try:
        lines = selector.read_text(encoding="utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError("selector must be UTF-8") from exc
    if not lines or any(not line or line.strip() != line for line in lines):
        raise ValueError("selector must contain exact nonempty stem lines")
    if lines != sorted(lines) or len(lines) != len(set(lines)):
        raise ValueError("selector stems must be sorted and unique")
    if not set(lines).issubset(evidence_stems):
        raise ValueError("selector contains unknown stem")
    return lines, {"scope": "canary", "path": str(selector), "sha256": _sha256_file(selector)}


def _strict_suffix_stems(directory: Path, suffix: str) -> tuple[set[str], list[str]]:
    if not directory.is_dir() or directory.is_symlink():
        return set(), [f"missing/non-ordinary directory: {directory}"]
    stems: set[str] = set(); issues: list[str] = []
    for entry in directory.iterdir():
        if not entry.name.endswith(suffix):
            continue
        if entry.is_symlink() or not entry.is_file():
            issues.append(f"non-ordinary {suffix} entry: {entry}")
            continue
        stem = entry.name[:-len(suffix)]
        if not stem or stem in stems:
            issues.append(f"duplicate/invalid {suffix} stem: {entry.name}")
        stems.add(stem)
    return stems, issues


def strict_stage_denominator_issues(step_name: str, ctx: dict) -> list[str]:
    """Compare every materialized stage against the frozen import denominator."""
    expected = set(ctx.get("expected_stems", ()))
    if not expected:
        return []
    issues: list[str] = []

    def require_suffix(directory: Path, suffix: str, label: str) -> None:
        actual, local = _strict_suffix_stems(directory, suffix)
        issues.extend(local)
        if actual != expected:
            issues.append(
                f"{label} denominator mismatch: missing={len(expected - actual)}, "
                f"extra={len(actual - expected)}")

    raw_ctc = ctx["ctc_pretg"]
    for suffix in _CTC_SUFFIXES:
        require_suffix(raw_ctc, suffix, f"raw CTC {suffix}")
    require_suffix(raw_ctc, "_ref.txt", "workspace reference")

    runtime_dict = Path(ctx["mfa_dict"])
    try:
        runtime_dict.relative_to(ctx["workspace"])
    except ValueError:
        issues.append(f"runtime dictionary escaped workspace: {runtime_dict}")
    if runtime_dict.is_symlink() or not runtime_dict.is_file():
        issues.append(f"runtime dictionary missing/non-ordinary: {runtime_dict}")

    evidence = ctx.get("strict_ready_evidence")
    audio_dir = Path(ctx["audio_dir"])
    if not isinstance(evidence, dict) or evidence.get("_audio_root") != audio_dir:
        issues.append("active audio escaped the evidenced audio_view")
    else:
        for stem in expected:
            path = audio_dir / f"{stem}.wav"
            record = evidence["_artifacts"][stem]["audio"]
            try:
                resolved = _strict_regular_file(path, audio_dir)
            except ValueError as exc:
                issues.append(str(exc))
                continue
            if (resolved != record["path"] or path.stat().st_size != record["size"]):
                issues.append(f"selected authority WAV evidence mismatch: {stem}")

    padded = ctx["workspace"] / "padded_audio"
    if padded.exists() or padded.is_symlink():
        issues.append(f"padding output is forbidden in strict v4: {padded}")

    after_resample = {"resample", "adjust", "align", "align_en", "postprocess", "strict_ok"}
    after_adjust = {"adjust", "align", "align_en", "postprocess", "strict_ok"}
    after_align = {"align", "align_en", "postprocess", "strict_ok"}
    if step_name in after_resample:
        require_suffix(ctx["mfa_audio_dir"], ".wav", "MFA audio")
    if step_name in after_adjust and ctx["ctc_pretg_adj"] != raw_ctc:
        for suffix in _CTC_SUFFIXES:
            require_suffix(ctx["ctc_pretg_adj"], suffix, f"adjusted CTC {suffix}")
        require_suffix(ctx["ctc_pretg_adj"], "_ref.txt", "adjusted reference")
    if step_name in after_align:
        require_suffix(ctx["aligned_dir"], ".TextGrid", "MFA alignment")
    if step_name in {"postprocess", "strict_ok"}:
        output, local = _strict_suffix_stems(ctx["output_dir"], ".TextGrid")
        filtered, local_filtered = _strict_suffix_stems(ctx["filtered_dir"], ".TextGrid")
        issues.extend(local + local_filtered)
        if output & filtered or output | filtered != expected:
            issues.append(
                f"final partition mismatch: overlap={len(output & filtered)}, "
                f"missing={len(expected - (output | filtered))}, "
                f"extra={len((output | filtered) - expected)}")
    return issues


def _run_ready_verifier(cfg: dict, evidence: dict) -> int:
    if callable(STRICT_READY_VERIFY_HOOK):
        return int(STRICT_READY_VERIFY_HOOK(cfg, evidence))
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "verify_hecheng_english_ctc_ready_v4.py"),
        "--run-root", str(evidence["_run_root"]),
        "--source-dir", str(evidence["_authoritative_source_root"]),
        "--dictionary-source", str(evidence["_source_dictionary"]["path"]),
    ]
    try:
        return subprocess.run(
            command, timeout=cfg.get("ctc_ready", {}).get("verify_timeout", 14400)
        ).returncode
    except subprocess.TimeoutExpired:
        print("ERROR: independent v4 ready verifier timed out")
        return 124


def _copy_regular_verified(source: Path, destination: Path, record: dict) -> dict:
    import shutil
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"copy source must be ordinary: {source}")
    before = source.stat(); source_hash = _sha256_file(source)
    if before.st_size != record["size"] or source_hash != record["sha256"]:
        raise ValueError(f"source changed before copy: {source}")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"copy destination must not preexist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _strict_directory(destination.parent)
    shutil.copyfile(source, destination)
    after = source.stat(); dest_stat = destination.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or _sha256_file(source) != record["sha256"]
            or destination.is_symlink() or not destination.is_file()
            or dest_stat.st_size != record["size"] or _sha256_file(destination) != record["sha256"]
            or (after.st_dev, after.st_ino) == (dest_stat.st_dev, dest_stat.st_ino)):
        raise ValueError(f"copy verification/alias failure: {source}")
    return {"source": str(source), "destination": str(destination), "sha256": record["sha256"],
            "size": record["size"], "source_dev": after.st_dev, "source_ino": after.st_ino,
            "destination_dev": dest_stat.st_dev, "destination_ino": dest_stat.st_ino}


def _verify_imported_copy(record: dict, workspace: Path) -> None:
    source = Path(record["source"])
    destination = _strict_regular_file(Path(record["destination"]), workspace)
    source_stat = source.stat(); destination_stat = destination.stat()
    if (source.is_symlink() or not source.is_file()
            or source_stat.st_size != record["size"]
            or destination_stat.st_size != record["size"]
            or _sha256_file(source) != record["sha256"]
            or _sha256_file(destination) != record["sha256"]
            or (source_stat.st_dev, source_stat.st_ino)
            == (destination_stat.st_dev, destination_stat.st_ino)
            or (source_stat.st_dev, source_stat.st_ino)
            != (record["source_dev"], record["source_ino"])
            or (destination_stat.st_dev, destination_stat.st_ino)
            != (record["destination_dev"], record["destination_ino"])):
        raise ValueError(f"imported copy verification failed: {destination}")


def _step_link_ctc_strict(args, cfg: dict, ctx: dict) -> int:
    import json, shutil
    cr = cfg["ctc_ready"]
    workspace = ctx["workspace"]; target = ctx["ctc_pretg"]
    import_manifest = workspace / "ctc_ready_import_manifest.json"
    selected_path = workspace / "ctc_ready_selected_stems.txt"
    runtime_dict = workspace / cfg.get("runtime_mfa_dict", "runtime/mfa_ipa.dict")
    padded = workspace / "padded_audio"
    forbidden = [target, import_manifest, selected_path, runtime_dict, padded]
    if any(path.exists() or path.is_symlink() for path in forbidden):
        print("ERROR: strict CTC-ready workspace is not fresh")
        return 1
    try:
        evidence = load_and_validate_ready_evidence(cfg)
        if _strict_directory(Path(ctx["audio_dir"])) != evidence["_audio_root"]:
            raise ValueError("active audio is not the evidenced audio_view")
        selector_arg = getattr(args, "ctc_ready_stems_file", None)
        selected, selector = load_strict_stem_selection(
            Path(selector_arg) if selector_arg else None,
            evidence["authoritative_stems"])
        if selector["scope"] == "full" and len(selected) != cr["expected_count"]:
            raise ValueError("full import denominator mismatch")
        if _run_ready_verifier(cfg, evidence) != 0:
            raise ValueError("independent v4 verifier failed before import")
        evidence_before = _sha256_file(evidence["_path"])
        stage = workspace / f".ctc_ready_import_{os.getpid()}"
        if stage.exists() or stage.is_symlink():
            raise ValueError("import staging collision")
        stage_ctc = stage / "ctc_pretg"; stage_dict = stage / "runtime" / "mfa_ipa.dict"
        stage_ctc.mkdir(parents=True); copies = []
        for stem in selected:
            item = evidence["_artifacts"][stem]
            for suffix in _CTC_SUFFIXES:
                staged = stage_ctc / f"{stem}{suffix}"
                copied = _copy_regular_verified(item["ctc"][suffix]["path"], staged,
                                                item["ctc"][suffix])
                copied.update({"staging_destination": str(staged),
                               "destination": str(target / staged.name)})
                copies.append(copied)
            staged = stage_ctc / f"{stem}_ref.txt"
            copied = _copy_regular_verified(item["reference"]["path"], staged, item["reference"])
            copied.update({"staging_destination": str(staged),
                           "destination": str(target / staged.name)})
            copies.append(copied)
        dictionary_copy = _copy_regular_verified(evidence["_dictionary"]["path"],
                                                 stage_dict, evidence["_dictionary"])
        dictionary_copy.update({"staging_destination": str(stage_dict),
                                "destination": str(runtime_dict)})
        expected_names = {f"{stem}{suffix}" for stem in selected for suffix in _CTC_SUFFIXES}
        expected_names |= {f"{stem}_ref.txt" for stem in selected}
        if {p.name for p in stage_ctc.iterdir() if p.is_file()} != expected_names:
            raise ValueError("staged CTC/reference namespace mismatch")
        if _sha256_file(evidence["_path"]) != evidence_before:
            raise ValueError("ready evidence changed during import")
        with selected_path.open("x", encoding="utf-8") as selected_handle:
            selected_handle.write("\n".join(selected) + "\n")
        os.replace(stage_ctc, target)
        runtime_dict.parent.mkdir(parents=True)
        _strict_directory(runtime_dict.parent)
        os.replace(stage_dict, runtime_dict)
        shutil.rmtree(stage, ignore_errors=True)
        _validate_exact_regular_namespace(target, expected_names, "workspace import")
        for copied in copies:
            _verify_imported_copy(copied, workspace)
        _verify_imported_copy(dictionary_copy, workspace)
        manifest_payload = {"schema": STRICT_READY_SCHEMA, "state": "imported",
                            "evidence_path": str(evidence["_path"]), "evidence_sha256": evidence_before,
                            "evidence_state": evidence["state"], "evidence_roots": evidence["roots"],
                            "independent_verifier_signature": evidence[
                                "independent_verifier_signature"],
                            "evidence_action_counts": evidence["action_counts"],
                            "evidence_taxonomy_sha256": evidence["taxonomy_sha256"],
                            "final_audio_axis": evidence["final_audio_axis"],
                            "padding_policy": evidence["padding_policy"],
                            "full_evidence_count": len(evidence["authoritative_stems"]),
                            "scope": selector, "selected_stems": selected,
                            "selected_count": len(selected),
                            "selected_stems_sha256": _sha256_file(selected_path), "copies": copies,
                            "runtime_dictionary": dictionary_copy,
                            "exact_destination_names": sorted(expected_names),
                            "checks": {"source_verify_before": True, "workspace_exact": True,
                                       "no_inode_alias": True}}
        if (_run_ready_verifier(cfg, evidence) != 0
                or _sha256_file(evidence["_path"]) != evidence_before):
            raise ValueError("independent v4 verifier failed after import")
        manifest_payload["checks"]["source_verify_after"] = True
        tmp_manifest = import_manifest.with_suffix(".json.tmp")
        with tmp_manifest.open("x", encoding="utf-8") as manifest_handle:
            manifest_handle.write(json.dumps(
                manifest_payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp_manifest, import_manifest)
        ctx.update({"expected_stems": tuple(selected), "strict_ready_evidence": evidence,
                    "strict_ready_evidence_sha256": evidence_before, "mfa_dict": runtime_dict,
                    "raw_text_dir": target, "strict_selected_stems_file": selected_path,
                    "audio_dir": evidence["_audio_root"]})
        print(f"  Strict CTC-ready import: {len(selected)} stems ({selector['scope']})")
        return 0
    except Exception as exc:
        print(f"ERROR: strict CTC-ready import failed: {exc}")
        return 1


def step_link_ctc(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Validate pre-existing NVASR CTC output and prepare workspace.

    Scans the CTC directory for ``.lab`` files (single-level, no recursion),
    matches audio by stem, validates all 6 NVASR output files per stem, then
    hard-links audio + CTC files into the workspace so the pipeline can
    proceed from ``resample`` onward.
    """
    import json as _json

    if _strict_ready_mode(cfg):
        return _step_link_ctc_strict(args, cfg, ctx)

    cr = cfg.get("ctc_ready", {})

    # -- Fast path: if manifest exists from a previous run, skip re-scanning --
    ctc_out_early = ctx["ctc_pretg"]
    manifest_early = ctc_out_early / "ctc_ready_manifest.json"
    if manifest_early.exists() and not args.overwrite:
        try:
            prev = _json.loads(manifest_early.read_text())
            n_stems = len(prev.get("stems", []))
            print(f"  Link already done ({n_stems} stems in manifest)."
                  f" Use --overwrite to re-link.")
            return 0
        except Exception:
            pass  # corrupt manifest, proceed with scan

    # ── Resolve source directories ──
    ctc_dir_src = resolve_input_path(cr["ctc_dir"], PROJECT_ROOT)
    if not ctc_dir_src.exists():
        print(f"ERROR: CTC directory not found: {ctc_dir_src}")
        return 1

    audio_src = ctx["data_dir"]  # data_dir IS the audio source in ctc_ready
    if not audio_src.exists():
        print(f"ERROR: Audio directory not found: {audio_src}")
        return 1

    text_src = resolve_input_path(cr.get("text_dir", ""), PROJECT_ROOT) if cr.get("text_dir") else audio_src

    print(f"  CTC dir:   {ctc_dir_src}")
    print(f"  Audio dir: {audio_src}")
    if text_src != audio_src:
        print(f"  Text dir:  {text_src}")

    # ── 1. Resolve filters ──
    stem_filter = cr.get("stems", None)      # explicit list
    stem_range = cr.get("stem_range", None)  # [start, end] inclusive
    stem_prefix = cr.get("stem_prefix", "")  # prepended to numeric stems
    is_filtered = stem_filter is not None or stem_range is not None
    require_all = cr.get("require_all", True)

    # ── 2. Single-pass matching: discover stems + match audio/text + validate ──
    # When filtered: generate candidates, probe directly — no directory scan.
    # When unfiltered: scan CTC dir for .lab files, then match audio/text.
    audio_index: dict[str, Path] = {}
    text_index: dict[str, Path] = {}
    valid: list[str] = []
    missing_audio: list[str] = []
    incomplete_ctc: list[tuple[str, str]] = []
    total_candidates = 0
    _ctc_base_cache: dict[str, Path] = {}  # stem -> resolved CTC base dir

    if is_filtered:
        # -- Filtered path -- direct probe (no directory scan) --
        # Build CTC presence sets once for O(1) completeness checks
        ctc_files_flat, ctc_files_nested = build_ctc_presence(ctc_dir_src)
        print(f"  CTC presence index: {len(ctc_files_flat)} flat + "
              f"{sum(len(v) for v in ctc_files_nested.values())} nested files")
        if stem_filter is not None:
            candidates = [f"{stem_prefix}{s}" for s in stem_filter]
        else:
            lo, hi = stem_range
            prefix = str(stem_prefix) if stem_prefix else ""
            candidates = [f"{prefix}{i}" for i in range(int(lo), int(hi) + 1)]
        total_candidates = len(candidates)
        print(f"  Probing {total_candidates} candidates"
              f" ({candidates[0]}–{candidates[-1]}) ...")

        for stem in candidates:
            # ── Resolve CTC base: flat (dir/{stem}.lab) or nested (dir/{stem}/{stem}.lab) ──
            ctc_base = ctc_dir_src
            lab_path = ctc_base / f"{stem}.lab"
            if not lab_path.exists():
                lab_path = ctc_base / stem / f"{stem}.lab"
                if lab_path.exists():
                    ctc_base = ctc_base / stem
                else:
                    continue

            # ── Match audio (exact -> nested -> zero-padded -> glob fallback) ──
            wav_path = find_wav(audio_src, stem)
            if wav_path is None:
                missing_audio.append(stem)
                continue
            audio_index[stem] = wav_path

            # ── Match text ──
            txt_path = text_src / f"{stem}.txt"
            if not txt_path.exists():
                txt_path = text_src / stem / f"{stem}.txt"
            if txt_path.exists():
                text_index[stem] = txt_path

            # -- Validate CTC completeness (O(1) set lookup, no per-file exists()) --
            if require_all:
                ctc_ok = all(
                    f"{stem}{suffix}" in ctc_files_flat
                    or (stem in ctc_files_nested
                        and f"{stem}{suffix}" in ctc_files_nested[stem])
                    for suffix in _CTC_SUFFIXES
                )
                if not ctc_ok:
                    # Determine which suffix is missing for the report
                    for suffix in _CTC_SUFFIXES:
                        in_flat = f"{stem}{suffix}" in ctc_files_flat
                        in_nested = (stem in ctc_files_nested
                                      and f"{stem}{suffix}" in ctc_files_nested[stem])
                        if not in_flat and not in_nested:
                            incomplete_ctc.append((stem, suffix))
                            break
                    continue

            # Store resolved ctc_base for linking step
            valid.append(stem)
            _ctc_base_cache[stem] = ctc_base

    else:
        # -- Unfiltered path -- scan CTC dir for .lab files --
        # Handles both flat (dir/{stem}.lab) and nested (dir/{stem}/{stem}.lab)
        print("  Scanning CTC directory for .lab files ...")

        # Build CTC presence sets once for O(1) completeness checks
        ctc_files_flat, ctc_files_nested = build_ctc_presence(ctc_dir_src)
        print(f"  CTC presence index: {len(ctc_files_flat)} flat files, "
              f"{len(ctc_files_nested)} nested dirs")
        stems_all: list[str] = []
        layout_kind = "flat"
        try:
            with os.scandir(str(ctc_dir_src)) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".lab"):
                        stems_all.append(entry.name[:-4])
                        _ctc_base_cache[entry.name[:-4]] = ctc_dir_src
                    elif entry.is_dir():
                        # Nested: {dir}/{stem}/{stem}.lab
                        nested_lab = Path(entry.path) / f"{entry.name}.lab"
                        if nested_lab.exists():
                            stems_all.append(entry.name)
                            _ctc_base_cache[entry.name] = Path(entry.path)
                            layout_kind = "nested"
        except OSError as e:
            print(f"ERROR: Cannot read CTC directory: {e}")
            return 1
        stems_all.sort()
        total_candidates = len(stems_all)
        print(f"  Found {total_candidates} stems via .lab scan ({layout_kind})")

        # Build audio/text indices (single-level scan, then match in memory)
        # Check for pre-built wav_index.json first (for deeply nested audio on CIFS)
        wav_index_path = ctc_dir_src / "wav_index.json"
        audio_index: dict[str, Path] = {}
        if wav_index_path.exists():
            try:
                raw = json.loads(wav_index_path.read_text(encoding='utf-8'))
                audio_index = {stem: Path(p) for stem, p in raw.items()}
                print(f"  Audio index: {len(audio_index)} WAV files (from wav_index.json)")
            except Exception:
                pass
        if not audio_index:
            audio_index = build_file_index(audio_src, ".wav")
            print(f"  Audio index: {len(audio_index)} WAV files")
        if text_src.exists():
            text_index = build_file_index(text_src, ".txt")
            print(f"  Text index:  {len(text_index)} TXT files")

        for stem in stems_all:
            if stem not in audio_index:
                missing_audio.append(stem)
                continue
            if require_all:
                ctc_ok = all(
                    f"{stem}{suffix}" in ctc_files_flat
                    or (stem in ctc_files_nested
                        and f"{stem}{suffix}" in ctc_files_nested[stem])
                    for suffix in _CTC_SUFFIXES
                )
                if not ctc_ok:
                    for suffix in _CTC_SUFFIXES:
                        in_flat = f"{stem}{suffix}" in ctc_files_flat
                        in_nested = (stem in ctc_files_nested
                                      and f"{stem}{suffix}" in ctc_files_nested[stem])
                        if not in_flat and not in_nested:
                            incomplete_ctc.append((stem, suffix))
                            break
                    continue
            valid.append(stem)

    # ── 3. Report ──
    n_missing_lab = total_candidates - len(valid) - len(missing_audio) - len(incomplete_ctc)
    print(f"\n  Candidates:     {total_candidates}")
    print(f"  Valid stems:    {len(valid)}")
    if n_missing_lab > 0:
        print(f"  Missing .lab:   {n_missing_lab}")
    if missing_audio:
        print(f"  Missing audio:  {len(missing_audio)}")
        for s in missing_audio[:5]:
            print(f"    - {s}")
        if len(missing_audio) > 5:
            print(f"    ... and {len(missing_audio) - 5} more")
    if incomplete_ctc:
        print(f"  Incomplete CTC: {len(incomplete_ctc)}")
        for s, suffix in incomplete_ctc[:5]:
            print(f"    - {s}{suffix}")
        if len(incomplete_ctc) > 5:
            print(f"    ... and {len(incomplete_ctc) - 5} more")

    if not valid:
        print("ERROR: No valid stems — nothing to process.")
        return 1

    # In scan-only mode, skip file linking — only validate + write manifest
    scan_only = getattr(args, 'scan_only', False)
    if scan_only:
        print(f"\n  Scan-only: skipping file linking for {len(valid)} stems")

    # ── 4-6. File linking (skipped in scan-only mode) ──
    ctc_out = ctx["ctc_pretg"]
    ctc_out.mkdir(parents=True, exist_ok=True)
    if not scan_only:
        audio_out = ctx["audio_dir"]
        if audio_out.resolve() == audio_src.resolve():
            # Audio dir IS the source — no linking needed
            # Verify all stems have audio accessible
            n_present = sum(1 for stem in valid if stem in audio_index)
            print(f"\n  Audio in-place: {n_present}/{len(valid)} stems indexed (audio at {audio_out})")
            if n_present < len(valid):
                print(f"  WARNING: {len(valid) - n_present} stems missing audio")
        else:
            audio_out.mkdir(parents=True, exist_ok=True)
            linked = 0
            for stem in valid:
                if _link_or_copy(audio_index[stem], audio_out / f"{stem}.wav"):
                    linked += 1
            print(f"\n  Audio linked: {linked} -> {audio_out}")

        # ── 5. Link CTC files -> workspace/ctc_pretg/ ──
        ctc_out.mkdir(parents=True, exist_ok=True)
        ctc_linked = 0
        ctc_missing: list[str] = []
        for stem in valid:
            ctc_base = _ctc_base_cache.get(stem, ctc_dir_src)
            for suffix in _CTC_SUFFIXES:
                src = ctc_base / f"{stem}{suffix}"
                if src.exists():
                    if _link_or_copy(src, ctc_out / f"{stem}{suffix}"):
                        ctc_linked += 1
                    else:
                        ctc_missing.append(f"{stem}{suffix}")
                else:
                    ctc_missing.append(f"{stem}{suffix} (source gone)")
        if ctc_missing:
            print(f"  WARNING: {len(ctc_missing)} CTC file(s) missing at link time "
                  f"(previously passed validation):")
            for f in ctc_missing[:10]:
                print(f"    - {f}")
            if len(ctc_missing) > 10:
                print(f"    ... and {len(ctc_missing) - 10} more")
        print(f"  CTC linked:  {ctc_linked} -> {ctc_out}")

        # Preserve a reference transcript already bundled with generated CTC
        # output.  It is optional for legacy CTC directories, but when
        # present it is the only authoritative text if text_dir is omitted.
        bundled_refs = 0
        for stem in valid:
            ctc_base = _ctc_base_cache.get(stem, ctc_dir_src)
            src_ref = ctc_base / f"{stem}_ref.txt"
            dst_ref = ctc_out / f"{stem}_ref.txt"
            if src_ref.exists() and (not dst_ref.exists() or args.overwrite):
                if _link_or_copy(src_ref, dst_ref):
                    bundled_refs += 1
        if bundled_refs:
            print(f"  Bundled refs: {bundled_refs} -> {ctc_out}")

        # ── 6. Copy/link reference text (.txt from text_dir) ──
        if text_index:
            txt_linked = 0
            for stem in valid:
                if stem in text_index:
                    # Copy to ctc_pretg so postprocess can find it
                    dst = ctc_out / f"{stem}_ref.txt"
                    if not dst.exists() or args.overwrite:
                        _link_or_copy(text_index[stem], dst)
                        txt_linked += 1
            if txt_linked:
                print(f"  Text refs:   {txt_linked} -> {ctc_out}")

    # ── 7. Save manifest (always, even in scan-only mode) ──
    manifest = {
        "mode": "ctc_ready",
        "ctc_dir_src": str(ctc_dir_src),
        "audio_src": str(audio_src),
        "n_candidates": total_candidates,
        "n_valid": len(valid),
        "n_missing_audio": len(missing_audio),
        "n_incomplete_ctc": len(incomplete_ctc),
        "stems": valid,
    }
    (ctc_out / "ctc_ready_manifest.json").write_text(
        _json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Ready for MFA pipeline ({len(valid)} stems)")
    return 0


def step_pad_silence(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Pad/trim head and tail silence to 0.5s, shift CTC timestamps.

    Modifies audio in-place: audio_dir is replaced with padded versions.
    All CTC timestamps (.TextGrid, _tokens.jsonl, _punct.json) are shifted
    by the net head change so downstream steps see consistent alignment.
    """
    pc = cfg.get("pad_silence", {})
    if not pc.get("enabled", True):
        print("  pad_silence disabled in config. Skipping.")
        return 0
    target_silence_sec = pc.get("target_edge_silence_sec", 0.5)

    ctc_dir = ctx["ctc_pretg"]
    padded_audio_dir = ctx["workspace"] / "padded_audio"

    pad_args = [
        "--ctc-dir", str(ctc_dir),
        "--audio-dir", str(ctx["audio_dir"]),
        "--padded-audio-dir", str(padded_audio_dir),
        "--target-silence-sec", str(target_silence_sec),
    ]
    # Optional: also write padded audio to output dir (default off)
    if pc.get("output_audio", False):
        output_audio_dir = ctx["output_dir"] / "padded_audio"
        pad_args += ["--output-audio-dir", str(output_audio_dir)]

    expected_stems = tuple(ctx.get("expected_stems", ()))
    if expected_stems:
        pad_args += ["--stems-file", str(ctx["strict_selected_stems_file"])]
    else:
        # Legacy-only optimization.  Strict evidence mode may never redirect
        # audio lookup through a mutable external index.
        cr = cfg.get("ctc_ready", {})
        ctc_src = resolve_input_path(cr.get("ctc_dir", ""), PROJECT_ROOT)
        wav_index_path = ctc_src / "wav_index.json"
        if wav_index_path.exists():
            pad_args += ["--wav-index", str(wav_index_path)]

    rc = run_python(SCRIPTS_DIR / "pad_silence_edges.py", pad_args, mfa_python,
                     ctx["models_dir"], desc="Pad/trim silence edges")

    if rc == 0 and expected_stems:
        expected_names = {f"{stem}.wav" for stem in expected_stems}
        entries = list(padded_audio_dir.iterdir()) if padded_audio_dir.is_dir() else []
        actual_names = {entry.name for entry in entries
                        if entry.is_file() and not entry.is_symlink()}
        lab_stems = {path.stem for path in ctc_dir.glob("*.lab")
                     if path.is_file() and not path.is_symlink()}
        if (actual_names != expected_names or len(entries) != len(expected_names)
                or any(entry.is_symlink() or not entry.is_file() for entry in entries)
                or lab_stems != set(expected_stems)):
            print("  ERROR: padding success did not preserve the frozen denominator")
            return 1

    if rc == 0:
        # Switch audio_dir to padded versions for all downstream steps
        ctx["audio_dir"] = padded_audio_dir
        print(f"  Switched audio_dir → {padded_audio_dir}")

    return rc


# ---------------------------------------------------------------------------
# Step registry — must come after all step functions are defined
# ---------------------------------------------------------------------------

STEPS = {
    "link": ("Link pre-existing CTC output (ctc_ready mode)", step_link_ctc),
    "pad_silence": ("Pad/trim head+tail silence to 0.5s", step_pad_silence),
    "trim": ("Audio preprocessing", step_trim_silence),
    "resample": ("Resample to 16kHz for MFA", step_resample_for_mfa),
    "prealign": ("CTC pre-alignment (NVASR -> MFA anchors)", step_prealign),
    "normalize_punct": ("Normalize punctuation (ASCII -> CJK)", step_normalize_punct),
    "normalize": ("Normalize numerals (Arabic -> Chinese)", step_normalize_text),
    "normalize_ria": ("Normalize ria transliterations (瑞娅/瑞亚/瑞雅/瑞啊 -> ria)", step_normalize_ria),
    "normalize_en": ("Normalise English-word fragments in CTC output", step_normalize_en),
    "adjust": ("Adjust CTC boundaries (energy-based)", step_adjust_ctc),
    "validate": ("MFA validate", step_mfa_validate),
    "align": ("MFA align (NVASR corpus + CTC anchors)", step_mfa_align),
    "align_en": ("English MFA align (English-only segments)", step_mfa_align_en),
    "postprocess": ("Post-processing (includes NVV brackets + sp1 normalization)", step_postprocess),
    "strict_ok": ("Independent strict-ok audit and manifest", step_strict_ok),
}

FULL_STEP_ORDER = ["trim", "resample", "prealign", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "adjust", "align", "align_en", "postprocess", "strict_ok"]
CTC_READY_STEP_ORDER = ["link", "pad_silence", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess", "strict_ok"]
STRICT_CTC_READY_STEP_ORDER = ["link", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess", "strict_ok"]
NVASR_FALLBACK_STEP_ORDER = ["prealign", "pad_silence", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess", "strict_ok"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Chinese MFA forced alignment pipeline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help=f"Config file path (default: {DEFAULT_CONFIG})")
    parser.add_argument("--step", type=str, default=None)
    parser.add_argument("--skip-to", type=str, default=None)
    for s in STEPS:
        parser.add_argument(f"--skip-{s}", action="store_true", help=f"Skip {s}")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override input directory from config.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory from config.")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Override workspace root (default: <project>/output/<workspace_name>).")
    parser.add_argument("--python", type=str, default=None,
                        help="Override Python path from config.")
    parser.add_argument("--device", type=str, default=None,
                        help="GPU device for NVASR CTC pre-alignment (e.g. cuda:0, cuda:1). "
                             "Overrides ctc_prealign.device in config. "
                             "Use with streaming --gpus for multi-GPU scheduling.")
    parser.add_argument("--nvme-cache", type=str, default=None, metavar="DIR",
                        help="Path to NVMe audio cache (default: auto-detect"
                             " /mnt/nvme3/mfa_audio_cache).")
    parser.add_argument("--auto-cache", action="store_true",
                        help="Auto-create temp audio cache on NVMe if permanent"
                             " cache not found (cleaned after pipeline completes).")
    parser.add_argument("--output-staging", action="store_true",
                        help="Write final output to NVMe staging first, then rsync to NAS."
                             " Avoids per-file CIFS write latency for postprocess.")
    parser.add_argument("--no-output-staging", action="store_true",
                        help="Disable output staging (override config).")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output structure after each step (uses output_spec in config).")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["full", "ctc_ready", "batch_ctc_ready", "nvrasr_fallback"],
                        help="Pipeline mode (default: from config, or 'full').")
    parser.add_argument("--ctc-ready", type=str, default=None, metavar="CTC_DIR",
                        help="Enable ctc_ready mode: path to pre-existing NVASR CTC output.")
    parser.add_argument("--ctc-ready-stems-file", type=str, default=None, metavar="FILE",
                        help="Strict mode only: sorted UTF-8 evidence subset for a fresh canary workspace.")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use pre-built scan cache (default: enabled, controlled by config 'use_cache').")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-scan, ignore cache and config setting.")
    parser.add_argument("--scan-only", action="store_true",
                        help="Pre-scan only: discover + validate + write cache, then exit.")
    parser.add_argument("--stop-after", type=str, default=None, metavar="STEP",
                        choices=["prealign", "normalize_en", "resample", "adjust", "align", "align_en"],
                        help="Stop pipeline after completing this step (for pipelined GPU/CPU split).")
    parser.add_argument("--dataset-offset", type=int, default=0,
                        help="Skip first N datasets (for multi-GPU slicing in batch mode).")
    parser.add_argument("--dataset-limit", type=int, default=0,
                        help="Process at most N datasets (0=all). "
                             "Use with --dataset-offset for multi-GPU slicing.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom cache directory (default: <project>/cache/).")
    args = parser.parse_args()

    if args.list_steps:
        for name, (desc, _) in STEPS.items():
            print(f"  {name:12s} - {desc}")
        return 0

    # Load config
    cfg = load_config(Path(args.config))
    print(f"Config: {args.config}")

    # Resolve cache paths (used by both batch and single modes)
    config_path = Path(args.config)
    cache_dir = _get_cache_dir(config_path, args.cache_dir)
    cache_path = _get_cache_path(config_path, cache_dir)
    # Cache default: enabled. Disable via config "use_cache: false" or CLI --no-cache.
    # --use-cache forces it on even if config says false.
    if args.no_cache:
        use_cache = False
    elif args.use_cache:
        use_cache = True
    else:
        use_cache = cfg.get("use_cache", True)
    if not use_cache:
        print("  Scan cache: DISABLED (use_cache=false or --no-cache)")

    # ── Resolve pipeline mode ──
    mode = args.mode or cfg.get("mode", "full")
    if args.ctc_ready:
        mode = "ctc_ready"
        cfg.setdefault("ctc_ready", {})["ctc_dir"] = args.ctc_ready
        print(f"ctc_ready mode: CTC dir = {args.ctc_ready}")

    if mode not in ("full", "ctc_ready", "batch_ctc_ready", "nvrasr_fallback"):
        print(f"ERROR: Unknown mode: {mode}")
        sys.exit(1)
    print(f"Pipeline mode: {mode}")

    # Evidence mode is fail-closed before model probing, directory creation,
    # cache discovery, or execution of any pipeline step.  The returned path
    # is reused below so the fresh-workspace decision cannot drift.
    _strict_ready = _strict_ready_mode(cfg)
    _strict_workspace = Path()
    if _strict_ready:
        try:
            _strict_workspace = validate_strict_ready_invocation(args, cfg, mode, use_cache)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1

    # Models & dicts: relative to PROJECT_ROOT (must be resolved before batch/single modes)
    models_dir = resolve_path(PROJECT_ROOT, cfg.get("models_dir", "models/mfa"))
    mfa_dict = resolve_path(PROJECT_ROOT, cfg.get("mfa_dict", "dict/mfa_ipa.dict"))

    # Find MFA Python
    if args.python:
        mfa_python = Path(args.python)
    else:
        mfa_python = find_mfa_python(cfg.get("python_path", ""))
    if not mfa_python or not mfa_python.exists():
        print("ERROR: Cannot find Python with MFA installed.")
        print("Set python_path in config.yaml or use --python PATH")
        sys.exit(1)
    print(f"Using Python: {mfa_python}")

    # ═══════════════════════════════════════════════════════════════════════════
    # batch_ctc_ready: discover all datasets and process each one
    # ═══════════════════════════════════════════════════════════════════════════
    if mode == "batch_ctc_ready":
        bc = cfg.get("batch", {})
        ctc_root = resolve_input_path(bc.get("ctc_root", ""), PROJECT_ROOT)
        audio_root = resolve_input_path(bc.get("audio_root", ""), PROJECT_ROOT)
        output_root_path = resolve_input_path(bc.get("output_root", ""), PROJECT_ROOT)

        if not ctc_root.exists():
            print(f"ERROR: CTC root not found: {ctc_root}")
            sys.exit(1)

        # Discover datasets -- use cache if available
        datasets: list[str] = []
        batch_cache_data: dict | None = None
        datasets_from_cache = False

        if use_cache and not args.scan_only:
            batch_cache_data = load_scan_cache(cache_path)
            if batch_cache_data and batch_cache_data.get("datasets"):
                datasets = [d["name"] for d in batch_cache_data["datasets"]]
                datasets_from_cache = True
                print(f"\nBatch: {len(datasets)} datasets (from cache)")

        if not datasets_from_cache:
            # Scan: directories under ctc_root that have wavs/
            try:
                for entry in os.scandir(str(ctc_root)):
                    if entry.is_dir():
                        ctc_wavs = Path(entry.path) / "wavs"
                        if ctc_wavs.exists():
                            datasets.append(entry.name)
            except OSError:
                pass
            datasets.sort()
            print(f"\nBatch: {len(datasets)} datasets discovered")

        # Filter: optional include/exclude
        include = bc.get("include", None)
        exclude = bc.get("exclude", [])
        if include:
            datasets = [d for d in datasets if d in include]
        if exclude:
            datasets = [d for d in datasets if d not in exclude]

        if datasets:
            print(f"  First: {datasets[0]}")
            print(f"  Last:  {datasets[-1]}")

        if not datasets:
            print("ERROR: No datasets found!")
            sys.exit(1)

        # Limit: config > CLI override (0 = all)
        limit = bc.get("limit", 0)
        if args.dataset_limit > 0:
            limit = args.dataset_limit
        # Apply CLI offset first, then limit
        if args.dataset_offset > 0:
            if args.dataset_offset >= len(datasets):
                print(f"ERROR: --dataset-offset={args.dataset_offset} >= {len(datasets)} datasets")
                sys.exit(1)
            datasets = datasets[args.dataset_offset:]
            print(f"  Offset {args.dataset_offset} → {len(datasets)} remaining")
        if limit > 0:
            datasets = datasets[:limit]
            print(f"  Limited to first {limit}")

        # Process each dataset
        ok_count = 0
        fail_list: list[str] = []
        batch_cache_entries: list[dict] = []  # accumulate cache info per dataset
        for i, ds_name in enumerate(datasets):
            audiodir = audio_root / ds_name / "wavs"
            if not audiodir.exists():
                print(f"\n[{i+1}/{len(datasets)}] {ds_name} — SKIP (no audio)")
                continue

            n_files = count_files_fast(audiodir, ".wav")
            print(f"\n{'='*60}")
            print(f"  [{i+1}/{len(datasets)}] {ds_name} ({n_files} files)")
            print(f"{'='*60}")

            # Build sub-config for this dataset
            sub_cfg = dict(cfg)  # shallow copy of top-level keys
            sub_cfg["workspace"] = ds_name
            sub_cfg["data_dir"] = str(audiodir)
            sub_cfg.setdefault("ctc_ready", {})["ctc_dir"] = str(ctc_root / ds_name / "wavs")
            sub_cfg["output_dir"] = str(output_root_path / ds_name)

            # Resolve workspace
            sub_output_root = PROJECT_ROOT / "output"
            sub_output_root.mkdir(parents=True, exist_ok=True)
            sub_ws_name = ds_name
            if not sub_ws_name.isascii():
                sub_ws_name = __import__('re').sub(r'[^\x00-\x7F]+', '_', sub_ws_name).strip('_') or "workspace"
            sub_workspace = sub_output_root / sub_ws_name
            sub_workspace.mkdir(parents=True, exist_ok=True)

            # Resolve paths for this dataset
            sub_data_dir = resolve_input_path(sub_cfg["data_dir"], PROJECT_ROOT)
            sub_audio_dir = sub_data_dir  # ctc_ready: in-place audio
            sub_output_dir = resolve_input_path(sub_cfg.get("output_dir", "output"), sub_workspace)
            if not sub_output_dir.is_absolute():
                sub_output_dir = sub_workspace / sub_cfg.get("output_dir", "output")
            sub_aligned_dir = sub_workspace / sub_cfg.get("aligned_dir", "aligned")
            sub_filtered_dir = sub_workspace / sub_cfg.get("filtered_dir", "filtered")
            sub_validate_dir = sub_workspace / sub_cfg.get("validate_dir", "validate")
            sub_temp_dir = sub_workspace / sub_cfg.get("temp_dir", "temp")
            sub_ctc_pretg = sub_workspace / sub_cfg.get("ctc_pretg", "ctc_pretg")
            sub_ctc_pretg_adj = sub_workspace / sub_cfg.get("ctc_pretg_adj", "ctc_pretg_adj")

            for d in [sub_output_dir, sub_aligned_dir, sub_filtered_dir,
                       sub_validate_dir, sub_temp_dir, sub_ctc_pretg,
                       sub_ctc_pretg_adj, sub_workspace]:
                d.mkdir(parents=True, exist_ok=True)

            sub_ctx = {
                "data_dir": sub_data_dir,
                "audio_dir": sub_audio_dir,
                "pinyin_dir": sub_workspace / sub_cfg.get("pinyin_dir", "pinyin"),
                "aligned_dir": sub_aligned_dir,
                "output_dir": sub_output_dir,
                "filtered_dir": sub_filtered_dir,
                "validate_dir": sub_validate_dir,
                "models_dir": models_dir,
                "temp_dir": sub_temp_dir,
                "workspace": sub_workspace,
                "mfa_dict": mfa_dict,
                "mfa_audio_dir": sub_workspace / "audio_16k",
                "ctc_pretg": sub_ctc_pretg,
                "ctc_pretg_adj": sub_ctc_pretg_adj,
                # In ctc_ready mode the reference may be external.  If no
                # external directory is configured, step_link_ctc copies it
                # as *_ref.txt into the workspace CTC directory.
                "raw_text_dir": (
                    resolve_input_path(sub_cfg["ctc_ready"]["text_dir"], PROJECT_ROOT)
                    if sub_cfg.get("ctc_ready", {}).get("text_dir")
                    else sub_ctc_pretg
                ),
            }

            # Run all ctc_ready steps
            sub_args = argparse.Namespace(
                force=args.force, overwrite=args.overwrite,
                scan_only=args.scan_only, validate=False,
                **( {k: getattr(args, k, False)
                    for k in [f"skip_{s}" for s in STEPS]} )
            )
            for skip_s in ("trim", "prealign"):
                setattr(sub_args, f"skip_{skip_s}", True)

            sub_failed = []
            # In --scan-only mode, only run the link step
            scan_only_steps = ["link"] if args.scan_only else CTC_READY_STEP_ORDER
            for step_name in scan_only_steps:
                if getattr(sub_args, f"skip_{step_name}", False):
                    continue
                desc, func = STEPS[step_name]
                print(f"\n  [{step_name}] {desc}")
                rc = func(sub_args, sub_cfg, mfa_python, sub_ctx)
                if rc != 0:
                    sub_failed.append(step_name)
                    if not sub_args.force:
                        break

            if not sub_failed:
                print(f"  [{i+1}/{len(datasets)}] {ds_name} -- DONE")
                ok_count += 1
                # Record cache entry for this dataset
                batch_cache_entries.append({
                    "name": ds_name,
                    "audio_dir": str(audiodir),
                    "ctc_dir": str(ctc_root / ds_name / "wavs"),
                })
            else:
                print(f"  [{i+1}/{len(datasets)}] {ds_name} -- FAILED: {sub_failed}")
                fail_list.append(ds_name)
        # Save batch-level scan cache for future --use-cache runs
        if not datasets_from_cache or args.scan_only:
            batch_cache = {
                "config_file": str(config_path),
                "mode": "batch_ctc_ready",
                "ctc_root": str(ctc_root),
                "audio_root": str(audio_root),
                "output_root": str(output_root_path),
                "datasets": batch_cache_entries,
            }
            save_scan_cache(cache_path, batch_cache)

        print(f"\n{'#'*60}")
        print(f"  BATCH COMPLETE: {ok_count}/{len(datasets)} OK")
        if fail_list:
            print(f"  Failed: {', '.join(fail_list)}")
        print(f"{'#'*60}")
        return 0 if not fail_list else 1

    # Resolve workspace and paths
    # --workspace override: point ALL intermediate output to a custom root
    # (e.g., local SSD).  When not set, defaults to <project>/output/<workspace>/.
    if _strict_ready:
        workspace = _strict_workspace
        try:
            workspace.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"ERROR: strict workspace raced/preexists: {workspace}")
            return 1
    elif args.workspace:
        workspace = Path(args.workspace)
        if not workspace.is_absolute():
            workspace = PROJECT_ROOT / workspace
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        output_root = PROJECT_ROOT / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        workspace_name = cfg.get("workspace", "default")
        # MFA's C++ backend (pywrapfst) does not support non-ASCII paths on
        # Windows.  Warn and fall back to a safe ASCII name if needed.
        if not workspace_name.isascii():
            import re as _re
            safe = _re.sub(r'[^\x00-\x7F]+', '_', workspace_name).strip('_') or "workspace"
            print(f"WARNING: workspace name '{workspace_name}' contains non-ASCII chars.")
            print(f"  MFA cannot handle Unicode paths. Using '{safe}' instead.")
            workspace_name = safe
        workspace = output_root / workspace_name
        workspace.mkdir(parents=True, exist_ok=True)

    # Input: apply UNC->Linux translation, then resolve relative to PROJECT_ROOT
    data_dir = resolve_input_path(args.data_dir) if args.data_dir else resolve_input_path(cfg.get("data_dir", "data_dir"), PROJECT_ROOT)

    # ── NVMe audio cache detection ──
    # Check for a pre-populated audio cache on local NVMe (created by
    # scripts/cache_audio_to_nvme.py).  If found, use it as the audio
    # source to eliminate NAS I/O contention.
    # Priority: CLI --nvme-cache > config nvme_cache > auto-detect
    _nvme_override = getattr(args, "nvme_cache", None) or cfg.get("nvme_cache")
    _auto_cache = getattr(args, "auto_cache", False)
    if _strict_ready:
        _nvme_cache_dir, _nvme_is_temp = None, False
        print("  NVMe audio cache: DISABLED by strict ready evidence contract")
    else:
        _nvme_cache_dir, _nvme_is_temp = _resolve_nvme_cache(
            data_dir,
            nvme_override=None if _nvme_override is None else str(_nvme_override),
            auto_cache=_auto_cache,
        )
    if _nvme_cache_dir:
        print(f"  NVMe audio cache: {_nvme_cache_dir}"
              f"{' (temp, auto-clean)' if _nvme_is_temp else ' (permanent)'}")

    # In ctc_ready mode, audio_dir points to the source data_dir (already trimmed)
    # to avoid copying 100k+ files across SMB mounts
    # Resample reads from here and writes 16k audio locally
    if mode in ("ctc_ready", "nvrasr_fallback"):
        audio_dir = _nvme_cache_dir or data_dir  # prefer NVMe if available
    else:
        audio_dir = _nvme_cache_dir or (workspace / cfg.get("audio_dir", "audio"))
    pinyin_dir = workspace / cfg.get("pinyin_dir", "pinyin")
    aligned_dir = workspace / cfg.get("aligned_dir", "aligned")
    if args.output_dir:
        output_dir = resolve_input_path(args.output_dir, workspace)
    else:
        raw_out = cfg.get("output_dir", "output")
        out_p = resolve_input_path(raw_out, workspace)
        # If resolve_input_path returned a non-absolute path (relative), make it relative to workspace
        if not out_p.is_absolute():
            out_p = workspace / raw_out
        output_dir = out_p
    # strict-ok always uses a private run directory for *both* result sets.
    # The configured output is only a version-root selector; no existing NAS
    # result is a target for writes or merges.
    configured_output_dir = output_dir
    _nas_output_dir = None
    _strict_ok = bool(cfg.get("postprocess", {}).get("strict_ok", True))
    _run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}"
    _output_staging = (getattr(args, "output_staging", False)
                       or cfg.get("output_staging", False)) \
                      and not getattr(args, "no_output_staging", False)
    if _strict_ok:
        output_dir, filtered_dir, _nas_output_dir = strict_run_paths(
            workspace, configured_output_dir, _run_id, _output_staging)
        print(f"  Strict run output: {output_dir}")
        print(f"  Strict run filtered: {filtered_dir}")
        if _nas_output_dir is not None:
            print(f"  Versioned publish target: {_nas_output_dir}")
        else:
            print("  Strict run publication disabled (--no-output-staging or config)")
    elif _output_staging and output_dir:
        _nas_output_dir = output_dir
        output_dir = workspace / "output_staging" / f"{_run_id}_{os.getpid()}"
        print(f"  Run-specific output staging: {output_dir}")
        print(f"  Versioned publish target: {_nas_output_dir}")
    else:
        filtered_dir = workspace / cfg.get("filtered_dir", "filtered")
    validate_dir = workspace / cfg.get("validate_dir", "validate")
    temp_dir = workspace / cfg.get("temp_dir", "temp")

    # Check models (already resolved above)
    if not mfa_dict.exists():
        print(f"ERROR: MFA dictionary not found at {mfa_dict}")
        sys.exit(1)

    # Resolve steps — order depends on pipeline mode
    if mode == "ctc_ready":
        step_order = (STRICT_CTC_READY_STEP_ORDER
                      if _strict_ready else CTC_READY_STEP_ORDER)
    elif mode == "nvrasr_fallback":
        step_order = NVASR_FALLBACK_STEP_ORDER
    else:
        step_order = FULL_STEP_ORDER
    if not _strict_ok and "strict_ok" in step_order:
        step_order.remove("strict_ok")

    # ctc_ready mode: skip trim/prealign unconditionally (CTC already exists)
    if mode == "ctc_ready":
        for skip_s in ("trim", "prealign"):
            setattr(args, f"skip_{skip_s}", True)

    # nvrasr_fallback mode: skip trim (audio is pre-trimmed), keep prealign
    if mode == "nvrasr_fallback":
        setattr(args, "skip_trim", True)

    # Skip standalone MFA validate when configured (align validates internally)
    if cfg.get("mfa", {}).get("skip_validate", True):
        setattr(args, "skip_validate", True)
        if "validate" in step_order:
            step_order.remove("validate")

    if args.step:
        if args.step not in STEPS:
            print(f"Unknown step: {args.step}")
            sys.exit(1)
        run_list = [args.step]
    elif args.skip_to:
        if args.skip_to not in STEPS:
            print(f"Unknown step: {args.skip_to}")
            sys.exit(1)
        if args.skip_to not in step_order:
            step_order.append(args.skip_to)
        run_list = step_order[step_order.index(args.skip_to):]
    else:
        run_list = list(step_order)

    # --scan-only: only run the link step (single-dataset mode)
    if args.scan_only and mode in ("full", "ctc_ready"):
        if "link" in run_list:
            run_list = ["link"]
        else:
            run_list = run_list[:1]  # keep the first step
        print(f"  Scan-only mode: running only {run_list}")
    elif args.scan_only and mode == "nvrasr_fallback":
        print("  Scan-only mode: nvrasr_fallback has no link step, nothing to scan.")
        return 0

    run_list = [s for s in run_list if not getattr(args, f"skip_{s}", False)]

    if (_strict_ready and (run_list != STRICT_CTC_READY_STEP_ORDER
                           or "pad_silence" in run_list)):
        print("ERROR: strict v4 requires the complete no-padding stage route")
        return 1

    if not run_list:
        print("No steps to run.")
        return 0

    # Only create dirs needed by the steps being run
    _ctc_pretg_dir = workspace / cfg.get("ctc_pretg", "ctc_pretg")
    _ctc_pretg_adj_dir = workspace / cfg.get("ctc_pretg_adj", "ctc_pretg_adj")
    step_dirs = {
        "link": [audio_dir, _ctc_pretg_dir],
        "pad_silence": [workspace / "padded_audio", output_dir / "padded_audio"],
        "trim": [audio_dir, temp_dir],
        "resample": [temp_dir],
        "prealign": [_ctc_pretg_dir],
        "adjust": [_ctc_pretg_adj_dir],
        "validate": [validate_dir, temp_dir, _ctc_pretg_dir],
        "align": [aligned_dir, temp_dir, _ctc_pretg_dir],
        "align_en": [workspace / "en_phones", temp_dir],
        "postprocess": [output_dir, filtered_dir],
        "strict_ok": [output_dir, filtered_dir],
    }
    if _strict_ready:
        # The import step owns the first creation of every mutable target.
        # Precreating ctc_pretg would defeat the nonfresh-target gate.
        step_dirs = {name: [] for name in step_dirs}
    created: set[Path] = set()
    for s in run_list:
        for d in step_dirs.get(s, []):
            if d not in created:
                d.mkdir(parents=True, exist_ok=True)
                created.add(d)

    print(f"\n{'#'*60}")
    print(f"  Chinese MFA Pipeline  [{mode}]")
    print(f"  Input:  {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Steps:  {' -> '.join(run_list)}")
    print(f"{'#'*60}")

    ctx = {
        "data_dir": data_dir, "audio_dir": audio_dir,
        "pinyin_dir": pinyin_dir, "aligned_dir": aligned_dir,
        "output_dir": output_dir, "filtered_dir": filtered_dir,
        "validate_dir": validate_dir, "models_dir": models_dir,
        "temp_dir": temp_dir, "mfa_dict": mfa_dict,
        "workspace": workspace,
        "strict_ready": _strict_ready,
        "mfa_audio_dir": workspace / "audio_16k",
        "ctc_pretg": workspace / cfg.get("ctc_pretg", "ctc_pretg"),
        "ctc_pretg_adj": workspace / cfg.get("ctc_pretg_adj", "ctc_pretg_adj"),
        # Keep the reference transcript available to postprocess.  Full and
        # fallback modes read {stem}.txt from data_dir; ctc_ready uses the
        # configured external text_dir or the linked *_ref.txt files.
        "raw_text_dir": (
            resolve_input_path(cfg.get("ctc_ready", {}).get("text_dir"), PROJECT_ROOT)
            if mode == "ctc_ready" and cfg.get("ctc_ready", {}).get("text_dir")
            else (workspace / cfg.get("ctc_pretg", "ctc_pretg")
                  if mode == "ctc_ready" else data_dir)
        ),
    }

    failed = []
    for step_name in run_list:
        desc, func = STEPS[step_name]
        print(f"\n  >>> [{step_name}] {desc}")
        try:
            rc = func(args, cfg, mfa_python, ctx)
        except Exception as exc:
            if not _strict_ready:
                raise
            print(f"  ERROR: strict step {step_name} raised: {exc}")
            rc = 1
        if rc == 0 and _strict_ready:
            denominator_issues = strict_stage_denominator_issues(step_name, ctx)
            if denominator_issues:
                print(f"  ERROR: strict denominator gate failed after {step_name}")
                for issue in denominator_issues[:30]:
                    print(f"    - {issue}")
                if len(denominator_issues) > 30:
                    print(f"    ... and {len(denominator_issues) - 30} more")
                rc = 1
        if rc != 0:
            failed.append(step_name)
            if not args.force:
                print("  Stopping. Use --force to continue on errors.")
                break
        elif args.validate:
            issues = validate_step_output(step_name, workspace,
                                          cfg.get("output_spec", {}))
            if issues:
                print(f"  [VALIDATE] {step_name} — output check failed:")
                for issue in issues:
                    print(f"    {issue}")
            else:
                print(f"  [VALIDATE] {step_name} — OK")
        if args.stop_after and step_name == args.stop_after:
            print(f"\n  Stopped after '{step_name}' (--stop-after). Pipeline partial complete.")
            break

    # Re-prove that the immutable ready source and its dictionary were not
    # changed by any downstream stage.  This gate runs even when a later step
    # failed, as long as import itself completed, and always precedes publish.
    if _strict_ready and ctx.get("strict_ready_evidence") is not None:
        evidence = ctx["strict_ready_evidence"]
        source_ok = True
        try:
            if _sha256_file(evidence["_path"]) != ctx["strict_ready_evidence_sha256"]:
                print("  ERROR: pinned ready evidence changed during the pipeline")
                source_ok = False
            if _run_ready_verifier(cfg, evidence) != 0:
                print("  ERROR: authoritative verify-ready failed at pipeline end")
                source_ok = False
            if _sha256_file(evidence["_path"]) != ctx["strict_ready_evidence_sha256"]:
                print("  ERROR: pinned ready evidence changed during final verification")
                source_ok = False
        except Exception as exc:
            print(f"  ERROR: final ready-source verification raised: {exc}")
            source_ok = False
        if not source_ok and "ready_source_verify" not in failed:
            failed.append("ready_source_verify")

    # Clean up temporary 16kHz audio (default keep, configurable via keep_16k_audio)
    keep_16k = cfg.get("keep_16k_audio", True)
    if "resample" in run_list:
        mfa_audio = workspace / "audio_16k"
        if mfa_audio.exists() and not keep_16k:
            import shutil
            shutil.rmtree(str(mfa_audio))
            print(f"  Cleaned temp: {mfa_audio}")
        elif mfa_audio.exists():
            print(f"  Kept 16kHz audio: {mfa_audio}")

    # Save scan cache for future --use-cache runs (single-dataset mode).
    # Skip when running as a subprocess of streaming_pipeline (config mode
    # is batch_ctc_ready but --mode ctc_ready was passed on command line).
    _config_mode = cfg.get("mode", "")
    if (not _strict_ready and mode in ("ctc_ready", "full") and not failed
            and _config_mode != "batch_ctc_ready"):
        import json as _json
        manifest_path = workspace / cfg.get("ctc_pretg", "ctc_pretg") / "ctc_ready_manifest.json"
        n_stems = 0
        if manifest_path.exists():
            try:
                n_stems = len(_json.loads(manifest_path.read_text()).get("stems", []))
            except Exception:
                pass
        single_cache = {
            "config_file": str(config_path),
            "mode": mode,
            "workspace": cfg.get("workspace", "default"),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "n_stems": n_stems,
            "manifest_path": str(manifest_path),
        }
        if mode == "ctc_ready":
            single_cache["ctc_dir"] = cfg.get("ctc_ready", {}).get("ctc_dir", "")
        save_scan_cache(cache_path, single_cache)

    # ── Clean up temp NVMe cache ──
    if _nvme_cache_dir and _nvme_is_temp:
        _cleanup_nvme_cache(_nvme_cache_dir, _nvme_is_temp)

    # ── Publish output staging to a new versioned NAS directory ──
    _final_output = output_dir
    _should_publish = (
        _nas_output_dir is not None
        and "strict_ok" in run_list
        and output_dir.exists()
    )
    if _should_publish and failed:
        print(f"\n  Publish skipped because the pipeline failed: {', '.join(failed)}")
    elif _should_publish:
        print(f"\n  Publishing validated output: {_nas_output_dir}")
        from pipeline_utils import publish_output_versioned, write_publish_manifest
        manifest_path = write_publish_manifest(output_dir)
        print(f"  Publish manifest: {manifest_path}")
        if publish_output_versioned(output_dir, _nas_output_dir):
            print(f"  Published and verified: {_nas_output_dir}")
            _final_output = _nas_output_dir
        else:
            print(f"  WARNING: Publish refused/failed; staging remains at {output_dir}")
            failed.append("output_publish")

    print(f"\n{'#'*60}")
    print(f"  {'FAILED' if failed else 'DONE'}: {', '.join(failed) if failed else 'Success'}")
    print(f"  Output: {_final_output}")
    print(f"{'#'*60}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
