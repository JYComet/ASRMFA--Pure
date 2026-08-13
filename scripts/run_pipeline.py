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
import math
import os
import platform
import subprocess
import sys
import time
import shutil
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
    build_ctc_presence, build_file_index, build_flat_file_names,
    count_files_fast, find_wav,
    is_punct, is_word_like, is_english_token,
    load_ctc_token_entries, repair_degenerate_ctc_token_intervals,
    normalize_reference_numerals,
    read_ctc_textgrid_words, rebuild_lab_from_tokens,
    validate_ctc_transcript_bundle,
    validate_strict_mfa_textgrid,
    make_pipeline_run_id, write_pipeline_run_receipt,
    PIPELINE_ACCOUNTING_SCHEMA, make_pipeline_accounting_receipt,
    validate_pipeline_accounting_receipt,
    read_pipeline_accounting_receipt, write_pipeline_accounting_receipt,
    compute_model_tree_digest, stable_json_digest,
    write_ctc_run_receipt, make_audio_transform_receipt,
    write_audio_transform_receipt, make_mfa_input_axis_receipt,
    make_mfa_alignment_axis_receipt, validate_mfa_axis_receipts,
    validate_ctc_run_receipt_v2, validate_audio_transform_receipt,
    CTC_RUN_RECEIPT_SCHEMA,
    MFA_ALIGNMENT_AXIS_RECEIPT_V2_SCHEMA,
    make_mfa_alignment_axis_receipt_v2,
    MFA_INPUT_AXIS_RECEIPT_SCHEMA, MFA_ALIGNMENT_AXIS_RECEIPT_SCHEMA,
    _axis_audio_metadata,
    _textgrid_global_bounds,
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
        # MFA can legitimately omit a small number of utterances that cannot
        # be aligned.  Keep the default fail-closed, but allow task configs to
        # opt into explicit partial accounting (missing stems are rejected,
        # never silently dropped).
        "allow_partial": False,
        "min_output_ratio": 1.0,
    },
    "mfa_en": {
        "enabled": True,
        "num_jobs": 4,
        "normalize_workers": 0,       # 0=auto: min(32, cpu_count)
        "corpus_workers": 0,          # 0=auto: min(16, cpu_count)
        "padding_ms": 50,
        "min_segment_dur_ms": 150,
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
        stems_for_receipts = sorted(ctx.get("expected_stems", ()) or {p.stem for p in wavs})
        return _ensure_mfa_transform_receipts(ctx, stems_for_receipts)

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
    stems_for_receipts = sorted(ctx.get("expected_stems", ()) or {p.stem for p in wavs})
    if _ensure_mfa_transform_receipts(ctx, stems_for_receipts) != 0:
        return 1
    return 0 if done > 0 or skipped > 0 else 1


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
    if ctc_out.exists() and not args.overwrite:
        _marker = ctc_out / ".ctc_normalized"
        _manifest = ctc_out / "manifest.json"
        if _marker.is_file() and _manifest.is_file():
            _marker_data = parse_ctc_normalization_marker(
                _marker.read_text(encoding="utf-8"))
            if _marker_data is not None:
                # v4 marker carries content identity — verify manifest integrity
                import hashlib as _hashlib
                _actual_digest = _hashlib.sha256(
                    _manifest.read_bytes()).hexdigest()
                if _actual_digest == _marker_data.get("manifest_sha256"):
                    print(f"  CTC prealign complete ({_marker_data['stems']} stems,"
                          f" marker v4). Use --overwrite to re-run.")
                    rc = _load_ctc_accounting(ctx, required=True)
                    return rc if rc else _ensure_ctc_axis_receipt(ctx)
                print(f"  CTC .ctc_normalized manifest digest mismatch —"
                      f" re-running (stale marker).")
            else:
                # v3 legacy marker: accept (no content-identity to verify)
                print(f"  CTC prealign has legacy v3 marker — re-using."
                      f" Use --overwrite to upgrade to v4 provenance marker.")
                rc = _load_ctc_accounting(ctx, required=True)
                return rc if rc else _ensure_ctc_axis_receipt(ctx)
        else:
            _tg_count = len(list(ctc_out.glob("*.TextGrid")))
            _lab_count = len(list(ctc_out.glob("*.lab")))
            if _tg_count > 0:
                print(f"  CTC TextGrids ({_tg_count}) exist without normalization"
                      f" marker — re-running (incomplete bundle).")
            # fall through to re-run

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
    if pc.get("allow_missing_reference", False):
        prealign_args.append("--allow-missing-reference")
    if pc.get("limit", 0) > 0:
        prealign_args += ["--limit", str(pc["limit"])]
    if pc.get("offset", 0) > 0:
        prealign_args += ["--offset", str(pc["offset"])]
    if args.overwrite:
        prealign_args.append("--overwrite")
    if pc.get("all_gpus", False):
        prealign_args.append("--all-gpus")

    # Use run_python with the NVASR Python, not mfa_python
    rc = run_python(SCRIPTS_DIR / "ctc_prealign.py", prealign_args, nvras_py_path,
                    ctx["models_dir"], "Step 4: CTC Pre-alignment (NVASR -> MFA anchors)",
                    timeout=pc.get("timeout", 3600))
    if rc != 0:
        return rc
    rc = _load_ctc_accounting(ctx, required=True)
    return rc if rc else _ensure_ctc_axis_receipt(ctx)


def _ensure_ctc_axis_receipt(ctx: dict) -> int:
    """Bind the CTC producer receipt to its actual input audio axis.

    This function runs before resampling.  It therefore writes only CTC input
    bindings; the MFA input-axis receipt is created later by ``_guard_mfa_axis``
    from the resampled audio and its explicit transform receipts.
    """
    ctc_dir = Path(ctx["ctc_pretg"])
    audio_dir = Path(ctx.get("audio_dir", ctx.get("mfa_audio_dir", Path())))
    receipt_path = ctc_dir / ".ctc_run_receipt.json"
    if not receipt_path.is_file():
        print(f"  ERROR: missing CTC run receipt: {receipt_path}")
        return 1
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != CTC_RUN_RECEIPT_SCHEMA:
            raise ValueError("legacy CTC receipt is not trusted; v2 required")
        stems = sorted(receipt.get("input_stems", []))
        if not stems:
            stems = sorted(path.stem for path in ctc_dir.glob("*.lab"))
        bindings = []
        for stem in stems:
            wav = audio_dir / f"{stem}.wav"
            if wav.is_symlink() or not wav.is_file():
                raise ValueError(f"missing CTC MFA-axis audio: {stem}")
            metadata = _axis_audio_metadata(wav)
            bounds: list[float] = []
            token_path = ctc_dir / f"{stem}_tokens.jsonl"
            if token_path.is_file():
                for line in token_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    for key in ("start_s", "end_s"):
                        value = row.get(key)
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            bounds.append(float(value))
            row = {"stem": stem, "path": str(wav.resolve()), **metadata,
                   "ctc_bounds": {"xmin": min(bounds) if bounds else 0.0,
                                  "xmax": max(bounds) if bounds else metadata["duration_s"]},
                   "token_min_s": min(bounds) if bounds else 0.0,
                   "token_max_s": max(bounds) if bounds else metadata["duration_s"]}
            if token_path.is_file():
                row.update({"tokens_path": str(token_path.resolve()),
                            "tokens_sha256": _sha256_file(token_path)})
            for field, suffix in (("lab_sha256", ".lab"), ("punct_sha256", "_punct.json"), ("reference_sha256", "_ref.txt")):
                artifact = ctc_dir / f"{stem}{suffix}"
                if artifact.is_file() and not artifact.is_symlink():
                    row[field] = _sha256_file(artifact)
                    row[field + "_path"] = str(artifact.resolve())
            textgrid = ctc_dir / f"{stem}.TextGrid"
            if textgrid.is_file() and not textgrid.is_symlink():
                xmin, xmax = _textgrid_global_bounds(textgrid)
                row.update({"textgrid_path": str(textgrid.resolve()),
                            "textgrid_sha256": _sha256_file(textgrid),
                            "textgrid_xmin": xmin, "textgrid_xmax": xmax,
                            # CTC timeline bounds are the global TextGrid
                            # axis; token extrema remain separate evidence.
                            "ctc_bounds": {"xmin": xmin, "xmax": xmax}})
            bindings.append(row)
        receipt["audio_bindings"] = bindings
        receipt["audio_axis_role"] = "ctc_input_audio"
        receipt_errors = validate_ctc_run_receipt_v2(
            receipt, expected_stems=stems, audio_root=audio_dir)
        if receipt_errors:
            raise ValueError("; ".join(receipt_errors))
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        ctx["ctc_axis_receipt"] = receipt
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  ERROR: invalid CTC MFA-axis receipt: {exc}")
        return 1


def _ensure_mfa_transform_receipts(ctx: dict, stems: list[str]) -> int:
    """Create/validate identity resample lineage for every MFA WAV."""
    ctc_receipt = ctx.get("ctc_axis_receipt")
    if not isinstance(ctc_receipt, dict):
        try:
            ctc_receipt = json.loads(
                (Path(ctx["ctc_pretg"]) / ".ctc_run_receipt.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ERROR: CTC v2 receipt unavailable for resample lineage: {exc}")
            return 1
    if ctc_receipt.get("schema") != CTC_RUN_RECEIPT_SCHEMA:
        print("  ERROR: legacy CTC receipt cannot seed MFA transform lineage")
        return 1
    input_rows = {row.get("stem"): row for row in ctc_receipt.get("audio_bindings", [])
                  if isinstance(row, dict)}
    audio_root = Path(ctx["audio_dir"])
    mfa_root = Path(ctx["mfa_audio_dir"])
    receipt_dir = Path(ctx["workspace"]) / "audio_transform_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    try:
        for stem in sorted(stems):
            row = input_rows.get(stem)
            source = audio_root / f"{stem}.wav"
            output = mfa_root / f"{stem}.wav"
            if not isinstance(row, dict):
                raise ValueError(f"missing CTC input binding: {stem}")
            if Path(str(row.get("path", ""))).resolve() != source.resolve():
                raise ValueError(f"CTC input binding path mismatch: {stem}")
            if not source.is_file() or source.is_symlink() or not output.is_file() or output.is_symlink():
                raise ValueError(f"resample input/output missing or unsafe: {stem}")
            transform = make_audio_transform_receipt(source, output)
            receipt_path = receipt_dir / f"{stem}.{transform['output']['sha256']}.json"
            if receipt_path.is_file() and not receipt_path.is_symlink():
                transform = json.loads(receipt_path.read_text(encoding="utf-8"))
            else:
                write_audio_transform_receipt(receipt_path, transform)
            errors = validate_audio_transform_receipt(transform, input_audio=source,
                                                      output_audio=output)
            if errors:
                raise ValueError(f"{stem}: {'; '.join(errors)}")
            if transform["input"].get("sha256") != row.get("sha256"):
                raise ValueError(f"CTC/resample input hash mismatch: {stem}")
            records[stem] = {"path": str(receipt_path.resolve()),
                             "sha256": _sha256_file(receipt_path),
                             "receipt": transform}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  ERROR: resample transform lineage failed: {exc}")
        return 1
    ctx["mfa_transform_receipts"] = records
    return 0


def _guard_mfa_axis(ctx: dict, stems: list[str], ctc_dir: Path) -> int:
    """Build and validate the actual MFA input axis before any MFA process."""
    ctc_receipt = ctx.get("ctc_axis_receipt")
    if not isinstance(ctc_receipt, dict):
        try:
            ctc_receipt = json.loads((ctc_dir / ".ctc_run_receipt.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ERROR: MFA axis CTC receipt unavailable: {exc}")
            return 1
    if ctc_receipt.get("schema") != CTC_RUN_RECEIPT_SCHEMA:
        print("  ERROR: MFA axis requires CTC run receipt v2")
        return 1
    mfa_root = Path(ctx["mfa_audio_dir"])
    if _ensure_mfa_transform_receipts(ctx, stems) != 0:
        return 1
    input_rows = {row.get("stem"): row for row in ctc_receipt.get("audio_bindings", [])
                  if isinstance(row, dict)}
    axis_rows = []
    errors: list[str] = []
    for stem in sorted(stems):
        row = input_rows.get(stem)
        transform_record = ctx.get("mfa_transform_receipts", {}).get(stem, {})
        transform = transform_record.get("receipt")
        output = mfa_root / f"{stem}.wav"
        if not isinstance(row, dict) or not isinstance(transform, dict):
            errors.append(f"missing CTC/transform binding:{stem}")
            continue
        if transform.get("schema") != "audio-transform-receipt-v1":
            errors.append(f"transform schema mismatch:{stem}")
            continue
        if transform.get("scale") != 1.0 or any(transform.get(k) != 0.0 for k in ("head_transform_s", "tail_transform_s", "shift_s")):
            errors.append(f"non-identity audio transform:{stem}")
        if transform["input"].get("sha256") != row.get("sha256") or Path(transform["input"].get("path", "")).resolve() != Path(row.get("path", "")).resolve():
            errors.append(f"CTC/transform input mismatch:{stem}")
        if Path(transform["output"].get("path", "")).resolve() != output.resolve():
            errors.append(f"transform/MFA output path mismatch:{stem}")
        try:
            mfa_metadata = _axis_audio_metadata(output)
            if transform["output"].get("sha256") != mfa_metadata.get("sha256"):
                errors.append(f"MFA transform output hash mismatch:{stem}")
            if abs(float(row.get("ctc_bounds", {}).get("xmax", 0.0)) - float(mfa_metadata["duration_s"])) > 0.003:
                errors.append(f"CTC bound exceeds MFA audio axis:{stem}")
            axis_rows.append({"stem": stem, "path": str(output.resolve()), **mfa_metadata,
                              "ctc_bounds": row.get("ctc_bounds", {}),
                              "transform_receipt": transform_record.get("path")})
        except (OSError, ValueError, TypeError):
            errors.append(f"MFA axis audio unreadable:{stem}")
    if not errors:
        axis = make_mfa_input_axis_receipt(sorted(stems), axis_rows, axis_root=mfa_root,
                                           transform_receipts=[ctx["mfa_transform_receipts"][s]["path"] for s in sorted(stems)])
        axis["tts_authoritative_audio_root"] = str(Path(ctx.get("tts_authoritative_audio_dir", ctx["audio_dir"])).resolve())
        axis_path = ctc_dir / ".mfa_input_axis_receipt.json"
        axis_path.write_text(json.dumps(axis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        errors.extend(validate_mfa_axis_receipts(axis, stems=stems, audio_root=mfa_root,
                                                 ctc_receipt=ctc_receipt))
    if errors:
        print("  ERROR: MFA audio-axis guard failed")
        for error in errors[:20]:
            print(f"    - {error}")
        return 1
    ctx["mfa_input_axis_receipt"] = axis
    ctx["mfa_input_axis_receipt_path"] = ctc_dir / ".mfa_input_axis_receipt.json"
    ctx["mfa_axis_audio_dir"] = mfa_root
    return 0


def _write_mfa_alignment_axis_receipt(ctx: dict, aligned_dir: Path) -> int:
    axis = ctx.get("mfa_input_axis_receipt")
    if not isinstance(axis, dict):
        print("  ERROR: MFA input axis missing after alignment")
        return 1
    rows = []
    for stem in axis.get("stems", []):
        tg = aligned_dir / f"{stem}.TextGrid"
        if not tg.is_file() or tg.is_symlink():
            continue
        xmax = None
        for line in tg.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("xmax ="):
                try:
                    value = float(line.split("=", 1)[1].strip())
                    xmax = value if xmax is None else max(xmax, value)
                except ValueError:
                    pass
        binding = next((row for row in axis.get("audio", []) if row.get("stem") == stem), None)
        if xmax is None or not binding:
            print(f"  ERROR: MFA alignment axis metadata missing: {stem}")
            return 1
        if abs(float(xmax) - float(binding.get("duration_s"))) > 0.003:
            print(f"  ERROR: TextGrid/audio axis drift >3ms: {stem}")
            return 1
        rows.append({"stem": stem, "path": str(tg.resolve()),
                     "sha256": _sha256_file(tg), "xmax": xmax,
                     "audio_sha256": binding.get("sha256"),
                     "audio_duration_s": binding.get("duration_s")})
    expected = set(axis.get("stems", []))
    aligned = {row["stem"] for row in rows}
    missing = sorted(expected - aligned)
    declared_missing = set(ctx.get("mfa_missing_stems", ()))
    if missing and declared_missing != set(missing):
        print("  ERROR: MFA alignment axis missing ledger mismatch")
        return 1
    if missing:
        receipt = make_mfa_alignment_axis_receipt_v2(
            axis, rows, missing, alignment_root=aligned_dir)
    else:
        receipt = make_mfa_alignment_axis_receipt(axis, rows, alignment_root=aligned_dir)
    path = aligned_dir.parent / ".mfa_alignment_axis_receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ctx["mfa_alignment_axis_receipt_path"] = path
    return 0


def _axis_interface_args(ctx: dict) -> list[str] | None:
    """Return explicit axis bindings for downstream B-owned consumers.

    Production and strict-replay stages must pass all four role paths as
    validated artifacts.  No sibling-directory discovery is permitted here;
    callers fail before invoking the external postprocess/audit process when
    any binding is absent, unsafe, or inconsistent with its receipt.
    """
    required = bool(ctx.get("axis_contract_required", False)
                    or ctx.get("accounting_required", False)
                    or ctx.get("strict_replay_mode", False))
    if not required:
        return []

    input_path = Path(str(ctx.get("mfa_input_axis_receipt_path", "")))
    alignment_path = Path(str(ctx.get("mfa_alignment_axis_receipt_path", "")))
    try:
        mfa_root = Path(str(ctx.get("mfa_axis_audio_dir", "")))
        tts_root = Path(str(ctx.get("tts_authoritative_audio_dir", "")))
        if not input_path.is_absolute() or input_path.is_symlink() or not input_path.is_file():
            raise ValueError("MFA input-axis receipt unavailable")
        if not alignment_path.is_absolute() or alignment_path.is_symlink() or not alignment_path.is_file():
            raise ValueError("MFA alignment-axis receipt unavailable")
        if not mfa_root.is_absolute() or mfa_root.is_symlink() or not mfa_root.is_dir():
            raise ValueError("MFA axis audio root unavailable")
        if not tts_root.is_absolute() or tts_root.is_symlink() or not tts_root.is_dir():
            raise ValueError("TTS authoritative audio root unavailable")
        input_axis = json.loads(input_path.read_text(encoding="utf-8"))
        alignment_axis = json.loads(alignment_path.read_text(encoding="utf-8"))
        if input_axis.get("schema") != MFA_INPUT_AXIS_RECEIPT_SCHEMA:
            raise ValueError("MFA input-axis receipt schema mismatch")
        if alignment_axis.get("schema") not in (MFA_ALIGNMENT_AXIS_RECEIPT_SCHEMA,
                                                   MFA_ALIGNMENT_AXIS_RECEIPT_V2_SCHEMA):
            raise ValueError("MFA alignment-axis receipt schema mismatch")
        if Path(str(input_axis.get("axis_root", ""))).resolve() != mfa_root.resolve():
            raise ValueError("MFA axis root does not match input receipt")
        if Path(str(input_axis.get("tts_authoritative_audio_root", ""))).resolve() != tts_root.resolve():
            raise ValueError("TTS authoritative root does not match input receipt")
        if Path(str(alignment_axis.get("alignment_root", ""))).resolve() != Path(
                str(ctx.get("aligned_dir", ""))).resolve():
            raise ValueError("MFA alignment root does not match aligned directory")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  ERROR: axis interface binding failed: {exc}")
        return None
    return [
        "--mfa-input-axis-receipt", str(input_path.resolve()),
        "--mfa-alignment-axis-receipt", str(alignment_path.resolve()),
        "--mfa-axis-audio-root", str(mfa_root.resolve()),
        "--tts-authoritative-audio-root", str(tts_root.resolve()),
    ]


def _write_strict_replay_axis_receipts(ctx: dict) -> int:
    """Freeze minimal CTC/input/alignment axis receipts for imported replay data."""
    ctc_dir = Path(ctx["ctc_pretg"])
    audio_root = Path(ctx["mfa_axis_audio_dir"])
    stems = sorted(ctx.get("expected_stems", ()))
    bindings = []
    try:
        for stem in stems:
            wav = audio_root / f"{stem}.wav"
            metadata = _axis_audio_metadata(wav)
            bounds: list[float] = []
            token_path = ctc_dir / f"{stem}_tokens.jsonl"
            if token_path.is_file():
                for line in token_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    for key in ("start_s", "end_s"):
                        value = row.get(key)
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            bounds.append(float(value))
            bindings.append({"stem": stem, "path": str(wav.resolve()), **metadata,
                             "ctc_bounds": {"xmin": min(bounds) if bounds else 0.0,
                                            "xmax": max(bounds) if bounds else metadata["duration_s"]}})
        ctc_receipt = {"schema": "ctc-run-receipt-v2", "input_stems": stems,
                       "output_stems": stems, "audio_bindings": bindings}
        errors = validate_ctc_run_receipt_v2(ctc_receipt, expected_stems=stems,
                                             audio_root=audio_root)
        if errors:
            raise ValueError("; ".join(errors))
        ctc_receipt_path = ctc_dir / ".ctc_run_receipt.json"
        ctc_receipt_path.write_text(json.dumps(ctc_receipt, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
        axis = make_mfa_input_axis_receipt(stems, bindings, axis_root=audio_root)
        axis["tts_authoritative_audio_root"] = str(
            Path(ctx["tts_authoritative_audio_dir"]).resolve())
        axis_path = ctc_dir / ".mfa_input_axis_receipt.json"
        axis_path.write_text(json.dumps(axis, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        ctx["ctc_axis_receipt"] = ctc_receipt
        ctx["mfa_input_axis_receipt"] = axis
        ctx["mfa_input_axis_receipt_path"] = axis_path
        return _write_mfa_alignment_axis_receipt(ctx, Path(ctx["aligned_dir"]))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  ERROR: strict replay axis receipt generation failed: {exc}")
        return 1


def _load_ctc_accounting(ctx: dict, *, required: bool = False) -> int:
    """Load and pin the frozen v2 CTC denominator for downstream/resume use."""
    receipt_path = ctx.get("accounting_receipt_path")
    if receipt_path is None:
        if required:
            print("  ERROR: accounting receipt path was not explicitly bound")
            return 1
        return 0
    receipt_path = Path(receipt_path)
    if not receipt_path.is_file():
        # Initial full/NVASR stages emit their frozen source receipt in the
        # explicitly bound CTC path.  Promote that receipt to the formal
        # output path; never scan workspace siblings.
        source_path = ctx.get("accounting_source_receipt_path")
        if source_path is not None and Path(source_path).is_file():
            source_path = Path(source_path)
            try:
                source_receipt = read_pipeline_accounting_receipt(source_path)
                write_pipeline_accounting_receipt(receipt_path.parent, source_receipt)
            except Exception as exc:
                print(f"  ERROR: invalid/tampered source accounting receipt: {exc}")
                return 1
    if not receipt_path.is_file():
        if required:
            print(f"  ERROR: missing frozen v2 accounting receipt: {receipt_path}")
            return 1
        return 0
    try:
        receipt = read_pipeline_accounting_receipt(receipt_path)
    except Exception as exc:
        print(f"  ERROR: invalid/tampered v2 accounting receipt: {exc}")
        return 1
    eligible = tuple(receipt["eligible"]["stems"])
    existing = tuple(ctx.get("expected_stems", ()))
    if existing and set(existing) != set(eligible):
        print("  ERROR: frozen v2 eligible stems differ from pipeline denominator")
        return 1
    ctx.update({"accounting_receipt": receipt,
                "accounting_receipt_path": receipt_path,
                "accounting_source_stems": tuple(receipt["source"]["stems"]),
                "accounting_eligible_stems": eligible,
                "accounting_exclusions": tuple(receipt.get("exclusions", ())),
                "expected_stems": eligible})
    return 0


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
    import re
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
        repaired = repair_degenerate_ctc_token_intervals(tokens_path)
        if repaired:
            print(f"  Repaired {repaired} degenerate CTC token interval(s): {tokens_path.name}")
        stem = tokens_path.name[:-len("_tokens.jsonl")]
        lab_path = ctc_dir / f"{stem}.lab"
        textgrid_path = ctc_dir / f"{stem}.TextGrid"
        try:
            token_entries = load_ctc_token_entries(tokens_path)
            token_words = [entry["word"].strip() for entry in token_entries]
            tg_words = read_ctc_textgrid_words(textgrid_path)
            if token_words != tg_words:
                # NVASR's RIA normalizer can merge rui3+ya5 into one ``ria``
                # token before this validation step, while the producer's
                # TextGrid still contains the two original intervals.  This
                # is the same deterministic rewrite performed by
                # normalize_ria; apply it here so the bundle reaches that
                # step instead of being rejected prematurely.
                expanded_matches = True
                _ti = 0
                for _word in token_words:
                    if _word == "ria":
                        if _ti < len(tg_words) and tg_words[_ti] == "ria":
                            _ti += 1
                        elif (_ti + 1 < len(tg_words)
                              and re.fullmatch(r"rui[0-5]", tg_words[_ti])
                              and re.fullmatch(r"(?:ya|a)[0-5]", tg_words[_ti + 1])):
                            _ti += 2
                        else:
                            expanded_matches = False
                            break
                    else:
                        if _ti >= len(tg_words) or tg_words[_ti] != _word:
                            expanded_matches = False
                            break
                        _ti += 1
                if expanded_matches and _ti == len(tg_words):
                    from normalize_english_tokens import rewrite_ctc_textgrid_words
                    rewrite_ctc_textgrid_words(textgrid_path, token_entries)
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

    if ctc_out.exists() and not args.overwrite:
        # Verify adjusted output contains ALL input stems — not just any .TextGrid
        _in_labs = {p.stem for p in ctc_in.glob("*.lab")}
        _out_tgs = {p.stem for p in ctc_out.glob("*.TextGrid")}
        if _in_labs and _in_labs == _out_tgs:
            print(f"  Adjusted CTC anchors complete ({len(_out_tgs)} stems)."
                  f" Use --overwrite to re-run.")
            ctx["ctc_pretg_adj"] = ctc_out
            return 0
        elif _out_tgs:
            _missing = _in_labs - _out_tgs
            _extra = _out_tgs - _in_labs
            _detail = []
            if _missing:
                _detail.append(f"missing={len(_missing)}")
            if _extra:
                _detail.append(f"extra={len(_extra)}")
            print(f"  Adjusted CTC anchors incomplete ({', '.join(_detail)}) —"
                  f" re-running.")
        # fall through to re-run

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


def _validate_mfa_shard_axis_links(stems: list[str], link_tasks: list[tuple[Path, Path]],
                                   mfa_axis_receipt: dict) -> list[str]:
    """Validate every shard WAV symlink against the frozen MFA axis."""
    axis_rows = {row.get("stem"): row for row in mfa_axis_receipt.get("audio", [])}
    # ``link_tasks`` contains three artifacts per stem (lab, WAV, anchor).
    # Filtering the full task list once per stem is O(N²) at 54k scale and
    # can prevent MFA from ever launching.  Build the WAV destination index
    # once, then validate each frozen stem in O(1).
    wav_links: dict[str, list[Path]] = {}
    for src, dst in link_tasks:
        if src.suffix == ".wav":
            wav_links.setdefault(src.name, []).append(dst)
    errors: list[str] = []
    for stem in stems:
        row = axis_rows.get(stem)
        links = wav_links.get(f"{stem}.wav", [])
        if not isinstance(row, dict) or len(links) != 1:
            errors.append(f"MFA shard audio-axis binding missing: {stem}")
            continue
        link = links[0]
        try:
            if (not link.is_symlink()
                    or link.resolve(strict=True) != Path(row["path"]).resolve(strict=True)):
                errors.append(f"MFA shard audio symlink target mismatch: {stem}")
            elif _sha256_file(link.resolve(strict=True)) != row.get("sha256"):
                errors.append(f"MFA shard audio symlink hash mismatch: {stem}")
        except (OSError, ValueError, KeyError):
            errors.append(f"MFA shard audio symlink unreadable: {stem}")
    return errors


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
    allow_partial: bool = False,
    min_output_ratio: float = 1.0,
    desc: str = "MFA Align",
    mfa_axis_receipt: dict | None = None,
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

    if mfa_axis_receipt is not None:
        _axis_errors = _validate_mfa_shard_axis_links(stems, _link_tasks, mfa_axis_receipt)
        if _axis_errors:
            for _error in _axis_errors[:20]:
                print(f"  ERROR: {_error}")
            return 1

    # ── Launch parallel MFA instances ──
    _procs: list[tuple] = []
    _failed: list[int] = []
    _return_codes: dict[int, int | str] = {}
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
        # ── OSError capture (Case 83 / R7) ─────────────────────────
        try:
            _mfa_env = get_mfa_env(mfa_python, models_dir)
            _mfa_root = _sd / "mfa_root"
            _mfa_root.mkdir(parents=True, exist_ok=True)
            _mfa_env["MFA_ROOT_DIR"] = str(_mfa_root)
            _proc = subprocess.Popen(
                _cmd,
                env=_mfa_env,
                stdout=_log_handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as _os_err:
            _log_handle.close()
            _return_codes[_si] = f"os_error:{_os_err}"
            _failed.append(_si)
            print(f"  [shard {_si}] OSError starting MFA: {_os_err}")
            _procs.append((
                _si, None, _sd, None, _log_path,
                set(_ss), time.time(),
            ))
            continue
        # ─────────────────────────────────────────────────────────────
        _procs.append((
            _si, _proc, _sd, _log_handle, _log_path,
            set(_ss), time.time(),
        ))

    # ── Wait for all shards ──
    for (_si, _proc, _sd, _log_handle, _log_path,
         _expected, _started_at) in _procs:
        # ── Handle OSError from Popen (Case 83 / R7) ──────────────
        if _proc is None:
            # Already recorded as failed with os_error return code
            continue
        # ─────────────────────────────────────────────────────────────
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
            if _log_handle is not None:
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
    _usable_by_shard: dict[int, set[str]] = {}
    for (_si, _proc, _sd, _log_handle, _log_path,
         _expected, _started_at) in _procs:
        # Skip shards that failed at Popen (already recorded)
        if _proc is None:
            _manifest_rows.append({
                "shard": _si,
                "return_code": _return_codes.get(_si),
                "log": str(_log_path),
                "expected_count": len(_expected),
                "expected": sorted(_expected),
                "produced": [],
                "produced_count": 0,
                "missing": sorted(_expected),
                "extra": [],
                "invalid": [],
                "invalid_detail": [],
            })
            continue
        _tg_paths = sorted((_sd / "output").glob("*.TextGrid"))
        _produced = {path.stem for path in _tg_paths}
        _missing = sorted(_expected - _produced)
        _extra = sorted(_produced - _expected)
        _invalid: list[str] = []
        _invalid_detail: list[dict] = []
        for _tg in _tg_paths:
            try:
                _errors = validate_strict_mfa_textgrid(_tg)
                if _errors:
                    _invalid.append(_tg.stem)
                    _invalid_detail.append({"stem": _tg.stem, "errors": _errors})
            except OSError:
                _invalid.append(_tg.stem)
                _invalid_detail.append({"stem": _tg.stem, "errors": ["OSError reading TextGrid"]})
        _all_missing.extend(_missing)
        _all_extra.extend(_extra)
        _all_invalid.extend(_invalid)
        # Only rc=0 shards with valid TextGrids are eligible for a partial
        # merge.  A nonzero shard is fail-closed even if it left artifacts.
        if _return_codes.get(_si) == 0:
            _usable_by_shard[_si] = _produced - set(_invalid)
        else:
            _usable_by_shard[_si] = set()
        if _missing or _extra or _invalid:
            if _si not in _failed:
                _failed.append(_si)
        _manifest_rows.append({
            "shard": _si,
            "return_code": _return_codes.get(_si),
            "log": str(_log_path),
            "expected_count": len(_expected),
            "expected": sorted(_expected),
            "produced": sorted(_produced),
            "produced_count": len(_produced),
            "missing": _missing,
            "extra": _extra,
            "invalid": _invalid,
            "invalid_detail": _invalid_detail,
        })

    _manifest_path = _log_dir / "mfa_output_manifest.json"
    _reconciliations = [reconcile_mfa_outputs(row.get("expected", []), row.get("produced", []),
                                               return_code=(row.get("return_code") if isinstance(row.get("return_code"), int) else 1),
                                               invalid_stems=row.get("invalid", []))
                        for row in _manifest_rows]
    _retry_missing = sorted({stem for rec in _reconciliations
                             if rec.get("retry_missing") for stem in rec.get("missing", [])})
    if _retry_missing:
        _retry_invocation = 0
        def _execute_retained_missing(_missing, *, _beam=beam, _retry_beam=retry_beam):
            """Retry only retained missing shard inputs with resolved MFA context."""
            nonlocal _retry_invocation
            _retry_invocation += 1
            _root = _shard_root / f"retry_missing_{_retry_invocation:02d}"
            _root.mkdir(parents=True, exist_ok=False)
            _corpus, _audio, _out = _root / "corpus", _root / "audio", _root / "output"
            _corpus.mkdir(); _audio.mkdir(); _out.mkdir()
            for _stem in _missing:
                _src_shard = next((_sd for (_i, _p, _sd, _lh, _lp, _ex, _st) in _procs if _stem in _ex), None)
                if _src_shard is None: continue
                for _suffix in (".lab", ".txt"):
                    _src = _src_shard / "corpus" / f"{_stem}{_suffix}"
                    if _src.is_file(): shutil.copy2(_src, _corpus / _src.name)
                _src = _src_shard / "audio" / f"{_stem}.wav"
                if _src.is_file(): shutil.copy2(_src, _audio / _src.name)
            _cmd = [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa", "align",
                    str(_corpus), str(dict_path), acoustic_model, str(_out), "--audio_directory", str(_audio),
                    "--temporary_directory", str(_root / "temp"), "--output_format", output_format,
                    "--num_jobs", str(_jobs_per_shard), "--overwrite", "--no_textgrid_cleanup", "--single_speaker",
                    "--no_tokenization", "--beam", str(_beam), "--retry_beam", str(_retry_beam),
                    "--boost_silence", str(boost_silence)]
            _retry_env = get_mfa_env(mfa_python, models_dir)
            _retry_mfa_root = _root / "mfa_root"
            _retry_mfa_root.mkdir(parents=True, exist_ok=True)
            _retry_env["MFA_ROOT_DIR"] = str(_retry_mfa_root)
            _proc = subprocess.run(_cmd, cwd=str(PROJECT_ROOT), env=_retry_env,
                                   capture_output=True, text=True)
            (_root / "retry.stdout.log").write_text(_proc.stdout or "", encoding="utf-8")
            (_root / "retry.stderr.log").write_text(_proc.stderr or "", encoding="utf-8")
            _produced = [] ; _invalid = []
            for _tg in _out.glob("*.TextGrid"):
                if validate_strict_mfa_textgrid(_tg): _invalid.append(_tg.stem)
                else: _produced.append(_tg.stem)
            return {"return_code": _proc.returncode, "produced": _produced,
                    "invalid": _invalid, "exception": _proc.stderr[-500:],
                    "output_dir": str(_out)}
        _initial_attempt = {"return_code": 0, "produced": sorted(set().union(*(_usable_by_shard.values() or [set()]))),
                            "invalid": [], "exception": "rc0_incomplete"}
        _retry_state = run_mfa_retry_coordinator(
            sorted(set().union(*[set(row.get("expected", [])) for row in _manifest_rows])),
            _initial_attempt,
            _execute_retained_missing,
            # The eligible singleton is derived only after the cumulative
            # batch retry.  The executor is therefore supplied unconditionally
            # but can be reached solely after the coordinator's explicit
            # nonzero NoAlignmentsError isolation gate.
            rescue_executor=(lambda stem: _execute_retained_missing([stem], _beam=200, _retry_beam=800)))
        _manifest_path.write_text(_json.dumps({"schema": "mfa-output-manifest-v2", "run_id": _run_id,
            "expected_total": _n, "shards": _manifest_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        _retry_plan = _log_dir / "mfa_missing_retry_plan.json"
        _retry_plan.write_text(_json.dumps({"schema": MFA_RETRY_SCHEMA,
            "stems": _retry_missing, "reason": "rc0_incomplete_before_partial_admission",
            "workspaces_retained": True, "reconciliation": _reconciliations,
            "coordinator": _retry_state}, ensure_ascii=False, indent=2), encoding="utf-8")
        if _retry_state.get("merge_allowed"):
            # Each attempt is retained independently; merge every successful
            # attempt's valid grids so a singleton isolation/rescue cannot
            # discard the batch's individually recovered stems.
            _retry_outputs = [Path(_attempt["output_dir"])
                              for _attempt in _retry_state.get("attempts", [])[1:]
                              if _attempt.get("output_dir")]
            for _retry_out in _retry_outputs:
                for _tg in _retry_out.glob("*.TextGrid"):
                    _target_shard = next((_sd for (_i, _p, _sd, _lh, _lp, _ex, _st) in _procs if _tg.stem in _ex), None)
                    if _target_shard is not None and not validate_strict_mfa_textgrid(_tg):
                        shutil.copy2(_tg, _target_shard / "output" / _tg.name)
                        _usable_by_shard[next(_i for (_i, _p, _sd, _lh, _lp, _ex, _st) in _procs if _sd == _target_shard)].add(_tg.stem)
            _all_missing = [s for s in _all_missing if s not in _retry_state["history"][-1].get("produced", [])]
            _failed = [si for si, usable in _usable_by_shard.items()
                       if usable != set(_shard_stems[si])]
            print(f"  MFA retry coordinator recovered exact missing set; continuing strict merge")
        else:
            # A clean retry may still leave a small, explicit missing set.
            # If partial output is enabled and the configured ratio is met,
            # admit the recovered grids and carry the remainder as the
            # declared MFA-missing ledger.  Do not fabricate TextGrids for
            # stems that MFA could not align.
            _last_retry = _retry_state.get("history", [])[-1] if _retry_state.get("history") else {}
            _last_missing = set(_last_retry.get("missing", []))
            _last_extra = set(_last_retry.get("extra", []))
            _last_invalid = set(_last_retry.get("invalid", []))
            _last_produced = set(_last_retry.get("produced", []))
            _last_ratio = len(_last_produced) / len(stems) if stems else 0.0
            if (allow_partial and _last_missing and not _last_extra
                    and not _last_invalid and _last_ratio >= min_output_ratio):
                for _attempt in _retry_state.get("attempts", [])[1:]:
                    _retry_out = Path(_attempt.get("output_dir", ""))
                    if not _retry_out.is_dir():
                        continue
                    for _tg in _retry_out.glob("*.TextGrid"):
                        _target_shard = next(
                            (_sd for (_i, _p, _sd, _lh, _lp, _ex, _st) in _procs
                             if _tg.stem in _ex), None)
                        if _target_shard is not None and not validate_strict_mfa_textgrid(_tg):
                            shutil.copy2(_tg, _target_shard / "output" / _tg.name)
                            _target_i = next(
                                _i for (_i, _p, _sd, _lh, _lp, _ex, _st) in _procs
                                if _sd == _target_shard)
                            _usable_by_shard[_target_i].add(_tg.stem)
                _all_missing = sorted(_last_missing)
                _failed = [si for si, usable in _usable_by_shard.items()
                           if usable != set(_shard_stems[si])]
                print(f"  MFA retry recovered {_last_ratio:.2%}; continuing partial merge"
                      f" with {len(_all_missing)} declared missing stems")
            else:
                print(f"  MFA rc=0 incomplete; retained shard workspaces and wrote retry plan: {_retry_plan}")
                return 1
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

    _usable_stems = set().union(*_usable_by_shard.values()) if _usable_by_shard else set()
    _usable_ratio = (len(_usable_stems) / len(stems)) if stems else 0.0
    if _failed:
        if (not allow_partial or _usable_ratio < min_output_ratio
                or not _usable_stems):
            print(f"  ERROR: {len(set(_failed))}/{_n_shards} shards failed "
                  f"or produced an incomplete set")
            print(f"  Manifest: {_manifest_path}")
            print(f"  Shard workspace retained: {_shard_root}")
            return 1
        _missing_partial = sorted(set(stems) - _usable_stems)
        print(f"  WARNING: accepting partial MFA output: "
              f"{len(_usable_stems)}/{len(stems)} ({_usable_ratio:.2%}) "
              f"valid; skipping {len(_missing_partial)} missing/invalid stems")
        print(f"  Partial-MFA manifest: {_manifest_path}")

    # ── Merge aligned TextGrids ──
    _merged = 0
    for _si, _sd in enumerate(_shard_dirs):
        _usable = _usable_by_shard.get(_si, set())
        for _tg in (_sd / "output").glob("*.TextGrid"):
            if _tg.stem not in _usable:
                continue
            _dest = aligned_dir / _tg.name
            if overwrite or not _dest.exists():
                import shutil as _shutil
                _shutil.copy2(str(_tg), str(_dest))
                _merged += 1

    _aligned_now = {path.stem for path in aligned_dir.glob("*.TextGrid")}
    _expected_all = set(stems)
    if _aligned_now != _expected_all and not allow_partial:
        _missing_after = sorted(_expected_all - _aligned_now)
        _extra_after = sorted(_aligned_now - _expected_all)
        print("  ERROR: merged aligned set is inconsistent")
        print(f"    missing ({len(_missing_after)}): {_missing_after[:10]}")
        print(f"    extra ({len(_extra_after)}): {_extra_after[:10]}")
        print(f"  Shard workspace retained: {_shard_root}")
        return 1
    if _aligned_now != _expected_all:
        _missing_after = sorted(_expected_all - _aligned_now)
        print(f"  Partial MFA merge: {_merged} TextGrids; "
              f"missing {len(_missing_after)} stems are explicitly rejected")

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
    if _guard_mfa_axis(ctx, _stems, ctc_dir) != 0:
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
        allow_partial=bool(mc.get("allow_partial", False)),
        min_output_ratio=float(mc.get("min_output_ratio", 1.0)),
        desc="Step 6: MFA Align (sharded)",
        mfa_axis_receipt=ctx.get("mfa_input_axis_receipt"),
        **_extra,
    )
    if _rc is not None:
        if _rc == 0:
            _aligned_now = {path.stem for path in ctx["aligned_dir"].glob("*.TextGrid")}
            _missing_now = set(_stems) - _aligned_now
            if _missing_now:
                if not mc.get("allow_partial", False):
                    print(f"  ERROR: MFA output set incomplete: "
                          f"missing {len(_missing_now)} stems")
                    return 1
                ctx["mfa_missing_stems"] = tuple(sorted(_missing_now))
                ctx["mfa_aligned_stems"] = tuple(sorted(_aligned_now))
                print(f"  MFA partial mode: {len(_aligned_now)}/{len(_stems)} "
                      f"aligned; {len(_missing_now)} explicitly skipped")
            if _write_mfa_alignment_axis_receipt(ctx, ctx["aligned_dir"]) != 0:
                return 1
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
    if _write_mfa_alignment_axis_receipt(ctx, ctx["aligned_dir"]) != 0:
        return 1
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
    # ── Strict TextGrid validation (Case 83 / R7) ──────────────────
    _invalid_count = 0
    for _tg_path in sorted(ctx["aligned_dir"].glob("*.TextGrid")):
        _errors = validate_strict_mfa_textgrid(_tg_path)
        if _errors:
            if _invalid_count < 5:
                for _err in _errors[:3]:
                    print(f"    [{_tg_path.stem}] {_err}")
            _invalid_count += 1
    if _invalid_count:
        print(f"  ERROR: {_invalid_count} MFA TextGrid(s) failed strict validation")
        return 1
    # ─────────────────────────────────────────────────────────────────
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
        "--min-segment-dur-ms", str(en_cfg.get("min_segment_dur_ms", 150)),
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


def _refresh_postprocess_accounting(ctx: dict, output_stems: set[str],
                                    filtered_stems: set[str]) -> int:
    """Atomically refresh the formal v2 receipt after postprocess output."""
    receipt_path = Path(ctx.get("accounting_receipt_path", ""))
    source = ctx.get("accounting_receipt")
    try:
        if not isinstance(source, dict):
            if not receipt_path.is_file():
                raise ValueError("frozen accounting receipt unavailable")
            source = read_pipeline_accounting_receipt(receipt_path)
        if validate_pipeline_accounting_receipt(source):
            raise ValueError("frozen accounting receipt invalid")
        eligible = set(source["eligible"]["stems"])
        if (output_stems & filtered_stems
                or output_stems | filtered_stems != eligible):
            raise ValueError("postprocess output/filtered conservation mismatch")
        paths = dict(source.get("paths", {}))
        paths.update({"output": str(Path(ctx["output_dir"]).resolve()),
                      "filtered": str(Path(ctx["filtered_dir"]).resolve()),
                      "report": str((Path(ctx["output_dir"]) / "postprocess_report.jsonl").resolve())})
        route = list(source.get("route", []))
        if "postprocess" not in route:
            route.append("postprocess")
        refreshed = make_pipeline_accounting_receipt(
            list(source["source"]["stems"]), list(source["eligible"]["stems"]),
            source.get("exclusions", []), sorted(output_stems), sorted(filtered_stems),
            run_id=str(source.get("run_id", "")), mode=str(source.get("mode", "")),
            route=route, paths=paths, shards=source.get("shards"),
            extra=source.get("extra", {}))
        write_pipeline_accounting_receipt(receipt_path, refreshed)
        ctx["accounting_receipt"] = refreshed
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"  ERROR: postprocess formal accounting refresh failed: {exc}")
        return 1


def _refresh_strict_manifest_accounting_binding(output_dir: Path) -> int:
    """Rebind strict-ok evidence after the audit isolates late rejects.

    The audit may move a candidate from output/ to filtered/ after reading the
    pre-audit accounting receipt.  The final runner receipt is then refreshed
    with the post-audit partition, so the manifest's receipt hash and derived
    counts must be updated atomically before publication.
    """
    manifest_path = output_dir / "strict_ok_manifest.json"
    receipt_path = output_dir / ".pipeline_run_receipt_v2.json"
    try:
        if not manifest_path.is_file() or not receipt_path.is_file():
            raise ValueError("strict-ok manifest or accounting receipt missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = read_pipeline_accounting_receipt(receipt_path)
        receipt_errors = validate_pipeline_accounting_receipt(receipt)
        if receipt_errors:
            raise ValueError("accounting receipt invalid: " + "; ".join(receipt_errors))
        expected = set(manifest.get("expected_stems", []))
        output = set(receipt["output"]["stems"])
        filtered = set(receipt["filtered"]["stems"])
        if (expected != set(receipt["eligible"]["stems"])
                or output != {entry["stem"] for entry in manifest.get("ok", [])}
                or filtered != set(manifest.get("rejected", {}))):
            raise ValueError("strict manifest and final accounting sets disagree")
        manifest["pipeline_accounting_receipt"] = {
            "path": str(receipt_path.resolve()),
            "sha256": _sha256_file(receipt_path),
            "schema": PIPELINE_ACCOUNTING_SCHEMA,
        }
        manifest["pipeline_accounting"] = receipt.get("derived", {})
        tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, manifest_path)
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"  ERROR: strict-ok accounting binding refresh failed: {exc}")
        return 1


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
    # A resume/postprocess-only invocation must consume the frozen v2 CTC
    # evidence.  It may not rebuild a denominator from labs/audio subsets.
    _accounting_required = bool(ctx.get("accounting_required", False))
    if _accounting_required:
        if _load_ctc_accounting(ctx, required=True) != 0:
            return 1
    expected_stems = set(ctx.get("accounting_eligible_stems",
                                ctx.get("expected_stems", ())))
    mfa_missing_stems = set(ctx.get("mfa_missing_stems", ()))
    allow_partial_mfa = bool(cfg.get("mfa", {}).get("allow_partial", False))
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
    # A postprocess-only resume has no in-memory alignment context.  Recover
    # the exact missing-MFA ledger from the workspace, but only accept it when
    # it exactly matches the current aligned/expected partition.
    if allow_partial_mfa and not mfa_missing_stems and missing_aligned:
        _missing_set = set(missing_aligned)
        for _ledger in sorted(
                (ctx["workspace"] / "mfa_logs").glob("*/mfa_missing_stems.json"),
                reverse=True):
            try:
                _payload = json.loads(_ledger.read_text(encoding="utf-8"))
                _candidate = set(_payload.get("missing", ()))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if _candidate == _missing_set:
                mfa_missing_stems = _candidate
                print(f"  Recovered MFA missing ledger: {len(mfa_missing_stems)} stems "
                      f"from {_ledger}")
                break
    allowed_missing_aligned = bool(
        allow_partial_mfa and mfa_missing_stems
        and set(missing_aligned).issubset(mfa_missing_stems)
    )
    if (not expected_stems or corpus_stems != expected_stems
            or missing_audio or unexpected_aligned
            or (missing_aligned and not allowed_missing_aligned)):
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
    # word_in_silence has an explicit tri-state CLI because strict QC treats
    # an omitted value as enabled for legacy invocations.  Always serialize
    # the configured boolean, even when the broader suspicious-filter bundle
    # is disabled; otherwise ``filter_suspicious: false`` accidentally
    # resurrects this detector in the child process.
    if pc.get("enable_word_in_silence_filter", False):
        pp_args.append("--enable-word-in-silence-filter")
        pp_args += ["--filter-word-energy-ratio", str(pc.get("filter_word_energy_ratio", 2.0))]
    else:
        pp_args.append("--no-enable-word-in-silence-filter")
        pp_args += ["--filter-word-energy-ratio", "0"]
    # Quality filters
    if pc.get("filter_suspicious", True):
        if pc.get("filter_short_phone", True):
            pp_args += ["--filter-short-phone-sec", str(pc.get("filter_short_phone_sec", 0.015))]
        else:
            pp_args.append("--no-filter-short-phone")
        pp_args += ["--filter-long-word-sec", str(pc.get("filter_long_word_sec", 1.0))]
        pp_args += ["--filter-min-word-sec", str(pc.get("filter_min_word_sec", 0.15))]
        pp_args += ["--filter-min-word-dur-sec", str(pc.get("filter_min_word_dur_sec", 0.02))]
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
    if pc.get("allow_filtered_integrity_failures", False):
        pp_args.append("--allow-filtered-integrity-failures")
    # Text correction & unexpected silence handling
    if not pc.get("enable_text_correction", True):
        pp_args.append("--no-enable-text-correction")
    if not pc.get("handle_unexpected_sil", True):
        pp_args.append("--no-handle-unexpected-sil")
    if pc.get("workers", 0) > 0:
        pp_args += ["--workers", str(pc["workers"])]
    axis_args = _axis_interface_args(ctx)
    if axis_args is None:
        return 1
    pp_args += axis_args
    if args.overwrite:
        pp_args.append("--overwrite")
    rc = run_python(SCRIPTS_DIR / "postprocess_textgrids.py", pp_args, mfa_python,
                    ctx["models_dir"], "Step 7: Post-processing")
    if rc != 0:
        return rc

    # Preserve full-set accounting when MFA could not emit a TextGrid.  These
    # are rejected bookkeeping artifacts, not publication candidates: copy
    # the authoritative CTC anchor into filtered/ and add an explicit report
    # row. This prevents missing stems from silently disappearing.
    if allowed_missing_aligned:
        import json as _json
        import shutil as _shutil
        report_path = ctx["output_dir"] / "postprocess_report.jsonl"
        with report_path.open("a", encoding="utf-8") as _report_handle:
            for _stem in sorted(mfa_missing_stems):
                _src = ctc_dir / f"{_stem}.TextGrid"
                _dst = ctx["filtered_dir"] / f"{_stem}.TextGrid"
                if not _src.is_file():
                    print(f"  ERROR: missing CTC anchor for skipped MFA stem: {_stem}")
                    return 1
                if _dst.exists() and not args.overwrite:
                    print(f"  ERROR: filtered placeholder already exists: {_dst}")
                    return 1
                _shutil.copy2(_src, _dst)
                _report_handle.write(_json.dumps({
                    "stem": _stem,
                    "status": "filtered_missing_mfa_alignment",
                    "filter_reasons": ["missing_mfa_alignment"],
                    "output": str(_dst),
                    "reference_source": "mfa_missing_manifest",
                }, ensure_ascii=False) + "\n")
        print(f"  Partial MFA accounting: {len(mfa_missing_stems)} stems "
              f"placed in filtered/ as missing_mfa_alignment")

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

    if _refresh_postprocess_accounting(ctx, output_stems, filtered_stems) != 0:
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
    # The receipt is frozen at link/prealign in ctc_pretg; adjusted CTC dirs
    # intentionally contain only derived transcript artifacts.
    receipt_path = ctx.get("accounting_receipt_path") or (
        ctx["ctc_pretg"] / ".pipeline_run_receipt_v2.json")
    strict_args += ["--pipeline-receipt", str(receipt_path)]
    axis_args = _axis_interface_args(ctx)
    if axis_args is None:
        return 1
    strict_args += axis_args
    replay_en = ctx.get("strict_replay_english_import_path")
    if replay_en is not None:
        workspace = Path(ctx["workspace"]).resolve(strict=True)
        replay_en = Path(replay_en)
        try:
            replay_en = replay_en.resolve(strict=True)
            if (replay_en.is_symlink() or not replay_en.is_file()
                    or replay_en != workspace / "strict_replay_english_import.json"):
                raise ValueError("strict replay English import path unsafe")
            payload = json.loads(replay_en.read_text(encoding="utf-8"))
            if payload.get("schema") != STRICT_REPLAY_ENGLISH_IMPORT_SCHEMA or payload.get("scope") != "strict_replay":
                raise ValueError("strict replay English import schema invalid")
            import_path = workspace / "strict_replay_import.json"
            import_hash = _sha256_file(import_path)
            if payload.get("replay_import_manifest_sha256") != import_hash:
                raise ValueError("strict replay English import binding mismatch")
            formal = read_pipeline_accounting_receipt(Path(receipt_path))
            eligible = formal.get("eligible")
            if not isinstance(eligible, dict) or not isinstance(eligible.get("stems"), list):
                raise ValueError("strict replay formal eligible vector missing")
            expected = sorted(eligible["stems"])
            if (payload.get("eligible_stems") != expected
                    or payload.get("eligible_count") != len(expected)
                    or payload.get("eligible_digest") != stable_json_digest(expected)):
                raise ValueError("strict replay English eligible vector/count/digest mismatch")
            # Replay must bind every downstream evidence path explicitly.  In
            # particular, audit may not infer the English import, formal
            # receipt, immutable import, or report from sibling directories.
            formal_path = Path(ctx.get("accounting_receipt_path", ""))
            immutable_path = Path(ctx.get(
                "strict_replay_immutable_import_path", workspace / "strict_replay_import.json"))
            replay_report = Path(ctx.get(
                "strict_replay_postprocess_report_path",
                ctx["output_dir"] / "postprocess_report.jsonl"))
            strict_args += [
                "--strict-replay-english-import", str(replay_en),
                "--strict-replay-english-manifest", str(
                    ctx.get("strict_replay_english_manifest_path",
                           workspace / "en_phones" / "en_alignment_manifest.json")),
                "--strict-replay-english-subset", str(
                    ctx.get("strict_replay_english_subset_path",
                           workspace / "strict_replay_english_alignment_subset.json")),
                "--strict-replay-english-subset-sha256", _sha256_file(Path(
                    ctx.get("strict_replay_english_subset_path",
                            workspace / "strict_replay_english_alignment_subset.json"))),
                "--strict-replay-parent-english-sha256", _sha256_file(Path(
                    ctx.get("strict_replay_english_manifest_path",
                            workspace / "en_phones" / "en_alignment_manifest.json"))),
                "--strict-replay-formal-receipt", str(formal_path),
                "--strict-replay-immutable-import", str(immutable_path),
                "--strict-replay-postprocess-report", str(replay_report),
            ]
            ctx["strict_replay_strict_argv"] = list(strict_args)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ERROR: strict replay English import validation failed: {exc}")
            return 1
    elif ctx.get("strict_replay_mode"):
        print("  ERROR: strict replay English import path missing")
        return 1
    return run_python(SCRIPTS_DIR / "audit_strict_ok.py", strict_args, mfa_python,
                      ctx["models_dir"], "Step 8: strict-ok independent audit")


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_step_output(step_name: str, workspace: Path, spec: dict,
                         ctx: dict | None = None) -> list[str]:
    """Check that expected output files exist for *step_name*.

    Returns a list of failure descriptions (empty = all OK).
    """
    patterns = list(spec.get(step_name, []))
    # A disabled adjust step deliberately aliases ctc_pretg_adj to ctc_pretg.
    # Validate the active directory rather than a stale configured sibling;
    # missing artifacts still fail through the same pattern checks.
    if (step_name == "adjust" and isinstance(ctx, dict)
            and "ctc_pretg" in ctx and "ctc_pretg_adj" in ctx
            and Path(ctx.get("ctc_pretg_adj", "")) == Path(ctx.get("ctc_pretg", ""))):
        patterns = [pattern.replace("ctc_pretg_adj/", "ctc_pretg/", 1)
                    for pattern in patterns]
    if not patterns:
        return []

    failures: list[str] = []
    workspace_root = Path(workspace).resolve()
    active_roots: dict[str, Path] = {}
    if isinstance(ctx, dict):
        for label in ("output", "filtered"):
            key = f"{label}_dir"
            if key not in ctx:
                continue
            candidate = Path(ctx[key])
            try:
                resolved = candidate.resolve()
            except OSError as exc:
                failures.append(f"  UNSAFE: {key} cannot resolve: {exc}")
                continue
            if (not candidate.is_absolute() or candidate.is_symlink()
                    or not resolved.is_relative_to(workspace_root)):
                failures.append(f"  UNSAFE: {key} escapes active workspace: {candidate}")
                continue
            active_roots[label] = resolved

    def _zero_filtered_proof() -> bool:
        """Prove an intentionally empty filtered partition for strict runs."""
        filtered_root = active_roots.get("filtered")
        output_root = active_roots.get("output")
        if filtered_root is None or output_root is None:
            return False
        if (not filtered_root.is_dir() or filtered_root.is_symlink()
                or any(path.is_symlink() for path in filtered_root.glob("*.TextGrid"))
                or any(path.is_file() for path in filtered_root.glob("*.TextGrid"))):
            return False
        receipt_path = output_root / ".pipeline_run_receipt_v2.json"
        try:
            if receipt_path.is_symlink() or not receipt_path.is_file():
                return False
            receipt = read_pipeline_accounting_receipt(receipt_path)
            if validate_pipeline_accounting_receipt(receipt):
                return False
            filtered = receipt.get("filtered", {})
            output = receipt.get("output", {})
            eligible = receipt.get("eligible", {})
            filtered_stems = filtered.get("stems", [])
            output_stems = set(output.get("stems", []))
            eligible_stems = set(eligible.get("stems", []))
            return (filtered.get("count") == 0 and filtered_stems == []
                    and output_stems == eligible_stems
                    and output.get("count") == len(output_stems))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
    for pattern in patterns:
        root = workspace_root
        relative_pattern = pattern
        for label in ("output", "filtered"):
            prefix = f"{label}/"
            if pattern.startswith(prefix) and label in active_roots:
                root = active_roots[label]
                relative_pattern = pattern[len(prefix):]
                break
        matches = list(root.glob(relative_pattern))
        if not matches:
            if (pattern.startswith("filtered/") and label == "filtered"
                    and _zero_filtered_proof()):
                continue
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
    """Materialize an independent copy of *src* at *dst*.

    CTC-ready normalization and padding mutate files in the run workspace.
    Symlinks and hard links would therefore mutate the authoritative source
    (and could make concurrent runs share a Kaldi/CTC database).  The small
    amount of copy I/O is intentional: input provenance must be immutable.

    Returns True on success, False if *src* does not exist.
    """
    if not src.exists():
        return False
    if src.resolve() == dst.resolve():
        return True    # same file — nothing to do
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
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
        # Materialize the frozen v2 denominator alongside the strict import
        # after exact-namespace validation (the receipt is an allowed control
        # artifact, not a per-stem CTC artifact).
        _missing_refs = list(evidence.get("missing_reference", ()))
        _strict_receipt = make_pipeline_accounting_receipt(
            source_stems=sorted(set(selected) | set(_missing_refs)),
            eligible_stems=sorted(selected),
            exclusions={stem: "missing_reference" for stem in _missing_refs},
            output_stems=sorted(selected), filtered_stems=[],
            run_id=make_pipeline_run_id(), mode="ctc_ready",
            route=["link", "strict_import"],
            paths={"ctc": str(target), "audio": str(ctx["audio_dir"])},
            shards=[{"shard_id": "strict_import", "stems": sorted(selected)}],
            extra={"source_frozen": True, "strict_ready_evidence": str(evidence["_path"])},
        )
        _strict_receipt_path = workspace / ".pipeline_run_receipt_v2.json"
        write_pipeline_accounting_receipt(_strict_receipt_path, _strict_receipt)
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
                    "audio_dir": evidence["_audio_root"],
                    "accounting_receipt": _strict_receipt,
                    "accounting_receipt_path": _strict_receipt_path,
                    "accounting_source_stems": tuple(_strict_receipt["source"]["stems"]),
                    "accounting_eligible_stems": tuple(selected),
                    "accounting_exclusions": tuple(_strict_receipt["exclusions"])})
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

    def _bind_reuse_ctc_receipt(ctc_source: Path) -> int:
        """Require and bind a source CTC v2 receipt before reuse/resample."""
        source_receipt = ctc_source / ".ctc_run_receipt.json"
        target_receipt = Path(ctx["ctc_pretg"]) / ".ctc_run_receipt.json"
        try:
            if source_receipt.is_symlink() or not source_receipt.is_file():
                raise ValueError("reuse CTC requires source .ctc_run_receipt.json v2")
            receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
            if receipt.get("schema") != CTC_RUN_RECEIPT_SCHEMA:
                raise ValueError("reuse CTC legacy receipt is not trusted")
            stems = sorted(ctx.get("accounting_eligible_stems", receipt.get("input_stems", [])))
            # NVASR can leave a producer receipt for the requested batch even
            # when one stem has no complete six-file CTC bundle.  ``link`` has
            # already frozen the complete eligible subset, so project the
            # receipt onto that subset before validating it.  The omitted
            # source stem remains an explicit accounting exclusion instead of
            # poisoning the entire batch.
            stem_set = set(stems)
            receipt["input_stems"] = stems
            receipt["input_stems_digest"] = stable_json_digest(stems)
            receipt["output_stems"] = sorted(
                stem for stem in receipt.get("output_stems", [])
                if stem in stem_set)
            receipt["output_stems_digest"] = stable_json_digest(
                receipt["output_stems"])
            receipt["audio_bindings"] = sorted(
                (row for row in receipt.get("audio_bindings", [])
                 if isinstance(row, dict) and row.get("stem") in stem_set),
                key=lambda row: row.get("stem", ""))
            errors = validate_ctc_run_receipt_v2(receipt, expected_stems=stems,
                                                 audio_root=Path(ctx["audio_dir"]))
            if errors:
                raise ValueError("; ".join(errors))
            target_receipt.parent.mkdir(parents=True, exist_ok=True)
            target_receipt.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            ctx["ctc_axis_receipt"] = receipt
            return 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"ERROR: CTC reuse receipt validation failed: {exc}")
            return 1

    # Formal output accounting is a replay-only capability.  Do not infer
    # replay from the presence/shape of ``output_dir`` or scattered fields:
    # ordinary callers (including legacy unit fixtures) must retain their
    # historical CTC-only receipt behavior.  Conversely, an explicit replay
    # marker is fail-closed until every immutable role binding is present.
    _replay_marker = ctx.get("strict_replay_mode")
    _replay_accounting = False
    _replay_output: Path | None = None
    if _replay_marker is True:
        _replay_scope = ctx.get("strict_replay_scope")
        if _replay_scope != "strict_replay":
            print("ERROR: strict_replay_missing_output_dir: invalid replay scope")
            return 1
        required_replay = {
            "workspace": ctx.get("workspace"),
            "output_dir": ctx.get("output_dir"),
            "ctc_pretg": ctx.get("ctc_pretg"),
            "immutable_import": ctx.get("strict_replay_immutable_import_path"),
            "english_import": ctx.get("strict_replay_english_import_path"),
            "formal_receipt": ctx.get("strict_replay_formal_receipt_path"),
        }
        if any(not isinstance(value, Path) or not value.is_absolute()
               for value in required_replay.values()):
            print("ERROR: strict_replay_missing_output_dir: replay path bindings missing/type-invalid")
            return 1
        workspace = required_replay["workspace"]
        _replay_output = required_replay["output_dir"]
        ctc_target = required_replay["ctc_pretg"]
        immutable = required_replay["immutable_import"]
        english = required_replay["english_import"]
        formal = required_replay["formal_receipt"]
        if (_replay_output == workspace or _replay_output.is_symlink()
                or ctc_target.is_symlink()
                or workspace.is_symlink()
                or immutable != workspace / "strict_replay_import.json"
                or english != workspace / "strict_replay_english_import.json"
                or formal != _replay_output / ".pipeline_run_receipt_v2.json"):
            print("ERROR: strict_replay_missing_output_dir: unsafe replay role binding")
            return 1
        # Existing parents must be ordinary directories; output itself may be
        # created by the caller immediately before link.
        for label, directory in (("workspace", workspace), ("output", _replay_output),
                                 ("ctc", ctc_target)):
            parent = directory if directory.exists() else directory.parent
            if parent.is_symlink() or not parent.is_dir():
                print(f"ERROR: strict_replay_missing_output_dir: unsafe {label} parent")
                return 1
        if (not isinstance(cfg.get("ctc_ready"), dict)
                or not isinstance(cfg["ctc_ready"].get("ctc_dir"), str)
                or not cfg["ctc_ready"].get("ctc_dir")):
            print("ERROR: strict_replay_missing_output_dir: ctc_ready binding missing")
            return 1
        _replay_accounting = True

    if _strict_ready_mode(cfg):
        rc = _step_link_ctc_strict(args, cfg, ctx)
        if rc != 0:
            return rc
        return _bind_reuse_ctc_receipt(resolve_input_path(cfg["ctc_ready"]["ctc_dir"], PROJECT_ROOT))

    cr = cfg.get("ctc_ready", {})

    # -- Fast path: if manifest exists from a previous run, verify integrity --
    ctc_out_early = ctx["ctc_pretg"]
    manifest_early = ctc_out_early / "ctc_ready_manifest.json"
    if manifest_early.exists() and not args.overwrite:
        try:
            prev = _json.loads(manifest_early.read_text())
            prev_stems = prev.get("stems", [])
            if not prev_stems:
                raise ValueError("empty stems list")
            # Verify the flat manifest in one directory pass.  The previous
            # per-stem ``Path.is_file`` loop issued one metadata syscall per
            # expected stem; this preserves flat-layout semantics while
            # reusing a single inventory for the complete check.
            _present_stems = build_flat_file_names(ctc_out_early, ".lab")
            _missing = [s for s in prev_stems if s not in _present_stems]
            if _missing:
                print(f"  Link manifest has {len(_missing)}/{len(prev_stems)}"
                      f" missing .lab files — re-scanning.")
            else:
                print(f"  Link already done ({len(prev_stems)} stems verified)."
                      f" Use --overwrite to re-link.")
                rc = _load_ctc_accounting(ctx, required=True)
                return rc if rc else _bind_reuse_ctc_receipt(resolve_input_path(cr["ctc_dir"], PROJECT_ROOT))
        except Exception:
            pass  # corrupt/incomplete manifest, proceed with scan

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

    # Freeze the complete source WAV universe before applying CTC/reference
    # eligibility.  This evidence is reused by postprocess-only resumes.
    _source_audio_index = build_file_index(audio_src, ".wav")
    _source_stems = tuple(sorted(_source_audio_index))
    if len(_source_stems) != len(_source_audio_index):
        print("ERROR: duplicate WAV stems in source universe")
        return 1

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

    # CTC-ready runs are reference-authoritative by default.  The pipelined
    # NVASR fallback route can explicitly opt into ASR-only bundles, which
    # carry no TXT/_ref.txt but do have a complete CTC artifact set.
    _allow_missing_reference = bool(cr.get("allow_missing_reference", False))
    _authoritative_valid: list[str] = []
    _source_exclusions: dict[str, str] = {}
    for _stem in _source_stems:
        _txt = text_index.get(_stem)
        _bundled = (_ctc_base_cache.get(_stem, ctc_dir_src) / f"{_stem}_ref.txt")
        if _txt is None and not _bundled.is_file():
            if _allow_missing_reference and _stem in valid:
                _authoritative_valid.append(_stem)
            else:
                _source_exclusions[_stem] = "missing_reference"
        elif _stem in valid:
            _authoritative_valid.append(_stem)
        else:
            _source_exclusions[_stem] = "missing_ctc_bundle"
    # Stems present in CTC but absent from the WAV universe are not source
    # denominator members and are rejected by the existing manifest checks.
    valid = sorted(_authoritative_valid)
    ctx.update({"accounting_source_stems": _source_stems,
                "accounting_eligible_stems": tuple(valid),
                "accounting_exclusions": tuple(
                    {"stem": s, "reason": r}
                    for s, r in sorted(_source_exclusions.items()))})

    scan_only = getattr(args, 'scan_only', False)
    if not valid:
        print("ERROR: No valid stems — nothing to process.")
        return 1

    if scan_only:
        # Scan-only must still reject legacy/missing source axis receipts; it
        # may not create a synthetic trusted v2 receipt from directory layout.
        if _bind_reuse_ctc_receipt(ctc_dir_src) != 0:
            return 1

    # In scan-only mode, skip file linking — only validate + write manifest
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

    # Freeze source/eligible/exclusion evidence at link time.  Downstream
    # stages must consume this receipt rather than reconstructing a denominator
    # from whichever subset of files happens to remain on disk.
    try:
        _link_receipt = make_pipeline_accounting_receipt(
            source_stems=list(ctx.get("accounting_source_stems", _source_stems)),
            eligible_stems=list(valid),
            exclusions=list(ctx.get("accounting_exclusions", ())),
            output_stems=list(valid), filtered_stems=[],
            run_id=make_pipeline_run_id(), mode="ctc_ready",
            route=["link"],
            paths={"ctc": str(ctc_out), "audio": str(audio_src)},
            shards=[{"shard_id": "link", "stems": sorted(valid)}],
            extra={"source_frozen": True, "reference_authority": "txt_or_bundled_ref"},
        )
        write_pipeline_accounting_receipt(ctc_out, _link_receipt)
        ctx["accounting_receipt"] = _link_receipt
        if _replay_accounting:
            # Replay alone owns a formal output receipt.  All role paths were
            # validated before any import work, so this write cannot fall
            # back to an ordinary output_dir-shaped context.
            assert _replay_output is not None
            write_pipeline_accounting_receipt(_replay_output, _link_receipt)
            ctx["accounting_receipt_path"] = _replay_output / ".pipeline_run_receipt_v2.json"
        else:
            ctx["accounting_receipt_path"] = ctc_out / ".pipeline_run_receipt_v2.json"
        if _bind_reuse_ctc_receipt(ctc_dir_src) != 0:
            return 1
    except (TypeError, ValueError) as exc:
        print(f"ERROR: unable to write v2 accounting receipt: {exc}")
        return 1

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
    pre_ctc = not bool(expected_stems) and ctx.get("mode") in ("nvrasr_fallback", "full")
    if pre_ctc:
        try:
            # The fresh RIA cache is speaker-partitioned (one direct child per
            # speaker), so a direct ``iterdir`` scan silently produced an
            # empty denominator.  Freeze the physical source universe
            # recursively, while still rejecting duplicate *stems* because
            # every downstream CTC/MFA artifact is flat and stem-addressed.
            entries = sorted(
                (entry for entry in ctx["audio_dir"].rglob("*.wav")
                 if entry.is_file() and not entry.is_symlink()),
                key=lambda p: str(p),
            )
            by_stem: dict[str, list[Path]] = {}
            for entry in entries:
                by_stem.setdefault(entry.stem, []).append(entry)
            duplicates = {stem: paths for stem, paths in by_stem.items()
                          if len(paths) > 1}
            expected_stems = tuple(sorted(by_stem))
            if not expected_stems:
                print("  ERROR: pre-CTC physical WAV denominator is empty")
                return 1
            if duplicates:
                sample = "; ".join(
                    f"{stem}: {', '.join(str(p) for p in paths[:3])}"
                    for stem, paths in sorted(duplicates.items())[:5])
                print("  ERROR: pre-CTC physical WAV denominator has duplicate stems")
                print(f"    {sample}")
                return 1
            manifest = ctx["workspace"] / "pre_ctc_stems.txt"
            manifest.write_text("\n".join(expected_stems) + "\n", encoding="utf-8")
            ctx["expected_stems"] = expected_stems
            ctx["pre_ctc_stems_file"] = manifest
            # Keep the recursive inventory for the receipt pass below.  Calling
            # find_wav() once per stem would recursively rescan the entire
            # speaker-partitioned cache for every item (O(N^2)); on a 54k run
            # that can appear hung and is easy to interrupt after padding has
            # already completed.
            ctx["pre_ctc_audio_index"] = {
                stem: paths[0] for stem, paths in by_stem.items()
            }
        except OSError as exc:
            print(f"  ERROR: unable to freeze pre-CTC WAV denominator: {exc}")
            return 1
    if expected_stems:
        stems_file = ctx.get("strict_selected_stems_file", ctx.get("pre_ctc_stems_file"))
        if stems_file is None:
            print("  ERROR: padding denominator manifest is unavailable")
            return 1
        pad_args += ["--stems-file", str(stems_file)]
        if pre_ctc:
            pad_args.append("--pre-ctc")
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
                or (not pre_ctc and lab_stems != set(expected_stems))):
            print("  ERROR: padding success did not preserve the frozen denominator")
            return 1

    if rc == 0:
        transform_dir = ctx["workspace"] / "audio_transform_receipts"
        transform_dir.mkdir(parents=True, exist_ok=True)
        source_index = ctx.get("pre_ctc_audio_index")
        if not isinstance(source_index, dict):
            source_index = build_file_index(ctx["audio_dir"], ".wav")
        if len(source_index) < len(expected_stems):
            print("  ERROR: source WAV index is incomplete before receipt generation")
            return 1
        for stem in sorted(expected_stems):
            # Pre-CTC inputs may live below speaker subdirectories; resolve
            # them from the single recursive inventory created above.
            source_wav = source_index.get(stem)
            padded_wav = padded_audio_dir / f"{stem}.wav"
            try:
                if source_wav is None:
                    raise FileNotFoundError(f"source WAV not found for {stem}")
                transform = make_audio_transform_receipt(source_wav, padded_wav)
                write_audio_transform_receipt(transform_dir / f"{stem}.json", transform)
            except (OSError, ValueError) as exc:
                print(f"  ERROR: audio transform receipt failed for {stem}: {exc}")
                return 1
        # Switch audio_dir to padded versions for all downstream steps
        ctx["audio_dir"] = padded_audio_dir
        ctx["tts_authoritative_audio_dir"] = padded_audio_dir
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
CTC_READY_STEP_ORDER = ["link", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess", "strict_ok"]
STRICT_CTC_READY_STEP_ORDER = ["link", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess", "strict_ok"]
NVASR_FALLBACK_STEP_ORDER = ["pad_silence", "prealign", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess", "strict_ok"]


# ═══════════════════════════════════════════════════════════════════════════════
# Config schema validation (R8)
# ═══════════════════════════════════════════════════════════════════════════════

_KNOWN_TOP_KEYS: set[str] = {
    "mode", "workspace", "data_dir", "audio_dir", "pinyin_dir",
    "aligned_dir", "output_dir", "filtered_dir", "validate_dir",
    "temp_dir", "models_dir", "mfa_dict", "acoustic_model",
    "mfa", "mfa_en", "ctc_prealign", "ctc_adjust", "ctc_ready",
    "normalize", "normalize_ria", "normalize_en",
    "normalize_punct", "pad_silence", "postprocess",
    "audio_axis",
    "output_staging", "nvme_cache", "auto_cache", "keep_16k_audio",
    "strict_ctc_ready", "streaming", "pipelined", "batch", "pipeline",
    "use_cache",
    # Deprecated but still parsed
    "ctc_pretg", "ctc_pretg_adj", "output_spec",
    "skip_validate", "fine_tune", "fine_tune_boundary_tolerance",
    # Legacy top-level keys from DEFAULT_CFG
    "pinyin_dict", "python_path", "txt_suffix",
    "trim", "prepare", "resample",
}

_CONFIG_TYPES: dict[str, type | tuple] = {
    "mode": str,
    "workspace": str,
    "data_dir": str,
    "audio_dir": str,
    "output_dir": str,
    "filtered_dir": str,
    "acoustic_model": str,
    "mfa_dict": str,
    "mfa": dict,
    "mfa_en": dict,
    "ctc_prealign": dict,
    "ctc_adjust": dict,
    "ctc_ready": dict,
    "normalize": dict,
    "normalize_ria": dict,
    "normalize_en": dict,
    "normalize_punct": dict,
    "pad_silence": dict,
    "postprocess": dict,
    "audio_axis": dict,
    "strict_ctc_ready": dict,
    "streaming": dict,
    "pipelined": dict,
    "batch": dict,
    "pipeline": dict,
    "output_staging": bool,
    "use_cache": bool,
    "nvme_cache": (str, type(None)),
    "auto_cache": bool,
    "keep_16k_audio": bool,
    "disable_nvme_cache": bool,
}


def validate_config(cfg: dict, mode: str) -> list[str]:
    """Validate config schema: unknown keys, types, and cross-field rules.

    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    # 1. Unknown top-level keys
    for key in sorted(cfg):
        if key not in _KNOWN_TOP_KEYS:
            errors.append(f"unknown config key: {key!r}")

    # 2. Type checks
    for key, expected_type in _CONFIG_TYPES.items():
        if key in cfg:
            val = cfg[key]
            if not isinstance(val, expected_type):
                errors.append(
                    f"config key {key!r}: expected {expected_type},"
                    f" got {type(val).__name__}"
                )

    # 3. Required path existence (informational — paths may not exist at config time)
    for path_key in ("data_dir", "mfa_dict", "models_dir"):
        if path_key in cfg:
            p = Path(str(cfg[path_key]))
            if not p.is_absolute() and not p.exists():
                pass  # relative paths are resolved against PROJECT_ROOT later

    # 4. Mode/step incompatibilities
    pc = cfg.get("ctc_prealign", {})
    if pc.get("all_gpus") and pc.get("enabled") is False:
        errors.append("ctc_prealign.all_gpus=true but ctc_prealign.enabled=false")
    if (pc.get("enabled", True) and pc.get("all_gpus")
            and mode not in ("full", "ctc_ready", "nvrasr_fallback", "strict_replay", "filtered_recovery")):
        errors.append(
            f"ctc_prealign.all_gpus=true requires full or nvrasr_fallback mode,"
            f" got {mode!r}"
        )

    # 5. NVV configuration conflicts
    if pc.get("nvv_enabled") is False and pc.get("nvv_bias", 0) > 0:
        errors.append(
            "ctc_prealign.nvv_bias > 0 but nvv_enabled is false;"
            " bias will have no effect"
        )

    # Reference NVV labels are part of the authoritative transcript contract
    # and must not be controlled by the optional audio-discovery switch.
    if pc.get("reference_nvv_enabled", True) is not True:
        errors.append(
            "ctc_prealign.reference_nvv_enabled must remain true;"
            " reference NVV labels are always preserved"
        )

    # 6. strict_ctc_ready configuration
    scr = cfg.get("strict_ctc_ready", {})
    if scr.get("enabled") and mode != "ctc_ready":
        errors.append(
            f"strict_ctc_ready.enabled=true only applies to ctc_ready mode,"
            f" got {mode!r}"
        )

    # 7. Pad-silence + strict evidence conflict
    if cfg.get("strict_ctc_ready", {}).get("enabled"):
        ps = cfg.get("pad_silence", {})
        if ps.get("enabled", True):
            errors.append(
                "strict_ctc_ready requires pad_silence.enabled=false"
                " (strict evidence contract does not allow padding)"
            )

    # 8. Streaming configuration
    sc = cfg.get("streaming", {})
    if sc.get("enabled") and not sc.get("local_work"):
        errors.append("streaming.enabled=true requires streaming.local_work")

    # 9. CTC adjust + pad_silence ordering
    ca = cfg.get("ctc_adjust", {})
    if ca.get("enabled", True) and mode == "nvrasr_fallback":
        # adjust runs after pad_silence in nvrasr_fallback — intentional
        pass

    # 10. Single MFA-axis audio contract.  Reuse-CTC may not silently consume
    # post-CTC padded audio; an explicit v2 axis receipt is mandatory.
    axis = cfg.get("audio_axis", {})
    if mode == "strict_replay" or axis:
        if axis.get("mfa_axis_role") != "mfa_axis_audio":
            errors.append("audio_axis.mfa_axis_role must be mfa_axis_audio")
        if axis.get("tts_authoritative_role") != "tts_authoritative_audio":
            errors.append("audio_axis.tts_authoritative_role must be tts_authoritative_audio")
        if axis.get("reuse_ctc_requires_receipt") is not True:
            errors.append("audio_axis.reuse_ctc_requires_receipt must be true")
        if axis.get("post_ctc_pad_silence") != "forbidden":
            errors.append("audio_axis.post_ctc_pad_silence must be forbidden")
    # Reuse is selected by the resolved top-level mode, not by a nested
    # ctc_ready.enabled flag (which is not part of the routing contract).
    if (mode == "ctc_ready" or bool(scr.get("enabled"))) \
            and cfg.get("pad_silence", {}).get("enabled", True):
        errors.append("reuse-CTC with post-CTC pad_silence is forbidden")

    return errors


# ---------------------------------------------------------------------------
# Strict replay import (first-class, opt-in, fail-closed)
# ---------------------------------------------------------------------------

STRICT_REPLAY_CANONICAL_SHA256 = "d88b9ac874283dbc67dc38003fb78d872b799597ce940175a8301f78aa2c5bcf"
STRICT_REPLAY_CONFIG_PATH = PROJECT_ROOT / "configs" / "hecheng_ria_fresh.yaml"
STRICT_REPLAY_CONFIG_SHA256 = "78053e0455711ae943e8fabf2926014006f9f9449a96f8414c91b857a15ad553"
STRICT_REPLAY_CANONICAL_SCHEMA = "mfa-quality-canonical-samples-v1"
STRICT_REPLAY_SCHEMA = "strict-replay-import-v2.1"
STRICT_REPLAY_CTC_SUFFIXES = (".TextGrid", ".lab", "_tokens.jsonl", "_punct.json",
                               "_text_cn.txt", "_text_raw.txt")
STRICT_REPLAY_CATEGORIES = (
    "accepted", "missing_mfa", "pp_tier_gaps", "word_in_silence",
    "tier_discontinuity", "english_phone_deficit", "short_word",
    "english_provenance_rejected",
)
STRICT_REPLAY_RANGES = ("000000-017999", "018000-035999", "036000-053999")

# ---------------------------------------------------------------------------
# Filtered recovery (provenance-quarantined, non-publishing)
# ---------------------------------------------------------------------------

FILTERED_RECOVERY_SCHEMA = "filtered-recovery-import-v1"
MFA_RETRY_SCHEMA = "mfa-retry-evidence-v1"
MFA_RESCUE_SCHEMA = "mfa-rescue-evidence-v1"


def reconcile_mfa_outputs(expected_stems, produced_stems, *, return_code: int,
                          invalid_stems=()) -> dict:
    """Reconcile an MFA attempt before any partial merge or cleanup."""
    expected = set(_filtered_recovery_sorted_unique(list(expected_stems), "MFA expected"))
    produced = set(_filtered_recovery_sorted_unique(list(produced_stems), "MFA produced"))
    invalid = set(_filtered_recovery_sorted_unique(list(invalid_stems), "MFA invalid"))
    return {"return_code": return_code, "expected": sorted(expected),
            "produced": sorted(produced), "missing": sorted(expected - produced),
            "extra": sorted(produced - expected), "invalid": sorted(invalid),
            "complete": return_code == 0 and produced == expected and not invalid,
            "retry_missing": return_code == 0 and bool(expected - produced) and not (produced - expected)}


def mfa_retry_state_machine(expected_stems, attempts, *, rescue_stem=None) -> dict:
    """Fail-closed retry policy with explicit batch/isolation/rescue phases.

    ``produced`` in an executor result is that attempt's individual output;
    ``produced_individual`` is accepted as an explicit alias when callers
    have already annotated it.  Reconciliation always uses the cumulative
    union, while retaining each attempt in ``history`` for auditability.
    """
    expected = set(_filtered_recovery_sorted_unique(list(expected_stems), "MFA expected"))
    state = "initial"; rescue_used = False; history = []; cumulative: set[str] = set()
    for index, attempt in enumerate(attempts):
        individual = set(_filtered_recovery_sorted_unique(
            list(attempt.get("produced_individual", attempt.get("produced", []))),
            f"MFA produced attempt {index}"))
        cumulative.update(individual)
        result = reconcile_mfa_outputs(expected, sorted(cumulative),
                                       return_code=int(attempt.get("return_code", 1)),
                                       invalid_stems=attempt.get("invalid", []))
        result["attempt_index"] = index
        result["produced_individual"] = sorted(individual)
        result["produced"] = sorted(cumulative)
        history.append(result)
        if result["complete"]:
            state = "complete"; break
        clean_missing = (not result["extra"] and not result["invalid"]
                         and bool(result["missing"]))
        if index == 0 and result["retry_missing"]:
            state = "unchanged_retry"; continue
        # A batch retry that leaves exactly one cumulative gap must be
        # followed by an unchanged 20/80 singleton isolation attempt.
        if index == 1 and clean_missing and len(result["missing"]) == 1:
            state = "singleton_isolation"; continue
        # Only a nonzero, explicit NoAlignmentsError from that isolation
        # attempt can authorize the single 200/800 rescue.
        if (index == 2 and rescue_stem and clean_missing
                and result["missing"] == [rescue_stem]
                and int(attempt.get("return_code", 1)) != 0
                and "NoAlignmentsError" in str(attempt.get("exception", ""))):
            rescue_used = True; state = "singleton_rescue"; continue
        state = "permanent_failure"; break
    return {"state": state, "rescue_used": rescue_used, "history": history,
            "merge_allowed": state == "complete"}


def run_mfa_retry_coordinator(expected_stems, initial_attempt: dict, retry_executor,
                              *, rescue_stem=None, rescue_executor=None) -> dict:
    """Execute batch retry, singleton isolation, then one guarded rescue."""
    expected = set(_filtered_recovery_sorted_unique(list(expected_stems), "MFA expected"))
    attempts = [dict(initial_attempt)]
    effective_rescue_stem = rescue_stem
    initial_rec = reconcile_mfa_outputs(
        expected, initial_attempt.get("produced_individual", initial_attempt.get("produced", [])),
        return_code=int(initial_attempt.get("return_code", 1)),
        invalid_stems=initial_attempt.get("invalid", []))
    if initial_rec["retry_missing"]:
        batch_missing = initial_rec["missing"]
        batch = dict(retry_executor(batch_missing))
        batch["produced_individual"] = sorted(set(batch.get("produced", [])))
        attempts.append(batch)
        # Evaluate the cumulative batch result before deciding whether the
        # unchanged 20/80 singleton isolation is warranted.
        batch_state = mfa_retry_state_machine(expected, attempts, rescue_stem=rescue_stem)
        batch_history = batch_state["history"][-1]
        if (batch_state["state"] == "singleton_isolation"
                and len(batch_history["missing"]) == 1):
            # In production the initial missing set can be large, so the
            # singleton eligible for rescue is only knowable after the batch
            # union.  A caller-supplied stem remains a strict binding check.
            effective_rescue_stem = (batch_history["missing"][0]
                                     if rescue_stem is None else rescue_stem)
            isolation = dict(retry_executor(batch_history["missing"]))
            isolation["produced_individual"] = sorted(set(isolation.get("produced", [])))
            attempts.append(isolation)
            isolation_state = mfa_retry_state_machine(expected, attempts,
                                                      rescue_stem=effective_rescue_stem)
            isolation_history = isolation_state["history"][-1]
            explicit_noalign = (
                isolation_history["missing"] == [effective_rescue_stem]
                and not isolation_history["extra"] and not isolation_history["invalid"]
                and int(isolation.get("return_code", 1)) != 0
                and "NoAlignmentsError" in str(isolation.get("exception", "")))
            if explicit_noalign and rescue_executor and effective_rescue_stem:
                rescue = dict(rescue_executor(effective_rescue_stem))
                rescue["produced_individual"] = sorted(set(rescue.get("produced", [])))
                attempts.append(rescue)
    state = mfa_retry_state_machine(expected, attempts, rescue_stem=effective_rescue_stem)
    state["attempts"] = attempts
    return state


def _filtered_recovery_sorted_unique(stems, label: str) -> list[str]:
    """Normalize a recovery stem sequence and reject duplicates/path escapes."""
    if not isinstance(stems, (list, tuple, set, frozenset)):
        raise ValueError(f"filtered recovery {label} must be a stem sequence")
    values = [str(stem) for stem in stems]
    if any(not stem or Path(stem).name != stem for stem in values):
        raise ValueError(f"filtered recovery {label} contains malformed stem")
    if len(values) != len(set(values)):
        raise ValueError(f"filtered recovery {label} contains duplicate stems")
    return sorted(values)


def validate_filtered_recovery_partition(
    frozen_stems, accepted_stems, recovered_stems, still_filtered_stems,
    *, expected_count: int | None = None,
) -> dict:
    """Validate the independent final recovery accounting for any frozen set.

    The parent accepted set is deliberately supplied separately and can never
    be copied into recovery.  Recovery rows must form a disjoint union of the
    frozen parent-filtered set, with no omissions, extras, or duplicate rows.
    """
    frozen = set(_filtered_recovery_sorted_unique(frozen_stems, "frozen"))
    accepted = set(_filtered_recovery_sorted_unique(accepted_stems, "accepted"))
    recovered = set(_filtered_recovery_sorted_unique(recovered_stems, "recovered"))
    filtered = set(_filtered_recovery_sorted_unique(still_filtered_stems, "still_filtered"))
    if expected_count is None:
        expected_count = len(frozen)
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("filtered recovery expected_count must be a positive integer")
    if len(frozen) != expected_count:
        raise ValueError(f"filtered recovery frozen set must contain exactly {expected_count} stems")
    if frozen & accepted:
        raise ValueError("filtered recovery frozen/parent accepted intersection is non-empty")
    if recovered & filtered:
        raise ValueError("filtered recovery recovered/still-filtered overlap")
    if recovered | filtered != frozen:
        missing = sorted(frozen - (recovered | filtered))
        extra = sorted((recovered | filtered) - frozen)
        raise ValueError(f"filtered recovery union mismatch: missing={missing[:5]} extra={extra[:5]}")
    return {
        "source": expected_count,
        "eligible": expected_count,
        "exclusions": 0,
        "output": len(recovered),
        "filtered": len(filtered),
        "recovered_stems": sorted(recovered),
        "still_filtered_stems": sorted(filtered),
        "parent_accepted_intersection": 0,
    }


def validate_filtered_recovery_rows(report_rows, frozen_stems, accepted_stems) -> dict:
    """Validate exactly one final taxonomy/report row per frozen stem."""
    frozen = _filtered_recovery_sorted_unique(frozen_stems, "frozen")
    if not isinstance(report_rows, list) or len(report_rows) != len(frozen):
        raise ValueError(f"filtered recovery final report must contain exactly {len(frozen)} rows")
    by_stem: dict[str, dict] = {}
    for row in report_rows:
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
            raise ValueError("filtered recovery final report row malformed")
        stem = row["stem"]
        if stem in by_stem:
            raise ValueError("filtered recovery final report contains duplicate stem")
        by_stem[stem] = row
    recovered = [stem for stem, row in by_stem.items() if row.get("status") == "ok"]
    still_filtered = [stem for stem, row in by_stem.items()
                      if isinstance(row.get("status"), str)
                      and row["status"].startswith("filtered")]
    if len(recovered) + len(still_filtered) != len(frozen):
        raise ValueError("filtered recovery final report has invalid taxonomy statuses")
    return validate_filtered_recovery_partition(
        frozen_stems, accepted_stems, recovered, still_filtered)


def validate_filtered_recovery_manifest(
    manifest: dict, frozen_stems, accepted_stems, *,
    parent_hashes: dict[str, str] | None = None,
    observed_parent_hashes: dict[str, str] | None = None,
    expected_mismatch: dict[str, str] | None = None,
) -> dict:
    """Fail-closed validation for a subset/full frozen retry manifest."""
    if not isinstance(manifest, dict) or manifest.get("schema") != FILTERED_RECOVERY_SCHEMA:
        raise ValueError("filtered recovery manifest schema mismatch")
    frozen = set(_filtered_recovery_sorted_unique(frozen_stems, "frozen"))
    accepted = set(_filtered_recovery_sorted_unique(accepted_stems, "accepted"))
    selected = _filtered_recovery_sorted_unique(manifest.get("stems", []), "selected")
    if not selected:
        raise ValueError("filtered recovery manifest selected stems are empty")
    if set(selected) - frozen:
        raise ValueError("filtered recovery manifest contains stem outside frozen set")
    if set(selected) & accepted:
        raise ValueError("filtered recovery manifest contains parent accepted stem")
    if manifest.get("source") != len(selected) or manifest.get("eligible") != len(selected):
        raise ValueError("filtered recovery manifest source/eligible must equal selected count")
    if manifest.get("exclusions", 0) != 0:
        raise ValueError("filtered recovery manifest exclusions must be zero")
    mismatch = _validate_filtered_recovery_mismatch(manifest.get("declared_vs_actual_inner_receipt"))
    if expected_mismatch is not None and mismatch != expected_mismatch:
        raise ValueError("filtered recovery declared/actual mismatch does not match evidence receipt")
    if parent_hashes is not None:
        if observed_parent_hashes != parent_hashes:
            raise ValueError("filtered recovery parent artifact hash changed")
    return {"stems": selected, "count": len(selected), "frozen_count": len(frozen),
            "declared_vs_actual_inner_receipt": mismatch}


def import_filtered_recovery_assets(
    assets: dict[str, Path], destination: Path, *, allowlist: set[str],
    destination_names: dict[str, str] | None = None,
) -> list[dict]:
    """Copy allowlisted frozen assets into a fresh recovery namespace.

    Symlink and hardlink aliases are rejected before and after copying; each
    row records source/destination hashes, inode and link-count evidence.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"filtered recovery destination must be fresh: {destination}")
    destination.mkdir(parents=True)
    rows: list[dict] = []
    for label, source in sorted(assets.items()):
        if label not in allowlist:
            raise ValueError(f"filtered recovery asset is not allowlisted: {label}")
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"filtered recovery source is not an ordinary file: {source}")
        dst_name = (destination_names.get(label, source.name)
                    if destination_names else source.name)
        if Path(dst_name).name != dst_name:
            # Category-prefixed relative names are permitted but must remain
            # beneath the fresh import root.
            if Path(dst_name).is_absolute() or ".." in Path(dst_name).parts:
                raise ValueError(f"filtered recovery destination name escapes namespace: {label}")
        dst = destination / dst_name
        if dst.exists() or dst.is_symlink():
            raise ValueError(f"filtered recovery destination collision: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_stat = source.stat()
        if src_stat.st_nlink != 1:
            raise ValueError(f"filtered recovery source has hardlink aliases: {label}")
        src_hash = _sha256_file(source)
        shutil.copy2(source, dst)
        dst_stat = dst.stat()
        if _sha256_file(dst) != src_hash:
            raise ValueError(f"filtered recovery copy hash mismatch: {label}")
        if os.path.samestat(src_stat, dst_stat) or dst.is_symlink() or dst_stat.st_nlink != 1:
            raise ValueError(f"filtered recovery copy inode/link alias: {label}")
        dst_hash = _sha256_file(dst)
        rows.append({"label": label, "source": str(source.resolve()), "destination": str(dst.resolve()),
                     "sha256": src_hash, "source_sha256": src_hash,
                     "destination_sha256": dst_hash, "source_inode": src_stat.st_ino,
                     "destination_inode": dst_stat.st_ino, "source_nlink": src_stat.st_nlink,
                     "destination_nlink": dst_stat.st_nlink, "size": src_stat.st_size})
    return rows


def make_filtered_recovery_receipt(
    partition: dict, imports: list[dict], parent_hashes: dict[str, str],
    parent_hashes_after: dict[str, str] | None = None, *, strict_id: str | None = None,
    declared_vs_actual_inner_receipt: dict[str, str] | None = None,
) -> dict:
    """Build durable, explicitly non-sealing recovery evidence."""
    after = dict(parent_hashes if parent_hashes_after is None else parent_hashes_after)
    if declared_vs_actual_inner_receipt is None:
        raise ValueError("filtered recovery receipt requires explicit inner-receipt evidence")
    return {"schema": FILTERED_RECOVERY_SCHEMA, "scope": "filtered_recovery",
            "publishing": False, "reseal_parent": False,
            "evidence_mode": "quarantined_independent_reconstruction",
            "parent_certification_status": "invalid_receipt_binding",
            "partition": partition, "imports": imports,
            "parent_artifact_sha256": dict(sorted(parent_hashes.items())),
            "parent_artifact_sha256_before": dict(sorted(parent_hashes.items())),
            "parent_artifact_sha256_after": dict(sorted(after.items())),
            "parent_hashes_unchanged": parent_hashes == after,
            "strict_id": strict_id,
            "declared_vs_actual_inner_receipt": dict(declared_vs_actual_inner_receipt)}


def _filtered_recovery_root(path: Path, label: str) -> Path:
    raw = Path(os.path.abspath(path))
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError(f"{label} must be an ordinary directory: {raw}")
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} has symlink ancestor: {cursor}")
    return raw.resolve(strict=True)


def _mfa_retry_regular_copy(source: Path, target: Path) -> dict:
    source = Path(source)
    if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
        raise ValueError(f"MFA retry source is not an ordinary file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ValueError(f"MFA retry destination collision: {target}")
    shutil.copy2(source, target)
    return {"source": str(source.resolve()), "destination": str(target.resolve()),
            "sha256": _sha256_file(source), "destination_sha256": _sha256_file(target),
            "size": source.stat().st_size}


def prepare_mfa_retry_packet(parent_root: Path, retry_root: Path, stems: list[str], *,
                             frozen_stems: list[str], accepted_stems: list[str],
                             execute: bool = False, mfa_python: Path | None = None,
                             model_path: Path | None = None, dictionary_path: Path | None = None) -> dict:
    """Create/run an exact-missing MFA retry in a fresh quarantine packet."""
    parent = _filtered_recovery_root(parent_root, "MFA retry parent")
    retry_root = _strict_replay_path_new(Path(retry_root), "MFA retry workspace")
    stems = _filtered_recovery_sorted_unique(stems, "MFA retry stems")
    frozen = set(_filtered_recovery_sorted_unique(frozen_stems, "frozen"))
    accepted = set(_filtered_recovery_sorted_unique(accepted_stems, "accepted"))
    if not set(stems) <= frozen or set(stems) & accepted:
        raise ValueError("MFA retry stems must be a frozen subset and exclude parent accepted stems")
    ws = parent / "workspace"
    axis_path = ws / ".mfa_alignment_axis_receipt.json"
    post_path = next(iter(ws.glob("strict_ok_runs/*/output/postprocess_report.jsonl")), None)
    manifest_path = next(iter(ws.glob("mfa_logs/*/mfa_output_manifest.json")), None)
    strict_path = next(iter(ws.glob("strict_ok_runs/*/output/strict_ok_manifest.json")), None)
    for path, label in ((axis_path, "MFA axis receipt"), (post_path, "postprocess report"),
                        (manifest_path, "MFA output manifest"), (strict_path, "strict manifest")):
        if path is None or path.is_symlink() or not path.is_file():
            raise ValueError(f"MFA retry {label} missing")
    axis = json.loads(axis_path.read_text(encoding="utf-8"))
    axis_missing = {r.get("stem") for r in axis.get("alignments", [])
                    if isinstance(r, dict) and r.get("status") == "missing_mfa_alignment"}
    post_missing = {r.get("stem") for r in (json.loads(line) for line in post_path.read_text(encoding="utf-8").splitlines() if line.strip())
                    if isinstance(r, dict) and "missing_mfa_alignment" in str(r.get("filter_reasons", []))}
    mfa_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_missing = {stem for row in mfa_payload.get("shards", []) for stem in row.get("missing", [])}
    evidence_missing = axis_missing & post_missing & manifest_missing
    if not (axis_missing == post_missing == manifest_missing) or not set(stems) <= evidence_missing:
        raise ValueError("MFA retry exact-missing evidence disagreement")
    strict_payload = json.loads(strict_path.read_text(encoding="utf-8"))
    strict_ok = {row.get("stem") for row in strict_payload.get("ok", []) if isinstance(row, dict)}
    if strict_ok & set(stems):
        raise ValueError("MFA retry exact set intersects accepted strict output")
    retry_root.mkdir(parents=True)
    corpus = retry_root / "corpus"; audio = retry_root / "audio"; output = retry_root / "aligned"
    copied = []
    for stem in stems:
        for suffix in (".lab", ".txt"):
            source = ws / "ctc_pretg" / f"{stem}{suffix}"
            if source.is_file(): copied.append({"stem": stem, "role": f"anchor{suffix}", **_mfa_retry_regular_copy(source, corpus / source.name)})
        wav = ws / "audio_16k" / f"{stem}.wav"
        copied.append({"stem": stem, "role": "mfa_axis_audio", **_mfa_retry_regular_copy(wav, audio / wav.name)})
    model = Path(model_path or "/home/user/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip")
    dictionary = Path(dictionary_path or PROJECT_ROOT / "dict" / "mfa_ipa.dict")
    mfa_bin = Path(mfa_python or "/home/user/miniconda3/envs/mfa-dev/bin/mfa")
    mfa_bin_dir = mfa_bin.parent
    mfa_python = mfa_bin_dir / "python"
    fstcompile = mfa_bin_dir / "fstcompile"
    if not mfa_bin.is_file() or not os.access(mfa_bin, os.X_OK) or not mfa_python.is_file():
        raise ValueError(f"MFA executable unavailable: {mfa_bin}")
    if not fstcompile.is_file() or not os.access(fstcompile, os.X_OK):
        raise ValueError(f"MFA dependency unavailable: {fstcompile}")
    numba_cache = retry_root / "numba_cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    preflight_env = os.environ.copy()
    preflight_env["PATH"] = str(mfa_bin_dir) + os.pathsep + preflight_env.get("PATH", "")
    preflight_env["NUMBA_CACHE_DIR"] = str(numba_cache)
    preflight = subprocess.run(
        [str(mfa_python), "-c", "import librosa; assert librosa.note_to_midi('C4') == 60"],
        cwd=str(PROJECT_ROOT), text=True, capture_output=True, env=preflight_env)
    if preflight.returncode != 0:
        raise ValueError(f"MFA librosa preflight failed: {preflight.stderr[-500:]}")
    command = [str(mfa_bin), "align", str(corpus), str(dictionary), str(model), str(output),
               "--audio_directory", str(audio), "--num_jobs", "12", "--single_speaker",
               "--no_tokenization", "--beam", "20", "--retry_beam", "80",
               "--boost_silence", "1.0", "--no_fine_tune", "--no_clean",
               "--output_format", "long_textgrid"]
    receipt = {"schema": MFA_RETRY_SCHEMA, "scope": "exact_missing_mfa_retry",
               "parent_root": str(parent), "stems": stems,
               "evidence_missing_stems": sorted(evidence_missing),
               "frozen_count": len(frozen), "accepted_intersection": 0,
               "evidence": {"axis": str(axis_path), "postprocess": str(post_path),
                            "mfa_manifest": str(manifest_path), "strict_manifest": str(strict_path),
                            "hashes": {str(p): _sha256_file(p) for p in (axis_path, post_path, manifest_path, strict_path)}},
               "inputs": copied, "command": command, "cwd": str(PROJECT_ROOT),
               "model": {"path": str(model), "sha256": _sha256_file(model)},
               "dictionary": {"path": str(dictionary), "sha256": _sha256_file(dictionary)},
               "mfa_executable": {"path": str(mfa_bin), "sha256": _sha256_file(mfa_bin)},
               "mfa_dependency": {"name": "fstcompile", "path": str(fstcompile), "sha256": _sha256_file(fstcompile)},
               "preflight": {"librosa_note_to_midi": "C4=60", "python": str(mfa_python),
                             "return_code": preflight.returncode, "stdout": preflight.stdout[-200:],
                             "stderr": preflight.stderr[-200:]},
               "options": {"num_jobs": 12, "single_speaker": True, "no_tokenization": True,
                           "beam": 20, "retry_beam": 80, "boost_silence": 1.0,
                           "fine_tune": False, "clean": False, "output_format": "long_textgrid"},
               "environment": {"PATH_prefix": str(mfa_bin_dir),
                               "MFA_ROOT_DIR": str(retry_root / "mfa_root"),
                               "NUMBA_CACHE_DIR": str(numba_cache),
                               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                               "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
               "execution": {"attempted": False}}
    if execute:
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        retry_env = os.environ.copy()
        retry_env["PATH"] = str(mfa_bin_dir) + os.pathsep + retry_env.get("PATH", "")
        mfa_root_dir = retry_root / "mfa_root"
        mfa_root_dir.mkdir(parents=True, exist_ok=True)
        retry_env["MFA_ROOT_DIR"] = str(mfa_root_dir)
        retry_env["NUMBA_CACHE_DIR"] = str(numba_cache)
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True, env=retry_env)
        (retry_root / "mfa.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (retry_root / "mfa.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        produced = sorted(p.stem for p in output.glob("*.TextGrid") if p.is_file() and not p.is_symlink())
        produced_set = set(produced)
        per_stem = []
        for stem in stems:
            artifact = output / f"{stem}.TextGrid"
            per_stem.append({"stem": stem, "produced": stem in produced_set,
                             "sha256": _sha256_file(artifact) if artifact.is_file() else None,
                             "oov": [], "invalid": bool(artifact.is_symlink()),
                             "return_code": proc.returncode, "started": started, "finished": time.time()})
        receipt["execution"] = {"attempted": True, "return_code": proc.returncode,
                                 "started": started, "finished": time.time(),
                                 "produced": produced, "missing": sorted(set(stems) - set(produced)),
                                 "extra": sorted(set(produced) - set(stems)),
                                 "invalid": [row["stem"] for row in per_stem if row["invalid"]],
                                 "per_stem": per_stem,
                                 "stdout": str(retry_root / "mfa.stdout.log"),
                                 "stderr": str(retry_root / "mfa.stderr.log"),
                                 "path_prefix": str(mfa_bin_dir),
                                 "mfa_root_dir": str(mfa_root_dir)}
    receipt_path = retry_root / "mfa_retry_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def run_mfa_singleton_rescue(parent_root: Path, rescue_root: Path, stem: str,
                             *, frozen_stems: list[str], accepted_stems: list[str],
                             prior_receipt_path: Path) -> dict:
    """Run the one authorized NoAlignmentsError singleton rescue (200/800)."""
    prior_receipt_path = Path(prior_receipt_path)
    if not prior_receipt_path.is_file() or prior_receipt_path.is_symlink():
        raise ValueError("MFA rescue prior receipt missing")
    prior = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    prior_exec = prior.get("execution", {})
    prior_stems = prior.get("stems", [])
    if prior_stems != [stem] or prior_exec.get("return_code") != 1 or prior_exec.get("produced"):
        raise ValueError("MFA rescue prior receipt is not the unchanged isolated failure")
    prior_log = Path(str(prior_exec.get("stderr", "")))
    if not prior_log.is_file() or "NoAlignmentsError" not in prior_log.read_text(encoding="utf-8"):
        raise ValueError("MFA rescue prior failure is not NoAlignmentsError")
    rescue_root = Path(rescue_root)
    packet = prepare_mfa_retry_packet(parent_root, rescue_root, [stem],
                                      frozen_stems=frozen_stems,
                                      accepted_stems=accepted_stems, execute=False)
    command = list(packet["command"])
    command[command.index("20")] = "200"
    command[command.index("80")] = "800"
    # Guard the singleton policy against accidental generic/batch widening.
    if packet["stems"] != [stem] or command.count("200") != 1 or command.count("800") != 1:
        raise ValueError("MFA rescue policy widening guard failed")
    output = rescue_root / "aligned"
    started = time.time()
    mfa_bin_dir = Path(command[0]).parent
    env = os.environ.copy()
    env["PATH"] = str(mfa_bin_dir) + os.pathsep + env.get("PATH", "")
    env["MFA_ROOT_DIR"] = str(rescue_root / "mfa_root")
    env["NUMBA_CACHE_DIR"] = str(rescue_root / "numba_cache")
    (rescue_root / "mfa_root").mkdir(parents=True, exist_ok=True)
    (rescue_root / "numba_cache").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True, env=env)
    (rescue_root / "mfa.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (rescue_root / "mfa.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    produced = sorted(p.stem for p in output.glob("*.TextGrid") if p.is_file() and not p.is_symlink())
    receipt = {"schema": MFA_RESCUE_SCHEMA, "stem": stem,
               "policy": {"attempts": 1, "beam": 200, "retry_beam": 800,
                           "reason": "NoAlignmentsError at unchanged beam20/retry80"},
               "prior_receipt": {"path": str(prior_receipt_path.resolve()),
                                 "sha256": _sha256_file(prior_receipt_path),
                                 "stderr_sha256": _sha256_file(prior_log)},
               "command": command, "cwd": str(PROJECT_ROOT),
               "environment": {"PATH_prefix": str(mfa_bin_dir),
                               "MFA_ROOT_DIR": str(rescue_root / "mfa_root"),
                               "NUMBA_CACHE_DIR": str(rescue_root / "numba_cache")},
               "model": packet["model"], "dictionary": packet["dictionary"],
               "mfa_executable": packet["mfa_executable"],
               "execution": {"attempted": True, "attempts": 1,
                             "return_code": proc.returncode, "started": started,
                             "finished": time.time(), "produced": produced,
                             "missing": sorted({stem} - set(produced)), "extra": sorted(set(produced) - {stem}),
                             "stdout": str(rescue_root / "mfa.stdout.log"),
                             "stderr": str(rescue_root / "mfa.stderr.log"),
                             "textgrid_sha256": _sha256_file(output / f"{stem}.TextGrid") if (output / f"{stem}.TextGrid").is_file() else None}}
    path = rescue_root / "mfa_rescue_receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def _filtered_recovery_stem_digest(stems) -> str:
    """Digest the canonical sorted frozen partition, independent of count."""
    payload = json.dumps(_filtered_recovery_sorted_unique(stems, "frozen"),
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_filtered_recovery_mismatch(mismatch) -> dict[str, str]:
    if not isinstance(mismatch, dict) or set(mismatch) != {"declared_sha256", "actual_sha256"}:
        raise ValueError("filtered recovery declared/actual mismatch must contain two SHA-256 digests")
    for key, value in mismatch.items():
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"filtered recovery mismatch {key} is not a lowercase SHA-256 digest (known value/evidence digest required)")
    if mismatch["declared_sha256"] == mismatch["actual_sha256"]:
        raise ValueError("filtered recovery declared/actual mismatch must be non-equal")
    return {"declared_sha256": mismatch["declared_sha256"],
            "actual_sha256": mismatch["actual_sha256"]}


def _read_filtered_recovery_evidence(path: Path, frozen, plan: dict) -> dict:
    """Load explicit evidence binding for a recovery plan.

    The evidence receipt is intentionally external to the plan so a stale
    historical digest cannot silently become an approved production value.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"filtered recovery evidence receipt missing: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("schema") != "filtered-recovery-evidence-v1":
        raise ValueError("filtered recovery evidence receipt schema mismatch")
    digest = evidence.get("frozen_stems_sha256")
    if digest != _filtered_recovery_stem_digest(frozen):
        raise ValueError("filtered recovery frozen partition digest does not match evidence")
    evidence_hashes = evidence.get("parent_artifact_sha256")
    plan_hashes = plan.get("parent_artifact_sha256")
    if not isinstance(evidence_hashes, dict) or evidence_hashes != plan_hashes:
        raise ValueError("filtered recovery parent artifact hashes do not match evidence")
    mismatch = _validate_filtered_recovery_mismatch(evidence.get("declared_vs_actual_inner_receipt"))
    if plan.get("declared_vs_actual_inner_receipt") != mismatch:
        raise ValueError("filtered recovery inner-receipt mismatch does not match evidence")
    return evidence


def _validate_filtered_recovery_english_ledger_scope(source_ledgers, frozen, accepted) -> set[str]:
    """Validate the sealed English ledger universe before frozen-only import.

    A producer manifest may be either an earlier frozen-only manifest or the
    parent-global manifest.  The latter is safe only because its exact bytes
    are already bound by the filtered-recovery evidence receipt.  No partial
    or expanded ledger universe is accepted.
    """
    if not isinstance(source_ledgers, list):
        raise ValueError("filtered recovery English source manifest ledgers missing")
    ledger_stems: list[str] = []
    for row in source_ledgers:
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str) or not row["stem"]:
            raise ValueError("filtered recovery English source ledger stem invalid")
        ledger_stems.append(row["stem"])
    source_stems = set(ledger_stems)
    if len(source_stems) != len(ledger_stems):
        raise ValueError("filtered recovery English source manifest contains duplicate ledger stems")
    frozen_stems = set(frozen)
    accepted_stems = set(accepted)
    if source_stems not in (frozen_stems, frozen_stems | accepted_stems):
        raise ValueError("filtered recovery English source ledger universe is not sealed frozen/full partition")
    return source_stems


def run_filtered_recovery(args, cfg: dict, config_path: Path | None = None) -> int:
    """Validate a frozen recovery plan in a fresh quarantine namespace.

    This entry point intentionally performs no MFA/CTC execution.  It imports
    only explicitly supplied allowlisted evidence and writes a non-publishing
    receipt.  ``--validate-only`` revalidates an existing import receipt
    without touching the parent or copying files.
    """
    try:
        parent = _filtered_recovery_root(Path(args.filtered_recovery_parent_root), "filtered recovery parent")
        workspace = Path(args.workspace) if args.workspace else parent / "workspace" / "filtered_recovery"
        output = Path(args.output_dir) if args.output_dir else workspace / "output"
        if getattr(args, "filtered_recovery_validate_only", False):
            receipt_arg = getattr(args, "filtered_recovery_import_receipt", None)
            if not receipt_arg:
                raise ValueError("filtered_recovery --validate-only requires --filtered-recovery-import-receipt")
            receipt_path = Path(receipt_arg)
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ValueError(f"filtered recovery import receipt missing: {receipt_path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("schema") != FILTERED_RECOVERY_SCHEMA or receipt.get("publishing") is not False or receipt.get("reseal_parent") is not False:
                raise ValueError("filtered recovery import receipt is not quarantined")
            partition = receipt.get("partition")
            if not isinstance(partition, dict):
                raise ValueError("filtered recovery import receipt partition missing")
            frozen_path = Path(args.filtered_recovery_frozen_manifest) if args.filtered_recovery_frozen_manifest else parent / "frozen_filtered.json"
            accepted_path = Path(args.filtered_recovery_accepted_manifest) if args.filtered_recovery_accepted_manifest else parent / "strict_ok_manifest.json"
            for candidate, label in ((frozen_path, "frozen manifest"), (accepted_path, "accepted manifest")):
                if not candidate.is_file() or candidate.is_symlink():
                    raise ValueError(f"filtered recovery {label} missing: {candidate}")
            frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
            accepted_payload = json.loads(accepted_path.read_text(encoding="utf-8"))
            frozen = frozen_payload.get("stems", frozen_payload.get("filtered", {}).get("stems", []))
            accepted = accepted_payload.get("ok", accepted_payload.get("output", {}).get("stems", []))
            if isinstance(accepted, list) and accepted and isinstance(accepted[0], dict):
                accepted = [row.get("stem") for row in accepted]
            if isinstance(frozen, list) and frozen and isinstance(frozen[0], dict):
                frozen = [row.get("stem") for row in frozen]
            mismatch = receipt.get("declared_vs_actual_inner_receipt")
            if not getattr(args, "filtered_recovery_evidence_receipt", None):
                raise ValueError("filtered recovery --validate-only requires --filtered-recovery-evidence-receipt")
            validate_filtered_recovery_partition(frozen, accepted,
                                                 partition.get("recovered_stems", []),
                                                 partition.get("still_filtered_stems", []))
            if receipt.get("parent_hashes_unchanged") is not True:
                raise ValueError("filtered recovery parent hashes are not unchanged")
            _read_filtered_recovery_evidence(Path(args.filtered_recovery_evidence_receipt), frozen, receipt)
            for name, expected in receipt.get("parent_artifact_sha256", {}).items():
                path = (parent / name).resolve(strict=True)
                try:
                    path.relative_to(parent)
                except ValueError as exc:
                    raise ValueError(f"filtered recovery parent artifact escapes parent: {name}") from exc
                if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
                    raise ValueError(f"filtered recovery parent artifact changed: {name}")
            print(f"filtered_recovery validate-only OK: {receipt_path}")
            return 0
        if workspace.exists() or output.exists():
            raise ValueError("filtered recovery workspace/output must be fresh non-existing paths")
        frozen_path = Path(args.filtered_recovery_frozen_manifest) if args.filtered_recovery_frozen_manifest else parent / "frozen_filtered.json"
        accepted_path = Path(args.filtered_recovery_accepted_manifest) if args.filtered_recovery_accepted_manifest else parent / "strict_ok_manifest.json"
        plan_path = Path(args.filtered_recovery_manifest) if args.filtered_recovery_manifest else parent / "filtered_recovery_manifest.json"
        evidence_path = getattr(args, "filtered_recovery_evidence_receipt", None)
        if not evidence_path:
            raise ValueError("filtered_recovery requires explicit --filtered-recovery-evidence-receipt")
        for candidate, label in ((frozen_path, "frozen manifest"), (accepted_path, "accepted manifest"), (plan_path, "recovery manifest")):
            if not candidate.is_file() or candidate.is_symlink():
                raise ValueError(f"filtered recovery {label} missing: {candidate}")
        frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
        accepted_payload = json.loads(accepted_path.read_text(encoding="utf-8"))
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        frozen = frozen_payload.get("stems", frozen_payload.get("filtered", {}).get("stems", []))
        accepted = accepted_payload.get("ok", accepted_payload.get("output", {}).get("stems", []))
        if isinstance(accepted, list) and accepted and isinstance(accepted[0], dict):
            accepted = [row.get("stem") for row in accepted]
        if isinstance(frozen, list) and frozen and isinstance(frozen[0], dict):
            frozen = [row.get("stem") for row in frozen]
        evidence = _read_filtered_recovery_evidence(Path(evidence_path), frozen, plan_payload)
        validated = validate_filtered_recovery_manifest(
            plan_payload, frozen, accepted,
            expected_mismatch=evidence["declared_vs_actual_inner_receipt"])
        # Hash every parent evidence file named by the plan before any import.
        parent_hashes = plan_payload.get("parent_artifact_sha256", {})
        if not isinstance(parent_hashes, dict):
            raise ValueError("filtered recovery parent_artifact_sha256 must be an object")
        observed = {}
        for name in sorted(parent_hashes):
            raw_path = parent / name
            try:
                path = raw_path.resolve(strict=True)
                path.relative_to(parent)
            except (OSError, ValueError) as exc:
                raise ValueError(f"filtered recovery parent artifact escapes parent: {name}") from exc
            if not path.is_file() or raw_path.is_symlink():
                raise ValueError(f"filtered recovery parent artifact missing: {name}")
            observed[name] = _sha256_file(path)
        if observed != parent_hashes:
            raise ValueError("filtered recovery parent artifact hash changed")
        # Import only the frozen stems' replay inputs.  Accepted-parent stems
        # are never enumerated or copied.  These four axes are the minimum
        # authoritative inputs consumed by a filtered replay.
        strict_output = Path(accepted_payload.get("output_dir", ""))
        if not strict_output.is_absolute():
            strict_output = parent / strict_output
        try:
            strict_output = strict_output.resolve(strict=True)
            strict_output.relative_to(parent)
        except (OSError, ValueError) as exc:
            raise ValueError("filtered recovery strict output escapes parent root") from exc
        filtered_root = strict_output.parent / "filtered"
        aligned_root = Path(args.filtered_recovery_aligned_root) if getattr(args, "filtered_recovery_aligned_root", None) else parent / "workspace" / "aligned"
        aligned_root = _filtered_recovery_root(aligned_root, "filtered recovery aligned root")
        ctc_root = parent / "workspace" / "ctc_pretg"
        en_root = parent / "workspace" / "en_phones"
        ctc_override_root = None
        en_override_root = None
        en_aligned_override_root = None
        if getattr(args, "filtered_recovery_ctc_root", None):
            ctc_override_root = _filtered_recovery_root(
                Path(args.filtered_recovery_ctc_root),
                "filtered recovery CTC override root")
        if getattr(args, "filtered_recovery_english_root", None):
            en_override_root = _filtered_recovery_root(
                Path(args.filtered_recovery_english_root),
                "filtered recovery English override root")
        if getattr(args, "filtered_recovery_english_aligned_root", None):
            en_aligned_override_root = _filtered_recovery_root(
                Path(args.filtered_recovery_english_aligned_root),
                "filtered recovery English aligned override root")
        input_root = parent / "input"
        mfa_audio_root = parent / "workspace" / "audio_16k"
        tts_audio_root = parent / "workspace" / "padded_audio"
        rejected_rows = accepted_payload.get("rejected", {})
        missing_mfa = {stem for stem, reasons in rejected_rows.items()
                       if "missing_mfa_alignment" in str(reasons)} if isinstance(rejected_rows, dict) else set()
        if not missing_mfa:
            axis_path = parent / "workspace" / ".mfa_alignment_axis_receipt.json"
            try:
                axis_rows = json.loads(axis_path.read_text(encoding="utf-8")).get("alignments", [])
                missing_mfa = {row.get("stem") for row in axis_rows
                               if isinstance(row, dict) and row.get("status") == "missing_mfa_alignment"}
            except (OSError, TypeError, json.JSONDecodeError):
                pass
        assets: dict[str, Path] = {}
        destination_names: dict[str, str] = {}
        for stem in frozen:
            required = {
                f"filtered_textgrid:{stem}": filtered_root / f"{stem}.TextGrid",
                f"audio:{stem}": input_root / f"{stem}.wav",
                f"reference:{stem}": input_root / f"{stem}.txt",
            }
            mfa_audio = mfa_audio_root / f"{stem}.wav"
            tts_audio = tts_audio_root / f"{stem}.wav"
            if not mfa_audio.is_file() or mfa_audio.is_symlink():
                raise ValueError(f"filtered recovery MFA-axis audio missing: {mfa_audio}")
            if not tts_audio.is_file() or tts_audio.is_symlink():
                raise ValueError(f"filtered recovery TTS-authoritative audio missing: {tts_audio}")
            required[f"mfa_audio:{stem}"] = mfa_audio
            required[f"tts_audio:{stem}"] = tts_audio
            transform_candidates = sorted((parent / "workspace" / "audio_transform_receipts").glob(f"{stem}.*.json"))
            if transform_candidates:
                required[f"audio_transform:{stem}"] = transform_candidates[0]
            for suffix in (".txt", "_ref.txt", ".lab", ".TextGrid", "_tokens.jsonl", "_punct.json", "_text_cn.txt", "_text_raw.txt"):
                ctc = ctc_root / f"{stem}{suffix}"
                if ctc_override_root is not None:
                    override = ctc_override_root / f"{stem}{suffix}"
                    if override.is_file() and not override.is_symlink():
                        ctc = override
                if ctc.is_file() and not ctc.is_symlink():
                    required[f"ctc_authority:{stem}:{suffix}"] = ctc
            aligned = aligned_root / f"{stem}.TextGrid"
            if aligned.is_file() and not aligned.is_symlink():
                required[f"mfa_aligned_textgrid:{stem}"] = aligned
            elif stem not in missing_mfa:
                raise ValueError(f"filtered recovery MFA aligned asset missing: {aligned}")
            en_phone = en_root / f"{stem}_en_phones.json"
            if en_override_root is not None:
                override = en_override_root / f"{stem}_en_phones.json"
                if override.is_file() and not override.is_symlink():
                    en_phone = override
            if en_phone.is_file() and not en_phone.is_symlink():
                required[f"english_phones:{stem}"] = en_phone
            if en_aligned_override_root is not None:
                for segment_grid in sorted(en_aligned_override_root.glob(f"{stem}_seg*.TextGrid")):
                    if segment_grid.is_file() and not segment_grid.is_symlink():
                        required[f"english_aligned_segment:{stem}:{segment_grid.name}"] = segment_grid
            for label, source in required.items():
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"filtered recovery required asset missing: {source}")
                assets[label] = source
                category = label.split(":", 1)[0]
                destination_names[label] = f"{category}/{source.name}"
        en_manifest = en_root / "en_alignment_manifest.json"
        if en_override_root is not None:
            override_manifest = en_override_root / "en_alignment_manifest.json"
            if override_manifest.is_file() and not override_manifest.is_symlink():
                en_manifest = override_manifest
        if en_manifest.is_file() and not en_manifest.is_symlink():
            assets["english_manifest_source"] = en_manifest
            destination_names["english_manifest_source"] = "english_phones/en_alignment_manifest.source.json"
        import_root = workspace / "imports"
        imports = import_filtered_recovery_assets(
            assets, import_root, allowlist=set(assets),
            destination_names=destination_names)
        # Localize the fresh English producer manifest and its ledger/grid
        # references into the quarantine namespace.  The source manifest is
        # retained verbatim as ``*.source.json``; the canonical manifest is a
        # frozen-stem subset whose every path/hash resolves to an imported
        # regular file (never a historical absolute path).
        localized_en_manifest = None
        localized_en_ledgers = {}
        source_en_manifest_path = import_root / "english_phones" / "en_alignment_manifest.source.json"
        if source_en_manifest_path.is_file():
            source_en_manifest = json.loads(source_en_manifest_path.read_text(encoding="utf-8"))
            if (source_en_manifest.get("schema") != "strict-en-mfa-v1"
                    or source_en_manifest.get("strict_provenance") is not True
                    or source_en_manifest.get("status") not in {"success", "no_english"}):
                raise ValueError("filtered recovery English source manifest is not strict-en-mfa-v1")
            source_ledgers = source_en_manifest.get("stem_ledgers")
            _validate_filtered_recovery_english_ledger_scope(source_ledgers, frozen, accepted)
            expected_ids = [item for item in source_en_manifest.get("expected_segments", [])
                            if isinstance(item, str) and item.rsplit(":s", 1)[0] in frozen]
            produced_ids = [item for item in source_en_manifest.get("produced_segments", [])
                            if isinstance(item, str) and item.rsplit(":s", 1)[0] in frozen]
            rejected_ids = [item for item in source_en_manifest.get("rejected_segments", [])
                            if isinstance(item, dict) and isinstance(item.get("id"), str)
                            and item["id"].rsplit(":s", 1)[0] in frozen]
            localized_rows = []
            localized_grid_refs = []
            english_words = verified_words = rejected_words = 0
            for source_row in source_ledgers:
                if not isinstance(source_row, dict) or source_row.get("stem") not in frozen:
                    continue
                stem = source_row["stem"]
                imported_ledger = import_root / "english_phones" / Path(source_row["path"]).name
                if not imported_ledger.is_file() or imported_ledger.is_symlink():
                    raise ValueError(f"filtered recovery English ledger import missing: {stem}")
                ledger_payload = json.loads(imported_ledger.read_text(encoding="utf-8"))
                if ledger_payload.get("schema") != "strict-en-mfa-v1" or ledger_payload.get("stem") != stem:
                    raise ValueError(f"filtered recovery English ledger invalid: {stem}")
                for segment in ledger_payload.get("segments", []):
                    if not isinstance(segment, dict):
                        raise ValueError(f"filtered recovery English segment invalid: {stem}")
                    words = segment.get("words", [])
                    if isinstance(words, list):
                        english_words += len(words)
                        verified_words += sum(1 for word in words if isinstance(word, dict) and word.get("status") == "verified")
                        rejected_words += sum(1 for word in words if isinstance(word, dict) and word.get("status") != "verified")
                    source_grid = segment.get("mfa_textgrid")
                    if segment.get("status") == "verified":
                        if not isinstance(source_grid, dict) or not isinstance(source_grid.get("path"), str):
                            raise ValueError(f"filtered recovery English source grid missing: {stem}")
                        grid_name = Path(source_grid["path"]).name
                        imported_grid = import_root / "english_aligned_segment" / grid_name
                        if not imported_grid.is_file() or imported_grid.is_symlink():
                            raise ValueError(f"filtered recovery English aligned grid import missing: {grid_name}")
                        segment["mfa_textgrid"] = {"path": str(imported_grid.resolve()),
                                                     "sha256": _sha256_file(imported_grid)}
                        localized_grid_refs.append({"path": str(imported_grid.resolve()),
                                                    "sha256": _sha256_file(imported_grid),
                                                    "segment_id": segment.get("segment_id")})
                # Rewrite the imported ledger itself so downstream provenance
                # validation never follows the producer's temporary paths.
                _strict_replay_replace_json(imported_ledger, ledger_payload)
                ledger_hash = _sha256_file(imported_ledger)
                localized_en_ledgers[stem] = {"path": str(imported_ledger.resolve()), "sha256": ledger_hash}
                localized_rows.append({"stem": stem, "path": str(imported_ledger.resolve()), "sha256": ledger_hash})
            localized_en_manifest = json.loads(json.dumps(source_en_manifest))
            localized_en_manifest.update({
                "expected_segments": sorted(set(expected_ids)),
                "produced_segments": sorted(set(produced_ids)),
                "rejected_segments": sorted(rejected_ids, key=lambda item: item.get("id", "")),
                "stem_ledgers": sorted(localized_rows, key=lambda item: item["stem"]),
                "counts": {"english_stems": len(localized_rows),
                            "english_segments": len(set(expected_ids)),
                            "english_words": english_words,
                            "verified_words": verified_words,
                            "rejected_words": rejected_words},
                "localized_scope": "filtered_recovery_frozen_only",
                "source_manifest": {"path": str(source_en_manifest_path.resolve()),
                                    "sha256": _sha256_file(source_en_manifest_path)},
                "paths_rewritten": True,
                "segment_grids": sorted(localized_grid_refs, key=lambda item: item.get("segment_id", "")),
            })
            localized_en_manifest_path = import_root / "english_phones" / "en_alignment_manifest.json"
            _strict_replay_write_once_json(localized_en_manifest_path, localized_en_manifest)
            imports.append({"label": "english_localized_manifest", "source": str(source_en_manifest_path.resolve()),
                            "destination": str(localized_en_manifest_path.resolve()),
                            "sha256": _sha256_file(source_en_manifest_path),
                            "destination_sha256": _sha256_file(localized_en_manifest_path),
                            "localized": True, "scope": "frozen_only"})
        # Bind a localized CTC v2 receipt to the frozen import namespace.  The
        # parent receipt is read-only evidence; this receipt rewrites every
        # per-stem path/hash to the copied CTC/audio assets and records any
        # fresh frozen-only producer override without admitting accepted stems.
        parent_ctc_receipt_path = ctc_root / ".ctc_run_receipt.json"
        if not parent_ctc_receipt_path.is_file() or parent_ctc_receipt_path.is_symlink():
            raise ValueError("filtered recovery parent CTC v2 receipt missing")
        parent_ctc_receipt = json.loads(parent_ctc_receipt_path.read_text(encoding="utf-8"))
        if parent_ctc_receipt.get("schema") != CTC_RUN_RECEIPT_SCHEMA:
            raise ValueError("filtered recovery parent CTC receipt is not v2")
        override_ctc_receipt = None
        override_ctc_receipt_path = (ctc_override_root / ".ctc_run_receipt.json"
                                     if ctc_override_root is not None else None)
        if override_ctc_receipt_path is not None and override_ctc_receipt_path.is_file():
            override_ctc_receipt = json.loads(override_ctc_receipt_path.read_text(encoding="utf-8"))
            if override_ctc_receipt.get("schema") != CTC_RUN_RECEIPT_SCHEMA:
                raise ValueError("filtered recovery CTC override receipt is not v2")
        parent_ctc_rows = {row.get("stem"): row for row in parent_ctc_receipt.get("audio_bindings", [])
                           if isinstance(row, dict)}
        override_ctc_rows = {row.get("stem"): row for row in (override_ctc_receipt or {}).get("audio_bindings", [])
                             if isinstance(row, dict)}
        localized_bindings = []
        for stem in sorted(frozen):
            source_row = dict(override_ctc_rows.get(stem) or parent_ctc_rows.get(stem) or {})
            audio_path = import_root / "mfa_audio" / f"{stem}.wav"
            lab_path = import_root / "ctc_authority" / f"{stem}.lab"
            textgrid_path = import_root / "ctc_authority" / f"{stem}.TextGrid"
            token_path = import_root / "ctc_authority" / f"{stem}_tokens.jsonl"
            punct_path = import_root / "ctc_authority" / f"{stem}_punct.json"
            reference_path = import_root / "ctc_authority" / f"{stem}_ref.txt"
            for required_path, label in ((audio_path, "MFA audio"), (lab_path, "CTC lab"),
                                          (textgrid_path, "CTC TextGrid"), (token_path, "CTC tokens"),
                                          (punct_path, "CTC punctuation"), (reference_path, "CTC reference")):
                if not required_path.is_file() or required_path.is_symlink():
                    raise ValueError(f"filtered recovery localized CTC {label} missing: {required_path}")
            audio_meta = _axis_audio_metadata(audio_path)
            tg_xmin, tg_xmax = _textgrid_global_bounds(textgrid_path)
            token_rows = []
            for line in token_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    token_rows.append(json.loads(line))
            token_bounds = [float(row[key]) for row in token_rows
                            for key in ("start_s", "end_s")
                            if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
            source_row.update({
                "stem": stem, "path": str(audio_path.resolve()),
                "sha256": audio_meta["sha256"], "duration_s": audio_meta["duration_s"],
                "sample_rate": audio_meta["sample_rate"], "frames": audio_meta["frames"],
                "channels": audio_meta["channels"], "sample_width": audio_meta["sample_width"],
                "ctc_bounds": {"xmin": min(token_bounds) if token_bounds else tg_xmin,
                               "xmax": max(token_bounds) if token_bounds else tg_xmax},
                "token_min_s": min(token_bounds) if token_bounds else None,
                "token_max_s": max(token_bounds) if token_bounds else None,
                "tokens_path": str(token_path.resolve()), "tokens_sha256": _sha256_file(token_path),
                "lab_sha256": _sha256_file(lab_path), "lab_sha256_path": str(lab_path.resolve()),
                "punct_sha256": _sha256_file(punct_path), "punct_sha256_path": str(punct_path.resolve()),
                "reference_sha256": _sha256_file(reference_path),
                "reference_sha256_path": str(reference_path.resolve()),
                "textgrid_path": str(textgrid_path.resolve()), "textgrid_sha256": _sha256_file(textgrid_path),
                "textgrid_xmin": tg_xmin, "textgrid_xmax": tg_xmax,
            })
            localized_bindings.append(source_row)
        localized_ctc_receipt = dict(parent_ctc_receipt)
        localized_ctc_receipt.update({
            "input_stems": sorted(frozen), "output_stems": sorted(frozen),
            "input_stems_digest": stable_json_digest(sorted(frozen)),
            "output_stems_digest": stable_json_digest(sorted(frozen)),
            "audio_bindings": sorted(localized_bindings, key=lambda row: row.get("stem", "")),
            "localized_scope": "filtered_recovery_frozen_only",
            "parent_receipt": {"path": str(parent_ctc_receipt_path.resolve()),
                               "sha256": _sha256_file(parent_ctc_receipt_path)},
            "override_receipt": ({"path": str(override_ctc_receipt_path.resolve()),
                                  "sha256": _sha256_file(override_ctc_receipt_path),
                                  "stems": sorted(override_ctc_rows)}
                                 if override_ctc_receipt_path is not None and override_ctc_receipt_path.is_file()
                                 else None),
        })
        ctc_receipt_errors = validate_ctc_run_receipt_v2(
            localized_ctc_receipt, expected_stems=sorted(frozen),
            audio_root=import_root / "mfa_audio")
        if ctc_receipt_errors:
            raise ValueError("localized CTC v2 receipt invalid: " + "; ".join(ctc_receipt_errors[:8]))
        localized_ctc_path = import_root / "ctc_authority" / ".ctc_run_receipt.json"
        _strict_replay_write_once_json(localized_ctc_path, localized_ctc_receipt)
        imports.append({"label": "ctc_localized_receipt", "source": str(parent_ctc_receipt_path.resolve()),
                        "destination": str(localized_ctc_path.resolve()),
                        "sha256": _sha256_file(parent_ctc_receipt_path),
                        "destination_sha256": _sha256_file(localized_ctc_path),
                        "localized": True, "scope": "frozen_only"})
        # Materialize a stem-restricted axis contract in quarantine so the
        # committed entry point can run the filtered-only postprocess without
        # consulting accepted-parent assets.
        import_rows = {row["label"]: row for row in imports}
        parent_input_axis = json.loads((ctc_root / ".mfa_input_axis_receipt.json").read_text(encoding="utf-8"))
        parent_alignment_axis = json.loads((parent / "workspace" / ".mfa_alignment_axis_receipt.json").read_text(encoding="utf-8"))
        input_rows = []
        replay_transform_dir = import_root / "axis" / "audio_transform_receipts"
        replay_transform_dir.mkdir(parents=True, exist_ok=True)
        for stem in sorted(frozen):
            source_row = next((row for row in parent_input_axis.get("audio", []) if row.get("stem") == stem), None)
            if not isinstance(source_row, dict):
                raise ValueError(f"filtered recovery input axis row missing: {stem}")
            row = dict(source_row)
            row["path"] = str((import_root / "mfa_audio" / f"{stem}.wav").resolve())
            transform = import_rows.get(f"audio_transform:{stem}")
            if transform:
                transform_payload = json.loads(Path(transform["destination"]).read_text(encoding="utf-8"))
                transform_payload["input"]["path"] = str((import_root / "tts_audio" / f"{stem}.wav").resolve())
                transform_payload["output"]["path"] = str((import_root / "mfa_audio" / f"{stem}.wav").resolve())
                replay_transform = replay_transform_dir / Path(transform["destination"]).name
                _strict_replay_write_once_json(replay_transform, transform_payload)
                row["transform_receipt"] = str(replay_transform.resolve())
            input_rows.append(row)
        input_axis = make_mfa_input_axis_receipt(sorted(frozen), input_rows,
                                                  axis_root=import_root / "mfa_audio")
        input_axis["tts_authoritative_audio_root"] = str((import_root / "tts_audio").resolve())
        input_axis["transform_receipts"] = [row["transform_receipt"] for row in input_rows if row.get("transform_receipt")]
        aligned_rows, missing_rows = [], []
        for stem in sorted(frozen):
            source_row = next((row for row in parent_alignment_axis.get("alignments", []) if row.get("stem") == stem), None)
            aligned_path = import_root / "mfa_aligned_textgrid" / f"{stem}.TextGrid"
            if not aligned_path.is_file():
                missing_rows.append(stem)
                continue
            row = dict(source_row or {})
            audio_row = next(item for item in input_rows if item.get("stem") == stem)
            from postprocess_textgrids import parse_textgrid as _parse_recovery_textgrid
            recovery_grid = _parse_recovery_textgrid(aligned_path)
            row.update({"stem": stem, "status": "aligned", "path": str(aligned_path.resolve()),
                        "sha256": _sha256_file(aligned_path),
                        "audio_sha256": audio_row.get("sha256"),
                        "audio_duration_s": audio_row.get("duration_s"),
                        "xmax": recovery_grid.xmax})
            aligned_rows.append(row)
        alignment_axis = make_mfa_alignment_axis_receipt_v2(
            input_axis, aligned_rows, missing_rows,
            alignment_root=import_root / "mfa_aligned_textgrid")
        axis_dir = import_root / "axis"
        _strict_replay_write_once_json(axis_dir / ".mfa_input_axis_receipt.json", input_axis)
        _strict_replay_write_once_json(axis_dir / ".mfa_alignment_axis_receipt.json", alignment_axis)
        # Replay only the frozen namespace through postprocess.  Failures stay
        # in filtered/ and are never promoted or resealed into the parent.
        replay_filtered = workspace / "filtered"
        postprocess_report = output / "postprocess_report.jsonl"
        postprocess_cmd = [sys.executable, str(SCRIPTS_DIR / "postprocess_textgrids.py"),
            "--txt-dir", str(import_root / "ctc_authority"),
            "--textgrid-dir", str(import_root / "mfa_aligned_textgrid"),
            "--output-dir", str(output), "--filtered-dir", str(replay_filtered),
            "--wav-dir", str(import_root / "mfa_audio"),
            "--raw-text-dir", str(import_root / "reference"),
            "--original-txt-dir", str(import_root / "reference"),
            "--pinyin-dict", str(PROJECT_ROOT / "dict" / "fullpinyin_enword.dict"),
            "--ipa-dict", str(PROJECT_ROOT / "dict" / "mfa_ipa.dict"),
            "--en-phones-dir", str(import_root / "english_phones"),
            "--no-fix-short-word", "--no-detect-bgm", "--no-filter-suspicious",
            "--strict-ok", "--allow-filtered-integrity-failures",
            "--no-handle-unexpected-sil", "--no-enable-word-in-silence-filter",
            "--filter-word-energy-ratio", "0", "--mfa-input-axis-receipt",
            str(axis_dir / ".mfa_input_axis_receipt.json"), "--mfa-alignment-axis-receipt",
            str(axis_dir / ".mfa_alignment_axis_receipt.json"), "--mfa-axis-audio-root",
            str(import_root / "mfa_audio"), "--tts-authoritative-audio-root",
            str(import_root / "tts_audio")]
        proc = subprocess.run(postprocess_cmd, cwd=str(PROJECT_ROOT), text=True,
                              capture_output=True)
        if proc.returncode != 0:
            raise ValueError(f"filtered recovery postprocess failed (rc={proc.returncode}): {proc.stderr[-500:]}")
        if not postprocess_report.is_file():
            raise ValueError("filtered recovery postprocess report missing")
        report_rows = [json.loads(line) for line in postprocess_report.read_text(encoding="utf-8").splitlines() if line.strip()]
        present = {row.get("stem") for row in report_rows}
        for stem in sorted(set(frozen) - present):
            report_rows.append({"stem": stem, "status": "filtered_missing_mfa_alignment",
                                "filter_reasons": ["missing_mfa_alignment"], "warnings": []})
        report_rows.sort(key=lambda row: row.get("stem", ""))
        final_report = workspace / "filtered_recovery_final_report.jsonl"
        final_report.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in report_rows), encoding="utf-8")
        # Preserve exact frozen-set filesystem accounting for missing MFA rows
        # with copied filtered placeholders; they remain quarantined.
        for stem in sorted(set(frozen) - present):
            placeholder = import_root / "filtered_textgrid" / f"{stem}.TextGrid"
            target = replay_filtered / f"{stem}.TextGrid"
            if placeholder.is_file() and not target.exists():
                shutil.copy2(placeholder, target)
        recovered = [row["stem"] for row in report_rows if row.get("status") == "ok"]
        still_filtered = [row["stem"] for row in report_rows if row.get("status", "").startswith("filtered")]
        final_partition = validate_filtered_recovery_partition(frozen, accepted, recovered, still_filtered)
        taxonomy_path = workspace / "filtered_recovery_taxonomy.jsonl"
        taxonomy_path.write_text("".join(json.dumps({"stem": row["stem"], "final_status": row.get("status"),
            "partition": "output" if row["stem"] in recovered else "filtered",
            "final_report_sha256": _sha256_file(final_report)}, ensure_ascii=False, sort_keys=True) + "\n" for row in report_rows), encoding="utf-8")
        from analyze_gpu1000_run import audit_filtered_recovery_logic
        independent_audit_path = workspace / "filtered_recovery_independent_audit.json"
        independent_audit = audit_filtered_recovery_logic(
            output, replay_filtered, final_report, sorted(frozen), sorted(accepted))
        _strict_replay_write_once_json(independent_audit_path, independent_audit)
        if not independent_audit.get("ok"):
            raise ValueError(f"filtered recovery independent audit failed: {independent_audit.get('errors')}")
        accounting = make_pipeline_accounting_receipt(sorted(frozen), sorted(frozen), [], sorted(recovered), sorted(still_filtered), mode="filtered_recovery", route=["filtered_only_postprocess"], paths={"report": str(final_report), "filtered": str(replay_filtered)}, extra={"quarantine": True})
        accounting_path = workspace / "filtered_recovery_pipeline_accounting_receipt.json"
        write_pipeline_accounting_receipt(accounting_path, accounting)
        english_ledger = output / "filtered_recovery_english_evidence.json"
        english_evidence_dir = output / "english_provenance"
        english_evidence_dir.mkdir(parents=True, exist_ok=True)
        localized_manifest_ref = ({"path": str((import_root / "english_phones" / "en_alignment_manifest.json").resolve()),
                                   "sha256": _sha256_file(import_root / "english_phones" / "en_alignment_manifest.json")}
                                  if localized_en_manifest is not None else None)
        english_evidence_rows = []
        for stem in sorted(recovered):
            ledger_ref = localized_en_ledgers.get(stem)
            if not ledger_ref:
                continue
            source_ledger = Path(ledger_ref["path"])
            copied_ledger = english_evidence_dir / source_ledger.name
            if copied_ledger.exists() or copied_ledger.is_symlink():
                raise ValueError(f"filtered recovery English evidence collision: {copied_ledger}")
            shutil.copy2(source_ledger, copied_ledger)
            if _sha256_file(copied_ledger) != ledger_ref["sha256"]:
                raise ValueError(f"filtered recovery English evidence ledger hash mismatch: {stem}")
            source_grids = []
            try:
                ledger_payload = json.loads(source_ledger.read_text(encoding="utf-8"))
                for segment in ledger_payload.get("segments", []):
                    source = segment.get("mfa_textgrid") if isinstance(segment, dict) else None
                    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                        continue
                    grid = Path(source["path"])
                    if not grid.is_file() or grid.is_symlink():
                        raise ValueError(f"filtered recovery English evidence grid missing: {grid}")
                    copied_grid = english_evidence_dir / grid.name
                    if not copied_grid.exists() and not copied_grid.is_symlink():
                        shutil.copy2(grid, copied_grid)
                    if _sha256_file(copied_grid) != source.get("sha256"):
                        raise ValueError(f"filtered recovery English evidence grid hash mismatch: {grid.name}")
                    source_grids.append({"path": str(copied_grid.relative_to(output)),
                                         "sha256": _sha256_file(copied_grid),
                                         "segment_id": segment.get("segment_id")})
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"filtered recovery English evidence staging failed: {stem}: {exc}") from exc
            english_evidence_rows.append({"stem": stem,
                                          "ledger": {"path": str(copied_ledger.relative_to(output)),
                                                      "sha256": _sha256_file(copied_ledger)},
                                          "source_textgrids": source_grids})
        _strict_replay_write_once_json(english_ledger, {
            "schema": "strict-en-mfa-v1", "source": "localized_filtered_recovery",
            "localized_manifest": localized_manifest_ref,
            "stems": english_evidence_rows,
        })
        ok_entries = []
        for stem in sorted(recovered):
            tg = output / f"{stem}.TextGrid"
            ref = import_root / "reference" / f"{stem}.txt"
            entry = {"stem": stem, "textgrid_sha256": _sha256_file(tg),
                     "reference": {"path": str(ref), "sha256": _sha256_file(ref)}}
            evidence_row = next((row for row in english_evidence_rows if row["stem"] == stem), None)
            if evidence_row is not None:
                entry["english_provenance"] = {"schema": "strict-en-mfa-v1",
                    "ledger": evidence_row["ledger"],
                    "source_textgrids": evidence_row["source_textgrids"],
                    "localized_manifest": localized_manifest_ref}
            ok_entries.append(entry)
        strict_manifest = {"policy_version": "strict-ok-v3.2", "english_provenance_policy": {"schema": "strict-en-mfa-v1", "required": True, "evidence_root": str(output)}, "output_dir": str(output), "filtered_dir": str(replay_filtered), "expected_stems": sorted(frozen), "ok": ok_entries, "rejected": {stem: ["filtered"] for stem in sorted(still_filtered)}, "safe_empty": False, "safe_empty_applied": False, "global_reasons": [], "pipeline_accounting_receipt": {"schema": accounting["schema"], "path": str(accounting_path), "sha256": _sha256_file(accounting_path)}}
        _strict_replay_write_once_json(workspace / "filtered_recovery_strict_manifest.json", strict_manifest)
        observed_after = {name: _sha256_file((parent / name).resolve(strict=True)) for name in sorted(parent_hashes)}
        if observed_after != observed:
            raise ValueError("filtered recovery parent artifact changed after validation")
        partition = validate_filtered_recovery_partition(frozen, accepted, [], frozen)
        receipt = make_filtered_recovery_receipt(
            final_partition, imports, observed, observed_after,
            strict_id=args.filtered_recovery_strict_id,
            declared_vs_actual_inner_receipt=evidence["declared_vs_actual_inner_receipt"])
        receipt["manifest"] = validated
        receipt["execution"] = {"attempted": True, "mode": "filtered_only_postprocess", "return_code": 0,
                                 "report": str(final_report), "taxonomy": str(taxonomy_path),
                                 "strict_manifest": str(workspace / "filtered_recovery_strict_manifest.json"),
                                 "accounting_receipt": str(accounting_path)}
        receipt_path = workspace / "filtered_recovery_import_receipt.json"
        _strict_replay_write_once_json(receipt_path, receipt)
        print(f"filtered_recovery replayed {len(validated['stems'])} frozen stems in quarantine: {receipt_path}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: filtered_recovery: {exc}")
        return 1


def _strict_replay_path_new(path: Path, label: str) -> Path:
    """Require a fresh absolute /tmp path with no symlink ancestor."""
    if ".." in Path(path).parts:
        raise ValueError(f"{label} textual parent traversal forbidden")
    raw = Path(os.path.abspath(path))
    if not raw.is_absolute() or raw == Path(raw.anchor):
        raise ValueError(f"{label} must be an absolute new /tmp directory")
    try:
        raw.relative_to(Path("/tmp"))
    except ValueError as exc:
        raise ValueError(f"{label} must be under /tmp: {raw}") from exc
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} has symlink ancestor: {cursor}")
    if raw.exists() or raw.is_symlink():
        raise ValueError(f"{label} must not preexist: {raw}")
    return raw


def _strict_replay_root(path: Path, label: str) -> Path:
    """Validate an existing read-only source root (no recursive discovery)."""
    raw = Path(os.path.abspath(path))
    if not raw.is_dir() or raw.is_symlink():
        raise ValueError(f"{label} must be an ordinary directory: {raw}")
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} has symlink ancestor: {cursor}")
    return raw.resolve(strict=True)


def _strict_replay_file(path: Path, root: Path, label: str) -> Path:
    root = _strict_replay_root(root, f"{label} root")
    candidate = Path(os.path.abspath(path))
    try:
        rel = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes root: {candidate}") from exc
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} symlink component: {cursor}")
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{label} must be an ordinary file: {candidate}")
    return candidate.resolve(strict=True)


def _strict_replay_authority_file(root: Path, stem: str, suffix: str, label: str) -> Path:
    """Resolve one authoritative asset at root or one speaker direct child only."""
    root = _strict_replay_root(root, "authoritative source root")
    direct = root / f"{stem}{suffix}"
    if direct.exists():
        return _strict_replay_file(direct, root, label)
    matches: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and not child.is_symlink():
            candidate = child / f"{stem}{suffix}"
            if candidate.exists():
                matches.append(_strict_replay_file(candidate, root, label))
    if len(matches) != 1:
        raise ValueError(f"{label} missing or ambiguous in authoritative direct children")
    return matches[0]


def _strict_replay_hash(path: Path) -> str:
    return _sha256_file(path)


def _strict_replay_copy(src: Path, dst: Path, label: str) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise ValueError(f"strict replay destination preexists: {dst}")
    src_hash = _strict_replay_hash(src)
    shutil.copy2(src, dst)
    if not dst.is_file() or dst.is_symlink() or _strict_replay_hash(dst) != src_hash:
        raise ValueError(f"strict replay copy hash mismatch: {label}")
    if os.path.samestat(src.stat(), dst.stat()):
        raise ValueError(f"strict replay copy is inode alias: {label}")
    return {"source": str(src), "copy": str(dst), "size": src.stat().st_size,
            "sha256": src_hash}


def _strict_replay_write_once_json(path: Path, payload: dict) -> None:
    """Atomically create a JSON artifact exactly once; never overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError(f"strict replay artifact already exists: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if tmp.exists() or tmp.is_symlink():
        raise ValueError(f"strict replay temporary artifact collision: {tmp}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise ValueError(f"strict replay artifact already exists: {path}") from exc
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _strict_replay_replace_json(path: Path, payload: dict) -> None:
    """Atomically replace mutable stage-state JSON only."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _strict_replay_canonical(path: Path) -> tuple[dict, list[dict]]:
    manifest = _strict_replay_file(path, path.parent, "canonical manifest")
    if _sha256_file(manifest) != STRICT_REPLAY_CANONICAL_SHA256:
        raise ValueError("canonical manifest SHA256 mismatch")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"canonical manifest unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != STRICT_REPLAY_CANONICAL_SCHEMA:
        raise ValueError("canonical manifest schema mismatch")
    entries = payload.get("entries")
    if payload.get("count") != 96 or not isinstance(entries, list) or len(entries) != 96:
        raise ValueError("canonical manifest must contain exactly 96 entries")
    if payload.get("selection", {}).get("sort") != "sha256(stem UTF-8)":
        raise ValueError("canonical selection sort contract mismatch")
    counts: dict[tuple[str, str], int] = {}
    slots: list[dict] = []
    for index, row in enumerate(entries):
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
            raise ValueError(f"canonical slot {index} malformed")
        category, range_name, stem = row.get("category"), row.get("range"), row["stem"]
        if category not in STRICT_REPLAY_CATEGORIES or range_name not in STRICT_REPLAY_RANGES:
            raise ValueError(f"canonical slot {index} category/range invalid")
        if Path(stem).name != stem or not stem:
            raise ValueError(f"canonical slot {index} stem invalid")
        expected_key = hashlib.sha256(stem.encode("utf-8")).hexdigest()
        if row.get("stable_sort_key") != expected_key:
            raise ValueError(f"canonical slot {index} stable sort key mismatch")
        key = (category, range_name)
        ordinal = counts.get(key, 0)
        counts[key] = ordinal + 1
        slots.append({"slot": index, "category": category, "range": range_name,
                      "ordinal": ordinal, "stem": stem})
    if any(counts.get((cat, rng), 0) != 4 for cat in STRICT_REPLAY_CATEGORIES for rng in STRICT_REPLAY_RANGES):
        raise ValueError("canonical manifest does not provide 8x3x4 slots")
    return payload, slots


def _strict_replay_select_pilot(canonical: dict, slots: list[dict], pilot: bool) -> tuple[list[dict], dict]:
    """Select only the frozen canonical pilot slots; never accept a selector file."""
    if not pilot:
        return list(slots), {"version": "strict-replay-selector-v1", "pilot": False}
    selected: list[dict] = []
    rank_rows: list[dict] = []
    for category in STRICT_REPLAY_CATEGORIES:
        for range_name in STRICT_REPLAY_RANGES:
            candidates = [s for s in slots if s["category"] == category and s["range"] == range_name]
            if len(candidates) != 4:
                raise ValueError("pilot selector category/range cardinality drift")
            ranked = []
            for item in candidates:
                source = canonical["entries"][item["slot"]]
                explicit = source.get("selection_rank")
                if explicit is not None and (type(explicit) is not int or explicit < 0):
                    raise ValueError("pilot selection_rank invalid")
                digest = hashlib.sha256(item["stem"].encode("utf-8")).hexdigest()
                ranked.append((explicit if explicit is not None else digest, digest, item))
            ranked.sort(key=lambda row: (row[0], row[1], row[2]["slot"]))
            chosen = ranked[0][2]
            selected.append(chosen)
            rank_rows.append({"slot": chosen["slot"], "category": category,
                              "range": range_name, "selection_rank": canonical["entries"][chosen["slot"]].get("selection_rank"),
                              "rank_hash": ranked[0][1], "stem": chosen["stem"]})
    selected.sort(key=lambda row: row["slot"])
    return selected, {"version": "strict-replay-selector-v1", "pilot": True,
                      "rank_rows": rank_rows}


def _strict_replay_fingerprint(path: Path | None) -> dict:
    if path is None:
        return {"path": None, "sha256": None, "status": "unset"}
    path = Path(path)
    if path.is_file() and not path.is_symlink():
        return {"path": str(path.resolve()), "size": path.stat().st_size,
                "sha256": _sha256_file(path), "status": "ok"}
    if path.is_dir() and not path.is_symlink():
        digest, files = compute_model_tree_digest(path)
        return {"path": str(path.resolve()), "sha256": digest,
                "file_count": len(files), "status": "ok"}
    return {"path": str(path), "sha256": None, "status": "missing"}


def _strict_replay_english_subset(
    en_root: Path, en_stage: Path, stems: set[str], *,
    selection_slot_records: list[dict] | None = None,
    source_stems: list[str] | None = None,
    exclusion_records: list[dict] | None = None,
) -> dict:
    """Freeze parent-global English evidence and a selected-stem subset."""
    source_manifest = _strict_replay_file(en_root / "en_alignment_manifest.json", en_root,
                                          "English producer manifest")
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    if raw.get("schema") != "strict-en-mfa-v1" or raw.get("strict_provenance") is not True:
        raise ValueError("English producer manifest schema/provenance invalid")
    parent_hash = _sha256_file(source_manifest)
    parent_copy = en_stage / "en_alignment_manifest.json"
    if parent_copy.exists() or parent_copy.is_symlink():
        if parent_copy.is_symlink() or not parent_copy.is_file() or _sha256_file(parent_copy) != parent_hash:
            raise ValueError("parent English manifest copy collision/hash mismatch")
    else:
        shutil.copy2(source_manifest, parent_copy)
        if _sha256_file(parent_copy) != parent_hash:
            raise ValueError("parent English manifest copy hash mismatch")
    source_manifest_path = source_manifest.resolve()
    parent_copy_path = parent_copy.resolve()
    parent_copy_hash = _sha256_file(parent_copy)
    if source_manifest_path == parent_copy_path or parent_copy_hash != parent_hash:
        raise ValueError("parent English manifest role/path identity mismatch")
    expected = [x for x in raw.get("expected_segments", []) if isinstance(x, str)
                and x.split(":s", 1)[0] in stems]
    produced = [x for x in raw.get("produced_segments", []) if isinstance(x, str)
                and x.split(":s", 1)[0] in stems]
    rejected = [x for x in raw.get("rejected_segments", []) if isinstance(x, dict)
                and isinstance(x.get("id"), str)
                and x["id"].split(":s", 1)[0] in stems]
    ledgers = []
    for entry in raw.get("stem_ledgers", []):
        if not isinstance(entry, dict) or entry.get("stem") not in stems:
            continue
        stem = entry["stem"]
        src = _strict_replay_file(en_root / f"{stem}_en_phones.json", en_root,
                                  f"{stem} English ledger")
        dst = en_stage / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        digest = _sha256_file(dst)
        if digest != entry.get("sha256"):
            raise ValueError(f"English ledger source hash mismatch: {stem}")
        ledgers.append({"stem": stem, "path": str(dst), "sha256": digest})
    if set(x["stem"] for x in ledgers) != stems:
        raise ValueError("English subset ledger membership mismatch")
    entries = []
    for ledger in ledgers:
        ledger_payload = json.loads((en_stage / Path(ledger["path"]).name).read_text(encoding="utf-8"))
        segments = [segment for segment in ledger_payload.get("segments", [])
                    if isinstance(segment, dict)]
        entries.append({"stem": ledger["stem"], "ledger": ledger,
                        "segments": segments})
    counts = {
        "english_stems": len(stems), "english_segments": len(expected),
        "english_words": sum(len(segment.get("words", [])) for entry in entries for segment in entry["segments"]),
        "verified_words": sum(len([word for word in segment.get("words", [])
                                   if isinstance(word, dict) and word.get("status") == "verified"])
                               for entry in entries for segment in entry["segments"]),
        "rejected_words": 0,
    }
    counts["rejected_words"] = counts["english_words"] - counts["verified_words"]
    source_vector = sorted(source_stems if source_stems is not None else stems)
    exclusion_vector = sorted(exclusion_records or [], key=lambda row: row.get("stem", ""))
    if any(not isinstance(row, dict) or set(row) != {"stem", "reason"}
           or not isinstance(row.get("stem"), str)
           or not isinstance(row.get("reason"), str) or not row.get("reason")
           for row in exclusion_vector):
        raise ValueError("strict replay exclusion records must be {stem,reason}")
    excluded_stems_vector = sorted(row["stem"] for row in exclusion_vector)
    if len(set(excluded_stems_vector)) != len(excluded_stems_vector):
        raise ValueError("strict replay exclusion records contain duplicate stems")
    eligible_vector = sorted(stems)
    if source_stems is not None and (len(source_vector), len(exclusion_vector), len(eligible_vector)) != (21, 3, 18):
        raise ValueError("strict replay English denominator must be source21/exclusion3/eligible18")
    english_required_vector = sorted(eligible_vector)
    english_entries_vector = sorted(entry["stem"] for entry in entries)
    selection_records = list(selection_slot_records or [])
    if selection_slot_records is not None and len(selection_records) != 24:
        raise ValueError("strict replay English selection must contain exactly 24 slot records")
    if len({(row.get("slot"), row.get("stem")) for row in selection_records
            if isinstance(row, dict)}) != len(selection_records):
        raise ValueError("strict replay English selection contains duplicate/malformed records")
    subset = {
        "schema": "strict-replay-english-alignment-subset-v2.1",
        "selection_slot_records": selection_records,
        "selection_slot_count": len(selection_records),
        "selection_slot_digest": stable_json_digest(selection_records),
        "parent_global_manifest": {
            "authoritative_source": {"path": str(source_manifest_path),
                                      "sha256": parent_hash,
                                      "immutable_import_path": str(source_manifest_path),
                                      "immutable_import_sha256": parent_hash},
            "workspace_copy": {"path": str(parent_copy_path),
                                "sha256": parent_copy_hash,
                                "immutable_import_path": str(parent_copy_path),
                                "immutable_import_sha256": parent_copy_hash},
            "content_identity_sha256": parent_hash,
        },
        "expected_segments": expected, "produced_segments": produced,
        "rejected_segments": rejected, "entries": entries, "counts": counts,
        "source_stems": source_vector, "source_count": len(source_vector),
        "source_digest": stable_json_digest(source_vector),
        "exclusion_records": exclusion_vector,
        "exclusion_count": len(exclusion_vector),
        "exclusion_digest": stable_json_digest(exclusion_vector),
        "excluded_stems": excluded_stems_vector,
        "excluded_count": len(excluded_stems_vector),
        "excluded_digest": stable_json_digest(excluded_stems_vector),
        "eligible_stems": eligible_vector, "eligible_count": len(eligible_vector),
        "eligible_digest": stable_json_digest(eligible_vector),
        "english_required_stems": english_required_vector,
        "english_required_count": len(english_required_vector),
        "english_required_digest": stable_json_digest(english_required_vector),
        "english_entries_stems": english_entries_vector,
        "english_entries_count": len(english_entries_vector),
        "english_entries_digest": stable_json_digest(english_entries_vector),
        "digests": {
            "expected_segments": stable_json_digest(sorted(expected)),
            "produced_segments": stable_json_digest(sorted(produced)),
            "rejected_segments": stable_json_digest(sorted(x["id"] for x in rejected)),
        },
    }
    subset_path = en_stage.parent / "strict_replay_english_alignment_subset.json"
    if subset_path.exists() or subset_path.is_symlink():
        raise ValueError("English alignment subset path collision")
    subset_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"schema": subset["schema"], "subset_path": str(subset_path),
            "subset_sha256": _sha256_file(subset_path), "parent_path": str(source_manifest),
            "parent_sha256": parent_hash, "parent_copy_path": str(parent_copy),
            "parent_copy_sha256": _sha256_file(parent_copy),
            "parent_global_manifest": subset["parent_global_manifest"],
            "validation": "passed"}


STRICT_REPLAY_ENGLISH_IMPORT_SCHEMA = "strict-replay-english-import-v2.1"
STRICT_REPLAY_ENGLISH_PRODUCER_REVISION = "strict-replay-english-producer-v4.2.13"


def write_strict_replay_english_import(
    replay_import_manifest_path: Path,
    *, config_path: Path, dictionary_path: Path | None = None,
    dictionary_roles: dict[str, dict] | None = None,
    english_subset_path: Path | None = None,
    parent_english_manifest_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Freeze the English-required subset from an immutable replay import.

    This producer is deliberately pre-postprocess: it reads only the import
    receipt and copied authoritative references/ledgers/TextGrids, and never
    binds to later output, filtered, or strict manifests.
    """
    import re
    import datetime as _dt
    import copy
    import os
    import json as _json
    import hashlib as _hashlib

    import_path = Path(replay_import_manifest_path).resolve(strict=True)
    if import_path.name != "strict_replay_import.json" or import_path.is_symlink():
        raise ValueError("replay import manifest path invalid")
    payload = _json.loads(import_path.read_text(encoding="utf-8"))
    if payload.get("schema") != STRICT_REPLAY_SCHEMA:
        raise ValueError("replay import must be strict-replay-import-v2.1")
    canonical = payload.get("canonical", {})
    if (canonical.get("schema") != STRICT_REPLAY_CANONICAL_SCHEMA
            or canonical.get("sha256") != STRICT_REPLAY_CANONICAL_SHA256):
        raise ValueError("canonical identity invalid")
    cpath = Path(str(canonical.get("path", ""))).resolve(strict=True)
    if _sha256_file(cpath) != STRICT_REPLAY_CANONICAL_SHA256:
        raise ValueError("canonical external hash mismatch")
    import_hash = _sha256_file(import_path)
    workspace = Path(str(payload.get("paths", {}).get("workspace", ""))).resolve(strict=True)
    paths = payload.get("paths", {})
    payload_workspace = Path(str(paths.get("workspace", "")))
    immutable_raw = paths.get("immutable_import")
    if (not isinstance(immutable_raw, str) or not immutable_raw
            or not Path(immutable_raw).is_absolute()
            or ".." in Path(immutable_raw).parts
            or Path(immutable_raw).name != "strict_replay_import.json"):
        raise ValueError("immutable import path contract invalid")
    immutable_path = Path(immutable_raw).resolve(strict=True)
    if (not payload_workspace.is_absolute() or ".." in payload_workspace.parts
            or payload_workspace.name == ""):
        raise ValueError("workspace path contract invalid")
    payload_workspace = payload_workspace.resolve(strict=True)
    payload_output = Path(str(paths.get("output", "")))
    if (not payload_output.is_absolute() or ".." in payload_output.parts
            or payload_output.resolve(strict=True) == payload_workspace):
        raise ValueError("workspace/output role contract invalid")
    if immutable_path != import_path or immutable_path.parent != payload_workspace:
        raise ValueError("immutable import path/ctx mismatch")
    if output_path is None:
        output_path = workspace / "en_phones" / "strict_replay_english_import.json"
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("English import output must be new")
    config_path = Path(config_path).resolve(strict=True)
    config_hash = _sha256_file(config_path)
    dictionary_hash = None
    if dictionary_path is not None:
        dictionary_path = Path(dictionary_path).resolve(strict=True)
        dictionary_hash = _sha256_file(dictionary_path)
    if not isinstance(dictionary_roles, dict) or set(dictionary_roles) != {
            "chinese_mfa_dictionary", "pinyin_projection_dictionary",
            "english_pronunciation_dictionary"}:
        raise ValueError("complete dictionary_roles matrix is required")
    dictionary_roles = json.loads(json.dumps(dictionary_roles))
    for role_name, role in dictionary_roles.items():
        if (not isinstance(role, dict) or set(role) - {"path", "sha256"}
                or not isinstance(role.get("path"), str)
                or not isinstance(role.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", role["sha256"])):
            raise ValueError(f"dictionary role malformed: {role_name}")
    if dictionary_path is not None:
        english_role = dictionary_roles["english_pronunciation_dictionary"]
        if english_role["sha256"] != dictionary_hash:
            raise ValueError("English dictionary role/hash mismatch")
    payload_roles = payload.get("dictionary_roles")
    if (payload_roles != dictionary_roles
            or payload.get("dictionary_roles_digest") != stable_json_digest(dictionary_roles)):
        raise ValueError("replay import dictionary_roles binding mismatch")
    if english_subset_path is None or parent_english_manifest_path is None:
        raise ValueError("English subset/parent manifest bindings are required")
    english_subset_path = Path(english_subset_path).resolve(strict=True)
    parent_english_manifest_path = Path(parent_english_manifest_path).resolve(strict=True)
    if english_subset_path.parent != workspace or parent_english_manifest_path.parent != workspace / "en_phones":
        raise ValueError("English subset/parent manifest path role mismatch")
    subset_hash = _sha256_file(english_subset_path)
    parent_hash = _sha256_file(parent_english_manifest_path)
    subset_payload = _json.loads(english_subset_path.read_text(encoding="utf-8"))
    if subset_payload.get("schema") != "strict-replay-english-alignment-subset-v2.1":
        raise ValueError("English alignment subset must be v2.1")
    parent_global = subset_payload.get("parent_global_manifest")
    if not isinstance(parent_global, dict):
        raise ValueError("English parent_global_manifest missing")
    source_role = parent_global.get("authoritative_source")
    copy_role = parent_global.get("workspace_copy")
    identity_hash = parent_global.get("content_identity_sha256")
    if (not isinstance(source_role, dict) or not isinstance(copy_role, dict)
            or not isinstance(identity_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", identity_hash)):
        raise ValueError("English parent_global_manifest roles malformed")
    def _normalized_role(role: dict, label: str) -> tuple[Path, str]:
        required_keys = {"path", "sha256", "immutable_import_path", "immutable_import_sha256"}
        if set(role) != required_keys:
            raise ValueError(f"English parent_global_manifest {label} fields invalid")
        path_raw = role["path"]
        immutable_raw = role["immutable_import_path"]
        if (not isinstance(path_raw, str) or not isinstance(immutable_raw, str)
                or not Path(path_raw).is_absolute() or not Path(immutable_raw).is_absolute()
                or ".." in Path(path_raw).parts or ".." in Path(immutable_raw).parts):
            raise ValueError(f"English parent_global_manifest {label} path invalid")
        path = Path(path_raw).resolve(strict=True)
        immutable = Path(immutable_raw).resolve(strict=True)
        if path != immutable:
            raise ValueError(f"English parent_global_manifest {label} path/immutable mismatch")
        sha = role["sha256"]
        immutable_sha = role["immutable_import_sha256"]
        if (not isinstance(sha, str) or not isinstance(immutable_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", sha)
                or immutable_sha != sha or _sha256_file(path) != sha):
            raise ValueError(f"English parent_global_manifest {label} hash invalid")
        return path, sha
    source_role_path, source_role_hash = _normalized_role(source_role, "authoritative_source")
    copy_role_path, copy_role_hash = _normalized_role(copy_role, "workspace_copy")
    if source_role_path == copy_role_path or source_role_hash != copy_role_hash or source_role_hash != identity_hash:
        raise ValueError("English parent_global_manifest identity mismatch")
    if copy_role_path != parent_english_manifest_path:
        raise ValueError("English parent_global_manifest workspace copy binding mismatch")
    if (subset_payload.get("parent_global_manifest", {}).get("workspace_copy", {}).get("path")
            != str(parent_english_manifest_path)):
        raise ValueError("English parent_global_manifest path must be normalized")
    parent_payload = _json.loads(parent_english_manifest_path.read_text(encoding="utf-8"))
    if (parent_payload.get("schema") != "strict-en-mfa-v1"
            or parent_payload.get("strict_provenance") is not True):
        raise ValueError("English parent manifest schema/provenance invalid")
    assets = payload.get("assets")
    if not isinstance(assets, list) or len({row.get("stem") for row in assets}) != len(assets):
        raise ValueError("replay asset membership duplicate/malformed")
    missing = set(payload.get("missing_mfa_alignment", []))
    source_stems = sorted(row["stem"] for row in assets)
    eligible_stems = sorted(set(source_stems) - missing)
    excluded_stems = sorted(missing)
    if (len(source_stems), len(excluded_stems), len(eligible_stems)) != (21, 3, 18):
        raise ValueError("strict replay English denominator must be source21/exclusion3/eligible18")
    if (payload.get("source_stems") != source_stems
            or payload.get("source_count") != 21
            or payload.get("source_digest") != stable_json_digest(source_stems)
            or payload.get("excluded_stems") != excluded_stems
            or payload.get("excluded_count") != 3
            or payload.get("excluded_digest") != stable_json_digest(excluded_stems)
            or payload.get("eligible_stems") != eligible_stems
            or payload.get("eligible_count") != 18
            or payload.get("eligible_digest") != stable_json_digest(eligible_stems)):
        raise ValueError("replay import denominator vectors mismatch")
    payload_exclusions = payload.get("exclusion_records")
    if (not isinstance(payload_exclusions, list)
            or payload.get("exclusion_count") != 3
            or payload.get("exclusion_digest") != stable_json_digest(payload_exclusions)
            or [row.get("stem") for row in payload_exclusions] != excluded_stems
            or any(not isinstance(row, dict) or set(row) != {"stem", "reason"}
                   or not isinstance(row.get("reason"), str) or not row.get("reason")
                   for row in payload_exclusions)):
        raise ValueError("replay import exclusion records mismatch")
    if subset_payload.get("source_stems") != source_stems or subset_payload.get("source_count") != 21:
        raise ValueError("English subset source vector mismatch")
    if subset_payload.get("eligible_stems") != eligible_stems or subset_payload.get("eligible_count") != 18:
        raise ValueError("English subset eligible vector mismatch")
    if subset_payload.get("excluded_stems") != excluded_stems or subset_payload.get("excluded_count") != 3:
        raise ValueError("English subset excluded_stems vector mismatch")
    exclusion_records = subset_payload.get("exclusion_records")
    if (not isinstance(exclusion_records, list)
            or subset_payload.get("exclusion_count") != 3
            or subset_payload.get("exclusion_digest") != stable_json_digest(exclusion_records)
            or [row.get("stem") for row in exclusion_records] != excluded_stems
            or any(not isinstance(row, dict) or set(row) != {"stem", "reason"}
                   or not isinstance(row.get("reason"), str) or not row.get("reason")
                   for row in exclusion_records)):
        raise ValueError("English subset exclusion_records vector mismatch")
    required: list[str] = []
    records: list[dict] = []
    for row in sorted(assets, key=lambda item: item.get("stem", "")):
        stem = row.get("stem")
        if stem not in eligible_stems:
            continue
        txt_rec = row.get("assets", {}).get("authoritative_txt", {})
        txt_path = Path(str(txt_rec.get("copy", ""))).resolve(strict=True)
        if txt_path.is_symlink() or not txt_path.is_file() or txt_path.is_relative_to(Path(str(payload["paths"]["source_root"])).resolve()):
            raise ValueError(f"reference path invalid or external: {stem}")
        text = txt_path.read_text(encoding="utf-8")
        lexical = [token for token in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
                   if is_english_token(token)]
        if not lexical:
            continue
        required.append(stem)
        ledger = row.get("english", {}).get("ledger", {})
        ledger_path = Path(str(ledger.get("copy", ""))).resolve(strict=True)
        if ledger_path.is_symlink() or not ledger_path.is_file() or not ledger_path.is_relative_to(workspace):
            raise ValueError(f"ledger path invalid: {stem}")
        ledger_payload = _json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger_payload.get("schema") != "strict-en-mfa-v1" or ledger_payload.get("stem") != stem:
            raise ValueError(f"ledger schema/stem invalid: {stem}")
        source_tgs = []
        for key in ("ctc",):
            tg = row.get("assets", {}).get(key, {}).get(".TextGrid")
            if tg:
                tg_path = Path(str(tg.get("copy", ""))).resolve(strict=True)
                if not tg_path.is_relative_to(workspace) or tg_path.is_symlink():
                    raise ValueError(f"source TextGrid path invalid: {stem}")
                source_tgs.append({"path": str(tg_path.relative_to(workspace)), "sha256": _sha256_file(tg_path)})
        aligned = row.get("aligned", {})
        if aligned.get("status") != "missing_mfa":
            ap = Path(str(aligned.get("copy", ""))).resolve(strict=True)
            if not ap.is_relative_to(workspace) or ap.is_symlink():
                raise ValueError(f"aligned TextGrid path invalid: {stem}")
            source_tgs.append({"path": str(ap.relative_to(workspace)), "sha256": _sha256_file(ap)})
        records.append({"stem": stem, "status": "english_required",
                        "ledger": {"path": str(ledger_path.relative_to(workspace)),
                                   "sha256": _sha256_file(ledger_path), "schema": ledger_payload["schema"]},
                        "source_textgrids": source_tgs,
                        "validation": copy.deepcopy(row.get("english", {}).get("validation", {}))})
    if records != sorted(records, key=lambda item: item["stem"]) or len({r["stem"] for r in records}) != len(records):
        raise ValueError("English records order/duplicate invalid")
    if (len(required), len(records)) != (18, 18):
        raise ValueError("strict replay English denominator must be english_required18/english_entries18")
    if (subset_payload.get("english_required_stems") != sorted(required)
            or subset_payload.get("english_required_count") != 18
            or subset_payload.get("english_entries_stems") != sorted(record["stem"] for record in records)
            or subset_payload.get("english_entries_count") != 18):
        raise ValueError("English subset required/entry vectors mismatch")
    selection_records = subset_payload.get("selection_slot_records")
    if (not isinstance(selection_records, list) or len(selection_records) != 24
            or subset_payload.get("selection_slot_count") != 24
            or subset_payload.get("selection_slot_digest") != stable_json_digest(selection_records)):
        raise ValueError("English subset selection slot vector mismatch")
    def digest(values: list[str]) -> str:
        return stable_json_digest(sorted(values))
    result = {"schema": STRICT_REPLAY_ENGLISH_IMPORT_SCHEMA, "scope": "strict_replay",
              "run_id": "strict-replay", "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "canonical_manifest_path": str(cpath), "canonical_manifest_sha256": STRICT_REPLAY_CANONICAL_SHA256,
              "replay_import_manifest_path": str(import_path), "replay_import_manifest_sha256": import_hash,
              "selection_slot_records": subset_payload.get("selection_slot_records", []),
              "selection_slot_count": subset_payload.get("selection_slot_count"),
              "selection_slot_digest": subset_payload.get("selection_slot_digest"),
              "source_stems": source_stems, "source_count": len(source_stems),
              "source_digest": digest(source_stems),
              "exclusion_records": exclusion_records,
              "exclusion_count": len(exclusion_records),
              "exclusion_digest": stable_json_digest(exclusion_records),
              "excluded_stems": excluded_stems, "excluded_count": len(excluded_stems),
              "excluded_digest": digest(excluded_stems),
              "eligible_stems": eligible_stems, "eligible_count": len(eligible_stems),
              "eligible_digest": digest(eligible_stems),
              "english_required_stems": sorted(required),
              "english_required_count": len(required),
              "english_required_digest": digest(required),
              "english_entries_stems": sorted(record["stem"] for record in records),
              "english_entries_count": len(records),
              "english_entries_digest": digest(sorted(record["stem"] for record in records)),
              "producer_revision": STRICT_REPLAY_ENGLISH_PRODUCER_REVISION,
              "config_sha256": config_hash,
              "dictionary_roles": dictionary_roles,
              "dictionary_roles_digest": stable_json_digest(dictionary_roles),
              "parent_global_manifest": copy.deepcopy(parent_global),
              "english_alignment_subset_path": str(english_subset_path) if english_subset_path else None,
              "english_alignment_subset_sha256": subset_hash if 'subset_hash' in locals() else None,
              "parent_english_manifest_path": str(parent_english_manifest_path) if parent_english_manifest_path else None,
              "parent_english_manifest_sha256": parent_hash if 'parent_hash' in locals() else None,
              "records": records}
    if output_path.parent.exists():
        if output_path.parent.is_symlink() or not output_path.parent.is_dir():
            raise ValueError("English import output parent invalid")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=False)
    result.update({
        "english_alignment_subset_path": str(english_subset_path),
        "english_alignment_subset_sha256": subset_hash,
        "parent_english_manifest_path": str(parent_english_manifest_path),
        "parent_english_manifest_sha256": parent_hash,
    })
    output_path.write_text(_json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_strict_replay(args, cfg: dict, config_path: Path) -> int:
    """Import the selected canonical slots into a fresh, isolated run root."""
    required = ("strict_replay_manifest", "strict_replay_source_root",
                "strict_replay_ctc_dir", "strict_replay_aligned_dir",
                "strict_replay_en_dir", "strict_replay_mfa_audio_root",
                "workspace", "output_dir")
    if any(getattr(args, name, None) in (None, "") for name in required):
        print("ERROR: strict_replay requires all frozen CLI paths")
        return 1
    forbidden = ("overwrite", "force", "step", "skip_to", "dataset_limit",
                 "dataset_offset", "ctc_ready", "output_staging", "use_cache",
                 "auto_cache", "scan_only", "data_dir", "python", "device",
                 "nvme_cache", "no_cache", "no_output_staging", "validate",
                 "mfa_jobs", "mfa_en_jobs", "ctc_ready_stems_file", "stop_after",
                 "cache_dir", "list_steps")
    if any(getattr(args, name, False) not in (False, None, 0, "") for name in forbidden):
        print("ERROR: strict_replay forbids overwrite/skip/limit/cache/staging overrides")
        return 1
    if args.strict_replay_pilot:
        if config_path.resolve() != STRICT_REPLAY_CONFIG_PATH.resolve():
            print("ERROR: strict_replay pilot requires configs/hecheng_ria_fresh.yaml")
            return 1
        if _sha256_file(config_path) != STRICT_REPLAY_CONFIG_SHA256:
            print("ERROR: strict_replay pilot config SHA256 mismatch")
            return 1
        expected_post = {"strict_ok": True, "allow_filtered_integrity_failures": True,
                         "merge_silence": True, "fix_short_word": False,
                         "detect_bgm": False, "filter_suspicious": False,
                         "enable_text_correction": True, "handle_unexpected_sil": False}
        actual_post = cfg.get("postprocess", {})
        if any(actual_post.get(key) != value for key, value in expected_post.items()):
            print("ERROR: strict_replay pilot postprocess config contract mismatch")
            return 1
    try:
        workspace = _strict_replay_path_new(Path(args.workspace), "workspace")
        output = _strict_replay_path_new(Path(args.output_dir), "output-dir")
        if workspace == output:
            raise ValueError("workspace and output must be distinct roles")
        source_root = _strict_replay_root(Path(args.strict_replay_source_root), "source root")
        mfa_axis_root = _strict_replay_root(Path(args.strict_replay_mfa_audio_root), "MFA axis audio root")
        ctc_root = _strict_replay_root(Path(args.strict_replay_ctc_dir), "CTC root")
        aligned_root = _strict_replay_root(Path(args.strict_replay_aligned_dir), "aligned root")
        en_root = _strict_replay_root(Path(args.strict_replay_en_dir), "English root")
        canonical, slots = _strict_replay_canonical(Path(args.strict_replay_manifest))
        selected_slots, selector = _strict_replay_select_pilot(canonical, slots, bool(args.strict_replay_pilot))
        workspace.mkdir(parents=True, exist_ok=False)
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        print(f"ERROR: strict_replay preflight failed: {exc}")
        return 1

    unique_stems = sorted({slot["stem"] for slot in selected_slots})
    slot_assets: list[dict] = []
    missing_mfa: list[str] = []
    try:
        for slot in selected_slots:
            stem = slot["stem"]
            if not any(item.get("stem") == stem for item in slot_assets):
                bundle = {"stem": stem, "slot": slot["slot"], "assets": {}}
                wav = _strict_replay_authority_file(source_root, stem, ".wav", f"{stem} WAV")
                txt = _strict_replay_authority_file(source_root, stem, ".txt", f"{stem} TXT")
                bundle["assets"]["tts_authoritative_audio"] = _strict_replay_copy(
                    wav, workspace / "inputs" / stem / "tts_authoritative_audio" / f"{stem}.wav",
                    f"{stem} TTS authoritative WAV")
                # Legacy metadata alias; downstream MFA consumes only the
                # explicit mfa_axis_audio role below.
                bundle["assets"]["authoritative_wav"] = bundle["assets"]["tts_authoritative_audio"]
                mfa_wav = _strict_replay_authority_file(
                    mfa_axis_root, stem, ".wav", f"{stem} MFA-axis WAV")
                bundle["assets"]["mfa_axis_audio"] = _strict_replay_copy(
                    mfa_wav, workspace / "inputs" / stem / "mfa_axis_audio" / f"{stem}.wav",
                    f"{stem} MFA-axis WAV")
                bundle["assets"]["authoritative_txt"] = _strict_replay_copy(txt, workspace / "inputs" / stem / f"{stem}.txt", f"{stem} TXT")
                ctc_assets = {}
                for suffix in STRICT_REPLAY_CTC_SUFFIXES:
                    src = _strict_replay_file(ctc_root / f"{stem}{suffix}", ctc_root, f"{stem} CTC {suffix}")
                    ctc_assets[suffix] = _strict_replay_copy(src, workspace / "inputs" / stem / "ctc" / f"{stem}{suffix}", f"{stem} CTC {suffix}")
                bundle["assets"]["ctc"] = ctc_assets
                aligned = aligned_root / f"{stem}.TextGrid"
                if aligned.exists():
                    aligned = _strict_replay_file(aligned, aligned_root, f"{stem} aligned")
                    bundle["aligned"] = _strict_replay_copy(aligned, workspace / "inputs" / stem / "aligned" / aligned.name, f"{stem} aligned")
                else:
                    bundle["aligned"] = {"status": "missing_mfa", "reason": "missing_mfa_alignment"}
                    missing_mfa.append(stem)
                en_manifest = _strict_replay_file(en_root / "en_alignment_manifest.json", en_root, "English producer manifest")
                ledger = _strict_replay_file(en_root / f"{stem}_en_phones.json", en_root, f"{stem} English ledger")
                bundle["english"] = {
                    "producer_manifest": _strict_replay_copy(en_manifest, workspace / "inputs" / stem / "english" / en_manifest.name, "English producer manifest"),
                    "ledger": _strict_replay_copy(ledger, workspace / "inputs" / stem / "english" / ledger.name, f"{stem} English ledger"),
                }
                try:
                    em = json.loads(en_manifest.read_text(encoding="utf-8"))
                    ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
                    bundle["english"]["validation"] = {
                        "manifest_schema": em.get("schema"), "manifest_status": em.get("status"),
                        "ledger_schema": ledger_payload.get("schema"), "ledger_stem": ledger_payload.get("stem"),
                        "source_manifest_sha256": em.get("source_manifest_sha256", em.get("source_manifest_hash", _sha256_file(en_manifest))),
                        "producer_manifest_sha256": em.get("producer_manifest_sha256", em.get("producer_manifest_hash", _sha256_file(en_manifest))),
                        "valid": ledger_payload.get("stem") == stem,
                    }
                except Exception as exc:
                    raise ValueError(f"English provenance validation failed for {stem}: {exc}") from exc
                slot_assets.append(bundle)
    except (OSError, ValueError) as exc:
        print(f"ERROR: strict_replay asset import failed: {exc}")
        return 1

    # Freeze the imported eligible subset into the exact directories consumed
    # by the official postprocess and strict-ok stages.  Missing-MFA stems are
    # intentionally omitted from every downstream directory.
    eligible_stems = set(unique_stems) - set(missing_mfa)
    stage_cfg = json.loads(json.dumps(cfg))
    stage_cfg.setdefault("mfa", {})["allow_partial"] = False
    stage_cfg.setdefault("postprocess", {})["strict_ok"] = True
    stage_ctc = workspace / "ctc_pretg"
    stage_aligned = workspace / "aligned"
    stage_audio = workspace / "audio_16k"
    stage_tts_audio = workspace / "tts_authoritative_audio"
    stage_raw = workspace / "raw_text"
    stage_en = workspace / "en_phones"
    stage_output = output
    stage_filtered = output / "filtered"
    stage_temp = workspace / "temp"
    for directory in (stage_ctc, stage_aligned, stage_audio, stage_tts_audio, stage_raw, stage_en,
                      stage_temp, stage_filtered):
        directory.mkdir(parents=True, exist_ok=True)
    by_stem = {row["stem"]: row for row in slot_assets}
    try:
        for stem in sorted(eligible_stems):
            bundle = by_stem[stem]
            for suffix, record in bundle["assets"]["ctc"].items():
                shutil.copy2(record["copy"], stage_ctc / f"{stem}{suffix}")
            shutil.copy2(bundle["assets"]["mfa_axis_audio"]["copy"], stage_audio / f"{stem}.wav")
            shutil.copy2(bundle["assets"]["tts_authoritative_audio"]["copy"],
                         stage_tts_audio / f"{stem}.wav")
            shutil.copy2(bundle["assets"]["authoritative_txt"]["copy"], stage_raw / f"{stem}.txt")
            if bundle["aligned"].get("status") != "missing_mfa":
                shutil.copy2(bundle["aligned"]["copy"], stage_aligned / f"{stem}.TextGrid")
            shutil.copy2(bundle["english"]["ledger"]["copy"], stage_en / f"{stem}_en_phones.json")
        english_subset = _strict_replay_english_subset(
            en_root, stage_en, eligible_stems,
            selection_slot_records=selected_slots,
            source_stems=unique_stems,
            exclusion_records=[{"stem": stem, "reason": "missing_mfa_alignment"}
                               for stem in sorted(missing_mfa)],
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: strict_replay stage freeze failed: {exc}")
        return 1

    # Provisional v2 denominator is written before postprocess so the official
    # helper can load frozen accounting; it is replaced with final stage sets.
    provisional_accounting = make_pipeline_accounting_receipt(
        unique_stems, sorted(eligible_stems),
        [(stem, "missing_mfa_alignment") for stem in sorted(missing_mfa)],
        [], sorted(eligible_stems), run_id="strict-replay", mode="strict_replay",
        route=["import", "postprocess", "strict_ok"],
        paths={"output": str(output), "filtered": str(stage_filtered),
               "report": str(output / "postprocess_report.jsonl")},
        extra={"strict_replay_receipt": str(workspace / "strict_replay_import.json"),
               "failed_steps": []})
    write_pipeline_accounting_receipt(stage_ctc / ".pipeline_run_receipt_v2.json", provisional_accounting)
    ctx = {"workspace": workspace, "ctc_pretg": stage_ctc, "ctc_pretg_adj": stage_ctc,
           "aligned_dir": stage_aligned, "mfa_audio_dir": stage_audio,
           "mfa_axis_audio_dir": stage_audio, "tts_authoritative_audio_dir": stage_tts_audio,
           "raw_text_dir": stage_raw, "data_dir": stage_raw, "output_dir": stage_output,
           "filtered_dir": stage_filtered, "temp_dir": stage_temp,
           "mfa_dict": dict_path if 'dict_path' in locals() else resolve_path(PROJECT_ROOT, cfg.get("mfa_dict")),
           "models_dir": resolve_path(PROJECT_ROOT, cfg.get("models_dir", "models/mfa")),
           "accounting_required": True, "accounting_receipt_path": stage_ctc / ".pipeline_run_receipt_v2.json",
           "accounting_source_stems": tuple(unique_stems), "accounting_eligible_stems": tuple(sorted(eligible_stems)),
           "accounting_exclusions": tuple((stem, "missing_mfa_alignment") for stem in sorted(missing_mfa)),
           "expected_stems": tuple(sorted(eligible_stems)), "mfa_missing_stems": set()}
    ctx["strict_replay_mode"] = True
    ctx["axis_contract_required"] = True
    ctx["strict_replay_scope"] = "strict_replay"
    ctx["mode"] = "strict_replay"
    if _write_strict_replay_axis_receipts(ctx) != 0:
        return 1
    try:
        repo_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        repo_head = None
    config_fp = _strict_replay_fingerprint(config_path)
    model_path = resolve_path(PROJECT_ROOT, cfg.get("models_dir"))
    dict_path = resolve_path(PROJECT_ROOT, cfg.get("mfa_dict"))
    english_dictionary_path = resolve_path(
        PROJECT_ROOT, cfg.get("mfa_en", {}).get("dictionary", "dict/cmudict.dict"))
    dictionary_role_fingerprints = {
        "chinese_mfa_dictionary": _strict_replay_fingerprint(dict_path),
        "pinyin_projection_dictionary": _strict_replay_fingerprint(
            resolve_path(PROJECT_ROOT, cfg.get("pinyin_dict", "dict/fullpinyin_enword.dict"))),
        "english_pronunciation_dictionary": _strict_replay_fingerprint(english_dictionary_path),
    }
    output_receipt = workspace / "strict_replay_import.json"
    def _write_replay_receipt(stage_records: list[dict]) -> dict:
        output_stems = sorted(p.stem for p in output.glob("*.TextGrid"))
        filtered_stems = sorted(p.stem for p in stage_filtered.glob("*.TextGrid"))
        receipt = {
            "schema": STRICT_REPLAY_SCHEMA,
            "canonical": {"schema": canonical["schema"], "path": str(Path(args.strict_replay_manifest).resolve()), "sha256": STRICT_REPLAY_CANONICAL_SHA256, "count": 96},
            "selection_slot_records": selected_slots,
            "selection_slot_count": len(selected_slots),
            "selection_slot_digest": stable_json_digest(selected_slots),
            "source_stems": unique_stems, "source_count": len(unique_stems),
            "source_digest": stable_json_digest(unique_stems),
            "exclusion_records": [{"stem": stem, "reason": "missing_mfa_alignment"}
                                  for stem in sorted(set(missing_mfa))],
            "exclusion_count": len(set(missing_mfa)),
            "exclusion_digest": stable_json_digest([{"stem": stem, "reason": "missing_mfa_alignment"}
                                                      for stem in sorted(set(missing_mfa))]),
            "excluded_stems": sorted(set(missing_mfa)),
            "excluded_count": len(set(missing_mfa)),
            "excluded_digest": stable_json_digest(sorted(set(missing_mfa))),
            "eligible_stems": sorted(eligible_stems),
            "eligible_count": len(eligible_stems),
            "eligible_digest": stable_json_digest(sorted(eligible_stems)),
            "dictionary_roles": {name: {"path": fp.get("path"), "sha256": fp.get("sha256")}
                                 for name, fp in dictionary_role_fingerprints.items()},
            "dictionary_roles_digest": stable_json_digest({name: {"path": fp.get("path"), "sha256": fp.get("sha256")}
                                                            for name, fp in dictionary_role_fingerprints.items()}),
            "slot_stem_mapping": [{"slot": s["slot"], "stem": s["stem"]} for s in slots],
            "slot_assets": [{"slot": s["slot"], "stem": s["stem"], "bundle_stem": s["stem"]} for s in selected_slots],
            "source_manifest_slots": 96,
            "pilot_selector_version": selector["version"], "pilot_selector": selector,
            "fingerprints": {"repo_head": repo_head, "config": config_fp, "model": _strict_replay_fingerprint(model_path), "dictionary": _strict_replay_fingerprint(dict_path), "dictionary_roles": dictionary_role_fingerprints},
            "dictionary_roles_digest": stable_json_digest({name: {"path": fp.get("path"), "sha256": fp.get("sha256")}
                                                             for name, fp in dictionary_role_fingerprints.items()}),
            "config_contract": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path),
                                "repo_head": repo_head, "postprocess": {key: stage_cfg.get("postprocess", {}).get(key) for key in ("strict_ok", "allow_filtered_integrity_failures", "merge_silence", "fix_short_word", "detect_bgm", "filter_suspicious", "enable_text_correction", "handle_unexpected_sil")}},
            "argv": list(sys.argv), "paths": {"workspace": str(workspace), "output": str(output), "immutable_import": str(output_receipt), "source_root": str(source_root), "mfa_axis_audio_root": str(mfa_axis_root), "ctc_root": str(ctc_root), "aligned_root": str(aligned_root), "english_root": str(en_root)},
            "assets": slot_assets, "missing_mfa_alignment": sorted(set(missing_mfa)),
            "report": {"source": len(unique_stems), "eligible": len(eligible_stems), "output": len(output_stems), "filtered": len(filtered_stems)},
            "english_subset": english_subset, "stages": stage_records, "global_reasons": []}
        _strict_replay_write_once_json(output_receipt, receipt)
        (workspace / "strict_replay_import.sha256").write_text(_sha256_file(output_receipt) + "\n", encoding="ascii")
        return receipt
    _write_replay_receipt([{"stage": "import", "argv": list(sys.argv), "workspace": str(workspace), "return_code": 0, "output": str(output), "filtered": str(stage_filtered), "reasons": []}])
    try:
        english_dictionary = english_dictionary_path
        dictionary_roles = {
            name: {"path": fingerprint.get("path"), "sha256": fingerprint.get("sha256")}
            for name, fingerprint in dictionary_role_fingerprints.items()
        }
        write_strict_replay_english_import(output_receipt, config_path=config_path,
                                           dictionary_path=english_dictionary,
                                           dictionary_roles=dictionary_roles,
                                           english_subset_path=Path(english_subset["subset_path"]),
                                           parent_english_manifest_path=Path(english_subset["parent_copy_path"]),
                                           output_path=workspace / "strict_replay_english_import.json")
    except (OSError, ValueError) as exc:
        print(f"ERROR: strict_replay English import producer failed: {exc}")
        return 1
    stage_state_path = workspace / ".strict_replay_stage_state.json"
    _strict_replay_write_once_json(stage_state_path, {"authoritative": False, "stages": [], "run_id": "strict-replay"})
    stage_results: list[dict] = []
    mfa_python = find_mfa_python(cfg.get("python_path", "")) or Path(sys.executable)
    for stage_name, stage_func in (("postprocess", step_postprocess),):
        try:
            rc = stage_func(args, stage_cfg, mfa_python, ctx)
            reason = [] if rc == 0 else [f"{stage_name}_return_code_{rc}"]
        except Exception as exc:
            rc, reason = 1, [f"{stage_name}_exception:{exc}"]
        stage_results.append({"stage": stage_name, "argv": list(sys.argv),
                              "workspace": str(workspace), "return_code": rc,
                              "output": str(output), "filtered": str(stage_filtered),
                              "reasons": reason})
        if rc != 0:
            print(f"ERROR: strict_replay official stage {stage_name} failed: {reason}")
            return 1
    _strict_replay_replace_json(stage_state_path, {"authoritative": False, "stages": stage_results, "run_id": "strict-replay"})
    post_output_stems = sorted(p.stem for p in output.glob("*.TextGrid"))
    post_filtered_stems = sorted(p.stem for p in stage_filtered.glob("*.TextGrid"))
    post_accounting = make_pipeline_accounting_receipt(
        unique_stems, sorted(eligible_stems),
        [(stem, "missing_mfa_alignment") for stem in sorted(missing_mfa)],
        post_output_stems, post_filtered_stems, run_id="strict-replay", mode="strict_replay",
        route=["import", "postprocess", "strict_ok"],
        paths={"output": str(output), "filtered": str(stage_filtered),
               "report": str(output / "postprocess_report.jsonl")},
        extra={"strict_replay_receipt": str(output_receipt), "failed_steps": []})
    post_accounting.setdefault("extra", {})["strict_replay_evidence"] = {
        "import_manifest": str(output_receipt), "import_sha256": _sha256_file(output_receipt),
        "english_import": str(workspace / "strict_replay_english_import.json"),
        "english_sha256": _sha256_file(workspace / "strict_replay_english_import.json"),
        "english_subset": str(workspace / "strict_replay_english_alignment_subset.json"),
        "english_subset_sha256": _sha256_file(workspace / "strict_replay_english_alignment_subset.json"),
        "parent_english_manifest": str(workspace / "en_phones" / "en_alignment_manifest.json"),
        "parent_english_sha256": _sha256_file(workspace / "en_phones" / "en_alignment_manifest.json")}
    _strict_replay_write_once_json(output / ".pipeline_run_receipt_v2.json", post_accounting)
    ctx["accounting_receipt_path"] = output / ".pipeline_run_receipt_v2.json"
    ctx["strict_replay_english_import_path"] = workspace / "strict_replay_english_import.json"
    ctx["strict_replay_immutable_import_path"] = output_receipt
    ctx["strict_replay_english_manifest_path"] = workspace / "en_phones" / "en_alignment_manifest.json"
    ctx["strict_replay_english_subset_path"] = workspace / "strict_replay_english_alignment_subset.json"
    ctx["strict_replay_formal_receipt_path"] = output / ".pipeline_run_receipt_v2.json"
    ctx["strict_replay_postprocess_report_path"] = output / "postprocess_report.jsonl"
    strict_name, strict_func = "strict_ok", step_strict_ok
    try:
        rc = strict_func(args, stage_cfg, mfa_python, ctx)
        reason = [] if rc == 0 else [f"{strict_name}_return_code_{rc}"]
    except Exception as exc:
        rc, reason = 1, [f"{strict_name}_exception:{exc}"]
    stage_results.append({"stage": strict_name, "argv": list(ctx.get("strict_replay_strict_argv", sys.argv)), "workspace": str(workspace), "return_code": rc,
                          "output": str(output), "filtered": str(stage_filtered), "reasons": reason})
    strict_binding = {"status": "missing", "expected_path": str(output / "strict_ok_manifest.json"),
                      "sha256": None, "missing_reason": "not_created"}
    if (output / "strict_ok_manifest.json").exists():
        manifest_path = output / "strict_ok_manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            strict_binding = {"status": "present", "expected_path": str(manifest_path),
                              "path": str(manifest_path),
                              "sha256": _sha256_file(manifest_path)}
        else:
            strict_binding = {"status": "missing", "expected_path": str(manifest_path),
                              "sha256": None, "missing_reason": "unsafe_file_type"}
    if rc == 0 and strict_binding["status"] != "present":
        reason = [*reason, "strict_manifest_missing"]
        rc = 1
    if rc != 0:
        print(f"ERROR: strict_replay official stage {strict_name} failed: {reason}")
        _strict_replay_write_once_json(output / "strict_replay_final_evidence.json", {
            "schema": "strict-replay-final-evidence-v1", "authoritative": False,
            "import_sha256": _sha256_file(output_receipt),
            "english_import_sha256": _sha256_file(workspace / "strict_replay_english_import.json"),
            "english_subset_sha256": _sha256_file(workspace / "strict_replay_english_alignment_subset.json"),
            "parent_english_sha256": _sha256_file(workspace / "en_phones" / "en_alignment_manifest.json"),
            "formal_receipt_sha256": _sha256_file(output / ".pipeline_run_receipt_v2.json"),
            "strict_manifest_binding": strict_binding, "stage_results": stage_results, "global_reasons": reason})
        return 1
    _strict_replay_write_once_json(output / "strict_replay_final_evidence.json", {
        "schema": "strict-replay-final-evidence-v1", "authoritative": False,
        "import_sha256": _sha256_file(output_receipt),
        "english_import_sha256": _sha256_file(workspace / "strict_replay_english_import.json"),
        "english_subset_sha256": _sha256_file(workspace / "strict_replay_english_alignment_subset.json"),
        "parent_english_sha256": _sha256_file(workspace / "en_phones" / "en_alignment_manifest.json"),
        "formal_receipt_sha256": _sha256_file(output / ".pipeline_run_receipt_v2.json"),
        "strict_manifest_sha256": _sha256_file(output / "strict_ok_manifest.json"),
        "strict_manifest_binding": strict_binding, "stage_results": stage_results, "global_reasons": []})
    print(f"strict_replay imported/staged {len(selected_slots)} slots ({len(unique_stems)} stems) -> {output_receipt}")
    return 0


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
    parser.add_argument("--mfa-jobs", type=int, default=None, metavar="N",
                        help="Override mfa.num_jobs for this invocation (must be positive).")
    parser.add_argument("--mfa-en-jobs", type=int, default=None, metavar="N",
                        help="Override mfa_en.num_jobs for this invocation (must be positive).")
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
                        choices=["full", "ctc_ready", "batch_ctc_ready", "nvrasr_fallback", "strict_replay", "filtered_recovery", "mfa_retry", "mfa_rescue"],
                        help="Pipeline mode (default: from config, or 'full').")
    parser.add_argument("--strict-replay-manifest", type=str, default=None,
                        help="Canonical 96-slot manifest for strict_replay.")
    parser.add_argument("--strict-replay-source-root", type=str, default=None,
                        help="Authoritative WAV/TXT root for strict_replay.")
    parser.add_argument("--strict-replay-ctc-dir", type=str, default=None,
                        help="Authoritative CTC artifact root for strict_replay.")
    parser.add_argument("--strict-replay-aligned-dir", type=str, default=None,
                        help="MFA aligned TextGrid root for strict_replay.")
    parser.add_argument("--strict-replay-en-dir", type=str, default=None,
                        help="English ledger/producer manifest root for strict_replay.")
    parser.add_argument("--strict-replay-mfa-audio-root", type=str, default=None,
                        help="Explicit MFA-axis audio root for strict_replay (distinct from TTS authority).")
    parser.add_argument("--strict-replay-pilot", action="store_true",
                        help="Select the frozen 24-slot strict_replay pilot subset.")
    parser.add_argument("--filtered-recovery-parent-root", type=str, default=None,
                        help="Read-only sealed parent root for filtered_recovery.")
    parser.add_argument("--filtered-recovery-strict-id", type=str, default=None,
                        help="Opaque sealed-parent strict run identifier for evidence binding.")
    parser.add_argument("--filtered-recovery-frozen-manifest", type=str, default=None,
                        help="Frozen rejected stem manifest (read-only; denominator is derived from contents).")
    parser.add_argument("--filtered-recovery-accepted-manifest", type=str, default=None,
                        help="Sealed parent accepted-776 manifest (read-only).")
    parser.add_argument("--filtered-recovery-manifest", type=str, default=None,
                        help="Digest-bound subset/full filtered recovery manifest.")
    parser.add_argument("--filtered-recovery-evidence-receipt", type=str, default=None,
                        help="Explicit filtered-recovery evidence receipt binding the frozen partition, parent digests, and inner-receipt mismatch.")
    parser.add_argument("--filtered-recovery-import-receipt", type=str, default=None,
                        help="Existing filtered-recovery import receipt for --filtered-recovery-validate-only.")
    parser.add_argument("--filtered-recovery-validate-only", action="store_true",
                        help="Validate an existing quarantined filtered-recovery import receipt without copying or replaying.")
    parser.add_argument("--filtered-recovery-aligned-root", type=str, default=None,
                        help="Optional fresh staged aligned-TextGrid root for frozen-only replay; never writes parent.")
    parser.add_argument("--filtered-recovery-ctc-root", type=str, default=None,
                        help="Optional fresh frozen-only CTC producer root; per-stem assets override parent CTC inputs.")
    parser.add_argument("--filtered-recovery-english-root", type=str, default=None,
                        help="Optional fresh frozen-only English strict-ledger root; per-stem ledgers override parent inputs.")
    parser.add_argument("--filtered-recovery-english-aligned-root", type=str, default=None,
                        help="Optional fresh English MFA aligned segment-TextGrid root for provenance staging.")
    parser.add_argument("--mfa-retry-parent-root", type=str, default=None,
                        help="Read-only parent root for exact-missing MFA retry packet.")
    parser.add_argument("--mfa-retry-stems-file", type=str, default=None,
                        help="Exact frozen stem list for MFA retry.")
    parser.add_argument("--mfa-retry-frozen-manifest", type=str, default=None)
    parser.add_argument("--mfa-retry-accepted-manifest", type=str, default=None)
    parser.add_argument("--mfa-retry-workspace", type=str, default=None)
    parser.add_argument("--mfa-retry-execute", action="store_true",
                        help="Execute the approved exact-missing MFA command in fresh quarantine workspace.")
    parser.add_argument("--mfa-retry-mfa", type=str, default=None,
                        help="Explicit MFA executable for manual retry packet (otherwise environment default).")
    parser.add_argument("--mfa-retry-model", type=str, default=None,
                        help="Explicit acoustic model for manual retry packet.")
    parser.add_argument("--mfa-retry-dict", type=str, default=None,
                        help="Explicit pronunciation dictionary for manual retry packet.")
    parser.add_argument("--mfa-rescue-prior-receipt", type=str, default=None)
    parser.add_argument("--mfa-rescue-stem", type=str, default=None)
    parser.add_argument("--mfa-rescue-workspace", type=str, default=None)
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
    for option, section, flag in ((args.mfa_jobs, "mfa", "--mfa-jobs"),
                                  (args.mfa_en_jobs, "mfa_en", "--mfa-en-jobs")):
        if option is not None:
            if option <= 0:
                parser.error(f"{flag} must be positive")
            cfg.setdefault(section, {})["num_jobs"] = option

    # ── Config schema validation (R8) — deferred until mode is resolved ──

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

    if mode not in ("full", "ctc_ready", "batch_ctc_ready", "nvrasr_fallback", "strict_replay", "filtered_recovery", "mfa_retry", "mfa_rescue"):
        print(f"ERROR: Unknown mode: {mode}")
        sys.exit(1)
    print(f"Pipeline mode: {mode}")

    # ── Config schema validation (R8) ────────────────────────────────
    _config_errors = validate_config(cfg, mode)
    if _config_errors:
        print(f"ERROR: Config validation failed ({len(_config_errors)} issues):")
        for err in _config_errors[:20]:
            print(f"  - {err}")
        if len(_config_errors) > 20:
            print(f"  ... and {len(_config_errors) - 20} more")
        return 1
    # ─────────────────────────────────────────────────────────────────

    # Strict replay is an isolated import route.  It must not probe MFA,
    # create ordinary production workspaces, discover caches, or execute any
    # acoustic/postprocess step.  All paths and assets are frozen by the
    # dedicated import routine above.
    if mode == "strict_replay":
        return run_strict_replay(args, cfg, config_path)
    if mode == "filtered_recovery":
        if not args.filtered_recovery_parent_root:
            print("ERROR: filtered_recovery requires --filtered-recovery-parent-root")
            return 1
        return run_filtered_recovery(args, cfg, config_path)
    if mode == "mfa_retry":
        if not args.mfa_retry_parent_root or not args.mfa_retry_stems_file or not args.mfa_retry_workspace:
            print("ERROR: mfa_retry requires --mfa-retry-parent-root, --mfa-retry-stems-file, --mfa-retry-workspace")
            return 1
        try:
            parent = Path(args.mfa_retry_parent_root)
            stems = [line.strip() for line in Path(args.mfa_retry_stems_file).read_text(encoding="utf-8").splitlines() if line.strip()]
            frozen_path = Path(args.mfa_retry_frozen_manifest) if args.mfa_retry_frozen_manifest else parent / "frozen_filtered.json"
            accepted_path = Path(args.mfa_retry_accepted_manifest) if args.mfa_retry_accepted_manifest else parent / "strict_ok_manifest.json"
            frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
            accepted_payload = json.loads(accepted_path.read_text(encoding="utf-8"))
            frozen = frozen_payload.get("stems", frozen_payload.get("filtered", {}).get("stems", []))
            accepted = accepted_payload.get("ok", accepted_payload.get("output", {}).get("stems", []))
            if accepted and isinstance(accepted[0], dict):
                accepted = [row.get("stem") for row in accepted]
            prepare_mfa_retry_packet(
                parent, Path(args.mfa_retry_workspace), stems,
                frozen_stems=frozen, accepted_stems=accepted,
                execute=args.mfa_retry_execute,
                mfa_python=Path(args.mfa_retry_mfa) if args.mfa_retry_mfa else None,
                model_path=Path(args.mfa_retry_model) if args.mfa_retry_model else None,
                dictionary_path=Path(args.mfa_retry_dict) if args.mfa_retry_dict else None)
            print(f"mfa_retry packet ready: {args.mfa_retry_workspace}/mfa_retry_receipt.json")
            return 0
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: mfa_retry: {exc}")
            return 1
    if mode == "mfa_rescue":
        if not args.mfa_retry_parent_root or not args.mfa_rescue_stem or not args.mfa_rescue_workspace or not args.mfa_rescue_prior_receipt:
            print("ERROR: mfa_rescue requires parent, stem, workspace, and prior receipt")
            return 1
        try:
            parent = Path(args.mfa_retry_parent_root)
            frozen_path = Path(args.mfa_retry_frozen_manifest) if args.mfa_retry_frozen_manifest else parent / "frozen_filtered.json"
            accepted_path = Path(args.mfa_retry_accepted_manifest) if args.mfa_retry_accepted_manifest else parent / "strict_ok_manifest.json"
            frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
            accepted_payload = json.loads(accepted_path.read_text(encoding="utf-8"))
            frozen = frozen_payload.get("stems", frozen_payload.get("filtered", {}).get("stems", []))
            accepted = accepted_payload.get("ok", accepted_payload.get("output", {}).get("stems", []))
            if accepted and isinstance(accepted[0], dict): accepted = [row.get("stem") for row in accepted]
            receipt = run_mfa_singleton_rescue(parent, Path(args.mfa_rescue_workspace), args.mfa_rescue_stem,
                                               frozen_stems=frozen, accepted_stems=accepted,
                                               prior_receipt_path=Path(args.mfa_rescue_prior_receipt))
            print(f"mfa_rescue complete: {args.mfa_rescue_workspace}/mfa_rescue_receipt.json")
            return 0
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: mfa_rescue: {exc}")
            return 1

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
                fail_list.append(ds_name)
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
        _stage_root = workspace / "output_staging" / f"{_run_id}_{os.getpid()}"
        output_dir = _stage_root
        # Non-strict staging still needs an isolated filtered partition.  Keep
        # it beside the workspace so the streaming uploader can transfer it
        # together with the published output directory.
        filtered_dir = workspace / "filtered"
        print(f"  Run-specific output staging: {output_dir}")
        print(f"  Versioned publish target: {_nas_output_dir}")
    else:
        # Non-strict, non-staging: use a run-specific subdirectory inside
        # the configured output/filtered roots to prevent accidental merging
        # of old and new results.
        _run_root = workspace / "runs" / _run_id
        _run_root.mkdir(parents=True, exist_ok=False)
        output_dir = _run_root / "output"
        filtered_dir = _run_root / "filtered"
        print(f"  Run-specific output: {output_dir}")
        print(f"  Run-specific filtered: {filtered_dir}")
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
            print(f"ERROR: --skip-to '{args.skip_to}' is not in the '{mode}' route.")
            print(f"  Allowed steps: {' '.join(step_order)}")
            return 1
        run_list = step_order[step_order.index(args.skip_to):]
    else:
        run_list = list(step_order)

    # --scan-only: only run the link step (single-dataset mode)
    if args.scan_only and mode in ("full", "ctc_ready"):
        if "link" in run_list:
            run_list = ["link"]
        elif mode == "ctc_ready":
            run_list = run_list[:1]  # ctc_ready first step is link (read-only)
        else:
            print("ERROR: --scan-only in full mode without link would run trim"
                  " (modifies audio). Use ctc_ready mode for read-only scan.")
            return 1
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
        "mode": mode,
        "strict_ready": _strict_ready,
        # v2 accounting is mandatory for pipeline execution/resume.  Direct
        # unit callers that omit this flag retain legacy fixture behaviour.
        "accounting_required": True,
        # Every downstream stage receives the exact formal output receipt
        # path.  Consumers must never infer a workspace/ctc sibling receipt.
        "accounting_receipt_path": output_dir / ".pipeline_run_receipt_v2.json",
        "accounting_source_receipt_path": workspace / cfg.get("ctc_pretg", "ctc_pretg") / ".pipeline_run_receipt_v2.json",
        "mfa_audio_dir": workspace / "audio_16k",
        # Explicit axis-role roots consumed by postprocess/audit.  These are
        # replaced by validated receipt-bound paths once each stage freezes.
        "mfa_axis_audio_dir": workspace / "audio_16k",
        "tts_authoritative_audio_dir": audio_dir,
        "axis_contract_required": True,
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
                                          cfg.get("output_spec", {}), ctx)
            if issues:
                print(f"  [VALIDATE] {step_name} — output check failed:")
                for issue in issues:
                    print(f"    {issue}")
                # --validate contract violations are real failures
                failed.append(f"validate:{step_name}")
                if not args.force:
                    print("  Stopping. Use --force to continue on validation errors.")
                    break
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
    # Skip when running as a subprocess of streaming_pipeline.  Child runs
    # receive an explicit --workspace; writing their single-batch scan cache
    # here would overwrite the parent's multi-dataset batch cache.
    _config_mode = cfg.get("mode", "")
    if (not _strict_ready and mode in ("ctc_ready", "full") and not failed
            and _config_mode != "batch_ctc_ready"
            and not getattr(args, "workspace", None)):
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

    # ── Write pipeline run receipt (v2 frozen source denominator) ──
    _input_stems = tuple(ctx.get("accounting_source_stems",
                                ctx.get("expected_stems",
                                        sorted({p.stem for p in audio_dir.rglob("*.wav")}
                                               if audio_dir.exists() else ()))))
    _eligible_stems = tuple(ctx.get("accounting_eligible_stems",
                                  ctx.get("expected_stems", _input_stems)))
    _exclusions = list(ctx.get("accounting_exclusions", ()))
    _output_stems = sorted({p.stem for p in output_dir.glob("*.TextGrid")}
                           if output_dir.exists() else ()) if not failed else []
    _filtered_stems = sorted({p.stem for p in filtered_dir.glob("*.TextGrid")}
                             if filtered_dir.exists() else ())
    # Preserve denominator conservation even when a failed/partial pipeline
    # leaves no publication artifact: every eligible stem absent from output
    # is represented in filtered evidence (with failed_steps in ``extra``).
    _eligible_set = set(_eligible_stems)
    _observed_filtered = (set(_filtered_stems) | (_eligible_set - set(_output_stems))) & _eligible_set
    try:
        _pipeline_accounting = make_pipeline_accounting_receipt(
            source_stems=list(_input_stems), eligible_stems=list(_eligible_stems),
            exclusions=_exclusions, output_stems=_output_stems,
            filtered_stems=sorted(_observed_filtered), run_id=_run_id, mode=mode,
            route=run_list,
            paths={"output": str(output_dir), "filtered": str(filtered_dir)},
            extra={"failed_steps": list(failed), "source_frozen": True},
        )
        write_pipeline_accounting_receipt(output_dir, _pipeline_accounting)
    except (TypeError, ValueError) as exc:
        print(f"  ERROR: failed to write v2 pipeline accounting receipt: {exc}")
        failed.append("receipt_accounting")

    # strict_ok can isolate additional candidates after it reads the
    # pre-audit receipt.  Rebind the audited manifest to the final receipt
    # before the publication guard checks both partition sets and its hash.
    if (not failed and "strict_ok" in run_list
            and (output_dir / "strict_ok_manifest.json").is_file()):
        if _refresh_strict_manifest_accounting_binding(output_dir) != 0:
            failed.append("strict_manifest_accounting_refresh")

    # ── Clean up temp NVMe cache ──
    if _nvme_cache_dir and _nvme_is_temp:
        _cleanup_nvme_cache(_nvme_cache_dir, _nvme_is_temp)

    # ── Publish output staging to a new versioned NAS directory ──
    _final_output = output_dir
    _should_publish = (
        _nas_output_dir is not None
        and ("strict_ok" in run_list or _output_staging)
        and not args.stop_after
        and output_dir.exists()
    )
    if _should_publish and failed:
        print(f"\n  Publish skipped because the pipeline failed: {', '.join(failed)}")
    elif _should_publish:
        print(f"\n  Publishing validated output: {_nas_output_dir}")
        from pipeline_utils import publish_output_versioned, write_publish_manifest
        manifest_path = write_publish_manifest(output_dir)
        print(f"  Publish manifest: {manifest_path}")
        if "strict_ok" in run_list:
            _published = publish_output_versioned(output_dir, _nas_output_dir)
        else:
            # In pipelined no-reference mode the CLI output target is the
            # batch-local handoff directory, not a versioned NAS root.  The
            # latter is uploaded by streaming_pipeline after this subprocess
            # returns, so use an ordinary local copy here instead of the
            # versioned-publish safety gate.
            import shutil as _publish_shutil
            _target = Path(_nas_output_dir)
            if (_target.exists() and any(_target.iterdir())
                    and not args.overwrite):
                print(f"  Refusing non-empty local publish target: {_target}")
                _published = False
            else:
                _target.mkdir(parents=True, exist_ok=True)
                _publish_shutil.copytree(output_dir, _target,
                                         dirs_exist_ok=True)
                _published = True
        if _published:
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
