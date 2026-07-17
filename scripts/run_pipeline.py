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
import os
import platform
import subprocess
import sys
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
    find_mfa_python, get_mfa_env,
    build_ctc_presence, build_file_index, count_files_fast, find_wav,
    is_punct, is_word_like,
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
    "prepare": {"copy_wav": False, "keep_punctuation": True},
    "ctc_prealign": {
        "enabled": True,
        "model_path": "models/Multilingual-NVASR",
        "device": "cuda:0",
        "python": "",
        "limit": 0,
        "timeout": 3600,
    },
    "ctc_adjust": {"enabled": True, "limit": 0},
    "mfa": {
        "num_jobs": 0,               # 0 = auto (os.cpu_count())
        "single_speaker": True,
        "output_format": "long_textgrid",
        "clean": False,              # keep feature cache for faster re-runs
        "no_tokenization": True,
        "skip_validate": True,       # MFA align internally validates; standalone validate is redundant
    },
    "mfa_en": {
        "enabled": True,
        "num_jobs": 4,
        "padding_ms": 50,
        "min_segment_dur_ms": 200,
        "max_gap_merge_s": 0.35,
        "beam": 10,
        "retry_beam": 40,
        "acoustic_model": "pretrained_models/acoustic/english_us_arpa.zip",
        "dictionary": "dict/cmudict.dict",
        "g2p_model": "pretrained_models/g2p/english_us_arpa.zip",
    },
    "postprocess": {
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


def resolve_num_jobs(cfg_val: int) -> int:
    """Resolve *num_jobs* config value (0 = auto -> os.cpu_count())."""
    if cfg_val <= 0:
        import multiprocessing as mp
        return mp.cpu_count()
    return cfg_val


# ---------------------------------------------------------------------------
# MFA environment — imported from pipeline_utils (find_mfa_python, get_mfa_env)
# ---------------------------------------------------------------------------


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
            desc: str = "", timeout: int = 1800) -> int:
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

    # Use scandir for fast flat-listing (common case)
    wavs: list[Path] = []

    # Fast path: read stems from ctc_ready manifest (no directory scan)
    manifest_path = ctx.get("ctc_pretg", Path()) / "ctc_ready_manifest.json"
    if manifest_path.exists():
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
        "--device", pc.get("device", "cuda:0"),
        "--dict-path", str(ctx["mfa_dict"]),
    ]
    if pc.get("nvv_bias", 0) > 0:
        prealign_args += ["--nvv-bias", str(pc["nvv_bias"])]
    if pc.get("limit", 0) > 0:
        prealign_args += ["--limit", str(pc["limit"])]
    if args.overwrite:
        prealign_args.append("--overwrite")

    # Use run_python with the NVASR Python, not mfa_python
    return run_python(SCRIPTS_DIR / "ctc_prealign.py", prealign_args, nvras_py_path,
                      ctx["models_dir"], "Step 4: CTC Pre-alignment (NVASR -> MFA anchors)",
                      timeout=pc.get("timeout", 3600))


def step_normalize_punct(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Normalize punctuation in CTC output text and sync with punct.json anchors.

    1. ASCII -> CJK equivalents (existing)
    2. Non-whitelist punctuation -> ，(fullwidth comma)
    3. Merge adjacent punctuation — no two puncts side by side;
       timestamps in _punct.json are merged to span the combined range.
    """
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
        # char_info[i] = ("punct"|"other", is_allowed | None, char)
        char_info: list[tuple[str, bool | None, str]] = []
        for ch in text:
            if is_punct(ch):
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
    """Normalize Arabic numerals to Chinese in CTC output text and .lab files."""
    try:
        import cn2an
    except ImportError:
        print("  cn2an not installed, skipping numeral normalization.")
        return 0
    ctc_dir = ctx["ctc_pretg"]
    count = 0
    missing = 0
    for txt_file in sorted(ctc_dir.glob("*_text_cn.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            missing += 1
            print(f"  WARNING: Skipping {txt_file.name} — file missing (symlink target gone?)")
            continue
        normalized = cn2an.transform(text, "an2cn")
        if normalized != text:
            txt_file.write_text(normalized + "\n", encoding="utf-8")
            lab_file = ctc_dir / txt_file.name.replace("_text_cn.txt", ".lab")
            if lab_file.exists():
                try:
                    lab_text = lab_file.read_text(encoding="utf-8").strip()
                    lab_file.write_text(cn2an.transform(lab_text, "an2cn") + "\n", encoding="utf-8")
                except FileNotFoundError:
                    pass  # lab file disappeared, non-critical
            count += 1
    if missing:
        print(f"  WARNING: {missing} _text_cn.txt file(s) not found, skipped")
    print(f"  Normalized numerals in {count} files")
    return 0


def step_normalize_ria(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Fix ria pinyin fragments in .lab + merge CTC anchors in _tokens.jsonl.

    Safety net for old CTC output.  New data is handled inline by
    ctc_prealign (align_text gets CJK→ria before tokenizer).
    Does NOT modify _text_cn.txt / _text_raw.txt (ASR archive).
    """
    import json, re

    ctc_dir = ctx["ctc_pretg"]
    if not ctc_dir or not ctc_dir.exists():
        return 0

    lab_changed = 0
    tokens_changed = 0

    for lab_file in sorted(ctc_dir.rglob("*.lab")):
        try:
            lab_text = lab_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue

        new_lab = re.sub(r'rui[0-5]\s+ya[0-5]', 'ria', lab_text)
        new_lab = re.sub(r'rui[0-5]\s+a[0-5]', 'ria', new_lab)

        if new_lab == lab_text:
            continue

        lab_file.write_text(new_lab + "\n", encoding="utf-8")
        lab_changed += 1

        # Merge tokens.jsonl: ruiN + yaN → ria
        tokens_path = lab_file.with_name(lab_file.stem + "_tokens.jsonl")
        if not tokens_path.exists():
            tokens_path = lab_file.with_suffix(".jsonl")
        if tokens_path.exists():
            try:
                entries = [json.loads(l) for l in
                           tokens_path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
                new_entries, i, changed = [], 0, False
                while i < len(entries):
                    w = entries[i]["word"]
                    if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
                            and re.match(r'^ya[0-5]$', entries[i + 1]["word"])):
                        a, b = entries[i], entries[i + 1]
                        new_entries.append({
                            "word": "ria", "start_ms": a["start_ms"], "end_ms": b["end_ms"],
                            "start_s": a["start_s"], "end_s": b["end_s"],
                            "type": a.get("type", "word")})
                        i += 2; changed = True
                    else:
                        new_entries.append(entries[i]); i += 1
                if changed:
                    tokens_path.write_text(
                        "\n".join(json.dumps(e, ensure_ascii=False) for e in new_entries) + "\n",
                        encoding="utf-8")
                    tokens_changed += 1
            except Exception:
                pass

    if lab_changed:
        print(f"  [normalize_ria] {lab_changed} .lab + {tokens_changed} tokens.jsonl (safety net)")
    return 0


def step_normalize_en(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Normalise English-word phonetic fragments in .lab and _tokens.jsonl.

    NVASR tokenizer breaks OOV English words into pinyin approximations
    (e.g. "ria"->"rui4"+"ya4").  This step merges them back into the
    canonical spelling before MFA alignment.
    """
    ctc_dir = ctx["ctc_pretg"]
    if not ctc_dir or not ctc_dir.exists():
        return 0

    script = SCRIPTS_DIR / "normalize_english_tokens.py"
    norm_en_args = ["--txt-dir", str(ctc_dir)]
    if ctx.get("mfa_dict"):
        norm_en_args += ["--dict-path", str(ctx["mfa_dict"])]
    return run_python(
        script, norm_en_args,
        mfa_python, ctx["models_dir"],
        "Step 2b: Normalise English tokens")


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


def step_mfa_align(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """MFA align — uses NVASR .lab as corpus + CTC TextGrid as anchors.

    NVASR produces both the transcript (.lab) and the word boundaries (TextGrid)
    from the same ASR text.  This guarantees 100% word matching between corpus and
    anchors, so MFA uses every CTC word boundary for phone-level refinement.
    """
    mc = cfg["mfa"]
    ctc_dir = ctx.get("ctc_pretg_adj", ctx["ctc_pretg"])  # use adjusted if available

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
    if mc.get("fine_tune", True):
        mfa_args.append("--fine_tune")
    fine_tune_tolerance = mc.get("fine_tune_boundary_tolerance", 0.1)
    if fine_tune_tolerance is not None and fine_tune_tolerance > 0:
        mfa_args += ["--fine_tune_boundary_tolerance", str(fine_tune_tolerance)]
    return run_mfa(mfa_args, mfa_python, ctx["models_dir"],
                   "Step 6: MFA Align" + (" (NVASR corpus + CTC anchors)" if use_nvasr_corpus and use_anchors else ""))


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
    audio_dir = ctx["mfa_audio_dir"]
    output_dir = ctx["workspace"] / "en_phones"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve English model paths
    en_acoustic = en_cfg.get("acoustic_model", str(PROJECT_ROOT / "pretrained_models" / "acoustic" / "english_us_arpa.zip"))
    en_dict = en_cfg.get("dictionary", str(PROJECT_ROOT / "dict" / "cmudict.dict"))
    en_g2p = en_cfg.get("g2p_model", str(PROJECT_ROOT / "pretrained_models" / "g2p" / "english_us_arpa.zip"))

    # Resolve relative paths
    for val, key in [(en_acoustic, "acoustic_model"), (en_dict, "dictionary"), (en_g2p, "g2p_model")]:
        p = Path(val)
        if not p.is_absolute():
            resolved = PROJECT_ROOT / val
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
    ]
    if args.python:
        align_en_args += ["--python", str(mfa_python)]

    script = SCRIPTS_DIR / "align_english_mfa.py"
    return run_python(script, align_en_args, mfa_python, ctx["models_dir"],
                      desc="English MFA Alignment")


def step_postprocess(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Post-process MFA aligned TextGrids.

    NVV and punctuation are self-referential in the MFA dictionary,
    so MFA preserves them natively — no post-injection needed.
    """
    pc = cfg["postprocess"]
    ctc_dir = ctx.get("ctc_pretg_adj", ctx["ctc_pretg"])  # use adjusted if available
    aligned_dir = ctx["aligned_dir"]

    pp_args = [
        "--txt-dir", str(ctc_dir),
        "--textgrid-dir", str(aligned_dir),
        "--output-dir", str(ctx["output_dir"]),
        "--filtered-dir", str(ctx["filtered_dir"]),
        "--wav-dir", str(ctx["mfa_audio_dir"]),
        "--raw-text-dir", str(ctc_dir),  # adjusted dir has _text_cn.txt too
        "--pinyin-dict", str(resolve_path(PROJECT_ROOT, cfg.get("pinyin_dict", "dict/fullpinyin_enword.dict"))),
        "--ipa-dict", str(resolve_path(PROJECT_ROOT, cfg.get("mfa_dict", "dict/mfa_ipa.dict"))),
        "--en-phones-dir", str(ctx["workspace"] / "en_phones"),
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
    # Text correction & unexpected silence handling
    if not pc.get("enable_text_correction", True):
        pp_args.append("--no-enable-text-correction")
    if not pc.get("handle_unexpected_sil", True):
        pp_args.append("--no-handle-unexpected-sil")
    if pc.get("workers", 0) > 0:
        pp_args += ["--workers", str(pc["workers"])]
    if args.overwrite:
        pp_args.append("--overwrite")
    return run_python(SCRIPTS_DIR / "postprocess_textgrids.py", pp_args, mfa_python,
                      ctx["models_dir"], "Step 7: Post-processing")


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


def step_link_ctc(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Validate pre-existing NVASR CTC output and prepare workspace.

    Scans the CTC directory for ``.lab`` files (single-level, no recursion),
    matches audio by stem, validates all 6 NVASR output files per stem, then
    hard-links audio + CTC files into the workspace so the pipeline can
    proceed from ``resample`` onward.
    """
    import json as _json

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


# ---------------------------------------------------------------------------
# Step registry — must come after all step functions are defined
# ---------------------------------------------------------------------------

STEPS = {
    "link": ("Link pre-existing CTC output (ctc_ready mode)", step_link_ctc),
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
}

FULL_STEP_ORDER = list(STEPS.keys())
CTC_READY_STEP_ORDER = ["link", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess"]
NVASR_FALLBACK_STEP_ORDER = ["prealign", "normalize_punct", "normalize", "normalize_ria", "normalize_en", "resample", "adjust", "align", "align_en", "postprocess"]


def main():
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
    parser.add_argument("--validate", action="store_true",
                        help="Validate output structure after each step (uses output_spec in config).")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["full", "ctc_ready", "batch_ctc_ready", "nvrasr_fallback"],
                        help="Pipeline mode (default: from config, or 'full').")
    parser.add_argument("--ctc-ready", type=str, default=None, metavar="CTC_DIR",
                        help="Enable ctc_ready mode: path to pre-existing NVASR CTC output.")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use pre-built scan cache (default: enabled, controlled by config 'use_cache').")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-scan, ignore cache and config setting.")
    parser.add_argument("--scan-only", action="store_true",
                        help="Pre-scan only: discover + validate + write cache, then exit.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom cache directory (default: <project>/cache/).")
    args = parser.parse_args()

    if args.list_steps:
        for name, (desc, _) in STEPS.items():
            print(f"  {name:12s} - {desc}")
        return

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

        # Optional limit for testing
        limit = bc.get("limit", 0)
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
    if args.workspace:
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

    # In ctc_ready mode, audio_dir points to the source data_dir (already trimmed)
    # to avoid copying 100k+ files across SMB mounts
    # Resample reads from here and writes 16k audio locally
    if mode in ("ctc_ready", "nvrasr_fallback"):
        audio_dir = data_dir  # use source audio in-place, no copy
    else:
        audio_dir = workspace / cfg.get("audio_dir", "audio")
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
    filtered_dir = workspace / cfg.get("filtered_dir", "filtered")
    validate_dir = workspace / cfg.get("validate_dir", "validate")
    temp_dir = workspace / cfg.get("temp_dir", "temp")

    # Check models (already resolved above)
    if not mfa_dict.exists():
        print(f"ERROR: MFA dictionary not found at {mfa_dict}")
        sys.exit(1)

    # Resolve steps — order depends on pipeline mode
    if mode == "ctc_ready":
        step_order = CTC_READY_STEP_ORDER
    elif mode == "nvrasr_fallback":
        step_order = NVASR_FALLBACK_STEP_ORDER
    else:
        step_order = FULL_STEP_ORDER

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
        return

    run_list = [s for s in run_list if not getattr(args, f"skip_{s}", False)]

    if not run_list:
        print("No steps to run.")
        return

    # Only create dirs needed by the steps being run
    _ctc_pretg_dir = workspace / cfg.get("ctc_pretg", "ctc_pretg")
    _ctc_pretg_adj_dir = workspace / cfg.get("ctc_pretg_adj", "ctc_pretg_adj")
    step_dirs = {
        "link": [audio_dir, _ctc_pretg_dir],
        "trim": [audio_dir, temp_dir],
        "resample": [temp_dir],
        "prealign": [_ctc_pretg_dir],
        "adjust": [_ctc_pretg_adj_dir],
        "validate": [validate_dir, temp_dir, _ctc_pretg_dir],
        "align": [aligned_dir, temp_dir, _ctc_pretg_dir],
        "align_en": [workspace / "en_phones", temp_dir],
        "postprocess": [output_dir, filtered_dir],
    }
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
        "mfa_audio_dir": workspace / "audio_16k",
        "ctc_pretg": workspace / cfg.get("ctc_pretg", "ctc_pretg"),
        "ctc_pretg_adj": workspace / cfg.get("ctc_pretg_adj", "ctc_pretg_adj"),
    }

    failed = []
    for step_name in run_list:
        desc, func = STEPS[step_name]
        print(f"\n  >>> [{step_name}] {desc}")
        rc = func(args, cfg, mfa_python, ctx)
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
    if mode in ("ctc_ready", "full") and not failed and _config_mode != "batch_ctc_ready":
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

    print(f"\n{'#'*60}")
    print(f"  {'FAILED' if failed else 'DONE'}: {', '.join(failed) if failed else 'Success'}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
