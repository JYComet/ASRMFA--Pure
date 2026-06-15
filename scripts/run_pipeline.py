#!/usr/bin/env python3
"""
Complete Chinese MFA forced alignment pipeline.

Steps: trim -> prepare -> resample -> validate -> align -> postprocess

Usage:
  python scripts/run_pipeline.py                              # all steps, use config.yaml
  python scripts/run_pipeline.py --data-dir E:/path/to/data   # override input
  python scripts/run_pipeline.py --step align                 # single step
  python scripts/run_pipeline.py --skip-to align              # from align onward
  python scripts/run_pipeline.py --config my_config.yaml      # custom config file
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    print("ERROR: pyyaml is required. Run: pip install pyyaml")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, value: str | None) -> Path | None:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


# ---------------------------------------------------------------------------
# MFA environment
# ---------------------------------------------------------------------------

def find_mfa_python(cfg_python: str = "") -> Path | None:
    import shutil as _shutil
    if cfg_python:
        p = Path(cfg_python)
        if p.exists():
            return p
    candidates = [
        Path(os.path.expandvars(r"%USERPROFILE%\miniconda3\envs\mfa_mandarin\python.exe")),
        Path(os.path.expandvars(r"%USERPROFILE%\anaconda3\envs\mfa_mandarin\python.exe")),
        Path(os.path.expandvars(r"%USERPROFILE%\miniconda3\envs\mfa_chinese\python.exe")),
    ]
    mfa_on_path = _shutil.which("mfa")
    if mfa_on_path:
        candidates.insert(0, Path(mfa_on_path).parent / "python.exe")
    for c in candidates:
        if c.exists():
            return c
    return None


def get_mfa_env(mfa_python: Path, models_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(models_dir)
    lib_bin = mfa_python.parent / "Library" / "bin"
    paths = [str(mfa_python.parent)]
    if lib_bin.exists():
        paths.append(str(lib_bin))
    if "PATH" in env:
        paths.append(env["PATH"])
    env["PATH"] = os.pathsep.join(paths)
    return env


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_python(script: Path, script_args: list[str], mfa_python: Path,
               models_dir: Path, desc: str = "") -> int:
    cmd = [str(mfa_python), str(script)] + script_args
    print(f"\n{'='*60}\n  {desc or script.name}\n  {' '.join(cmd)}\n{'='*60}\n")
    return subprocess.run(cmd, env=get_mfa_env(mfa_python, models_dir)).returncode


def run_mfa(mfa_args: list[str], mfa_python: Path, models_dir: Path, desc: str = "") -> int:
    print(f"\n{'='*60}\n  {desc or 'MFA: ' + ' '.join(mfa_args)}\n{'='*60}\n")
    return subprocess.run(
        [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa"] + mfa_args,
        env=get_mfa_env(mfa_python, models_dir),
    ).returncode


# ---------------------------------------------------------------------------
# Pipeline steps — all take (args, cfg, mfa_python, ctx)
# ctx = {data_dir, audio_dir, pinyin_dir, mfa_audio_dir, aligned_dir,
#        output_dir, filtered_dir, validate_dir, models_dir, temp_dir, mfa_dict}
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


def step_prepare_corpus(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    pc = cfg["prepare"]
    txt_suffix = cfg.get("txt_suffix", "")
    prep_args = ["--data-dir", str(ctx["data_dir"]), "--corpus-dir", str(ctx["pinyin_dir"])]
    if not pc.get("copy_wav", False):
        prep_args.append("--no-copy-wav")
    if txt_suffix:
        prep_args += ["--txt-suffix", txt_suffix]
    if args.overwrite:
        prep_args.append("--overwrite")
    if not pc.get("keep_punctuation", True):
        prep_args.append("--no-punct")
    return run_python(SCRIPTS_DIR / "prepare_corpus.py", prep_args, mfa_python,
                      ctx["models_dir"],
                      "Step 2: Corpus Preparation")


def step_resample_for_mfa(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Resample trimmed audio to 16kHz for MFA (keeps original sample rate intact)."""
    import shutil
    import numpy as np
    import soundfile as sf
    from scipy.signal import decimate, resample_poly

    audio_dir = ctx["audio_dir"]
    mfa_audio_dir = ctx["mfa_audio_dir"]
    target_sr = 16000

    wavs = list(audio_dir.rglob("*.wav"))
    if not wavs:
        print("  No WAVs found in audio dir.")
        return 1

    # Check if already done
    existing = list(mfa_audio_dir.rglob("*.wav")) if mfa_audio_dir.exists() else []
    if len(existing) >= len(wavs) and not args.overwrite:
        print(f"  {len(existing)} resampled WAVs already exist. Use --overwrite to redo.")
        return 0

    mfa_audio_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for wav in wavs:
        rel = wav.relative_to(audio_dir)
        out = mfa_audio_dir / rel
        if out.exists() and not args.overwrite:
            done += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(str(wav))
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr == target_sr:
            shutil.copy2(str(wav), str(out))
        elif sr % target_sr == 0:
            audio = decimate(audio.astype('float64'), sr // target_sr, ftype='iir').astype('float32')
            sf.write(str(out), audio, target_sr)
        else:
            gcd = np.gcd(sr, target_sr)
            audio = resample_poly(audio.astype('float64'), target_sr // gcd, sr // gcd).astype('float32')
            sf.write(str(out), audio, target_sr)
        done += 1
    print(f"  Resampled {done} WAVs to {target_sr}Hz → {mfa_audio_dir}")
    return 0


def step_mfa_validate(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    mc = cfg["mfa"]
    if not list(ctx["pinyin_dir"].glob("*.txt")):
        print("ERROR: No txt files in pinyin dir.")
        return 1
    mfa_args = [
        "validate", str(ctx["pinyin_dir"]), str(ctx["mfa_dict"]),
        "--acoustic_model_path", cfg["acoustic_model"],
        "--audio_directory", str(ctx["mfa_audio_dir"]),
        "--temporary_directory", str(ctx["temp_dir"]),
        "--num_jobs", str(mc["num_jobs"]),
        "--overwrite",
    ]
    if mc.get("clean"):
        mfa_args.append("--clean")
    if mc.get("single_speaker"):
        mfa_args.append("--single_speaker")
    return run_mfa(mfa_args, mfa_python, ctx["models_dir"], "Step 4: MFA Validate")


def step_mfa_align(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    mc = cfg["mfa"]
    if not list(ctx["pinyin_dir"].glob("*.txt")):
        print("ERROR: No txt files in pinyin dir.")
        return 1
    mfa_args = [
        "align", str(ctx["pinyin_dir"]), str(ctx["mfa_dict"]),
        cfg["acoustic_model"], str(ctx["aligned_dir"]),
        "--audio_directory", str(ctx["mfa_audio_dir"]),
        "--temporary_directory", str(ctx["temp_dir"]),
        "--output_format", mc.get("output_format", "long_textgrid"),
        "--num_jobs", str(mc["num_jobs"]),
        "--overwrite", "--no_textgrid_cleanup",
    ]
    if mc.get("clean"):
        mfa_args.append("--clean")
    if mc.get("single_speaker"):
        mfa_args.append("--single_speaker")
    if mc.get("no_tokenization"):
        mfa_args.append("--no_tokenization")
    return run_mfa(mfa_args, mfa_python, ctx["models_dir"], "Step 5: MFA Align")


def step_postprocess(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    pc = cfg["postprocess"]
    pp_args = [
        "--txt-dir", str(ctx["pinyin_dir"]),
        "--textgrid-dir", str(ctx["aligned_dir"]),
        "--output-dir", str(ctx["output_dir"]),
        "--filtered-dir", str(ctx["filtered_dir"]),
        "--wav-dir", str(ctx["audio_dir"]),   # 32kHz original for BGM/energy analysis
        "--raw-text-dir", str(ctx["data_dir"]),
        "--pinyin-dict", str(resolve_path(PROJECT_ROOT, cfg.get("pinyin_dict", "dict/fullpinyin_enword.dict"))),
        "--ipa-dict", str(resolve_path(PROJECT_ROOT, cfg.get("mfa_dict", "dict/mfa_ipa.dict"))),
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
        pp_args += ["--filter-short-phone-sec", str(pc.get("filter_short_phone_sec", 0.015))]
        pp_args += ["--filter-long-word-sec", str(pc.get("filter_long_word_sec", 1.0))]
        pp_args += ["--filter-min-word-sec", str(pc.get("filter_min_word_sec", 0.15))]
        pp_args += ["--filter-min-phone-coverage", str(pc.get("filter_min_phone_coverage", 0.35))]
        pp_args += ["--filter-edge-gap-sec", str(pc.get("filter_edge_gap_sec", 0.25))]
        pp_args += ["--filter-flank-silence-sec", str(pc.get("filter_flank_silence_sec", 0.4))]
        pp_args += ["--filter-long-consonant-sec", str(pc.get("filter_long_consonant_sec", 999.0))]
        pp_args += ["--filter-long-vowel-sec", str(pc.get("filter_long_vowel_sec", 999.0))]
    else:
        pp_args.append("--no-filter-suspicious")
    if args.overwrite:
        pp_args.append("--overwrite")
    return run_python(SCRIPTS_DIR / "postprocess_textgrids.py", pp_args, mfa_python,
                      ctx["models_dir"], "Step 6: Post-processing")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = {
    "trim": ("Audio preprocessing", step_trim_silence),
    "prepare": ("Corpus preparation", step_prepare_corpus),
    "resample": ("Resample to 16kHz for MFA", step_resample_for_mfa),
    "validate": ("MFA validate", step_mfa_validate),
    "align": ("MFA align", step_mfa_align),
    "postprocess": ("Post-processing", step_postprocess),
}

SCRIPTS_DIR = PROJECT_ROOT / "scripts"


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
    parser.add_argument("--python", type=str, default=None,
                        help="Override Python path from config.")
    args = parser.parse_args()

    if args.list_steps:
        for name, (desc, _) in STEPS.items():
            print(f"  {name:12s} - {desc}")
        return

    # Load config
    cfg = load_config(Path(args.config))
    print(f"Config: {args.config}")

    # Resolve workspace and paths
    workspace = Path(cfg.get("workspace", str(PROJECT_ROOT / "workspace")))
    workspace.mkdir(parents=True, exist_ok=True)

    # Input: relative to PROJECT_ROOT (or absolute / CLI override)
    data_dir = Path(args.data_dir) if args.data_dir else resolve_path(PROJECT_ROOT, cfg.get("data_dir", "data_dir"))

    # Outputs: relative to workspace
    audio_dir = workspace / cfg.get("audio_dir", "audio")
    pinyin_dir = workspace / cfg.get("pinyin_dir", "pinyin")
    aligned_dir = workspace / cfg.get("aligned_dir", "aligned")
    output_dir = Path(args.output_dir) if args.output_dir else workspace / cfg.get("output_dir", "output")
    filtered_dir = workspace / cfg.get("filtered_dir", "filtered")
    validate_dir = workspace / cfg.get("validate_dir", "validate")
    temp_dir = workspace / cfg.get("temp_dir", "temp")

    # Models & dicts: relative to PROJECT_ROOT
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

    # Check models
    if not mfa_dict.exists():
        print(f"ERROR: MFA dictionary not found at {mfa_dict}")
        sys.exit(1)

    # Resolve steps
    step_order = list(STEPS.keys())
    if args.step:
        if args.step not in STEPS:
            print(f"Unknown step: {args.step}")
            sys.exit(1)
        run_list = [args.step]
    elif args.skip_to:
        if args.skip_to not in STEPS:
            print(f"Unknown step: {args.skip_to}")
            sys.exit(1)
        run_list = step_order[step_order.index(args.skip_to):]
    else:
        run_list = list(step_order)
    run_list = [s for s in run_list if not getattr(args, f"skip_{s}", False)]

    if not run_list:
        print("No steps to run.")
        return

    for d in [audio_dir, pinyin_dir, aligned_dir, validate_dir, output_dir, filtered_dir, temp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Chinese MFA Pipeline")
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
        "mfa_audio_dir": temp_dir / "audio_16k",
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

    # Clean up temporary 16kHz audio
    mfa_audio = temp_dir / "audio_16k"
    if mfa_audio.exists():
        import shutil
        shutil.rmtree(str(mfa_audio))
        print(f"  Cleaned temp: {mfa_audio}")

    print(f"\n{'#'*60}")
    print(f"  {'FAILED' if failed else 'DONE'}: {', '.join(failed) if failed else 'Success'}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
