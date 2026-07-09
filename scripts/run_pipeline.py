#!/usr/bin/env python3
"""
Complete Chinese MFA forced alignment pipeline.

Steps: trim -> resample -> prealign -> normalize -> adjust -> validate -> align -> postprocess

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

    # Try config/env-sourced Python (mfa on PATH)
    mfa_on_path = _shutil.which("mfa")
    if mfa_on_path:
        parent = Path(mfa_on_path).parent
        py = parent / ("python.exe" if os.name == "nt" else "python3")
        if py.exists():
            return py

    # Search common conda envs
    home = Path.home()
    conda_roots = [
        home / "miniconda3",
        home / "anaconda3",
        home / "opt" / "miniconda3",
        home / "opt" / "anaconda3",
        Path("/opt/conda"),
        Path("/usr/local/anaconda3"),
    ]
    env_names = ["mfa_chinese", "mfa_mandarin", "mfa"]
    is_win = os.name == "nt"

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
               models_dir: Path, desc: str = "", timeout: int = 600) -> int:
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


def step_resample_for_mfa(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Resample trimmed audio to 16kHz for MFA."""
    import shutil
    import soundfile as sf
    sys.path.insert(0, str(SCRIPTS_DIR))
    from audio_utils import resample_audio

    audio_dir = ctx["audio_dir"]
    mfa_audio_dir = ctx["mfa_audio_dir"]
    target_sr = 16000

    wavs = list(audio_dir.rglob("*.wav"))
    if not wavs:
        print("  No WAVs found in audio dir.")
        return 1

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
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
        else:
            shutil.copy2(str(wav), str(out))
            done += 1
            continue
        sf.write(str(out), audio, target_sr)
        done += 1
    print(f"  Resampled {done} WAVs to {target_sr}Hz → {mfa_audio_dir}")
    return 0


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
    mfa_args = [
        "validate", str(corpus_dir), str(ctx["mfa_dict"]),
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
    return run_mfa(mfa_args, mfa_python, ctx["models_dir"], "Step 5: MFA Validate")


def step_prealign(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Run NVASR CTC forced alignment → produce MFA anchor TextGrids."""
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
                      ctx["models_dir"], "Step 4: CTC Pre-alignment (NVASR → MFA anchors)",
                      timeout=pc.get("timeout", 3600))


def step_normalize_text(args, cfg: dict, mfa_python: Path, ctx: dict) -> int:
    """Normalize Arabic numerals to Chinese in CTC output text and .lab files."""
    try:
        import cn2an
    except ImportError:
        print("  cn2an not installed, skipping numeral normalization.")
        return 0
    ctc_dir = ctx["ctc_pretg"]
    count = 0
    for txt_file in sorted(ctc_dir.glob("*_text_cn.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        normalized = cn2an.transform(text, "an2cn")
        if normalized != text:
            txt_file.write_text(normalized + "\n", encoding="utf-8")
            lab_file = ctc_dir / txt_file.name.replace("_text_cn.txt", ".lab")
            if lab_file.exists():
                lab_text = lab_file.read_text(encoding="utf-8").strip()
                lab_file.write_text(cn2an.transform(lab_text, "an2cn") + "\n", encoding="utf-8")
            count += 1
    print(f"  Normalized numerals in {count} files")
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

    # Clean temp dir when overwriting to avoid stale MFA cache
    import shutil
    if args.overwrite and ctx["temp_dir"].exists():
        shutil.rmtree(ctx["temp_dir"], ignore_errors=True)
        ctx["temp_dir"].mkdir(parents=True, exist_ok=True)

    if not list(corpus_dir.glob("*.lab" if use_nvasr_corpus else "*.txt")):
        print("ERROR: No corpus files found.")
        return 1

    # Check for CTC anchors
    use_anchors = ctc_dir.exists() and any(ctc_dir.glob("*.TextGrid"))

    if use_nvasr_corpus:
        print(f"  NVASR corpus: {ctc_dir} (.lab files from ASR text)")
    if use_anchors:
        print(f"  CTC anchors:  {ctc_dir}")
        print(f"  Transcript and anchors from SAME source → 100% word match")

    mfa_args = [
        "align", str(corpus_dir), str(ctx["mfa_dict"]),
        cfg["acoustic_model"], str(ctx["aligned_dir"]),
        "--audio_directory", str(ctx["mfa_audio_dir"]),
        "--temporary_directory", str(ctx["temp_dir"]),
        "--output_format", mc.get("output_format", "long_textgrid"),
        "--num_jobs", str(mc["num_jobs"]),
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
    return run_mfa(mfa_args, mfa_python, ctx["models_dir"],
                   "Step 6: MFA Align" + (" (NVASR corpus + CTC anchors)" if use_nvasr_corpus and use_anchors else ""))


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
        "--wav-dir", str(ctx["audio_dir"]),
        "--raw-text-dir", str(ctc_dir),  # adjusted dir has _text_cn.txt too
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
        if pc.get("filter_short_phone", True):
            pp_args += ["--filter-short-phone-sec", str(pc.get("filter_short_phone_sec", 0.015))]
        else:
            pp_args.append("--no-filter-short-phone")
        pp_args += ["--filter-long-word-sec", str(pc.get("filter_long_word_sec", 1.0))]
        pp_args += ["--filter-min-word-sec", str(pc.get("filter_min_word_sec", 0.15))]
        pp_args += ["--filter-min-word-dur-sec", str(pc.get("filter_min_word_dur_sec", 0.02))]
        pp_args += ["--filter-word-energy-ratio", str(pc.get("filter_word_energy_ratio", 2.0))]
        pp_args += ["--filter-min-phone-coverage", str(pc.get("filter_min_phone_coverage", 0.35))]
        pp_args += ["--filter-edge-gap-sec", str(pc.get("filter_edge_gap_sec", 0.25))]
        pp_args += ["--filter-flank-silence-sec", str(pc.get("filter_flank_silence_sec", 0.4))]
        pp_args += ["--filter-long-consonant-sec", str(pc.get("filter_long_consonant_sec", 999.0))]
        pp_args += ["--filter-long-vowel-sec", str(pc.get("filter_long_vowel_sec", 999.0))]
    else:
        pp_args.append("--no-filter-suspicious")
    # Text correction & unexpected silence handling
    if not pc.get("enable_text_correction", True):
        pp_args.append("--no-enable-text-correction")
    if not pc.get("handle_unexpected_sil", True):
        pp_args.append("--no-handle-unexpected-sil")
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

STEPS = {
    "trim": ("Audio preprocessing", step_trim_silence),
    "resample": ("Resample to 16kHz for MFA", step_resample_for_mfa),
    "prealign": ("CTC pre-alignment (NVASR → MFA anchors)", step_prealign),
    "normalize": ("Normalize numerals (Arabic → Chinese)", step_normalize_text),
    "adjust": ("Adjust CTC boundaries (energy-based)", step_adjust_ctc),
    "validate": ("MFA validate", step_mfa_validate),
    "align": ("MFA align (NVASR corpus + CTC anchors)", step_mfa_align),
    "postprocess": ("Post-processing (includes NVV brackets + sp1 normalization)", step_postprocess),
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
    parser.add_argument("--validate", action="store_true",
                        help="Validate output structure after each step (uses output_spec in config).")
    args = parser.parse_args()

    if args.list_steps:
        for name, (desc, _) in STEPS.items():
            print(f"  {name:12s} - {desc}")
        return

    # Load config
    cfg = load_config(Path(args.config))
    print(f"Config: {args.config}")

    # Resolve workspace and paths
    # All pipeline output goes under output/<workspace>/ — never pollutes
    # the project root or random external directories.
    output_root = PROJECT_ROOT / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    workspace_name = cfg.get("workspace", "default")
    workspace = output_root / workspace_name
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

    # Only create dirs needed by the steps being run
    _ctc_pretg_dir = workspace / cfg.get("ctc_pretg", "ctc_pretg")
    _ctc_pretg_adj_dir = workspace / cfg.get("ctc_pretg_adj", "ctc_pretg_adj")
    step_dirs = {
        "trim": [audio_dir, temp_dir],
        "resample": [temp_dir],
        "prealign": [_ctc_pretg_dir],
        "adjust": [_ctc_pretg_adj_dir],
        "validate": [validate_dir, temp_dir, _ctc_pretg_dir],
        "align": [aligned_dir, temp_dir, _ctc_pretg_dir],
        "postprocess": [output_dir, filtered_dir],
    }
    created: set[Path] = set()
    for s in run_list:
        for d in step_dirs.get(s, []):
            if d not in created:
                d.mkdir(parents=True, exist_ok=True)
                created.add(d)

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

    print(f"\n{'#'*60}")
    print(f"  {'FAILED' if failed else 'DONE'}: {', '.join(failed) if failed else 'Success'}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
