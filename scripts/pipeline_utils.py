#!/usr/bin/env python3
"""
共享工具 — 路径翻译、文件发现、MFA 环境。

被 run_pipeline.py 和 streaming_pipeline.py 共同导入。
"""

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


def get_mfa_env(mfa_python: Path, models_dir: Path) -> dict[str, str]:
    """Build environment dict for MFA subprocess calls."""
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
    return index


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


# ═══════════════════════════════════════════════════════════════
# Shared constants — canonical definitions used across the pipeline.
# Edit HERE when adding/changing NVV names, IPA mappings, etc.
# ═══════════════════════════════════════════════════════════════

import re as _re

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
    return '一' <= ch <= '鿿'


def is_nvv_token(token: str) -> bool:
    """Check if *token* is an NVV label (BREATHING, QUESTION-YI, etc.)."""
    return token.strip().strip('<>').upper() in NVV_NAMES


def is_english_token(token: str) -> bool:
    """Token is English alpha: not NVV, not CJK, not pinyin syllable with tone."""
    if not token or not token.isalpha():
        return False
    if not token.isascii():
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
    return is_cjk(s) or s[0].isalpha() or s.isdigit() or is_nvv_token(s)


def is_punct(s: str) -> bool:
    """True if *s* is a non-word token (punctuation / symbol)."""
    return bool(s.strip()) and not is_word_like(s)


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
