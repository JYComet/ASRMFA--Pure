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
# UNC → Linux 路径翻译
# ═══════════════════════════════════════════════════════════════

_WIN_UNC_MAP: dict[str, str] = {}


def _detect_smb_mounts() -> dict[str, str]:
    """Parse /proc/mounts for CIFS mounts → UNC→linux mapping."""
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
    """Convert Windows UNC → Linux mount path."""
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
    """Translate UNC + resolve relative → absolute Path."""
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

    Checks (in order): explicit config path → ``mfa`` on PATH →
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
    """单次 os.scandir → O(1) 文件名查找。

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
    """Find {stem}.wav — flat → nested → zero-padded → glob fallback."""
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
    """Best-effort: hard-link → symlink → copy."""
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
    """Sync local → NAS with 3 retries + exponential backoff."""
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
SILENCE_LABELS: set[str] = {"<eps>", "<sil>", "sil", "<sp0>", "<sp1>", "<sp2>", "<sp3>"}

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

# ── IPA → pinyin mapping tables ─────────────────────────────────
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
