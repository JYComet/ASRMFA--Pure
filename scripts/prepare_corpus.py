#!/usr/bin/env python3
"""
Prepare MFA corpus for Chinese forced alignment.

Scans data_dir for wav and txt files matched within each subdirectory.
Keeps original Chinese text (MFA handles punctuation natively).

Input:  data_dir/  (wav + txt in subdirectories)
Output: corpus_clean/wav/, corpus_clean/txt/, corpus_clean/corpus_report.json
"""

import argparse
import json
import re
import shutil
from pathlib import Path

try:
    from pypinyin import lazy_pinyin, Style
except ModuleNotFoundError:
    raise SystemExit("pypinyin is not installed. Run: pip install pypinyin")

SUFFIX_PATTERN = re.compile(r"_(firered|qwen3|qwen3-api)$")

# 管线支持的标点白名单
ALLOWED_PUNCT = set("，。！？、；：…,.!?;:")


def clean_unsupported_punct(text: str) -> str:
    """过滤白名单外的非 CJK 标点 (如 」『』【】等), 保留 CJK/英文/数字/标签."""
    result: list[str] = []
    for ch in text:
        if ch in ALLOWED_PUNCT:
            result.append(ch)
        elif '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            result.append(ch)
        elif ch.isalpha() or ch.isdigit() or ch == '-':
            result.append(ch)
        elif ch.isspace():
            result.append(ch)
        elif ch in '<|>[]':
            result.append(ch)
    return ''.join(result)


# ─── NVV 标签 → MFA 大写 token 映射 (与 ctc_prealign.py 共享逻辑) ───

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
    "Dissatisfaction-hnn": "DISSATISFACTION-HNN", "Pause": "PAUSE",
}

_NVV_RE = re.compile(r'\[([A-Za-z][^\]]*?)\]')

def _nvv_replace(m: re.Match) -> str:
    inner = m.group(1)
    return NVV_TO_MFA.get(inner, inner.upper().replace(" ", "-"))


def text_to_pinyin(text: str, keep_punctuation: bool = True) -> str:
    """Convert Chinese text to pinyin with tone numbers.

    Pre-processes [NVV] labels → UPPERCASE tokens for MFA dictionary matching.
    """
    # 预处理: [Question-yi] → QUESTION-YI (避免被 pypinyin 拆散)
    text = _NVV_RE.sub(_nvv_replace, text)
    parts = []
    buf = ""
    for ch in text:
        if '一' <= ch <= '鿿' or ch.isalpha() or ch.isdigit():
            buf += ch
        elif ch.isspace():
            if buf.strip():
                parts.append(" ".join(lazy_pinyin(buf.strip(), style=Style.TONE3,
                                                   neutral_tone_with_five=True, errors="default")))
                buf = ""
            if keep_punctuation:
                parts.append(ch)
        else:
            if buf.strip():
                parts.append(" ".join(lazy_pinyin(buf.strip(), style=Style.TONE3,
                                                   neutral_tone_with_five=True, errors="default")))
                buf = ""
            if keep_punctuation:
                parts.append(ch)
    if buf.strip():
        parts.append(" ".join(lazy_pinyin(buf.strip(), style=Style.TONE3,
                                           neutral_tone_with_five=True, errors="default")))
    return " ".join(parts)

# Resolve project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def match_by_directory(data_dir: Path, txt_suffix: str | None = None) -> list[dict]:
    """Match wav and txt files within each subdirectory's txt/ folder."""
    matches = []
    txt_dirs = [p for p in data_dir.rglob("txt") if p.is_dir()]

    for txt_dir in sorted(txt_dirs):
        sub_dir = txt_dir.parent
        wav_files = sorted(sub_dir.glob("*.wav"))
        if not wav_files:
            continue
        txt_files = sorted(txt_dir.glob("*.txt"))
        if not txt_files:
            continue

        txt_map: dict[str, list[tuple[str, Path]]] = {}
        for tp in txt_files:
            stem = tp.stem
            m = SUFFIX_PATTERN.search(stem)
            if m:
                base = stem[:m.start()]
                sfx = m.group(1)
                if txt_suffix and sfx != txt_suffix:
                    continue
            else:
                base = stem
                sfx = ""
                if txt_suffix:
                    continue
            txt_map.setdefault(base, []).append((sfx, tp))

        for wp in wav_files:
            wav_stem = wp.stem
            if wav_stem in txt_map:
                for sfx, tp in txt_map[wav_stem]:
                    matches.append(dict(wav_stem=wav_stem, wav_path=str(wp),
                                        txt_path=str(tp), suffix=sfx,
                                        sub_dir=str(sub_dir)))
                continue
            for base_stem, txt_list in txt_map.items():
                if wav_stem.startswith(base_stem) or base_stem.startswith(wav_stem):
                    for sfx, tp in txt_list:
                        matches.append(dict(wav_stem=wav_stem, wav_path=str(wp),
                                            txt_path=str(tp), suffix=sfx,
                                            sub_dir=str(sub_dir)))
                    break
            else:
                print(f"  Warning: wav '{wav_stem}' in {sub_dir.name} has no matching txt")
    return matches


def prepare_corpus(data_dir: Path, corpus_dir: Path, overwrite: bool = False,
                   keep_punctuation: bool = True, copy_wav: bool = True,
                   txt_suffix: str | None = None) -> dict[str, object]:
    if copy_wav:
        wav_dir = corpus_dir / "wav"
        txt_dir = corpus_dir / "txt"
        wav_dir.mkdir(parents=True, exist_ok=True)
    else:
        wav_dir = None
        txt_dir = corpus_dir  # write pinyin directly to output dir
    txt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} for matched wav/txt pairs...")
    matches = match_by_directory(data_dir, txt_suffix=txt_suffix)
    print(f"  Found {len(matches)} wav-txt pairs across subdirectories.")

    written = 0
    skipped = 0
    entries = []

    for m in matches:
        suffix = m["suffix"]
        if suffix and not copy_wav:
            # No suffix needed when not copying WAVs — txt matches original WAV name
            out_stem = m["wav_stem"]
        else:
            out_stem = f"{m['wav_stem']}_{suffix}" if suffix else m["wav_stem"]
        out_txt = txt_dir / f"{out_stem}.txt"

        if out_txt.exists() and not overwrite:
            skipped += 1
            continue

        raw_text = Path(m["txt_path"]).read_text(encoding="utf-8").strip()
        raw_text = clean_unsupported_punct(raw_text)
        if not raw_text:
            skipped += 1
            continue

        # Convert to pinyin with tone numbers
        pinyin_text = text_to_pinyin(raw_text, keep_punctuation=keep_punctuation)
        out_txt.write_text(pinyin_text + "\n", encoding="utf-8")

        if copy_wav:
            out_wav = wav_dir / f"{out_stem}.wav"
            wav_src = Path(m["wav_path"])
            if not out_wav.exists() or overwrite:
                shutil.copy2(wav_src, out_wav)

        written += 1
        entries.append(dict(out_stem=out_stem, wav_stem=m["wav_stem"],
                            txt_source=m["txt_path"], suffix=suffix,
                            raw_text=raw_text[:100]))

    report = dict(matched_pairs=len(matches), written=written,
                  skipped=skipped, entries=entries)
    report_path = corpus_dir / "corpus_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Done. written={written}, skipped={skipped}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare Chinese MFA corpus.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data_dir")
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "corpus_clean")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-punct", action="store_true",
                        help="Strip punctuation instead of keeping it.")
    parser.add_argument("--no-copy-wav", action="store_true",
                        help="Don't copy WAV files (use when trim already wrote them).")
    parser.add_argument("--txt-suffix", type=str, default=None,
                        help="Only match txt files with this suffix (e.g. qwen3-api).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if args.dry_run:
        matches = match_by_directory(args.data_dir, txt_suffix=args.txt_suffix)
        print(f"Matched pairs: {len(matches)}")
        for m in matches[:20]:
            print(f"  [{Path(m['sub_dir']).name}] {m['wav_stem']} <- {Path(m['txt_path']).name}")
        if len(matches) > 20:
            print(f"  ... and {len(matches) - 20} more")
        return

    prepare_corpus(data_dir=args.data_dir, corpus_dir=args.corpus_dir,
                   overwrite=args.overwrite, keep_punctuation=not args.no_punct,
                   copy_wav=not args.no_copy_wav, txt_suffix=args.txt_suffix)


if __name__ == "__main__":
    main()
