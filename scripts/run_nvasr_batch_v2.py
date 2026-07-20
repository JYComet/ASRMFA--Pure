#!/usr/bin/env python3
"""
批量 NVASR CTC 预对齐 V2 — 单次模型加载，按 speaker 分组处理

优化点 (对比 V1):
  1. NVASR 模型只加载一次 (而非每个叶子目录重新加载)
  2. 按 speaker 目录分组批量推理 (GPU 利用率最大化)
  3. 跳过已有输出 (断点续跑)
  4. 输出目录自动镜像输入结构: {stem}/{stem}.TextGrid ...

用法:
  # 测试 10 条
  python3 scripts/run_nvasr_batch_v2.py --limit 10

  # 全量 (跳过已完成)
  python3 scripts/run_nvasr_batch_v2.py

  # 全量 + 覆盖
  python3 scripts/run_nvasr_batch_v2.py --overwrite
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Path translation (UNC -> Linux mount) ──
from pipeline_utils import (
    NVV_NAMES, NVV_TO_MFA,
    is_nvv_token, is_english_token, is_pinyin_syllable, is_punct,
    RIA_VARIANTS, replace_ria_variants, normalize_punct_inline,
    _ASCII_TO_CJK_PUNCT,
)

# ── Constants from ctc_prealign.py ──
NVV_START, NVV_END = 25025, 25054
BLANK_ID = 0
ELLIPSIS_ID = 9724
FRAME_MS = 60
QUERY_FRAMES = 4
PAUSE_FRAMES_DEFAULT = 8
NVV_BIAS_DEFAULT = 4.0
ALLOWED_PUNCT = set("，。！？、；：…, .!?;:")
ALLOWED_PUNCT_CJK = "，。！？、；：…"


# ═══════════════════════════════════════════════════════════════
# Path translation
# ═══════════════════════════════════════════════════════════════

def _detect_smb_mounts() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if sys.platform == "win32":
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
            if "192.168.102.202" in clean:
                _, _, suffix = clean.partition("192.168.102.202")
                mapping[f"//RS3621{suffix}"] = mnt
                _ws = suffix.replace("/", "\\")
                mapping[f"\\\\RS3621{_ws}"] = mnt
    except Exception:
        pass
    return mapping


_SMB_MOUNTS = _detect_smb_mounts()


def translate_path(path_str: str) -> str:
    if not path_str or sys.platform == "win32":
        return path_str
    normalized = path_str.replace("\\", "/")
    for unc_raw, linux_mnt in sorted(_SMB_MOUNTS.items(), key=lambda x: -len(x[0])):
        unc_norm = unc_raw.replace("\\", "/")
        if normalized.startswith(unc_norm):
            rest = normalized[len(unc_norm):].lstrip("/")
            return f"{linux_mnt}/{rest}" if rest else linux_mnt
    return path_str


def resolve_path(raw: str) -> Path:
    if not raw:
        return PROJECT_ROOT
    p = Path(translate_path(raw))
    return p if p.is_absolute() else PROJECT_ROOT / p


# ═══════════════════════════════════════════════════════════════
# Core functions imported from ctc_prealign
# ═══════════════════════════════════════════════════════════════

# Import the monkeypatch factory directly
sys.path.insert(0, str(SCRIPTS_DIR))
from ctc_prealign import (
    make_patched_inference, write_textgrid, load_mfa_word_set,
    nvv_to_mfa, valid_mfa_word, chars_and_pinyin,
    clean_unsupported_punct, has_japanese,
    _normalize_punct, _normalize_numerals, _normalize_ria,
    _normalize_english, _vad_speech_end, ALLOWED_PUNCT as _AP,
)


# ═══════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════

def discover_speakers(input_dir: Path) -> list[Path]:
    """返回 input_dir 下的所有一级子目录 (speaker 目录)."""
    speakers = []
    try:
        with os.scandir(str(input_dir)) as it:
            for entry in sorted(it, key=lambda e: e.name):
                if entry.is_dir():
                    speakers.append(Path(entry.path))
    except OSError:
        pass
    return speakers


def count_wavs(speaker_dir: Path) -> int:
    """快速统计 speaker 目录下的 WAV 文件数."""
    n = 0
    for _ in speaker_dir.rglob("*.wav"):
        n += 1
    return n


def count_outputs(output_speaker_dir: Path) -> int:
    """统计已有输出文件数."""
    if not output_speaker_dir.exists():
        return 0
    n = 0
    for _ in output_speaker_dir.rglob("*.TextGrid"):
        n += 1
    return n


# ═══════════════════════════════════════════════════════════════
# Reorganize: flat output -> mirrored per-stem dirs
# ═══════════════════════════════════════════════════════════════

_CTC_SUFFIXES = [
    (".TextGrid", ""),
    (".lab", ""),
    ("_tokens.jsonl", ""),
    ("_punct.json", ""),
    ("_text_cn.txt", ""),
    ("_text_raw.txt", ""),
]


def reorganize_output(flat_dir: Path, wav_index: dict[str, tuple[Path, Path]],
                      output_root: Path, overwrite: bool = False) -> int:
    """将扁平输出目录中的文件按 stem 分组移动到镜像子文件夹.

    wav_index: {stem: (wav_path, relative_to_data_dir)}

    输出结构: output_root / {relative_to_data_dir} / {stem} / {stem}.TextGrid ...
    """
    # Group files by stem
    stem_files: dict[str, list[Path]] = {}
    for f in flat_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        # Determine stem
        stem = None
        for suffix, _ in _CTC_SUFFIXES:
            if name.endswith(suffix):
                stem = name[:-len(suffix)]
                break
        if stem is None:
            continue
        stem_files.setdefault(stem, []).append(f)

    copied = 0
    for stem, files in stem_files.items():
        if stem not in wav_index:
            # Unknown stem — copy flat under output root
            stem_dir = flat_dir / stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest = stem_dir / f.name
                if not dest.exists() or overwrite:
                    shutil.copy2(str(f), str(dest))
                    copied += 1
            continue

        wav_path, rel_parent = wav_index[stem]
        stem_dir = output_root / rel_parent / stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest = stem_dir / f.name
            if not dest.exists() or overwrite:
                shutil.copy2(str(f), str(dest))
                copied += 1

    return copied


# ═══════════════════════════════════════════════════════════════
# Single-speaker processing (model already loaded)
# ═══════════════════════════════════════════════════════════════

def process_speaker(speaker_dir: Path, output_speaker: Path,
                    model, orig_inference, mfa_words: set | None,
                    dict_path: Path | None, device: str, limit: int,
                    overwrite: bool, tmp_base: Path) -> tuple[int, int, float]:
    """处理一个 speaker 目录下的所有 WAV 文件.

    返回: (ok, fail, elapsed_seconds)
    """
    t0 = time.time()

    # ── Scan WAVs ──
    wav_files = sorted(speaker_dir.rglob("*.wav"))
    if limit > 0:
        wav_files = wav_files[:limit]

    if not wav_files:
        return 0, 0, 0.0

    speaker_name = speaker_dir.name
    total = len(wav_files)
    print(f"  [{speaker_name}] {total} WAVs", end="", flush=True)

    # ── Build reference text index ──
    ref_texts: dict[str, str] = {}
    for wav_path in wav_files:
        stem = wav_path.stem
        # Look for .txt with same name or suffixed
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            if not has_japanese(text):
                ref_texts[stem] = clean_unsupported_punct(text)

    # ── Patch inference with this speaker's ref texts ──
    patched = make_patched_inference(ref_texts, NVV_BIAS_DEFAULT)
    model.model.inference = patched.__get__(model.model, type(model.model))

    # ── Batch inference ──
    paths = [str(p) for p in wav_files]
    stems = [p.stem for p in wav_files]

    # Auto batch size
    try:
        mem_gb = torch.cuda.get_device_properties(device).total_mem / 1024**3
        BATCH = 64 if mem_gb >= 40 else 32 if mem_gb >= 24 else 16 if mem_gb >= 8 else 8
    except Exception:
        BATCH = 16

    all_results = []
    for bs in range(0, len(paths), BATCH):
        batch = paths[bs:bs + BATCH]
        batch_size_s = min(300, max(60, len(batch) * 30))
        res = model.generate(input=batch, language="zh", use_itn=True,
                             batch_size_s=batch_size_s)
        all_results.extend(res)

    infer_time = time.time() - t0
    speed = total / infer_time if infer_time > 0 else 0

    # ── Build wav_index for output reorganization ──
    wav_index: dict[str, tuple[Path, Path]] = {}
    for wav_path in wav_files:
        stem = wav_path.stem
        rel_parent = wav_path.parent.relative_to(speaker_dir)
        wav_index[stem] = (wav_path, rel_parent)

    # ── Generate output files ──
    tmp_dir = tmp_base / speaker_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for i, r in enumerate(all_results):
        stem = stems[i] if i < len(stems) else Path(r["key"]).stem
        words_aligned = r["words"]
        duration_s = r["duration_s"]

        if not r.get("english_complete", True):
            fail += 1
            continue

        try:
            # Token -> pinyin/NVV mapping (same as ctc_prealign.py)
            words_pinyin = []
            punct_entries = []
            for w in words_aligned:
                token_str = w["word"].strip()
                if not token_str:
                    continue
                token_clean = token_str.lstrip("▁")

                if token_clean.startswith("[") and token_clean.endswith("]"):
                    words_pinyin.append({"word": nvv_to_mfa(token_clean),
                                         "start": w["start"], "end": w["end"]})
                elif is_nvv_token(token_clean):
                    words_pinyin.append({"word": token_clean,
                                         "start": w["start"], "end": w["end"]})
                elif is_punct(token_clean):
                    punct_entries.append({"word": token_clean,
                                          "start": w["start"], "end": w["end"]})
                else:
                    # Alpha/English token — keep as-is
                    clean = token_clean.strip()
                    if clean and (clean.isalpha() or clean.isdigit()
                                  or any(c.isalpha() for c in clean)):
                        words_pinyin.append({"word": clean,
                                             "start": w["start"], "end": w["end"]})

            # ── Merge single ASCII letters ──
            if words_pinyin:
                merged = []
                i2 = 0
                while i2 < len(words_pinyin):
                    w2 = words_pinyin[i2]
                    t2 = w2["word"]
                    if (len(t2) == 1 and t2.isascii() and t2.isalpha()
                            and not is_nvv_token(t2)):
                        letters = [t2]
                        j = i2 + 1
                        while j < len(words_pinyin):
                            nt = words_pinyin[j]["word"]
                            if (len(nt) == 1 and nt.isascii() and nt.isalpha()
                                    and not is_nvv_token(nt)):
                                letters.append(nt)
                                j += 1
                            else:
                                break
                        merged.append({"word": "".join(letters),
                                       "start": w2["start"],
                                       "end": words_pinyin[j - 1]["end"]})
                        i2 = j
                    else:
                        merged.append(w2)
                        i2 += 1
                words_pinyin = merged

            # ── Dedup adjacent NVV ──
            if words_pinyin:
                deduped = []
                i3 = 0
                while i3 < len(words_pinyin):
                    w3 = words_pinyin[i3]
                    if is_nvv_token(w3["word"]):
                        j3 = i3 + 1
                        while j3 < len(words_pinyin) and words_pinyin[j3]["word"] == w3["word"]:
                            j3 += 1
                        if j3 > i3 + 1:
                            deduped.append({"word": w3["word"],
                                            "start": w3["start"],
                                            "end": words_pinyin[j3 - 1]["end"]})
                            i3 = j3
                        else:
                            deduped.append(w3)
                            i3 += 1
                    else:
                        deduped.append(w3)
                        i3 += 1
                words_pinyin = deduped

            # ── Write TextGrid ──
            blank_runs = r.get("blank_runs", [])
            pauses = []
            for s, e in blank_runs:
                dur_ms = (e - s) * 60
                if dur_ms >= 200:
                    pauses.append({"start_ms": s * 60, "end_ms": e * 60,
                                   "duration_ms": dur_ms})

            out_tg = tmp_dir / f"{stem}.TextGrid"
            write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

            # ── Write .lab ──
            out_lab = tmp_dir / f"{stem}.lab"
            lab_tokens = " ".join(w["word"] for w in words_pinyin)
            out_lab.write_text(lab_tokens + "\n", encoding="utf-8")

            # ── Write punct.json ──
            punct_path = tmp_dir / f"{stem}_punct.json"
            if punct_entries:
                punct_data = []
                for p in punct_entries:
                    punct_data.append({
                        "word": p["word"],
                        "start_ms": round(p["start"] * 1000, 1),
                        "end_ms": round(p["end"] * 1000, 1),
                        "start_s": p["start"],
                        "end_s": p["end"],
                    })
                punct_path.write_text(json.dumps(punct_data, ensure_ascii=False),
                                    encoding="utf-8")

            # ── Write text files ──
            text_asr = r.get("text_asr", "")
            text_asr = clean_unsupported_punct(text_asr)
            (tmp_dir / f"{stem}_text_raw.txt").write_text(text_asr + "\n",
                                                          encoding="utf-8")

            text_cn = re.sub(r"<\|[^|]+\|>", "", text_asr).strip()
            text_cn = clean_unsupported_punct(text_cn)
            text_cn = re.sub(r'\[([A-Za-z][^\]]*?)\]',
                            lambda m: ' ' + nvv_to_mfa(m.group(0)) + ' ', text_cn)
            text_cn = re.sub(r'\s+', ' ', text_cn).strip()
            text_cn = re.sub(r'\b([A-Z][A-Z0-9-]+)\s+\1\b', r'\1', text_cn)
            (tmp_dir / f"{stem}_text_cn.txt").write_text(text_cn + "\n",
                                                         encoding="utf-8")

            # ── Write tokens.jsonl ──
            tokens_path = tmp_dir / f"{stem}_tokens.jsonl"
            with open(tokens_path, "w", encoding="utf-8") as f:
                for wi2, w in enumerate(words_pinyin):
                    if wi2 < len(words_pinyin) - 1:
                        end_s = words_pinyin[wi2 + 1]["start"]
                    else:
                        end_s = w["end"]
                    line = {"word": w["word"],
                            "start_ms": round(w["start"] * 1000, 1),
                            "end_ms": round(end_s * 1000, 1),
                            "start_s": w["start"], "end_s": end_s, "type": "word"}
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")

            ok += 1
        except Exception as e:
            print(f"\n    FAIL {stem}: {e}")
            fail += 1

    # ── Restore original inference ──
    model.model.inference = orig_inference

    # ── Post-processing on flat temp output ──
    if ok > 0:
        _normalize_punct(tmp_dir)
        _normalize_numerals(tmp_dir)
        _normalize_ria(tmp_dir)
        _normalize_english(tmp_dir, dict_path)

    # ── Reorganize: flat temp -> mirrored output ──
    copied = reorganize_output(tmp_dir, wav_index, output_speaker, overwrite)
    print(f"  → {copied} 文件, {infer_time:.1f}s ({speed:.1f} wav/s)")

    # Clean temp
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return ok, fail, infer_time


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NVASR 批量预对齐 V2 (单模型加载)")
    parser.add_argument("--input-dir", type=str,
                        default=r"\\RS3621\CompanyShare-Confidential\Persons\jiangyichen\英文文本素材")
    parser.add_argument("--output-root", type=str,
                        default=r"\\RS3621\CompanyShare-Confidential\Persons\jiangyichen\英文nvasr文本")
    parser.add_argument("--model-path", type=str,
                        default="/mnt/project/nvvasr_standalone/models/Multilingual-NVASR")
    parser.add_argument("--dict-path", type=str, default="dict/mfa_ipa.dict")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tmp-dir", type=str,
                        default="output/nvasr_batch_tmp")
    args = parser.parse_args()

    # ── Resolve paths ──
    input_dir = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_root)
    model_path = Path(args.model_path)
    dict_path = PROJECT_ROOT / args.dict_path
    tmp_base = PROJECT_ROOT / args.tmp_dir

    print(f"输入:     {input_dir}")
    print(f"输出:     {output_root}")
    print(f"模型:     {model_path}")
    print(f"设备:     {args.device}")

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    # ── Discover speakers ──
    speakers = discover_speakers(input_dir)
    total_wavs = 0
    speaker_wavs: dict[str, int] = {}
    for sp in speakers:
        n = count_wavs(sp)
        speaker_wavs[sp.name] = n
        total_wavs += n

    already_done = 0
    for sp in speakers:
        out_sp = output_root / sp.name
        n_tg = count_outputs(out_sp)
        n_wav = speaker_wavs[sp.name]
        if n_tg >= n_wav:
            already_done += 1

    print(f"\nSpeakers:  {len(speakers)} ({already_done} 已完成, "
          f"{len(speakers) - already_done} 待处理)")
    print(f"总 WAV:    {total_wavs}")

    if args.dry_run:
        print("\n[Dry-run] 待处理 speakers:")
        for sp in speakers:
            out_sp = output_root / sp.name
            n_wav = speaker_wavs[sp.name]
            n_done = count_outputs(out_sp)
            status = "✅" if n_done >= n_wav else f"⏳ ({n_done}/{n_wav})"
            print(f"  {sp.name}: {n_wav} WAVs {status}")
        return

    # ── Load NVASR model ONCE ──
    print(f"\n加载 NVASR 模型: {model_path}")
    from funasr import AutoModel
    model = AutoModel(model=str(model_path), device=args.device, disable_update=True)
    orig_inference = model.model.inference
    print("模型加载完成")

    # ── Load MFA dict ──
    mfa_words = load_mfa_word_set(dict_path) if dict_path.exists() else None
    if mfa_words:
        print(f"MFA 词典: {len(mfa_words)} 词条")

    # ── Process each speaker ──
    total_ok = 0
    total_fail = 0
    t_start = time.time()
    processed = 0

    for sp in speakers:
        out_sp = output_root / sp.name
        n_wav = speaker_wavs[sp.name]
        n_done = count_outputs(out_sp)

        if n_done >= n_wav and not args.overwrite:
            print(f"  [跳过] {sp.name} ({n_done}/{n_wav} 已完成)")
            total_ok += n_wav
            continue

        sp_limit = min(args.limit, n_wav) if args.limit > 0 else 0
        ok, fail, elapsed = process_speaker(
            speaker_dir=sp,
            output_speaker=out_sp,
            model=model,
            orig_inference=orig_inference,
            mfa_words=mfa_words,
            dict_path=dict_path,
            device=args.device,
            limit=sp_limit,
            overwrite=args.overwrite,
            tmp_base=tmp_base,
        )
        total_ok += ok
        total_fail += fail
        processed += 1

    # ── Restore original inference ──
    model.model.inference = orig_inference

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  完成: {total_ok} OK, {total_fail} 失败")
    print(f"  处理 {processed} 个 speaker, 耗时 {elapsed_total/3600:.1f}h")
    print(f"  输出: {output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
