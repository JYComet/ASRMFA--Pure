#!/usr/bin/env python3
"""
批量 NVASR CTC 预对齐 — 嵌套目录 → 镜像输出

遍历输入目录下的所有 .wav 文件，运行 NVASR CTC 强制对齐，
输出按输入目录结构镜像到输出根目录。

每条音频输出 6 个文件:
  {stem}.TextGrid        — MFA 锚点 (words tier)
  {stem}.lab             — MFA 语料文本
  {stem}_tokens.jsonl    — 逐词 CTC 时间戳
  {stem}_punct.json      — 标点 CTC 锚点
  {stem}_text_cn.txt     — ASR 文本
  {stem}_text_raw.txt    — 原始 ASR 文本 (含情绪标签)

用法:
  # 测试 2 条
  python3 scripts/run_nvasr_batch.py --limit 2

  # 全量处理
  python3 scripts/run_nvasr_batch.py

  # 自定义路径
  python3 scripts/run_nvasr_batch.py \
    --input-dir "//RS3621/CompanyShare-Confidential/Persons/jiangyichen/英文文本素材" \
    --output-root "//RS3621/CompanyShare-Confidential/Persons/jiangyichen/英文nvasr文本" \
    --model-path models/Multilingual-NVASR \
    --limit 100
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# ── 复用管线路径翻译 (UNC → Linux mount) ──
from pipeline_utils import find_mfa_python


# ═══════════════════════════════════════════════════════════════
# 路径翻译: Windows UNC → Linux mount
# ═══════════════════════════════════════════════════════════════

def _detect_smb_mounts() -> dict[str, str]:
    """解析 /proc/mounts 建立 UNC→Linux 映射."""
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
                server_share = dev_path
                unc = f"//{server_share}"
                mapping[unc] = mnt
                mapping[unc.replace("/", "\\")] = mnt
        for unc, mnt in list(mapping.items()):
            clean = unc.replace("\\", "/")
            if "192.168.102.202" in clean:
                parts_after = clean.split("192.168.102.202", 1)
                if len(parts_after) > 1:
                    suffix = parts_after[1]
                    mapping[f"//RS3621{suffix}"] = mnt
                    _win_suf = suffix.replace("/", "\\")
                    mapping[f"\\\\RS3621{_win_suf}"] = mnt
    except Exception:
        pass
    return mapping


_SMB_MOUNTS = _detect_smb_mounts()


def translate_path(path_str: str) -> str:
    """UNC 路径 → Linux mount 路径."""
    if not path_str or sys.platform == "win32":
        return path_str
    normalized = path_str.replace("\\", "/")
    for unc_raw, linux_mnt in sorted(_SMB_MOUNTS.items(),
                                     key=lambda x: -len(x[0])):
        unc_norm = unc_raw.replace("\\", "/")
        if normalized.startswith(unc_norm):
            rest = normalized[len(unc_norm):]
            rest = rest.lstrip("/")
            return f"{linux_mnt}/{rest}" if rest else linux_mnt
    return path_str


def resolve_path(raw: str) -> Path:
    """解析路径: UNC→Linux + 相对→绝对."""
    if not raw:
        return PROJECT_ROOT
    translated = translate_path(raw)
    p = Path(translated)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ═══════════════════════════════════════════════════════════════
# WAV 发现 + 目录映射
# ═══════════════════════════════════════════════════════════════

def discover_wav_files(input_dir: Path, limit: int = 0
                       ) -> list[tuple[Path, Path]]:
    """递归发现所有 .wav 文件，返回 (wav_path, relative_parent).

    对于 ``input_dir/A/B/C/file.wav``, relative_parent = ``A/B/C``.
    """
    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(input_dir.rglob("*.wav")):
        rel_parent = wav.parent.relative_to(input_dir)
        pairs.append((wav, rel_parent))
        if limit > 0 and len(pairs) >= limit:
            break
    return pairs


# ═══════════════════════════════════════════════════════════════
# 单目录 NVASR 处理 (调用 ctc_prealign.py)
# ═══════════════════════════════════════════════════════════════

def run_nvasr_flat(wav_dir: Path, output_dir: Path,
                   model_path: Path, dict_path: Path,
                   device: str = "cuda:0", limit: int = 0,
                   overwrite: bool = False) -> int:
    """对单个目录运行 ctc_prealign.py."""
    import subprocess

    prealign_script = SCRIPTS_DIR / "ctc_prealign.py"
    cmd = [
        sys.executable, str(prealign_script),
        "--data-dir", str(wav_dir),
        "--pinyin-dir", str(output_dir.parent / "pinyin"),
        "--output-dir", str(output_dir),
        "--model-path", str(model_path),
        "--device", device,
        "--dict-path", str(dict_path),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if overwrite:
        cmd.append("--overwrite")

    print(f"\n{'='*60}")
    print(f"  NVASR: {wav_dir}")
    print(f"  → {output_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, timeout=7200)
    return result.returncode


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="批量 NVASR CTC 预对齐 (嵌套目录 → 镜像输出)")
    parser.add_argument("--input-dir", type=str,
                        default=r"\\RS3621\CompanyShare-Confidential\Persons\jiangyichen\英文文本素材")
    parser.add_argument("--output-root", type=str,
                        default=r"\\RS3621\CompanyShare-Confidential\Persons\jiangyichen\英文nvasr文本")
    parser.add_argument("--model-path", type=str,
                        default="/mnt/project/nvvasr_standalone/models/Multilingual-NVASR")
    parser.add_argument("--dict-path", type=str,
                        default="dict/mfa_ipa.dict")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理条数, 0=全部")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅扫描, 不执行")
    parser.add_argument("--tmp-dir", type=str,
                        default="output/nvasr_batch_tmp",
                        help="临时工作目录 (相对项目根)")
    args = parser.parse_args()

    # ── 路径解析 ──
    input_dir = resolve_path(args.input_dir)
    output_root = resolve_path(args.output_root)
    model_path = resolve_path(args.model_path)
    dict_path = resolve_path(args.dict_path)
    tmp_dir = PROJECT_ROOT / args.tmp_dir

    print(f"输入目录:   {input_dir}")
    print(f"输出根目录: {output_root}")
    print(f"NVASR 模型: {model_path}")
    print(f"MFA 词典:   {dict_path}")

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)
    if not model_path.exists():
        print(f"错误: NVASR 模型不存在: {model_path}")
        sys.exit(1)

    # ── 发现所有 WAV 文件 ──
    print(f"\n扫描 WAV 文件...")
    wav_pairs = discover_wav_files(input_dir, limit=args.limit)
    print(f"  发现 {len(wav_pairs)} 个 WAV 文件")

    if not wav_pairs:
        print("没有可处理的文件。")
        return

    # ── 按父目录分组 ──
    from collections import defaultdict
    groups: dict[Path, list[Path]] = defaultdict(list)
    for wav_path, rel_parent in wav_pairs:
        groups[rel_parent].append(wav_path)

    print(f"  分布在 {len(groups)} 个子目录中")
    for rel_parent, wavs in sorted(groups.items())[:5]:
        print(f"    {rel_parent}/  ({len(wavs)} WAVs)")
    if len(groups) > 5:
        print(f"    ... 还有 {len(groups) - 5} 个目录")

    if args.dry_run:
        print("\n[Dry-run] 完成。使用 --limit N 控制数量。")
        return

    # ── 批量处理: 所有 WAV 一次性丢给 NVASR (避免重复加载模型) ──
    # ctc_prealign.py 的 rglob 能处理嵌套结构，但输出是扁平的。
    # 策略: 创建一个临时扁平目录 (符号链接), NVASR 处理完后再按
    #        manifest.json 记录的原始路径映射回镜像输出目录。

    # 简洁方案: 按 leaf directory 逐个处理。
    # 模型加载 ~3-5s, 100+ 个目录的开销可接受。
    # 对于大目录 (>100 WAV), 合并处理以减少 reload 次数。

    t0 = time.time()
    ok = fail = 0
    total = len(wav_pairs)

    for rel_parent, wavs in sorted(groups.items()):
        leaf_input = input_dir / rel_parent
        leaf_output = output_root / rel_parent
        leaf_tmp = tmp_dir / rel_parent

        n_wavs = len(wavs)
        leaf_tmp.mkdir(parents=True, exist_ok=True)
        leaf_output.mkdir(parents=True, exist_ok=True)

        # 跳过已有输出的目录 (除非 --overwrite)
        existing_tg = list(leaf_output.glob("*.TextGrid"))
        if existing_tg and len(existing_tg) >= n_wavs and not args.overwrite:
            print(f"  [跳过] {rel_parent}/ ({n_wavs} WAVs, {len(existing_tg)} 已有)")
            ok += n_wavs
            continue

        print(f"\n  [{rel_parent}] {n_wavs} WAVs")

        # 运行 NVASR → 输出到临时目录
        rc = run_nvasr_flat(
            wav_dir=leaf_input,
            output_dir=leaf_tmp,
            model_path=model_path,
            dict_path=dict_path,
            device=args.device,
            limit=0,  # 处理该目录下全部
            overwrite=True,  # tmp dir 总是覆盖
        )

        if rc != 0:
            print(f"  [失败] {rel_parent}/")
            fail += n_wavs
            continue

        # ── 将临时输出按 stem 分文件夹复制到镜像输出目录 ──
        # 每个音频的输出放入同名子文件夹: {stem}/{stem}.TextGrid, ...
        copied = 0
        stem_files: dict[str, list[Path]] = {}
        for f in leaf_tmp.iterdir():
            if f.is_file() and not f.name.startswith("manifest") and not f.name.startswith("summary"):
                # 从文件名提取 stem (去掉后缀链)
                name = f.name
                # TextGrid 的 stem
                if name.endswith(".TextGrid"):
                    stem = name[:-9]
                elif name.endswith(".lab"):
                    stem = name[:-4]
                elif name.endswith("_tokens.jsonl"):
                    stem = name[:-14]
                elif name.endswith("_punct.json"):
                    stem = name[:-11]
                elif name.endswith("_text_cn.txt"):
                    stem = name[:-12]
                elif name.endswith("_text_raw.txt"):
                    stem = name[:-13]
                else:
                    continue
                stem_files.setdefault(stem, []).append(f)

        for stem, files in stem_files.items():
            stem_dir = leaf_output / stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest = stem_dir / f.name
                if not dest.exists() or args.overwrite:
                    shutil.copy2(str(f), str(dest))
                    copied += 1

        print(f"  [完成] {rel_parent}/ → {copied} 文件 → {len(stem_files)} 个子文件夹")
        ok += n_wavs

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"  批量完成: {ok}/{total} OK, {fail} 失败")
    print(f"  耗时: {elapsed:.1f}s ({total/elapsed:.1f} files/s)" if elapsed > 0 else "")
    print(f"  输出根目录: {output_root}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
