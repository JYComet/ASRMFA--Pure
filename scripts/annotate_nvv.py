#!/usr/bin/env python3
"""
Multilingual-NVASR 副语言标注脚本
基于 SenseVoice-Small 微调的 NVV 检测模型，识别笑声、呼吸声、咳嗽等30类副语言事件。

用法:
    python annotate_nvv.py [--input_dir DIR] [--output_dir DIR] [--device cuda:0]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_model(model_path, device="cuda:0"):
    from funasr import AutoModel

    print(f"[1/4] 加载模型: {model_path}")
    model = AutoModel(
        model=model_path,
        device=device,
        disable_update=True,
    )
    print(f"      模型加载完成 (device={device})")
    return model


def get_audio_duration(path):
    import wave
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def parse_nvv_output(text):
    """解析模型输出，提取各类标签"""
    result = {"raw_text": text, "language": None, "emotion": None,
              "nvv_events": [], "clean_text": "", "itn_mode": None}

    # 提取 <|xxx|> 格式的标签
    pipe_tags = re.findall(r"<\|([^|]+)\|>", text)

    # 语言标签
    lang_tags = {"zh", "en", "yue", "ja", "ko", "nospeech", "zh/en", "en/zh", "dialect",
                 "minnan", "wuyu"}
    for tag in pipe_tags:
        if tag in lang_tags:
            result["language"] = tag
        elif tag in {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED",
                      "SURPRISED", "EMO_UNKNOWN"}:
            result["emotion"] = tag
        elif tag in {"withitn", "woitn"}:
            result["itn_mode"] = tag

    # 提取 [...] 格式的 NVV 标签及其在文本中的位置
    for m in re.finditer(r"\[([A-Za-z][^\]]*?)\]", text):
        tag = m.group(1)
        result["nvv_events"].append({
            "type": tag,
            "char_position": m.start(),
        })

    # 生成纯净文本（移除标签）
    clean = text
    clean = re.sub(r"<\|[^|]+\|>", "", clean)
    clean = re.sub(r"\[[A-Za-z][^\]]*?\]", "", clean)
    result["clean_text"] = clean.strip()

    # 统计各类 NVV 事件数量
    event_counts = {}
    for evt in result["nvv_events"]:
        event_counts[evt["type"]] = event_counts.get(evt["type"], 0) + 1
    result["event_counts"] = event_counts

    return result


def process_files(model, input_dir, output_dir):
    """
    批量标注所有音频文件
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        wav_files = sorted(input_dir.glob("*.mp3"))
    if not wav_files:
        print(f"错误: {input_dir} 中未找到 .wav 或 .mp3 文件")
        sys.exit(1)

    print(f"\n[2/4] 扫描到 {len(wav_files)} 个音频文件")
    print(f"      输入目录: {input_dir}")
    print(f"      输出目录: {output_dir}")

    # 批量推理
    print(f"\n[3/4] 开始标注...")
    start_time = time.time()
    all_results = []

    # 分批次处理，避免内存问题
    BATCH_SIZE = 32
    for batch_start in range(0, len(wav_files), BATCH_SIZE):
        batch = wav_files[batch_start: batch_start + BATCH_SIZE]
        a_paths = [str(p) for p in batch]

        # 使用 FunASR 批量推理
        res = model.generate(
            input=a_paths,
            language="zh",
            use_itn=True,
            batch_size_s=min(300, max(60, sum(1 for _ in batch) * 30)),
        )

        for i, r in enumerate(res):
            wav_path = a_paths[i]
            text = r.get("text", "")
            parsed = parse_nvv_output(text)
            duration = get_audio_duration(wav_path)

            entry = {
                "audio": str(wav_path),
                "filename": os.path.basename(wav_path),
                "duration_seconds": round(duration, 2) if duration else None,
                "text": text,
                "clean_text": parsed["clean_text"],
                "language": parsed["language"],
                "emotion": parsed["emotion"],
                "nvv_events": parsed["nvv_events"],
                "nvv_event_types": list(parsed["event_counts"].keys()),
                "nvv_event_counts": parsed["event_counts"],
            }
            all_results.append(entry)

        elapsed = time.time() - start_time
        done = min(batch_start + BATCH_SIZE, len(wav_files))
        speed = done / elapsed if elapsed > 0 else 0
        print(f"      进度: {done}/{len(wav_files)} ({speed:.1f} 文件/秒)",
              end="\r")

    total = time.time() - start_time
    print(f"\n      完成! {len(wav_files)} 个文件, 耗时 {total:.1f} 秒 "
          f"({len(wav_files) / total:.1f} 文件/秒)")

    return all_results


def save_results(results, output_dir):
    """
    保存结果: JSONL, 摘要, 纯文本转录
    """
    output_dir = Path(output_dir)
    print(f"\n[4/4] 保存结果...")

    # 1. JSONL: 完整标注
    jsonl_path = output_dir / "nvv_annotations.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"      JSONL 标注文件: {jsonl_path}")

    # 2. TXT: 纯文本转录（含 NVV 标签）
    txt_path = output_dir / "transcripts_with_nvv.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r['filename']}]\n")
            f.write(f"{r['text']}\n\n")
    print(f"      转录文本文件: {txt_path}")

    # 3. TXT: 纯净文本（不含标签）
    clean_path = output_dir / "transcripts_clean.txt"
    with open(clean_path, "w", encoding="utf-8") as f:
        for r in results:
            if r["clean_text"]:
                f.write(f"[{r['filename']}]\n")
                f.write(f"{r['clean_text']}\n\n")
    print(f"      纯净文本文件: {clean_path}")

    # 4. 摘要报告
    total_files = len(results)
    files_with_nvv = sum(1 for r in results if r["nvv_events"])
    total_events = sum(len(r["nvv_events"]) for r in results)

    # 统计各事件类型
    all_event_counts = {}
    emotion_counts = {}
    for r in results:
        for etype, count in r["nvv_event_counts"].items():
            all_event_counts[etype] = all_event_counts.get(etype, 0) + count
        if r["emotion"]:
            emotion_counts[r["emotion"]] = emotion_counts.get(r["emotion"], 0) + 1

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Multilingual-NVASR 副语言标注报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总文件数:       {total_files}\n")
        f.write(f"含 NVV 事件:    {files_with_nvv} ({100*files_with_nvv/max(1,total_files):.0f}%)\n")
        f.write(f"总 NVV 事件数:  {total_events}\n")
        f.write(f"平均每文件:     {total_events/max(1,total_files):.1f}\n\n")

        # NVV 事件分布
        f.write("-" * 40 + "\n")
        f.write("NVV 事件类型分布:\n")
        f.write("-" * 40 + "\n")
        for etype, count in sorted(all_event_counts.items(), key=lambda x: -x[1]):
            f.write(f"  [{etype:25s}] {count:5d} 次\n")

        # 情绪分布
        if emotion_counts:
            f.write("\n" + "-" * 40 + "\n")
            f.write("情绪分布:\n")
            f.write("-" * 40 + "\n")
            for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {emo:15s} {count:5d} 段\n")

        # 含 NVV 的文件列表
        f.write("\n" + "-" * 40 + "\n")
        f.write("含 NVV 事件的文件:\n")
        f.write("-" * 40 + "\n")
        for r in results:
            if r["nvv_events"]:
                evt_list = ", ".join(r["nvv_event_types"])
                f.write(f"  {r['filename']}\n")
                f.write(f"    事件: {evt_list}\n")
                f.write(f"    文本: {r['clean_text'][:120]}...\n" if len(r['clean_text']) > 120
                        else f"    文本: {r['clean_text']}\n")

    print(f"      摘要报告:   {summary_path}")

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Multilingual-NVASR 副语言标注工具")
    parser.add_argument(
        "--input_dir",
        default=str(PROJECT_ROOT / "data_dir"),
        help="输入音频目录 (WAV/MP3)"
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "output" / "nvv_annotations"),
        help="输出目录"
    )
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "models" / "Multilingual-NVASR"),
        help="模型路径"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="推理设备 (cuda:0 / cpu)"
    )
    args = parser.parse_args()

    # 1. 加载模型
    model = load_model(args.model_path, args.device)

    # 2. 批量处理
    results = process_files(model, args.input_dir, args.output_dir)

    # 3. 保存结果
    summary_path = save_results(results, args.output_dir)

    # 打印摘要
    with open(summary_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
