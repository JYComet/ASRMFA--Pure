#!/usr/bin/env python3
"""
音频标注可视化工具启动器

用法:
    # 启动可视化界面 (手动选择文件)
    python visualize.py

    # 预加载音频 + JSONL
    python visualize.py --audio video/xiaoyuan_100/xxx.wav --jsonl output/xiaoyuan_mfa/timestamp_test.jsonl

    # 用 Praat 打开 TextGrid
    python visualize.py --praat --textgrid output/xiaoyuan_mfa/textgrids/xxx.TextGrid --audio video/xiaoyuan_100/xxx.wav

    # 批量: 对指定目录所有 wav 生成 TextGrid → 用 Praat 打开
    python export_mfa_textgrid.py --input_dir video/xiaoyuan_100 --output_dir output/xiaoyuan_mfa
    python visualize.py --praat-dir output/xiaoyuan_mfa/textgrids/
"""

import argparse
import http.server
import json
import os
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent  # chinese_mfa_pipeline/
PRAAT_EXE = ROOT / "Praat.exe"
HTML_FILE = ROOT / "visualize.html"
DEFAULT_PORT = 8765


def start_server(port):
    """在项目根目录启动 HTTP 静态文件服务器"""
    os.chdir(str(ROOT))
    handler = http.server.SimpleHTTPRequestHandler
    # 禁用日志
    handler.log_message = lambda self, fmt, *args: None
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    return httpd


def open_visualizer(audio_path=None, jsonl_path=None, port=DEFAULT_PORT):
    """启动可视化界面"""
    params = []
    if audio_path:
        abs_path = os.path.abspath(audio_path)
        rel = os.path.relpath(abs_path, ROOT).replace(os.sep, "/")
        params.append(f"audio={quote(rel)}")
    if jsonl_path:
        abs_path = os.path.abspath(jsonl_path)
        rel = os.path.relpath(abs_path, ROOT).replace(os.sep, "/")
        params.append(f"jsonl={quote(rel)}")

    url = f"http://127.0.0.1:{port}/visualize.html"
    if params:
        url += "?" + "&".join(params)

    print(f"  可视化界面: {url}")
    webbrowser.open(url)


def open_praat(audio_path, textgrid_path):
    """用 Praat 打开音频 + TextGrid"""
    if not PRAAT_EXE.exists():
        print(f"错误: 找不到 Praat.exe ({PRAAT_EXE})")
        return False

    audio_abs = os.path.abspath(audio_path)
    tg_abs = os.path.abspath(textgrid_path)

    if not os.path.exists(audio_abs):
        print(f"错误: 音频文件不存在: {audio_abs}")
        return False
    if not os.path.exists(tg_abs):
        print(f"错误: TextGrid 文件不存在: {tg_abs}")
        return False

    # Praat 启动时打开文件: Praat.exe --open audio.wav textgrid.TextGrid
    # 或者通过 sendpraat 脚本控制
    # Praat 6.3+ 支持命令行参数打开文件
    cmd = [str(PRAAT_EXE), "--open", audio_abs, tg_abs]

    print(f"  启动 Praat:")
    print(f"    音频:     {audio_abs}")
    print(f"    TextGrid: {tg_abs}")

    subprocess.Popen(cmd, cwd=str(ROOT))
    return True


def open_praat_dir(textgrid_dir, audio_dir=None):
    """匹配 textgrid 目录中的 .TextGrid 与对应的 .wav 文件，批量用 Praat 打开第一对"""
    tg_dir = Path(textgrid_dir)
    if not tg_dir.exists():
        print(f"错误: 目录不存在: {tg_dir}")
        return

    textgrids = sorted(tg_dir.glob("*.TextGrid"))
    if not textgrids:
        print(f"错误: {tg_dir} 中未找到 .TextGrid 文件")
        return

    # 尝试找到对应的 wav 文件
    if audio_dir:
        wav_dir = Path(audio_dir)
    else:
        # 假设 textgrid 目录中有对应的 .lab 文件记录了原始音频路径
        # 或者音频和 textgrid 同名
        wav_dir = tg_dir

    wavs = {p.stem: p for p in wav_dir.glob("*.wav")}

    print(f"找到 {len(textgrids)} 个 TextGrid 文件")

    if len(textgrids) == 1:
        tg = textgrids[0]
        stem = tg.stem
        wav = wavs.get(stem)
        if wav:
            open_praat(str(wav), str(tg))
        else:
            print(f"警告: 找不到匹配的音频文件 ({stem}.wav)")
            # 尝试用 manifest.json 查找
            manifest_path = tg_dir.parent / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                for entry in manifest:
                    if Path(entry.get("textgrid", "")).stem == stem:
                        wav_path = entry.get("audio", "")
                        if os.path.exists(wav_path):
                            open_praat(wav_path, str(tg))
                            return
                print("  manifest.json 中也未找到匹配项")
    else:
        # 列出可用的 TextGrid, 打开第一个
        print(f"\n可用 TextGrid (共 {len(textgrids)} 个):")
        for i, tg in enumerate(textgrids[:20]):
            print(f"  [{i+1:3d}] {tg.name}")
        if len(textgrids) > 20:
            print(f"  ... 还有 {len(textgrids) - 20} 个")
        print(f"\n打开第一个...")
        tg = textgrids[0]
        stem = tg.stem
        wav = wavs.get(stem)
        if wav:
            open_praat(str(wav), str(tg))
        else:
            print(f"警告: 找不到匹配的音频 ({stem}.wav)")


def list_pairs():
    """列出项目中的音频/标注配对"""
    output_dirs = [
        ROOT / "output" / "xiaoyuan_mfa",
        ROOT / "output" / "xiaoyuan_v5",
    ]

    print("\n可用的音频/标注配对:")
    print("-" * 60)

    for out_dir in output_dirs:
        if not out_dir.exists():
            continue

        textgrid_dir = out_dir / "textgrids"
        manifest_path = out_dir / "manifest.json"

        if textgrid_dir.exists():
            tgs = list(textgrid_dir.glob("*.TextGrid"))
            print(f"\n[{out_dir.name}] {len(tgs)} 个 TextGrid")
            print(f"  → python visualize.py --praat-dir {textgrid_dir}")

        if manifest_path.exists():
            print(f"  manifest.json 可用")

        jsonl_path = out_dir / "nvv_annotations.jsonl"
        if jsonl_path.exists():
            print(f"  JSONL 标注可用: {jsonl_path.name}")

    # 检查是否有已有的配对可直接打开
    mfa_dir = ROOT / "output" / "xiaoyuan_mfa"
    tg_dir = mfa_dir / "textgrids"
    if tg_dir.exists():
        tgs = sorted(tg_dir.glob("*.TextGrid"))
        if tgs:
            print(f"\n最快启动方式:")
            print(f"  python visualize.py --praat-dir {tg_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="音频标注可视化工具启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize.py                                    # 打开交互式可视化
  python visualize.py --audio video/xxx/audio.wav        # 预加载音频
  python visualize.py --audio a.wav --jsonl t.jsonl      # 加载音频+标注
  python visualize.py --praat --textgrid t.TextGrid --audio a.wav  # Praat打开
  python visualize.py --praat-dir output/xiaoyuan_mfa/textgrids/   # 批量匹配
  python visualize.py --list                              # 列出可用配对
        """
    )
    parser.add_argument("--audio", help="音频文件路径 (.wav/.mp3)")
    parser.add_argument("--jsonl", help="时间戳 JSONL 文件路径")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP 服务器端口 (默认: {DEFAULT_PORT})")

    # Praat 模式
    parser.add_argument("--praat", action="store_true",
                        help="用 Praat.exe 打开 TextGrid + 音频")
    parser.add_argument("--textgrid", help="TextGrid 文件路径 (配合 --praat)")
    parser.add_argument("--praat-dir", help="TextGrid 目录 (批量匹配音频，配合 --praat)")

    # 信息
    parser.add_argument("--list", action="store_true",
                        help="列出可用的音频/标注配对")

    args = parser.parse_args()

    # --list: 信息模式
    if args.list:
        list_pairs()
        return

    # --praat 模式
    if args.praat:
        if args.textgrid and args.audio:
            open_praat(args.audio, args.textgrid)
        elif args.praat_dir:
            audio_dir = args.audio if args.audio else None
            open_praat_dir(args.praat_dir, audio_dir)
        else:
            # 自动查找
            mfa_dir = ROOT / "output" / "xiaoyuan_mfa" / "textgrids"
            if mfa_dir.exists():
                print("自动查找 TextGrid...")
                open_praat_dir(str(mfa_dir), args.audio)
            else:
                print("错误: --praat 需要 --textgrid + --audio 或 --praat-dir")
                print("      或手动: python visualize.py --list 查看可用配对")
        return

    # 默认: 启动可视化服务器
    print("=" * 60)
    print("  音频标注可视化工具")
    print("  Audio Annotation Viewer (Praat-like)")
    print("=" * 60)
    print()

    if args.audio:
        print(f"  预加载音频: {args.audio}")
    if args.jsonl:
        print(f"  预加载标注: {args.jsonl}")
    print(f"  项目根目录: {ROOT}")
    print(f"  Praat 位置: {PRAAT_EXE} {'✓' if PRAAT_EXE.exists() else '✗ 未找到'}")
    print()

    # 启动 HTTP 服务器
    httpd = start_server(args.port)
    print(f"  HTTP 服务: http://127.0.0.1:{args.port}")
    print()

    # 打开浏览器
    open_visualizer(args.audio, args.jsonl, args.port)

    print(f"  按 Ctrl+C 停止服务器")
    print()
    print("  使用方法:")
    print(f"    1. 拖放 WAV + JSONL 文件到浏览器窗口")
    print(f"    2. 或点击'选择音频文件'/'选择 JSONL 文件'按钮")
    print(f"    3. 空格键播放/暂停, 滚轮缩放, 双击 token 播放该段")
    print(f"    4. 按 F 切换频谱图")
    print(f"    5. 或运行: python visualize.py --praat-dir output/xiaoyuan_mfa/textgrids/")
    print(f"       直接用 Praat 打开 TextGrid + 音频")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        httpd.server_close()


if __name__ == "__main__":
    main()
