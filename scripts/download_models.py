#!/usr/bin/env python3
"""
按官方来源下载管线所需的预训练模型。

覆盖的模型
----------
MFA 预训练模型 — 走 `mfa model download`, 落到 models/mfa/:
    acoustic    mandarin_mfa
    acoustic    english_us_arpa
    g2p         mandarin_china_pinyin_mfa
    g2p         english_us_arpa
    dictionary  mandarin_china_mfa

Qwen3 原生权重 — 走 HuggingFace Hub, 落到 models/qwen3/:
    Qwen/Qwen3-ASR-1.7B-hf
    Qwen/Qwen3-ForcedAligner-0.6B-hf

不在本脚本范围内
----------------
Multilingual-NVASR 是 anchored_nvv 模式的可选组件, 主流程 (mode: full) 不依赖
它。该仓库在 HuggingFace 上受 gated 保护 — 必须先人工接受 CC-BY-NC-4.0 许可
并分享联系方式 — 因此无法自动下载, 需要时请按 README「NVASR 环境」一节手动获取。

用法
----
    python scripts/download_models.py                # 下载全部缺失项
    python scripts/download_models.py --check        # 只检查, 不下载
    python scripts/download_models.py --only mfa     # 只下 MFA 模型
    python scripts/download_models.py --only qwen    # 只下 Qwen3 权重
    python scripts/download_models.py --qwen-dir /data/models/qwen3

国内镜像
--------
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py

退出码
------
    0  全部就位 (或已成功下载)
    1  仍有缺失项 — --check 模式下表示本机不完整
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 模型清单 ─────────────────────────────────────────────────────────────
# MFA: (model_kind, model_name, 相对 models/mfa 的落盘路径)
MFA_MODELS: list[tuple[str, str, str]] = [
    ("acoustic", "mandarin_mfa", "pretrained_models/acoustic/mandarin_mfa.zip"),
    ("acoustic", "english_us_arpa", "pretrained_models/acoustic/english_us_arpa.zip"),
    ("g2p", "mandarin_china_pinyin_mfa",
     "pretrained_models/g2p/mandarin_china_pinyin_mfa.zip"),
    ("g2p", "english_us_arpa", "pretrained_models/g2p/english_us_arpa.zip"),
    ("dictionary", "mandarin_china_mfa",
     "pretrained_models/dictionary/mandarin_china_mfa.dict"),
]

# Qwen3: (HuggingFace repo_id, 目标子目录名)
HF_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen3-ASR-1.7B-hf", "Qwen3-ASR-1.7B-hf"),
    ("Qwen/Qwen3-ForcedAligner-0.6B-hf", "Qwen3-ForcedAligner-0.6B-hf"),
]

# Japanese/English v3 assets are deliberately opt-in.  The normal downloader
# path below remains byte-for-byte compatible with the existing MFA/Qwen
# workflow; these URLs are only contacted when ``--only ja-en`` is explicit.
JA_EN_ASSETS: list[dict[str, str]] = [
    {"id": "japanese_mfa_v3_acoustic", "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-japanese_mfa-v3.0.0/japanese_mfa.zip", "filename": "japanese_mfa-v3.0.0.zip", "sha256": "85928ffb1024486872a677a92e1fa94d2f997462d0b84c73a58aa9bb0e35179a"},
    {"id": "japanese_mfa_v3_dictionary", "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-japanese_mfa-v3.0.0/japanese_mfa.dict", "filename": "japanese_mfa-v3.0.0.dict", "sha256": "4a0c66760576e4b7f3748f3169e7d5217c0135f857f0b5d34637c93a85fd1c91"},
    {"id": "english_us_arpa_v3_acoustic", "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_us_arpa-v3.0.0/english_us_arpa.zip", "filename": "english_us_arpa-v3.0.0.zip", "sha256": "d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b"},
    {"id": "english_us_arpa_v3_dictionary", "url": "https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_us_arpa-v3.0.0/english_us_arpa.dict", "filename": "english_us_arpa-v3.0.0.dict", "sha256": "e8c6c7b036ae2b7c78d2768b8dc6b1f9359175b842956d00b48c53c9c332e6b0"},
]

# 配置文件里 ctc_prealign.*_model_path 需要指向的目录名, 用于打印提示。
_QWEN_CONFIG_KEYS = {
    "Qwen3-ASR-1.7B-hf": "model_path",
    "Qwen3-ForcedAligner-0.6B-hf": "forced_aligner_model_path",
}


# ═══════════════════════════════════════════════════════════════════════
# 存在性判定
# ═══════════════════════════════════════════════════════════════════════

def mfa_artifact_present(target: Path) -> bool:
    """MFA 模型的落盘形态。

    ``mfa model download dictionary`` 可能写出 ``<name>.dict`` 文件或同名
    目录, 两种都算命中。
    """
    if target.exists():
        return True
    return target.with_suffix("").is_dir()


def hf_model_present(target: Path) -> bool:
    """Qwen 权重目录是否完整。

    只认权重实际落盘的标志 (``config.json`` + 至少一个 safetensors 分片),
    避免把半途中断的空目录当成已完成。
    """
    if not (target / "config.json").is_file():
        return False
    return any(target.glob("*.safetensors"))


# ═══════════════════════════════════════════════════════════════════════
# MFA 模型下载
# ═══════════════════════════════════════════════════════════════════════

def find_mfa_exe(explicit_python: str = "") -> Path | None:
    """定位 ``mfa`` 可执行文件。

    顺序: ``--mfa-python`` 同级的 mfa -> 复用的环境探测 (pipeline_utils
    ``find_mfa_python``) -> PATH 上的 mfa。
    """
    pythons: list[Path] = []
    if explicit_python:
        pythons.append(Path(explicit_python))

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline_utils import find_mfa_python  # noqa: PLC0415
        detected = find_mfa_python("")
        if detected:
            pythons.append(detected)
    except Exception:
        # 探测失败不是致命错误 — 后面还有 PATH 兜底。
        pass

    for py in pythons:
        if not py.exists():
            continue
        # Linux/macOS: <env>/bin/mfa ; Windows conda: <env>/Scripts/mfa.exe
        for exe in (py.parent / "mfa", py.parent / "mfa.exe",
                    py.parent / "Scripts" / "mfa.exe"):
            if exe.exists():
                return exe

    on_path = shutil.which("mfa")
    return Path(on_path) if on_path else None


def download_mfa(models_root: Path, mfa_exe: Path, *, force: bool) -> list[str]:
    """逐个下载缺失的 MFA 模型, 返回失败项。"""
    env = os.environ.copy()
    # `mfa model download` resolves its target from MFA_ROOT_DIR/pretrained_models;
    # pinning it here keeps the download independent of the caller's shell.
    env["MFA_ROOT_DIR"] = str(models_root)

    failures: list[str] = []
    for kind, name, rel in MFA_MODELS:
        target = models_root / rel
        label = f"{kind}/{name}"
        try:
            shown = target.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = target

        if mfa_artifact_present(target) and not force:
            print(f"  [skip] {label} — 已存在 ({shown})")
            continue

        print(f"  [get ] {label} ...", flush=True)
        proc = subprocess.run([str(mfa_exe), "model", "download", kind, name],
                              env=env)
        if proc.returncode != 0 or not mfa_artifact_present(target):
            failures.append(label)
            print(f"  [FAIL] {label} — 见上方 mfa 输出")
        else:
            print(f"  [ok  ] {label} ({shown})")
    return failures


# ═══════════════════════════════════════════════════════════════════════
# Qwen3 权重下载
# ═══════════════════════════════════════════════════════════════════════

def prune_hf_local_cache(target: Path) -> None:
    """清掉 ``snapshot_download(local_dir=...)`` 写入的 ``.cache/``。

    ``compute_model_tree_digest`` 会把模型目录下的每个普通文件都算进
    provenance 摘要, 残留的下载元数据会让同一份权重在不同机器上得到不同的
    tree digest。
    """
    cache = target / ".cache"
    if cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def assert_no_symlinks(target: Path) -> None:
    """管线拒绝模型树里的符号链接 — 下载后立刻把关。"""
    bad = sorted(p for p in target.rglob("*") if p.is_symlink())
    if bad:
        listing = ", ".join(str(p.relative_to(target)) for p in bad[:5])
        raise RuntimeError(
            f"模型目录含符号链接, 管线 provenance 会拒绝: {target} — {listing}")


def download_hf(repo_id: str, target: Path, *, force: bool) -> bool:
    """下载单个 HuggingFace 仓库到 *target*。"""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError:
        print("  [FAIL] 缺少 huggingface_hub。请先安装:")
        print("           pip install huggingface_hub")
        return False

    if force and target.exists():
        shutil.rmtree(target, ignore_errors=True)

    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target))
    except Exception as exc:
        print(f"  [FAIL] {repo_id} — {type(exc).__name__}: {exc}")
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            print("         该仓库需要授权。请先在 HuggingFace 页面接受许可,")
            print("         然后 export HF_TOKEN=<你的 token> 重跑。")
        return False

    prune_hf_local_cache(target)
    assert_no_symlinks(target)
    return True


def download_qwen(qwen_dir: Path, *, force: bool) -> list[str]:
    """下载缺失的 Qwen3 权重, 返回失败项。"""
    failures: list[str] = []
    for repo_id, subdir in HF_MODELS:
        target = qwen_dir / subdir
        try:
            shown = target.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = target

        if hf_model_present(target) and not force:
            print(f"  [skip] {repo_id} — 已存在 ({shown})")
            continue

        print(f"  [get ] {repo_id} → {shown} ...", flush=True)
        if download_hf(repo_id, target, force=force):
            print(f"  [ok  ] {repo_id} ({shown})")
        else:
            failures.append(repo_id)
    return failures


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_ja_en(target_dir: Path, *, force: bool) -> list[str]:
    """Download fixed JA/EN v3 assets and write a hash receipt.

    This function is never called by default.  A failed URL or missing asset
    remains a failure; no unverified model is silently substituted.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    receipt: list[dict[str, str]] = []
    for asset in JA_EN_ASSETS:
        target = target_dir / asset["filename"]
        if target.exists() and not force:
            digest = _file_sha256(target)
            status = "present" if digest == asset["sha256"] else "hash_mismatch"
            if status != "present":
                failures.append(asset["id"])
            receipt.append({"id": asset["id"], "url": asset["url"], "path": str(target), "sha256": digest, "expected_sha256": asset["sha256"], "status": status})
            continue
        try:
            urllib.request.urlretrieve(asset["url"], target)
            digest = _file_sha256(target)
            if digest != asset["sha256"]:
                failures.append(asset["id"])
                receipt.append({"id": asset["id"], "url": asset["url"], "path": str(target), "sha256": digest, "expected_sha256": asset["sha256"], "status": "hash_mismatch"})
            else:
                receipt.append({"id": asset["id"], "url": asset["url"], "path": str(target), "sha256": digest, "expected_sha256": asset["sha256"], "status": "downloaded"})
        except Exception as exc:
            failures.append(asset["id"])
            receipt.append({"id": asset["id"], "url": asset["url"], "path": str(target), "sha256": "", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    (target_dir / "ja_en_assets.receipt.json").write_text(json.dumps({"schema": "ja-en-assets-receipt-v1", "assets": receipt}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return failures


# ═══════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════

def _report_status(models_root: Path, qwen_dir: Path, only: str | None, ja_en_dir: Path | None = None) -> int:
    """打印当前就位情况, 返回缺失数量。"""
    missing = 0
    if only in (None, "mfa"):
        print("MFA 预训练模型:")
        for kind, name, rel in MFA_MODELS:
            ok = mfa_artifact_present(models_root / rel)
            missing += 0 if ok else 1
            print(f"  [{'ok  ' if ok else 'MISS'}] {kind}/{name}")
    if only in (None, "qwen"):
        print("Qwen3 权重:")
        for _repo_id, subdir in HF_MODELS:
            ok = hf_model_present(qwen_dir / subdir)
            missing += 0 if ok else 1
            print(f"  [{'ok  ' if ok else 'MISS'}] {qwen_dir / subdir}")
    if only == "ja-en":
        ja_en_dir = ja_en_dir or qwen_dir.parent / "ja-en"
        print("Japanese/English MFA v3 assets:")
        for asset in JA_EN_ASSETS:
            target = ja_en_dir / asset["filename"]
            ok = target.is_file() and _file_sha256(target) == asset["sha256"]
            missing += 0 if ok else 1
            print(f"  [{'ok  ' if ok else 'MISS'}] {target}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按官方来源下载管线所需的预训练模型。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="只检查本地是否齐全, 不下载")
    parser.add_argument("--only", choices=("mfa", "qwen", "ja-en"), default=None,
                        help="只处理指定的一类模型")
    parser.add_argument("--force", action="store_true",
                        help="已存在也重新下载")
    parser.add_argument("--models-root", type=Path,
                        default=PROJECT_ROOT / "models" / "mfa",
                        help="MFA_ROOT_DIR (默认 models/mfa)")
    parser.add_argument("--qwen-dir", type=Path,
                        default=PROJECT_ROOT / "models" / "qwen3",
                        help="Qwen3 权重落盘目录 (默认 models/qwen3)")
    parser.add_argument("--ja-en-dir", type=Path,
                        default=PROJECT_ROOT / "models" / "ja-en",
                        help="Japanese/English v3 assets (仅 --only ja-en 时使用)")
    parser.add_argument("--mfa-python", default="",
                        help="MFA 所在 conda 环境的 python 路径")
    args = parser.parse_args(argv)

    models_root = args.models_root.resolve()
    qwen_dir = args.qwen_dir.resolve()

    if args.check:
        print("=" * 60)
        print("  模型就位检查")
        print("=" * 60)
        missing = _report_status(models_root, qwen_dir, args.only, args.ja_en_dir.resolve())
        print()
        if missing:
            print(f"{missing} 项缺失 — 运行 `python scripts/download_models.py` 补齐。")
            return 1
        print("全部就位。")
        return 0

    want_mfa = args.only in (None, "mfa")
    want_qwen = args.only in (None, "qwen")
    want_ja_en = args.only == "ja-en"
    failures: list[str] = []

    if want_mfa:
        print("=" * 60)
        print(f"  MFA 预训练模型 → {models_root}")
        print("=" * 60)
        mfa_exe = find_mfa_exe(args.mfa_python)
        if mfa_exe is None:
            print("  [FAIL] 找不到 mfa 可执行文件。")
            print("         请先运行 setup.sh / setup_env.bat 建好 mfa_chinese 环境,")
            print("         或用 --mfa-python 指定该环境的 python 路径。")
            failures.extend(f"{k}/{n}" for k, n, _ in MFA_MODELS)
        else:
            print(f"  使用: {mfa_exe}")
            failures.extend(download_mfa(models_root, mfa_exe, force=args.force))
        print()

    if want_qwen:
        print("=" * 60)
        print(f"  Qwen3 权重 → {qwen_dir}")
        if os.environ.get("HF_ENDPOINT"):
            print(f"  HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
        print("=" * 60)
        failures.extend(download_qwen(qwen_dir, force=args.force))
        print()

    if want_ja_en:
        print("=" * 60)
        print(f"  Japanese/English v3 assets → {args.ja_en_dir.resolve()}")
        print("  opt-in download; hashes are recorded in ja_en_assets.receipt.json")
        print("=" * 60)
        failures.extend(download_ja_en(args.ja_en_dir.resolve(), force=args.force))
        print()

    if failures:
        print(f"{len(failures)} 项未能就位:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("=" * 60)
    print("  模型全部就位")
    print("=" * 60)
    print()
    print("配置里引用 Qwen3 权重时使用以下路径:")
    for _repo_id, subdir in HF_MODELS:
        key = _QWEN_CONFIG_KEYS.get(subdir, "model_path")
        print(f"  ctc_prealign.{key}: {qwen_dir / subdir}")
    print()
    print("Multilingual-NVASR 不在此脚本范围内 (HuggingFace gated, 可选组件)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
