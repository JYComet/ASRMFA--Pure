#!/usr/bin/env python3
"""
CTC Pre-alignment — NVASR 强制对齐 → MFA 锚点 TextGrid

把 NVASR 的 CTC 逐帧预测转为 MFA 可用的初始词边界锚点, MFA 只需在 ±60ms
窗口内做音素级精调, 跳过已标记的停顿段。

NV V 标签处理:
  NVASR 检测到的 [Question-yi]/[Breathing] 等 → 去括号大写 → QUESTION-YI/BREATHING
  → 作为单 phone 词条写入 MFA 词典 → CTC 强制对齐保留其时间戳
  → MFA 对齐时作为 phone 级标注输出

数据流:
  audio.wav
    → NVASR encoder → CTC logits (加 blank-frame NVV bias)
    → ASR 解码得到 text_asr (含 [NVV] 标签)
    → [NVV] → UPPERCASE 预处理 → 强制对齐 token 序列
    → 汉字 → 拼音, NVV → 保持大写
    → blank-run 停顿检测
    → MFA 锚点 TextGrid (words=拼音+NVV大写, pauses=停顿段)
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import torch

# ─── NVV 标签范围 & CTC 常量 ───
NVV_START, NVV_END = 25025, 25054   # 30 类 NVV: [Breathing]..[Crying]
BLANK_ID = 0
ELLIPSIS_ID = 9724                    # "…" 省略号 token
PAUSE_FRAMES_DEFAULT = 8              # ≥8 帧 ≈ 480ms 注入省略号, 可改
NVV_BIAS_DEFAULT = 4.0                # blank 帧 NVV logit 偏置
FRAME_MS = 60                         # CTC 帧长 (LFR m=7 n=6 → ~60ms)
QUERY_FRAMES = 4                      # 编码器前的 lang/emo/textnorm query 帧

# Known VTuber/proper names that NVASR tokenizer splits into single
# uppercase letters.  Force lowercase after merge so downstream
# reference matching doesn't see a case mismatch (e.g. "RIA"≠"ria").
_SINGLE_LETTER_LOWERCASE_NAMES: frozenset[str] = frozenset({
    "ria", "noa", "mila",
})

# 管线支持的标点白名单 — 只有这些字符会被保留为标点
ALLOWED_PUNCT_CJK = "，。！？、；：…"
ALLOWED_PUNCT_ASCII = ",.!?;:"
ALLOWED_PUNCT = set(ALLOWED_PUNCT_CJK + ALLOWED_PUNCT_ASCII)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_selected_stems_manifest(path: Path, expected: set[str]) -> str:
    """Validate shard denominator metadata and return its content digest."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selected stem manifest missing/unsafe: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines != sorted(set(lines)) or set(lines) != set(expected):
        raise ValueError(f"selected stem manifest mismatch: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_shard_accounting_receipt(path: Path, expected: set[str],
                                      eligible_universe: set[str]) -> None:
    """Validate a child receipt before quarantining it during all-GPU merge.

    A child sees the complete input directory, so its source/eligible buckets
    describe the parent universe while its output/processed buckets must be
    exactly the stems assigned to that shard.  The child receipt is evidence,
    not a merge artifact: the parent writes the sole authoritative receipt.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"shard accounting receipt missing/unsafe: {path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid shard accounting receipt JSON: {path}") from exc
    errors = validate_pipeline_accounting_receipt(receipt)
    if errors:
        raise ValueError(
            f"invalid shard accounting receipt: {path}: {'; '.join(errors)}"
        )
    if receipt.get("mode") != "ctc_prealign":
        raise ValueError(f"shard accounting mode mismatch: {path}")
    if set(receipt["eligible"]["stems"]) != set(eligible_universe):
        raise ValueError(f"shard accounting eligible universe mismatch: {path}")
    if set(receipt["output"]["stems"]) != set(expected):
        raise ValueError(f"shard accounting output mismatch: {path}")
    processed = receipt.get("extra", {}).get("processed_stems")
    if not isinstance(processed, list) or set(processed) != set(expected):
        raise ValueError(f"shard accounting processed stems mismatch: {path}")


# ═══════════════════════════════════════════════════════════════
# 拼音映射 (汉字 → 拼音音节)
# ═══════════════════════════════════════════════════════════════

def chars_and_pinyin(text: str):
    """将中文文本拆分为字符列表和对应的拼音音节列表.

    返回 (chars, pinyins), 两者长度相等.
    - CJK 字符: 1 char → 1 pinyin syllable (tone number)
    - 标点: 保持原样
    - 英文/数字: 原样保留
    - 空白: 跳过
    """
    try:
        from pypinyin import lazy_pinyin, Style
    except ModuleNotFoundError:
        raise SystemExit("pypinyin is required. Run: pip install pypinyin")

    chars: list[str] = []
    pinyins: list[str] = []

    for ch in text:
        if ch.isspace():
            continue
        chars.append(ch)
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            py = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True,
                             errors="default")
            pinyins.append(py[0] if py else ch)
        else:
            pinyins.append(ch)  # 标点/英文/数字 原样保留

    return chars, pinyins


# ── Import shared constants from pipeline_utils ──
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from pipeline_utils import (
    CTC_NORMALIZATION_MARKER,
    make_ctc_normalization_marker, parse_ctc_normalization_marker,
    NVV_NAMES, NVV_TO_MFA,
    is_nvv_token, is_english_token, is_pinyin_syllable, is_punct,
    RIA_VARIANTS, replace_ria_variants, normalize_punct_inline,
    _ASCII_TO_CJK_PUNCT,
    normalize_reference_numerals, normalize_authority_reference_numerals,
    validate_ctc_transcript_bundle,
    validate_ctc_authority_bundle,
    make_pipeline_accounting_receipt, write_pipeline_accounting_receipt,
    read_pipeline_accounting_receipt, validate_pipeline_accounting_receipt,
    make_pipeline_run_id, cuda_visible_token,
)

try:
    from english_units import (
        EnglishUnitError,
        merge_authority_fragment_group,
        parse_english_units,
    )

except ImportError:  # package-style imports in tests/tools
    from scripts.english_units import (
        EnglishUnitError,
        merge_authority_fragment_group,
        parse_english_units,
    )

try:
    if "ctc_processed_geometry" in sys.modules:
        import ctc_processed_geometry as _processed_geometry
    else:
        from scripts import ctc_processed_geometry as _processed_geometry
except ImportError:  # direct-script imports outside the repository root
    import ctc_processed_geometry as _processed_geometry

# Keep package and direct-script imports on the same module object.
sys.modules["ctc_processed_geometry"] = _processed_geometry
sys.modules["scripts.ctc_processed_geometry"] = _processed_geometry
_vad_speech_end = _processed_geometry._vad_speech_end
resolve_processed_english_spans = _processed_geometry.resolve_processed_english_spans

_resolve_processed_english_spans = resolve_processed_english_spans


def _reference_inventory(wav_files: list[Path], data_dir: Path,
                        txt_index: dict[str, Path]) -> tuple[list[Path], dict[str, str], dict[str, str]]:
    """Freeze the WAV universe before applying reference-only eligibility.

    The returned exclusion map is explicit evidence: a WAV without an
    authoritative TXT is ``missing_reference`` and is never sent to ASR as a
    fallback.  Unsupported reference text is kept separate so it cannot be
    mistaken for a missing source artifact.
    """
    by_stem: dict[str, Path] = {}
    for wav in wav_files:
        if wav.stem in by_stem:
            raise ValueError(f"duplicate WAV stem in frozen source universe: {wav.stem}")
        by_stem[wav.stem] = wav
    eligible: dict[str, str] = {}
    exclusions: dict[str, str] = {}
    for stem in sorted(by_stem):
        ref = find_ref_text(stem, data_dir, txt_index)
        if not ref:
            exclusions[stem] = "missing_reference"
        elif has_japanese(ref):
            exclusions[stem] = "unsupported_reference"
        else:
            eligible[stem] = clean_unsupported_punct(ref)
    return [by_stem[s] for s in sorted(eligible)], eligible, exclusions


def _source_inventory(wav_files: list[Path], data_dir: Path,
                      txt_index: dict[str, Path],
                      allow_missing_reference: bool = False,
                      reference_mode: str = "auto") -> tuple[list[Path], dict[str, str], dict[str, str]]:
    """Build either a reference-authoritative or free-ASR inventory."""
    if reference_mode == "fallback":
        # An explicit no-reference run must not become a mixed batch merely
        # because a stale/accidental {stem}.txt is present beside one WAV.
        # ASR remains the sole transcript source for this policy.
        by_stem: dict[str, Path] = {}
        for wav in wav_files:
            if wav.stem in by_stem:
                raise ValueError(f"duplicate WAV stem in frozen source universe: {wav.stem}")
            by_stem[wav.stem] = wav
        return [by_stem[stem] for stem in sorted(by_stem)], {}, {}
    if not allow_missing_reference:
        return _reference_inventory(wav_files, data_dir, txt_index)
    by_stem: dict[str, Path] = {}
    for wav in wav_files:
        if wav.stem in by_stem:
            raise ValueError(f"duplicate WAV stem in frozen source universe: {wav.stem}")
        by_stem[wav.stem] = wav
    optional_refs: dict[str, str] = {}
    for stem in sorted(by_stem):
        ref = find_ref_text(stem, data_dir, txt_index)
        if ref and not has_japanese(ref):
            optional_refs[stem] = clean_unsupported_punct(ref)
    return [by_stem[stem] for stem in sorted(by_stem)], optional_refs, {}


def load_mfa_word_set(dict_path: Path | None) -> set[str] | None:
    """加载 MFA 词典词条集合 (若提供)."""
    if not dict_path or not dict_path.exists():
        return None
    words: set[str] = set()
    with open(dict_path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                words.add(line.split()[0])
    return words


def valid_mfa_word(token: str, mfa_words: set[str] | None = None) -> bool:
    """Check if *token* should be kept in the MFA words tier."""
    if is_nvv_token(token):
        return token in mfa_words if mfa_words is not None else True
    if is_pinyin_syllable(token):
        return token in mfa_words if mfa_words is not None else True
    if is_english_token(token):
        return True
    if token.isdigit():
        return True
    return False


def nvv_to_mfa(label: str) -> str:
    """[Question-yi] → QUESTION-YI, [Breathing] → BREATHING."""
    inner = label.strip("[]")
    return NVV_TO_MFA.get(inner, inner.upper().replace(" ", "-"))


def preprocess_asr_for_mfa(text: str) -> str:
    """将 ASR 输出中的 [NVV] 标签转换为 MFA 大写 token.

    "[Question-yi]" → "QUESTION-YI"
    "[Breathing]" → "BREATHING"
    其他文本 (汉字/标点) 保持不变.
    """
    text = re.sub(
        r'\[([A-Za-z][^\]]*?)\]',
        lambda m: ' ' + nvv_to_mfa(m.group(0)) + ' ',
        text)
    return re.sub(r'\s+', ' ', text).strip()


# ═══════════════════════════════════════════════════════════════
# Monkey-patch: 用参考文本做 CTC 强制对齐, 非 ASR 解码
# ═══════════════════════════════════════════════════════════════

def _free_decode_logits(logits: torch.Tensor, *, reference_only: bool,
                        enable_nvv: bool, bias_value: float,
                        blank_id: int = BLANK_ID) -> torch.Tensor:
    """Prepare only the logits clone used by free CTC decoding."""
    decoded = logits.clone()
    if reference_only:
        decoded[..., NVV_START:NVV_END + 1] = -float("inf")
    elif enable_nvv:
        top_pred = decoded.argmax(dim=-1)
        is_blank = top_pred == blank_id
        decoded[is_blank, NVV_START:NVV_END + 1] += bias_value
    return decoded

def make_patched_inference(ref_texts: dict[str, str],
                           bias_value: float = NVV_BIAS_DEFAULT,
                           pause_threshold: int = PAUSE_FRAMES_DEFAULT,
                           enable_nvv: bool = True,
                           reference_only: bool = False):
    """
    创建打了补丁的 inference 方法.

    与原版 export_mfa_textgrid.py 的核心区别:
    - 不从 CTC 解码 ASR 文本, 而是从 ref_texts 字典查找参考中文文本
    - 对参考文本做 CTC 强制对齐 → 汉字级别时间戳
    - 同样做 blank-frame NVV bias + 停顿检测 + 省略号注入

    ref_texts: {stem: chinese_text}  — 键为音频文件 stem (无扩展名)
    enable_nvv: 非 reference-only 模式是否启用 blank-frame NVV bias.
    reference_only: 仅在自由解码 logits clone 上屏蔽 NVV；forced alignment
                    继续使用干净原 logits 和 reference target.
    """
    try:
        import cn2an as _cn2an
    except ImportError:
        _cn2an = None

    def patched(self, data_in, data_lengths=None,
                key=["wav_file_tmp_name"], tokenizer=None,
                frontend=None, **kwargs):
        from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

        meta = {}
        time1 = time.perf_counter()
        samples = load_audio_text_image_video(
            data_in, fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer)
        meta["load_data"] = f"{time.perf_counter() - time1:.3f}"

        speech, lens = extract_fbank(
            samples, data_type=kwargs.get("data_type", "sound"),
            frontend=frontend)
        meta["extract_feat"] = f"{time.perf_counter() - time1:.3f}"
        speech, lens = speech.to(kwargs["device"]), lens.to(kwargs["device"])

        # ── 添加 query embedding (lang/emo/textnorm, 共 4 帧) ──
        lang = kwargs.get("language", "auto")
        lq = self.embed(
            torch.LongTensor([[self.lid_dict.get(lang, 0)]])
            .to(speech.device)).repeat(speech.size(0), 1, 1)
        tn = "withitn" if kwargs.get("use_itn", False) else "woitn"
        tq = self.embed(
            torch.LongTensor([[self.textnorm_dict[tn]]])
            .to(speech.device)).repeat(speech.size(0), 1, 1)
        speech, lens = torch.cat((tq, speech), 1), lens + 1
        eq = self.embed(
            torch.LongTensor([[1, 2]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech, lens = torch.cat((torch.cat((lq, eq), 1), speech), 1), lens + 3

        # ── Encoder ──
        enc, elens = self.encoder(speech, lens)
        if isinstance(enc, tuple):
            enc = enc[0]
        ctc_logits = self.ctc.log_softmax(enc)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b = enc.size(0)
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key *= b

        for i in range(b):
            # Free decode gets an isolated clone; forced alignment below keeps
            # the clean ctc_logits and reference target.
            x = _free_decode_logits(
                ctc_logits[i, :elens[i].item(), :],
                reference_only=reference_only,
                enable_nvv=enable_nvv,
                bias_value=bias_value,
                blank_id=self.blank_id,
            )

            raw_y = x.argmax(dim=-1).tolist()

            # ── 记录 blank 段 + 长空白注入省略号 (单次扫描) ──
            blank_runs = []
            yseq_pause = torch.tensor(raw_y).to(x.device)
            jj = 0
            while jj < len(raw_y):
                if raw_y[jj] == BLANK_ID:
                    s = jj
                    while jj < len(raw_y) and raw_y[jj] == BLANK_ID:
                        jj += 1
                    run_len = jj - s
                    blank_runs.append((s, jj))
                    if run_len >= pause_threshold:
                        yseq_pause[s + run_len // 2] = ELLIPSIS_ID
                else:
                    jj += 1

            yseq_unique = torch.unique_consecutive(yseq_pause, dim=-1)
            mask = yseq_unique != self.blank_id
            token_int = yseq_unique[mask].tolist()
            asr_text = tokenizer.decode(token_int)  # 仅用于显示/nvv标签

            # ── 后处理 (省略号标点去重等) ──
            # Adjacent ellipsis+punct → punct only
            asr_text = re.sub(r'…([，。！？、；：,\.!\?;:])', r'\1', asr_text)
            asr_text = re.sub(r'([，。！？、；：,\.!\?;:])…', r'\1', asr_text)
            # ，NVV… → ，NVV  (ellipsis redundant when punct already exists
            # before the NVV token; the comma already marks the pause)
            asr_text = re.sub(
                r'([，。！？、；：,\.!\?;:])([A-Z][A-Z0-9-]*[A-Z0-9])…',
                r'\1\2', asr_text)
            asr_text = re.sub(r'…{2,}', '…', asr_text)
            asr_text = re.sub(r'^((?:<\|[^|]+\|>|\[[^\]]+\])*)…+', r'\1', asr_text)
            asr_text = re.sub(
                r'\[([A-Za-z][^\]]*?)\]\s*([，。！？、；：…,\.!\?;:\-]+)\s*\[\1\]',
                r'\2[\1]', asr_text)
            asr_text = re.sub(r'\[([A-Za-z][^\]]*?)\]\s+\[\1\]', r'[\1]', asr_text)

            # ── 强制对齐: 优先使用参考文本 (准确), 纯 ASR 作后备 ──
            total_frames = elens[i].item()
            duration_s = total_frames * FRAME_MS / 1000 - QUERY_FRAMES * FRAME_MS / 1000

            asr_final = asr_text.lstrip('…')
            asr_clean = re.sub(r'<\|[^|]+\|>', '', asr_final).strip()

            # FunASR may return the filename stem without its final suffix.
            # Preserve an exact key match first because valid filenames can
            # themselves contain ".wav" (e.g. source.wav.tmp_clip0001.wav).
            _raw_key_stem = str(key[i])
            stem = (_raw_key_stem if _raw_key_stem in ref_texts
                    else Path(_raw_key_stem).stem)
            if stem in ref_texts:
                # 使用参考文本 (ground truth) → 更准确的 CJK 字符级强制对齐
                align_text = ref_texts[stem].strip()
            else:
                # 无参考文本, 使用 ASR 解码文本
                align_text = asr_clean

            # Reference-only keeps the canonical reference surface exactly;
            # non-reference mode retains its historical deterministic transforms.
            if _cn2an is not None and not reference_only:
                parts = re.split(r'(\[[^\]]+\]|[A-Z][A-Z0-9-]*[A-Z0-9])', align_text)
                for k, part in enumerate(parts):
                    if re.match(r'^(\[[^\]]+\]|[A-Z][A-Z0-9-]*[A-Z0-9])$', part):
                        continue
                    try:
                        parts[k] = _cn2an.transform(part, 'an2cn')
                    except Exception:
                        pass
                align_text = ''.join(parts)

            # ria and punctuation transforms are non-reference-mode only.
            if not reference_only:
                align_text = replace_ria_variants(align_text)

            # Preserve reference punctuation/order in reference-only mode.
            if not reference_only:
                align_text = normalize_punct_inline(align_text)

            words_aligned = []  # token 级别时间戳
            speech_tokens = []  # initialize to avoid UnboundLocalError
            if align_text:
                tokens = tokenizer.text2tokens(align_text)
                speech_tokens = tokens
                token_ids_list = tokenizer.tokens2ids(speech_tokens)
                token_ids_flat = []
                # ``tokens2ids`` may expand one logical tokenizer token into
                # multiple CTC ids.  Keep a label for every flattened id;
                # indexing ``speech_tokens`` by the flattened position
                # silently shifts all following timestamps in that case.
                flat_token_labels = []
                for token, tids in zip(speech_tokens, token_ids_list):
                    if tids:
                        token_ids_flat.extend(tids)
                        flat_token_labels.extend([token] * len(tids))
                    else:
                        token_ids_flat.append(124)  # space token
                        flat_token_labels.append(token)

                if token_ids_flat:
                    # 准备 logits: 去掉 query 帧 (复用 L213 的 ctc_logits,
                    # 因 x 已改为 .clone()，NVV bias 不会污染 ctc_logits)
                    logits_speech = ctc_logits[i, QUERY_FRAMES:total_frames, :]
                    total_speech_frames = total_frames - QUERY_FRAMES

                    from funasr.models.sense_voice.utils.ctc_alignment import ctc_forced_align
                    # 零化 blank 高置信帧的 blank logit, 防止对齐塌缩
                    pred = logits_speech.argmax(dim=-1)
                    align_logits = logits_speech.clone()
                    align_logits[pred == self.blank_id, self.blank_id] = 0

                    align = ctc_forced_align(
                        align_logits.unsqueeze(0).float(),
                        torch.LongTensor(token_ids_flat).unsqueeze(0).to(kwargs["device"]),
                        torch.LongTensor([total_speech_frames]).to(kwargs["device"]),
                        torch.LongTensor([len(token_ids_flat)]).to(kwargs["device"]),
                        ignore_id=self.ignore_id,
                    )

                    # 分组提取 token 边界。 ctc_forced_align 返回的是
                    # target token id，不是 target 序号；某个 target 可能
                    # 获得 0 帧。按“第几个非 blank group”取标签会在此处
                    # 将后续所有时间戳左移，因此必须按 token id 在目标
                    # 序列中的下一个出现位置恢复逻辑 token 标签。
                    pred_grp = groupby(align[0, :total_speech_frames].tolist())
                    _s = 0
                    target_cursor = 0
                    matched_target_positions: set[int] = set()
                    for ptok, pframe_iter in pred_grp:
                        frame_indices = list(pframe_iter)
                        _e = _s + len(frame_indices)
                        if ptok != 0:
                            matched = next(
                                (pos for pos in range(target_cursor, len(token_ids_flat))
                                 if token_ids_flat[pos] == ptok),
                                None,
                            )
                            if matched is not None:
                                t_left = max((_s * FRAME_MS - 30) / 1000, 0)
                                t_right = min((_e * FRAME_MS - 30) / 1000, duration_s)
                                words_aligned.append({
                                    "word": flat_token_labels[matched],
                                    "start": round(t_left, 3),
                                    "end": round(t_right, 3),
                                    "source_ctc_ordinal": matched,
                                })
                                matched_target_positions.add(matched)
                                target_cursor = matched + 1
                        _s = _e

                    missing_target_positions = [
                        pos for pos in range(len(token_ids_flat))
                        if pos not in matched_target_positions
                    ]
                    missing_ctc_tokens = [
                        flat_token_labels[pos] for pos in missing_target_positions
                    ]
                else:
                    missing_ctc_tokens = list(speech_tokens)
            else:
                missing_ctc_tokens = []

            ctc_alignment_complete = not missing_ctc_tokens

            # ── Check English token completeness ──
            # CTC forced alignment can drop tokens that get 0 frames
            # (e.g. "live"→"li"+"ve" but only "li" survives).  Detect
            # this by comparing tokenizer output against aligned tokens.
            expected_eng: list[str] = []
            for t in speech_tokens:
                t_clean = str(t).lstrip('▁')
                if is_english_token(t_clean):
                    expected_eng.append(t_clean)
            actual_eng = [w['word'].lstrip('▁') for w in words_aligned
                          if is_english_token(w['word'].lstrip('▁'))]
            exp_counts = Counter(expected_eng)
            act_counts = Counter(actual_eng)
            english_complete = True
            missing_english: list[str] = []
            for tok, count in exp_counts.items():
                if act_counts.get(tok, 0) < count:
                    english_complete = False
                    missing_english.append(
                        f"{tok}(expected {count}, got {act_counts.get(tok, 0)})")

            # Blank runs were measured in encoder coordinates; published
            # pause coordinates are the speech slice [QUERY_FRAMES, total).
            # A run touching query frames is not a physical speech pause.
            # Drop it instead of publishing a synthetic pause at speech zero.
            blank_runs_speech = [(s - QUERY_FRAMES, e - QUERY_FRAMES)
                                 for s, e in blank_runs
                                 if s >= QUERY_FRAMES and e > QUERY_FRAMES]
            results.append({
                "key": key[i],
                "text_asr": asr_final,
                "text_asr_clean": asr_clean,
                "reference_text": ref_texts.get(stem),
                "reference_only": reference_only,
                "duration_s": round(duration_s, 3),
                "words": words_aligned,
                "blank_runs": blank_runs_speech,
                "english_complete": english_complete,
                "missing_english": missing_english,
                "ctc_alignment_complete": ctc_alignment_complete,
                "missing_ctc_tokens": missing_ctc_tokens,
            })

        return results, meta

    return patched


# ═══════════════════════════════════════════════════════════════
# TextGrid 写入 (Praat 格式, MFA 兼容)
# ═══════════════════════════════════════════════════════════════

def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one CTC artifact without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def write_textgrid(words_pinyin: list[dict], duration_s: float,
                   out_path: Path, pauses: list[dict] | None = None) -> None:
    """生成双层 TextGrid: words tier + pauses tier.

    MFA 只读 words 层做锚点对齐, pauses 层供下游参考 (≥200ms CTC 空白段).
    """
    n_tiers = 2 if pauses is not None else 1
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = 0",
        f"xmax = {duration_s:.6f}",
        "tiers? <exists>",
        f"size = {n_tiers}",
        "item []:",
    ]

    # ── words tier: 每个词延申到下一个词的 start ──
    lines.append("    item [1]:")
    lines.append('        class = "IntervalTier"')
    lines.append('        name = "words"')
    lines.append(f"        xmin = 0")
    lines.append(f"        xmax = {duration_s:.6f}")
    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for i, w in enumerate(words_pinyin):
        ws = w["start"]
        # A canonical authority unit owns only its normalized CTC span.  Do
        # not let a following punctuation/gap or lexical row enlarge that
        # interval merely to make the tier visually contiguous.
        if "canonical_unit" in w:
            we = w["end"]
        else:
            we = (words_pinyin[i + 1]["start"]
                  if i + 1 < len(words_pinyin) else w["end"])
        if ws > cursor + 0.005:
            intervals.append((cursor, ws, ""))
        intervals.append((ws, we, w["word"]))
        cursor = we
    if cursor < duration_s - 0.005:
        intervals.append((cursor, duration_s, ""))
    lines.append(f"        intervals: size = {len(intervals)}")
    for k, (s, e, txt) in enumerate(intervals):
        lines.append(f"        intervals [{k + 1}]:")
        lines.append(f"            xmin = {s:.6f}")
        lines.append(f"            xmax = {e:.6f}")
        txt_escaped = txt.replace('"', '""')
        lines.append(f'            text = "{txt_escaped}"')

    # ── pauses tier: CTC 空白段 ≥200ms ──
    if pauses is not None:
        lines.append("    item [2]:")
        lines.append('        class = "IntervalTier"')
        lines.append('        name = "pauses"')
        lines.append(f"        xmin = 0")
        lines.append(f"        xmax = {duration_s:.6f}")
        p_intervals: list[tuple[float, float, str]] = []
        pc = 0.0
        for p in pauses:
            ps = float(p["start_ms"]) / 1000
            pe = float(p["end_ms"]) / 1000
            if (not math.isfinite(ps) or not math.isfinite(pe)
                    or ps < 0 or pe <= ps or ps >= duration_s):
                raise ValueError(f"invalid pause endpoint: {ps}..{pe} / {duration_s}")
            if pe > duration_s:
                if pe - duration_s > FRAME_MS / 1000 + 1e-6:
                    raise ValueError(f"pause endpoint exceeds WAV axis: {pe} > {duration_s}")
                pe = duration_s
            if pe <= ps:
                raise ValueError("pause endpoint clips to empty interval")
            if ps > pc + 0.005:
                p_intervals.append((pc, ps, ""))
            p_intervals.append((ps, pe, f'{(pe - ps) * 1000:.1f}ms'))
            pc = pe
        if pc < duration_s - 0.005:
            p_intervals.append((pc, duration_s, ""))
        if not p_intervals:
            p_intervals = [(0, duration_s, "")]
        lines.append(f"        intervals: size = {len(p_intervals)}")
        for k, (s, e, label) in enumerate(p_intervals):
            lines.append(f"        intervals [{k + 1}]:")
            lines.append(f"            xmin = {s:.6f}")
            lines.append(f"            xmax = {e:.6f}")
            lines.append(f'            text = "{label}"')

    _atomic_write_text(out_path, "\n".join(lines) + "\n")


def _clamp_words_to_wav_axis(words: list[dict], duration_s: float) -> list[dict]:
    """Keep encoder-derived endpoints inside the authoritative WAV axis."""
    result = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= duration_s or end <= start:
            raise ValueError(f"invalid token endpoint: {word['word']}")
        if end > duration_s:
            if end - duration_s > FRAME_MS / 1000 + 1e-6:
                raise ValueError(f"token endpoint exceeds WAV axis: {word['word']}")
            end = duration_s
        if end <= start:
            raise ValueError(f"token endpoint clips to empty interval: {word['word']}")
        result.append({**word, "start": start, "end": end})
    return result


# ═══════════════════════════════════════════════════════════════
# 主流程: 批量处理
# ═══════════════════════════════════════════════════════════════

def has_japanese(text: str) -> bool:
    """检测文本是否含日语假名 (ひらがな / カタカナ)."""
    for ch in text:
        if '぀' <= ch <= 'ゟ':   # Hiragana U+3040..U+309F
            return True
        if '゠' <= ch <= 'ヿ':   # Katakana U+30A0..U+30FF
            return True
    return False


def clean_unsupported_punct(text: str) -> str:
    """过滤掉白名单外的非 CJK 标点符号 (如 」『』【】《》\"' 等)。

    NVASR 词表包含大量符号 token, CTC 强制对齐会把它们当标点输出,
    导致 _punct.json / .lab / TextGrid 中出现预期外的字符。
    此函数在文本入口处过滤, 保证所有下游文件一致。
    """
    result: list[str] = []
    for ch in text:
        if ch in ALLOWED_PUNCT:
            result.append(ch)
        elif '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            result.append(ch)       # CJK 汉字
        elif ch.isalpha() or ch.isdigit():
            result.append(ch)       # 英文/数字
        elif ch == '-':
            result.append(ch)       # NVV 标签内的连字符 (QUESTION-YI 等)
        elif ch.isspace():
            result.append(ch)       # 空格
        elif ch in '<|>[]':
            result.append(ch)       # 保留 emotion/lang/NVV 标签结构字符
        # 其余符号类字符 (」『』【】《》"' 等) 直接丢弃
    return ''.join(result)


def _build_txt_index(data_dir: Path) -> dict[str, Path]:
    """Build a deterministic recursive ``{stem: path}`` reference index.

    The fresh corpus contains ``ria新增/ria/*.txt`` below the usual speaker
    directory depth.  A one-level fallback silently omitted that subtree,
    causing a frozen 1000-WAV authority selection to become 500 eligible
    stems at CTC time.  Scan the complete source tree once and keep the first
    path in stable lexical order; duplicate stems remain an explicit source
    error in the WAV denominator rather than being resolved nondeterministically.
    """
    index: dict[str, Path] = {}
    try:
        entries = sorted(
            (path for path in data_dir.rglob("*.txt")
             if path.is_file() and not path.is_symlink()),
            key=lambda path: path.as_posix(),
        )
    except OSError:
        return index
    for path in entries:
        stem = path.stem
        if stem not in index:
            index[stem] = path
    return index


def find_ref_text(stem: str, data_dir: Path,
                  txt_index: dict[str, Path] | None = None) -> str | None:
    """Look up reference text for *stem* using pre-built index or rglob fallback."""
    # Use index if provided (O(1) lookup)
    if txt_index is not None:
        path = txt_index.get(stem)
        if path:
            return path.read_text(encoding="utf-8").strip()
        for suffix in ("_qwen3-api", "_qwen3", "_firered"):
            path = txt_index.get(f"{stem}{suffix}")
            if path:
                return path.read_text(encoding="utf-8").strip()
        m = re.search(r"_(firered|qwen3|qwen3-api)$", stem)
        if m:
            path = txt_index.get(stem[:m.start()])
            if path:
                return path.read_text(encoding="utf-8").strip()
        return None

    # Fallback: slow rglob (for backward compatibility)
    candidates = list(data_dir.rglob(f"{stem}.txt"))
    if candidates:
        return candidates[0].read_text(encoding="utf-8").strip()
    for suffix in ("_qwen3-api", "_qwen3", "_firered"):
        candidates = list(data_dir.rglob(f"{stem}{suffix}.txt"))
        if candidates:
            return candidates[0].read_text(encoding="utf-8").strip()
    m = re.search(r"_(firered|qwen3|qwen3-api)$", stem)
    if m:
        base = stem[:m.start()]
        candidates = list(data_dir.rglob(f"{base}.txt"))
        if candidates:
            return candidates[0].read_text(encoding="utf-8").strip()
    return None


# ═══════════════════════════════════════════════════════════════
# 输出后处理 (与 run_pipeline.py 的 normalize_punct / normalize / normalize_en 等价)
# 两种模式走不同逻辑, 这里独立实现避免导入 run_pipeline 的副作用.
# ═══════════════════════════════════════════════════════════════

_NORM_ALLOWED_PUNCT = frozenset("，。！？、；：…")
_ASCII_TO_CJK = {
    ",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "：",
}


def _normalize_punct(ctc_dir: Path) -> int:
    """ASCII→CJK 标点映射 + 相邻标点合并 + 非白名单标点替换."""
    changed = 0
    for txt_file in sorted(ctc_dir.glob("*_text_cn.txt")):
        stem = txt_file.stem.replace("_text_cn", "")
        try:
            text = txt_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue

        punct_file = ctc_dir / f"{stem}_punct.json"
        punct_entries: list[dict] = []
        if punct_file.exists():
            try:
                punct_entries = json.loads(punct_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Phase 1: ASCII → CJK
        text = text.translate(str.maketrans(_ASCII_TO_CJK))
        for p in punct_entries:
            w = p.get("word", "")
            if w in _ASCII_TO_CJK:
                p["word"] = _ASCII_TO_CJK[w]

        # Phase 2: classify each character
        # '-' between two ASCII letters is part of an NVV token
        # (e.g. SURPRISE-OH, QUESTION-YI) — never treat as punct.
        # Regression Case 17.
        char_info: list[tuple[str, bool | None, str]] = []
        for i, ch in enumerate(text):
            is_hyphen_in_nvv = (
                ch == '-'
                and i > 0 and i + 1 < len(text)
                and text[i - 1].isascii() and text[i - 1].isalpha()
                and text[i + 1].isascii() and text[i + 1].isalpha()
            )
            if is_punct(ch) and not is_hyphen_in_nvv:
                char_info.append(("punct", ch in _NORM_ALLOWED_PUNCT, ch))
            else:
                char_info.append(("other", None, ch))

        for p in punct_entries:
            p["_merge_del"] = False

        # Phase 3: replace abnormal + merge adjacent
        new_chars: list[str] = []
        i = 0
        punct_seq = 0
        while i < len(char_info):
            kind, is_allowed, ch = char_info[i]
            if kind != "punct":
                new_chars.append(ch)
                i += 1
                continue

            group: list[tuple[int, bool, str]] = []
            j = i
            while j < len(char_info) and char_info[j][0] == "punct":
                group.append((j, char_info[j][1], char_info[j][2]))
                j += 1

            if len(group) == 1:
                _, ia, ch = group[0]
                if ia:
                    new_chars.append(ch)
                else:
                    new_chars.append("，")
                    if punct_seq < len(punct_entries):
                        punct_entries[punct_seq]["word"] = "，"
                punct_seq += 1
                i = j
                continue

            new_chars.append("，")
            first_seq = punct_seq
            last_seq = punct_seq + len(group) - 1
            if first_seq < len(punct_entries) and last_seq < len(punct_entries):
                first = punct_entries[first_seq]
                last = punct_entries[last_seq]
                first["word"] = "，"
                first["end_ms"] = last["end_ms"]
                first["end_s"] = last["end_s"]
                for k in range(first_seq + 1, last_seq + 1):
                    if k < len(punct_entries):
                        punct_entries[k]["_merge_del"] = True
            punct_seq += len(group)
            i = j

        # Phase 4: write back
        new_text = "".join(new_chars)
        new_punct = [p for p in punct_entries if not p.pop("_merge_del", False)]

        if new_text != text or len(new_punct) != len(punct_entries):
            txt_file.write_text(new_text + "\n", encoding="utf-8")
            if punct_file.exists() or new_punct:
                punct_file.write_text(
                    json.dumps(new_punct, ensure_ascii=False), encoding="utf-8")
            changed += 1

    if changed:
        print(f"  [normalize_punct] {changed} files")
    return changed


def _normalize_numerals(ctc_dir: Path) -> int:
    """阿拉伯数字→中文数字，只处理人类可读的 _text_cn.txt。

    .lab 已经是 MFA token 序列；声调数字绝不能再交给 cn2an。
    """
    try:
        import cn2an as _cn2an
    except ImportError:
        print("  [normalize_numerals] cn2an not installed, skipping")
        return 0

    changed = 0
    for txt_file in sorted(ctc_dir.glob("*_text_cn.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8-sig").strip()
        except FileNotFoundError:
            continue

        normalized = normalize_reference_numerals(text, _cn2an.transform)

        changed_file = normalized != text
        if changed_file:
            txt_file.write_text(normalized + "\n", encoding="utf-8")
        if changed_file:
            changed += 1

    if changed:
        print(f"  [normalize_numerals] {changed} files")
    return changed


def _validate_all_ctc_bundles(
    ctc_dir: Path, authority_references: dict[str, str] | None = None,
) -> bool:
    """Validate every CTC bundle before publishing a normalization marker."""
    lab_paths = sorted(ctc_dir.glob("*.lab"))
    invalid: list[tuple[str, list[str]]] = []
    for lab_path in lab_paths:
        # This is the producer/raw bundle.  It must validate raw CTC
        # authority without requiring the processed geometry that is created
        # later in ctc_pretg_adj.
        errors = validate_ctc_transcript_bundle(
            ctc_dir, lab_path.stem, _include_authority=False)
        if not errors and authority_references and lab_path.stem in authority_references:
            errors.extend(validate_ctc_authority_bundle(
                ctc_dir, lab_path.stem, authority_references[lab_path.stem],
                require_processed=False))
        if errors:
            invalid.append((lab_path.stem, errors))
    if invalid:
        print(f"  ERROR: {len(invalid)} invalid CTC transcript bundle(s)")
        for stem, errors in invalid[:20]:
            print(f"    - {stem}: {'; '.join(errors)}")
        if len(invalid) > 20:
            print(f"    ... and {len(invalid) - 20} more")
        return False
    print(f"  CTC bundle validation: {len(lab_paths)} OK")
    return bool(lab_paths)


def _wav_duration_s(path: Path) -> float:
    """Return the authoritative physical WAV duration, never encoder length."""
    # ``wave`` rejects IEEE-float WAVs (format tag 3), while soundfile is
    # already a runtime audio reader used by this script and supports both
    # float and PCM WAV metadata.
    import soundfile as sf

    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise ValueError(f"invalid WAV sample rate: {path}")
    return info.frames / info.samplerate


def _rebuild_final_manifest(ctc_dir: Path, audio_dir: Path,
                            wav_files: list[Path] | None = None) -> None:
    """Atomically publish a manifest from final normalized CTC artifacts."""
    # Build stem→WAV mapping (supports subdirectory layout)
    wav_map: dict[str, Path] = {}
    # Callers that already discovered the input bundle can pass that immutable
    # list to avoid rescanning a large audio tree during final publication.
    # The default keeps the historical standalone-call behaviour unchanged.
    for p in (wav_files if wav_files is not None else audio_dir.rglob("*.wav")):
        if p.stem not in wav_map:
            wav_map[p.stem] = p

    entries = []
    for lab in sorted(ctc_dir.glob("*.lab")):
        stem = lab.stem; tokens = ctc_dir / f"{stem}_tokens.jsonl"
        audio = wav_map.get(stem)
        if audio is None or not audio.is_file() or not tokens.is_file():
            raise ValueError(f"cannot build final manifest for {stem}")
        rows = [json.loads(line) for line in tokens.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        duration = _wav_duration_s(audio)
        punct_path = ctc_dir / f"{stem}_punct.json"
        try:
            punct_count = len(json.loads(punct_path.read_text(encoding="utf-8-sig")))
        except Exception as exc:
            raise ValueError(f"cannot rebuild final manifest punctuation for {stem}") from exc
        entries.append({"audio": str(audio), "textgrid": str(ctc_dir / f"{stem}.TextGrid"),
                        "lab": str(lab), "duration_s": duration, "n_words": len(rows),
                        "n_punct": punct_count,
                        "_words": [{"word": row["word"], "start": row["start_s"], "end": row["end_s"]} for row in rows]})
    temporary = ctc_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, ctc_dir / "manifest.json")


def _merge_ria_tokens(tokens_path: Path) -> bool:
    """合并 tokens.jsonl 中相邻的 ruiN + yaN 条目为单个 ria 条目."""
    try:
        lines = tokens_path.read_text(encoding="utf-8").strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]
    except Exception:
        return False
    if len(entries) < 2:
        return False

    new_entries, i, changed = [], 0, False
    while i < len(entries):
        w = entries[i]["word"]
        if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
                and re.match(r'^(ya|a)[0-5]$', entries[i + 1]["word"])):
            a, b = entries[i], entries[i + 1]
            new_entries.append({
                "word": "ria",
                "start_ms": a["start_ms"], "end_ms": b["end_ms"],
                "start_s": a["start_s"], "end_s": b["end_s"],
                "type": a.get("type", "word"),
            })
            i += 2
            changed = True
        else:
            new_entries.append(entries[i])
            i += 1

    if changed:
        tokens_path.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in new_entries) + "\n",
            encoding="utf-8")
    return changed


def _protect_ria(words_pinyin: list[dict]) -> list[dict]:
    """Ensure "ria" (VTuber name) is always a complete, lowercase, standalone token.

    The NVASR tokenizer can split OOV "ria" into several patterns:
      - "R" + "I" + "A"  (single letters → handled by single-letter merge)
      - "R" + "ia"        (mixed — single-letter merge misses this)
      - "RIA"             (already merged but uppercase)
      - "rui4" + "ya4"    (pinyin phonetic — also handled by _normalize_ria)

    This function scans for ANY adjacent tokens whose lowercase letters
    form "ria" and merges them into a single lowercase "ria" token.
    It also normalises standalone "RIA"/"Ria" → "ria".
    """
    if not words_pinyin:
        return words_pinyin

    _RIA_TARGET = "ria"
    _RIA_LETTERS = frozenset(_RIA_TARGET)  # {'r', 'i', 'a'}

    result: list[dict] = []
    i = 0
    while i < len(words_pinyin):
        w = words_pinyin[i]
        token = w["word"]
        token_lower = token.lower()

        # Already correct
        if token == _RIA_TARGET:
            result.append(w)
            i += 1
            continue

        # Standalone case-fix: "RIA" / "Ria" → "ria"
        if token_lower == _RIA_TARGET:
            result.append({**w, "word": _RIA_TARGET})
            i += 1
            continue

        # Fragment detection: collect consecutive fragments whose
        # letters are a subset of "ria"'s letter set.
        if (len(token) <= 4 and token.isascii()
                and all(c.lower() in _RIA_LETTERS for c in token if c.isalpha())):
            fragments = [w]
            j = i + 1
            while j < len(words_pinyin):
                nt = words_pinyin[j]
                nt_word = nt["word"]
                if (len(nt_word) <= 4 and nt_word.isascii()
                        and all(c.lower() in _RIA_LETTERS for c in nt_word if c.isalpha())):
                    fragments.append(nt)
                    j += 1
                else:
                    break

            # Check whether the collected fragments form "ria"
            combined = "".join(f["word"] for f in fragments).lower()
            if combined == _RIA_TARGET or all(
                    c in _RIA_LETTERS for c in combined if c.isalpha()):
                # Merge: one "ria" token spanning the full time range
                result.append({
                    "word": _RIA_TARGET,
                    "start": fragments[0]["start"],
                    "end": fragments[-1]["end"],
                })
                i = j
                continue

        result.append(w)
        i += 1

    return result


def _merge_reference_english_fragments(words_pinyin: list[dict], reference: str) -> list[dict]:
    """Merge reference English units with exact, contiguous CTC fragments.

    ``english_units.py`` owns the spelling and merge contract.  This adapter
    only maps timed CTC rows into that contract and serializes the resulting
    canonical metadata.  It intentionally does not search for a later match:
    the first English row at the current lexical position owns the span, and
    every source ordinal in the candidate must be contiguous.  Consequently
    partial, reordered, punctuation-separated, CJK/NVV-crossing, and extra
    fragment candidates fail closed instead of being repaired heuristically.
    """
    if not reference:
        return list(words_pinyin)

    authorities = parse_english_units(reference)
    english_indexes = [
        index for index, row in enumerate(words_pinyin)
        if is_english_token(str(row.get("word", "")).strip())
    ]
    if not authorities:
        if english_indexes:
            raise ValueError("authority English candidate has no reference unit")
        return list(words_pinyin)

    def source_ordinals(row: dict, fallback: int) -> tuple[int, ...]:
        values = row.get("source_ctc_ordinals")
        if values is None:
            value = row.get("source_ctc_ordinal", fallback)
            values = [value]
        elif isinstance(values, int):
            values = [values]
        if (not isinstance(values, (list, tuple))
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value < 0 for value in values)):
            raise ValueError("invalid source CTC ordinal metadata")
        normalized = tuple(values)
        if not normalized or tuple(sorted(set(normalized))) != normalized:
            raise ValueError("non-monotonic source CTC ordinal metadata")
        return normalized

    def row_fragment(row: dict, fallback: int) -> dict:
        text = str(row.get("word", "")).strip().lstrip("▁")
        ordinals = source_ordinals(row, fallback)
        fragment = {"text": text, "ordinal": ordinals[0]}
        if len(ordinals) != 1:
            # A pre-merged row is not one CTC fragment and cannot be used as
            # an authority candidate.  It must be re-emitted from raw rows.
            raise ValueError("pre-merged source CTC row in authority candidate")
        if "start" not in row or "end" not in row:
            raise ValueError("authority CTC candidate is missing timing")
        fragment.update(start=float(row["start"]), end=float(row["end"]))
        return fragment

    def content_hash(value: object) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    reference_identity = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    result = list(words_pinyin)
    cursor = 0
    for authority in authorities:
        start = next((index for index in english_indexes if index >= cursor), None)
        if start is None:
            raise ValueError(f"missing authority English unit {authority.surface_text}")

        # Collect the smallest source-ordered candidate whose lexical ASCII
        # length reaches the authority token.  Non-English rows are retained
        # in the candidate so CJK/NVV crossings are rejected by the shared
        # validator rather than skipped.
        candidate_rows: list[dict] = []
        compact_length = 0
        end = start
        while end < len(result):
            row = result[end]
            candidate_rows.append(row)
            token = str(row.get("word", "")).strip().lstrip("▁")
            if is_english_token(token):
                compact_length += len(token.replace("-", ""))
            end += 1
            if compact_length >= len(authority.alignment_token):
                break
        if compact_length < len(authority.alignment_token):
            raise ValueError(f"partial authority English unit {authority.surface_text}")

        fragments = [row_fragment(row, index)
                     for index, row in enumerate(candidate_rows, start)]
        try:
            merged = merge_authority_fragment_group(authority, fragments)
        except EnglishUnitError as exc:
            raise ValueError(
                f"authority English unit {authority.surface_text}: {exc.code}") from exc

        canonical = merged.to_dict()
        canonical_json_hash = content_hash(canonical)
        merged_row = dict(candidate_rows[0])
        canonical_start = merged.canonical_start
        canonical_end = merged.canonical_end
        if not isinstance(canonical_start, (int, float)):
            canonical_start = float(candidate_rows[0]["start"])
        if not isinstance(canonical_end, (int, float)):
            canonical_end = float(candidate_rows[-1]["end"])
        merged_row.update({
            "word": merged.alignment_token,
            "surface_text": merged.surface_text,
            "start": float(canonical_start),
            "end": float(canonical_end),
            "source_ctc_ordinals": list(merged.source_ctc_ordinals),
            "canonical_span": [merged.canonical_start, merged.canonical_end],
            "canonical_unit": canonical,
            "canonical_unit_sha256": canonical_json_hash,
            "reference_identity": reference_identity,
            "reference_ordinal": merged.reference_ordinal,
        })
        if ("-" in authority.surface_text
                and any(right > left + 1
                        for left, right in zip(merged.source_ctc_ordinals,
                                              merged.source_ctc_ordinals[1:]))
                and all(re.fullmatch(r"[A-Za-z]+(?:[0-9]+)?", str(row.get("word", "")).strip())
                        for row in candidate_rows)):
            merged_row["hyphen_separator_omitted"] = True
        del result[start:end]
        result.insert(start, merged_row)
        cursor = start + 1
        # The index list is based on the original row list.  Recompute it
        # after each contraction so a second authority unit cannot consume an
        # old position or silently skip an extra English candidate.
        english_indexes = [
            index for index, row in enumerate(result)
            if is_english_token(str(row.get("word", "")).strip())
        ]
    remaining = [
        row for row in result[cursor:]
        if is_english_token(str(row.get("word", "")).strip())
    ]
    if remaining:
        raise ValueError("extra authority English fragment")
    return result


def _normalize_ria(ctc_dir: Path) -> int:
    """ASR 后处理安全网: 修复旧 CTC 输出中 ria 的拼音碎片和 CTC 锚点.

    新数据已在 align_text 上通过 replace_ria_variants() 实时处理,
    .lab / _tokens.jsonl 从源头就是 ria. 此函数覆盖旧版 pipeline 输出:
      1. 修复 .lab:          rui4 ya4 → ria
      2. 合并 _tokens.jsonl: rui4 + ya4 → ria (含 CTC 时间戳)

    _text_cn.txt 和 _text_raw.txt 保持原样 (ASR 原始输出存档).

    使用 rglob 同时支持 flat 和嵌套目录结构.
    """
    lab_changed = 0
    tokens_changed = 0

    for lab_file in sorted(ctc_dir.rglob("*.lab")):
        try:
            lab_text = lab_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue

        new_lab = re.sub(r'rui[0-5]\s+ya[0-5]', 'ria', lab_text)
        new_lab = re.sub(r'rui[0-5]\s+a[0-5]', 'ria', new_lab)

        if new_lab != lab_text:
            lab_file.write_text(new_lab + "\n", encoding="utf-8")
            lab_changed += 1

            # 同步合并 _tokens.jsonl
            tokens_path = lab_file.with_suffix(".jsonl")
            if not tokens_path.exists():
                tokens_path = lab_file.parent / f"{lab_file.stem}_tokens.jsonl"
            if tokens_path.exists() and _merge_ria_tokens(tokens_path):
                tokens_changed += 1

    if lab_changed:
        print(f"  [normalize_ria] {lab_changed} .lab + {tokens_changed} tokens.jsonl (safety net)")
    return lab_changed


def _normalize_english(ctc_dir: Path, dict_path: Path | None = None,
                       update_dict: bool = True) -> int:
    """英文 token 碎片合并 (rui4+ya4 → ria), 同步更新 .lab / .TextGrid / _tokens.jsonl."""
    try:
        from normalize_english_tokens import normalize_stem
    except ImportError:
        print("  [normalize_en] normalize_english_tokens not found, skipping")
        return 0

    stems = set()
    for f in ctc_dir.glob("*_text_cn.txt"):
        stems.add(f.name.replace("_text_cn.txt", ""))
    if not stems:
        for f in ctc_dir.glob("*.lab"):
            if (ctc_dir / f"{f.stem}_text_cn.txt").exists():
                stems.add(f.stem)

    changed = 0
    for stem in sorted(stems):
        if normalize_stem(ctc_dir, stem, dry_run=False):
            changed += 1

    if changed:
        print(f"  [normalize_en] {changed} files")

    # Auto-add English tokens to MFA dictionary
    if update_dict and dict_path and dict_path.exists() and changed:
        english_tokens_found: set[str] = set()
        for lab_path in sorted(ctc_dir.glob("*.lab")):
            tokens = lab_path.read_text(encoding="utf-8").strip().split()
            for t in tokens:
                if is_english_token(t):
                    english_tokens_found.add(t)
        if english_tokens_found:
            existing = set()
            with open(dict_path, encoding='utf-8-sig') as f:
                for line in f:
                    if line.strip():
                        existing.add(line.split()[0])
            new_tokens = sorted(t for t in english_tokens_found if t not in existing)
            if new_tokens:
                with open(dict_path, 'a', encoding='utf-8') as f:
                    for t in new_tokens:
                        f.write(f"{t} {t}\n")
                print(f"  [normalize_en] Added {len(new_tokens)} tokens to MFA dict: "
                      f"{', '.join(new_tokens)}")

    return changed


def _plan_all_gpu_shard(output_dir: Path, gpu_id: int, limit: int) -> tuple[Path, bool, bool]:
    """Plan a shard directory without filesystem mutation.

    Returns ``(path, reused, recovered)``.  ``recovered`` denotes a clean
    staging replacement for an incomplete ``_shard_gpuN``.  An existing
    staging directory is treated as ambiguous active/stale ownership and is
    rejected; callers must resolve it explicitly.
    """
    canonical = output_dir / f"_shard_gpu{gpu_id}"
    if not canonical.exists():
        return canonical, False, False

    existing_labs = len(list(canonical.glob("*.lab")))
    if existing_labs >= limit:
        return canonical, True, False

    staging = output_dir / f"_shard_gpu{gpu_id}_staging"
    if staging.exists():
        raise RuntimeError(
            "incomplete shard and existing recovery staging directory require "
            f"explicit operator resolution: {canonical} ({existing_labs}/{limit} labs), "
            f"{staging}"
        )
    return staging, False, True


def _plan_all_gpu_shards(
    output_dir: Path, shard_specs: list[tuple[int, int]]
) -> list[tuple[int, int, Path, bool, bool]]:
    """Read-only plan for every shard; no earlier plan may mutate state."""
    plans: list[tuple[int, int, Path, bool, bool]] = []
    for gpu_id, limit in shard_specs:
        shard_dir, reused, recovered = _plan_all_gpu_shard(output_dir, gpu_id, limit)
        plans.append((gpu_id, limit, shard_dir, reused, recovered))
    return plans


def _prepare_dictionary_candidate(
    dict_path: Path | None,
    merged_manifest: list[dict],
    *,
    no_update: bool,
) -> tuple[Path | None, list[str]]:
    """Build and validate a dictionary candidate without touching the live file."""
    if no_update or not dict_path or not dict_path.is_file():
        return None, []
    english_tokens = sorted({
        word["word"]
        for entry in merged_manifest
        for word in entry.get("_words", [])
        if isinstance(word, dict) and is_english_token(str(word.get("word", "")))
    })
    existing: set[str] = set()
    with open(dict_path, encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                existing.add(line.split()[0])
    new_tokens = [token for token in english_tokens if token not in existing]
    if not new_tokens:
        return None, []

    candidate = dict_path.with_name(
        f".{dict_path.name}.candidate-{os.getpid()}"
    )
    if candidate.exists() or candidate.is_symlink():
        raise RuntimeError(f"dictionary candidate already exists: {candidate}")
    shutil.copy2(str(dict_path), str(candidate))
    with open(candidate, "a", encoding="utf-8") as handle:
        for token in new_tokens:
            handle.write(f"{token} {token}\n")
    candidate_words = load_mfa_word_set(candidate) or set()
    if not set(existing).issubset(candidate_words) \
            or not set(new_tokens).issubset(candidate_words):
        raise RuntimeError(f"dictionary candidate validation failed: {candidate}")
    return candidate, new_tokens


def _commit_all_gpu_candidate(
    live_output: Path,
    candidate_output: Path,
    old_output_backup: Path,
    dict_path: Path | None,
    dict_candidate: Path | None,
) -> tuple[Path, Path | None]:
    """Commit output and dictionary as a recoverable pair.

    The two filesystem namespaces cannot be made one OS-level atomic rename.
    Keep the old output and dictionary as recoverable backups and quarantine
    the new candidate on any failure, restoring the old pair before raising.
    """
    dict_backup = (dict_path.with_name(
        f".{dict_path.name}.previous-{os.getpid()}"
    ) if dict_path and dict_candidate else None)
    if old_output_backup.exists() or old_output_backup.is_symlink():
        raise RuntimeError(f"output backup already exists: {old_output_backup}")
    if dict_backup and (dict_backup.exists() or dict_backup.is_symlink()):
        raise RuntimeError(f"dictionary backup already exists: {dict_backup}")

    output_moved = False
    output_published = False
    dictionary_moved = False
    dictionary_published = False
    failed_output = candidate_output.with_name(
        f"{candidate_output.name}.FAILED-{os.getpid()}"
    )
    failed_dictionary = (dict_candidate.with_name(
        f"{dict_candidate.name}.FAILED-{os.getpid()}"
    ) if dict_candidate else None)
    try:
        os.replace(live_output, old_output_backup)
        output_moved = True
        os.replace(candidate_output, live_output)
        output_published = True
        if dict_candidate and dict_path and dict_backup:
            os.replace(dict_path, dict_backup)
            dictionary_moved = True
            os.replace(dict_candidate, dict_path)
            dictionary_published = True
        return old_output_backup, dict_backup
    except Exception:
        # Preserve every newly-created artifact; never delete user data.
        if output_published and live_output.exists():
            if failed_output.exists() or failed_output.is_symlink():
                raise RuntimeError(
                    f"cannot quarantine failed output candidate: {failed_output}"
                )
            os.replace(live_output, failed_output)
        if output_moved and old_output_backup.exists() and not live_output.exists():
            os.replace(old_output_backup, live_output)

        if dictionary_published and dict_path and failed_dictionary:
            if dict_path.exists():
                os.replace(dict_path, failed_dictionary)
        if dictionary_moved and dict_path and dict_backup \
                and dict_backup.exists() and not dict_path.exists():
            os.replace(dict_backup, dict_path)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="CTC Pre-alignment: NVASR → MFA anchor TextGrids (pinyin)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="原始数据目录 (含 wav + 中文 txt)")
    parser.add_argument("--pinyin-dir", type=Path, required=True,
                        help="拼音语料目录 (用于 fallback)")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="处理后的音频目录 (trim 输出), 默认同 data-dir")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "workspace" / "ctc_pretg")
    parser.add_argument("--model-path", type=str,
                        default=str(PROJECT_ROOT / "models" / "Multilingual-NVASR"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dict-path", type=Path, default=None,
                        help="MFA 词典路径 (用于过滤标点等非词条)")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理数量, 0=全部")
    parser.add_argument("--offset", type=int, default=0,
                        help="跳过前 N 个文件 (配合 --limit 实现多 GPU 分片)")
    parser.add_argument("--stems-file", type=Path, default=None,
                        help="Frozen eligible stem manifest; disables implicit slicing")
    parser.add_argument("--all-gpus", action="store_true",
                        help="自动检测所有 GPU 并均匀分片并行处理")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-fresh-output", action="store_true",
                        help="Fail before GPU/model work if --output-dir already exists (v4 gate).")
    parser.add_argument("--nvv-bias", type=float, default=NVV_BIAS_DEFAULT,
                        help=f"NVV blank-frame bias (default: {NVV_BIAS_DEFAULT}).")
    parser.add_argument("--no-nvv", action="store_true",
                        help="禁用 NVV 标签检测, 仅用 CTC 锚点给参考文本做时间戳.")
    parser.add_argument("--allow-missing-reference", action="store_true",
                        help="无参考文本全管线：所有 WAV 进入 NVASR，自由 ASR 文本作为 MFA 语料.")
    parser.add_argument("--reference-mode", choices=("auto", "authority", "fallback"),
                        default="auto",
                        help="Transcript authority policy; fallback ignores reference TXT files.")
    parser.add_argument("--no-dict-update", action="store_true",
                        help="Do not append discovered English tokens to --dict-path.")
    args = parser.parse_args()
    if args.reference_mode == "authority" and args.allow_missing_reference:
        print("ERROR: reference-mode=authority conflicts with --allow-missing-reference",
              file=sys.stderr)
        return 2
    if args.reference_mode == "fallback" and args.no_nvv:
        print("ERROR: reference-mode=fallback requires NVASR free decode (omit --no-nvv)",
              file=sys.stderr)
        return 2
    # The explicit policy is the source of truth; retain the legacy flag as a
    # compatible command-line alias for auto mode.
    if args.reference_mode == "fallback":
        args.allow_missing_reference = True
    if args.require_fresh_output and args.output_dir.exists():
        print(f"ERROR: --require-fresh-output refuses existing output: {args.output_dir}", file=sys.stderr)
        return 2

    # ── Model tree provenance (Case 99 / R5) ──────────────────────────
    # Compute once at startup so both all-GPU parent and single-GPU child
    # paths share the same frozen identity.
    from pipeline_utils import (compute_model_tree_digest,
                                write_ctc_run_receipt,
                                validate_ctc_receipts_same_identity)
    _model_path = Path(args.model_path).resolve()
    if not _model_path.is_dir():
        print(f"ERROR: model path is not a directory: {_model_path}", file=sys.stderr)
        return 2
    _model_tree_digest, _model_file_manifest = compute_model_tree_digest(_model_path)
    _dict_digest = hashlib.sha256(
        Path(args.dict_path).read_bytes() if args.dict_path and args.dict_path.is_file()
        else b""
    ).hexdigest() if args.dict_path else ""

    # ── --all-gpus: auto-detect GPUs, split files, launch parallel subprocesses ──
    if args.all_gpus:
        if not torch.cuda.is_available():
            print("ERROR: --all-gpus requires CUDA. Falling back to single-GPU mode.")
            args.all_gpus = False
        else:
            import subprocess as _sp
            import shutil as _shutil

            num_gpus = torch.cuda.device_count()

            # Detect the correct Python for subprocesses.
            # sys.executable might be a base conda Python without pypinyin/cn2an.
            # Prefer the Python that has torch + ctc_prealign dependencies.
            _child_python = sys.executable
            try:
                import pypinyin  # noqa: F401
            except ImportError:
                # Current Python lacks pypinyin — try common ASR env paths
                _candidates = [
                    Path.home() / "miniconda3/envs/asr/bin/python",
                    Path.home() / "miniconda3/envs/asr/bin/python3",
                ]
                for _c in _candidates:
                    if _c.exists():
                        _child_python = str(_c)
                        print(f"  Note: using {_child_python} for subprocesses"
                              f" (current Python lacks pypinyin)")
                        break
                else:
                    print("  WARNING: pypinyin not available. English token"
                          " normalization may be skipped.")

            audio_dir = args.audio_dir or args.data_dir
            # Freeze the complete source WAV universe before any reference
            # filtering.  Children receive only eligible stems; the parent
            # publishes the single authoritative v2 receipt after merge.
            _source_wavs = sorted(audio_dir.rglob("*.wav"))
            _txt_index = _build_txt_index(args.data_dir)
            try:
                all_wavs, _all_ref_texts, _all_exclusions = _source_inventory(
                    _source_wavs, args.data_dir, _txt_index,
                    args.allow_missing_reference, args.reference_mode)
            except ValueError as _exc:
                print(f"ERROR: {_exc}", file=sys.stderr)
                return 2
            if _all_exclusions:
                print(f"  Frozen source: {len(_source_wavs)} WAVs; "
                      f"eligible={len(all_wavs)}, exclusions={len(_all_exclusions)}")
            # Apply operator offset/limit to the frozen eligible denominator
            # before shard construction, never inside each child.
            if args.offset > 0:
                all_wavs = all_wavs[args.offset:]
            if args.limit > 0:
                all_wavs = all_wavs[:args.limit]
            total = len(all_wavs)
            if total == 0:
                print("ERROR: no eligible WAVs with authoritative references", file=sys.stderr)
                return 2
            per_gpu = (total + num_gpus - 1) // num_gpus
            print(f"--all-gpus: {num_gpus} GPUs detected, {total} WAVs → ~{per_gpu}/GPU")

            _procs: list[tuple[int, _sp.Popen, Path]] = []

            # Build base argv from parsed namespace
            _base_argv = [
                _child_python, __file__,
                "--data-dir", str(args.data_dir),
                "--pinyin-dir", str(args.pinyin_dir),
                "--model-path", str(args.model_path),
                "--nvv-bias", str(args.nvv_bias),
            ]
            if args.audio_dir:
                _base_argv += ["--audio-dir", str(args.audio_dir)]
            if args.no_nvv:
                _base_argv += ["--no-nvv"]
            if args.allow_missing_reference:
                _base_argv += ["--allow-missing-reference"]
            _base_argv += ["--reference-mode", args.reference_mode]
            if args.no_dict_update:
                _base_argv += ["--no-dict-update"]

            _shard_specs = [
                (gpu_id, min(per_gpu, total - gpu_id * per_gpu))
                for gpu_id in range(num_gpus)
                if min(per_gpu, total - gpu_id * per_gpu) > 0
            ]
            try:
                _shard_plans = _plan_all_gpu_shards(args.output_dir, _shard_specs)
            except RuntimeError as _exc:
                print(f"ERROR: {_exc}", file=sys.stderr)
                return 2

            # Materialize and launch only after every GPU's read-only plan
            # succeeds.  A race that changes a planned fresh path is loud and
            # leaves all prior plans untouched.
            for gpu_id, limit, shard_dir, _reused, _recovered in _shard_plans:
                offset = gpu_id * per_gpu
                if _reused:
                    _existing_labs = len(list(shard_dir.glob("*.lab")))
                    print(f"  GPU {gpu_id}: reuse existing shard ({_existing_labs} labs)")
                    _procs.append((gpu_id, None, shard_dir))
                    continue
                if shard_dir.exists():
                    print(f"ERROR: planned fresh shard path became occupied: {shard_dir}",
                          file=sys.stderr)
                    return 2
                shard_dir.mkdir(parents=True)
                if _recovered:
                    _legacy = args.output_dir / f"_shard_gpu{gpu_id}"
                    _existing_labs = len(list(_legacy.glob("*.lab")))
                    print(
                        f"  GPU {gpu_id}: incomplete legacy shard preserved "
                        f"({_legacy}, {_existing_labs}/{limit} labs); "
                        f"clean recovery staged at {shard_dir}"
                    )

                child_argv = list(_base_argv)
                child_argv += [
                    "--device", "cuda:0",
                    "--output-dir", str(shard_dir),
                ]
                shard_manifest = shard_dir / "selected_stems.txt"
                shard_stems = [p.stem for p in all_wavs[offset:offset + limit]]
                shard_manifest.write_text("\n".join(sorted(shard_stems)) + "\n", encoding="utf-8")
                child_argv += ["--stems-file", str(shard_manifest)]
                # Copy dict to shard dir (avoids concurrent write races on shared dict)
                if args.dict_path:
                    shard_dict = shard_dir / args.dict_path.name
                    if not shard_dict.exists():
                        _shutil.copy2(str(args.dict_path), str(shard_dict))
                    child_argv += ["--dict-path", str(shard_dict)]

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = cuda_visible_token(gpu_id, env)
                print(f"  GPU {gpu_id}: offset={offset} limit={limit} "
                      f"→ {shard_dir}")
                _proc = _sp.Popen(child_argv, env=env, cwd=str(PROJECT_ROOT))
                _procs.append((gpu_id, _proc, shard_dir))

            # Wait for all GPUs
            print(f"\n  等待 {len(_procs)} 个 GPU 完成...")
            failed: list[tuple[int, int]] = []
            for gpu_id, _proc, _shard_dir in _procs:
                if _proc is None:
                    print(f"  GPU {gpu_id}: REUSED")
                    continue
                _rc = _proc.wait()
                if _rc != 0:
                    failed.append((gpu_id, _rc))
                    print(f"  GPU {gpu_id}: FAILED (rc={_rc})")
                else:
                    print(f"  GPU {gpu_id}: DONE")

            if failed:
                print(f"\n  FAILURES: {len(failed)} GPU(s)")
                for gpu_id, rc in failed:
                    print(f"    GPU {gpu_id}: rc={rc}")
                sys.exit(1)

            # ── Preflight every shard before touching the parent namespace ──
            # This is the transaction boundary for all-GPU mode: no shard file
            # may be moved until every shard has an exact namespace, manifest,
            # summary and mutually exclusive expected stem set.
            import json as _json
            _artifact_suffixes = [".TextGrid", ".lab", "_tokens.jsonl",
                                  "_punct.json", "_text_cn.txt", "_text_raw.txt"]
            if args.no_nvv:
                _artifact_suffixes.append("_ref.txt")
            _preflight_shards = []
            _seen_shard_stems: set[str] = set()
            _expected_all_stems: set[str] = set()
            _selected_manifest_digests: dict[str, str] = {}
            for _gpu_id, _, _shard_dir in _procs:
                _start = _gpu_id * per_gpu
                _expected = {p.stem for p in all_wavs[_start:_start + per_gpu]}
                _expected_all_stems |= _expected
                _allowed = {s + suffix for s in _expected for suffix in _artifact_suffixes}
                _allowed |= {"manifest.json", "summary.txt", ".ctc_normalized",
                             ".ctc_run_receipt.json",
                             ".pipeline_run_receipt_v2.json", "selected_stems.txt"}
                if args.dict_path:
                    _allowed.add(args.dict_path.name)
                _files = list(_shard_dir.iterdir())
                if any(p.is_symlink() or not p.is_file() for p in _files):
                    raise RuntimeError(f"shard contains non-regular artifact: {_shard_dir}")
                if {p.name for p in _files} != _allowed:
                    raise RuntimeError(f"shard namespace mismatch: {_shard_dir}")
                _selected_path = _shard_dir / "selected_stems.txt"
                try:
                    _selected_manifest_digests[_shard_dir.name] = validate_selected_stems_manifest(
                        _selected_path, _expected)
                except (OSError, ValueError) as _exc:
                    raise RuntimeError(str(_exc)) from _exc
                try:
                    validate_shard_accounting_receipt(
                        _shard_dir / ".pipeline_run_receipt_v2.json",
                        _expected,
                        {p.stem for p in all_wavs},
                    )
                except (OSError, ValueError) as _exc:
                    raise RuntimeError(str(_exc)) from _exc
                try:
                    _shard_manifest = _json.loads(
                        (_shard_dir / "manifest.json").read_text(encoding="utf-8"))
                except Exception as _exc:
                    raise RuntimeError(f"invalid shard manifest: {_shard_dir}") from _exc
                if not isinstance(_shard_manifest, list):
                    raise RuntimeError(f"shard manifest is not a list: {_shard_dir}")
                _manifest_stems = []
                for _entry in _shard_manifest:
                    if not isinstance(_entry, dict):
                        raise RuntimeError(f"invalid shard manifest entry: {_shard_dir}")
                    _audio_stem = Path(str(_entry.get("audio", ""))).stem
                    _manifest_stems.append(_audio_stem)
                    if _audio_stem not in _expected:
                        raise RuntimeError(f"shard manifest stem mismatch: {_audio_stem}")
                    for _key in ("textgrid", "lab"):
                        _path = Path(str(_entry.get(_key, "")))
                        if _path.name not in _allowed or not (_shard_dir / _path.name).is_file():
                            raise RuntimeError(f"shard manifest artifact mismatch: {_entry}")
                if _manifest_stems != sorted(_expected) or len(_manifest_stems) != len(set(_manifest_stems)):
                    raise RuntimeError(f"shard manifest stem set mismatch: {_shard_dir}")
                _summary_text = (_shard_dir / "summary.txt").read_text(encoding="utf-8")
                _summary_match = re.search(
                    r"^Files:\s+(\d+)\s+total,\s+(\d+)\s+OK,\s+(\d+)\s+failed$",
                    _summary_text, re.MULTILINE)
                if not _summary_match or tuple(map(int, _summary_match.groups())) != (len(_expected), len(_expected), 0):
                    raise RuntimeError(f"shard summary mismatch: {_shard_dir}")
                _marker_text = (_shard_dir / ".ctc_normalized").read_text(encoding="utf-8")
                # Accept v3 (legacy) or v4 (content-identity) marker.
                _marker_ok = (
                    _marker_text == CTC_NORMALIZATION_MARKER
                    or parse_ctc_normalization_marker(_marker_text) is not None
                )
                if not _marker_ok:
                    raise RuntimeError(f"shard normalization marker mismatch: {_shard_dir}")
                # ── Shard receipt validation (Case 99 / R5) ──────────
                _receipt_path = _shard_dir / ".ctc_run_receipt.json"
                if not _receipt_path.is_file():
                    raise RuntimeError(f"shard missing run receipt: {_shard_dir}")
                try:
                    _receipt = _json.loads(_receipt_path.read_text(encoding="utf-8"))
                except Exception as _exc:
                    raise RuntimeError(f"invalid shard run receipt: {_shard_dir}") from _exc
                _receipt_model = _receipt.get("model", {}).get("tree_digest", "")
                _receipt_dict = _receipt.get("dictionary", {}).get("digest", "")
                if _receipt_model != _model_tree_digest:
                    raise RuntimeError(
                        f"shard model tree digest mismatch: "
                        f"{_receipt_model!r} != parent {_model_tree_digest!r}"
                    )
                if _receipt_dict != _dict_digest:
                    raise RuntimeError(
                        f"shard dict digest mismatch: "
                        f"{_receipt_dict!r} != parent {_dict_digest!r}"
                    )
                # ───────────────────────────────────────────────────────
                if _seen_shard_stems & _expected:
                    raise RuntimeError(f"duplicate shard stem set: {_shard_dir}")
                _seen_shard_stems |= _expected
                _preflight_shards.append((_shard_dir, _shard_manifest))
            if _seen_shard_stems != _expected_all_stems or _expected_all_stems != {p.stem for p in all_wavs}:
                raise RuntimeError("all-GPU shard stem union is not exact")
            for _shard_dir, _ in _preflight_shards:
                for _f in _shard_dir.iterdir():
                    if _f.name in {"manifest.json", "summary.txt",
                                   ".ctc_normalized", ".ctc_run_receipt.json",
                                   ".pipeline_run_receipt_v2.json", "selected_stems.txt"}:
                        continue
                    if (args.output_dir / _f.name).exists() or (args.output_dir / _f.name).is_symlink():
                        raise FileExistsError(f"all-GPU target collision: {args.output_dir / _f.name}")

            # ── Merge shard outputs into an isolated publish candidate ──
            # Never move a shard artifact into the live output namespace.  A
            # later validation failure must leave the original output path
            # untouched and recoverable.
            _merge_output_dir = args.output_dir.parent / (
                f".{args.output_dir.name}.merge-{os.getpid()}"
            )
            if _merge_output_dir.exists():
                raise RuntimeError(
                    f"merge staging already exists: {_merge_output_dir}"
                )
            _merge_output_dir.mkdir(parents=False)
            _merge_lock = args.output_dir / ".merge_lock"
            try:
                _lock_fd = os.open(
                    str(_merge_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                os.close(_lock_fd)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"all-GPU merge lock exists — another merge may be in progress: {_merge_lock}"
                ) from exc
            try:
                print(f"\n  合并 {len(_procs)} 个 shard 输出...")
                _merged_entries = 0
                _total_files = _total_ok = _total_fail = _total_time = 0
                _all_manifests: list[tuple[Path, Path]] = []
                _all_summaries: list[Path] = []

                for _gpu_id, _, _shard_dir in _procs:
                    for _f in _shard_dir.glob("*"):
                        if _f.name == "manifest.json":
                            _all_manifests.append((_shard_dir, _f))
                        elif _f.name == "summary.txt":
                            _all_summaries.append(_f)
                        elif _f.name == ".ctc_normalized":
                            pass  # parent marker is published last, after full validation
                        elif _f.name == ".ctc_run_receipt.json":
                            pass  # parent receipt is published last, after full validation
                        elif _f.name == ".pipeline_run_receipt_v2.json":
                            pass  # validated child evidence remains quarantined
                        elif _f.name == "selected_stems.txt":
                            pass  # shard denominator metadata remains quarantined
                        elif args.dict_path and _f.name == args.dict_path.name:
                            pass  # skip shard-local dict copy
                        else:
                            _dest = _merge_output_dir / _f.name
                            if _dest.exists() or _dest.is_symlink():
                                raise FileExistsError(f"all-GPU target collision: {_dest}")
                            _shutil.copy2(str(_f), str(_dest))
                            _merged_entries += 1

                # ── Merge manifests with path rewriting (BEFORE shard cleanup) ──
                import json as _json
                _merged = []
                for _shard_dir, _m in sorted(_all_manifests, key=lambda p: p[1].parent.name):
                    _data = _json.loads(_m.read_text(encoding="utf-8"))
                    _shard_s = str(_shard_dir)
                    _main_s = str(args.output_dir)
                    for _entry in _data:
                        for _key in ("textgrid", "lab"):
                            if _key in _entry:
                                _entry[_key] = _entry[_key].replace(_shard_s, _main_s)
                    _merged.extend(_data)

                # Extract stats from per-shard summaries
                for _s in _all_summaries:
                    for _line in _s.read_text(encoding="utf-8").splitlines():
                        if _line.startswith("Files:"):
                            _p = _line.split()
                            _total_files += int(_p[1])
                            _total_ok += int(_p[3])
                            _total_fail += int(_p[5])
                        elif _line.startswith("Time:"):
                            _total_time += float(_line.split()[1].rstrip("s"))

                # Write combined summary
                _summary = (
                    f"CTC Pre-alignment Report (--all-gpus, {num_gpus}x GPU)\n"
                    f"{'=' * 40}\n"
                    f"Files: {_total_files} total, {_total_ok} OK, {_total_fail} failed\n"
                    f"Time: {_total_time:.1f}s (wall-clock total)\n\n"
                    f"Output: {args.output_dir}\n"
                )
                # summary/manifest are published only after final bundle validation.

                # Keep the shard evidence under the quarantined partial tree
                # until the final publish is complete.  This is useful for
                # diagnosing a failed GPU without mutating the old evidence.

                print(f"  Merged {len(_merged)} manifest entries, {_merged_entries} files")
                print(f"\n{_summary}")

                # ── Write normalization marker ──
                # Tells run_pipeline.py that normalize_* steps were already done
                # by ctc_prealign. pad_silence only shifts timestamps, doesn't
                # change token content, so re-normalizing is redundant.
                if not _validate_all_ctc_bundles(_merge_output_dir, _all_ref_texts):
                    print("ERROR: refusing to write CTC normalization marker")
                    sys.exit(1)
                _rebuild_final_manifest(_merge_output_dir, audio_dir,
                                        wav_files=all_wavs)
                (_merge_output_dir / "summary.txt.tmp").write_text(_summary, encoding="utf-8")
                os.replace(_merge_output_dir / "summary.txt.tmp", _merge_output_dir / "summary.txt")
                _stem_count = len(list(_merge_output_dir.glob("*.lab")))
                _manifest_digest = hashlib.sha256(
                    (_merge_output_dir / "manifest.json").read_bytes()).hexdigest()
                (_merge_output_dir / ".ctc_normalized").write_text(
                    make_ctc_normalization_marker(_stem_count, _manifest_digest),
                    encoding="utf-8",
                )
                # ── Parent run receipt (Case 99 / R5) ──────────────────
                # The CTC receipt describes the eligible/processed audio
                # axis, not the frozen source universe.  The latter may
                # include reference-less exclusions and is recorded in the
                # pipeline accounting receipt below.  Binding the receipt to
                # _all_stems would make ctc_ready reject a valid partial
                # reference run because its audio_bindings cannot match the
                # eligible CTC output set.
                _all_stems = sorted({p.stem for p in _source_wavs})
                _output_stems_v2 = sorted(p.stem for p in _merge_output_dir.glob("*.lab"))
                from pipeline_utils import _axis_audio_metadata, _textgrid_global_bounds
                _audio_bindings_v2 = []
                for _stem in _output_stems_v2:
                    _wav = audio_dir / f"{_stem}.wav"
                    _xmin, _xmax = _textgrid_global_bounds(
                        _merge_output_dir / f"{_stem}.TextGrid")
                    _audio_bindings_v2.append({
                        "stem": _stem,
                        "path": str(_wav.resolve()),
                        **_axis_audio_metadata(_wav),
                        "ctc_bounds": {"xmin": _xmin, "xmax": _xmax},
                    })
                write_ctc_run_receipt(
                    _merge_output_dir,
                    actual_argv=sys.argv,
                    asr_python=sys.executable,
                    model_path=_model_path,
                    model_tree_digest=_model_tree_digest,
                    model_file_manifest=_model_file_manifest,
                    dict_path=Path(args.dict_path) if args.dict_path else Path(""),
                    dict_digest=_dict_digest,
                    input_stems=_output_stems_v2,
                    output_stems=_output_stems_v2,
                    audio_bindings=_audio_bindings_v2,
                )
                _eligible_stems = sorted(p.stem for p in all_wavs)
                _filtered_stems_v2 = sorted(set(_eligible_stems) - set(_output_stems_v2))
                _shard_rows = [
                    {"shard_id": f"gpu{_gpu_id}", "stems": sorted(
                        p.stem for p in all_wavs[_gpu_id * per_gpu:
                                                  _gpu_id * per_gpu + per_gpu])}
                    for _gpu_id, _, _ in _procs
                ]
                _accounting = make_pipeline_accounting_receipt(
                    source_stems=_all_stems,
                    eligible_stems=_eligible_stems,
                    exclusions=_all_exclusions,
                    output_stems=_output_stems_v2,
                    filtered_stems=_filtered_stems_v2,
                    run_id=make_pipeline_run_id(), mode="ctc_prealign",
                    route=["ctc_prealign", "all_gpus"],
                    paths={"output": str(args.output_dir), "filtered": str(args.output_dir)},
                    shards=_shard_rows,
                    extra={"source_frozen": True,
                           "reference_only": not args.allow_missing_reference,
                           "reference_mode": args.reference_mode,
                           "selected_stems_manifest_sha256": _selected_manifest_digests},
                )
                write_pipeline_accounting_receipt(_merge_output_dir, _accounting)
                _partial_output_dir = args.output_dir.parent / (
                    f"{args.output_dir.name}.partial-{os.getpid()}"
                )
                _dict_candidate, _new_dict_tokens = _prepare_dictionary_candidate(
                    Path(args.dict_path) if args.dict_path else None,
                    _merged,
                    no_update=args.no_dict_update,
                )
                _commit_all_gpu_candidate(
                    live_output=args.output_dir,
                    candidate_output=_merge_output_dir,
                    old_output_backup=_partial_output_dir,
                    dict_path=Path(args.dict_path) if args.dict_path else None,
                    dict_candidate=_dict_candidate,
                )
                if _new_dict_tokens:
                    print(f"  Added {len(_new_dict_tokens)} English tokens to dict")
                print(f"  Published atomically; partial shard evidence retained at {_partial_output_dir}")
                # ─────────────────────────────────────────────────────────
            finally:
                try:
                    _merge_lock.unlink()
                except OSError:
                    pass
            print(f"完成! 输出: {args.output_dir}")
            sys.exit(0)
            sys.exit(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 扫描音频文件 ──
    audio_dir = args.audio_dir or args.data_dir
    # Freeze the complete WAV universe before reference-only prefiltering.
    source_wav_files = sorted(audio_dir.rglob("*.wav"))
    print(f"扫描到 {len(source_wav_files)} 个 WAV 文件")
    txt_index = _build_txt_index(args.data_dir)
    try:
        wav_files, ref_texts, source_exclusions = _source_inventory(
            source_wav_files, args.data_dir, txt_index,
            args.allow_missing_reference, args.reference_mode)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if source_exclusions:
        print(f"  冻结来源: {len(source_wav_files)} WAV; "
              f"eligible={len(wav_files)}, exclusions={len(source_exclusions)}")

    if args.reference_mode == "authority":
        # Reference text is the semantic authority.  Normalize Arabic
        # numerals before CTC words are projected to pinyin; otherwise a
        # surface such as ``target1`` is parsed as one English unit and the
        # CTC ``target``/``1`` evidence cannot be reconciled.  This is an
        # in-memory projection only; source text files and raw CTC evidence
        # remain immutable and are audited separately downstream.
        try:
            import cn2an
            _numeral_transform = cn2an.transform
        except ImportError:
            _numeral_transform = None
        ref_texts = {
            stem: normalize_authority_reference_numerals(
                text, _numeral_transform)
            for stem, text in ref_texts.items()
        }

    # 建立 stem → WAV 路径映射 (支持子目录布局)
    wav_map: dict[str, Path] = {}
    for p in wav_files:
        if p.stem not in wav_map:
            wav_map[p.stem] = p

    # Reference text is authoritative for every eligible stem.  Missing
    # references are explicit exclusions; ASR fallback is intentionally not
    # used for accounting-safe runs.
    skipped: dict[str, list[str]] = {}  # reason → stems (unified skip tracking)
    for _stem, _reason in source_exclusions.items():
        skipped.setdefault(_reason, []).append(_stem)
    print(f"  已索引 {len(txt_index)} 个文本文件")
    if not wav_files:
        print("ERROR: 所有音频均无可用权威参考文本", file=sys.stderr)
        return 2
    if args.stems_file:
        try:
            selected = [line.strip() for line in args.stems_file.read_text(encoding="utf-8").splitlines()]
            if not selected or selected != sorted(set(selected)):
                raise ValueError("stems file must be sorted and unique")
            available = {p.stem: p for p in wav_files}
            missing = [stem for stem in selected if stem not in available]
            if missing:
                raise ValueError(f"stems file contains unavailable stems: {missing[:5]}")
            wav_files = [available[stem] for stem in selected]
        except (OSError, ValueError) as exc:
            print(f"ERROR: invalid --stems-file: {exc}", file=sys.stderr)
            return 2
    # Apply offset/limit AFTER filtering so they count processable stems
    if args.offset > 0:
        wav_files = wav_files[args.offset:]
    if args.limit > 0:
        wav_files = wav_files[:args.limit]
    print(f"  已索引 {len(ref_texts)} 个参考文本, 共 {len(wav_files)} 个音频")

    # ── 加载 NVASR 模型 ──
    print(f"加载 NVASR 模型: {args.model_path}")
    from funasr import AutoModel
    model = AutoModel(model=args.model_path, device=args.device, disable_update=True)
    orig_inf = model.model.inference
    patched = make_patched_inference(ref_texts, args.nvv_bias,
                                      enable_nvv=not args.no_nvv,
                                      reference_only=args.no_nvv)
    model.model.inference = patched.__get__(model.model, type(model.model))

    # ── 处理所有音频文件 (有参考文本用参考, 无则纯靠 ASR) ──
    paths = [str(p) for p in wav_files]
    stems = [p.stem for p in wav_files]
    if not paths:
        print("错误: 没有可处理的音频文件")
        model.model.inference = orig_inf
        return 1

    # ── 批量推理 ──
    t0 = time.time()
    all_results = []

    # Auto-select batch size based on GPU memory (if CUDA available)
    def _detect_batch_size(device: str) -> int:
        if device == "cpu" or not device.startswith("cuda"):
            return 4
        try:
            # Parse numeric index from "cuda:N" string; PyTorch 2.9+
            # renamed total_mem → total_memory
            idx = int(device.split(":", 1)[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(idx)
            mem_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
            mem_gb = mem_bytes / 1024**3
            if mem_gb >= 40:   return 64   # A100, RTX 6000 Ada
            if mem_gb >= 24:   return 32   # RTX 3090/4090
            if mem_gb >= 16:   return 24   # RTX 3080/4080, A4000
            if mem_gb >= 8:    return 16   # RTX 2070/3070
            return 8
        except Exception:
            return 16

    BATCH = _detect_batch_size(args.device)
    print(f"  推理 batch size: {BATCH} (device: {args.device})")
    for bs in range(0, len(paths), BATCH):
        batch = paths[bs:bs + BATCH]
        res = model.generate(input=batch, language="zh", use_itn=True,
                             batch_size_s=min(300, max(60, len(batch) * 30)))
        all_results.extend(res)
        n_done = min(bs + BATCH, len(paths))
        elapsed = time.time() - t0
        speed = n_done / elapsed if elapsed > 0 else 0
        print(f"  推理: {n_done}/{len(paths)} ({speed:.1f} files/s)" if speed > 0
              else f"  推理: {n_done}/{len(paths)}")

    infer_time = time.time() - t0
    print(f"推理完成: {len(all_results)} 结果, {infer_time:.1f}s "
          f"({len(all_results) / infer_time:.1f} files/s)")

    # ── 加载 MFA 词典 (用于过滤) ──
    mfa_words = load_mfa_word_set(args.dict_path)
    if mfa_words:
        print(f"MFA 词典: {len(mfa_words)} 词条 (将过滤非词条token)")

    # ── 拼音映射 + 写 TextGrid ──
    print("生成 TextGrid (拼音映射)...")
    manifest = []
    ok = fail = 0

    input_stems = set(stems)
    seen_result_stems: set[str] = set()
    for i, r in enumerate(all_results):
        # FunASR normally preserves batch order, but the result key is the
        # only reliable ownership relation.  Positional mapping can write
        # one file's CTC anchors under another audio after a reordered or
        # partially failed batch.
        result_key = str(r.get("key", ""))
        # Prefer the exact returned key: Path(...).stem would incorrectly
        # strip an embedded ".wav" from names such as
        # source.wav.tmp_clip0001.wav when FunASR returns the stem
        # source.wav.tmp_clip0001.
        result_stem = ""
        if result_key:
            result_stem = (result_key if result_key in input_stems
                           else Path(result_key).stem)
        if not result_stem or result_stem not in input_stems:
            print(f"  FAIL result[{i}]: invalid/unmatched key {result_key!r}")
            fail += 1
            continue
        if result_stem in seen_result_stems:
            print(f"  FAIL result[{i}]: duplicate result key for {result_stem!r}")
            fail += 1
            continue
        seen_result_stems.add(result_stem)
        stem = result_stem
        words_aligned = r["words"]
        # Encoder frame duration can differ at its endpoint.  Every published
        # artifact is instead clamped to the physical WAV header duration.
        duration_s = _wav_duration_s(wav_map[stem])

        # ── Reject incomplete target alignment before writing any anchor ──
        # A zero-frame target cannot be repaired by shifting the following
        # labels; emitting it would create a text/time mismatch.
        if not r.get("ctc_alignment_complete", True):
            missing = r.get("missing_ctc_tokens", [])
            print(f"  FAIL {stem}: incomplete CTC target alignment - {', '.join(map(str, missing[:12]))}")
            skipped.setdefault("incomplete_ctc_alignment", []).append(stem)
            fail += 1
            continue

        # ── Skip stems with incomplete English fragments ──
        # CTC forced alignment can drop OOV English fragments (e.g.
        # "live"→"li"+"ve" but only "li" gets frames).  These stems
        # would produce incomplete transcriptions — filter them out.
        if not r.get("english_complete", True):
            missing = r.get("missing_english", [])
            print(f"  SKIP {stem}: incomplete English fragments - {', '.join(missing)}")
            skipped.setdefault("incomplete_english", []).append(stem)
            continue

        # 将 CTC 对齐 token 映射到 MFA 词条
        # 策略: 遍历 words_aligned, 检测并合并 [NVV] 模式:
        #   - 遇到 "[" token → 进入 NVV 合并, 收集直到 "]", 输出大写 NVV token
        #   - CJK 单字 → 查 pypinyin → 拼音
        #   - 标点 → 跳过
        #   - 多字符 (英文/数字) → 保留

        # ── token → 统一 words tier (拼音 + NVV, 不含标点) ──
        # 标点不进 MFA: 没有声学实现, 后处理从 CTC 锚点注入
        words_pinyin = []
        punct_entries = []

        def _source_metadata(row: dict) -> dict:
            metadata = {}
            if "source_ctc_ordinals" in row:
                metadata["source_ctc_ordinals"] = list(row["source_ctc_ordinals"])
            elif "source_ctc_ordinal" in row:
                metadata["source_ctc_ordinal"] = row["source_ctc_ordinal"]
            return metadata

        for w in words_aligned:
            token_str = w["word"].strip()
            if not token_str:
                continue
            token_clean = token_str.lstrip("▁")

            # 情况 0: [NVV] 格式 token → uppercase → words tier
            #   [Question-yi] → QUESTION-YI, [Breathing] → BREATHING
            if token_clean.startswith("[") and token_clean.endswith("]"):
                mfa_token = nvv_to_mfa(token_clean)
                words_pinyin.append({
                    "word": mfa_token,
                    "start": w["start"],
                    "end": w["end"],
                    **_source_metadata(w),
                })
                continue

            # 情况 1: NVV 大写 token → words tier (兜底, 通常由情况0处理)
            if is_nvv_token(token_clean):
                words_pinyin.append({
                    "word": token_clean,
                    "start": w["start"],
                    "end": w["end"],
                    **_source_metadata(w),
                })
                continue

            # 情况 2: 单个 CJK 字符 → pinyin
            if len(token_clean) == 1 and ('一' <= token_clean <= '鿿' or '㐀' <= token_clean <= '䶿'):
                try:
                    from pypinyin import lazy_pinyin, Style
                    py = lazy_pinyin(token_clean, style=Style.TONE3,
                                     neutral_tone_with_five=True, errors="default")
                    py_token = py[0] if py else token_clean
                except Exception:
                    py_token = token_clean
                words_pinyin.append({
                    "word": py_token,
                    "start": w["start"],
                    "end": w["end"],
                    **_source_metadata(w),
                })
                continue

            # 情况 3: 英文/数字 token → 原样保留
            if token_clean.isalpha() or token_clean.isdigit():
                words_pinyin.append({
                    "word": token_clean,
                    "start": w["start"],
                    "end": w["end"],
                    **_source_metadata(w),
                })
                continue

            # 情况 4: 标点 (白名单内) → 不进 MFA, 仅记录 CTC 锚点
            #   标点没有声学实现, 不进 .lab 和 TextGrid words tier,
            #   避免 MFA 打乱 phone 层. CTC 时间戳由 postprocess 后注入.
            #   只保留白名单内的标点, 其余单字符符号 (」『』【】等) 直接丢弃.
            if token_clean and len(token_clean) == 1 and token_clean in ALLOWED_PUNCT:
                # ASCII→CJK 标点 inline 转换 (安全网: align_text 已规范化, 兜底 tokenizer 输出的 ASCII)
                word_cjk = (token_clean if args.no_nvv else
                            _ASCII_TO_CJK_PUNCT.get(token_clean, token_clean))
                punct_entries.append({
                    "word": word_cjk,
                    "start": w["start"],
                    "end": w["end"],
                })

        # Canonicalize reference English before any generic CTC token merge.
        # The shared authority validator must see raw source rows so source
        # ordinals remain available for punctuation/gap and crossing checks.
        if stem in ref_texts:
            words_pinyin = _merge_reference_english_fragments(
                words_pinyin, ref_texts[stem])

        # ── Merge consecutive single-ASCII-letter tokens ──
        # The NVASR tokenizer splits OOV English words into individual
        # letter tokens (e.g. "ria"→"R"+"I"+"A").  Merge them back so
        # the token count stays aligned with reference-text word units.
        if words_pinyin:
            merged_pinyin = []
            i = 0
            while i < len(words_pinyin):
                w = words_pinyin[i]
                token = w["word"]
                if (len(token) == 1 and token.isascii() and token.isalpha()
                        and not is_nvv_token(token)
                        and "canonical_unit" not in w):
                    letters = [token]
                    j = i + 1
                    while j < len(words_pinyin):
                        nt = words_pinyin[j]["word"]
                        if (len(nt) == 1 and nt.isascii() and nt.isalpha()
                                and not is_nvv_token(nt)
                                and "canonical_unit" not in words_pinyin[j]):
                            letters.append(nt)
                            j += 1
                        else:
                            break
                    merged_word = "".join(letters)
                    # Force lowercase for known VTuber/proper names that the
                    # NVASR tokenizer splits into single uppercase letters
                    # (e.g. "R"+"I"+"A" → "ria", not "RIA").
                    if merged_word.lower() in _SINGLE_LETTER_LOWERCASE_NAMES:
                        merged_word = merged_word.lower()
                    merged_pinyin.append({
                        "word": merged_word,
                        "start": w["start"],
                        "end": words_pinyin[j - 1]["end"],
                    })
                    i = j
                else:
                    merged_pinyin.append(w)
                    i += 1
            words_pinyin = merged_pinyin

        # ── Dedup adjacent identical NVV tokens ──
        # NVASR can produce [Breathing][Breathing] from reference text or
        # ASR output where punctuation between two identical NVV tags is
        # stripped.  Keep the first token, extend its end to cover the
        # last duplicate so the time span encompasses the full event.
        if words_pinyin and not args.no_nvv:
            deduped_pinyin = []
            i = 0
            while i < len(words_pinyin):
                w = words_pinyin[i]
                token = w["word"]
                if is_nvv_token(token):
                    j = i + 1
                    while j < len(words_pinyin) and words_pinyin[j]["word"] == token:
                        j += 1
                    if j > i + 1:
                        deduped_pinyin.append({
                            "word": token,
                            "start": w["start"],
                            "end": words_pinyin[j - 1]["end"],
                        })
                        i = j
                    else:
                        deduped_pinyin.append(w)
                        i += 1
                else:
                    deduped_pinyin.append(w)
                    i += 1
            words_pinyin = deduped_pinyin

        # ── ria name integrity protection ──
        # Merge any remaining ria fragments that survived single-letter
        # merge (e.g. "R"+"ia"), normalise case ("RIA"→"ria").
        # Regression Case 31 Fix-4d.
        if not args.no_nvv:
            words_pinyin = _protect_ria(words_pinyin)

        words_pinyin = _clamp_words_to_wav_axis(words_pinyin, duration_s)

        # 写 TextGrid — 含 pauses tier
        try:
            # 从 blank_runs 计算 ≥200ms 的停顿段
            blank_runs = r.get("blank_runs", [])
            pauses = []
            for s, e in blank_runs:
                dur_ms = (e - s) * 60  # 60ms/frame
                if dur_ms >= 200:
                    pauses.append({
                        "start_ms": s * 60,
                        "end_ms": e * 60,
                        "duration_ms": dur_ms,
                    })

            # The producer writes immutable raw CTC evidence.  English
            # processed spans are resolved only after this bundle is copied
            # into ctc_pretg_adj by the energy/geometry stage; otherwise the
            # raw CTC root would already contain downstream geometry.
            words_pinyin = _clamp_words_to_wav_axis(words_pinyin, duration_s)

            # ── Strip leading punctuation (part A: remove from punct_entries) ──
            # NVASR sometimes inserts a comma / period right after emotion
            # tags (e.g. <|SAD|>，) at the very start of an utterance.
            # Detect it early so we can drop the CTC anchor before _punct.json
            # is written.  The text_asr string itself is cleaned in part B below.
            # Subsequent word timestamps are NOT shifted.
            _raw_text_check = "" if args.no_nvv else re.sub(
                r"<\|[^|]+\|>", "", r.get("text_asr", "")).strip()
            _leading_punct = None
            if _raw_text_check and is_punct(_raw_text_check[0]):
                _leading_punct = _raw_text_check[0]
                if punct_entries:
                    _first_punct = min(punct_entries, key=lambda x: x["start"])
                    if (_first_punct["word"] == _leading_punct
                            and _first_punct["start"] < 0.100):
                        punct_entries.remove(_first_punct)

            # ── ASR 空文本检测: 无内容词时跳过输出, 不进 MFA ──
            lab_tokens = " ".join(w["word"] for w in words_pinyin)
            if not lab_tokens.strip():
                print(f"  SKIP {stem}: ASR produced no text — skipping MFA alignment")
                skipped.setdefault("empty_asr", []).append(stem)
                continue

            # 写 TextGrid — 含 pauses tier (空检测之后, 避免孤立的空 TextGrid)
            out_tg = args.output_dir / f"{stem}.TextGrid"
            write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

            # 写 .lab — MFA 将此作为 transcript, 与 TextGrid words tier 同源
            out_lab = args.output_dir / f"{stem}.lab"
            _atomic_write_text(out_lab, lab_tokens + "\n")

            # 写标点锚点文件 (供 postprocess 后注入)
            # end = 下一个 token 的 start, 与词 token 同规则
            punct_path = args.output_dir / f"{stem}_punct.json"
            if punct_entries:
                all_seq = []
                for w in words_pinyin:
                    all_seq.append({"text": w["word"], "start": w["start"], "kind": "w"})
                for p in punct_entries:
                    all_seq.append({"text": p["word"], "start": p["start"], "kind": "p"})
                all_seq.sort(key=lambda x: x["start"])
                # 最後标点: end = max(start, duration - 0.5s), 尾部留给静音
                last_punct = max(punct_entries, key=lambda x: x["start"]) if punct_entries else None
                # Pre-extract sorted start times for O(P+T) monotonic lookup
                seq_starts = [t["start"] for t in all_seq]
                punct_data = []
                next_idx = 0  # monotonic pointer
                for p in punct_entries:
                    next_start = None
                    while next_idx < len(seq_starts) and seq_starts[next_idx] <= p["start"] + 0.001:
                        next_idx += 1
                    if next_idx < len(seq_starts):
                        next_start = seq_starts[next_idx]
                    if p is last_punct:
                        # 最后标点直接延续到音频结束
                        end_s = duration_s
                    else:
                        end_s = next_start if next_start is not None else p["end"]
                    # Guard: CTC overlap (e.g. NVV inside word) can make
                    # end_s < start_s, producing invalid intervals.
                    if end_s <= p["start"]:
                        end_s = p["start"] + 0.060  # min 1 frame width
                    punct_data.append({
                        "word": p["word"],
                        "start_ms": round(p["start"] * 1000, 1),
                        "end_ms": round(end_s * 1000, 1),
                        "start_s": p["start"],
                        "end_s": end_s,
                    })
                # Dedup: remove … if it overlaps with a real punctuation mark
                # (comma, period, etc.) — the punct mark already serves the
                # pause-marking function.  NVV tokens must be preserved.
                non_ellipsis = [p for p in punct_data if p["word"] != "…"]
                ellipsis_only = [p for p in punct_data if p["word"] == "…"]
                if not args.no_nvv and non_ellipsis and ellipsis_only:
                    kept_ellipsis = []
                    for ep in ellipsis_only:
                        overlap = any(
                            nep["start_s"] < ep["end_s"]
                            and nep["end_s"] > ep["start_s"]
                            for nep in non_ellipsis)
                        if not overlap:
                            kept_ellipsis.append(ep)
                    punct_data = non_ellipsis + kept_ellipsis

                _atomic_write_text(
                    punct_path, json.dumps(punct_data, ensure_ascii=False))
            else:
                # v4 requires a deterministic sidecar for every rerun stem.
                _atomic_write_text(punct_path, "[]")

            # Required sidecars never receive free ASR content in
            # reference-only mode, even transiently before the canonical
            # overwrite below.  ASR remains diagnostic in the manifest only.
            text_asr = (ref_texts.get(stem) if args.no_nvv
                        else r.get("text_asr", ""))
            if text_asr is None:
                raise ValueError(f"reference-only stem has no canonical reference: {stem}")
            text_asr = clean_unsupported_punct(text_asr)
            # ── Strip leading punctuation (part B: remove from text_asr) ──
            # Detection mirroring part A above; now that text_asr is available,
            # delete the leading punct character so _text_raw.txt / _text_cn.txt
            # are clean.  Subsequent token positions are not shifted.
            _text_clean_b = re.sub(r"<\|[^|]+\|>", "", text_asr).strip()
            if not args.no_nvv and _text_clean_b and is_punct(_text_clean_b[0]):
                _lp = _text_clean_b[0]
                _tag_end = 0
                for _m in re.finditer(r"<\|[^|]+\|>", text_asr):
                    _tag_end = _m.end()
                _pos = _tag_end
                while _pos < len(text_asr) and text_asr[_pos] in ' \t':
                    _pos += 1
                if _pos < len(text_asr) and text_asr[_pos] == _lp:
                    text_asr = text_asr[:_pos] + text_asr[_pos + 1:]
            raw_path = args.output_dir / f"{stem}_text_raw.txt"
            _atomic_write_text(raw_path, text_asr + "\n")

            # 中文文本 (raw_text tier 用): 去掉 <|lang|> <|emo|> 标签, NVV→大写
            text_cn = re.sub(r"<\|[^|]+\|>", "", text_asr).strip()
            text_cn = clean_unsupported_punct(text_cn)
            # 兜底 ITN: 阿拉伯数字 → 中文数字 (优先 cn2an, 兜底简单映射)
            try:
                import cn2an
                text_cn = cn2an.transform(text_cn, "an2cn")
            except ImportError:
                text_cn = re.sub(r'\d+',
                                 lambda m: ''.join("零一二三四五六七八九"[int(d)] for d in m.group(0)),
                                 text_cn)
            text_cn = re.sub(
                r'\[([A-Za-z][^\]]*?)\]',
                lambda m: ' ' + nvv_to_mfa(m.group(0)) + ' ', text_cn)
            text_cn = re.sub(r'\s+', ' ', text_cn).strip()
            # Dedup adjacent identical NVV tokens (BREATHING BREATHING → BREATHING)
            text_cn = re.sub(r'\b([A-Z][A-Z0-9-]+)\s+\1\b', r'\1', text_cn)
            # Same ellipsis-punct dedup as asr_text (lines 308-316)
            text_cn = re.sub(r'…([，。！？、；：,\.!\?;:])', r'\1', text_cn)
            text_cn = re.sub(r'([，。！？、；：,\.!\?;:])…', r'\1', text_cn)
            text_cn = re.sub(
                r'([，。！？、；：,\.!\?;:])([A-Z][A-Z0-9-]*[A-Z0-9])…',
                r'\1\2', text_cn)
            text_cn = re.sub(r'…{2,}', '…', text_cn)
            cn_path = args.output_dir / f"{stem}_text_cn.txt"
            _atomic_write_text(cn_path, text_cn + "\n")

            # Reference-only required sidecars are canonical reference content;
            # free ASR text remains diagnostic only.
            if args.no_nvv:
                canonical = ref_texts.get(stem)
                if canonical is None:
                    raise ValueError(f"reference-only stem has no canonical reference: {stem}")
                canonical_bytes = canonical.strip() + "\n"
                _atomic_write_text(raw_path, canonical_bytes)
                _atomic_write_text(cn_path, canonical_bytes)
                _atomic_write_text(
                    args.output_dir / f"{stem}_ref.txt", canonical_bytes)

            # Preserve the source transcript beside CTC output.  The ASR
            # text above is diagnostic only; downstream normalization and
            # post-processing must be able to recover the authoritative
            # reference even when the original data directory is not passed.
            if stem in ref_texts:
                _atomic_write_text(
                    args.output_dir / f"{stem}_ref.txt",
                    ref_texts[stem].strip() + "\n")

            manifest.append({
                "audio": str(wav_map[stem]),
                "textgrid": str(out_tg),
                "lab": str(out_lab),
                "text_asr": r.get("text_asr", ""),
                "nvv_mode": "reference_only" if args.no_nvv else "asr_added_allowed",
                "asr_nvv_bias": False if args.no_nvv else bool(args.nvv_bias),
                "content_authority": "reference" if args.no_nvv else "asr_or_reference",
                "duration_s": duration_s,
                "n_words": len(words_pinyin),
                "n_pauses": len(pauses),
                "pauses": pauses,
                "n_punct": len(punct_entries),
                "_words": words_pinyin,
            })
            ok += 1
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
            fail += 1

    # ── Auto-add English tokens to MFA dictionary ──
    # English tokens (like "li", "ve", "A", "I") are self-referential
    # in the MFA dict — MFA can't model them acoustically, so they get
    # CTC-only boundaries like NVV tokens.
    if not args.no_dict_update and args.dict_path and args.dict_path.exists():
        english_tokens_found: set[str] = set()
        for entry in manifest:
            for w in entry.get("_words", []):
                token = w["word"]
                if is_english_token(token):
                    english_tokens_found.add(token)

        if english_tokens_found:
            existing = set()
            with open(args.dict_path, encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.add(line.split()[0])
            new_tokens = sorted(t for t in english_tokens_found if t not in existing)
            if new_tokens:
                with open(args.dict_path, 'a', encoding='utf-8') as f:
                    for t in new_tokens:
                        f.write(f"{t} {t}\n")
                print(f"  Added {len(new_tokens)} English tokens to MFA dict: {', '.join(new_tokens)}")
                mfa_words = load_mfa_word_set(args.dict_path)
            else:
                print(f"  English tokens already in MFA dict: {', '.join(sorted(english_tokens_found))}")

    for entry in manifest:
        stem = Path(entry["audio"]).stem
        words = entry["_words"]
        # 合并 words + punct, 按时间排序: 每个 token 的 end = 下一个 token 的 start
        all_tokens = []
        for w in words:
            all_tokens.append({"text": w["word"], "start": w["start"], "kind": "word"})
        # punct_entries 来自 per-file 写入的 _punct.json, 需要重建
        punct_path = args.output_dir / f"{stem}_punct.json"
        punct_for_end: list[dict] = []
        if punct_path.exists():
            punct_for_end = json.loads(punct_path.read_text(encoding="utf-8"))
            for p in punct_for_end:
                all_tokens.append({"text": p["word"], "start": p["start_s"], "kind": "punct"})
        all_tokens.sort(key=lambda x: x["start"])

        # 最后一个词: VAD 检测语音结束点, 不留尾静音
        last_word_vad_end = None
        audio_path = entry["audio"]
        if words:
            last_word_vad_end = _vad_speech_end(audio_path, words[-1]["start"])

        tokens_path = args.output_dir / f"{stem}_tokens.jsonl"
        token_lines: list[str] = []
        for i, w in enumerate(words):
            canonical_span = w.get("canonical_span")
            has_timed_canonical_span = (
                "canonical_unit" in w
                and isinstance(canonical_span, (list, tuple))
                and len(canonical_span) == 2
                and all(isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        for value in canonical_span))
            if has_timed_canonical_span:
                # Raw CTC is an immutable evidence artifact.  Its TextGrid
                # must never require or consume mutable processed geometry;
                # that geometry is created later in the copied work root by
                # adjust_ctc_boundaries.py.  In particular, a 60 ms raw
                # anchor is evidence, not the final English word duration.
                start_s = float(w["start"])
                end_s = float(canonical_span[1])
            elif (w is words[-1] and last_word_vad_end is not None
                    and "canonical_unit" not in w):
                start_s = w["start"]
                end_s = last_word_vad_end
            else:
                start_s = w["start"]
                next_start = None
                for t in all_tokens:
                    if t["start"] > w["start"] + 0.001:
                        next_start = t["start"]
                        break
                end_s = next_start if next_start is not None else w["end"]
            line = {
                "word": w["word"],
                "start_ms": round(start_s * 1000, 1),
                "end_ms": round(end_s * 1000, 1),
                "start_s": start_s,
                "end_s": end_s,
                "type": "word",
            }
            # Authority English rows carry the immutable Wave 1 unit
            # ledger into the CTC bundle.  Keep the ordinary rows
            # byte-compatible apart from their timing/source metadata.
            for key in (
                    "surface_text", "source_ctc_ordinals", "canonical_span",
                    "canonical_unit", "canonical_unit_sha256",
                    "reference_identity", "reference_ordinal",
                    "hyphen_separator_omitted", "processed_ctc_span",
                    "processed_ctc_boundary_source"):
                if key in w:
                    line[key] = w[key]
            token_lines.append(json.dumps(line, ensure_ascii=False))
        _atomic_write_text(tokens_path, "\n".join(token_lines) + "\n")

    # ── Build skip summary lines ──
    skip_lines = ""
    skip_labels = {
        "missing_reference": "无权威参考文本 (missing_reference)",
        "unsupported_reference": "参考文本不受支持",
        "japanese": "含假名 (管线不支持)",
        "incomplete_english": "英文碎片不完整",
        "incomplete_ctc_alignment": "CTC 目标 token 未完整对齐",
        "empty_asr": "ASR 无输出文本",
    }
    for reason, label in skip_labels.items():
        if reason in skipped:
            skip_lines += f"  Skipped ({label}): {len(skipped[reason])}\n"

    summary = (
        f"CTC Pre-alignment Report\n"
        f"{'=' * 40}\n"
        f"Files: {len(paths)} total, {ok} OK, {fail} failed\n"
        + (skip_lines if skip_lines else "")
        + f"Time: {infer_time:.1f}s\n\n"
        f"Output: {args.output_dir}\n"
        f"  *.TextGrid  → MFA anchors (words=pinyin+punct+NVV)\n"
        f"  *.lab       → MFA corpus (same source as anchors, 100% match)\n"
        f"  manifest.json    → full file index\n"
        f"  *_tokens.jsonl   → per-word CTC timestamps (ms)\n\n"
        f"Pipeline: reference text (when available) → CTC anchors + .lab;\n"
        f"  ASR text is diagnostic/fallback only\n"
        f"  → MFA reads .lab as transcript, TextGrid as anchors\n"
        f"  → 100% word match → every CTC boundary used for phone refinement\n"
    )
    print(f"\n{summary}")
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")

    # ── 输出后处理 (等价于 run_pipeline.py 的 normalize_punct → normalize → normalize_en) ──
    if ok > 0:
        print("\n── 输出后处理 ──")
        if not args.no_nvv:
            _normalize_punct(args.output_dir)
            _normalize_numerals(args.output_dir)
            _normalize_ria(args.output_dir)
            _normalize_english(args.output_dir, args.dict_path,
                               update_dict=not args.no_dict_update)
        if _validate_all_ctc_bundles(args.output_dir, ref_texts):
            _rebuild_final_manifest(args.output_dir, audio_dir,
                                    wav_files=wav_files)
            # Marker: downstream pipeline steps can skip re-normalization.
            _stem_count = len(list(args.output_dir.glob("*.lab")))
            _manifest_digest = hashlib.sha256(
                (args.output_dir / "manifest.json").read_bytes()).hexdigest()
            (args.output_dir / ".ctc_normalized").write_text(
                make_ctc_normalization_marker(_stem_count, _manifest_digest),
                encoding="utf-8",
            )
            # ── CTC run receipt (Case 99 / R5) ─────────────────────
            _all_output_stems = sorted(
                p.stem for p in args.output_dir.glob("*.lab")
            )
            _all_input_stems = sorted(Path(p).stem for p in paths)
            write_ctc_run_receipt(
                args.output_dir,
                actual_argv=sys.argv,
                asr_python=sys.executable,
                model_path=_model_path,
                model_tree_digest=_model_tree_digest,
                model_file_manifest=_model_file_manifest,
                dict_path=Path(args.dict_path) if args.dict_path else Path(""),
                dict_digest=_dict_digest,
                input_stems=_all_input_stems,
                output_stems=_all_output_stems,
            )
            # v2 source-denominator receipt: the source universe was frozen
            # before filtering, and skipped eligible stems are explicit
            # filtered evidence rather than silent loss.
            _eligible_all = sorted(stems) if args.allow_missing_reference else sorted(ref_texts)
            _output_v2 = sorted(_all_output_stems)
            _filtered_v2 = sorted(set(_eligible_all) - set(_output_v2))
            _accounting = make_pipeline_accounting_receipt(
                source_stems=sorted(p.stem for p in source_wav_files),
                eligible_stems=_eligible_all,
                exclusions=source_exclusions,
                output_stems=_output_v2,
                filtered_stems=_filtered_v2,
                run_id=make_pipeline_run_id(), mode="ctc_prealign",
                route=["ctc_prealign"],
                paths={"output": str(args.output_dir), "filtered": str(args.output_dir)},
                shards=[{"shard_id": "single", "stems": _eligible_all}],
                extra={"source_frozen": True,
                       "reference_only": not args.allow_missing_reference,
                       "reference_mode": args.reference_mode,
                       "processed_stems": sorted(Path(p).stem for p in paths)},
            )
            write_pipeline_accounting_receipt(args.output_dir, _accounting)
        else:
            print("ERROR: refusing to write CTC normalization marker")
            fail += 1

    # ── 恢复模型 ──
    model.model.inference = orig_inf
    print(f"完成! 输出: {args.output_dir}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
