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
import json
import os
import re
import time
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

# 管线支持的标点白名单 — 只有这些字符会被保留为标点
ALLOWED_PUNCT_CJK = "，。！？、；：…"
ALLOWED_PUNCT_ASCII = ",.!?;:"
ALLOWED_PUNCT = set(ALLOWED_PUNCT_CJK + ALLOWED_PUNCT_ASCII)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


# ── NVV known names (shared with postprocess_textgrids.py) ──
NVV_NAMES: set[str] = {
    "BREATHING", "LAUGHTER", "BURP", "COUGH", "CRYING", "GROAN",
    "HISS", "HUM", "SHH", "SIGH", "SNEEZE", "SNIFF", "SNORE",
    "TSK", "UHM", "WHISTLE", "YAWN",
    "QUESTION-YI", "QUESTION-EN", "QUESTION-OH", "QUESTION-AH",
    "QUESTION-EI", "QUESTION-HUH",
    "SURPRISE-OH", "SURPRISE-AH", "SURPRISE-WA", "SURPRISE-YO",
    "CONFIRMATION-EN", "DISSATISFACTION-HNN",
}

_NVV_NAMES_UPPER = {n.upper() for n in NVV_NAMES}


def is_pinyin_syllable(token: str) -> bool:
    """检查 token 是否为有效的拼音音节 (可被 MFA 词典识别).

    拼音音节: 小写字母序列 + 可选声调数字 1-5.
    例如: kuai4, ni3, hao3, e2, a1, nv3, lve4
    不匹配: , . ? ! ... (标点), hello (英文), 123 (数字)
    """
    import re
    return bool(re.match(r'^[a-z]+[1-5]$', token))


def is_nvv_token(token: str) -> bool:
    """检查 token 是否为 NVV 标签 (BREATHING, QUESTION-YI 等)."""
    return token.upper() in _NVV_NAMES_UPPER


def is_english_token(token: str) -> bool:
    """Token is English alpha: not CJK, not NVV, not pinyin syllable with tone.

    English tokens (like "li", "ve", "A", "I", "AI", "live") are treated
    as self-referential MFA dict entries and use CTC-only boundaries.
    """
    if not token or not token.isalpha():
        return False
    if not token.isascii():
        return False  # CJK chars are alpha in Python but not English
    if is_nvv_token(token):
        return False
    if is_pinyin_syllable(token):
        return False
    return True


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


# ─── NVV 标签 → MFA 大写 token 映射 ───

NVV_TO_MFA: dict[str, str] = {
    "Breathing": "BREATHING",
    "Laughter": "LAUGHTER",
    "Burp": "BURP",
    "Cough": "COUGH",
    "Crying": "CRYING",
    "Groan": "GROAN",
    "Hiss": "HISS",
    "Hum": "HUM",
    "Shh": "SHH",
    "Sigh": "SIGH",
    "Sneeze": "SNEEZE",
    "Sniff": "SNIFF",
    "Snore": "SNORE",
    "Tsk": "TSK",
    "Uhm": "UHM",
    "Whistle": "WHISTLE",
    "Yawn": "YAWN",
    "Question-yi": "QUESTION-YI",
    "Question-en": "QUESTION-EN",
    "Question-oh": "QUESTION-OH",
    "Question-ah": "QUESTION-AH",
    "Question-ei": "QUESTION-EI",
    "Question-huh": "QUESTION-HUH",
    "Surprise-oh": "SURPRISE-OH",
    "Surprise-ah": "SURPRISE-AH",
    "Surprise-wa": "SURPRISE-WA",
    "Surprise-yo": "SURPRISE-YO",
    "Confirmation-en": "CONFIRMATION-EN",
    "Dissatisfaction-hnn": "DISSATISFACTION-HNN",
}


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
    return re.sub(
        r'\[([A-Za-z][^\]]*?)\]',
        lambda m: nvv_to_mfa(m.group(0)),
        text
    )


# ═══════════════════════════════════════════════════════════════
# Monkey-patch: 用参考文本做 CTC 强制对齐, 非 ASR 解码
# ═══════════════════════════════════════════════════════════════

def make_patched_inference(ref_texts: dict[str, str],
                           bias_value: float = NVV_BIAS_DEFAULT,
                           pause_threshold: int = PAUSE_FRAMES_DEFAULT):
    """
    创建打了补丁的 inference 方法.

    与原版 export_mfa_textgrid.py 的核心区别:
    - 不从 CTC 解码 ASR 文本, 而是从 ref_texts 字典查找参考中文文本
    - 对参考文本做 CTC 强制对齐 → 汉字级别时间戳
    - 同样做 blank-frame NVV bias + 停顿检测 + 省略号注入

    ref_texts: {stem: chinese_text}  — 键为音频文件 stem (无扩展名)
    """

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
            x = ctc_logits[i, :elens[i].item(), :]

            # ── Blank-frame NVV bias ──
            top_pred = x.argmax(dim=-1)
            is_blank = (top_pred == BLANK_ID)
            x[is_blank, NVV_START:NVV_END + 1] += bias_value

            raw_y = x.argmax(dim=-1).tolist()

            # ── 记录 blank 段 (CTC 空白帧 → 停顿) ──
            blank_runs = []
            jj = 0
            while jj < len(raw_y):
                if raw_y[jj] == BLANK_ID:
                    s = jj
                    while jj < len(raw_y) and raw_y[jj] == BLANK_ID:
                        jj += 1
                    blank_runs.append((s, jj))
                else:
                    jj += 1

            # ── 长空白注入省略号 (供文本转录用, 不影响时间戳) ──
            yseq_pause = torch.tensor(raw_y).to(x.device)
            j = 0
            while j < len(raw_y):
                if raw_y[j] == BLANK_ID:
                    s = j
                    while j < len(raw_y) and raw_y[j] == BLANK_ID:
                        j += 1
                    if (j - s) >= pause_threshold:
                        yseq_pause[s + (j - s) // 2] = ELLIPSIS_ID
                else:
                    j += 1

            yseq_unique = torch.unique_consecutive(yseq_pause, dim=-1)
            mask = yseq_unique != self.blank_id
            token_int = yseq_unique[mask].tolist()
            asr_text = tokenizer.decode(token_int)  # 仅用于显示/nvv标签

            # ── 后处理 (省略号标点去重等) ──
            asr_text = re.sub(r'…([，。！？、；：,\.!\?;:])', r'\1', asr_text)
            asr_text = re.sub(r'([，。！？、；：,\.!\?;:])…', r'\1', asr_text)
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

            stem = Path(key[i]).stem
            if stem in ref_texts:
                # 使用参考文本 (ground truth) → 更准确的 CJK 字符级强制对齐
                align_text = ref_texts[stem].strip()
            else:
                # 无参考文本, 使用 ASR 解码文本
                align_text = asr_clean

            # cn2an 数字正则化 (参考文本和 ASR 文本都可能含阿拉伯数字)
            try:
                import cn2an
                parts = re.split(r'(\[[^\]]+\]|[A-Z][A-Z0-9-]*[A-Z0-9])', align_text)
                for k, part in enumerate(parts):
                    if re.match(r'^(\[[^\]]+\]|[A-Z][A-Z0-9-]*[A-Z0-9])$', part):
                        continue
                    try:
                        parts[k] = cn2an.transform(part, 'an2cn')
                    except Exception:
                        pass
                align_text = ''.join(parts)
            except ImportError:
                pass

            words_aligned = []  # token 级别时间戳
            if align_text:
                tokens = tokenizer.text2tokens(align_text)
                speech_tokens = tokens
                token_ids_list = tokenizer.tokens2ids(speech_tokens)
                token_ids_flat = []
                for tids in token_ids_list:
                    if tids:
                        token_ids_flat.extend(tids)
                    else:
                        token_ids_flat.append(124)  # space token

                if token_ids_flat:
                    # 准备 logits: 去掉 query 帧
                    logits_speech = self.ctc.log_softmax(enc)[
                        i, QUERY_FRAMES:total_frames, :
                    ]
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

                    # 分组提取 token 边界
                    pred_grp = groupby(align[0, :total_speech_frames].tolist())
                    _s = 0
                    tid = 0
                    for ptok, pframe_iter in pred_grp:
                        frame_indices = list(pframe_iter)
                        _e = _s + len(frame_indices)
                        if ptok != 0 and tid < len(token_ids_flat):
                            t_left = max((_s * FRAME_MS - 30) / 1000, 0)
                            t_right = min((_e * FRAME_MS - 30) / 1000, duration_s)
                            token_str = speech_tokens[tid] if tid < len(speech_tokens) else ""
                            words_aligned.append({
                                "word": token_str,
                                "start": round(t_left, 3),
                                "end": round(t_right, 3),
                            })
                            tid += 1
                        _s = _e

            results.append({
                "key": key[i],
                "text_asr": asr_final,
                "text_asr_clean": asr_clean,
                "duration_s": round(duration_s, 3),
                "words": words_aligned,
                "blank_runs": blank_runs,
            })

        return results, meta

    return patched


# ═══════════════════════════════════════════════════════════════
# TextGrid 写入 (Praat 格式, MFA 兼容)
# ═══════════════════════════════════════════════════════════════

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
        we = words_pinyin[i + 1]["start"] if i + 1 < len(words_pinyin) else w["end"]
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
            ps = p["start_ms"] / 1000
            pe = p["end_ms"] / 1000
            if ps > pc + 0.005:
                p_intervals.append((pc, ps, ""))
            p_intervals.append((ps, pe, f'{p["duration_ms"]}ms'))
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

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# 主流程: 批量处理
# ═══════════════════════════════════════════════════════════════

def _vad_speech_end(wav_path: str, search_from_s: float) -> float | None:
    """用能量 VAD 从 audio 末尾反向搜索语音结束点."""
    try:
        import soundfile as sf
        import numpy as np
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        frame_ms = 0.01  # 10ms frames
        frame_len = int(sr * frame_ms)
        hop = frame_len // 2
        # 从 search_from_s 开始向后搜索
        start_sample = int(search_from_s * sr)
        if start_sample >= len(audio):
            return None
        segment = audio[start_sample:]
        rms = np.array([np.sqrt(np.mean(segment[i:i+frame_len]**2))
                        for i in range(0, len(segment) - frame_len, hop)])
        if len(rms) == 0:
            return None
        # 阈值: 最大 RMS 的 5%
        threshold = np.max(rms) * 0.05
        # 从后向前找最后一个超过阈值的帧
        last_speech_frame = len(rms) - 1
        for i in range(len(rms) - 1, -1, -1):
            if rms[i] > threshold:
                last_speech_frame = i
                break
        end_s = search_from_s + (last_speech_frame * hop) / sr
        return round(end_s, 3)
    except Exception:
        return None


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
    """Build {stem: path} index for .txt files — single scan, O(1) lookup."""
    index: dict[str, Path] = {}
    # Single-level scan first (fast, handles flat directories)
    try:
        with os.scandir(str(data_dir)) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".txt"):
                    stem = entry.name[:-4]
                    if stem not in index:
                        index[stem] = Path(entry.path)
    except OSError:
        pass
    # One level of subdirectories if top-level empty
    if not index:
        try:
            with os.scandir(str(data_dir)) as it:
                for entry in it:
                    if entry.is_dir():
                        try:
                            with os.scandir(entry.path) as it2:
                                for e2 in it2:
                                    if e2.is_file() and e2.name.endswith('.txt'):
                                        stem = e2.name[:-4]
                                        if stem not in index:
                                            index[stem] = Path(e2.path)
                        except OSError:
                            pass
        except OSError:
            pass
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


def main():
    parser = argparse.ArgumentParser(
        description="CTC Pre-alignment: NVASR → MFA anchor TextGrids (pinyin)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="原始数据目录 (含 wav + 中文 txt)")
    parser.add_argument("--pinyin-dir", type=Path, required=True,
                        help="拼音语料目录 (prepare_corpus 输出, 用于 fallback)")
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nvv-bias", type=float, default=NVV_BIAS_DEFAULT,
                        help=f"NVV blank-frame bias (default: {NVV_BIAS_DEFAULT}).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 扫描音频文件 ──
    audio_dir = args.audio_dir or args.data_dir
    wav_files = sorted(audio_dir.rglob("*.wav"))
    if args.limit > 0:
        wav_files = wav_files[:args.limit]
    print(f"扫描到 {len(wav_files)} 个 WAV 文件")

    # ── 构建参考文本查找表 {stem: chinese_text} ──
    print("构建参考文本查找表...")
    ref_texts: dict[str, str] = {}
    missing_ref = []
    skipped_jp = []
    # Build text index once (O(1) lookup per stem, no repeated rglob)
    txt_index = _build_txt_index(args.data_dir)
    print(f"  已索引 {len(txt_index)} 个文本文件")

    for wav_path in wav_files:
        stem = wav_path.stem
        ref = find_ref_text(stem, args.data_dir, txt_index)
        if ref:
            if has_japanese(ref):
                skipped_jp.append(stem)
                continue
            ref = clean_unsupported_punct(ref)
            ref_texts[stem] = ref
        else:
            missing_ref.append(stem)

    if skipped_jp:
        print(f"  跳过日语: {len(skipped_jp)} 个文件 (含假名, 管线不支持)")
    if missing_ref:
        print(f"  注意: {len(missing_ref)} 个文件无参考文本, 将纯靠 ASR 文本")
    print(f"  已索引 {len(ref_texts)} 个参考文本, 共 {len(wav_files)} 个音频")

    # ── 加载 NVASR 模型 ──
    print(f"加载 NVASR 模型: {args.model_path}")
    from funasr import AutoModel
    model = AutoModel(model=args.model_path, device=args.device, disable_update=True)
    orig_inf = model.model.inference
    patched = make_patched_inference(ref_texts, args.nvv_bias)
    model.model.inference = patched.__get__(model.model, type(model.model))

    # ── 处理所有音频文件 (有参考文本用参考, 无则纯靠 ASR) ──
    paths = [str(p) for p in wav_files]
    stems = [p.stem for p in wav_files]
    if not paths:
        print("错误: 没有可处理的音频文件")
        model.model.inference = orig_inf
        return

    # ── 批量推理 ──
    t0 = time.time()
    all_results = []
    BATCH = 16  # 比 export_mfa_textgrid.py 稍小, 留 GPU 空间给 forced alignment
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

    for i, r in enumerate(all_results):
        stem = stems[i] if i < len(stems) else Path(r["key"]).stem
        words_aligned = r["words"]
        duration_s = r["duration_s"]

        # 判断 token 类型并决定是否保留到 TextGrid
        def valid_word(token: str) -> bool:
            """token 在 MFA 词典中 → 保留; 标点/空白 → 剔除."""
            # NVV 大写标签
            if is_nvv_token(token):
                if mfa_words is not None:
                    return token in mfa_words
                return True  # 无词典时也保留 NVV
            # 拼音音节
            if is_pinyin_syllable(token):
                if mfa_words is not None:
                    return token in mfa_words
                return True
            # 英文 token — self-referential, 自动加入 MFA 词典
            if is_english_token(token):
                return True
            # 数字
            if token.isdigit():
                return True
            return False

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
                })
                continue

            # 情况 1: NVV 大写 token → words tier (兜底, 通常由情况0处理)
            if is_nvv_token(token_clean):
                words_pinyin.append({
                    "word": token_clean,
                    "start": w["start"],
                    "end": w["end"],
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
                })
                continue

            # 情况 3: 英文/数字 token → 原样保留
            if token_clean.isalpha() or token_clean.isdigit():
                words_pinyin.append({
                    "word": token_clean,
                    "start": w["start"],
                    "end": w["end"],
                })
                continue

            # 情况 4: 标点 (白名单内) → 不进 MFA, 仅记录 CTC 锚点
            #   标点没有声学实现, 不进 .lab 和 TextGrid words tier,
            #   避免 MFA 打乱 phone 层. CTC 时间戳由 postprocess 后注入.
            #   只保留白名单内的标点, 其余单字符符号 (」『』【】等) 直接丢弃.
            if token_clean and len(token_clean) == 1 and token_clean in ALLOWED_PUNCT:
                punct_entries.append({
                    "word": token_clean,
                    "start": w["start"],
                    "end": w["end"],
                })

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

            out_tg = args.output_dir / f"{stem}.TextGrid"
            write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

            # 写 .lab — MFA 将此作为 transcript, 与 TextGrid words tier 同源
            out_lab = args.output_dir / f"{stem}.lab"
            lab_tokens = " ".join(w["word"] for w in words_pinyin)
            out_lab.write_text(lab_tokens + "\n", encoding="utf-8")

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
                # 最后标点: end = max(start, duration - 0.5s), 尾部留给静音
                last_punct = max(punct_entries, key=lambda x: x["start"]) if punct_entries else None
                trailing_silence_s = max(0, duration_s - 0.5)
                punct_data = []
                for p in punct_entries:
                    next_start = None
                    for t in all_seq:
                        if t["start"] > p["start"] + 0.001:
                            next_start = t["start"]
                            break
                    if p is last_punct:
                        # 最后标点直接延续到音频结束
                        end_s = duration_s
                    else:
                        end_s = next_start if next_start is not None else p["end"]
                    punct_data.append({
                        "word": p["word"],
                        "start_ms": round(p["start"] * 1000, 1),
                        "end_ms": round(end_s * 1000, 1),
                        "start_s": p["start"],
                        "end_s": end_s,
                    })
                punct_path.write_text(json.dumps(punct_data, ensure_ascii=False),
                                     encoding="utf-8")

            # 含情绪标签的原始文本 (保留 <|HAPPY|> <|zh|> 等, NVV 保持 [Bracket] 格式)
            text_asr = r.get("text_asr", "")
            text_asr = clean_unsupported_punct(text_asr)
            raw_path = args.output_dir / f"{stem}_text_raw.txt"
            raw_path.write_text(text_asr + "\n", encoding="utf-8")

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
                lambda m: nvv_to_mfa(m.group(0)), text_cn)
            cn_path = args.output_dir / f"{stem}_text_cn.txt"
            cn_path.write_text(text_cn + "\n", encoding="utf-8")

            manifest.append({
                "audio": str(audio_dir / f"{stem}.wav"),
                "textgrid": str(out_tg),
                "lab": str(out_lab),
                "text_asr": r.get("text_asr", ""),
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
    if args.dict_path and args.dict_path.exists():
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

    # ── 保存 manifest + 逐词 tokens JSONL ──
    with open(args.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

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
        with open(tokens_path, "w", encoding="utf-8") as f:
            for i, w in enumerate(words):
                if w is words[-1] and last_word_vad_end is not None:
                    end_s = last_word_vad_end
                else:
                    next_start = None
                    for t in all_tokens:
                        if t["start"] > w["start"] + 0.001:
                            next_start = t["start"]
                            break
                    end_s = next_start if next_start is not None else w["end"]
                line = {
                    "word": w["word"],
                    "start_ms": round(w["start"] * 1000, 1),
                    "end_ms": round(end_s * 1000, 1),
                    "start_s": w["start"],
                    "end_s": end_s,
                    "type": "word",
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    summary = (
        f"CTC Pre-alignment Report\n"
        f"{'=' * 40}\n"
        f"Files: {len(paths)} total, {ok} OK, {fail} failed\n"
        f"Time: {infer_time:.1f}s\n\n"
        f"Output: {args.output_dir}\n"
        f"  *.TextGrid  → MFA anchors (words=pinyin+punct+NVV)\n"
        f"  *.lab       → MFA corpus (same source as anchors, 100% match)\n"
        f"  manifest.json    → full file index\n"
        f"  *_tokens.jsonl   → per-word CTC timestamps (ms)\n\n"
        f"Pipeline: NVASR ASR text → TextGrid + .lab (same text)\n"
        f"  → MFA reads .lab as transcript, TextGrid as anchors\n"
        f"  → 100% word match → every CTC boundary used for phone refinement\n"
    )
    print(f"\n{summary}")
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")

    # ── 恢复模型 ──
    model.model.inference = orig_inf
    print(f"完成! 输出: {args.output_dir}")


if __name__ == "__main__":
    main()
