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
from decimal import Decimal, ROUND_HALF_UP
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
# 疑问/惊讶/确认/不满 语气词 token id — 这些由 Qwen 转写为 CJK 原词
# (哎/咦/嗯/啊), 必须从 NVV 输出中屏蔽, 仅保留 vegetative 生理发声类.
NVV_SUPPRESSED_IDS: frozenset[int] = frozenset({
    25036,  # [Question-huh]
    25038,  # [Confirmation-en]
    25041,  # [Surprise-ah]
    25042,  # [Surprise-oh]
    25044,  # [Dissatisfaction-hnn]
    25045,  # [Surprise-wa]
    25046,  # [Question-yi]
    25047,  # [Question-ei]
    25049,  # [Question-ah]
    25050,  # [Question-oh]
    25051,  # [Surprise-yo]
    25052,  # [Question-en]
})
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
PUNCTUATION_EVIDENCE_SCHEMA = "ctc-punctuation-evidence-v2"
NVASR_CANDIDATE_PROVENANCE_SCHEMA = "nvasr-candidate-provenance-v1"
NVASR_MAPPING_BASIS = "raw_ctc_label_neighbors_forced_overlap-v2"
NVASR_CANDIDATE_SCHEMA_VERSION = 3
NVASR_MAPPING_AXIS = "non_nvv_compact_v1"
NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA = "nvasr-raw-timeline-neighbors-v1"
NVASR_SPIKE_ANCHOR_SCHEMA = "ctc_spike_anchor_v2"
CTC_RAW_TOKEN_ROW_SCHEMA = "ctc_raw_token_row_v1"
# Kept as an import-compatible name for existing consumers.  The value is
# the exact compact-axis contract, rather than a descriptive metadata schema.
NVASR_SEMANTIC_AXIS_SCHEMA = NVASR_MAPPING_AXIS
NVASR_ANCHOR_COORDINATE_SYSTEM = "speech_seconds_from_ctc_encoder_frames"
NVASR_ANCHOR_QUANTIZATION = "half_open_60ms_frames_centered_30ms_round_half_up"
CTC_TOKEN_SIDECAR_PASSTHROUGH_FIELDS = (
    "surface_text", "source_ctc_ordinals", "canonical_span",
    "canonical_unit", "canonical_unit_sha256", "reference_identity",
    "reference_ordinal", "hyphen_separator_omitted", "processed_ctc_span",
    "processed_ctc_boundary_source", "provenance_schema", "candidate_id",
    "candidate_kind", "candidate_surface", "candidate_token_id",
    "candidate_token_ids", "raw_span", "speech_span", "raw_start_frame",
    "raw_end_frame", "speech_start_frame", "speech_end_frame",
    "raw_start_s", "raw_end_s", "speech_start_s", "speech_end_s",
    "raw_frame_count", "speech_frame_count", "query_frames", "frame_ms",
    "mapping_basis", "mapping_axis", "mapping_key", "mapping_outcome", "forced_span",
    "raw_timeline_mapping_key",
    "nvasr_candidate_schema_version", "ordered_semantic_neighbors",
    "candidate_source", "raw_timeline_neighbors",
    "raw_timeline_neighbors_schema", "raw_timeline_index",
    "raw_timeline_event_count", "raw_timeline_evidence_sha256",
    "ctc_spike_anchor",
    "mapping_selection", "mapping_forced_speech_overlap_s",
    "mapping_forced_ctc_anchor_overlap_s", "ctc_lexical_ordinal",
    "ctc_raw_token_row",
    "semantic_occurrence_id", "semantic_surface_occurrence",
    "nvv_deduplication", "adjusted_span", "adjusted_span_basis",
    "adjusted_span_is_acoustic_evidence", "adjusted_mapping_outcome",
)

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

    A child sees the complete input directory, but its operator-bounded
    accounting universe is exactly the shard assigned to it.  The parent
    universe is only an allow-list for the expected shard; the child receipt
    itself must have source == eligible == expected and no exclusions.  The
    child receipt is evidence, not a merge artifact: the parent writes the
    sole authoritative receipt.
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
    if not set(expected) <= set(eligible_universe):
        raise ValueError(f"shard accounting expected stems outside parent universe: {path}")
    if set(receipt["source"]["stems"]) != set(expected):
        raise ValueError(f"shard accounting source mismatch: {path}")
    if set(receipt["eligible"]["stems"]) != set(expected):
        raise ValueError(f"shard accounting eligible shard mismatch: {path}")
    if receipt.get("exclusions") != []:
        raise ValueError(f"shard accounting exclusions are not empty: {path}")
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
    is_silence,
    RIA_VARIANTS, replace_ria_variants, normalize_punct_inline,
    _ASCII_TO_CJK_PUNCT,
    normalize_reference_numerals, normalize_authority_reference_numerals,
    validate_ctc_transcript_bundle,
    validate_ctc_authority_bundle,
    make_pipeline_accounting_receipt, write_pipeline_accounting_receipt,
    read_pipeline_accounting_receipt, validate_pipeline_accounting_receipt,
    make_pipeline_run_id, cuda_visible_token,
)

NVASR_NONLEXICAL_MFA_LABELS = frozenset({"sil", "sp", "spn", "<eps>"})

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


def _operator_bounded_accounting_universe(
    source_stems: list[str],
    eligible_stems: list[str],
    source_exclusions: dict[str, str],
    selected_stems: list[str] | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return the v2 accounting universe for one operator invocation.

    ``selected_stems is None`` means the invocation is unbounded and the
    frozen source/eligible/exclusion partition is retained.  Any explicit
    stems-file or offset/limit selection passes the actual selected eligible
    stems and receives an exact, exclusion-free shard universe.  The helper
    validates its inputs rather than normalizing away duplicate or malformed
    evidence.
    """
    def canonical_stems(value: object, field: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        if any(not isinstance(stem, str) or not stem or Path(stem).name != stem
               for stem in value):
            raise ValueError(f"{field} contains invalid stem")
        if len(value) != len(set(value)):
            raise ValueError(f"{field} contains duplicate stems")
        if value != sorted(value):
            raise ValueError(f"{field} must be sorted")
        return list(value)

    source = canonical_stems(source_stems, "source_stems")
    eligible = canonical_stems(eligible_stems, "eligible_stems")
    if not isinstance(source_exclusions, dict):
        raise ValueError("source_exclusions must be a mapping")
    exclusion_stems = canonical_stems(list(source_exclusions), "exclusion_stems")
    if any(not isinstance(reason, str) or not reason.strip()
           for reason in source_exclusions.values()):
        raise ValueError("source_exclusions contains an invalid reason")
    source_set = set(source)
    eligible_set = set(eligible)
    exclusion_set = set(exclusion_stems)
    if eligible_set & exclusion_set:
        raise ValueError("eligible and excluded stems overlap")
    if not eligible_set <= source_set or not exclusion_set <= source_set:
        raise ValueError("eligible/excluded stem is outside source universe")
    if source_set != eligible_set | exclusion_set:
        raise ValueError("source is not the exact eligible/exclusion partition")

    if selected_stems is None:
        return source, eligible, dict(source_exclusions)
    selected = canonical_stems(selected_stems, "selected_stems")
    if not selected:
        raise ValueError("selected accounting universe is empty")
    if not set(selected) <= eligible_set:
        raise ValueError("selected stem is outside eligible universe")
    return selected, selected.copy(), {}


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


def _nvasr_semantic_axis_member(surface: object, *, candidate_kind=None) -> bool:
    """Return whether a row occupies the canonical compact semantic axis."""
    text = str(surface or "").strip()
    return bool(
        text
        and candidate_kind is None
        and not re.fullmatch(r"<\|[^|]+\|>", text)
        and not is_nvv_token(text)
        and text.casefold() not in NVASR_NONLEXICAL_MFA_LABELS
        and not is_silence(text)
        and not is_punct(text)
    )


def _nvasr_semantic_identity(surface: object) -> str:
    """Return the stable identity used by the compact semantic axis."""
    return str(surface or "").strip()


def _nvasr_anchor_from_frames(start_frame: int, end_frame: int, *,
                              query_frames: int = QUERY_FRAMES,
                              frame_ms: int = FRAME_MS) -> dict:
    """Build immutable speech-time evidence from a half-open CTC frame run.

    Coordinates are rounded to six decimal places using the producer's
    documented half-open-frame convention.  The resulting interval is an
    evidence anchor only; its width is never a physiological duration.
    """
    half_frame_s = frame_ms / 2000.0
    start = (start_frame - query_frames) * frame_ms / 1000.0 - half_frame_s
    end = (end_frame - query_frames) * frame_ms / 1000.0 - half_frame_s

    def round_half_up(value: float) -> float:
        return float(Decimal(str(value)).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP))

    speech_start_frame = start_frame - query_frames
    speech_end_frame = end_frame - query_frames
    return {
        "schema": NVASR_SPIKE_ANCHOR_SCHEMA,
        "raw_start_frame": start_frame,
        "raw_end_frame": end_frame,
        "start": round_half_up(start),
        "end": round_half_up(end),
        "ordered_source_frame_ids": list(range(start_frame, end_frame)),
        "raw_frame_count": end_frame - start_frame,
        "query_frames": query_frames,
        "frame_ms": frame_ms,
        "speech_start_frame": speech_start_frame,
        "speech_end_frame": speech_end_frame,
        "coordinate_system": NVASR_ANCHOR_COORDINATE_SYSTEM,
        "quantization": NVASR_ANCHOR_QUANTIZATION,
    }


def _nvasr_valid_anchor(value: object) -> bool:
    """Validate the complete v2 anchor against locked producer runtime values."""
    if not isinstance(value, dict):
        return False
    required = {
        "schema", "raw_start_frame", "raw_end_frame",
        "start", "end", "ordered_source_frame_ids", "raw_frame_count",
        "query_frames", "frame_ms", "speech_start_frame", "speech_end_frame",
        "coordinate_system", "quantization",
    }
    if set(value) != required:
        return False
    start, end = value.get("start"), value.get("end")
    frame_ids = value.get("ordered_source_frame_ids")
    count = value.get("raw_frame_count")
    raw_start = value.get("raw_start_frame")
    raw_end = value.get("raw_end_frame")
    query_frames = value.get("query_frames")
    frame_ms = value.get("frame_ms")
    speech_start = value.get("speech_start_frame")
    speech_end = value.get("speech_end_frame")
    if (not isinstance(start, (int, float)) or isinstance(start, bool)
            or not math.isfinite(float(start))
            or not isinstance(end, (int, float)) or isinstance(end, bool)
            or not math.isfinite(float(end)) or end <= start
            or not isinstance(frame_ids, list)):
        return False
    if (not frame_ids
            or any(not isinstance(frame, int) or isinstance(frame, bool)
                   or frame < 0 for frame in frame_ids)
            or frame_ids != list(range(frame_ids[0], frame_ids[-1] + 1))):
        return False
    if (not isinstance(count, int) or isinstance(count, bool)
            or count <= 0 or count != len(frame_ids)
            or not isinstance(raw_start, int) or isinstance(raw_start, bool)
            or not isinstance(raw_end, int) or isinstance(raw_end, bool)
            or raw_start < 0 or raw_end <= raw_start
            or frame_ids != list(range(raw_start, raw_end))
            or count != raw_end - raw_start
            or query_frames != QUERY_FRAMES
            or frame_ms != FRAME_MS
            or speech_start != raw_start - QUERY_FRAMES
            or speech_end != raw_end - QUERY_FRAMES
            or value.get("schema") != NVASR_SPIKE_ANCHOR_SCHEMA
            or value.get("coordinate_system") != NVASR_ANCHOR_COORDINATE_SYSTEM
            or value.get("quantization") != NVASR_ANCHOR_QUANTIZATION):
        return False
    expected = _nvasr_anchor_from_frames(
        raw_start, raw_end, query_frames=QUERY_FRAMES, frame_ms=FRAME_MS)
    if value != expected:
        return False
    return math.isclose(
        float(end) - float(start), count * FRAME_MS / 1000.0,
        abs_tol=1e-9)


def _nvasr_anchor_binding_reasons(row: dict) -> list[str]:
    """Bind a complete spike anchor to its candidate and runtime constants."""
    reasons: list[str] = []
    if row.get("query_frames") != QUERY_FRAMES:
        reasons.append("ctc_spike_anchor_query_frames_mismatch")
    if row.get("frame_ms") != FRAME_MS:
        reasons.append("ctc_spike_anchor_frame_ms_mismatch")
    anchor = row.get("ctc_spike_anchor")
    if not isinstance(anchor, dict):
        return reasons + ["ctc_spike_anchor_v2_malformed"]
    raw_start = row.get("raw_start_frame")
    raw_end = row.get("raw_end_frame")
    if (not isinstance(raw_start, int) or isinstance(raw_start, bool)
            or not isinstance(raw_end, int) or isinstance(raw_end, bool)
            or raw_start < 0 or raw_end <= raw_start):
        return reasons + ["ctc_spike_anchor_candidate_frames_invalid"]
    if (anchor.get("raw_start_frame") != raw_start
            or anchor.get("raw_end_frame") != raw_end
            or anchor.get("ordered_source_frame_ids") != list(range(
                raw_start, raw_end))
            or anchor.get("raw_frame_count") != raw_end - raw_start
            or row.get("raw_frame_count") != raw_end - raw_start
            or anchor.get("speech_start_frame") != row.get(
                "speech_start_frame")
            or anchor.get("speech_end_frame") != row.get("speech_end_frame")
            or row.get("speech_frame_count") != raw_end - raw_start):
        reasons.append("ctc_spike_anchor_frame_binding_invalid")
    expected = _nvasr_anchor_from_frames(
        raw_start, raw_end, query_frames=QUERY_FRAMES, frame_ms=FRAME_MS)
    if anchor != expected:
        reasons.append("ctc_spike_anchor_coordinate_binding_invalid")
    if not _nvasr_valid_anchor(anchor):
        reasons.append("ctc_spike_anchor_v2_malformed")
    return list(dict.fromkeys(reasons))


def _nvasr_candidate_coordinate_binding_reasons(row: dict) -> list[str]:
    """Validate every immutable raw/speech frame and second coordinate."""
    reasons: list[str] = []
    raw_start = row.get("raw_start_frame")
    raw_end = row.get("raw_end_frame")
    speech_start = row.get("speech_start_frame")
    speech_end = row.get("speech_end_frame")
    if (not isinstance(raw_start, int) or isinstance(raw_start, bool)
            or not isinstance(raw_end, int) or isinstance(raw_end, bool)
            or not isinstance(speech_start, int)
            or isinstance(speech_start, bool)
            or not isinstance(speech_end, int) or isinstance(speech_end, bool)
            or raw_start < 0 or raw_end <= raw_start):
        return ["nvasr_candidate_frame_coordinates_invalid"]
    count = raw_end - raw_start
    if (row.get("raw_frame_count") != count
            or row.get("speech_frame_count") != count
            or speech_start != raw_start - QUERY_FRAMES
            or speech_end != raw_end - QUERY_FRAMES):
        reasons.append("nvasr_candidate_frame_count_or_offset_invalid")
    if row.get("query_frames") != QUERY_FRAMES:
        reasons.append("nvasr_candidate_query_frames_invalid")
    if row.get("frame_ms") != FRAME_MS:
        reasons.append("nvasr_candidate_frame_ms_invalid")
    expected_raw = [raw_start * FRAME_MS / 1000.0,
                    raw_end * FRAME_MS / 1000.0]
    expected_speech = [speech_start * FRAME_MS / 1000.0,
                       speech_end * FRAME_MS / 1000.0]
    for values, expected, label in (
            ([row.get("raw_start_s"), row.get("raw_end_s")],
             expected_raw, "raw_coordinates"),
            (row.get("raw_span"), expected_raw, "raw_span"),
            ([row.get("speech_start_s"), row.get("speech_end_s")],
             expected_speech, "speech_coordinates"),
            (row.get("speech_span"), expected_speech, "speech_span")):
        if (not isinstance(values, (list, tuple)) or len(values) != 2
                or any(not isinstance(value, (int, float))
                       or isinstance(value, bool)
                       or not math.isfinite(float(value))
                       or not math.isclose(float(value), expected[index],
                                           abs_tol=1e-9)
                       for index, value in enumerate(values))):
            reasons.append(f"nvasr_candidate_{label}_binding_invalid")
    return list(dict.fromkeys(reasons))


_NVASR_RAW_EVENT_FIELDS = frozenset({
    "surface", "source", "token_id", "ordered_source_frame_ids",
})
_NVASR_RAW_EVENT_SOURCES = frozenset({"ctc", "blank_run"})


def _nvasr_raw_event_diagnostic(event: dict) -> dict:
    """Project one raw event without assigning canonical semantics to it."""
    return {
        "surface": event["surface"],
        "source": event["source"],
        "token_id": event["token_id"],
        "ordered_source_frame_ids": list(
            range(event["start"], event["end"])),
    }


def _nvasr_raw_event_valid(value: object) -> bool:
    """Validate one exact raw-timeline event diagnostic."""
    if not isinstance(value, dict) or set(value) != _NVASR_RAW_EVENT_FIELDS:
        return False
    frames = value.get("ordered_source_frame_ids")
    token_id = value.get("token_id")
    return (
        isinstance(value.get("surface"), str)
        and value.get("source") in _NVASR_RAW_EVENT_SOURCES
        and isinstance(token_id, int) and not isinstance(token_id, bool)
        and token_id >= 0
        and isinstance(frames, list) and bool(frames)
        and all(isinstance(frame, int) and not isinstance(frame, bool)
                and frame >= 0 for frame in frames)
        and frames == list(range(frames[0], frames[-1] + 1))
    )


def _nvasr_raw_mapping_key(row: dict) -> dict | None:
    raw_mapping_key = row.get("raw_timeline_mapping_key")
    if raw_mapping_key is not None:
        return raw_mapping_key if isinstance(raw_mapping_key, dict) else None
    mapping_key = row.get("mapping_key")
    if mapping_key is not None:
        return mapping_key if isinstance(mapping_key, dict) else None
    if ("left_lexical_ordinal" not in row
            and "right_lexical_ordinal" not in row):
        return None
    return {
        "left_lexical_ordinal": row.get("left_lexical_ordinal"),
        "right_lexical_ordinal": row.get("right_lexical_ordinal"),
    }


def _nvasr_raw_candidate_value(row: dict, durable: str, producer: str):
    return row.get(durable) if durable in row else row.get(producer)


def _nvasr_raw_timeline_evidence_material(row: dict) -> dict:
    """Return the canonical JSON material binding a candidate raw window."""
    return {
        "schema": row.get("raw_timeline_neighbors_schema"),
        "candidate_id": row.get("candidate_id"),
        "candidate_surface": _nvasr_raw_candidate_value(
            row, "candidate_surface", "surface"),
        "candidate_source": _nvasr_raw_candidate_value(
            row, "candidate_source", "source"),
        "candidate_token_id": _nvasr_raw_candidate_value(
            row, "candidate_token_id", "token_id"),
        "candidate_token_ids": _nvasr_raw_candidate_value(
            row, "candidate_token_ids", "token_ids"),
        "raw_start_frame": row.get("raw_start_frame"),
        "raw_end_frame": row.get("raw_end_frame"),
        "query_frames": row.get("query_frames"),
        "frame_ms": row.get("frame_ms"),
        "raw_timeline_index": row.get("raw_timeline_index"),
        "raw_timeline_event_count": row.get("raw_timeline_event_count"),
        "mapping_key": _nvasr_raw_mapping_key(row),
        "raw_timeline_neighbors": row.get("raw_timeline_neighbors"),
    }


def _nvasr_raw_timeline_evidence_sha256(row: dict) -> str:
    """Hash the exact candidate/raw-neighbour evidence contract."""
    payload = json.dumps(
        _nvasr_raw_timeline_evidence_material(row),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nvasr_raw_timeline_sequence_reasons(value: object) -> list[str]:
    """Validate the producer's complete ordered raw-event sequence."""
    if not isinstance(value, list) or not value:
        return ["raw_timeline_events_empty_or_malformed"]
    reasons: list[str] = []
    for index, event in enumerate(value):
        if not _nvasr_raw_event_valid(event):
            reasons.append(f"raw_timeline_event_invalid:{index}")
    if reasons:
        return reasons
    for index, (left, right) in enumerate(zip(value, value[1:])):
        if (left["ordered_source_frame_ids"][-1]
                >= right["ordered_source_frame_ids"][0]):
            reasons.append(f"raw_timeline_event_order_invalid:{index}")
    return reasons


def _nvasr_raw_timeline_contract_reasons(
        row: dict, *, timeline_events: list[dict] | None = None) -> list[str]:
    """Validate one immutable raw-neighbour window and its candidate binding.

    Raw neighbours are direct decoder-timeline events, not canonical semantic
    neighbours. Their side nullability is determined solely by the persisted
    raw event index/count. The digest makes the exact identities, token IDs,
    frame runs, candidate coordinates, and raw mapping key tamper-evident after
    the full producer timeline is no longer available to a consumer.
    """
    reasons: list[str] = []
    if (row.get("raw_timeline_neighbors_schema") !=
            NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA):
        reasons.append("raw_timeline_neighbors_schema_invalid")

    index = row.get("raw_timeline_index")
    count = row.get("raw_timeline_event_count")
    if (not isinstance(index, int) or isinstance(index, bool)
            or not isinstance(count, int) or isinstance(count, bool)
            or count <= 0 or index < 0 or index >= count):
        reasons.append("raw_timeline_position_invalid")

    neighbors = row.get("raw_timeline_neighbors")
    if not isinstance(neighbors, dict) or set(neighbors) != {"left", "right"}:
        reasons.append("raw_timeline_neighbors_structure_invalid")
        neighbors = {"left": None, "right": None}

    mapping_key = _nvasr_raw_mapping_key(row)
    if (row.get("nvasr_candidate_schema_version") ==
            NVASR_CANDIDATE_SCHEMA_VERSION
            and "raw_timeline_mapping_key" not in row):
        reasons.append("raw_timeline_mapping_key_required")
    if (not isinstance(mapping_key, dict)
            or set(mapping_key) != {
                "left_lexical_ordinal", "right_lexical_ordinal"}):
        reasons.append("raw_timeline_mapping_key_invalid")
        mapping_key = {"left_lexical_ordinal": None,
                       "right_lexical_ordinal": None}
    elif any(value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0)
            for value in mapping_key.values()):
        reasons.append("raw_timeline_mapping_key_invalid")

    raw_start = row.get("raw_start_frame")
    raw_end = row.get("raw_end_frame")
    token_id = _nvasr_raw_candidate_value(
        row, "candidate_token_id", "token_id")
    token_ids = _nvasr_raw_candidate_value(
        row, "candidate_token_ids", "token_ids")
    surface = _nvasr_raw_candidate_value(
        row, "candidate_surface", "surface")
    source = _nvasr_raw_candidate_value(
        row, "candidate_source", "source")
    candidate_valid = (
        isinstance(row.get("candidate_id"), str) and bool(row["candidate_id"])
        and isinstance(surface, str) and bool(surface)
        and source in _NVASR_RAW_EVENT_SOURCES
        and isinstance(token_id, int) and not isinstance(token_id, bool)
        and token_id >= 0
        and isinstance(raw_start, int) and not isinstance(raw_start, bool)
        and isinstance(raw_end, int) and not isinstance(raw_end, bool)
        and 0 <= raw_start < raw_end
        and row.get("query_frames") == QUERY_FRAMES
        and row.get("frame_ms") == FRAME_MS
        and isinstance(token_ids, list)
        and token_ids == [token_id] * (raw_end - raw_start)
    )
    if not candidate_valid:
        reasons.append("raw_timeline_candidate_identity_invalid")

    for side in ("left", "right"):
        neighbor = neighbors[side]
        if neighbor is not None and not _nvasr_raw_event_valid(neighbor):
            reasons.append(f"raw_timeline_{side}_event_invalid")

    if (isinstance(index, int) and not isinstance(index, bool)
            and isinstance(count, int) and not isinstance(count, bool)
            and count > 0 and 0 <= index < count):
        left_required = index > 0
        right_required = index + 1 < count
        if (neighbors["left"] is None) != (not left_required):
            reasons.append("raw_timeline_left_presence_invalid")
        if (neighbors["right"] is None) != (not right_required):
            reasons.append("raw_timeline_right_presence_invalid")
    for side in ("left", "right"):
        if (mapping_key.get(f"{side}_lexical_ordinal") is not None
                and neighbors[side] is None):
            reasons.append(f"raw_timeline_{side}_required_by_mapping_key")

    if candidate_valid and _nvasr_raw_event_valid(neighbors["left"]):
        if neighbors["left"]["ordered_source_frame_ids"][-1] >= raw_start:
            reasons.append("raw_timeline_left_frame_order_invalid")
    if candidate_valid and _nvasr_raw_event_valid(neighbors["right"]):
        if neighbors["right"]["ordered_source_frame_ids"][0] < raw_end:
            reasons.append("raw_timeline_right_frame_order_invalid")

    digest = row.get("raw_timeline_evidence_sha256")
    if (not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        reasons.append("raw_timeline_evidence_digest_invalid")
    else:
        try:
            expected_digest = _nvasr_raw_timeline_evidence_sha256(row)
        except (TypeError, ValueError):
            expected_digest = None
        if digest != expected_digest:
            reasons.append("raw_timeline_evidence_digest_mismatch")

    if timeline_events is not None:
        sequence_reasons = _nvasr_raw_timeline_sequence_reasons(
            timeline_events)
        reasons.extend(sequence_reasons)
        if (not sequence_reasons and isinstance(index, int)
                and not isinstance(index, bool) and isinstance(count, int)
                and not isinstance(count, bool) and count == len(timeline_events)
                and 0 <= index < count and candidate_valid):
            candidate_event = {
                "surface": surface,
                "source": source,
                "token_id": token_id,
                "ordered_source_frame_ids": list(range(raw_start, raw_end)),
            }
            expected_neighbors = {
                "left": timeline_events[index - 1] if index else None,
                "right": (timeline_events[index + 1]
                          if index + 1 < count else None),
            }
            if timeline_events[index] != candidate_event:
                reasons.append("raw_timeline_candidate_event_mismatch")
            if neighbors != expected_neighbors:
                reasons.append("raw_timeline_neighbor_identity_mismatch")
        elif (not sequence_reasons and isinstance(count, int)
              and not isinstance(count, bool) and count != len(timeline_events)):
            reasons.append("raw_timeline_event_count_mismatch")
    return list(dict.fromkeys(reasons))


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
    # 疑问/惊讶/确认语气词永不为 NVV (由 Qwen 转写为 CJK 原词), 屏蔽放在
    # bias 之后, 避免 bias 复活这些槽位.
    if NVV_SUPPRESSED_IDS:
        decoded[..., sorted(NVV_SUPPRESSED_IDS)] = -float("inf")
    return decoded


def _decode_candidate_surface(token_id: int, token_decoder) -> str:
    """Decode one CTC token without importing or constructing an ASR provider."""
    if token_decoder is None:
        return ""
    if callable(token_decoder):
        value = token_decoder(token_id)
    else:
        value = token_decoder[token_id]
    if value is None:
        return ""
    return str(value).strip()


def _candidate_coordinates(start_frame: int, end_frame: int, *,
                           query_frames: int, frame_ms: int) -> dict:
    """Return immutable raw and speech-relative coordinates for one occurrence."""
    raw_start_s = start_frame * frame_ms / 1000
    raw_end_s = end_frame * frame_ms / 1000
    speech_start_frame = start_frame - query_frames
    speech_end_frame = end_frame - query_frames
    return {
        "raw_start_frame": start_frame,
        "raw_end_frame": end_frame,
        "speech_start_frame": speech_start_frame,
        "speech_end_frame": speech_end_frame,
        "raw_start_s": round(raw_start_s, 6),
        "raw_end_s": round(raw_end_s, 6),
        "speech_start_s": round(speech_start_frame * frame_ms / 1000, 6),
        "speech_end_s": round(speech_end_frame * frame_ms / 1000, 6),
        "raw_span": [round(raw_start_s, 6), round(raw_end_s, 6)],
        "speech_span": [
            round(speech_start_frame * frame_ms / 1000, 6),
            round(speech_end_frame * frame_ms / 1000, 6),
        ],
        "raw_frame_count": end_frame - start_frame,
        "speech_frame_count": end_frame - start_frame,
    }


def extract_nvasr_candidate_timeline(
    frame_token_ids,
    diagnostic_text: str = "",
    *,
    token_decoder=None,
    token_surfaces=None,
    stem: str | None = None,
    query_frames: int = QUERY_FRAMES,
    frame_ms: int = FRAME_MS,
    blank_id: int = BLANK_ID,
    ellipsis_id: int = ELLIPSIS_ID,
    pause_threshold: int = PAUSE_FRAMES_DEFAULT,
) -> dict:
    """Extract lexical and NVV/punctuation evidence without a provider.

    ``frame_token_ids`` are encoder-frame CTC argmax IDs, so their coordinates
    are the raw encoder axis.  Every non-blank lexical CTC run becomes one
    ordered lexical occurrence; separated duplicate surfaces remain separate
    occurrences.  NVV/punctuation candidates are emitted once per run and
    carry neighboring lexical ordinals.  Long speech-only blank runs add one
    synthetic ellipsis occurrence at the same midpoint used by the normal CTC
    path.  Raw coordinates are never rewritten; speech coordinates are the
    raw coordinates translated by ``query_frames``.

    ``token_decoder`` may be a callable accepting one token ID, or
    ``token_surfaces`` may map IDs to already-decoded surfaces.  Neither seam
    imports, loads, or constructs an ASR provider.
    """
    if not isinstance(diagnostic_text, str):
        raise TypeError("diagnostic_text must be a string")
    if query_frames != QUERY_FRAMES:
        raise ValueError(f"query_frames must be exactly {QUERY_FRAMES}")
    if frame_ms != FRAME_MS:
        raise ValueError(f"frame_ms must be exactly {FRAME_MS}")
    if not isinstance(pause_threshold, int) or pause_threshold <= 0:
        raise ValueError("pause_threshold must be a positive integer")
    if token_decoder is not None and token_surfaces is not None:
        raise ValueError("pass token_decoder or token_surfaces, not both")

    if hasattr(frame_token_ids, "tolist"):
        frame_token_ids = frame_token_ids.tolist()
    frame_token_ids = [int(token_id) for token_id in frame_token_ids]
    decoder = token_decoder if token_decoder is not None else token_surfaces

    def surface_for(token_id: int) -> str:
        try:
            return _decode_candidate_surface(token_id, decoder)
        except (KeyError, IndexError, TypeError, ValueError):
            # Unknown IDs are not lexical candidates, but must not disturb the
            # diagnostic timeline or the ordinary CTC path.
            return ""

    def candidate_kind(surface: str, token_id: int) -> str | None:
        stripped = surface.strip()
        if token_id == ellipsis_id or stripped == "…":
            return "punctuation"
        if token_id in NVV_SUPPRESSED_IDS:
            return None
        if NVV_START <= token_id <= NVV_END:
            return "nvv"
        if re.fullmatch(r"\[[A-Za-z][^\]]*\]", stripped):
            return "nvv"
        if is_nvv_token(stripped):
            return "nvv"
        if len(stripped) == 1 and stripped in ALLOWED_PUNCT:
            return "punctuation"
        return None

    raw_events: list[dict] = []
    index = 0
    while index < len(frame_token_ids):
        token_id = frame_token_ids[index]
        end = index + 1
        while end < len(frame_token_ids) and frame_token_ids[end] == token_id:
            end += 1
        if token_id != blank_id:
            surface = surface_for(token_id)
            kind = candidate_kind(surface, token_id)
            raw_events.append({
                "start": index,
                "end": end,
                "token_id": token_id,
                "token_ids": [token_id] * (end - index),
                "surface": surface,
                "candidate_kind": kind,
                "source": "ctc",
            })
        index = end

    # Keep the synthetic ellipsis provenance separate from the CTC token run.
    blank_start = 0
    while blank_start < len(frame_token_ids):
        if frame_token_ids[blank_start] != blank_id:
            blank_start += 1
            continue
        blank_end = blank_start + 1
        while (blank_end < len(frame_token_ids)
               and frame_token_ids[blank_end] == blank_id):
            blank_end += 1
        if (blank_start >= query_frames
                and blank_end - blank_start >= pause_threshold):
            midpoint = blank_start + (blank_end - blank_start) // 2
            raw_events.append({
                "start": midpoint,
                "end": midpoint + 1,
                "token_id": ellipsis_id,
                "token_ids": [ellipsis_id],
                "surface": "…",
                "candidate_kind": "punctuation",
                "source": "blank_run",
            })
        blank_start = blank_end

    # Query-only occurrences are encoder metadata, not speech candidates.  A
    # candidate or lexical group crossing the prefix is likewise excluded: the
    # locked fusion envelope requires every published raw span to begin on the
    # speech axis.
    raw_events = [event for event in raw_events
                  if event["start"] >= query_frames]
    raw_events.sort(key=lambda event: (
        event["start"], event["end"], event["token_id"]))

    def is_lexical_surface(surface: str, kind: str | None) -> bool:
        return _nvasr_semantic_axis_member(surface, candidate_kind=kind)

    lexical_events = [
        event for event in raw_events
        if is_lexical_surface(event["surface"], event["candidate_kind"])
    ]
    lexical_event_ids = {id(event) for event in lexical_events}
    lexical_occurrences = []
    surface_occurrences: dict[str, int] = {}
    for lexical_ordinal, event in enumerate(lexical_events):
        surface = event["surface"]
        surface_occurrence = surface_occurrences.get(surface, 0)
        surface_occurrences[surface] = surface_occurrence + 1
        lexical_id = f"nvasr-lexical-{lexical_ordinal:04d}"
        lexical_row = {
            "lexical_ordinal": lexical_ordinal,
            "lexical_occurrence_id": lexical_id,
            "occurrence_id": lexical_id,
            "surface": surface,
            "source": event["source"],
            "kind": "lexical",
            "token_id": event["token_id"],
            "token_ids": list(event["token_ids"]),
            "query_frames": query_frames,
            "frame_ms": frame_ms,
            "surface_occurrence": surface_occurrence,
            "surface_occurrence_index": surface_occurrence,
            "surface_occurrence_id": f"{surface}#{surface_occurrence}",
        }
        lexical_row.update(_candidate_coordinates(
            event["start"], event["end"],
            query_frames=query_frames, frame_ms=frame_ms))
        lexical_occurrences.append(lexical_row)

    lexical_by_event_id = {
        id(event): row["lexical_ordinal"]
        for event, row in zip(lexical_events, lexical_occurrences)
    }
    lexical_by_ordinal = {
        row["lexical_ordinal"]: row for row in lexical_occurrences
    }
    right_lexical_by_event_id: dict[int, int | None] = {}
    next_lexical_ordinal: int | None = None
    for event in reversed(raw_events):
        right_lexical_by_event_id[id(event)] = next_lexical_ordinal
        if id(event) in lexical_event_ids:
            next_lexical_ordinal = lexical_by_event_id[id(event)]

    raw_timeline_events = [
        _nvasr_raw_event_diagnostic(event) for event in raw_events
    ]
    candidates = []
    left_lexical_ordinal: int | None = None
    for ordinal, event in enumerate(raw_events):
        if id(event) in lexical_event_ids:
            left_lexical_ordinal = lexical_by_event_id[id(event)]
            continue
        kind = event["candidate_kind"]
        if kind is None:
            continue
        start = event["start"]
        end = event["end"]
        surface = event["surface"]
        left_event = next((raw_events[pos - 1] for pos in range(len(raw_events))
                           if raw_events[pos] is event and pos > 0), None)
        right_event = next((raw_events[pos + 1] for pos in range(len(raw_events))
                            if raw_events[pos] is event
                            and pos + 1 < len(raw_events)), None)

        semantic_neighbors = []
        for side, lexical_ordinal in (
                ("left", left_lexical_ordinal),
                ("right", right_lexical_by_event_id[id(event)])):
            if lexical_ordinal is None:
                continue
            lexical = lexical_by_ordinal[lexical_ordinal]
            semantic_neighbors.append({
                "side": side,
                "lexical_ordinal": lexical_ordinal,
                "occurrence_id": lexical["occurrence_id"],
                "surface": lexical["surface"],
                "surface_occurrence": lexical["surface_occurrence"],
            })
        candidate = {
            "candidate_id": f"nvasr-candidate-{len(candidates):04d}",
            "occurrence": len(candidates),
            "label": surface,
            "surface": surface,
            "kind": kind,
            "source": event["source"],
            "mapping_basis": NVASR_MAPPING_BASIS,
            "mapping_axis": NVASR_MAPPING_AXIS,
            "nvasr_candidate_schema_version": NVASR_CANDIDATE_SCHEMA_VERSION,
            "query_frames": query_frames,
            "frame_ms": frame_ms,
            "token_id": event["token_id"],
            "token_ids": list(event["token_ids"]),
            "diagnostic": diagnostic_text,
            "left_lexical_ordinal": left_lexical_ordinal,
            "right_lexical_ordinal": right_lexical_by_event_id[id(event)],
            "raw_timeline_mapping_key": {
                "left_lexical_ordinal": left_lexical_ordinal,
                "right_lexical_ordinal": right_lexical_by_event_id[id(event)],
            },
            "ordered_semantic_neighbors": semantic_neighbors,
            "raw_timeline_neighbors_schema":
                NVASR_RAW_TIMELINE_NEIGHBORS_SCHEMA,
            "raw_timeline_index": ordinal,
            "raw_timeline_event_count": len(raw_events),
            "raw_timeline_neighbors": {
                "left": (_nvasr_raw_event_diagnostic(left_event)
                         if left_event is not None else None),
                "right": (_nvasr_raw_event_diagnostic(right_event)
                          if right_event is not None else None),
            },
        }
        candidate.update(_candidate_coordinates(
            start, end, query_frames=query_frames, frame_ms=frame_ms))
        candidate["ctc_spike_anchor"] = _nvasr_anchor_from_frames(
            int(candidate["raw_start_frame"]),
            int(candidate["raw_end_frame"]),
            query_frames=query_frames, frame_ms=frame_ms)
        candidate["raw_timeline_evidence_sha256"] = (
            _nvasr_raw_timeline_evidence_sha256(candidate))
        candidates.append(candidate)

    timeline = {
        # The envelope name remains stable for readers that dispatch on it;
        # the mandatory numeric version below carries the contract revision.
        "schema": "nvasr-candidate-timeline-v1",
        "nvasr_candidate_schema_version": NVASR_CANDIDATE_SCHEMA_VERSION,
        "mapping_axis": NVASR_MAPPING_AXIS,
        "duration_s": round(
            max(0, len(frame_token_ids) - query_frames) * frame_ms / 1000,
            6,
        ),
        "query_frames": query_frames,
        "frame_ms": frame_ms,
        "semantic_axis": {
            "schema": NVASR_MAPPING_AXIS,
            "name": "non_nvv_compact_semantic",
            "ordinal_field": "lexical_ordinal",
            "excluded_kinds": ["nvv", "punctuation", "silence", "special"],
        },
        "ordered_semantic_neighbors": [
            {
                "lexical_ordinal": row["lexical_ordinal"],
                "occurrence_id": row["occurrence_id"],
                "surface": row["surface"],
                "surface_occurrence": row["surface_occurrence"],
            }
            for row in lexical_occurrences
        ],
        "raw_timeline_neighbors": raw_timeline_events,
        "diagnostic": diagnostic_text,
        "diagnostic_text": diagnostic_text,
        "lexical_occurrences": lexical_occurrences,
        "candidates": candidates,
    }
    if stem is not None:
        if not isinstance(stem, str) or not stem:
            raise ValueError("stem must be a non-empty string or null")
        timeline["stem"] = stem
    return timeline


# Descriptive alias retained for callers that use the envelope's verb.
build_nvasr_candidate_timeline = extract_nvasr_candidate_timeline


def _nvasr_candidate_provenance(candidate: dict, *, forced_span=None) -> dict:
    """Project one immutable timeline candidate onto a durable CTC row."""
    raw_span = [float(candidate["raw_start_s"]),
                float(candidate["raw_end_s"])]
    speech_span = [float(candidate["speech_start_s"]),
                   float(candidate["speech_end_s"])]
    anchor = candidate.get("ctc_spike_anchor")
    if not isinstance(anchor, dict):
        # Compatibility for direct legacy helper fixtures.  Extracted
        # producer timelines always carry the mandatory v2 anchor.
        anchor = _nvasr_anchor_from_frames(
            int(candidate["raw_start_frame"]),
            int(candidate["raw_end_frame"]))
    result = {
        "provenance_schema": NVASR_CANDIDATE_PROVENANCE_SCHEMA,
        "candidate_id": str(candidate["candidate_id"]),
        "nvasr_candidate_schema_version": NVASR_CANDIDATE_SCHEMA_VERSION,
        "candidate_kind": str(candidate["kind"]),
        "candidate_surface": str(candidate["surface"]),
        "candidate_source": str(candidate.get("source", "")),
        "candidate_token_id": int(candidate["token_id"]),
        "candidate_token_ids": list(candidate["token_ids"]),
        "raw_span": raw_span,
        "speech_span": speech_span,
        "raw_start_frame": int(candidate["raw_start_frame"]),
        "raw_end_frame": int(candidate["raw_end_frame"]),
        "speech_start_frame": int(candidate["speech_start_frame"]),
        "speech_end_frame": int(candidate["speech_end_frame"]),
        "raw_start_s": raw_span[0],
        "raw_end_s": raw_span[1],
        "speech_start_s": speech_span[0],
        "speech_end_s": speech_span[1],
        "raw_frame_count": (int(candidate["raw_end_frame"])
                            - int(candidate["raw_start_frame"])),
        "speech_frame_count": (int(candidate["speech_end_frame"])
                               - int(candidate["speech_start_frame"])),
        "query_frames": int(candidate.get("query_frames", QUERY_FRAMES)),
        "frame_ms": int(candidate.get("frame_ms", FRAME_MS)),
        "mapping_basis": NVASR_MAPPING_BASIS,
        "mapping_axis": NVASR_MAPPING_AXIS,
        "ordered_semantic_neighbors": list(
            candidate.get("ordered_semantic_neighbors", ())),
        "raw_timeline_neighbors": json.loads(json.dumps(
            candidate.get("raw_timeline_neighbors", {}),
            ensure_ascii=False)),
        "ctc_spike_anchor": dict(anchor),
        "mapping_key": {
            "left_lexical_ordinal": candidate.get("left_lexical_ordinal"),
            "right_lexical_ordinal": candidate.get("right_lexical_ordinal"),
        },
        "raw_timeline_mapping_key": dict(
            candidate.get("raw_timeline_mapping_key", {
                "left_lexical_ordinal": candidate.get(
                    "left_lexical_ordinal"),
                "right_lexical_ordinal": candidate.get(
                    "right_lexical_ordinal"),
            })),
        "mapping_outcome": "unique",
    }
    for key in (
            "raw_timeline_neighbors_schema", "raw_timeline_index",
            "raw_timeline_event_count", "raw_timeline_evidence_sha256"):
        if key in candidate:
            result[key] = candidate[key]
    if forced_span is not None:
        result["forced_span"] = [float(forced_span[0]), float(forced_span[1])]
    return result


def _deduplicate_adjacent_nvv_rows(words_pinyin: list[dict]) -> list[dict]:
    """Collapse decoder-duplicate NVV targets before candidate binding.

    SentencePiece preserves ``[Laughter][Laughter]`` as two forced targets.
    The user-facing surface intentionally owns one adjacent event, but that
    contraction must happen before provenance is attached.  Contracting bound
    rows afterwards used to construct a bare ``word/start/end`` dictionary and
    silently discard both acoustic candidate ledgers.

    The merged forced envelope remains explicit.  Candidate binding then has
    to select one unique raw candidate for that final output row; a tie remains
    a hard mapping error instead of being hidden by the deduplication.
    """
    result: list[dict] = []
    index = 0
    while index < len(words_pinyin):
        first = words_pinyin[index]
        label = str(first.get("word", ""))
        if not is_nvv_token(label):
            result.append(first)
            index += 1
            continue

        end = index + 1
        while (end < len(words_pinyin)
               and str(words_pinyin[end].get("word", "")) == label):
            end += 1
        group = words_pinyin[index:end]
        if len(group) == 1:
            result.append(first)
            index = end
            continue
        if any("candidate_kind" in row or "candidate_id" in row for row in group):
            raise ValueError("adjacent NVV deduplication must precede candidate binding")

        spans: list[list[float]] = []
        source_ordinals: list[int] = []
        source_ordinals_complete = True
        for row in group:
            try:
                start = float(row["start"])
                stop = float(row["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("adjacent NVV row has malformed forced span") from exc
            if (not math.isfinite(start) or not math.isfinite(stop)
                    or stop <= start):
                raise ValueError("adjacent NVV row has malformed forced span")
            spans.append([start, stop])
            ordinals = row.get("source_ctc_ordinals")
            if ordinals is None and "source_ctc_ordinal" in row:
                ordinals = [row["source_ctc_ordinal"]]
            if ordinals is None:
                source_ordinals_complete = False
            elif (not isinstance(ordinals, (list, tuple))
                  or any(not isinstance(value, int) or isinstance(value, bool)
                         or value < 0 for value in ordinals)):
                raise ValueError("adjacent NVV row has malformed source CTC ordinal")
            else:
                source_ordinals.extend(ordinals)

        if any(right[0] < left[0] or right[1] <= left[1]
               for left, right in zip(spans, spans[1:])):
            raise ValueError("adjacent NVV forced spans are not monotonic")
        if (source_ordinals_complete
                and source_ordinals != sorted(set(source_ordinals))):
            raise ValueError("adjacent NVV source CTC ordinals are not unique/monotonic")

        merged = dict(first)
        merged["end"] = spans[-1][1]
        merged.pop("source_ctc_ordinal", None)
        if source_ordinals_complete:
            merged["source_ctc_ordinals"] = source_ordinals
        merged["nvv_deduplication"] = {
            "schema": "nvv-adjacent-deduplication-v1",
            "label": label,
            "occurrence_count": len(group),
            "forced_occurrence_spans": spans,
        }
        result.append(merged)
        index = end
    return result


def _merge_ctc_row_metadata(rows: list[dict], *, word: str,
                            start: float, end: float,
                            fallback_ordinals: list[int] | None = None) -> dict:
    """Retain the leftmost row while unioning source CTC ordinals.

    Cardinality-changing normalizers must not manufacture a bare row: the
    leftmost row carries the durable producer evidence, while the consumed
    source ordinals explain the contraction.  Final physical/canonical
    coordinates are assigned by ``_rebase_final_token_sidecars``.
    """
    if not rows:
        raise ValueError("cannot merge an empty CTC row group")
    merged = dict(rows[0])
    ordinals: list[int] = []
    for position, row in enumerate(rows):
        values = row.get("source_ctc_ordinals")
        if values is None and "source_ctc_ordinal" in row:
            values = [row["source_ctc_ordinal"]]
        if values is None and fallback_ordinals is not None:
            values = [fallback_ordinals[position]]
        if values is None:
            continue
        if (not isinstance(values, (list, tuple))
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value < 0 for value in values)):
            raise ValueError("invalid source CTC ordinal metadata")
        ordinals.extend(values)
    if ordinals:
        merged["source_ctc_ordinals"] = sorted(set(ordinals))
        merged.pop("source_ctc_ordinal", None)
    merged.update({
        "word": word,
        "start": float(start),
        "end": float(end),
    })
    return merged


def _validate_emitted_nvasr_provenance(words_pinyin: list[dict]) -> list[str]:
    """Verify the final free-decode rows immediately before serialization."""
    required = (
        "provenance_schema", "candidate_id", "candidate_kind",
        "candidate_surface", "candidate_source", "raw_span", "speech_span",
        "forced_span",
        "mapping_basis", "mapping_axis", "mapping_outcome",
        "nvasr_candidate_schema_version", "ordered_semantic_neighbors",
        "raw_timeline_neighbors", "raw_timeline_neighbors_schema",
        "raw_timeline_index", "raw_timeline_event_count",
        "raw_timeline_evidence_sha256", "query_frames", "frame_ms",
        "ctc_spike_anchor",
    )
    errors: list[str] = []
    candidate_ids: set[str] = set()
    canonical_rows: list[tuple[int, dict, int]] = []
    canonical_surface_counts: dict[str, int] = {}
    canonical_ready = any("semantic_occurrence_id" in row
                          for row in words_pinyin)
    if canonical_ready:
        for position, row in enumerate(words_pinyin):
            text = str(row.get("word", "")).strip()
            if not _nvasr_semantic_axis_member(text):
                continue
            ordinal = len(canonical_rows)
            occurrence = canonical_surface_counts.get(text, 0)
            canonical_surface_counts[text] = occurrence + 1
            expected_id = f"nvasr-lexical-{ordinal:04d}"
            if (row.get("semantic_occurrence_id") != expected_id
                    or row.get("semantic_surface_occurrence") != occurrence):
                errors.append(
                    f"row {position}: canonical semantic identity/order invalid")
            canonical_rows.append((position, row, ordinal))
    canonical_neighbors: dict[int, list[dict]] = {}
    if canonical_ready:
        for position, row in enumerate(words_pinyin):
            if not is_nvv_token(str(row.get("word", "")).strip()):
                continue
            left = next((item for item in reversed(canonical_rows)
                         if item[0] < position), None)
            right = next((item for item in canonical_rows
                          if item[0] > position), None)
            expected = []
            for side, item in (("left", left), ("right", right)):
                if item is None:
                    continue
                _pos, neighbor, ordinal = item
                expected.append({
                    "side": side,
                    "lexical_ordinal": ordinal,
                    "occurrence_id": neighbor["semantic_occurrence_id"],
                    "surface": neighbor["word"],
                    "surface_occurrence": neighbor["semantic_surface_occurrence"],
                })
            canonical_neighbors[id(row)] = expected

    def valid_span(value: object) -> bool:
        return (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(point, (int, float))
                        and not isinstance(point, bool)
                        and math.isfinite(float(point)) for point in value)
                and 0 <= float(value[0]) < float(value[1]))

    for index, row in enumerate(words_pinyin):
        label = str(row.get("word", ""))
        nvv = is_nvv_token(label)
        has_nvv_provenance = (
            row.get("candidate_kind") == "nvv"
            or row.get("provenance_schema") == NVASR_CANDIDATE_PROVENANCE_SCHEMA)
        if not nvv:
            if has_nvv_provenance:
                errors.append(f"row {index}: NVV provenance attached to non-NVV {label!r}")
            continue
        missing = [key for key in required if key not in row]
        if missing:
            errors.append(
                f"row {index}: output NVV {label!r} missing provenance fields "
                f"{','.join(missing)}")
            continue
        candidate_id = row.get("candidate_id")
        if (not isinstance(candidate_id, str) or not candidate_id
                or candidate_id in candidate_ids):
            errors.append(f"row {index}: output NVV {label!r} has invalid candidate identity")
        else:
            candidate_ids.add(candidate_id)
        if (row.get("provenance_schema") != NVASR_CANDIDATE_PROVENANCE_SCHEMA
                or row.get("candidate_kind") != "nvv"
                or row.get("mapping_basis") != NVASR_MAPPING_BASIS
                or row.get("mapping_axis") != NVASR_MAPPING_AXIS
                or row.get("nvasr_candidate_schema_version") !=
                NVASR_CANDIDATE_SCHEMA_VERSION
                or row.get("mapping_outcome") != "unique"):
            errors.append(f"row {index}: output NVV {label!r} has invalid provenance contract")
        neighbors = row.get("ordered_semantic_neighbors")
        if (not isinstance(neighbors, list)
                or any(not isinstance(item, dict)
                       or item.get("side") not in {"left", "right"}
                       or not isinstance(item.get("lexical_ordinal"), int)
                       or not isinstance(item.get("occurrence_id"), str)
                       or not isinstance(item.get("surface"), str)
                       for item in neighbors)
                or [item.get("side") for item in neighbors]
                != sorted([item.get("side") for item in neighbors],
                          key=("left", "right").index)):
            errors.append(f"row {index}: output NVV {label!r} semantic neighbors invalid")
        if canonical_ready and neighbors != canonical_neighbors.get(id(row)):
            errors.append(
                f"row {index}: output NVV {label!r} canonical neighbor identity mismatch")
        raw_reasons = _nvasr_raw_timeline_contract_reasons(row)
        if raw_reasons:
            errors.append(
                f"row {index}: output NVV {label!r} raw neighbors invalid:"
                f"{','.join(raw_reasons)}")
        anchor_reasons = _nvasr_anchor_binding_reasons(row)
        if anchor_reasons:
            errors.append(
                f"row {index}: output NVV {label!r} spike anchor invalid:"
                f"{','.join(anchor_reasons)}")
        coordinate_reasons = _nvasr_candidate_coordinate_binding_reasons(row)
        if coordinate_reasons:
            errors.append(
                f"row {index}: output NVV {label!r} coordinates invalid:"
                f"{','.join(coordinate_reasons)}")
        if nvv_to_mfa(str(row.get("candidate_surface", ""))) != label:
            errors.append(f"row {index}: output NVV {label!r} candidate label mismatch")
        for span_key in ("raw_span", "speech_span", "forced_span"):
            if not valid_span(row.get(span_key)):
                errors.append(f"row {index}: output NVV {label!r} invalid {span_key}")
        forced = row.get("forced_span")
        if valid_span(forced):
            try:
                row_span = [float(row["start"]), float(row["end"])]
            except (KeyError, TypeError, ValueError):
                row_span = []
            if (len(row_span) != 2
                    or any(not math.isclose(float(forced[pos]), row_span[pos],
                                            abs_tol=1e-9)
                           for pos in range(2))):
                errors.append(f"row {index}: output NVV {label!r} forced span is stale")

        deduplication = row.get("nvv_deduplication")
        if deduplication is not None:
            spans = (deduplication.get("forced_occurrence_spans")
                     if isinstance(deduplication, dict) else None)
            count = (deduplication.get("occurrence_count")
                     if isinstance(deduplication, dict) else None)
            if (not isinstance(deduplication, dict)
                    or deduplication.get("schema") !=
                    "nvv-adjacent-deduplication-v1"
                    or deduplication.get("label") != label
                    or not isinstance(count, int) or isinstance(count, bool)
                    or count < 2 or not isinstance(spans, list)
                    or len(spans) != count
                    or any(not valid_span(span) for span in spans)
                    or not valid_span(forced)
                    or not math.isclose(float(spans[0][0]), float(forced[0]), abs_tol=1e-9)
                    or not math.isclose(float(spans[-1][1]), float(forced[1]), abs_tol=1e-9)):
                errors.append(f"row {index}: output NVV {label!r} invalid deduplication ledger")
    return errors


def attach_nvasr_candidate_provenance(
        words_pinyin: list[dict], punct_entries: list[dict],
        timeline: dict, *, strict_schema_v3: bool = False,
        strict_schema_v2: bool | None = None) -> list[str]:
    """Bind NVASR candidates to CTC rows by label and lexical neighbor ordinals.

    The candidate's raw encoder coordinates are copied verbatim.  Matching is
    occurrence-aware: two equal labels with the same lexical-neighbor key are
    not distinguishable and therefore reject the bundle.  This helper mutates
    only the in-memory producer rows and returns hard mapping errors for the
    caller to quarantine before writing any artifact.
    """
    candidates = list(timeline.get("candidates", ()))
    nvv_candidates = [row for row in candidates if row.get("kind") == "nvv"]
    nvv_candidate_positions = [index for index, row in enumerate(candidates)
                               if row.get("kind") == "nvv"]
    nvv_rows = [(index, row) for index, row in enumerate(words_pinyin)
                if is_nvv_token(row.get("word", ""))]

    # ``strict_schema_v2`` remains an API-only alias for older tests/callers;
    # it selects the current strict contract and never admits a version-2 row.
    strict_v3 = strict_schema_v3 or strict_schema_v2 is True
    semantic_axis = timeline.get("semantic_axis")
    if strict_v3 and (
            timeline.get("nvasr_candidate_schema_version") !=
            NVASR_CANDIDATE_SCHEMA_VERSION
            or timeline.get("mapping_axis") != NVASR_MAPPING_AXIS
            or timeline.get("query_frames") != QUERY_FRAMES
            or timeline.get("frame_ms") != FRAME_MS
            or not isinstance(semantic_axis, dict)
            or semantic_axis.get("schema") != NVASR_MAPPING_AXIS
            or semantic_axis.get("name") != "non_nvv_compact_semantic"
            or semantic_axis.get("ordinal_field") != "lexical_ordinal"
            or not isinstance(timeline.get("ordered_semantic_neighbors"), list)
            or not isinstance(timeline.get("raw_timeline_neighbors"), list)):
        return ["candidate timeline schema-v3 metadata is invalid"]
    if semantic_axis is not None and (
            not isinstance(semantic_axis, dict)
            or semantic_axis.get("schema") != NVASR_MAPPING_AXIS
            or semantic_axis.get("name") != "non_nvv_compact_semantic"
            or semantic_axis.get("ordinal_field") != "lexical_ordinal"):
        return ["candidate timeline semantic axis metadata is invalid"]
    timeline_lexical = timeline.get("lexical_occurrences")
    emitted_semantic = [
        row for row in words_pinyin
        if _nvasr_semantic_axis_member(row.get("word"))]
    if isinstance(timeline_lexical, list) and len(timeline_lexical) != len(
            emitted_semantic):
        return [
            "candidate timeline semantic axis round-trip count mismatch: "
            f"timeline={len(timeline_lexical)} emitted={len(emitted_semantic)}"
        ]
    if strict_v3:
        expected_axis = [{
            "lexical_ordinal": row.get("lexical_ordinal"),
            "occurrence_id": row.get("occurrence_id"),
            "surface": row.get("surface"),
            "surface_occurrence": row.get("surface_occurrence"),
        } for row in timeline_lexical or []]
        if timeline.get("ordered_semantic_neighbors") != expected_axis:
            return ["candidate timeline ordered semantic identity/order mismatch"]
        if not isinstance(timeline_lexical, list) or any(
                not isinstance(row, dict)
                or row.get("lexical_ordinal") != index
                or row.get("occurrence_id") != row.get("lexical_occurrence_id")
                or not isinstance(row.get("surface"), str)
                for index, row in enumerate(timeline_lexical)):
            return ["candidate timeline lexical identity/order is invalid"]
        timeline_raw = timeline.get("raw_timeline_neighbors")
        raw_sequence_reasons = _nvasr_raw_timeline_sequence_reasons(
            timeline_raw)
        if raw_sequence_reasons:
            return [
                "candidate timeline raw event contract invalid:"
                f"{','.join(raw_sequence_reasons)}"
            ]
        for candidate_index, candidate in enumerate(candidates):
            anchor_reasons = _nvasr_anchor_binding_reasons(candidate)
            if anchor_reasons:
                return [
                    f"candidate timeline spike-anchor contract invalid at "
                    f"{candidate_index}:{','.join(anchor_reasons)}"
                ]
            coordinate_reasons = _nvasr_candidate_coordinate_binding_reasons(
                candidate)
            if coordinate_reasons:
                return [
                    f"candidate timeline coordinate contract invalid at "
                    f"{candidate_index}:{','.join(coordinate_reasons)}"
                ]
            raw_reasons = _nvasr_raw_timeline_contract_reasons(
                candidate, timeline_events=timeline_raw)
            if raw_reasons:
                return [
                    f"candidate timeline raw neighbor contract invalid at "
                    f"{candidate_index}:{','.join(raw_reasons)}"
                ]

    def row_key(row: dict, rows: list[dict], index: int) -> tuple[int | None, int | None]:
        explicit = explicit_neighbor_key(row)
        if explicit is not None:
            return explicit
        lexical_before = sum(
            1 for prior in rows[:index]
            if _nvasr_semantic_axis_member(prior.get("word")))
        lexical_after = sum(
            1 for later in rows[index + 1:]
            if _nvasr_semantic_axis_member(later.get("word")))
        return (lexical_before - 1 if lexical_before else None,
                lexical_before if lexical_after else None)

    def valid_span(value) -> bool:
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(point, bool)
                       or not isinstance(point, (int, float))
                       or not math.isfinite(float(point)) for point in value)):
            return False
        return 0 <= float(value[0]) < float(value[1])

    def valid_candidate(candidate: dict) -> bool:
        frame_keys = (
            "raw_start_frame", "raw_end_frame",
            "speech_start_frame", "speech_end_frame")
        frames_valid = all(
            isinstance(candidate.get(key), int)
            and not isinstance(candidate.get(key), bool)
            for key in frame_keys)
        return (
            isinstance(candidate.get("candidate_id"), str)
            and bool(candidate.get("candidate_id"))
            and isinstance(candidate.get("kind"), str)
            and isinstance(candidate.get("surface"), str)
            and isinstance(candidate.get("token_id"), int)
            and isinstance(candidate.get("token_ids"), list)
            and valid_span([candidate.get("raw_start_s"), candidate.get("raw_end_s")])
            and valid_span([candidate.get("speech_start_s"),
                            candidate.get("speech_end_s")])
            and frames_valid
            and candidate["raw_start_frame"] < candidate["raw_end_frame"]
            and candidate["speech_start_frame"] < candidate["speech_end_frame"]
            and (not strict_v3 or (
                candidate.get("nvasr_candidate_schema_version") ==
                NVASR_CANDIDATE_SCHEMA_VERSION
                and candidate.get("mapping_axis") == NVASR_MAPPING_AXIS
                and isinstance(candidate.get("ordered_semantic_neighbors"), list)
                and not _nvasr_candidate_coordinate_binding_reasons(candidate)
                and not _nvasr_raw_timeline_contract_reasons(
                    candidate,
                    timeline_events=timeline.get("raw_timeline_neighbors"))
                and _nvasr_valid_anchor(candidate.get("ctc_spike_anchor")))))

    def candidate_v3_identity_valid(candidate: dict) -> bool:
        if not strict_v3:
            return True
        lexical_by_ordinal = {
            item.get("lexical_ordinal"): item
            for item in timeline_lexical or [] if isinstance(item, dict)}
        expected = []
        for side, ordinal in (
                ("left", candidate.get("left_lexical_ordinal")),
                ("right", candidate.get("right_lexical_ordinal"))):
            if ordinal is None:
                continue
            item = lexical_by_ordinal.get(ordinal)
            if item is None:
                return False
            expected.append({
                "side": side,
                "lexical_ordinal": ordinal,
                "occurrence_id": item.get("occurrence_id"),
                "surface": item.get("surface"),
                "surface_occurrence": item.get("surface_occurrence"),
            })
        return candidate.get("ordered_semantic_neighbors") == expected

    def explicit_neighbor_key(row: dict):
        mapping_key = row.get("mapping_key")
        if isinstance(mapping_key, dict):
            left = mapping_key.get("left_lexical_ordinal")
            right = mapping_key.get("right_lexical_ordinal")
            if (isinstance(left, int) or left is None) and \
                    (isinstance(right, int) or right is None):
                return left, right
        left = row.get("left_lexical_ordinal")
        right = row.get("right_lexical_ordinal")
        if ((isinstance(left, int) or left is None)
                and (isinstance(right, int) or right is None)
                and ("left_lexical_ordinal" in row
                     or "right_lexical_ordinal" in row)):
            return left, right
        return None

    # Punctuation is emitted outside ``words_pinyin``.  Keep a read-only
    # merged event view so topology can identify immediate neighbours and
    # derive their lexical-neighbour key without changing punctuation rows.
    emitted_events = [
        ("word", index, row) for index, row in enumerate(words_pinyin)
    ] + [
        ("punct", index, row) for index, row in enumerate(punct_entries)
    ]
    event_spans: dict[tuple[str, int], tuple[float, float]] = {}
    malformed_event = False
    for event_kind, event_index, event in emitted_events:
        try:
            start = float(event["start"])
            end = float(event["end"])
            if (not math.isfinite(start) or not math.isfinite(end)
                    or start >= end):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            malformed_event = True
            continue
        event_spans[(event_kind, event_index)] = (start, end)
    emitted_events = [event for event in emitted_events
                      if (event[0], event[1]) in event_spans]
    emitted_events.sort(key=lambda event: (
        event_spans[(event[0], event[1])][0],
        event_spans[(event[0], event[1])][1],
        0 if event[0] == "word" else 1,
        event[1],
    ))
    event_positions = {
        (event_kind, event_index): position
        for position, (event_kind, event_index, _event) in
        enumerate(emitted_events)
    }

    def event_neighbor_key(position: int):
        _kind, _index, event = emitted_events[position]
        explicit = explicit_neighbor_key(event)
        if explicit is not None:
            return explicit
        lexical_before = sum(
            1 for prior_kind, _prior_index, prior in emitted_events[:position]
            if prior_kind == "word"
            and _nvasr_semantic_axis_member(prior.get("word"))
        )
        lexical_after = sum(
            1 for later_kind, _later_index, later in emitted_events[position + 1:]
            if later_kind == "word"
            and _nvasr_semantic_axis_member(later.get("word"))
        )
        return (lexical_before - 1 if lexical_before else None,
                lexical_before if lexical_after else None)

    def bind_punctuation_entry(entry_index: int, target_key,
                               used_punctuation: set[int]):
        """Return one uniquely speech-overlapping timeline punctuation row."""
        entry = punct_entries[entry_index]
        if entry.get("source") != "ctc":
            return None
        event_position = event_positions.get(("punct", entry_index))
        if event_position is None or event_neighbor_key(event_position) != target_key:
            return None
        try:
            forced_start = float(entry["start"])
            forced_end = float(entry["end"])
            if (not math.isfinite(forced_start)
                    or not math.isfinite(forced_end)
                    or forced_start >= forced_end):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        matches = [
            (index, candidate) for index, candidate in enumerate(candidates)
            if index not in used_punctuation
            and candidate.get("kind") == "punctuation"
            and candidate.get("source") == "ctc"
            and candidate.get("surface") == entry.get("word")
            and (candidate.get("left_lexical_ordinal"),
                 candidate.get("right_lexical_ordinal")) == target_key
        ]
        if not matches or any(not valid_candidate(candidate)
                              for _index, candidate in matches):
            return None
        scored = []
        for candidate_index, candidate in matches:
            try:
                speech_start = float(candidate["speech_start_s"])
                speech_end = float(candidate["speech_end_s"])
                overlap = max(
                    0.0,
                    min(forced_end, speech_end)
                    - max(forced_start, speech_start),
                )
                if not math.isfinite(overlap):
                    return None
            except (KeyError, TypeError, ValueError):
                return None
            scored.append((overlap, candidate_index, candidate))
        positive = [item for item in scored if item[0] > 1e-9]
        if not positive:
            return None
        best_overlap = max(item[0] for item in positive)
        best = [item for item in positive
                if math.isclose(item[0], best_overlap, abs_tol=1e-9)]
        if len(best) != 1:
            return None
        return best[0]

    def punctuation_topology_selection(
            row_index: int, full_index: int, row: dict, target_key,
            matches: list[tuple[int, dict]], used_candidates: set[int]):
        if malformed_event:
            return None
        word_position = event_positions.get(("word", full_index))
        if word_position is None or word_position == 0 \
                or word_position + 1 >= len(emitted_events):
            return None
        before_kind, before_index, before = emitted_events[word_position - 1]
        after_kind, after_index, after = emitted_events[word_position + 1]
        if (before_kind != "punct" or after_kind != "punct"
                or before.get("source") != "ctc"
                or after.get("source") != "ctc"):
            return None
        try:
            forced = row.get("forced_span") or [row["start"], row["end"]]
            forced_start = float(forced[0])
            forced_end = float(forced[1])
            before_start, before_end = event_spans[("punct", before_index)]
            after_start, _after_end = event_spans[("punct", after_index)]
        except (KeyError, TypeError, ValueError):
            return None
        if (before_end > forced_start or after_start < forced_end):
            return None
        before_bound = bind_punctuation_entry(
            before_index, target_key, set())
        if before_bound is None:
            return None
        before_overlap, before_candidate_index, _before_candidate = before_bound
        after_bound = bind_punctuation_entry(
            after_index, target_key, {before_candidate_index})
        if after_bound is None:
            return None
        after_overlap, after_candidate_index, _after_candidate = after_bound
        if before_candidate_index >= after_candidate_index:
            return None
        inside = [
            (candidate_index, candidate)
            for candidate_index, candidate in matches
            if candidate_index not in used_candidates
            and before_candidate_index
            < nvv_candidate_positions[candidate_index]
            < after_candidate_index
        ]
        if len(inside) != 1:
            return None
        candidate_index, candidate = inside[0]
        return candidate_index, candidate

    errors: list[str] = []
    used_candidates: set[int] = set()
    pending_nvv_bindings: list[tuple[dict, dict, str, float | None]] = []
    # Output rows are authoritative for whether an NVV was emitted.  Free
    # logits can contain additional biased NVV runs that the decoder did not
    # select into ``text_asr``; requiring every such diagnostic candidate to
    # become a row falsely rejects otherwise complete utterances.  Bind in the
    # opposite direction instead: every emitted row must have exactly one raw
    # candidate, while unused candidates remain diagnostic timeline evidence.
    for row_index, (full_index, row) in enumerate(nvv_rows):
        label = str(row.get("word", ""))
        key = row_key(row, words_pinyin, full_index)
        forced = row.get("forced_span")
        if forced is None:
            forced = [row.get("start"), row.get("end")]
        if not valid_span(forced):
            errors.append(
                f"output NVV row {row_index}: malformed forced span "
                f"for {label!r}")
            continue
        matches = [
            (index, candidate) for index, candidate in enumerate(nvv_candidates)
            if index not in used_candidates
            and nvv_to_mfa(str(candidate.get("surface", ""))) == label
            and (candidate.get("left_lexical_ordinal"),
                 candidate.get("right_lexical_ordinal")) == key
        ]
        if strict_v3 and any(
                not valid_candidate(candidate)
                or not candidate_v3_identity_valid(candidate)
                for _candidate_index, candidate in matches):
            errors.append(
                f"output NVV row {row_index}: semantic candidate identity/order invalid")
            continue
        selection = "label_neighbors"
        selected_overlap = None
        topology_eligible = False
        if not matches and strict_v3:
            # Canonical English retokenisation can change the raw-neighbour
            # key.  The only permitted recovery is a unique label match with
            # positive forced/CTC-anchor overlap; raw-neighbour diagnostics do
            # not participate in this fallback proof.
            label_matches = [
                (index, candidate) for index, candidate in enumerate(nvv_candidates)
                if index not in used_candidates
                and nvv_to_mfa(str(candidate.get("surface", ""))) == label
                and valid_candidate(candidate)
                and candidate_v3_identity_valid(candidate)
            ]
            positive = []
            for candidate_index, candidate in label_matches:
                anchor = candidate["ctc_spike_anchor"]
                overlap = max(
                    0.0,
                    min(float(forced[1]), float(anchor["end"]))
                    - max(float(forced[0]), float(anchor["start"])),
                )
                if overlap > 1e-9:
                    positive.append((candidate_index, candidate, overlap))
            if len(positive) == 1:
                candidate_index, candidate, selected_overlap = positive[0]
                matches = [(candidate_index, candidate)]
                selection = "unique_label_forced_ctc_overlap"
        if len(matches) > 1:
            # An ambiguous key is only rankable when every matching candidate
            # is structurally trustworthy.  Do this before either coordinate
            # axis is scored: treating a malformed competitor as zero would
            # let an unknowable candidate lose by construction.
            malformed = [candidate for _index, candidate in matches
                         if not valid_candidate(candidate)]
            if malformed:
                errors.append(
                    f"output NVV row {row_index}: malformed candidate mapping "
                    f"for {label!r} at neighbors {key!r}")
                continue
            forced_start, forced_end = float(forced[0]), float(forced[1])
            scored = []
            for candidate_index, candidate in matches:
                try:
                    speech_start = float(candidate["speech_start_s"])
                    speech_end = float(candidate["speech_end_s"])
                    overlap = max(
                        0.0,
                        min(forced_end, speech_end) - max(forced_start, speech_start),
                    )
                except (KeyError, TypeError, ValueError):
                    overlap = 0.0
                scored.append((overlap, candidate_index, candidate))
            positive = [item for item in scored if item[0] > 1e-9]
            if positive:
                best_overlap = max(item[0] for item in positive)
                best = [item for item in positive
                        if math.isclose(item[0], best_overlap, abs_tol=1e-9)]
                if len(best) == 1:
                    selected_overlap, candidate_index, candidate = best[0]
                    matches = [(candidate_index, candidate)]
                    selection = "unique_max_forced_speech_overlap"
            else:
                # The free-decode speech axis is shifted by the immutable
                # query prefix.  When it cannot distinguish same-key
                # candidates, use the candidate's untouched raw encoder span
                # against the forced row.  This is deliberately a separate
                # basis: never store this score in the speech-overlap field.
                scored_raw = []
                malformed_raw = False
                for candidate_index, candidate in matches:
                    try:
                        raw_start = float(candidate["raw_start_s"])
                        raw_end = float(candidate["raw_end_s"])
                        if (not math.isfinite(raw_start)
                                or not math.isfinite(raw_end)
                                or raw_end <= raw_start):
                            raise ValueError("malformed raw candidate span")
                        overlap = max(
                            0.0,
                            min(forced_end, raw_end) - max(forced_start, raw_start),
                        )
                        if not math.isfinite(overlap):
                            raise ValueError("malformed forced/raw overlap")
                    except (KeyError, TypeError, ValueError):
                        malformed_raw = True
                        continue
                    scored_raw.append((overlap, candidate_index, candidate))
                if malformed_raw:
                    errors.append(
                        f"output NVV row {row_index}: malformed raw candidate "
                        f"mapping for {label!r} at neighbors {key!r}")
                    continue
                positive_raw = [item for item in scored_raw if item[0] > 1e-9]
                if positive_raw:
                    best_raw_overlap = max(item[0] for item in positive_raw)
                    best_raw = [item for item in positive_raw
                                if math.isclose(item[0], best_raw_overlap,
                                                abs_tol=1e-9)]
                    if len(best_raw) == 1:
                        _unused_overlap, candidate_index, candidate = best_raw[0]
                        matches = [(candidate_index, candidate)]
                        selection = "unique_max_forced_raw_overlap"
                else:
                    # Topology is a last-resort disambiguator only when both
                    # coordinate axes are entirely non-positive.  A positive
                    # tie on either axis is evidence of ambiguity, never an
                    # invitation to use surrounding punctuation.
                    topology_eligible = True
            if topology_eligible and len(matches) > 1:
                topology = punctuation_topology_selection(
                    row_index, full_index, row, key, matches, used_candidates)
                if topology is not None:
                    candidate_index, candidate = topology
                    matches = [(candidate_index, candidate)]
                    selection = "unique_punctuation_topology_bound"
        if len(matches) != 1:
            outcome = "missing" if not matches else "ambiguous"
            errors.append(
                f"output NVV row {row_index}: {outcome} candidate mapping "
                f"for {label!r} at neighbors {key!r}")
            continue
        candidate_index, candidate = matches[0]
        if not valid_candidate(candidate):
            errors.append(
                f"output NVV row {row_index}: malformed candidate mapping "
                f"for {label!r} at neighbors {key!r}")
            continue
        used_candidates.add(candidate_index)
        pending_nvv_bindings.append(
            (row, candidate, selection, selected_overlap))

    # Commit atomically.  A rejected bundle must not expose a mixture of
    # bound and unbound rows to later diagnostics or retry logic.
    if errors:
        return errors
    for row, candidate, selection, selected_overlap in pending_nvv_bindings:
        forced_span = row.get("forced_span") or [row["start"], row["end"]]
        row.update(_nvasr_candidate_provenance(
            candidate, forced_span=forced_span))
        # ``adjusted_span`` is the durable display/correspondence coordinate;
        # keep it bound to the emitted row without confusing it with the
        # immutable spike-anchor provenance derived by the consumer.
        row.setdefault("adjusted_span", [float(row["start"]),
                                          float(row["end"])])
        row["mapping_selection"] = selection
        if selected_overlap is not None:
            overlap_field = (
                "mapping_forced_ctc_anchor_overlap_s"
                if selection == "unique_label_forced_ctc_overlap"
                else "mapping_forced_speech_overlap_s")
            row[overlap_field] = round(selected_overlap, 6)

    # Punctuation already has its own ``ctc-punctuation-evidence-v2``
    # coordinate contract.  In particular, its ``raw_start_s`` means the
    # forced punctuation anchor, not the free-decode encoder axis used by
    # NVASR candidates.  Never project NVASR provenance onto these rows: doing
    # so moves punctuation and can then lengthen the preceding NVV token.
    return errors


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
            candidate_timeline = extract_nvasr_candidate_timeline(
                raw_y,
                asr_final,
                token_decoder=lambda token_id: tokenizer.decode([token_id]),
                stem=stem,
                query_frames=QUERY_FRAMES,
                frame_ms=FRAME_MS,
                blank_id=self.blank_id,
                ellipsis_id=ELLIPSIS_ID,
                pause_threshold=pause_threshold,
            )
            expected_timeline_duration = max(0.0, duration_s)
            if not math.isclose(
                    candidate_timeline["duration_s"],
                    expected_timeline_duration,
                    rel_tol=0.0,
                    abs_tol=1e-6):
                raise ValueError(
                    "NVASR candidate timeline duration does not match "
                    "the encoder speech-axis duration")
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
                "nvasr_candidate_timeline": candidate_timeline,
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


def _ctc_token_sidecar_row(
        word_row: dict, start_s: float, end_s: float, *,
        stem: str | None = None, row_ordinal: int | None = None) -> dict:
    """Serialize one durable token row without dropping provenance fields."""
    line = {
        "word": word_row["word"],
        "start_ms": round(start_s * 1000, 1),
        "end_ms": round(end_s * 1000, 1),
        "start_s": start_s,
        "end_s": end_s,
        "type": "word",
    }
    for key in CTC_TOKEN_SIDECAR_PASSTHROUGH_FIELDS:
        if key in word_row:
            line[key] = word_row[key]
    if stem is not None or row_ordinal is not None:
        if (not isinstance(stem, str) or not stem or Path(stem).name != stem
                or not isinstance(row_ordinal, int)
                or isinstance(row_ordinal, bool) or row_ordinal < 0):
            raise ValueError("raw CTC token locator arguments are invalid")
        line["ctc_raw_token_row"] = {
            "schema": CTC_RAW_TOKEN_ROW_SCHEMA,
            "stem": stem,
            "sidecar": f"{stem}_tokens.jsonl",
            "row_ordinal": row_ordinal,
        }
    return line

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
                # Keep the first candidate's raw CTC span immutable.  The
                # merged display envelope is separate processed evidence.
                first["processed_end_ms"] = last.get(
                    "processed_end_ms", last["end_ms"])
                first["processed_end_s"] = last.get(
                    "processed_end_s", last["end_s"])
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


def _rebase_final_token_sidecars(ctc_dir: Path) -> int:
    """Rebase every final token sidecar after cardinality-changing passes.

    Normalizers may contract rows, so their persisted coordinates and
    producer locators are no longer authoritative.  This single barrier
    assigns physical row locators, contiguous full lexical ordinals, and the
    final compact semantic identities.  Raw NVASR evidence is deliberately
    excluded from the rewrite; its digest is checked against
    ``raw_timeline_mapping_key`` and is never rehashed here.
    """
    plans: list[tuple[Path, list[dict], str, str]] = []
    for path in sorted(ctc_dir.rglob("*_tokens.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe final token sidecar: {path}")
        stem = path.name[:-len("_tokens.jsonl")]
        if not stem or Path(stem).name != stem:
            raise ValueError(f"invalid final token sidecar stem: {path}")
        try:
            original_text = path.read_text(encoding="utf-8-sig")
            rows = [json.loads(line) for line in original_text.splitlines()
                    if line.strip()]
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid final token sidecar: {path}") from exc
        if not rows or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"empty/non-object final token sidecar: {path}")
        plans.append((path, rows, stem, original_text))

    changed = 0
    serialized: list[tuple[Path, str, str]] = []
    for path, rows, stem, original_text in plans:
        sidecar = path.name
        for ordinal, row in enumerate(rows):
            row["ctc_raw_token_row"] = {
                "schema": CTC_RAW_TOKEN_ROW_SCHEMA,
                "stem": stem,
                "sidecar": sidecar,
                "row_ordinal": ordinal,
            }

        lexical_ordinal = 0
        semantic_rows: list[tuple[int, dict, int]] = []
        surface_counts: dict[str, int] = {}
        for position, row in enumerate(rows):
            text = str(row.get("word", "")).strip()
            if text and not is_silence(text) and not is_punct(text):
                row["ctc_lexical_ordinal"] = lexical_ordinal
                lexical_ordinal += 1
            if not _nvasr_semantic_axis_member(text):
                continue
            semantic_ordinal = len(semantic_rows)
            occurrence = surface_counts.get(text, 0)
            surface_counts[text] = occurrence + 1
            row["semantic_occurrence_id"] = (
                f"nvasr-lexical-{semantic_ordinal:04d}")
            row["semantic_surface_occurrence"] = occurrence
            semantic_rows.append((position, row, semantic_ordinal))

        for position, row in enumerate(rows):
            if not is_nvv_token(str(row.get("word", "")).strip()):
                continue
            left = next((item for item in reversed(semantic_rows)
                         if item[0] < position), None)
            right = next((item for item in semantic_rows
                          if item[0] > position), None)
            neighbors = []
            for side, item in (("left", left), ("right", right)):
                if item is None:
                    continue
                _position, neighbor, ordinal = item
                neighbors.append({
                    "side": side,
                    "lexical_ordinal": ordinal,
                    "occurrence_id": neighbor["semantic_occurrence_id"],
                    "surface": neighbor["word"],
                    "surface_occurrence": neighbor[
                        "semantic_surface_occurrence"],
                })
            if row.get("nvasr_candidate_schema_version") == \
                    NVASR_CANDIDATE_SCHEMA_VERSION:
                row["ordered_semantic_neighbors"] = neighbors
                row["mapping_key"] = {
                    "left_lexical_ordinal": (
                        left[2] if left is not None else None),
                    "right_lexical_ordinal": (
                        right[2] if right is not None else None),
                }

        # Validate the complete planned output before publishing any member.
        expected_ordinals = list(range(lexical_ordinal))
        actual_ordinals = [row.get("ctc_lexical_ordinal") for row in rows
                           if str(row.get("word", "")).strip()
                           and not is_silence(str(row.get("word", "")))
                           and not is_punct(str(row.get("word", "")))]
        if actual_ordinals != expected_ordinals:
            raise ValueError(f"non-contiguous final CTC lexical ordinals: {path}")
        for ordinal, row in enumerate(rows):
            locator = row.get("ctc_raw_token_row")
            if locator != {
                    "schema": CTC_RAW_TOKEN_ROW_SCHEMA,
                    "stem": stem, "sidecar": sidecar,
                    "row_ordinal": ordinal}:
                raise ValueError(f"final CTC locator validation failed: {path}:{ordinal}")
            if row.get("nvasr_candidate_schema_version") != \
                    NVASR_CANDIDATE_SCHEMA_VERSION:
                continue
            raw_key = row.get("raw_timeline_mapping_key")
            if (not isinstance(raw_key, dict)
                    or set(raw_key) != {
                        "left_lexical_ordinal", "right_lexical_ordinal"}):
                raise ValueError(
                    f"strict-v3 raw timeline mapping key missing: {path}:{ordinal}")
            raw_reasons = _nvasr_raw_timeline_contract_reasons(row)
            if raw_reasons:
                raise ValueError(
                    f"strict-v3 raw timeline evidence invalid: {path}:{ordinal}: "
                    f"{','.join(raw_reasons)}")
            if not is_nvv_token(str(row.get("word", "")).strip()):
                continue
            left = next((item for item in reversed(semantic_rows)
                         if item[0] < ordinal), None)
            right = next((item for item in semantic_rows if item[0] > ordinal), None)
            expected_key = {
                "left_lexical_ordinal": left[2] if left else None,
                "right_lexical_ordinal": right[2] if right else None,
            }
            if (row.get("mapping_key") != expected_key
                    or row.get("ordered_semantic_neighbors") != [
                        {
                            "side": side,
                            "lexical_ordinal": item[2],
                            "occurrence_id": item[1]["semantic_occurrence_id"],
                            "surface": item[1]["word"],
                            "surface_occurrence": item[1][
                                "semantic_surface_occurrence"],
                        }
                        for side, item in (("left", left), ("right", right))
                        if item is not None]):
                raise ValueError(
                    f"strict-v3 canonical mapping rebase validation failed: "
                    f"{path}:{ordinal}")
        output = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                         for row in rows)
        serialized.append((path, output, original_text))
        if output != original_text:
            changed += 1

    # No filesystem offers an atomic multi-file replace. Stage the complete
    # validated plan in memory, publish only changed members, and roll back
    # every already-published member if a later replace fails. A rollback
    # failure remains a hard error so the manifest/receipt layer fails closed.
    published: list[tuple[Path, str]] = []
    try:
        for path, output, original_text in serialized:
            if output == original_text:
                continue
            _atomic_write_text(path, output)
            published.append((path, original_text))
    except Exception as exc:
        rollback_errors = []
        for path, original_text in reversed(published):
            try:
                _atomic_write_text(path, original_text)
            except Exception as rollback_exc:
                rollback_errors.append(f"{path}:{rollback_exc}")
        detail = (f"; rollback failed: {', '.join(rollback_errors)}"
                  if rollback_errors else "; rollback complete")
        raise OSError(f"final CTC sidecar transaction failed{detail}") from exc
    return changed


# Explicit descriptive aliases for downstream callers/tests.
rebase_final_token_sidecars = _rebase_final_token_sidecars
_rebase_final_ctc_tokens = _rebase_final_token_sidecars


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
            merged = dict(a)
            merged.update({
                "word": "ria",
                "start_ms": a["start_ms"], "end_ms": b["end_ms"],
                "start_s": a["start_s"], "end_s": b["end_s"],
                "type": a.get("type", "word"),
            })
            source_ordinals: list[int] = []
            for fallback, row in ((i, a), (i + 1, b)):
                values = row.get("source_ctc_ordinals")
                if values is None and "source_ctc_ordinal" in row:
                    values = [row["source_ctc_ordinal"]]
                if values is None:
                    values = [fallback]
                if (not isinstance(values, (list, tuple))
                        or any(not isinstance(value, int)
                               or isinstance(value, bool) or value < 0
                               for value in values)):
                    return False
                source_ordinals.extend(values)
            merged["source_ctc_ordinals"] = sorted(set(source_ordinals))
            merged.pop("source_ctc_ordinal", None)
            new_entries.append(merged)
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

    This function scans for adjacent pure-ASCII alphabetic fragments whose
    ordered case-folded concatenation is exactly ``ria``.  It also normalises
    standalone ``RIA``/``Ria`` → ``ria``.  Digits and punctuation are never
    ignored while deciding whether a fragment is eligible: ``a5`` is a
    pinyin syllable, not an ``a`` fragment.
    """
    if not words_pinyin:
        return words_pinyin

    _RIA_TARGET = "ria"
    _RIA_LETTERS = frozenset(_RIA_TARGET)  # {'r', 'i', 'a'}

    def is_fragment(token: object) -> bool:
        return (isinstance(token, str) and 0 < len(token) <= 4
                and token.isascii() and token.isalpha()
                and all(char.casefold() in _RIA_LETTERS for char in token))

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

        # Fragment detection: collect only pure ASCII alphabetic fragments
        # whose letters are a subset of "ria"'s letter set.
        if is_fragment(token):
            fragments = [w]
            j = i + 1
            while j < len(words_pinyin):
                nt = words_pinyin[j]
                nt_word = nt["word"]
                if is_fragment(nt_word):
                    fragments.append(nt)
                    j += 1
                else:
                    break

            # Stop at the shortest exact match.  Any later fragment remains
            # available to the outer scan (e.g. ``r i a ria`` -> two tokens).
            combined = ""
            match_end = None
            for fragment_index, fragment in enumerate(fragments, start=1):
                combined += fragment["word"].casefold()
                if combined == _RIA_TARGET:
                    match_end = fragment_index
                    break
                if not _RIA_TARGET.startswith(combined):
                    break
            if match_end is not None:
                matched = fragments[:match_end]
                # Merge: one "ria" token spanning the full time range
                result.append(_merge_ctc_row_metadata(
                    matched, word=_RIA_TARGET,
                    start=matched[0]["start"], end=matched[-1]["end"],
                    fallback_ordinals=list(range(i, i + match_end))))
                i += match_end
                continue

        result.append(w)
        i += 1

    return result


def _finalize_nvasr_canonical_neighbors(words_pinyin: list[dict]) -> list[dict]:
    """Commit canonical semantic identities after all lexical merges.

    Candidate binding intentionally happens before English/RIA normalization so
    raw timeline diagnostics remain available.  This final pass rewrites only
    the semantic-neighbour identity to the emitted ordinary rows; the raw
    neighbour diagnostic and immutable spike anchor are never changed.
    """
    ordinary: list[tuple[int, dict]] = []
    surface_counts: dict[str, int] = {}
    ctc_ordinal = 0
    for index, row in enumerate(words_pinyin):
        text = str(row.get("word", "")).strip()
        if text and not is_silence(text) and not is_punct(text):
            row["ctc_lexical_ordinal"] = ctc_ordinal
            ctc_ordinal += 1
        if not _nvasr_semantic_axis_member(text):
            continue
        ordinal = len(ordinary)
        occurrence = surface_counts.get(text, 0)
        surface_counts[text] = occurrence + 1
        row["semantic_occurrence_id"] = f"nvasr-lexical-{ordinal:04d}"
        row["semantic_surface_occurrence"] = occurrence
        ordinary.append((index, row))

    for index, row in enumerate(words_pinyin):
        if not is_nvv_token(str(row.get("word", "")).strip()):
            continue
        left = next((item for item in reversed(ordinary) if item[0] < index), None)
        right = next((item for item in ordinary if item[0] > index), None)
        neighbors = []
        for side, item in (("left", left), ("right", right)):
            if item is None:
                continue
            _ordinary_index, neighbor = item
            ordinal = int(neighbor["semantic_occurrence_id"].rsplit("-", 1)[1])
            neighbors.append({
                "side": side,
                "lexical_ordinal": ordinal,
                "occurrence_id": neighbor["semantic_occurrence_id"],
                "surface": neighbor["word"],
                "surface_occurrence": neighbor["semantic_surface_occurrence"],
            })
        if "nvasr_candidate_schema_version" in row:
            row["ordered_semantic_neighbors"] = neighbors
    return words_pinyin


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
            _all_source_stems = sorted(p.stem for p in _source_wavs)
            _all_eligible_stems = sorted(p.stem for p in all_wavs)
            if _all_exclusions:
                print(f"  Frozen source: {len(_source_wavs)} WAVs; "
                      f"eligible={len(all_wavs)}, exclusions={len(_all_exclusions)}")
            # Apply explicit operator selection before shard construction,
            # never inside each child.  A parent stems file is authoritative
            # for this bounded invocation just as it is for a child.
            if args.stems_file:
                try:
                    _selected = [
                        line.strip() for line in args.stems_file.read_text(
                            encoding="utf-8").splitlines()
                    ]
                    if not _selected or _selected != sorted(set(_selected)):
                        raise ValueError("stems file must be sorted and unique")
                    _available = {p.stem: p for p in all_wavs}
                    _missing = [stem for stem in _selected if stem not in _available]
                    if _missing:
                        raise ValueError(
                            f"stems file contains unavailable stems: {_missing[:5]}"
                        )
                    all_wavs = [_available[stem] for stem in _selected]
                except (OSError, ValueError) as _exc:
                    print(f"ERROR: invalid --stems-file: {_exc}", file=sys.stderr)
                    return 2
            if args.offset > 0:
                all_wavs = all_wavs[args.offset:]
            if args.limit > 0:
                all_wavs = all_wavs[:args.limit]
            _selected_accounting_stems = (
                sorted(p.stem for p in all_wavs)
                if args.stems_file or args.offset > 0 or args.limit > 0
                else None
            )
            try:
                (_accounting_source_stems, _accounting_eligible_stems,
                 _accounting_exclusions) = _operator_bounded_accounting_universe(
                    _all_source_stems, _all_eligible_stems, _all_exclusions,
                    _selected_accounting_stems,
                )
            except ValueError as _exc:
                print(f"ERROR: invalid accounting universe: {_exc}", file=sys.stderr)
                return 2
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

                # All shard-local normalizers may have changed token
                # cardinality.  Rebase the merged publication once more so
                # physical locators and final semantic mappings describe the
                # exact parent sidecars before validation/manifest/receipt.
                _rebase_final_token_sidecars(_merge_output_dir)

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
                _filtered_stems_v2 = sorted(
                    set(_accounting_eligible_stems) - set(_output_stems_v2)
                )
                _shard_rows = [
                    {"shard_id": f"gpu{_gpu_id}", "stems": sorted(
                        p.stem for p in all_wavs[_gpu_id * per_gpu:
                                                  _gpu_id * per_gpu + per_gpu])}
                    for _gpu_id, _, _ in _procs
                ]
                _accounting = make_pipeline_accounting_receipt(
                    source_stems=_accounting_source_stems,
                    eligible_stems=_accounting_eligible_stems,
                    exclusions=_accounting_exclusions,
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
    _accounting_source_input = sorted(p.stem for p in source_wav_files)
    _accounting_eligible_input = sorted(p.stem for p in wav_files)

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
    _selected_accounting_stems = (
        sorted(p.stem for p in wav_files)
        if args.stems_file or args.offset > 0 or args.limit > 0
        else None
    )
    try:
        (_accounting_source_stems, _accounting_eligible_stems,
         _accounting_exclusions) = _operator_bounded_accounting_universe(
            _accounting_source_input,
            _accounting_eligible_input,
            source_exclusions,
            _selected_accounting_stems,
        )
    except ValueError as exc:
        print(f"ERROR: invalid accounting universe: {exc}", file=sys.stderr)
        return 2
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
                    # These are immutable raw CTC coordinates.  A later
                    # display owner may cover a larger silence gap, but it
                    # must never overwrite this evidence span.
                    "raw_start_s": float(w["start"]),
                    "raw_end_s": float(w["end"]),
                    "candidate_id": f"ctc-punct-{len(punct_entries):04d}",
                    "source": "ctc",
                })

        # Collapse adjacent decoder duplicates before binding.  The merged
        # forced envelope still has to select one unique raw occurrence; a tie
        # remains a hard error.  Binding continues to precede English/RIA
        # normalization so lexical-neighbour ordinals retain their raw meaning.
        if words_pinyin and not args.no_nvv:
            try:
                words_pinyin = _deduplicate_adjacent_nvv_rows(words_pinyin)
            except ValueError as exc:
                print(f"  FAIL {stem}: NVASR deduplication - {exc}")
                skipped.setdefault("nvasr_candidate_mapping", []).append(stem)
                fail += 1
                continue

        _candidate_mapping_errors = attach_nvasr_candidate_provenance(
            words_pinyin, punct_entries, r.get("nvasr_candidate_timeline", {}),
            strict_schema_v3=True)
        if _candidate_mapping_errors:
            print(f"  FAIL {stem}: NVASR candidate mapping - "
                  f"{'; '.join(_candidate_mapping_errors[:5])}")
            skipped.setdefault("nvasr_candidate_mapping", []).append(stem)
            fail += 1
            continue

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
                    merged_pinyin.append(_merge_ctc_row_metadata(
                        words_pinyin[i:j], word=merged_word,
                        start=w["start"], end=words_pinyin[j - 1]["end"],
                        fallback_ordinals=list(range(i, j))))
                    i = j
                else:
                    merged_pinyin.append(w)
                    i += 1
            words_pinyin = merged_pinyin

        # ── ria name integrity protection ──
        # Merge any remaining ria fragments that survived single-letter
        # merge (e.g. "R"+"ia"), normalise case ("RIA"→"ria").
        # Regression Case 31 Fix-4d.
        if not args.no_nvv:
            words_pinyin = _protect_ria(words_pinyin)

        # Canonical English/RIA merges are complete.  Preserve the raw
        # timeline diagnostics from attach, but publish semantic neighbours
        # only from this final ordinary-row sequence.
        words_pinyin = _finalize_nvasr_canonical_neighbors(words_pinyin)
        words_pinyin = _clamp_words_to_wav_axis(words_pinyin, duration_s)
        if not args.no_nvv:
            _final_provenance_errors = _validate_emitted_nvasr_provenance(
                words_pinyin)
            if _final_provenance_errors:
                print(f"  FAIL {stem}: final NVASR provenance - "
                      f"{'; '.join(_final_provenance_errors[:5])}")
                skipped.setdefault("nvasr_candidate_mapping", []).append(stem)
                fail += 1
                continue

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
                    # Keep the actual CTC frame span.  The old last-mark
                    # extension mixed raw evidence with processed display
                    # ownership and made stale punctuation look authoritative.
                    end_s = float(p.get("raw_end_s", p["end"]))
                    # Guard: CTC overlap (e.g. NVV inside word) can make
                    # end_s < start_s, producing invalid intervals.
                    if end_s <= p["start"]:
                        end_s = p["start"] + 0.060  # min 1 frame width
                    punct_data.append({
                        "schema": PUNCTUATION_EVIDENCE_SCHEMA,
                        "word": p["word"],
                        "start_ms": round(p["start"] * 1000, 1),
                        "end_ms": round(end_s * 1000, 1),
                        "start_s": float(p.get("raw_start_s", p["start"])),
                        "end_s": end_s,
                        "raw_start_s": float(p.get("raw_start_s", p["start"])),
                        "raw_end_s": end_s,
                        "candidate_id": p["candidate_id"],
                        "source": p.get("source", "ctc"),
                    })

                # Bind each candidate to lexical neighbors on the raw CTC
                # sequence.  Ordinals are computed from the post-tokenization
                # lexical stream and are never changed by display ownership.
                ordered_words = sorted(words_pinyin,
                                       key=lambda row: (row["start"], row["end"]))
                for candidate in punct_data:
                    start = candidate["raw_start_s"]
                    left = [row for row in ordered_words
                            if row["end"] <= start + 0.001]
                    right = [row for row in ordered_words
                             if row["start"] >= candidate["raw_end_s"] - 0.001]
                    candidate["left_lexical_ordinal"] = len(left) - 1 if left else None
                    candidate["right_lexical_ordinal"] = len(left) if right else None
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
        token_rows: list[dict] = []
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
            # Authority English and NVV rows carry their immutable evidence
            # ledgers into the sidecar.  Centralizing the passthrough list
            # prevents a successful in-memory binding from disappearing at
            # the producer serialization boundary.
            line = _ctc_token_sidecar_row(
                w, start_s, end_s, stem=stem, row_ordinal=i)
            token_rows.append(line)
        _serialized_candidates = [
            row["candidate_id"] for row in token_rows
            if row.get("candidate_kind") == "nvv"
        ]
        _serialized_locator_ordinals = [
            row.get("ctc_raw_token_row", {}).get("row_ordinal")
            for row in token_rows]
        if (_serialized_locator_ordinals != list(range(len(token_rows)))
                or _serialized_candidates != sorted(_serialized_candidates)
                or len(_serialized_candidates) != len(set(
                    _serialized_candidates))):
            print(f"  FAIL {stem}: serialized token locators/candidate IDs "
                  "are not unique and ordered")
            skipped.setdefault("nvasr_candidate_mapping", []).append(stem)
            fail += 1
            continue
        token_lines = [json.dumps(row, ensure_ascii=False)
                       for row in token_rows]
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
        _rebase_final_token_sidecars(args.output_dir)
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
            _output_v2 = sorted(_all_output_stems)
            _filtered_v2 = sorted(
                set(_accounting_eligible_stems) - set(_output_v2)
            )
            _accounting = make_pipeline_accounting_receipt(
                source_stems=_accounting_source_stems,
                eligible_stems=_accounting_eligible_stems,
                exclusions=_accounting_exclusions,
                output_stems=_output_v2,
                filtered_stems=_filtered_v2,
                run_id=make_pipeline_run_id(), mode="ctc_prealign",
                route=["ctc_prealign"],
                paths={"output": str(args.output_dir), "filtered": str(args.output_dir)},
                shards=[{"shard_id": "single", "stems": _accounting_eligible_stems}],
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
